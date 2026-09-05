//! The road network as a graph, and road ribbons as meshes.
//!
//! # The document's model, with two required changes
//!
//! The owner's document models a road network as a directed graph —
//! intersections are nodes, stretches of road are edges — and that is right; it
//! is how every routing system and every city generator represents roads. Its
//! struct shapes are adopted almost verbatim. Two things had to change:
//!
//! **1. `HashMap` must be `BTreeMap`.** The document writes
//! `segments: HashMap<u64, RoadSegment>`. Anything that reaches a cooked asset
//! must iterate in a deterministic order — the same law that made
//! `TerrainAssetBuilder` use a `BTreeMap` explicitly "so insertion order cannot
//! affect the output". With a `HashMap`, two cooks of the same Shapefile emit
//! different bytes, the cook-determinism gate goes red, and the reason would be
//! hard to find because the *geometry* would be identical every time. See
//! `the_graph_is_order_independent`.
//!
//! **2. `spine_coordinates: Vec<[f64; 2]>` must carry a Y.** The engine is Y-up
//! f64 (architecture rule 3), and a road that cannot say what height it is at is
//! a road that cannot cross a valley. Spines are `Vec<DVec3>` in world metres.
//!
//! # Intersections are derived, not read
//!
//! Published road layers do not carry intersection records. They carry segments
//! whose endpoints happen to coincide — to within whatever precision the agency
//! digitised at. So the graph derives its nodes by snapping endpoints onto a
//! quantised lattice ([`SNAP_TOLERANCE_M`]) and joining the ones that land
//! together. The tolerance is a real modelling decision and is documented where
//! it is defined, because too small leaves a city of disconnected stubs and too
//! large welds a bridge to the road beneath it.
//!
//! # Nothing here is persisted
//!
//! The graph is **derived at bake time from the vector layer**, not stored as its
//! own asset. That is deliberate and it follows `LoweredPcg`'s precedent: the
//! layer is the source of truth, the graph is a pure function of it, and a
//! derived thing that is also persisted is a thing that can disagree with its
//! own source. It also means this whole module costs **no schema ladder**.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec3, Vec3Swizzles};

use crate::feature::{GeoFeature, GeoGeometry, GeoLayer};

/// One traffic lane, in metres. The standard figure the document names, and the
/// one every highway authority designs to within a few tens of centimetres.
pub const LANE_WIDTH_M: f64 = 3.5;

/// How close two segment endpoints must be to be the same intersection.
///
/// **A real modelling decision, not a float epsilon.** Published centrelines are
/// digitised at metre-ish precision and two segments that meet at a junction
/// routinely disagree by a metre or two. Too small a tolerance leaves a network
/// of disconnected stubs — every junction becomes two dead ends — and the
/// symptom is a city whose roads visibly do not join. Too large welds an
/// overpass to the road beneath it, because a bridge and its underpass are metres
/// apart in plan and tens of metres apart in reality.
///
/// 2 m is chosen to sit above digitising noise and below any real road spacing.
/// Snapping is on the **XZ plane only** for exactly the overpass reason: two
/// endpoints at the same plan position but different heights are still snapped
/// together here, and separating them needs a bridge/tunnel attribute that
/// published layers rarely carry — a named limit, recorded in the disposition
/// memo rather than papered over.
pub const SNAP_TOLERANCE_M: f64 = 2.0;

/// The most a mitred corner may widen its cross-section, as a multiple of the
/// half-width.
///
/// A mitre offset is `half / cos(θ/2)`, which goes to infinity as a corner
/// approaches a hairpin. 4 is the usual limit (it is SVG's default) and admits
/// every corner down to about 29 degrees.
///
/// # What it does past that, said accurately (audit ROAD1)
///
/// This doc used to say that a sharper corner is "**clipped rather than
/// spiked**", and that is what a *stroker's* miter limit does: SVG falls back to
/// a bevel join and the spike disappears. This one does not. It clamps the
/// **ratio** at 4 and [`CrossFrame::at`] still multiplies the offset by it, so a
/// hairpin gets a spike exactly four half-widths long instead of an infinite
/// one — which was tolerable while the only thing offset from the centreline was
/// the carriageway, and stopped being tolerable when wave ROAD1 put a footway at
/// `built_half_width_m`.
///
/// Measured on the shipped island: the kerb-and-pavement mesh reaches **23.200 m
/// from the centreline**, which is 4.000 × an arterial's 5.800 m built
/// half-width, and **4 156.5 m² of the footway (2.43 %) sits past its own
/// route's built half-width**, worst **+17.400 m** at (-1057.7, 98.4, -453.7) —
/// a 17 m tongue of concrete pointing into open ground at a switchback. It is
/// distinct from the footway-on-the-carriageway defect
/// `clip_kerbs_to_open_ground` closes: a spike points *away* from every road, so
/// nothing clips it. Carried, with the remedy named: a real bevel join, which
/// changes the ribbon's topology at a corner and moves a committed
/// `.inf_mesh` — a wave, not an audit fix.
pub const MITER_LIMIT: f64 = 4.0;

/// The road classes the document names, plus the two a real layer always has.
///
/// # Freeze note
///
/// This enum does **not** reach a bincode wire today (the graph is derived, never
/// persisted). If that ever changes it becomes a freeze-pinned append-only wire
/// enum like `MovementMode`, and this comment is the warning that the change is
/// not free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RoadKind {
    /// Grade-separated, multi-lane.
    Highway,
    /// A major urban through-route.
    Arterial,
    /// A neighbourhood street. The default, because it is the commonest class
    /// and the least damaging thing to guess wrong.
    #[default]
    Residential,
    /// Unpaved rural track.
    DirtTrack,
    /// A footpath or cycleway — no vehicle lanes.
    Path,
    /// A railway. Present in every transportation layer and emphatically not a
    /// road; classifying it as one paves the tracks.
    Rail,
}

impl RoadKind {
    /// The default lane count for this class, used when the source has none.
    pub const fn default_lanes(self) -> u32 {
        match self {
            RoadKind::Highway => 4,
            RoadKind::Arterial => 4,
            RoadKind::Residential => 2,
            RoadKind::DirtTrack => 1,
            RoadKind::Path => 1,
            RoadKind::Rail => 1,
        }
    }

    /// **Whether this class is kerbed** — a street with a pavement — or open,
    /// with a shoulder (wave ROAD1).
    ///
    /// It is the class and not a per-feature attribute because published road
    /// layers do not carry one, and the class is the honest proxy: a motorway
    /// has no footway by definition, and a street in a town has one whether or
    /// not the survey said so.
    ///
    /// **The named limit**: this makes a rural arterial carry a pavement it does
    /// not need. Distinguishing "through a settlement" from "between two" wants
    /// a settlement mask the road layer has no field for, and `inf-gis` has no
    /// business knowing where a city is. Carried, not smuggled.
    pub const fn is_kerbed(self) -> bool {
        matches!(self, RoadKind::Arterial | RoadKind::Residential)
    }

    /// **The sealed shoulder this class carries**, metres a side — what an open
    /// road has where a street has a kerb (wave ROAD1).
    ///
    /// 2.5 m is the figure a rural highway is built to; a farm track's 0.75 m is
    /// the gravel edge a passing place is made of. A kerbed class answers zero,
    /// and [`is_kerbed`](RoadKind::is_kerbed) is the other half of the same
    /// decision — no class has both.
    pub const fn shoulder_m(self) -> f64 {
        match self {
            RoadKind::Highway => 2.5,
            RoadKind::DirtTrack => 0.75,
            _ => 0.0,
        }
    }

    /// **Whether this class carries painted markings** (wave ROAD1).
    ///
    /// A highway and an arterial do; a residential street in the country the
    /// island is in does **not** — a neighbourhood street is unmarked, and
    /// painting a centre line down one would be a road that reads as a
    /// through-route. A track, a footpath and a railway obviously carry none.
    pub const fn is_marked(self) -> bool {
        matches!(self, RoadKind::Highway | RoadKind::Arterial)
    }

    /// Surface width in metres for a lane count.
    ///
    /// Lanes are [`LANE_WIDTH_M`] except for the two classes that are not
    /// measured in lanes at all: a footpath is 2 m regardless, and a single
    /// railway track occupies about 4 m of formation.
    pub fn width_m(self, lanes: u32) -> f64 {
        let lanes = lanes.max(1) as f64;
        match self {
            RoadKind::Path => 2.0,
            RoadKind::Rail => 4.0 * lanes,
            RoadKind::DirtTrack => (LANE_WIDTH_M * lanes).max(3.0),
            _ => LANE_WIDTH_M * lanes,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            RoadKind::Highway => "highway",
            RoadKind::Arterial => "arterial",
            RoadKind::Residential => "residential",
            RoadKind::DirtTrack => "dirt",
            RoadKind::Path => "path",
            RoadKind::Rail => "rail",
        }
    }

    /// Classify a source's own road-type text.
    ///
    /// Deliberately generous about spelling: government layers use `FCC` codes,
    /// OSM-style `highway=` values, and plain English, sometimes in the same
    /// file. An unrecognised value becomes [`RoadKind::Residential`] and the
    /// caller counts it — guessing quietly is fine here precisely *because* the
    /// count is reported.
    pub fn classify(text: &str) -> Self {
        let t = text.trim().to_ascii_lowercase();
        let has = |n: &str| t.contains(n);
        if has("rail") || has("train") || has("tram") || has("subway") {
            RoadKind::Rail
        } else if has("motorway")
            || has("freeway")
            || has("highway")
            || has("expressway")
            || has("trunk")
        {
            RoadKind::Highway
        } else if has("primary")
            || has("secondary")
            || has("arterial")
            || has("major")
            || has("avenue")
        {
            RoadKind::Arterial
        } else if has("path")
            || has("foot")
            || has("cycle")
            || has("pedestrian")
            || has("trail")
            || has("sidewalk")
        {
            RoadKind::Path
        } else if has("dirt") || has("gravel") || has("unpaved") || has("track") || has("4wd") {
            RoadKind::DirtTrack
        } else {
            RoadKind::Residential
        }
    }
}

/// A stretch of road between two intersections.
#[derive(Clone, Debug, PartialEq)]
pub struct RoadSegment {
    pub id: u64,
    pub name: String,
    pub kind: RoadKind,
    /// Posted speed in km/h where the source declared one.
    ///
    /// Kept in km/h rather than converted to m/s **because it is a sign, not a
    /// physical quantity** — it is the number painted on the road, and rounding
    /// it into SI and back would turn 50 into 49.99. Anything that simulates with
    /// it converts at the point of use, which is what the units doctrine asks for
    /// at a data boundary.
    pub speed_limit_kmh: Option<u32>,
    pub lane_count: u32,
    /// The centreline, in world metres, in the source's own vertex order.
    pub spine: Vec<DVec3>,
    pub start_node: u64,
    pub end_node: u64,
}

impl RoadSegment {
    /// Surface width in metres.
    pub fn width_m(&self) -> f64 {
        self.kind.width_m(self.lane_count)
    }

    /// Centreline length on the XZ plane, in metres.
    pub fn length_m(&self) -> f64 {
        self.spine
            .windows(2)
            .map(|w| (w[1].xz() - w[0].xz()).length())
            .sum()
    }
}

/// A junction where segments meet.
#[derive(Clone, Debug, PartialEq)]
pub struct Intersection {
    pub id: u64,
    pub position: DVec3,
    /// Ids of the segments converging here, ascending. A `BTreeSet` so the order
    /// is a function of the data rather than of insertion.
    pub segments: BTreeSet<u64>,
}

impl Intersection {
    /// How many segments meet here. 1 is a dead end, 2 is a bend, 3+ is a real
    /// junction.
    pub fn degree(&self) -> usize {
        self.segments.len()
    }
}

/// The road network.
///
/// `BTreeMap` throughout — see the module docs for why that is load-bearing
/// rather than a preference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadGraph {
    pub segments: BTreeMap<u64, RoadSegment>,
    pub intersections: BTreeMap<u64, Intersection>,
    /// Features skipped while building, with the reason.
    pub skipped: Vec<String>,
    /// How many segments took a default road class because the source's value
    /// was not recognised. Reported so "everything came out residential" is a
    /// visible fact rather than a mystery.
    pub unclassified: usize,
}

/// Quantise an XZ position onto the snap lattice. Two endpoints that quantise
/// equal are the same node.
fn snap_key(p: DVec3) -> (i64, i64) {
    (
        (p.x / SNAP_TOLERANCE_M).round() as i64,
        (p.z / SNAP_TOLERANCE_M).round() as i64,
    )
}

impl RoadGraph {
    /// Build a graph from a vector layer's polylines.
    ///
    /// Attribute names are looked up over the spellings real layers use; a source
    /// with none of them still imports, with defaults, and the count of defaulted
    /// segments is reported in [`unclassified`](Self::unclassified).
    pub fn from_layer(layer: &GeoLayer) -> Self {
        let mut graph = RoadGraph::default();
        // Nodes are assigned in lattice order, not encounter order, so the ids
        // are a function of the geometry alone. Collect first, number after.
        let mut node_ids: BTreeMap<(i64, i64), u64> = BTreeMap::new();
        let mut pending: Vec<(usize, Vec<DVec3>)> = Vec::new();

        for (i, f) in layer.features.iter().enumerate() {
            let points = match &f.geometry {
                GeoGeometry::Polyline { points, .. } => points.clone(),
                GeoGeometry::Polygon { exterior, .. } => exterior.clone(),
                GeoGeometry::Point(_) => {
                    graph
                        .skipped
                        .push(format!("feature {i}: a point is not a road segment"));
                    continue;
                }
            };
            if points.len() < 2 {
                graph.skipped.push(format!(
                    "feature {i}: a road needs at least 2 positions, this has {}",
                    points.len()
                ));
                continue;
            }
            if let Some(bad) = points.iter().find(|p| !p.is_finite()) {
                graph
                    .skipped
                    .push(format!("feature {i}: non-finite position {bad:?}"));
                continue;
            }
            node_ids.insert(snap_key(points[0]), 0);
            node_ids.insert(snap_key(points[points.len() - 1]), 0);
            pending.push((i, points));
        }

        // Number the nodes in lattice order.
        for (n, (_, id)) in node_ids.iter_mut().enumerate() {
            *id = n as u64;
        }

        for (seg_id, (i, points)) in pending.into_iter().enumerate() {
            let f = &layer.features[i];
            let kind_text = f.attr_text(&ROAD_CLASS_FIELDS);
            let kind = match kind_text {
                Some(t) => {
                    let k = RoadKind::classify(t);
                    // `RoadKind::default()` IS `Residential`, so the second half
                    // of the old condition (`classify(t) == default()`) was a
                    // restatement of the first plus a redundant re-classify. What
                    // it MEANT is "classify fell through to the default" — which
                    // is what `k == default()` says on its own.
                    if k == RoadKind::default() {
                        // The text was present but matched nothing specific.
                        // Lower-cased once per feature rather than once per
                        // needle: this runs over 10^5-feature county layers.
                        let lower = t.to_ascii_lowercase();
                        let recognised = ["resid", "street", "local", "minor", "road"]
                            .iter()
                            .any(|n| lower.contains(n));
                        if !recognised {
                            graph.unclassified += 1;
                        }
                    }
                    k
                }
                None => {
                    graph.unclassified += 1;
                    RoadKind::default()
                }
            };
            let lane_count = f
                .attr_number(&ROAD_LANE_FIELDS)
                .filter(|v| *v >= 1.0 && *v <= 24.0)
                .map(|v| v as u32)
                .unwrap_or_else(|| kind.default_lanes());
            let speed_limit_kmh = f
                .attr_number(&[
                    "speed_limit_kmh",
                    "speed",
                    "maxspeed",
                    "spd_lim",
                    "speedlimit",
                ])
                .filter(|v| *v > 0.0 && *v <= 200.0)
                .map(|v| v as u32);
            let name = f
                .attr_text(&[
                    "name",
                    "street_name",
                    "full_name",
                    "fullname",
                    "streetname",
                    "label",
                ])
                .unwrap_or("")
                .to_string();

            let start_node = node_ids[&snap_key(points[0])];
            let end_node = node_ids[&snap_key(points[points.len() - 1])];
            let id = seg_id as u64;
            graph.segments.insert(
                id,
                RoadSegment {
                    id,
                    name,
                    kind,
                    speed_limit_kmh,
                    lane_count,
                    spine: points,
                    start_node,
                    end_node,
                },
            );
        }

        // Materialise the intersections from the segments that reached them.
        for ((kx, kz), id) in &node_ids {
            graph.intersections.insert(
                *id,
                Intersection {
                    id: *id,
                    position: DVec3::new(
                        *kx as f64 * SNAP_TOLERANCE_M,
                        0.0,
                        *kz as f64 * SNAP_TOLERANCE_M,
                    ),
                    segments: BTreeSet::new(),
                },
            );
        }
        // Positions come from the real endpoints rather than the lattice, so a
        // junction sits where the road actually is and not where it rounded to.
        let mut sums: BTreeMap<u64, (DVec3, f64)> = BTreeMap::new();
        for s in graph.segments.values() {
            for (node, p) in [
                (s.start_node, s.spine[0]),
                (s.end_node, s.spine[s.spine.len() - 1]),
            ] {
                let e = sums.entry(node).or_insert((DVec3::ZERO, 0.0));
                e.0 += p;
                e.1 += 1.0;
            }
        }
        let links: Vec<(u64, u64)> = graph
            .segments
            .values()
            .flat_map(|s| [(s.start_node, s.id), (s.end_node, s.id)])
            .collect();
        for (node, seg) in links {
            if let Some(n) = graph.intersections.get_mut(&node) {
                n.segments.insert(seg);
            }
        }
        for (node, (sum, count)) in sums {
            if let Some(n) = graph.intersections.get_mut(&node) {
                if count > 0.0 {
                    n.position = sum / count;
                }
            }
        }
        // An intersection nothing reached is not an intersection.
        graph.intersections.retain(|_, n| !n.segments.is_empty());
        graph
    }

    /// Total centreline length, metres.
    pub fn total_length_m(&self) -> f64 {
        self.segments.values().map(RoadSegment::length_m).sum()
    }

    /// Junctions where three or more segments meet — the real intersections, as
    /// opposed to the bends and dead ends.
    pub fn junctions(&self) -> impl Iterator<Item = &Intersection> {
        self.intersections.values().filter(|n| n.degree() >= 3)
    }

    /// The nearest point on any segment's centreline to `p`, as
    /// `(segment_id, position, distance_m)`.
    ///
    /// This is the query the document's building-orientation step needs ("query
    /// the road graph for the nearest segment, rotate the house so the door faces
    /// the street"). Linear over segments: `O(subjects)` in the segments, which
    /// is the right shape for the tens of thousands a city layer holds. A spatial
    /// index earns its place only if this ever measures slow, and the house rule
    /// is to measure before prescribing.
    pub fn nearest_on_centreline(&self, p: DVec3) -> Option<(u64, DVec3, f64)> {
        let mut best: Option<(u64, DVec3, f64)> = None;
        for s in self.segments.values() {
            for w in s.spine.windows(2) {
                let (a, b) = (w[0], w[1]);
                let ab = b.xz() - a.xz();
                let len_sq = ab.length_squared();
                let t = if len_sq <= f64::EPSILON {
                    0.0
                } else {
                    ((p.xz() - a.xz()).dot(ab) / len_sq).clamp(0.0, 1.0)
                };
                let q = a + (b - a) * t;
                let d = (p.xz() - q.xz()).length();
                if best.as_ref().is_none_or(|(_, _, bd)| d < *bd) {
                    best = Some((s.id, q, d));
                }
            }
        }
        best
    }

    /// **The road network as an [`inf_nav::NavGraph`]** — the same adjacency
    /// this struct has always been, in the one shape a search runs on.
    ///
    /// Nothing here is re-derived. An intersection is already a node with a
    /// world position; a segment is already an edge naming two of them and
    /// carrying a surveyed centreline. All this does is tag the ids into
    /// `inf-nav`'s namespace ([`inf_nav::domain::ROAD`], which exists because
    /// three producers mint ids independently and `NavGraph::absorb` joins on id
    /// *equality*) and hand each segment's **interior** spine points over as the
    /// edge's `via`, so a route across a switchback climbs it rather than
    /// cutting the corner. The spine is stored in the source's own vertex order
    /// and `from_layer` derives `start_node` from `spine[0]`, so `via` is
    /// already in `start → end` order and needs no reversal;
    /// `NavGraph::link_with_cost` reverses it for the other half of the link.
    ///
    /// The graph stays **derived and never persisted**, exactly as this one is —
    /// so exposing it costs no schema ladder either (see the module docs).
    ///
    /// # The cost is `length_m`, which is the PLAN length
    ///
    /// [`RoadSegment::length_m`] measures on the XZ plane, and the spine it
    /// measures is three-dimensional — so a road climbing a grade is priced
    /// here slightly under the chain a body actually walks. That is deliberate
    /// and it is why this uses `link_with_cost` rather than `link`, which would
    /// measure the chain: `length_m` is the number this layer reports
    /// everywhere else — `total_length_m`, the import report, the wizard's
    /// preview — and a route whose `cost_m` did not add up to the lengths of the
    /// segments it names would be a second opinion about the same road. The
    /// geometry is still carried, so `NavRoute::path.length_m()` is the number
    /// to read when the climb is what matters, and the two are the same on the
    /// flat.
    ///
    /// # A segment that begins and ends at one junction carries no route
    ///
    /// A loop — a roundabout digitised as one closed feature, a cul-de-sac ring
    /// — snaps both of its endpoints into the same lattice cell and therefore
    /// onto the same node, and `link_with_cost` refuses `a == b` because a
    /// self-edge can only ever lengthen a route. Its *geometry* is untouched: it
    /// still builds a ribbon and still paves ground. What it cannot do is be
    /// walked, and splitting it would need a second node this layer gave it no
    /// endpoint for — a named limit rather than a silent one, pinned by
    /// `a_self_looping_segment_carries_no_route`.
    pub fn nav_graph(&self) -> inf_nav::NavGraph {
        let mut g = inf_nav::NavGraph::new();
        for n in self.intersections.values() {
            g.add_node(
                inf_nav::domain::ROAD | n.id,
                n.position,
                inf_nav::NavKind::Road,
            );
        }
        for s in self.segments.values() {
            // The points strictly BETWEEN the two junctions. A two-point spine
            // has none, which costs a `Vec` header and no allocation.
            let via: Vec<DVec3> = if s.spine.len() > 2 {
                s.spine[1..s.spine.len() - 1].to_vec()
            } else {
                Vec::new()
            };
            g.link_with_cost(
                inf_nav::domain::ROAD | s.start_node,
                inf_nav::domain::ROAD | s.end_node,
                inf_nav::NavKind::Road,
                via,
                s.length_m(),
            );
        }
        g
    }

    /// **The lanes this road layer carries** (wave VEH2b) — the same graph
    /// [`nav_graph`](Self::nav_graph) hands back, with a carriageway on each
    /// side of every spine.
    ///
    /// This is where the two numbers the importer already probes off the source
    /// features finally do something: [`RoadSegment::lane_count`] decides how
    /// many carriageways a road has, and [`RoadSegment::speed_limit_kmh`] the
    /// sign on them. Before this they reached the *width* of the ribbon and
    /// nothing else, so a four-lane highway and a two-lane arterial imported
    /// from the same shapefile differed only in how much tarmac was drawn.
    ///
    /// # `Rail` carries no lanes
    ///
    /// A railway is in the road layer because the source data puts it there
    /// (`RoadKind::classify` maps `"rail"`), and it is drawn as a four-metre
    /// ribbon per track. It is not a carriageway, nothing in this engine drives
    /// a train, and a car routed onto one would be a car on a railway line — so
    /// the spec closure answers `None` for it, which is the door
    /// [`LaneNetwork::from_graph`] documents for a one-way street.
    ///
    /// # The arithmetic is not here
    ///
    /// `inf-nav` owns the offset, the mitre and the junction join
    /// ([`inf_nav::lane`]), because the settlement grids re-derived at runtime
    /// need exactly the same three things and this crate is banned from the
    /// shipped player. What lives here is the *translation*: a road class into a
    /// [`LaneSpec`], which is the one fact `inf-nav` must not know.
    ///
    /// [`LaneNetwork::from_graph`]: inf_nav::LaneNetwork::from_graph
    /// [`LaneSpec`]: inf_nav::LaneSpec
    pub fn lane_network(&self) -> inf_nav::LaneNetwork {
        let graph = self.nav_graph();
        // Keyed by the pair of tagged node ids, both ways, so the lookup inside
        // the closure is a `BTreeMap` hit rather than a walk over the segments.
        let mut spec_by_pair: BTreeMap<(u64, u64), inf_nav::LaneSpec> = BTreeMap::new();
        for s in self.segments.values() {
            if s.kind == RoadKind::Rail {
                continue;
            }
            let spec = inf_nav::LaneSpec {
                lane_count: s.lane_count,
                width_m: LANE_WIDTH_M,
                speed_limit_kmh: s.speed_limit_kmh.unwrap_or(default_speed_kmh(s.kind)),
            };
            let (a, b) = (
                inf_nav::domain::ROAD | s.start_node,
                inf_nav::domain::ROAD | s.end_node,
            );
            // Two segments between one pair of junctions — a dual carriageway
            // digitised as two features — keep the WIDER of the two, because a
            // lane the network refuses is a lane no car can use and a lane it
            // invents is a car in a hedge.
            let entry = spec_by_pair.entry((a.min(b), a.max(b))).or_insert(spec);
            if spec.lane_count > entry.lane_count {
                *entry = spec;
            }
        }
        inf_nav::LaneNetwork::from_graph(&graph, |from, edge| {
            spec_by_pair
                .get(&(from.min(edge.to), from.max(edge.to)))
                .copied()
        })
    }
}

/// **The sign a road class wears when the source data has none**, km/h.
///
/// Every committed road layer in this tree is one of these: `write_roads` emits
/// `name` and `road_type` and nothing else, so on the island *every* limit comes
/// from this function. Stated as a table rather than as one default, because a
/// highway and a farm track sharing a fifty is the kind of number that makes a
/// whole island's traffic move at one speed.
pub fn default_speed_kmh(kind: RoadKind) -> u32 {
    match kind {
        RoadKind::Highway => 90,
        RoadKind::Arterial => 60,
        RoadKind::Residential => 50,
        RoadKind::DirtTrack => 30,
        RoadKind::Path => 10,
        // Not a carriageway; `lane_network` never asks. Answered rather than
        // panicked, so the match stays total and a future caller gets a number.
        RoadKind::Rail => 0,
    }
}

/// A quad-ribbon mesh generated along a road spine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadRibbon {
    /// World-space vertices, left/right alternating along the spine.
    pub vertices: Vec<DVec3>,
    /// `(u, v)` per vertex: `u` is 0/1 across the road, `v` is **arc length in
    /// metres divided by the road width**, so road markings tile at a constant
    /// physical size regardless of how long the segment is.
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl RoadRibbon {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Extrude a road spine into a terrain-conforming quad ribbon.
///
/// `height_at` is asked for the ground elevation at each cross-section — the
/// engine's own `TerrainData::height_at` or `inf_voxel::ground_height_at` at the
/// call site, so this module needs no dependency on either. `lift_m` raises the
/// surface off the ground so it does not z-fight the terrain it follows.
///
/// This closes a follow-up `inf-math`'s own spline module had already written
/// down as unbuilt: "baking a spline to a renderable mesh (tube / ribbon)".
pub fn build_ribbon(
    spine: &[DVec3],
    width_m: f64,
    lift_m: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> Result<RoadRibbon, crate::GisError> {
    build_ribbon_across(spine, width_m, lift_m, 1, 0.0, height_at)
}

/// [`build_ribbon`], with the cross-section split into `strips` quads.
///
/// # Why a road has to be subdivided ACROSS as well as along
///
/// Resampling the spine makes a road follow the ground *along* its length and
/// does nothing at all for its width, and that is where the daylight actually
/// is: a 14 m arterial crossing a hill with 0.004 m⁻¹ of curvature leaves
/// **49 mm** between its crown and the ground, measured, against 0.4 mm along
/// its length at a 1 m step. The first version of this builder had one quad
/// across, and the along-the-spine resampling that closes the longitudinal gap
/// hid the transverse one perfectly, because both errors are "the road does not
/// follow the ground" and only one of them was being measured.
///
/// The alternative — a road that is planar across its width, which is what a
/// graded carriageway really is — needs the terrain cut and filled to meet it,
/// and that is the road-to-terrain blend the disposition memo lists as unbuilt.
/// Conforming is the honest thing an importer can do on its own.
pub fn build_ribbon_across(
    spine: &[DVec3],
    width_m: f64,
    lift_m: f64,
    strips: usize,
    crown_fall: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> Result<RoadRibbon, crate::GisError> {
    let strips = strips.max(1);
    // Drop consecutive duplicates: a zero-length segment has no direction, and a
    // perpendicular of a zero vector is where a NaN enters a mesh.
    let mut pts: Vec<DVec3> = Vec::with_capacity(spine.len());
    for &p in spine {
        if !p.is_finite() {
            return Err(crate::GisError::NotFinite(format!(
                "the road spine carries the non-finite position {p:?}"
            )));
        }
        if pts
            .last()
            .is_none_or(|q: &DVec3| (p.xz() - q.xz()).length_squared() > 1e-12)
        {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return Err(crate::GisError::Geometry(format!(
            "a road ribbon needs at least 2 distinct positions; this spine has {} \
             after removing repeated points",
            pts.len()
        )));
    }
    if !(width_m.is_finite() && width_m > 0.0) {
        return Err(crate::GisError::Geometry(format!(
            "road width {width_m} is not a positive finite length"
        )));
    }
    // The spine and the width were guarded and `lift_m` was not — and a NaN lift
    // makes every vertex's `y` a NaN that `f32::min`/`max` then ignore, so the
    // ribbon's bounds come back looking perfectly healthy. Same door, same
    // reason, one line earlier than the place it would have hurt.
    if !lift_m.is_finite() {
        return Err(crate::GisError::NotFinite(format!(
            "the road's lift above the ground ({lift_m} m) is not a finite length"
        )));
    }

    if !(crown_fall.is_finite() && crown_fall >= 0.0) {
        return Err(crate::GisError::Geometry(format!(
            "the carriageway's cross-fall ({crown_fall}) is not a non-negative finite fraction"
        )));
    }

    let half = width_m * 0.5;
    let mut vertices = Vec::with_capacity(pts.len() * 2);
    let mut uvs = Vec::with_capacity(pts.len() * 2);
    let mut indices = Vec::with_capacity((pts.len() - 1) * 6);
    // **One frame list, and every part of the road reads it** (wave ROAD1). The
    // mitre used to be worked out inline here and nowhere else; a kerb that
    // mitred differently from the carriageway it bounds opens a wedge at every
    // bend, so the corner geometry is now computed once, by `cross_frames`, and
    // the kerb, the pavement, the lines and the crossings all take it from
    // there.
    let frames = cross_frames(&pts);
    let opts = SurfaceOptions {
        lift_m,
        crown_fall,
        // The plateau this builder was told about — `build_ribbon_across` takes
        // the cross-fall as an argument and not the whole options block, so a
        // direct caller grades over a plateau exactly as wide as its
        // carriageway. `build_surface` passes its own.
        graded_half_m: if crown_fall > 0.0 { half } else { 0.0 },
        ..SurfaceOptions::default()
    };

    for (i, frame) in frames.iter().enumerate() {
        for k in 0..=strips {
            // `u` runs 0 (left kerb) to 1 (right kerb) across the carriageway.
            let t = k as f64 / strips as f64;
            let offset = (t * 2.0 - 1.0) * half;
            let xz = frame.at(offset);
            // The terrain callback is somebody else's function reading somebody
            // else's heightfield; a query over a voxel hole or an unloaded tile
            // is exactly where a `Some(NaN)` comes from, and it would become a
            // NaN vertex with healthy-looking bounds. `carriageway_y` is the one
            // door every part of a road takes its height from — including this
            // one — so a crown authored here cannot disagree with the kerb
            // sitting on it.
            let ground_probe = height_at(xz.x, xz.y);
            if let Some(h) = ground_probe {
                if !h.is_finite() {
                    return Err(crate::GisError::NotFinite(format!(
                        "the ground query under the road at ({}, {}) returned the non-finite \
                         height {h}",
                        xz.x, xz.y
                    )));
                }
            }
            let y = carriageway_y(frame, offset, half, &opts, height_at);
            vertices.push(DVec3::new(xz.x, y, xz.y));
            // **uv IS METRES** (wave ROAD1): `u` is the offset across the
            // carriageway and `v` the arc along it, both in world metres, so a
            // material's `uv_tiling_m` is the whole tiling rule and the road no
            // longer tiles at its own width. Before this wave `v` was
            // `arc / width_m` and `u` was 0..1, which made one uv unit 14.0 m on
            // the island and the asphalt read three and a half times life size.
            uvs.push([offset as f32, frame.arc as f32]);
        }
        if i + 1 < frames.len() {
            let row = (strips + 1) as u32;
            let base = (i * (strips + 1)) as u32;
            for k in 0..strips as u32 {
                let a = base + k;
                // Wound so the surface faces up (+Y), matching the triangulator.
                // Identical to the two-vertex form for `strips == 1`.
                //
                // **The order flipped in wave ROAD1 and the geometry did not**:
                // the across parameter used to run to the road's LEFT (`u = 0`
                // was its right edge), and `CrossFrame::at` runs to its RIGHT,
                // because that is the sense `inf_nav::lane::right_of` uses and a
                // kerb offset had to mean the same thing in both crates. Same
                // triangles, mirrored index order, so the faces still point up —
                // and `a_ribbon_conforms_to_the_ground_and_tiles_by_arc_length`
                // is the arm that caught it pointing down.
                indices.extend_from_slice(&[a, a + row, a + row + 1, a, a + row + 1, a + 1]);
            }
        }
    }

    Ok(RoadRibbon {
        vertices,
        uvs,
        indices,
    })
}

/// Build every segment's ribbon, keyed by segment id, plus **what could not be
/// built and why**.
///
/// The first cut wrote `if let Ok(r) = …` and dropped the errors on the floor —
/// so a segment whose spine collapsed to one distinct point, or whose width came
/// out non-finite, simply vanished from the output with no count and no name.
/// This crate says three separate times that a skipped feature is a reported
/// feature ([`RoadGraph::skipped`], `GeoLayer::skipped`, the vector reader's own
/// doctrine); a silent hole in a road network is exactly the case that doctrine
/// is for, because the symptom is a street that stops in the middle of a block.
pub fn build_all_ribbons(
    graph: &RoadGraph,
    lift_m: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> (BTreeMap<u64, RoadRibbon>, Vec<String>) {
    let mut out = BTreeMap::new();
    let mut skipped = Vec::new();
    for s in graph.segments.values() {
        match build_ribbon(&s.spine, s.width_m(), lift_m, height_at) {
            Ok(r) => {
                out.insert(s.id, r);
            }
            Err(e) => skipped.push(format!(
                "segment {} ({}) has no buildable surface: {e}",
                s.id,
                if s.name.is_empty() {
                    "unnamed"
                } else {
                    &s.name
                }
            )),
        }
    }
    (out, skipped)
}

/// **The attribute spellings a lane count may be stored under**, most specific
/// first.
///
/// Hoisted to a constant in wave ROAD1 for the reason [`ROAD_CLASS_FIELDS`] was:
/// two readers of one fact have to read it through one door. `RoadGraph::from_
/// layer` probed this list inline, and `inf_island::layers::routes_of` — which
/// reads the same committed layer to plan a terrain corridor with — probed
/// nothing at all, so a layer stating `nlanes` would have built a four-lane road
/// through a two-lane corridor.
pub const ROAD_LANE_FIELDS: [&str; 5] = [
    "lane_count",
    "lanes",
    "nlanes",
    "num_lanes",
    "through_lanes",
];

/// The attribute spellings a road class may be stored under, most specific
/// first.
///
/// **One list, read from two places.** `RoadGraph::from_layer` probed ten
/// spellings and [`kind_of`] — the wizard's preview of the same decision —
/// probed six, so on any TIGER-style layer whose class lives in `RTTYP` or
/// `MTFCC` the preview said "residential" and the import built a highway. Two
/// readers of one fact have to read it through one door.
pub const ROAD_CLASS_FIELDS: [&str; 10] = [
    "road_type",
    "roadtype",
    "highway",
    "fclass",
    "type",
    "class",
    "rttyp",
    "mtfcc",
    "surface",
    "category",
];

/// Classify a feature's road kind without building a whole graph — the wizard's
/// preview. Reads [`ROAD_CLASS_FIELDS`], the same list the import uses.
pub fn kind_of(f: &GeoFeature) -> RoadKind {
    f.attr_text(&ROAD_CLASS_FIELDS)
        .map(RoadKind::classify)
        .unwrap_or_default()
}

// ── the road SURFACE: a whole network, draped, joined and merged ────────────

/// How far a road may run between ground queries before it stops following the
/// ground, in metres.
///
/// **This is the watertightness knob, and it is the terrain's number, not the
/// road's.** A published centreline has a vertex where the street bends, which
/// is nowhere near often enough to follow a heightfield: a 300 m straight has
/// two vertices, the ground under it has three hundred samples, and a ribbon
/// built on the centreline's own vertices spans every hill between them as a
/// chord — metres of daylight under the middle of the block. Resampling at the
/// terrain's own sample pitch makes the road's error the terrain's own
/// interpolation error and nothing more.
pub const DEFAULT_GROUND_STEP_M: f64 = 1.0;

/// How far a road surface floats above the ground it follows, in metres.
///
/// Two centimetres: enough that the two coplanar surfaces cannot z-fight at any
/// distance the depth buffer resolves, small enough that no wheel and no foot
/// notices a step. It is the only gap between a road and its terrain, and the
/// watertightness arm measures exactly it.
pub const DEFAULT_ROAD_LIFT_M: f64 = 0.02;

/// Resample a spine so no two consecutive positions are more than `step_m`
/// apart on the XZ plane.
///
/// Every original vertex is kept — a survey's vertices *are* the road (the same
/// reason the imported spline interpolates linearly) — and the inserted ones
/// divide each leg into equal parts, so the result is a pure function of the
/// input and the step.
pub fn densify_spine(spine: &[DVec3], step_m: f64) -> Vec<DVec3> {
    if spine.len() < 2 || !(step_m.is_finite() && step_m > 0.0) {
        return spine.to_vec();
    }
    let mut out: Vec<DVec3> = Vec::with_capacity(spine.len());
    out.push(spine[0]);
    for w in spine.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = (b.xz() - a.xz()).length();
        // `ceil` on the ratio, so the sub-step is at most `step_m` and never a
        // hair over it: 2.0001 m at a 1 m step becomes three legs of 0.667 m,
        // not two of 1.00005.
        let n = if len.is_finite() && len > step_m {
            (len / step_m).ceil() as usize
        } else {
            1
        };
        for i in 1..=n {
            let t = i as f64 / n as f64;
            out.push(a + (b - a) * t);
        }
    }
    out
}

/// A monotone stand-in for `atan2(d.y, d.x)`, in `[0, 4)`.
///
/// **Trig-free by law.** The P14 portability law bans `atan2` (and its whole
/// family) on anything that reaches committed content, and a junction fan's
/// vertex order decides its triangles, which reach a `.inf_mesh`. This is the
/// standard L1 pseudo-angle: it is not an angle, but it is *ordered like* one,
/// which is all a radial sort needs, and it is exact rational arithmetic.
fn pseudo_angle(d: glam::DVec2) -> f64 {
    let denom = d.x.abs() + d.y.abs();
    if !(denom.is_finite() && denom > 0.0) {
        return 0.0;
    }
    let p = d.x / denom;
    if d.y < 0.0 {
        3.0 + p
    } else {
        1.0 - p
    }
}

// ── wave ROAD1: the road is more than its carriageway ───────────────────────

/// **The kerb's height above the carriageway edge**, metres.
///
/// 150 mm is the standard upstand a highway authority specifies: high enough to
/// stop a wheel and to hold a gutter's water, low enough that a person steps up
/// it without thinking. Lower and it reads as a painted line from any distance;
/// higher and it is a wall.
pub const KERB_HEIGHT_M: f64 = 0.15;

/// **The width of a kerb stone's top**, metres — the flat between the road's
/// edge and the pavement's.
pub const KERB_WIDTH_M: f64 = 0.30;

/// **The pavement behind the kerb**, metres.
///
/// # One authority, two consumers (the ROAD1 one-door clause)
///
/// This is `inf_ecs::society::PAVEMENT_M` **by value**, and it has to be: that
/// constant lays the eight nav nodes of a city block's pavement ring, and this
/// lays the slab a person walks on. A crowd routed two metres out from a
/// building line while the concrete it walks on is one and a half is a crowd
/// walking beside its own pavement.
///
/// `inf-gis` cannot name `inf-ecs` (neither depends on the other, by design —
/// see `inf-ecs`'s own manifest note), so the equality is pinned where a crate
/// can see both: `the_kerb_geometry_and_the_nav_ring_are_one_pavement` in
/// `editor/crates/inf-editor-core/tests/road_authority.rs`. Same arrangement as
/// [`LANE_WIDTH_M`] and `inf_nav::lane::DEFAULT_LANE_WIDTH_M`, which are pinned
/// one file over for the same reason.
pub const PAVEMENT_M: f64 = 2.0;

/// **The carriageway's cross-fall**, as a fraction — 2 %, the figure every
/// highway manual specifies.
///
/// A road is not flat across: it is crowned so rain runs to the gutter rather
/// than standing in the wheel tracks. On a 14 m carriageway 2 % is **140 mm**
/// from crown to channel, which is the difference between a surface that reads
/// as a road and one that reads as a painted strip of ground.
///
/// It is a `SurfaceOptions` field rather than a constant applied everywhere, and
/// `0.0` — no crown, conform to the ground at every point — stays the default.
/// See [`SurfaceOptions::crown_fall`] for why that matters.
pub const DEFAULT_CROWN_FALL: f64 = 0.02;

/// **A painted line's width**, metres. 100 mm is the standard lane marking.
pub const LINE_WIDTH_M: f64 = 0.10;

/// The gap between the two lines of a **double** centre line, metres.
pub const DOUBLE_LINE_GAP_M: f64 = 0.10;

/// How far a marking floats above the carriageway it is painted on, metres.
///
/// **Four millimetres, and the number is doing real work.** Thermoplastic road
/// marking really is 3 mm proud, so this is not a z-fighting hack wearing a
/// physical name — but it also has to beat the depth buffer at the distance a
/// road is seen from, and 4 mm over a surface already lifted
/// [`DEFAULT_ROAD_LIFT_M`] off the terrain is what reverse-Z resolves at a
/// kilometre. Any smaller and the line dashes in and out as the camera moves.
pub const MARKING_LIFT_M: f64 = 0.004;

/// A dashed lane divider's painted length, metres.
pub const DASH_M: f64 = 3.0;

/// A dashed lane divider's gap, metres. 3 painted / 6 clear is the urban
/// pattern; a motorway's is longer, and one pattern is honest at this scale.
pub const DASH_GAP_M: f64 = 6.0;

/// How far the edge line sits inside the carriageway's edge, metres.
pub const EDGE_LINE_INSET_M: f64 = 0.20;

/// A crosswalk bar's width across the road, metres (the "continental" ladder
/// pattern every North American city paints).
pub const CROSSWALK_BAR_M: f64 = 0.50;

/// The gap between crosswalk bars, metres.
pub const CROSSWALK_GAP_M: f64 = 0.50;

/// How far a crosswalk's near edge sits from the junction node, metres.
///
/// Far enough out that the bars clear the fan that paves the intersection, close
/// enough that it reads as that junction's crossing and not as a mid-block one.
pub const CROSSWALK_SETBACK_M: f64 = 6.0;

/// How far a crosswalk runs along the road, metres — the depth a person crosses
/// through.
pub const CROSSWALK_DEPTH_M: f64 = 3.0;

/// **The half-width of everything a road of this class draws**, metres — the
/// carriageway, plus whichever of a sealed shoulder or a kerb-and-pavement it
/// carries (wave ROAD1).
///
/// No class carries both: [`RoadKind::is_kerbed`] and
/// [`RoadKind::shoulder_m`] are the two halves of one decision. This is the
/// number a terrain has to be levelled flat to under a **graded** road — see
/// [`SurfaceOptions::crown_fall`] — because a planar carriageway over ground
/// that is not planar under all of it puts a kerb in the air at one end of the
/// section and a pavement in a hillside at the other.
pub fn built_half_width_m(kind: RoadKind, lanes: u32) -> f64 {
    let half = kind.width_m(lanes) * 0.5;
    if kind.is_kerbed() {
        half + KERB_WIDTH_M + PAVEMENT_M
    } else {
        half + kind.shoulder_m()
    }
}

/// **A road surface's material groups** — one mesh, and one entity, per group.
///
/// # Why this is an enum and not four fields
///
/// Because an `inf_ecs::Material` component binds **one** `.inf_mat`, so a road
/// that wears asphalt, concrete, white paint and yellow paint is four entities
/// however the geometry is stored. Naming the groups makes the split a fact
/// about the road rather than a convention four call sites have to remember,
/// and it is what lets one loop write all four meshes.
///
/// [`Carriageway`](RoadPart::Carriageway) is the surface [`RoadSurface::parts`]
/// holds — one submesh per [`RoadKind`] — and the other three come out of
/// [`RoadSurface::furniture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoadPart {
    /// The drivable surface and the junction fans that join it: asphalt.
    Carriageway,
    /// Kerb stones and the pavement slab behind them: concrete.
    Kerb,
    /// Edge lines, lane dashes and crosswalks: white paint.
    MarkingWhite,
    /// The centre line that separates opposing traffic: yellow paint.
    MarkingYellow,
}

/// The furniture groups, in the frozen order the island writes their meshes in.
pub const FURNITURE_PARTS: [RoadPart; 3] = [
    RoadPart::Kerb,
    RoadPart::MarkingWhite,
    RoadPart::MarkingYellow,
];

impl RoadPart {
    /// The stem an asset for this part is written under, and the submesh name.
    pub const fn label(self) -> &'static str {
        match self {
            RoadPart::Carriageway => "carriageway",
            RoadPart::Kerb => "kerb",
            RoadPart::MarkingWhite => "marking white",
            RoadPart::MarkingYellow => "marking yellow",
        }
    }
}

/// **One cross-section station along a road**: where the centreline is, which
/// way is across it, how far along it is, and how much the corner mitre widens
/// it.
///
/// # Why this is extracted rather than recomputed
///
/// [`build_ribbon_across`] worked the mitre out inline, and until this wave it
/// was the only thing that needed it. Now the kerb, the pavement, the edge
/// lines, the centre line and the lane dashes all have to sit at a stated offset
/// from the same centreline **through the same corners** — and a kerb that
/// mitred differently from the carriageway it bounds would open a wedge at every
/// bend. One frame list, computed once, read by six builders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossFrame {
    /// The spine position. `y` is the *authored* elevation, not the ground.
    pub centre: DVec3,
    /// Unit vector across the road, on the XZ plane, pointing to its **left**.
    pub perp: glam::DVec2,
    /// The mitre's widening factor — see [`build_ribbon_across`]'s note on why
    /// a mitre must be scaled and not merely aimed. `1.0` on a straight run.
    pub miter: f64,
    /// Distance along the centreline from the segment's start, metres.
    pub arc: f64,
}

impl CrossFrame {
    /// The world XZ position `offset_m` to the road's **right** of the
    /// centreline (negative is left), with the corner mitre applied.
    #[inline]
    pub fn at(&self, offset_m: f64) -> glam::DVec2 {
        self.centre.xz() - self.perp * (offset_m * self.miter)
    }
}

/// **The cross-section stations of a densified spine.**
///
/// The frames are a pure function of the positions: no ground query, no options.
/// A spine with fewer than two distinct XZ positions yields an empty list rather
/// than a frame whose perpendicular is a normalized zero.
pub fn cross_frames(pts: &[DVec3]) -> Vec<CrossFrame> {
    let mut out = Vec::with_capacity(pts.len());
    let mut arc = 0.0f64;
    for i in 0..pts.len() {
        if i > 0 {
            arc += (pts[i].xz() - pts[i - 1].xz()).length();
        }
        let back_dir = if i > 0 {
            (pts[i].xz() - pts[i - 1].xz()).normalize_or_zero()
        } else {
            glam::DVec2::ZERO
        };
        let fwd_dir = if i + 1 < pts.len() {
            (pts[i + 1].xz() - pts[i].xz()).normalize_or_zero()
        } else {
            glam::DVec2::ZERO
        };
        let dir = {
            let d = back_dir + fwd_dir;
            if d.length_squared() > 1e-12 {
                d.normalize()
            } else if fwd_dir.length_squared() > 1e-12 {
                fwd_dir
            } else if back_dir.length_squared() > 1e-12 {
                back_dir
            } else {
                glam::DVec2::X
            }
        };
        // Perpendicular on the XZ plane. Rotating (x, z) by 90 degrees gives
        // (-z, x); with +X east and +Z south that points to the road's left.
        let perp = glam::DVec2::new(-dir.y, dir.x);
        let miter = {
            let leg = if fwd_dir.length_squared() > 1e-12 {
                fwd_dir
            } else {
                back_dir
            };
            let c = dir.dot(leg).abs();
            if c > 1.0 / MITER_LIMIT {
                1.0 / c
            } else {
                MITER_LIMIT
            }
        };
        out.push(CrossFrame {
            centre: pts[i],
            perp,
            miter,
            arc,
        });
    }
    out
}

/// A pavement's cross-fall back toward the kerb, as a fraction — 2 %, the same
/// figure the carriageway's crown uses and for the same reason: water.
pub const PAVEMENT_FALL: f64 = 0.02;

/// A shoulder's cross-fall where the carriageway states none, as a fraction.
/// 4 % is the usual figure — a shoulder falls harder than a running lane so it
/// sheds the water the lane gave it.
pub const SHOULDER_FALL: f64 = 0.04;

/// The ceiling on dashes one segment may paint.
///
/// Not a tuning knob: it is the guard that stops a period that somehow came out
/// zero from spinning over a county's worth of segments. 20 000 dashes at the
/// 9 m period is 180 km, which is longer than any digitised segment.
const MAX_DASHES_PER_SEGMENT: usize = 20_000;

/// The ceiling on bars one crossing may paint — a 60 m carriageway's worth at
/// the 1 m period, which is past every real road.
const MAX_CROSSWALK_BARS: usize = 64;

/// A drivable road surface: merged triangles per road class, plus the junction
/// fans that close the holes between them.
///
/// One [`RoadRibbon`] per [`RoadKind`] rather than one per segment, because a
/// mesh asset per segment is ten thousand files for a county and the classes are
/// exactly the material split a road surface wants.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadSurface {
    /// The merged geometry, keyed by class. `BTreeMap`, so the submesh order in
    /// the asset is a function of the data.
    pub parts: BTreeMap<RoadKind, RoadRibbon>,
    /// **Everything beside the carriageway** (wave ROAD1), keyed by the material
    /// group it wears — kerbs and pavements in concrete, edge lines and lane
    /// dashes and crossings in white, the centre line in yellow.
    ///
    /// It is a second map rather than more entries in [`parts`](Self::parts)
    /// because the two are keyed by different things: `parts` is keyed by *road
    /// class* and its whole map is one material, while this is keyed by
    /// *material* and does not care what class a piece of kerb came from. A
    /// single map would have to be keyed by a pair, and every consumer would
    /// then have to know that half the key is ignored.
    ///
    /// Empty unless [`SurfaceOptions::furniture`] asked for it.
    pub furniture: BTreeMap<RoadPart, RoadRibbon>,
    /// How many junctions got a fan.
    pub junctions_filled: usize,
    /// Junctions that could not be filled, and why.
    pub junctions_skipped: usize,
    /// **Footway triangles dropped because they lay on a carriageway** (audit
    /// ROAD1) — see [`clip_kerbs_to_open_ground`]. Reported rather than silent,
    /// because the number is a measure of how much of the road network runs over
    /// itself, which is a routing fact nothing in this module can fix.
    pub kerbs_clipped: usize,
    /// Segments with no buildable surface, named.
    pub skipped: Vec<String>,
}

impl RoadSurface {
    /// Triangles in the **carriageway** — the drivable surface and its junction
    /// fans. Deliberately not the furniture's: this number is what
    /// [`surface_to_mesh`] produces, and several arms read it as that.
    pub fn triangle_count(&self) -> usize {
        self.parts.values().map(RoadRibbon::triangle_count).sum()
    }
    /// Vertices in the **carriageway** — see [`triangle_count`](Self::triangle_count).
    pub fn vertex_count(&self) -> usize {
        self.parts.values().map(|r| r.vertices.len()).sum()
    }
    /// Triangles in the road furniture (wave ROAD1), all groups together.
    pub fn furniture_triangle_count(&self) -> usize {
        self.furniture
            .values()
            .map(RoadRibbon::triangle_count)
            .sum()
    }
    /// Vertices in the road furniture (wave ROAD1), all groups together.
    pub fn furniture_vertex_count(&self) -> usize {
        self.furniture.values().map(|r| r.vertices.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.parts.values().all(|r| r.indices.is_empty())
    }
}

/// Options for [`build_surface`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceOptions {
    /// See [`DEFAULT_ROAD_LIFT_M`].
    pub lift_m: f64,
    /// See [`DEFAULT_GROUND_STEP_M`].
    pub ground_step_m: f64,
    /// Fill junctions of degree 3 and above with a fan. A degree-2 node is a
    /// bend and the ribbon's own mitre already closes it.
    pub fill_junctions: bool,
    /// **The carriageway's cross-fall**, as a fraction — see
    /// [`DEFAULT_CROWN_FALL`] (wave ROAD1).
    ///
    /// # Zero is the default, and it is not a cop-out
    ///
    /// `0.0` means *conform*: the ground is sampled at every cross-section
    /// point and the surface follows it, which is the pre-ROAD1 behaviour
    /// byte for byte and the only honest thing an **importer** can do. A road
    /// dropped onto somebody else's terrain has no right to a design surface;
    /// grading it planar would leave its edges floating over the hillside it
    /// crosses by exactly the terrain's own cross-slope.
    ///
    /// Above zero the section is **graded**: one ground sample, on the
    /// centreline, and a plane rising to the crown. That is what a built road
    /// is, and it is only truthful where the caller **owns the terrain and has
    /// levelled a corridor under it** — which the island recipe does, at
    /// `inf_island::terrain`'s corridor plateau, and which the editor's GIS
    /// import wizard cannot. So the island states 2 % and the wizard states
    /// nothing.
    ///
    /// It also buys the road-vs-drawn-terrain conformance the wave was called
    /// for: a locally planar corridor decimates to itself, so the clipmap's LOD
    /// morph moves the ground under a graded road by nothing.
    pub crown_fall: f64,
    /// **How far from the centreline the caller has levelled a plateau**,
    /// metres — the half-width over which grading is legal (wave ROAD1).
    ///
    /// # The corner that made this a field
    ///
    /// A graded section takes ONE ground sample, on the centreline, and that is
    /// only truthful where the ground under the whole section is planar. It is,
    /// inside the corridor plateau — and a **mitred corner is not inside it**:
    /// `MITER_LIMIT` lets a cross-section reach four half-widths from the
    /// centreline at a hairpin, which on the island's own fixture put ribbon
    /// vertices **14 m out** and floating **1.56 m** over the batter they landed
    /// on. Measured; it is what `road1_gate`'s first run reported.
    ///
    /// So past this distance the section **conforms** again, eased in over half
    /// a plateau so the transition is not a crease. `0.0` — the default — means
    /// "no plateau", and with `crown_fall` at zero beside it that is the
    /// pre-ROAD1 conforming road exactly.
    ///
    /// It is the *same number* the terrain was levelled with, and it has to be:
    /// two answers to "how wide is the flat" is a road graded over ground that
    /// is not.
    pub graded_half_m: f64,
    /// Build kerbs, pavements, shoulders, markings and crossings beside the
    /// carriageway (wave ROAD1). `false` reproduces the pre-ROAD1 surface.
    pub furniture: bool,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self {
            lift_m: DEFAULT_ROAD_LIFT_M,
            ground_step_m: DEFAULT_GROUND_STEP_M,
            fill_junctions: true,
            // **Both default to the pre-ROAD1 road**, so every caller that has
            // not thought about a design surface gets the conforming one it
            // already had, and the two callers that have — the island's recipe
            // and this crate's own arms — say so out loud.
            crown_fall: 0.0,
            graded_half_m: 0.0,
            furniture: false,
        }
    }
}

/// How many quads a carriageway of `width_m` needs across it at `step_m`.
///
/// The same rule as the spine's resampling, applied to the other axis, and
/// capped at [`MAX_CROSS_STRIPS`] so a 60 m motorway interchange at a 10 cm step
/// cannot ask for six hundred columns of vertices.
pub fn cross_strips(width_m: f64, step_m: f64) -> usize {
    if !(width_m.is_finite() && width_m > 0.0 && step_m.is_finite() && step_m > 0.0) {
        return 1;
    }
    ((width_m / step_m).ceil() as usize).clamp(1, MAX_CROSS_STRIPS)
}

/// The ceiling on cross-section subdivision. A road wider than
/// `MAX_CROSS_STRIPS × step` conforms to the ground at a coarser pitch across
/// than along — reported by `MeshBuildReport`'s triangle count rather than
/// hidden, and 32 is past every real carriageway at every sane step.
pub const MAX_CROSS_STRIPS: usize = 32;

/// Append `src` into `dst`, rebasing the indices.
fn append_ribbon(dst: &mut RoadRibbon, src: &RoadRibbon) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.uvs.extend_from_slice(&src.uvs);
    dst.indices.extend(src.indices.iter().map(|i| i + base));
}

/// One point of a road's cross-section: how far to the right of the centreline
/// it is, and how high.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ProfilePoint {
    offset_m: f64,
    y: f64,
}

/// **Emit a quad strip along a run of frames, through a cross-section profile.**
///
/// The profile's points are in **increasing offset order** on both sides of the
/// road, which is the whole reason the left kerb is authored outside-in: the
/// winding below is the carriageway's own, and a profile that ran the other way
/// would face its triangles into the ground.
///
/// Consecutive points at the same offset make a **vertical** face — that is how
/// a kerb's upstand and a pavement's outer skirt are built, and it is why this
/// takes a profile rather than a left and a right offset.
///
/// `uv` is `(distance ALONG the cross-section in metres, arc in metres)`, in
/// the same physical space the carriageway's is, so every part of a road tiles
/// at one rate and a material's `uv_tiling_m` means the same thing on all of
/// them.
///
/// # It is the cross-section's arc length, not the offset, and a kerb is why
///
/// A horizontal run has `u == offset` and the two are the same number — which
/// is what makes the kerb's uv continuous with the carriageway's beside it. A
/// **vertical** face does not: the kerb's 150 mm upstand and the pavement's
/// outer skirt have the same offset at both ends, so a uv taken from the offset
/// alone is *constant across the face* and the texture stretches one texel over
/// the whole of it. Measured as a defect before it was one: the kerb face is
/// 150 mm of a 2 m concrete tile and would have drawn as a smear.
fn emit_profile_strip(
    out: &mut RoadRibbon,
    frames: &[CrossFrame],
    profile: &mut dyn FnMut(&CrossFrame) -> Vec<ProfilePoint>,
) {
    if frames.len() < 2 {
        return;
    }
    let base = out.vertices.len() as u32;
    let mut row_len = 0usize;
    for (i, f) in frames.iter().enumerate() {
        let pts = profile(f);
        if i == 0 {
            row_len = pts.len();
            if row_len < 2 {
                return;
            }
        } else if pts.len() != row_len {
            // A profile whose point count varies along the run cannot be
            // stripped. Refusing is right: the alternative is an index buffer
            // that walks off a row.
            return;
        }
        // The cross-section's own arc length, anchored on its first point so a
        // horizontal profile's `u` IS its offset — see the note above.
        let mut u = pts[0].offset_m;
        for (k, p) in pts.iter().enumerate() {
            if k > 0 {
                let d_off = p.offset_m - pts[k - 1].offset_m;
                let d_y = p.y - pts[k - 1].y;
                u += (d_off * d_off + d_y * d_y).sqrt();
            }
            let xz = f.at(p.offset_m);
            out.vertices.push(DVec3::new(xz.x, p.y, xz.y));
            out.uvs.push([u as f32, f.arc as f32]);
        }
    }
    let row = row_len as u32;
    for i in 0..(frames.len() as u32 - 1) {
        for k in 0..(row - 1) {
            let a = base + i * row + k;
            // The carriageway's own winding — see `build_ribbon_across`. A
            // profile in increasing-offset order therefore faces up, which is
            // why the left kerb below is authored outside-in.
            out.indices
                .extend_from_slice(&[a, a + row, a + row + 1, a, a + row + 1, a + 1]);
        }
    }
}

/// The stations of `frames` whose arc lies in `[from, to]`, as a slice range.
///
/// Half-open at neither end: a crosswalk that lost its last station would be one
/// bar short of the kerb it runs to.
fn frames_between(frames: &[CrossFrame], from: f64, to: f64) -> &[CrossFrame] {
    let lo = frames.partition_point(|f| f.arc < from);
    let hi = frames.partition_point(|f| f.arc <= to);
    let lo = lo.min(frames.len());
    let hi = hi.min(frames.len());
    if hi > lo {
        &frames[lo..hi]
    } else {
        &[]
    }
}

/// **The carriageway's finished surface height** at `offset_m` from the crown.
///
/// # The one function every part of a road takes its height from
///
/// The kerb sits on the channel, the markings sit on the wearing course, and the
/// crosswalk sits on both — so all of them ask this, and none of them re-derives
/// a crown. `crown_fall == 0.0` reproduces the pre-ROAD1 arithmetic exactly:
/// the ground is sampled **at the point**, and the crown term is zero.
///
/// Above zero, the section is **graded**: the ground is sampled once, on the
/// centreline, and the cross-section is a plane rising to the crown. That is
/// what a built road is — cut and filled to a design surface — and it is only
/// honest where the caller owns the terrain and has levelled a corridor under
/// it. `SurfaceOptions::crown_fall`'s own note is the record of that condition.
fn carriageway_y(
    frame: &CrossFrame,
    offset_m: f64,
    half_m: f64,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> f64 {
    // **The distance from the centreline is the MITRED one**, not the offset:
    // at a corner the cross-section is scaled out by up to `MITER_LIMIT`, and it
    // is the real distance that decides whether this vertex is still over the
    // plateau. Reading the offset instead is what put a hairpin's rim 14 m out
    // and 1.56 m in the air.
    let lateral = (offset_m * frame.miter).abs();
    // How much this vertex conforms rather than grades: 0 over the plateau,
    // 1 past it, eased over half a plateau so the hand-off is not a crease.
    let conform = if opts.crown_fall > 0.0 && opts.graded_half_m > 0.0 {
        let ease = opts.graded_half_m * GRADED_EASE;
        smooth01(((lateral - opts.graded_half_m) / ease.max(1e-9)).clamp(0.0, 1.0))
    } else {
        // No plateau: the pre-ROAD1 road, which conforms at every point.
        1.0
    };
    let mut ground_at = |off: f64| -> f64 {
        let xz = frame.at(off);
        match height_at(xz.x, xz.y) {
            Some(h) if h.is_finite() => h,
            _ => frame.centre.y,
        }
    };
    let local = ground_at(offset_m);
    let ground = if conform >= 1.0 {
        local
    } else {
        let graded = ground_at(0.0);
        let blended = graded + (local - graded) * conform;
        // **A graded section may not leave the ground it covers by more than
        // its own crown**, and this clamp is what makes that a bound rather
        // than a hope.
        //
        // The plateau is levelled to the NEAREST route, and a grade-limited
        // router builds switchbacks — two limbs of one road passing within a
        // few metres of each other at different heights. There the plateau can
        // only serve one of them, and the other's graded height is the wrong
        // side of a 1.78 m step (measured, on the island fixture, at
        // (-378.3, 318.1)). Unclamped, that limb's carriageway hangs in the air.
        //
        // Clamped, the worst a road can do is sit its own crown above the
        // ground or the same distance below it, and where the plateau IS flat —
        // every straight and every gentle bend, which is nearly all of it — the
        // clamp is inactive and the section is exactly planar.
        let slack = opts.crown_fall * half_m;
        blended.clamp(local - slack, local + slack)
    };
    ground + opts.lift_m + opts.crown_fall * (half_m - offset_m.abs()).max(0.0)
}

/// How far past [`SurfaceOptions::graded_half_m`] a section eases from graded
/// back to conforming, as a fraction of the plateau's own half-width.
const GRADED_EASE: f64 = 0.5;

/// The smoothstep this module eases with — `t·t·(3 − 2t)`, trig-free.
#[inline]
fn smooth01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The ground under a cross-section offset, falling back to the spine's own
/// authored elevation where nothing answers.
fn ground_at_offset(
    frame: &CrossFrame,
    offset_m: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> f64 {
    let xz = frame.at(offset_m);
    match height_at(xz.x, xz.y) {
        Some(h) if h.is_finite() => h,
        _ => frame.centre.y,
    }
}

/// **The foot of the pavement's outer skirt**, never above the slab it hangs
/// from.
///
/// The skirt exists to close the 190 mm cliff between a footway and the ground
/// behind it, and it does that by dropping a vertical face to the terrain. Where
/// the road is in a **cutting** the terrain behind the footway is *higher* than
/// the slab, and an unguarded skirt turns inside out: the face rises above the
/// pavement and its triangles wind the other way, so it draws as a black wall
/// standing on the footway. Clamped, it degenerates to zero height instead —
/// which is the right answer, because a cut slope already covers what the skirt
/// was there to hide.
fn skirt_foot(
    frame: &CrossFrame,
    offset_m: f64,
    slab_y: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> f64 {
    ground_at_offset(frame, offset_m, height_at).min(slab_y)
}

/// **Kerb stones and the pavement slab behind them**, both sides.
///
/// The profile, outward from the channel: the kerb's 150 mm upstand, its 300 mm
/// top, the [`PAVEMENT_M`] slab falling 2 % back toward the kerb the way a real
/// footway drains, and a vertical skirt from the slab's back edge down to the
/// ground — without which you see straight through a 190 mm cliff of nothing at
/// every property line.
fn build_kerbs(
    out: &mut RoadRibbon,
    frames: &[CrossFrame],
    half_m: f64,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    let back = half_m + KERB_WIDTH_M + PAVEMENT_M;
    // The right-hand kerb: offsets increase outward, so the profile is already
    // in the order `emit_profile_strip` winds.
    emit_profile_strip(out, frames, &mut |f| {
        let channel = carriageway_y(f, half_m, half_m, opts, height_at);
        let top = channel + KERB_HEIGHT_M;
        vec![
            ProfilePoint {
                offset_m: half_m,
                y: channel,
            },
            ProfilePoint {
                offset_m: half_m,
                y: top,
            },
            ProfilePoint {
                offset_m: half_m + KERB_WIDTH_M,
                y: top,
            },
            ProfilePoint {
                offset_m: back,
                y: top + PAVEMENT_M * PAVEMENT_FALL,
            },
            ProfilePoint {
                offset_m: back,
                y: skirt_foot(f, back, top + PAVEMENT_M * PAVEMENT_FALL, height_at),
            },
        ]
    });
    // The left-hand kerb, authored **outside-in** so its offsets also increase.
    emit_profile_strip(out, frames, &mut |f| {
        let channel = carriageway_y(f, -half_m, half_m, opts, height_at);
        let top = channel + KERB_HEIGHT_M;
        vec![
            ProfilePoint {
                offset_m: -back,
                y: skirt_foot(f, -back, top + PAVEMENT_M * PAVEMENT_FALL, height_at),
            },
            ProfilePoint {
                offset_m: -back,
                y: top + PAVEMENT_M * PAVEMENT_FALL,
            },
            ProfilePoint {
                offset_m: -half_m - KERB_WIDTH_M,
                y: top,
            },
            ProfilePoint {
                offset_m: -half_m,
                y: top,
            },
            ProfilePoint {
                offset_m: -half_m,
                y: channel,
            },
        ]
    });
}

/// **A paved shoulder**, both sides — what an open road has where a street has a
/// kerb.
///
/// It is drawn in the carriageway's own ribbon and therefore in asphalt, because
/// that is what a sealed shoulder is; what separates it from a running lane is
/// the solid white edge line, which [`build_markings`] paints at the
/// carriageway's edge and not at the shoulder's.
fn build_shoulders(
    out: &mut RoadRibbon,
    frames: &[CrossFrame],
    half_m: f64,
    shoulder_m: f64,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    if shoulder_m <= 0.0 {
        return;
    }
    for side in [1.0f64, -1.0] {
        emit_profile_strip(out, frames, &mut |f| {
            let edge = carriageway_y(f, side * half_m, half_m, opts, height_at);
            // The shoulder continues the carriageway's cross-fall, so water that
            // ran off the crown keeps running.
            let outer = edge - shoulder_m * opts.crown_fall.max(SHOULDER_FALL);
            let (a, b) = if side > 0.0 {
                (
                    ProfilePoint {
                        offset_m: half_m,
                        y: edge,
                    },
                    ProfilePoint {
                        offset_m: half_m + shoulder_m,
                        y: outer,
                    },
                )
            } else {
                (
                    ProfilePoint {
                        offset_m: -half_m - shoulder_m,
                        y: outer,
                    },
                    ProfilePoint {
                        offset_m: -half_m,
                        y: edge,
                    },
                )
            };
            vec![a, b]
        });
    }
}

/// Paint one longitudinal line of `width_m` centred at `offset_m`, over the
/// stations `frames`.
fn paint_line(
    out: &mut RoadRibbon,
    frames: &[CrossFrame],
    offset_m: f64,
    width_m: f64,
    half_m: f64,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    let (a, b) = (offset_m - width_m * 0.5, offset_m + width_m * 0.5);
    emit_profile_strip(out, frames, &mut |f| {
        vec![
            ProfilePoint {
                offset_m: a,
                y: carriageway_y(f, a, half_m, opts, height_at) + MARKING_LIFT_M,
            },
            ProfilePoint {
                offset_m: b,
                y: carriageway_y(f, b, half_m, opts, height_at) + MARKING_LIFT_M,
            },
        ]
    });
}

/// Paint a **dashed** longitudinal line: [`DASH_M`] painted, [`DASH_GAP_M`]
/// clear, on a lattice measured from the segment's own start.
fn paint_dashes(
    out: &mut RoadRibbon,
    frames: &[CrossFrame],
    offset_m: f64,
    half_m: f64,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    let period = DASH_M + DASH_GAP_M;
    let total = frames.last().map(|f| f.arc).unwrap_or(0.0);
    let mut start = 0.0f64;
    // A guard on the count as well as on the arc: a period that somehow came out
    // zero would spin here, and this runs over a county's worth of segments.
    let mut guard = 0usize;
    while start < total && guard < MAX_DASHES_PER_SEGMENT {
        let run = frames_between(frames, start, (start + DASH_M).min(total));
        if run.len() >= 2 {
            paint_line(out, run, offset_m, LINE_WIDTH_M, half_m, opts, height_at);
        }
        start += period;
        guard += 1;
    }
}

/// **Every marking a road of this class carries**, split white from yellow.
///
/// | class | centre | lane dividers | edge lines |
/// |---|---|---|---|
/// | highway (4 lanes) | double **yellow** | white dashes at ±3.5 m | solid white |
/// | arterial (2 lanes) | single **yellow** | — | solid white |
/// | residential | — | — | — |
/// | dirt track, path, rail | — | — | — |
///
/// **Yellow separates opposing traffic and white does not** — the North American
/// convention, which is the one the island is in (British Columbia) and the one
/// `frames/driving/0006` shows: a double yellow down the middle of a four-lane
/// street and white at both edges. A residential street is deliberately
/// **unmarked**, because a neighbourhood street in that country is.
fn build_markings(
    white: &mut RoadRibbon,
    yellow: &mut RoadRibbon,
    frames: &[CrossFrame],
    seg: &RoadSegment,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    if !seg.kind.is_marked() {
        return;
    }
    let half = seg.width_m() * 0.5;
    let lanes = seg.lane_count.max(1);

    // ── the centre line ──────────────────────────────────────────────────────
    if lanes >= 4 {
        // Double solid yellow: two lines, a line's width apart.
        let off = (DOUBLE_LINE_GAP_M + LINE_WIDTH_M) * 0.5;
        for s in [-1.0f64, 1.0] {
            paint_line(yellow, frames, s * off, LINE_WIDTH_M, half, opts, height_at);
        }
    } else if lanes >= 2 {
        paint_line(yellow, frames, 0.0, LINE_WIDTH_M, half, opts, height_at);
    }

    // ── the lane dividers, dashed white ──────────────────────────────────────
    // Lane `k` from the crown ends at `k · LANE_WIDTH_M`; the divider is that
    // line, and the outermost one is the carriageway edge rather than a marking.
    for k in 1..(lanes / 2) {
        let off = f64::from(k) * LANE_WIDTH_M;
        for s in [-1.0f64, 1.0] {
            paint_dashes(white, frames, s * off, half, opts, height_at);
        }
    }

    // ── the edge lines, solid white ──────────────────────────────────────────
    let edge = half - EDGE_LINE_INSET_M;
    if edge > LINE_WIDTH_M {
        for s in [-1.0f64, 1.0] {
            paint_line(white, frames, s * edge, LINE_WIDTH_M, half, opts, height_at);
        }
    }
}

/// **A pedestrian crossing across the carriageway**, in the continental ladder
/// pattern: bars along the direction of travel, spaced across the road.
///
/// `from_start` says which end of the segment the junction is at, because a
/// crossing belongs to a junction and sits [`CROSSWALK_SETBACK_M`] out from it —
/// far enough to clear the fan that paves the intersection.
fn build_crosswalk(
    white: &mut RoadRibbon,
    frames: &[CrossFrame],
    seg: &RoadSegment,
    from_start: bool,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    let total = frames.last().map(|f| f.arc).unwrap_or(0.0);
    // A crossing needs room between the junction and the far end of the block.
    if total < CROSSWALK_SETBACK_M + CROSSWALK_DEPTH_M {
        return;
    }
    let (from, to) = if from_start {
        (CROSSWALK_SETBACK_M, CROSSWALK_SETBACK_M + CROSSWALK_DEPTH_M)
    } else {
        (
            total - CROSSWALK_SETBACK_M - CROSSWALK_DEPTH_M,
            total - CROSSWALK_SETBACK_M,
        )
    };
    let run = frames_between(frames, from, to);
    if run.len() < 2 {
        return;
    }
    let half = seg.width_m() * 0.5;
    let period = CROSSWALK_BAR_M + CROSSWALK_GAP_M;
    let mut off = -half + CROSSWALK_GAP_M * 0.5;
    let mut guard = 0usize;
    while off + CROSSWALK_BAR_M <= half && guard < MAX_CROSSWALK_BARS {
        paint_line(
            white,
            run,
            off + CROSSWALK_BAR_M * 0.5,
            CROSSWALK_BAR_M,
            half,
            opts,
            height_at,
        );
        off += period;
        guard += 1;
    }
}

/// **Build a whole road network's surface**: drape every segment on the ground,
/// merge by class, and fan the junctions.
///
/// `height_at` is the engine's own ground query — `inf_voxel::ground_height_at`
/// at the call site, which since IB-15 is the *topmost terrain that answers*, so
/// a road crossing from one terrain onto another follows both. `None` means
/// nothing answers there, and the spine's own elevation is used, which is the
/// published centreline's best guess.
///
/// # What "watertight with the terrain" means here, exactly
///
/// Every ribbon vertex is placed at `ground(x, z) + lift_m`, at both edges of
/// the road, at a spacing of `ground_step_m`. So the road touches the terrain to
/// within `lift_m` at every one of its own vertices by construction, and between
/// them it differs from the terrain by the terrain's own chord error over one
/// sample — which is why the step is the terrain's pitch and not the road's.
/// `roads_follow_the_ground_they_are_draped_on` measures both.
pub fn build_surface(
    graph: &RoadGraph,
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> RoadSurface {
    let mut out = RoadSurface::default();
    // node id -> the cross-section corners of each incident segment end, as
    // (left, right) world positions. A `BTreeMap` of `Vec`s, filled in segment
    // id order, so a fan's vertex set is a function of the graph.
    let mut ends: BTreeMap<u64, Vec<(DVec3, DVec3)>> = BTreeMap::new();

    for s in graph.segments.values() {
        let dense = densify_spine(&s.spine, opts.ground_step_m);
        // The cross-section is subdivided at the same pitch as the spine — see
        // `build_ribbon_across` for the 49 mm this closes.
        let strips = cross_strips(s.width_m(), opts.ground_step_m);
        match build_ribbon_across(
            &dense,
            s.width_m(),
            opts.lift_m,
            strips,
            opts.crown_fall,
            height_at,
        ) {
            Ok(r) => {
                let row = strips + 1;
                if r.vertices.len() >= 2 * row {
                    let n = r.vertices.len();
                    ends.entry(s.start_node)
                        .or_default()
                        .push((r.vertices[0], r.vertices[row - 1]));
                    ends.entry(s.end_node)
                        .or_default()
                        .push((r.vertices[n - row], r.vertices[n - 1]));
                }
                append_ribbon(out.parts.entry(s.kind).or_default(), &r);
                if opts.furniture {
                    build_segment_furniture(&mut out, graph, s, &dense, opts, height_at);
                }
            }
            Err(e) => out.skipped.push(format!(
                "segment {} ({}) has no buildable surface: {e}",
                s.id,
                if s.name.is_empty() {
                    "unnamed"
                } else {
                    &s.name
                }
            )),
        }
    }

    if opts.fill_junctions {
        for (node, corners) in &ends {
            let Some(inter) = graph.intersections.get(node) else {
                continue;
            };
            // A degree-2 node is a bend, and `build_ribbon`'s mitre already
            // closes it exactly. A degree-1 node is a dead end with nothing on
            // the far side to close against.
            if inter.degree() < 3 || corners.len() < 3 {
                continue;
            }
            // The fan's class is the highest-order road at the junction — a
            // motorway crossing a residential street paves the intersection in
            // motorway, which is what the ground actually looks like. `RoadKind`
            // ordering is declaration order, most important first.
            let kind = inter
                .segments
                .iter()
                .filter_map(|id| graph.segments.get(id).map(|s| s.kind))
                .min()
                .unwrap_or_default();
            match fan_at(corners, opts.lift_m, height_at) {
                Some(fan) => {
                    append_ribbon(out.parts.entry(kind).or_default(), &fan);
                    out.junctions_filled += 1;
                }
                None => out.junctions_skipped += 1,
            }
        }
    }
    if opts.furniture {
        out.kerbs_clipped = clip_kerbs_to_open_ground(&mut out);
    }
    out
}

/// **How far inside a carriageway a footway triangle has to sit before it is
/// dropped**, metres (audit ROAD1).
///
/// It is not zero, and the reason is the profile: `build_kerbs`' first two
/// points are both at `offset = half`, so the kerb's own vertical face has a
/// centroid *exactly on* the carriageway's edge. A zero inset would delete every
/// kerb face on the island. 100 mm is a third of a kerb stone — far inside the
/// resolution of "is this concrete beside the road or on it", and far outside
/// the f64 noise of a boundary that two builders compute from one frame list.
///
/// # It is an inset into the UNION, and that distinction cost a debugging pass
///
/// The first version of this clip applied the inset **per triangle**, and a road
/// mesh is a lattice of them: a point 24 mm from the internal diagonal two
/// carriageway triangles share is a tenth of a metre inside *neither*, so the
/// test leaked along every shared edge and left a grid of 200 mm-wide strips of
/// pavement standing on the road. Measured on the two-crossing-roads fixture
/// below, which is exactly why that fixture exists. The inset is therefore
/// sampled as a small cross about the centroid against the union — see
/// [`is_covered_by`].
const KERB_CLIP_INSET_M: f64 = 0.10;

/// The cell of the carriageway lookup grid the clip walks, metres.
///
/// Four metres is a little over one cross-section step, so a road triangle lands
/// in one or two cells and a cell holds a couple of dozen. Smaller costs more
/// cells for the same work; larger makes every lookup walk the whole road.
const CLIP_GRID_M: f64 = 4.0;

/// The most grid cells one carriageway triangle may claim.
///
/// A mitred corner can stretch a triangle to four half-widths, and a guard is
/// cheaper than the pathological case: past this the triangle is left out of the
/// index rather than allowed to fill it. 256 cells is a 64 m square, past any
/// road triangle this builder makes.
const MAX_CLIP_CELLS_PER_TRIANGLE: i64 = 256;

/// Whether `p` lies inside XZ triangle `t`, edges included.
///
/// The three edge cross products must agree in sign — the standard test — with a
/// zero treated as agreeing with either, because a strip's triangles meet on the
/// cross-section lines and a probe that lands on one is on *both* of them and
/// must not fall between two `false`s.
///
/// No transcendental and no division: the P14 portability law reaches here,
/// because what this decides is which triangles a committed `.inf_mesh`
/// contains.
fn plan_contains(t: &[glam::DVec2; 3], p: glam::DVec2) -> bool {
    let mut sign = 0.0f64;
    for k in 0..3 {
        let a = t[k];
        let e = t[(k + 1) % 3] - a;
        if !(e.length_squared() > 1.0e-24) {
            // A degenerate edge means a triangle with no plan area — a kerb
            // face or a skirt seen from above. It covers nothing.
            return false;
        }
        let cross = e.x * (p.y - a.y) - e.y * (p.x - a.x);
        let s = if cross > 0.0 {
            1.0
        } else if cross < 0.0 {
            -1.0
        } else {
            0.0
        };
        if s != 0.0 {
            if sign == 0.0 {
                sign = s;
            } else if s != sign {
                return false;
            }
        }
    }
    true
}

/// **Is `p` at least `inset_m` inside the union covered by `tris`?**
///
/// Sampled as a five-point cross rather than as a per-triangle inset, for the
/// reason [`KERB_CLIP_INSET_M`] records: an inset applied to one triangle leaks
/// along every internal edge a mesh's triangles share, and a road surface is
/// nothing but internal edges. A cross asks the *union* the question instead —
/// the centre and four points `inset_m` away — so a shared diagonal is covered
/// by the neighbour and only a real boundary answers no.
///
/// `candidates` is a closure so the caller can hand it a grid bucket per probe:
/// the four arms can land in a different cell from the centre.
fn is_covered_by(
    p: glam::DVec2,
    inset_m: f64,
    mut candidates: impl FnMut(glam::DVec2) -> bool,
) -> bool {
    [
        p,
        glam::DVec2::new(p.x + inset_m, p.y),
        glam::DVec2::new(p.x - inset_m, p.y),
        glam::DVec2::new(p.x, p.y + inset_m),
        glam::DVec2::new(p.x, p.y - inset_m),
    ]
    .into_iter()
    .all(&mut candidates)
}

/// **A footway is not drawn on a road** (audit ROAD1) — drop every kerb and
/// pavement triangle that lies on a carriageway, and say how many.
///
/// # What this is for, measured
///
/// `build_segment_furniture` lays a kerb and 2 m of footway beside *its own*
/// segment and knows nothing about any other. Two roads that pass within
/// `built_half_width + half` of each other therefore each lay concrete over the
/// other's asphalt — and so does **one** road that folds back on itself, which a
/// grade-limited router does at every switchback. Before this pass, on the
/// island: **19 754.6 m² of the 170 901.7 m² footway (11.56 %) was drawn on top
/// of a carriageway**, floating the kerb's 190 mm above it, reaching within
/// 0.02 m of a 7 m road's own crown; on the fixture, 969.5 m² of 8 247.6 m²
/// (11.75 %), all of it in the switchback at (-400, 300). It is what a 1080p
/// capture of Harbour City's main street shows as a grey plank lying diagonally
/// across the road, and it is the defect this audit was pointed at.
///
/// # Why dropping triangles is the right answer and not a patch
///
/// Two roads occupying one piece of ground is a **routing** fact, carried since
/// this wave (a road crossing a stream crosses at grade; a switchback folds).
/// Nothing here can fix that. What it can decide is what gets drawn where they
/// overlap, and asphalt-under-nothing is strictly better than asphalt-under-
/// concrete: the carriageway is the surface a car drives on and a person crosses
/// at the painted crossing, so where a footway would cover it the footway is the
/// piece that is wrong. Where the overlap is a junction fan this is also what
/// the wave's own carried item 8 asked for — "kerbs stop at the rim" — made true
/// rather than described.
///
/// Markings are deliberately **not** clipped: a marking's whole job is to lie on
/// a carriageway, so the same test would delete all of them.
///
/// # Determinism
///
/// The index is a `BTreeMap` and the verdict is an `any` over its bucket, so the
/// output is a function of the input alone at every candidate order — and the
/// arithmetic is exact f64 with no transcendental in it (`is_inside_by`).
fn clip_kerbs_to_open_ground(out: &mut RoadSurface) -> usize {
    if out
        .furniture
        .get(&RoadPart::Kerb)
        .is_none_or(|r| r.indices.is_empty())
    {
        return 0;
    }
    // The carriageway — every class, plus the shoulders and the junction fans
    // that were appended into the same ribbons — as plan-view triangles.
    let mut tris: Vec<[glam::DVec2; 3]> = Vec::new();
    for r in out.parts.values() {
        for t in r.indices.chunks_exact(3) {
            tris.push([
                r.vertices[t[0] as usize].xz(),
                r.vertices[t[1] as usize].xz(),
                r.vertices[t[2] as usize].xz(),
            ]);
        }
    }
    if tris.is_empty() {
        return 0;
    }
    let mut grid: BTreeMap<(i64, i64), Vec<u32>> = BTreeMap::new();
    for (i, t) in tris.iter().enumerate() {
        let lo = t[0].min(t[1]).min(t[2]);
        let hi = t[0].max(t[1]).max(t[2]);
        if !(lo.is_finite() && hi.is_finite()) {
            continue;
        }
        let x0 = (lo.x / CLIP_GRID_M).floor() as i64;
        let x1 = (hi.x / CLIP_GRID_M).floor() as i64;
        let y0 = (lo.y / CLIP_GRID_M).floor() as i64;
        let y1 = (hi.y / CLIP_GRID_M).floor() as i64;
        if (x1 - x0 + 1).saturating_mul(y1 - y0 + 1) > MAX_CLIP_CELLS_PER_TRIANGLE {
            continue;
        }
        for gx in x0..=x1 {
            for gy in y0..=y1 {
                grid.entry((gx, gy)).or_default().push(i as u32);
            }
        }
    }

    let kerb = out
        .furniture
        .get_mut(&RoadPart::Kerb)
        .expect("checked above");
    let mut keep: Vec<u32> = Vec::with_capacity(kerb.indices.len());
    let mut dropped = 0usize;
    for t in kerb.indices.chunks_exact(3) {
        let c = (kerb.vertices[t[0] as usize]
            + kerb.vertices[t[1] as usize]
            + kerb.vertices[t[2] as usize])
            / 3.0;
        let p = c.xz();
        let on_road = is_covered_by(p, KERB_CLIP_INSET_M, |q| {
            let cell = (
                (q.x / CLIP_GRID_M).floor() as i64,
                (q.y / CLIP_GRID_M).floor() as i64,
            );
            grid.get(&cell)
                .is_some_and(|bucket| bucket.iter().any(|&i| plan_contains(&tris[i as usize], q)))
        });
        if on_road {
            dropped += 1;
        } else {
            keep.extend_from_slice(t);
        }
    }
    if dropped == 0 {
        return 0;
    }
    // Compact, because an orphaned vertex is a vertex a `.inf_mesh` still pays
    // for and a `MeshBuildReport` still counts.
    let mut remap = vec![u32::MAX; kerb.vertices.len()];
    let mut vertices = Vec::with_capacity(kerb.vertices.len());
    let mut uvs = Vec::with_capacity(kerb.uvs.len());
    for i in keep.iter_mut() {
        let old = *i as usize;
        if remap[old] == u32::MAX {
            remap[old] = vertices.len() as u32;
            vertices.push(kerb.vertices[old]);
            uvs.push(kerb.uvs[old]);
        }
        *i = remap[old];
    }
    kerb.vertices = vertices;
    kerb.uvs = uvs;
    kerb.indices = keep;
    dropped
}

// ── the surface becomes a mesh asset ────────────────────────────────────────

/// What building a mesh out of a road surface cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshBuildReport {
    pub vertices: usize,
    pub triangles: usize,
    /// The furthest any vertex sits from the mesh's local origin, in metres.
    pub max_offset_m: f64,
    /// The f32 spacing at that distance — the mesh's own positional resolution.
    ///
    /// A `MeshVertex` is `[f32; 3]` (it is uploaded to a GPU buffer verbatim),
    /// and this engine's world is f64 for exactly the reason that matters here:
    /// a UTM easting is ~5×10⁵ and an f32 resolves it to 30 mm. Centring the
    /// mesh on its own content makes the offset the *extent*, not the
    /// coordinate — 25 km at island scale, where the spacing is ~3 mm — and
    /// reporting it means nobody has to guess.
    pub quantisation_m: f64,
}

/// The f32 spacing at a given magnitude — one ulp of the mantissa.
fn f32_quantisation(magnitude_m: f64) -> f64 {
    let m = magnitude_m.abs().max(1.0) as f32;
    (f32::from_bits(m.to_bits() + 1) - m) as f64
}

/// **Everything one segment carries beside its carriageway** (wave ROAD1).
///
/// The shoulder goes into the **carriageway's own** ribbon because a sealed
/// shoulder is asphalt; the kerb, the pavement and the paint go into their own
/// material groups. What separates a shoulder from a running lane is the edge
/// line, not a material.
fn build_segment_furniture(
    out: &mut RoadSurface,
    graph: &RoadGraph,
    seg: &RoadSegment,
    dense: &[DVec3],
    opts: &SurfaceOptions,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) {
    let frames = cross_frames(dense);
    if frames.len() < 2 {
        return;
    }
    let half = seg.width_m() * 0.5;
    // **The furniture grades over the same plateau the road does**, and that
    // plateau is this class's own built half-width — carriageway plus kerb and
    // footway, or plus shoulder. A kerb that conformed while the carriageway it
    // bounds was graded would ride up and down the terrain's cross-slope beside
    // a channel that does not.
    let owned = SurfaceOptions {
        graded_half_m: built_half_width_m(seg.kind, seg.lane_count),
        ..*opts
    };
    let opts = if opts.crown_fall > 0.0 { &owned } else { opts };

    if seg.kind.is_kerbed() {
        build_kerbs(
            out.furniture.entry(RoadPart::Kerb).or_default(),
            &frames,
            half,
            opts,
            height_at,
        );
    } else {
        build_shoulders(
            out.parts.entry(seg.kind).or_default(),
            &frames,
            half,
            seg.kind.shoulder_m(),
            opts,
            height_at,
        );
    }

    // The paint. Two ribbons because two colours, and two colours because
    // `inf_ecs::Material` binds one `.inf_mat` — see `RoadPart`.
    let mut white = std::mem::take(out.furniture.entry(RoadPart::MarkingWhite).or_default());
    let mut yellow = std::mem::take(out.furniture.entry(RoadPart::MarkingYellow).or_default());
    build_markings(&mut white, &mut yellow, &frames, seg, opts, height_at);

    // **A crossing belongs to a JUNCTION, and it is painted from the leg.**
    // Degree 3 is the same threshold `fill_junctions` uses: a degree-2 node is a
    // bend in one road and nobody crosses a bend. A dead end gets none either —
    // there is nothing on the far side to cross to.
    if seg.kind.is_kerbed() {
        for (node, from_start) in [(seg.start_node, true), (seg.end_node, false)] {
            let crossable = graph
                .intersections
                .get(&node)
                .is_some_and(|i| i.degree() >= 3);
            if crossable {
                build_crosswalk(&mut white, &frames, seg, from_start, opts, height_at);
            }
        }
    }

    if !white.indices.is_empty() {
        *out.furniture.entry(RoadPart::MarkingWhite).or_default() = white;
    }
    if !yellow.indices.is_empty() {
        *out.furniture.entry(RoadPart::MarkingYellow).or_default() = yellow;
    }
    // An entry that was created and stayed empty would become a zero-triangle
    // submesh, which `surface_to_mesh` skips but `furniture_to_mesh` would
    // otherwise turn into a mesh asset with no geometry in it.
    out.furniture.retain(|_, r| !r.indices.is_empty());
}

/// **One furniture group as its own `MeshAsset`** (wave ROAD1) — `None` when
/// this road has none of that part.
///
/// A separate function from [`surface_to_mesh`] rather than a fourth submesh in
/// it, because a mesh's submeshes all draw with **one** `Material` component:
/// four material groups is four assets and four entities however the geometry is
/// stored. Positions are local to `origin`, the same f32-mantissa reason.
pub fn furniture_to_mesh(
    surface: &RoadSurface,
    origin: DVec3,
    part: RoadPart,
) -> Result<Option<(inf_mesh::MeshAsset, MeshBuildReport)>, crate::GisError> {
    if !origin.is_finite() {
        return Err(crate::GisError::NotFinite(format!(
            "the road mesh's local origin {origin:?} is not finite"
        )));
    }
    let Some(ribbon) = surface.furniture.get(&part) else {
        return Ok(None);
    };
    if ribbon.indices.is_empty() {
        return Ok(None);
    }
    let mut report = MeshBuildReport::default();
    let sub = ribbon_to_submesh(part.label(), ribbon, origin, 0, &mut report)?;
    report.quantisation_m = f32_quantisation(report.max_offset_m);
    Ok(Some((
        inf_mesh::MeshAsset::new(vec![sub], vec![part.label().to_string()]),
        report,
    )))
}

/// **The road surface as a real `MeshAsset`** — the thing Wave G's ribbon
/// builder stopped one step short of.
///
/// One submesh per road class, named by the class, with the class list as the
/// mesh's material slots — so assigning a material to slot *k* means "this is
/// what an arterial looks like" on both sides of the asset.
///
/// Positions are **local to `origin`**, because a `MeshVertex` is f32 and a
/// world-space UTM coordinate spends its whole mantissa on the false easting.
/// The caller places the entity at `origin`; see [`MeshBuildReport`].
pub fn surface_to_mesh(
    surface: &RoadSurface,
    origin: DVec3,
) -> Result<(inf_mesh::MeshAsset, MeshBuildReport), crate::GisError> {
    if !origin.is_finite() {
        return Err(crate::GisError::NotFinite(format!(
            "the road mesh's local origin {origin:?} is not finite"
        )));
    }
    if surface.is_empty() {
        return Err(crate::GisError::Geometry(
            "the road network produced no surface at all — every segment was \
             skipped, or the layer had none. The per-segment reasons are in the \
             import report."
                .to_string(),
        ));
    }
    let mut report = MeshBuildReport::default();
    let mut submeshes = Vec::new();
    let mut slots = Vec::new();
    for (kind, ribbon) in &surface.parts {
        if ribbon.indices.is_empty() {
            continue;
        }
        let slot = slots.len() as u32;
        slots.push(kind.label().to_string());
        let sub = ribbon_to_submesh(kind.label(), ribbon, origin, slot, &mut report)?;
        submeshes.push(sub);
    }
    if submeshes.is_empty() {
        return Err(crate::GisError::Geometry(
            "every road class in this network produced an empty surface".to_string(),
        ));
    }
    report.quantisation_m = f32_quantisation(report.max_offset_m);
    Ok((inf_mesh::MeshAsset::new(submeshes, slots), report))
}

/// One ribbon → one submesh, with per-vertex normals and UV-derived tangents.
fn ribbon_to_submesh(
    name: &str,
    ribbon: &RoadRibbon,
    origin: DVec3,
    slot: u32,
    report: &mut MeshBuildReport,
) -> Result<inf_mesh::SubMesh, crate::GisError> {
    let n = ribbon.vertices.len();
    if ribbon.uvs.len() != n {
        return Err(crate::GisError::Geometry(format!(
            "the {name} ribbon has {n} vertices and {} UVs — they are index \
             aligned by construction, so this is a builder defect",
            ribbon.uvs.len()
        )));
    }
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(n);
    for v in &ribbon.vertices {
        if !v.is_finite() {
            return Err(crate::GisError::NotFinite(format!(
                "the {name} surface carries the non-finite vertex {v:?}"
            )));
        }
        let l = *v - origin;
        report.max_offset_m = report
            .max_offset_m
            .max(l.x.abs().max(l.y.abs()).max(l.z.abs()));
        pos.push([l.x as f32, l.y as f32, l.z as f32]);
    }
    // Accumulate face normals and UV tangents per vertex. Both are the textbook
    // constructions; the only thing worth saying is that a degenerate triangle
    // (a sliver at a hairpin, a fan spoke of zero area) contributes nothing
    // rather than a NaN, which is the same door every other producer in this
    // crate has.
    let mut nrm = vec![glam::Vec3::ZERO; n];
    let mut tan = vec![glam::Vec3::ZERO; n];
    for t in ribbon.indices.chunks_exact(3) {
        let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n {
            return Err(crate::GisError::Geometry(format!(
                "the {name} surface indexes vertex {} of {n}",
                t.iter().copied().max().unwrap_or(0)
            )));
        }
        let p = [
            glam::Vec3::from(pos[i0]),
            glam::Vec3::from(pos[i1]),
            glam::Vec3::from(pos[i2]),
        ];
        let face = (p[1] - p[0]).cross(p[2] - p[0]);
        if face.length_squared() > 0.0 && face.is_finite() {
            for i in [i0, i1, i2] {
                nrm[i] += face;
            }
        }
        let uv = [
            glam::Vec2::from(ribbon.uvs[i0]),
            glam::Vec2::from(ribbon.uvs[i1]),
            glam::Vec2::from(ribbon.uvs[i2]),
        ];
        let (duv1, duv2) = (uv[1] - uv[0], uv[2] - uv[0]);
        let det = duv1.x * duv2.y - duv2.x * duv1.y;
        if det.abs() > 1e-12 {
            let r = 1.0 / det;
            let t3 = ((p[1] - p[0]) * duv2.y - (p[2] - p[0]) * duv1.y) * r;
            if t3.is_finite() {
                for i in [i0, i1, i2] {
                    tan[i] += t3;
                }
            }
        }
        report.triangles += 1;
    }

    let vertices: Vec<inf_mesh::MeshVertex> = (0..n)
        .map(|i| {
            let normal = nrm[i].normalize_or(glam::Vec3::Y);
            // Gram-Schmidt against the normal, so the tangent frame is
            // orthonormal even where two ribbons meet at a fold.
            let t = tan[i] - normal * normal.dot(tan[i]);
            let tangent = if t.length_squared() > 1e-20 {
                let t = t.normalize();
                [t.x, t.y, t.z, 1.0]
            } else {
                inf_mesh::TANGENT_PLACEHOLDER
            };
            inf_mesh::MeshVertex {
                position: pos[i],
                normal: [normal.x, normal.y, normal.z],
                uv: ribbon.uvs[i],
                tangent,
            }
        })
        .collect();
    report.vertices += vertices.len();

    Ok(inf_mesh::SubMesh {
        name: name.to_string(),
        vertices,
        indices: ribbon.indices.clone(),
        material_slot: Some(slot),
        skin: Vec::new(),
    })
}

/// A triangle fan over the cross-section corners meeting at one junction.
///
/// # Why a fan and not a mitre
///
/// A mitre joins **two** legs and `build_ribbon` already does it. Three or more
/// legs have no bisector to mitre onto: the honest shapes are a fan (paves the
/// convex area the legs enclose) or a proper intersection mesh with kerb radii
/// per leg pair, which is a road-modelling project rather than an import. The
/// fan is the one that is right about the thing that matters here — there is no
/// hole in the ground at the junction — and wrong only about the corner radii,
/// which is a texture-scale defect on a surface a car drives over.
///
/// The corners are sorted radially about their own centroid by
/// [`pseudo_angle`], which is trig-free by law, and each triangle is emitted
/// with the winding that faces **up**, tested per triangle rather than reasoned
/// about.
fn fan_at(
    corners: &[(DVec3, DVec3)],
    lift_m: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> Option<RoadRibbon> {
    let mut pts: Vec<DVec3> = Vec::with_capacity(corners.len() * 2);
    for (l, r) in corners {
        pts.push(*l);
        pts.push(*r);
    }
    if pts.len() < 3 || pts.iter().any(|p| !p.is_finite()) {
        return None;
    }
    let mut centre = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    if !centre.is_finite() {
        return None;
    }
    // **The hub sits on the GROUND, not on the average of its rim.** Averaging
    // was the first version and it is wrong in exactly the case a junction meets:
    // a corner whose XZ falls outside the terrain keeps the published
    // centreline's own elevation, and one such corner in six dragged a hub 7.78 m
    // below the ground under it — measured, on a road running along a terrain's
    // edge. Asking the same query every other vertex asks makes the hub a vertex
    // like the rest, and lets the import's own arm assert *every* vertex sits at
    // `ground + lift`.
    if let Some(g) = height_at(centre.x, centre.z) {
        if g.is_finite() {
            centre.y = g + lift_m;
        }
    }
    // Sort radially. `total_cmp` because a pseudo-angle is an f64 and
    // `partial_cmp().unwrap()` is the panic this crate has already paid for.
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|a, b| {
        let ka = pseudo_angle(pts[*a].xz() - centre.xz());
        let kb = pseudo_angle(pts[*b].xz() - centre.xz());
        ka.total_cmp(&kb).then(a.cmp(b))
    });

    let mut ribbon = RoadRibbon {
        vertices: vec![centre],
        uvs: vec![[0.5, 0.5]],
        indices: Vec::new(),
    };
    for i in &order {
        ribbon.vertices.push(pts[*i]);
        // The junction's UVs are a unit-square projection about the centre, so
        // a road texture continues across it at roughly its own scale rather
        // than stretching.
        let d = pts[*i].xz() - centre.xz();
        ribbon
            .uvs
            .push([(0.5 + d.x * 0.1) as f32, (0.5 + d.y * 0.1) as f32]);
    }
    let n = order.len() as u32;
    for i in 0..n {
        let a = 0u32;
        let b = 1 + i;
        let c = 1 + (i + 1) % n;
        let (pb, pc) = (ribbon.vertices[b as usize], ribbon.vertices[c as usize]);
        // The vertical component of (pb - centre) x (pc - centre): positive is
        // up. Tested rather than assumed — the shoelace sign on the XZ plane is
        // the NEGATION of the usual 2-D orientation term, which this crate has
        // already had backwards once.
        let (u, v) = (pb - centre, pc - centre);
        let up = u.z * v.x - u.x * v.z;
        if up.abs() <= 0.0 {
            continue; // a degenerate sliver contributes nothing
        }
        if up > 0.0 {
            ribbon.indices.extend_from_slice(&[a, b, c]);
        } else {
            ribbon.indices.extend_from_slice(&[a, c, b]);
        }
    }
    (!ribbon.indices.is_empty()).then_some(ribbon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{Attr, LayerKind};

    fn line(pts: &[(f64, f64)]) -> GeoGeometry {
        GeoGeometry::Polyline {
            points: pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect(),
            closed: false,
        }
    }

    fn layer(features: Vec<GeoFeature>) -> GeoLayer {
        let mut l = GeoLayer::new("Roads", LayerKind::Roads, "EPSG:32610");
        l.features = features;
        l
    }

    /// One classified road feature, for the wave-ROAD1 arms below.
    fn road(pts: &[(f64, f64)], class: &str, name: &str) -> GeoFeature {
        let mut f = GeoFeature::new(line(pts));
        f.attributes.insert("name".into(), Attr::Text(name.into()));
        f.attributes
            .insert("road_type".into(), Attr::Text(class.into()));
        f
    }

    /// A synthetic hill with real curvature, so a chord across it is visibly
    /// wrong and a resampled ribbon is not. Portable (no trig): a quadratic
    /// bump, whose second derivative is a constant.
    fn hill(x: f64, z: f64) -> Option<f64> {
        Some(20.0 - 0.002 * (x - 100.0) * (x - 100.0) - 0.001 * z * z)
    }

    /// A three-legged junction at (100, 0).
    ///
    /// **The through-street is TWO features, not one**, and that is how published
    /// layers are digitised: `RoadGraph` derives its nodes from segment
    /// *endpoints*, so a street that merely passes through a vertex at the
    /// crossing creates no junction there. A fixture that wrote Broadway as one
    /// 200 m feature produced a degree-2 node and the fan gate had nothing to
    /// measure.
    fn t_junction() -> RoadGraph {
        let arterial = |pts: &[(f64, f64)], name: &str| {
            let mut f = GeoFeature::new(line(pts));
            f.attributes.insert("name".into(), Attr::Text(name.into()));
            f.attributes
                .insert("road_type".into(), Attr::Text("arterial".into()));
            f
        };
        let mut b = GeoFeature::new(line(&[(100.0, 0.0), (100.0, 120.0)]));
        b.attributes.insert("name".into(), Attr::Text("Elm".into()));
        RoadGraph::from_layer(&layer(vec![
            arterial(&[(0.0, 0.0), (60.0, 0.0), (100.0, 0.0)], "Broadway West"),
            arterial(&[(100.0, 0.0), (200.0, 0.0)], "Broadway East"),
            b,
        ]))
    }

    // ── the lanes (wave VEH2b) ──────────────────────────────────────────────

    /// The one number two crates share: a lane derived in `inf-nav` has to fit
    /// inside the ribbon drawn here, and `inf-nav` restates the width rather
    /// than importing it (it depends on `glam` alone, on purpose). This is the
    /// arm that keeps the restatement honest — the `inf_nav::lane` doc names it.
    #[test]
    fn the_lane_width_agrees_with_the_surface_it_is_drawn_on() {
        assert_eq!(LANE_WIDTH_M, inf_nav::lane::DEFAULT_LANE_WIDTH_M);
        // …and the carriageway a class implies is the width the ribbon is built
        // at, so lane 0 and lane `n-1` both land on tarmac.
        for kind in [RoadKind::Highway, RoadKind::Arterial, RoadKind::Residential] {
            let lanes = kind.default_lanes();
            assert_eq!(kind.width_m(lanes), LANE_WIDTH_M * f64::from(lanes));
        }
    }

    /// A two-lane arterial gets one carriageway each way, offset half a lane
    /// from the paint — and the two run in opposite directions, which is the
    /// whole point.
    #[test]
    fn a_two_way_street_gets_a_carriageway_on_each_side_of_its_paint() {
        let net = t_junction().lane_network();
        // The two Broadway halves are arterials — four lanes, so two each way —
        // and Elm carries no `road_type`, so it is residential: two lanes, one
        // each way. 4 + 4 + 2 = 10, and the arithmetic is the class table's.
        assert_eq!(net.len(), 10);
        assert_eq!(net.worst_fold_m(), 0.0);
        // Broadway East runs +X from the junction at (100, 0) to (200, 0), so
        // its inner lane sits at z = -1.75; the other way sits at +1.75.
        let mut east = None;
        let mut west = None;
        for lane in net.lanes().filter(|l| l.id.index == 0) {
            let a = lane.entry();
            let b = lane.exit();
            if (a.x - 100.0).abs() < 2.0 && (b.x - 200.0).abs() < 2.0 {
                east = Some(lane.clone());
            }
            if (a.x - 200.0).abs() < 2.0 && (b.x - 100.0).abs() < 2.0 {
                west = Some(lane.clone());
            }
        }
        let (east, west) = (east.expect("eastbound"), west.expect("westbound"));
        assert!((east.entry().z + 1.75).abs() < 1e-9, "{:?}", east.entry());
        assert!((west.entry().z - 1.75).abs() < 1e-9, "{:?}", west.entry());
        // …and neither of them is on the paint, which is where VEH1a parked.
        assert!(east.entry().z != 0.0 && west.entry().z != 0.0);
    }

    /// The source's own attributes finally reach something a car reads.
    #[test]
    fn the_layers_lane_count_and_speed_limit_reach_the_carriageway() {
        let mut f = GeoFeature::new(line(&[(0.0, 0.0), (400.0, 0.0)]));
        f.attributes
            .insert("road_type".into(), Attr::Text("highway".into()));
        f.attributes.insert("lanes".into(), Attr::Number(4.0));
        f.attributes
            .insert("speed_limit_kmh".into(), Attr::Number(90.0));
        let net = RoadGraph::from_layer(&layer(vec![f])).lane_network();
        // Four lanes: two out, two back.
        assert_eq!(net.len(), 4);
        for lane in net.lanes() {
            assert_eq!(lane.speed_limit_kmh, 90);
            assert!((lane.speed_limit_mps() - 25.0).abs() < 1e-12);
        }
        let outer: Vec<f64> = net
            .lanes()
            .filter(|l| l.id.index == 1)
            .map(|l| l.entry().z.abs())
            .collect();
        assert_eq!(outer.len(), 2);
        for z in outer {
            assert!((z - 5.25).abs() < 1e-9, "{z}");
        }
    }

    /// A source with no `speed_limit` attribute — which is every committed
    /// layer in this tree — gets the class's own sign, not one number for the
    /// whole island.
    #[test]
    fn a_layer_with_no_signs_takes_the_class_default() {
        let of = |class: &str| {
            let mut f = GeoFeature::new(line(&[(0.0, 0.0), (300.0, 0.0)]));
            f.attributes
                .insert("road_type".into(), Attr::Text(class.into()));
            RoadGraph::from_layer(&layer(vec![f]))
                .lane_network()
                .lanes()
                .next()
                .map(|l| l.speed_limit_kmh)
        };
        assert_eq!(of("highway"), Some(90));
        assert_eq!(of("arterial"), Some(60));
        assert_eq!(of("residential"), Some(50));
        assert_eq!(of("track"), Some(30));
    }

    /// Nothing in this engine drives a train, and a car routed onto a railway
    /// line is worse than a car with nowhere to go.
    #[test]
    fn a_railway_carries_no_carriageway() {
        let mut f = GeoFeature::new(line(&[(0.0, 0.0), (500.0, 0.0)]));
        f.attributes
            .insert("road_type".into(), Attr::Text("rail".into()));
        let graph = RoadGraph::from_layer(&layer(vec![f]));
        // It is still a road segment and still paves ground…
        assert_eq!(graph.segments.len(), 1);
        // …and it is not somewhere to drive.
        assert!(graph.lane_network().is_empty());
    }

    /// **The road follows the ground, and the step that makes it do so is the
    /// TERRAIN's number.**
    ///
    /// Every ribbon vertex sits at `ground + lift` by construction, so the arm
    /// that means something is the one about the space *between* vertices: a
    /// ribbon built on the centreline's own vertices spans a hill as a chord.
    /// The alternative is priced in the test, in metres, because a gate against
    /// a cheaper alternative has to price the alternative (the I1 audit law).
    #[test]
    fn roads_follow_the_ground_they_are_draped_on() {
        let graph = t_junction();
        let mut h = |x: f64, z: f64| hill(x, z);
        let opts = SurfaceOptions::default();
        let s = build_surface(&graph, &opts, &mut h);
        assert!(!s.is_empty() && s.skipped.is_empty(), "{s:?}");

        // (a) EVERY vertex — junction hubs included — is exactly `lift` above
        // the ground under it.
        let mut worst_vertex = 0.0f64;
        for r in s.parts.values() {
            for v in &r.vertices {
                let g = hill(v.x, v.z).unwrap();
                worst_vertex = worst_vertex.max((v.y - g - opts.lift_m).abs());
            }
        }
        assert!(
            worst_vertex < 1e-9,
            "every road vertex must sit exactly {} m above the ground; worst was \
             {worst_vertex} m",
            opts.lift_m
        );
        let ribbons = build_surface(
            &graph,
            &SurfaceOptions {
                fill_junctions: false,
                ..opts
            },
            &mut h,
        );

        // (b) Between vertices — the claim that matters. Walk each ribbon edge's
        // midpoint and compare the interpolated road against the real ground.
        fn ribbon_error(r: &RoadRibbon) -> f64 {
            let mut worst = 0.0f64;
            for t in r.indices.chunks_exact(3) {
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    let a = r.vertices[t[i] as usize];
                    let b = r.vertices[t[j] as usize];
                    let m = (a + b) * 0.5;
                    let g = hill(m.x, m.z).unwrap();
                    worst = worst.max((m.y - g - 0.02).abs());
                }
            }
            worst
        }
        let midspan_error = |surf: &RoadSurface| -> f64 {
            surf.parts.values().map(ribbon_error).fold(0.0f64, f64::max)
        };
        let dense = midspan_error(&ribbons);
        let coarse = midspan_error(&build_surface(
            &graph,
            &SurfaceOptions {
                fill_junctions: false,
                ground_step_m: 10_000.0, // i.e. the centreline's own vertices
                ..opts
            },
            &mut h,
        ));
        assert!(
            dense < 0.002,
            "at a 1 m step the road should hug the ground to millimetres; it was \
             {dense:.6} m"
        );
        assert!(
            coarse > 1.0,
            "THE ALTERNATIVE MUST BE PRICED: on the centreline's own vertices the \
             ribbon spans the hill as a chord, and that has to be metres for this \
             gate to mean anything — it measured {coarse:.4} m"
        );
        assert!(
            coarse / dense > 100.0,
            "dense {dense:.6} m vs coarse {coarse:.4} m"
        );

        // (c) **THE OTHER AXIS, PRICED THE SAME WAY** (I2 audit). The wave's own
        // law is that closing the longitudinal gap *hid* the transverse one,
        // and the only thing standing behind it was the `dense` bound above —
        // which does catch a one-quad ribbon, but never says by how much, so
        // the number the law is named for lived in a ledger rather than in a
        // test. Here it is: the SAME densified spine through `build_ribbon`,
        // which is one quad across, i.e. exactly what this builder had before
        // IB-4.
        let arterial = graph
            .segments
            .values()
            .find(|s| s.kind == RoadKind::Arterial)
            .expect("the fixture has an arterial");
        let spine = densify_spine(&arterial.spine, opts.ground_step_m);
        let one_quad =
            build_ribbon(&spine, arterial.width_m(), opts.lift_m, &mut h).expect("a ribbon builds");
        let across = ribbon_error(&one_quad);
        let subdivided = build_ribbon_across(
            &spine,
            arterial.width_m(),
            opts.lift_m,
            cross_strips(arterial.width_m(), opts.ground_step_m),
            // Conforming, which is what this arm measures: a graded section
            // would sit on one ground sample by design and its "error" would be
            // the terrain's cross-slope rather than the builder's.
            0.0,
            &mut h,
        )
        .expect("a ribbon builds");
        let subdivided = ribbon_error(&subdivided);
        assert!(
            across > 0.02,
            "THE TRANSVERSE ALTERNATIVE MUST BE PRICED: one quad across a {} m \
             carriageway leaves its crown off the ground by the half-width's own \
             chord, and that has to be centimetres for this arm to mean anything \
             — it measured {across:.6} m",
            arterial.width_m()
        );
        assert!(
            across / subdivided > 20.0,
            "subdividing across must be the fix, not a rounding: one quad \
             {across:.6} m vs {} strips {subdivided:.6} m",
            cross_strips(arterial.width_m(), opts.ground_step_m)
        );
        println!(
            "IB-4 draping on a {} m arterial: dense {dense:.6} m (both axes) vs \
             centreline-only {coarse:.4} m ALONG vs one-quad-across \
             {across:.4} m ACROSS; subdivided across = {subdivided:.6} m",
            arterial.width_m()
        );
    }

    /// Densifying keeps every surveyed vertex and bounds the sub-step.
    #[test]
    fn densify_keeps_the_survey_and_bounds_the_step() {
        let spine = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.5, 0.0, 0.0),
            DVec3::new(2.5, 0.0, 30.0),
        ];
        let d = densify_spine(&spine, 1.0);
        for p in &spine {
            assert!(
                d.iter().any(|q| (*q - *p).length() < 1e-12),
                "the survey's own vertex {p:?} must survive"
            );
        }
        let worst = d
            .windows(2)
            .map(|w| (w[1].xz() - w[0].xz()).length())
            .fold(0.0f64, f64::max);
        assert!(worst <= 1.0 + 1e-12, "worst sub-step {worst}");
        // Degenerate steps are refused by returning the input rather than
        // looping forever.
        assert_eq!(densify_spine(&spine, 0.0), spine);
        assert_eq!(densify_spine(&spine, f64::NAN), spine);
        assert_eq!(densify_spine(&spine[..1], 1.0), spine[..1].to_vec());
    }

    /// Is `p` (XZ) inside any triangle of the surface?
    fn covered_at(s: &RoadSurface, p: glam::DVec2) -> bool {
        for r in s.parts.values() {
            for t in r.indices.chunks_exact(3) {
                let a = r.vertices[t[0] as usize].xz();
                let b = r.vertices[t[1] as usize].xz();
                let c = r.vertices[t[2] as usize].xz();
                let s1 = (b - a).perp_dot(p - a);
                let s2 = (c - b).perp_dot(p - b);
                let s3 = (a - c).perp_dot(p - c);
                if (s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0) || (s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0) {
                    return true;
                }
            }
        }
        false
    }

    /// **The road MESH is a function of the layer, not of a walk over it**
    /// (I2 audit).
    ///
    /// `the_graph_is_order_independent` holds the graph to this and stops there;
    /// what reaches committed content is the `.inf_mesh` two crates further on,
    /// which adds a `BTreeMap` per class, a `BTreeMap` of junction ends, a radial
    /// sort and a normal/tangent accumulation on top. So the arm goes all the way
    /// to the bytes the asset writer would write: build the same layer twice and
    /// encode both. An iteration order that stopped being a `BTreeMap`, a sort
    /// that stopped breaking its ties, or a fold that picked up a `HashMap`
    /// would show here and nowhere else.
    ///
    /// It is deliberately **not** an order-independence claim: segment ids are
    /// encounter-order, so reversing the file renumbers them and the vertex
    /// order moves with them — the graph arm says exactly that.
    #[test]
    fn the_same_layer_builds_a_bit_identical_road_mesh() {
        // FNV-1a over every field of the asset, floats as `to_bits` — the same
        // construction `SpawnPlan::digest` uses, and for the same reason: a
        // comparison of the exact numbers rather than of printed ones.
        fn mesh_digest(m: &inf_mesh::MeshAsset) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut w = |v: u64| {
                for b in v.to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            };
            w(u64::from(m.schema_version));
            w(m.material_slots.len() as u64);
            for s in &m.material_slots {
                w(s.len() as u64);
                for b in s.as_bytes() {
                    w(u64::from(*b));
                }
            }
            w(m.submeshes.len() as u64);
            for sub in &m.submeshes {
                w(sub.name.len() as u64);
                for b in sub.name.as_bytes() {
                    w(u64::from(*b));
                }
                w(sub.material_slot.map_or(u64::MAX, u64::from));
                w(sub.indices.len() as u64);
                for i in &sub.indices {
                    w(u64::from(*i));
                }
                w(sub.vertices.len() as u64);
                for v in &sub.vertices {
                    for f in v
                        .position
                        .iter()
                        .chain(&v.normal)
                        .chain(&v.uv)
                        .chain(&v.tangent)
                    {
                        w(u64::from(f.to_bits()));
                    }
                }
            }
            h
        }

        let graph = t_junction();
        let build = || {
            let mut h = |x: f64, z: f64| hill(x, z);
            let s = build_surface(&graph, &SurfaceOptions::default(), &mut h);
            surface_to_mesh(&s, DVec3::new(100.0, 20.0, 0.0)).unwrap()
        };
        let (a, ra) = build();
        let (b, rb) = build();
        assert_eq!(ra, rb, "one layer, one report");
        assert_eq!(
            mesh_digest(&a),
            mesh_digest(&b),
            "two builds of one layer are one asset, bit for bit"
        );
        assert_eq!(
            a, b,
            "…and structurally, so the digest is not the only reader"
        );
        // …and the digest is not a constant: a different lift is a different
        // asset, so an arm that stopped building would not pass by accident.
        let mut h = |x: f64, z: f64| hill(x, z);
        let other = build_surface(
            &graph,
            &SurfaceOptions {
                lift_m: 0.05,
                ..Default::default()
            },
            &mut h,
        );
        let (other, _) = surface_to_mesh(&other, DVec3::new(100.0, 20.0, 0.0)).unwrap();
        assert_ne!(mesh_digest(&other), mesh_digest(&a));
        println!(
            "IB-4 determinism: {} vertices / {} triangles, digest {:016x}, \
             identical across two builds",
            ra.vertices,
            ra.triangles,
            mesh_digest(&a)
        );
    }

    /// **A junction's core is PAVED, not holed** — and the arm measures *ground*
    /// on a grid rather than asking the report whether it ran.
    ///
    /// # The fixture is an acute fork, and that is a finding
    ///
    /// The obvious fixture — a symmetric T or a `+` crossing — turns out to need
    /// **no** fan at all: when two opposed legs of one street both end at the
    /// node, their ribbons tile the whole crossing between them and every probe
    /// is already covered, so a gate built on one measures nothing and passes
    /// whatever the fan does. Holes appear where the legs do *not* span the
    /// node's neighbourhood between them: an acute fork (a slip road leaving a
    /// through street) with a third leg elsewhere leaves an open wedge on the
    /// far side of the fork, inside the junction's own corner hull. That is the
    /// case this measures.
    ///
    /// What the fan does **not** fix is stated where it is built: the wedges
    /// *outside* the corner hull, between adjacent kerbs, need kerb-radius
    /// fillets per leg pair, which is a road-modelling project rather than an
    /// import.
    #[test]
    fn a_junction_of_three_roads_is_paved_rather_than_holed() {
        // Node at (100, 0): a through street heading east, a slip road forking
        // off it at ~11 degrees, and a cross street heading north.
        let leg = |to: (f64, f64), name: &str| {
            let mut f = GeoFeature::new(line(&[(100.0, 0.0), to]));
            f.attributes.insert("name".into(), Attr::Text(name.into()));
            f
        };
        let graph = RoadGraph::from_layer(&layer(vec![
            leg((300.0, 0.0), "Kingsway"),
            leg((298.0, 40.0), "Kingsway Slip"),
            leg((100.0, 200.0), "Main"),
        ]));
        let node = graph
            .intersections
            .values()
            .find(|i| i.degree() >= 3)
            .expect("three legs share one endpoint");
        assert_eq!(node.degree(), 3);

        let mut h = |x: f64, z: f64| hill(x, z);
        let filled = build_surface(&graph, &SurfaceOptions::default(), &mut h);
        let holed = build_surface(
            &graph,
            &SurfaceOptions {
                fill_junctions: false,
                ..Default::default()
            },
            &mut h,
        );
        assert_eq!(filled.junctions_filled, 1, "{filled:?}");
        assert_eq!(filled.junctions_skipped, 0);

        // Sweep a 16 m box around the node at 10 cm and count open ground.
        let mut open_before = 0usize;
        let mut open_after = 0usize;
        let mut newly_paved = 0usize;
        for ix in -80..=80 {
            for iz in -80..=80 {
                let p = node.position.xz() + glam::DVec2::new(ix as f64 * 0.1, iz as f64 * 0.1);
                let b = covered_at(&holed, p);
                let a = covered_at(&filled, p);
                if !b {
                    open_before += 1;
                }
                if !a {
                    open_after += 1;
                }
                if !b && a {
                    newly_paved += 1;
                }
                assert!(b <= a, "the fan must never REMOVE ground, at {p:?}");
            }
        }
        assert!(
            newly_paved > 0,
            "THE FAN MUST PAVE GROUND THAT WAS OPEN. It paved {newly_paved} of the \
             {open_before} open samples in the box — if that is zero, the legs \
             already tiled the junction and this gate measures nothing."
        );
        assert!(open_after < open_before);
        println!(
            "IB-4 junction: {newly_paved} of {open_before} open samples paved by the \
             fan ({open_after} remain, outside the corner hull)"
        );
        // The fan takes the class of the most important road meeting there; all
        // three legs here are residential.
        assert!(
            filled.parts.contains_key(&RoadKind::Residential),
            "{:?}",
            filled.parts.keys().collect::<Vec<_>>()
        );
        assert!(filled.triangle_count() > holed.triangle_count());
    }

    /// **The ribbon becomes a real `MeshAsset`** — the step Wave G stopped one
    /// short of — with per-class submeshes, upward normals and a stated
    /// positional resolution.
    #[test]
    fn a_road_network_becomes_a_mesh_asset_with_a_submesh_per_class() {
        let graph = t_junction();
        let mut h = |x: f64, z: f64| hill(x, z);
        let s = build_surface(&graph, &SurfaceOptions::default(), &mut h);
        // Centre the mesh on its own content — the reason is f32.
        let origin = DVec3::new(100.0, 20.0, 0.0);
        let (mesh, report) = surface_to_mesh(&s, origin).unwrap();

        assert_eq!(mesh.schema_version, inf_mesh::MeshAsset::CURRENT_VERSION);
        assert_eq!(
            mesh.material_slots,
            vec!["arterial".to_string(), "residential".into()],
            "one slot per class present, in class order"
        );
        assert_eq!(mesh.submeshes.len(), 2);
        assert_eq!(mesh.triangle_count(), s.triangle_count());
        assert!(mesh.validate().is_ok(), "the asset's own door accepts it");
        for (i, sub) in mesh.submeshes.iter().enumerate() {
            assert_eq!(sub.material_slot, Some(i as u32));
            for v in &sub.vertices {
                assert!(v.position.iter().all(|c| c.is_finite()));
                assert!(
                    v.normal[1] > 0.5,
                    "a road surface faces UP; {:?} does not",
                    v.normal
                );
                let t = glam::Vec3::from_slice(&v.tangent[..3]);
                assert!((t.length() - 1.0).abs() < 1e-3, "unit tangent, got {t:?}");
            }
        }
        // The report says what an f32 vertex cost at this extent.
        assert!(report.max_offset_m > 100.0 && report.max_offset_m < 200.0);
        assert!(
            report.quantisation_m > 0.0 && report.quantisation_m < 1e-4,
            "{report:?}"
        );
        assert_eq!(report.triangles, s.triangle_count());

        // An empty surface is a refusal naming where the reasons are.
        let e = surface_to_mesh(&RoadSurface::default(), origin).unwrap_err();
        assert!(e.to_string().contains("import report"), "{e}");
        assert!(surface_to_mesh(&s, DVec3::new(f64::NAN, 0.0, 0.0)).is_err());
    }

    /// **The determinism gate** — the reason `HashMap` became `BTreeMap`.
    ///
    /// The same roads presented in a different order must produce the same graph,
    /// because a graph that depends on insertion order produces different cooked
    /// bytes on every run while the geometry looks identical.
    ///
    /// Un-fix mutation: number the nodes in encounter order instead of lattice
    /// order and the node-id assertion fails.
    #[test]
    fn the_graph_is_order_independent() {
        let mk = |pts: &[(f64, f64)]| GeoFeature::new(line(pts));
        let a = vec![
            mk(&[(0.0, 0.0), (100.0, 0.0)]),
            mk(&[(100.0, 0.0), (100.0, 100.0)]),
            mk(&[(100.0, 100.0), (0.0, 100.0)]),
        ];
        let mut b = a.clone();
        b.reverse();

        let ga = RoadGraph::from_layer(&layer(a));
        let gb = RoadGraph::from_layer(&layer(b));

        // The node set and their positions are identical.
        assert_eq!(
            ga.intersections.keys().collect::<Vec<_>>(),
            gb.intersections.keys().collect::<Vec<_>>()
        );
        for (id, n) in &ga.intersections {
            assert_eq!(
                n.position, gb.intersections[id].position,
                "node {id} moved when the input order changed"
            );
        }
        // Same total length and same node count, whatever order the file was in.
        assert!((ga.total_length_m() - gb.total_length_m()).abs() < 1e-9);
        assert_eq!(ga.intersections.len(), gb.intersections.len());
        // And the containers really are ordered maps: iteration is ascending.
        let keys: Vec<u64> = ga.segments.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);

        // **What is NOT order-independent, stated rather than left implied.**
        // Segment ids are assigned in encounter order (`pending.into_iter()
        // .enumerate()`), so reversing the file renumbers them — the geometry is
        // the same, the labels are not. The node ids ARE geometric (they come
        // out of the snap lattice, which is a BTreeMap keyed on position), which
        // is why the assertions above are on nodes.
        //
        // This matters the moment anything keys a cooked artefact by segment id:
        // `build_all_ribbons` returns a `BTreeMap<u64, RoadRibbon>` and two cooks
        // of the same file in different record orders would pair the same meshes
        // with different keys. Recorded here, and asserted so the day the ids
        // become geometric this test tells somebody the note is stale.
        let geom_a: Vec<Vec<DVec3>> = ga.segments.values().map(|s| s.spine.clone()).collect();
        let geom_b: Vec<Vec<DVec3>> = gb.segments.values().map(|s| s.spine.clone()).collect();
        assert_ne!(
            geom_a, geom_b,
            "segment IDS are encounter-order today, so reversing the file must \
             pair different geometry with the same id. If this now passes, the \
             ids became geometric — good news, and this comment plus the module \
             header need updating."
        );
        let mut sa = geom_a.clone();
        let mut sb = geom_b.clone();
        let key = |v: &Vec<DVec3>| (v[0].x.to_bits(), v[0].z.to_bits());
        sa.sort_by_key(key);
        sb.sort_by_key(key);
        assert_eq!(
            sa, sb,
            "the SET of segment geometry must be identical whatever order the \
             file was in — only the numbering moves"
        );
    }

    /// **Two doors onto one decision have to read the same fields**, and a
    /// ribbon that cannot be built is reported rather than dropped.
    ///
    /// `kind_of` is the wizard's preview of the class `from_layer` will assign.
    /// It probed six attribute spellings where the import probed ten, so on a
    /// TIGER-style layer whose class lives in `RTTYP` or `MTFCC` the preview said
    /// one thing and the import built another.
    #[test]
    fn the_preview_classifier_agrees_with_the_import_and_ribbons_report_failures() {
        for field in ROAD_CLASS_FIELDS {
            let mut f = GeoFeature::new(line(&[(0.0, 0.0), (100.0, 0.0)]));
            f.attributes
                .insert(field.to_uppercase(), Attr::Text("motorway".into()));
            assert_eq!(
                kind_of(&f),
                RoadKind::Highway,
                "the preview must read `{field}`"
            );
            let g = RoadGraph::from_layer(&layer(vec![f]));
            assert_eq!(
                g.segments.values().next().unwrap().kind,
                RoadKind::Highway,
                "the import must read `{field}` too — a preview that disagrees \
                 with the import is worse than no preview"
            );
        }

        // A segment whose spine collapses has no surface, and that is REPORTED.
        let g = RoadGraph::from_layer(&layer(vec![
            GeoFeature::new(line(&[(0.0, 0.0), (100.0, 0.0)])),
            GeoFeature::new(line(&[(0.0, 500.0), (300.0, 500.0)])),
        ]));
        let mut flat = |_: f64, _: f64| Some(0.0);
        let (ribbons, skipped) = build_all_ribbons(&g, 0.05, &mut flat);
        assert_eq!(ribbons.len(), 2, "both roads build");
        assert!(skipped.is_empty(), "{skipped:?}");

        // …and a non-finite lift takes every one of them out, by name.
        let mut sick = |_: f64, _: f64| Some(f64::NAN);
        let (ribbons, skipped) = build_all_ribbons(&g, 0.0, &mut sick);
        assert!(ribbons.is_empty());
        assert_eq!(skipped.len(), 2, "every failure is counted");
        assert!(
            skipped[0].contains("segment") && skipped[0].contains("non-finite"),
            "a dropped ribbon must name itself and its cause: {skipped:?}"
        );
    }

    /// Endpoints that nearly coincide become one junction — the property that
    /// makes a published layer into a connected network.
    #[test]
    fn near_endpoints_snap_into_one_junction() {
        // Four segments meeting at (100, 100), each digitised slightly off.
        let g = RoadGraph::from_layer(&layer(vec![
            GeoFeature::new(line(&[(0.0, 100.0), (99.6, 100.2)])),
            GeoFeature::new(line(&[(100.4, 99.8), (200.0, 100.0)])),
            GeoFeature::new(line(&[(100.1, 0.0), (100.0, 99.5)])),
            GeoFeature::new(line(&[(99.9, 100.5), (100.0, 200.0)])),
        ]));
        let junctions: Vec<_> = g.junctions().collect();
        assert_eq!(
            junctions.len(),
            1,
            "the four stubs must meet at ONE junction, got {} (positions {:?})",
            junctions.len(),
            g.intersections
                .values()
                .map(|n| n.position)
                .collect::<Vec<_>>()
        );
        assert_eq!(junctions[0].degree(), 4);
        // The junction sits at the average of the real endpoints, not on the
        // snap lattice.
        let p = junctions[0].position;
        assert!(
            (p.x - 100.0).abs() < 1.0 && (p.z - 100.0).abs() < 1.0,
            "the junction landed at {p:?}"
        );
        // Total nodes: the one junction plus four far ends.
        assert_eq!(g.intersections.len(), 5);

        // Roads that genuinely do not meet stay separate.
        let g = RoadGraph::from_layer(&layer(vec![
            GeoFeature::new(line(&[(0.0, 0.0), (50.0, 0.0)])),
            GeoFeature::new(line(&[(60.0, 0.0), (100.0, 0.0)])),
        ]));
        assert_eq!(g.junctions().count(), 0, "a 10 m gap is not a junction");
        assert_eq!(g.intersections.len(), 4);
    }

    /// Attributes drive width and class, over the field spellings real layers
    /// use — and an unrecognised class is COUNTED, not hidden.
    #[test]
    fn attributes_drive_class_and_width_and_unknowns_are_counted() {
        let mut hw = GeoFeature::new(line(&[(0.0, 0.0), (1000.0, 0.0)]));
        hw.attributes
            .insert("HIGHWAY".into(), Attr::Text("motorway".into()));
        hw.attributes.insert("LANES".into(), Attr::Number(6.0));
        hw.attributes
            .insert("MAXSPEED".into(), Attr::Text("100".into()));
        hw.attributes
            .insert("NAME".into(), Attr::Text("Trans-Canada".into()));

        let mut path = GeoFeature::new(line(&[(0.0, 50.0), (100.0, 50.0)]));
        path.attributes
            .insert("fclass".into(), Attr::Text("cycleway".into()));

        let mut weird = GeoFeature::new(line(&[(0.0, 90.0), (100.0, 90.0)]));
        weird
            .attributes
            .insert("type".into(), Attr::Text("zamboni run".into()));

        let bare = GeoFeature::new(line(&[(0.0, 120.0), (100.0, 120.0)]));

        let g = RoadGraph::from_layer(&layer(vec![hw, path, weird, bare]));
        let by_name = |n: &str| {
            g.segments
                .values()
                .find(|s| s.name == n)
                .expect("segment present")
                .clone()
        };
        let h = by_name("Trans-Canada");
        assert_eq!(h.kind, RoadKind::Highway);
        assert_eq!(h.lane_count, 6);
        assert_eq!(h.speed_limit_kmh, Some(100));
        assert!(
            (h.width_m() - 21.0).abs() < 1e-9,
            "6 lanes at 3.5 m, got {}",
            h.width_m()
        );
        assert!((h.length_m() - 1000.0).abs() < 1e-9);

        let kinds: Vec<RoadKind> = g.segments.values().map(|s| s.kind).collect();
        assert!(kinds.contains(&RoadKind::Path));
        // A path is 2 m wide regardless of lane arithmetic.
        let p = g
            .segments
            .values()
            .find(|s| s.kind == RoadKind::Path)
            .unwrap();
        assert_eq!(p.width_m(), 2.0);

        // Two segments could not be classified: the nonsense value and the one
        // with no attribute at all. Both defaulted, and both were counted.
        assert_eq!(
            g.unclassified, 2,
            "an unrecognised class must be COUNTED, not silently defaulted"
        );

        // Rail is recognised and is not paved as a road.
        assert_eq!(RoadKind::classify("Light Rail Transit"), RoadKind::Rail);
        assert_eq!(
            RoadKind::classify("residential street"),
            RoadKind::Residential
        );
        assert_eq!(RoadKind::classify("unpaved track"), RoadKind::DirtTrack);
    }

    /// The ribbon: a real surface, terrain-conforming, with tiling UVs.
    #[test]
    fn a_ribbon_conforms_to_the_ground_and_tiles_by_arc_length() {
        // A ramp: ground rises 1 m per 10 m of x.
        let mut ground = |x: f64, _z: f64| Some(x * 0.1);
        let spine: Vec<DVec3> = (0..=10)
            .map(|i| DVec3::new(i as f64 * 10.0, 0.0, 0.0))
            .collect();
        let r = build_ribbon(&spine, 7.0, 0.05, &mut ground).unwrap();

        assert_eq!(r.vertices.len(), 22, "two vertices per cross-section");
        assert_eq!(r.triangle_count(), 20, "two triangles per span");
        // Every vertex sits on the ground plus the lift.
        for v in &r.vertices {
            assert!(
                (v.y - (v.x * 0.1 + 0.05)).abs() < 1e-9,
                "vertex {v:?} is not on the ground"
            );
        }
        // The road is 7 m wide across, measured between the paired vertices.
        for pair in r.vertices.chunks_exact(2) {
            let w = (pair[1].xz() - pair[0].xz()).length();
            assert!(
                (w - 7.0).abs() < 1e-9,
                "cross-section is {w} m wide, want 7"
            );
        }
        // **uv IS METRES** (wave ROAD1): `u` is the offset across the
        // carriageway, `v` the arc along it, both in world metres — so a
        // material's `uv_tiling_m` is the whole tiling rule and the road no
        // longer tiles at its own width. It used to be `(0..1, arc / width_m)`,
        // which made one uv unit 14.0 m on the island and the committed asphalt
        // read three and a half times life size (the ASSET0 audit's finding).
        assert_eq!(
            r.uvs[0],
            [-3.5, 0.0],
            "u is the offset in metres, left edge"
        );
        assert_eq!(r.uvs[1], [3.5, 0.0], "…and the right edge is +half");
        let last_v = r.uvs[r.uvs.len() - 1][1];
        assert!(
            (last_v - 100.0).abs() < 1e-4,
            "v at the far end is {last_v} m, want the arc length 100"
        );
        // Faces point up.
        for t in r.indices.chunks_exact(3) {
            let (a, b, c) = (
                r.vertices[t[0] as usize],
                r.vertices[t[1] as usize],
                r.vertices[t[2] as usize],
            );
            assert!((b - a).cross(c - a).y > 0.0, "triangle {t:?} faces down");
        }

        // With no ground answer the spine's own height is used, not zero.
        let mut none = |_: f64, _: f64| None;
        let raised: Vec<DVec3> = (0..3)
            .map(|i| DVec3::new(i as f64 * 10.0, 42.0, 0.0))
            .collect();
        let r = build_ribbon(&raised, 5.0, 0.0, &mut none).unwrap();
        assert!(r.vertices.iter().all(|v| (v.y - 42.0).abs() < 1e-9));
    }

    /// A corner miters rather than producing a torn ribbon, and degenerate
    /// spines are refused instead of emitting NaN vertices.
    #[test]
    fn corners_miter_and_degenerate_spines_are_refused() {
        let mut flat = |_: f64, _: f64| Some(0.0);
        // A right-angle corner.
        let spine = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(50.0, 0.0, 0.0),
            DVec3::new(50.0, 0.0, 50.0),
        ];
        let r = build_ribbon(&spine, 10.0, 0.0, &mut flat).unwrap();
        assert!(
            r.vertices.iter().all(|v| v.is_finite()),
            "a corner produced a NaN vertex"
        );
        // **The corner test has to measure the corner, exactly.**
        //
        // The first version of this arm asserted `(10.0..20.0).contains(&w)` on
        // a 10 m road, and `Range::contains` is `start <= x` — so the *pinched*
        // answer (exactly 10.0, which is what an unscaled bisector offset gives
        // at every corner angle) satisfied it, and so would a runaway. The
        // assertion was structurally incapable of failing in either direction
        // while its own message named the right answer, 14.1. The right answer
        // is the one that gets asserted.
        let corner_w = (r.vertices[3].xz() - r.vertices[2].xz()).length();
        assert!(
            (corner_w - 10.0 * std::f64::consts::SQRT_2).abs() < 1e-9,
            "the mitred corner is {corner_w} m across; a right-angle corner on a \
             10 m road must be exactly 10*sqrt(2) = 14.142. An unscaled bisector \
             offset gives exactly 10.0 here — and then the road's width MEASURED \
             AGAINST EACH LEG is 7.07 m, a visible notch at every bend."
        );

        // …and the property that actually matters: the ribbon is `width_m` wide
        // measured perpendicular to each LEG, through the corner as well as
        // along the straights. That is what a mitre is for, and it is the claim
        // the cross-section width alone cannot make.
        let leg = glam::DVec2::new(1.0, 0.0); // the incoming leg's heading
        let leg_perp = glam::DVec2::new(-leg.y, leg.x);
        let across = (r.vertices[3].xz() - r.vertices[2].xz())
            .dot(leg_perp)
            .abs();
        assert!(
            (across - 10.0).abs() < 1e-9,
            "measured across the incoming leg the corner is {across} m wide, not 10"
        );

        // A hairpin is CLIPPED, not spiked — the miter limit is a real bound.
        let hairpin = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(50.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.5),
        ];
        let h = build_ribbon(&hairpin, 10.0, 0.0, &mut flat).unwrap();
        let spike = (h.vertices[3].xz() - h.vertices[2].xz()).length();
        assert!(
            spike <= 10.0 * MITER_LIMIT + 1e-9,
            "a hairpin ran the miter away to {spike} m; the limit is {}",
            10.0 * MITER_LIMIT
        );
        assert!(h.vertices.iter().all(|v| v.is_finite()));

        // Repeated points collapse, and a spine that is all one point is refused.
        let dup = vec![DVec3::ZERO, DVec3::ZERO, DVec3::ZERO];
        let e = build_ribbon(&dup, 5.0, 0.0, &mut flat)
            .unwrap_err()
            .to_string();
        assert!(e.contains("at least 2 distinct"), "{e}");
        // Non-finite input never becomes a vertex.
        let bad = vec![DVec3::ZERO, DVec3::new(f64::NAN, 0.0, 1.0)];
        assert!(matches!(
            build_ribbon(&bad, 5.0, 0.0, &mut flat),
            Err(crate::GisError::NotFinite(_))
        ));
        // A nonsense width is refused rather than producing an inverted ribbon.
        assert!(build_ribbon(&spine, 0.0, 0.0, &mut flat).is_err());
        assert!(build_ribbon(&spine, f64::NAN, 0.0, &mut flat).is_err());
        // …and so is a nonsense LIFT, which the width check used to leave open:
        // every vertex would carry a NaN `y` whose bounds look perfectly healthy
        // because `f32::min`/`max` ignore NaN.
        assert!(matches!(
            build_ribbon(&spine, 5.0, f64::NAN, &mut flat),
            Err(crate::GisError::NotFinite(_))
        ));
        // A ground query is somebody else's heightfield — a hole or an unloaded
        // tile is exactly where a `Some(NaN)` comes from, and it must not become
        // a vertex either.
        let mut sick = |_: f64, _: f64| Some(f64::NAN);
        assert!(matches!(
            build_ribbon(&spine, 5.0, 0.0, &mut sick),
            Err(crate::GisError::NotFinite(_))
        ));
    }

    /// The nearest-segment query the building-orientation step needs.
    #[test]
    fn the_nearest_centreline_query_finds_the_right_road() {
        let g = RoadGraph::from_layer(&layer(vec![
            GeoFeature::new(line(&[(0.0, 0.0), (100.0, 0.0)])),
            GeoFeature::new(line(&[(0.0, 60.0), (100.0, 60.0)])),
        ]));
        // A lot at z = 10 is nearest the first road, 10 m away, at x = 50.
        let (id, q, d) = g
            .nearest_on_centreline(DVec3::new(50.0, 0.0, 10.0))
            .unwrap();
        assert!((d - 10.0).abs() < 1e-9, "distance {d}");
        assert!(
            (q.x - 50.0).abs() < 1e-9 && q.z.abs() < 1e-9,
            "closest point {q:?}"
        );
        assert_eq!(g.segments[&id].spine[0].z, 0.0, "picked the wrong road");
        // A lot at z = 50 is nearest the second.
        let (id2, _, d2) = g
            .nearest_on_centreline(DVec3::new(50.0, 0.0, 50.0))
            .unwrap();
        assert!((d2 - 10.0).abs() < 1e-9);
        assert_ne!(id, id2);
        // Past the end of a segment the query clamps to the endpoint rather than
        // projecting onto the infinite line.
        let (_, q, _) = g
            .nearest_on_centreline(DVec3::new(500.0, 0.0, 0.0))
            .unwrap();
        assert!(
            (q.x - 100.0).abs() < 1e-9,
            "clamped to the endpoint, got {q:?}"
        );
        // An empty graph has no answer rather than a wrong one.
        assert!(RoadGraph::default()
            .nearest_on_centreline(DVec3::ZERO)
            .is_none());
    }

    /// Unusable features are skipped, counted and NAMED — never dropped quietly.
    #[test]
    fn unusable_features_are_skipped_with_their_reasons() {
        let g = RoadGraph::from_layer(&layer(vec![
            GeoFeature::new(GeoGeometry::Point(DVec3::ZERO)),
            GeoFeature::new(GeoGeometry::Polyline {
                points: vec![DVec3::ZERO],
                closed: false,
            }),
            GeoFeature::new(GeoGeometry::Polyline {
                points: vec![DVec3::ZERO, DVec3::new(f64::NAN, 0.0, 0.0)],
                closed: false,
            }),
            GeoFeature::new(line(&[(0.0, 0.0), (10.0, 0.0)])),
        ]));
        assert_eq!(g.segments.len(), 1, "only the real road survives");
        assert_eq!(g.skipped.len(), 3);
        assert!(g.skipped[0].contains("point"), "{:?}", g.skipped);
        assert!(g.skipped[1].contains("at least 2"), "{:?}", g.skipped);
        assert!(g.skipped[2].contains("non-finite"), "{:?}", g.skipped);
    }

    // ── the network as a NavGraph (NPC1c) ───────────────────────────────────

    /// **One node per junction, one link per segment** — the road network handed
    /// to the search unchanged, not a second graph derived beside it.
    #[test]
    fn the_road_network_is_a_nav_graph_junction_for_junction() {
        let graph = t_junction();
        let nav = graph.nav_graph();
        assert_eq!(
            nav.len(),
            graph.intersections.len(),
            "a junction that is not a node is a place no route can turn at"
        );
        // `edge_count` counts directed halves and a link adds both, so three
        // segments between three distinct pairs are six halves.
        assert_eq!(nav.edge_count(), 2 * graph.segments.len());
        for n in graph.intersections.values() {
            let id = inf_nav::domain::ROAD | n.id;
            assert!(nav.contains(id), "junction {} is missing", n.id);
            assert_eq!(nav.node(id).unwrap().position, n.position);
            assert_eq!(nav.node(id).unwrap().kind, inf_nav::NavKind::Road);
        }
        // Every id belongs to the road domain, which is the whole reason
        // `absorb` can fold a town's grid into an island's roads without welding
        // a junction to a bedroom.
        for n in nav.nodes() {
            assert_eq!(inf_nav::domain::of(n.id), inf_nav::domain::ROAD);
        }
        println!(
            "NPC1c roads: {} junctions ({} of degree 3+), {} segments -> {} nodes / \
             {} directed edges",
            graph.intersections.len(),
            graph.junctions().count(),
            graph.segments.len(),
            nav.len(),
            nav.edge_count()
        );
    }

    /// **A route's cost is the layer's own `length_m`, summed** — the claim
    /// `link_with_cost` is used for, measured rather than argued.
    #[test]
    fn a_route_across_a_chain_costs_the_segments_own_lengths() {
        let graph = t_junction();
        let nav = graph.nav_graph();
        let west = nav.nearest(DVec3::ZERO).expect("Broadway's west end");
        let east = nav
            .nearest(DVec3::new(200.0, 0.0, 0.0))
            .expect("Broadway's east end");
        let r = inf_nav::route(&nav, west, east)
            .route()
            .expect("Broadway is one street in two features");
        // Three nodes: the two ends and the junction the two features share.
        assert_eq!(r.nodes.len(), 3, "{:?}", r.nodes);
        let want: f64 = graph
            .segments
            .values()
            .filter(|s| s.name.starts_with("Broadway"))
            .map(RoadSegment::length_m)
            .sum();
        assert!(
            (r.cost_m - want).abs() < 1e-9,
            "the route cost {} m and the two segments are {want} m",
            r.cost_m
        );
        assert!((want - 200.0).abs() < 1e-9, "the fixture moved: {want} m");
        println!(
            "NPC1c roads: Broadway west->east is {} nodes, {:.3} m of cost against \
             {:.3} m of `length_m`, {:.3} m of walked chain",
            r.nodes.len(),
            r.cost_m,
            want,
            r.path.length_m()
        );
    }

    /// **The spine is spliced into the route**, so an agent handed a switchback
    /// climbs it instead of walking through the hill it goes round.
    #[test]
    fn a_switchbacks_spine_is_spliced_into_the_route() {
        let graph = RoadGraph::from_layer(&layer(vec![GeoFeature::new(line(&[
            (0.0, 0.0),
            (50.0, 80.0),
            (100.0, 0.0),
        ]))]));
        assert_eq!(graph.segments.len(), 1);
        let nav = graph.nav_graph();
        let a = nav.nearest(DVec3::ZERO).unwrap();
        let b = nav.nearest(DVec3::new(100.0, 0.0, 0.0)).unwrap();
        let r = inf_nav::route(&nav, a, b).route().expect("one segment");
        assert_eq!(r.path.points().len(), 3, "the bend was dropped");
        assert_eq!(r.path.points()[1], DVec3::new(50.0, 0.0, 80.0));
        let chord = 100.0;
        assert!(
            r.path.length_m() > chord * 1.8,
            "THE CHORD MUST BE PRICED: a route that cut this corner would be \
             {chord} m and the survey is {} m",
            r.path.length_m()
        );
        // …and the cost the search reports is the segment's own length, which on
        // a flat spine is the chain it walked.
        let seg = graph.segments.values().next().unwrap();
        assert!((r.cost_m - seg.length_m()).abs() < 1e-9);
        assert!((r.cost_m - r.path.length_m()).abs() < 1e-9);
        println!(
            "NPC1c roads: a switchback is {:.3} m of route against a {chord} m chord",
            r.path.length_m()
        );
    }

    /// **A loop carries no route** — both of its endpoints snap onto one
    /// junction, and a self-edge can only ever lengthen a walk. The geometry
    /// survives; only the link is refused.
    #[test]
    fn a_self_looping_segment_carries_no_route() {
        let graph = RoadGraph::from_layer(&layer(vec![GeoFeature::new(line(&[
            (0.0, 0.0),
            (50.0, 0.0),
            (50.0, 50.0),
            (0.0, 0.0),
        ]))]));
        assert_eq!(graph.segments.len(), 1, "the loop is still a road");
        let seg = graph.segments.values().next().unwrap();
        assert_eq!(
            seg.start_node, seg.end_node,
            "the fixture must really close on itself or this arm proves nothing"
        );
        assert!(seg.length_m() > 0.0);
        let nav = graph.nav_graph();
        assert_eq!(nav.len(), 1);
        assert_eq!(nav.edge_count(), 0, "a self-edge reached the graph");
        println!(
            "NPC1c roads: a {:.1} m closed loop is 1 node and 0 edges",
            seg.length_m()
        );
    }

    /// **The road is more than its carriageway** (wave ROAD1, clause 1) — and
    /// every part of it is measured in metres against the thing it models.
    ///
    /// A straight 200 m arterial on a plane, so every number below is a fact
    /// about the builder and not about a terrain. What is asserted:
    ///
    /// * the kerb's upstand is [`KERB_HEIGHT_M`] above the **channel** (which is
    ///   the carriageway's edge, not its crown, and on a cambered road those are
    ///   different heights — reading the crown would report 150 mm and build 80);
    /// * the pavement is [`PAVEMENT_M`] wide and behind the kerb, so a person
    ///   walking the nav ring is walking on concrete;
    /// * the crown is `crown_fall · half` above the channel;
    /// * the paint sits above the surface it is painted on and nowhere else;
    /// * the two colours are the two colours: yellow only down the middle.
    #[test]
    fn a_kerbed_road_carries_a_kerb_a_pavement_a_crown_and_its_paint() {
        let graph = RoadGraph::from_layer(&layer(vec![road(
            &[(0.0, 0.0), (200.0, 0.0)],
            "arterial",
            "Main",
        )]));
        let seg = graph.segments.values().next().expect("one segment");
        let half = seg.width_m() * 0.5;
        let opts = SurfaceOptions {
            crown_fall: DEFAULT_CROWN_FALL,
            furniture: true,
            ..SurfaceOptions::default()
        };
        // A plane at y = 0, so the ground contributes nothing to any number.
        let mut flat = |_x: f64, _z: f64| Some(0.0);
        let surface = build_surface(&graph, &opts, &mut flat);

        // ── the carriageway is CROWNED ───────────────────────────────────────
        let road = surface
            .parts
            .get(&RoadKind::Arterial)
            .expect("the arterial paved");
        let crown = road
            .vertices
            .iter()
            .map(|v| v.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let channel = road
            .vertices
            .iter()
            .map(|v| v.y)
            .fold(f64::INFINITY, f64::min);
        let want_crown = opts.lift_m + DEFAULT_CROWN_FALL * half;
        assert!(
            (crown - want_crown).abs() < 1e-9 && (channel - opts.lift_m).abs() < 1e-9,
            "the carriageway's crown is {crown} m and its channel {channel} m; \
             want {want_crown} and {} — a road that is flat across is a painted \
             strip of ground, and one whose EDGES are raised is a gutter down \
             the middle",
            opts.lift_m
        );

        // ── the kerb ────────────────────────────────────────────────────────
        let kerb = surface
            .furniture
            .get(&RoadPart::Kerb)
            .expect("a kerbed class builds a kerb");
        assert!(
            RoadKind::Arterial.is_kerbed() && !RoadKind::Highway.is_kerbed(),
            "a street is kerbed and a motorway is not"
        );
        // The kerb's top is the channel plus the upstand — measured against the
        // CHANNEL, because that is what a kerb is laid against.
        let top = kerb
            .vertices
            .iter()
            .map(|v| v.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let want_top = opts.lift_m + KERB_HEIGHT_M + PAVEMENT_M * PAVEMENT_FALL;
        assert!(
            (top - want_top).abs() < 1e-9,
            "the pavement's back edge is at {top} m; want {want_top} — the \
             kerb's {KERB_HEIGHT_M} m upstand over the channel plus the \
             footway's own {PAVEMENT_FALL} cross-fall over {PAVEMENT_M} m"
        );
        // …and it reaches out exactly as far as the nav ring does.
        let reach = kerb
            .vertices
            .iter()
            .map(|v| v.z.abs())
            .fold(0.0f64, f64::max);
        let want_reach = half + KERB_WIDTH_M + PAVEMENT_M;
        assert!(
            (reach - want_reach).abs() < 1e-6,
            "the pavement's back edge is {reach} m from the centreline; want \
             {want_reach} — half the {} m carriageway, a {KERB_WIDTH_M} m kerb \
             stone and {PAVEMENT_M} m of footway",
            seg.width_m()
        );
        // The skirt closes the cliff: some vertex at the back edge is ON the
        // ground, or you see through a 190 mm gap at every property line.
        assert!(
            kerb.vertices
                .iter()
                .any(|v| v.z.abs() > want_reach - 1e-6 && v.y.abs() < 1e-9),
            "the pavement's outer edge does not reach the ground — the slab is \
             a floating slice with nothing under it"
        );

        // ── the paint ───────────────────────────────────────────────────────
        let yellow = surface
            .furniture
            .get(&RoadPart::MarkingYellow)
            .expect("an arterial has a centre line");
        let white = surface
            .furniture
            .get(&RoadPart::MarkingWhite)
            .expect("…and edge lines");
        // **Yellow is only ever down the middle.** It is what separates opposing
        // traffic, and a yellow edge line would be a road that reads as a
        // one-way carriageway.
        let widest_yellow = yellow
            .vertices
            .iter()
            .map(|v| v.z.abs())
            .fold(0.0f64, f64::max);
        // The island's arterials take `default_lanes` = 4 until the layer says
        // otherwise, so the centre line here is a DOUBLE yellow: two lines a
        // `DOUBLE_LINE_GAP_M` apart, reaching half that plus a line's width.
        let centre_reach = (DOUBLE_LINE_GAP_M + 2.0 * LINE_WIDTH_M) * 0.5;
        assert!(
            widest_yellow <= centre_reach + 1e-9,
            "yellow paint reaches {widest_yellow} m from the crown and the \
             centre line is {centre_reach} m half-wide; yellow separates \
             OPPOSING traffic and must never reach an edge"
        );
        // White is at the edges, inside the carriageway.
        let widest_white = white
            .vertices
            .iter()
            .map(|v| v.z.abs())
            .fold(0.0f64, f64::max);
        assert!(
            (widest_white - (half - EDGE_LINE_INSET_M + LINE_WIDTH_M * 0.5)).abs() < 1e-6,
            "the edge line's outer side is {widest_white} m out; want \
             {} — inside the carriageway by {EDGE_LINE_INSET_M} m",
            half - EDGE_LINE_INSET_M + LINE_WIDTH_M * 0.5
        );
        // Every painted vertex floats above the surface under it and by exactly
        // the marking lift — paint that conformed would z-fight, and paint that
        // ignored the crown would float at the crown and sink at the channel.
        for r in [yellow, white] {
            for v in &r.vertices {
                let under = opts.lift_m + DEFAULT_CROWN_FALL * (half - v.z.abs()).max(0.0);
                assert!(
                    (v.y - (under + MARKING_LIFT_M)).abs() < 1e-9,
                    "a marking at {} m across sits {} m up; the cambered surface \
                     under it is at {under} m and the paint is {MARKING_LIFT_M} m \
                     proud of it",
                    v.z,
                    v.y
                );
            }
        }

        // ── and a road that asks for none gets none ──────────────────────────
        let bare = build_surface(&graph, &SurfaceOptions::default(), &mut flat);
        assert!(
            bare.furniture.is_empty() && bare.furniture_triangle_count() == 0,
            "`SurfaceOptions::furniture` defaults to false, so every pre-ROAD1 \
             caller builds the road it already had"
        );
        assert_eq!(
            bare.parts
                .get(&RoadKind::Arterial)
                .map(|r| r.vertices.len()),
            road.vertices.len().into(),
            "…and the carriageway itself is the same mesh either way"
        );
    }

    /// **An open road has a shoulder where a street has a kerb** (wave ROAD1),
    /// and the shoulder is asphalt rather than a fourth material.
    ///
    /// The distinction is the class's, not an attribute's: a motorway has no
    /// footway by definition. What separates the shoulder from a running lane is
    /// the solid white edge line, which is painted at the **carriageway's** edge
    /// and not at the shoulder's — measured here, because painting it at the
    /// outer edge is the mistake that makes a hard shoulder look like a lane.
    #[test]
    fn an_open_road_gets_a_shoulder_and_a_street_gets_a_kerb() {
        let graph = RoadGraph::from_layer(&layer(vec![road(
            &[(0.0, 0.0), (200.0, 0.0)],
            "highway",
            "Trunk",
        )]));
        let seg = graph.segments.values().next().expect("one segment");
        let half = seg.width_m() * 0.5;
        let opts = SurfaceOptions {
            crown_fall: DEFAULT_CROWN_FALL,
            furniture: true,
            ..SurfaceOptions::default()
        };
        let mut flat = |_x: f64, _z: f64| Some(0.0);
        let surface = build_surface(&graph, &opts, &mut flat);

        assert!(
            !surface.furniture.contains_key(&RoadPart::Kerb),
            "a motorway does not have a footway"
        );
        let road = surface.parts.get(&RoadKind::Highway).expect("paved");
        let reach = road
            .vertices
            .iter()
            .map(|v| v.z.abs())
            .fold(0.0f64, f64::max);
        let want = half + RoadKind::Highway.shoulder_m();
        assert!(
            (reach - want).abs() < 1e-6,
            "the sealed surface reaches {reach} m; want {want} — half the \
             carriageway plus a {} m shoulder, in the SAME ribbon because a \
             sealed shoulder is asphalt",
            RoadKind::Highway.shoulder_m()
        );

        // The edge line marks the carriageway, not the shoulder.
        let white = surface
            .furniture
            .get(&RoadPart::MarkingWhite)
            .expect("a highway is marked");
        let widest = white
            .vertices
            .iter()
            .map(|v| v.z.abs())
            .fold(0.0f64, f64::max);
        assert!(
            widest < half,
            "the edge line's outer side is {widest} m out and the carriageway's \
             edge is {half} m — a line painted at the shoulder's edge makes the \
             shoulder read as a running lane"
        );
        // Four lanes, so a DOUBLE yellow and one white lane divider each side.
        assert_eq!(seg.lane_count, 4, "the class's default");
        let yellow = surface
            .furniture
            .get(&RoadPart::MarkingYellow)
            .expect("opposing traffic is separated in yellow");
        let mut centres: Vec<f64> = yellow.vertices.iter().map(|v| v.z).collect();
        centres.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let span = centres.last().unwrap() - centres.first().unwrap();
        assert!(
            (span - (DOUBLE_LINE_GAP_M + 2.0 * LINE_WIDTH_M)).abs() < 1e-9,
            "the centre line spans {span} m; a double yellow is two \
             {LINE_WIDTH_M} m lines {DOUBLE_LINE_GAP_M} m apart"
        );
        // The lane divider is dashed: painted 3 m in 9, so it is broken up into
        // many disjoint runs rather than one strip.
        let dashes = white.indices.len() / 6;
        assert!(
            dashes > 20,
            "the white paint is {dashes} quads — a dashed divider over 200 m at \
             a {DASH_M} m dash and a {DASH_GAP_M} m gap is many, and one long \
             strip would be a solid line"
        );
    }

    /// **A junction of three or more legs gets its crossings** (wave ROAD1) —
    /// and a bend and a dead end do not.
    ///
    /// Degree 3 is the same threshold `fill_junctions` uses, for the same
    /// reason: a degree-2 node is a bend in one road and nobody crosses a bend.
    #[test]
    fn a_real_junction_gets_a_crossing_and_a_bend_does_not() {
        // A T: two arms meeting a through-route at one node.
        let graph = RoadGraph::from_layer(&layer(vec![
            road(&[(-200.0, 0.0), (0.0, 0.0)], "arterial", "West"),
            road(&[(0.0, 0.0), (200.0, 0.0)], "arterial", "East"),
            road(&[(0.0, 0.0), (0.0, 200.0)], "arterial", "South"),
        ]));
        let opts = SurfaceOptions {
            crown_fall: DEFAULT_CROWN_FALL,
            furniture: true,
            ..SurfaceOptions::default()
        };
        let mut flat = |_x: f64, _z: f64| Some(0.0);
        let crossed = build_surface(&graph, &opts, &mut flat);
        let with_junction = crossed
            .furniture
            .get(&RoadPart::MarkingWhite)
            .map(|r| r.triangle_count())
            .unwrap_or(0);

        // The control: the same three roads with the junction pulled apart, so
        // no node has three legs. Same length of road, same edge lines, no
        // crossings.
        let apart = layer(vec![
            road(&[(-200.0, -1000.0), (0.0, -1000.0)], "arterial", "West"),
            road(&[(0.0, 1000.0), (200.0, 1000.0)], "arterial", "East"),
            road(&[(900.0, 0.0), (900.0, 200.0)], "arterial", "South"),
        ]);
        let lone = build_surface(&RoadGraph::from_layer(&apart), &opts, &mut flat);
        let without = lone
            .furniture
            .get(&RoadPart::MarkingWhite)
            .map(|r| r.triangle_count())
            .unwrap_or(0);

        assert!(
            with_junction > without,
            "the T junction painted {with_junction} white triangles and three \
             disjoint roads of the same length painted {without} — a crossing \
             is what the difference is, and no difference means none was laid"
        );
        // And the bars really are at the junction: white paint within the
        // setback band, across the carriageway, on all three legs.
        let white = crossed
            .furniture
            .get(&RoadPart::MarkingWhite)
            .expect("white paint");
        let near_node = white
            .vertices
            .iter()
            .filter(|v| {
                let d = (v.x * v.x + v.z * v.z).sqrt();
                d > CROSSWALK_SETBACK_M - 1.0 && d < CROSSWALK_SETBACK_M + CROSSWALK_DEPTH_M + 1.0
            })
            .count();
        assert!(
            near_node >= 3 * 4,
            "only {near_node} painted vertices sit in the crossing band around \
             the junction; three legs of bars is many more"
        );
    }

    // ── the footway is not drawn on the road (audit ROAD1) ──────────────────

    /// Two arterials that **cross without meeting**: 200 m along +X at `z = 0`,
    /// 200 m along +Z at `x = 100`, sharing no endpoint.
    ///
    /// That is the graph's own rule, stated one fixture above: nodes come from
    /// segment *endpoints*, so a road that merely passes through another's line
    /// makes no junction and gets no fan. Both ribbons are simply laid, and they
    /// overlap — which is the case the island's own network is full of, at every
    /// switchback and every pair of routes that share a valley.
    fn crossing_pair() -> RoadGraph {
        RoadGraph::from_layer(&layer(vec![
            road(&[(0.0, 0.0), (200.0, 0.0)], "arterial", "Broadway"),
            road(&[(100.0, -100.0), (100.0, 100.0)], "arterial", "Elm"),
        ]))
    }

    /// Every carriageway triangle of `s`, in plan.
    fn carriageway_plan(s: &RoadSurface) -> Vec<[glam::DVec2; 3]> {
        let mut out = Vec::new();
        for r in s.parts.values() {
            for t in r.indices.chunks_exact(3) {
                out.push([
                    r.vertices[t[0] as usize].xz(),
                    r.vertices[t[1] as usize].xz(),
                    r.vertices[t[2] as usize].xz(),
                ]);
            }
        }
        out
    }

    /// Does any triangle of `r` cover `p` in plan?
    fn ribbon_covers(r: &RoadRibbon, p: glam::DVec2) -> bool {
        r.indices.chunks_exact(3).any(|t| {
            plan_contains(
                &[
                    r.vertices[t[0] as usize].xz(),
                    r.vertices[t[1] as usize].xz(),
                    r.vertices[t[2] as usize].xz(),
                ],
                p,
            )
        })
    }

    /// **A FOOTWAY IS NEVER DRAWN ON A CARRIAGEWAY** (audit ROAD1).
    ///
    /// # What this caught
    ///
    /// `build_segment_furniture` lays a kerb and 2 m of concrete beside *its
    /// own* segment and cannot see any other, so where two roads pass within
    /// `built_half_width + half` — two crossing routes, or the two limbs of one
    /// road at a switchback — each laid its footway across the other's asphalt,
    /// 190 mm above it. Measured on the shipped island before this clip:
    /// **19 754.6 m² of 170 901.7 m² (11.56 %)**, reaching within 0.02 m of a
    /// 7 m road's own crown; on the fixture, 969.5 m² of 8 247.6 m² (11.75 %).
    /// A 1080p capture of Harbour City's main street showed it as a grey plank
    /// lying diagonally across the road.
    ///
    /// # The three claims, and why the third one is here
    ///
    /// (a) the clip **did something** on a fixture built to make it — a clip
    /// that fired zero times would satisfy (b) vacuously; (b) nothing of the
    /// footway is left on the asphalt; and (c) the footway **beside** each road
    /// survived, because a clip that deleted the whole ribbon would also satisfy
    /// (b). (c) is checked at the exact pair of points that separates the two:
    /// one slab position on Broadway's north footway inside Elm's carriageway,
    /// which must be gone, and the same footway 20 m west of Elm, which must
    /// not be.
    ///
    /// Falsification, measured both ways: dropping the
    /// `clip_kerbs_to_open_ground` call in `build_surface` reds (a); applying
    /// the inset per triangle instead of to the union reds (b), which is the
    /// defect the first version of this clip actually had; and setting
    /// [`KERB_CLIP_INSET_M`] to zero reds (d) — and *only* (d), which is why (d)
    /// is here: the kerb's own vertical face has a plan footprint that IS the
    /// carriageway's edge, so a zero inset deletes the upstand off the whole
    /// island while (a), (b) and (c) all stay green.
    #[test]
    fn a_footway_is_never_drawn_on_a_carriageway() {
        let opts = SurfaceOptions {
            furniture: true,
            ..Default::default()
        };
        let mut flat = |_x: f64, _z: f64| Some(0.0);
        let surface = build_surface(&crossing_pair(), &opts, &mut flat);
        let kerb = surface
            .furniture
            .get(&RoadPart::Kerb)
            .expect("two arterials carry kerbs");

        // (a) THE ANTI-VACUITY FLOOR. Four footways crossing two 7 m
        //     carriageways is a real overlap, and a fixture that produced none
        //     would make (b) mean nothing.
        assert!(
            surface.kerbs_clipped > 0,
            "the clip dropped nothing on a fixture built out of two roads that \
             cross — either the crossing stopped overlapping or the clip stopped \
             running, and (b) below proves nothing either way"
        );

        // (b) THE PROPERTY.
        let road = carriageway_plan(&surface);
        for t in kerb.indices.chunks_exact(3) {
            let c = (kerb.vertices[t[0] as usize]
                + kerb.vertices[t[1] as usize]
                + kerb.vertices[t[2] as usize])
                / 3.0;
            let p = c.xz();
            assert!(
                !is_covered_by(p, KERB_CLIP_INSET_M, |q| road
                    .iter()
                    .any(|tri| plan_contains(tri, q))),
                "a footway triangle sits on the carriageway at ({:.2}, {:.2}) — \
                 that is 190 mm of concrete over the surface a car drives on",
                p.x,
                p.y
            );
        }

        // (c) THE CONTROL. The middle of Broadway's north slab: on Elm's
        //     carriageway at x = 100, and 20 m clear of it at x = 80.
        // The layer states no lane count, so both roads take the class default —
        // derived here rather than written as a number, so a change to the
        // default moves the probe with it.
        let half = RoadKind::Arterial.width_m(RoadKind::Arterial.default_lanes()) * 0.5;
        let slab = half + KERB_WIDTH_M + PAVEMENT_M * 0.5;
        assert!(
            !ribbon_covers(kerb, glam::DVec2::new(100.0, slab)),
            "Broadway's footway is still drawn across Elm at (100.0, {slab:.2})"
        );
        assert!(
            ribbon_covers(kerb, glam::DVec2::new(80.0, slab)),
            "Broadway's footway is gone at (80.0, {slab:.2}), 20 m clear of any \
             other road — the clip ate the pavement it was meant to protect"
        );
        assert!(
            ribbon_covers(kerb, glam::DVec2::new(100.0 + slab, -50.0)),
            "Elm's own footway is gone at ({:.2}, -50.0)",
            100.0 + slab
        );

        // (d) THE KERB'S OWN FACE, which is the whole reason the inset is not
        //     zero. It is a vertical wall whose plan footprint IS the
        //     carriageway's edge line, so a containment test taken at the
        //     centroid with no inset finds every one of them "on the road" and
        //     deletes the kerb from the entire island. Counted on a stretch of
        //     Broadway well clear of Elm.
        let faces = kerb
            .indices
            .chunks_exact(3)
            .filter(|t| {
                t.iter().all(|&i| {
                    let v = kerb.vertices[i as usize];
                    (60.0..90.0).contains(&v.x) && (v.z.abs() - half).abs() < 1.0e-9
                })
            })
            .count();
        assert!(
            faces > 0,
            "no kerb face survives on the 30 m of Broadway between x = 60 and \
             x = 90 — the clip measured containment with no inset and took the \
             150 mm upstand with it"
        );
    }
}
