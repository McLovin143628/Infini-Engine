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
/// every corner down to about 29 degrees; sharper than that the corner is
/// **clipped rather than spiked**, which is wrong by a little at a bend no
/// survey puts a road through, instead of wrong by a lot.
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
                .attr_number(&[
                    "lane_count",
                    "lanes",
                    "nlanes",
                    "num_lanes",
                    "through_lanes",
                ])
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
    build_ribbon_across(spine, width_m, lift_m, 1, height_at)
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

    let half = width_m * 0.5;
    let mut vertices = Vec::with_capacity(pts.len() * 2);
    let mut uvs = Vec::with_capacity(pts.len() * 2);
    let mut indices = Vec::with_capacity((pts.len() - 1) * 6);
    let mut arc = 0.0f64;

    for i in 0..pts.len() {
        if i > 0 {
            arc += (pts[i].xz() - pts[i - 1].xz()).length();
        }
        // The cross-section direction is the average of the incoming and outgoing
        // headings, so an interior corner miters rather than producing two
        // overlapping quads with a gap between them.
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
        // **A mitre has to be SCALED, not just aimed.** Offsetting by `half`
        // along the bisector is the mistake that looks right: the bisector is a
        // unit vector, so the two cross-section vertices are always exactly
        // `width_m` apart — and their distance from each *leg*'s centreline is
        // `half · cos(θ/2)`, so the road PINCHES at every corner. Measured on a
        // right-angle bend of a 10 m road: 7.07 m through the corner, a visible
        // notch at every bend in every ribbon the importer produced. The
        // correction is `1 / (bisector · leg)`, which is 1 on a straight run and
        // √2 at a right angle.
        let miter = {
            let leg = if fwd_dir.length_squared() > 1e-12 {
                fwd_dir
            } else {
                back_dir
            };
            let c = dir.dot(leg).abs();
            // A hairpin has `c → 0` and would spike to infinity. Clipped rather
            // than spiked, at the usual limit — a corner sharper than ~29° gets
            // a blunt end instead of a spear, which is wrong by a little in a
            // place no survey puts a road, rather than wrong by a lot.
            if c > 1.0 / MITER_LIMIT {
                1.0 / c
            } else {
                MITER_LIMIT
            }
        };
        for k in 0..=strips {
            // `u` runs 0 (left kerb) to 1 (right kerb) across the carriageway.
            let u = k as f64 / strips as f64;
            let sign = u * 2.0 - 1.0;
            let xz = pts[i].xz() + perp * (half * miter * sign);
            // The terrain callback is somebody else's function reading somebody
            // else's heightfield; a query over a voxel hole or an unloaded tile
            // is exactly where a `Some(NaN)` comes from, and it would become a
            // NaN vertex with healthy-looking bounds.
            let ground = match height_at(xz.x, xz.y) {
                Some(h) if h.is_finite() => h,
                Some(h) => {
                    return Err(crate::GisError::NotFinite(format!(
                        "the ground query under the road at ({}, {}) returned the \
                         non-finite height {h}",
                        xz.x, xz.y
                    )))
                }
                None => pts[i].y,
            };
            vertices.push(DVec3::new(xz.x, ground + lift_m, xz.y));
            uvs.push([u as f32, (arc / width_m) as f32]);
        }
        if i + 1 < pts.len() {
            let row = (strips + 1) as u32;
            let base = (i * (strips + 1)) as u32;
            for k in 0..strips as u32 {
                let a = base + k;
                // Wound so the surface faces up (+Y), matching the triangulator.
                // Identical to the two-vertex form for `strips == 1`.
                indices.extend_from_slice(&[a, a + 1, a + row + 1, a, a + row + 1, a + row]);
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
    /// How many junctions got a fan.
    pub junctions_filled: usize,
    /// Junctions that could not be filled, and why.
    pub junctions_skipped: usize,
    /// Segments with no buildable surface, named.
    pub skipped: Vec<String>,
}

impl RoadSurface {
    pub fn triangle_count(&self) -> usize {
        self.parts.values().map(RoadRibbon::triangle_count).sum()
    }
    pub fn vertex_count(&self) -> usize {
        self.parts.values().map(|r| r.vertices.len()).sum()
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
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self {
            lift_m: DEFAULT_ROAD_LIFT_M,
            ground_step_m: DEFAULT_GROUND_STEP_M,
            fill_junctions: true,
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
        match build_ribbon_across(&dense, s.width_m(), opts.lift_m, strips, height_at) {
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
    out
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
        // v tiles by arc length / width, so markings are a constant physical size.
        assert_eq!(r.uvs[0], [0.0, 0.0]);
        assert_eq!(r.uvs[1], [1.0, 0.0]);
        let last_v = r.uvs[r.uvs.len() - 1][1];
        assert!(
            (last_v - (100.0 / 7.0) as f32).abs() < 1e-4,
            "v at the far end is {last_v}, want 100/7"
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
}
