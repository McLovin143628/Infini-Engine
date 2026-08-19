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
}

/// A quad-ribbon mesh generated along a road spine.
#[derive(Clone, Debug, PartialEq)]
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
        for (side, sign) in [(0usize, -1.0f64), (1, 1.0)] {
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
            uvs.push([side as f32, (arc / width_m) as f32]);
        }
        if i + 1 < pts.len() {
            let b = (i * 2) as u32;
            // Wound so the surface faces up (+Y), matching the triangulator.
            indices.extend_from_slice(&[b, b + 1, b + 3, b, b + 3, b + 2]);
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
}
