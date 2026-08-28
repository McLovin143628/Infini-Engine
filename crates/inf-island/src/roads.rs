//! The road network: routed against a grade bound, then audited against it.
//!
//! # "Designed data, not OSM-imported" — what that means in practice
//!
//! The *design* is the site list, the class of each link and the grade ceiling.
//! The *route* is derived, because a route is a fact about the ground and the
//! ground is 51 million samples of real survey. A human drawing a highway across
//! this island by eye would draw one that climbs at 30 % somewhere, and the way
//! you find out is a car that cannot get up it.
//!
//! So: [`route`] plans the links once against the carved terrain, the result is
//! committed as `roads.geojson` — the design artifact — and every build
//! thereafter reads that file and **audits** it with [`grade_audit`]. The
//! derivation is re-runnable; the committed layer is what ships.
//!
//! # Where the switchbacks come from
//!
//! Nowhere special. The router forbids any step whose grade exceeds the ceiling,
//! so on a face too steep to climb directly the only remaining moves are across
//! it — and a path that has to gain height by traversing is a switchback. It is
//! not a feature that was added; it is what a shortest path under a slope
//! constraint *is*.
//!
//! # What is NOT built, stated plainly
//!
//! **There are no bridges and no tunnels.** A link whose two ends have no
//! land route between them under the grade ceiling is **refused by name** with
//! both ends' positions, because the alternative — routing it through the sea and
//! draping the result on the sea floor — is a road at the bottom of the ocean
//! that nothing in the pipeline would complain about.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::{DVec2, DVec3};

use crate::recipe::{IslandRecipe, Site};
use crate::terrain::CoarseHeights;
use crate::IslandError;

/// One planned or committed link.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub name: String,
    /// `"highway"`, `"arterial"`, … — an `inf_gis::RoadKind` label.
    pub class: String,
    /// World XZ + the ground height the route was planned at.
    pub points: Vec<DVec3>,
}

impl Route {
    /// Plan-view length, metres.
    pub fn length_m(&self) -> f64 {
        self.points
            .windows(2)
            .map(|w| (DVec2::new(w[1].x, w[1].z) - DVec2::new(w[0].x, w[0].z)).length())
            .sum()
    }
}

/// The grade the audit found on one stretch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradeSpan {
    /// Which route, by index into the audited set.
    pub route: usize,
    /// Distance along the route where the stretch starts, metres.
    pub from_m: f64,
    /// Rise over run.
    pub grade: f64,
    /// Where it is, so an author can go and look.
    pub at: DVec2,
}

/// What the grade audit measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GradeAudit {
    /// The steepest grade found anywhere.
    pub worst: f64,
    /// Where the worst one is.
    pub worst_at: DVec2,
    /// Every stretch above the recipe's ceiling.
    pub over: Vec<GradeSpan>,
    /// How many stretches were measured.
    pub samples: usize,
    /// The ceiling that was applied.
    pub ceiling: f64,
    /// Stretches whose ground could not be sampled (off the terrain).
    pub off_terrain: usize,
}

impl GradeAudit {
    /// `true` when nothing exceeded the ceiling.
    pub fn is_clean(&self) -> bool {
        self.over.is_empty()
    }

    /// The fraction of measured stretches above the ceiling.
    pub fn over_fraction(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.over.len() as f64 / self.samples as f64
    }
}

/// The network, in numbers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadReport {
    /// `(class label, kilometres)`, ascending by class.
    pub km_by_class: Vec<(String, f64)>,
    pub total_km: f64,
    pub segments: usize,
    pub junctions: usize,
    pub audit: GradeAudit,
}

/// A cell's cost of being on a route.
///
/// The gentler a road is the longer it is, so a router that only minimised
/// length would take every slope it was allowed to. The grade term is what makes
/// it prefer a flatter line when one exists — 8× the run at the ceiling, which
/// says "a road at the legal limit costs as much as eight level ones", and that
/// is what produces a switchback rather than a staircase at exactly the ceiling.
const GRADE_COST_GAIN: f64 = 8.0;

/// Plan at this fraction of the ceiling the audit will apply.
///
/// # THE ROUTER AND THE AUDIT DO NOT LOOK AT THE SAME GROUND
///
/// The router plans on the [`DERIVATION_PITCH_M`](crate::DERIVATION_PITCH_M)
/// lattice — eight metres — and the audit measures the terrain itself, which is
/// one metre a sample and bilinear between them. Detail the lattice cannot see is
/// detail the route was planned in ignorance of, and it is exactly where a
/// grade goes over.
///
/// The margin is that gap, priced conservatively, and **it is not the hairpin**:
/// the apex problem is a real one and it is fixed in `smooth`, not paid for
/// here. The two were confused in this wave's first draft, and the confusion is
/// worth recording — a margin that stands in for a defect is a defect that never
/// gets fixed. Measured: with the apex guard in place, a route planned *at* the
/// ceiling on the 15 % fixture audits **clean at 0.0671**; without it, the same
/// route audits at **0.1500, the full fall line, on 24 of 336 stretches**.
pub const PLAN_GRADE_MARGIN: f64 = 0.85;

/// Refuse a step that would put a road under water by this much.
///
/// Zero, because "under water" is not a matter of degree. Named as a constant so
/// the day a bridge exists there is one place to relax it.
const MAX_SUBMERGENCE_M: f64 = 0.0;

/// How far a single routing step may reach, in lattice cells.
///
/// # THE EIGHT-NEIGHBOUR ROUTER CANNOT BUILD A SWITCHBACK, and this is why
///
/// On ground of uniform gradient `g`, a D8 step achieves exactly two grades:
/// `g` along the fall line and `g/√2 = 0.707 g` across it. **There is no third
/// option.** So a ceiling below `0.707 g` makes the slope impassable and the
/// router answers "no route" — which is what it did on a 15 % hillside under an
/// 8 % ceiling, a hillside every real road in the world crosses by traversing.
///
/// A step of `(1, r)` cells achieves `g / √(1 + r²)`. At `r = 4` that is
/// `0.243 g`, so an 8 % road can be built on ground up to 33 %. That number is
/// the reach's whole justification and
/// `an_eight_neighbour_router_cannot_traverse_a_uniform_slope` measures it from
/// both sides.
///
/// The cost: a step is a **chord**, so the grade it is admitted on is the
/// chord's *average*. Every cell the chord crosses is still checked for land, so
/// a long step cannot leap a river — but it can average over a dip, which is
/// exactly what a road on an embankment does, and what the corridor levelling
/// then builds.
pub const ROUTE_REACH_CELLS: i32 = 4;

/// The step set: every primitive vector inside the reach.
///
/// Primitive (`gcd == 1`) because `(2, 2)` is `(1, 1)` twice and admitting both
/// doubles the edge count for no reachable position — and, worse, lets a
/// composite step average over a cell a primitive one would have had to check.
fn steps() -> Vec<(i32, i32)> {
    fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut v = Vec::new();
    for dz in -ROUTE_REACH_CELLS..=ROUTE_REACH_CELLS {
        for dx in -ROUTE_REACH_CELLS..=ROUTE_REACH_CELLS {
            if (dx, dz) == (0, 0) {
                continue;
            }
            if gcd(dx.abs(), dz.abs()) == 1 {
                v.push((dx, dz));
            }
        }
    }
    v
}

/// Plan one link between two world positions, under a grade ceiling.
///
/// Dijkstra on the coarse height grid with steps past the ceiling deleted. The
/// answer is a polyline through cell centres, decimated and smoothed.
pub fn route(
    heights: &CoarseHeights,
    sea_level_m: f64,
    from: DVec2,
    to: DVec2,
    max_grade: f64,
    stride: usize,
) -> Result<Vec<DVec3>, IslandError> {
    let (nx, nz) = (heights.nx, heights.nz);
    let n = nx * nz;
    let cell = |p: DVec2| -> Option<usize> {
        let i = ((p.x - heights.min.x) / heights.pitch).round();
        let j = ((p.y - heights.min.y) / heights.pitch).round();
        if i < 0.0 || j < 0.0 || i as usize >= nx || j as usize >= nz {
            return None;
        }
        Some(j as usize * nx + i as usize)
    };
    let start = cell(from).ok_or_else(|| {
        IslandError::Settings(format!("the route start {from:?} is off the world"))
    })?;
    let goal = cell(to)
        .ok_or_else(|| IslandError::Settings(format!("the route end {to:?} is off the world")))?;

    let passable =
        |k: usize| heights.known[k] && f64::from(heights.h[k]) > sea_level_m + MAX_SUBMERGENCE_M;
    if !passable(start) || !passable(goal) {
        return Err(IslandError::Settings(format!(
            "a route endpoint is under water: {from:?} -> {to:?}. There are no \
             bridges in this generator, so a link needs land at both ends."
        )));
    }

    let steps = steps();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![u32::MAX; n];
    let mut heap: BinaryHeap<Reverse<(OrderedF64, u32)>> = BinaryHeap::new();
    dist[start] = 0.0;
    heap.push(Reverse((OrderedF64(0.0), start as u32)));
    while let Some(Reverse((OrderedF64(d), ki))) = heap.pop() {
        let k = ki as usize;
        if d > dist[k] {
            continue;
        }
        if k == goal {
            break;
        }
        let (i, j) = (k % nx, k / nx);
        for (dx, dz) in &steps {
            let (ni, nj) = (i as i64 + *dx as i64, j as i64 + *dz as i64);
            if ni < 0 || nj < 0 || ni as usize >= nx || nj as usize >= nz {
                continue;
            }
            let nk = nj as usize * nx + ni as usize;
            if !passable(nk) {
                continue;
            }
            // A long step is a CHORD. Every cell it crosses has to be land, or a
            // road would leap a river; the grade it is admitted on is the
            // chord's own average, which is what an embankment is.
            let span = dx.abs().max(dz.abs());
            let mut blocked = false;
            for t in 1..span {
                let mi = i as i64 + (*dx as i64 * t as i64) / span as i64;
                let mj = j as i64 + (*dz as i64 * t as i64) / span as i64;
                if mi < 0 || mj < 0 || mi as usize >= nx || mj as usize >= nz {
                    blocked = true;
                    break;
                }
                if !passable(mj as usize * nx + mi as usize) {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }
            let run = (f64::from(*dx) * f64::from(*dx) + f64::from(*dz) * f64::from(*dz)).sqrt()
                * heights.pitch;
            let rise = (f64::from(heights.h[nk]) - f64::from(heights.h[k])).abs();
            let grade = rise / run;
            if grade > max_grade {
                continue;
            }
            let cost = run * (1.0 + GRADE_COST_GAIN * grade / max_grade);
            let nd = d + cost;
            if nd < dist[nk] {
                dist[nk] = nd;
                prev[nk] = k as u32;
                heap.push(Reverse((OrderedF64(nd), nk as u32)));
            }
        }
    }
    if !dist[goal].is_finite() {
        return Err(IslandError::Settings(format!(
            "no route from {from:?} to {to:?} holds a grade of {max_grade:.3} \
             without leaving land. Either the sites are across water (this \
             generator builds no bridges), or the ceiling is tighter than the \
             ground allows — raise `[roads] max_grade`, or move the site."
        )));
    }

    let mut cells = vec![goal];
    let mut k = goal;
    while k != start {
        k = prev[k] as usize;
        cells.push(k);
    }
    cells.reverse();

    let stride = stride.max(1);
    let mut pts: Vec<DVec3> = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        if i % stride == 0 || i + 1 == cells.len() {
            let p = heights.position(c % nx, c / nx);
            pts.push(DVec3::new(p.x, f64::from(heights.h[*c]), p.y));
        }
    }
    Ok(smooth(&pts))
}

/// One pass of a corner-cutting smooth, keeping the endpoints **and the
/// apexes**.
///
/// A chord-stepping path is still a little angular; a road is not. Chaikin-style
/// corner cutting with the ends pinned turns it into a line a car can follow, and
/// it is pure arithmetic (the portability law).
///
/// # A SWITCHBACK'S APEX MUST NOT BE CUT
///
/// The apex of a switchback is a **reversal**: the two chords that meet there
/// point back along each other. Averaging that vertex with its neighbours moves
/// it toward their midpoint, which lies straight up the fall line — so cutting
/// the corner turns two legal traverses into one illegal climb. Measured on the
/// 15 % fixture: cutting every corner produced apex grades of **0.1500 — the
/// full fall line — on 24 of 336 stretches** of a route planned at 5.6 %.
///
/// So a vertex whose turn reverses is kept. Everything else is cut.
fn smooth(p: &[DVec3]) -> Vec<DVec3> {
    if p.len() < 3 {
        return p.to_vec();
    }
    let mut out = Vec::with_capacity(p.len());
    out.push(p[0]);
    for w in p.windows(3) {
        let a = glam::DVec2::new(w[1].x - w[0].x, w[1].z - w[0].z);
        let b = glam::DVec2::new(w[2].x - w[1].x, w[2].z - w[1].z);
        let turn = if a.length_squared() > 0.0 && b.length_squared() > 0.0 {
            a.normalize().dot(b.normalize())
        } else {
            1.0
        };
        if turn <= 0.0 {
            out.push(w[1]);
        } else {
            out.push(w[0] * 0.25 + w[1] * 0.5 + w[2] * 0.25);
        }
    }
    out.push(p[p.len() - 1]);
    out
}

/// `f64` with a total order for the heap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF64(f64);
impl Eq for OrderedF64 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// **The nearest planned route vertex to `p`, and the direction the route runs
/// there** (island wave VEH1a) — one door, three callers.
///
/// The routes are the only *committed* thing on this island that carries a
/// ground height ([`player_start`](crate::build::player_start)'s own argument:
/// the terrain is a build artifact and the design is not), so anything that
/// wants to put something **on the ground** at author time asks here. Its
/// callers are the player start, the fleet the level parks at each settlement,
/// and the connectivity walk.
///
/// The walk is over routes in order with a **strict `<`** on the squared
/// distance, so a tie between two vertices at the same distance is broken by the
/// earlier route rather than by a float comparison's mood. The direction is the
/// segment the vertex belongs to, normalized in XZ; a single-vertex route
/// answers `+Z`, which is this engine's forward.
pub fn nearest_route_vertex(routes: &[Route], p: DVec2) -> Option<(DVec3, DVec2)> {
    let mut best: Option<(f64, DVec3, DVec2)> = None;
    for r in routes {
        for (i, v) in r.points.iter().enumerate() {
            let d = (DVec2::new(v.x, v.z) - p).length_squared();
            if best.is_some_and(|(bd, _, _)| d >= bd) {
                continue;
            }
            // The segment this vertex belongs to: the one ahead of it, or the
            // one behind it at the end of the line.
            let (a, b) = if i + 1 < r.points.len() {
                (r.points[i], r.points[i + 1])
            } else if i > 0 {
                (r.points[i - 1], r.points[i])
            } else {
                (*v, *v + DVec3::Z)
            };
            let dir = DVec2::new(b.x - a.x, b.z - a.z);
            let dir = if dir.length_squared() > 0.0 {
                dir.normalize()
            } else {
                DVec2::new(0.0, 1.0)
            };
            best = Some((d, *v, dir));
        }
    }
    best.map(|(_, v, d)| (v, d))
}

/// Plan every link the recipe's sites imply.
///
/// The topology is the design and it is stated here rather than in the recipe,
/// because it is a *sentence about the island* rather than a number: **the two
/// cities are joined by a highway; every town is joined to the nearest city by
/// an arterial; and the towns are strung together into a circuit** so a drive
/// that leaves one settlement arrives at another rather than at a dead end.
pub fn plan_network(
    recipe: &IslandRecipe,
    heights: &CoarseHeights,
) -> Result<Vec<Route>, IslandError> {
    use crate::recipe::SiteKind;
    let cities: Vec<&Site> = recipe.sites_of(SiteKind::City).collect();
    let towns: Vec<&Site> = recipe.sites_of(SiteKind::Town).collect();
    let sea = recipe.sea.level_m;
    // Plan under the ceiling the audit will apply — see `PLAN_GRADE_MARGIN`.
    let g = recipe.roads.max_grade * PLAN_GRADE_MARGIN;
    // One, because a chord-stepping router already emits few vertices and
    // decimating a sparse path deletes the switchback it just built.
    let stride = 1;
    let mut out = Vec::new();

    let link = |a: &Site, b: &Site, class: &str| -> Result<Route, IslandError> {
        let pts = route(
            heights,
            sea,
            DVec2::new(a.x, a.z),
            DVec2::new(b.x, b.z),
            g,
            stride,
        )
        .map_err(|e| {
            IslandError::Settings(format!("the {class} from {} to {}: {e}", a.name, b.name))
        })?;
        Ok(Route {
            name: format!("{} - {}", a.name, b.name),
            class: class.to_string(),
            points: pts,
        })
    };

    // The trunk: city to city.
    for w in cities.windows(2) {
        out.push(link(w[0], w[1], "highway")?);
    }
    // Each town to its nearest city.
    for t in &towns {
        let Some(c) = cities.iter().min_by(|a, b| {
            let da = (DVec2::new(a.x, a.z) - DVec2::new(t.x, t.z)).length();
            let db = (DVec2::new(b.x, b.z) - DVec2::new(t.x, t.z)).length();
            da.total_cmp(&db).then(a.name.cmp(&b.name))
        }) else {
            continue;
        };
        out.push(link(t, c, "arterial")?);
    }
    // The circuit: town to town, in recipe order, closing back to the first.
    for w in towns.windows(2) {
        out.push(link(w[0], w[1], "arterial")?);
    }
    if towns.len() > 2 {
        out.push(link(towns[towns.len() - 1], towns[0], "arterial")?);
    }
    Ok(out)
}

/// Measure a set of routes against the ground they will actually sit on.
///
/// `height_at` is the **terrain's** query, not the route's own recorded
/// elevation: a route planned on a coarse grid and draped on the fine one is a
/// different line, and the whole point of the audit is to measure the second.
pub fn grade_audit(
    routes: &[Route],
    ceiling: f64,
    step_m: f64,
    mut height_at: impl FnMut(DVec2) -> Option<f64>,
) -> GradeAudit {
    let mut a = GradeAudit {
        ceiling,
        ..Default::default()
    };
    let step = if step_m.is_finite() && step_m > 0.0 {
        step_m
    } else {
        20.0
    };
    for (ri, r) in routes.iter().enumerate() {
        let mut s = 0.0f64;
        let mut prev: Option<(DVec2, f64)> = None;
        for w in r.points.windows(2) {
            let p0 = DVec2::new(w[0].x, w[0].z);
            let p1 = DVec2::new(w[1].x, w[1].z);
            let len = (p1 - p0).length();
            if len <= 0.0 {
                continue;
            }
            let n = (len / step).ceil().max(1.0) as usize;
            for k in 0..=n {
                let t = k as f64 / n as f64;
                let p = p0 + (p1 - p0) * t;
                let Some(h) = height_at(p) else {
                    a.off_terrain += 1;
                    prev = None;
                    continue;
                };
                if let Some((pp, ph)) = prev {
                    let run = (p - pp).length();
                    if run > 0.0 {
                        let grade = (h - ph).abs() / run;
                        a.samples += 1;
                        if grade > a.worst {
                            a.worst = grade;
                            a.worst_at = p;
                        }
                        if grade > ceiling {
                            a.over.push(GradeSpan {
                                route: ri,
                                from_m: s,
                                grade,
                                at: p,
                            });
                        }
                        s += run;
                    }
                }
                prev = Some((p, h));
            }
        }
    }
    a
}

/// Build the road surface mesh through `inf-gis`'s own door.
///
/// `RoadGraph::from_layer` → `build_surface` → `surface_to_mesh`, which is the
/// path IB-4 proved at 3 758 vertices and 0.000000 m of deviation from
/// `ground + lift`. Nothing here re-derives a ribbon; the island is a *caller*.
pub fn build_mesh(
    layer: &inf_gis::GeoLayer,
    ground_step_m: f64,
    height_at: &mut dyn FnMut(f64, f64) -> Option<f64>,
) -> Result<(inf_mesh::MeshAsset, inf_gis::MeshBuildReport, RoadReport), IslandError> {
    let graph = inf_gis::RoadGraph::from_layer(layer);
    let opts = inf_gis::SurfaceOptions {
        ground_step_m,
        ..Default::default()
    };
    let surface = inf_gis::build_surface(&graph, &opts, height_at);
    let (mesh, report) = inf_gis::surface_to_mesh(&surface, DVec3::ZERO)?;

    let mut by_class: std::collections::BTreeMap<String, f64> = Default::default();
    for s in graph.segments.values() {
        *by_class.entry(s.kind.label().to_string()).or_default() += s.length_m();
    }
    let rr = RoadReport {
        km_by_class: by_class.into_iter().map(|(k, v)| (k, v / 1000.0)).collect(),
        total_km: graph.total_length_m() / 1000.0,
        segments: graph.segments.len(),
        junctions: graph.junctions().count(),
        audit: GradeAudit::default(),
    };
    Ok((mesh, report, rr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{tests::tiny_recipe_text, Site, SiteKind};
    use std::path::Path;

    /// A hillside of uniform 15 % grade, climbing along +X and constant along
    /// +Z — the shape every real mountain road crosses by traversing.
    fn hillside(nx: usize, nz: usize, pitch: f64, grade: f64) -> CoarseHeights {
        let mut h = vec![0.0f32; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                h[j * nx + i] = (10.0 + i as f64 * pitch * grade) as f32;
            }
        }
        CoarseHeights {
            min: DVec2::ZERO,
            pitch,
            nx,
            nz,
            h,
            known: vec![true; nx * nz],
        }
    }

    /// **An eight-neighbour router cannot traverse a uniform slope**, which is
    /// the whole reason [`ROUTE_REACH_CELLS`] is 4 — measured from both sides.
    ///
    /// On ground of gradient `g` a D8 step achieves `g` or `g/√2`; a step of
    /// `(1, r)` achieves `g/√(1 + r²)`. So the shallowest road a reach of `r`
    /// can build on gradient `g` is `g/√(1 + r²)`, and everything below it is
    /// impassable **however long the detour**.
    #[test]
    fn an_eight_neighbour_router_cannot_traverse_a_uniform_slope() {
        let g = 0.15;
        let d8 = g / std::f64::consts::SQRT_2;
        let r4 = g / (1.0 + f64::from(ROUTE_REACH_CELLS).powi(2)).sqrt();
        println!(
            "REACH: on a {g:.2} slope, D8 bottoms out at {d8:.4} and reach \
             {ROUTE_REACH_CELLS} at {r4:.4}"
        );
        assert!(
            d8 > 0.08 && r4 < 0.08,
            "the fixture must straddle the 8 % ceiling: d8 {d8}, r4 {r4}"
        );
        // The step set is primitive and symmetric.
        let s = steps();
        assert!(s.contains(&(1, 4)) && s.contains(&(-4, 1)) && s.contains(&(1, 0)));
        assert!(!s.contains(&(2, 2)), "(2,2) is (1,1) twice");
        assert!(!s.contains(&(0, 0)));
        assert!(!s.contains(&(2, 4)));
        for (dx, dz) in &s {
            assert!(s.contains(&(-dx, -dz)), "the step set must be symmetric");
        }
        println!("REACH: {} primitive steps", s.len());
    }

    /// **The grade ceiling is honoured, and honouring it is what makes a
    /// switchback.**
    ///
    /// Un-fix mutation: delete the `if grade > max_grade { continue }` and the
    /// route below goes straight up the hill — its length collapses to the
    /// straight line and the tight-ceiling control below stops discriminating.
    #[test]
    fn a_route_under_a_grade_ceiling_switchbacks_instead_of_climbing() {
        let h = hillside(120, 240, 8.0, 0.15);
        let a = DVec2::new(80.0, 960.0);
        let b = DVec2::new(880.0, 960.0);
        let straight = (b - a).length();

        let pts = route(&h, 0.0, a, b, 0.08, 1).expect("a route exists under 8 %");
        let r = Route {
            name: "x".into(),
            class: "highway".into(),
            points: pts,
        };
        let len = r.length_m();
        // 120 m of rise at 8 % is 1 500 m of road, over an 800 m separation.
        let rise = 0.15 * straight;
        let need = rise / 0.08;
        println!(
            "ROUTE: {len:.0} m of road for {straight:.0} m of separation \
             ({:.2}x); {rise:.0} m of rise at 8 % needs {need:.0} m",
            len / straight
        );
        assert!(
            len >= need * 0.98,
            "a road gaining {rise:.0} m at 8 % cannot be shorter than {need:.0} m; \
             this one is {len:.0} m, so the ceiling is not being enforced"
        );
        assert!(
            len > straight * 1.8,
            "…and it must be much longer than the straight line ({len} vs {straight})"
        );
        // It is a TRAVERSE: the route wanders across the fall line, which is what
        // a switchback is. A straight climb would hold one contour.
        let zs: Vec<f64> = r.points.iter().map(|p| p.z).collect();
        let z_span = zs.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            - zs.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        println!("ROUTE: it wanders {z_span:.0} m across the fall line");
        assert!(
            z_span > 200.0,
            "the route stays within {z_span} m of one line — that is a staircase, \
             not a traverse"
        );

        // And the ground it sits on holds the ceiling, measured through the audit.
        //
        // **The audit reads the CONTINUOUS surface**, which is what
        // `TerrainData::height_at` is (bilinear). Reading a nearest-neighbour
        // sample of the routing lattice instead measures a staircase: two probes
        // 8 m apart that straddle a cell edge see a 1.2 m step over a 5.66 m run
        // and report 0.2121 — `0.15 × √2`, a number about the sampler and not
        // about the road. The first version of this arm did exactly that and the
        // 0.2121 is why this comment exists.
        let ground = |p: DVec2| -> Option<f64> {
            (p.x >= h.min.x && p.x <= h.min.x + (h.nx - 1) as f64 * h.pitch)
                .then_some(10.0 + p.x * 0.15)
        };
        let audit = grade_audit(std::slice::from_ref(&r), 0.08, 8.0, ground);
        println!(
            "AUDIT (planned AT the ceiling): {} samples, worst {:.4}, {} over the \
             {:.3} ceiling ({:.2} %)",
            audit.samples,
            audit.worst,
            audit.over.len(),
            audit.ceiling,
            audit.over_fraction() * 100.0
        );
        assert!(audit.samples > 50);
        assert_eq!(audit.off_terrain, 0);
        assert!(audit.is_clean(), "{:?}", audit.over.first());
        assert_eq!(audit.over_fraction(), 0.0);
        assert!(
            audit.worst <= 0.08,
            "the audit found {:.4} on a route planned at 0.08",
            audit.worst
        );

        // **THE APEX GUARD, priced against the alternative it replaced.** Cut
        // every corner — including the reversals — and the same route's apexes
        // run straight up the fall line. This is the mutation the guard exists
        // for, run here rather than described.
        let cut_all = {
            let p = &r.points;
            let mut out = vec![p[0]];
            for w in p.windows(3) {
                out.push(w[0] * 0.25 + w[1] * 0.5 + w[2] * 0.25);
            }
            out.push(p[p.len() - 1]);
            Route {
                name: "cut".into(),
                class: "highway".into(),
                points: out,
            }
        };
        let bad = grade_audit(std::slice::from_ref(&cut_all), 0.08, 8.0, ground);
        println!(
            "APEX: cutting every corner audits worst {:.4} ({} over of {}); the \
             fall line is 0.1500",
            bad.worst,
            bad.over.len(),
            bad.samples
        );
        assert!(
            bad.worst > audit.worst * 1.5,
            "cutting the apexes must make it materially worse ({:.4} vs {:.4}); if \
             it does not, the guard in `smooth` is guarding nothing",
            bad.worst,
            audit.worst
        );
        assert!(!bad.is_clean());

        // A ceiling BELOW what the reach can achieve on this slope is refused,
        // not silently exceeded — the second half of the reach measurement.
        let e = route(&h, 0.0, a, b, 0.02, 1).unwrap_err().to_string();
        assert!(
            e.contains("0.020") && e.contains("tighter than the ground allows"),
            "{e}"
        );

        // The CONTROL that says the audit can fail: the same route measured
        // against a tighter ceiling reports the stretches it now exceeds.
        let tight = grade_audit(std::slice::from_ref(&r), 0.01, 8.0, ground);
        assert!(
            !tight.is_clean(),
            "a 1 % ceiling on a 15 % hillside is clean?"
        );
        assert!(tight.over_fraction() > 0.0);
        assert_eq!(
            tight.worst, audit.worst,
            "the same ground, the same worst grade"
        );
    }

    /// A link with no land route is refused **by name**, not routed through the
    /// sea.
    #[test]
    fn a_link_with_no_land_route_is_refused_with_both_ends() {
        // Two plateaus separated by open water.
        let (nx, nz, pitch) = (80usize, 40usize, 8.0);
        let mut h = vec![-20.0f32; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                if !(20..=60).contains(&i) {
                    h[j * nx + i] = 30.0;
                }
            }
        }
        let ch = CoarseHeights {
            min: DVec2::ZERO,
            pitch,
            nx,
            nz,
            h,
            known: vec![true; nx * nz],
        };
        let e = route(
            &ch,
            0.0,
            DVec2::new(40.0, 160.0),
            DVec2::new(600.0, 160.0),
            0.5,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("no bridges") && e.contains("0.500"), "{e}");

        // An endpoint under water is refused earlier and differently.
        let e2 = route(
            &ch,
            0.0,
            DVec2::new(300.0, 160.0),
            DVec2::new(600.0, 160.0),
            0.5,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(e2.contains("under water"), "{e2}");

        // Off the world is its own refusal.
        assert!(route(
            &ch,
            0.0,
            DVec2::new(-9e9, 0.0),
            DVec2::new(40.0, 160.0),
            0.5,
            1
        )
        .unwrap_err()
        .to_string()
        .contains("off the world"));
    }

    #[test]
    fn the_network_topology_joins_the_cities_and_strings_the_towns() {
        let mut r = IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/i")).unwrap();
        r.grid.tiles = 8;
        r.grid.tile_resolution = 33;
        r.grid.meters_per_sample = 4.0; // 128 m tiles -> 1024 m world
        r.roads.max_grade = 0.5;
        let half = r.grid.half_extent_m();
        let mut push = |n: &str, k, x, z| {
            r.sites.push(Site {
                name: n.into(),
                kind: k,
                x,
                z,
                radius_m: 20.0,
            })
        };
        push("A", SiteKind::City, -300.0, -300.0);
        push("B", SiteKind::City, 300.0, 300.0);
        push("T1", SiteKind::Town, -300.0, 300.0);
        push("T2", SiteKind::Town, 300.0, -300.0);
        push("T3", SiteKind::Town, 0.0, 400.0);

        let (nx, nz) = (129usize, 129usize);
        let h = CoarseHeights {
            min: DVec2::splat(-half),
            pitch: 8.0,
            nx,
            nz,
            h: vec![50.0; nx * nz],
            known: vec![true; nx * nz],
        };
        let net = plan_network(&r, &h).expect("a flat world routes everything");
        let names: Vec<&str> = net.iter().map(|x| x.name.as_str()).collect();
        println!("NETWORK: {names:?}");
        // one city-city highway
        assert_eq!(net.iter().filter(|x| x.class == "highway").count(), 1);
        // three town->city arterials + two town-town + one closing = 6
        assert_eq!(net.iter().filter(|x| x.class == "arterial").count(), 6);
        assert!(names.contains(&"A - B"));
        assert!(names.contains(&"T3 - T1"), "the circuit closes: {names:?}");
        for x in &net {
            assert!(
                x.points.len() >= 2,
                "{} has {} points",
                x.name,
                x.points.len()
            );
            assert!(x.length_m() > 0.0);
        }
    }

    #[test]
    fn the_audit_reports_ground_it_cannot_sample_rather_than_skipping_it() {
        let r = Route {
            name: "x".into(),
            class: "highway".into(),
            points: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)],
        };
        let a = grade_audit(std::slice::from_ref(&r), 0.08, 10.0, |p| {
            (p.x < 50.0).then_some(0.0)
        });
        assert!(
            a.off_terrain > 0,
            "half the route is off the terrain and unreported"
        );
        assert!(a.samples > 0, "…and the other half was still measured");
        assert!(a.is_clean());
        // A degenerate step falls back to a sane one rather than dividing by zero.
        let b = grade_audit(std::slice::from_ref(&r), 0.08, 0.0, |_| Some(0.0));
        assert!(b.samples > 0);
        let empty = grade_audit(&[], 0.08, 10.0, |_| Some(0.0));
        assert_eq!(empty.samples, 0);
        assert_eq!(empty.over_fraction(), 0.0);
    }

    #[test]
    fn the_mesh_goes_through_the_gis_door_and_reports_its_classes() {
        // Two crossing roads, split at the crossing the way a published layer is.
        let mut layer = inf_gis::GeoLayer::new("roads", inf_gis::LayerKind::Roads, "EPSG:32610");
        let mk = |pts: Vec<DVec3>, class: &str| {
            let mut f = inf_gis::GeoFeature::new(inf_gis::GeoGeometry::Polyline {
                points: pts,
                closed: false,
            });
            f.attributes
                .insert("road_type".into(), inf_gis::Attr::Text(class.to_string()));
            f
        };
        layer.features.push(mk(
            vec![DVec3::new(-100.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 0.0)],
            "highway",
        ));
        layer.features.push(mk(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)],
            "highway",
        ));
        layer.features.push(mk(
            vec![DVec3::new(0.0, 0.0, -100.0), DVec3::new(0.0, 0.0, 0.0)],
            "arterial",
        ));
        layer.features.push(mk(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 100.0)],
            "arterial",
        ));

        let mut ground = |x: f64, z: f64| Some((x + z) * 0.01);
        let (mesh, report, rr) = build_mesh(&layer, 2.0, &mut ground).expect("the door builds");
        assert!(report.vertices > 0 && report.triangles > 0);
        assert!(!mesh.submeshes.is_empty());
        assert_eq!(rr.segments, 4);
        assert_eq!(rr.junctions, 1, "four legs meet at one crossing");
        let classes: Vec<&str> = rr.km_by_class.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(classes, vec!["arterial", "highway"], "BTree order");
        assert!((rr.total_km - 0.4).abs() < 1e-9, "{}", rr.total_km);
        println!(
            "ROAD MESH: {} vertices, {} triangles, {:.3} km over {} segments, \
             quantisation {:.4} m",
            report.vertices, report.triangles, rr.total_km, rr.segments, report.quantisation_m
        );
        // The road really is ON the ground it was draped on.
        assert!(
            report.max_offset_m > 0.0,
            "a road across a sloped ground has a non-zero extent"
        );
    }
}
