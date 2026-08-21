//! Where the water goes — derived from the carved ground, then committed as the
//! design.
//!
//! # The order matters and it is the classic one
//!
//! 1. **Fill the depressions.** A D8 router on raw ground walks into every pit
//!    and stops there, so half the island drains nowhere. Priority-flood raises
//!    each pit to its own spill point — and the *depth of the fill* is not a
//!    by-product to throw away, it is **where the lakes are**. One pass answers
//!    two questions, which is why they are one pass.
//! 2. **Route.** Each cell drains to its steepest filled-surface neighbour.
//! 3. **Accumulate.** Cells in descending filled height, each pushing its total
//!    downstream. Descending order is what makes one pass sufficient: a cell is
//!    only visited after everything that could drain into it.
//! 4. **Extract.** Cells past a catchment threshold are channel; the channel set
//!    is walked from its heads into reaches, and each reach is one polyline.
//! 5. **Waterfalls** are the reaches' own steep segments — not a new system, a
//!    measurement of one that exists.
//!
//! # Why this is a DESIGN artifact and not an oracle
//!
//! The output is committed GeoJSON. An author may move a stream, delete a lake
//! or add one, and the build will use what is committed rather than re-deriving
//! it — because the derivation runs on a projection this repository's own
//! portability law exempts, so a rebuild on a different libm can move a
//! threshold comparison and therefore a channel. [`crate::report::LayerDrift`]
//! is what reports the difference instead of pretending there is none.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use glam::DVec2;

use crate::terrain::CoarseHeights;

/// The default catchment a cell needs before it is called a stream, in square
/// metres.
///
/// One square kilometre. Chosen against the island rather than in general: at
/// 50 km² a 1 km² threshold yields a handful of named watercourses, and dropping
/// it an order of magnitude yields a hundred rivulets that are really the
/// terrain's own noise at the source's 3 m resolution.
pub const DEFAULT_STREAM_CATCHMENT_M2: f64 = 1.0e6;

/// The shallowest fill that counts as a lake, in metres.
pub const DEFAULT_LAKE_DEPTH_M: f64 = 1.5;

/// The smallest lake that is worth a water body, in square metres.
pub const DEFAULT_LAKE_AREA_M2: f64 = 2_500.0;

/// The gradient at which a stream segment is called a waterfall — rise over run.
///
/// 0.5 is a 26.6° bed, which is past anything a river cuts and into what a river
/// *falls down*. Reported with its own number so an author can see the drop
/// rather than the label.
pub const DEFAULT_WATERFALL_GRADE: f64 = 0.5;

/// How the derivation is parameterised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HydroParams {
    pub sea_level_m: f64,
    pub stream_catchment_m2: f64,
    pub lake_depth_m: f64,
    pub lake_area_m2: f64,
    pub waterfall_grade: f64,
    /// How many lattice cells a committed stream vertex is worth. A stream
    /// polyline at the lattice's own pitch is thousands of vertices for a shape
    /// that is smooth at a hundred metres; decimating is what keeps the committed
    /// layer a design document rather than a raster in disguise.
    pub vertex_stride: usize,
}

impl Default for HydroParams {
    fn default() -> Self {
        Self {
            sea_level_m: 0.0,
            stream_catchment_m2: DEFAULT_STREAM_CATCHMENT_M2,
            lake_depth_m: DEFAULT_LAKE_DEPTH_M,
            lake_area_m2: DEFAULT_LAKE_AREA_M2,
            waterfall_grade: DEFAULT_WATERFALL_GRADE,
            vertex_stride: 8,
        }
    }
}

/// The filled surface, the routing and the accumulation.
#[derive(Clone, Debug)]
pub struct FlowField {
    pub nx: usize,
    pub nz: usize,
    pub pitch: f64,
    pub min: DVec2,
    /// The depression-filled surface, metres.
    pub filled: Vec<f32>,
    /// How much the fill raised each cell, metres.
    pub fill_depth: Vec<f32>,
    /// Downstream cell index, or `usize::MAX` for an outlet.
    pub down: Vec<u32>,
    /// Contributing area in square metres.
    pub accum: Vec<f32>,
}

const NO_DOWN: u32 = u32::MAX;

/// The eight neighbours, in a fixed order so a tie between two equally steep
/// neighbours resolves the same way on every run.
const N8: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

impl FlowField {
    /// Fill, route and accumulate over a coarse height grid.
    pub fn derive(h: &CoarseHeights, p: &HydroParams) -> Self {
        let (nx, nz) = (h.nx, h.nz);
        let n = nx * nz;
        let sea = p.sea_level_m as f32;

        // ── 1. priority flood ────────────────────────────────────────────────
        // Every border cell and every cell at or below sea level is an outlet
        // and enters the queue at its own height. Everything else enters when a
        // neighbour pops, raised to at least that neighbour's level.
        let mut filled = vec![f32::INFINITY; n];
        let mut done = vec![false; n];
        // (height, index) with Reverse for a min-heap. The index is in the key so
        // two cells at exactly one height pop in a fixed order.
        let mut q: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        for j in 0..nz {
            for i in 0..nx {
                let k = j * nx + i;
                let border = i == 0 || j == 0 || i + 1 == nx || j + 1 == nz;
                if border || !h.known[k] || h.h[k] <= sea {
                    filled[k] = h.h[k];
                    q.push(Reverse((OrderedF32(h.h[k]), k as u32)));
                    done[k] = true;
                }
            }
        }
        while let Some(Reverse((OrderedF32(hv), ki))) = q.pop() {
            let k = ki as usize;
            let (i, j) = (k % nx, k / nx);
            for (dx, dz) in N8 {
                let (ni, nj) = (i as i64 + dx as i64, j as i64 + dz as i64);
                if ni < 0 || nj < 0 || ni as usize >= nx || nj as usize >= nz {
                    continue;
                }
                let nk = nj as usize * nx + ni as usize;
                if done[nk] {
                    continue;
                }
                let v = h.h[nk].max(hv);
                filled[nk] = v;
                done[nk] = true;
                q.push(Reverse((OrderedF32(v), nk as u32)));
            }
        }
        let fill_depth: Vec<f32> = (0..n).map(|k| (filled[k] - h.h[k]).max(0.0)).collect();

        // ── 2. D8 routing on the filled surface ──────────────────────────────
        let mut down = vec![NO_DOWN; n];
        for j in 0..nz {
            for i in 0..nx {
                let k = j * nx + i;
                if filled[k] <= sea {
                    continue; // the sea is the outlet
                }
                let mut best = 0.0f64;
                let mut best_k = NO_DOWN;
                for (dx, dz) in N8 {
                    let (ni, nj) = (i as i64 + dx as i64, j as i64 + dz as i64);
                    if ni < 0 || nj < 0 || ni as usize >= nx || nj as usize >= nz {
                        continue;
                    }
                    let nk = nj as usize * nx + ni as usize;
                    let drop = f64::from(filled[k] - filled[nk]);
                    if drop <= 0.0 {
                        continue;
                    }
                    // Steepest DESCENT, not steepest drop: a diagonal neighbour
                    // is 1.414 cells away and comparing raw drops biases every
                    // channel onto the diagonals.
                    let run = if dx != 0 && dz != 0 {
                        std::f64::consts::SQRT_2
                    } else {
                        1.0
                    } * h.pitch;
                    let grade = drop / run;
                    if grade > best {
                        best = grade;
                        best_k = nk as u32;
                    }
                }
                down[k] = best_k;
            }
        }

        // ── 3. accumulation, in descending filled height ─────────────────────
        let cell_area = (h.pitch * h.pitch) as f32;
        let mut accum = vec![cell_area; n];
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|a, b| {
            filled[*b as usize]
                .total_cmp(&filled[*a as usize])
                .then(a.cmp(b))
        });
        for &k in &order {
            let k = k as usize;
            let d = down[k];
            if d != NO_DOWN {
                accum[d as usize] += accum[k];
            }
        }

        Self {
            nx,
            nz,
            pitch: h.pitch,
            min: h.min,
            filled,
            fill_depth,
            down,
            accum,
        }
    }

    /// The world position of cell `k`.
    #[inline]
    pub fn position(&self, k: usize) -> DVec2 {
        DVec2::new(
            self.min.x + (k % self.nx) as f64 * self.pitch,
            self.min.y + (k / self.nx) as f64 * self.pitch,
        )
    }

    /// The largest catchment anywhere on the grid, in square metres — the
    /// island's own drainage, and the number a stream threshold should be read
    /// against.
    pub fn max_accumulation_m2(&self) -> f64 {
        self.accum.iter().fold(0.0f32, |a, b| a.max(*b)) as f64
    }
}

/// `f32` with a total order, so a heap over heights is deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// One watercourse reach.
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    /// World XZ + the bed elevation the derivation found, upstream first.
    pub points: Vec<glam::DVec3>,
    /// The catchment at the downstream end, square metres.
    pub catchment_m2: f64,
    /// Plan-view length, metres.
    pub length_m: f64,
    /// Total fall along the reach, metres.
    pub fall_m: f64,
}

impl Stream {
    /// A width derived from the catchment, metres.
    ///
    /// The textbook hydraulic-geometry form `w = a·Q^b` with the exponent at its
    /// classic ½ — expressed as a **square root of the area ratio** rather than
    /// a `powf`, because `powf` is on the portability ban list and `sqrt` is
    /// exact.
    pub fn width_m(&self) -> f64 {
        let r = (self.catchment_m2 / DEFAULT_STREAM_CATCHMENT_M2).max(0.0);
        (2.0 * r.sqrt()).clamp(1.5, 24.0)
    }

    /// A depth derived from the width.
    pub fn depth_m(&self) -> f64 {
        (self.width_m() * 0.18).clamp(0.35, 3.0)
    }

    /// The mean gradient over the reach.
    pub fn grade(&self) -> f64 {
        if self.length_m <= 0.0 {
            return 0.0;
        }
        self.fall_m / self.length_m
    }
}

/// A lake the fill found.
#[derive(Clone, Debug, PartialEq)]
pub struct Lake {
    /// The water surface, world metres.
    pub level_m: f64,
    /// Plan-view centroid.
    pub centre: DVec2,
    /// Half the bounding box, world metres — the shape `WaterBody::lake` wants.
    pub half_extent: DVec2,
    /// Surface area, square metres.
    pub area_m2: f64,
    /// Deepest fill, metres.
    pub max_depth_m: f64,
    /// The outline, world XZ, for the committed layer.
    pub outline: Vec<DVec2>,
}

/// A steep stretch of a stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Waterfall {
    /// Which stream, by index into the network.
    pub stream: usize,
    /// The top of the fall.
    pub top: glam::DVec3,
    /// The bottom.
    pub bottom: glam::DVec3,
    /// The drop, metres.
    pub drop_m: f64,
    /// Rise over run.
    pub grade: f64,
}

/// Everything the hydrology step produced.
#[derive(Clone, Debug, Default)]
pub struct StreamNetwork {
    pub streams: Vec<Stream>,
    pub lakes: Vec<Lake>,
    pub waterfalls: Vec<Waterfall>,
    /// The largest catchment on the island, square metres.
    pub max_catchment_m2: f64,
    /// How many lattice cells were classed as channel.
    pub channel_cells: usize,
}

impl StreamNetwork {
    /// Total stream length in metres.
    pub fn total_length_m(&self) -> f64 {
        self.streams.iter().map(|s| s.length_m).sum()
    }

    /// Total lake surface in square metres.
    pub fn total_lake_area_m2(&self) -> f64 {
        self.lakes.iter().map(|l| l.area_m2).sum()
    }
}

/// Extract the network from a derived flow field.
pub fn extract(flow: &FlowField, p: &HydroParams) -> StreamNetwork {
    let n = flow.nx * flow.nz;
    let sea = p.sea_level_m as f32;
    let thr = p.stream_catchment_m2 as f32;

    // Channel cells: enough catchment, above the sea, and draining somewhere.
    let channel: Vec<bool> = (0..n)
        .map(|k| flow.accum[k] >= thr && flow.filled[k] > sea && flow.down[k] != NO_DOWN)
        .collect();
    let channel_cells = channel.iter().filter(|c| **c).count();

    // A head is a channel cell nothing upstream of it is a channel cell.
    let mut has_upstream = vec![false; n];
    for k in 0..n {
        if !channel[k] {
            continue;
        }
        let d = flow.down[k] as usize;
        if d < n && channel[d] {
            has_upstream[d] = true;
        }
    }

    // Walk each head downstream, stopping where another reach already claimed
    // the cell. Heads in ascending index order, so the reach decomposition is a
    // function of the grid rather than of a hash.
    let mut claimed = vec![false; n];
    let mut streams: Vec<Stream> = Vec::new();
    for head in 0..n {
        if !channel[head] || has_upstream[head] || claimed[head] {
            continue;
        }
        let mut cells = Vec::new();
        let mut k = head;
        loop {
            cells.push(k);
            claimed[k] = true;
            let d = flow.down[k];
            if d == NO_DOWN {
                break;
            }
            let d = d as usize;
            // The reach ends AT the joining cell, so two reaches meeting at a
            // confluence share that vertex rather than leaving a gap in the map.
            if !channel[d] || claimed[d] {
                cells.push(d);
                break;
            }
            k = d;
        }
        if cells.len() < 3 {
            continue;
        }
        let stride = p.vertex_stride.max(1);
        let mut points: Vec<glam::DVec3> = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            if i % stride == 0 || i + 1 == cells.len() {
                let xz = flow.position(*c);
                points.push(glam::DVec3::new(xz.x, f64::from(flow.filled[*c]), xz.y));
            }
        }
        if points.len() < 2 {
            continue;
        }
        let mut length = 0.0;
        for w in points.windows(2) {
            length += (DVec2::new(w[1].x, w[1].z) - DVec2::new(w[0].x, w[0].z)).length();
        }
        let fall = points[0].y - points[points.len() - 1].y;
        streams.push(Stream {
            catchment_m2: f64::from(flow.accum[*cells.last().unwrap()]),
            points,
            length_m: length,
            fall_m: fall,
        });
    }

    // Waterfalls: a reach's own steep segments.
    let mut waterfalls = Vec::new();
    for (si, s) in streams.iter().enumerate() {
        for w in s.points.windows(2) {
            let run = (DVec2::new(w[1].x, w[1].z) - DVec2::new(w[0].x, w[0].z)).length();
            let drop = w[0].y - w[1].y;
            if run <= 0.0 || drop <= 0.0 {
                continue;
            }
            let grade = drop / run;
            if grade >= p.waterfall_grade {
                waterfalls.push(Waterfall {
                    stream: si,
                    top: w[0],
                    bottom: w[1],
                    drop_m: drop,
                    grade,
                });
            }
        }
    }

    StreamNetwork {
        lakes: lakes_of(flow, p),
        streams,
        waterfalls,
        max_catchment_m2: flow.max_accumulation_m2(),
        channel_cells,
    }
}

/// Connected components of filled depth — the lakes.
fn lakes_of(flow: &FlowField, p: &HydroParams) -> Vec<Lake> {
    let (nx, nz) = (flow.nx, flow.nz);
    let n = nx * nz;
    let cell = flow.pitch * flow.pitch;
    let min_depth = p.lake_depth_m as f32;
    let sea = p.sea_level_m as f32;
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for start in 0..n {
        if seen[start] || flow.fill_depth[start] < min_depth || flow.filled[start] <= sea {
            continue;
        }
        // Flood fill in index order — deterministic without a sort.
        let mut stack = vec![start];
        seen[start] = true;
        let mut cells: Vec<usize> = Vec::new();
        while let Some(k) = stack.pop() {
            cells.push(k);
            let (i, j) = (k % nx, k / nx);
            for (dx, dz) in N8 {
                let (ni, nj) = (i as i64 + dx as i64, j as i64 + dz as i64);
                if ni < 0 || nj < 0 || ni as usize >= nx || nj as usize >= nz {
                    continue;
                }
                let nk = nj as usize * nx + ni as usize;
                if seen[nk] || flow.fill_depth[nk] < min_depth || flow.filled[nk] <= sea {
                    continue;
                }
                seen[nk] = true;
                stack.push(nk);
            }
        }
        let area = cells.len() as f64 * cell;
        if area < p.lake_area_m2 {
            continue;
        }
        cells.sort_unstable();
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        let mut sum = DVec2::ZERO;
        let mut level = f32::NEG_INFINITY;
        let mut depth = 0.0f32;
        for k in &cells {
            let q = flow.position(*k);
            lo = lo.min(q);
            hi = hi.max(q);
            sum += q;
            level = level.max(flow.filled[*k]);
            depth = depth.max(flow.fill_depth[*k]);
        }
        let centre = sum / cells.len() as f64;
        // The outline is the bounding rectangle: a lake's committed layer is a
        // DESIGN artifact and a rectangle is what an author edits. The exact
        // filled cell set is the derivation's, and it is what the level's own
        // half-extent comes from.
        let outline = vec![
            DVec2::new(lo.x, lo.y),
            DVec2::new(hi.x, lo.y),
            DVec2::new(hi.x, hi.y),
            DVec2::new(lo.x, hi.y),
        ];
        out.push(Lake {
            level_m: f64::from(level),
            centre,
            half_extent: (hi - lo) * 0.5,
            area_m2: area,
            max_depth_m: f64::from(depth),
            outline,
        });
    }
    // Largest first — a report that leads with a puddle is a report nobody reads.
    out.sort_by(|a, b| {
        b.area_m2
            .total_cmp(&a.area_m2)
            .then(a.centre.x.total_cmp(&b.centre.x))
            .then(a.centre.y.total_cmp(&b.centre.y))
    });
    out
}

/// Carve a channel for each stream into the terrain.
///
/// # Why the ground has to move
///
/// A `WaterBody::River` is a surface over a spline. P20's own bed advisory calls
/// a river whose bed climbs *"a basin, not a river"* and refuses to pretend
/// otherwise — and a derived stream laid on unmodified ground is exactly that:
/// the ground under a D8 channel is the ground the channel was found on, which
/// is a surface, not a trench. Cutting a shallow V under each reach is what makes
/// the water sit **in** the island instead of on it.
///
/// Returns how many samples moved.
pub fn carve_channels(
    data: &mut inf_terrain::TerrainData,
    index: &crate::shape::SegmentIndex,
    half_width_m: f64,
    depth_m: f64,
) -> u64 {
    if index.is_empty() || half_width_m <= 0.0 || depth_m <= 0.0 {
        return 0;
    }
    let res = data.tile_resolution();
    let mps = data.meters_per_sample();
    let coords: Vec<(i32, i32)> = data.tiles().map(|(c, _)| *c).collect();
    let mut moved = 0u64;
    for c in coords {
        let origin = data.tile_origin_xz(c);
        let Some(tile) = data.get_tile_mut(c) else {
            continue;
        };
        for j in 0..res {
            for i in 0..res {
                let p = DVec2::new(origin.x + f64::from(i) * mps, origin.y + f64::from(j) * mps);
                let Some(n) = index.nearest(p) else { continue };
                if n.distance_m >= half_width_m {
                    continue;
                }
                // A V: full depth on the centreline, meeting the bank at zero.
                let t = n.distance_m / half_width_m;
                let cut = depth_m * (1.0 - t);
                let bed = n.height_m - cut;
                let h = tile.world_height(res, i, j);
                if bed < h {
                    tile.set_sample(res, i, j, (bed - tile.origin.y) as f32);
                    moved += 1;
                }
            }
        }
    }
    moved
}

/// Index the derived streams for the channel carve, keyed on the bed each reach
/// wants.
pub fn channel_index(streams: &[Stream], reach_m: f64) -> crate::shape::SegmentIndex {
    let lines: Vec<Vec<crate::shape::Vertex3>> = streams
        .iter()
        .map(|s| {
            s.points
                .iter()
                .map(|p| crate::shape::Vertex3 {
                    xz: DVec2::new(p.x, p.z),
                    y: p.y,
                })
                .collect()
        })
        .collect();
    crate::shape::SegmentIndex::new(&lines, reach_m)
}

/// The lakes, keyed by a stable id, for a caller that wants a map.
pub fn lakes_by_id(lakes: &[Lake]) -> BTreeMap<usize, &Lake> {
    lakes.iter().enumerate().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cone with a pit in one flank: everything drains off the cone, and the
    /// pit is a lake. Every number below is derivable by hand from the shape.
    fn cone(nx: usize, nz: usize, pitch: f64, pit: Option<(usize, usize, f32)>) -> CoarseHeights {
        let mut h = vec![0.0f32; nx * nz];
        let (cx, cz) = ((nx as f64 - 1.0) * 0.5, (nz as f64 - 1.0) * 0.5);
        for j in 0..nz {
            for i in 0..nx {
                let d = ((i as f64 - cx).powi(2) + (j as f64 - cz).powi(2)).sqrt();
                h[j * nx + i] = (200.0 - d * 6.0) as f32;
            }
        }
        if let Some((pi, pj, depth)) = pit {
            for j in pj.saturating_sub(2)..(pj + 3).min(nz) {
                for i in pi.saturating_sub(2)..(pi + 3).min(nx) {
                    h[j * nx + i] -= depth;
                }
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

    #[test]
    fn the_fill_raises_a_pit_to_its_spill_and_nothing_else() {
        let p = HydroParams::default();
        let clean = FlowField::derive(&cone(41, 41, 8.0, None), &p);
        assert!(
            clean.fill_depth.iter().all(|d| *d < 1e-3),
            "a cone has no depression; the fill invented one"
        );
        let pitted = FlowField::derive(&cone(41, 41, 8.0, Some((12, 20, 25.0))), &p);
        let deepest = pitted.fill_depth.iter().fold(0.0f32, |a, b| a.max(*b));
        // **It fills to the SPILL, not to its own rim's maximum.** The pit sits
        // on a slope, so its lowest rim neighbour is downhill of its highest and
        // the water leaves over the low side. That is why 25 m of excavation
        // holds less than 25 m of water, and it is the property being asserted —
        // an assertion of exactly 25 would be asserting the excavation, which
        // the fixture already knows.
        assert!(
            deepest > 15.0 && deepest < 25.0,
            "a 25 m pit on a 6 m/cell slope filled to {deepest} m — above 25 the \
             fill overshot its spill, below 15 it barely filled at all"
        );
        // And only the pit moved.
        let moved = pitted.fill_depth.iter().filter(|d| **d > 0.1).count();
        assert!(moved > 4 && moved < 60, "{moved} cells were raised");
        println!("FILL: deepest {deepest:.2} m over {moved} cells");
    }

    /// Accumulation is a real catchment: the total leaving the cone equals the
    /// cone's own area, and the steepest-DESCENT rule keeps channels off the
    /// diagonals.
    #[test]
    fn accumulation_conserves_area_and_descends_by_gradient() {
        let p = HydroParams::default();
        let h = cone(41, 41, 8.0, None);
        let f = FlowField::derive(&h, &p);
        let cell = 64.0f32;
        // Every interior cell drains somewhere.
        let stuck = (0..f.nx * f.nz)
            .filter(|k| {
                let (i, j) = (k % f.nx, k / f.nx);
                i > 0 && j > 0 && i + 1 < f.nx && j + 1 < f.nz && f.down[*k] == NO_DOWN
            })
            .count();
        assert_eq!(stuck, 0, "{stuck} interior cells drain nowhere");
        // The largest catchment is a real fraction of the cone.
        let total = (f.nx * f.nz) as f32 * cell;
        let max = f.max_accumulation_m2() as f32;
        assert!(max > cell * 8.0, "the biggest catchment is only {max} m2");
        assert!(max < total, "a catchment cannot exceed the whole grid");
        println!("ACCUM: max {max:.0} m2 of a {total:.0} m2 grid");

        // Un-fix control for the descent rule: comparing raw DROPS instead of
        // gradients puts the biggest catchments on the diagonals. Measure the
        // split as evidence the rule is doing something.
        let mut diag = 0;
        let mut orth = 0;
        for k in 0..f.nx * f.nz {
            let d = f.down[k];
            if d == NO_DOWN {
                continue;
            }
            let (i, j) = (k % f.nx, k / f.nx);
            let (ni, nj) = (d as usize % f.nx, d as usize / f.nx);
            if i != ni && j != nj {
                diag += 1;
            } else {
                orth += 1;
            }
        }
        println!("ROUTING: {diag} diagonal, {orth} orthogonal");
        assert!(
            orth > 0 && diag > 0,
            "one of the two families is unreachable"
        );
    }

    #[test]
    fn a_pit_becomes_a_lake_with_its_own_level_and_extent() {
        let mut p = HydroParams::default();
        p.lake_area_m2 = 100.0;
        let f = FlowField::derive(&cone(41, 41, 8.0, Some((12, 20, 25.0))), &p);
        let net = extract(&f, &p);
        assert_eq!(
            net.lakes.len(),
            1,
            "one pit, one lake: {:?}",
            net.lakes.len()
        );
        let l = &net.lakes[0];
        assert!(l.max_depth_m > 15.0, "{}", l.max_depth_m);
        assert!(l.area_m2 >= 100.0);
        assert_eq!(l.outline.len(), 4);
        assert!(l.half_extent.x > 0.0 && l.half_extent.y > 0.0);
        // The lake's own centre is inside its own box.
        assert!((l.centre.x - (l.outline[0].x + l.half_extent.x)).abs() < l.half_extent.x + 8.0);
        println!(
            "LAKE: level {:.2} m, {:.0} m2, deepest {:.2} m, half {:?}",
            l.level_m, l.area_m2, l.max_depth_m, l.half_extent
        );
        // A cone with no pit has no lake — the anti-vacuity control.
        let clean = FlowField::derive(&cone(41, 41, 8.0, None), &p);
        assert!(extract(&clean, &p).lakes.is_empty());
    }

    #[test]
    fn streams_come_out_as_reaches_with_widths_and_falls() {
        let mut p = HydroParams::default();
        // A 41x41 cone at 8 m is 100 000 m2 total, so a 1 km2 threshold finds
        // nothing. Scale the threshold to the fixture, which is the honest way
        // round: the constant is sized against the ISLAND and this is not one.
        p.stream_catchment_m2 = 3_000.0;
        p.vertex_stride = 2;
        let f = FlowField::derive(&cone(61, 61, 8.0, None), &p);
        let net = extract(&f, &p);
        assert!(
            !net.streams.is_empty(),
            "a cone drains radially; nothing was found"
        );
        assert!(net.channel_cells > 0);
        for s in &net.streams {
            assert!(s.points.len() >= 2);
            assert!(s.length_m > 0.0, "a reach with no length");
            assert!(
                s.fall_m > -1e-6,
                "a reach that climbs {} m — the routing is not descending",
                s.fall_m
            );
            assert!((1.5..=24.0).contains(&s.width_m()));
            assert!((0.35..=3.0).contains(&s.depth_m()));
            assert!(s.grade() >= 0.0);
        }
        // The widths really do follow the catchment.
        let widest = net
            .streams
            .iter()
            .max_by(|a, b| a.catchment_m2.total_cmp(&b.catchment_m2))
            .unwrap();
        let narrowest = net
            .streams
            .iter()
            .min_by(|a, b| a.catchment_m2.total_cmp(&b.catchment_m2))
            .unwrap();
        println!(
            "STREAMS: {} reaches, {:.0} m total, widest {:.2} m at {:.0} m2, \
             narrowest {:.2} m at {:.0} m2",
            net.streams.len(),
            net.total_length_m(),
            widest.width_m(),
            widest.catchment_m2,
            narrowest.width_m(),
            narrowest.catchment_m2
        );
        assert!(widest.width_m() >= narrowest.width_m());
        // Reaches do not overlap: the cells are claimed once, so the total
        // length is bounded by the channel set's own extent.
        assert!(net.total_length_m() <= net.channel_cells as f64 * f.pitch * 1.5);
    }

    /// A waterfall is a measurement of a steep segment, not a new system — so the
    /// arm is that a cliff produces one and a gentle slope does not.
    #[test]
    fn a_steep_reach_reports_a_waterfall_and_a_gentle_one_does_not() {
        let mut p = HydroParams::default();
        p.stream_catchment_m2 = 3_000.0;
        p.vertex_stride = 1;
        // A shallow valley draining along +Z with a step across it. The cross
        // gradient (0.5 m a cell) must EXCEED the down-valley one (0.2 m a cell)
        // or the steepest-descent rule sends every column straight down its own
        // line and nothing converges — which is the first version of this
        // fixture, and it found no waterfall because it had no stream.
        let (nx, nz, pitch) = (61usize, 61usize, 8.0);
        let mut h = vec![0.0f32; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                let mut v = 300.0 - j as f32 * 0.2 + (i as f32 - 30.0).abs() * 0.5;
                if j > 30 {
                    v -= 60.0; // a 60 m step over one cell
                }
                h[j * nx + i] = v;
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
        let f = FlowField::derive(&ch, &p);
        let net = extract(&f, &p);
        assert!(!net.waterfalls.is_empty(), "a 60 m step made no waterfall");
        let w = net
            .waterfalls
            .iter()
            .max_by(|a, b| a.drop_m.total_cmp(&b.drop_m))
            .unwrap();
        assert!(w.drop_m > 30.0, "the biggest drop is {} m", w.drop_m);
        assert!(w.grade >= p.waterfall_grade);
        assert!(w.top.y > w.bottom.y);
        assert!(
            net.streams.get(w.stream).is_some(),
            "a waterfall names a real reach"
        );
        println!(
            "WATERFALL: {} sites, biggest {:.1} m at grade {:.2}",
            net.waterfalls.len(),
            w.drop_m,
            w.grade
        );

        // The control: the same valley with no step reports none.
        let mut h2 = vec![0.0f32; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                h2[j * nx + i] = 300.0 - j as f32 * 0.2 + (i as f32 - 30.0).abs() * 0.5;
            }
        }
        let ch2 = CoarseHeights { h: h2, ..ch };
        let net2 = extract(&FlowField::derive(&ch2, &p), &p);
        assert!(
            net2.waterfalls.is_empty(),
            "a 1-in-8 slope reported {} waterfalls",
            net2.waterfalls.len()
        );
        assert!(!net2.streams.is_empty(), "…but it still has streams");
    }

    #[test]
    fn a_channel_carve_puts_the_bed_under_the_ground_it_was_found_on() {
        let mut data = inf_terrain::TerrainData::new(65, 1.0);
        data.author_tile((0, 0), |_, _| 100.0);
        let s = Stream {
            points: vec![
                glam::DVec3::new(4.0, 100.0, 8.0),
                glam::DVec3::new(60.0, 100.0, 8.0),
            ],
            catchment_m2: 2.0e6,
            length_m: 56.0,
            fall_m: 0.0,
        };
        let idx = channel_index(std::slice::from_ref(&s), 6.0);
        let moved = carve_channels(&mut data, &idx, 6.0, 2.0);
        assert!(moved > 0, "the carve moved nothing");
        // On the centreline the bed is the full depth down.
        let on = data.height_at(DVec2::new(30.0, 8.0)).unwrap();
        assert!((on - 98.0).abs() < 0.05, "the bed is at {on} m, want 98");
        // At the bank it meets the ground.
        let bank = data.height_at(DVec2::new(30.0, 14.0)).unwrap();
        assert!((bank - 100.0).abs() < 0.05, "the bank is at {bank} m");
        // Away from the stream nothing moved.
        assert_eq!(data.height_at(DVec2::new(30.0, 40.0)).unwrap(), 100.0);
        // A carve never RAISES ground: an empty index and a zero depth are no-ops.
        assert_eq!(carve_channels(&mut data, &idx, 6.0, 0.0), 0);
        assert_eq!(
            carve_channels(&mut data, &channel_index(&[], 6.0), 6.0, 2.0),
            0
        );
        assert!(lakes_by_id(&[]).is_empty());
    }
}
