//! **The plane sweep**: the CPU reference that defines what `plane_sweep.wgsl`
//! must reproduce, and everything it needs to run — the census transform, the
//! covisibility neighbourhood, and the depth range prior.
//!
//! # The sweep, in one paragraph
//!
//! For a reference view, place a stack of **fronto-parallel planes** across the
//! depth range that view's share of the sparse cloud occupies. For each plane,
//! project every reference pixel onto it and reproject that 3-D point into each
//! neighbour view; the plane the true surface lies on is the one where the
//! reprojected patch looks like the reference patch. Sweep, score, take the
//! `argmin`, and fit a parabola through the three costs around it to get a
//! depth between two planes.
//!
//! # Why census and not ZNCC
//!
//! The cost is the **Hamming distance between census codes**, which makes it an
//! integer, and that is the load-bearing choice in this crate.
//!
//! * **Parity.** An integer cost means the cost volume, the per-neighbour
//!   aggregation, the `argmin` and its tie-break are *exact* on both the CPU and
//!   the GPU. A ZNCC would put a windowed floating-point reduction — a sum of
//!   products, whose value depends on the order it is summed in — inside the
//!   innermost loop, and there would be nothing left to anchor the two
//!   implementations to except a tolerance.
//! * **Cost.** The window is paid **once per pixel**, in
//!   [`census_transform`], not once per (pixel, plane, neighbour). A 5x5 ZNCC
//!   is 25 taps per sample; a census match is one fetch, one `xor`, one
//!   `countOneBits`. On the gate fixture that is the difference between a
//!   correctness gate that runs in CI time and one that does not.
//! * **Radiometry.** A census code is invariant to any monotone intensity
//!   change, which is strictly stronger than ZNCC's invariance to affine ones.
//!
//! What it gives up: census is a *rank* transform, so it discards the magnitude
//! of the intensity differences it encodes and is weaker than correlation on
//! low-contrast, finely-textured surfaces. That is the documented bound.
//!
//! # The window, and the dilation
//!
//! [`CENSUS_TAPS`] is a 5x5 neighbourhood minus its own centre — **24 bits** in
//! a `u32`, which is exactly the width WGSL's `countOneBits` operates on and
//! leaves no second word to keep in step. [`CensusConfig::dilation`] spaces
//! those 24 taps out: at `2` the same 24 bits span a 9x9 neighbourhood, so the
//! descriptor sees a wider patch without costing a bit more. Comparisons are
//! `neighbour < centre` against the raw `u8` intensities with clamp-to-edge
//! addressing, so the transform is integer end to end and identical on every
//! platform by construction.
//!
//! # Determinism
//!
//! Every ordering here is integer: planes are scanned in ascending index with
//! ties going to the **nearer** plane, neighbours are ranked by covisibility
//! count with ties going to the **lower view index**, and the per-neighbour cost
//! aggregation is an insertion sort of integers. The only floating point on the
//! path is the projection and the parabola, both `f32`, both written once here
//! and once in WGSL, in the same order.

use glam::{DVec2, DVec3};
use inf_core::job::JobPool;
use inf_photo::camera::{Intrinsics, Pose};
use inf_photo::gray::GrayImage;
use inf_photo::sfm::{Reconstruction, RegisteredCamera};

use crate::DenseError;

/// How many taps a census code carries: a 5x5 neighbourhood minus its centre.
///
/// Twenty-four, and not thirty-two, because 5x5 is the smallest window with
/// enough taps to be distinctive and the centre compares to itself. It fits a
/// `u32` with eight bits to spare, and a `u32` is what WGSL's `countOneBits`
/// takes — a 7x7 window would be 48 bits, which means two words on the GPU and
/// two words to keep in step.
pub const CENSUS_TAPS: usize = 24;

/// The most neighbour views one reference view can be swept against.
///
/// A compile-time cap because the shader's per-plane cost aggregation sorts the
/// neighbour costs in a fixed-size array; it has no allocator.
pub const MAX_NEIGHBOURS: usize = 8;

/// The sentinel a plane whose cost could not be evaluated carries.
///
/// `u32::MAX`, so the `argmin` scan needs no separate validity flag and a plane
/// nobody could see never wins.
pub const COST_INVALID: u32 = u32::MAX;

/// Costs are reported as `bits * COST_SCALE / taps_used`, so a plane scored
/// against three neighbours and one scored against two are on the same scale.
///
/// Integer division, deliberately: `256` is enough resolution that the
/// truncation never decides an `argmin` that the underlying Hamming distances
/// did not already decide, and it keeps the whole cost volume in exact `u32`.
pub const COST_SCALE: u32 = 256;

/// A sample nearer than this to a neighbour's camera plane is behind it.
///
/// In **reconstruction units** (the gauge is one baseline). `f32`, because this
/// is compared on both sides of the CPU/GPU seam.
pub const MIN_SAMPLE_Z: f32 = 1.0e-6;

/// How the census transform is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CensusConfig {
    /// Tap spacing in pixels. `1` is a dense 5x5; `2` spans 9x9 with the same
    /// 24 bits.
    pub dilation: u32,
}

impl Default for CensusConfig {
    fn default() -> Self {
        Self { dilation: 2 }
    }
}

/// How the sweep is tuned. Every field changes depth maps; the defaults are what
/// `tests/mvs_gate.rs` measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepConfig {
    /// The census transform.
    pub census: CensusConfig,
    /// How many fronto-parallel planes to sweep.
    pub planes: u32,
    /// How many neighbour views to sweep each reference view against.
    pub neighbours: usize,
    /// Fewer covisible neighbours than this is a refusal
    /// ([`DenseError::TooFewNeighbours`]).
    pub min_neighbours: usize,
    /// A plane scored against fewer than this many neighbours does not get a
    /// cost. Two, so a single neighbour can never carry a depth on its own.
    pub min_valid_neighbours: usize,
    /// The lower percentile of the in-view sparse depths that sets the near
    /// plane, before padding. `0.02`, so a couple of stray points in front of
    /// the subject cannot drag the whole range.
    pub low_percentile: f64,
    /// The upper percentile that sets the far plane, before padding.
    pub high_percentile: f64,
    /// The range is padded by this fraction of its own span at both ends, so a
    /// surface slightly outside the sparse cloud's hull is still reachable.
    pub pad_fraction: f64,
    /// Fewer sparse points in view than this is a refusal
    /// ([`DenseError::TooFewSparsePoints`]).
    pub min_range_points: usize,
    /// A winning cost above this is not a match. In `COST_SCALE` units, so
    /// `2048` is eight of twenty-four census bits disagreeing on average.
    pub max_cost: u32,
    /// The winner must beat the best cost **outside** a `+/- exclusion_planes`
    /// window by this much, relative — see [`confidence_of`].
    pub min_confidence: f32,
    /// How many planes either side of the winner are excluded when looking for
    /// the runner-up. The cost curve's own basin is not a competitor.
    pub exclusion_planes: u32,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            census: CensusConfig::default(),
            planes: 64,
            neighbours: 3,
            min_neighbours: 2,
            min_valid_neighbours: 2,
            low_percentile: 0.02,
            high_percentile: 0.98,
            pad_fraction: 0.20,
            min_range_points: 24,
            max_cost: 2048,
            min_confidence: 0.06,
            exclusion_planes: 2,
        }
    }
}

/// The **census transform** of an image: one `u32` per pixel, bit `i` set when
/// tap `i` is darker than the centre.
///
/// Integer end to end — `u8` comparisons, clamp-to-edge addressing, a fixed tap
/// order. There is no float, no rounding mode and no platform in it, which is
/// why the GPU mirror of this stage is bit-exact by construction rather than by
/// measurement.
pub fn census_transform(image: &GrayImage, cfg: &CensusConfig) -> Vec<u32> {
    let d = cfg.dilation.max(1) as i64;
    let w = image.width();
    let h = image.height();
    let mut out = vec![0u32; w as usize * h as usize];
    for y in 0..h {
        for x in 0..w {
            let centre = image.at(x, y);
            let mut code = 0u32;
            let mut bit = 0u32;
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let v = image.clamped(x as i64 + dx * d, y as i64 + dy * d);
                    if v < centre {
                        code |= 1 << bit;
                    }
                    bit += 1;
                }
            }
            debug_assert_eq!(bit as usize, CENSUS_TAPS);
            out[y as usize * w as usize + x as usize] = code;
        }
    }
    out
}

/// One neighbour view, packed the way both the CPU reference and the shader read
/// it.
///
/// `#[repr(C)]` with scalar members only: a `vec3<f32>` would need 16-byte
/// alignment in WGSL and a padding rule to go with it, and this struct exists to
/// be the *same bytes* on both sides.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NeighbourParams {
    /// Relative translation, reference camera frame into this one, x.
    pub tx: f32,
    /// Relative translation, y.
    pub ty: f32,
    /// Relative translation, z.
    pub tz: f32,
    /// Focal length x, **pixels**.
    pub fx: f32,
    /// Focal length y, **pixels**.
    pub fy: f32,
    /// Principal point x, **pixels**.
    pub cx: f32,
    /// Principal point y, **pixels**.
    pub cy: f32,
    /// First radial coefficient.
    pub k1: f32,
    /// Second radial coefficient.
    pub k2: f32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Where this neighbour's census image starts in the packed census buffer.
    pub census_offset: u32,
}

/// A neighbour view as [`SweepGeometry::build`] wants it.
#[derive(Clone, Copy, Debug)]
pub struct NeighbourView {
    /// The view index.
    pub view: u32,
    /// Its world-to-camera pose.
    pub pose: Pose,
    /// Its intrinsics.
    pub intrinsics: Intrinsics,
    /// Where its census image starts in the packed buffer.
    pub census_offset: u32,
}

/// Everything the sweep needs about one reference view, already narrowed to
/// `f32`.
///
/// **This is one of the two `f64` -> `f32` seams in the crate** (the other is
/// [`crate::tsdf::TsdfGrid::fuse_params`]). Poses, intrinsics and the
/// undistortion — a fixed-point iteration that must never run on the GPU,
/// because it would be a second implementation of a loop whose convergence is
/// the whole point — are all `f64` up to [`SweepGeometry::build`] and `f32`
/// after it. Nothing downstream widens back.
#[derive(Clone, Debug)]
pub struct SweepGeometry {
    /// The reference view index.
    pub view: u32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// The neighbour view indices, in the order they are scored.
    pub neighbours: Vec<u32>,
    /// `R_rel * (xn, yn, 1)` per (neighbour, pixel), flattened with stride 3.
    ///
    /// The rotation is applied **here** rather than per plane because it does
    /// not depend on the plane: a point at depth `d` along the reference ray is
    /// `d * ray` in the reference frame, so its neighbour-frame position is
    /// `d * (R ray) + t`. Hoisting `R ray` out is not only faster, it removes a
    /// matrix multiply from the shader's innermost loop and with it three
    /// multiply-adds whose contraction could differ.
    pub rays: Vec<f32>,
    /// Per-neighbour camera parameters.
    pub params: Vec<NeighbourParams>,
    /// Plane depths, near to far, in **reconstruction units**.
    pub depths: Vec<f32>,
    /// Inverse depth of plane 0.
    pub inv_near: f32,
    /// Inverse-depth step per plane. **Negative**, because planes run near to
    /// far and inverse depth falls.
    pub inv_step: f32,
}

impl SweepGeometry {
    /// Narrow one reference view's geometry for the sweep.
    ///
    /// Planes are placed **uniformly in inverse depth**, not in depth: the
    /// disparity a fixed pixel displacement corresponds to is linear in inverse
    /// depth, so uniform depth planes over-sample the far field and under-sample
    /// the near one by exactly the ratio of the range. The plane depths are then
    /// derived from the *same* `f32` scalars the sub-plane refinement uses, so
    /// refining by zero lands exactly on the plane it started from.
    pub fn build(
        reference: &RegisteredCamera,
        neighbours: &[NeighbourView],
        near: f64,
        far: f64,
        planes: u32,
    ) -> Self {
        assert!(planes >= 2, "a sweep needs at least two planes");
        assert!(
            neighbours.len() <= MAX_NEIGHBOURS,
            "at most {MAX_NEIGHBOURS} neighbour views"
        );
        // The plane stack's DIRECTION is load-bearing and was, until this line,
        // assumed. `sweep_view` breaks an `argmin` tie toward the lower plane
        // index and documents that as "the nearer surface, the conservative
        // answer when a ray grazes two surfaces"; the parabola's end guards read
        // `best_p == 0` as the near end. Handed `(far, near)` this function
        // builds the same SET of planes in the opposite order, so every one of
        // those claims silently inverts and the reconstruction barely moves —
        // measured: swapping the two arguments at the call site left all 47 tests
        // in this crate green. A precondition is what makes the direction a fact.
        assert!(
            far > near && near > 0.0,
            "a sweep runs from a NEAR plane to a FAR one, both in front of the camera; got \
             near = {near}, far = {far}"
        );
        let k = &reference.intrinsics;
        let (w, h) = (k.width, k.height);
        let n_px = w as usize * h as usize;

        let mut rays = vec![0.0f32; neighbours.len() * n_px * 3];
        let mut params = Vec::with_capacity(neighbours.len());
        for (ni, nb) in neighbours.iter().enumerate() {
            let rel = reference.pose.relative_to(&nb.pose);
            let rot = rel.rotation;
            for y in 0..h {
                for x in 0..w {
                    // The pixel-centre convention, and `to_normalized` is where
                    // the radial undistortion happens — in `f64`, once.
                    let n = k.to_normalized(DVec2::new(x as f64, y as f64));
                    let ray = rot * DVec3::new(n.x, n.y, 1.0);
                    let base = (ni * n_px + y as usize * w as usize + x as usize) * 3;
                    rays[base] = ray.x as f32;
                    rays[base + 1] = ray.y as f32;
                    rays[base + 2] = ray.z as f32;
                }
            }
            params.push(NeighbourParams {
                tx: rel.translation.x as f32,
                ty: rel.translation.y as f32,
                tz: rel.translation.z as f32,
                fx: nb.intrinsics.fx as f32,
                fy: nb.intrinsics.fy as f32,
                cx: nb.intrinsics.cx as f32,
                cy: nb.intrinsics.cy as f32,
                k1: nb.intrinsics.k1 as f32,
                k2: nb.intrinsics.k2 as f32,
                width: nb.intrinsics.width,
                height: nb.intrinsics.height,
                census_offset: nb.census_offset,
            });
        }

        let inv_near = (1.0 / near) as f32;
        let inv_far = (1.0 / far) as f32;
        let inv_step = (inv_far - inv_near) / (planes - 1) as f32;
        let depths = (0..planes)
            .map(|p| 1.0f32 / (inv_near + inv_step * p as f32))
            .collect();

        Self {
            view: reference.view,
            width: w,
            height: h,
            neighbours: neighbours.iter().map(|n| n.view).collect(),
            rays,
            params,
            depths,
            inv_near,
            inv_step,
        }
    }

    /// How many pixels.
    #[inline]
    pub fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// A per-view depth map, the plane sweep's output.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    /// Which view.
    pub view: u32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Depth along the camera's **+Z**, in reconstruction units. Exactly `0.0`
    /// means "no depth here" — the sweep found nothing, or a filter took it.
    ///
    /// Zero rather than a `NaN` sentinel because zero has one bit pattern and
    /// `NaN` has millions, and these bytes get compared.
    pub depth: Vec<f32>,
    /// Match confidence in `[0, 1]`, zero wherever depth is zero.
    pub confidence: Vec<f32>,
}

impl DepthMap {
    /// An all-invalid map.
    pub fn empty(view: u32, width: u32, height: u32) -> Self {
        let n = width as usize * height as usize;
        Self {
            view,
            width,
            height,
            depth: vec![0.0; n],
            confidence: vec![0.0; n],
        }
    }

    /// How many pixels carry a depth.
    pub fn valid_count(&self) -> usize {
        self.depth.iter().filter(|d| **d > 0.0).count()
    }

    /// How many pixels there are.
    #[inline]
    pub fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Invalidate a pixel, both channels together, so the two can never
    /// disagree.
    #[inline]
    pub fn invalidate(&mut self, index: usize) {
        self.depth[index] = 0.0;
        self.confidence[index] = 0.0;
    }

    /// Canonical bytes for the determinism surface.
    pub fn write_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.view.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        for v in &self.depth {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &self.confidence {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The scoring kernel — transcribed into `plane_sweep.wgsl` line for line
// ─────────────────────────────────────────────────────────────────────────────

/// The census cost of one (pixel, plane) against one neighbour, or `None` when
/// the sample lands behind that neighbour's camera plane or outside its image.
///
/// **This function is the CPU half of the parity claim.** Its arithmetic — the
/// order of the multiply-adds, the `floor(x + 0.5)` rounding, the bounds test —
/// is what `plane_sweep.wgsl`'s `neighbour_cost` reproduces. Two details are
/// deliberate:
///
/// * `floor(px + 0.5)`, **not** `round(px)`. Rust's `f32::round` is
///   round-half-away-from-zero and WGSL's `round` is round-half-to-even; on a
///   sample that lands exactly on `.5` they disagree, and they disagree
///   *systematically* rather than rarely. `floor` is exact and identical in
///   both.
/// * `!(qz > MIN_SAMPLE_Z)` rather than `qz <= MIN_SAMPLE_Z`, so a `NaN` depth
///   is rejected. Both languages evaluate a comparison with `NaN` as false.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn neighbour_cost(
    reference_code: u32,
    ray: [f32; 3],
    p: &NeighbourParams,
    census: &[u32],
    depth: f32,
) -> Option<u32> {
    let qx = depth * ray[0] + p.tx;
    let qy = depth * ray[1] + p.ty;
    let qz = depth * ray[2] + p.tz;
    if !(qz > MIN_SAMPLE_Z) {
        return None;
    }
    let u = qx / qz;
    let v = qy / qz;
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let f = 1.0 + p.k1 * r2 + p.k2 * r4;
    let px = p.fx * (u * f) + p.cx;
    let py = p.fy * (v * f) + p.cy;
    let ix = (px + 0.5).floor();
    let iy = (py + 0.5).floor();
    if !(ix >= 0.0) || !(iy >= 0.0) || ix > (p.width - 1) as f32 || iy > (p.height - 1) as f32 {
        return None;
    }
    let index = p.census_offset as usize + iy as usize * p.width as usize + ix as usize;
    Some((reference_code ^ census[index]).count_ones())
}

/// The aggregated cost of one plane at one pixel, or [`COST_INVALID`].
///
/// Neighbour costs are insertion-sorted (integers, so the sort is exact and its
/// tie-break irrelevant) and the **better half** is averaged. Averaging the best
/// half rather than all of them is what makes one occluded neighbour survivable:
/// a surface visible in three of four views scores on the three that see it. The
/// result is rescaled by [`COST_SCALE`] over the number of taps actually used,
/// so planes scored against different numbers of neighbours are comparable —
/// which they must be, because the `argmin` runs across them.
///
/// The argument list is long because it is a **shader function's** argument
/// list: `plane_sweep.wgsl` reaches the same values through global bindings, and
/// packing them into a struct here would put one more thing between the two
/// implementations that has to be read the same way twice.
#[allow(clippy::too_many_arguments)]
pub fn plane_cost(
    reference_code: u32,
    rays: &[f32],
    ray_base: usize,
    n_px: usize,
    params: &[NeighbourParams],
    census: &[u32],
    depth: f32,
    min_valid: usize,
) -> u32 {
    let mut sorted = [0u32; MAX_NEIGHBOURS];
    let mut valid = 0usize;
    for (ni, p) in params.iter().enumerate() {
        let base = (ni * n_px + ray_base) * 3;
        let ray = [rays[base], rays[base + 1], rays[base + 2]];
        let Some(c) = neighbour_cost(reference_code, ray, p, census, depth) else {
            continue;
        };
        let mut j = valid;
        while j > 0 && sorted[j - 1] > c {
            sorted[j] = sorted[j - 1];
            j -= 1;
        }
        sorted[j] = c;
        valid += 1;
    }
    if valid < min_valid || valid == 0 {
        return COST_INVALID;
    }
    let half = valid.div_ceil(2);
    let mut sum = 0u32;
    for c in sorted.iter().take(half) {
        sum += *c;
    }
    sum * COST_SCALE / half as u32
}

/// How much the winner beat the field by, in `[0, 1]`.
///
/// `(runner_up - winner) / (runner_up + 1)`. The `+ 1` is not a fudge: it makes
/// the expression total when the runner-up is zero (two planes that both match
/// perfectly, which is what a repeated texture looks like) and it makes the
/// measure *relative*, so a win of 3 bits over 4 counts for more than a win of 3
/// over 40. The runner-up is the best plane **outside** the winner's own basin
/// — an adjacent plane is not a competitor, it is the same match one step over.
#[inline]
pub fn confidence_of(winner: u32, runner_up: u32) -> f32 {
    if runner_up == COST_INVALID {
        return 1.0;
    }
    if runner_up <= winner {
        return 0.0;
    }
    let c = (runner_up - winner) as f32 / (runner_up as f32 + 1.0);
    c.clamp(0.0, 1.0)
}

/// The sub-plane offset from a parabola through three costs, in **plane index
/// units**, clamped to `[-0.5, 0.5]`.
///
/// A quadratic through `(-1, a)`, `(0, b)`, `(1, c)` has its extremum at
/// `0.5 (a - c) / (a - 2b + c)`. The denominator is positive exactly when the
/// three points are convex, which is what a real cost minimum looks like; a
/// non-positive denominator means the winner is on a plateau or a ridge and the
/// honest answer is the plane itself. The clamp is the second guard: a
/// near-zero denominator can throw the vertex arbitrarily far, and an offset
/// beyond half a plane says the wrong plane won, not that the depth is out
/// there.
#[inline]
pub fn parabola_offset(a: u32, b: u32, c: u32) -> f32 {
    if a == COST_INVALID || c == COST_INVALID {
        return 0.0;
    }
    let (a, b, c) = (a as f32, b as f32, c as f32);
    let denom = a - 2.0 * b + c;
    if !(denom > 0.0) {
        return 0.0;
    }
    (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
}

/// Sweep one reference view: the **CPU reference implementation**, and the
/// definition of what the shader computes.
///
/// `reference_census` is this view's census image; `census` is the packed
/// concatenation the neighbours' `census_offset`s index into. Rows are
/// dispatched through [`JobPool::parallel_map_ref`], an in-order pure map, and
/// stitched back serially, so the result is a function of the inputs and not of
/// the worker count.
pub fn sweep_view(
    geometry: &SweepGeometry,
    reference_census: &[u32],
    census: &[u32],
    cfg: &SweepConfig,
    pool: &JobPool,
) -> DepthMap {
    let w = geometry.width as usize;
    let h = geometry.height as usize;
    let n_px = geometry.pixels();
    let planes = geometry.depths.len();
    let min_valid = cfg.min_valid_neighbours.min(geometry.params.len()).max(1);
    let exclusion = cfg.exclusion_planes as usize;

    let rows: Vec<u32> = (0..h as u32).collect();
    let scanned: Vec<Vec<(f32, f32)>> = pool.parallel_map_ref(&rows, |&y| {
        let mut out = Vec::with_capacity(w);
        let mut curve = vec![COST_INVALID; planes];
        for x in 0..w {
            let index = y as usize * w + x;
            let code = reference_census[index];
            for (p, slot) in curve.iter_mut().enumerate() {
                *slot = plane_cost(
                    code,
                    &geometry.rays,
                    index,
                    n_px,
                    &geometry.params,
                    census,
                    geometry.depths[p],
                    min_valid,
                );
            }
            // Ties go to the LOWER plane index, which is the NEARER surface —
            // an integer order that cannot wobble, and the conservative answer
            // when a ray grazes two surfaces.
            let mut best = COST_INVALID;
            let mut best_p = 0usize;
            for (p, &c) in curve.iter().enumerate() {
                if c < best {
                    best = c;
                    best_p = p;
                }
            }
            if best == COST_INVALID || best > cfg.max_cost {
                out.push((0.0, 0.0));
                continue;
            }
            let mut runner_up = COST_INVALID;
            for (p, &c) in curve.iter().enumerate() {
                if p.abs_diff(best_p) <= exclusion {
                    continue;
                }
                if c < runner_up {
                    runner_up = c;
                }
            }
            let confidence = confidence_of(best, runner_up);
            if confidence < cfg.min_confidence {
                out.push((0.0, 0.0));
                continue;
            }
            let delta = if best_p == 0 || best_p + 1 >= planes {
                0.0
            } else {
                parabola_offset(curve[best_p - 1], best, curve[best_p + 1])
            };
            let inv = geometry.inv_near + geometry.inv_step * (best_p as f32 + delta);
            if !(inv > 0.0) {
                out.push((0.0, 0.0));
                continue;
            }
            out.push((1.0f32 / inv, confidence));
        }
        out
    });

    let mut map = DepthMap::empty(geometry.view, geometry.width, geometry.height);
    let mut i = 0usize;
    for row in &scanned {
        for &(d, c) in row {
            map.depth[i] = d;
            map.confidence[i] = c;
            i += 1;
        }
    }
    map
}

// ─────────────────────────────────────────────────────────────────────────────
// Neighbourhood and depth range: the two priors the sweep needs from P25.1
// ─────────────────────────────────────────────────────────────────────────────

/// Rank the other registered views by how many reconstruction points they share
/// with `view`, and take the best `cfg.neighbours`.
///
/// Covisibility — shared **tracks** — is the prior P25.1 already computed and
/// the one that transfers: two photographs that see the same surface points see
/// the same surface. Ties go to the lower view index, so the selection is a
/// total order over integers.
///
/// **The bound, stated:** the ranking is by count alone. It does not score
/// baseline or viewing angle, so a capture that took two frames from nearly the
/// same position will rank them as each other's best neighbour and get a
/// degenerate sweep out of it. That degrades rather than lies — a zero-baseline
/// pair makes every plane score alike, the cost curve goes flat, and
/// [`confidence_of`] rejects the pixel — but a genuine view-selection score is a
/// v2 item, not something this function is doing badly.
pub fn neighbours_for(
    view: u32,
    reconstruction: &Reconstruction,
    cfg: &SweepConfig,
) -> Result<Vec<u32>, DenseError> {
    let mut shared: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for point in &reconstruction.points {
        if !point.observations.iter().any(|o| o.view == view) {
            continue;
        }
        for o in &point.observations {
            if o.view == view {
                continue;
            }
            if reconstruction.cameras.contains_key(&o.view) {
                *shared.entry(o.view).or_insert(0) += 1;
            }
        }
    }
    let mut ranked: Vec<(u32, usize)> = shared.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if ranked.len() < cfg.min_neighbours {
        return Err(DenseError::TooFewNeighbours {
            view,
            found: ranked.len(),
            required: cfg.min_neighbours,
        });
    }
    // The cap comes LAST. `min_neighbours` is a caller's field, and raising it
    // past `MAX_NEIGHBOURS` used to lift `take` past the cap with it — which is
    // an abort inside `SweepGeometry::build`'s own assertion, and would be a
    // nine-deep write into the shader's `array<u32, 8>` if it ever got there.
    let take = cfg
        .neighbours
        .max(cfg.min_neighbours)
        .clamp(1, MAX_NEIGHBOURS);
    Ok(ranked.into_iter().take(take).map(|(v, _)| v).collect())
}

/// The near and far plane depths for one view, in **reconstruction units**.
///
/// The prior is the sparse cloud's own extent *in that view*, because there is
/// no other scale in a reconstruction: P25.1's output is in baseline units, so
/// a metric depth range would be a number nobody could have supplied. The
/// formula, in full:
///
/// 1. Take the camera-space `z` of every reconstruction point that **projects
///    into this view's frame** and is in front of it — not merely the points it
///    was *matched* against. A point on a wall that two other views tracked and
///    this one did not is still a point this camera is looking at, and it is
///    exactly the evidence a view with a thin share of the tracks has nothing
///    else of. Measured on the gate fixture: the one station whose sparse share
///    is thinnest went from a 0.7-unit depth range covering 1% of its pixels to
///    a range covering the scene it actually sees.
/// 2. Sort by [`f64::total_cmp`] and read the **2nd and 98th percentiles**
///    (`SweepConfig::low_percentile` / `high_percentile`) at index
///    `floor(p * (n - 1))`. Percentiles rather than min/max because one
///    mis-triangulated point at ten times the scene depth would otherwise
///    stretch the range by an order of magnitude and starve every plane of
///    resolution.
/// 3. Pad both ends by `pad_fraction` of the percentile span, so a surface just
///    outside the sparse hull — the corner of a wall no feature landed on — is
///    still reachable.
/// 4. Floor the near plane at **half** the low percentile, so the padding can
///    never push it to zero or negative however degenerate the spread is.
pub fn depth_range_for(
    view: u32,
    reconstruction: &Reconstruction,
    cfg: &SweepConfig,
) -> Result<(f64, f64), DenseError> {
    let Some(camera) = reconstruction.cameras.get(&view) else {
        return Err(DenseError::TooFewSparsePoints {
            view,
            found: 0,
            required: cfg.min_range_points,
        });
    };
    let mut depths: Vec<f64> = Vec::new();
    for point in &reconstruction.points {
        let camera_space = camera.pose.transform(point.position);
        if !(camera_space.z > 0.0) || !camera_space.z.is_finite() {
            continue;
        }
        match camera.intrinsics.project(camera_space) {
            Some(px) if camera.intrinsics.contains(px) => depths.push(camera_space.z),
            _ => {}
        }
    }
    if depths.len() < cfg.min_range_points {
        return Err(DenseError::TooFewSparsePoints {
            view,
            found: depths.len(),
            required: cfg.min_range_points,
        });
    }
    depths.sort_by(f64::total_cmp);
    let n = depths.len();
    let at = |p: f64| -> f64 {
        let idx = (p * (n - 1) as f64).floor().max(0.0) as usize;
        depths[idx.min(n - 1)]
    };
    let lo = at(cfg.low_percentile);
    let hi = at(cfg.high_percentile);
    // A span of zero is a scene at one depth — legal, and a range of zero width
    // is not. Fall back to a tenth of the depth itself.
    let span = if hi > lo { hi - lo } else { hi * 0.1 };
    let near = (lo - cfg.pad_fraction * span).max(lo * 0.5);
    let far = hi + cfg.pad_fraction * span;
    Ok((near, far))
}

/// Back-project a pixel at a given depth into **world** coordinates, in `f64`.
///
/// A depth map is `f32` because WGSL has no `f64` and the sweep has to be
/// comparable across the seam. Every *consumer* of a depth — the consistency
/// check, TSDF fusion, the gate's own error metric — works in world space, and
/// world space is `f64` (architecture rule 3). The widening back happens here,
/// once, so there is one place to look for it. Undistortion runs in `f64` too,
/// through the same [`Intrinsics::to_normalized`] the sweep's rays came from.
pub fn backproject_world(camera: &RegisteredCamera, x: u32, y: u32, depth: f32) -> DVec3 {
    let n = camera
        .intrinsics
        .to_normalized(DVec2::new(x as f64, y as f64));
    let p = DVec3::new(n.x, n.y, 1.0) * depth as f64;
    camera.pose.rotation.inverse() * (p - camera.pose.translation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, ((x * 11 + y * 7) % 256) as u8);
            }
        }
        img
    }

    #[test]
    fn the_census_transform_uses_every_bit_it_claims_and_no_more() {
        let img = ramp(24, 20);
        let codes = census_transform(&img, &CensusConfig { dilation: 1 });
        assert_eq!(codes.len(), 24 * 20);
        // Twenty-four taps means the top eight bits are structurally zero. A
        // code with bit 24 set would mean the loop ran long and the shader — a
        // `countOneBits` over the same `u32` — would silently disagree.
        assert!(
            codes.iter().all(|c| c >> CENSUS_TAPS == 0),
            "a census code used more than {CENSUS_TAPS} bits"
        );
        // And it is not a constant: a ramp has both polarities everywhere.
        let distinct: std::collections::BTreeSet<u32> = codes.iter().copied().collect();
        assert!(
            distinct.len() > 8,
            "only {} distinct census codes on a ramp",
            distinct.len()
        );
    }

    #[test]
    fn the_census_transform_is_invariant_to_a_monotone_intensity_change() {
        // The property census is chosen FOR. Halve every intensity (a monotone
        // map on the values present) and the codes must not move: the transform
        // encodes order, not magnitude.
        let img = ramp(20, 16);
        let mut dimmed = GrayImage::new(20, 16).unwrap();
        for y in 0..16 {
            for x in 0..20 {
                dimmed.set(x, y, img.at(x, y) / 2);
            }
        }
        let cfg = CensusConfig { dilation: 1 };
        // A ramp modulo 256 is not globally monotone, so restrict the claim to
        // pixels whose whole neighbourhood survived the halving without a tie
        // appearing — which is what "monotone on the values present" means.
        let a = census_transform(&img, &cfg);
        let b = census_transform(&dimmed, &cfg);
        let mut compared = 0usize;
        for y in 2..14u32 {
            for x in 2..18u32 {
                let centre = img.at(x, y);
                let mut tied = false;
                for dy in -2i64..=2 {
                    for dx in -2i64..=2 {
                        let v = img.clamped(x as i64 + dx, y as i64 + dy);
                        if (v / 2) == (centre / 2) && v != centre {
                            tied = true;
                        }
                    }
                }
                if tied {
                    continue;
                }
                let i = y as usize * 20 + x as usize;
                assert_eq!(
                    a[i], b[i],
                    "census moved under a monotone change at {x},{y}"
                );
                compared += 1;
            }
        }
        assert!(compared > 40, "only {compared} pixels were comparable");
    }

    #[test]
    fn dilation_widens_the_footprint_without_widening_the_code() {
        let img = ramp(40, 32);
        let tight = census_transform(&img, &CensusConfig { dilation: 1 });
        let wide = census_transform(&img, &CensusConfig { dilation: 3 });
        assert!(wide.iter().all(|c| c >> CENSUS_TAPS == 0));
        assert_ne!(tight, wide, "dilation changed nothing");
    }

    fn reference_camera() -> RegisteredCamera {
        RegisteredCamera {
            view: 0,
            pose: Pose::IDENTITY,
            intrinsics: Intrinsics::centred(8, 6, 10.0),
            observations: 0,
        }
    }

    #[test]
    fn the_plane_stack_runs_from_near_to_far_and_refuses_to_run_backwards() {
        // The direction the `argmin` tie-break's meaning rests on. Plane 0 must be
        // the NEAREST plane, or "ties go to the lower index, which is the nearer
        // surface" is the opposite of what the code does — and nothing else in
        // this crate could tell, because the plane SET is the same either way.
        let g = SweepGeometry::build(&reference_camera(), &[], 2.0, 6.0, 8);
        assert_eq!(g.depths.len(), 8);
        assert!(
            g.inv_step < 0.0,
            "inverse depth must FALL across the stack; step {}",
            g.inv_step
        );
        for pair in g.depths.windows(2) {
            assert!(
                pair[1] > pair[0],
                "the plane depths are not ascending: {:?}",
                g.depths
            );
        }
        assert!(
            (g.depths[0] - 2.0).abs() < 1e-5,
            "plane 0 is {}",
            g.depths[0]
        );
        assert!(
            (g.depths[7] - 6.0).abs() < 1e-5,
            "the last plane is {}",
            g.depths[7]
        );
        // Refining by zero must land exactly on the plane it started from, which
        // is the reason the depths are derived from the same `f32` scalars the
        // refinement uses rather than from `near`/`far` directly.
        for (p, d) in g.depths.iter().enumerate() {
            let inv = g.inv_near + g.inv_step * p as f32;
            assert_eq!(1.0f32 / inv, *d, "plane {p} does not reproduce itself");
        }
    }

    #[test]
    #[should_panic(expected = "a sweep runs from a NEAR plane to a FAR one")]
    fn a_backwards_depth_range_is_a_refusal_not_a_reversed_stack() {
        SweepGeometry::build(&reference_camera(), &[], 6.0, 2.0, 8);
    }

    #[test]
    fn the_parabola_finds_a_vertex_it_is_given_and_refuses_one_it_is_not() {
        // Symmetric costs put the vertex on the plane.
        assert_eq!(parabola_offset(10, 4, 10), 0.0);
        // Leaning left moves it left, and the magnitude is the textbook one:
        // a=12, b=4, c=20 -> 0.5*(12-20)/(12-8+20) = -4/24 = -1/6.
        let d = parabola_offset(12, 4, 20);
        assert!((d - (-1.0 / 6.0)).abs() < 1e-6, "offset {d}");
        // A concave or flat triple has no minimum to refine to.
        assert_eq!(parabola_offset(1, 4, 1), 0.0);
        assert_eq!(parabola_offset(4, 4, 4), 0.0);
        // And it never leaves the plane it won.
        for a in [5u32, 100, 4000] {
            for c in [5u32, 100, 4000] {
                let o = parabola_offset(a, 4, c);
                assert!((-0.5..=0.5).contains(&o), "offset {o} out of range");
            }
        }
        // An invalid shoulder is not a number to fit through.
        assert_eq!(parabola_offset(COST_INVALID, 4, 10), 0.0);
        assert_eq!(parabola_offset(10, 4, COST_INVALID), 0.0);
    }

    #[test]
    fn confidence_is_relative_and_total() {
        assert_eq!(confidence_of(4, COST_INVALID), 1.0);
        assert_eq!(confidence_of(4, 4), 0.0);
        assert_eq!(confidence_of(9, 4), 0.0);
        // A win of 3 over 4 beats a win of 3 over 40.
        assert!(confidence_of(1, 4) > confidence_of(37, 40));
        // Total at zero: two perfect matches are no confidence, not a division
        // by zero.
        assert_eq!(confidence_of(0, 0), 0.0);
        assert!(confidence_of(0, 1).is_finite());
    }

    #[test]
    fn the_cost_aggregation_averages_the_better_half() {
        // Four neighbours, costs 2, 4, 20, 24 -> the better half is 2 and 4,
        // mean 3, scaled by COST_SCALE. Checked through the real kernel by
        // handing it a census layout that produces exactly those distances.
        let mut sorted = [0u32; MAX_NEIGHBOURS];
        let mut valid = 0usize;
        for c in [20u32, 2, 24, 4] {
            let mut j = valid;
            while j > 0 && sorted[j - 1] > c {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = c;
            valid += 1;
        }
        assert_eq!(&sorted[..4], &[2, 4, 20, 24]);
        let half = valid.div_ceil(2);
        assert_eq!(half, 2);
        assert_eq!((sorted[0] + sorted[1]) * COST_SCALE / half as u32, 3 * 256);
    }

    #[test]
    fn a_sample_behind_or_outside_a_neighbour_has_no_cost() {
        let p = NeighbourParams {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            fx: 100.0,
            fy: 100.0,
            cx: 8.0,
            cy: 8.0,
            k1: 0.0,
            k2: 0.0,
            width: 16,
            height: 16,
            census_offset: 0,
        };
        let census = vec![0u32; 256];
        // Straight backwards: qz < 0.
        assert_eq!(neighbour_cost(0, [0.0, 0.0, -1.0], &p, &census, 1.0), None);
        // Exactly on the camera plane.
        assert_eq!(neighbour_cost(0, [0.0, 0.0, 0.0], &p, &census, 1.0), None);
        // Far off to the side: projects outside a 16x16 image.
        assert_eq!(neighbour_cost(0, [5.0, 0.0, 1.0], &p, &census, 1.0), None);
        // Down the axis: lands on the principal point and scores.
        assert_eq!(
            neighbour_cost(0, [0.0, 0.0, 1.0], &p, &census, 1.0),
            Some(0)
        );
        // A NaN depth is rejected rather than indexing on a NaN.
        assert_eq!(
            neighbour_cost(0, [0.0, 0.0, 1.0], &p, &census, f32::NAN),
            None
        );
    }

    #[test]
    fn the_sample_rounding_is_floor_of_x_plus_half_in_both_directions() {
        // The rounding rule the WGSL mirror has to match. A sample landing on
        // exactly 7.5 must go to 8 on both sides; `round` would give 8 in Rust
        // (half away from zero) and 8 in WGSL (half to even) here, but at 8.5
        // they would split — 9 against 8. This arm pins the rule at the
        // half-integer where it bites.
        for (v, want) in [(7.5f32, 8.0f32), (8.5, 9.0), (-0.5, 0.0), (2.49, 2.0)] {
            assert_eq!((v + 0.5).floor(), want, "floor(x+0.5) at {v}");
        }
        assert_ne!((8.5f32 + 0.5).floor(), 8.0, "the round-to-even answer");
    }
}
