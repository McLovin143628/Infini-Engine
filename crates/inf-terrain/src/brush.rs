//! Sculpt **brushes**: pure, deterministic, resolution-independent height edits.
//!
//! Every brush is a function of its [`BrushParams`] and [`BrushOp`] evaluated at
//! **world positions** on the terrain's own sample lattice, so the same stroke
//! produces the same heights regardless of tile layout, tile resolution, or the
//! order dabs are applied (for non-overlapping dabs). Each application returns a
//! [`HeightDelta`] — the sparse before/after record the editor's undo stack
//! stores and [`TerrainData::revert_delta`] replays.
//!
//! * **Local ops** ([`BrushOp::Raise`]/[`Lower`](BrushOp::Lower)/[`Noise`](BrushOp::Noise))
//!   are per-sample and drive tile iteration directly (they may auto-author tiles).
//! * **Neighbourhood ops** ([`BrushOp::Smooth`]/[`Flatten`](BrushOp::Flatten)) run
//!   through a [`HeightRegion`](crate::HeightRegion) read-modify-write and never
//!   author outside existing space.
//!
//! A [`Stroke`] merges the dabs of one mouse-down→up gesture into a single
//! [`HeightDelta`] (before = first touch, after = final).

use glam::{DVec2, DVec3};

use crate::data::TerrainData;
use crate::delta::{DeltaBuilder, HeightDelta};
use crate::noise::fbm_signed;

/// How a brush's influence falls off from full weight at the centre to zero at
/// the radius. `weight(0) == 1`, `weight(≥ radius) == 0`, monotonically decreasing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Falloff {
    /// Smoothstep `s²(3 − 2s)` in `s = 1 − d/r` — the soft default (C¹ at both ends).
    Smooth,
    /// Hemispherical dome `√(1 − (d/r)²)`.
    Sphere,
    /// Linear ramp `1 − d/r`.
    Linear,
    /// Quadratic `(1 − d/r)²` — a sharper, more pointed peak.
    Sharp,
    /// Full weight `1` within `flat_fraction · r`, then smoothstep down to `0` at
    /// `r`. `flat_fraction` is clamped to `[0, 1]`.
    Plateau(f64),
}

impl Falloff {
    /// Brush weight in `[0, 1]` at distance `dist` from the centre for radius
    /// `radius`. `0` at or beyond the radius, `1` at the centre.
    ///
    /// # A non-finite argument weighs nothing (C4-36)
    ///
    /// This was the house's standing NaN-blind-guard class, verbatim, on the
    /// `.inf_terrain` write path. `radius <= 0.0` is false for NaN; `dist >=
    /// radius` is false for NaN; and `(dist / radius).clamp(0.0, 1.0)`
    /// **propagates** a NaN rather than rejecting it, because `f64::clamp`
    /// returns `self` when `self` is NaN. So a NaN reached every arm below and
    /// came back out as a weight — which the caller then multiplied a height by
    /// and wrote into a tile.
    ///
    /// The only thing standing in front of it was an accidental `.max(0.0)` at
    /// the Tauri door, which guards `radius` and not `dist`, and which nothing
    /// in Ring 0 goes through at all. Refusing here is the check that is true
    /// for every caller.
    pub fn weight(&self, dist: f64, radius: f64) -> f64 {
        if !(dist.is_finite() && radius.is_finite()) {
            return 0.0;
        }
        if radius <= 0.0 {
            return if dist <= 0.0 { 1.0 } else { 0.0 };
        }
        if dist >= radius {
            return 0.0;
        }
        let t = (dist / radius).clamp(0.0, 1.0); // 0 centre .. 1 edge
        let s = 1.0 - t; // 1 centre .. 0 edge
        match self {
            Falloff::Linear => s,
            Falloff::Sharp => s * s,
            Falloff::Smooth => s * s * (3.0 - 2.0 * s),
            Falloff::Sphere => (1.0 - t * t).max(0.0).sqrt(),
            Falloff::Plateau(frac) => {
                let f = frac.clamp(0.0, 1.0);
                if t <= f || f >= 1.0 {
                    1.0
                } else {
                    let u = (t - f) / (1.0 - f); // 0 at plateau edge .. 1 at radius
                    let su = 1.0 - u;
                    su * su * (3.0 - 2.0 * su)
                }
            }
        }
    }
}

/// The spatial parameters shared by every brush op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushParams {
    /// Brush centre in world XZ.
    pub center: DVec2,
    /// Brush radius in world metres.
    pub radius: f64,
    /// Strength. **Metres at full weight** for [`BrushOp::Raise`]/[`Lower`]/
    /// [`Noise`](BrushOp::Noise) amplitude scaling context; a **blend fraction**
    /// (clamped to `[0, 1]`) for [`Smooth`](BrushOp::Smooth)/[`Flatten`](BrushOp::Flatten).
    pub strength: f64,
    /// The falloff curve.
    pub falloff: Falloff,
}

impl BrushParams {
    /// A brush at `center` with the smooth falloff.
    pub fn new(center: DVec2, radius: f64, strength: f64) -> Self {
        Self {
            center,
            radius,
            strength,
            falloff: Falloff::Smooth,
        }
    }

    /// World-space AABB `[min, max]` the brush disk can touch.
    #[inline]
    pub fn aabb(&self) -> (DVec2, DVec2) {
        let r = DVec2::splat(self.radius.max(0.0));
        (self.center - r, self.center + r)
    }
}

/// The target height a [`BrushOp::Flatten`] converges toward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlattenTarget {
    /// A caller-picked absolute world height (e.g. the height under the cursor).
    PickedHeight(f64),
    /// The mean world height of the affected (in-radius, authored) samples.
    Mean,
}

/// The sculpt operation a brush performs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrushOp {
    /// Add `strength · weight` metres. Auto-authors tiles where it writes nonzero.
    Raise,
    /// Subtract `strength · weight` metres. Auto-authors like [`Raise`](BrushOp::Raise).
    Lower,
    /// Blend each in-radius sample toward its neighbourhood mean, `iterations`
    /// times, by `weight · strength`. Reads across tile seams seamlessly; never
    /// authors new tiles.
    Smooth {
        /// Number of box-blur passes.
        iterations: u32,
    },
    /// Blend each in-radius sample toward `target` by `weight · strength`. Never
    /// authors new tiles.
    Flatten {
        /// What height to flatten toward.
        target: FlattenTarget,
    },
    /// Add seeded fBm displacement `amplitude · weight · fbm(worldpos)` — signed,
    /// world-anchored (independent of the stroke path). Auto-authors like
    /// [`Raise`](BrushOp::Raise).
    Noise {
        /// fBm seed.
        seed: u64,
        /// Base spatial frequency (cycles per world unit).
        frequency: f64,
        /// Number of fBm octaves.
        octaves: u32,
        /// Peak displacement in metres at full weight.
        amplitude: f64,
    },
}

/// Apply one brush dab to `data`, returning the reversible [`HeightDelta`].
pub fn apply_brush(data: &mut TerrainData, op: BrushOp, params: BrushParams) -> HeightDelta {
    let mut builder = DeltaBuilder::default();
    apply_op(data, &op, &params, &mut builder);
    builder.finalize(data)
}

/// Dispatch one dab into a (possibly shared, for [`Stroke`]) delta builder.
fn apply_op(
    data: &mut TerrainData,
    op: &BrushOp,
    params: &BrushParams,
    builder: &mut DeltaBuilder,
) {
    match *op {
        BrushOp::Raise => {
            let strength = params.strength;
            apply_local(data, params, builder, move |_, _, old, w| {
                old + strength * w
            });
        }
        BrushOp::Lower => {
            let strength = params.strength;
            apply_local(data, params, builder, move |_, _, old, w| {
                old - strength * w
            });
        }
        BrushOp::Noise {
            seed,
            frequency,
            octaves,
            amplitude,
        } => {
            apply_local(data, params, builder, move |wx, wz, old, w| {
                old + amplitude * w * fbm_signed(seed, frequency, octaves, wx, wz)
            });
        }
        BrushOp::Smooth { iterations } => apply_smooth(data, params, iterations, builder),
        BrushOp::Flatten { target } => apply_flatten(data, params, target, builder),
    }
}

/// Clamped `[lo, hi]` local-sample range along one axis intersecting `[minw, maxw]`.
/// Returns `None` when no sample of the tile falls in the range.
fn axis_range(minw: f64, maxw: f64, origin: f64, mps: f64, res: u32) -> Option<(u32, u32)> {
    let lo = ((minw - origin) / mps).ceil().max(0.0);
    let hi = ((maxw - origin) / mps).floor().min((res - 1) as f64);
    if lo > hi {
        None
    } else {
        Some((lo as u32, hi as u32))
    }
}

/// Drive a per-sample **local** op across every tile the brush overlaps. `value`
/// maps `(world_x, world_z, old_world_height, weight)` → new world height; a
/// sample whose new `f32` offset differs from the old is written. A tile that
/// does not yet exist is created only if at least one of its samples would change
/// (so a pure-zero write never authors empty tiles).
fn apply_local<F>(
    data: &mut TerrainData,
    params: &BrushParams,
    builder: &mut DeltaBuilder,
    value: F,
) where
    F: Fn(f64, f64, f64, f64) -> f64,
{
    let (min, max) = params.aabb();
    let res = data.tile_resolution();
    let mps = data.meters_per_sample();
    let c0 = data.tile_coord_of(min.x, min.y);
    let c1 = data.tile_coord_of(max.x, max.y);
    let (cx, cz, radius, falloff) = (
        params.center.x,
        params.center.y,
        params.radius,
        params.falloff,
    );

    for tz in c0.1..=c1.1 {
        for tx in c0.0..=c1.0 {
            let coord = (tx, tz);
            let o = data.tile_origin_xz(coord);
            let Some((i_lo, i_hi)) = axis_range(min.x, max.x, o.x, mps, res) else {
                continue;
            };
            let Some((j_lo, j_hi)) = axis_range(min.y, max.y, o.y, mps, res) else {
                continue;
            };

            // Gather the changed samples first (immutable read), then commit.
            let existed = data.has_tile(coord);
            let writes: Vec<(u32, u32, f32, f32)> = {
                let existing = data.get_tile(coord);
                let base_y = existing.map(|t| t.origin.y).unwrap_or(0.0);
                let mut ws = Vec::new();
                for j in j_lo..=j_hi {
                    let wz = o.y + j as f64 * mps;
                    for i in i_lo..=i_hi {
                        let wx = o.x + i as f64 * mps;
                        let d = ((wx - cx).powi(2) + (wz - cz).powi(2)).sqrt();
                        let w = falloff.weight(d, radius);
                        if w <= 0.0 {
                            continue;
                        }
                        // A P21.2 **holed** sample has no surface to raise or
                        // lower, so the brush passes straight over it — exactly
                        // as it passes over an unauthored one. Sculpting the lip
                        // of a cave mouth must not quietly re-grow its roof.
                        if existing.map(|t| t.is_hole(res, i, j)).unwrap_or(false) {
                            continue;
                        }
                        let old_off = existing.map(|t| t.sample(res, i, j)).unwrap_or(0.0);
                        let old_world = base_y + old_off as f64;
                        let new_off = (value(wx, wz, old_world, w) - base_y) as f32;
                        // **A sample the arithmetic could not compute is not
                        // written** (C4-35). `as f32` SATURATES to
                        // `f32::INFINITY` past `f32::MAX`, so an unbounded
                        // strength turns a finite tile into an infinite one; a
                        // later Smooth dab then computes `inf - inf` = NaN and a
                        // Flatten does the same. Both of them sail past the
                        // `w <= 0.0` guard above (false for NaN) and past the
                        // `new_off != old_off` test below (ALWAYS true for NaN,
                        // since NaN equals nothing) — so every sample in the
                        // footprint would be committed, and `encode_tile`
                        // bincodes it into the `.inf_terrain` without a word.
                        if !new_off.is_finite() {
                            continue;
                        }
                        if new_off != old_off {
                            ws.push((i, j, old_off, new_off));
                        }
                    }
                }
                ws
            };

            if writes.is_empty() {
                continue; // nothing changed → never author an empty tile
            }
            if !existed {
                builder.mark_created(coord);
            }
            let tile = data.get_or_create_tile(coord);
            for (i, j, before, new_off) in writes {
                builder.touch(coord, i, j, before);
                tile.set_sample(res, i, j, new_off);
            }
        }
    }
}

/// Smooth: `iterations` box-blur passes toward the neighbourhood mean, blended by
/// `weight · strength`. Runs in a [`HeightRegion`](crate::HeightRegion) so tile
/// seams are interior columns (seamless) and holes are excluded from the mean.
fn apply_smooth(
    data: &mut TerrainData,
    params: &BrushParams,
    iterations: u32,
    builder: &mut DeltaBuilder,
) {
    let (min, max) = params.aabb();
    // One-ring margin so in-radius edge samples have valid neighbours to read.
    let mut region = data.extract_region(min, max, 1);
    let (nx, nz) = region.dims();
    let strength = params.strength.clamp(0.0, 1.0);
    let radius = params.radius;
    let falloff = params.falloff;

    for _ in 0..iterations {
        let src = region.heights().to_vec();
        for r in 0..nz {
            for c in 0..nx {
                if !region.is_authored(c, r) {
                    continue;
                }
                let p = region.world_xz(c, r);
                let w = falloff.weight((p - params.center).length(), radius);
                if w <= 0.0 {
                    continue;
                }
                // 3×3 box mean over authored neighbours (holes excluded).
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let nc = c as i32 + di;
                        let nr = r as i32 + dj;
                        if nc < 0 || nr < 0 || nc >= nx as i32 || nr >= nz as i32 {
                            continue;
                        }
                        let (nc, nr) = (nc as u32, nr as u32);
                        if !region.is_authored(nc, nr) {
                            continue;
                        }
                        sum += src[(nr * nx + nc) as usize];
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let mean = sum / cnt as f32;
                let old = src[(r * nx + c) as usize];
                let blend = (w * strength) as f32;
                region.set_height(c, r, old + (mean - old) * blend);
            }
        }
    }
    region.write_back(data, builder);
}

/// Flatten: blend each in-radius authored sample toward `target` by
/// `weight · strength`.
fn apply_flatten(
    data: &mut TerrainData,
    params: &BrushParams,
    target: FlattenTarget,
    builder: &mut DeltaBuilder,
) {
    let (min, max) = params.aabb();
    let mut region = data.extract_region(min, max, 0);
    let (nx, nz) = region.dims();
    let strength = params.strength.clamp(0.0, 1.0);
    let radius = params.radius;
    let falloff = params.falloff;

    // Target in region-offset space (offset from the region anchor).
    let target_off = match target {
        FlattenTarget::PickedHeight(h) => (h - region.anchor_y()) as f32,
        FlattenTarget::Mean => {
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            for r in 0..nz {
                for c in 0..nx {
                    if !region.is_authored(c, r) {
                        continue;
                    }
                    let w =
                        falloff.weight((region.world_xz(c, r) - params.center).length(), radius);
                    if w <= 0.0 {
                        continue;
                    }
                    sum += region.height(c, r) as f64;
                    cnt += 1;
                }
            }
            if cnt == 0 {
                return; // nothing to flatten
            }
            (sum / cnt as f64) as f32
        }
    };

    for r in 0..nz {
        for c in 0..nx {
            if !region.is_authored(c, r) {
                continue;
            }
            let w = falloff.weight((region.world_xz(c, r) - params.center).length(), radius);
            if w <= 0.0 {
                continue;
            }
            let old = region.height(c, r);
            let blend = (w * strength).clamp(0.0, 1.0) as f32;
            region.set_height(c, r, old + (target_off - old) * blend);
        }
    }
    region.write_back(data, builder);
}

/// A brush **stroke**: the accumulated dabs of one mouse-down→up gesture, merged
/// into a single [`HeightDelta`]. Where dabs overlap, `before` is the value at the
/// first touch and `after` is the final value — one undo step per gesture.
#[derive(Default)]
pub struct Stroke {
    builder: DeltaBuilder,
}

impl Stroke {
    /// Begin an empty stroke.
    pub fn begin() -> Self {
        Self::default()
    }

    /// `true` before any dab has changed a sample.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }

    /// Apply one dab into the stroke, mutating `data` and accumulating the delta.
    pub fn add_dab(&mut self, data: &mut TerrainData, op: BrushOp, params: BrushParams) {
        apply_op(data, &op, &params, &mut self.builder);
    }

    /// Finish the stroke, producing the merged [`HeightDelta`]. Reads the final
    /// (`after`) heights from `data`.
    pub fn finish(self, data: &TerrainData) -> HeightDelta {
        self.builder.finalize(data)
    }
}

/// Resample a drag path into evenly spaced dab centres: the first path point, then
/// one every `spacing` world units of arc length. Callers pass `spacing =
/// spacing_fraction · radius`. A non-positive `spacing` returns the path points
/// unchanged; an empty path returns empty.
pub fn dab_positions(path: &[DVec2], spacing: f64) -> Vec<DVec2> {
    if path.is_empty() {
        return Vec::new();
    }
    if spacing <= 0.0 {
        return path.to_vec();
    }
    let mut out = vec![path[0]];
    let mut carry = 0.0; // arc length already consumed since the last dab
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = b - a;
        let len = seg.length();
        if len <= 0.0 {
            continue;
        }
        let dir = seg / len;
        let mut t = spacing - carry; // distance into this segment to the next dab
        while t <= len {
            out.push(a + dir * t);
            t += spacing;
        }
        carry = len - (t - spacing); // leftover past the last dab on this segment
    }
    out
}

// ── the 3D resampler (hoisted to Ring 0, P23.5 audit) ──────────────────────

/// Resample a **3D** drag path at even arc length — [`dab_positions`] lifted off
/// the XZ plane.
///
/// # Why this lives here and not beside its callers
///
/// It had two callers before it had a home: `inf_editor_core::voxel_tool`
/// (P21.2's carve brush) and `inf_dcc::sculpt` (P23.5's mesh brush). Both needed
/// "the same spacing rule as the terrain brush, in three dimensions", and both
/// wrote their own lift — which is how two resamplers come to disagree, and they
/// already had: one returned the whole path for a non-finite spacing and the
/// other returned its first point. Hoisting it into the crate that owns
/// [`dab_positions`] makes the spacing rule, the carry across segments and the
/// boundary conventions **one function with one set of tests**, and leaves each
/// caller owning only what a distance means to it.
///
/// The path is flattened onto its own arc-length axis, handed to
/// [`dab_positions`], and the answers are lifted back onto the polyline. The 2D
/// function decides *where* the dabs fall; this one only decides how far apart
/// two 3D points are.
pub fn dab_positions_3d(path: &[DVec3], spacing: f64) -> Vec<DVec3> {
    dab_positions_3d_capped(path, spacing, usize::MAX)
}

/// [`dab_positions_3d`], **bounded before it allocates** — the door an
/// interactive brush uses (the P21.3 audit's rule).
///
/// A `.take(n)` on the full list is a *filter*, not a bound: the whole list is
/// built first. At a small radius a drag of a metre asks for tens of thousands of
/// points, and at a very small one for enough to exhaust memory — taking the
/// unsaved session with it. So the **path** is trimmed to `max_dabs · spacing` of
/// arc length before the resampler ever sees it, and the remainder is carried by
/// the caller (a brush resumes from the last dab actually placed; a one-shot
/// stroke reports the truncation).
///
/// Returns at most `max_dabs + 1` points — the first is the path's own start.
pub fn dab_positions_3d_capped(path: &[DVec3], spacing: f64, max_dabs: usize) -> Vec<DVec3> {
    if path.is_empty() {
        return Vec::new();
    }
    if path.len() == 1 || !spacing.is_finite() || spacing <= 0.0 {
        return path.to_vec();
    }
    let total: f64 = path
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .filter(|d| d.is_finite())
        .sum();
    let allowed = {
        // `usize::MAX * spacing` must not become `inf` and then `NaN`.
        let v = max_dabs as f64 * spacing;
        if v.is_finite() {
            v
        } else {
            f64::MAX
        }
    };
    if allowed < total {
        let end = point_at_arc(path, allowed);
        let mut trimmed: Vec<DVec3> = vec![path[0]];
        let mut walked = 0.0;
        for w in path.windows(2) {
            let seg = (w[1] - w[0]).length();
            if !seg.is_finite() || seg <= 0.0 {
                continue;
            }
            if walked + seg >= allowed {
                break;
            }
            walked += seg;
            trimmed.push(w[1]);
        }
        trimmed.push(end);
        return resample_3d(&trimmed, spacing);
    }
    resample_3d(path, spacing)
}

fn resample_3d(path: &[DVec3], spacing: f64) -> Vec<DVec3> {
    // Cumulative arc length, as a degenerate 2D path along +X. `dab_positions`
    // then does all the work.
    let mut arc = Vec::with_capacity(path.len());
    let mut total = 0.0;
    arc.push(DVec2::ZERO);
    for w in path.windows(2) {
        let seg = (w[1] - w[0]).length();
        total += if seg.is_finite() { seg } else { 0.0 };
        arc.push(DVec2::new(total, 0.0));
    }
    dab_positions(&arc, spacing)
        .into_iter()
        .map(|p| point_at_arc(path, p.x))
        .collect()
}

/// The point `s` metres along the polyline `path`, clamped to its ends.
fn point_at_arc(path: &[DVec3], s: f64) -> DVec3 {
    let mut remaining = s.max(0.0);
    for w in path.windows(2) {
        let seg = w[1] - w[0];
        let len = seg.length();
        if !len.is_finite() || len <= 0.0 {
            continue;
        }
        if remaining <= len {
            return w[0] + seg * (remaining / len);
        }
        remaining -= len;
    }
    *path.last().unwrap_or(&DVec3::ZERO)
}
