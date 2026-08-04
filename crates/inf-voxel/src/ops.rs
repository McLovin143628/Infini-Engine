//! **Carve and fill**: the three brush shapes, the CSG rule they apply, and the
//! exact accounting they return.
//!
//! # The model
//!
//! A [`VoxelShape`] is an analytic signed-distance function in **world metres**.
//! Applying it to a volume converts that distance to **voxels** (the field's own
//! unit) and combines it with what is already there:
//!
//! * [`Fill`](VoxelOpKind::Fill) is CSG **union**: `new = min(old, shape)`, and
//!   every sample the fill makes (or keeps) solid takes the fill's material.
//! * [`Carve`](VoxelOpKind::Carve) is CSG **difference**: `new = max(old, −shape)`
//!   — the complement of the shape intersected with what was there. A sample the
//!   carve empties is normalized back to [`DEFAULT_MATERIAL`], so two histories
//!   that reach the same geometry reach the same **bytes** (an empty voxel has no
//!   material, and letting one linger would make an undone carve serialize
//!   differently from a never-carved volume).
//!
//! Both are `min`/`max` on a clamped band, which makes them **idempotent**:
//! applying the same op twice changes nothing the second time, and
//! [`the_same_op_twice_is_a_no_op`] pins it. That is what makes a replayed
//! gameplay carve (P21.4) safe to run again after a rollback.
//!
//! # The accounting is exact, and that is the gate
//!
//! [`OpReport`] counts **sample transitions**, not an estimate:
//! `filled[m]` is the number of samples that went empty → solid and now carry
//! material `m`; `carved[m]` is the number that went solid → empty and *had*
//! material `m`. P21.3's soil-displacement conservation test is going to subtract
//! one from the other and demand they balance, exactly as the P10 erosion mass
//! gates do — so an approximate count here is a wrong answer there.
//!
//! # Determinism
//!
//! The affected region is an integer box derived from the shape's world AABB, and
//! the pass is a gather over it in a fixed order. Nothing is scattered, nothing is
//! hashed, and the arithmetic is `f64` down to the single `as f32` store.

use glam::DVec3;

use crate::chunk::{clamp_sdf, ChunkKey, DEFAULT_MATERIAL, MATERIAL_COUNT, SDF_BAND};
use crate::data::{local_of, VoxelData};
use crate::delta::{ChunkPatch, VoxelDelta};

/// An analytic brush volume, in **world metres**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoxelShape {
    /// A ball.
    Sphere { center: DVec3, radius_m: f64 },
    /// An **axis-aligned** box. Oriented boxes are a P21.3 tool concern (an
    /// author drags a rotated foundation pit); the primitive stays axis-aligned so
    /// its distance function is exact and cheap, and a tool that needs rotation
    /// composes one from capsules or waits for the oriented variant to be added
    /// beside this one.
    Box { center: DVec3, half_extents: DVec3 },
    /// A capsule (a swept sphere) — the spline-tunnel and brush-stroke primitive:
    /// dragging a carve across the world is a chain of these, and a chain of
    /// capsules has no gaps where a chain of spheres would.
    Capsule { a: DVec3, b: DVec3, radius_m: f64 },
    /// A **swept rectangular trench** (P21.3): the box you get by dragging a
    /// `2·half_width_m × 2·half_height_m` cross-section along the segment
    /// `a → b`. The utility-trench / road-cut primitive, and the oriented box
    /// [`Box`](VoxelShape::Box)'s doc comment promised would be "added beside
    /// it" rather than composed from capsules.
    ///
    /// **Yaw is free, roll is not.** The frame is derived from the centreline:
    /// local +X is `normalize(b − a)`, local +Z is the horizontal perpendicular
    /// `normalize(u × Y)`, and local +Y closes the basis. So a trench can run in
    /// any direction and dive, and its floor stays perpendicular to the run —
    /// but it cannot be banked, which is the one degree of freedom an excavation
    /// has no use for. A **vertical** centreline (a shaft) has no horizontal
    /// perpendicular; the basis falls back to world +X, deterministically.
    ///
    /// The distance function is exact (a box SDF in the rotated frame), the
    /// basis is built from square roots and dot/cross products only — **no `std`
    /// trig anywhere**, which is what makes a committed trench bit-identical on
    /// every platform (the P14 law).
    Trench {
        /// Centreline start, world metres.
        a: DVec3,
        /// Centreline end, world metres.
        b: DVec3,
        /// Half the cross-section's width, metres (local Z).
        half_width_m: f64,
        /// Half the cross-section's height, metres (local Y).
        half_height_m: f64,
    },
}

/// The orthonormal frame of a [`Trench`](VoxelShape::Trench): `(u, v, w,
/// half_length_m)`, where `u` runs along the centreline, `w` is the horizontal
/// perpendicular and `v = w × u` closes it.
///
/// `None` for a degenerate centreline (zero length, or non-finite input) — the
/// same "write nothing rather than a plane of garbage" contract every other
/// shape's validity check has.
///
/// Public because the tools draw the trench's preview box from exactly this
/// frame, and a second copy of the basis rule is how a preview comes to show a
/// different trench from the one that gets cut.
pub fn trench_basis(a: DVec3, b: DVec3) -> Option<(DVec3, DVec3, DVec3, f64)> {
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    let ab = b - a;
    let len = ab.length();
    // `<=` plus an explicit NaN guard rather than `!(len > 0.0)`: the two say the
    // same thing and only one of them reads as a deliberate statement about a
    // partially-ordered type.
    if len.is_nan() || len <= 0.0 || len.is_infinite() {
        return None;
    }
    let u = ab / len;
    let cross = u.cross(DVec3::Y);
    // A vertical run has no horizontal perpendicular. Falling back to world +X
    // is arbitrary but *fixed*, which is the property that matters: a shaft cut
    // twice is cut the same way twice.
    let w = if cross.length_squared() > 1e-18 {
        cross.normalize()
    } else {
        DVec3::X
    };
    let v = w.cross(u);
    Some((u, v, w, len * 0.5))
}

impl VoxelShape {
    /// Signed distance from `p` to the shape's surface, in **world metres**
    /// (negative inside).
    pub fn distance_m(&self, p: DVec3) -> f64 {
        match *self {
            VoxelShape::Sphere { center, radius_m } => p.distance(center) - radius_m,
            VoxelShape::Box {
                center,
                half_extents,
            } => {
                // The exact box SDF: outside is the length of the positive part of
                // the componentwise overshoot, inside is the (negative) distance to
                // the nearest face.
                let q = (p - center).abs() - half_extents.abs();
                q.max(DVec3::ZERO).length() + q.max_element().min(0.0)
            }
            VoxelShape::Capsule { a, b, radius_m } => {
                let ab = b - a;
                let len2 = ab.length_squared();
                let t = if len2 > 0.0 {
                    ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                p.distance(a + ab * t) - radius_m
            }
            VoxelShape::Trench {
                a,
                b,
                half_width_m,
                half_height_m,
            } => {
                let Some((u, v, w, half_len)) = trench_basis(a, b) else {
                    return f64::INFINITY;
                };
                // The box SDF, evaluated in the trench's own frame. `d` is the
                // centreline midpoint offset expressed in `(u, v, w)`, so this
                // is the axis-aligned formula one rotation later.
                let d = p - (a + b) * 0.5;
                let local = DVec3::new(d.dot(u), d.dot(v), d.dot(w));
                let q = local.abs() - DVec3::new(half_len, half_height_m.abs(), half_width_m.abs());
                q.max(DVec3::ZERO).length() + q.max_element().min(0.0)
            }
        }
    }

    /// Inclusive world-space AABB of the shape, grown by `margin_m`.
    pub fn aabb_m(&self, margin_m: f64) -> (DVec3, DVec3) {
        let (lo, hi) = match *self {
            VoxelShape::Sphere { center, radius_m } => {
                let r = DVec3::splat(radius_m.abs());
                (center - r, center + r)
            }
            VoxelShape::Box {
                center,
                half_extents,
            } => {
                let h = half_extents.abs();
                (center - h, center + h)
            }
            VoxelShape::Capsule { a, b, radius_m } => {
                let r = DVec3::splat(radius_m.abs());
                (a.min(b) - r, a.max(b) + r)
            }
            VoxelShape::Trench {
                a,
                b,
                half_width_m,
                half_height_m,
            } => {
                let Some((u, v, w, half_len)) = trench_basis(a, b) else {
                    return (DVec3::ZERO, DVec3::ZERO);
                };
                // The world AABB of an oriented box: the support function of a
                // box in each axis direction, which is the sum of |axis
                // component| × half extent — exact, not a sphere bound.
                let ext = u.abs() * half_len
                    + v.abs() * half_height_m.abs()
                    + w.abs() * half_width_m.abs();
                let c = (a + b) * 0.5;
                (c - ext, c + ext)
            }
        };
        let m = DVec3::splat(margin_m.max(0.0));
        (lo - m, hi + m)
    }

    /// How many lattice samples [`VoxelData::apply_op`] would visit for this
    /// shape at `voxel_size_m` — the **cost of a cut, computable before cutting
    /// it**.
    ///
    /// The excavation gate (P21.3) needs an upper bound on a dig *before* it
    /// starts, because a dig is judged whole: refusing a foundation pit halfway
    /// through would leave exactly the half-excavated hole `a4e5844` ruled
    /// against, and the spoil pile that must balance it is bounded by the same
    /// number. This is that bound — the op's own affected region (the AABB grown
    /// by [`SDF_BAND`]), so it is never an underestimate of the work.
    ///
    /// Saturating throughout: an absurd shape answers `u64::MAX` rather than
    /// wrapping into a small number that would wave it through.
    pub fn affected_sample_count(&self, voxel_size_m: f64) -> u64 {
        if !self.is_valid() || voxel_size_m.is_nan() || voxel_size_m <= 0.0 {
            return 0;
        }
        let (lo, hi) = self.aabb_m(SDF_BAND as f64 * voxel_size_m);
        let mut total: u64 = 1;
        for a in 0..3 {
            let l = (lo.to_array()[a] / voxel_size_m).floor();
            let h = (hi.to_array()[a] / voxel_size_m).ceil();
            let span = h - l + 1.0;
            if span.is_nan() || span <= 0.0 {
                return 0;
            }
            // `u64::MAX` for anything past the integer range, so the gate above
            // sees "too large" rather than a truncated small answer.
            let n = if span >= u64::MAX as f64 {
                u64::MAX
            } else {
                span as u64
            };
            total = total.saturating_mul(n);
        }
        total
    }

    /// `true` when the shape encloses a positive volume (a degenerate brush —
    /// zero radius, zero extent, a NaN dragged in from a UI — writes nothing
    /// rather than writing a plane of garbage).
    pub fn is_valid(&self) -> bool {
        match *self {
            VoxelShape::Sphere { center, radius_m } => {
                center.is_finite() && radius_m.is_finite() && radius_m > 0.0
            }
            VoxelShape::Box {
                center,
                half_extents,
            } => center.is_finite() && half_extents.is_finite() && half_extents.min_element() > 0.0,
            VoxelShape::Capsule { a, b, radius_m } => {
                a.is_finite() && b.is_finite() && radius_m.is_finite() && radius_m > 0.0
            }
            VoxelShape::Trench {
                a,
                b,
                half_width_m,
                half_height_m,
            } => {
                trench_basis(a, b).is_some()
                    && half_width_m.is_finite()
                    && half_width_m > 0.0
                    && half_height_m.is_finite()
                    && half_height_m > 0.0
            }
        }
    }

    /// The shape's centre in world metres — what a readout quotes and what the
    /// default spoil rule measures from.
    pub fn center_m(&self) -> DVec3 {
        match *self {
            VoxelShape::Sphere { center, .. } => center,
            VoxelShape::Box { center, .. } => center,
            VoxelShape::Capsule { a, b, .. } => (a + b) * 0.5,
            VoxelShape::Trench { a, b, .. } => (a + b) * 0.5,
        }
    }
}

/// What an op does where its shape covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelOpKind {
    /// Remove material (CSG difference).
    Carve,
    /// Add material of the given splat index (CSG union).
    Fill { material: u8 },
}

/// One brush application.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelOp {
    pub shape: VoxelShape,
    pub kind: VoxelOpKind,
}

impl VoxelOp {
    /// Carve `shape` out of the volume.
    pub fn carve(shape: VoxelShape) -> Self {
        Self {
            shape,
            kind: VoxelOpKind::Carve,
        }
    }

    /// Fill `shape` with `material`.
    pub fn fill(shape: VoxelShape, material: u8) -> Self {
        Self {
            shape,
            kind: VoxelOpKind::Fill { material },
        }
    }
}

/// Exact per-material accounting for one op — see the module docs on why this is
/// counted rather than estimated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpReport {
    /// Samples whose signed distance or material actually changed.
    pub touched: u64,
    /// Samples that went **empty → solid**, by the material they now carry.
    pub filled: [u64; MATERIAL_COUNT],
    /// Samples that went **solid → empty**, by the material they carried before.
    pub carved: [u64; MATERIAL_COUNT],
    /// Chunks the op materialized from nothing.
    pub created_chunks: u64,
}

impl OpReport {
    /// Total samples added to the solid set.
    pub fn total_filled(&self) -> u64 {
        self.filled.iter().sum()
    }

    /// Total samples removed from the solid set.
    pub fn total_carved(&self) -> u64 {
        self.carved.iter().sum()
    }

    /// Net change in solid samples (positive = the world gained material).
    pub fn net_solid(&self) -> i64 {
        self.total_filled() as i64 - self.total_carved() as i64
    }

    /// `true` when the op changed nothing at all.
    pub fn is_noop(&self) -> bool {
        self.touched == 0
    }
}

impl VoxelData {
    /// Apply `op`, returning its exact accounting and a reversible
    /// [`VoxelDelta`].
    ///
    /// The delta is the undo record: `data.revert_delta(&delta)` restores the
    /// volume **byte-identically**, chunk set included. A degenerate shape is a
    /// no-op with an empty delta rather than an error — a UI can hand this a
    /// zero-radius brush mid-drag and must not have to guard every call.
    ///
    /// The one-dab form of [`apply_op_into`](Self::apply_op_into); a brush stroke
    /// wants that one.
    pub fn apply_op(&mut self, op: &VoxelOp) -> (OpReport, VoxelDelta) {
        let mut builder = VoxelDeltaBuilder::new();
        let report = self.apply_op_into(op, &mut builder);
        let delta = builder.finalize(self);
        (report, delta)
    }

    /// Apply `op`, accumulating its undo record into `builder` — **the stroke
    /// form**.
    ///
    /// One builder held across every dab of a mouse-down is what makes a stroke
    /// undo in a single step *to where it started*: `before` is the value at the
    /// **first** touch of each sample and `after` is read at
    /// [`finalize`](VoxelDeltaBuilder::finalize) time, so overlapping dabs merge
    /// instead of stacking N deltas whose intermediate states nobody wants back.
    /// (It is also what keeps a long drag from costing one dense sub-box per dab
    /// on the undo stack.)
    pub fn apply_op_into(&mut self, op: &VoxelOp, builder: &mut VoxelDeltaBuilder) -> OpReport {
        let mut report = OpReport::default();
        if !op.shape.is_valid() {
            return report;
        }

        // The affected region: the shape's AABB grown by the distance band, so the
        // gradient around the new surface is written and the mesher never reads a
        // stale slope one voxel out. In grid units, floored/ceiled to integers.
        let margin_m = SDF_BAND as f64 * self.voxel_size_m();
        let (lo_m, hi_m) = op.shape.aabb_m(margin_m);
        let lo_g = self.world_to_grid(lo_m);
        let hi_g = self.world_to_grid(hi_m);
        let Some((lo, hi)) = integer_box(lo_g, hi_g) else {
            return report;
        };

        let inv = 1.0 / self.voxel_size_m();
        // THIS op's touched chunks, kept apart from the builder's running set: a
        // stroke's builder accumulates every dab, and compacting all of them on
        // every dab would make a long drag quadratic in the chunks it has
        // already passed over.
        let mut touched: std::collections::BTreeSet<ChunkKey> = std::collections::BTreeSet::new();

        for gz in lo[2]..=hi[2] {
            for gy in lo[1]..=hi[1] {
                for gx in lo[0]..=hi[0] {
                    let world = self.grid_to_world(DVec3::new(gx as f64, gy as f64, gz as f64));
                    // The shape's distance in the field's own unit.
                    let shape_v = op.shape.distance_m(world) * inv;
                    let old_sdf = self.sample_global(gx, gy, gz);
                    let old_mat = self.material_global(gx, gy, gz);
                    let (new_sdf, new_mat) = match op.kind {
                        VoxelOpKind::Fill { material } => {
                            let v = clamp_sdf((old_sdf as f64).min(shape_v) as f32);
                            // Inside the brush, the material becomes the fill's;
                            // outside it, whatever was already there survives.
                            let m = if shape_v < 0.0 {
                                material.min(MATERIAL_COUNT as u8 - 1)
                            } else {
                                old_mat
                            };
                            (v, if v < 0.0 { m } else { DEFAULT_MATERIAL })
                        }
                        VoxelOpKind::Carve => {
                            let v = clamp_sdf((old_sdf as f64).max(-shape_v) as f32);
                            // An empty sample carries no material — normalizing it
                            // is what makes an undone carve byte-identical to a
                            // never-carved volume.
                            (v, if v < 0.0 { old_mat } else { DEFAULT_MATERIAL })
                        }
                    };
                    if new_sdf == old_sdf && new_mat == old_mat {
                        continue;
                    }

                    let key = ChunkKey::of_sample(gx, gy, gz);
                    if !self.is_resident(key) {
                        // Materializing a chunk is part of the op, and the delta
                        // has to know so a revert can remove it again.
                        builder.mark_created(key);
                        report.created_chunks += 1;
                    }
                    let (i, j, k) = local_of(gx, gy, gz);
                    builder.touch(key, i, j, k, old_sdf, old_mat);
                    touched.insert(key);

                    let chunk = self.get_or_create_chunk(key);
                    chunk.set_sample(i, j, k, new_sdf);
                    chunk.set_material(i, j, k, new_mat);

                    report.touched += 1;
                    let was_solid = old_sdf < 0.0;
                    let is_solid = new_sdf < 0.0;
                    if !was_solid && is_solid {
                        report.filled[(new_mat as usize).min(MATERIAL_COUNT - 1)] += 1;
                    } else if was_solid && !is_solid {
                        report.carved[(old_mat as usize).min(MATERIAL_COUNT - 1)] += 1;
                    }
                }
            }
        }

        // Compact every touched chunk's material buffer, so a chunk whose paint the
        // op removed returns to the sparse form (and therefore to fewer bytes).
        //
        // `raw_chunk_mut` rather than `get_chunk_mut`: the op has already stamped
        // every chunk it touched, and compaction cannot change what any reader
        // sees (the sparse and dense forms hold the same values), so a second
        // stamp would only buy a redundant re-mesh.
        for key in touched {
            if let Some(c) = self.raw_chunk_mut(key) {
                c.compact_materials();
            }
        }
        report
    }
}

/// Clamp a fractional grid box to inclusive integer sample bounds, or `None` when
/// it is empty or too far out to name real samples.
pub(crate) fn integer_box(lo: DVec3, hi: DVec3) -> Option<([i32; 3], [i32; 3])> {
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let limit = i32::MAX as f64 - 4.0;
    let mut out_lo = [0i32; 3];
    let mut out_hi = [0i32; 3];
    for a in 0..3 {
        let l = lo.to_array()[a].floor().clamp(-limit, limit);
        let h = hi.to_array()[a].ceil().clamp(-limit, limit);
        if h < l {
            return None;
        }
        out_lo[a] = l as i32;
        out_hi[a] = h as i32;
    }
    Some((out_lo, out_hi))
}

/// Accumulates the *first-touch* `(sdf, material)` per changed sample, then
/// compresses the result into per-chunk dense sub-boxes.
///
/// A brush stroke keeps one builder alive across every dab so that, where dabs
/// overlap, `before` is the value at the **first** touch and `after` (read at
/// [`finalize`](VoxelDeltaBuilder::finalize) time) is the **final** value — the
/// merge contract an undo stack relies on. The P21.2 carve brush holds one
/// across a drag through [`VoxelData::apply_op_into`]; a single
/// [`VoxelData::apply_op`] uses one for exactly one op.
///
/// The exact shape — and the exact contract — of
/// [`HoleDeltaBuilder`](inf_terrain::HoleDeltaBuilder), one layer over, which is
/// what lets a carve's two halves merge the same way and undo as one step.
#[derive(Default)]
pub struct VoxelDeltaBuilder {
    /// `key → (i, j, k) → first-touch (sdf, material)`.
    touched: TouchedSamples,
    /// Chunks materialized from nothing during the stroke.
    created: std::collections::BTreeSet<ChunkKey>,
}

/// First-touch `(sdf, material)` per changed sample, grouped by chunk. Named so
/// the nesting reads once here rather than at every use site.
type TouchedSamples =
    std::collections::BTreeMap<ChunkKey, std::collections::BTreeMap<(u32, u32, u32), (f32, u8)>>;

impl VoxelDeltaBuilder {
    /// A builder with nothing touched.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when no sample has been touched yet — an empty stroke, which
    /// records no undo entry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.touched.is_empty() && self.created.is_empty()
    }

    #[inline]
    pub(crate) fn touch(&mut self, key: ChunkKey, i: u32, j: u32, k: u32, sdf: f32, mat: u8) {
        self.touched
            .entry(key)
            .or_default()
            .entry((i, j, k))
            .or_insert((sdf, mat));
    }

    #[inline]
    pub(crate) fn mark_created(&mut self, key: ChunkKey) {
        self.created.insert(key);
    }

    /// Compress the accumulated touches into per-chunk dense sub-boxes, reading
    /// final (`after`) and filler values from `data`'s current chunks.
    pub fn finalize(&self, data: &VoxelData) -> VoxelDelta {
        let mut patches = Vec::with_capacity(self.touched.len());
        for (&key, samples) in &self.touched {
            let Some(chunk) = data.get_chunk(key) else {
                continue;
            };
            let mut lo = [u32::MAX; 3];
            let mut hi = [0u32; 3];
            for &(i, j, k) in samples.keys() {
                for (a, v) in [i, j, k].into_iter().enumerate() {
                    lo[a] = lo[a].min(v);
                    hi[a] = hi[a].max(v);
                }
            }
            let (w, h, d) = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1);
            let n = (w * h * d) as usize;
            let mut sdf_before = vec![0.0f32; n];
            let mut sdf_after = vec![0.0f32; n];
            let mut mat_before = vec![0u8; n];
            let mut mat_after = vec![0u8; n];
            for dk in 0..d {
                for dj in 0..h {
                    for di in 0..w {
                        let idx = ((dk * h + dj) * w + di) as usize;
                        let (i, j, k) = (lo[0] + di, lo[1] + dj, lo[2] + dk);
                        // `after` is the chunk's current value; `before` is the
                        // recorded first touch, or the current value for filler
                        // cells inside the box that never actually changed.
                        let cur = (chunk.sample(i, j, k), chunk.material(i, j, k));
                        sdf_after[idx] = cur.0;
                        mat_after[idx] = cur.1;
                        let b = samples.get(&(i, j, k)).copied().unwrap_or(cur);
                        sdf_before[idx] = b.0;
                        mat_before[idx] = b.1;
                    }
                }
            }
            patches.push(ChunkPatch {
                key,
                i0: lo[0],
                j0: lo[1],
                k0: lo[2],
                w,
                h,
                d,
                sdf_before,
                sdf_after,
                mat_before,
                mat_after,
            });
        }
        VoxelDelta {
            patches,
            created_chunks: self.created.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{VoxelChunk, CHUNK_DIM, EMPTY_SDF};
    use crate::residency::chunk_range;

    /// The whole volume's chunk bytes — the object every "byte-identical"
    /// assertion below actually compares.
    fn image(v: &VoxelData) -> Vec<(ChunkKey, Vec<u8>)> {
        v.chunks()
            .map(|(&k, c)| (k, crate::asset::encode_chunk(c).unwrap()))
            .collect()
    }

    fn solid_block() -> VoxelData {
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty();
        v
    }

    /// The distance functions are the analytic ones, not approximations.
    #[test]
    fn the_shape_distance_functions_are_exact() {
        let s = VoxelShape::Sphere {
            center: DVec3::ZERO,
            radius_m: 2.0,
        };
        assert_eq!(s.distance_m(DVec3::ZERO), -2.0);
        assert_eq!(s.distance_m(DVec3::new(2.0, 0.0, 0.0)), 0.0);
        assert_eq!(s.distance_m(DVec3::new(5.0, 0.0, 0.0)), 3.0);

        let b = VoxelShape::Box {
            center: DVec3::ZERO,
            half_extents: DVec3::new(1.0, 2.0, 3.0),
        };
        assert_eq!(b.distance_m(DVec3::ZERO), -1.0, "nearest face is x");
        assert_eq!(b.distance_m(DVec3::new(1.0, 0.0, 0.0)), 0.0);
        assert!((b.distance_m(DVec3::new(4.0, 6.0, 0.0)) - 5.0).abs() < 1e-12);

        let c = VoxelShape::Capsule {
            a: DVec3::new(-3.0, 0.0, 0.0),
            b: DVec3::new(3.0, 0.0, 0.0),
            radius_m: 1.0,
        };
        assert_eq!(c.distance_m(DVec3::ZERO), -1.0);
        assert_eq!(c.distance_m(DVec3::new(0.0, 1.0, 0.0)), 0.0);
        assert_eq!(c.distance_m(DVec3::new(5.0, 0.0, 0.0)), 1.0, "the cap");
        // A degenerate capsule is a sphere, never a divide by zero.
        let dot = VoxelShape::Capsule {
            a: DVec3::ZERO,
            b: DVec3::ZERO,
            radius_m: 1.0,
        };
        assert_eq!(dot.distance_m(DVec3::new(3.0, 0.0, 0.0)), 2.0);
    }

    /// The **trench** primitive (P21.3): a box in a frame the centreline
    /// defines, checked against an independently rotated evaluation of the
    /// axis-aligned box SDF.
    #[test]
    fn the_trench_is_an_exact_box_in_its_own_frame() {
        // A trench running due +X: its frame is the world's, so it must agree
        // with the axis-aligned box exactly.
        let along_x = VoxelShape::Trench {
            a: DVec3::new(-3.0, 0.0, 0.0),
            b: DVec3::new(3.0, 0.0, 0.0),
            half_width_m: 1.0,
            half_height_m: 2.0,
        };
        let equiv = VoxelShape::Box {
            center: DVec3::ZERO,
            half_extents: DVec3::new(3.0, 2.0, 1.0),
        };
        for p in [
            DVec3::ZERO,
            DVec3::new(1.0, 1.0, 0.5),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 4.0),
            DVec3::new(-7.0, 9.0, -3.0),
        ] {
            assert!(
                (along_x.distance_m(p) - equiv.distance_m(p)).abs() < 1e-12,
                "{p:?}: {} vs {}",
                along_x.distance_m(p),
                equiv.distance_m(p)
            );
        }

        // Rotated 90° about Y (running due +Z): the same distances, with the
        // query point rotated the same way.
        let along_z = VoxelShape::Trench {
            a: DVec3::new(0.0, 0.0, -3.0),
            b: DVec3::new(0.0, 0.0, 3.0),
            half_width_m: 1.0,
            half_height_m: 2.0,
        };
        for p in [
            DVec3::new(1.0, 1.0, 0.5),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::new(-7.0, 9.0, -3.0),
        ] {
            // (x, y, z) → (−z, y, x) is the same rotation the centreline took.
            let rotated = DVec3::new(-p.z, p.y, p.x);
            assert!(
                (along_z.distance_m(rotated) - along_x.distance_m(p)).abs() < 1e-12,
                "{p:?}"
            );
        }

        // The AABB really contains the solid, on a diagonal run where a naive
        // "endpoints ± half extents" bound would clip the corners off.
        let diag = VoxelShape::Trench {
            a: DVec3::new(-4.0, 1.0, -4.0),
            b: DVec3::new(4.0, 1.0, 4.0),
            half_width_m: 1.5,
            half_height_m: 0.75,
        };
        let (lo, hi) = diag.aabb_m(0.0);
        let mut inside = 0;
        for gz in -80..=80 {
            for gy in -20..=20 {
                for gx in -80..=80 {
                    let p = DVec3::new(gx as f64 * 0.1, gy as f64 * 0.1, gz as f64 * 0.1);
                    if diag.distance_m(p) >= 0.0 {
                        continue;
                    }
                    inside += 1;
                    assert!(
                        p.cmpge(lo).all() && p.cmple(hi).all(),
                        "{p:?} is inside the trench but outside its AABB {lo:?}..{hi:?}"
                    );
                }
            }
        }
        assert!(inside > 500, "the fixture trench is empty ({inside})");

        // A **vertical** run (a shaft) has no horizontal perpendicular and falls
        // back to a fixed basis rather than a NaN.
        let shaft = VoxelShape::Trench {
            a: DVec3::new(0.0, 0.0, 0.0),
            b: DVec3::new(0.0, -6.0, 0.0),
            half_width_m: 1.0,
            half_height_m: 1.0,
        };
        assert!(shaft.is_valid());
        assert!(shaft.distance_m(DVec3::new(0.0, -3.0, 0.0)) < 0.0);
        assert!(shaft.distance_m(DVec3::new(0.0, -3.0, 5.0)) > 0.0);
        assert!(shaft.distance_m(DVec3::new(0.0, -3.0, 0.0)).is_finite());
    }

    /// A degenerate trench writes nothing — a zero-length centreline, a zero
    /// section, a NaN dragged in from a drag that has not moved yet.
    #[test]
    fn a_degenerate_trench_is_a_clean_no_op() {
        let mut v = solid_block();
        let before = image(&v);
        for shape in [
            VoxelShape::Trench {
                a: DVec3::ZERO,
                b: DVec3::ZERO,
                half_width_m: 1.0,
                half_height_m: 1.0,
            },
            VoxelShape::Trench {
                a: DVec3::ZERO,
                b: DVec3::new(4.0, 0.0, 0.0),
                half_width_m: 0.0,
                half_height_m: 1.0,
            },
            VoxelShape::Trench {
                a: DVec3::splat(f64::NAN),
                b: DVec3::new(4.0, 0.0, 0.0),
                half_width_m: 1.0,
                half_height_m: 1.0,
            },
        ] {
            assert!(!shape.is_valid(), "{shape:?}");
            let (r, d) = v.apply_op(&VoxelOp::carve(shape));
            assert!(r.is_noop(), "{shape:?} → {r:?}");
            assert!(d.is_empty());
            assert_eq!(shape.affected_sample_count(0.5), 0);
        }
        assert_eq!(image(&v), before);
    }

    /// **The exact-accounting gate for the trench.** Carving a swept rectangle
    /// out of rock empties precisely the samples the SDF contains, counted by an
    /// independent walk.
    #[test]
    fn a_trench_carve_reports_exactly_the_samples_it_emptied() {
        let mut v = solid_block();
        let shape = VoxelShape::Trench {
            a: v.grid_to_world(DVec3::new(1.0, 8.0, 3.0)),
            b: v.grid_to_world(DVec3::new(26.0, 10.0, 22.0)),
            half_width_m: 1.25,
            half_height_m: 0.75,
        };
        let mut want = 0u64;
        for gz in 0..(2 * CHUNK_DIM as i32) {
            for gy in 0..(2 * CHUNK_DIM as i32) {
                for gx in 0..(2 * CHUNK_DIM as i32) {
                    let p = v.grid_to_world(DVec3::new(gx as f64, gy as f64, gz as f64));
                    if shape.distance_m(p) <= 0.0 && v.sample_global(gx, gy, gz) < 0.0 {
                        want += 1;
                    }
                }
            }
        }
        assert!(want > 100, "the fixture trench carved {want} samples");
        let (report, _) = v.apply_op(&VoxelOp::carve(shape));
        assert_eq!(report.carved[1], want);
        assert_eq!(report.total_carved(), want);
        assert_eq!(report.total_filled(), 0);
    }

    /// [`VoxelShape::affected_sample_count`] really bounds the work: it is never
    /// less than what the op actually touched, and it is a pure function of the
    /// shape (so a dig can be judged before it cuts).
    #[test]
    fn the_affected_sample_bound_is_never_an_underestimate() {
        for shape in [
            VoxelShape::Sphere {
                center: DVec3::new(4.0, 4.0, 4.0),
                radius_m: 3.0,
            },
            VoxelShape::Box {
                center: DVec3::new(4.0, 4.0, 4.0),
                half_extents: DVec3::new(2.0, 1.0, 3.0),
            },
            VoxelShape::Capsule {
                a: DVec3::new(1.0, 4.0, 4.0),
                b: DVec3::new(9.0, 6.0, 5.0),
                radius_m: 1.5,
            },
            VoxelShape::Trench {
                a: DVec3::new(1.0, 4.0, 4.0),
                b: DVec3::new(11.0, 4.0, 9.0),
                half_width_m: 1.0,
                half_height_m: 1.0,
            },
        ] {
            let mut v = solid_block();
            let bound = shape.affected_sample_count(v.voxel_size_m());
            let (report, _) = v.apply_op(&VoxelOp::carve(shape));
            assert!(report.touched > 0, "{shape:?}");
            assert!(
                report.touched <= bound,
                "{shape:?}: touched {} > bound {bound}",
                report.touched
            );
            assert_eq!(
                bound,
                shape.affected_sample_count(v.voxel_size_m()),
                "the bound is not a pure function of the shape"
            );
        }
    }

    /// **The exact-accounting gate**, sphere. Carving a ball out of solid rock
    /// must empty *precisely* the samples whose centres the ball contains, tallied
    /// against the material they had — counted here by an independent brute-force
    /// walk, so the report cannot be "correct" by construction.
    #[test]
    fn a_carve_reports_exactly_the_samples_it_emptied() {
        let mut v = solid_block();
        let center = v.grid_to_world(DVec3::splat(8.0));
        let radius_m = 3.0;
        let shape = VoxelShape::Sphere { center, radius_m };

        // Independent count: every global sample in the volume that the ball
        // reaches — `<= 0`, not `< 0`, because a carve writes `max(old, -shape)`
        // and the sign convention is that `0` is **not** solid, so a sample landing
        // exactly on the brush surface is emptied. (On a 6-voxel radius that is 30
        // lattice points, which is exactly the discrepancy a `<` here would show.)
        let mut want = 0u64;
        for gz in 0..(2 * CHUNK_DIM as i32) {
            for gy in 0..(2 * CHUNK_DIM as i32) {
                for gx in 0..(2 * CHUNK_DIM as i32) {
                    let p = v.grid_to_world(DVec3::new(gx as f64, gy as f64, gz as f64));
                    if shape.distance_m(p) <= 0.0 && v.sample_global(gx, gy, gz) < 0.0 {
                        want += 1;
                    }
                }
            }
        }
        assert!(want > 100, "the fixture must actually carve something");

        let (report, _) = v.apply_op(&VoxelOp::carve(shape));
        assert_eq!(
            report.carved[1], want,
            "material-1 carve count is not exact"
        );
        assert_eq!(report.carved[0], 0);
        assert_eq!(report.total_carved(), want);
        assert_eq!(report.total_filled(), 0);
        assert_eq!(report.net_solid(), -(want as i64));
        assert!(report.touched >= want, "the band widens the touched set");
        assert!(!report.is_noop());
        assert_eq!(report.created_chunks, 0, "the block was already resident");
    }

    /// The fill half, on all three shapes, tallied per material — and filling into
    /// empty space **materializes** chunks, which the report says.
    #[test]
    fn fills_report_exactly_what_they_made_solid_per_material() {
        for (label, shape) in [
            (
                "sphere",
                VoxelShape::Sphere {
                    center: DVec3::new(4.0, 4.0, 4.0),
                    radius_m: 2.0,
                },
            ),
            (
                "box",
                VoxelShape::Box {
                    center: DVec3::new(4.0, 4.0, 4.0),
                    half_extents: DVec3::new(1.5, 1.0, 2.0),
                },
            ),
            (
                "capsule",
                VoxelShape::Capsule {
                    a: DVec3::new(2.0, 4.0, 4.0),
                    b: DVec3::new(6.0, 4.0, 4.0),
                    radius_m: 1.0,
                },
            ),
        ] {
            let mut v = VoxelData::new(0.5);
            let (report, _) = v.apply_op(&VoxelOp::fill(shape, 2));
            let mut want = 0u64;
            let r = 40;
            for gz in -r..r {
                for gy in -r..r {
                    for gx in -r..r {
                        let p = v.grid_to_world(DVec3::new(gx as f64, gy as f64, gz as f64));
                        if shape.distance_m(p) < 0.0 {
                            want += 1;
                        }
                    }
                }
            }
            assert!(want > 20, "{label}: the fixture must fill something");
            assert_eq!(report.filled[2], want, "{label} filled count");
            assert_eq!(report.filled[0], 0, "{label}");
            assert_eq!(report.total_carved(), 0, "{label}");
            assert_eq!(report.net_solid(), want as i64, "{label}");
            assert!(report.created_chunks > 0, "{label} must materialize chunks");
            assert_eq!(report.created_chunks as usize, v.chunk_count(), "{label}");
            // The material really landed.
            let inside = shape.center_m();
            let g = v.world_to_grid(inside);
            assert!(v.is_solid_at(inside), "{label}");
            assert_eq!(
                v.material_global(g.x.round() as i32, g.y.round() as i32, g.z.round() as i32),
                2,
                "{label} material"
            );
        }
    }

    /// **The undo gate**: an op's delta reverts the volume **byte-identically**,
    /// including the chunks the op materialized and the sparse form of every
    /// material buffer. Run for a carve out of rock and a fill into air, because
    /// the two exercise opposite halves of the normalization.
    #[test]
    fn every_op_reverts_byte_identically_and_redoes_byte_identically() {
        for (label, mut v, op) in [
            (
                "carve",
                solid_block(),
                VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(4.0, 4.0, 4.0),
                    radius_m: 2.5,
                }),
            ),
            (
                "fill",
                VoxelData::new(0.5),
                VoxelOp::fill(
                    VoxelShape::Capsule {
                        a: DVec3::new(0.0, 2.0, 2.0),
                        b: DVec3::new(9.0, 2.0, 2.0),
                        radius_m: 1.25,
                    },
                    3,
                ),
            ),
            (
                "paint-over",
                solid_block(),
                VoxelOp::fill(
                    VoxelShape::Box {
                        center: DVec3::new(4.0, 4.0, 4.0),
                        half_extents: DVec3::splat(2.0),
                    },
                    2,
                ),
            ),
        ] {
            let before = image(&v);
            let (report, delta) = v.apply_op(&op);
            assert!(!report.is_noop(), "{label}: the fixture changed nothing");
            assert!(!delta.is_empty(), "{label}");
            let after = image(&v);
            assert_ne!(before, after, "{label}");

            assert_eq!(
                v.revert_delta(&delta),
                0,
                "{label}: an op's own delta is always well-formed"
            );
            assert_eq!(
                image(&v),
                before,
                "{label}: an undo must restore the exact bytes, chunk set included"
            );

            assert_eq!(v.apply_delta(&delta), 0);
            assert_eq!(
                image(&v),
                after,
                "{label}: redo must restore the exact bytes"
            );

            // …and a second undo still lands on the original.
            assert_eq!(v.revert_delta(&delta), 0);
            assert_eq!(image(&v), before, "{label}: undo is not idempotent");
        }
    }

    /// The ops are `min`/`max` on a clamped band, so **applying one twice changes
    /// nothing the second time** — which is what lets a rolled-back replay
    /// re-apply a gameplay carve (P21.4) without diverging.
    #[test]
    fn the_same_op_twice_is_a_no_op() {
        for (label, mut v, op) in [
            (
                "carve",
                solid_block(),
                VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(4.0, 4.0, 4.0),
                    radius_m: 2.5,
                }),
            ),
            (
                "fill",
                VoxelData::new(0.5),
                VoxelOp::fill(
                    VoxelShape::Box {
                        center: DVec3::new(3.0, 3.0, 3.0),
                        half_extents: DVec3::splat(1.5),
                    },
                    1,
                ),
            ),
        ] {
            let (first, _) = v.apply_op(&op);
            assert!(!first.is_noop(), "{label}");
            let settled = image(&v);
            let (second, delta) = v.apply_op(&op);
            assert!(second.is_noop(), "{label}: {second:?}");
            assert!(delta.is_empty(), "{label}");
            assert_eq!(image(&v), settled, "{label}");
        }
    }

    /// **The stroke merge contract.** One builder held across a chain of
    /// overlapping dabs undoes to where the *stroke* started, not to where the
    /// last dab started — and it does so in one step, whatever the dabs did to
    /// each other on the way.
    ///
    /// The overlap is the point: with disjoint dabs any per-dab record would
    /// pass. Here every dab re-carves ground the previous one already emptied,
    /// so a builder that kept the *latest* `before` would restore a volume with
    /// the first dab's hole still in it.
    #[test]
    fn a_stroke_of_overlapping_dabs_undoes_to_where_it_started() {
        let mut v = solid_block();
        let before = image(&v);
        let mut builder = VoxelDeltaBuilder::new();
        assert!(builder.is_empty());

        // Six dabs walking half a radius at a time along +X — each overlaps its
        // predecessor by more than half its volume.
        for step in 0..6 {
            let center = v.grid_to_world(DVec3::new(4.0 + step as f64 * 1.5, 8.0, 8.0));
            let report = v.apply_op_into(
                &VoxelOp::carve(VoxelShape::Sphere {
                    center,
                    radius_m: 1.5,
                }),
                &mut builder,
            );
            assert!(!report.is_noop(), "dab {step} carved nothing");
        }
        assert!(!builder.is_empty());
        let after = image(&v);
        assert_ne!(after, before);

        let delta = builder.finalize(&v);
        assert_eq!(v.revert_delta(&delta), 0);
        assert_eq!(
            image(&v),
            before,
            "the stroke undid to the last dab's state rather than to the stroke's"
        );
        assert_eq!(v.apply_delta(&delta), 0);
        assert_eq!(image(&v), after, "redo must restore the whole stroke");
    }

    /// Two runs of the same op on the same volume produce identical bytes AND
    /// identical reports — the determinism law, applied to the ops.
    #[test]
    fn ops_are_deterministic_across_runs() {
        let op = VoxelOp::carve(VoxelShape::Capsule {
            a: DVec3::new(1.0, 4.0, 4.0),
            b: DVec3::new(11.0, 5.0, 6.0),
            radius_m: 1.75,
        });
        let mut a = solid_block();
        let mut b = solid_block();
        let (ra, da) = a.apply_op(&op);
        let (rb, db) = b.apply_op(&op);
        assert_eq!(ra, rb);
        assert_eq!(da, db);
        assert_eq!(image(&a), image(&b));
        assert!(ra.total_carved() > 100);
    }

    /// A degenerate brush writes nothing rather than a plane of garbage — a UI
    /// hands this a zero-radius shape on the first frame of every drag.
    #[test]
    fn a_degenerate_shape_is_a_clean_no_op() {
        let mut v = solid_block();
        let before = image(&v);
        for shape in [
            VoxelShape::Sphere {
                center: DVec3::ZERO,
                radius_m: 0.0,
            },
            VoxelShape::Sphere {
                center: DVec3::ZERO,
                radius_m: f64::NAN,
            },
            VoxelShape::Sphere {
                center: DVec3::splat(f64::INFINITY),
                radius_m: 1.0,
            },
            VoxelShape::Box {
                center: DVec3::ZERO,
                half_extents: DVec3::new(1.0, 0.0, 1.0),
            },
            VoxelShape::Capsule {
                a: DVec3::ZERO,
                b: DVec3::ZERO,
                radius_m: -1.0,
            },
        ] {
            assert!(!shape.is_valid(), "{shape:?}");
            let (r, d) = v.apply_op(&VoxelOp::carve(shape));
            assert!(r.is_noop(), "{shape:?} → {r:?}");
            assert!(d.is_empty());
        }
        assert_eq!(image(&v), before);
    }

    /// An empty sample carries **no** material: carving through painted rock and
    /// undoing it must not leave a ghost material behind, or the same geometry
    /// would serialize two different ways depending on its history.
    #[test]
    fn carving_normalizes_the_material_of_emptied_samples() {
        let mut v = VoxelData::new(0.5);
        v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(3));
        v.clear_dirty();
        let virgin = crate::asset::encode_chunk(&VoxelChunk::empty()).unwrap();

        // Carve the entire chunk away (a ball that swallows it whole).
        let (r, _) = v.apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: v.grid_to_world(DVec3::splat(7.5)),
            radius_m: 40.0,
        }));
        assert!(r.carved[3] > 0);
        let c = v.get_chunk(ChunkKey::new(0, 0, 0)).unwrap();
        assert!(c.is_empty_space());
        assert!(
            c.materials_are_default(),
            "a fully carved chunk must be indistinguishable from a never-carved one"
        );
        assert_eq!(crate::asset::encode_chunk(c).unwrap(), virgin);
        assert_eq!(c.sample(0, 0, 0), EMPTY_SDF);
    }

    /// The affected region is derived from the shape's AABB plus the band — a
    /// brush entirely outside every resident chunk still materializes exactly the
    /// chunks it needs and no more.
    #[test]
    fn a_fill_materializes_only_the_chunks_it_reaches() {
        let mut v = VoxelData::new(1.0);
        let far = DVec3::new(1000.0, -500.0, 250.0);
        let (r, delta) = v.apply_op(&VoxelOp::fill(
            VoxelShape::Sphere {
                center: far,
                radius_m: 2.0,
            },
            1,
        ));
        assert!(r.total_filled() > 0);
        assert_eq!(delta.created_chunks.len(), v.chunk_count());
        assert!(v.chunk_count() <= 8, "{} chunks", v.chunk_count());
        assert!(v.is_solid_at(far));
        // Every materialized chunk is within one chunk of the brush.
        let hit = ChunkKey::of_sample(1000, -500, 250);
        for k in v.resident_keys() {
            assert!(
                (k.x - hit.x).abs() <= 1 && (k.y - hit.y).abs() <= 1 && (k.z - hit.z).abs() <= 1,
                "{k:?} is nowhere near {hit:?}"
            );
        }
        // …and the whole thing undoes to nothing at all.
        assert_eq!(v.revert_delta(&delta), 0);
        assert!(v.is_empty());
    }
}
