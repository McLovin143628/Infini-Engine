//! **TSDF fusion, and the mesher this crate did not write.**
//!
//! Depth maps are per-view opinions about where a surface is. Fusion is the step
//! that turns a stack of opinions into one surface: every view's depth map votes
//! on every voxel near it, the votes are averaged by confidence, and the
//! zero-crossing of the resulting signed distance field *is* the reconstruction.
//!
//! # The reuse ruling, and the outcome
//!
//! P25.2 was asked whether TSDF fusion could fuse into [`inf_voxel`]'s SDF
//! representation and extract through its **existing** deterministic
//! Surface-Nets mesher, rather than growing a second marching-cubes in the
//! workspace. It can, and it does. Measured against the API:
//!
//! * [`inf_voxel::mesh_chunk`] is a pure function of a bare
//!   [`inf_voxel::VoxelData`] and one [`inf_voxel::ChunkKey`] — it needs no
//!   store, no residency, no streaming and no asset. `inf-physics` already calls
//!   it that way to build runtime colliders, so this is a second caller of an
//!   existing door rather than a new arrangement.
//! * Its seams are **bit-identical from both sides** — a property `inf-voxel`
//!   gates with `assert_eq!` on `f64` positions, not an epsilon — so
//!   [`TsdfGrid::extract`] can weld the per-chunk meshes into one buffer by
//!   **exact** global-cell identity and get a watertight result, with no
//!   epsilon-radius vertex merge anywhere.
//! * Its samples are `f32` signed distances in **voxels**, negative inside, hard
//!   clamped to `±SDF_BAND` (4 voxels) — which is a truncated signed distance
//!   field. The representation this stage produces is the representation that
//!   crate already stores.
//!
//! **The one gap**, stated: `VoxelChunk` has no per-voxel *weight* channel, and
//! a TSDF's whole method is a weighted running average. Adding one would be an
//! `.inf_voxel` schema change for the benefit of a stage that does not persist.
//! So [`TsdfGrid`] keeps its own dense `(distance, weight)` accumulator for the
//! duration of a fusion and writes only the **resolved** distance into
//! `inf-voxel` at the end. That is the whole adapter.
//!
//! # Determinism, and what it costs
//!
//! Fusion is **serial across views and parallel within one**. Both halves are
//! deliberate:
//!
//! * *Serial across views* because the accumulation `d += w·tsdf; W += w` is a
//!   floating-point sum, and a sum's value depends on the order it is taken in.
//!   Six views is six passes; the alternative is a reduction whose answer
//!   depends on how many workers ran.
//! * *Parallel within a view* by **slab**, and the parallelism is free of
//!   ordering concerns for a stronger reason than care: within one view each
//!   voxel is touched **exactly once**, so the per-voxel contributions are
//!   disjoint and the fold that applies them is a copy, not a sum. There is no
//!   order for it to depend on.
//!
//! And it is a **gather**: a voxel reads the depth map. The depth map never
//! scatters into voxels. That is what removes `atomicAdd` from the GPU mirror,
//! and `atomicAdd` over floats is precisely the order-dependent construct the
//! house rules exist to keep out of anything that becomes content.
//!
//! # The voxel size, and where it comes from
//!
//! A fusion grid finer than the photographs is wasted memory; one coarser throws
//! away resolution that was paid for. The size is therefore derived from the
//! **pixel footprint at the scene's own depth**:
//!
//! ```text
//! footprint = mean_depth / mean_focal_px          // units per pixel, at depth
//! wanted    = voxel_scale * footprint
//! floor     = longest_axis_extent / max_grid_side // the memory cap
//! voxel     = max(wanted, floor)
//! ```
//!
//! `mean_depth` is the mean over reference views of that view's **median** valid
//! depth (a median, so one runaway depth cannot move it), and `mean_focal_px` is
//! the mean of the cameras' own `mean_focal()`. `voxel_scale` defaults to `2`:
//! one voxel per two pixels of footprint, because a plane sweep's depth
//! resolution is coarser than its lateral resolution and a grid at exactly one
//! footprint would resolve noise. When the memory cap binds instead of the
//! formula, [`crate::DenseAdvisory::VoxelSizeClamped`] says so — a fusion that
//! is quietly coarser than its inputs is the kind of thing nobody notices for a
//! phase.
//!
//! The truncation band is [`TsdfConfig::truncation_voxels`] voxels, defaulting
//! to `3` and capped below `inf-voxel`'s own `SDF_BAND` of `4` so the band this
//! module reasons about is never silently re-clamped by the store it writes into.
//!
//! # What the band leaves behind, said out loud
//!
//! A voxel more than one truncation *behind* a surface is not written at all,
//! because nothing observed it: a single-sided capture knows where a wall is and
//! knows nothing about the far side of it. Unwritten voxels read as `EMPTY_SDF`,
//! which is "outside" — so the extracted surface is a **slab** one truncation
//! thick, whose front face is the reconstruction and whose back face is the
//! band's own far edge. That is not a defect to hide behind a metric: it is what
//! a TSDF of photographs taken from one side geometrically is, and
//! `tests/mvs_gate.rs` measures the front face against truth *and* asserts the
//! slab's thickness, so the day a capture surrounds its subject the arm changes
//! rather than the claim.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_core::job::JobPool;
use inf_photo::sfm::RegisteredCamera;
use inf_voxel::{mesh_chunk, mesh_keys_for, VoxelData, SDF_BAND};

use crate::sweep::{backproject_world, DepthMap};

/// The incidence factor a pixel gets when its neighbours are missing and no
/// surface normal could be estimated.
///
/// Half, not one: a pixel on the edge of a valid region is exactly the pixel
/// most likely to be straddling a depth discontinuity, so it votes, and it votes
/// quietly.
pub const FALLBACK_INCIDENCE: f32 = 0.5;

/// How fusion is tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TsdfConfig {
    /// Voxels per pixel footprint at the mean scene depth. See the module docs
    /// for the formula this sits in.
    pub voxel_scale: f64,
    /// The memory cap: no axis of the grid exceeds this many voxels.
    pub max_grid_side: u32,
    /// The truncation band, in **voxels**. Clamped below `inf-voxel`'s
    /// `SDF_BAND` so this module's band is the one that applies.
    pub truncation_voxels: f64,
    /// A voxel whose accumulated weight is below this has not been seen well
    /// enough to carry a surface.
    pub min_weight: f32,
    /// How far outside the observed depth samples the grid extends, in
    /// truncations. Two, so both faces of the slab fit inside the grid and the
    /// surface never gets clipped by the box rather than by the field.
    pub bounds_pad_truncations: f64,
    /// The per-axis percentile the grid's bounds are trimmed to. A handful of
    /// depths at ten times the scene's extent would otherwise size the whole
    /// grid, and the memory cap would then coarsen everything else to pay for
    /// them.
    pub bounds_percentile: f64,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_scale: 2.0,
            max_grid_side: 256,
            truncation_voxels: 3.0,
            min_weight: 0.10,
            bounds_pad_truncations: 2.0,
            bounds_percentile: 0.002,
        }
    }
}

/// Per-pixel surface normals and the fusion weights derived from them.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceHints {
    /// Unit surface normals in **camera space**, pointing back toward the
    /// camera. `[0, 0, 0]` where there is no depth.
    pub normals: Vec<[f32; 3]>,
    /// `confidence * incidence`, the weight a pixel's vote carries. Zero where
    /// there is no depth.
    pub weights: Vec<f32>,
}

/// Estimate a normal per pixel from the depth map's own local geometry, and turn
/// it into a fusion weight.
///
/// The normal is the cross product of the two forward differences of the
/// camera-space surface point, which is the cheapest estimator that is still a
/// real normal rather than a smoothed one. It is only taken when all three
/// pixels carry depths **within `link_tolerance` of each other**: across a
/// depth discontinuity the two differences span the step rather than the
/// surface, and the "normal" that comes out points along the ray, which would
/// then weight an occlusion boundary as though it were seen face-on.
///
/// The weight is `confidence * max(0, cos(incidence))` — a surface seen at a
/// grazing angle is measured badly by every stereo method there is, and this is
/// how fusion is told so. Where no normal could be taken the incidence factor is
/// [`FALLBACK_INCIDENCE`].
pub fn surface_hints(
    camera: &RegisteredCamera,
    map: &DepthMap,
    link_tolerance: f32,
) -> SurfaceHints {
    let w = map.width as usize;
    let h = map.height as usize;
    let n_px = w * h;
    let mut normals = vec![[0.0f32; 3]; n_px];
    let mut weights = vec![0.0f32; n_px];
    // Camera-space surface points, in `f32` — the same space the fusion kernel
    // works in, so the normal and the sample it weights are consistent.
    let point = |x: usize, y: usize| -> Option<DVec3> {
        let i = y * w + x;
        let d = map.depth[i];
        if !(d > 0.0) {
            return None;
        }
        let n = camera
            .intrinsics
            .to_normalized(glam::DVec2::new(x as f64, y as f64));
        Some(DVec3::new(n.x, n.y, 1.0) * d as f64)
    };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let Some(p) = point(x, y) else { continue };
            let mut incidence = FALLBACK_INCIDENCE;
            let mut normal = [0.0f32; 3];
            if x + 1 < w && y + 1 < h {
                let (a, b) = (point(x + 1, y), point(x, y + 1));
                if let (Some(a), Some(b)) = (a, b) {
                    let d0 = map.depth[i];
                    let da = map.depth[i + 1];
                    let db = map.depth[i + w];
                    let step_ok = (da - d0).abs() <= link_tolerance * d0.max(da)
                        && (db - d0).abs() <= link_tolerance * d0.max(db);
                    if step_ok {
                        let n = (a - p).cross(b - p);
                        let len = n.length();
                        if len > 0.0 {
                            // Orient toward the camera: the camera looks down
                            // +Z, so a visible surface's normal has z < 0.
                            let n = if n.z > 0.0 { -n / len } else { n / len };
                            let view = p.normalize();
                            incidence = (-view.dot(n)).clamp(0.0, 1.0) as f32;
                            normal = [n.x as f32, n.y as f32, n.z as f32];
                        }
                    }
                }
            }
            normals[i] = normal;
            weights[i] = map.confidence[i] * incidence;
        }
    }
    SurfaceHints { normals, weights }
}

/// The camera parameters one fusion pass needs, already narrowed to `f32` and
/// expressed **relative to the grid's own origin**.
///
/// `#[repr(C)]`, scalar members only, and 112 bytes — a multiple of 16, which is
/// what a WGSL uniform block needs. It is the same bytes on both sides of the
/// seam by construction rather than by a layout rule nobody re-reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FuseParams {
    /// Camera-space position of grid voxel `(0, 0, 0)`, x.
    pub tx: f32,
    /// ... y.
    pub ty: f32,
    /// ... z.
    pub tz: f32,
    /// Camera-space displacement of one voxel step along grid x, x component.
    pub ax: f32,
    /// ... y.
    pub ay: f32,
    /// ... z.
    pub az: f32,
    /// One voxel step along grid y, x component.
    pub bx: f32,
    /// ... y.
    pub by: f32,
    /// ... z.
    pub bz: f32,
    /// One voxel step along grid z, x component.
    pub cx_: f32,
    /// ... y.
    pub cy_: f32,
    /// ... z.
    pub cz_: f32,
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
    /// The truncation band, in **reconstruction units**.
    pub truncation: f32,
    /// Depth-map width.
    pub width: u32,
    /// Depth-map height.
    pub height: u32,
    /// Grid dimension x.
    pub dim_x: u32,
    /// Grid dimension y.
    pub dim_y: u32,
    /// Grid dimension z.
    pub dim_z: u32,
    /// Padding to a 16-byte multiple.
    pub _pad: [u32; 4],
}

/// The dense `(distance, weight)` accumulator a fusion runs in.
#[derive(Clone, Debug)]
pub struct TsdfGrid {
    /// World position of voxel `(0, 0, 0)`'s **centre**, in `f64`.
    ///
    /// **This is the crate's second `f64` -> `f32` seam** (the first is
    /// [`crate::sweep::SweepGeometry::build`]). The origin stays `f64`; every
    /// voxel is an `f32` offset from it. That is the engine's floating-origin
    /// doctrine, applied to a scan volume.
    pub origin: DVec3,
    /// Edge length of one voxel, in **reconstruction units**.
    pub voxel_size: f64,
    /// Grid dimensions in voxels.
    pub dims: [u32; 3],
    /// The truncation band, in **reconstruction units**.
    pub truncation: f32,
    /// Weighted sum of truncated signed distances, one per voxel.
    pub distance: Vec<f32>,
    /// Sum of weights, one per voxel.
    pub weight: Vec<f32>,
}

impl TsdfGrid {
    /// An empty grid.
    pub fn new(origin: DVec3, voxel_size: f64, dims: [u32; 3], truncation: f32) -> Self {
        let n = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        Self {
            origin,
            voxel_size,
            dims,
            truncation,
            distance: vec![0.0; n],
            weight: vec![0.0; n],
        }
    }

    /// How many voxels.
    #[inline]
    pub fn len(&self) -> usize {
        self.distance.len()
    }

    /// Is the grid empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.distance.is_empty()
    }

    /// The `f32` camera parameters for integrating one view, relative to this
    /// grid's origin.
    pub fn fuse_params(&self, camera: &RegisteredCamera, map: &DepthMap) -> FuseParams {
        let r = camera.pose.rotation_matrix();
        let t = camera.pose.transform(self.origin);
        let step = |axis: DVec3| -> DVec3 { r.mul_vec3(axis) * self.voxel_size };
        let a = step(DVec3::X);
        let b = step(DVec3::Y);
        let c = step(DVec3::Z);
        FuseParams {
            tx: t.x as f32,
            ty: t.y as f32,
            tz: t.z as f32,
            ax: a.x as f32,
            ay: a.y as f32,
            az: a.z as f32,
            bx: b.x as f32,
            by: b.y as f32,
            bz: b.z as f32,
            cx_: c.x as f32,
            cy_: c.y as f32,
            cz_: c.z as f32,
            fx: camera.intrinsics.fx as f32,
            fy: camera.intrinsics.fy as f32,
            cx: camera.intrinsics.cx as f32,
            cy: camera.intrinsics.cy as f32,
            k1: camera.intrinsics.k1 as f32,
            k2: camera.intrinsics.k2 as f32,
            truncation: self.truncation,
            width: map.width,
            height: map.height,
            dim_x: self.dims[0],
            dim_y: self.dims[1],
            dim_z: self.dims[2],
            _pad: [0; 4],
        }
    }

    /// Integrate one view: the **CPU reference**, and the definition of what
    /// `tsdf.wgsl` computes.
    ///
    /// Returns how many voxels this view contributed to. Parallel over grid
    /// z-slabs — disjoint per voxel, so the fold that applies them is a copy
    /// rather than a sum and has no order to depend on.
    pub fn integrate_view(
        &mut self,
        params: &FuseParams,
        map: &DepthMap,
        weights: &[f32],
        pool: &JobPool,
    ) -> usize {
        let dims = self.dims;
        let slab = dims[0] as usize * dims[1] as usize;
        let slices: Vec<u32> = (0..dims[2]).collect();
        let contributions: Vec<Vec<(f32, f32)>> = pool.parallel_map_ref(&slices, |&k| {
            let mut out = vec![(0.0f32, 0.0f32); slab];
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let local = j as usize * dims[0] as usize + i as usize;
                    if let Some(v) = voxel_vote(params, map, weights, i, j, k) {
                        out[local] = v;
                    }
                }
            }
            out
        });
        let mut touched = 0usize;
        for (k, slice) in contributions.iter().enumerate() {
            let base = k * slab;
            for (local, &(d, w)) in slice.iter().enumerate() {
                if w > 0.0 {
                    self.distance[base + local] += d;
                    self.weight[base + local] += w;
                    touched += 1;
                }
            }
        }
        touched
    }

    /// The resolved signed distance at a voxel, in **reconstruction units**, or
    /// `None` when nothing observed it well enough.
    #[inline]
    pub fn resolved(&self, index: usize, min_weight: f32) -> Option<f32> {
        let w = self.weight[index];
        if !(w >= min_weight) || w <= 0.0 {
            return None;
        }
        Some(self.distance[index] / w)
    }

    /// Write the resolved field into an [`inf_voxel::VoxelData`] and mesh it
    /// through the mesher that is already there.
    ///
    /// Distances go in **in voxels** (`inf-voxel`'s unit, not metres) with the
    /// crate's own sign convention — negative inside — which is the convention
    /// this module already accumulates in. Voxels below `min_weight` are left
    /// **absent**, which `inf-voxel` reads as `EMPTY_SDF`: outside, unknown, and
    /// costing nothing.
    pub fn extract(&self, min_weight: f32) -> (VoxelData, DenseMesh) {
        let mut data = VoxelData::new(self.voxel_size).with_origin(self.origin);
        let inv_voxel = 1.0 / self.voxel_size as f32;
        let band = SDF_BAND;
        for k in 0..self.dims[2] {
            for j in 0..self.dims[1] {
                for i in 0..self.dims[0] {
                    let index = self.index(i, j, k);
                    let Some(d) = self.resolved(index, min_weight) else {
                        continue;
                    };
                    let voxels = (d * inv_voxel).clamp(-band, band);
                    data.set_sample_global(i as i32, j as i32, k as i32, voxels);
                }
            }
        }
        let mesh = self.mesh_from(&data, min_weight);
        (data, mesh)
    }

    /// Mesh a written volume, welding the per-chunk buffers into one.
    ///
    /// The weld key is the mesher's own **global cell coordinate**, which is the
    /// identity `inf-voxel` guarantees is bit-identical from both sides of a
    /// chunk seam. So this is an exact integer merge, not an epsilon-radius
    /// vertex weld — and the result is watertight for the same reason the
    /// per-chunk seams are.
    fn mesh_from(&self, data: &VoxelData, min_weight: f32) -> DenseMesh {
        let mut positions: Vec<DVec3> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut confidence: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut index_of: BTreeMap<[i32; 3], u32> = BTreeMap::new();
        for key in mesh_keys_for(data) {
            let chunk = mesh_chunk(data, key);
            if chunk.is_empty() {
                continue;
            }
            let mut local = vec![0u32; chunk.positions.len()];
            for (vi, cell) in chunk.cells.iter().enumerate() {
                local[vi] = match index_of.get(cell) {
                    Some(&id) => id,
                    None => {
                        let id = positions.len() as u32;
                        let g = chunk.positions[vi];
                        // Mesher positions are in GLOBAL VOXEL-GRID units; the
                        // grid's own origin and voxel size put them back in the
                        // reconstruction's frame.
                        positions
                            .push(self.origin + DVec3::new(g[0], g[1], g[2]) * self.voxel_size);
                        normals.push(chunk.normals[vi]);
                        confidence.push(self.support_at(g, min_weight));
                        index_of.insert(*cell, id);
                        id
                    }
                };
            }
            for tri in chunk.indices.chunks_exact(3) {
                indices.push(local[tri[0] as usize]);
                indices.push(local[tri[1] as usize]);
                indices.push(local[tri[2] as usize]);
            }
        }
        DenseMesh {
            positions,
            normals,
            confidence,
            indices,
        }
    }

    /// A saturating `[0, 1)` view of the accumulated fusion weight nearest a
    /// vertex: `w / (w + 1)`.
    ///
    /// Saturating rather than normalised because the maximum possible weight
    /// depends on how many views saw a place, and dividing by that would make a
    /// vertex seen well by two cameras look identical to one seen well by six.
    /// P25.3's bakes want to know which is which.
    fn support_at(&self, grid: [f64; 3], min_weight: f32) -> f32 {
        let i = (grid[0] + 0.5).floor();
        let j = (grid[1] + 0.5).floor();
        let k = (grid[2] + 0.5).floor();
        if i < 0.0 || j < 0.0 || k < 0.0 {
            return 0.0;
        }
        let (i, j, k) = (i as u32, j as u32, k as u32);
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return 0.0;
        }
        let index = self.index(i, j, k);
        let w = self.weight[index];
        if w < min_weight {
            return 0.0;
        }
        w / (w + 1.0)
    }

    /// Flat index of a voxel, x fastest — the same order `inf-voxel` and the
    /// shader use.
    #[inline]
    pub fn index(&self, i: u32, j: u32, k: u32) -> usize {
        (k as usize * self.dims[1] as usize + j as usize) * self.dims[0] as usize + i as usize
    }
}

/// One voxel's vote from one view, or `None` when that view has nothing to say
/// about it.
///
/// **This is the CPU half of the fusion parity claim**; `tsdf.wgsl`'s
/// `voxel_vote` reproduces it. Deliberate details, in the order they bite:
///
/// * The voxel's camera-space position is built as `t + i·a + j·b + k·c` from
///   the pre-multiplied per-axis steps, so the whole grid-to-camera transform is
///   three multiply-adds rather than a matrix product with a translation.
/// * `floor(x + 0.5)`, for the reason [`crate::sweep::neighbour_cost`] gives.
/// * A sample **more than one truncation behind** the observed surface is not a
///   vote at all: it is either occluded, or on the far side of a wall nobody
///   photographed. Returning `None` there is what keeps the fused volume a slab
///   around the surface instead of a solid that fills the frustum.
pub fn voxel_vote(
    params: &FuseParams,
    map: &DepthMap,
    weights: &[f32],
    i: u32,
    j: u32,
    k: u32,
) -> Option<(f32, f32)> {
    let (fi, fj, fk) = (i as f32, j as f32, k as f32);
    let qx = params.tx + fi * params.ax + fj * params.bx + fk * params.cx_;
    let qy = params.ty + fi * params.ay + fj * params.by + fk * params.cy_;
    let qz = params.tz + fi * params.az + fj * params.bz + fk * params.cz_;
    if !(qz > crate::sweep::MIN_SAMPLE_Z) {
        return None;
    }
    let u = qx / qz;
    let v = qy / qz;
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let f = 1.0 + params.k1 * r2 + params.k2 * r4;
    let px = params.fx * (u * f) + params.cx;
    let py = params.fy * (v * f) + params.cy;
    let ix = (px + 0.5).floor();
    let iy = (py + 0.5).floor();
    if !(ix >= 0.0)
        || !(iy >= 0.0)
        || ix > (params.width - 1) as f32
        || iy > (params.height - 1) as f32
    {
        return None;
    }
    let index = iy as usize * params.width as usize + ix as usize;
    let observed = map.depth[index];
    if !(observed > 0.0) {
        return None;
    }
    let w = weights[index];
    if !(w > 0.0) {
        return None;
    }
    // Positive in front of the surface (outside), negative behind it (inside):
    // `inf-voxel`'s convention, adopted here so nothing has to flip a sign at
    // the boundary between the two crates.
    let sdf = observed - qz;
    if sdf < -params.truncation {
        return None;
    }
    let clamped = if sdf > params.truncation {
        params.truncation
    } else {
        sdf
    };
    Some((w * clamped, w))
}

/// The fused surface: one welded triangle mesh in **reconstruction units**.
///
/// This is the type P25.3's finish pipeline consumes. It carries no UVs and no
/// tangents on purpose — decimation comes before unwrapping, and unwrapping is
/// P23.5's LSCM, which wants a decimated mesh rather than a fusion-resolution
/// one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DenseMesh {
    /// Vertex positions in world space, `f64`, in the reconstruction's own
    /// **baseline units**.
    pub positions: Vec<DVec3>,
    /// Unit normals from the SDF gradient, `f32` — the mesher's own.
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex support in `[0, 1)`: a saturating view of how much fusion
    /// weight backs this vertex. P25.3's bakes can use it to prefer well-seen
    /// geometry; it is **not** a probability.
    pub confidence: Vec<f32>,
    /// Triangle indices, counter-clockwise from outside.
    pub indices: Vec<u32>,
}

impl DenseMesh {
    /// How many vertices.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// How many triangles.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Is there any surface at all?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Every edge of the mesh, canonicalised low-index-first, with how many
    /// triangles use it.
    ///
    /// A closed surface uses every edge **exactly twice**. This is what
    /// "watertight" means as a computable claim, and the gate asserts it rather
    /// than trusting the mesher's promise across a weld this crate performed.
    pub fn edge_use_counts(&self) -> BTreeMap<(u32, u32), usize> {
        let mut counts: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for tri in self.indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Canonical bytes for the determinism surface.
    pub fn write_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.positions.len() as u64).to_le_bytes());
        for p in &self.positions {
            for v in p.to_array() {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for n in &self.normals {
            for v in n {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for c in &self.confidence {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&(self.indices.len() as u64).to_le_bytes());
        for i in &self.indices {
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
}

/// The world-space bounds of every valid depth sample, trimmed to a percentile
/// per axis.
///
/// Returns `None` when nothing is valid anywhere.
pub fn observed_bounds(
    maps: &[DepthMap],
    cameras: &[RegisteredCamera],
    percentile: f64,
) -> Option<(DVec3, DVec3)> {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut zs: Vec<f64> = Vec::new();
    for (map, cam) in maps.iter().zip(cameras) {
        for y in 0..map.height {
            for x in 0..map.width {
                let i = y as usize * map.width as usize + x as usize;
                let d = map.depth[i];
                if !(d > 0.0) {
                    continue;
                }
                let p = backproject_world(cam, x, y, d);
                xs.push(p.x);
                ys.push(p.y);
                zs.push(p.z);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let axis = |v: &mut Vec<f64>| -> (f64, f64) {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let lo = ((percentile * (n - 1) as f64).floor().max(0.0) as usize).min(n - 1);
        let hi = (((1.0 - percentile) * (n - 1) as f64).floor().max(0.0) as usize).min(n - 1);
        (v[lo], v[hi.max(lo)])
    };
    let (x0, x1) = axis(&mut xs);
    let (y0, y1) = axis(&mut ys);
    let (z0, z1) = axis(&mut zs);
    Some((DVec3::new(x0, y0, z0), DVec3::new(x1, y1, z1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_photo::camera::{Intrinsics, Pose};

    fn camera(view: u32, eye: DVec3) -> RegisteredCamera {
        RegisteredCamera {
            view,
            pose: Pose::look_at(eye, DVec3::new(0.0, 0.0, 3.0), DVec3::NEG_Y),
            intrinsics: Intrinsics::centred(48, 36, 40.0),
            observations: 0,
        }
    }

    /// A depth map of the world plane `z = plane`.
    fn plane_map(cam: &RegisteredCamera, plane: f64) -> DepthMap {
        let mut map = DepthMap::empty(cam.view, 48, 36);
        let eye = cam.pose.center();
        for y in 0..36u32 {
            for x in 0..48u32 {
                let n = cam
                    .intrinsics
                    .to_normalized(glam::DVec2::new(x as f64, y as f64));
                let dir = cam.pose.rotation.inverse() * DVec3::new(n.x, n.y, 1.0);
                if dir.z.abs() < 1e-9 {
                    continue;
                }
                let t = (plane - eye.z) / dir.z;
                if t <= 0.0 {
                    continue;
                }
                let i = y as usize * 48 + x as usize;
                map.depth[i] = t as f32;
                map.confidence[i] = 1.0;
            }
        }
        map
    }

    #[test]
    fn a_plane_fuses_to_a_field_whose_zero_crossing_is_that_plane() {
        let cams = [
            camera(0, DVec3::new(0.0, 0.0, 0.0)),
            camera(1, DVec3::new(0.4, 0.05, 0.0)),
        ];
        let maps: Vec<DepthMap> = cams.iter().map(|c| plane_map(c, 3.0)).collect();
        let voxel = 0.05;
        let dims = [40u32, 30, 24];
        let origin = DVec3::new(-1.0, -0.75, 2.4);
        let mut grid = TsdfGrid::new(origin, voxel, dims, (3.0 * voxel) as f32);
        let pool = JobPool::new(2);
        for (cam, map) in cams.iter().zip(&maps) {
            let hints = surface_hints(cam, map, 0.02);
            let params = grid.fuse_params(cam, map);
            let touched = grid.integrate_view(&params, map, &hints.weights, &pool);
            assert!(
                touched > 1000,
                "a whole plane only touched {touched} voxels"
            );
        }
        // Sample down the middle column: the resolved distance must cross zero
        // exactly where the plane is.
        let (i, j) = (20u32, 15u32);
        let mut crossing = None;
        for k in 0..dims[2] - 1 {
            let a = grid.resolved(grid.index(i, j, k), 0.1);
            let b = grid.resolved(grid.index(i, j, k + 1), 0.1);
            if let (Some(a), Some(b)) = (a, b) {
                if a > 0.0 && b <= 0.0 {
                    let t = a / (a - b);
                    crossing = Some(origin.z + (k as f64 + t as f64) * voxel);
                }
            }
        }
        let z = crossing.expect("the field never crossed zero");
        assert!(
            (z - 3.0).abs() < voxel,
            "the zero crossing is at z = {z}, the plane is at 3.0"
        );
    }

    #[test]
    fn nothing_behind_the_truncation_band_is_written() {
        // The ruling the module docs make: a voxel a long way behind the
        // surface is unknown, not inside. Sever this and the fused volume fills
        // the frustum.
        let cam = camera(0, DVec3::ZERO);
        let map = plane_map(&cam, 3.0);
        let hints = surface_hints(&cam, &map, 0.02);
        let voxel = 0.05f64;
        let grid = TsdfGrid::new(
            DVec3::new(-0.1, -0.1, 2.5),
            voxel,
            [4, 4, 40],
            (3.0 * voxel) as f32,
        );
        let params = grid.fuse_params(&cam, &map);
        let mut written = 0usize;
        let mut deepest = 0.0f64;
        for k in 0..40u32 {
            if voxel_vote(&params, &map, &hints.weights, 2, 2, k).is_some() {
                written += 1;
                deepest = 2.5 + k as f64 * voxel;
            }
        }
        assert!(written > 5, "nothing was written at all");
        assert!(
            deepest <= 3.0 + 3.0 * voxel + 1e-9,
            "a voxel {deepest} was written, which is more than one truncation behind z = 3.0"
        );
    }

    #[test]
    fn a_grazing_surface_is_weighted_below_a_face_on_one() {
        // Two views of the same plane, one square on and one at a slant. The
        // slanted one's weights must be lower — this is the incidence factor
        // doing its job, and the arm that goes red if the cross product is
        // dropped.
        let face_on = camera(0, DVec3::ZERO);
        let mut slanted = camera(1, DVec3::new(2.6, 0.0, 0.2));
        slanted.pose = Pose::look_at(
            DVec3::new(2.6, 0.0, 0.2),
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::NEG_Y,
        );
        let a = surface_hints(&face_on, &plane_map(&face_on, 3.0), 0.05);
        let b = surface_hints(&slanted, &plane_map(&slanted, 3.0), 0.05);
        let mean = |h: &SurfaceHints| -> f32 {
            let v: Vec<f32> = h.weights.iter().copied().filter(|w| *w > 0.0).collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        let (ma, mb) = (mean(&a), mean(&b));
        assert!(
            ma > mb * 1.3,
            "face-on weight {ma} is not meaningfully above the grazing {mb}"
        );
        assert!(ma <= 1.0 && mb > 0.0, "{ma} / {mb}");
    }

    #[test]
    fn the_grid_index_is_x_fastest_and_total() {
        let g = TsdfGrid::new(DVec3::ZERO, 1.0, [3, 4, 5], 1.0);
        assert_eq!(g.len(), 60);
        assert_eq!(g.index(0, 0, 0), 0);
        assert_eq!(g.index(1, 0, 0), 1);
        assert_eq!(g.index(0, 1, 0), 3);
        assert_eq!(g.index(0, 0, 1), 12);
        let mut seen = std::collections::BTreeSet::new();
        for k in 0..5 {
            for j in 0..4 {
                for i in 0..3 {
                    assert!(seen.insert(g.index(i, j, k)), "index collision");
                }
            }
        }
        assert_eq!(seen.len(), 60);
    }

    #[test]
    fn a_closed_surface_uses_every_edge_exactly_twice() {
        // A tetrahedron, by hand: the property `edge_use_counts` measures, and
        // the arm that proves it is not vacuously true of any index list.
        let mesh = DenseMesh {
            positions: vec![DVec3::ZERO; 4],
            normals: vec![[0.0; 3]; 4],
            confidence: vec![0.0; 4],
            indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        };
        let counts = mesh.edge_use_counts();
        assert_eq!(counts.len(), 6);
        assert!(counts.values().all(|&c| c == 2), "counts {counts:?}");
        // Open it and the claim fails.
        let open = DenseMesh {
            indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1],
            ..mesh
        };
        assert!(open.edge_use_counts().values().any(|&c| c == 1));
    }
}
