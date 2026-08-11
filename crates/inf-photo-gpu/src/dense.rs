//! **The dense pipeline**: the one order of operations, and the two backends
//! that can execute it.
//!
//! # The order
//!
//! 1. **Validate.** Too few cameras, a camera with no view, or intrinsics that
//!    do not describe the image they are attached to is a named
//!    [`DenseError`] rather than a degenerate result.
//! 2. **Census.** Every registered view's image is census-transformed once, on
//!    the job pool, and the results are packed into one buffer the sweep indexes
//!    with per-view offsets. Once, because a census transform is a property of
//!    an image, not of a sweep, and a view is a neighbour of several references.
//! 3. **Sweep**, per reference view in ascending view order: pick the
//!    neighbourhood by covisibility, derive the depth range from that view's
//!    share of the sparse cloud, narrow the geometry to `f32`, and run the
//!    plane sweep on whichever backend was handed in.
//! 4. **Filter.** Left-right consistency, then speckle removal — both computed
//!    from the **unfiltered** maps and applied afterwards, so no view's result
//!    depends on which views were filtered before it.
//! 5. **Fuse.** Derive the voxel size and the grid bounds from what survived,
//!    then integrate every view in ascending order into one TSDF.
//! 6. **Extract.** Write the resolved field into an [`inf_voxel::VoxelData`] and
//!    mesh it with the mesher that is already there.
//!
//! # One orchestrator, two backends
//!
//! [`DenseBackend`] is the seam. `CpuBackend` is the semantic definition;
//! [`crate::gpu::GpuBackend`] is a mirror of it. They are not two pipelines with
//! a flag — they are one pipeline with two implementations of two functions, so
//! "the GPU path agrees with the CPU path" is a claim about those two functions
//! and nothing else can drift underneath it.
//!
//! The reference-view loop is **serial**. Each view's sweep is internally
//! parallel (rows on the job pool, or a GPU dispatch), and running the views
//! themselves in parallel would put several GPU submissions in flight against
//! one queue for no measurable gain on captures of this size.

use std::collections::BTreeMap;

use inf_core::job::JobPool;
use inf_photo::gray::GrayImage;
use inf_photo::sfm::{Reconstruction, RegisteredCamera, View};

use crate::filter::{apply_mask, consistency_mask, speckle_mask, FilterConfig};
use crate::sweep::{
    census_transform, depth_range_for, neighbours_for, sweep_view, CensusConfig, DepthMap,
    NeighbourView, SweepConfig, SweepGeometry,
};
use crate::tsdf::{
    observed_bounds, surface_hints, DenseMesh, FuseParams, SurfaceHints, TsdfConfig, TsdfGrid,
};
use crate::{DenseAdvisory, DenseError};

/// How the whole dense stage is tuned. Every field changes reconstructions; the
/// defaults are what `tests/mvs_gate.rs` measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseConfig {
    /// The plane sweep.
    pub sweep: SweepConfig,
    /// The two filters.
    pub filter: FilterConfig,
    /// Fusion and extraction.
    pub tsdf: TsdfConfig,
    /// Fewer registered cameras than this is a refusal.
    ///
    /// Three, matching [`inf_photo::SfmConfig::min_views`]: a dense stage over
    /// two cameras has one baseline, no third opinion, and no left-right check
    /// worth running.
    pub min_cameras: usize,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            sweep: SweepConfig::default(),
            filter: FilterConfig::default(),
            tsdf: TsdfConfig::default(),
            min_cameras: 3,
        }
    }
}

/// What the dense stage did, in numbers a diagnostics panel can show.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseReport {
    /// How many cameras the sparse reconstruction registered.
    pub cameras: usize,
    /// How many views were swept.
    pub swept: usize,
    /// How many planes each sweep used.
    pub planes: u32,
    /// Valid pixels per depth map after filtering, in view order.
    pub valid_per_view: Vec<usize>,
    /// The near plane of each sweep, in view order, **reconstruction units**.
    pub near_per_view: Vec<f64>,
    /// The far plane of each sweep, in view order.
    pub far_per_view: Vec<f64>,
    /// The fusion voxel size, in **reconstruction units**.
    pub voxel_size: f64,
    /// The fusion grid's dimensions in voxels.
    pub grid_dims: [u32; 3],
    /// How many voxel-votes were cast in total, over every view.
    pub votes: usize,
    /// How many voxels ended up carrying a resolved distance.
    pub surface_voxels: usize,
    /// Vertices in the extracted mesh.
    pub vertices: usize,
    /// Triangles in the extracted mesh.
    pub triangles: usize,
}

/// Depth maps, surface hints, the fused mesh, and the tale of how they were
/// found.
#[derive(Clone, Debug)]
pub struct DenseReconstruction {
    /// One depth map per registered view, in ascending view order.
    pub depth_maps: Vec<DepthMap>,
    /// Per-pixel normals and fusion weights, parallel to `depth_maps`.
    pub surface: Vec<SurfaceHints>,
    /// The fused, welded dense mesh.
    pub mesh: DenseMesh,
    /// Non-fatal findings, in canonical order.
    pub advisories: Vec<DenseAdvisory>,
    /// The numbers.
    pub report: DenseReport,
}

impl DenseReconstruction {
    /// The dense reconstruction's **canonical byte image**: the assertion
    /// surface for determinism.
    ///
    /// Everything that defines the result goes in, in a fixed order — depth
    /// maps, normals, weights, the mesh, the advisories and **the whole
    /// report**. Floats are little-endian IEEE bit patterns, never formatted
    /// decimals, so a one-ULP difference cannot hide behind a rounded digit.
    ///
    /// P25.1 learned this the expensive way: its `canonical_bytes` wrote one
    /// field of its report, so both of its byte-identity arms were comparing an
    /// image that omitted a *branch point* of the algorithm. `tests/mvs_gate.rs`
    /// therefore asserts the coverage field by field rather than trusting this
    /// list to stay complete.
    ///
    /// **The P25.3 ruling, which this note used to pre-empt.** It read "the day
    /// a dense reconstruction is written into an asset, that exemption ends and
    /// every function that touched these numbers comes under the portable
    /// family". That day arrived — P25.3 bakes these numbers into a `.inf_mesh`
    /// and three `.inf_tex` — and the notice was settled rather than executed:
    /// a photogrammetry import is the glTF-import class, so the solvers keep
    /// their exemption, and what the portable family still governs is anything a
    /// **gate** re-derives and any **table the committed bytes are a function
    /// of**. The whole ruling is written once, in [`inf_photo`]'s crate docs
    /// under "The line, as P25.3 settled it"; the two crate-level notices point
    /// at it and so does this one.
    ///
    /// What survives unchanged is the practical rule for *these* bytes: they are
    /// in-memory and compared within one run or within one machine, never
    /// against a committed number.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"INF_PHOTO_DENSE\x01");
        out.extend_from_slice(&(self.depth_maps.len() as u64).to_le_bytes());
        for map in &self.depth_maps {
            map.write_canonical(&mut out);
        }
        out.extend_from_slice(&(self.surface.len() as u64).to_le_bytes());
        for hints in &self.surface {
            out.extend_from_slice(&(hints.normals.len() as u64).to_le_bytes());
            for n in &hints.normals {
                for v in n {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
            for w in &hints.weights {
                out.extend_from_slice(&w.to_bits().to_le_bytes());
            }
        }
        self.mesh.write_canonical(&mut out);
        out.extend_from_slice(&(self.advisories.len() as u64).to_le_bytes());
        for a in &self.advisories {
            out.extend_from_slice(format!("{a}").as_bytes());
            out.push(0);
        }
        let r = &self.report;
        for v in [
            r.cameras as u64,
            r.swept as u64,
            r.planes as u64,
            r.valid_per_view.len() as u64,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for n in &r.valid_per_view {
            out.extend_from_slice(&(*n as u64).to_le_bytes());
        }
        for v in &r.near_per_view {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in &r.far_per_view {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&r.voxel_size.to_bits().to_le_bytes());
        for d in r.grid_dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for v in [
            r.votes as u64,
            r.surface_voxels as u64,
            r.vertices as u64,
            r.triangles as u64,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// The depth map of a view, if it was swept.
    pub fn depth_map(&self, view: u32) -> Option<&DepthMap> {
        self.depth_maps.iter().find(|m| m.view == view)
    }
}

/// The seam between the pipeline and the machine that runs its two heavy
/// stages.
///
/// Implemented by [`CpuBackend`] (the definition) and
/// [`crate::gpu::GpuBackend`] (the mirror). Both take `&mut self` because a GPU
/// backend caches pipelines and buffers between calls; the CPU one holds
/// nothing.
pub trait DenseBackend {
    /// A name for reports and test output.
    fn name(&self) -> &'static str;

    /// Census-transform every registered view's image, in view order.
    fn census_all(
        &mut self,
        images: &[&GrayImage],
        cfg: &CensusConfig,
        pool: &JobPool,
    ) -> Vec<Vec<u32>>;

    /// Sweep one reference view.
    fn sweep(
        &mut self,
        geometry: &SweepGeometry,
        reference_census: &[u32],
        census: &[u32],
        cfg: &SweepConfig,
        pool: &JobPool,
    ) -> DepthMap;

    /// Integrate one view into the TSDF. Returns how many voxels it touched.
    fn integrate(
        &mut self,
        grid: &mut TsdfGrid,
        params: &FuseParams,
        map: &DepthMap,
        weights: &[f32],
        pool: &JobPool,
    ) -> usize;
}

/// The CPU backend: the semantic definition of every stage.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl DenseBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn census_all(
        &mut self,
        images: &[&GrayImage],
        cfg: &CensusConfig,
        pool: &JobPool,
    ) -> Vec<Vec<u32>> {
        let cfg = *cfg;
        pool.parallel_map_ref(images, |img| census_transform(img, &cfg))
    }

    fn sweep(
        &mut self,
        geometry: &SweepGeometry,
        reference_census: &[u32],
        census: &[u32],
        cfg: &SweepConfig,
        pool: &JobPool,
    ) -> DepthMap {
        sweep_view(geometry, reference_census, census, cfg, pool)
    }

    fn integrate(
        &mut self,
        grid: &mut TsdfGrid,
        params: &FuseParams,
        map: &DepthMap,
        weights: &[f32],
        pool: &JobPool,
    ) -> usize {
        grid.integrate_view(params, map, weights, pool)
    }
}

/// Run the dense stage on the CPU.
pub fn reconstruct_dense(
    views: &[View],
    reconstruction: &Reconstruction,
    cfg: &DenseConfig,
    pool: &JobPool,
) -> Result<DenseReconstruction, DenseError> {
    reconstruct_dense_with(views, reconstruction, cfg, pool, &mut CpuBackend)
}

/// Run the dense stage on a given backend.
pub fn reconstruct_dense_with(
    views: &[View],
    reconstruction: &Reconstruction,
    cfg: &DenseConfig,
    pool: &JobPool,
    backend: &mut dyn DenseBackend,
) -> Result<DenseReconstruction, DenseError> {
    let cameras: Vec<RegisteredCamera> = reconstruction.cameras.values().copied().collect();
    if cameras.len() < cfg.min_cameras {
        return Err(DenseError::TooFewCameras {
            given: cameras.len(),
            required: cfg.min_cameras,
        });
    }
    for cam in &cameras {
        let Some(view) = views.get(cam.view as usize) else {
            return Err(DenseError::MissingView {
                view: cam.view,
                given: views.len(),
            });
        };
        if view.image.width() != cam.intrinsics.width
            || view.image.height() != cam.intrinsics.height
        {
            return Err(DenseError::IntrinsicsMismatch {
                view: cam.view,
                img_w: view.image.width(),
                img_h: view.image.height(),
                int_w: cam.intrinsics.width,
                int_h: cam.intrinsics.height,
            });
        }
    }
    // Registered position of each view index — the sweep works in positions,
    // the reconstruction speaks in view indices, and this is the one map
    // between them.
    let position_of: BTreeMap<u32, usize> = cameras
        .iter()
        .enumerate()
        .map(|(i, c)| (c.view, i))
        .collect();

    // ── Census, once per view ──────────────────────────────────────────────
    let images: Vec<&GrayImage> = cameras
        .iter()
        .map(|cam| &views[cam.view as usize].image)
        .collect();
    let per_view: Vec<Vec<u32>> = backend.census_all(&images, &cfg.sweep.census, pool);
    let mut offsets = Vec::with_capacity(per_view.len());
    let mut census: Vec<u32> = Vec::new();
    for codes in &per_view {
        offsets.push(census.len() as u32);
        census.extend_from_slice(codes);
    }

    // ── Sweep, per reference view ──────────────────────────────────────────
    let mut advisories: Vec<DenseAdvisory> = Vec::new();
    let mut maps: Vec<DepthMap> = Vec::with_capacity(cameras.len());
    let mut neighbourhoods: Vec<Vec<usize>> = Vec::with_capacity(cameras.len());
    let mut near_per_view = Vec::with_capacity(cameras.len());
    let mut far_per_view = Vec::with_capacity(cameras.len());
    for (pos, cam) in cameras.iter().enumerate() {
        let neighbour_views = neighbours_for(cam.view, reconstruction, &cfg.sweep)?;
        let (near, far) = depth_range_for(cam.view, reconstruction, &cfg.sweep)?;
        if neighbour_views.len() < cfg.sweep.neighbours {
            advisories.push(DenseAdvisory::ThinNeighbourhood {
                view: cam.view,
                used: neighbour_views.len(),
                wanted: cfg.sweep.neighbours,
            });
        }
        let positions: Vec<usize> = neighbour_views.iter().map(|v| position_of[v]).collect();
        let neighbours: Vec<NeighbourView> = positions
            .iter()
            .map(|&p| NeighbourView {
                view: cameras[p].view,
                pose: cameras[p].pose,
                intrinsics: cameras[p].intrinsics,
                census_offset: offsets[p],
            })
            .collect();
        let geometry = SweepGeometry::build(cam, &neighbours, near, far, cfg.sweep.planes);
        maps.push(backend.sweep(&geometry, &per_view[pos], &census, &cfg.sweep, pool));
        neighbourhoods.push(positions);
        near_per_view.push(near);
        far_per_view.push(far);
    }

    // ── Filter ─────────────────────────────────────────────────────────────
    // Every mask is computed from the UNFILTERED maps and applied afterwards.
    let consistency: Vec<Vec<bool>> = (0..maps.len())
        .map(|i| consistency_mask(&maps, &cameras, i, &neighbourhoods[i], &cfg.filter))
        .collect();
    for (i, mask) in consistency.iter().enumerate() {
        let dropped = apply_mask(&mut maps[i], mask);
        if dropped > 0 {
            advisories.push(DenseAdvisory::ConsistencyRejected {
                view: cameras[i].view,
                dropped,
            });
        }
    }
    let speckle: Vec<(Vec<bool>, crate::filter::SpeckleReport)> =
        maps.iter().map(|m| speckle_mask(m, &cfg.filter)).collect();
    for (i, (mask, report)) in speckle.iter().enumerate() {
        apply_mask(&mut maps[i], mask);
        if report.dropped > 0 {
            advisories.push(DenseAdvisory::SpeckleRemoved {
                view: cameras[i].view,
                dropped: report.dropped,
                regions: report.regions,
            });
        }
    }
    for (i, map) in maps.iter().enumerate() {
        let valid = map.valid_count();
        if valid * 4 < map.pixels() {
            advisories.push(DenseAdvisory::SparseDepthMap {
                view: cameras[i].view,
                valid,
                total: map.pixels(),
            });
        }
    }
    let total_valid: usize = maps.iter().map(|m| m.valid_count()).sum();
    if total_valid == 0 {
        return Err(DenseError::NoDepthSurvived { swept: maps.len() });
    }

    // ── Fuse ───────────────────────────────────────────────────────────────
    let surface: Vec<SurfaceHints> = maps
        .iter()
        .zip(&cameras)
        .map(|(m, c)| surface_hints(c, m, cfg.filter.speckle_delta))
        .collect();
    let (voxel_size, clamped) = voxel_size_for(&maps, &cameras, &cfg.tsdf);
    if let Some((wanted, used)) = clamped {
        advisories.push(DenseAdvisory::VoxelSizeClamped {
            wanted_bits: wanted.to_bits(),
            used_bits: used.to_bits(),
        });
    }
    let truncation = voxel_size * cfg.tsdf.truncation_voxels.min(inf_voxel::SDF_BAND as f64);
    let Some((lo, hi)) = observed_bounds(&maps, &cameras, cfg.tsdf.bounds_percentile) else {
        return Err(DenseError::NoDepthSurvived { swept: maps.len() });
    };
    let pad = truncation * cfg.tsdf.bounds_pad_truncations;
    let origin = lo - glam::DVec3::splat(pad);
    let extent = (hi + glam::DVec3::splat(pad)) - origin;
    let dims = [
        ((extent.x / voxel_size).ceil() as i64 + 1).clamp(2, cfg.tsdf.max_grid_side as i64) as u32,
        ((extent.y / voxel_size).ceil() as i64 + 1).clamp(2, cfg.tsdf.max_grid_side as i64) as u32,
        ((extent.z / voxel_size).ceil() as i64 + 1).clamp(2, cfg.tsdf.max_grid_side as i64) as u32,
    ];
    let mut grid = TsdfGrid::new(origin, voxel_size, dims, truncation as f32);
    let mut votes = 0usize;
    for ((cam, map), hints) in cameras.iter().zip(&maps).zip(&surface) {
        let params = grid.fuse_params(cam, map);
        votes += backend.integrate(&mut grid, &params, map, &hints.weights, pool);
    }
    if votes == 0 {
        return Err(DenseError::EmptyFusion {
            samples: total_valid,
        });
    }

    // ── Extract ────────────────────────────────────────────────────────────
    let (_data, mesh) = grid.extract(cfg.tsdf.min_weight);
    let surface_voxels = (0..grid.len())
        .filter(|&i| grid.resolved(i, cfg.tsdf.min_weight).is_some())
        .count();

    advisories.sort();
    advisories.dedup();

    Ok(DenseReconstruction {
        report: DenseReport {
            cameras: cameras.len(),
            swept: maps.len(),
            planes: cfg.sweep.planes,
            valid_per_view: maps.iter().map(|m| m.valid_count()).collect(),
            near_per_view,
            far_per_view,
            voxel_size,
            grid_dims: dims,
            votes,
            surface_voxels,
            vertices: mesh.vertex_count(),
            triangles: mesh.triangle_count(),
        },
        depth_maps: maps,
        surface,
        mesh,
        advisories,
    })
}

/// The fusion voxel size, and what the pixel-footprint formula wanted if the
/// memory cap overruled it.
///
/// See [`crate::tsdf`]'s module docs for the formula. `mean_depth` is the mean
/// over views of that view's **median** valid depth: a median because one
/// runaway depth in one map must not size the whole grid, and a mean over views
/// because a capture whose stations are at different distances should get a
/// grid sized for the capture rather than for its closest photograph.
pub fn voxel_size_for(
    maps: &[DepthMap],
    cameras: &[RegisteredCamera],
    cfg: &TsdfConfig,
) -> (f64, Option<(f64, f64)>) {
    let mut depth_sum = 0.0f64;
    let mut depth_n = 0usize;
    for map in maps {
        let mut valid: Vec<f32> = map.depth.iter().copied().filter(|d| *d > 0.0).collect();
        if valid.is_empty() {
            continue;
        }
        valid.sort_by(f32::total_cmp);
        depth_sum += valid[valid.len() / 2] as f64;
        depth_n += 1;
    }
    let focal: f64 = cameras
        .iter()
        .map(|c| c.intrinsics.mean_focal())
        .sum::<f64>()
        / cameras.len().max(1) as f64;
    let mean_depth = if depth_n > 0 {
        depth_sum / depth_n as f64
    } else {
        1.0
    };
    let wanted = cfg.voxel_scale * mean_depth / focal.max(1.0);

    // The memory cap, expressed as a floor on the voxel size: whatever the
    // longest axis of the observed extent is, it must fit in `max_grid_side`
    // voxels.
    let extent = match observed_bounds(maps, cameras, cfg.bounds_percentile) {
        Some((lo, hi)) => {
            let e = hi - lo;
            e.x.max(e.y).max(e.z)
        }
        None => 0.0,
    };
    let floor = extent / cfg.max_grid_side as f64;
    if floor > wanted {
        (floor, Some((wanted, floor)))
    } else {
        (wanted, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_photo::camera::Intrinsics;
    use inf_photo::gray::GrayImage;

    fn tiny_reconstruction() -> (Vec<View>, Reconstruction) {
        let image = GrayImage::new(16, 12).unwrap();
        let intrinsics = Intrinsics::centred(16, 12, 20.0);
        let views = (0..3)
            .map(|_| View {
                image: image.clone(),
                intrinsics,
            })
            .collect();
        let cameras = (0..3u32)
            .map(|v| {
                (
                    v,
                    RegisteredCamera {
                        view: v,
                        pose: inf_photo::camera::Pose::IDENTITY,
                        intrinsics,
                        observations: 0,
                    },
                )
            })
            .collect();
        let reconstruction = Reconstruction {
            cameras,
            points: Vec::new(),
            advisories: Vec::new(),
            report: inf_photo::sfm::SfmReport {
                views: 3,
                registered: 3,
                features_per_view: vec![0; 3],
                pairs: 0,
                matches: 0,
                tracks: 0,
                points: 0,
                observations: 0,
                reprojection_rms_px: 0.0,
                initial_pair: (0, 1),
                final_bundle: inf_photo::bundle::BundleReport::empty(),
            },
        };
        (views, reconstruction)
    }

    #[test]
    fn too_few_cameras_refuses_by_name() {
        let (views, mut recon) = tiny_reconstruction();
        recon.cameras.remove(&2);
        let err = reconstruct_dense(&views, &recon, &DenseConfig::default(), &JobPool::new(1))
            .unwrap_err();
        assert_eq!(
            err,
            DenseError::TooFewCameras {
                given: 2,
                required: 3
            }
        );
        assert!(format!("{err}").contains("at least 3"));
    }

    #[test]
    fn a_camera_with_no_neighbours_refuses_by_name() {
        // No points at all means no covisibility, which is the shape of a
        // capture whose photographs do not overlap. It must refuse by name
        // rather than sweep against nothing.
        let (views, recon) = tiny_reconstruction();
        let err = reconstruct_dense(&views, &recon, &DenseConfig::default(), &JobPool::new(1))
            .unwrap_err();
        assert!(
            matches!(
                err,
                DenseError::TooFewNeighbours {
                    view: 0,
                    found: 0,
                    required: 2
                }
            ),
            "expected a named neighbour refusal, got {err:?}"
        );
        assert!(format!("{err}").contains("covisible"));
    }

    #[test]
    fn mismatched_intrinsics_refuse_by_name() {
        let (views, mut recon) = tiny_reconstruction();
        recon.cameras.get_mut(&1).unwrap().intrinsics.width = 17;
        let err = reconstruct_dense(&views, &recon, &DenseConfig::default(), &JobPool::new(1))
            .unwrap_err();
        assert!(
            matches!(err, DenseError::IntrinsicsMismatch { view: 1, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn the_committed_defaults_are_the_ones_the_gate_measured() {
        // `DenseConfig::default()` is not a convenience — it is the
        // configuration `tests/mvs_gate.rs` states its numbers for, and every
        // field of it changes depth maps. Pinned field by field, so a
        // "harmless" tweak is a red test beside the gate's measured block
        // rather than a silent drift in every number this crate reports.
        let c = DenseConfig::default();
        assert_eq!(c.min_cameras, 3);

        assert_eq!(c.sweep.census.dilation, 2);
        assert_eq!(c.sweep.planes, 64);
        assert_eq!(c.sweep.neighbours, 3);
        assert_eq!(c.sweep.min_neighbours, 2);
        assert_eq!(c.sweep.min_valid_neighbours, 2);
        assert_eq!(c.sweep.low_percentile, 0.02);
        assert_eq!(c.sweep.high_percentile, 0.98);
        assert_eq!(c.sweep.pad_fraction, 0.20);
        assert_eq!(c.sweep.min_range_points, 24);
        assert_eq!(c.sweep.max_cost, 2048);
        assert_eq!(c.sweep.min_confidence, 0.06);
        assert_eq!(c.sweep.exclusion_planes, 2);

        assert_eq!(c.filter.lr_tolerance, 0.02);
        assert_eq!(c.filter.lr_min_agree, 1);
        assert_eq!(c.filter.speckle_delta, 0.02);
        assert_eq!(c.filter.speckle_min_region, 24);

        assert_eq!(c.tsdf.voxel_scale, 2.0);
        assert_eq!(c.tsdf.max_grid_side, 256);
        assert_eq!(c.tsdf.truncation_voxels, 3.0);
        assert_eq!(c.tsdf.min_weight, 0.10);
        assert_eq!(c.tsdf.bounds_pad_truncations, 2.0);
        assert_eq!(c.tsdf.bounds_percentile, 0.002);

        // And the truncation this crate reasons about really does fit inside
        // the band `inf-voxel` clamps to, or the store would silently re-clamp
        // it and the two crates would disagree about where the band ends.
        assert!(c.tsdf.truncation_voxels <= inf_voxel::SDF_BAND as f64);
    }
}
