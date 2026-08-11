//! **Photogrammetry: the dense stage** (P25.2) — poses and a sparse cloud in,
//! per-view depth maps and one fused dense mesh out, with the expensive half
//! mirrored into WGSL and gated against the CPU that defines it.
//!
//! # Where this sits
//!
//! [`inf_photo`] is the *first* half of Phase 25: it turns photographs into
//! camera poses and a sparse point cloud, in `f64`, with **no `wgpu`
//! dependency** — a line its crate docs draw deliberately. This crate is the
//! second half and is where the GPU enters. It consumes an
//! [`inf_photo::Reconstruction`] and produces a [`DenseReconstruction`].
//!
//! Serialization is **not** here. P25.3 owns the finish pipeline (decimation,
//! UV unwrap, bakes) and P25.4 the wizard; everything this crate produces is an
//! in-memory structure. See "transcendentals" below for the notice that attaches
//! to the day those bytes get committed.
//!
//! # The CPU is the definition; the GPU is a mirror
//!
//! Every stage that has a WGSL form has a CPU form first, and the CPU form is
//! the semantic definition:
//!
//! | stage | CPU reference | WGSL mirror | parity |
//! |---|---|---|---|
//! | census transform | [`sweep::census_transform`] | `census.wgsl` | **bit-exact by construction** — integer comparisons of `u8` intensities packed into a `u32`. No float anywhere. |
//! | cost volume + depth selection | [`sweep::sweep_view`] | `plane_sweep.wgsl` | bit-exact **when the adapter does not contract**; measured, not assumed. See below. |
//! | TSDF integration | [`tsdf::integrate_view`] | `tsdf.wgsl` | same story as the sweep: the arithmetic is a fixed `f32` sequence written twice. |
//!
//! The costs themselves are **integers** — a census Hamming distance — which is
//! the single most load-bearing choice in this crate. It means the cost volume,
//! the per-neighbour aggregation, the `argmin` and the tie-breaks are exact on
//! both sides, and the only floating point the sweep does at all is the
//! projection of a pixel through a plane and the sub-plane parabola. A ZNCC cost
//! would have put a windowed float reduction inside the innermost loop, where a
//! single re-association is a different answer and there is nothing left to
//! anchor it to.
//!
//! ## The op that can break the parity, named
//!
//! `q = d * rray + t` (and, in the distortion polynomial, `1 + k1 r² + k2 r⁴`)
//! is a multiply-add. WGSL is written without `fma()` here and naga emits
//! separate `OpFMul`/`OpFAdd`, but a driver's back-end compiler is permitted to
//! **contract** them into a fused multiply-add, which is a *different* `f32`
//! result by up to one ULP. One ULP in `u` matters only when `floor(u + 0.5)`
//! is within one ULP of an integer, and then it is a whole different census
//! fetch. That is the entire parity risk, it is one named operation, and
//! `tests/mvs_gate.rs` measures it rather than asserting a hope: the arm reports
//! the fraction of pixels whose depth is not bit-identical and holds it to a
//! bound it also proves is not vacuous.
//!
//! # The `f64` -> `f32` seam, stated once
//!
//! World geometry is `f64` (architecture rule 3) and WGSL has no `f64`. The
//! narrowing therefore happens at **exactly two places**, both of them named:
//!
//! 1. [`sweep::SweepGeometry::build`] — per reference pixel and per neighbour
//!    view, the rotated undistorted ray `R_rel · (xₙ, yₙ, 1)` and the relative
//!    translation `t_rel`, computed in `f64` from the reconstruction's poses and
//!    narrowed once. Undistortion (a fixed-point iteration) never runs on the
//!    GPU: it is folded into the ray here.
//! 2. [`tsdf::TsdfGrid::new`] — the grid's world origin stays `f64`; every voxel
//!    centre is an `f32` offset *from that origin*, which is the engine's
//!    floating-origin doctrine applied to a scan volume.
//!
//! Everything downstream of those two points is `f32` on both sides, so "the CPU
//! reference" and "the shader" are comparable at all.
//!
//! # Determinism
//!
//! The [`inf_photo`] rules carry over unchanged and are structural here too:
//!
//! * Parallel stages go through [`inf_core::job::JobPool::parallel_map`] /
//!   `parallel_map_ref`, which are in-order pure maps. No float sum is
//!   accumulated in parallel.
//! * TSDF fusion is **serial across views** and parallel only *within* one view,
//!   and even there it is a **gather**: a voxel reads the depth map, the depth
//!   map never scatters into voxels. There is no atomic float add anywhere,
//!   because `atomicAdd` over floats is exactly the order-dependent construct
//!   the house rules exist to keep out of committed content. What that costs is
//!   measured in [`tsdf`]'s module docs.
//! * Every collection that reaches a result is a `Vec` in a fixed index order or
//!   a `BTreeMap`. There is no `HashMap` on any path that reaches
//!   [`DenseReconstruction`].
//! * [`DenseReconstruction::canonical_bytes`] is the assertion surface, the same
//!   pattern (and the same lesson — write *every* field) as
//!   [`inf_photo::Reconstruction::canonical_bytes`].
//!
//! # Transcendentals
//!
//! The P14 law: `std` trigonometry is not bit-portable, so anything that lands
//! in committed bytes uses [`inf_math`](https://docs.rs/)'s portable family.
//! P25.2's outputs are in-memory, exactly as P25.1's were, so the exemption
//! carries — and in practice this crate calls **no** transcendental at all. The
//! only non-arithmetic function on any path is `sqrt`, which is IEEE-754
//! correctly rounded and identical everywhere. The parabola, the projection, the
//! distortion polynomial and the TSDF blend are `+ - * /`.
//!
//! **The day a dense reconstruction is serialized** — P25.3's bakes, P25.4's
//! import — the portable family applies to everything that touches those bytes,
//! and this paragraph is the notice that it was a line rather than an oversight.
//!
//! # Units
//!
//! # One lint, allowed once, for one reason
//!
//! `clippy::neg_cmp_op_on_partial_ord` is allowed crate-wide. Every guard in the
//! numeric path here is written `!(x > 0.0)` rather than `x <= 0.0`, and the
//! difference is **not** style: a `NaN` compares false against everything, so
//! `!(x > 0.0)` rejects it and `x <= 0.0` accepts it. Both languages this crate
//! is written in agree on that, which is what lets the CPU reference and the
//! WGSL mirror share a guard. Rewriting these to read more smoothly would change
//! what they do, in the one direction that produces an out-of-bounds index from
//! a `NaN` depth.
//!
//! # Units
//!
//! [`inf_photo::Reconstruction`] is in **baseline units**, not metres (structure
//! from motion is scale-ambiguous and the gauge is the first camera pair). This
//! crate inherits that: depths, the voxel size and the mesh are all in the
//! reconstruction's own unit. The metric scale arrives with P25.4's scale step,
//! as one `metres_per_unit` multiplier over the whole thing. Anything here whose
//! unit is genuinely physical says so; **pixels** are pixels.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod dense;
pub mod filter;
pub mod fixture;
pub mod gpu;
pub mod sweep;
pub mod tsdf;

pub use dense::{
    reconstruct_dense, reconstruct_dense_with, CpuBackend, DenseBackend, DenseConfig,
    DenseReconstruction, DenseReport,
};
pub use filter::FilterConfig;
pub use gpu::{DenseGpu, GpuBackend};
pub use sweep::{CensusConfig, DepthMap, SweepConfig, SweepGeometry};
pub use tsdf::{DenseMesh, SurfaceHints, TsdfConfig, TsdfGrid};

use thiserror::Error;

/// Why a dense reconstruction **refused**.
///
/// Every variant names the shortfall and the bound it fell under. A dense stage
/// that quietly returns an empty mesh is worse than one that stops, because an
/// empty mesh is exactly what a *successful* reconstruction of a featureless
/// wall also looks like.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DenseError {
    /// A reference view could not find enough covisible neighbours to sweep
    /// against.
    ///
    /// A plane sweep with one neighbour has no way to tell a true match from a
    /// repeated texture, and no left-right check to run afterwards. This is the
    /// refusal a capture that orbited too far between shots earns.
    #[error(
        "view {view} has {found} covisible neighbour views, need {required} — the photographs \
         around it may not overlap enough to sweep against"
    )]
    TooFewNeighbours {
        /// The reference view.
        view: u32,
        /// How many neighbours shared at least one track with it.
        found: usize,
        /// [`SweepConfig::min_neighbours`].
        required: usize,
    },
    /// A reference view saw too few reconstructed points to derive a depth
    /// range from.
    ///
    /// The sweep's near and far planes are percentiles of the sparse cloud's own
    /// extent *in that view*; with a handful of points those percentiles are
    /// noise, and a plane sweep over the wrong range finds nothing anywhere.
    #[error(
        "view {view} sees {found} reconstructed points, need {required} to derive a depth range"
    )]
    TooFewSparsePoints {
        /// The reference view.
        view: u32,
        /// How many reconstruction points project into it.
        found: usize,
        /// [`SweepConfig::min_range_points`].
        required: usize,
    },
    /// Fewer registered cameras than a dense stage can work with.
    #[error("dense reconstruction needs at least {required} registered cameras, {given} given")]
    TooFewCameras {
        /// How many the reconstruction registered.
        given: usize,
        /// The minimum.
        required: usize,
    },
    /// The caller's view list does not line up with the reconstruction.
    #[error("registered camera {view} has no view in the {given} views supplied")]
    MissingView {
        /// The camera's view index.
        view: u32,
        /// How many views the caller passed.
        given: usize,
    },
    /// A view's image does not match the intrinsics the reconstruction carries
    /// for it.
    #[error("view {view} is {img_w}x{img_h} but its camera declares {int_w}x{int_h}")]
    IntrinsicsMismatch {
        /// The offending view.
        view: u32,
        /// Image width.
        img_w: u32,
        /// Image height.
        img_h: u32,
        /// Declared width.
        int_w: u32,
        /// Declared height.
        int_h: u32,
    },
    /// No depth survived filtering anywhere, so there is nothing to fuse.
    #[error(
        "every depth map is empty after filtering ({swept} views swept) — the photographs may \
         have no overlapping texture"
    )]
    NoDepthSurvived {
        /// How many views were swept.
        swept: usize,
    },
    /// Fusion produced no surface.
    #[error("TSDF fusion over {samples} depth samples produced no surface voxels")]
    EmptyFusion {
        /// How many depth samples were integrated.
        samples: usize,
    },
    /// A GPU stage was asked for and the adapter refused.
    ///
    /// Carried as a string because that is what
    /// [`inf_render::GpuContext::headless`] reports and re-wording an adapter's
    /// own complaint helps nobody.
    #[error("GPU dense stage unavailable: {message}")]
    NoGpu {
        /// The adapter door's own message.
        message: String,
    },
}

/// A non-fatal finding about a dense reconstruction: something the caller (and
/// P25.4's diagnostics panel) should be told, that did not stop the solve.
///
/// The list is ordered and canonical — it is part of
/// [`DenseReconstruction::canonical_bytes`], so an advisory appearing or
/// disappearing is a visible change rather than a log line nobody reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseAdvisory {
    /// A view was swept against fewer neighbours than were asked for.
    ThinNeighbourhood {
        /// The reference view.
        view: u32,
        /// How many neighbours it got.
        used: usize,
        /// How many were requested.
        wanted: usize,
    },
    /// A view's depth map came back sparse. Its share of the fused surface is
    /// small, and a hole in the mesh probably belongs to it.
    SparseDepthMap {
        /// The reference view.
        view: u32,
        /// Valid pixels after filtering.
        valid: usize,
        /// Pixels in the image.
        total: usize,
    },
    /// The left-right consistency check rejected a large share of a view's
    /// depths. Usually an occlusion boundary or a pose that is slightly out.
    ConsistencyRejected {
        /// The reference view.
        view: u32,
        /// How many pixels the check dropped.
        dropped: usize,
    },
    /// Speckle removal dropped isolated regions.
    SpeckleRemoved {
        /// The reference view.
        view: u32,
        /// How many pixels went.
        dropped: usize,
        /// How many separate regions.
        regions: usize,
    },
    /// The voxel size hit the grid-side cap rather than the pixel-footprint
    /// formula, so the fusion is coarser than the photographs support.
    VoxelSizeClamped {
        /// What the pixel-footprint formula asked for, in reconstruction units,
        /// as `to_bits` so the advisory has a total order.
        wanted_bits: u64,
        /// What the cap allowed, same encoding.
        used_bits: u64,
    },
}

impl DenseAdvisory {
    /// A total-order key. Hand-written for the same reason
    /// [`inf_photo::Advisory`]'s is: one variant carries floats, and `f64` is
    /// neither `Eq` nor `Ord`. They are already stored as `to_bits`, which makes
    /// the derive possible but not the *ordering*, so the key is explicit.
    fn sort_key(&self) -> (u8, u64, u64, u64) {
        match self {
            DenseAdvisory::ThinNeighbourhood { view, used, wanted } => {
                (0, *view as u64, *used as u64, *wanted as u64)
            }
            DenseAdvisory::SparseDepthMap { view, valid, total } => {
                (1, *view as u64, *valid as u64, *total as u64)
            }
            DenseAdvisory::ConsistencyRejected { view, dropped } => {
                (2, *view as u64, *dropped as u64, 0)
            }
            DenseAdvisory::SpeckleRemoved {
                view,
                dropped,
                regions,
            } => (3, *view as u64, *dropped as u64, *regions as u64),
            DenseAdvisory::VoxelSizeClamped {
                wanted_bits,
                used_bits,
            } => (4, *wanted_bits, *used_bits, 0),
        }
    }
}

impl PartialOrd for DenseAdvisory {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DenseAdvisory {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl std::fmt::Display for DenseAdvisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseAdvisory::ThinNeighbourhood { view, used, wanted } => write!(
                f,
                "view {view} was swept against {used} neighbour views, not the {wanted} asked \
                 for; its depths are the least redundant in the reconstruction"
            ),
            DenseAdvisory::SparseDepthMap { view, valid, total } => write!(
                f,
                "view {view} kept {valid} of {total} pixels after filtering; holes in the fused \
                 mesh near it are expected"
            ),
            DenseAdvisory::ConsistencyRejected { view, dropped } => write!(
                f,
                "view {view} lost {dropped} pixels to the left-right consistency check"
            ),
            DenseAdvisory::SpeckleRemoved {
                view,
                dropped,
                regions,
            } => write!(
                f,
                "view {view} lost {dropped} pixels in {regions} isolated regions to speckle removal"
            ),
            DenseAdvisory::VoxelSizeClamped {
                wanted_bits,
                used_bits,
            } => write!(
                f,
                "the pixel footprint asked for a {} unit voxel and the grid-side cap allowed {}; \
                 the fusion is coarser than the photographs support",
                f64::from_bits(*wanted_bits),
                f64::from_bits(*used_bits)
            ),
        }
    }
}
