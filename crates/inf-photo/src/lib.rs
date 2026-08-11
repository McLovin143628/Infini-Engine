//! **Photogrammetry: the structure-from-motion core** (P25.1) — photographs in,
//! camera poses and a sparse point cloud out, with nothing outside this
//! workspace doing the reconstructing.
//!
//! # The stance, stated once
//!
//! Decided 2026-07-31 (ROADMAP §12, Phase 25): **native classical SfM + GPU
//! MVS**, in Rust and WGSL. No COLMAP, no OpenMVG, no OpenCV — not because they
//! are bad (they are excellent) but because a reconstruction that runs out of
//! process is a reconstruction the engine cannot make deterministic, cannot
//! ship, and cannot put behind a wizard. Everything here is owned math with
//! tests against it.
//!
//! There is deliberately **no `wgpu` dependency**: dense stereo arrives with
//! P25.2, and the sparse core has no business linking a graphics API.
//!
//! # What is here so far
//!
//! The **front end** and the linear algebra it stands on:
//!
//! * [`linalg`] — symmetric Jacobi eigen, a one-sided-Jacobi 3x3 SVD, `LDLᵀ`.
//!   Four small routines, hand-rolled, in place of a linear-algebra dependency.
//! * [`hash`] — the counter-based mixer that is this crate's only source of
//!   pseudo-randomness.
//! * [`camera`] — pinhole plus radial `k1`/`k2`, `f64`, with the analytic
//!   projection Jacobian a bundle adjuster needs.
//! * [`gray`] — 8-bit images and the scale pyramid the detector runs on.
//! * [`features`] — multi-scale FAST-9 ranked by Harris, oriented by intensity
//!   centroid, described by a 256-bit BRIEF-class descriptor over a
//!   *compile-time constant* sampling pattern. ORB-class.
//! * [`matching`] — brute-force Hamming matching with Lowe's ratio test and a
//!   mutual-best check, then tracks across views by union-find.
//!
//! The geometry (essential matrix, triangulation, PnP), bundle adjustment and
//! the incremental loop that drives them land in the next commit.
//!
//! # Determinism is the headline requirement, not a nicety
//!
//! A reconstruction is a **pure function of its input**. That is enforced
//! structurally, not hoped for:
//!
//! * Every parallel stage goes through [`inf_core::job::JobPool::parallel_map`]
//!   / `parallel_map_ref`, which are in-order pure maps. **No floating-point sum
//!   is ever accumulated in parallel** — parallel work produces per-item results
//!   which a serial loop then folds in index order.
//! * There is no stateful RNG. Every pseudo-random draw is a pure function of
//!   the integers folded into [`hash::Hash64`].
//! * Every collection that feeds a result is a `BTreeMap` or a vector sorted by
//!   a total order with integer tie-breaks. There is no `HashMap` anywhere.
//! * Every float comparison used for sorting goes through [`f64::total_cmp`].
//!
//! # Transcendentals, and the line this crate sits on
//!
//! The P14 law is that `f32`/`f64` `std` trigonometry is **not bit-portable
//! across platforms**, so anything that lands in *committed bytes* must use
//! [`inf_math`]'s portable family. This crate's outputs are intermediate
//! in-memory results and test data; nothing here is serialized into an asset.
//! `std` transcendentals are therefore permitted **inside the solvers**, and in
//! practice almost none are used: `sqrt` is IEEE-754 correctly rounded and
//! identical everywhere, and feature orientation carries `(cos, sin)` **directly
//! from the intensity centroid** as a normalized 2-vector, so the detector calls
//! no trigonometry at all.
//!
//! **The day a reconstruction is serialized into an asset** (P25.3 bakes, P25.4
//! imports) the portable family applies to everything that touches those bytes,
//! and this paragraph is the notice that it was a deliberate line rather than an
//! oversight.
//!
//! # Units
//!
//! [`camera::Intrinsics`] is in **pixels**; `k1`/`k2` are dimensionless.
//! Residuals and thresholds are in **pixels**. Structure from motion is
//! scale-ambiguous, so reconstruction coordinates will be **baseline units, not
//! metres** — the gauge ruling lands with the pipeline.

pub mod camera;
pub mod features;
pub mod gray;
pub mod hash;
pub mod linalg;
pub mod matching;

pub use camera::{Intrinsics, Pose};
pub use features::{detect_and_describe, Feature, FeatureConfig, FeatureSet};
pub use gray::{GrayImage, Pyramid};
pub use matching::{Match, MatchConfig, Observation, PairMatches, Track};

use thiserror::Error;

/// Everything that can go wrong loading or decoding an input photograph.
#[derive(Debug, Error)]
pub enum PhotoError {
    /// The file could not be read.
    #[error("photo read failed for {path}: {source}")]
    Read {
        /// The path that failed.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The bytes could not be decoded as an image.
    #[error("photo decode failed for {path}: {message}")]
    Decode {
        /// The path that failed.
        path: String,
        /// The decoder's complaint.
        message: String,
    },
    /// A pixel buffer's length did not match its declared dimensions.
    #[error("image is {width}x{height} = {expected} pixels but {given} bytes were given")]
    PixelCount {
        /// Declared width.
        width: u32,
        /// Declared height.
        height: u32,
        /// `width * height`.
        expected: usize,
        /// The buffer length actually handed over.
        given: usize,
    },
    /// A zero-width or zero-height image.
    #[error("image dimensions must both be non-zero, got {width}x{height}")]
    EmptyImage {
        /// Declared width.
        width: u32,
        /// Declared height.
        height: u32,
    },
}

/// A non-fatal observation about a reconstruction: something the caller (and,
/// in P25.4, the wizard's diagnostics panel) should be told, that did not stop
/// the solve.
///
/// The list is ordered and canonical, and lands inside the reconstruction's
/// byte image — so an advisory appearing or disappearing is a visible change,
/// not a log line nobody reads.
#[derive(Debug, Clone)]
pub enum Advisory {
    /// Too many views for exhaustive pairing, so a sliding window was used.
    /// Recall drops for photographs that overlap non-sequentially.
    WindowedPairing {
        /// How many views were supplied.
        views: usize,
        /// The cap that was exceeded.
        cap: usize,
        /// The half-window each view was matched against.
        window: usize,
    },
    /// A view never got a pose: too few 2D–3D correspondences, or PnP failed.
    ViewNotRegistered {
        /// The view index.
        view: u32,
        /// How many 2D–3D correspondences it had when it was last considered.
        correspondences: usize,
    },
    /// A track saw the same view twice (a matching inconsistency). Those
    /// observations were dropped from the track; the rest of it survived.
    ConflictingTrackViews {
        /// How many tracks were affected.
        tracks: usize,
        /// How many observations were dropped in total.
        dropped: usize,
    },
    /// Observations pruned for reprojection error after the final bundle.
    PrunedObservations {
        /// How many were dropped.
        dropped: usize,
        /// The threshold in **pixels**.
        threshold_px: f64,
    },
    /// A view registered on fewer correspondences than is comfortable. Its pose
    /// is the least trustworthy in the reconstruction.
    ThinRegistration {
        /// The view index.
        view: u32,
        /// How many PnP inliers it registered on.
        inliers: usize,
    },
}

impl Advisory {
    /// A total-order key.
    ///
    /// Hand-written rather than derived because [`Advisory::PrunedObservations`]
    /// carries an `f64` threshold, and `f64` is neither `Eq` nor `Ord`. Folding
    /// it in as `to_bits` gives a genuine total order (advisory thresholds are
    /// never `NaN`, and a `NaN` would sort consistently rather than making
    /// `sort` misbehave), which is what lets the advisory list be sorted and
    /// deduplicated deterministically before it lands in
    /// [`Reconstruction::canonical_bytes`].
    fn sort_key(&self) -> (u8, u64, u64, u64) {
        match self {
            Advisory::WindowedPairing { views, cap, window } => {
                (0, *views as u64, *cap as u64, *window as u64)
            }
            Advisory::ViewNotRegistered {
                view,
                correspondences,
            } => (1, *view as u64, *correspondences as u64, 0),
            Advisory::ConflictingTrackViews { tracks, dropped } => {
                (2, *tracks as u64, *dropped as u64, 0)
            }
            Advisory::PrunedObservations {
                dropped,
                threshold_px,
            } => (3, *dropped as u64, threshold_px.to_bits(), 0),
            Advisory::ThinRegistration { view, inliers } => (4, *view as u64, *inliers as u64, 0),
        }
    }
}

impl PartialEq for Advisory {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for Advisory {}

impl PartialOrd for Advisory {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Advisory {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl std::fmt::Display for Advisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Advisory::WindowedPairing { views, cap, window } => write!(
                f,
                "{views} views exceeds the exhaustive-pairing cap of {cap}; matched each view \
                 against its {window} neighbours instead — non-sequential overlap may be missed"
            ),
            Advisory::ViewNotRegistered {
                view,
                correspondences,
            } => write!(
                f,
                "view {view} was not registered ({correspondences} 2D-3D correspondences); it \
                 contributes no pose and no points"
            ),
            Advisory::ConflictingTrackViews { tracks, dropped } => write!(
                f,
                "{tracks} tracks saw one view twice; {dropped} observations were dropped from them"
            ),
            Advisory::PrunedObservations {
                dropped,
                threshold_px,
            } => write!(
                f,
                "{dropped} observations pruned above {threshold_px} px reprojection error"
            ),
            Advisory::ThinRegistration { view, inliers } => write!(
                f,
                "view {view} registered on only {inliers} PnP inliers; its pose is the weakest in \
                 the reconstruction"
            ),
        }
    }
}
