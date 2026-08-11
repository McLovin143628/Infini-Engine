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
//! This crate is the *first* half of that promise and only the first half: it
//! produces **poses + a sparse cloud**. Dense depth (plane-sweep MVS in WGSL),
//! fusion, retopology, bake and the capture wizard are P25.2–P25.4 and are not
//! here. There is deliberately **no `wgpu` dependency** in this batch.
//!
//! # The pipeline
//!
//! [`reconstruct`] runs, in order:
//!
//! 1. **[`features`]** — a multi-scale FAST-9 detector ranked by Harris
//!    response, oriented by intensity centroid, described by a 256-bit
//!    BRIEF-class binary descriptor over a *compile-time constant* sampling
//!    pattern. ORB-class. Its bounds are documented on the module.
//! 2. **[`matching`]** — brute-force Hamming matching with Lowe's ratio test and
//!    a mutual-best check, then tracks across views by union-find.
//! 3. **[`geometry`]** — a normalized eight-point essential matrix under
//!    seeded RANSAC for the initial pair, cheirality-disambiguated relative
//!    pose, multi-view DLT triangulation, and DLT-PnP under RANSAC for each
//!    subsequent view.
//! 4. **[`bundle`]** — Levenberg–Marquardt bundle adjustment with the Schur
//!    complement over 6-DOF camera blocks and 3-DOF point blocks, robustified by
//!    a Huber kernel.
//!
//! [`synth`] is the known-truth dataset the gates run on: an analytic scene and
//! a small deterministic software renderer, so the whole pipeline can be
//! measured against poses that are *given*, not estimated, with no GPU and no
//! committed image fixtures.
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
//! * RANSAC uses a **counter-based** hash ([`hash::Hash64`]), never a stateful
//!   RNG: the sample drawn by iteration `i` is a pure function of
//!   `(seed, stage, iteration)`. Iteration counts are **fixed**; there is no
//!   early exit on an inlier ratio and none on elapsed time.
//! * Every collection that feeds a result is a `BTreeMap` or a vector sorted by
//!   a total order with integer tie-breaks. There is no `HashMap` on any path
//!   that reaches [`Reconstruction`].
//! * Every float comparison used for sorting goes through [`f64::total_cmp`].
//!
//! [`Reconstruction::canonical_bytes`] is the assertion surface: two runs, and
//! runs on pools of one / two / eight workers, produce **byte-identical** output.
//! `crates/inf-photo/tests/sfm_gate.rs` gates exactly that.
//!
//! # Transcendentals, and the line this crate sits on
//!
//! The P14 law is that `f32`/`f64` `std` trigonometry is **not bit-portable
//! across platforms**, so anything that lands in *committed bytes* must use
//! [`inf_math`]'s portable family. P25.1's outputs — poses, points, tracks — are
//! intermediate in-memory results and test data; nothing here is serialized into
//! an asset. `std` transcendentals are therefore permitted **inside the
//! solvers**, and in practice almost none are used:
//!
//! * `sqrt` is IEEE-754 correctly rounded and identical everywhere. It is used
//!   freely.
//! * Feature orientation carries `(cos, sin)` **directly from the intensity
//!   centroid** (a normalized 2-vector), so the detector calls no trig at all.
//! * The bundle adjuster's rotation retraction is the **normalized-quaternion**
//!   one, `q <- normalize((1, w/2)) * q`, not the exponential map, so it calls no
//!   trig either.
//! * The only transcendental in the crate is [`inf_math::patan2_64`] in
//!   [`align::rotation_angle_deg`] — the *portable* one — and that is a
//!   reporting metric.
//!
//! **The day a reconstruction is serialized into an asset** (P25.3 bakes, P25.4
//! imports) the portable family applies to everything that touches those bytes,
//! and this paragraph is the notice that it was a deliberate line rather than an
//! oversight.
//!
//! # Gauge, and where SI comes back
//!
//! Structure from motion is **scale-ambiguous**: photographs of a doll's house
//! and of a house are the same photographs. The reconstruction therefore fixes
//! its gauge by convention:
//!
//! * The **world frame is the first registered camera** — its pose is exactly
//!   the identity and it is held fixed through every bundle adjustment (it
//!   contributes residuals, never parameters).
//! * The **scale is one unit of baseline to the second registered camera**:
//!   `|C1 - C0| == 1` exactly, re-imposed after each solve. Scale is an exact
//!   null direction of reprojection cost (scaling all points and all camera
//!   centres about the origin leaves every projection untouched), so pinning it
//!   costs the fit nothing.
//!
//! Consequently **[`Reconstruction`] positions are NOT metres.** They are
//! baseline units. Architecture rule 6 (1 world unit = 1 metre, SI everywhere)
//! is not violated: this type never reaches world space. The metric scale is
//! recovered by the capture wizard's scale step (P25.4) — a known distance, a
//! marker, or camera EXIF — which multiplies the whole reconstruction by one
//! `metres_per_unit` factor. Any field here whose unit *is* physical says so on
//! its doc comment ([`camera::Intrinsics`] is in **pixels**; residuals and
//! thresholds are in **pixels**).
//!
//! # What this is not
//!
//! Honest bounds, so nobody discovers them by being surprised:
//!
//! * **Intrinsics are an input and are held fixed.** Nothing here does
//!   self-calibration; `fx, fy, cx, cy, k1, k2` come from the caller and come out
//!   unchanged. Refining them is a bundle-parameter block that does not exist yet.
//! * **The initial pair uses the eight-point algorithm**, not the five-point.
//!   Eight-point needs more correspondences, is more noise-sensitive, and
//!   **degenerates on a purely planar scene**; five-point does not. It is a
//!   fraction of the code, and the incremental loop plus bundle adjustment
//!   absorbs its extra noise. Stated, measured, and the upgrade path if a planar
//!   capture ever needs it.
//! * **PnP is DLT-based** (linear, six correspondences minimum) rather than P3P,
//!   and inherits the same planar degeneracy. It is always followed by a
//!   nonlinear pose refinement.
//! * **Pair selection is exhaustive** below [`matching::MatchConfig::exhaustive_view_cap`]
//!   views and falls back to a sliding window above it, with an advisory. There
//!   is no vocabulary tree and no image retrieval.
//! * **Descriptors are BRIEF-class**: rotation-invariant by construction, and
//!   invariant to scale only across the pyramid's span. Strong affine or
//!   perspective foreshortening between two photographs will lose matches.

pub mod align;
pub mod bundle;
pub mod camera;
pub mod features;
pub mod geometry;
pub mod gray;
pub mod hash;
pub mod linalg;
pub mod matching;
pub mod sfm;
pub mod synth;

pub use align::{rotation_angle_deg, similarity_from_points, Similarity};
pub use bundle::{bundle_adjust, BundleConfig, BundleObservation, BundleReport};
pub use camera::{Intrinsics, Pose};
pub use features::{detect_and_describe, Feature, FeatureConfig, FeatureSet};
pub use gray::{GrayImage, Pyramid};
pub use matching::{Match, MatchConfig, Observation, PairMatches, Track};
pub use sfm::{
    reconstruct, Reconstruction, RegisteredCamera, ScenePoint, SfmConfig, SfmReport, View,
};

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

/// Why a reconstruction **refused**. Every variant names the shortfall and the
/// bound it fell under, because a reconstruction that quietly returns garbage is
/// worse than one that stops.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SfmError {
    /// Fewer photographs than [`SfmConfig::min_views`] were handed in.
    ///
    /// Two views *are* geometrically solvable — that is a stereo pair, not a
    /// photogrammetric capture: one baseline gives no redundancy, no way to
    /// detect a bad pose, and a dense stage nothing to cross-check. The pipeline
    /// refuses rather than returning a result the rest of Phase 25 cannot use.
    #[error("photogrammetry needs at least {required} views, {given} given")]
    TooFewViews {
        /// How many views the caller supplied.
        given: usize,
        /// [`SfmConfig::min_views`].
        required: usize,
    },
    /// An input view's declared intrinsics do not match its image.
    #[error("view {view} has {img_w}x{img_h} pixels but intrinsics declare {int_w}x{int_h}")]
    IntrinsicsMismatch {
        /// The offending view index.
        view: usize,
        /// Image width.
        img_w: u32,
        /// Image height.
        img_h: u32,
        /// Declared intrinsics width.
        int_w: u32,
        /// Declared intrinsics height.
        int_h: u32,
    },
    /// No pair of views survived essential-matrix RANSAC with enough inliers and
    /// enough parallax to start from.
    #[error(
        "no viable initial pair: {examined} candidate pairs examined, best had {best_inliers} \
         inliers (need {required}) — the photographs may not overlap, or may all be taken from \
         one position"
    )]
    NoViableInitialPair {
        /// How many candidate pairs were scored.
        examined: usize,
        /// The best inlier count seen.
        best_inliers: usize,
        /// The minimum required.
        required: usize,
    },
    /// The initial pair produced no usable triangulated points.
    #[error("initial pair ({a}, {b}) triangulated {points} points, need {required}")]
    InitializationFailed {
        /// First view of the pair.
        a: u32,
        /// Second view of the pair.
        b: u32,
        /// How many points survived cheirality and parallax.
        points: usize,
        /// The minimum required.
        required: usize,
    },
    /// The incremental loop registered fewer views than the minimum.
    #[error("only {registered} of {given} views registered, need {required}")]
    TooFewRegistered {
        /// How many views got a pose.
        registered: usize,
        /// How many were supplied.
        given: usize,
        /// [`SfmConfig::min_views`].
        required: usize,
    },
}

/// A non-fatal observation about a reconstruction: something the caller (and,
/// in P25.4, the wizard's diagnostics panel) should be told, that did not stop
/// the solve.
///
/// The list is ordered and canonical — it is part of
/// [`Reconstruction::canonical_bytes`], so an advisory appearing or disappearing
/// is a visible change, not a log line nobody reads.
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
    /// Observations pruned for reprojection error, summed over **every** prune
    /// pass the solve made — the incremental ones after each intermediate
    /// bundle as well as the final one.
    PrunedObservations {
        /// How many were dropped, over the whole solve.
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
                "{dropped} observations pruned above {threshold_px} px reprojection error over the \
                 whole solve"
            ),
            Advisory::ThinRegistration { view, inliers } => write!(
                f,
                "view {view} registered on only {inliers} PnP inliers; its pose is the weakest in \
                 the reconstruction"
            ),
        }
    }
}
