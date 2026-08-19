//! **The capture wizard's Ring-1 half** (P25.4): photographs on disk in, a
//! finished asset in a project out, with progress and diagnostics all the way.
//!
//! The pipeline itself is already built and is not re-implemented here. This
//! module is the *session* that drives it:
//!
//! ```text
//!   Load ──▶ Sfm ──────▶ Dense ─────────▶ Finish ──────────▶ Write
//!   files    inf_photo   inf_photo_gpu    crate::photo-       AssetProject
//!   decoded  ::reconstr  ::reconstruct_   grammetry::finish_  (five assets
//!   + gray   uct         dense            reconstruction      or none)
//! ```
//!
//! # Progress, and exactly how fine it is
//!
//! The four automatic stages run on **one worker thread** (the P4 `ImportQueue`
//! shape) and report on a channel Ring 2 drains onto `photogrammetry://progress`.
//! Granularity differs by stage, and the difference is a fact about who owns the
//! code rather than a choice:
//!
//! * **Load** reports per photograph — this module reads them.
//! * **Sfm** and **Dense** report *start* and *finish* only. Their entry points
//!   ([`inf_photo::reconstruct`], [`inf_photo_gpu::reconstruct_dense`]) are
//!   single blocking calls in Ring-0 crates, and threading a callback through
//!   two crates the shipped player links is not this batch's business.
//! * **Finish** reports each of [`FinishStep::ALL`], because *that* orchestrator
//!   is ours — [`finish_reconstruction_with_progress`] is one function with a
//!   callback, deliberately not a second copy of the pipeline.
//! * **Write** is not automatic at all: it runs when the user presses Import,
//!   synchronously, on the command thread, and emits the same two events so the
//!   stage sequence a caller sees is the whole five.
//!
//! # Cancellation is BETWEEN stages, and that is stated rather than dressed up
//!
//! [`PhotogrammetrySession::cancel`] sets a flag the worker reads **between**
//! stages. There is no mid-stage cancellation because there is no mid-stage: a
//! stage is one blocking call into a pool-parallel kernel, so pressing Cancel
//! during a four-minute dense solve stops the run when that solve returns, not
//! when the button is pressed. What the guarantee *is* worth is the thing that
//! matters: a cancelled run has written **nothing**, because writing is a
//! separate stage the user starts.
//!
//! `Finish` is the last **automatic** stage, so a Cancel arriving while it runs
//! finds no stage left to skip: the run completes and settles
//! [`CaptureState::Ready`] with its product in hand. Same for a
//! [`refinish`](PhotogrammetrySession::refinish), which is that stage alone.
//! The guarantee is unchanged — `Write` is the user's — and throwing away
//! minutes of finished work to honour a button that arrived at the end would be
//! the expensive kind of literalism. It is written here, gated in
//! `capture_wizard_gate`, and said in the panel, because a Cancel whose effect
//! depends on when it lands has to say so.
//!
//! # The worker is the only thing that can end a run, so it may not die quietly
//!
//! Every terminal state is published by the worker, which means a worker that
//! **panics** would leave the session `Running` for the life of the process,
//! with Cancel answering `true` to nobody and Start refusing `Busy` for ever.
//! [`run_guarded`] catches the unwind and settles the run `Failed` carrying the
//! panic's own words, so a bug in a solver reaches a user in the same shape a
//! refusal does. The doors that do not go through the worker
//! ([`PhotogrammetrySession::load_photos`],
//! [`PhotogrammetrySession::reset`]) refuse or wait rather than reaching around
//! a run in flight, for the same reason: two things that each believe they own
//! the session is how a Cancel button stops working.
//!
//! # Diagnostics are the second half of the spec, not a log
//!
//! Everything that can be said about a capture is a typed [`CaptureIssue`] with
//! a severity, a stage and a `Display` that carries a **remedy**. Pre-flight
//! catches what a run should not be started over (unreadable files by name, too
//! few photographs, a scale that is not a scale); the solve stages add the
//! Ring-0 advisories (a view that never registered, named with its file and its
//! correspondence count); and the finish adds [`FinishAdvisory`] and
//! [`CoverageReport`] — "what each camera saw", measured through the pipeline's
//! own [`view_sees`], which is why that function is public.
//!
//! # Units
//!
//! A reconstruction is in **baseline units** (structure from motion is
//! scale-ambiguous). [`FinishConfig::metres_per_unit`] is the scale step and the
//! wizard exposes it; [`CaptureProduct::extent_units`] is what a user measures
//! it against, and [`scale_for_longest_side`] is the arithmetic that turns a
//! known real-world length into that multiplier.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use glam::DVec3;
use inf_core::job::JobPool;
use inf_mesh::{MeshAsset, MeshVertex};
use inf_photo::{
    reconstruct, Advisory, GrayImage, Intrinsics, Reconstruction, RgbImage, SfmConfig, View,
};
use inf_photo_gpu::{reconstruct_dense, AlbedoView, DenseConfig, DenseReconstruction, DenseReport};

use crate::assets::terrain_import::CancelToken;
use crate::assets::AssetProject;
use crate::photogrammetry::{
    finish_reconstruction_with_progress, view_sees, write_finished, FinishAdvisory, FinishConfig,
    FinishStep, FinishedAsset, FinishedIds,
};

/// Where a capture's assets land, relative to the content root.
///
/// A constant rather than a frontend string for the reason
/// [`crate::character::CHARACTER_FOLDER`] is one: a second spelling in the panel
/// goes stale silently.
pub const SCAN_FOLDER: &str = "Scans";

/// The assumed camera a photograph is reconstructed with when nothing calibrated
/// it.
///
/// Structure from motion refines poses and **never** intrinsics
/// ([`inf_photo::View::intrinsics`] is an input), so these three numbers are the
/// one thing the wizard cannot recover and must be told. They are expressed as a
/// *ratio* of the image's longer side rather than in pixels so one setting
/// covers a whole shoot at one resolution and reads the same at any.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssumedCamera {
    /// Focal length as a fraction of `max(width, height)`.
    ///
    /// `1.2` is a phone-camera-shaped default (a ~26 mm-equivalent lens on a 3:2
    /// frame lands near it) and is honest rather than right: it is a guess, and
    /// the panel says so.
    pub focal_ratio: f64,
    /// First radial-distortion coefficient. Zero means "a pinhole", which is
    /// what an uncalibrated capture gets.
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
}

impl Default for AssumedCamera {
    fn default() -> Self {
        Self {
            focal_ratio: 1.2,
            k1: 0.0,
            k2: 0.0,
        }
    }
}

impl AssumedCamera {
    /// The smallest focal ratio that describes a lens rather than a singularity.
    ///
    /// `fx = fy = focal_ratio * max(w, h)`, and `to_normalized` divides by it —
    /// so a zero gives `0.0 / 0.0` = **NaN** at the principal point and ±inf
    /// everywhere else, and `undistort`'s `if f.abs() < 1e-12` is NaN-blind.
    /// A hundredth of the frame is roughly a 180°-plus fisheye: below it the
    /// number is a typo, not a lens.
    pub const MIN_FOCAL_RATIO: f64 = 0.01;

    /// Whether these three numbers describe a usable lens (C4-44).
    ///
    /// The wizard's `metres_per_unit`, two fields further down the same DTO, has
    /// been validated since P25.4 — these three were copied through unchecked,
    /// and they are the ones that reach the projection.
    pub fn validate(&self) -> Result<(), String> {
        if !self.focal_ratio.is_finite() || self.focal_ratio < Self::MIN_FOCAL_RATIO {
            return Err(format!(
                "focal ratio {} is not a lens: it must be finite and at least {} of the \
                 image's longer side, or the projection divides by ~zero",
                self.focal_ratio,
                Self::MIN_FOCAL_RATIO
            ));
        }
        if !self.k1.is_finite() || !self.k2.is_finite() {
            return Err(format!(
                "radial coefficients must be finite (k1 = {}, k2 = {})",
                self.k1, self.k2
            ));
        }
        Ok(())
    }

    /// The intrinsics this describes for a `width x height` photograph.
    ///
    /// Falls back to [`Default`] when [`validate`](Self::validate) refuses — the
    /// door that *takes* the numbers refuses first (`CaptureSettingsDto::
    /// to_config`), so reaching this fallback means an internal caller built an
    /// impossible camera, and a default lens is a wrong picture where NaN
    /// intrinsics are a wrong picture that also poisons every pose.
    pub fn intrinsics(&self, width: u32, height: u32) -> Intrinsics {
        let safe = match self.validate() {
            Ok(()) => *self,
            Err(why) => {
                tracing::warn!("assumed camera rejected ({why}); using the default lens");
                Self::default()
            }
        };
        let focal = safe.focal_ratio * width.max(height) as f64;
        Intrinsics::centred(width, height, focal).with_radial(safe.k1, safe.k2)
    }
}

/// Everything a capture run is tuned by, in one place.
///
/// `Default` is derived, and it is each part's own default: the assumed camera's
/// guess, `SfmConfig`'s committed constants, `DenseConfig`'s, and
/// `FinishConfig`'s `metres_per_unit = 1.0`. A hand-written impl here would be a
/// second place those defaults live.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CaptureConfig {
    /// What the photographs were taken with.
    pub camera: AssumedCamera,
    /// Structure from motion.
    pub sfm: SfmConfig,
    /// The dense stage.
    pub dense: DenseConfig,
    /// The finish, including [`FinishConfig::metres_per_unit`] — the scale step.
    pub finish: FinishConfig,
}

// ── stages and progress ─────────────────────────────────────────────────────

/// A stage of a capture, in the order they run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CaptureStage {
    /// Reading and decoding the photographs.
    Load,
    /// Structure from motion: features, matches, poses.
    Sfm,
    /// Multi-view stereo: depth maps and the fused surface.
    Dense,
    /// Retopology, unwrap, the three bakes.
    Finish,
    /// Writing the five assets into the project.
    Write,
}

impl CaptureStage {
    /// Every stage, in order.
    pub const ALL: [CaptureStage; 5] = [
        CaptureStage::Load,
        CaptureStage::Sfm,
        CaptureStage::Dense,
        CaptureStage::Finish,
        CaptureStage::Write,
    ];

    /// The wire name, and the label a progress bar shows.
    pub fn name(self) -> &'static str {
        match self {
            CaptureStage::Load => "load",
            CaptureStage::Sfm => "sfm",
            CaptureStage::Dense => "dense",
            CaptureStage::Finish => "finish",
            CaptureStage::Write => "write",
        }
    }

    /// Its position in [`ALL`](CaptureStage::ALL) — the monotonicity a progress
    /// reader checks.
    pub fn index(self) -> usize {
        CaptureStage::ALL
            .iter()
            .position(|s| *s == self)
            .expect("every stage is in ALL")
    }
}

/// What happened to a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePhase {
    /// The stage began.
    Started,
    /// The stage is part-way through. Only `Load`, `Finish` and (trivially)
    /// `Write` ever send this — see the module docs on granularity.
    Progress,
    /// The stage completed.
    Finished,
    /// The stage refused. The run is over.
    Failed,
    /// The run was cancelled between stages, before this one ran.
    Cancelled,
}

impl CapturePhase {
    /// The wire name.
    pub fn name(self) -> &'static str {
        match self {
            CapturePhase::Started => "started",
            CapturePhase::Progress => "progress",
            CapturePhase::Finished => "finished",
            CapturePhase::Failed => "failed",
            CapturePhase::Cancelled => "cancelled",
        }
    }

    /// Whether this phase ends the run.
    pub fn is_terminal(self) -> bool {
        matches!(self, CapturePhase::Failed | CapturePhase::Cancelled)
    }
}

/// One progress event.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureProgress {
    /// Which run — [`PhotogrammetrySession::start`] bumps it, so a late event
    /// from a cancelled run can be told from the current one's.
    pub run: u64,
    /// Which stage.
    pub stage: CaptureStage,
    /// What happened to it.
    pub phase: CapturePhase,
    /// Units done, for a `Progress` tick.
    pub done: u64,
    /// Units in total.
    pub total: u64,
    /// A short human label ("photograph 3 of 6", "albedo bake").
    pub detail: String,
    /// The refusal, on `Failed`.
    pub error: Option<String>,
}

// ── diagnostics ─────────────────────────────────────────────────────────────

/// How much a [`CaptureIssue`] matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureSeverity {
    /// The run cannot start (or could not finish) until this is fixed.
    Blocking,
    /// The result exists and is worse than it looks.
    Warning,
    /// Something a caller should know and need not act on.
    Note,
}

impl CaptureSeverity {
    /// The wire name.
    pub fn name(self) -> &'static str {
        match self {
            CaptureSeverity::Blocking => "blocking",
            CaptureSeverity::Warning => "warning",
            CaptureSeverity::Note => "note",
        }
    }
}

/// Something the wizard has to say about a capture.
///
/// Typed rather than a string list so a gate can assert *which* finding it got,
/// and every `Display` carries a **remedy** — the standard P25.3's audit
/// withdrew a claim about and this batch makes true.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureIssue {
    /// A file could not be read or decoded.
    Unreadable {
        /// The path, as the caller gave it.
        path: String,
        /// The decoder's complaint.
        message: String,
    },
    /// Fewer usable photographs than structure from motion needs.
    TooFewPhotos {
        /// How many decoded.
        given: usize,
        /// [`SfmConfig::min_views`].
        required: usize,
    },
    /// The set is not all one size.
    MixedResolutions {
        /// The most common size.
        common: (u32, u32),
        /// One that differs.
        odd: (u32, u32),
        /// How many distinct sizes there are.
        sizes: usize,
    },
    /// [`FinishConfig::metres_per_unit`] is not a scale. Caught here so a
    /// multi-minute run is not started to learn it.
    BadScale {
        /// What the caller asked for.
        metres_per_unit: f64,
    },
    /// A photograph never got a pose.
    ViewNotRegistered {
        /// Its view index.
        view: u32,
        /// Its file name.
        photo: String,
        /// How many 2D-3D correspondences it had when last considered.
        correspondences: usize,
    },
    /// A photograph registered on uncomfortably few inliers.
    ThinRegistration {
        /// Its view index.
        view: u32,
        /// Its file name.
        photo: String,
        /// The PnP inlier count.
        inliers: usize,
    },
    /// Anything else structure from motion said, in its own words.
    Sparse {
        /// The advisory, rendered.
        message: String,
    },
    /// Anything the dense stage said, in its own words.
    Dense {
        /// The advisory, rendered.
        message: String,
    },
    /// Geometry only **one** camera photographed — the overlap warning the
    /// spec names.
    SingleCoverage {
        /// Triangles seen by exactly one camera.
        triangles: usize,
        /// Triangles examined.
        examined: usize,
    },
    /// Geometry **no** camera photographed. Only reachable with
    /// [`FinishConfig::trim_unseen`] off, which is why it is separate from the
    /// trim advisory.
    NoCoverage {
        /// Triangles seen by no camera.
        triangles: usize,
        /// Triangles examined.
        examined: usize,
    },
    /// A finding of the finish pipeline, in its own words.
    Finish(FinishAdvisory),
    /// The written scan has no derived meshlet DAG, so the editor viewport
    /// draws it as a placeholder cube.
    ///
    /// A **note** rather than a warning: the asset is correct, complete and
    /// re-openable. It is said out loud because the alternative is a user
    /// watching a cube appear where their scan should be and concluding the
    /// wizard failed.
    ///
    /// **Wave D: raised only when it is TRUE.** The import now derives the DAG
    /// synchronously, so this fires for the two cases where a cube really is
    /// what the viewport will draw — a scan below `[vgeom] min_triangles`
    /// (`VmeshDerivation::Skipped`) and a derivation that failed. It used to
    /// fire unconditionally, because nothing derived anything.
    NoMeshletDag,
}

impl CaptureIssue {
    /// How much it matters.
    pub fn severity(&self) -> CaptureSeverity {
        match self {
            CaptureIssue::Unreadable { .. }
            | CaptureIssue::TooFewPhotos { .. }
            | CaptureIssue::BadScale { .. } => CaptureSeverity::Blocking,
            CaptureIssue::NoMeshletDag => CaptureSeverity::Note,
            _ => CaptureSeverity::Warning,
        }
    }

    /// Which stage raised it.
    pub fn stage(&self) -> CaptureStage {
        match self {
            CaptureIssue::Unreadable { .. }
            | CaptureIssue::TooFewPhotos { .. }
            | CaptureIssue::MixedResolutions { .. }
            | CaptureIssue::BadScale { .. } => CaptureStage::Load,
            CaptureIssue::ViewNotRegistered { .. }
            | CaptureIssue::ThinRegistration { .. }
            | CaptureIssue::Sparse { .. } => CaptureStage::Sfm,
            CaptureIssue::Dense { .. } => CaptureStage::Dense,
            CaptureIssue::SingleCoverage { .. }
            | CaptureIssue::NoCoverage { .. }
            | CaptureIssue::Finish(_) => CaptureStage::Finish,
            CaptureIssue::NoMeshletDag => CaptureStage::Write,
        }
    }

    /// Whether this stops a run from starting.
    pub fn blocks(&self) -> bool {
        self.severity() == CaptureSeverity::Blocking
    }
}

impl std::fmt::Display for CaptureIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureIssue::Unreadable { path, message } => write!(
                f,
                "{path} could not be read as a photograph ({message}) — remove it from the set, or \
                 re-export it as PNG, JPEG, TGA, BMP, HDR or EXR"
            ),
            CaptureIssue::TooFewPhotos { given, required } => write!(
                f,
                "photogrammetry needs at least {required} photographs and {given} decoded — add \
                 more, taken from positions that see the same surfaces from different angles, \
                 because one baseline gives the solve nothing to cross-check against"
            ),
            CaptureIssue::MixedResolutions { common, odd, sizes } => write!(
                f,
                "the set has {sizes} different image sizes ({}x{} and {}x{} among them) and one \
                 assumed focal length is applied to all of them, so the odd ones out are \
                 reconstructed with the wrong lens — crop or resize the set to one size, or \
                 reconstruct the sizes separately",
                common.0, common.1, odd.0, odd.1
            ),
            CaptureIssue::BadScale { metres_per_unit } => write!(
                f,
                "metres per unit is {metres_per_unit}, which is not a scale — zero collapses the \
                 mesh onto its own origin and a negative value mirrors it away from its own \
                 normals; leave it at 1.0 to keep the reconstruction's own units"
            ),
            CaptureIssue::ViewNotRegistered {
                view,
                photo,
                correspondences,
            } => write!(
                f,
                "{photo} (view {view}) never got a pose: it had {correspondences} 2D-3D \
                 correspondences with the rest of the set, which is what too little overlap looks \
                 like — re-shoot it with more of the same surface in frame, or drop it, because \
                 its pixels contribute no colour and no depth as things stand"
            ),
            CaptureIssue::ThinRegistration {
                view,
                photo,
                inliers,
            } => write!(
                f,
                "{photo} (view {view}) registered on only {inliers} inliers, so its pose is the \
                 least trustworthy in the set and any smearing in the texture is likely to be \
                 its — add a photograph between it and its neighbours"
            ),
            CaptureIssue::Sparse { message } => write!(f, "{message}"),
            CaptureIssue::Dense { message } => write!(f, "{message}"),
            CaptureIssue::SingleCoverage {
                triangles,
                examined,
            } => write!(
                f,
                "{triangles} of {examined} finished triangles were photographed by exactly ONE \
                 camera; nothing cross-checks their colour or their depth, so that is where a \
                 scan usually looks smeared — add photographs that see those surfaces from a \
                 second angle (the coverage list shows which cameras are carrying the set alone)"
            ),
            CaptureIssue::NoCoverage {
                triangles,
                examined,
            } => write!(
                f,
                "{triangles} of {examined} finished triangles were photographed by NO camera and \
                 their colour is entirely invented by dilation — photograph the missing side, or \
                 turn the unphotographed trim back on so the geometry nobody captured is removed \
                 rather than textured"
            ),
            CaptureIssue::Finish(advisory) => write!(f, "{advisory}"),
            CaptureIssue::NoMeshletDag => write!(
                f,
                "the scan's mesh has no derived meshlet DAG yet, and that is the editor \
                 viewport's only real-geometry path, so it draws as a PLACEHOLDER CUBE until one \
                 exists; the project-open sweep derives one for every mesh that lacks it, so \
                 re-opening the project is what makes the scan visible"
            ),
        }
    }
}

// ── coverage ────────────────────────────────────────────────────────────────

/// What one camera saw of the finished mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewCoverage {
    /// The view index, which is the photograph's position in the input list.
    pub view: u32,
    /// Its file name.
    pub photo: String,
    /// Whether it got a pose at all.
    pub registered: bool,
    /// Finished triangles this camera can see.
    pub triangles_seen: usize,
    /// That, as a fraction of the finished mesh.
    pub fraction: f64,
}

/// **The coverage overlay's data**: which camera saw what, and how much of the
/// result rests on a single opinion.
///
/// Measured through [`view_sees`] — the pipeline's own visibility test, not a
/// second one — so "the wizard says this camera saw it" and "the bake took
/// colour from this camera there" are the same claim.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageReport {
    /// Triangles in the finished mesh.
    pub triangles: usize,
    /// One entry per input photograph, in view order.
    pub views: Vec<ViewCoverage>,
    /// Triangles no camera can see.
    pub seen_by_none: usize,
    /// Triangles exactly one camera can see.
    pub seen_by_one: usize,
    /// Triangles two or more cameras can see — the redundant ones.
    pub seen_by_two_or_more: usize,
    /// Atlas texels no camera contributed to (from the albedo bake).
    pub unseen_texels: usize,
    /// Atlas texels the charts cover.
    pub covered_texels: usize,
}

impl CoverageReport {
    /// The share of the finished mesh at least one camera saw.
    pub fn covered_fraction(&self) -> f64 {
        if self.triangles == 0 {
            return 0.0;
        }
        (self.triangles - self.seen_by_none) as f64 / self.triangles as f64
    }

    /// The share that two or more cameras saw — the redundancy the whole method
    /// rests on.
    pub fn overlap_fraction(&self) -> f64 {
        if self.triangles == 0 {
            return 0.0;
        }
        self.seen_by_two_or_more as f64 / self.triangles as f64
    }
}

/// Finished triangles' centroids, back in the reconstruction's own frame.
fn centroids(mesh: &MeshAsset, origin: DVec3, metres_per_unit: f64) -> Vec<DVec3> {
    // The mesh has already been through the scale step, so undo it before
    // comparing against depth maps that never were: `view_sees` projects with
    // the reconstruction's poses, which are in baseline units.
    let inv = if metres_per_unit.is_finite() && metres_per_unit != 0.0 {
        1.0 / metres_per_unit
    } else {
        1.0
    };
    let mut out = Vec::new();
    for sm in &mesh.submeshes {
        for tri in sm.indices.chunks_exact(3) {
            let mut c = DVec3::ZERO;
            for &i in tri {
                let p = sm.vertices[i as usize].position;
                c += DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            }
            out.push(c / 3.0 * inv + origin);
        }
    }
    out
}

/// Measure [`CoverageReport`] for a finished asset.
///
/// `photo_names` is indexed by **view index**, like `photos` everywhere else in
/// this pipeline, so an unregistered view still gets a row that names its file.
///
/// `metres_per_unit` is the [`FinishConfig`] the asset was finished at: the mesh
/// has been through the scale step and the depth maps have not, so the centroids
/// are divided back into baseline units before they are projected. Handing it in
/// rather than reading it off the asset is deliberate —
/// [`FinishedAsset`] does not carry it, and a coverage report computed against
/// the wrong scale would say every camera saw nothing.
pub fn measure_coverage(
    finished: &FinishedAsset,
    reconstruction: &Reconstruction,
    dense: &DenseReconstruction,
    photos: &[RgbImage],
    photo_names: &[String],
    tolerance: f32,
    metres_per_unit: f64,
) -> CoverageReport {
    let points = centroids(&finished.mesh, finished.origin_units, metres_per_unit);
    let mut per_triangle = vec![0u32; points.len()];
    let mut views: Vec<ViewCoverage> = photo_names
        .iter()
        .enumerate()
        .map(|(i, name)| ViewCoverage {
            view: i as u32,
            photo: name.clone(),
            registered: false,
            triangles_seen: 0,
            fraction: 0.0,
        })
        .collect();

    for (slot, camera) in reconstruction.cameras.values().enumerate() {
        let Some(image) = photos.get(camera.view as usize) else {
            continue;
        };
        let (Some(depth), Some(hints)) = (dense.depth_maps.get(slot), dense.surface.get(slot))
        else {
            continue;
        };
        let view = AlbedoView {
            camera,
            image,
            depth,
            hints,
        };
        let mut seen = 0usize;
        for (i, p) in points.iter().enumerate() {
            if view_sees(&view, *p, tolerance) {
                per_triangle[i] += 1;
                seen += 1;
            }
        }
        if let Some(row) = views.get_mut(camera.view as usize) {
            row.registered = true;
            row.triangles_seen = seen;
            row.fraction = if points.is_empty() {
                0.0
            } else {
                seen as f64 / points.len() as f64
            };
        }
    }

    let seen_by_none = per_triangle.iter().filter(|n| **n == 0).count();
    let seen_by_one = per_triangle.iter().filter(|n| **n == 1).count();
    let seen_by_two_or_more = per_triangle.iter().filter(|n| **n >= 2).count();
    let (unseen_texels, covered_texels) = finished
        .advisories
        .iter()
        .find_map(|a| match a {
            FinishAdvisory::UnseenTexels { unseen, covered } => Some((*unseen, *covered)),
            _ => None,
        })
        .unwrap_or((0, 0));

    CoverageReport {
        triangles: points.len(),
        views,
        seen_by_none,
        seen_by_one,
        seen_by_two_or_more,
        unseen_texels,
        covered_texels,
    }
}

/// The multiplier that makes a reconstruction's longest side measure
/// `known_metres` — **the scale step's arithmetic**.
///
/// The honest v1 affordance the P25.4 ruling settled for: run once at `1.0`,
/// read [`CaptureProduct::extent_units`] off the result, type the real length of
/// the object's longest side, and re-run the finish stage alone. Picking two
/// points on the preview is the better tool and is not this batch.
///
/// `None` when either number is not a length, so a caller cannot divide its way
/// to a scale [`FinishConfig::metres_per_unit`] would refuse.
pub fn scale_for_longest_side(extent_units: f64, known_metres: f64) -> Option<f64> {
    if !(extent_units.is_finite() && extent_units > 0.0) {
        return None;
    }
    if !(known_metres.is_finite() && known_metres > 0.0) {
        return None;
    }
    Some(known_metres / extent_units)
}

// ── the product ─────────────────────────────────────────────────────────────

/// Everything a completed run holds, in memory, before anything is written.
pub struct CaptureProduct {
    /// The poses.
    pub reconstruction: Reconstruction,
    /// The depth maps and the fused surface.
    pub dense: DenseReconstruction,
    /// The five payloads, un-written.
    pub finished: FinishedAsset,
    /// Who saw what.
    pub coverage: CoverageReport,
    /// Everything worth saying, in severity-then-stage order.
    pub issues: Vec<CaptureIssue>,
    /// The longest side of the finished mesh in **baseline units**, which is
    /// what a known real-world length is divided by to get the scale step.
    pub extent_units: f64,
    /// Wall-clock milliseconds per stage, in [`CaptureStage::ALL`] order.
    pub elapsed_ms: [u64; 5],
}

impl CaptureProduct {
    /// The dense stage's numbers, for the readout.
    pub fn dense_report(&self) -> &DenseReport {
        &self.dense.report
    }

    /// The finished mesh's geometry, flattened for the offscreen preview.
    ///
    /// One buffer pair for the whole asset, on the thumbnailer's own convention
    /// (its `combined_geometry` does the same for a mesh thumbnail): the preview
    /// draws one surface and has no material slots to switch between.
    pub fn preview_geometry(&self) -> (Vec<MeshVertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for sm in &self.finished.mesh.submeshes {
            let base = verts.len() as u32;
            verts.extend_from_slice(&sm.vertices);
            indices.extend(sm.indices.iter().map(|i| i + base));
        }
        (verts, indices)
    }
}

// ── the session ─────────────────────────────────────────────────────────────

/// Where a session is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    /// No run has been started (photographs may or may not be loaded).
    Idle,
    /// A run is in this stage.
    Running(CaptureStage),
    /// A run finished and its product is in hand, un-written.
    Ready,
    /// The product has been written into a project.
    Imported,
    /// A run refused.
    Failed,
    /// A run was cancelled between stages.
    Cancelled,
}

impl CaptureState {
    /// The wire name.
    pub fn name(self) -> &'static str {
        match self {
            CaptureState::Idle => "idle",
            CaptureState::Running(_) => "running",
            CaptureState::Ready => "ready",
            CaptureState::Imported => "imported",
            CaptureState::Failed => "failed",
            CaptureState::Cancelled => "cancelled",
        }
    }
}

/// Why a session refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CaptureError {
    /// The pre-flight found something blocking.
    #[error("the photographs cannot be reconstructed as they are: {0}")]
    Preflight(String),
    /// A run is already in flight.
    #[error("a reconstruction is already running — cancel it before starting another")]
    Busy,
    /// Something was asked for that needs a finished run.
    #[error("there is no finished reconstruction to {0}")]
    NoProduct(&'static str),
    /// A stage refused.
    #[error("{stage} failed: {message}")]
    Stage {
        /// Which stage.
        stage: &'static str,
        /// Its own complaint.
        message: String,
    },
}

/// One decoded photograph. The path stays in [`PhotoEntry`]; what the pipeline
/// needs is the two decodes.
struct LoadedPhoto {
    name: String,
    colour: RgbImage,
    gray: GrayImage,
}

/// What the worker publishes back.
#[derive(Default)]
struct Shared {
    state: Option<CaptureState>,
    product: Option<Box<CaptureProduct>>,
    error: Option<String>,
    /// Findings collected by a run that **did not reach a product**.
    ///
    /// A successful run carries its findings on [`CaptureProduct::issues`]; a
    /// failed one has no product, and the advisories the stages before the
    /// refusal produced are exactly what a user needs to read next. Published by
    /// the worker on the way out rather than dropped with the stack.
    issues: Vec<CaptureIssue>,
}

/// **The capture wizard's session.** One per editor process; the wizard opens
/// it, drops photographs on it, runs it, previews it and imports it.
pub struct PhotogrammetrySession {
    photos: Arc<Vec<LoadedPhoto>>,
    /// The pre-flight for the photographs currently loaded, including the rows
    /// for files that failed to decode (which never reach `photos`).
    entries: Vec<PhotoEntry>,
    load_issues: Vec<CaptureIssue>,
    cfg: CaptureConfig,
    state: CaptureState,
    shared: Arc<Mutex<Shared>>,
    tx: Sender<CaptureProgress>,
    rx: Receiver<CaptureProgress>,
    cancel: CancelToken,
    worker: Option<JoinHandle<()>>,
    run: u64,
}

/// One row of the pre-flight table.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotoEntry {
    /// The path as the caller gave it.
    pub path: PathBuf,
    /// Its file name.
    pub name: String,
    /// Pixel width, `0` when it did not decode.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Why it did not decode, when it did not.
    pub error: Option<String>,
}

impl Default for PhotogrammetrySession {
    fn default() -> Self {
        Self::new()
    }
}

impl PhotogrammetrySession {
    /// An empty session.
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            photos: Arc::new(Vec::new()),
            entries: Vec::new(),
            load_issues: Vec::new(),
            cfg: CaptureConfig::default(),
            state: CaptureState::Idle,
            shared: Arc::new(Mutex::new(Shared::default())),
            tx,
            rx,
            cancel: CancelToken::new(),
            worker: None,
            run: 0,
        }
    }

    /// The configuration the next run will use.
    pub fn config(&self) -> &CaptureConfig {
        &self.cfg
    }

    /// Replace it. Takes effect on the next [`start`](Self::start) or
    /// [`refinish`](Self::refinish); a run in flight keeps the config it began
    /// with, because half a run tuned two ways is not a reconstruction.
    pub fn set_config(&mut self, cfg: CaptureConfig) {
        self.cfg = cfg;
    }

    /// The pre-flight table.
    pub fn photos(&self) -> &[PhotoEntry] {
        &self.entries
    }

    /// Where the session is.
    ///
    /// Read through the worker's own published slot rather than a field this
    /// side keeps in step: the worker is the only thing that knows a stage
    /// finished, and a second copy of "is it running" is the kind of drift a
    /// Cancel button is judged by. `self.state` is the fallback for the states
    /// only this side ever sets (`Idle` after a load, `Imported` after a write).
    pub fn state(&self) -> CaptureState {
        self.shared
            .lock()
            .ok()
            .and_then(|g| g.state)
            .unwrap_or(self.state)
    }

    /// The current run id.
    pub fn run_id(&self) -> u64 {
        self.run
    }

    /// **Load and validate a set of photographs**, replacing whatever was
    /// loaded. Returns the blocking-and-warning list; nothing is reconstructed.
    ///
    /// Decoding is what makes this the pre-flight rather than a guess: a file
    /// that cannot be decoded is named here, at the moment it is dropped on the
    /// dialog, rather than three minutes into a solve.
    ///
    /// **Refuses [`CaptureError::Busy`] while a run is in flight**, by name,
    /// like [`start`](Self::start) does. Not a formality: this replaces the
    /// published state as well as the photographs, so a load during a solve used
    /// to leave the session reading `Idle` over a worker still running — Cancel
    /// would answer `false`, and minutes later that worker would publish a
    /// product for photographs nobody has loaded any more.
    pub fn load_photos(&mut self, paths: &[PathBuf]) -> Result<Vec<CaptureIssue>, CaptureError> {
        if matches!(self.state(), CaptureState::Running(_)) {
            return Err(CaptureError::Busy);
        }
        let mut entries = Vec::with_capacity(paths.len());
        let mut loaded = Vec::new();
        let mut issues = Vec::new();
        for path in paths {
            let name = file_name_of(path);
            match RgbImage::load(path) {
                Ok(colour) => {
                    let gray = colour.to_gray();
                    entries.push(PhotoEntry {
                        path: path.clone(),
                        name: name.clone(),
                        width: colour.width(),
                        height: colour.height(),
                        error: None,
                    });
                    loaded.push(LoadedPhoto { name, colour, gray });
                }
                Err(e) => {
                    let message = e.to_string();
                    entries.push(PhotoEntry {
                        path: path.clone(),
                        name,
                        width: 0,
                        height: 0,
                        error: Some(message.clone()),
                    });
                    issues.push(CaptureIssue::Unreadable {
                        path: path.display().to_string(),
                        message,
                    });
                }
            }
        }
        self.entries = entries;
        self.photos = Arc::new(loaded);
        // ONLY the per-file findings are remembered: a file that would not
        // decode is a fact about that file and cannot be recomputed once it is
        // out of the list. Everything else the pre-flight says is a function of
        // what is loaded AND of the configuration, so it is computed on demand —
        // `min_views` is a config field, and a cached "too few photographs"
        // would keep saying three after a caller lowered it.
        self.load_issues = issues;
        self.state = CaptureState::Idle;
        if let Ok(mut shared) = self.shared.lock() {
            shared.product = None;
            shared.error = None;
            shared.state = None;
            // The last run's findings are about the last run's photographs.
            shared.issues.clear();
        }
        Ok(self.preflight())
    }

    /// Everything the pre-flight has to say about what is loaded **and** the
    /// configuration it would be run with.
    ///
    /// The scale check lives here rather than only in the finish because
    /// `finish_reconstruction` is four stages away: a wizard that starts a
    /// multi-minute run to learn that a text field said `0` is a wizard that
    /// wastes four minutes. Its wording defers to
    /// [`FinishError::BadScale`](crate::photogrammetry::FinishError::BadScale)'s
    /// reasoning rather than inventing a second rule.
    pub fn preflight(&self) -> Vec<CaptureIssue> {
        let mut out = self.load_issues.clone();
        if self.photos.len() < self.cfg.sfm.min_views {
            out.push(CaptureIssue::TooFewPhotos {
                given: self.photos.len(),
                required: self.cfg.sfm.min_views,
            });
        }
        if let Some(issue) = resolution_issue(&self.photos) {
            out.push(issue);
        }
        let scale = self.cfg.finish.metres_per_unit;
        if !(scale.is_finite() && scale > 0.0) {
            out.push(CaptureIssue::BadScale {
                metres_per_unit: scale,
            });
        }
        out
    }

    /// **Start a run**: load-validate, structure from motion, dense, finish.
    ///
    /// Returns the run id. Refuses when the pre-flight blocks or a run is
    /// already in flight — a refusal by name rather than a queue, because two
    /// reconstructions of two photo sets in one session is not a thing the
    /// wizard can show.
    pub fn start(&mut self, pool: Arc<JobPool>) -> Result<u64, CaptureError> {
        self.begin(pool, false)
    }

    /// **Re-run the finish stage alone**, over the reconstruction already in
    /// hand — how the scale step is applied without paying for structure from
    /// motion and the dense solve a second time.
    ///
    /// [`FinishConfig::metres_per_unit`] is applied once, at the end, so a
    /// re-finish is exactly what changing it costs; so is a new triangle budget,
    /// a bigger atlas, or turning de-lighting on to see it refuse.
    pub fn refinish(&mut self, pool: Arc<JobPool>) -> Result<u64, CaptureError> {
        self.begin(pool, true)
    }

    fn begin(&mut self, pool: Arc<JobPool>, finish_only: bool) -> Result<u64, CaptureError> {
        if matches!(self.state(), CaptureState::Running(_)) {
            return Err(CaptureError::Busy);
        }
        let blocking: Vec<String> = self
            .preflight()
            .iter()
            .filter(|i| i.blocks())
            .map(|i| i.to_string())
            .collect();
        if !blocking.is_empty() {
            return Err(CaptureError::Preflight(blocking.join("; ")));
        }
        // A re-finish needs the two solves it is skipping.
        let seed = if finish_only {
            let guard = self.shared.lock().map_err(|e| CaptureError::Stage {
                stage: "finish",
                message: e.to_string(),
            })?;
            let product = guard
                .product
                .as_ref()
                .ok_or(CaptureError::NoProduct("re-finish"))?;
            Some((product.reconstruction.clone(), product.dense.clone()))
        } else {
            None
        };

        self.join_worker();
        self.run += 1;
        self.cancel = CancelToken::new();
        let run = self.run;
        let photos = self.photos.clone();
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        let shared = self.shared.clone();
        let cancel = self.cancel.clone();
        {
            let mut guard = self.shared.lock().map_err(|e| CaptureError::Stage {
                stage: "start",
                message: e.to_string(),
            })?;
            guard.error = None;
            guard.issues.clear();
            guard.state = Some(CaptureState::Running(if finish_only {
                CaptureStage::Finish
            } else {
                CaptureStage::Load
            }));
            if !finish_only {
                guard.product = None;
            }
        }
        self.state = CaptureState::Running(if finish_only {
            CaptureStage::Finish
        } else {
            CaptureStage::Load
        });
        let handle = std::thread::Builder::new()
            .name("photogrammetry".into())
            .spawn(move || {
                run_guarded(run, &tx, &shared, || {
                    run_capture(run, photos, cfg, seed, &pool, &cancel, &tx, &shared);
                });
            })
            .map_err(|e| CaptureError::Stage {
                stage: "start",
                message: e.to_string(),
            })?;
        self.worker = Some(handle);
        Ok(run)
    }

    /// Ask an in-flight run to stop. `false` when nothing is running.
    ///
    /// It stops **between stages** — see the module docs. The one guarantee that
    /// matters is unconditional: a cancelled run has written nothing, because
    /// [`import`](Self::import) is a separate call the user makes.
    pub fn cancel(&mut self) -> bool {
        if !matches!(self.state(), CaptureState::Running(_)) {
            return false;
        }
        self.cancel.cancel();
        true
    }

    /// Drain the progress events emitted since the last drain.
    pub fn drain(&self) -> Vec<CaptureProgress> {
        let mut out = Vec::new();
        while let Ok(ev) = self.rx.try_recv() {
            out.push(ev);
        }
        out
    }

    fn join_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    /// Block until the run in flight has finished, if there is one.
    ///
    /// The wizard never calls this — it drains events on a tick — but a gate
    /// does, and so does [`reset`](Self::reset). Named after what it costs.
    pub fn wait(&mut self) {
        self.join_worker();
    }

    /// Run something against the finished product.
    ///
    /// A closure rather than a `&CaptureProduct` because the product lives
    /// behind the worker's mutex: handing out a reference would hand out the
    /// guard's lifetime, and every caller would have to know that.
    pub fn with_product<T>(&self, f: impl FnOnce(&CaptureProduct) -> T) -> Option<T> {
        let guard = self.shared.lock().ok()?;
        let product = guard.product.as_ref()?;
        Some(f(product))
    }

    /// The run's refusal, when it failed.
    pub fn error(&self) -> Option<String> {
        self.shared.lock().ok().and_then(|g| g.error.clone())
    }

    /// **Everything the wizard has to say right now**, in one rule, so a panel
    /// does not have to know which of three places a finding lives in.
    ///
    /// * A finished run's are on its product, which already contains everything
    ///   the pre-flight said that still matters.
    /// * A **failed** run's are the ones the stages before the refusal produced.
    ///   They used to be dropped with the worker's stack, which meant the one
    ///   moment a user most needs "station3.png never got a pose" was the one
    ///   moment they could not see it — the pre-flight cannot know it, and there
    ///   is no product to carry it.
    /// * Otherwise they are the pre-flight's, recomputed rather than remembered
    ///   (the `min_views` reason, on [`preflight`](Self::preflight)).
    ///
    /// A failed run's list is followed by the pre-flight, because what is loaded
    /// is still loaded and a mixed-resolution set is still mixed.
    pub fn findings(&self) -> Vec<CaptureIssue> {
        if let Ok(guard) = self.shared.lock() {
            if let Some(product) = guard.product.as_ref() {
                return product.issues.clone();
            }
            if !guard.issues.is_empty() {
                let mut out = guard.issues.clone();
                drop(guard);
                out.extend(self.preflight());
                return out;
            }
        }
        self.preflight()
    }

    /// **Write the finished scan into a project** — the fifth stage, and the one
    /// the user starts.
    ///
    /// `dir` is joined onto the project's own content root here, because
    /// [`AssetProject::write_asset`] takes its directory **verbatim** and a
    /// relative path would resolve against the process's working directory. That
    /// trap is documented at both ends and is closed at this one.
    pub fn import(
        &mut self,
        project: &mut AssetProject,
        name: &str,
    ) -> Result<FinishedIds, CaptureError> {
        let run = self.run;
        // A clone rather than `&self.tx`: the closure outlives the `&mut self`
        // writes below, and borrowing the field would make the state update a
        // borrow-checker error rather than a design.
        let tx = self.tx.clone();
        let send = |phase: CapturePhase, detail: &str, error: Option<String>| {
            let _ = tx.send(CaptureProgress {
                run,
                stage: CaptureStage::Write,
                phase,
                done: if phase == CapturePhase::Finished {
                    5
                } else {
                    0
                },
                total: 5,
                detail: detail.to_string(),
                error,
            });
        };
        send(CapturePhase::Started, "writing five assets", None);
        let dir = match project.content_dir(SCAN_FOLDER) {
            Ok(dir) => dir,
            Err(e) => {
                let message = e.to_string();
                send(
                    CapturePhase::Failed,
                    "content folder",
                    Some(message.clone()),
                );
                return Err(CaptureError::Stage {
                    stage: "write",
                    message,
                });
            }
        };
        let written = {
            let guard = self.shared.lock().map_err(|e| CaptureError::Stage {
                stage: "write",
                message: e.to_string(),
            })?;
            let product = guard
                .product
                .as_ref()
                .ok_or(CaptureError::NoProduct("import"))?;
            write_finished(project, &dir, name, &product.finished)
        };
        match written {
            Ok(ids) => {
                // **The meshlet DAG, derived here** (Wave D) — closing the P25
                // carried remainder exactly where `write_finished`'s own doc
                // says it belongs: *"P25.4's wizard is the door that places a
                // finish in a scene and is where the derivation belongs, on the
                // pattern the import orchestrator already uses."*
                //
                // Synchronously, and after the all-or-none write rather than
                // inside it — the reason that doc gives for keeping it out of
                // `write_finished` is that a sixth artifact would join a set
                // whose whole property is that it lands together or not at all.
                // A derivation that fails leaves five correct assets and a note,
                // which is a different and much smaller failure.
                //
                // It is seconds on a fifteen-thousand-triangle scan. That is the
                // right place to spend them: the author pressed Import and the
                // next thing they do is look at the viewport, and a queued
                // derivation is a window in which their scan is a cube.
                send(CapturePhase::Finished, "deriving the meshlet DAG", None);
                let dag = crate::assets::vmesh::ensure_vmesh(project, ids.mesh);
                let drew_a_cube = match &dag {
                    Ok(crate::assets::vmesh::VmeshDerivation::Skipped) | Err(_) => true,
                    Ok(_) => false,
                };
                if let Err(e) = &dag {
                    tracing::warn!("the scan's meshlet DAG could not be derived: {e}");
                }
                send(CapturePhase::Finished, "five assets written", None);
                self.state = CaptureState::Imported;
                if let Ok(mut guard) = self.shared.lock() {
                    guard.state = Some(CaptureState::Imported);
                    if let Some(product) = guard.product.as_mut() {
                        // The note is raised only when the viewport really will
                        // draw a cube. Raising it unconditionally — which is what
                        // it did while nothing derived the DAG — would now be
                        // telling the author about a hazard that is not there.
                        if drew_a_cube && !product.issues.contains(&CaptureIssue::NoMeshletDag) {
                            product.issues.push(CaptureIssue::NoMeshletDag);
                        }
                    }
                }
                Ok(ids)
            }
            Err(e) => {
                let message = e.to_string();
                send(CapturePhase::Failed, "asset write", Some(message.clone()));
                Err(CaptureError::Stage {
                    stage: "write",
                    message,
                })
            }
        }
    }

    /// Forget everything: photographs, product, state.
    ///
    /// Cancels and then **waits**, rather than refusing like
    /// [`load_photos`](Self::load_photos) does: a caller resetting is throwing
    /// the session away, and the only coherent way to do that is to outlive the
    /// worker that is writing into it. Named after what it costs — cancellation
    /// is between stages, so this blocks for the rest of the stage in flight, and
    /// a Ring-2 caller holding a session mutex across it holds it for that long
    /// too. P25.4's dialog therefore disables every path to it while a run is
    /// running.
    pub fn reset(&mut self) {
        self.cancel.cancel();
        self.join_worker();
        self.photos = Arc::new(Vec::new());
        self.entries.clear();
        self.load_issues.clear();
        self.state = CaptureState::Idle;
        if let Ok(mut guard) = self.shared.lock() {
            *guard = Shared::default();
        }
        // Drop stale events so the next run's stream starts clean.
        let _ = self.drain();
    }
}

impl Drop for PhotogrammetrySession {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.join_worker();
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// The mixed-resolution warning, or `None` when the set is uniform.
fn resolution_issue(photos: &[LoadedPhoto]) -> Option<CaptureIssue> {
    let mut sizes: Vec<((u32, u32), usize)> = Vec::new();
    for p in photos {
        let key = (p.colour.width(), p.colour.height());
        match sizes.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => *n += 1,
            None => sizes.push((key, 1)),
        }
    }
    if sizes.len() < 2 {
        return None;
    }
    // The commonest size, with ties broken by the first seen so the message is a
    // function of the input order alone.
    let common = sizes
        .iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| *k)
        .unwrap_or((0, 0));
    let odd = sizes
        .iter()
        .find(|(k, _)| *k != common)
        .map(|(k, _)| *k)
        .unwrap_or(common);
    Some(CaptureIssue::MixedResolutions {
        common,
        odd,
        sizes: sizes.len(),
    })
}

/// Run the worker body and turn a **panic** into a failed run.
///
/// # Why a session needs this and an import job does not
///
/// Every terminal state a session can reach is published by the worker itself,
/// so a worker that unwinds publishes nothing and the session stays
/// `Running(stage)` for ever: [`state`](PhotogrammetrySession::state) keeps
/// saying running, [`cancel`](PhotogrammetrySession::cancel) sets a flag no
/// thread will read again and answers `true`, and
/// [`start`](PhotogrammetrySession::start) refuses `Busy` for the rest of the
/// process. The wizard on top of it disables Close, Escape, the backdrop and
/// "Choose other photographs" *while running*, so the only way out of a panicked
/// stage would be restarting the editor.
///
/// The stages are four blocking calls into Ring-0 solvers over real photographs.
/// A refusal there is an `Err` and is reported; a panic there is a bug, and the
/// thing a user is owed when a bug happens is the same shape as a refusal —
/// a failed run, carrying what went wrong, with nothing written. That is all
/// this does. It does **not** make the session panic-safe in any deeper sense:
/// the product is not published, so nothing half-built survives.
///
/// The lock is taken through the poison, deliberately: the invariant [`Shared`]
/// carries is "the last thing the worker published", and this is the worker
/// publishing that it died.
fn run_guarded(
    run: u64,
    tx: &Sender<CaptureProgress>,
    shared: &Arc<Mutex<Shared>>,
    body: impl FnOnce(),
) {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) else {
        return;
    };
    let message = panic_message(&*payload);
    let mut guard = match shared.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    // The stage that was in flight is the one that panicked, because the worker
    // publishes `Running(stage)` as it enters each one.
    let stage = match guard.state {
        Some(CaptureState::Running(stage)) => stage,
        _ => CaptureStage::Load,
    };
    let message = format!("{} panicked: {message}", stage.name());
    guard.state = Some(CaptureState::Failed);
    guard.error = Some(message.clone());
    guard.product = None;
    drop(guard);
    let _ = tx.send(CaptureProgress {
        run,
        stage,
        phase: CapturePhase::Failed,
        done: 0,
        total: 0,
        detail: String::new(),
        error: Some(message),
    });
}

/// What a caught panic's payload says, for the two shapes `panic!` produces.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "a panic with no message".to_string()
}

/// The whole worker body: four stages, cancellable between them.
#[allow(clippy::too_many_arguments)]
fn run_capture(
    run: u64,
    photos: Arc<Vec<LoadedPhoto>>,
    cfg: CaptureConfig,
    seed: Option<(Reconstruction, DenseReconstruction)>,
    pool: &JobPool,
    cancel: &CancelToken,
    tx: &Sender<CaptureProgress>,
    shared: &Arc<Mutex<Shared>>,
) {
    let mut elapsed = [0u64; 5];
    let publish = |state: CaptureState, error: Option<String>| {
        if let Ok(mut guard) = shared.lock() {
            guard.state = Some(state);
            if error.is_some() {
                guard.error = error;
            }
        }
    };
    let emit = |stage: CaptureStage,
                phase: CapturePhase,
                done: u64,
                total: u64,
                detail: String,
                error: Option<String>| {
        let _ = tx.send(CaptureProgress {
            run,
            stage,
            phase,
            done,
            total,
            detail,
            error,
        });
    };
    // A cancellation, said once, in one shape.
    let stop = |stage: CaptureStage| {
        emit(
            stage,
            CapturePhase::Cancelled,
            0,
            0,
            "cancelled between stages".into(),
            None,
        );
        publish(CaptureState::Cancelled, None);
    };
    // A refusal, said once, in one shape — and carrying the findings the stages
    // BEFORE it produced, because a run that fails at the dense solve is exactly
    // when "station3.png never got a pose" is the sentence a user needs, and
    // there is no product to hang it on.
    let fail = |stage: CaptureStage, message: String, collected: &[CaptureIssue]| {
        emit(
            stage,
            CapturePhase::Failed,
            0,
            0,
            String::new(),
            Some(message.clone()),
        );
        if let Ok(mut guard) = shared.lock() {
            guard.issues = collected.to_vec();
        }
        publish(CaptureState::Failed, Some(message));
    };

    let mut issues: Vec<CaptureIssue> = Vec::new();
    let names: Vec<String> = photos.iter().map(|p| p.name.clone()).collect();
    let colours: Vec<RgbImage> = photos.iter().map(|p| p.colour.clone()).collect();

    let (reconstruction, dense) = match seed {
        // ── the re-finish path: the two solves are already in hand ──────────
        Some(pair) => pair,
        None => {
            // ── 1. load ─────────────────────────────────────────────────────
            let started = std::time::Instant::now();
            publish(CaptureState::Running(CaptureStage::Load), None);
            emit(
                CaptureStage::Load,
                CapturePhase::Started,
                0,
                photos.len() as u64,
                format!("{} photographs", photos.len()),
                None,
            );
            let mut views = Vec::with_capacity(photos.len());
            for (i, p) in photos.iter().enumerate() {
                views.push(View {
                    image: p.gray.clone(),
                    intrinsics: cfg.camera.intrinsics(p.gray.width(), p.gray.height()),
                });
                emit(
                    CaptureStage::Load,
                    CapturePhase::Progress,
                    i as u64 + 1,
                    photos.len() as u64,
                    p.name.clone(),
                    None,
                );
            }
            elapsed[CaptureStage::Load.index()] = started.elapsed().as_millis() as u64;
            emit(
                CaptureStage::Load,
                CapturePhase::Finished,
                photos.len() as u64,
                photos.len() as u64,
                format!("{} views prepared", views.len()),
                None,
            );
            if cancel.is_cancelled() {
                stop(CaptureStage::Sfm);
                return;
            }

            // ── 2. structure from motion ────────────────────────────────────
            let started = std::time::Instant::now();
            publish(CaptureState::Running(CaptureStage::Sfm), None);
            emit(
                CaptureStage::Sfm,
                CapturePhase::Started,
                0,
                1,
                "features, matches and poses".into(),
                None,
            );
            let reconstruction = match reconstruct(&views, &cfg.sfm, pool) {
                Ok(r) => r,
                Err(e) => {
                    fail(CaptureStage::Sfm, e.to_string(), &issues);
                    return;
                }
            };
            elapsed[CaptureStage::Sfm.index()] = started.elapsed().as_millis() as u64;
            for advisory in &reconstruction.advisories {
                issues.push(sparse_issue(advisory, &names));
            }
            emit(
                CaptureStage::Sfm,
                CapturePhase::Finished,
                1,
                1,
                format!(
                    "{} of {} views registered, {} points, RMS {:.3} px",
                    reconstruction.report.registered,
                    reconstruction.report.views,
                    reconstruction.report.points,
                    reconstruction.report.reprojection_rms_px
                ),
                None,
            );
            if cancel.is_cancelled() {
                stop(CaptureStage::Dense);
                return;
            }

            // ── 3. dense ────────────────────────────────────────────────────
            let started = std::time::Instant::now();
            publish(CaptureState::Running(CaptureStage::Dense), None);
            emit(
                CaptureStage::Dense,
                CapturePhase::Started,
                0,
                1,
                "depth maps and fusion".into(),
                None,
            );
            let dense = match reconstruct_dense(&views, &reconstruction, &cfg.dense, pool) {
                Ok(d) => d,
                Err(e) => {
                    fail(CaptureStage::Dense, e.to_string(), &issues);
                    return;
                }
            };
            elapsed[CaptureStage::Dense.index()] = started.elapsed().as_millis() as u64;
            for advisory in &dense.advisories {
                issues.push(CaptureIssue::Dense {
                    message: advisory.to_string(),
                });
            }
            emit(
                CaptureStage::Dense,
                CapturePhase::Finished,
                1,
                1,
                format!(
                    "{} triangles at a {:.4} unit voxel",
                    dense.report.triangles, dense.report.voxel_size
                ),
                None,
            );
            if cancel.is_cancelled() {
                stop(CaptureStage::Finish);
                return;
            }
            (reconstruction, dense)
        }
    };

    // ── 4. finish ───────────────────────────────────────────────────────────
    let started = std::time::Instant::now();
    publish(CaptureState::Running(CaptureStage::Finish), None);
    let steps = FinishStep::ALL.len() as u64;
    emit(
        CaptureStage::Finish,
        CapturePhase::Started,
        0,
        steps,
        "retopology, unwrap and three bakes".into(),
        None,
    );
    let finished = {
        let mut on_step = |step: FinishStep| {
            emit(
                CaptureStage::Finish,
                CapturePhase::Progress,
                step.index() as u64,
                steps,
                step.label().to_string(),
                None,
            );
        };
        finish_reconstruction_with_progress(
            &dense,
            &reconstruction,
            &colours,
            &cfg.finish,
            pool,
            &mut on_step,
        )
    };
    let finished = match finished {
        Ok(f) => f,
        Err(e) => {
            fail(CaptureStage::Finish, e.to_string(), &issues);
            return;
        }
    };
    elapsed[CaptureStage::Finish.index()] = started.elapsed().as_millis() as u64;

    let coverage = measure_coverage(
        &finished,
        &reconstruction,
        &dense,
        &colours,
        &names,
        cfg.finish.bake.occlusion_tolerance,
        cfg.finish.metres_per_unit,
    );
    if coverage.seen_by_none > 0 {
        issues.push(CaptureIssue::NoCoverage {
            triangles: coverage.seen_by_none,
            examined: coverage.triangles,
        });
    }
    if coverage.seen_by_one > 0 {
        issues.push(CaptureIssue::SingleCoverage {
            triangles: coverage.seen_by_one,
            examined: coverage.triangles,
        });
    }
    for advisory in &finished.advisories {
        issues.push(CaptureIssue::Finish(advisory.clone()));
    }
    issues.sort_by_key(|i| (i.severity(), i.stage(), i.to_string()));

    let extent_units = longest_side_units(&finished, cfg.finish.metres_per_unit);
    emit(
        CaptureStage::Finish,
        CapturePhase::Finished,
        steps,
        steps,
        format!(
            "{} triangles, {} charts, {} of {} triangles seen by two cameras or more",
            finished.report.final_triangles,
            finished.report.charts,
            coverage.seen_by_two_or_more,
            coverage.triangles
        ),
        None,
    );

    if let Ok(mut guard) = shared.lock() {
        guard.product = Some(Box::new(CaptureProduct {
            reconstruction,
            dense,
            finished,
            coverage,
            issues,
            extent_units,
            elapsed_ms: elapsed,
        }));
        guard.state = Some(CaptureState::Ready);
    }
}

/// The longest side of the finished mesh in **baseline units**.
///
/// Divided back out of the scale step, so a caller that ran at `2.5` and one
/// that ran at `1.0` read the same number and can compute the same correction.
fn longest_side_units(finished: &FinishedAsset, metres_per_unit: f64) -> f64 {
    let b = finished.mesh.bounds;
    let extent = [
        (b.max[0] - b.min[0]) as f64,
        (b.max[1] - b.min[1]) as f64,
        (b.max[2] - b.min[2]) as f64,
    ];
    let longest = extent.iter().copied().fold(0.0f64, f64::max);
    if metres_per_unit.is_finite() && metres_per_unit > 0.0 {
        longest / metres_per_unit
    } else {
        longest
    }
}

/// One structure-from-motion advisory, given a face and a remedy where this
/// module has one to add.
fn sparse_issue(advisory: &Advisory, names: &[String]) -> CaptureIssue {
    match advisory {
        Advisory::ViewNotRegistered {
            view,
            correspondences,
        } => CaptureIssue::ViewNotRegistered {
            view: *view,
            photo: names
                .get(*view as usize)
                .cloned()
                .unwrap_or_else(|| format!("view {view}")),
            correspondences: *correspondences,
        },
        Advisory::ThinRegistration { view, inliers } => CaptureIssue::ThinRegistration {
            view: *view,
            photo: names
                .get(*view as usize)
                .cloned()
                .unwrap_or_else(|| format!("view {view}")),
            inliers: *inliers,
        },
        other => CaptureIssue::Sparse {
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stage_and_step_orders_are_their_own_index() {
        for (i, stage) in CaptureStage::ALL.iter().enumerate() {
            assert_eq!(stage.index(), i, "{} is out of order", stage.name());
        }
        for (i, step) in FinishStep::ALL.iter().enumerate() {
            assert_eq!(step.index(), i, "{} is out of order", step.label());
        }
    }

    #[test]
    fn an_assumed_camera_scales_its_focal_with_the_longer_side() {
        let cam = AssumedCamera::default();
        let wide = cam.intrinsics(4000, 3000);
        let tall = cam.intrinsics(3000, 4000);
        assert!((wide.fx - 1.2 * 4000.0).abs() < 1e-9);
        assert!((tall.fx - 1.2 * 4000.0).abs() < 1e-9);
        // The principal point follows the pixel-centre convention, which is
        // `(w - 1) / 2` and NOT `w / 2` — P25.1's F2, which the real-photo path
        // is exactly what could see.
        assert!((wide.cx - 1999.5).abs() < 1e-9);
        // And distortion is carried, not dropped.
        let lens = AssumedCamera {
            focal_ratio: 0.9375,
            k1: -0.09,
            k2: 0.02,
        };
        let i = lens.intrinsics(320, 240);
        assert!((i.fx - 300.0).abs() < 1e-9);
        assert_eq!((i.k1, i.k2), (-0.09, 0.02));
    }

    #[test]
    fn the_scale_helper_refuses_everything_the_finish_would() {
        assert_eq!(scale_for_longest_side(4.0, 2.0), Some(0.5));
        assert_eq!(scale_for_longest_side(0.0, 2.0), None);
        assert_eq!(scale_for_longest_side(4.0, 0.0), None);
        assert_eq!(scale_for_longest_side(-4.0, 2.0), None);
        assert_eq!(scale_for_longest_side(4.0, -2.0), None);
        assert_eq!(scale_for_longest_side(f64::NAN, 2.0), None);
        assert_eq!(scale_for_longest_side(f64::INFINITY, 2.0), None);
    }

    #[test]
    fn every_issue_renders_a_remedy_and_never_a_double_space() {
        // The P25.3 arm, met again on this module's own enum: a run of spaces is
        // an eaten `\` continuation, and these strings are user-facing.
        let issues = vec![
            CaptureIssue::Unreadable {
                path: "a.png".into(),
                message: "bad".into(),
            },
            CaptureIssue::TooFewPhotos {
                given: 2,
                required: 3,
            },
            CaptureIssue::MixedResolutions {
                common: (4000, 3000),
                odd: (2000, 1500),
                sizes: 2,
            },
            CaptureIssue::BadScale {
                metres_per_unit: 0.0,
            },
            CaptureIssue::ViewNotRegistered {
                view: 3,
                photo: "d.jpg".into(),
                correspondences: 4,
            },
            CaptureIssue::ThinRegistration {
                view: 2,
                photo: "c.jpg".into(),
                inliers: 9,
            },
            CaptureIssue::SingleCoverage {
                triangles: 10,
                examined: 100,
            },
            CaptureIssue::NoCoverage {
                triangles: 5,
                examined: 100,
            },
            CaptureIssue::NoMeshletDag,
        ];
        for issue in &issues {
            let text = issue.to_string();
            assert!(
                !text.contains("  "),
                "a run of spaces in {issue:?}: {text:?}"
            );
            assert!(
                text.contains(" — ") || text.contains("; "),
                "{issue:?} carries no remedy: {text:?}"
            );
        }
        // The two pass-throughs render their source's words and add none.
        let sparse = CaptureIssue::Sparse {
            message: "anything at all".into(),
        };
        assert_eq!(sparse.to_string(), "anything at all");
    }

    #[test]
    fn the_severity_and_stage_of_every_issue_is_the_one_it_is_raised_at() {
        assert!(CaptureIssue::TooFewPhotos {
            given: 0,
            required: 3
        }
        .blocks());
        assert!(CaptureIssue::BadScale {
            metres_per_unit: -1.0
        }
        .blocks());
        assert!(CaptureIssue::Unreadable {
            path: "x".into(),
            message: "y".into()
        }
        .blocks());
        // A coverage warning must NOT block: the scan exists and is worse than
        // it looks, which is a different thing from unbuildable.
        let single = CaptureIssue::SingleCoverage {
            triangles: 1,
            examined: 2,
        };
        assert!(!single.blocks());
        assert_eq!(single.stage(), CaptureStage::Finish);
        assert_eq!(CaptureIssue::NoMeshletDag.severity(), CaptureSeverity::Note);
        assert_eq!(CaptureIssue::NoMeshletDag.stage(), CaptureStage::Write);
    }

    #[test]
    fn a_session_with_no_photographs_refuses_to_start_and_names_the_shortfall() {
        let mut session = PhotogrammetrySession::new();
        let issues = session.load_photos(&[]).expect("nothing is running");
        assert!(issues
            .iter()
            .any(|i| matches!(i, CaptureIssue::TooFewPhotos { given: 0, .. })));
        let err = session
            .start(Arc::new(JobPool::new(1)))
            .expect_err("an empty set must refuse");
        let text = err.to_string();
        assert!(text.contains("at least 3"), "{text}");
        assert_eq!(session.state(), CaptureState::Idle);
    }

    #[test]
    fn an_unreadable_file_is_named_and_the_readable_ones_survive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = dir.path().join("ok.png");
        write_test_png(&good, 8, 8);
        let bad = dir.path().join("broken.png");
        std::fs::write(&bad, b"not a png").expect("write");
        let missing = dir.path().join("gone.png");

        let mut session = PhotogrammetrySession::new();
        let issues = session
            .load_photos(&[good.clone(), bad.clone(), missing.clone()])
            .expect("nothing is running");
        let unreadable: Vec<&CaptureIssue> = issues
            .iter()
            .filter(|i| matches!(i, CaptureIssue::Unreadable { .. }))
            .collect();
        assert_eq!(unreadable.len(), 2, "{issues:?}");
        for issue in unreadable {
            let text = issue.to_string();
            assert!(
                text.contains("broken.png") || text.contains("gone.png"),
                "an unreadable file was not named: {text}"
            );
        }
        // The table keeps a row per input, in order, with the failures marked.
        assert_eq!(session.photos().len(), 3);
        assert!(session.photos()[0].error.is_none());
        assert_eq!(
            (session.photos()[0].width, session.photos()[0].height),
            (8, 8)
        );
        assert!(session.photos()[1].error.is_some());
        assert!(session.photos()[2].error.is_some());
    }

    #[test]
    fn a_mixed_resolution_set_warns_and_names_both_sizes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.png");
        let b = dir.path().join("b.png");
        let c = dir.path().join("c.png");
        write_test_png(&a, 16, 12);
        write_test_png(&b, 16, 12);
        write_test_png(&c, 8, 8);
        let mut session = PhotogrammetrySession::new();
        let issues = session.load_photos(&[a, b, c]).expect("nothing is running");
        let issue = issues
            .iter()
            .find(|i| matches!(i, CaptureIssue::MixedResolutions { .. }))
            .expect("a mixed set must warn");
        let text = issue.to_string();
        assert!(text.contains("16x12"), "{text}");
        assert!(text.contains("8x8"), "{text}");
        // …and a uniform set does not.
        let dir2 = tempfile::tempdir().expect("tempdir");
        let (x, y, z) = (
            dir2.path().join("x.png"),
            dir2.path().join("y.png"),
            dir2.path().join("z.png"),
        );
        for p in [&x, &y, &z] {
            write_test_png(p, 16, 12);
        }
        let issues = session.load_photos(&[x, y, z]).expect("nothing is running");
        assert!(
            !issues
                .iter()
                .any(|i| matches!(i, CaptureIssue::MixedResolutions { .. })),
            "{issues:?}"
        );
    }

    #[test]
    fn a_scale_that_is_not_a_scale_is_caught_before_a_run_starts() {
        let mut session = PhotogrammetrySession::new();
        let mut cfg = CaptureConfig::default();
        cfg.finish.metres_per_unit = 0.0;
        session.set_config(cfg);
        assert!(session
            .preflight()
            .iter()
            .any(|i| matches!(i, CaptureIssue::BadScale { .. })));
        let mut cfg = session.config().clone();
        cfg.finish.metres_per_unit = -1.0;
        session.set_config(cfg);
        assert!(session
            .preflight()
            .iter()
            .any(|i| matches!(i, CaptureIssue::BadScale { .. })));
        let mut cfg = session.config().clone();
        cfg.finish.metres_per_unit = 1.0;
        session.set_config(cfg);
        assert!(!session
            .preflight()
            .iter()
            .any(|i| matches!(i, CaptureIssue::BadScale { .. })));
    }

    #[test]
    fn a_refinish_with_nothing_reconstructed_refuses_by_name() {
        let mut session = PhotogrammetrySession::new();
        let dir = tempfile::tempdir().expect("tempdir");
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| {
                let p = dir.path().join(format!("p{i}.png"));
                write_test_png(&p, 8, 8);
                p
            })
            .collect();
        session.load_photos(&paths).expect("nothing is running");
        let err = session
            .refinish(Arc::new(JobPool::new(1)))
            .expect_err("nothing to re-finish");
        assert!(matches!(err, CaptureError::NoProduct("re-finish")), "{err}");
    }

    #[test]
    fn a_stage_that_panics_settles_the_run_failed_rather_than_running_for_ever() {
        // The worker is the ONLY thing that publishes a terminal state, so an
        // unwind out of a solver used to leave the session `Running` for the
        // life of the process — Cancel answering `true` to nobody, Start
        // refusing `Busy` for ever, and a dialog whose Close button is disabled
        // exactly while that is true. This drives `run_guarded` itself, which is
        // the wrapper `begin` spawns, rather than a copy of it.
        let (tx, rx) = channel();
        let shared: Arc<Mutex<Shared>> = Arc::new(Mutex::new(Shared::default()));
        shared.lock().expect("fresh").state = Some(CaptureState::Running(CaptureStage::Dense));
        run_guarded(7, &tx, &shared, || panic!("the plane sweep tripped"));

        let guard = shared
            .lock()
            .expect("the mutex is taken through the poison");
        assert_eq!(guard.state, Some(CaptureState::Failed));
        let error = guard
            .error
            .clone()
            .expect("a failed run carries its refusal");
        // Named with the stage that was in flight AND the panic's own words.
        assert!(error.contains("dense panicked"), "{error}");
        assert!(error.contains("the plane sweep tripped"), "{error}");
        // Nothing half-built survives.
        assert!(guard.product.is_none());
        drop(guard);

        // …and it reaches the stream, on the stage that raised it, for the run
        // that raised it — the shape the panel already knows how to show.
        let events: Vec<CaptureProgress> = rx.try_iter().collect();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].phase, CapturePhase::Failed);
        assert_eq!(events[0].stage, CaptureStage::Dense);
        assert_eq!(events[0].run, 7);
        assert!(events[0].error.is_some());
    }

    #[test]
    fn a_panic_with_a_formatted_message_keeps_its_words() {
        // `panic!("{}", x)` boxes a `String` and `panic!("literal")` boxes a
        // `&'static str`; both are what a solver actually produces, and a
        // downcast that only handles one reports "a panic with no message" for
        // the other.
        let owned =
            std::panic::catch_unwind(|| panic!("{} views registered", 0)).expect_err("it panicked");
        assert_eq!(panic_message(&*owned), "0 views registered");
        let literal =
            std::panic::catch_unwind(|| panic!("index out of bounds")).expect_err("it panicked");
        assert_eq!(panic_message(&*literal), "index out of bounds");
    }

    #[test]
    fn a_failed_run_keeps_the_findings_the_stages_before_it_produced() {
        // The diagnostics clause's worst case: a run that refuses at the dense
        // solve has no product, so the solver advisories it already collected
        // used to go out with the worker's stack and the panel fell back to a
        // pre-flight that cannot know any of them. `findings` is the one rule
        // that decides which of the three places a finding lives in.
        let mut session = PhotogrammetrySession::new();
        // Nothing loaded, so the pre-flight blocks — the third branch.
        assert!(session.findings().iter().any(|i| i.blocks()));

        let collected = vec![
            CaptureIssue::ViewNotRegistered {
                view: 3,
                photo: "station3.png".into(),
                correspondences: 4,
            },
            CaptureIssue::Dense {
                message: "two views agreed on nothing".into(),
            },
        ];
        {
            let mut guard = session.shared.lock().expect("fresh");
            guard.state = Some(CaptureState::Failed);
            guard.error = Some("dense failed: no depth".into());
            guard.issues = collected.clone();
        }
        let findings = session.findings();
        for issue in &collected {
            assert!(
                findings.contains(issue),
                "a failed run dropped {issue:?}: {findings:?}"
            );
        }
        // …and what is loaded is still said, because it is still loaded.
        assert!(
            findings
                .iter()
                .any(|i| matches!(i, CaptureIssue::TooFewPhotos { .. })),
            "{findings:?}"
        );

        // A new photograph set retires them: they are about the last set. (The
        // remaining branch — a product wins over both — is asserted through the
        // session door in `capture_wizard_gate`, because building a
        // `CaptureProduct` means running the pipeline.)
        session.load_photos(&[]).expect("nothing is running");
        assert!(
            !session
                .findings()
                .iter()
                .any(|i| matches!(i, CaptureIssue::ViewNotRegistered { .. })),
            "a new load kept the last run's findings"
        );
    }

    #[test]
    fn a_load_during_a_run_is_refused_by_name_rather_than_reaching_around_it() {
        // Loading replaces the PUBLISHED state as well as the photographs, so a
        // load during a solve used to leave the session reading `Idle` over a
        // worker still running: Cancel would answer `false`, and minutes later
        // that worker would publish a product for photographs nobody has loaded.
        let dir = tempfile::tempdir().expect("tempdir");
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| {
                let p = dir.path().join(format!("p{i}.png"));
                write_test_png(&p, 24, 24);
                p
            })
            .collect();
        let mut session = PhotogrammetrySession::new();
        session.load_photos(&paths).expect("an idle session loads");
        // Pretend a run owns the session, which is what the worker publishes as
        // it enters its first stage.
        session.shared.lock().expect("fresh").state =
            Some(CaptureState::Running(CaptureStage::Sfm));

        let err = session
            .load_photos(&paths[..1])
            .expect_err("a load during a run must refuse");
        assert!(matches!(err, CaptureError::Busy), "{err}");
        // …and it refused rather than half-applying: the three are still loaded
        // and the run still owns the session.
        assert_eq!(session.photos().len(), 3);
        assert_eq!(session.state(), CaptureState::Running(CaptureStage::Sfm));
    }

    /// A tiny valid PNG, written through the `png` encoder this crate already
    /// depends on.
    fn write_test_png(path: &Path, w: u32, h: u32) {
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, w, h);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().expect("header");
            let pixels: Vec<u8> = (0..(w * h))
                .flat_map(|i| [(i % 251) as u8, (i % 199) as u8, (i % 173) as u8])
                .collect();
            writer.write_image_data(&pixels).expect("data");
        }
        std::fs::write(path, bytes).expect("write png");
    }
}
