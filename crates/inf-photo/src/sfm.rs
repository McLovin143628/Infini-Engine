//! **Incremental structure from motion**: the pipeline that turns a set of
//! photographs into camera poses and a sparse point cloud.
//!
//! # The order of operations
//!
//! 1. **Validate.** Fewer than [`SfmConfig::min_views`] photographs, or
//!    intrinsics that do not describe the image they are attached to, is a named
//!    refusal ([`SfmError`]) rather than a degenerate result.
//! 2. **Extract** features from every view, on the job pool.
//! 3. **Match** every candidate pair, on the job pool, and chain the matches
//!    into tracks.
//! 4. **Initialise** from the best pair: essential matrix under RANSAC,
//!    cheirality-resolved relative pose, and the tracks both views share,
//!    triangulated.
//! 5. **Grow.** Repeatedly pick the unregistered view with the most 2D–3D
//!    correspondences, solve its pose by PnP under RANSAC, refine it, and
//!    triangulate every track that just became visible from two registered
//!    views. Bundle-adjust every [`SfmConfig::bundle_every`] registrations.
//! 6. **Finish.** One last bundle over everything, then prune observations and
//!    points the converged solution disagrees with, then re-impose the gauge.
//!
//! # Why the choice of next view is a *count*
//!
//! Ranking candidate views by how many already-triangulated points they see is
//! the standard incremental heuristic, and it is deterministic in a way that
//! "best score" heuristics involving floating-point uncertainty estimates are
//! not: an integer count with a tie broken by the lower view index is a total
//! order that cannot wobble. It is also the right heuristic — a view seeing many
//! reconstructed points is a view whose pose is well constrained.
//!
//! # The gauge, precisely
//!
//! * The **lowest-indexed registered view** is the world frame: its pose is
//!   exactly [`Pose::IDENTITY`]. It is the camera bundle adjustment holds fixed,
//!   and [`SfmConfig`] never lets that be ambiguous, because view indices are
//!   integers.
//! * The distance from it to the **next-lowest-indexed registered view** is
//!   exactly `1`.
//!
//! Reconstruction coordinates are therefore **baseline units, not metres**; see
//! the crate docs for where SI comes back.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec2, DVec3};
use inf_core::job::JobPool;

use crate::bundle::{bundle_adjust, BundleConfig, BundleObservation, BundleReport};
use crate::camera::{Intrinsics, Pose};
use crate::features::{describe_pyramid, FeatureConfig, FeatureSet};
use crate::geometry::{
    parallax_cos, pose_from_essential, ransac_essential, ransac_pnp, reprojection_error,
    triangulate_dlt, RansacConfig,
};
use crate::gray::{GrayImage, Pyramid};
use crate::hash::Hash64;
use crate::matching::{build_tracks, match_all, MatchConfig, Observation, PairMatches, Track};
use crate::{Advisory, SfmError};

/// One input photograph and what is known about the camera that took it.
#[derive(Clone, Debug)]
pub struct View {
    /// The photograph, already grayscale.
    pub image: GrayImage,
    /// The camera's intrinsics. An **input**; never refined.
    pub intrinsics: Intrinsics,
}

impl View {
    /// A view whose intrinsics are a centred pinhole of the given focal length
    /// in **pixels** — the honest fallback when no calibration is available.
    pub fn assumed(image: GrayImage, focal_px: f64) -> Self {
        let intrinsics = Intrinsics::centred(image.width(), image.height(), focal_px);
        Self { image, intrinsics }
    }
}

/// How the whole pipeline is tuned. Every field is a **committed constant** in
/// the sense that changing it changes reconstructions; the defaults are what the
/// gate measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SfmConfig {
    /// Feature extraction.
    pub features: FeatureConfig,
    /// Descriptor matching.
    pub matching: MatchConfig,
    /// The robust estimators.
    pub ransac: RansacConfig,
    /// Bundle adjustment.
    pub bundle: BundleConfig,
    /// The seed every RANSAC draws from. Committed; changing it changes every
    /// reconstruction this crate has ever produced.
    pub seed: u64,
    /// Fewer photographs than this is a refusal.
    pub min_views: usize,
    /// A track shorter than this is not carried.
    pub min_track_length: usize,
    /// How many candidate pairs are tried for the initialisation.
    pub init_candidates: usize,
    /// A triangulated point whose widest ray pair has a cosine at or above this
    /// has no usable depth. `cos(2°) ≈ 0.99939`.
    pub max_parallax_cos: f64,
    /// Observations further than this from their point are dropped, in
    /// **pixels**.
    pub max_reprojection_px: f64,
    /// Bundle-adjust after this many registrations.
    pub bundle_every: usize,
    /// A view registering on fewer inliers than this earns an advisory.
    pub thin_registration_inliers: usize,
}

impl Default for SfmConfig {
    fn default() -> Self {
        Self {
            features: FeatureConfig::default(),
            matching: MatchConfig::default(),
            ransac: RansacConfig::default(),
            bundle: BundleConfig::default(),
            seed: 0x494e_465f_5044_4d47,
            min_views: 3,
            min_track_length: 2,
            init_candidates: 4,
            max_parallax_cos: 0.999_390,
            max_reprojection_px: 4.0,
            bundle_every: 2,
            thin_registration_inliers: 25,
        }
    }
}

/// A camera that got a pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisteredCamera {
    /// Which input view.
    pub view: u32,
    /// The pose, world-to-camera, in **baseline units**.
    pub pose: Pose,
    /// The intrinsics it was reconstructed with (copied, so a consumer needs
    /// nothing but this struct).
    pub intrinsics: Intrinsics,
    /// How many observations survived to the end.
    pub observations: usize,
}

/// One triangulated point.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenePoint {
    /// Position in **baseline units**.
    pub position: DVec3,
    /// The track it came from.
    pub track: u32,
    /// Which views see it, ascending.
    pub observations: Vec<Observation>,
}

/// What the pipeline did, in numbers a diagnostics panel can show.
#[derive(Clone, Debug, PartialEq)]
pub struct SfmReport {
    /// How many views were supplied.
    pub views: usize,
    /// How many got a pose.
    pub registered: usize,
    /// Features found per view, in view order.
    pub features_per_view: Vec<usize>,
    /// How many view pairs survived matching.
    pub pairs: usize,
    /// How many correspondences survived matching, in total.
    pub matches: usize,
    /// How many tracks were built.
    pub tracks: usize,
    /// How many points survived to the end.
    pub points: usize,
    /// How many observations survived to the end.
    pub observations: usize,
    /// Final reprojection RMS, in **pixels**.
    pub reprojection_rms_px: f64,
    /// The pair the reconstruction started from.
    pub initial_pair: (u32, u32),
    /// The final bundle's report.
    pub final_bundle: BundleReport,
}

/// Poses, points and the tale of how they were found.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    /// Registered cameras, keyed by view index.
    pub cameras: BTreeMap<u32, RegisteredCamera>,
    /// Points, ordered by their track index.
    pub points: Vec<ScenePoint>,
    /// Non-fatal findings, in canonical order.
    pub advisories: Vec<Advisory>,
    /// The numbers.
    pub report: SfmReport,
}

impl Reconstruction {
    /// The reconstruction's **canonical byte image**: the assertion surface for
    /// determinism.
    ///
    /// Everything that defines the result is written, in a fixed order, as
    /// little-endian IEEE bit patterns — never as formatted decimals, which
    /// would hide a one-ULP difference. Two runs of [`reconstruct`] on the same
    /// input produce identical bytes here, and so do runs on pools of different
    /// sizes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"INF_PHOTO_SFM\x01");
        out.extend_from_slice(&(self.cameras.len() as u64).to_le_bytes());
        for (view, cam) in &self.cameras {
            out.extend_from_slice(&view.to_le_bytes());
            cam.pose.write_canonical(&mut out);
            cam.intrinsics.write_canonical(&mut out);
            out.extend_from_slice(&(cam.observations as u64).to_le_bytes());
        }
        out.extend_from_slice(&(self.points.len() as u64).to_le_bytes());
        for p in &self.points {
            out.extend_from_slice(&p.track.to_le_bytes());
            for v in p.position.to_array() {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            out.extend_from_slice(&(p.observations.len() as u64).to_le_bytes());
            for o in &p.observations {
                out.extend_from_slice(&o.view.to_le_bytes());
                out.extend_from_slice(&o.feature.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.advisories.len() as u64).to_le_bytes());
        for a in &self.advisories {
            out.extend_from_slice(format!("{a}").as_bytes());
            out.push(0);
        }
        out.extend_from_slice(&self.report.reprojection_rms_px.to_bits().to_le_bytes());
        out
    }

    /// The registered poses, in ascending view order — what
    /// [`crate::align::pose_error`] and the P25.2 dense stage both want.
    pub fn poses(&self) -> Vec<Pose> {
        self.cameras.values().map(|c| c.pose).collect()
    }

    /// The registered view indices, ascending.
    pub fn registered_views(&self) -> Vec<u32> {
        self.cameras.keys().copied().collect()
    }
}

/// Run the whole pipeline.
pub fn reconstruct(
    views: &[View],
    cfg: &SfmConfig,
    pool: &JobPool,
) -> Result<Reconstruction, SfmError> {
    if views.len() < cfg.min_views {
        return Err(SfmError::TooFewViews {
            given: views.len(),
            required: cfg.min_views,
        });
    }
    for (i, v) in views.iter().enumerate() {
        if v.intrinsics.width != v.image.width() || v.intrinsics.height != v.image.height() {
            return Err(SfmError::IntrinsicsMismatch {
                view: i,
                img_w: v.image.width(),
                img_h: v.image.height(),
                int_w: v.intrinsics.width,
                int_h: v.intrinsics.height,
            });
        }
    }

    // ── 1. Features ────────────────────────────────────────────────────────
    let fcfg = cfg.features;
    let sets: Vec<FeatureSet> = pool.parallel_map_ref(views, |v| {
        // `Pyramid::build` only fails on a zero dimension, which the loop above
        // has already excluded, so an empty set here is a texture-poor
        // photograph rather than an error.
        match Pyramid::build(
            &v.image,
            fcfg.levels,
            fcfg.scale_factor,
            fcfg.min_level_side,
        ) {
            Ok(p) => describe_pyramid(&p, &fcfg),
            Err(_) => FeatureSet::default(),
        }
    });
    let features_per_view: Vec<usize> = sets.iter().map(|s| s.len()).collect();

    // ── 2. Matching and tracks ─────────────────────────────────────────────
    let (pairs, mut advisories) = match_all(&sets, &cfg.matching, pool);
    let total_matches: usize = pairs.iter().map(|p| p.matches.len()).sum();
    let (tracks, track_advisories) = build_tracks(&sets, &pairs, cfg.min_track_length);
    advisories.extend(track_advisories);

    // Pixel and normalized coordinates, once. Every estimator downstream works
    // in normalized coordinates and every residual is reported in pixels;
    // computing them per hypothesis would be slow and, worse, would be two code
    // paths that could disagree.
    let pixels: Vec<Vec<DVec2>> = sets
        .iter()
        .map(|s| s.features.iter().map(|f| DVec2::new(f.x, f.y)).collect())
        .collect();
    let normalized: Vec<Vec<DVec2>> = views
        .iter()
        .zip(&pixels)
        .map(|(v, px)| px.iter().map(|p| v.intrinsics.to_normalized(*p)).collect())
        .collect();

    // ── 3. Initialisation ──────────────────────────────────────────────────
    let init = choose_initial_pair(&pairs, &normalized, views, cfg, pool)?;
    let mut state = State::new(views, &tracks, pixels);
    state.register(init.view_a, Pose::IDENTITY);
    state.register(init.view_b, init.pose);

    let seeded = state.triangulate_pending(&normalized, views, cfg);
    if seeded < MIN_SEED_POINTS {
        return Err(SfmError::InitializationFailed {
            a: init.view_a,
            b: init.view_b,
            points: seeded,
            required: MIN_SEED_POINTS,
        });
    }

    // ── 4. Grow ────────────────────────────────────────────────────────────
    let mut since_bundle = 0usize;
    while let Some((next, correspondences)) = state.best_unregistered() {
        if correspondences < cfg.ransac.min_pnp_inliers {
            for view in state.unregistered() {
                advisories.push(Advisory::ViewNotRegistered {
                    view,
                    correspondences: state.correspondence_count(view),
                });
            }
            break;
        }
        let (world, obs, _tracks) = state.correspondences(next, &normalized);
        let threshold = cfg.ransac.pnp_threshold_px / views[next as usize].intrinsics.mean_focal();
        let seed = Hash64::new(cfg.seed)
            .mix_u64(0x504e_5000)
            .mix_u64(next as u64)
            .finish();
        match ransac_pnp(&world, &obs, threshold, &cfg.ransac, seed, pool) {
            Some(m) if m.inliers.len() >= cfg.ransac.min_pnp_inliers => {
                if m.inliers.len() < cfg.thin_registration_inliers {
                    advisories.push(Advisory::ThinRegistration {
                        view: next,
                        inliers: m.inliers.len(),
                    });
                }
                state.register(next, m.pose);
                state.triangulate_pending(&normalized, views, cfg);
                since_bundle += 1;
                if since_bundle >= cfg.bundle_every {
                    state.bundle(views, cfg, pool);
                    state.prune(views, cfg);
                    since_bundle = 0;
                }
            }
            _ => {
                advisories.push(Advisory::ViewNotRegistered {
                    view: next,
                    correspondences,
                });
                state.give_up(next);
            }
        }
    }

    if state.registered_count() < cfg.min_views {
        return Err(SfmError::TooFewRegistered {
            registered: state.registered_count(),
            given: views.len(),
            required: cfg.min_views,
        });
    }

    // ── 5. Finish ──────────────────────────────────────────────────────────
    let final_bundle = state.bundle(views, cfg, pool);
    let pruned = state.prune(views, cfg);
    if pruned > 0 {
        advisories.push(Advisory::PrunedObservations {
            dropped: pruned,
            threshold_px: cfg.max_reprojection_px,
        });
    }
    state.normalize_gauge();
    let rms = state.reprojection_rms(views);

    advisories.sort();
    advisories.dedup();

    let cameras = state.finish_cameras(views);
    let points = state.finish_points();
    let observations = points.iter().map(|p| p.observations.len()).sum();
    Ok(Reconstruction {
        report: SfmReport {
            views: views.len(),
            registered: cameras.len(),
            features_per_view,
            pairs: pairs.len(),
            matches: total_matches,
            tracks: tracks.len(),
            points: points.len(),
            observations,
            reprojection_rms_px: rms,
            initial_pair: (init.view_a, init.view_b),
            final_bundle,
        },
        cameras,
        points,
        advisories,
    })
}

/// The initial pair must seed at least this many points, or there is nothing to
/// register a third view against.
const MIN_SEED_POINTS: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

struct InitialPair {
    view_a: u32,
    view_b: u32,
    pose: Pose,
}

/// Pick the pair to start from.
///
/// Candidates are the [`SfmConfig::init_candidates`] pairs with the most
/// matches — a pair with few correspondences cannot produce a well-conditioned
/// essential matrix however good those correspondences are. Each candidate is
/// scored by its **essential inlier count**, and a pair whose points are mostly
/// at near-zero parallax scores zero however many inliers it has, because a
/// rotation-only pair fits an epipolar model perfectly and reconstructs nothing.
/// Ties go to the pair with more matches, then to the lower `(a, b)`.
fn choose_initial_pair(
    pairs: &[PairMatches],
    normalized: &[Vec<DVec2>],
    views: &[View],
    cfg: &SfmConfig,
    pool: &JobPool,
) -> Result<InitialPair, SfmError> {
    let mut ranked: Vec<&PairMatches> = pairs.iter().collect();
    ranked.sort_by(|a, b| {
        b.matches
            .len()
            .cmp(&a.matches.len())
            .then(a.view_a.cmp(&b.view_a))
            .then(a.view_b.cmp(&b.view_b))
    });

    let mut best: Option<(usize, InitialPair)> = None;
    let mut best_seen = 0usize;
    let examined = ranked.len().min(cfg.init_candidates);
    for pair in ranked.iter().take(cfg.init_candidates) {
        let a: Vec<DVec2> = pair
            .matches
            .iter()
            .map(|m| normalized[pair.view_a as usize][m.a as usize])
            .collect();
        let b: Vec<DVec2> = pair
            .matches
            .iter()
            .map(|m| normalized[pair.view_b as usize][m.b as usize])
            .collect();
        let focal = 0.5
            * (views[pair.view_a as usize].intrinsics.mean_focal()
                + views[pair.view_b as usize].intrinsics.mean_focal());
        let threshold = cfg.ransac.essential_threshold_px / focal;
        let seed = Hash64::new(cfg.seed)
            .mix_u64(0x4553_5300)
            .mix_u64(pair.view_a as u64)
            .mix_u64(pair.view_b as u64)
            .finish();
        let Some(model) = ransac_essential(&a, &b, threshold, &cfg.ransac, seed, pool) else {
            continue;
        };
        best_seen = best_seen.max(model.inliers.len());
        if model.inliers.len() < cfg.ransac.min_essential_inliers {
            continue;
        }
        let Some((pose, in_front)) = pose_from_essential(&model.essential, &a, &b, &model.inliers)
        else {
            continue;
        };
        let poses = [Pose::IDENTITY, pose];
        let mut with_parallax = 0usize;
        for &i in &model.inliers {
            let obs = [(Pose::IDENTITY, a[i as usize]), (pose, b[i as usize])];
            if let Some(x) = triangulate_dlt(&obs) {
                if parallax_cos(x, &poses) < cfg.max_parallax_cos {
                    with_parallax += 1;
                }
            }
        }
        if with_parallax * 2 < in_front {
            continue;
        }
        let score = model.inliers.len();
        if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((
                score,
                InitialPair {
                    view_a: pair.view_a,
                    view_b: pair.view_b,
                    pose,
                },
            ));
        }
    }

    best.map(|(_, p)| p).ok_or(SfmError::NoViableInitialPair {
        examined,
        best_inliers: best_seen,
        required: cfg.ransac.min_essential_inliers,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// The growing reconstruction
// ─────────────────────────────────────────────────────────────────────────────

/// A track that has been triangulated.
#[derive(Clone, Debug)]
struct LivePoint {
    track: u32,
    position: DVec3,
    observations: Vec<Observation>,
}

struct State<'a> {
    tracks: &'a [Track],
    /// Feature pixel coordinates, per view, in level-0 pixels.
    pixels: Vec<Vec<DVec2>>,
    /// Registered poses, keyed by view.
    poses: BTreeMap<u32, Pose>,
    /// Views PnP has already failed on, so the loop cannot spin on them.
    abandoned: BTreeSet<u32>,
    /// Triangulated tracks, keyed by track index (so iteration is canonical).
    points: BTreeMap<u32, LivePoint>,
    /// Which tracks each view participates in, ascending.
    tracks_of_view: Vec<Vec<u32>>,
    n_views: usize,
}

impl<'a> State<'a> {
    fn new(views: &[View], tracks: &'a [Track], pixels: Vec<Vec<DVec2>>) -> Self {
        let mut tracks_of_view = vec![Vec::new(); views.len()];
        for (ti, t) in tracks.iter().enumerate() {
            for o in &t.observations {
                tracks_of_view[o.view as usize].push(ti as u32);
            }
        }
        Self {
            tracks,
            pixels,
            poses: BTreeMap::new(),
            abandoned: BTreeSet::new(),
            points: BTreeMap::new(),
            tracks_of_view,
            n_views: views.len(),
        }
    }

    fn register(&mut self, view: u32, pose: Pose) {
        self.poses.insert(view, pose);
    }

    fn give_up(&mut self, view: u32) {
        self.abandoned.insert(view);
    }

    fn registered_count(&self) -> usize {
        self.poses.len()
    }

    fn unregistered(&self) -> Vec<u32> {
        (0..self.n_views as u32)
            .filter(|v| !self.poses.contains_key(v))
            .collect()
    }

    /// How many already-triangulated points a view can see.
    fn correspondence_count(&self, view: u32) -> usize {
        self.tracks_of_view[view as usize]
            .iter()
            .filter(|t| self.points.contains_key(t))
            .count()
    }

    /// The unregistered, un-abandoned view seeing the most points. Ties go to
    /// the lower view index — an integer order, so it cannot wobble.
    fn best_unregistered(&self) -> Option<(u32, usize)> {
        let mut best: Option<(u32, usize)> = None;
        for view in 0..self.n_views as u32 {
            if self.poses.contains_key(&view) || self.abandoned.contains(&view) {
                continue;
            }
            let count = self.correspondence_count(view);
            if best.map(|(_, c)| count > c).unwrap_or(true) {
                best = Some((view, count));
            }
        }
        best
    }

    /// The 2D–3D correspondences a view has, in track order.
    fn correspondences(
        &self,
        view: u32,
        normalized: &[Vec<DVec2>],
    ) -> (Vec<DVec3>, Vec<DVec2>, Vec<u32>) {
        let mut world = Vec::new();
        let mut obs = Vec::new();
        let mut track_ids = Vec::new();
        for &t in &self.tracks_of_view[view as usize] {
            let Some(point) = self.points.get(&t) else {
                continue;
            };
            let Some(feature) = self.tracks[t as usize].observation_in(view) else {
                continue;
            };
            world.push(point.position);
            obs.push(normalized[view as usize][feature as usize]);
            track_ids.push(t);
        }
        (world, obs, track_ids)
    }

    /// **Point continuation**: give every already-triangulated point the
    /// observations of views that have registered since it was created, then
    /// re-triangulate it from all of them.
    ///
    /// Without this the pipeline still runs and still registers every view — and
    /// produces a reconstruction in which **every point has exactly two
    /// observations**, because a point's observation list would be frozen at the
    /// moment it was first triangulated. That was measured on the gate fixture
    /// before this existed: 435 points, 870 observations, a reprojection RMS of
    /// 0.31 px that looked perfectly healthy, and camera rotations **5.6 degrees**
    /// out. A two-view point constrains the pair that made it and contributes
    /// nothing to any later camera; bundle adjustment over such a reconstruction
    /// is exactly determined and converges in one iteration having changed
    /// nothing. Multi-view points are where a bundle's redundancy comes from, and
    /// this is the function that creates them.
    ///
    /// A new observation must reproject inside
    /// [`SfmConfig::max_reprojection_px`] against the point's *current*
    /// position before it is accepted, so a bad match cannot drag a good point.
    fn continue_points(
        &mut self,
        normalized: &[Vec<DVec2>],
        views: &[View],
        cfg: &SfmConfig,
    ) -> usize {
        let poses = &self.poses;
        let tracks = self.tracks;
        let mut extended = 0usize;
        for (ti, point) in self.points.iter_mut() {
            let track = &tracks[*ti as usize];
            let mut grew = false;
            for o in &track.observations {
                let Some(pose) = poses.get(&o.view) else {
                    continue;
                };
                if point.observations.iter().any(|e| e.view == o.view) {
                    continue;
                }
                if pose.transform(point.position).z <= 0.0 {
                    continue;
                }
                let threshold =
                    cfg.max_reprojection_px / views[o.view as usize].intrinsics.mean_focal();
                let obs = normalized[o.view as usize][o.feature as usize];
                if reprojection_error(pose, point.position, obs) > threshold {
                    continue;
                }
                point.observations.push(*o);
                grew = true;
            }
            if !grew {
                continue;
            }
            extended += 1;
            point.observations.sort_unstable();
            let rays: Vec<(Pose, DVec2)> = point
                .observations
                .iter()
                .map(|o| {
                    (
                        poses[&o.view],
                        normalized[o.view as usize][o.feature as usize],
                    )
                })
                .collect();
            if let Some(x) = triangulate_dlt(&rays) {
                point.position = x;
            }
        }
        extended
    }

    /// Triangulate every track visible from at least two registered views that
    /// has not been triangulated yet. Returns how many were added.
    fn triangulate_pending(
        &mut self,
        normalized: &[Vec<DVec2>],
        views: &[View],
        cfg: &SfmConfig,
    ) -> usize {
        self.continue_points(normalized, views, cfg);
        let mut added = 0usize;
        for (ti, track) in self.tracks.iter().enumerate() {
            let ti = ti as u32;
            if self.points.contains_key(&ti) {
                continue;
            }
            let mut rays: Vec<(Pose, DVec2)> = Vec::new();
            let mut observations: Vec<Observation> = Vec::new();
            for o in &track.observations {
                let Some(pose) = self.poses.get(&o.view) else {
                    continue;
                };
                rays.push((*pose, normalized[o.view as usize][o.feature as usize]));
                observations.push(*o);
            }
            if rays.len() < 2 {
                continue;
            }
            let Some(position) = triangulate_dlt(&rays) else {
                continue;
            };
            let poses: Vec<Pose> = rays.iter().map(|(p, _)| *p).collect();
            if parallax_cos(position, &poses) >= cfg.max_parallax_cos {
                continue;
            }
            // Cheirality and reprojection, checked against **every** view that
            // saw it. A point accepted on one pair and wrong for a third view is
            // exactly how a bundle gets poisoned.
            let mut ok = true;
            for (o, (pose, obs)) in observations.iter().zip(&rays) {
                let k = &views[o.view as usize].intrinsics;
                if pose.transform(position).z <= 0.0 {
                    ok = false;
                    break;
                }
                let threshold = cfg.max_reprojection_px / k.mean_focal();
                if reprojection_error(pose, position, *obs) > threshold {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            self.points.insert(
                ti,
                LivePoint {
                    track: ti,
                    position,
                    observations,
                },
            );
            added += 1;
        }
        added
    }

    /// Assemble the bundle problem, solve it, and write the result back.
    ///
    /// The parameter block is ordered by view index, so
    /// [`BundleConfig::fixed_cameras`] pins the **lowest-indexed registered
    /// view** — which is the gauge this module documents.
    fn bundle(&mut self, views: &[View], cfg: &SfmConfig, pool: &JobPool) -> BundleReport {
        let view_ids: Vec<u32> = self.poses.keys().copied().collect();
        if view_ids.len() < 2 || self.points.is_empty() {
            return BundleReport::empty();
        }
        let index_of: BTreeMap<u32, u32> = view_ids
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();

        let mut cameras: Vec<Pose> = view_ids.iter().map(|v| self.poses[v]).collect();
        let intrinsics: Vec<Intrinsics> = view_ids
            .iter()
            .map(|v| views[*v as usize].intrinsics)
            .collect();
        let track_ids: Vec<u32> = self.points.keys().copied().collect();
        let mut points: Vec<DVec3> = track_ids.iter().map(|t| self.points[t].position).collect();

        let mut observations = Vec::new();
        for (pi, t) in track_ids.iter().enumerate() {
            for o in &self.points[t].observations {
                let Some(&ci) = index_of.get(&o.view) else {
                    continue;
                };
                observations.push(BundleObservation {
                    camera: ci,
                    point: pi as u32,
                    pixel: self.pixels[o.view as usize][o.feature as usize],
                });
            }
        }
        if observations.is_empty() {
            return BundleReport::empty();
        }

        let report = bundle_adjust(
            &mut cameras,
            &intrinsics,
            &mut points,
            &observations,
            &cfg.bundle,
            pool,
        );
        for (i, v) in view_ids.iter().enumerate() {
            self.poses.insert(*v, cameras[i]);
        }
        for (i, t) in track_ids.iter().enumerate() {
            if let Some(p) = self.points.get_mut(t) {
                p.position = points[i];
            }
        }
        report
    }

    /// Drop observations the converged solution disagrees with, then drop any
    /// point that lost too many. Returns how many observations went.
    fn prune(&mut self, views: &[View], cfg: &SfmConfig) -> usize {
        let poses = &self.poses;
        let pixels = &self.pixels;
        let mut dropped = 0usize;
        let mut dead: Vec<u32> = Vec::new();
        for (ti, point) in self.points.iter_mut() {
            let before = point.observations.len();
            let position = point.position;
            point.observations.retain(|o| {
                let Some(pose) = poses.get(&o.view) else {
                    return false;
                };
                let camera_space = pose.transform(position);
                if camera_space.z <= 0.0 {
                    return false;
                }
                match views[o.view as usize].intrinsics.project(camera_space) {
                    Some(px) => {
                        (px - pixels[o.view as usize][o.feature as usize]).length()
                            <= cfg.max_reprojection_px
                    }
                    None => false,
                }
            });
            dropped += before - point.observations.len();
            if point.observations.len() < cfg.min_track_length {
                dead.push(*ti);
            }
        }
        for t in dead {
            self.points.remove(&t);
        }
        dropped
    }

    /// Re-impose the gauge: the lowest-indexed registered camera at the
    /// identity, and a unit baseline to the next-lowest.
    fn normalize_gauge(&mut self) {
        let ids: Vec<u32> = self.poses.keys().copied().collect();
        if ids.len() < 2 {
            return;
        }
        let first = self.poses[&ids[0]];
        // New world coordinates are the first camera's frame: X' = R0 X + t0,
        // so R_i' = R_i R0⁻¹ and t_i' = t_i - R_i' t0.
        let inverse = first.rotation.inverse();
        for id in &ids {
            let pose = self.poses[id];
            let rotation = pose.rotation * inverse;
            let translation = pose.translation - rotation * first.translation;
            self.poses.insert(*id, Pose::from_qt(rotation, translation));
        }
        for p in self.points.values_mut() {
            p.position = first.rotation * p.position + first.translation;
        }
        // The gauge camera is set *exactly*, not computed. `R0 R0⁻¹` is the
        // identity in algebra and a quaternion a few ULP away from it in `f64`,
        // which left the first camera's translation at 1e-16 rather than zero —
        // measured, on the arm that asserts the gauge is where the module says
        // it is. A documented convention that is only nearly true is a
        // convention nobody downstream can rely on.
        self.poses.insert(ids[0], Pose::IDENTITY);

        // Then scale so the baseline to the second camera is exactly one. Scale
        // is an exact null direction of reprojection cost, so this costs the fit
        // nothing (see `Pose::scaled`).
        let baseline = self.poses[&ids[1]].center().length();
        if baseline > 1e-12 && baseline.is_finite() {
            let s = 1.0 / baseline;
            for id in &ids {
                let pose = self.poses[id];
                self.poses.insert(*id, pose.scaled(s));
            }
            for p in self.points.values_mut() {
                p.position *= s;
            }
        }
    }

    fn reprojection_rms(&self, views: &[View]) -> f64 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for point in self.points.values() {
            for o in &point.observations {
                let Some(pose) = self.poses.get(&o.view) else {
                    continue;
                };
                let k = &views[o.view as usize].intrinsics;
                if let Some(px) = k.project(pose.transform(point.position)) {
                    let d = (px - self.pixels[o.view as usize][o.feature as usize]).length();
                    sum += d * d;
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            (sum / n as f64).sqrt()
        }
    }

    fn finish_cameras(&self, views: &[View]) -> BTreeMap<u32, RegisteredCamera> {
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for p in self.points.values() {
            for o in &p.observations {
                *counts.entry(o.view).or_insert(0) += 1;
            }
        }
        self.poses
            .iter()
            .map(|(&view, &pose)| {
                (
                    view,
                    RegisteredCamera {
                        view,
                        pose,
                        intrinsics: views[view as usize].intrinsics,
                        observations: counts.get(&view).copied().unwrap_or(0),
                    },
                )
            })
            .collect()
    }

    fn finish_points(&self) -> Vec<ScenePoint> {
        self.points
            .values()
            .map(|p| ScenePoint {
                position: p.position,
                track: p.track,
                observations: p.observations.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_defaults_are_the_ones_the_gate_measured() {
        // `SfmConfig::default()` is not a convenience — it is the configuration
        // `crates/inf-photo/tests/sfm_gate.rs` states its numbers for, and every
        // field of it changes reconstructions. Pinned field by field, including
        // the nested configs, so a "harmless" default tweak is a red test with
        // the gate's measured block beside it rather than a silent drift in
        // every number this crate has ever reported.
        //
        // Measured: turning `matching.mutual` off — a plausible speed-up —
        // leaves every arm of the gate green. This is the arm that sees it.
        let c = SfmConfig::default();
        assert_eq!(c.seed, 0x494e_465f_5044_4d47);
        assert_eq!(c.min_views, 3);
        assert_eq!(c.min_track_length, 2);
        assert_eq!(c.init_candidates, 4);
        assert_eq!(c.max_parallax_cos, 0.999_390);
        assert_eq!(c.max_reprojection_px, 4.0);
        assert_eq!(c.bundle_every, 2);
        assert_eq!(c.thin_registration_inliers, 25);

        assert_eq!(c.features.levels, 4);
        assert_eq!(c.features.scale_factor, 1.2);
        assert_eq!(c.features.min_level_side, 64);
        assert_eq!(c.features.fast_threshold, 18);
        assert_eq!(c.features.harris_k, 0.04);
        assert_eq!(c.features.max_features, 512);

        assert_eq!(c.matching.ratio, 0.8);
        assert_eq!(c.matching.max_distance, 80);
        assert!(
            c.matching.mutual,
            "the mutual-best check is part of the default"
        );
        assert_eq!(c.matching.min_pair_matches, 16);
        assert_eq!(c.matching.exhaustive_view_cap, 60);
        assert_eq!(c.matching.window, 8);

        assert_eq!(c.ransac.essential_iterations, 384);
        assert_eq!(c.ransac.essential_threshold_px, 1.5);
        assert_eq!(c.ransac.pnp_iterations, 192);
        assert_eq!(c.ransac.pnp_threshold_px, 3.0);
        assert_eq!(c.ransac.min_essential_inliers, 30);
        assert_eq!(c.ransac.min_pnp_inliers, 12);
        assert_eq!(c.ransac.refine_iterations, 20);
        assert_eq!(c.ransac.refine_huber_px, 2.0);

        assert_eq!(c.bundle.max_iterations, 25);
        assert_eq!(c.bundle.max_damping_retries, 8);
        assert_eq!(c.bundle.initial_lambda, 1e-4);
        assert_eq!(c.bundle.huber_delta_px, 2.0);
        assert_eq!(c.bundle.relative_tolerance, 1e-10);
        assert_eq!(c.bundle.fixed_cameras, 1);
    }

    #[test]
    fn too_few_views_is_a_named_refusal() {
        let img = GrayImage::new(64, 64).unwrap();
        let views = vec![View::assumed(img.clone(), 50.0), View::assumed(img, 50.0)];
        let err = reconstruct(&views, &SfmConfig::default(), &JobPool::new(1)).unwrap_err();
        assert_eq!(
            err,
            SfmError::TooFewViews {
                given: 2,
                required: 3
            }
        );
        assert!(format!("{err}").contains("at least 3 views"));
    }

    #[test]
    fn mismatched_intrinsics_are_a_named_refusal() {
        let img = GrayImage::new(64, 48).unwrap();
        let mut views = vec![
            View::assumed(img.clone(), 50.0),
            View::assumed(img.clone(), 50.0),
            View::assumed(img, 50.0),
        ];
        views[1].intrinsics.width = 65;
        let err = reconstruct(&views, &SfmConfig::default(), &JobPool::new(1)).unwrap_err();
        assert!(matches!(err, SfmError::IntrinsicsMismatch { view: 1, .. }));
        assert!(format!("{err}").contains("65"));
    }

    #[test]
    fn featureless_photographs_refuse_by_name_rather_than_reconstructing_noise() {
        let mut img = GrayImage::new(160, 120).unwrap();
        for y in 0..120 {
            for x in 0..160 {
                img.set(x, y, 128);
            }
        }
        let views: Vec<View> = (0..4).map(|_| View::assumed(img.clone(), 150.0)).collect();
        let err = reconstruct(&views, &SfmConfig::default(), &JobPool::new(1)).unwrap_err();
        assert!(
            matches!(err, SfmError::NoViableInitialPair { .. }),
            "expected a named refusal, got {err:?}"
        );
    }
}
