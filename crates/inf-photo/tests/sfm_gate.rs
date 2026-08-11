//! The P25.1 gate: **photographs in, poses and points out, within bounds, the
//! same bytes every time.**
//!
//! The dataset is `inf_photo::synth`'s alcove — three textured planes rendered
//! from six lopsided camera stations by a deterministic analytic renderer. The
//! poses are *given*, so the reconstruction can be measured rather than merely
//! sanity-checked, and there is no GPU, no adapter and no committed image file
//! anywhere in it.
//!
//! # What limits the numbers, measured rather than assumed
//!
//! `the_solver_reaches_the_data_s_own_noise_floor` is the arm that says what the
//! bounds below are really made of. It triangulates every track using the
//! **truth** poses and measures how well those poses explain the *detected*
//! feature positions: 0.39 px RMS. That is not the solver's error — it is the
//! feature detector's, and no reconstruction can beat it. The absolute
//! reprojection bound is therefore a bound on the *fixture*, and the arm that
//! actually holds the solver to account is the one comparing it against that
//! floor.
//!
//! Every arm here is mutation-verified; the batch report records which arms go
//! red for which severing.

use std::sync::OnceLock;

use glam::DVec2;
use inf_core::job::JobPool;
use inf_photo::align::pose_error;
use inf_photo::geometry::triangulate_dlt;
use inf_photo::matching::{build_tracks, match_all};
use inf_photo::synth::{SynthConfig, SynthDataset};
use inf_photo::{
    detect_and_describe, reconstruct, FeatureConfig, MatchConfig, Reconstruction, SfmConfig,
    SfmError,
};

/// The gate's dataset. Rendering six 320x240 views costs a couple of seconds in
/// a debug build and every arm wants the same pixels, so it is rendered once.
fn dataset() -> &'static SynthDataset {
    static DATA: OnceLock<SynthDataset> = OnceLock::new();
    DATA.get_or_init(|| SynthDataset::alcove(&SynthConfig::default()))
}

fn solve(threads: usize) -> Reconstruction {
    let data = dataset();
    reconstruct(&data.views, &SfmConfig::default(), &JobPool::new(threads))
        .expect("the committed fixture reconstructs")
}

// ─────────────────────────────────────────────────────────────────────────────
// The bounds
//
// Measured on the committed fixture and then set with roughly threefold margin.
// They are floors and ceilings, not descriptions: a change that makes the
// reconstruction worse than this is a regression, and a change that makes it
// much better should tighten them rather than leave slack nobody notices.
//
// Measured 2026-08-11 (six 320x240 views, four pyramid levels, 512 features per
// view): 6/6 cameras, 430 points, 1280 observations (mean track 2.98), 0.364 px
// RMS, worst camera centre 0.0161 m over a 4.2 m station spread, worst
// *relative* camera rotation 0.305 deg, worst centre-aligned rotation 0.412 deg.
// ─────────────────────────────────────────────────────────────────────────────

/// Every station must get a pose.
const REQUIRED_CAMERAS: usize = 6;
/// Worst camera-centre error after similarity alignment, in **metres** (the
/// truth's units; the reconstruction's own unit is its first baseline). The
/// stations span 4.2 m, so this is under one percent of the configuration.
const MAX_CENTER_ERROR_M: f64 = 0.040;
/// Worst **gauge-free** relative camera-rotation error, in **degrees**.
const MAX_RELATIVE_ROTATION_DEG: f64 = 0.80;
/// Worst rotation error after the centre-fitted alignment, in **degrees**. Kept
/// looser than the relative bound on purpose: with six cameras it inherits the
/// alignment's own uncertainty (see `inf_photo::align::PoseError`).
const MAX_ALIGNED_ROTATION_DEG: f64 = 1.10;
/// Reprojection RMS ceiling, in **pixels**.
const MAX_REPROJECTION_RMS_PX: f64 = 0.90;
/// A floor on the sparse cloud, so a reconstruction cannot pass by keeping four
/// perfect points.
const MIN_POINTS: usize = 280;
/// A floor on observations, for the same reason — and it is more than twice
/// `MIN_POINTS`, so a reconstruction of two-view-only points fails it.
const MIN_OBSERVATIONS: usize = 850;
/// The mean track length floor. Two-view points are the failure mode that once
/// looked perfectly healthy on every other number here.
const MIN_MEAN_TRACK_LENGTH: f64 = 2.4;

#[test]
fn the_fixture_reconstructs_within_bounds() {
    let data = dataset();
    let r = solve(2);

    assert_eq!(
        r.cameras.len(),
        REQUIRED_CAMERAS,
        "only {} of {} stations registered; advisories: {:?}",
        r.cameras.len(),
        data.views.len(),
        r.advisories
    );

    // Truth restricted to the registered views, in the same order.
    let truth: Vec<_> = r
        .registered_views()
        .iter()
        .map(|&v| data.truth[v as usize])
        .collect();
    let err = pose_error(&r.poses(), &truth).expect("the poses align");
    let mean_track = r.report.observations as f64 / r.report.points.max(1) as f64;

    println!(
        "P25.1 gate: {} cameras, {} points, {} observations (mean track {:.2}), \
         rms {:.4} px, centre max {:.5} m mean {:.5} m, relative rotation max {:.5} deg \
         mean {:.5} deg, aligned rotation max {:.5} deg, init pair {:?}",
        r.cameras.len(),
        r.report.points,
        r.report.observations,
        mean_track,
        r.report.reprojection_rms_px,
        err.max_center,
        err.mean_center,
        err.max_relative_rotation_deg,
        err.mean_relative_rotation_deg,
        err.max_aligned_rotation_deg,
        r.report.initial_pair,
    );

    assert!(
        err.max_center <= MAX_CENTER_ERROR_M,
        "worst camera centre is {:.5} m out (bound {MAX_CENTER_ERROR_M} m)",
        err.max_center
    );
    assert!(
        err.max_relative_rotation_deg <= MAX_RELATIVE_ROTATION_DEG,
        "worst relative camera rotation is {:.5} deg out (bound {MAX_RELATIVE_ROTATION_DEG} deg)",
        err.max_relative_rotation_deg
    );
    assert!(
        err.max_aligned_rotation_deg <= MAX_ALIGNED_ROTATION_DEG,
        "worst aligned camera rotation is {:.5} deg out (bound {MAX_ALIGNED_ROTATION_DEG} deg)",
        err.max_aligned_rotation_deg
    );
    assert!(
        r.report.reprojection_rms_px <= MAX_REPROJECTION_RMS_PX,
        "reprojection RMS is {:.4} px (bound {MAX_REPROJECTION_RMS_PX} px)",
        r.report.reprojection_rms_px
    );
    assert!(
        r.report.points >= MIN_POINTS,
        "only {} points survived (floor {MIN_POINTS})",
        r.report.points
    );
    assert!(
        r.report.observations >= MIN_OBSERVATIONS,
        "only {} observations survived (floor {MIN_OBSERVATIONS})",
        r.report.observations
    );
    assert!(
        mean_track >= MIN_MEAN_TRACK_LENGTH,
        "mean track length is {mean_track:.2} (floor {MIN_MEAN_TRACK_LENGTH}) — points are \
         not being carried across views"
    );
}

#[test]
fn the_solver_reaches_the_data_s_own_noise_floor() {
    // The arm that makes the absolute bounds mean something. Triangulate every
    // track through the TRUTH poses and see how well they explain the detected
    // features. Whatever that number is, no reconstruction can beat it — so the
    // real claim about the solver is that it lands on it rather than above it.
    let data = dataset();
    let fcfg = FeatureConfig::default();
    let sets: Vec<_> = data
        .views
        .iter()
        .map(|v| detect_and_describe(&v.image, &fcfg).expect("features"))
        .collect();
    let pool = JobPool::new(2);
    let (pairs, _) = match_all(&sets, &MatchConfig::default(), &pool);
    let (tracks, _) = build_tracks(&sets, &pairs, 2);

    let pixels: Vec<Vec<DVec2>> = sets
        .iter()
        .map(|s| s.features.iter().map(|f| DVec2::new(f.x, f.y)).collect())
        .collect();
    let normalized: Vec<Vec<DVec2>> = data
        .views
        .iter()
        .zip(&pixels)
        .map(|(v, p)| p.iter().map(|q| v.intrinsics.to_normalized(*q)).collect())
        .collect();

    let mut squared = 0.0f64;
    let mut n = 0usize;
    for t in &tracks {
        let rays: Vec<_> = t
            .observations
            .iter()
            .map(|o| {
                (
                    data.truth[o.view as usize],
                    normalized[o.view as usize][o.feature as usize],
                )
            })
            .collect();
        let Some(x) = triangulate_dlt(&rays) else {
            continue;
        };
        let mut local = 0.0;
        let mut ok = true;
        for o in &t.observations {
            match data.views[o.view as usize]
                .intrinsics
                .project(data.truth[o.view as usize].transform(x))
            {
                Some(p) => {
                    local += (p - pixels[o.view as usize][o.feature as usize]).length_squared()
                }
                None => ok = false,
            }
        }
        // A track whose members are metres apart is a mismatch, not a
        // localisation error; the solver rejects those too.
        if !ok || (local / t.observations.len() as f64).sqrt() > 3.0 {
            continue;
        }
        squared += local;
        n += t.observations.len();
    }
    assert!(n > 500, "only {n} observations survived the truth-pose fit");
    let floor = (squared / n as f64).sqrt();
    let solver = solve(2).report.reprojection_rms_px;
    println!(
        "noise floor {floor:.4} px (truth poses), solver {solver:.4} px over {n} observations"
    );
    assert!(
        floor > 0.05,
        "the fixture has no feature noise at all ({floor:.5} px) — this arm is measuring nothing"
    );
    assert!(
        solver <= floor * 1.25,
        "the solver's {solver:.4} px is well above the data's own {floor:.4} px floor"
    );
}

#[test]
fn the_gauge_is_where_it_says_it_is() {
    let r = solve(1);
    let views = r.registered_views();
    let first = r.cameras[&views[0]].pose;
    assert_eq!(
        first.translation.to_array().map(f64::to_bits),
        [0.0f64; 3].map(f64::to_bits),
        "the first registered camera is not at the origin"
    );
    assert_eq!(
        first.rotation.to_array().map(f64::to_bits),
        [0.0f64, 0.0, 0.0, 1.0].map(f64::to_bits),
        "the first registered camera is not the identity"
    );
    let baseline = r.cameras[&views[1]].pose.center().length();
    assert!(
        (baseline - 1.0).abs() < 1e-9,
        "the gauge baseline is {baseline}, not 1"
    );
    // And the rest of the reconstruction lives in that unit: the scene is a few
    // baselines across, not a few thousand and not a few thousandths.
    let spread = r
        .cameras
        .values()
        .map(|c| c.pose.center().length())
        .fold(0.0f64, f64::max);
    assert!(
        (0.5..20.0).contains(&spread),
        "the reconstruction's scale is {spread} baselines across"
    );
}

#[test]
fn the_same_input_gives_the_same_bytes_twice() {
    let a = solve(2).canonical_bytes();
    let b = solve(2).canonical_bytes();
    assert_eq!(
        a.len(),
        b.len(),
        "two runs produced different-sized reconstructions"
    );
    assert!(
        a == b,
        "two runs of the same input differed at byte {:?}",
        a.iter().zip(&b).position(|(x, y)| x != y)
    );
    assert!(
        a.len() > 4096,
        "the byte image is suspiciously small: {} bytes",
        a.len()
    );
}

#[test]
fn the_result_does_not_depend_on_the_worker_count() {
    // The P16.4 law. Every parallel stage in this crate goes through an in-order
    // pure map and every float sum is folded serially, so this is a structural
    // property, not a hope.
    let one = solve(1).canonical_bytes();
    let two = solve(2).canonical_bytes();
    let eight = solve(8).canonical_bytes();
    let mismatch = |a: &[u8], b: &[u8]| a.iter().zip(b).position(|(x, y)| x != y);
    assert!(
        one == two,
        "1 vs 2 workers differ at byte {:?} (lengths {} and {})",
        mismatch(&one, &two),
        one.len(),
        two.len()
    );
    assert!(
        one == eight,
        "1 vs 8 workers differ at byte {:?} (lengths {} and {})",
        mismatch(&one, &eight),
        one.len(),
        eight.len()
    );
}

#[test]
fn a_wrong_intrinsic_measurably_moves_the_reconstruction() {
    // Anti-vacuity. If the bounds above could be met by a solver that ignored
    // its inputs, they would measure nothing. Feed the *same photographs* with a
    // focal length four percent out and the answer must get worse — visibly, in
    // the metrics the gate actually asserts.
    let data = dataset();
    let mut perturbed = data.views.clone();
    for v in perturbed.iter_mut() {
        v.intrinsics.fx *= 1.04;
        v.intrinsics.fy *= 1.04;
    }
    let wrong = reconstruct(&perturbed, &SfmConfig::default(), &JobPool::new(2))
        .expect("a wrong focal length still reconstructs *something*");
    let wrong_truth: Vec<_> = wrong
        .registered_views()
        .iter()
        .map(|&v| data.truth[v as usize])
        .collect();
    let err = pose_error(&wrong.poses(), &wrong_truth).expect("the poses align");

    let baseline = solve(2);
    let baseline_truth: Vec<_> = baseline
        .registered_views()
        .iter()
        .map(|&v| data.truth[v as usize])
        .collect();
    let base_err = pose_error(&baseline.poses(), &baseline_truth).expect("aligns");

    println!(
        "perturbed fx +4%: centre {:.5} m (was {:.5}), relative rotation {:.5} deg (was {:.5}), \
         rms {:.4} px (was {:.4})",
        err.max_center,
        base_err.max_center,
        err.max_relative_rotation_deg,
        base_err.max_relative_rotation_deg,
        wrong.report.reprojection_rms_px,
        baseline.report.reprojection_rms_px
    );
    assert!(
        wrong.canonical_bytes() != baseline.canonical_bytes(),
        "a 4% focal-length error produced a byte-identical reconstruction"
    );
    assert!(
        err.max_relative_rotation_deg > base_err.max_relative_rotation_deg * 2.0,
        "a 4% focal-length error barely moved the rotation error: {:.5} deg vs {:.5} deg",
        err.max_relative_rotation_deg,
        base_err.max_relative_rotation_deg
    );
    assert!(
        err.max_center > MAX_CENTER_ERROR_M
            || err.max_relative_rotation_deg > MAX_RELATIVE_ROTATION_DEG
            || wrong.report.reprojection_rms_px > MAX_REPROJECTION_RMS_PX,
        "a 4% focal-length error stayed inside every bound this gate asserts — \
         centre {:.5} m, relative rotation {:.5} deg, rms {:.4} px",
        err.max_center,
        err.max_relative_rotation_deg,
        wrong.report.reprojection_rms_px
    );
}

#[test]
fn too_few_views_refuses_by_name() {
    let data = dataset();
    let two = data.views[..2].to_vec();
    let err = reconstruct(&two, &SfmConfig::default(), &JobPool::new(1)).unwrap_err();
    assert_eq!(
        err,
        SfmError::TooFewViews {
            given: 2,
            required: 3
        },
        "a two-view capture must refuse by name, not reconstruct"
    );
    assert!(format!("{err}").contains("at least 3 views"));
    // And three views — the minimum — must still work, so the refusal is a
    // *bound* rather than a blanket rejection of small captures.
    let three = data.views[..3].to_vec();
    let r = reconstruct(&three, &SfmConfig::default(), &JobPool::new(1))
        .expect("three views is enough");
    assert_eq!(r.cameras.len(), 3, "advisories: {:?}", r.advisories);
}

#[test]
fn the_reconstruction_reports_what_it_did() {
    let r = solve(1);
    assert_eq!(r.report.views, 6);
    assert_eq!(r.report.registered, r.cameras.len());
    assert_eq!(r.report.points, r.points.len());
    assert_eq!(r.report.features_per_view.len(), 6);
    assert!(
        r.report.features_per_view.iter().all(|&n| n > 100),
        "a station found too few features: {:?}",
        r.report.features_per_view
    );
    assert!(
        r.report.pairs >= 10,
        "only {} pairs matched",
        r.report.pairs
    );
    assert!(r.report.tracks >= r.report.points);
    let (a, b) = r.report.initial_pair;
    assert!(a < b && (b as usize) < 6, "nonsense initial pair {a},{b}");

    // Every surviving point is seen by at least two registered cameras, never
    // twice by one, and every camera's observation count is its own.
    let registered = r.registered_views();
    let mut counted = std::collections::BTreeMap::<u32, usize>::new();
    for p in &r.points {
        assert!(p.observations.len() >= 2, "a point survived on one view");
        let mut views: Vec<u32> = p.observations.iter().map(|o| o.view).collect();
        let before = views.len();
        views.dedup();
        assert_eq!(views.len(), before, "a point sees one view twice");
        for o in &p.observations {
            assert!(
                registered.contains(&o.view),
                "a point references unregistered view {}",
                o.view
            );
            *counted.entry(o.view).or_insert(0) += 1;
        }
    }
    for (view, cam) in &r.cameras {
        assert_eq!(
            cam.observations,
            counted.get(view).copied().unwrap_or(0),
            "camera {view} misreports its observation count"
        );
        assert_eq!(cam.view, *view);
    }
    assert_eq!(
        r.report.observations,
        counted.values().sum::<usize>(),
        "the report's observation total is not the sum of the points'"
    );
}
