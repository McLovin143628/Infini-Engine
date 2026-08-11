//! The P25.4 gate: **files on disk to assets in a project, through the wizard's
//! own door**, with the progress stream, the cancellation guarantee and the
//! diagnostics asserted rather than described.
//!
//! # What this is a gate over, and what it is not
//!
//! It drives [`PhotogrammetrySession`] — the same Ring-1 object all nine Tauri
//! commands drive — and nothing else. It does **not** drive the UI: a
//! `#[tauri::command]` cannot be run from a test on any CI leg, so every rule
//! worth gating lives on this side of the wire and the command layer is the
//! string hop the typed-IPC law makes it (P23.4's ruling, met again). The nine
//! commands themselves are therefore **untested by design**, and the two rules
//! that live in them anyway — the status projection and the settings overlay —
//! have unit tests beside them in `commands/photogrammetry.rs` and
//! `ipc.rs`.
//!
//! The photographs are the P25.2/P25.3 fixture's six renders, **written to a
//! temporary directory as PNG files** and re-read through `RgbImage::load` —
//! which is the point: everything upstream of P25.4 has taken its photographs
//! from a function call, and a capture wizard's first stage is a filesystem.
//!
//! # The set is the GRAYSCALE renders, and the colour ones are an arm
//!
//! `fixture::photographs()` — the lit, tinted colour renders P25.3's albedo bake
//! re-reads — **do not reconstruct**, and the reason is worth a paragraph
//! because it is the wizard's own subject matter. MEASURED, over a gain applied
//! to the whole set before graying it:
//!
//! | peak luma | outcome |
//! |---|---|
//! | 135 (as rendered) | no viable initial pair, best 24 inliers of 30 needed |
//! | 189 | 2 of 6 registered |
//! | 222 | no viable initial pair, best 29 of 30 |
//! | 250 | 5 of 6 registered, RMS 0.449 px |
//! | grayscale renders | **6 of 6**, RMS 0.405 px |
//!
//! The colour renders peak at 135 of 255 — the fixture's lighting is sized so
//! nothing clips, which makes it a **half-exposed** capture — and this crate's
//! descriptor is not contrast-normalized, so half the range is half the
//! descriptor. That is a real finding about real photographs and it is gated as
//! one, in `the_wizard_reports_a_solver_refusal_by_name_and_writes_nothing`;
//! the flow arms use the grayscale renders, whose 6-of-6 is what a gate over
//! *the wizard* needs. The colour accuracy of a bake is P25.3's claim and is
//! measured there.
//!
//! # The camera is told, because it cannot be recovered
//!
//! Structure from motion refines poses and never intrinsics, so the wizard's
//! [`AssumedCamera`] is an input. The arms below run it at the fixture's real
//! lens (`focal_ratio = 300/320`, `k1 = -0.09`, `k2 = 0.02`) — the calibrated
//! case a user with a known camera has — and one arm measures the
//! **uncalibrated** default beside it, because that is what a phone in a pocket
//! actually gives and the honest answer is a measurement rather than a hope.
//!
//! # Within one run or within one machine
//!
//! `meshopt` is in this chain (P18's law) so nothing here is compared against a
//! committed number. Every bound is either structural (a stage order, an asset
//! count, a filesystem state) or a comparison between two runs in this process.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use inf_core::job::JobPool;
use inf_editor_core::assets::AssetProject;
use inf_editor_core::capture::{
    scale_for_longest_side, AssumedCamera, CaptureConfig, CaptureIssue, CapturePhase,
    CaptureProgress, CaptureStage, CaptureState, PhotogrammetrySession, SCAN_FOLDER,
};
use inf_editor_core::photogrammetry::{FinishAdvisory, FinishConfig};
use inf_photo::rgb::RgbImage;
use inf_photo_gpu::{fixture, BakeConfig};

// ─────────────────────────────────────────────────────────────────────────────
// The bounds
//
// MEASURED 2026-08-11 on the committed fixture, through the session door, at
// `capture_config()` (the fixture's own lens, a 20 000-triangle budget and a
// 256-texel atlas). Every number printed by the arms; the constants below are
// floors and ceilings with room, not descriptions.
//
// MEASURED, the full six-station run: 6 of 6 views registered at RMS 0.405 px,
// 225 564 fused triangles, 14 882 finished triangles over 624 charts, an extent
// of 8.2638 baseline units, and **twenty** findings. Stage wall clock, 4 job
// workers: load 0 ms, structure from motion 95 ms, dense 991 ms, finish 1844 ms
// — the eleven-arm gate ran in 8.4 s, and the thirteen the P25.4 audit left
// behind still never skip, because every stage of it is CPU (P25.3's ruling,
// inherited).
//
// MEASURED, coverage: **1.000** of the finished mesh seen by some camera (the
// unphotographed trim is why), **0.891** seen by two or more, 1 618 triangles
// seen by exactly one. Per station: 0.732 / 0.730 / 0.557 / 0.269 / 0.699 /
// 0.329 — the spread a lopsided rig produces, and the reason a per-view row
// exists rather than one number.
//
// MEASURED, the one-sided subset (stations 1-3, all on the right): overlap
// **0.187** against 0.891, 1 933 triangles seen by one camera, and 552 of every
// 1 000 covered texels invented by dilation.
//
// MEASURED, the uncalibrated default lens (1.2x the longer side, no distortion,
// against the fixture's true 0.9375x with k1 = -0.09): still **6 of 6**
// registered, 14 745 triangles against 14 882, overlap 0.900 against 0.891. A
// 28% focal error and a dropped distortion model cost this fixture almost
// nothing — which is a fact about a 320x240 frame with a modest field of view,
// not a licence, and the arm prints it rather than asserting the closeness.
// ─────────────────────────────────────────────────────────────────────────────

/// The fixture is 320 wide and was rendered at a 300 px focal length.
const FIXTURE_FOCAL_RATIO: f64 = 300.0 / 320.0;
/// A finished capture must produce a mesh with real geometry in it.
const MIN_FINAL_TRIANGLES: usize = 2_048;
/// Five assets: mesh, base colour, normal, ORM, material.
const SCAN_ASSETS: usize = 5;
/// How much of the finished mesh at least one camera must see. The trim removes
/// what nobody photographed, so this is near one by construction and is here to
/// catch a coverage measurement that has stopped measuring.
const MIN_COVERED_FRACTION: f64 = 0.98;
/// How much of the finished mesh two cameras or more must see on the FULL
/// six-station set. This is the number a one-sided capture destroys, and the
/// reason the overlap warning exists.
const MIN_OVERLAP_FRACTION: f64 = 0.50;

/// The configuration every arm runs at. One place, so no arm measures a
/// different pipeline from the one beside it.
fn capture_config() -> CaptureConfig {
    CaptureConfig {
        camera: AssumedCamera {
            focal_ratio: FIXTURE_FOCAL_RATIO,
            k1: -0.09,
            k2: 0.02,
        },
        finish: FinishConfig {
            target_triangles: 20_000,
            bake: BakeConfig {
                size: 256,
                ao_rays: 32,
                ..BakeConfig::default()
            },
            ..FinishConfig::default()
        },
        ..CaptureConfig::default()
    }
}

fn pool() -> Arc<JobPool> {
    Arc::new(JobPool::new(4))
}

/// The fixture's six photographs, written once to a directory this process owns
/// for its whole life.
///
/// A `OnceLock` over a leaked `TempDir`: every arm that runs the pipeline wants
/// the same six files, and encoding them per arm is pure waste. The directory
/// lives as long as the test binary, which is exactly the scope its readers do.
fn photo_files() -> &'static [PathBuf] {
    static FILES: OnceLock<Vec<PathBuf>> = OnceLock::new();
    FILES.get_or_init(|| {
        let dir = photo_dir("inf-capture-photos");
        fixture::dataset()
            .views
            .iter()
            .enumerate()
            .map(|(i, view)| {
                let path = dir.join(format!("station{i}.png"));
                write_gray_png(
                    &path,
                    view.image.width(),
                    view.image.height(),
                    view.image.pixels(),
                );
                path
            })
            .collect()
    })
}

/// The **colour** renders, as files. Half-exposed, and the header says what that
/// costs.
fn colour_files() -> &'static [PathBuf] {
    static FILES: OnceLock<Vec<PathBuf>> = OnceLock::new();
    FILES.get_or_init(|| {
        let dir = photo_dir("inf-capture-colour");
        fixture::photographs()
            .iter()
            .enumerate()
            .map(|(i, photo)| {
                let path = dir.join(format!("colour{i}.png"));
                write_rgb_png(&path, photo);
                path
            })
            .collect()
    })
}

/// A temp directory that outlives the test binary's use of it.
fn photo_dir(prefix: &str) -> &'static Path {
    let dir = tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("a temp dir for the photographs");
    Box::leak(Box::new(dir)).path()
}

/// Write an 8-bit grayscale PNG, through the encoder this crate already depends
/// on.
fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
    write_png(path, width, height, png::ColorType::Grayscale, pixels);
}

/// Write an 8-bit RGB PNG.
fn write_rgb_png(path: &Path, photo: &RgbImage) {
    write_png(
        path,
        photo.width(),
        photo.height(),
        png::ColorType::Rgb,
        photo.pixels(),
    );
}

fn write_png(path: &Path, width: u32, height: u32, colour: png::ColorType, pixels: &[u8]) {
    let mut bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut bytes, width, height);
        enc.set_color(colour);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(pixels).expect("png data");
    }
    std::fs::write(path, bytes).expect("write the photograph");
}

/// A session with the fixture's photographs loaded at `capture_config()`.
fn loaded_session(paths: &[PathBuf]) -> PhotogrammetrySession {
    let mut session = PhotogrammetrySession::new();
    session.set_config(capture_config());
    let issues = session.load_photos(paths).expect("an idle session loads");
    assert!(
        !issues.iter().any(|i| i.blocks()),
        "the fixture's own photographs did not pass the pre-flight: {issues:?}"
    );
    session
}

/// Run a session to completion, returning every progress event in order.
fn run_to_completion(session: &mut PhotogrammetrySession) -> Vec<CaptureProgress> {
    session.start(pool()).expect("the run starts");
    drain_until_done(session)
}

/// Poll-and-drain until the worker parks, which is what the Ring-2 tick does on
/// a timer.
fn drain_until_done(session: &mut PhotogrammetrySession) -> Vec<CaptureProgress> {
    let mut events = Vec::new();
    loop {
        events.extend(session.drain());
        if !matches!(session.state(), CaptureState::Running(_)) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    session.wait();
    events.extend(session.drain());
    events
}

/// The stage sequence an event stream visited, with repeats collapsed.
fn stage_sequence(events: &[CaptureProgress]) -> Vec<CaptureStage> {
    let mut out: Vec<CaptureStage> = Vec::new();
    for ev in events {
        if out.last() != Some(&ev.stage) {
            out.push(ev.stage);
        }
    }
    out
}

/// The one full run every read-only arm shares, and its events.
///
/// Computed once because it is minutes of work and every arm below asks a
/// different question of the same answer — the `OnceLock` discipline the P25.3
/// gate is built on.
struct FullRun {
    events: Vec<CaptureProgress>,
    triangles: usize,
    charts: usize,
    extent_units: f64,
    covered_fraction: f64,
    overlap_fraction: f64,
    seen_by_none: usize,
    seen_by_one: usize,
    views: Vec<(String, bool, f64)>,
    issues: Vec<CaptureIssue>,
    elapsed_ms: [u64; 5],
}

fn full_run() -> &'static FullRun {
    static RUN: OnceLock<FullRun> = OnceLock::new();
    RUN.get_or_init(|| {
        let mut session = loaded_session(photo_files());
        let events = run_to_completion(&mut session);
        assert_eq!(
            session.state(),
            CaptureState::Ready,
            "the fixture did not reconstruct: {:?}",
            session.error()
        );
        session
            .with_product(|p| FullRun {
                events: events.clone(),
                triangles: p.finished.report.final_triangles,
                charts: p.finished.report.charts,
                extent_units: p.extent_units,
                covered_fraction: p.coverage.covered_fraction(),
                overlap_fraction: p.coverage.overlap_fraction(),
                seen_by_none: p.coverage.seen_by_none,
                seen_by_one: p.coverage.seen_by_one,
                views: p
                    .coverage
                    .views
                    .iter()
                    .map(|v| (v.photo.clone(), v.registered, v.fraction))
                    .collect(),
                issues: p.issues.clone(),
                elapsed_ms: p.elapsed_ms,
            })
            .expect("a finished run has a product")
    })
}

// ── (a) the whole flow, through the session door ────────────────────────────

#[test]
fn six_photographs_on_disk_become_a_finished_scan_with_progress_all_the_way() {
    let run = full_run();
    println!(
        "P25.4 run: {} triangles, {} charts, extent {:.4} units, load {} ms / sfm {} ms / dense {} \
         ms / finish {} ms",
        run.triangles,
        run.charts,
        run.extent_units,
        run.elapsed_ms[0],
        run.elapsed_ms[1],
        run.elapsed_ms[2],
        run.elapsed_ms[3]
    );
    assert!(
        run.triangles >= MIN_FINAL_TRIANGLES,
        "{} triangles is not a scan",
        run.triangles
    );
    assert!(run.charts > 1, "{} charts is not an atlas", run.charts);
    assert!(
        run.extent_units > 0.0,
        "the reconstruction has no extent to scale against"
    );

    // THE STAGE SEQUENCE, exactly. An automatic run is four stages; `Write` is
    // the user's and is asserted in its own arm.
    assert_eq!(
        stage_sequence(&run.events),
        vec![
            CaptureStage::Load,
            CaptureStage::Sfm,
            CaptureStage::Dense,
            CaptureStage::Finish,
        ],
        "the stage sequence is not the pipeline's"
    );
    // MONOTONE: no event ever names a stage earlier than one already reported.
    let mut high = 0usize;
    for ev in &run.events {
        assert!(
            ev.stage.index() >= high,
            "{} came after {}",
            ev.stage.name(),
            CaptureStage::ALL[high].name()
        );
        high = ev.stage.index();
    }
    // TERMINAL: every stage that started finished, and none failed.
    for stage in [
        CaptureStage::Load,
        CaptureStage::Sfm,
        CaptureStage::Dense,
        CaptureStage::Finish,
    ] {
        let phases: Vec<CapturePhase> = run
            .events
            .iter()
            .filter(|e| e.stage == stage)
            .map(|e| e.phase)
            .collect();
        assert_eq!(
            phases.first(),
            Some(&CapturePhase::Started),
            "{} did not report a start",
            stage.name()
        );
        assert_eq!(
            phases.last(),
            Some(&CapturePhase::Finished),
            "{} did not report a finish",
            stage.name()
        );
        assert!(
            !phases.iter().any(|p| p.is_terminal()),
            "{} reported a terminal phase in a successful run",
            stage.name()
        );
    }
}

#[test]
fn the_two_stages_this_module_owns_report_inside_themselves() {
    // The granularity ruling, asserted rather than written down: `Load` reports
    // per photograph and `Finish` reports per step, because both orchestrators
    // are Ring 1. `Sfm` and `Dense` are single blocking calls into Ring 0 and
    // report start-and-finish only — so a progress bar that pretended otherwise
    // would be inventing motion.
    let run = full_run();
    let ticks = |stage: CaptureStage| -> Vec<&CaptureProgress> {
        run.events
            .iter()
            .filter(|e| e.stage == stage && e.phase == CapturePhase::Progress)
            .collect()
    };
    let load = ticks(CaptureStage::Load);
    assert_eq!(load.len(), 6, "one tick per photograph");
    assert!(
        load.iter().all(|t| t.total == 6),
        "a load tick does not know how many photographs there are"
    );
    assert!(
        load.iter().enumerate().all(|(i, t)| t.done == i as u64 + 1),
        "the load ticks are not counting"
    );
    assert!(
        load.iter().any(|t| t.detail.contains("station0.png")),
        "a load tick does not name its file: {:?}",
        load.iter().map(|t| &t.detail).collect::<Vec<_>>()
    );

    let finish = ticks(CaptureStage::Finish);
    assert_eq!(finish.len(), 8, "one tick per finish step");
    let labels: Vec<&str> = finish.iter().map(|t| t.detail.as_str()).collect();
    for expected in ["retopology", "UV unwrap", "albedo bake"] {
        assert!(
            labels.contains(&expected),
            "the finish never reported {expected}: {labels:?}"
        );
    }
    assert!(
        finish.iter().enumerate().all(|(i, t)| t.done == i as u64),
        "the finish ticks are not counting"
    );

    for stage in [CaptureStage::Sfm, CaptureStage::Dense] {
        assert!(
            ticks(stage).is_empty(),
            "{} claims progress it cannot measure",
            stage.name()
        );
    }
    // …and both of them still say what they produced when they end.
    let sfm = run
        .events
        .iter()
        .find(|e| e.stage == CaptureStage::Sfm && e.phase == CapturePhase::Finished)
        .expect("sfm finished");
    println!("P25.4 sfm: {}", sfm.detail);
    assert!(sfm.detail.contains("registered"), "{}", sfm.detail);
}

// ── (b) the write, and the five-or-none it inherits ─────────────────────────

#[test]
fn the_import_writes_five_assets_under_the_project_root_and_says_so() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut project = AssetProject::open(dir.path()).expect("a project");
    let before = project.db().len();

    let mut session = loaded_session(photo_files());
    let events = run_to_completion(&mut session);
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "{:?}",
        session.error()
    );
    assert!(!events.is_empty());

    let ids = session
        .import(&mut project, "Ridge")
        .expect("the import writes");
    assert_eq!(
        project.db().len() - before,
        SCAN_ASSETS,
        "an import is five assets or none"
    );

    // THE TRAP, closed at this end: `AssetProject::write_asset` takes its
    // directory verbatim, so a wizard that passed a relative folder would write
    // into the process's working directory — which is the source tree. Every
    // one of the five must be under the project root, in `Scans`.
    //
    // Canonicalized on both sides: the registered path is, and on Windows that
    // is the `\\?\` verbatim form of the same directory. Comparing the two
    // spellings would fail on a difference that is not the one this arm is
    // about, and passing on a *prefix* comparison of two different spellings
    // would be worse.
    let scans = std::fs::canonicalize(project.root().join(SCAN_FOLDER))
        .expect("the scan folder exists once a scan has been written");
    for id in [ids.mesh, ids.albedo, ids.normal, ids.orm, ids.material] {
        let entry = project.db().get(id).expect("a written asset");
        assert!(
            entry.path.starts_with(&scans),
            "{} landed at {} rather than under {}",
            entry.name,
            entry.path.display(),
            scans.display()
        );
        assert!(entry.path.is_file(), "{} is not on disk", entry.name);
    }
    assert_eq!(session.state(), CaptureState::Imported);

    // The write reports on the same channel as every other stage, so the
    // wizard's state machine needs no new case.
    let write: Vec<&CaptureProgress> = events.iter().collect();
    assert!(
        write.iter().all(|e| e.stage != CaptureStage::Write),
        "the automatic run wrote something"
    );
    let after = session.drain();
    assert_eq!(
        stage_sequence(&after),
        vec![CaptureStage::Write],
        "the import did not report as a stage"
    );
    assert_eq!(after.first().map(|e| e.phase), Some(CapturePhase::Started));
    assert_eq!(after.last().map(|e| e.phase), Some(CapturePhase::Finished));

    // And the honest note about what the viewport will draw is now on the
    // product, because that is the moment a user goes looking for their scan.
    let has_note = session
        .with_product(|p| p.issues.contains(&CaptureIssue::NoMeshletDag))
        .unwrap_or(false);
    assert!(
        has_note,
        "the wizard imported a scan and did not say it draws as a placeholder cube"
    );
}

// ── (c) cancellation ────────────────────────────────────────────────────────

#[test]
fn a_cancelled_run_writes_nothing_and_leaves_a_session_that_can_be_asked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut project = AssetProject::open(dir.path()).expect("a project");
    let before = project.db().len();

    let mut session = loaded_session(photo_files());
    session.start(pool()).expect("the run starts");
    // Cancel immediately: the worker reads the flag between stages, so this
    // lands at the first boundary — which is the guarantee, and is why the arm
    // asserts the FILESYSTEM rather than a timing.
    assert!(session.cancel(), "cancel found nothing running");
    let events = drain_until_done(&mut session);

    assert_eq!(
        session.state(),
        CaptureState::Cancelled,
        "a cancelled run did not settle as cancelled"
    );
    let cancelled: Vec<&CaptureProgress> = events
        .iter()
        .filter(|e| e.phase == CapturePhase::Cancelled)
        .collect();
    assert_eq!(cancelled.len(), 1, "a run stops once: {events:?}");
    println!(
        "P25.4 cancel: stopped before {} after {} events",
        cancelled[0].stage.name(),
        events.len()
    );
    // Nothing after the stop.
    let stop_at = events
        .iter()
        .position(|e| e.phase == CapturePhase::Cancelled)
        .expect("a stop");
    assert_eq!(
        stop_at,
        events.len() - 1,
        "events kept arriving after the run stopped"
    );

    // NO ASSETS. The whole point: writing is a separate stage the user starts,
    // so a cancelled run cannot have half-written anything — asserted against
    // the database AND against the directory, because a torn write is a
    // filesystem fact (the P25.3 law).
    assert_eq!(project.db().len(), before, "a cancelled run wrote assets");
    assert!(
        !dir.path().join(SCAN_FOLDER).exists(),
        "a cancelled run created the scan folder"
    );
    // And the session is asked, not dead: it refuses the import by name.
    let err = session
        .import(&mut project, "Ridge")
        .expect_err("there is nothing to import");
    assert!(err.to_string().contains("import"), "{err}");
    assert_eq!(
        project.db().len(),
        before,
        "the refused import wrote assets"
    );
}

// ── (d) the pre-flight refusals, by name ────────────────────────────────────

#[test]
fn the_preflight_refuses_by_name_before_a_run_is_ever_started() {
    let files = photo_files();

    // Too few.
    let mut session = PhotogrammetrySession::new();
    session.set_config(capture_config());
    let issues = session
        .load_photos(&files[..2])
        .expect("an idle session loads");
    let too_few = issues
        .iter()
        .find(|i| matches!(i, CaptureIssue::TooFewPhotos { given: 2, .. }))
        .expect("two photographs must refuse");
    assert!(too_few.blocks());
    let err = session
        .start(pool())
        .expect_err("two photographs must refuse");
    assert!(err.to_string().contains("at least 3"), "{err}");

    // A file that is not a photograph, named.
    let dir = tempfile::tempdir().expect("tempdir");
    let broken = dir.path().join("notes.png");
    std::fs::write(&broken, b"this is not a png").expect("write");
    let mut paths = files.to_vec();
    paths.push(broken);
    let issues = session.load_photos(&paths).expect("an idle session loads");
    let unreadable = issues
        .iter()
        .find(|i| matches!(i, CaptureIssue::Unreadable { .. }))
        .expect("an undecodable file must refuse");
    assert!(unreadable.to_string().contains("notes.png"), "{unreadable}");
    assert!(session.start(pool()).is_err());

    // A scale that is not a scale, caught before four stages of work.
    let mut session = loaded_session(files);
    let mut cfg = capture_config();
    cfg.finish.metres_per_unit = 0.0;
    session.set_config(cfg);
    let err = session.start(pool()).expect_err("a zero scale must refuse");
    assert!(err.to_string().contains("not a scale"), "{err}");
    assert_eq!(session.state(), CaptureState::Idle);
}

// ── (d2) a solver refusal, reported rather than swallowed ───────────────────

#[test]
fn the_wizard_reports_a_solver_refusal_by_name_and_writes_nothing() {
    // The **half-exposed** colour set (see the header's gain table). The
    // photographs decode, the pre-flight passes them, and structure from motion
    // refuses — which is the shape of failure a real capture actually has, and
    // the one a wizard has to be able to show. What is gated is that the
    // refusal reaches the user with the solver's own words on it.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut project = AssetProject::open(dir.path()).expect("a project");
    let before = project.db().len();

    let mut session = PhotogrammetrySession::new();
    session.set_config(capture_config());
    let issues = session
        .load_photos(colour_files())
        .expect("an idle session loads");
    assert!(
        !issues.iter().any(|i| i.blocks()),
        "the pre-flight cannot see an exposure problem and must not pretend to: {issues:?}"
    );

    let events = run_to_completion(&mut session);
    assert_eq!(
        session.state(),
        CaptureState::Failed,
        "the half-exposed set reconstructed — re-measure the header's gain table"
    );
    let error = session.error().expect("a failed run carries its refusal");
    println!("P25.4 refusal: {error}");
    assert!(
        error.contains("initial pair") || error.contains("registered"),
        "the refusal is not the solver's: {error}"
    );

    // It arrives on the event stream too, on the stage that raised it, and
    // nothing follows it.
    let failed = events
        .iter()
        .find(|e| e.phase == CapturePhase::Failed)
        .expect("a failed stage");
    assert_eq!(failed.stage, CaptureStage::Sfm);
    assert_eq!(failed.error.as_deref(), Some(error.as_str()));
    assert_eq!(
        events.last().map(|e| e.phase),
        Some(CapturePhase::Failed),
        "events kept arriving after the refusal"
    );
    // Load still finished before it — a refusal does not erase what worked.
    assert_eq!(
        stage_sequence(&events),
        vec![CaptureStage::Load, CaptureStage::Sfm]
    );

    assert_eq!(project.db().len(), before, "a failed run wrote assets");
    assert!(session.import(&mut project, "Ridge").is_err());
    assert_eq!(project.db().len(), before);
}

// ── (e) the coverage overlay, and the overlap warning ───────────────────────

#[test]
fn the_coverage_overlay_names_every_photograph_and_measures_what_it_saw() {
    let run = full_run();
    println!(
        "P25.4 coverage: {:.3} covered, {:.3} overlap, {} seen by none, {} seen by one",
        run.covered_fraction, run.overlap_fraction, run.seen_by_none, run.seen_by_one
    );
    for (name, registered, fraction) in &run.views {
        println!("  {name}: registered {registered}, sees {fraction:.3}");
    }
    assert_eq!(
        run.views.len(),
        6,
        "a row per photograph, registered or not"
    );
    assert!(
        run.views.iter().all(|(name, _, _)| name.ends_with(".png")),
        "a coverage row does not name its file"
    );
    assert!(
        run.covered_fraction >= MIN_COVERED_FRACTION,
        "only {:.3} of the finished mesh is seen by any camera",
        run.covered_fraction
    );
    assert!(
        run.overlap_fraction >= MIN_OVERLAP_FRACTION,
        "only {:.3} of the finished mesh is seen by two cameras or more",
        run.overlap_fraction
    );
    // Every registered camera must see SOMETHING, or the measurement is not
    // measuring: a row of zeroes is what a wrong frame or a wrong scale looks
    // like, and it would be invisible to a total.
    for (name, registered, fraction) in &run.views {
        if *registered {
            assert!(
                *fraction > 0.05,
                "{name} registered a pose and sees {fraction:.4} of the result"
            );
        }
    }
}

#[test]
fn a_one_sided_capture_says_which_surfaces_only_one_camera_saw() {
    // Stations 1, 2 and 3 sweep the RIGHT of the scene; nothing looks at it from
    // the left. That is the failure a capture wizard exists to diagnose, and the
    // arm asserts the specific finding rather than "some advisory appeared".
    let files = photo_files();
    let subset = vec![files[1].clone(), files[2].clone(), files[3].clone()];
    let mut session = loaded_session(&subset);
    let _ = run_to_completion(&mut session);
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "the one-sided subset did not reconstruct: {:?}",
        session.error()
    );

    let (overlap, single, unseen, issues) = session
        .with_product(|p| {
            (
                p.coverage.overlap_fraction(),
                p.coverage.seen_by_one,
                p.coverage
                    .unseen_texels
                    .checked_mul(1000)
                    .and_then(|n| n.checked_div(p.coverage.covered_texels.max(1)))
                    .unwrap_or(0),
                p.issues.clone(),
            )
        })
        .expect("a finished run has a product");
    let full = full_run();
    println!(
        "P25.4 one-sided: overlap {overlap:.3} against {:.3} on six stations, {single} triangles \
         seen by one, {} per-mille of covered texels unseen",
        full.overlap_fraction, unseen
    );

    // THE SPECIFIC FINDING: geometry only one camera photographed.
    let single_coverage = issues
        .iter()
        .find(|i| matches!(i, CaptureIssue::SingleCoverage { .. }))
        .expect("a three-station capture must raise the overlap warning");
    let text = single_coverage.to_string();
    assert!(text.contains("exactly ONE"), "{text}");
    assert!(
        text.contains("second angle"),
        "the warning carries no remedy: {text}"
    );
    assert!(
        !single_coverage.blocks(),
        "an overlap warning is not a refusal"
    );

    // …and it is WORSE than the full set's, which is what makes it a diagnosis
    // rather than a constant. A subset that overlapped as well as the whole set
    // would mean the measurement is not reading the cameras.
    assert!(
        overlap < full.overlap_fraction,
        "three stations overlap as well as six ({overlap:.3} against {:.3}) — the coverage \
         measurement is not reading the camera set",
        full.overlap_fraction
    );

    // The albedo's own version of the same shortfall, from the pipeline rather
    // than from this file.
    let unseen_texels = issues
        .iter()
        .any(|i| matches!(i, CaptureIssue::Finish(FinishAdvisory::UnseenTexels { .. })));
    assert!(
        unseen_texels,
        "a one-sided capture invented colour and no advisory said so: {issues:?}"
    );
}

// ── (f) the scale step ──────────────────────────────────────────────────────

#[test]
fn the_scale_step_is_a_refinish_and_costs_only_the_finish_stage() {
    let mut session = loaded_session(photo_files());
    let _ = run_to_completion(&mut session);
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "{:?}",
        session.error()
    );

    let (extent, first_vertex) = session
        .with_product(|p| {
            (
                p.extent_units,
                p.finished.mesh.submeshes[0].vertices[0].position,
            )
        })
        .expect("a product");

    // The affordance: "the longest side of this thing is 3 metres".
    let scale = scale_for_longest_side(extent, 3.0).expect("a real length is a real scale");
    println!("P25.4 scale: extent {extent:.4} units, 3 m across is {scale:.5} m/unit");
    let mut cfg = capture_config();
    cfg.finish.metres_per_unit = scale;
    session.set_config(cfg);

    let events = {
        session.refinish(pool()).expect("the re-finish starts");
        drain_until_done(&mut session)
    };
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "{:?}",
        session.error()
    );
    // ONLY the finish ran. Structure from motion and the dense solve are minutes
    // of work whose answer did not change, and a wizard that re-ran them to
    // multiply by a constant would be unusable.
    assert_eq!(
        stage_sequence(&events),
        vec![CaptureStage::Finish],
        "a re-finish ran stages it already had answers for"
    );

    let (scaled_extent, scaled_vertex) = session
        .with_product(|p| {
            (
                p.extent_units,
                p.finished.mesh.submeshes[0].vertices[0].position,
            )
        })
        .expect("a product");
    // The mesh moved by the scale…
    for c in 0..3 {
        let expected = first_vertex[c] as f64 * scale;
        assert!(
            (scaled_vertex[c] as f64 - expected).abs() < 1e-3,
            "the scale step did not multiply the positions: {} against {expected}",
            scaled_vertex[c]
        );
    }
    // …and the extent readout did NOT, because it is reported in baseline units
    // so a second correction is computed against the same number as the first.
    assert!(
        (scaled_extent - extent).abs() < 1e-6,
        "the extent readout is in metres, so a user who scales once cannot scale again: \
         {scaled_extent} against {extent}"
    );
}

// ── (g) the uncalibrated camera, measured rather than hoped ─────────────────

#[test]
fn an_uncalibrated_lens_is_measured_beside_the_calibrated_one() {
    // `AssumedCamera::default()` is a GUESS — 1.2 times the longer side, with no
    // distortion — and it is what a photograph with no EXIF and no calibration
    // actually gets. What that costs on this fixture is a measurement, not an
    // assumption, and it is recorded here so the wizard's "assumed lens" field
    // has a number behind its warning.
    let mut session = PhotogrammetrySession::new();
    let mut cfg = capture_config();
    cfg.camera = AssumedCamera::default();
    session.set_config(cfg);
    session
        .load_photos(photo_files())
        .expect("an idle session loads");
    let _ = run_to_completion(&mut session);

    let full = full_run();
    match session.state() {
        CaptureState::Ready => {
            let (registered, triangles, overlap) = session
                .with_product(|p| {
                    (
                        p.reconstruction.report.registered,
                        p.finished.report.final_triangles,
                        p.coverage.overlap_fraction(),
                    )
                })
                .expect("a product");
            println!(
                "P25.4 uncalibrated: {registered} of 6 registered, {triangles} triangles, overlap \
                 {overlap:.3} — against {} triangles and {:.3} at the fixture's own lens",
                full.triangles, full.overlap_fraction
            );
            assert!(
                registered >= 3,
                "a wrong focal length must still register the minimum or refuse by name"
            );
        }
        CaptureState::Failed => {
            let error = session.error().unwrap_or_default();
            println!("P25.4 uncalibrated: REFUSES — {error}");
            assert!(
                !error.is_empty(),
                "a failed run must carry the refusal that caused it"
            );
        }
        other => panic!("an uncalibrated run settled as {other:?}"),
    }
}

// ── (h) the diagnostics surface itself ──────────────────────────────────────

#[test]
fn every_finding_a_run_produces_carries_a_stage_a_severity_and_a_remedy() {
    let run = full_run();
    println!("P25.4 diagnostics: {} findings", run.issues.len());
    for issue in &run.issues {
        println!(
            "  [{}/{}] {issue}",
            issue.severity().name(),
            issue.stage().name()
        );
    }
    assert!(
        !run.issues.is_empty(),
        "a real reconstruction produced no findings at all, which is not a thing that happens"
    );
    for issue in &run.issues {
        let text = issue.to_string();
        // A finding nobody can act on is a log line. The two pass-throughs carry
        // their source's words, and those sources have their own arms.
        if !matches!(
            issue,
            CaptureIssue::Sparse { .. } | CaptureIssue::Dense { .. }
        ) {
            assert!(
                text.contains(" — ") || text.contains("; "),
                "no remedy in {issue:?}: {text}"
            );
        }
        // The eaten-`\` law: these are committed, user-facing literals.
        assert!(!text.contains("  "), "a run of spaces in {issue:?}: {text}");
        // Nothing blocking survives a successful run.
        assert!(
            !issue.blocks(),
            "a finished run carries a blocking finding: {issue}"
        );
    }
}

// ── (i) the session has ONE owner, and says so by name ──────────────────────

#[test]
fn a_run_in_flight_refuses_every_door_that_would_reach_around_it() {
    // `CaptureError::Busy` existed and nothing asserted it. The three doors that
    // can be pushed while a solve is running are a second start, a re-finish and
    // a LOAD — and the load is the one that mattered, because it replaces the
    // published state as well as the photographs: before P25.4's audit it left
    // the session reading `Idle` over a worker still running, which makes Cancel
    // answer `false` and lets that worker publish, minutes later, a product for
    // photographs nobody has loaded any more.
    let dir = tempfile::tempdir().expect("tempdir");
    let project = AssetProject::open(dir.path()).expect("a project");
    let before = project.db().len();

    let files = photo_files();
    let mut session = loaded_session(files);
    session.start(pool()).expect("the run starts");
    assert!(matches!(session.state(), CaptureState::Running(_)));

    let busy = |err: inf_editor_core::capture::CaptureError| {
        let text = err.to_string();
        assert!(
            text.contains("already running"),
            "a door refused with something other than Busy: {text}"
        );
    };
    busy(session.start(pool()).expect_err("a second run must refuse"));
    busy(
        session
            .refinish(pool())
            .expect_err("a re-finish must refuse"),
    );
    busy(
        session
            .load_photos(&files[..3])
            .expect_err("a load must refuse"),
    );
    // The refusals refused rather than half-applying: the six are still loaded
    // and the run still owns the session.
    assert_eq!(session.photos().len(), 6);
    assert!(matches!(session.state(), CaptureState::Running(_)));

    assert!(
        session.cancel(),
        "the run is still the one that was started"
    );
    let _ = drain_until_done(&mut session);
    assert_eq!(session.state(), CaptureState::Cancelled);
    assert_eq!(project.db().len(), before, "a refused door wrote assets");
    assert!(!dir.path().join(SCAN_FOLDER).exists());
}

// ── (j) the Cancel that arrives with nothing left to cancel ─────────────────

#[test]
fn a_cancel_with_no_stage_left_to_skip_lets_the_run_finish_and_settles_ready() {
    // Cancellation is read BETWEEN stages, and `Finish` is the last automatic
    // one — so a Cancel pressed while it runs finds no stage left to skip. This
    // pins the outcome the docs and the panel now state: the run COMPLETES and
    // settles `Ready` rather than pretending to have stopped, and the guarantee
    // that is actually worth something is untouched, because `Write` is the
    // user's and nothing was written.
    //
    // A re-finish is that stage on its own, so the cancel lands inside it
    // deterministically rather than on a timer.
    let dir = tempfile::tempdir().expect("tempdir");
    let project = AssetProject::open(dir.path()).expect("a project");
    let before = project.db().len();

    let mut session = PhotogrammetrySession::new();
    let mut cfg = capture_config();
    // Nothing here reads a bake; the state machine is the subject.
    cfg.finish.bake.size = 128;
    cfg.finish.bake.ao_rays = 4;
    session.set_config(cfg);
    session
        .load_photos(photo_files())
        .expect("an idle session loads");
    let _ = run_to_completion(&mut session);
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "{:?}",
        session.error()
    );

    session.refinish(pool()).expect("the re-finish starts");
    assert!(session.cancel(), "the re-finish is running");
    let events = drain_until_done(&mut session);
    println!(
        "P25.4 late cancel: settled {:?} after {} events",
        session.state(),
        events.len()
    );
    assert_eq!(
        session.state(),
        CaptureState::Ready,
        "a cancel with no stage left to skip did not let the finish complete"
    );
    assert!(
        !events.iter().any(|e| e.phase == CapturePhase::Cancelled),
        "the run reported a stop it did not make: {events:?}"
    );
    // The product is there and is a real one…
    let triangles = session
        .with_product(|p| p.finished.report.final_triangles)
        .expect("a completed re-finish has a product");
    assert!(triangles >= MIN_FINAL_TRIANGLES, "{triangles} triangles");
    // …and the thing the button actually guarantees still holds.
    assert_eq!(project.db().len(), before, "a cancelled run wrote assets");
    assert!(!dir.path().join(SCAN_FOLDER).exists());
}
