//! The P25.2 gate: **poses in, depth maps and one fused mesh out, within
//! measured bounds of geometry that is *given*, the same bytes every time, and
//! the same answer on the adapter as on the CPU.**
//!
//! The dataset is `inf_photo_gpu::fixture`'s dihedral — a convex ridge, a raised
//! block that occludes the floor, and a floor, rendered from six lopsided
//! stations by P25.1's deterministic analytic renderer. Every pixel of every
//! view has an **analytic depth** and every point in space has an **analytic
//! distance to the surface**, so the dense stage is measured rather than
//! sanity-checked.
//!
//! # What limits the numbers, measured rather than assumed
//!
//! [`the_dense_stage_reaches_the_fixtures_own_noise_floor`] is the arm that
//! makes the absolute bounds mean something. It runs the **identical pipeline**
//! on the **exact** camera poses the renderer used, so the only error left is
//! the census sampling, the plane quantisation and the parabola. That number is
//! the floor; no dense stage over these photographs can beat it, and the claim
//! about the pipeline is that it lands near it rather than above it.
//!
//! # The two tiers, for the GPU
//!
//! The golden harness's doctrine, applied to compute:
//!
//! * **Same-adapter determinism is asserted hard** — the GPU pipeline run twice
//!   in one process must be byte-identical, no tolerance.
//! * **CPU against GPU is measured**, because a driver may contract a
//!   multiply-add and a Vulkan `f32` divide is allowed 2.5 ULP. The census stage
//!   is bit-exact by construction (integers only) and *is* asserted hard; the
//!   sweep is held to a bound that was measured, is stated in the arm, and is
//!   proved non-vacuous.
//! * Every GPU arm **skips cleanly** with no adapter.
//!
//! Every arm here is mutation-verified; the batch report records which arms go
//! red for which severing.
//!
//! `clippy::neg_cmp_op_on_partial_ord` is allowed here for the reason the crate
//! allows it: `!(x > 0.0)` rejects a `NaN` and `x <= 0.0` accepts it, and a gate
//! that measured a `NaN` depth as valid would be measuring nothing.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::OnceLock;

use glam::DVec3;
use inf_core::job::JobPool;
use inf_photo::align::{pose_error, Similarity};
use inf_photo::synth::{SynthDataset, SynthScene};
use inf_photo::{reconstruct, Reconstruction, SfmConfig};
use inf_photo_gpu::dense::{reconstruct_dense, reconstruct_dense_with, CpuBackend, DenseBackend};
use inf_photo_gpu::sweep::{census_transform, CensusConfig};
use inf_photo_gpu::tsdf::surface_hints;
use inf_photo_gpu::{
    fixture, DenseAdvisory, DenseConfig, DenseError, DenseGpu, DenseReconstruction, GpuBackend,
    TsdfGrid,
};

// ─────────────────────────────────────────────────────────────────────────────
// The bounds
//
// Measured on the committed fixture 2026-08-11 (six 320x240 views, focal 300 px,
// k1 = -0.09, k2 = 0.02; `DenseConfig::default()`: 64 planes, 3 neighbours,
// 5x5 census at dilation 2) and then set with real margin. They are floors and
// ceilings, not descriptions: a change that makes the reconstruction worse than
// this is a regression, and one that makes it much better should tighten them
// rather than leave slack nobody notices.
//
// MEASURED, sparse stage first (this is P25.1 on this fixture, not a claim of
// this batch): 6/6 cameras, 449 points, 1113 observations, 0.4055 px RMS, worst
// camera centre 0.0144 m, worst relative rotation 0.3841 deg, one reconstruction
// unit = 0.8782 m.
//
// MEASURED, dense stage on those poses: 40.3% of all pixels carry a depth after
// filtering; absolute depth error against the analytic truth p50 0.0416 m,
// p75 0.0740, p90 0.1128, p95 0.1409, p99 0.2061; 57.9% inside 5 cm and 86.4%
// inside 10 cm. Voxel size 0.0355 m; grid 206 x 121 x 133; 2 378 388 voxel
// votes; 865 272 surface voxels; 104 206 vertices and 225 564 triangles; zero
// boundary edges and 8 221 of 330 125 edges non-manifold (2.5%).
//
// MEASURED, the fixture's own floor (the same pipeline on the *exact* poses):
// p50 0.0305 m, p90 0.0913 — so the solved-pose run sits at 1.36x the floor.
//
// MEASURED, the mesh: front-facing vertices mean 1.59 voxels from the truth
// surface, p90 3.47, p99 6.07; back-facing 2.90 (the slab's far face, against a
// 3-voxel truncation); 76.2% of 5 825 visible truth samples have a vertex within
// 4 voxels.
//
// MEASURED, GPU against CPU on an RTX 4070 Ti / Vulkan: census **0 of 460 800**
// words differ; TSDF fusion casts an identical 5 058 votes with **0 weight
// words** differing and 404 of 40 800 distance words differing by at most
// 3.6e-6 of a truncation; the sweep agrees on **validity for every pixel**, is
// bit-identical for 83.4% of them, is inside 1e-6 relative for 99.998%, and its
// worst disagreement is 3.94e-3 relative — under half of one plane step.
// ─────────────────────────────────────────────────────────────────────────────

/// Every station must get a depth map.
const REQUIRED_MAPS: usize = 6;
/// A floor on the share of all pixels that carry a depth after filtering. The
/// fixture's upper frame is genuinely empty background, so this can never be
/// near one.
const MIN_VALID_FRACTION: f64 = 0.30;
/// A floor on the **weakest** station's share, not the average — see the arm
/// for the mutation that made this necessary. Measured: the six stations keep
/// 47.4, 46.6, 40.1, 30.7, 44.7 and 32.0 percent of their pixels.
const MIN_VALID_FRACTION_PER_VIEW: f64 = 0.15;
/// Median absolute depth error against the analytic truth, in **metres**.
const MAX_MEDIAN_DEPTH_ERROR_M: f64 = 0.070;
/// 90th-percentile absolute depth error, in **metres**.
const MAX_P90_DEPTH_ERROR_M: f64 = 0.180;
/// Share of depths inside 5 cm of truth.
const MIN_WITHIN_5CM: f64 = 0.45;
/// Share of depths inside 10 cm of truth.
const MIN_WITHIN_10CM: f64 = 0.75;
/// How far above the fixture's own noise floor the solved-pose run may sit.
const MAX_FLOOR_RATIO: f64 = 2.0;

/// Mean distance from a **front-facing** mesh vertex to the truth surface, in
/// **voxels** — scale-free, because that is the unit the fusion's resolution is
/// set in.
const MAX_MEAN_FRONT_VOXELS: f64 = 3.0;
/// 90th percentile of the same.
const MAX_P90_FRONT_VOXELS: f64 = 6.0;
/// Share of edges allowed to be non-manifold (used by more than two triangles).
/// Surface Nets produces these where two sheets of surface pass through
/// neighbouring cells; **boundary** edges, which are the watertightness claim,
/// are asserted at exactly zero.
const MAX_NONMANIFOLD_FRACTION: f64 = 0.06;
/// Share of *visible* truth-surface samples that must have a mesh vertex within
/// [`COMPLETENESS_VOXELS`] of them.
const MIN_COMPLETENESS: f64 = 0.55;
/// The completeness radius, in **voxels**.
const COMPLETENESS_VOXELS: f64 = 4.0;

/// Share of both-valid pixels that must be **bit-identical** between the CPU and
/// the GPU.
const MIN_BITWISE_PARITY: f64 = 0.75;
/// Share that must agree to within one part in a million — a last-ULP
/// difference and nothing more.
const MIN_ULP_PARITY: f64 = 0.999;
/// The worst relative depth disagreement allowed between the two backends.
/// A plane step is about 1.3e-2 relative here, so this is under half a plane.
const MAX_PARITY_RELATIVE: f64 = 6.0e-3;
/// The worst allowed gap between the two backends' fused signed distances, as a
/// fraction of the **truncation band** — the only scale a TSDF distance has.
/// A last-ULP difference in the voxel-to-camera multiply-add is `~1e-7`.
const MAX_FUSION_DISTANCE_GAP: f64 = 1.0e-5;
/// Share of fused distance words allowed to differ at all.
const MAX_FUSION_DISTANCE_DIFFERING: f64 = 0.05;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures, rendered and solved once
// ─────────────────────────────────────────────────────────────────────────────

fn data() -> &'static SynthDataset {
    static D: OnceLock<SynthDataset> = OnceLock::new();
    D.get_or_init(fixture::dataset)
}

fn scene() -> &'static SynthScene {
    static S: OnceLock<SynthScene> = OnceLock::new();
    S.get_or_init(fixture::dihedral_scene)
}

/// The sparse reconstruction P25.1 produces from the fixture. Rendering six
/// views and solving them costs about a second, and every arm wants the same
/// answer, so it is done once.
fn sparse() -> &'static Reconstruction {
    static R: OnceLock<Reconstruction> = OnceLock::new();
    R.get_or_init(|| {
        reconstruct(&data().views, &SfmConfig::default(), &JobPool::new(2))
            .expect("the committed fixture reconstructs")
    })
}

/// The similarity taking the reconstruction's baseline units into the truth's
/// metres, and the pose error left over.
fn alignment() -> &'static Similarity {
    static A: OnceLock<Similarity> = OnceLock::new();
    A.get_or_init(|| {
        let r = sparse();
        let truth: Vec<_> = r
            .registered_views()
            .iter()
            .map(|&v| data().truth[v as usize])
            .collect();
        pose_error(&r.poses(), &truth)
            .expect("the poses align")
            .alignment
    })
}

fn dense_cpu(threads: usize) -> DenseReconstruction {
    reconstruct_dense(
        &data().views,
        sparse(),
        &DenseConfig::default(),
        &JobPool::new(threads),
    )
    .expect("the committed fixture densifies")
}

fn solved() -> &'static DenseReconstruction {
    static S: OnceLock<DenseReconstruction> = OnceLock::new();
    S.get_or_init(|| dense_cpu(2))
}

fn gpu_or_skip(name: &str) -> Option<DenseGpu> {
    match DenseGpu::new() {
        Ok(gpu) => {
            eprintln!("{name}: adapter {}", gpu.adapter_name());
            Some(gpu)
        }
        Err(e) => {
            eprintln!("SKIP {name}: no GPU adapter available ({e})");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement helpers
// ─────────────────────────────────────────────────────────────────────────────

struct Quantiles {
    n: usize,
    p50: f64,
    p90: f64,
    p99: f64,
    mean: f64,
}

fn quantiles(mut v: Vec<f64>) -> Quantiles {
    assert!(!v.is_empty(), "quantiles of an empty sample");
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let at = |q: f64| v[((q * (n - 1) as f64) as usize).min(n - 1)];
    Quantiles {
        n,
        p50: at(0.5),
        p90: at(0.9),
        p99: at(0.99),
        mean: v.iter().sum::<f64>() / n as f64,
    }
}

/// Absolute depth errors against the analytic truth, in **metres**, for every
/// pixel that carries a depth. `scale` converts the reconstruction's own unit
/// into metres; a pixel with a depth where the truth has none counts as
/// infinitely wrong rather than being quietly dropped.
fn depth_errors(dense: &DenseReconstruction, scale: f64) -> (Vec<f64>, usize, usize) {
    let d = data();
    let k = fixture::config().intrinsics();
    let mut errors = Vec::new();
    let mut valid = 0usize;
    let mut total = 0usize;
    for map in &dense.depth_maps {
        let truth = fixture::truth_depth(scene(), &k, &d.truth[map.view as usize]);
        for (depth, want) in map.depth.iter().zip(&truth) {
            total += 1;
            if !(*depth > 0.0) {
                continue;
            }
            valid += 1;
            if *want > 0.0 {
                errors.push((*depth as f64 * scale - *want).abs());
            } else {
                errors.push(f64::INFINITY);
            }
        }
    }
    (errors, valid, total)
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) Depth against analytic truth
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_depth_maps_land_on_the_analytic_truth() {
    let dense = solved();
    let scale = alignment().scale;
    assert_eq!(
        dense.depth_maps.len(),
        REQUIRED_MAPS,
        "only {} of six stations were swept; advisories {:?}",
        dense.depth_maps.len(),
        dense.advisories
    );
    let (errors, valid, total) = depth_errors(dense, scale);
    let fraction = valid as f64 / total as f64;
    let q = quantiles(errors.clone());
    let share = |bound: f64| errors.iter().filter(|e| **e <= bound).count() as f64 / q.n as f64;
    let (w5, w10) = (share(0.05), share(0.10));
    println!(
        "P25.2 depth: {valid}/{total} valid ({:.1}%), error m p50 {:.4} p90 {:.4} p99 {:.4}, \
         within 5 cm {:.1}%, within 10 cm {:.1}%, one unit = {scale:.4} m",
        100.0 * fraction,
        q.p50,
        q.p90,
        q.p99,
        100.0 * w5,
        100.0 * w10
    );
    assert!(
        fraction >= MIN_VALID_FRACTION,
        "only {:.1}% of pixels carry a depth (floor {:.0}%)",
        100.0 * fraction,
        100.0 * MIN_VALID_FRACTION
    );
    // **The weakest station, not the average.** Found by mutation: reverting the
    // depth-range prior to "points this view was MATCHED against" rather than
    // "points that project into this view's frame" leaves the total valid
    // fraction at 34.2% — comfortably over the floor above — while the one
    // station whose sparse share is thinnest collapses from 30.7% of its pixels
    // to 1.0%. An average hides a whole camera going dark. This is the arm that
    // does not, and it is joined by the advisory record asserted EMPTY, because
    // `SparseDepthMap` is precisely the finding that station would earn.
    let weakest = dense
        .depth_maps
        .iter()
        .map(|m| (m.valid_count() as f64 / m.pixels() as f64, m.view))
        .fold((1.0f64, u32::MAX), |a, b| if b.0 < a.0 { b } else { a });
    println!(
        "P25.2 weakest station: view {} keeps {:.1}% of its pixels",
        weakest.1,
        100.0 * weakest.0
    );
    assert!(
        weakest.0 >= MIN_VALID_FRACTION_PER_VIEW,
        "view {} keeps only {:.1}% of its pixels (floor {:.0}%) — one station is dark",
        weakest.1,
        100.0 * weakest.0,
        100.0 * MIN_VALID_FRACTION_PER_VIEW
    );
    let sparse_maps: Vec<&DenseAdvisory> = dense
        .advisories
        .iter()
        .filter(|a| matches!(a, DenseAdvisory::SparseDepthMap { .. }))
        .collect();
    assert!(
        sparse_maps.is_empty(),
        "a station came back sparse on the committed fixture: {sparse_maps:?}"
    );
    assert!(
        q.p50 <= MAX_MEDIAN_DEPTH_ERROR_M,
        "median depth error is {:.4} m (bound {MAX_MEDIAN_DEPTH_ERROR_M} m)",
        q.p50
    );
    assert!(
        q.p90 <= MAX_P90_DEPTH_ERROR_M,
        "p90 depth error is {:.4} m (bound {MAX_P90_DEPTH_ERROR_M} m)",
        q.p90
    );
    assert!(
        w5 >= MIN_WITHIN_5CM,
        "only {:.1}% of depths are inside 5 cm (floor {:.0}%)",
        100.0 * w5,
        100.0 * MIN_WITHIN_5CM
    );
    assert!(
        w10 >= MIN_WITHIN_10CM,
        "only {:.1}% of depths are inside 10 cm (floor {:.0}%)",
        100.0 * w10,
        100.0 * MIN_WITHIN_10CM
    );
    // And the sweep really used its range rather than pinning everything to one
    // plane: a depth map of a constant would pass every bound above on a scene
    // this flat if the bounds were loose enough.
    let mut spread: Vec<f64> = dense.depth_maps[0]
        .depth
        .iter()
        .filter(|d| **d > 0.0)
        .map(|d| *d as f64)
        .collect();
    spread.sort_by(f64::total_cmp);
    let (near, far) = (spread[0], spread[spread.len() - 1]);
    let distinct: std::collections::BTreeSet<u32> = dense.depth_maps[0]
        .depth
        .iter()
        .filter(|d| **d > 0.0)
        .map(|d| d.to_bits())
        .collect();
    println!(
        "P25.2 view 0 depths: {near:.3}..{far:.3} units, {} distinct",
        distinct.len()
    );
    assert!(
        far > near * 1.5,
        "view 0's depths span only {near:.3}..{far:.3} — the sweep is returning one plane"
    );
    // The sub-plane refinement really refines: without it a depth map could
    // hold at most `planes` distinct values.
    //
    // MEASURED, and worth writing down rather than being surprised by: view 0
    // has 3 333 distinct depths from 36 407 pixels, which is about 52 per plane
    // rather than the continuum a parabola suggests. That is not a defect — the
    // costs the parabola is fitted through are **integers** (a census Hamming
    // distance rescaled by `COST_SCALE / taps`), so `0.5 (a - c) / (a - 2b + c)`
    // takes values from a finite set. Discrete costs buy the CPU/GPU parity this
    // crate is built on, and this is what they cost.
    let planes = DenseConfig::default().sweep.planes as usize;
    assert!(
        distinct.len() > planes * 8,
        "view 0 has only {} distinct depths over {planes} planes — the sub-plane refinement is \
         not running",
        distinct.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) The fixture's own noise floor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_dense_stage_reaches_the_fixtures_own_noise_floor() {
    // The arm that makes the absolute bounds above mean something. Run the
    // IDENTICAL pipeline on the exact poses the renderer used: whatever error
    // is left is the census sampling, the plane quantisation and the parabola,
    // and no dense stage over these photographs can beat it. The real claim
    // about the pipeline is that it lands near that floor rather than above it.
    let d = data();
    let truth_recon = fixture::with_truth_poses(sparse(), &d.truth).expect("the poses align");
    let floor_run = reconstruct_dense(
        &d.views,
        &truth_recon,
        &DenseConfig::default(),
        &JobPool::new(2),
    )
    .expect("the exact poses densify");
    // Truth poses are already in metres, so the scale is exactly one.
    let (floor_errors, floor_valid, floor_total) = depth_errors(&floor_run, 1.0);
    let floor = quantiles(floor_errors);
    let (errors, _, _) = depth_errors(solved(), alignment().scale);
    let solved_q = quantiles(errors);
    println!(
        "P25.2 floor: exact poses p50 {:.4} m p90 {:.4} m over {floor_valid}/{floor_total} pixels; \
         solved poses p50 {:.4} m p90 {:.4} m; ratio {:.2}x",
        floor.p50,
        floor.p90,
        solved_q.p50,
        solved_q.p90,
        solved_q.p50 / floor.p50
    );
    assert!(
        floor.p50 > 0.005,
        "the fixture has no depth error at all at exact poses ({:.5} m) — this arm is measuring \
         nothing",
        floor.p50
    );
    assert!(
        solved_q.p50 <= floor.p50 * MAX_FLOOR_RATIO,
        "the solved-pose run's {:.4} m is {:.2}x the fixture's own {:.4} m floor (bound \
         {MAX_FLOOR_RATIO}x)",
        solved_q.p50,
        solved_q.p50 / floor.p50,
        floor.p50
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (c) The fused mesh against the truth surface
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_fused_mesh_lands_on_the_truth_surface_and_is_closed() {
    let dense = solved();
    let similarity = alignment();
    let voxel_m = dense.report.voxel_size * similarity.scale;
    let mesh = &dense.mesh;
    assert!(
        mesh.triangle_count() > 20_000,
        "the fused mesh has only {} triangles",
        mesh.triangle_count()
    );

    // Watertight, as a computed claim rather than as the mesher's promise: this
    // crate WELDED the per-chunk buffers, so the property has to be re-checked
    // over the weld it performed.
    let edges = mesh.edge_use_counts();
    let boundary = edges.values().filter(|c| **c == 1).count();
    let nonmanifold = edges.values().filter(|c| **c > 2).count();
    let nm_fraction = nonmanifold as f64 / edges.len() as f64;

    // Where a vertex is, in the truth's frame.
    let world: Vec<DVec3> = mesh
        .positions
        .iter()
        .map(|p| similarity.apply(*p))
        .collect();
    let centroid =
        data().truth.iter().fold(DVec3::ZERO, |a, p| a + p.center()) / data().truth.len() as f64;
    let mut front = Vec::new();
    let mut back = Vec::new();
    for (i, w) in world.iter().enumerate() {
        let n = mesh.normals[i];
        let normal =
            similarity
                .rotation
                .mul_vec3(DVec3::new(n[0] as f64, n[1] as f64, n[2] as f64));
        let distance = fixture::surface_distance(scene(), *w) / voxel_m;
        // "Facing the capture" is the dot product with the direction back to the
        // stations' centroid. The TSDF of a one-sided capture is a SLAB: this
        // face is the reconstruction, the other is the truncation band's far
        // edge, and the module docs say so.
        if normal.dot((centroid - *w).normalize()) > 0.3 {
            front.push(distance);
        } else {
            back.push(distance);
        }
    }
    let f = quantiles(front);
    let b = quantiles(back);

    // Completeness: every part of the truth surface at least two stations can
    // see should have a vertex near it. Without this, a reconstruction of one
    // square metre would pass the accuracy bounds perfectly.
    let samples = fixture::visible_surface_samples(scene(), &fixture::config(), 0.09);
    let radius = COMPLETENESS_VOXELS * voxel_m;
    let mut covered = 0usize;
    for s in &samples {
        if world
            .iter()
            .any(|w| (*w - *s).length_squared() <= radius * radius)
        {
            covered += 1;
        }
    }
    let completeness = covered as f64 / samples.len() as f64;

    println!(
        "P25.2 mesh: {} verts {} tris, voxel {:.4} m; front-facing n={} mean {:.2} p90 {:.2} \
         p99 {:.2} voxels; back-facing n={} p50 {:.2} voxels; boundary edges {boundary}, \
         non-manifold {nonmanifold}/{} ({:.2}%); completeness {:.1}% within {COMPLETENESS_VOXELS} \
         voxels of {} samples",
        mesh.vertex_count(),
        mesh.triangle_count(),
        voxel_m,
        f.n,
        f.mean,
        f.p90,
        f.p99,
        b.n,
        b.p50,
        edges.len(),
        100.0 * nm_fraction,
        100.0 * completeness,
        samples.len()
    );

    assert_eq!(
        boundary, 0,
        "the welded mesh has {boundary} boundary edges — it is not closed"
    );
    assert!(
        nm_fraction <= MAX_NONMANIFOLD_FRACTION,
        "{:.2}% of edges are non-manifold (bound {:.0}%)",
        100.0 * nm_fraction,
        100.0 * MAX_NONMANIFOLD_FRACTION
    );
    assert!(
        f.mean <= MAX_MEAN_FRONT_VOXELS,
        "front-facing vertices average {:.2} voxels from the truth surface (bound \
         {MAX_MEAN_FRONT_VOXELS})",
        f.mean
    );
    assert!(
        f.p90 <= MAX_P90_FRONT_VOXELS,
        "front-facing p90 is {:.2} voxels (bound {MAX_P90_FRONT_VOXELS})",
        f.p90
    );
    assert!(
        completeness >= MIN_COMPLETENESS,
        "only {:.1}% of visible truth samples have a vertex within {COMPLETENESS_VOXELS} voxels \
         (floor {:.0}%)",
        100.0 * completeness,
        100.0 * MIN_COMPLETENESS
    );
    // The slab, asserted rather than glossed: the far face sits about one
    // truncation behind the surface, which is what the band IS. If a capture
    // ever surrounds its subject this arm is the one that changes.
    let truncation_voxels = DenseConfig::default().tsdf.truncation_voxels;
    assert!(
        b.n > f.n / 4,
        "the mesh has only {} back-facing vertices to {} front-facing — the slab claim is not \
         being measured",
        b.n,
        f.n
    );
    assert!(
        b.p50 > f.p50 && b.p50 < truncation_voxels * 2.5,
        "the back face sits {:.2} voxels from the surface; the truncation band is \
         {truncation_voxels} voxels and the front face is at {:.2}",
        b.p50,
        f.p50
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (d) Determinism
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_dense_stage_is_byte_identical_twice_and_across_pool_sizes() {
    // The P16.4 law. Every parallel stage goes through an in-order pure map, no
    // float sum is accumulated in parallel, and TSDF fusion is serial across
    // views — so this is structural, not a hope.
    let a = dense_cpu(2).canonical_bytes();
    let b = dense_cpu(2).canonical_bytes();
    let one = dense_cpu(1).canonical_bytes();
    let eight = dense_cpu(8).canonical_bytes();
    let mismatch = |x: &[u8], y: &[u8]| x.iter().zip(y).position(|(p, q)| p != q);
    assert!(
        a == b,
        "two runs differ at byte {:?} (lengths {} and {})",
        mismatch(&a, &b),
        a.len(),
        b.len()
    );
    assert!(
        a == one,
        "2 vs 1 worker differ at byte {:?}",
        mismatch(&a, &one)
    );
    assert!(
        a == eight,
        "2 vs 8 workers differ at byte {:?}",
        mismatch(&a, &eight)
    );
    assert!(
        a.len() > 1_000_000,
        "the byte image is suspiciously small: {} bytes",
        a.len()
    );
}

#[test]
fn the_canonical_bytes_cover_every_field_they_claim_to() {
    // P25.1's F5: `canonical_bytes` wrote one field of its report, so both of
    // its byte-identity arms were comparing an image that omitted a branch
    // point of the algorithm. This arm changes one field at a time and insists
    // the bytes notice.
    let base = solved().clone();
    let bytes = base.canonical_bytes();
    let mut changed: Vec<&str> = Vec::new();
    macro_rules! mutate {
        ($name:literal, $r:ident, $body:block) => {{
            #[allow(unused_mut)]
            let mut $r = base.clone();
            $body
            assert_ne!(
                $r.canonical_bytes(),
                bytes,
                "changing {} left the canonical bytes identical",
                $name
            );
            changed.push($name);
        }};
    }

    mutate!("a depth", r, {
        let d = &mut r.depth_maps[0].depth;
        let i = d.iter().position(|v| *v > 0.0).expect("a depth");
        d[i] = f32::from_bits(d[i].to_bits() + 1);
    });
    mutate!("a confidence", r, {
        let c = &mut r.depth_maps[1].confidence;
        let i = c.iter().position(|v| *v > 0.0).expect("a confidence");
        c[i] = f32::from_bits(c[i].to_bits() + 1);
    });
    mutate!("a depth map's view index", r, {
        r.depth_maps[2].view += 100;
    });
    mutate!("a depth map's width", r, {
        r.depth_maps[3].width += 1;
    });
    mutate!("a surface normal", r, {
        r.surface[0].normals[0][1] = 0.25;
    });
    mutate!("a fusion weight", r, {
        let w = &mut r.surface[1].weights;
        let i = w.iter().position(|v| *v > 0.0).expect("a weight");
        w[i] = f32::from_bits(w[i].to_bits() + 1);
    });
    mutate!("a mesh position", r, {
        let p = &mut r.mesh.positions[0];
        p.z = f64::from_bits(p.z.to_bits() + 1);
    });
    mutate!("a mesh normal", r, {
        r.mesh.normals[0][0] = 0.5;
    });
    mutate!("a vertex confidence", r, {
        r.mesh.confidence[0] = 0.125;
    });
    mutate!("a mesh index", r, {
        r.mesh.indices[0] += 1;
    });
    mutate!("an advisory", r, {
        r.advisories.push(DenseAdvisory::ThinNeighbourhood {
            view: 0,
            used: 1,
            wanted: 3,
        });
    });
    mutate!("report.cameras", r, {
        r.report.cameras += 1;
    });
    mutate!("report.swept", r, {
        r.report.swept += 1;
    });
    mutate!("report.planes", r, {
        r.report.planes += 1;
    });
    mutate!("report.valid_per_view", r, {
        r.report.valid_per_view[0] += 1;
    });
    mutate!("report.near_per_view", r, {
        let v = &mut r.report.near_per_view[0];
        *v = f64::from_bits(v.to_bits() + 1);
    });
    mutate!("report.far_per_view", r, {
        let v = &mut r.report.far_per_view[0];
        *v = f64::from_bits(v.to_bits() + 1);
    });
    mutate!("report.voxel_size", r, {
        let v = &mut r.report.voxel_size;
        *v = f64::from_bits(v.to_bits() + 1);
    });
    mutate!("report.grid_dims", r, {
        r.report.grid_dims[2] += 1;
    });
    mutate!("report.votes", r, {
        r.report.votes += 1;
    });
    mutate!("report.surface_voxels", r, {
        r.report.surface_voxels += 1;
    });
    mutate!("report.vertices", r, {
        r.report.vertices += 1;
    });
    mutate!("report.triangles", r, {
        r.report.triangles += 1;
    });

    // Anti-vacuity from the other side: an untouched clone must still match, or
    // the arm above would pass for a `canonical_bytes` that simply differed on
    // every call.
    assert_eq!(
        base.clone().canonical_bytes(),
        bytes,
        "canonical_bytes is not a function of the reconstruction"
    );
    assert!(
        changed.len() >= 22,
        "only {} fields were exercised: {changed:?}",
        changed.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (e) GPU parity, stage by stage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_census_transform_is_bit_exact_on_the_gpu() {
    // The one stage where "bit-exact" is a claim about the ARITHMETIC rather
    // than about an adapter: every operation is an integer comparison of `u8`
    // intensities packed into a `u32`. No float, no rounding mode, no
    // contraction, nothing for a driver to decide. It is therefore asserted
    // hard, with no tolerance and no envelope.
    let Some(gpu) = gpu_or_skip("census parity") else {
        return;
    };
    let cfg = CensusConfig::default();
    let mut differed = 0usize;
    let mut words = 0usize;
    let mut distinct = std::collections::BTreeSet::new();
    for view in &data().views {
        let cpu = census_transform(&view.image, &cfg);
        let gpu_codes = gpu.census(&view.image, &cfg);
        assert_eq!(cpu.len(), gpu_codes.len());
        words += cpu.len();
        differed += cpu.iter().zip(&gpu_codes).filter(|(a, b)| a != b).count();
        distinct.extend(cpu.iter().copied());
    }
    println!(
        "P25.2 census parity: {differed} of {words} words differ, {} distinct codes",
        distinct.len()
    );
    assert_eq!(differed, 0, "the census transform is not bit-exact");
    // Anti-vacuity: an all-zero census would also be "bit-exact".
    assert!(
        distinct.len() > 1000,
        "only {} distinct census codes — the transform is not encoding anything",
        distinct.len()
    );
}

#[test]
fn the_gpu_sweep_agrees_with_the_cpu_within_a_measured_bound() {
    // The two-tier doctrine. The sweep's arithmetic is `f32` and its innermost
    // operations are multiply-adds and a division; a driver may CONTRACT a
    // multiply-add into an fma, and Vulkan allows an `f32` divide 2.5 ULP. One
    // ULP in the sample coordinate only matters where `floor(u + 0.5)` is within
    // one ULP of an integer — and there it is a whole different census word,
    // which can move the argmin by a plane. So the claim is measured:
    //
    //   * the two backends NEVER disagree about whether a pixel has a depth;
    //   * most pixels are bit-identical;
    //   * nearly all the rest differ by a last ULP;
    //   * and the worst disagreement stays under half a plane step.
    let Some(gpu) = gpu_or_skip("sweep parity") else {
        return;
    };
    let mut backend = GpuBackend::new(&gpu);
    let on_gpu = reconstruct_dense_with(
        &data().views,
        sparse(),
        &DenseConfig::default(),
        &JobPool::new(2),
        &mut backend,
    )
    .expect("the fixture densifies on the adapter");
    let on_cpu = solved();

    let mut validity = 0usize;
    let mut identical = 0usize;
    let mut both = 0usize;
    let mut relative: Vec<f64> = Vec::new();
    for (a, b) in on_cpu.depth_maps.iter().zip(&on_gpu.depth_maps) {
        assert_eq!(a.view, b.view);
        for i in 0..a.pixels() {
            let (av, bv) = (a.depth[i] > 0.0, b.depth[i] > 0.0);
            if av != bv {
                validity += 1;
                continue;
            }
            if !av {
                continue;
            }
            both += 1;
            if a.depth[i].to_bits() == b.depth[i].to_bits() {
                identical += 1;
            }
            relative.push(((a.depth[i] - b.depth[i]).abs() / a.depth[i]) as f64);
        }
    }
    assert!(both > 100_000, "only {both} pixels were comparable");
    let bitwise = identical as f64 / both as f64;
    let ulp = relative.iter().filter(|r| **r <= 1e-6).count() as f64 / both as f64;
    let worst = relative.iter().copied().fold(0.0f64, f64::max);
    println!(
        "P25.2 sweep parity: {both} comparable pixels, {validity} validity disagreements, \
         {:.3}% bit-identical, {:.4}% within 1e-6 relative, worst {worst:.3e} relative",
        100.0 * bitwise,
        100.0 * ulp
    );
    assert_eq!(
        validity, 0,
        "{validity} pixels have a depth on one backend and not the other"
    );
    assert!(
        bitwise >= MIN_BITWISE_PARITY,
        "only {:.2}% of pixels are bit-identical (floor {:.0}%)",
        100.0 * bitwise,
        100.0 * MIN_BITWISE_PARITY
    );
    assert!(
        ulp >= MIN_ULP_PARITY,
        "only {:.4}% of pixels agree to 1e-6 relative (floor {:.1}%)",
        100.0 * ulp,
        100.0 * MIN_ULP_PARITY
    );
    assert!(
        worst <= MAX_PARITY_RELATIVE,
        "the worst backend disagreement is {worst:.3e} relative (bound {MAX_PARITY_RELATIVE:.0e})"
    );
    // Anti-vacuity: a shader that wrote zeros everywhere would have zero
    // validity disagreements only if the CPU also did. The reports have to
    // describe the same reconstruction.
    assert!(
        on_gpu.report.vertices > 20_000 && on_gpu.report.votes > 1_000_000,
        "the GPU run produced {} vertices from {} votes",
        on_gpu.report.vertices,
        on_gpu.report.votes
    );
}

#[test]
fn the_gpu_fusion_agrees_with_the_cpu_on_identical_inputs() {
    // Fusion parity in isolation, so a difference here is the fusion kernel's
    // and not an inherited difference in the depth maps it was handed. Both
    // backends integrate the SAME depth maps and the SAME weights into the same
    // grid, view by view, in the same order.
    let Some(gpu) = gpu_or_skip("fusion parity") else {
        return;
    };
    let dense = solved();
    let cameras: Vec<_> = sparse().cameras.values().copied().collect();
    let pool = JobPool::new(2);
    // A grid deliberately smaller than the pipeline's, so this arm is quick and
    // its numbers are its own.
    let origin = dense.mesh.positions[0] - DVec3::splat(0.6);
    let mut a = TsdfGrid::new(origin, dense.report.voxel_size, [40, 34, 30], {
        (dense.report.voxel_size * DenseConfig::default().tsdf.truncation_voxels) as f32
    });
    let mut b = a.clone();
    let (mut votes_a, mut votes_b) = (0usize, 0usize);
    for (camera, map) in cameras.iter().zip(&dense.depth_maps) {
        let hints = surface_hints(camera, map, DenseConfig::default().filter.speckle_delta);
        let params = a.fuse_params(camera, map);
        votes_a += CpuBackend.integrate(&mut a, &params, map, &hints.weights, &pool);
        votes_b += gpu.integrate(&mut b, &params, map, &hints.weights);
    }
    let distance = a
        .distance
        .iter()
        .zip(&b.distance)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let weight = a
        .weight
        .iter()
        .zip(&b.weight)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    // The distances that DO differ, as a fraction of the truncation band — the
    // only scale a signed distance in a TSDF has.
    let worst = a
        .distance
        .iter()
        .zip(&b.distance)
        .map(|(x, y)| ((x - y).abs() / a.truncation) as f64)
        .fold(0.0f64, f64::max);
    println!(
        "P25.2 fusion parity: votes {votes_a} vs {votes_b}; {distance} of {} distance words and \
         {weight} weight words differ; worst distance gap {worst:.3e} of a truncation",
        a.len()
    );
    // Anti-vacuity first: a grid nothing voted into would agree perfectly.
    assert!(
        votes_a > 200,
        "only {votes_a} votes were cast — this arm is comparing two empty grids"
    );
    // The vote count and the weight field are **integer-selected**: which depth
    // sample a voxel reads is `floor(px + 0.5)` and what it adds is a value read
    // straight out of a buffer. Both are asserted hard, because a difference in
    // either means the two backends disagreed about WHICH pixel a voxel sees,
    // not about the last bit of an arithmetic result.
    assert_eq!(
        votes_a, votes_b,
        "the two backends cast different numbers of votes — they disagree about which depth \
         samples a voxel can see"
    );
    assert_eq!(
        weight, 0,
        "{weight} accumulated weights differ; the weight a voxel adds is read from a buffer, so a \
         difference here is a different sample, not a different sum"
    );
    // The distance is `w * (observed - qz)`, and `qz` is the end of a chain of
    // `f32` multiply-adds a driver may contract. THAT is the named operation.
    // Measured on an RTX 4070 Ti / Vulkan: 404 of 40 800 words differ (1.0%),
    // worst gap 3.6e-6 of a truncation.
    assert!(
        worst <= MAX_FUSION_DISTANCE_GAP,
        "the worst fused-distance gap is {worst:.3e} of a truncation (bound \
         {MAX_FUSION_DISTANCE_GAP:.0e}) — that is more than a last-ULP difference in the \
         voxel-to-camera multiply-add"
    );
    assert!(
        distance as f64 <= a.len() as f64 * MAX_FUSION_DISTANCE_DIFFERING,
        "{distance} of {} distance words differ (bound {:.0}%)",
        a.len(),
        100.0 * MAX_FUSION_DISTANCE_DIFFERING
    );
}

#[test]
fn the_gpu_is_byte_identical_twice_on_one_adapter() {
    // Same-adapter determinism, asserted HARD — the golden harness's first
    // tier. Cross-adapter is a different question and this crate does not claim
    // it; the arm above is where cross-implementation drift is measured.
    let Some(gpu) = gpu_or_skip("gpu determinism") else {
        return;
    };
    let run = || {
        let mut backend = GpuBackend::new(&gpu);
        reconstruct_dense_with(
            &data().views,
            sparse(),
            &DenseConfig::default(),
            &JobPool::new(2),
            &mut backend,
        )
        .expect("densifies")
        .canonical_bytes()
    };
    let a = run();
    let b = run();
    assert!(
        a == b,
        "two GPU runs differ at byte {:?} (lengths {} and {})",
        a.iter().zip(&b).position(|(x, y)| x != y),
        a.len(),
        b.len()
    );
    assert!(a.len() > 1_000_000, "byte image too small: {}", a.len());
}

// ─────────────────────────────────────────────────────────────────────────────
// (f) Anti-vacuity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_perturbed_pose_measurably_degrades_the_depth_maps() {
    // If the bounds above could be met by a stage that ignored its poses, they
    // would measure nothing. Rotate one camera by a fifth of a degree and the
    // depth maps must get worse, visibly.
    //
    // **The degradation shows up as COVERAGE, not as the median error** —
    // measured, and worth writing down rather than papering over. A wrong pose
    // makes a pixel's cost curve disagree with its neighbours' depth maps, and
    // the left-right consistency check is precisely the pass that deletes those
    // pixels. So the reconstruction loses a fifth of its depths and the ones
    // that survive are, if anything, slightly *better* than before (p50
    // 0.0416 -> 0.0389 m). That is the filter working, and an arm that only
    // asserted "the error goes up" would have been red for the right reason and
    // the wrong claim.
    let d = data();
    let mut wrong = sparse().clone();
    let view = *wrong.cameras.keys().next().expect("a camera");
    {
        let camera = wrong.cameras.get_mut(&view).expect("a camera");
        // A local rotation increment, through the pose type's own retraction so
        // the perturbed pose is still a valid rigid transform.
        camera.pose = camera
            .pose
            .retract(DVec3::new(0.0035, 0.0, 0.0), DVec3::ZERO);
    }
    let perturbed = reconstruct_dense(&d.views, &wrong, &DenseConfig::default(), &JobPool::new(2))
        .expect("a wrong pose still densifies *something*");
    let scale = alignment().scale;
    let (base_errors, base_valid, _) = depth_errors(solved(), scale);
    let (bad_errors, bad_valid, _) = depth_errors(&perturbed, scale);
    let base = quantiles(base_errors);
    let bad = quantiles(bad_errors);
    println!(
        "P25.2 perturbed pose: p50 {:.4} -> {:.4} m, p90 {:.4} -> {:.4} m, valid {base_valid} -> \
         {bad_valid}",
        base.p50, bad.p50, base.p90, bad.p90
    );
    assert!(
        perturbed.canonical_bytes() != solved().canonical_bytes(),
        "a rotated camera produced a byte-identical dense reconstruction"
    );
    assert!(
        bad_valid * 20 < base_valid * 19,
        "rotating a camera by 0.2 degrees cost only {} of {base_valid} depths — the consistency \
         check is not seeing the disagreement it exists for",
        base_valid - bad_valid
    );
    // And the surviving depths are still depths, not a handful of stragglers:
    // an arm satisfied by "everything was deleted" would measure nothing.
    assert!(
        bad_valid > base_valid / 2,
        "a 0.2 degree rotation deleted {bad_valid} of {base_valid} — that is a collapse, not a \
         degradation, and this fixture is supposed to be measuring the latter"
    );
}

#[test]
fn too_few_neighbour_views_refuses_by_name() {
    // A capture whose photographs do not overlap must stop, by name, rather
    // than sweep a reference view against nothing and return an empty mesh —
    // which is exactly what a *successful* reconstruction of a blank wall also
    // looks like.
    let d = data();
    let mut thin = sparse().clone();
    let target = thin.registered_views()[0];
    // Take away every track that view shares with anything: it now has no
    // covisible neighbour at all.
    for point in thin.points.iter_mut() {
        if point.observations.iter().any(|o| o.view == target) {
            point.observations.retain(|o| o.view == target);
        }
    }
    let err = reconstruct_dense(&d.views, &thin, &DenseConfig::default(), &JobPool::new(1))
        .expect_err("a view with no covisible neighbour must refuse");
    assert!(
        matches!(
            err,
            DenseError::TooFewNeighbours {
                found: 0,
                required: 2,
                ..
            }
        ),
        "expected a named neighbour refusal, got {err:?}"
    );
    assert!(format!("{err}").contains("covisible"));

    // And the *bound* is a bound rather than a blanket rejection: with the
    // minimum two neighbours it still runs.
    let two = DenseConfig {
        sweep: inf_photo_gpu::SweepConfig {
            neighbours: 2,
            ..DenseConfig::default().sweep
        },
        ..DenseConfig::default()
    };
    let ok = reconstruct_dense(&d.views, sparse(), &two, &JobPool::new(2))
        .expect("two neighbours is enough");
    assert_eq!(ok.depth_maps.len(), REQUIRED_MAPS);
    assert!(
        ok.report.votes > 100_000,
        "a two-neighbour sweep cast only {} votes",
        ok.report.votes
    );
}

#[test]
fn the_engagement_counters_are_zero_when_there_is_nothing_to_see() {
    // The house counter doctrine. Fusion can silently no-op — a grid behind the
    // cameras, or one every depth map projects outside of, integrates nothing
    // and reports success. So the counter is asserted BOTH ways: nonzero on the
    // real volume, and exactly zero on a volume nobody photographed.
    let dense = solved();
    assert!(
        dense.report.votes > 1_000_000 && dense.report.surface_voxels > 100_000,
        "the real fusion reports {} votes and {} surface voxels",
        dense.report.votes,
        dense.report.surface_voxels
    );

    let cameras: Vec<_> = sparse().cameras.values().copied().collect();
    let pool = JobPool::new(1);
    // A grid a long way BEHIND the capture: every voxel projects behind every
    // camera, so every vote must be refused at the `qz > 0` test.
    let behind = cameras[0].pose.center() - cameras[0].pose.forward() * 40.0;
    let mut grid = TsdfGrid::new(behind, dense.report.voxel_size, [12, 12, 12], 0.1);
    let mut votes = 0usize;
    for (camera, map) in cameras.iter().zip(&dense.depth_maps) {
        let hints = surface_hints(camera, map, 0.02);
        let params = grid.fuse_params(camera, map);
        votes += CpuBackend.integrate(&mut grid, &params, map, &hints.weights, &pool);
    }
    assert_eq!(
        votes, 0,
        "a grid behind every camera collected {votes} votes"
    );
    assert!(
        grid.weight.iter().all(|w| *w == 0.0),
        "a grid behind every camera accumulated weight"
    );
    // And the same on the adapter, if there is one: a kernel that wrote
    // unconditionally would pass the CPU half of this and fail here.
    if let Some(gpu) = gpu_or_skip("empty-volume counter") {
        let mut gpu_grid = TsdfGrid::new(behind, dense.report.voxel_size, [12, 12, 12], 0.1);
        let mut gpu_votes = 0usize;
        for (camera, map) in cameras.iter().zip(&dense.depth_maps) {
            let hints = surface_hints(camera, map, 0.02);
            let params = gpu_grid.fuse_params(camera, map);
            gpu_votes += gpu.integrate(&mut gpu_grid, &params, map, &hints.weights);
        }
        assert_eq!(
            gpu_votes, 0,
            "the GPU kernel voted {gpu_votes} times on an empty volume"
        );
        assert!(gpu_grid.weight.iter().all(|w| *w == 0.0));
    }
}

#[test]
fn the_advisories_say_what_the_filters_did() {
    // A filter that stopped filtering would leave every accuracy bound in this
    // file green (more pixels, all of them worse on average but not enough to
    // move a p50). The advisories are the record that both passes ran, and the
    // arm that severing either one turns red.
    let dense = solved();
    let consistency: usize = dense
        .advisories
        .iter()
        .filter_map(|a| match a {
            DenseAdvisory::ConsistencyRejected { dropped, .. } => Some(*dropped),
            _ => None,
        })
        .sum();
    let speckle: usize = dense
        .advisories
        .iter()
        .filter_map(|a| match a {
            DenseAdvisory::SpeckleRemoved { dropped, .. } => Some(*dropped),
            _ => None,
        })
        .sum();
    println!(
        "P25.2 filters: consistency dropped {consistency}, speckle dropped {speckle}; advisories \
         {:?}",
        dense.advisories
    );
    assert!(
        consistency > 10_000,
        "the left-right consistency check dropped only {consistency} pixels over six views"
    );
    assert!(
        speckle > 5_000,
        "speckle removal dropped only {speckle} pixels over six views"
    );
    // Both filters must leave something behind, or the arm above would be
    // satisfied by a pass that rejected everything.
    let kept: usize = dense.report.valid_per_view.iter().sum();
    assert!(
        kept > consistency && kept > speckle,
        "the filters dropped more than they kept: {kept} kept, {consistency} and {speckle} dropped"
    );
}

#[test]
fn the_reconstruction_reports_what_it_did() {
    let dense = solved();
    let r = &dense.report;
    assert_eq!(r.cameras, REQUIRED_MAPS);
    assert_eq!(r.swept, dense.depth_maps.len());
    assert_eq!(r.planes, DenseConfig::default().sweep.planes);
    assert_eq!(r.valid_per_view.len(), dense.depth_maps.len());
    assert_eq!(r.near_per_view.len(), dense.depth_maps.len());
    assert_eq!(r.vertices, dense.mesh.vertex_count());
    assert_eq!(r.triangles, dense.mesh.triangle_count());
    assert_eq!(dense.surface.len(), dense.depth_maps.len());
    for (i, map) in dense.depth_maps.iter().enumerate() {
        assert_eq!(r.valid_per_view[i], map.valid_count());
        assert!(
            r.near_per_view[i] > 0.0 && r.far_per_view[i] > r.near_per_view[i],
            "view {} has a depth range of {}..{}",
            map.view,
            r.near_per_view[i],
            r.far_per_view[i]
        );
        assert_eq!(dense.surface[i].weights.len(), map.pixels());
        // A depth and a confidence are set and cleared together, always.
        for j in 0..map.pixels() {
            assert_eq!(
                map.depth[j] > 0.0,
                map.confidence[j] > 0.0,
                "view {} pixel {j} has a depth without a confidence, or the reverse",
                map.view
            );
        }
    }
    // Every index in the mesh addresses a vertex that exists, and every vertex
    // carries all three of its channels.
    assert_eq!(dense.mesh.normals.len(), dense.mesh.positions.len());
    assert_eq!(dense.mesh.confidence.len(), dense.mesh.positions.len());
    assert!(dense
        .mesh
        .indices
        .iter()
        .all(|i| (*i as usize) < dense.mesh.positions.len()));
    assert!(dense.mesh.confidence.iter().all(|c| (0.0..1.0).contains(c)));
    // And the mesh really lives where the grid says it does.
    let dims = r.grid_dims;
    assert!(dims.iter().all(|d| *d >= 2 && *d <= 256), "dims {dims:?}");
    assert_eq!(
        dense.depth_map(dense.depth_maps[0].view).map(|m| m.view),
        Some(dense.depth_maps[0].view)
    );
}
