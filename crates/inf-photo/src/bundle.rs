//! **Bundle adjustment**: Levenberg–Marquardt over camera poses and 3D points,
//! reduced by the Schur complement, robustified by a Huber kernel.
//!
//! # The shape of the problem
//!
//! Every observation contributes a two-dimensional residual — the difference, in
//! **pixels**, between where a point projects and where it was seen. The
//! parameters are six numbers per camera (a rotation increment and a translation
//! increment) and three per point. A capture with 30 cameras and 20 000 points
//! has 60 060 parameters; the normal equations are 60 060 square, which is four
//! gigabytes if written down densely, and almost entirely zero.
//!
//! The saving structure is that **no point is coupled to another point**. The
//! normal matrix is
//!
//! ```text
//!     [ U   W ] [ dc ]   [ bc ]
//!     [ Wᵀ  V ] [ dp ] = [ bp ]
//! ```
//!
//! with `V` block-**diagonal** (one 3x3 per point). Eliminating `dp` gives the
//! *Schur complement*
//!
//! ```text
//!     (U - W V⁻¹ Wᵀ) dc = bc - W V⁻¹ bp
//! ```
//!
//! which is `6C` square — 180 for thirty cameras — and is solved densely by
//! `LDLᵀ`. Points come back by back-substitution, each from its own 3x3. That is
//! the whole reason a bundle adjuster is tractable, and it is what this module
//! implements.
//!
//! # Gauge
//!
//! The first [`BundleConfig::fixed_cameras`] cameras contribute residuals but
//! **no parameters**. With one camera fixed the only remaining freedom is
//! overall scale, which is an exact null direction of the cost (see
//! [`crate::camera::Pose::scaled`]); Levenberg damping keeps the reduced system
//! non-singular along it, and [`crate::sfm`] re-imposes the unit-baseline
//! convention afterwards. Fixing the camera rather than adding a prior means the
//! solved system has no artificial residual in it and the reported cost is the
//! reprojection cost and nothing else.
//!
//! # The robust kernel
//!
//! Huber, with the knee at [`BundleConfig::huber_delta_px`] **pixels**.
//! Quadratic below the knee, linear above it: an outlier that survived RANSAC
//! pulls with bounded force instead of dominating the solve. It is applied as
//! **IRLS** — the residual and its Jacobian are scaled by `rho'(s)` — which is
//! the standard first-order treatment. The second-order Triggs correction is not
//! applied; it changes the convergence *rate* near a large-residual minimum, not
//! the minimum. Written down so the omission is a decision rather than an
//! oversight.
//!
//! # Determinism
//!
//! Residuals and Jacobians are evaluated through
//! [`inf_core::job::JobPool::parallel_map_ref`] — an in-order pure map — and
//! every accumulation into `U`, `V`, `W` and the cost is then performed by a
//! **serial loop in observation order**. No floating-point sum in this module is
//! ever formed in parallel, which is why a bundle solved on one worker and on
//! eight is bit-identical rather than merely close.

use glam::{DVec2, DVec3};
use inf_core::job::JobPool;

use crate::camera::{Intrinsics, Pose};
use crate::linalg::{invert_sym3, mat3_from_rows, solve_ldlt};

/// One measurement: point `point`, seen by camera `camera`, at `pixel`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleObservation {
    /// Index into the camera array.
    pub camera: u32,
    /// Index into the point array.
    pub point: u32,
    /// Where it was seen, in **pixels**.
    pub pixel: DVec2,
}

/// How the solver is budgeted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleConfig {
    /// Outer Levenberg steps (one Jacobian evaluation each). Fixed — the solver
    /// never runs "until converged" on a wall clock.
    pub max_iterations: usize,
    /// Damping retries inside one outer step before giving up on it.
    pub max_damping_retries: usize,
    /// Starting Levenberg damping factor, relative to the diagonal.
    pub initial_lambda: f64,
    /// Huber knee, in **pixels**.
    pub huber_delta_px: f64,
    /// Stop when an accepted step improves the cost by less than this fraction.
    pub relative_tolerance: f64,
    /// How many leading cameras are held fixed to pin the gauge.
    pub fixed_cameras: usize,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            max_damping_retries: 8,
            initial_lambda: 1e-4,
            huber_delta_px: 2.0,
            relative_tolerance: 1e-10,
            fixed_cameras: 1,
        }
    }
}

/// What one bundle did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleReport {
    /// How many outer steps ran.
    pub iterations: usize,
    /// How many of them were accepted.
    pub accepted: usize,
    /// Robust cost before and after.
    pub initial_cost: f64,
    /// Robust cost after.
    pub final_cost: f64,
    /// Un-robustified reprojection RMS before, in **pixels**.
    pub initial_rms_px: f64,
    /// Un-robustified reprojection RMS after, in **pixels**.
    pub final_rms_px: f64,
    /// How many observations were in the problem.
    pub observations: usize,
}

impl BundleReport {
    /// A bundle that ran on nothing.
    pub fn empty() -> Self {
        Self {
            iterations: 0,
            accepted: 0,
            initial_cost: 0.0,
            final_cost: 0.0,
            initial_rms_px: 0.0,
            final_rms_px: 0.0,
            observations: 0,
        }
    }
}

/// The residual magnitude charged to an observation whose point is behind its
/// camera, in **pixels**. Large enough that any step which fixes it wins, finite
/// so the cost never becomes `inf` and stops comparing.
const BEHIND_CAMERA_PX: f64 = 1.0e4;

/// Per-observation evaluation: residual, Jacobians and robust weight.
#[derive(Clone, Copy)]
struct ObsEval {
    valid: bool,
    residual: [f64; 2],
    /// d(residual)/d(camera parameters), 2x6.
    jc: [[f64; 6]; 2],
    /// d(residual)/d(point), 2x3.
    jp: [[f64; 3]; 2],
    weight: f64,
}

fn evaluate(
    obs: &BundleObservation,
    cameras: &[Pose],
    intrinsics: &[Intrinsics],
    points: &[DVec3],
    huber: f64,
) -> ObsEval {
    let pose = &cameras[obs.camera as usize];
    let k = &intrinsics[obs.camera as usize];
    let x = points[obs.point as usize];
    let p = pose.transform(x);
    let Some((projected, jpix)) = k.project_with_jacobian(p) else {
        return ObsEval {
            valid: false,
            residual: [0.0; 2],
            jc: [[0.0; 6]; 2],
            jp: [[0.0; 3]; 2],
            weight: 0.0,
        };
    };
    let r = projected - obs.pixel;
    let norm = r.length();

    // d(p)/d(omega) = -[p]_x and d(p)/d(delta) = I — the perturbation model
    // `Pose::retract` implements.
    let cross = [[0.0, p.z, -p.y], [-p.z, 0.0, p.x], [p.y, -p.x, 0.0]];
    let rot = pose.rotation_matrix();
    let mut jc = [[0.0f64; 6]; 2];
    let mut jp = [[0.0f64; 3]; 2];
    for row in 0..2 {
        for c in 0..3 {
            let mut omega = 0.0;
            let mut point = 0.0;
            for k3 in 0..3 {
                omega += jpix[row][k3] * cross[k3][c];
                point += jpix[row][k3] * crate::linalg::mat3_at(&rot, k3, c);
            }
            jc[row][c] = omega;
            jc[row][3 + c] = jpix[row][c];
            jp[row][c] = point;
        }
    }

    let weight = if norm <= huber || !norm.is_finite() {
        1.0
    } else {
        huber / norm
    };
    ObsEval {
        valid: true,
        residual: [r.x, r.y],
        jc,
        jp,
        weight,
    }
}

/// The robust cost and the un-robustified RMS of a configuration, summed in
/// observation order.
fn cost_of(
    cameras: &[Pose],
    intrinsics: &[Intrinsics],
    points: &[DVec3],
    observations: &[BundleObservation],
    huber: f64,
    pool: &JobPool,
) -> (f64, f64) {
    let evals = pool.parallel_map_ref(observations, |o| {
        let pose = &cameras[o.camera as usize];
        let k = &intrinsics[o.camera as usize];
        let p = pose.transform(points[o.point as usize]);
        match k.project(p) {
            Some(projected) => {
                let n = (projected - o.pixel).length();
                let c = if n <= huber {
                    n * n
                } else {
                    2.0 * huber * n - huber * huber
                };
                (c, n * n)
            }
            None => (
                2.0 * huber * BEHIND_CAMERA_PX - huber * huber,
                BEHIND_CAMERA_PX * BEHIND_CAMERA_PX,
            ),
        }
    });
    let mut cost = 0.0;
    let mut squared = 0.0;
    for (c, s) in evals {
        cost += c;
        squared += s;
    }
    let rms = if observations.is_empty() {
        0.0
    } else {
        (squared / observations.len() as f64).sqrt()
    };
    (cost, rms)
}

/// Run bundle adjustment in place.
///
/// `intrinsics` is indexed by camera and is **not** refined — see the crate
/// docs. Cameras `0..cfg.fixed_cameras` are held fixed.
pub fn bundle_adjust(
    cameras: &mut [Pose],
    intrinsics: &[Intrinsics],
    points: &mut [DVec3],
    observations: &[BundleObservation],
    cfg: &BundleConfig,
    pool: &JobPool,
) -> BundleReport {
    let n_cams = cameras.len();
    let n_points = points.len();
    let fixed = cfg.fixed_cameras.min(n_cams);
    let free_cams = n_cams - fixed;
    if observations.is_empty() || n_points == 0 || free_cams == 0 {
        let (cost, rms) = cost_of(
            cameras,
            intrinsics,
            points,
            observations,
            cfg.huber_delta_px,
            pool,
        );
        return BundleReport {
            iterations: 0,
            accepted: 0,
            initial_cost: cost,
            final_cost: cost,
            initial_rms_px: rms,
            final_rms_px: rms,
            observations: observations.len(),
        };
    }

    // Which observations belong to which point, in observation order.
    let mut point_obs: Vec<Vec<u32>> = vec![Vec::new(); n_points];
    for (i, o) in observations.iter().enumerate() {
        point_obs[o.point as usize].push(i as u32);
    }

    let huber = cfg.huber_delta_px;
    let (initial_cost, initial_rms) =
        cost_of(cameras, intrinsics, points, observations, huber, pool);
    let mut cost = initial_cost;
    let mut lambda = cfg.initial_lambda;
    let mut accepted = 0usize;
    let mut iterations = 0usize;

    let dim = free_cams * 6;
    for _ in 0..cfg.max_iterations {
        iterations += 1;

        // ── Evaluate, in parallel; accumulate, serially. ────────────────────
        let evals = pool.parallel_map_ref(observations, |o| {
            evaluate(o, cameras, intrinsics, points, huber)
        });

        let mut u_blocks = vec![[0.0f64; 36]; free_cams];
        let mut v_blocks = vec![[0.0f64; 9]; n_points];
        let mut b_cam = vec![[0.0f64; 6]; free_cams];
        let mut b_pt = vec![[0.0f64; 3]; n_points];
        let mut w_blocks = vec![[0.0f64; 18]; observations.len()];

        for (i, (o, e)) in observations.iter().zip(&evals).enumerate() {
            if !e.valid {
                continue;
            }
            let cam = o.camera as usize;
            let pt = o.point as usize;
            let w = e.weight;
            let free = cam.checked_sub(fixed);
            for row in 0..2 {
                let r = e.residual[row];
                if let Some(fc) = free {
                    for c in 0..6 {
                        b_cam[fc][c] -= w * e.jc[row][c] * r;
                        for d in 0..6 {
                            u_blocks[fc][c * 6 + d] += w * e.jc[row][c] * e.jc[row][d];
                        }
                    }
                }
                for c in 0..3 {
                    b_pt[pt][c] -= w * e.jp[row][c] * r;
                    for d in 0..3 {
                        v_blocks[pt][c * 3 + d] += w * e.jp[row][c] * e.jp[row][d];
                    }
                }
                if free.is_some() {
                    for c in 0..6 {
                        for d in 0..3 {
                            w_blocks[i][c * 3 + d] += w * e.jc[row][c] * e.jp[row][d];
                        }
                    }
                }
            }
        }

        // ── Damping retries: the Jacobian is reused, only lambda changes. ───
        let mut step_taken = false;
        for _ in 0..cfg.max_damping_retries {
            let Some((delta_cam, delta_pt)) = solve_schur(
                &u_blocks,
                &v_blocks,
                &w_blocks,
                &b_cam,
                &b_pt,
                &point_obs,
                observations,
                fixed,
                lambda,
                dim,
            ) else {
                lambda *= 10.0;
                if lambda > 1e12 {
                    break;
                }
                continue;
            };

            let mut cand_cams = cameras.to_vec();
            for (k, pose) in cand_cams.iter_mut().enumerate().skip(fixed) {
                let base = (k - fixed) * 6;
                *pose = pose.retract(
                    DVec3::new(delta_cam[base], delta_cam[base + 1], delta_cam[base + 2]),
                    DVec3::new(
                        delta_cam[base + 3],
                        delta_cam[base + 4],
                        delta_cam[base + 5],
                    ),
                );
            }
            let mut cand_points = points.to_vec();
            for (p, x) in cand_points.iter_mut().enumerate() {
                *x += DVec3::new(delta_pt[p * 3], delta_pt[p * 3 + 1], delta_pt[p * 3 + 2]);
            }

            let (cand_cost, _) = cost_of(
                &cand_cams,
                intrinsics,
                &cand_points,
                observations,
                huber,
                pool,
            );
            if cand_cost.is_finite() && cand_cost < cost {
                let improvement = (cost - cand_cost) / cost.max(f64::MIN_POSITIVE);
                cameras.copy_from_slice(&cand_cams);
                points.copy_from_slice(&cand_points);
                cost = cand_cost;
                accepted += 1;
                step_taken = true;
                lambda = (lambda * 0.1).max(1e-12);
                if improvement < cfg.relative_tolerance {
                    step_taken = false; // converged; fall out of the outer loop
                }
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !step_taken {
            break;
        }
    }

    let (final_cost, final_rms) = cost_of(cameras, intrinsics, points, observations, huber, pool);
    BundleReport {
        iterations,
        accepted,
        initial_cost,
        final_cost,
        initial_rms_px: initial_rms,
        final_rms_px: final_rms,
        observations: observations.len(),
    }
}

/// Build and solve the Schur-reduced system, returning `(delta_cameras,
/// delta_points)`.
///
/// A point whose 3x3 block is singular even after damping — one seen from a
/// single camera, or from a line — is **frozen**: it contributes nothing to the
/// reduction and receives a zero step, rather than a division by a whisker.
#[allow(clippy::too_many_arguments)]
fn solve_schur(
    u_blocks: &[[f64; 36]],
    v_blocks: &[[f64; 9]],
    w_blocks: &[[f64; 18]],
    b_cam: &[[f64; 6]],
    b_pt: &[[f64; 3]],
    point_obs: &[Vec<u32>],
    observations: &[BundleObservation],
    fixed: usize,
    lambda: f64,
    dim: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let free_cams = u_blocks.len();
    let n_points = v_blocks.len();

    // S starts as the damped U, block-diagonal.
    let mut s = vec![0.0f64; dim * dim];
    for (fc, block) in u_blocks.iter().enumerate() {
        for c in 0..6 {
            for d in 0..6 {
                let mut v = block[c * 6 + d];
                if c == d {
                    v += lambda * v.abs().max(1e-9);
                }
                s[(fc * 6 + c) * dim + (fc * 6 + d)] = v;
            }
        }
    }
    let mut rhs = vec![0.0f64; dim];
    for (fc, b) in b_cam.iter().enumerate() {
        for c in 0..6 {
            rhs[fc * 6 + c] = b[c];
        }
    }

    // Reduce point by point, in point order.
    let mut v_inverses: Vec<Option<[f64; 9]>> = vec![None; n_points];
    for p in 0..n_points {
        let block = v_blocks[p];
        let mut damped = block;
        for c in 0..3 {
            damped[c * 3 + c] += lambda * block[c * 3 + c].abs().max(1e-9);
        }
        let m = mat3_from_rows(
            [damped[0], damped[1], damped[2]],
            [damped[3], damped[4], damped[5]],
            [damped[6], damped[7], damped[8]],
        );
        let Some(inv) = invert_sym3(&m) else {
            continue;
        };
        let inv_flat = [
            crate::linalg::mat3_at(&inv, 0, 0),
            crate::linalg::mat3_at(&inv, 0, 1),
            crate::linalg::mat3_at(&inv, 0, 2),
            crate::linalg::mat3_at(&inv, 1, 0),
            crate::linalg::mat3_at(&inv, 1, 1),
            crate::linalg::mat3_at(&inv, 1, 2),
            crate::linalg::mat3_at(&inv, 2, 0),
            crate::linalg::mat3_at(&inv, 2, 1),
            crate::linalg::mat3_at(&inv, 2, 2),
        ];
        v_inverses[p] = Some(inv_flat);

        // Y_i = W_i V⁻¹  (6x3), for every observation of this point on a free
        // camera. Then S -= Y_i W_jᵀ and rhs -= Y_i b_p.
        let free_obs: Vec<(usize, usize)> = point_obs[p]
            .iter()
            .filter_map(|&i| {
                let cam = observations[i as usize].camera as usize;
                cam.checked_sub(fixed).map(|fc| (i as usize, fc))
            })
            .collect();
        let mut y: Vec<[f64; 18]> = Vec::with_capacity(free_obs.len());
        for &(i, _) in &free_obs {
            let w = &w_blocks[i];
            let mut yi = [0.0f64; 18];
            for c in 0..6 {
                for d in 0..3 {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += w[c * 3 + k] * inv_flat[k * 3 + d];
                    }
                    yi[c * 3 + d] = acc;
                }
            }
            y.push(yi);
        }
        for (a, &(_, fc_a)) in free_obs.iter().enumerate() {
            for c in 0..6 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += y[a][c * 3 + k] * b_pt[p][k];
                }
                rhs[fc_a * 6 + c] -= acc;
            }
            for (b, &(ib, fc_b)) in free_obs.iter().enumerate() {
                let _ = b;
                let wb = &w_blocks[ib];
                for c in 0..6 {
                    for d in 0..6 {
                        let mut acc = 0.0;
                        for k in 0..3 {
                            acc += y[a][c * 3 + k] * wb[d * 3 + k];
                        }
                        s[(fc_a * 6 + c) * dim + (fc_b * 6 + d)] -= acc;
                    }
                }
            }
        }
    }

    let delta_cam = solve_ldlt(&mut s, &rhs, dim)?;
    if !delta_cam.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Back-substitute: dp = V⁻¹ (b_p - Σ Wᵀ dc).
    let mut delta_pt = vec![0.0f64; n_points * 3];
    for p in 0..n_points {
        let Some(inv) = v_inverses[p] else {
            continue;
        };
        let mut reduced = b_pt[p];
        for &i in &point_obs[p] {
            let cam = observations[i as usize].camera as usize;
            let Some(fc) = cam.checked_sub(fixed) else {
                continue;
            };
            let w = &w_blocks[i as usize];
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..6 {
                    acc += w[c * 3 + d] * delta_cam[fc * 6 + c];
                }
                reduced[d] -= acc;
            }
        }
        for d in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += inv[d * 3 + k] * reduced[k];
            }
            delta_pt[p * 3 + d] = acc;
        }
    }
    if !delta_pt.iter().all(|v| v.is_finite()) {
        return None;
    }
    let _ = free_cams;
    Some((delta_cam, delta_pt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Hash64;

    fn scene() -> (
        Vec<Pose>,
        Vec<Intrinsics>,
        Vec<DVec3>,
        Vec<BundleObservation>,
    ) {
        // An asymmetric point cloud and four cameras at unequal spacings — a
        // symmetric ring would hide a sign error in the update.
        let points: Vec<DVec3> = (0..40)
            .map(|i| {
                let h = Hash64::new(0xbeef).mix_usize(i);
                DVec3::new(
                    -1.6 + 3.1 * h.mix_u64(1).unit(),
                    -0.9 + 2.0 * h.mix_u64(2).unit(),
                    5.0 + 3.0 * h.mix_u64(3).unit(),
                )
            })
            .collect();
        let centres = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.15, -0.2),
            DVec3::new(2.3, -0.25, 0.35),
            DVec3::new(-0.9, 0.4, 0.15),
        ];
        let cameras: Vec<Pose> = centres
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                Pose::look_at(
                    c,
                    DVec3::new(0.1 * i as f64, 0.0, 6.0),
                    DVec3::new(0.02 * i as f64, -1.0, 0.01),
                )
            })
            .collect();
        let k = Intrinsics::centred(640, 480, 500.0).with_radial(-0.05, 0.01);
        let intrinsics = vec![k; cameras.len()];
        let mut observations = Vec::new();
        for (ci, cam) in cameras.iter().enumerate() {
            for (pi, x) in points.iter().enumerate() {
                if let Some(px) = k.project(cam.transform(*x)) {
                    observations.push(BundleObservation {
                        camera: ci as u32,
                        point: pi as u32,
                        pixel: px,
                    });
                }
            }
        }
        (cameras, intrinsics, points, observations)
    }

    #[test]
    fn a_perfect_bundle_is_already_at_its_minimum() {
        let (mut cams, intr, mut pts, obs) = scene();
        let pool = JobPool::new(2);
        let report = bundle_adjust(
            &mut cams,
            &intr,
            &mut pts,
            &obs,
            &BundleConfig::default(),
            &pool,
        );
        assert!(
            report.initial_rms_px < 1e-9,
            "the fixture is not noise-free: {} px",
            report.initial_rms_px
        );
        assert!(report.final_rms_px <= report.initial_rms_px + 1e-12);
    }

    #[test]
    fn a_perturbed_bundle_converges_back() {
        let (truth_cams, intr, truth_pts, obs) = scene();
        let mut cams = truth_cams.clone();
        let mut pts = truth_pts.clone();
        // Perturb everything but the fixed first camera.
        for (i, c) in cams.iter_mut().enumerate().skip(1) {
            let h = Hash64::new(0x9111).mix_usize(i);
            *c = c.retract(
                DVec3::new(
                    0.02 * (h.mix_u64(1).unit() - 0.5),
                    0.02 * (h.mix_u64(2).unit() - 0.5),
                    0.02 * (h.mix_u64(3).unit() - 0.5),
                ),
                DVec3::new(
                    0.05 * (h.mix_u64(4).unit() - 0.5),
                    0.05 * (h.mix_u64(5).unit() - 0.5),
                    0.05 * (h.mix_u64(6).unit() - 0.5),
                ),
            );
        }
        for (i, p) in pts.iter_mut().enumerate() {
            let h = Hash64::new(0x7222).mix_usize(i);
            *p += DVec3::new(
                0.08 * (h.mix_u64(1).unit() - 0.5),
                0.08 * (h.mix_u64(2).unit() - 0.5),
                0.12 * (h.mix_u64(3).unit() - 0.5),
            );
        }
        let pool = JobPool::new(2);
        let report = bundle_adjust(
            &mut cams,
            &intr,
            &mut pts,
            &obs,
            &BundleConfig::default(),
            &pool,
        );
        assert!(
            report.initial_rms_px > 1.0,
            "the perturbation was too small to test anything: {} px",
            report.initial_rms_px
        );
        assert!(
            report.final_rms_px < 0.02,
            "bundle did not converge: {} -> {} px ({} accepted of {} iterations)",
            report.initial_rms_px,
            report.final_rms_px,
            report.accepted,
            report.iterations
        );
        assert!(report.final_cost < report.initial_cost);
        // The recovered geometry, not just the cost: camera 0 is fixed and the
        // rest should be back where they started (the gauge is pinned by the
        // fixed camera plus the fact that the points carry the scale).
        for (i, (got, want)) in cams.iter().zip(&truth_cams).enumerate() {
            assert!(
                (got.center() - want.center()).length() < 0.02,
                "camera {i} centre {} vs {}",
                got.center(),
                want.center()
            );
        }
    }

    #[test]
    fn the_fixed_camera_never_moves() {
        let (mut cams, intr, mut pts, obs) = scene();
        let pinned = cams[0];
        for p in pts.iter_mut() {
            *p += DVec3::new(0.05, -0.04, 0.06);
        }
        let pool = JobPool::new(1);
        bundle_adjust(
            &mut cams,
            &intr,
            &mut pts,
            &obs,
            &BundleConfig::default(),
            &pool,
        );
        assert_eq!(
            cams[0].translation.to_array().map(f64::to_bits),
            pinned.translation.to_array().map(f64::to_bits),
            "the gauge camera moved"
        );
        assert_eq!(
            cams[0].rotation.to_array().map(f64::to_bits),
            pinned.rotation.to_array().map(f64::to_bits)
        );
    }

    #[test]
    fn the_bundle_is_pool_size_invariant() {
        let (base_cams, intr, base_pts, obs) = scene();
        let run = |threads: usize| {
            let mut cams = base_cams.clone();
            let mut pts = base_pts.clone();
            for (i, p) in pts.iter_mut().enumerate() {
                let h = Hash64::new(0x1234).mix_usize(i);
                *p += DVec3::new(
                    0.07 * (h.mix_u64(1).unit() - 0.5),
                    0.07 * (h.mix_u64(2).unit() - 0.5),
                    0.09 * (h.mix_u64(3).unit() - 0.5),
                );
            }
            for c in cams.iter_mut().skip(1) {
                *c = c.retract(
                    DVec3::new(0.01, -0.008, 0.006),
                    DVec3::new(0.02, 0.01, -0.015),
                );
            }
            let pool = JobPool::new(threads);
            let report = bundle_adjust(
                &mut cams,
                &intr,
                &mut pts,
                &obs,
                &BundleConfig::default(),
                &pool,
            );
            let mut bytes = Vec::new();
            for c in &cams {
                c.write_canonical(&mut bytes);
            }
            for p in &pts {
                for v in p.to_array() {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
            (bytes, report)
        };
        let (b1, r1) = run(1);
        let (b2, _) = run(2);
        let (b8, _) = run(8);
        assert!(r1.accepted > 0, "the fixture did not exercise the solver");
        assert_eq!(b1, b2, "bundle output differed between 1 and 2 workers");
        assert_eq!(b1, b8, "bundle output differed between 1 and 8 workers");
    }

    #[test]
    fn a_huber_kernel_bounds_one_bad_observation() {
        let (truth_cams, intr, truth_pts, obs) = scene();
        // Move one observation 200 px away — an outlier that survived RANSAC.
        let mut poisoned = obs.clone();
        poisoned[7].pixel += DVec2::new(200.0, -150.0);

        let solve = |huber: f64, observations: &[BundleObservation]| {
            let mut cams = truth_cams.clone();
            let mut pts = truth_pts.clone();
            for c in cams.iter_mut().skip(1) {
                *c = c.retract(DVec3::new(0.01, 0.0, 0.0), DVec3::new(0.01, 0.0, 0.0));
            }
            let cfg = BundleConfig {
                huber_delta_px: huber,
                ..BundleConfig::default()
            };
            bundle_adjust(
                &mut cams,
                &intr,
                &mut pts,
                observations,
                &cfg,
                &JobPool::new(1),
            );
            cams
        };

        let robust = solve(2.0, &poisoned);
        // A knee of 10^9 px is a plain least-squares solve: the outlier pulls.
        let naive = solve(1e9, &poisoned);
        let robust_err: f64 = robust
            .iter()
            .zip(&truth_cams)
            .map(|(a, b)| (a.center() - b.center()).length())
            .sum();
        let naive_err: f64 = naive
            .iter()
            .zip(&truth_cams)
            .map(|(a, b)| (a.center() - b.center()).length())
            .sum();
        assert!(
            robust_err < naive_err,
            "the Huber kernel did not help: robust {robust_err} vs least-squares {naive_err}"
        );
    }

    #[test]
    fn an_empty_problem_reports_rather_than_panicking() {
        let mut cams: Vec<Pose> = vec![Pose::IDENTITY];
        let intr = vec![Intrinsics::centred(64, 48, 50.0)];
        let mut pts: Vec<DVec3> = Vec::new();
        let report = bundle_adjust(
            &mut cams,
            &intr,
            &mut pts,
            &[],
            &BundleConfig::default(),
            &JobPool::new(1),
        );
        assert_eq!(report.observations, 0);
        assert_eq!(report.iterations, 0);
    }
}
