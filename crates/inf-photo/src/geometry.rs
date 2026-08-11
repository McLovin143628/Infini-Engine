//! Multi-view geometry: the essential matrix, relative pose, triangulation and
//! pose-from-points, each wrapped in a **seeded, fixed-iteration** RANSAC.
//!
//! # The estimators
//!
//! * **Essential matrix, normalized eight-point.** Correspondences are first
//!   Hartley-normalized (centroid to the origin, mean radius to `sqrt(2)`),
//!   because the raw system's condition number is what makes the naive
//!   eight-point algorithm's bad reputation. The linear solution is then
//!   projected onto the essential manifold — singular values forced to
//!   `(1, 1, 0)` — because a linear solve does not know it is estimating an
//!   essential matrix.
//! * **Relative pose.** Four candidates come out of the SVD; exactly one puts
//!   the points in front of both cameras, and that is the one chosen. Ties (a
//!   degenerate configuration where two candidates score equally) go to the
//!   lower candidate index, so the choice is total.
//! * **Triangulation, multi-view DLT.** Two rows per observation, null vector of
//!   the normal matrix. Accepted only if the point is in front of every camera
//!   that sees it, its reprojection is inside tolerance, and the **parallax**
//!   between its two most separated rays is real — a point triangulated from
//!   near-parallel rays has a depth with no information in it and poisons a
//!   bundle.
//! * **Pose from points, DLT-PnP.** The 3x4 projection matrix as the null
//!   vector of a 12-column system, decomposed into a rotation and a scale, then
//!   refined nonlinearly. Six correspondences minimum.
//!
//! # Why RANSAC here is a *fixed* loop
//!
//! Textbook RANSAC recomputes its own iteration budget from the running inlier
//! ratio and stops early. That makes the number of hypotheses a function of the
//! order the data happened to be sampled in, which makes the answer a function
//! of that order too. Here the budget is a **constant** ([`RansacConfig`]), every
//! hypothesis is drawn from `(seed, stage, iteration)` through
//! [`crate::hash::Hash64`], all hypotheses are scored, and the best is chosen by
//! a total order: **more inliers wins; equal inliers, lower total residual wins;
//! equal residual, lower iteration index wins**. Nothing about that depends on
//! evaluation order, so the loop can be — and is — fanned out over the job pool.
//!
//! # Units
//!
//! Every point in this module is a **normalized** image coordinate
//! (dimensionless, `x/z` of a camera-space ray), not a pixel. Thresholds are
//! given in **pixels** and divided by the mean focal length at the boundary,
//! which is the only place the two meet.

use glam::{DMat3, DVec2, DVec3};
use inf_core::job::JobPool;

use crate::camera::{projection_rows, Pose};
use crate::hash::{distinct_indices, Hash64};
use crate::linalg::{
    accumulate_normal, mat3_at, mat3_from_rows, nearest_rotation, normalize_or,
    project_to_essential, smallest_eigenvector, solve_ldlt, svd3,
};

/// How the robust estimators are budgeted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacConfig {
    /// Hypotheses drawn for the essential matrix. Fixed, never adaptive.
    pub essential_iterations: usize,
    /// Inlier threshold for the essential matrix, in **pixels**.
    pub essential_threshold_px: f64,
    /// Hypotheses drawn for PnP. Fixed.
    pub pnp_iterations: usize,
    /// Inlier threshold for PnP reprojection, in **pixels**.
    pub pnp_threshold_px: f64,
    /// A pair with fewer essential inliers than this cannot initialise.
    pub min_essential_inliers: usize,
    /// A view with fewer PnP inliers than this cannot register.
    pub min_pnp_inliers: usize,
    /// Gauss–Newton/LM steps in the nonlinear pose refinement. Fixed.
    pub refine_iterations: usize,
    /// Huber threshold for the pose refinement, in **pixels**.
    pub refine_huber_px: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            essential_iterations: 384,
            essential_threshold_px: 1.5,
            pnp_iterations: 192,
            pnp_threshold_px: 3.0,
            min_essential_inliers: 30,
            min_pnp_inliers: 12,
            refine_iterations: 20,
            refine_huber_px: 2.0,
        }
    }
}

/// A Hartley normalisation: `x_normalized = scale * (x - centre)`, carried as
/// the 3x3 it can be folded back through.
#[derive(Clone, Copy, Debug)]
struct Whitening {
    scale: f64,
    centre: DVec2,
}

impl Whitening {
    /// Centroid to the origin, mean distance to `sqrt(2)` — the conditioning
    /// that turns the eight-point algorithm from a curiosity into an estimator.
    fn of(points: &[DVec2]) -> Self {
        if points.is_empty() {
            return Self {
                scale: 1.0,
                centre: DVec2::ZERO,
            };
        }
        let mut centre = DVec2::ZERO;
        for p in points {
            centre += *p;
        }
        centre /= points.len() as f64;
        let mut mean_r = 0.0;
        for p in points {
            mean_r += (*p - centre).length();
        }
        mean_r /= points.len() as f64;
        let scale = if mean_r > 1e-12 {
            std::f64::consts::SQRT_2 / mean_r
        } else {
            1.0
        };
        Self { scale, centre }
    }

    #[inline]
    fn apply(&self, p: DVec2) -> DVec2 {
        (p - self.centre) * self.scale
    }

    /// The 3x3 form, so `E_original = T_bᵀ E_whitened T_a`.
    fn matrix(&self) -> DMat3 {
        mat3_from_rows(
            [self.scale, 0.0, -self.scale * self.centre.x],
            [0.0, self.scale, -self.scale * self.centre.y],
            [0.0, 0.0, 1.0],
        )
    }
}

/// The normalized eight-point essential matrix over the given correspondence
/// indices. `a` and `b` are normalized image coordinates of the two views.
pub fn essential_eight_point(a: &[DVec2], b: &[DVec2], indices: &[u32]) -> Option<DMat3> {
    if indices.len() < 8 {
        return None;
    }
    let sample_a: Vec<DVec2> = indices.iter().map(|&i| a[i as usize]).collect();
    let sample_b: Vec<DVec2> = indices.iter().map(|&i| b[i as usize]).collect();
    let ta = Whitening::of(&sample_a);
    let tb = Whitening::of(&sample_b);

    let mut ata = vec![0.0f64; 81];
    for (pa, pb) in sample_a.iter().zip(&sample_b) {
        let (x1, y1) = {
            let p = ta.apply(*pa);
            (p.x, p.y)
        };
        let (x2, y2) = {
            let p = tb.apply(*pb);
            (p.x, p.y)
        };
        // The row of `x2ᵀ E x1 = 0` in row-major vec(E).
        let row = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
        accumulate_normal(&mut ata, &row);
    }
    let e = smallest_eigenvector(&ata, 9);
    let whitened = mat3_from_rows([e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]]);
    let denormalized = tb
        .matrix()
        .transpose()
        .mul_mat3(&whitened)
        .mul_mat3(&ta.matrix());
    let projected = project_to_essential(&denormalized);
    if (0..3).all(|c| projected.col(c).is_finite()) {
        Some(projected)
    } else {
        None
    }
}

/// The first-order (Sampson) approximation to the geometric distance of a
/// correspondence from an epipolar model, in **normalized** units.
///
/// The algebraic residual `x2ᵀ E x1` on its own is meaningless as a threshold —
/// it scales with where in the image the point sits. Sampson divides it by the
/// gradient, which is what makes one threshold apply across the frame.
#[inline]
pub fn sampson_distance(e: &DMat3, a: DVec2, b: DVec2) -> f64 {
    let x1 = DVec3::new(a.x, a.y, 1.0);
    let x2 = DVec3::new(b.x, b.y, 1.0);
    let ex1 = e.mul_vec3(x1);
    let etx2 = e.transpose().mul_vec3(x2);
    let numerator = x2.dot(ex1);
    let denom = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    (numerator * numerator / denom).sqrt()
}

/// A robustly estimated essential matrix and the correspondences that support it.
#[derive(Clone, Debug)]
pub struct EssentialModel {
    /// The matrix, refit on its own inliers.
    pub essential: DMat3,
    /// Indices into the correspondence arrays, ascending.
    pub inliers: Vec<u32>,
    /// Sum of inlier Sampson distances (normalized units) — the tie-break.
    pub residual: f64,
}

/// Seeded, fixed-iteration RANSAC for the essential matrix.
///
/// `seed` should encode the pair's identity so two different pairs do not draw
/// the same hypotheses; [`crate::sfm`] folds the view indices in.
pub fn ransac_essential(
    a: &[DVec2],
    b: &[DVec2],
    threshold: f64,
    cfg: &RansacConfig,
    seed: u64,
    pool: &JobPool,
) -> Option<EssentialModel> {
    assert_eq!(a.len(), b.len(), "correspondence arrays must line up");
    let n = a.len();
    if n < 8 {
        return None;
    }

    let iterations: Vec<usize> = (0..cfg.essential_iterations).collect();
    let scored = pool.parallel_map(iterations, |iter| {
        let mut scratch = Vec::new();
        let base = Hash64::new(seed).mix_usize(iter);
        let sample = match distinct_indices(base, n, 8, &mut scratch) {
            Some(s) => s,
            None => return (0usize, f64::INFINITY),
        };
        let Some(e) = essential_eight_point(a, b, &sample) else {
            return (0usize, f64::INFINITY);
        };
        let mut count = 0usize;
        let mut residual = 0.0;
        for i in 0..n {
            let d = sampson_distance(&e, a[i], b[i]);
            if d <= threshold {
                count += 1;
                residual += d;
            }
        }
        (count, residual)
    });

    // Serial argmax over a total order, so the winner cannot depend on which
    // worker finished first.
    let mut best_iter = usize::MAX;
    let mut best_count = 0usize;
    let mut best_residual = f64::INFINITY;
    for (iter, &(count, residual)) in scored.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let better = count > best_count
            || (count == best_count && residual.total_cmp(&best_residual).is_lt());
        if better {
            best_iter = iter;
            best_count = count;
            best_residual = residual;
        }
    }
    if best_iter == usize::MAX {
        return None;
    }

    // Recover the winning hypothesis, then refit on all of its inliers — the
    // step that turns a minimal sample into an estimate.
    let mut scratch = Vec::new();
    let sample = distinct_indices(Hash64::new(seed).mix_usize(best_iter), n, 8, &mut scratch)?;
    let coarse = essential_eight_point(a, b, &sample)?;
    let mut inliers: Vec<u32> = (0..n as u32)
        .filter(|&i| sampson_distance(&coarse, a[i as usize], b[i as usize]) <= threshold)
        .collect();

    let mut essential = coarse;
    if inliers.len() >= 8 {
        if let Some(refit) = essential_eight_point(a, b, &inliers) {
            let refined: Vec<u32> = (0..n as u32)
                .filter(|&i| sampson_distance(&refit, a[i as usize], b[i as usize]) <= threshold)
                .collect();
            // Only adopt the refit if it did not lose support. A refit that
            // shrinks the inlier set has been dragged by an outlier.
            if refined.len() >= inliers.len() {
                essential = refit;
                inliers = refined;
            }
        }
    }
    let residual = inliers
        .iter()
        .map(|&i| sampson_distance(&essential, a[i as usize], b[i as usize]))
        .sum();
    Some(EssentialModel {
        essential,
        inliers,
        residual,
    })
}

/// The four relative poses an essential matrix admits, in a fixed order.
///
/// `E = [t]_× R` with the second camera at `(R, t)` and the first at the
/// identity. The decomposition gives two rotations and two translation signs;
/// three of the four combinations put points behind a camera.
pub fn pose_candidates(e: &DMat3) -> [Pose; 4] {
    let (u, _, v) = svd3(e);
    // Force both factors right-handed, or the "rotations" below are reflections.
    let u = if u.determinant() < 0.0 {
        DMat3::from_cols(u.col(0), u.col(1), -u.col(2))
    } else {
        u
    };
    let v = if v.determinant() < 0.0 {
        DMat3::from_cols(v.col(0), v.col(1), -v.col(2))
    } else {
        v
    };
    let w = mat3_from_rows([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let r1 = u.mul_mat3(&w).mul_mat3(&v.transpose());
    let r2 = u.mul_mat3(&w.transpose()).mul_mat3(&v.transpose());
    let t = normalize_or(u.col(2), DVec3::Z);
    [
        Pose::from_rt(r1, t),
        Pose::from_rt(r1, -t),
        Pose::from_rt(r2, t),
        Pose::from_rt(r2, -t),
    ]
}

/// Pick the relative pose that puts the most correspondences in front of both
/// cameras, with the identity as the first camera.
///
/// Returns the pose and how many correspondences it put in front. Ties go to the
/// lower candidate index.
pub fn pose_from_essential(
    e: &DMat3,
    a: &[DVec2],
    b: &[DVec2],
    inliers: &[u32],
) -> Option<(Pose, usize)> {
    let first = Pose::IDENTITY;
    let mut best: Option<(Pose, usize)> = None;
    for candidate in pose_candidates(e) {
        let mut count = 0usize;
        for &i in inliers {
            let obs = [(first, a[i as usize]), (candidate, b[i as usize])];
            if let Some(x) = triangulate_dlt(&obs) {
                if first.transform(x).z > 0.0 && candidate.transform(x).z > 0.0 {
                    count += 1;
                }
            }
        }
        if best.as_ref().map(|(_, c)| count > *c).unwrap_or(true) {
            best = Some((candidate, count));
        }
    }
    best.filter(|(_, count)| *count > 0)
}

/// Multi-view DLT triangulation. `observations` pairs a pose with the
/// **normalized** image coordinate that pose saw.
pub fn triangulate_dlt(observations: &[(Pose, DVec2)]) -> Option<DVec3> {
    if observations.len() < 2 {
        return None;
    }
    let mut ata = vec![0.0f64; 16];
    for (pose, obs) in observations {
        let p = projection_rows(pose);
        let row_u = [
            obs.x * p[2][0] - p[0][0],
            obs.x * p[2][1] - p[0][1],
            obs.x * p[2][2] - p[0][2],
            obs.x * p[2][3] - p[0][3],
        ];
        let row_v = [
            obs.y * p[2][0] - p[1][0],
            obs.y * p[2][1] - p[1][1],
            obs.y * p[2][2] - p[1][2],
            obs.y * p[2][3] - p[1][3],
        ];
        accumulate_normal(&mut ata, &row_u);
        accumulate_normal(&mut ata, &row_v);
    }
    let x = smallest_eigenvector(&ata, 4);
    if x[3].abs() < 1e-12 {
        // A point at infinity: two parallel rays. There is no position here.
        return None;
    }
    let point = DVec3::new(x[0] / x[3], x[1] / x[3], x[2] / x[3]);
    if point.is_finite() {
        Some(point)
    } else {
        None
    }
}

/// The **cosine of the widest angle** between any two rays that see a point.
///
/// Returned rather than the angle because a cosine needs no `acos`, and the test
/// downstream is always "is this below `cos(min_angle)`". `1.0` means every ray
/// is parallel — no parallax, no depth.
pub fn parallax_cos(point: DVec3, poses: &[Pose]) -> f64 {
    let mut widest = 1.0f64;
    let dirs: Vec<DVec3> = poses
        .iter()
        .map(|p| normalize_or(point - p.center(), DVec3::Z))
        .collect();
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            widest = widest.min(dirs[i].dot(dirs[j]));
        }
    }
    widest
}

/// The linear DLT solution to pose-from-points: `world` points against
/// **normalized** image coordinates. Six correspondences minimum.
///
/// Degenerate on a planar point set — the 12-column system loses rank — which is
/// the documented bound in the crate docs. RANSAC will simply fail to find
/// support rather than returning nonsense, because every hypothesis is scored by
/// reprojection.
pub fn pnp_dlt(world: &[DVec3], obs: &[DVec2], indices: &[u32]) -> Option<Pose> {
    if indices.len() < 6 {
        return None;
    }
    // Condition the world points the same way image points are conditioned:
    // an object a hundred units from the origin has a normal matrix whose
    // entries span ten orders of magnitude otherwise.
    let mut centre = DVec3::ZERO;
    for &i in indices {
        centre += world[i as usize];
    }
    centre /= indices.len() as f64;
    let mut mean_r = 0.0;
    for &i in indices {
        mean_r += (world[i as usize] - centre).length();
    }
    mean_r /= indices.len() as f64;
    let ws = if mean_r > 1e-12 {
        3.0f64.sqrt() / mean_r
    } else {
        1.0
    };

    let mut ata = vec![0.0f64; 144];
    for &i in indices {
        let p = (world[i as usize] - centre) * ws;
        let o = obs[i as usize];
        let mut row = [0.0f64; 12];
        row[0] = p.x;
        row[1] = p.y;
        row[2] = p.z;
        row[3] = 1.0;
        row[8] = -o.x * p.x;
        row[9] = -o.x * p.y;
        row[10] = -o.x * p.z;
        row[11] = -o.x;
        accumulate_normal(&mut ata, &row);
        let mut row = [0.0f64; 12];
        row[4] = p.x;
        row[5] = p.y;
        row[6] = p.z;
        row[7] = 1.0;
        row[8] = -o.y * p.x;
        row[9] = -o.y * p.y;
        row[10] = -o.y * p.z;
        row[11] = -o.y;
        accumulate_normal(&mut ata, &row);
    }
    let v = smallest_eigenvector(&ata, 12);
    let m = mat3_from_rows([v[0], v[1], v[2]], [v[4], v[5], v[6]], [v[8], v[9], v[10]]);
    let t_raw = DVec3::new(v[3], v[7], v[11]);

    // `m` is a rotation times a scale, possibly negated. `det(m) = s³`, so the
    // *sign* comes from the determinant and the *magnitude* from the mean
    // singular value — which avoids a cube root, and with it a libm call.
    let (_, sv, _) = svd3(&m);
    let magnitude = (sv.x + sv.y + sv.z) / 3.0;
    if !magnitude.is_finite() || magnitude < 1e-12 {
        return None;
    }
    let s = if m.determinant() < 0.0 {
        -magnitude
    } else {
        magnitude
    };
    let rotation = nearest_rotation(&(m * (1.0 / s)));
    let t_conditioned = t_raw / s;

    // Undo the world conditioning. The solved pose maps the *conditioned* point
    // `p = ws (X - centre)`:
    //
    //     proj ∝ R p + t_c = ws R X - ws R centre + t_c
    //
    // and a projection is homogeneous, so dividing through by `ws` leaves it
    // untouched:
    //
    //     proj ∝ R X + (t_c / ws - R centre)
    //
    // which is the unconditioned pose.
    let translation = t_conditioned / ws - rotation.mul_vec3(centre);
    if !translation.is_finite() {
        return None;
    }
    let pose = Pose::from_rt(rotation, translation);
    if pose.rotation.is_finite() {
        Some(pose)
    } else {
        None
    }
}

/// A robustly estimated pose and the correspondences that support it.
#[derive(Clone, Debug)]
pub struct PnpModel {
    /// The pose, refined nonlinearly on its inliers.
    pub pose: Pose,
    /// Indices into the correspondence arrays, ascending.
    pub inliers: Vec<u32>,
    /// Sum of inlier reprojection errors, in **normalized** units.
    pub residual: f64,
}

/// Seeded, fixed-iteration RANSAC for pose-from-points, followed by a nonlinear
/// refinement on the inlier set.
pub fn ransac_pnp(
    world: &[DVec3],
    obs: &[DVec2],
    threshold: f64,
    cfg: &RansacConfig,
    seed: u64,
    pool: &JobPool,
) -> Option<PnpModel> {
    assert_eq!(world.len(), obs.len(), "correspondence arrays must line up");
    let n = world.len();
    if n < 6 {
        return None;
    }

    let iterations: Vec<usize> = (0..cfg.pnp_iterations).collect();
    let scored = pool.parallel_map(iterations, |iter| {
        let mut scratch = Vec::new();
        let base = Hash64::new(seed).mix_usize(iter);
        let Some(sample) = distinct_indices(base, n, 6, &mut scratch) else {
            return (0usize, f64::INFINITY);
        };
        let Some(pose) = pnp_dlt(world, obs, &sample) else {
            return (0usize, f64::INFINITY);
        };
        let (count, residual) = score_pose(&pose, world, obs, threshold);
        (count, residual)
    });

    let mut best_iter = usize::MAX;
    let mut best_count = 0usize;
    let mut best_residual = f64::INFINITY;
    for (iter, &(count, residual)) in scored.iter().enumerate() {
        if count < 6 {
            continue;
        }
        let better = count > best_count
            || (count == best_count && residual.total_cmp(&best_residual).is_lt());
        if better {
            best_iter = iter;
            best_count = count;
            best_residual = residual;
        }
    }
    if best_iter == usize::MAX {
        return None;
    }

    let mut scratch = Vec::new();
    let sample = distinct_indices(Hash64::new(seed).mix_usize(best_iter), n, 6, &mut scratch)?;
    let coarse = pnp_dlt(world, obs, &sample)?;
    let inliers: Vec<u32> = (0..n as u32)
        .filter(|&i| reprojection_error(&coarse, world[i as usize], obs[i as usize]) <= threshold)
        .collect();
    if inliers.len() < 6 {
        return None;
    }

    let refined = refine_pose(&coarse, world, obs, &inliers, cfg, threshold);
    let pose = refined.unwrap_or(coarse);
    let inliers: Vec<u32> = (0..n as u32)
        .filter(|&i| reprojection_error(&pose, world[i as usize], obs[i as usize]) <= threshold)
        .collect();
    let residual = inliers
        .iter()
        .map(|&i| reprojection_error(&pose, world[i as usize], obs[i as usize]))
        .sum();
    Some(PnpModel {
        pose,
        inliers,
        residual,
    })
}

/// Reprojection error of one correspondence in **normalized** units, or
/// infinity when the point is behind the camera.
#[inline]
pub fn reprojection_error(pose: &Pose, world: DVec3, obs: DVec2) -> f64 {
    let p = pose.transform(world);
    if p.z <= 1e-9 || !p.z.is_finite() {
        return f64::INFINITY;
    }
    (DVec2::new(p.x / p.z, p.y / p.z) - obs).length()
}

fn score_pose(pose: &Pose, world: &[DVec3], obs: &[DVec2], threshold: f64) -> (usize, f64) {
    let mut count = 0usize;
    let mut residual = 0.0;
    for i in 0..world.len() {
        let e = reprojection_error(pose, world[i], obs[i]);
        if e <= threshold {
            count += 1;
            residual += e;
        }
    }
    (count, residual)
}

/// Nonlinear **motion-only** refinement: Levenberg–Marquardt on the six pose
/// degrees of freedom with a Huber kernel, the 3D points held fixed.
///
/// The same residual, the same retraction and the same robustifier as
/// [`crate::bundle`] — this is that solver with the structure block removed, and
/// it exists because a DLT pose is a good enough starting point and a poor
/// enough answer.
pub fn refine_pose(
    initial: &Pose,
    world: &[DVec3],
    obs: &[DVec2],
    inliers: &[u32],
    cfg: &RansacConfig,
    threshold: f64,
) -> Option<Pose> {
    if inliers.len() < 3 {
        return None;
    }
    // Thresholds arrive in normalized units here; the Huber knee is given in
    // pixels, so it is converted through the same ratio the caller used.
    let huber = threshold * (cfg.refine_huber_px / cfg.pnp_threshold_px).max(1e-6);
    let mut pose = *initial;
    let mut cost = robust_cost(&pose, world, obs, inliers, huber);
    let mut lambda = 1e-4f64;

    for _ in 0..cfg.refine_iterations {
        let mut normal = vec![0.0f64; 36];
        let mut gradient = [0.0f64; 6];
        let mut used = 0usize;
        for &i in inliers {
            let p = pose.transform(world[i as usize]);
            if p.z <= 1e-9 {
                continue;
            }
            let inv_z = 1.0 / p.z;
            let proj = DVec2::new(p.x * inv_z, p.y * inv_z);
            let r = proj - obs[i as usize];
            // d(proj)/d(p)
            let dn = [[inv_z, 0.0, -proj.x * inv_z], [0.0, inv_z, -proj.y * inv_z]];
            // d(p)/d(omega) = -[p]_×, d(p)/d(delta) = I
            let cross = [[0.0, p.z, -p.y], [-p.z, 0.0, p.x], [p.y, -p.x, 0.0]];
            let mut j = [[0.0f64; 6]; 2];
            for (row, jrow) in j.iter_mut().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += dn[row][k] * cross[k][c];
                    }
                    jrow[c] = acc;
                    jrow[3 + c] = dn[row][c];
                }
            }
            let w = huber_weight(r.length(), huber);
            used += 1;
            for (row, jrow) in j.iter().enumerate() {
                let ri = if row == 0 { r.x } else { r.y };
                for c in 0..6 {
                    gradient[c] -= w * jrow[c] * ri;
                    for d in 0..6 {
                        normal[c * 6 + d] += w * jrow[c] * jrow[d];
                    }
                }
            }
        }
        if used < 3 {
            return None;
        }
        let mut damped = normal.clone();
        for c in 0..6 {
            let d = damped[c * 6 + c];
            damped[c * 6 + c] = d + lambda * d.max(1e-9);
        }
        let Some(step) = solve_ldlt(&mut damped, &gradient, 6) else {
            lambda *= 10.0;
            continue;
        };
        let candidate = pose.retract(
            DVec3::new(step[0], step[1], step[2]),
            DVec3::new(step[3], step[4], step[5]),
        );
        let candidate_cost = robust_cost(&candidate, world, obs, inliers, huber);
        if candidate_cost < cost {
            pose = candidate;
            cost = candidate_cost;
            lambda = (lambda * 0.1).max(1e-10);
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    Some(pose)
}

/// The Huber IRLS weight for a residual of length `r`.
///
/// `rho(s) = s` below the knee and `2*delta*sqrt(s) - delta²` above it, with
/// `s = r²`; the weight applied to a residual is `rho'(s)`. The second-order
/// Triggs correction is **not** applied — it costs a per-observation eigen
/// decision and buys convergence speed, not a different minimum. Stated so the
/// omission is a decision.
#[inline]
pub fn huber_weight(r: f64, delta: f64) -> f64 {
    if r <= delta || !r.is_finite() {
        1.0
    } else {
        delta / r
    }
}

fn robust_cost(pose: &Pose, world: &[DVec3], obs: &[DVec2], inliers: &[u32], huber: f64) -> f64 {
    let mut total = 0.0;
    for &i in inliers {
        let p = pose.transform(world[i as usize]);
        if p.z <= 1e-9 || !p.z.is_finite() {
            total += huber * huber * 4.0;
            continue;
        }
        let r = (DVec2::new(p.x / p.z, p.y / p.z) - obs[i as usize]).length();
        total += if r <= huber {
            r * r
        } else {
            2.0 * huber * r - huber * huber
        };
    }
    total
}

/// `[t]_×`, the skew-symmetric cross-product matrix — exported because the
/// essential matrix's definition is written in terms of it and the tests check
/// against that definition rather than against this module's own output.
pub fn skew(v: DVec3) -> DMat3 {
    mat3_from_rows([0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0])
}

/// `E = [t]_× R` for a relative pose — the definition, for tests and for
/// anything that needs a synthetic epipolar model.
pub fn essential_from_pose(pose: &Pose) -> DMat3 {
    skew(pose.translation).mul_mat3(&pose.rotation_matrix())
}

/// A readable dump of a matrix, for assertion messages.
pub fn mat3_debug(m: &DMat3) -> String {
    let mut s = String::new();
    for r in 0..3 {
        s.push_str(&format!(
            "[{:.6} {:.6} {:.6}]",
            mat3_at(m, r, 0),
            mat3_at(m, r, 1),
            mat3_at(m, r, 2)
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Hash64;

    /// An asymmetric little cloud: no plane contains it, and no reflection or
    /// axis swap maps it to itself. A symmetric fixture cannot see a sign error.
    fn cloud(n: usize) -> Vec<DVec3> {
        (0..n)
            .map(|i| {
                let h = Hash64::new(0xc10d).mix_usize(i);
                DVec3::new(
                    -1.5 + 3.0 * h.mix_u64(1).unit(),
                    -0.8 + 1.9 * h.mix_u64(2).unit(),
                    4.0 + 3.5 * h.mix_u64(3).unit(),
                )
            })
            .collect()
    }

    fn project_all(pose: &Pose, points: &[DVec3]) -> Vec<DVec2> {
        points
            .iter()
            .map(|&x| {
                let p = pose.transform(x);
                DVec2::new(p.x / p.z, p.y / p.z)
            })
            .collect()
    }

    fn second_camera() -> Pose {
        // Asymmetric: translated in all three axes and rotated about a slanted
        // axis, so no coordinate permutation reproduces it.
        Pose::look_at(
            DVec3::new(0.9, 0.35, -0.4),
            DVec3::new(0.1, -0.05, 5.0),
            DVec3::new(0.06, -1.0, 0.03),
        )
    }

    #[test]
    fn the_eight_point_algorithm_recovers_a_planted_essential_matrix() {
        let points = cloud(40);
        let a = project_all(&Pose::IDENTITY, &points);
        let pose = second_camera();
        let b = project_all(&pose, &points);
        let indices: Vec<u32> = (0..40).collect();
        let e = essential_eight_point(&a, &b, &indices).expect("eight-point solves");
        for i in 0..40 {
            let d = sampson_distance(&e, a[i], b[i]);
            assert!(
                d < 1e-9,
                "correspondence {i} is {d} off its own epipolar model"
            );
        }
        // And it agrees with the definition, up to scale and sign. Both are
        // normalized to unit Frobenius norm and their signs aligned on the
        // element of largest magnitude, so the comparison is scale-free without
        // dividing by an element that might be near zero.
        let truth = essential_from_pose(&pose);
        let (na, nb) = (normalize_mat3(&e), normalize_mat3(&truth));
        let mut worst = 0.0f64;
        for r in 0..3 {
            for c in 0..3 {
                worst = worst.max((mat3_at(&na, r, c) - mat3_at(&nb, r, c)).abs());
            }
        }
        assert!(
            worst < 1e-8,
            "E disagrees with [t]x R by {worst}: {} vs {}",
            mat3_debug(&na),
            mat3_debug(&nb)
        );
    }

    /// Unit Frobenius norm with the largest-magnitude element made positive.
    fn normalize_mat3(m: &DMat3) -> DMat3 {
        let mut norm = 0.0f64;
        let mut pivot = (0usize, 0usize);
        for r in 0..3 {
            for c in 0..3 {
                let v = mat3_at(m, r, c);
                norm += v * v;
                if v.abs() > mat3_at(m, pivot.0, pivot.1).abs() {
                    pivot = (r, c);
                }
            }
        }
        let norm = norm.sqrt();
        let sign = if mat3_at(m, pivot.0, pivot.1) < 0.0 {
            -1.0
        } else {
            1.0
        };
        *m * (sign / norm)
    }

    #[test]
    fn pose_from_essential_recovers_the_planted_pose() {
        let points = cloud(40);
        let a = project_all(&Pose::IDENTITY, &points);
        let truth = second_camera();
        let b = project_all(&truth, &points);
        let indices: Vec<u32> = (0..40).collect();
        let e = essential_eight_point(&a, &b, &indices).unwrap();
        let (pose, count) = pose_from_essential(&e, &a, &b, &indices).expect("cheirality resolves");
        assert_eq!(count, 40, "not every point landed in front of both cameras");
        // Translation is only recoverable up to scale.
        let want_dir = truth.translation.normalize();
        let got_dir = pose.translation.normalize();
        assert!(
            (want_dir - got_dir).length() < 1e-6,
            "baseline direction {got_dir} vs {want_dir}"
        );
        let dq = pose.rotation * truth.rotation.inverse();
        assert!(
            dq.w.abs() > 1.0 - 1e-9,
            "rotation differs: w = {} (1.0 is identity)",
            dq.w.abs()
        );
    }

    #[test]
    fn the_three_wrong_cheirality_candidates_are_actually_wrong() {
        // Anti-vacuity: if the four candidates were all equivalent, the test
        // above would pass with any of them.
        let points = cloud(30);
        let a = project_all(&Pose::IDENTITY, &points);
        let truth = second_camera();
        let b = project_all(&truth, &points);
        let indices: Vec<u32> = (0..30).collect();
        let e = essential_eight_point(&a, &b, &indices).unwrap();
        let counts: Vec<usize> = pose_candidates(&e)
            .iter()
            .map(|cand| {
                indices
                    .iter()
                    .filter(|&&i| {
                        triangulate_dlt(&[(Pose::IDENTITY, a[i as usize]), (*cand, b[i as usize])])
                            .map(|x| {
                                Pose::IDENTITY.transform(x).z > 0.0 && cand.transform(x).z > 0.0
                            })
                            .unwrap_or(false)
                    })
                    .count()
            })
            .collect();
        let winners = counts.iter().filter(|&&c| c == 30).count();
        assert_eq!(winners, 1, "cheirality did not discriminate: {counts:?}");
    }

    #[test]
    fn triangulation_returns_the_planted_point() {
        let poses = [
            Pose::IDENTITY,
            second_camera(),
            Pose::look_at(
                DVec3::new(-0.8, 0.2, 0.3),
                DVec3::new(0.0, 0.0, 5.0),
                DVec3::new(-0.02, -1.0, 0.0),
            ),
        ];
        for x in cloud(12) {
            let obs: Vec<(Pose, DVec2)> = poses
                .iter()
                .map(|p| {
                    let c = p.transform(x);
                    (*p, DVec2::new(c.x / c.z, c.y / c.z))
                })
                .collect();
            let got = triangulate_dlt(&obs).expect("three rays intersect");
            assert!((got - x).length() < 1e-9, "triangulated {got} vs {x}");
        }
    }

    #[test]
    fn triangulation_refuses_parallel_rays() {
        // Two cameras at the same centre see the same ray twice.
        let a = Pose::IDENTITY;
        let b = Pose::from_qt(a.rotation, a.translation);
        let obs = [(a, DVec2::new(0.1, 0.2)), (b, DVec2::new(0.1, 0.2))];
        // Either no solution, or one whose parallax is exactly degenerate.
        match triangulate_dlt(&obs) {
            None => {}
            Some(x) => {
                let cos = parallax_cos(x, &[a, b]);
                assert!(cos > 0.999_999, "a zero baseline reported parallax: {cos}");
            }
        }
    }

    #[test]
    fn parallax_falls_as_the_baseline_shrinks() {
        let x = DVec3::new(0.0, 0.0, 10.0);
        let wide = [
            Pose::IDENTITY,
            Pose::from_rt(DMat3::IDENTITY, DVec3::new(-3.0, 0.0, 0.0)),
        ];
        let narrow = [
            Pose::IDENTITY,
            Pose::from_rt(DMat3::IDENTITY, DVec3::new(-0.05, 0.0, 0.0)),
        ];
        assert!(parallax_cos(x, &wide) < parallax_cos(x, &narrow));
        assert!(parallax_cos(x, &narrow) > 0.99);
    }

    #[test]
    fn pnp_recovers_a_planted_pose() {
        let points = cloud(20);
        let truth = Pose::look_at(
            DVec3::new(1.7, -0.6, -1.2),
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::new(0.05, -1.0, -0.02),
        );
        let obs = project_all(&truth, &points);
        let indices: Vec<u32> = (0..20).collect();
        let pose = pnp_dlt(&points, &obs, &indices).expect("DLT-PnP solves");
        assert!(
            (pose.center() - truth.center()).length() < 1e-7,
            "centre {} vs {}",
            pose.center(),
            truth.center()
        );
        let dq = pose.rotation * truth.rotation.inverse();
        assert!(
            dq.w.abs() > 1.0 - 1e-10,
            "rotation differs: w = {}",
            dq.w.abs()
        );
        for i in 0..20 {
            assert!(reprojection_error(&pose, points[i], obs[i]) < 1e-9);
        }
    }

    #[test]
    fn pnp_refuses_fewer_than_six_points() {
        let points = cloud(20);
        let truth = second_camera();
        let obs = project_all(&truth, &points);
        assert!(pnp_dlt(&points, &obs, &[0, 1, 2, 3, 4]).is_none());
    }

    #[test]
    fn ransac_essential_ignores_planted_outliers() {
        let pool = JobPool::new(2);
        let points = cloud(80);
        let truth = second_camera();
        let mut a = project_all(&Pose::IDENTITY, &points);
        let mut b = project_all(&truth, &points);
        // Corrupt a quarter of the correspondences by shuffling their partners.
        for i in (0..80).step_by(4) {
            let j = (i + 37) % 80;
            b[i] = b[j] + DVec2::new(0.05, -0.03);
        }
        let cfg = RansacConfig::default();
        let model = ransac_essential(&a, &b, 0.002, &cfg, 0xbeef, &pool).expect("a model");
        assert!(
            model.inliers.len() >= 58,
            "only {} of the 60 clean correspondences survived",
            model.inliers.len()
        );
        assert!(
            model.inliers.iter().all(|&i| i % 4 != 0),
            "an outlier was accepted as an inlier"
        );
        // A corrupted-beyond-repair set must not produce a confident model.
        for x in a.iter_mut() {
            *x += DVec2::new(0.4, -0.4);
        }
        let junk = ransac_essential(&a, &b, 0.0001, &cfg, 0xbeef, &pool);
        assert!(
            junk.map(|m| m.inliers.len()).unwrap_or(0) < 40,
            "a scrambled set produced a confident model"
        );
    }

    #[test]
    fn ransac_is_pool_size_invariant() {
        let points = cloud(60);
        let truth = second_camera();
        let a = project_all(&Pose::IDENTITY, &points);
        let mut b = project_all(&truth, &points);
        for i in (0..60).step_by(5) {
            b[i] += DVec2::new(0.07, 0.05);
        }
        let cfg = RansacConfig::default();
        let one = ransac_essential(&a, &b, 0.002, &cfg, 7, &JobPool::new(1)).unwrap();
        let eight = ransac_essential(&a, &b, 0.002, &cfg, 7, &JobPool::new(8)).unwrap();
        assert_eq!(one.inliers, eight.inliers);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    mat3_at(&one.essential, r, c).to_bits(),
                    mat3_at(&eight.essential, r, c).to_bits(),
                    "the essential matrix depended on the worker count"
                );
            }
        }
    }

    #[test]
    fn ransac_pnp_ignores_planted_outliers_and_refines() {
        let pool = JobPool::new(2);
        let points = cloud(60);
        let truth = Pose::look_at(
            DVec3::new(-1.1, 0.9, -0.7),
            DVec3::new(0.2, 0.0, 5.0),
            DVec3::new(0.04, -1.0, 0.01),
        );
        let mut obs = project_all(&truth, &points);
        for i in (0..60).step_by(3) {
            obs[i] += DVec2::new(0.11, -0.09);
        }
        let cfg = RansacConfig::default();
        let model = ransac_pnp(&points, &obs, 0.004, &cfg, 0xfeed, &pool).expect("a pose");
        assert!(
            model.inliers.len() >= 38,
            "only {} of 40 clean correspondences survived",
            model.inliers.len()
        );
        assert!(
            (model.pose.center() - truth.center()).length() < 1e-4,
            "centre {} vs {}",
            model.pose.center(),
            truth.center()
        );
        let one = ransac_pnp(&points, &obs, 0.004, &cfg, 0xfeed, &JobPool::new(1)).unwrap();
        let eight = ransac_pnp(&points, &obs, 0.004, &cfg, 0xfeed, &JobPool::new(8)).unwrap();
        assert_eq!(one.inliers, eight.inliers);
        assert_eq!(
            one.pose.translation.to_array().map(f64::to_bits),
            eight.pose.translation.to_array().map(f64::to_bits)
        );
    }

    #[test]
    fn the_huber_weight_is_one_below_the_knee_and_falls_above_it() {
        assert_eq!(huber_weight(0.5, 1.0), 1.0);
        assert_eq!(huber_weight(1.0, 1.0), 1.0);
        assert!((huber_weight(4.0, 1.0) - 0.25).abs() < 1e-15);
        assert_eq!(huber_weight(f64::INFINITY, 1.0), 1.0);
    }

    #[test]
    fn refinement_improves_a_perturbed_pose() {
        let points = cloud(40);
        let truth = second_camera();
        let obs = project_all(&truth, &points);
        let perturbed = truth.retract(
            DVec3::new(0.02, -0.015, 0.01),
            DVec3::new(0.03, 0.02, -0.025),
        );
        let inliers: Vec<u32> = (0..40).collect();
        let before: f64 = inliers
            .iter()
            .map(|&i| reprojection_error(&perturbed, points[i as usize], obs[i as usize]))
            .sum();
        let cfg = RansacConfig::default();
        let refined = refine_pose(&perturbed, &points, &obs, &inliers, &cfg, 0.01).unwrap();
        let after: f64 = inliers
            .iter()
            .map(|&i| reprojection_error(&refined, points[i as usize], obs[i as usize]))
            .sum();
        assert!(
            after < before * 1e-3,
            "refinement barely moved: {before} -> {after}"
        );
        assert!(
            (refined.center() - truth.center()).length() < 1e-6,
            "refined centre {} vs {}",
            refined.center(),
            truth.center()
        );
    }
}
