//! Small dense **f64** linear algebra, hand-rolled.
//!
//! # Why there is no `nalgebra` here
//!
//! Structure from motion needs exactly four things beyond `glam`:
//!
//! * the null vector of a small over-determined homogeneous system (the
//!   eight-point essential matrix at `n = 9`, DLT triangulation at `n = 4`,
//!   DLT-PnP at `n = 12`) — which is the eigenvector of `AᵀA` for its smallest
//!   eigenvalue;
//! * the SVD of a 3x3 (projecting a matrix onto the essential manifold, and
//!   onto the nearest rotation);
//! * a symmetric solve for the bundle adjuster's Schur-reduced camera system;
//! * a 3x3 symmetric inverse for its point blocks.
//!
//! Every one of them is a *fixed, small* problem. Pulling a general linear
//! algebra crate in for four routines buys a license review, a version pin, and
//! a build-time cost, in exchange for code this file can carry with tests
//! against it. The house prefers owned math with tests. If a future batch needs
//! sparse Cholesky over thousands of cameras, that ruling gets revisited on the
//! evidence and memo'd — it is not revisited here.
//!
//! # Determinism
//!
//! Everything is serial, in a fixed order, in `f64`, and uses only `+ - * /` and
//! `sqrt` (IEEE-754 correctly rounded, therefore identical on every platform).
//! No transcendental is called anywhere in this module. The Jacobi sweep count
//! is bounded by a constant and its early exit is a **value** comparison, not a
//! time or iteration budget that could differ between runs.
//!
//! # Storage convention
//!
//! General `n x n` matrices are **row-major** `&[f64]` of length `n*n`: element
//! `(row, col)` is `m[row * n + col]`. Eigenvectors come back as the **columns**
//! of a row-major matrix, i.e. component `i` of eigenvector `j` is `v[i*n + j]`.
//! 3x3 work uses `glam::DMat3`, which is **column**-major — [`mat3_from_rows`]
//! and [`mat3_at`] exist so that a formula written row-wise stays readable.

use glam::{DMat3, DVec3};

/// Hard cap on Jacobi sweeps. A cyclic Jacobi sweep squares the off-diagonal
/// norm once it is in the quadratic regime, so 8-12 sweeps is convergence for
/// every size this crate uses; 64 is the "something is pathological" ceiling and
/// exists so the routine is a *bounded* computation rather than a `while`.
pub const MAX_JACOBI_SWEEPS: usize = 64;

/// Symmetric eigen-decomposition by **cyclic Jacobi rotations**.
///
/// `a` is a row-major symmetric `n x n` matrix. Returns `(eigenvalues,
/// eigenvectors)` where the eigenvectors are the columns of a row-major `n x n`
/// matrix, **sorted by eigenvalue ascending** — so column `0` is the null
/// direction of a homogeneous system.
///
/// Jacobi rather than a tridiagonal QR because it is thirty lines, is
/// unconditionally stable for symmetric input, and delivers small eigenvalues to
/// full relative accuracy — which is the entire point when the answer *is* the
/// smallest eigenvector.
pub fn jacobi_eigen_sym(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        n * n,
        "jacobi_eigen_sym wants a square row-major matrix"
    );
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // A scale-relative convergence floor: once the off-diagonal energy is below
    // this the rotations are pure rounding noise.
    let mut frobenius = 0.0;
    for x in &m {
        frobenius += x * x;
    }
    let tol = frobenius * (f64::EPSILON * f64::EPSILON) * (n as f64);

    for _sweep in 0..MAX_JACOBI_SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                // tan of the rotation, via the numerically stable branch that
                // never subtracts nearly equal quantities.
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // m <- Jᵀ m J, applied as columns then rows.
                for k in 0..n {
                    let akp = m[k * n + p];
                    let akq = m[k * n + q];
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = m[p * n + k];
                    let aqk = m[q * n + k];
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                // v <- v J
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Sort ascending by eigenvalue, ties broken by column index so the
    // permutation is a total order and cannot depend on sort stability.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]).then(i.cmp(&j)));

    let eigenvalues: Vec<f64> = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0f64; n * n];
    for (dst, &src) in order.iter().enumerate() {
        // Sign-canonicalise: the first component with the largest magnitude is
        // made positive. An eigenvector is only defined up to sign, and a sign
        // that flips between runs would flip a reconstruction's bytes.
        let mut pivot = 0usize;
        for i in 1..n {
            if v[i * n + src].abs() > v[pivot * n + src].abs() {
                pivot = i;
            }
        }
        let sign = if v[pivot * n + src] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            vectors[i * n + dst] = sign * v[i * n + src];
        }
    }
    (eigenvalues, vectors)
}

/// The unit null vector of a homogeneous system `A x = 0`: the eigenvector of
/// `AᵀA` for its smallest eigenvalue.
///
/// `ata` is the row-major `n x n` normal matrix (the caller accumulates it, so
/// an over-determined `A` with thousands of rows never has to be materialised).
pub fn smallest_eigenvector(ata: &[f64], n: usize) -> Vec<f64> {
    let (_, vectors) = jacobi_eigen_sym(ata, n);
    (0..n).map(|i| vectors[i * n]).collect()
}

/// Accumulate `AᵀA += rowᵀ row` for one row of a homogeneous system.
#[inline]
pub fn accumulate_normal(ata: &mut [f64], row: &[f64]) {
    let n = row.len();
    for i in 0..n {
        let ri = row[i];
        if ri == 0.0 {
            continue;
        }
        for j in 0..n {
            ata[i * n + j] += ri * row[j];
        }
    }
}

/// Build a [`DMat3`] from **rows**, so a formula written row-wise transcribes
/// directly (glam stores columns).
#[inline]
pub fn mat3_from_rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(r0[0], r1[0], r2[0]),
        DVec3::new(r0[1], r1[1], r2[1]),
        DVec3::new(r0[2], r1[2], r2[2]),
    )
}

/// Element `(row, col)` of a [`DMat3`].
#[inline]
pub fn mat3_at(m: &DMat3, row: usize, col: usize) -> f64 {
    m.col(col)[row]
}

/// A 3x3 singular value decomposition: `m == u * diag(s) * vᵀ`, with `s` sorted
/// **descending** and `u`, `v` orthonormal.
///
/// # Why one-sided Jacobi and not `mᵀm`
///
/// The obvious route — eigen-decompose `mᵀm`, take square roots — was the first
/// implementation and it is **not accurate enough**. Forming `mᵀm` squares the
/// condition number, so a singular value that should be zero comes back at
/// `sqrt(eps)` relative to the largest, around `1e-8`. That was measured, not
/// assumed: `project_to_essential` left a third singular value of `5e-9` where
/// it needed a hard zero, and a reconstructed 3x3 differed from its input in the
/// eighth decimal.
///
/// **One-sided Jacobi** rotates the *columns of `m` itself* until they are
/// mutually orthogonal. The singular values are then the column norms, computed
/// without ever squaring the matrix, and small ones come out to full relative
/// accuracy. It is the same rotation formula, applied on the right only.
///
/// Columns of `u` whose singular value has collapsed are completed by
/// orthogonalising against the ones that survived, so a rank-deficient input
/// still comes back with an orthonormal `u` — an essential matrix's third
/// singular value is *supposed* to be zero, and [`project_to_essential`] would
/// be meaningless otherwise.
pub fn svd3(m: &DMat3) -> (DMat3, DVec3, DMat3) {
    let mut work = [m.col(0), m.col(1), m.col(2)];
    let mut rot = [DVec3::X, DVec3::Y, DVec3::Z];
    for _ in 0..MAX_JACOBI_SWEEPS {
        let mut worst = 0.0f64;
        for p in 0..3 {
            for q in (p + 1)..3 {
                let alpha = work[p].length_squared();
                let beta = work[q].length_squared();
                let gamma = work[p].dot(work[q]);
                let scale = (alpha * beta).sqrt();
                if scale > 0.0 {
                    worst = worst.max(gamma.abs() / scale);
                }
                if gamma == 0.0 {
                    continue;
                }
                // The rotation that orthogonalises columns p and q:
                // t² + 2 zeta t - 1 = 0, taking the root of smaller magnitude.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                let (ap, aq) = (work[p], work[q]);
                work[p] = ap * c - aq * s;
                work[q] = ap * s + aq * c;
                let (vp, vq) = (rot[p], rot[q]);
                rot[p] = vp * c - vq * s;
                rot[q] = vp * s + vq * c;
            }
        }
        if worst <= f64::EPSILON {
            break;
        }
    }

    // Column norms are the singular values; sort descending with the column
    // index as a tie-break so the permutation is a total order.
    let mut order = [0usize, 1, 2];
    let norms = [work[0].length(), work[1].length(), work[2].length()];
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]).then(i.cmp(&j)));
    let mut s = [0.0f64; 3];
    let mut v_cols = [DVec3::ZERO; 3];
    let mut u_raw = [DVec3::ZERO; 3];
    for (k, &src) in order.iter().enumerate() {
        s[k] = norms[src];
        v_cols[k] = rot[src];
        u_raw[k] = work[src];
    }
    let v = DMat3::from_cols(v_cols[0], v_cols[1], v_cols[2]);

    let scale = s[0].max(f64::MIN_POSITIVE);
    let mut u_cols = [DVec3::ZERO; 3];
    let mut solid = [false; 3];
    for k in 0..3 {
        if s[k] > scale * 1e-14 {
            u_cols[k] = u_raw[k] / s[k];
            solid[k] = true;
        } else {
            s[k] = 0.0;
        }
    }
    // Complete the basis for any collapsed direction.
    for k in 0..3 {
        if solid[k] {
            continue;
        }
        let (a, b) = match k {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        let candidate = if solid[a] && solid[b] {
            u_cols[a].cross(u_cols[b])
        } else if solid[a] {
            orthogonal_to(u_cols[a])
        } else if solid[b] {
            orthogonal_to(u_cols[b])
        } else {
            DVec3::new(
                if k == 0 { 1.0 } else { 0.0 },
                if k == 1 { 1.0 } else { 0.0 },
                if k == 2 { 1.0 } else { 0.0 },
            )
        };
        u_cols[k] = normalize_or(candidate, DVec3::Z);
        solid[k] = true;
    }
    // Re-orthonormalise in a fixed order (Gram-Schmidt), so `u` is orthonormal
    // to machine precision even when two singular values were near-equal.
    u_cols[0] = normalize_or(u_cols[0], DVec3::X);
    u_cols[1] = normalize_or(u_cols[1] - u_cols[0] * u_cols[0].dot(u_cols[1]), DVec3::Y);
    u_cols[2] = normalize_or(
        u_cols[2] - u_cols[0] * u_cols[0].dot(u_cols[2]) - u_cols[1] * u_cols[1].dot(u_cols[2]),
        u_cols[0].cross(u_cols[1]),
    );

    (
        DMat3::from_cols(u_cols[0], u_cols[1], u_cols[2]),
        DVec3::new(s[0], s[1], s[2]),
        v,
    )
}

/// Some unit vector perpendicular to `v`, chosen by a fixed rule so it is a
/// function of `v` alone.
#[inline]
fn orthogonal_to(v: DVec3) -> DVec3 {
    let axis = if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        DVec3::X
    } else if v.y.abs() <= v.z.abs() {
        DVec3::Y
    } else {
        DVec3::Z
    };
    v.cross(axis)
}

/// Normalize, falling back to `fallback` when the input is degenerate.
#[inline]
pub fn normalize_or(v: DVec3, fallback: DVec3) -> DVec3 {
    let len = v.length();
    if len > 0.0 && len.is_finite() {
        v / len
    } else {
        fallback
    }
}

/// The **nearest rotation** to `m` in the Frobenius sense: `u vᵀ` with the
/// determinant forced to `+1` by flipping the sign of the least significant
/// singular direction.
pub fn nearest_rotation(m: &DMat3) -> DMat3 {
    let (u, _, v) = svd3(m);
    let mut r = u.mul_mat3(&v.transpose());
    if r.determinant() < 0.0 {
        // Negate the third column of u, i.e. the direction carrying the
        // smallest singular value — the minimal-cost way to fix handedness.
        let u2 = DMat3::from_cols(u.col(0), u.col(1), -u.col(2));
        r = u2.mul_mat3(&v.transpose());
    }
    r
}

/// Project a 3x3 onto the **essential manifold**: singular values forced to
/// `(1, 1, 0)`.
///
/// An essential matrix has exactly two equal non-zero singular values. The
/// eight-point algorithm solves a linear system that does not know that, so its
/// answer must be projected back before it means anything geometric.
pub fn project_to_essential(m: &DMat3) -> DMat3 {
    let (u, _, v) = svd3(m);
    let d = DMat3::from_cols(DVec3::X, DVec3::Y, DVec3::ZERO);
    u.mul_mat3(&d).mul_mat3(&v.transpose())
}

/// Solve a **symmetric** dense system `a x = b` in place by `LDLᵀ` with a
/// diagonal-magnitude guard.
///
/// `a` is row-major `n x n` and is destroyed. Returns `None` if a pivot is too
/// small to divide by, which for the bundle adjuster means "raise lambda and try
/// again" rather than "produce a wrong step".
///
/// `LDLᵀ` rather than Cholesky because the Schur complement of a
/// gauge-constrained system is positive *semi*-definite before damping, and a
/// `sqrt` of a whisker-negative pivot is a `NaN` that propagates silently.
pub fn solve_ldlt(a: &mut [f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n);
    let mut scale = 0.0f64;
    for i in 0..n {
        scale = scale.max(a[i * n + i].abs());
    }
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let floor = scale * 1e-14;

    // In-place LDLᵀ: the strict lower triangle holds L, the diagonal holds D.
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k] * a[k * n + k];
        }
        if d.abs() < floor || !d.is_finite() {
            return None;
        }
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k] * a[k * n + k];
            }
            a[i * n + j] = s / d;
        }
    }

    // Forward substitution (L y = b), diagonal scale (D z = y), back (Lᵀ x = z).
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= a[i * n + k] * y[k];
        }
        y[i] = s;
    }
    for i in 0..n {
        y[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * y[k];
        }
        y[i] = s;
    }
    if y.iter().all(|v| v.is_finite()) {
        Some(y)
    } else {
        None
    }
}

/// Invert a symmetric 3x3 (the bundle adjuster's point blocks). Returns `None`
/// when the block is singular — a point seen from one place, or from a line.
pub fn invert_sym3(m: &DMat3) -> Option<DMat3> {
    let det = m.determinant();
    let scale = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .fold(0.0f64, |acc, (r, c)| acc.max(mat3_at(m, r, c).abs()));
    if !scale.is_finite() || scale <= 0.0 || det.abs() < scale * scale * scale * 1e-14 {
        return None;
    }
    let inv = m.inverse();
    if (0..3).all(|c| inv.col(c).is_finite()) {
        Some(inv)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i * n + k] * b[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    #[test]
    fn jacobi_reproduces_a_known_spectrum() {
        // A symmetric matrix built as Q diag(l) Qᵀ from an explicit rotation, so
        // the eigenvalues are given, not estimated.
        let l = [0.25f64, 2.0, 7.5];
        // A right-handed orthonormal basis with no zeros in it.
        let q0 = DVec3::new(2.0, 1.0, 2.0).normalize();
        let q1 = DVec3::new(-1.0, 2.0, 0.0).normalize();
        let q1 = (q1 - q0 * q0.dot(q1)).normalize();
        let q2 = q0.cross(q1);
        let mut a = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = l[0] * q0[i] * q0[j] + l[1] * q1[i] * q1[j] + l[2] * q2[i] * q2[j];
            }
        }
        let (eig, vecs) = jacobi_eigen_sym(&a, 3);
        let mut expect = l;
        expect.sort_by(f64::total_cmp);
        for k in 0..3 {
            assert!(
                (eig[k] - expect[k]).abs() < 1e-12,
                "eig {k}: {} vs {}",
                eig[k],
                expect[k]
            );
        }
        // A v == lambda v for every column.
        for k in 0..3 {
            let v = DVec3::new(vecs[k], vecs[3 + k], vecs[6 + k]);
            assert!(
                (v.length() - 1.0).abs() < 1e-12,
                "eigenvector {k} is not unit"
            );
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i * 3 + j] * v[j]).sum();
                assert!(
                    (av - eig[k] * v[i]).abs() < 1e-11,
                    "A v != l v at ({k},{i})"
                );
            }
        }
    }

    #[test]
    fn jacobi_is_bit_identical_across_calls() {
        let mut a = vec![0.0; 81];
        for i in 0..9 {
            for j in 0..9 {
                let v = ((i * 13 + j * 7) % 11) as f64 * 0.37 - 1.1;
                a[i * 9 + j] += v;
                a[j * 9 + i] += v;
            }
        }
        let (e1, v1) = jacobi_eigen_sym(&a, 9);
        let (e2, v2) = jacobi_eigen_sym(&a, 9);
        assert_eq!(
            e1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            v1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            v2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn smallest_eigenvector_finds_a_planted_null_space() {
        // Rows all orthogonal to a chosen direction: the null vector is that
        // direction, up to sign, and sign is canonicalised.
        let null = [0.3f64, -0.5, 0.2, 0.77];
        let n = 4;
        let mut ata = vec![0.0; n * n];
        for r in 0..12 {
            // A pseudo-random row, made orthogonal to `null`.
            let mut row = [0.0f64; 4];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = (((r * 5 + j * 3) % 7) as f64) - 3.0 + 0.11 * (j as f64);
            }
            let dot: f64 = (0..4).map(|j| row[j] * null[j]).sum();
            let nn: f64 = null.iter().map(|v| v * v).sum();
            for j in 0..4 {
                row[j] -= dot / nn * null[j];
            }
            accumulate_normal(&mut ata, &row);
        }
        let v = smallest_eigenvector(&ata, n);
        let scale: f64 =
            (0..4).map(|j| v[j] * null[j]).sum::<f64>() / null.iter().map(|x| x * x).sum::<f64>();
        for j in 0..4 {
            assert!(
                (v[j] - scale * null[j]).abs() < 1e-9,
                "null vector component {j}: {} vs {}",
                v[j],
                scale * null[j]
            );
        }
    }

    #[test]
    fn svd3_reconstructs_its_input() {
        let cases = [
            mat3_from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]),
            mat3_from_rows([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            // Rank 2 on purpose: the third row is the sum of the first two.
            mat3_from_rows([1.0, 0.5, -2.0], [0.25, 3.0, 1.0], [1.25, 3.5, -1.0]),
        ];
        for (idx, m) in cases.iter().enumerate() {
            let (u, s, v) = svd3(m);
            let rebuilt = u
                .mul_mat3(&DMat3::from_cols(
                    DVec3::new(s.x, 0.0, 0.0),
                    DVec3::new(0.0, s.y, 0.0),
                    DVec3::new(0.0, 0.0, s.z),
                ))
                .mul_mat3(&v.transpose());
            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        (mat3_at(&rebuilt, r, c) - mat3_at(m, r, c)).abs() < 1e-10,
                        "case {idx} element ({r},{c}): {} vs {}",
                        mat3_at(&rebuilt, r, c),
                        mat3_at(m, r, c)
                    );
                }
            }
            assert!(
                s.x >= s.y && s.y >= s.z,
                "case {idx} singular values unsorted: {s}"
            );
            let ortho = u.transpose().mul_mat3(&u);
            for r in 0..3 {
                for c in 0..3 {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (mat3_at(&ortho, r, c) - want).abs() < 1e-10,
                        "case {idx}: u is not orthonormal at ({r},{c})"
                    );
                }
            }
        }
    }

    #[test]
    fn project_to_essential_lands_on_the_manifold() {
        let m = mat3_from_rows([0.4, -1.2, 0.9], [2.0, 0.1, -0.3], [0.7, 0.6, 1.4]);
        let e = project_to_essential(&m);
        let (_, s, _) = svd3(&e);
        assert!((s.x - 1.0).abs() < 1e-10, "s0 = {}", s.x);
        assert!((s.y - 1.0).abs() < 1e-10, "s1 = {}", s.y);
        assert!(s.z < 1e-10, "s2 = {}", s.z);
        // The defining cubic constraint: 2 E Eᵀ E - tr(E Eᵀ) E == 0.
        let eet = e.mul_mat3(&e.transpose());
        let trace = mat3_at(&eet, 0, 0) + mat3_at(&eet, 1, 1) + mat3_at(&eet, 2, 2);
        let lhs = eet.mul_mat3(&e);
        for r in 0..3 {
            for c in 0..3 {
                let v = 2.0 * mat3_at(&lhs, r, c) - trace * mat3_at(&e, r, c);
                assert!(
                    v.abs() < 1e-9,
                    "essential constraint violated at ({r},{c}): {v}"
                );
            }
        }
    }

    #[test]
    fn nearest_rotation_is_a_rotation_and_fixes_handedness() {
        let m = mat3_from_rows(
            [0.98, 0.03, -0.01],
            [-0.02, 1.03, 0.04],
            [0.00, -0.05, 0.97],
        );
        let r = nearest_rotation(&m);
        assert!(
            (r.determinant() - 1.0).abs() < 1e-12,
            "det = {}",
            r.determinant()
        );
        // A reflection must come back right-handed, not mirrored.
        let mirrored = mat3_from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]);
        let fixed = nearest_rotation(&mirrored);
        assert!((fixed.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ldlt_solves_a_symmetric_system() {
        let n = 6;
        // A strictly diagonally dominant symmetric matrix, so it is PD.
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    10.0 + i as f64
                } else {
                    0.5 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        let x_true: Vec<f64> = (0..n).map(|i| 0.3 * i as f64 - 1.0).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x_true[j]).sum())
            .collect();
        let x = solve_ldlt(&mut a.clone(), &b, n).expect("PD system must solve");
        for i in 0..n {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-10,
                "x[{i}] = {} vs {}",
                x[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn ldlt_refuses_a_singular_system() {
        let n = 3;
        // Rank 2: the third row equals the first.
        let mut a = vec![
            2.0, 1.0, 2.0, //
            1.0, 3.0, 1.0, //
            2.0, 1.0, 2.0,
        ];
        assert!(solve_ldlt(&mut a, &[1.0, 1.0, 1.0], n).is_none());
        assert!(solve_ldlt(&mut [0.0; 9], &[1.0, 1.0, 1.0], 3).is_none());
    }

    #[test]
    fn invert_sym3_round_trips_and_refuses_singular() {
        let m = mat3_from_rows([4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]);
        let inv = invert_sym3(&m).expect("PD block inverts");
        let id = m.mul_mat3(&inv);
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((mat3_at(&id, r, c) - want).abs() < 1e-12);
            }
        }
        let singular = mat3_from_rows([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]);
        assert!(invert_sym3(&singular).is_none());
        assert!(invert_sym3(&DMat3::ZERO).is_none());
    }

    #[test]
    fn dense_mul_agrees_with_the_jacobi_reconstruction() {
        // v diag(e) vᵀ == a, which is the decomposition's contract at n = 5.
        let n = 5;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = ((i * 3 + j * 5) % 9) as f64 - 4.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (e, v) = jacobi_eigen_sym(&a, n);
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            d[i * n + i] = e[i];
        }
        let mut vt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vt[i * n + j] = v[j * n + i];
            }
        }
        let rebuilt = dense_mul(&dense_mul(&v, &d, n), &vt, n);
        for i in 0..n * n {
            assert!(
                (rebuilt[i] - a[i]).abs() < 1e-10,
                "element {i}: {} vs {}",
                rebuilt[i],
                a[i]
            );
        }
    }
}
