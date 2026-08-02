//! Shared fixture meshes for every vgeom test suite — the **one** place a
//! displaced-grid fixture is generated.
//!
//! # Why one generator, and why it is a crate feature
//!
//! This grid used to be copy-pasted into nine places: this crate's unit tests
//! (`asset`, `stream`, `model`), its integration tests (`tests/streaming.rs`),
//! `inf-render`'s three vgeom suites plus its frame-budget gate, and
//! `inf-player`'s activation gate. Nine copies of a *fixture* is normally just
//! noise; for this fixture it was a correctness problem, because the copies were
//! not textually identical (two normal conventions, one of them written with the
//! constant folded by hand) and nothing stopped one of them from drifting into a
//! shape whose DAG no longer exercised what its tests claimed. So the generator
//! lives here, behind the `test-support` feature, and the other crates' *dev*-
//! dependencies switch it on. It stays **test-only**: a shipping build never
//! enables the feature, and the module is host-only regardless because
//! [`build_vgeom`] links `meshopt`'s C++, which is not in the wasm player.
//!
//! # Why the displacement is `psin64`/`pcos64` and never `std` trig
//!
//! **The P14 LAW.** `f32::sin`/`f32::cos` are not bit-portable — MSVC's CRT,
//! glibc's libm and Apple's libm round the last bits differently. A fixture
//! displaced with `std` trig therefore hands `meshopt` *different vertices* on
//! each platform, and it simplifies them into a genuinely **different meshlet
//! DAG** — not a ULP difference, a structural one. Measured on the `n = 48`
//! fixture: 138 340 B of resident pages on x86_64-msvc against 138 176 B on
//! aarch64-apple-darwin, meshlet counts differing by several, and per-page
//! `max_parent_error` moving by percent. Two macOS-only CI failures came straight
//! out of that: a flythrough tuned clear of an error boundary on Windows landed on
//! the far side of it on macOS, and
//! `stream::tests::reimported_content_under_one_id_resets_residency`'s
//! anti-vacuity premise — two fixture builds agreeing on `(meshlet_count,
//! page_count)` — held on Windows (41, 5) and failed on macOS.
//!
//! Everything below is built from IEEE-exact operations only: `f32`
//! add/sub/mul/div and `f64` `sqrt` are correctly rounded (hence identical
//! everywhere), and the trig goes through [`inf_math::psin64`] /
//! [`inf_math::pcos64`], which are pure add/mul/floor polynomials. The grid
//! coordinates stay in `f32` exactly as they always were; only the transcendental
//! step is lifted into `f64` and cast back once per vertex — the same recipe
//! `inf_render::primitives` uses for committed primitive geometry.
//!
//! Consequence, and the whole point: **every platform now cooks these fixtures to
//! byte-identical vertices, hence to the identical meshlet DAG.** Any
//! `(meshlet_count, page_count, page_bytes)` fact a test establishes on one
//! machine is therefore true on all of them — which is what lets an anti-vacuity
//! guard like the re-import one above be a *construction* rather than a wish.
//! Cost of the fix, paid once: the DAG changed on every platform, so the two
//! vgeom goldens were deliberately re-blessed.

use inf_math::{pcos64, psin64};

use crate::build::{build_vgeom, BuildParams};
use crate::model::VgeomMesh;

/// `(positions, normals, uvs, indices)` — the four streams [`build_vgeom`] takes.
pub type MeshStreams = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>);

/// Which normal field [`displaced_grid`] writes.
///
/// Normals enter neither clusterization nor simplification, so this changes a
/// fixture's *shading* and never its DAG — the two variants page identically,
/// which is what lets the render suites and the paging suites quote each other's
/// page ladders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridNormals {
    /// Constant `+Y`. The DAG/paging fixtures use this: nothing they assert reads
    /// a normal, so the cheapest field that keeps the mesh well-formed wins.
    Flat,
    /// The analytic surface normal of the displacement. The **rendering** fixtures
    /// use this — those frames are lit, and a flat field would flatten the shading
    /// into something that proves much less about the meshlet raster path.
    Analytic,
}

/// The displacement amplitude [`dense_mesh`] and [`dense_grid_mesh`] use.
pub const DEFAULT_AMPLITUDE: f64 = 0.3;

/// Vertex streams for an `n × n` quad-plane spanning `x, z ∈ [-1, 1]`, displaced
/// by `amp · sin(3x) · cos(3z)`: `2n²` triangles over a surface with real
/// curvature, which is what makes the builder produce nontrivial normal cones and
/// several genuine LOD levels instead of one flat root. The outer perimeter is an
/// open boundary, so the build also exercises border locking.
///
/// All trig is bit-portable — see the module header.
pub fn displaced_grid(n: usize, amp: f64, normals: GridNormals) -> MeshStreams {
    let verts = (n + 1) * (n + 1);
    let mut positions = Vec::with_capacity(verts);
    let mut nrms = Vec::with_capacity(verts);
    let mut uvs = Vec::with_capacity(verts);
    for j in 0..=n {
        for i in 0..=n {
            // f32 divide/sub/mul are IEEE-exact, so these coordinates were already
            // portable; they are left bit-for-bit as they were.
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;

            let (fx, fz) = (x as f64 * 3.0, z as f64 * 3.0);
            let (sx, cx) = (psin64(fx), pcos64(fx));
            let (sz, cz) = (psin64(fz), pcos64(fz));
            positions.push([x, (amp * sx * cz) as f32, z]);

            nrms.push(match normals {
                GridNormals::Flat => [0.0, 1.0, 0.0],
                GridNormals::Analytic => {
                    // The surface normal of y(x,z) is (-∂y/∂x, 1, -∂y/∂z),
                    // normalized in f64 (`sqrt` is correctly rounded) and cast once.
                    let dydx = amp * 3.0 * cx * cz;
                    let dydz = -amp * 3.0 * sx * sz;
                    let (nx, ny, nz) = (-dydx, 1.0, -dydz);
                    let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
                    [(nx * inv) as f32, (ny * inv) as f32, (nz * inv) as f32]
                }
            });
            uvs.push([u, v]);
        }
    }

    let stride = (n + 1) as u32;
    let mut indices = Vec::with_capacity(n * n * 6);
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            indices.extend_from_slice(&[a, a + stride, a + 1, a + 1, a + stride, a + stride + 1]);
        }
    }
    (positions, nrms, uvs, indices)
}

/// [`displaced_grid`] run through the builder with `BuildParams::default()`.
pub fn build_grid(n: usize, amp: f64, normals: GridNormals) -> VgeomMesh {
    let (positions, nrms, uvs, indices) = displaced_grid(n, amp, normals);
    build_vgeom(&positions, &nrms, &uvs, &indices, BuildParams::default())
}

/// The **DAG** fixture: a dense displaced grid with flat normals → a multi-level
/// meshlet DAG (mirrors the cook input the `vgeom-demo` sample uses).
///
/// `n` is the grid resolution: `n = 24` gives a DAG several levels deep with roots
/// at more than one level, which is the interesting case the paging tests need.
pub fn dense_mesh(n: usize) -> VgeomMesh {
    build_grid(n, DEFAULT_AMPLITUDE, GridNormals::Flat)
}

/// [`dense_mesh`] with a **tunable amplitude** — the same topology, so the
/// clusterization (and therefore the meshlet and page *counts*) is unchanged,
/// while every simplification error differs.
///
/// That is what a re-import of an edited mesh looks like, and it is precisely the
/// case a count-only staleness check cannot see. Because the vertices are now
/// bit-identical on every platform, so is the count agreement between two
/// amplitudes — which is what
/// `stream::tests::reimported_content_under_one_id_resets_residency` rests on.
pub fn displaced_mesh(n: usize, amp: f64) -> VgeomMesh {
    build_grid(n, amp, GridNormals::Flat)
}

/// The **rendering** fixture: [`dense_mesh`]'s geometry carrying analytic normals,
/// so a lit frame shades the curvature. This is the mesh behind the `vgeom_dense`
/// / `vgeom_far` goldens, `inf-render`'s occlusion and streaming GPU gates, its
/// frame-budget gate, and the player's activation gate.
pub fn dense_grid_mesh(n: usize) -> VgeomMesh {
    build_grid(n, DEFAULT_AMPLITUDE, GridNormals::Analytic)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two normal conventions are the **same geometry**: same positions, same
    /// indices, therefore the same DAG. Stated as a test because every "the render
    /// fixture pages like the paging fixture" remark in the suites depends on it.
    #[test]
    fn the_two_normal_variants_are_the_same_geometry() {
        let (pa, na, ua, ia) = displaced_grid(12, DEFAULT_AMPLITUDE, GridNormals::Flat);
        let (pb, nb, ub, ib) = displaced_grid(12, DEFAULT_AMPLITUDE, GridNormals::Analytic);
        assert_eq!(pa, pb);
        assert_eq!(ua, ub);
        assert_eq!(ia, ib);
        assert_ne!(na, nb, "the analytic field must not be the flat one");
    }

    /// The displacement is a pure function of IEEE arithmetic — no `std` trig
    /// anywhere on the path from `(i, j)` to a committed vertex. Pinned by exact
    /// bits against the same portable expression written out longhand, which is
    /// the assertion that would have caught the macOS divergence at its source
    /// rather than in three suites downstream. (`psin64`/`pcos64` are themselves
    /// bit-pinned in `inf_math::portable`.)
    #[test]
    fn the_displacement_is_bit_pinned() {
        let (p, n, _, _) = displaced_grid(4, DEFAULT_AMPLITUDE, GridNormals::Analytic);
        // Vertex (i = 1, j = 2) of a 4×4 grid: u = 0.25, v = 0.5 ⇒ x = -0.5, z = 0.
        let v = p[2 * 5 + 1];
        assert_eq!((v[0], v[2]), (-0.5, 0.0));
        let want = (DEFAULT_AMPLITUDE * psin64(-0.5 * 3.0) * pcos64(0.0)) as f32;
        assert_eq!(
            v[1].to_bits(),
            want.to_bits(),
            "the height must be the portable f64 expression, cast exactly once"
        );

        let nrm = n[2 * 5 + 1];
        let len2 = nrm.iter().map(|c| *c as f64 * *c as f64).sum::<f64>();
        assert!(
            (len2.sqrt() - 1.0).abs() < 1e-6,
            "analytic normals are unit length"
        );
    }
}
