//! Shared fixtures for this crate's unit tests.
//!
//! Test-only (`#[cfg(test)]`), so it never reaches a shipped build — and the
//! builder it calls is host-only anyway (`meshopt`'s C++ is not in the wasm
//! player).

use crate::build::{build_vgeom, BuildParams};
use crate::model::VgeomMesh;

/// A dense displaced grid → a multi-level meshlet DAG (mirrors the cook input the
/// `vgeom-demo` sample uses). `n` is the grid resolution: `n = 24` gives a DAG
/// several levels deep with roots at more than one level, which is what the
/// paging tests need to be exercising the interesting case.
pub fn dense_mesh(n: usize) -> VgeomMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;
            let y = 0.3 * (x * 3.0).sin() * (z * 3.0).cos();
            positions.push([x, y, z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([u, v]);
        }
    }
    let stride = (n + 1) as u32;
    let mut indices = Vec::new();
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            indices.extend_from_slice(&[a, a + stride, a + 1, a + 1, a + stride, a + stride + 1]);
        }
    }
    build_vgeom(&positions, &normals, &uvs, &indices, BuildParams::default())
}
