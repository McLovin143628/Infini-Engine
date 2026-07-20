//! `meshopt` post-processing for imported geometry.
//!
//! The pipeline is the standard meshoptimizer order: weld duplicate vertices,
//! then optimize for the GPU vertex cache (maximize post-transform reuse), then
//! optimize vertex fetch (reorder the vertex buffer to match index access order,
//! improving memory locality). Overdraw optimization is deliberately skipped —
//! it needs a position adapter and matters most for opaque depth-heavy scenes,
//! a later tuning pass.

use crate::asset::MeshVertex;

/// Optimize one submesh's vertex + index buffers in place-of-return. Safe on
/// empty input (returns it unchanged).
pub fn optimize(vertices: Vec<MeshVertex>, indices: Vec<u32>) -> (Vec<MeshVertex>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (vertices, indices);
    }

    // 1. Weld: find unique vertices and a remap table over the index stream.
    let (unique_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(&indices));
    let mut verts = meshopt::remap_vertex_buffer(&vertices, unique_count, &remap);
    let mut idx = meshopt::remap_index_buffer(Some(&indices), indices.len(), &remap);

    // 2. Vertex-cache optimization (reorders indices only).
    idx = meshopt::optimize_vertex_cache(&idx, verts.len());

    // 3. Vertex-fetch optimization (reorders vertices, rewrites indices to match).
    verts = meshopt::optimize_vertex_fetch(&mut idx, &verts);

    (verts, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welds_duplicate_vertices() {
        // A quad authored as two triangles with duplicated corner vertices.
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        let verts = vec![
            v(0.0, 0.0),
            v(1.0, 0.0),
            v(1.0, 1.0), // tri 1
            v(0.0, 0.0),
            v(1.0, 1.0),
            v(0.0, 1.0), // tri 2 (2 dupes)
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let (out_v, out_i) = optimize(verts, indices);
        assert_eq!(out_v.len(), 4, "6 verts weld to 4 unique corners");
        assert_eq!(out_i.len(), 6, "still two triangles");
    }

    #[test]
    fn empty_is_passthrough() {
        let (v, i) = optimize(vec![], vec![]);
        assert!(v.is_empty() && i.is_empty());
    }
}
