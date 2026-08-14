//! `meshopt` post-processing for imported geometry.
//!
//! The pipeline is the standard meshoptimizer order: weld duplicate vertices,
//! then optimize for the GPU vertex cache (maximize post-transform reuse), then
//! optimize vertex fetch (reorder the vertex buffer to match index access order,
//! improving memory locality). Overdraw optimization is deliberately skipped —
//! it needs a position adapter and matters most for opaque depth-heavy scenes,
//! a later tuning pass.
//!
//! # The meshopt law, and which operations may call it
//!
//! **`meshopt` output is not cross-platform** (P18). It is a C library built
//! through `cc`, its cost heaps break ties on floating-point comparisons, and
//! two machines are not obliged to agree about the result. The consequence the
//! workspace draws from that is not "never call it" — it is a rule about *what
//! kind of operation* may:
//!
//! * **A one-shot import may.** glTF import, `.obj` import and (P25.3)
//!   photogrammetry finish all run **once**, on the author's machine, over a
//!   source the author supplies, and write a content-addressed asset that is
//!   thereafter committed and read back verbatim. Nothing re-derives it, so
//!   nothing can disagree about it.
//! * **Anything two machines re-derive may not.** That is why
//!   `inf_dcc::ExportOptions::optimize` is **off** by default and why the P23.3
//!   law keeps `meshopt` out of the op journal: a modelling session's saved
//!   bytes are replayed, and a replay that lands somewhere else is a corrupt
//!   document. `inf-vgeom`'s meshlet DAG sits on the import side of that line
//!   too — it is *derived at cook*, on one machine, and shipped.
//!
//! Every function in this module is therefore import-side by contract, and a
//! caller that is re-deriving rather than importing is calling the wrong door.
//!
//! # There is no LOD ladder in a `.inf_mesh`
//!
//! [`crate::MeshAsset`] has submeshes and no levels. The engine's LOD ladder is
//! `inf-vgeom`'s **meshlet DAG**, derived at cook from a mesh whose triangle
//! count clears `VgeomCookOptions::min_triangles` (2048). So "give me an
//! LOD-ready mesh" means one thing here — [`simplify`] to a budget the cook will
//! still virtualize — and the ladder itself is built downstream by somebody
//! else.

use crate::asset::MeshVertex;
use meshopt::{SimplifyOptions, VertexDataAdapter};

/// Optimize one submesh's vertex + index buffers in place-of-return. Safe on
/// empty input (returns it unchanged).
///
/// # It will not hand an out-of-range index to the FFI
///
/// `meshopt::generate_vertex_remap` is a raw `unsafe` call into a C library
/// with no Rust-side validation: it sizes its remap table from `vertices.len()`
/// and the C writes `remap[index]`, with the `assert` that would have caught an
/// overrun compiled out under `-DNDEBUG`. One index past the end of the vertex
/// buffer is therefore an out-of-bounds heap **write**, not a panic — and the
/// glTF importer used to collect its index accessor with `into_u32().collect()`
/// and pass it straight here (C4-1).
///
/// The refusal that matters is at the import door
/// ([`crate::validate::reject_out_of_range`]), where it can name the file and
/// the attribute. This is the check that stands between that door and the
/// `unsafe` call for every *other* caller — the DCC exporter, photogrammetry
/// finish — which build their own index buffers. Returning the input untouched
/// is the only honest answer available at a signature with no error channel: an
/// unoptimized mesh is a mesh, and the alternative is corrupting the allocator.
pub fn optimize(vertices: Vec<MeshVertex>, indices: Vec<u32>) -> (Vec<MeshVertex>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (vertices, indices);
    }
    if indices.iter().any(|&i| i as usize >= vertices.len()) {
        debug_assert!(
            false,
            "optimize() was handed an index outside its vertex buffer; the import door \
             (inf_mesh::validate) is supposed to have refused this file already"
        );
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

/// What one decimation did.
#[derive(Debug, Clone, PartialEq)]
pub struct Simplified {
    /// The new index buffer. It references the **original** vertex buffer —
    /// meshopt does not compact — so a caller who wants a tight mesh runs
    /// [`optimize`] afterwards, which welds and re-fetches in one step.
    pub indices: Vec<u32>,
    /// meshopt's own error estimate, **relative to the mesh's extent**. A
    /// simplification that halves a one-metre object and reports `0.01` moved
    /// its surface by roughly a centimetre.
    pub error: f32,
    /// Triangles asked for.
    pub target_triangles: usize,
    /// Triangles actually produced.
    ///
    /// Reported rather than enforced, because meshopt **stops when the topology
    /// stops it**: a mesh whose edge collapses would all produce non-manifold or
    /// inverted geometry cannot be reduced further, and a decimator that
    /// pretended otherwise would have to invent geometry. A caller who needs a
    /// hard budget compares this against `target_triangles` and says so.
    pub triangles: usize,
}

/// Decimate an index buffer toward a triangle budget.
///
/// Positions are read from `vertices` at offset 0 — the layout
/// [`MeshVertex`]'s `#[repr(C)]` fixes — so no copy is made.
///
/// # The error target
///
/// meshopt takes both a target count and a target error and stops at whichever
/// it reaches first. This passes a **large** relative error (`1.0`) so the
/// **count** is what governs, which is the same choice `inf_vgeom`'s meshlet
/// simplifier makes and for the same reason: a budget the caller can plan
/// against beats a quality knob nobody can convert into one. The error actually
/// incurred comes back in [`Simplified::error`].
///
/// Empty or sub-triangle input comes back unchanged with a zero error rather
/// than refusing — an empty mesh is already inside every budget.
pub fn simplify(vertices: &[MeshVertex], indices: &[u32], target_triangles: usize) -> Simplified {
    let target_triangles = target_triangles.max(1);
    if vertices.is_empty() || indices.len() < 3 {
        return Simplified {
            indices: indices.to_vec(),
            error: 0.0,
            target_triangles,
            triangles: indices.len() / 3,
        };
    }
    if indices.len() / 3 <= target_triangles {
        return Simplified {
            indices: indices.to_vec(),
            error: 0.0,
            target_triangles,
            triangles: indices.len() / 3,
        };
    }
    let bytes: &[u8] = bytemuck::cast_slice(vertices);
    let stride = std::mem::size_of::<MeshVertex>();
    let adapter = VertexDataAdapter::new(bytes, stride, 0)
        .expect("MeshVertex is repr(C) with position at offset 0 and a stride that divides it");
    let mut error = 0.0f32;
    let out = meshopt::simplify(
        indices,
        &adapter,
        target_triangles * 3,
        // Relative, and deliberately unreachable, so the count is the binding
        // constraint. See the doc comment.
        1.0,
        SimplifyOptions::None,
        Some(&mut error),
    );
    Simplified {
        triangles: out.len() / 3,
        indices: out,
        error,
        target_triangles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tessellated grid, `n` quads on a side, as a triangle soup with no
    /// shared vertices — the shape a decimator has real work to do on.
    fn grid(n: u32) -> (Vec<MeshVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(MeshVertex {
                    position: [i as f32 / n as f32, j as f32 / n as f32, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [i as f32 / n as f32, j as f32 / n as f32],
                    tangent: crate::TANGENT_PLACEHOLDER,
                });
            }
        }
        let idx = |i: u32, j: u32| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        (vertices, indices)
    }

    #[test]
    fn simplify_reaches_a_budget_and_reports_what_it_reached() {
        let (v, i) = grid(24);
        assert_eq!(i.len() / 3, 1152);
        let out = simplify(&v, &i, 200);
        assert_eq!(out.target_triangles, 200);
        assert_eq!(out.triangles, out.indices.len() / 3);
        assert!(
            out.triangles <= 200,
            "asked for 200 triangles, got {}",
            out.triangles
        );
        assert!(
            out.triangles > 1,
            "the decimator collapsed the mesh to {} triangles",
            out.triangles
        );
        // A planar grid decimates almost for free; the error must still be a
        // real number rather than a NaN that every bound accepts.
        assert!(out.error.is_finite(), "error is {}", out.error);
        // Every index still addresses the original buffer.
        assert!(out.indices.iter().all(|&x| (x as usize) < v.len()));
    }

    #[test]
    fn a_mesh_already_inside_the_budget_is_returned_untouched() {
        let (v, i) = grid(4);
        let out = simplify(&v, &i, 10_000);
        assert_eq!(out.indices, i, "an in-budget mesh was decimated anyway");
        assert_eq!(out.error, 0.0);
        assert_eq!(out.triangles, i.len() / 3);
    }

    #[test]
    fn degenerate_input_is_a_value_not_a_panic() {
        let out = simplify(&[], &[], 100);
        assert!(out.indices.is_empty());
        let (v, _) = grid(2);
        let out = simplify(&v, &[0, 1], 100);
        assert_eq!(out.indices, vec![0, 1], "a partial triangle was rewritten");
        // A zero budget is clamped to one rather than dividing by nothing.
        let (v, i) = grid(8);
        let out = simplify(&v, &i, 0);
        assert_eq!(out.target_triangles, 1);
        assert!(out.triangles >= 1);
    }

    #[test]
    fn decimation_then_optimize_compacts_the_vertex_buffer() {
        // The documented pairing: `simplify` leaves the vertex buffer alone, so
        // the vertices its output no longer references are still there until
        // `optimize` welds and re-fetches. A caller who skips that ships a mesh
        // whose buffer is mostly dead weight.
        let (v, i) = grid(24);
        let out = simplify(&v, &i, 150);
        let (verts, idx) = optimize(v.clone(), out.indices.clone());
        assert!(
            verts.len() < v.len(),
            "optimize kept all {} vertices for {} triangles",
            verts.len(),
            idx.len() / 3
        );
        assert_eq!(idx.len(), out.indices.len(), "triangles went missing");
        assert!(idx.iter().all(|&x| (x as usize) < verts.len()));
    }

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
