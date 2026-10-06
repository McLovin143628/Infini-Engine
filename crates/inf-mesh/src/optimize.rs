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

/// **One coarser rung of an index buffer over the SAME vertex buffer** (wave
/// PERF1, clause 3 -- the skinned LOD reader).
///
/// A rung re-indexes the vertices it was simplified from and moves none of
/// them, so every attribute a vertex carries -- a skinned vertex's joints and
/// weights, its uv -- is the original's, and the GPU buffer that holds them is
/// shared by every rung. `ranges` maps each input range (a submesh, a material
/// section) to its own sub-range of `indices`, in input order, so a draw that
/// addressed range `k` of the full buffer addresses range `k` of the rung.
///
/// # The meshopt law, and why this is on the right side of it
///
/// This module is import-side by contract because `meshopt` output is not
/// cross-platform (P18). [`index_lods`] is the one function here a host calls
/// at LOAD, every session -- and it is legal for the reason the law gives
/// rather than despite it: the rungs are **render-only**. They are never
/// persisted, never hashed into a world digest, never compared between two
/// hosts or two machines, and never read by the simulation; what a rung moves
/// is which triangles rasterize beyond the distance at which its error is
/// under [`INDEX_LOD_PIXEL_ERROR`] of a pixel. Below [`INDEX_LOD_MIN_TRIANGLES`]
/// no rung is built at all, which keeps every committed golden (the largest
/// skinned golden mesh is a few hundred triangles) on its full buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexLod {
    /// The rung's triangles, concatenated range by range.
    pub indices: Vec<u32>,
    /// Per input range, `(first index, index count)` within [`Self::indices`].
    pub ranges: Vec<(u32, u32)>,
    /// The simplifier's geometric error, in the mesh's own units (metres):
    /// `meshopt`'s relative error times `meshopt::simplify_scale` over the
    /// whole vertex buffer, maximised over the ranges.
    pub error_m: f32,
}

/// The triangle ratios of the rungs [`index_lods`] tries, finest first. A
/// rung that does not cut at least a fifth of the previous one's triangles
/// ends the ladder (the simplifier has run out of room).
///
/// Two rungs, a quarter and a twelfth -- UE's own ladder for the island's body
/// is 95 330 / 21 040 / 7 996 triangles, the same two steps. A third at 2.5 %
/// was built and measured (`perf1_budget_gate`'s pop arm): at its switch
/// distance, ~80 px tall, it moved 2 % of the body's INTERIOR pixels by more
/// than 8 / 255 -- a shading pop the half-pixel geometric bound does not see
/// -- so it is not built.
pub const INDEX_LOD_RATIOS: [f32; 2] = [0.25, 0.08];

/// The weight a vertex normal carries in the rung simplifier's error, against
/// a position error measured relative to the mesh's own extent.
pub const INDEX_LOD_NORMAL_WEIGHT: f32 = 0.5;

/// Meshes below this many triangles get no rung: their full buffer is already
/// cheap, and it keeps every committed skinned golden on its full buffer.
pub const INDEX_LOD_MIN_TRIANGLES: usize = 8192;

/// **The pop bound a rung is selected under**, in pixels of the frame it is
/// drawn into: a rung is drawn only where its [`IndexLod::error_m`], projected
/// at the instance's nearest distance, is under this. Half a pixel -- a
/// silhouette that moves by less than the pixel grid's own half-step.
pub const INDEX_LOD_PIXEL_ERROR: f32 = 0.5;

/// **The rungs of an index buffer** over `positions` -- see [`IndexLod`].
///
/// `ranges` are `(first index, index count)` into `indices` (a skinned mesh's
/// submeshes, concatenated); each range is simplified on its own, so a rung's
/// ranges line up with the full buffer's one for one. Empty (no rung) below
/// [`INDEX_LOD_MIN_TRIANGLES`], on a range that runs past the buffer, or on an
/// index past the vertex buffer -- the last is the FFI guard [`optimize`]
/// documents: the C writes through whatever it is handed.
pub fn index_lods(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
    ranges: &[(u32, u32)],
) -> Vec<IndexLod> {
    let tris = indices.len() / 3;
    if tris < INDEX_LOD_MIN_TRIANGLES || positions.is_empty() || normals.len() != positions.len() {
        return Vec::new();
    }
    let n = positions.len() as u32;
    if indices.iter().any(|&i| i >= n) {
        return Vec::new();
    }
    if ranges
        .iter()
        .any(|&(f, c)| (f as usize).saturating_add(c as usize) > indices.len() || c % 3 != 0)
    {
        return Vec::new();
    }
    let bytes: &[u8] = bytemuck::cast_slice(positions);
    let Ok(adapter) = VertexDataAdapter::new(bytes, 12, 0) else {
        return Vec::new();
    };
    let scale = meshopt::simplify_scale(&adapter);
    // The NORMALS ride as a weighted attribute: a geometric half-pixel says
    // nothing about shading, and a rung that bends the normals across a big
    // triangle changes the lit pixels inside a silhouette that has not moved
    // (measured on the PERF1 gate's body: the position-only coarsest rung moved
    // 8.5 % of the body's pixels by more than 8 / 255 at its switch distance).
    // The error meshopt returns then folds the normal deviation in, so a rung
    // that would bend the shading is selected farther away.
    let attributes: &[f32] = bytemuck::cast_slice(normals);
    let weights = [INDEX_LOD_NORMAL_WEIGHT; 3];
    // Every vertex free: the C reads the lock array whole, so it is sized to
    // the vertex buffer rather than left empty.
    let unlocked = vec![false; positions.len()];
    let mut lods: Vec<IndexLod> = Vec::new();
    let mut prev = tris;
    for ratio in INDEX_LOD_RATIOS {
        let mut out: Vec<u32> = Vec::new();
        let mut rr: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        let mut error = 0.0f32;
        for &(f, c) in ranges {
            let sub = &indices[f as usize..(f + c) as usize];
            let sub_tris = sub.len() / 3;
            let target = ((sub_tris as f32 * ratio).round() as usize).max(1);
            let mut e = 0.0f32;
            let simplified = if sub_tris <= target {
                sub.to_vec()
            } else {
                meshopt::simplify_with_attributes_and_locks(
                    sub,
                    &adapter,
                    attributes,
                    &weights,
                    12,
                    &unlocked,
                    target * 3,
                    1.0,
                    SimplifyOptions::None,
                    Some(&mut e),
                )
            };
            rr.push((out.len() as u32, simplified.len() as u32));
            out.extend_from_slice(&simplified);
            error = error.max(e);
        }
        let got = out.len() / 3;
        if got * 5 > prev * 4 {
            break;
        }
        prev = got;
        lods.push(IndexLod {
            indices: out,
            ranges: rr,
            error_m: error * scale,
        });
    }
    lods
}

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
    // **The partial triangle**, which is the same class one step further in.
    // `meshopt_optimizeVertexCacheTable` floors `index_count / 3` and then
    // enters its emit loop over a compiled-out `assert(output_triangle <
    // face_count)`, reading `indices[current_triangle * 3 + 2]` — with two
    // indices that is one `u32` past the end of the input AND past the end of
    // the destination `optimize_vertex_cache` allocated at `indices.len()`. A
    // glTF primitive with a count-2 index accessor produces exactly that.
    // See `crate::validate::reject_partial_triangle` for the enumeration.
    if !indices.len().is_multiple_of(3) {
        debug_assert!(
            false,
            "optimize() was handed a partial triangle; the import door \
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

    /// A closed, smooth surface (a UV sphere) with `n` rings -- the shape a
    /// body is closest to among the shapes a test can afford.
    fn sphere(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut p = Vec::new();
        let mut i = Vec::new();
        for r in 0..=n {
            let th = std::f32::consts::PI * r as f32 / n as f32;
            for s in 0..=2 * n {
                let ph = std::f32::consts::PI * s as f32 / n as f32;
                p.push([th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]);
            }
        }
        let w = 2 * n + 1;
        for r in 0..n {
            for s in 0..2 * n {
                let a = r * w + s;
                i.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
            }
        }
        (p, i)
    }

    /// **The rungs keep the ranges and the vertices, and get coarser** (wave
    /// PERF1). Two ranges (two submeshes of one buffer): every rung has two
    /// ranges, each indexes only vertices its full range used, the triangle
    /// count falls rung by rung, the error rises, and a mesh under the floor
    /// gets no rung at all.
    #[test]
    fn index_lods_keep_their_ranges_and_get_coarser() {
        let (p, i) = sphere(64);
        let tris = i.len() / 3;
        assert!(tris >= INDEX_LOD_MIN_TRIANGLES);
        let half = ((i.len() / 2) / 3 * 3) as u32;
        let ranges = [(0, half), (half, i.len() as u32 - half)];
        let lods = index_lods(&p, &p, &i, &ranges);
        assert!(
            lods.len() >= 2,
            "only {} rungs from {tris} triangles",
            lods.len()
        );
        let mut prev_tris = tris;
        let mut prev_err = 0.0f32;
        for (k, lod) in lods.iter().enumerate() {
            assert_eq!(lod.ranges.len(), 2, "rung {k} lost a range");
            let t = lod.indices.len() / 3;
            println!(
                "PERF1 index lod {k}: {t} tris (from {tris}), error {:.5} m",
                lod.error_m
            );
            assert!(
                t * 5 <= prev_tris * 4,
                "rung {k} did not get coarser: {t} vs {prev_tris}"
            );
            assert!(
                lod.error_m >= prev_err && lod.error_m > 0.0,
                "rung {k} error {}",
                lod.error_m
            );
            for (r, &(f, c)) in lod.ranges.iter().enumerate() {
                let full: std::collections::BTreeSet<u32> = i
                    [ranges[r].0 as usize..(ranges[r].0 + ranges[r].1) as usize]
                    .iter()
                    .copied()
                    .collect();
                for v in &lod.indices[f as usize..(f + c) as usize] {
                    assert!(
                        full.contains(v),
                        "rung {k} range {r} uses a vertex its full range never did"
                    );
                }
            }
            prev_tris = t;
            prev_err = lod.error_m;
        }
        let (sp, si) = sphere(16);
        assert!(
            index_lods(&sp, &sp, &si, &[(0, si.len() as u32)]).is_empty(),
            "a small mesh got a rung"
        );
        // The FFI guard: an index past the vertex buffer builds nothing.
        let mut bad = i.clone();
        bad[0] = p.len() as u32;
        assert!(index_lods(&p, &p, &bad, &[(0, bad.len() as u32)]).is_empty());
        assert!(index_lods(&p, &p[1..], &i, &[(0, i.len() as u32)]).is_empty());
    }

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
