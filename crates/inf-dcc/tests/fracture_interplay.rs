//! The one place P23.3 meets an existing consumer of `MeshAsset`.
//!
//! `inf_mesh::fracture_mesh` is the cook-time Voronoi pre-fracture pipeline
//! (P22.2). It has only ever been fed assets produced by the glTF/OBJ importers,
//! so "the kernel can write a `MeshAsset`" is not proven by a round trip through
//! the kernel's own reader — that only shows the writer agrees with itself. This
//! test feeds a kernel-written asset to a stranger.
//!
//! What it would catch: a writer that emits an index buffer the fracture point
//! cloud reads differently, degenerate or inside-out geometry whose convex hull
//! has no volume, or bounds that disagree with the vertices.

use inf_asset::AssetId;
use inf_dcc::{cube, cylinder, to_mesh_asset, ExportOptions};
use inf_mesh::{fracture_mesh, FractureParams, MIN_CHUNK_COUNT};

/// A cube modelled in the kernel, written out, and broken.
#[test]
fn a_kernel_written_cube_fractures_into_sane_chunks() {
    let (asset, report) = to_mesh_asset(&cube(2.0), &ExportOptions::default());
    assert_eq!(report.triangles, 12);
    assert_eq!(report.fan_fallbacks, 0);

    let params = FractureParams {
        seed: 7,
        chunk_count: 8,
    };
    let fractured =
        fracture_mesh(&asset, AssetId::NIL, params).expect("a 8 m³ cube is fracturable");

    assert!(
        fractured.chunks.len() >= MIN_CHUNK_COUNT as usize,
        "got {} chunks",
        fractured.chunks.len()
    );
    assert_eq!(fractured.bounds, asset.bounds);

    // The pieces account for the solid: 2 m cube = 8 m³. The Voronoi cells
    // partition the hull, so the sum is the whole volume up to hull-clipping
    // tolerance — a writer emitting inverted winding or duplicated geometry
    // would not land here.
    let total: f64 = fractured.chunks.iter().map(|c| c.volume_m3).sum();
    assert!(
        (total - 8.0).abs() < 0.05,
        "chunk volumes sum to {total} m³, expected 8"
    );

    for (i, chunk) in fractured.chunks.iter().enumerate() {
        assert!(chunk.volume_m3 > 0.0, "chunk {i} has no volume");
        assert!(chunk.hull_points.len() >= 4, "chunk {i} has no solid hull");
        assert_eq!(chunk.indices.len() % 3, 0);
        assert!(chunk
            .indices
            .iter()
            .all(|&i| (i as usize) < chunk.vertices.len()));
        for &n in &chunk.neighbors {
            assert!((n as usize) < fractured.chunks.len());
            assert_ne!(n as usize, i, "a chunk is not its own neighbour");
        }
        for p in &chunk.center_of_mass {
            assert!(p.is_finite());
        }
    }
}

/// A cylinder exercises the n-gon path: its caps are 16-gons in the kernel and
/// are ear-clipped on the way out, so this is the writer's triangulation meeting
/// a consumer that only understands triangles.
#[test]
fn a_kernel_written_cylinder_with_ngon_caps_also_fractures() {
    let (asset, report) = to_mesh_asset(&cylinder(1.0, 2.0, 16), &ExportOptions::default());
    assert_eq!(report.fan_fallbacks, 0, "convex caps ear-clip cleanly");
    assert_eq!(
        report.triangles,
        16 * 2 + 14 * 2,
        "16 side quads + two 16-gons"
    );

    let fractured = fracture_mesh(
        &asset,
        AssetId::NIL,
        FractureParams {
            seed: 3,
            chunk_count: 6,
        },
    )
    .expect("a ~6.2 m³ cylinder is fracturable");
    assert!(fractured.chunks.len() >= MIN_CHUNK_COUNT as usize);
    let total: f64 = fractured.chunks.iter().map(|c| c.volume_m3).sum();
    // The fracture works on the mesh's convex HULL, and a 16-gon prism's hull is
    // the prism itself: 16 · ½ · r² · sin(2π/16) · h ≈ 6.12 m³.
    assert!(
        (total - 6.12).abs() < 0.1,
        "chunk volumes sum to {total} m³"
    );
}
