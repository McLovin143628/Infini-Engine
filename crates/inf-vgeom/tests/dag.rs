//! Behavioral tests for the meshlet DAG builder (P13.1a deliverable 3).
//!
//! The load-bearing test is the **cut invariant**: for any LOD-selection
//! threshold, the selected meshlet set forms a *complete, non-overlapping,
//! crack-free* surface. We prove this via edge bookkeeping — a watertight surface
//! has every interior edge shared by exactly two triangles and every boundary
//! edge (the mesh's outer rim) used exactly once — which is precisely what the
//! group boundary-locking guarantees across levels. A crack (inconsistent
//! collapse across a group boundary) would expose an interior edge as
//! single-used; a hole (missing region) would drop boundary edges. Neither may
//! happen at *any* threshold.

use std::collections::BTreeMap;

use inf_vgeom::{build_vgeom, BuildParams, Meshlet, VgeomMesh};

/// `(positions, normals, uvs, indices)` — the raw streams a generator yields.
type MeshStreams = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>);

// ─────────────────────────── mesh generators ───────────────────────────────

/// A wavy `n × n` vertex grid: `2·(n-1)²` triangles, a curved (non-planar)
/// surface so simplification incurs real error and builds a multi-level DAG. Its
/// outer perimeter is an open boundary (exercises boundary locking).
fn wavy_grid(n: usize) -> MeshStreams {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let fx = x as f32;
            let fz = z as f32;
            let y = 0.6 * (fx * 0.5).sin() * (fz * 0.5).cos();
            positions.push([fx, y, fz]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([fx / (n as f32), fz / (n as f32)]);
        }
    }
    let mut indices = Vec::new();
    let idx = |x: usize, z: usize| (z * n + x) as u32;
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            let v00 = idx(x, z);
            let v10 = idx(x + 1, z);
            let v01 = idx(x, z + 1);
            let v11 = idx(x + 1, z + 1);
            indices.extend_from_slice(&[v00, v10, v11]);
            indices.extend_from_slice(&[v00, v11, v01]);
        }
    }
    (positions, normals, uvs, indices)
}

/// A unit cube (12 triangles) — a tiny mesh that should stay a single meshlet.
fn cube() -> MeshStreams {
    let p = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let positions: Vec<[f32; 3]> = p.to_vec();
    let normals = vec![[0.0, 1.0, 0.0]; 8];
    let uvs = vec![[0.0, 0.0]; 8];
    #[rustfmt::skip]
    let indices = vec![
        0,1,2, 0,2,3, // back
        4,6,5, 4,7,6, // front
        0,4,5, 0,5,1, // bottom
        3,2,6, 3,6,7, // top
        0,3,7, 0,7,4, // left
        1,5,6, 1,6,2, // right
    ];
    (positions, normals, uvs, indices)
}

// ─────────────────────────── edge helpers ──────────────────────────────────

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Undirected edge → incident-triangle count over a set of meshlets.
fn edge_counts(
    mesh: &VgeomMesh,
    meshlets: impl Iterator<Item = usize>,
) -> BTreeMap<(u32, u32), u32> {
    let mut counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for mi in meshlets {
        let m = &mesh.meshlets[mi];
        for t in 0..m.triangle_count as usize {
            let [a, b, c] = mesh.triangle(mi, t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *counts.entry(edge_key(u, v)).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// The set of single-used edges (the outer boundary) of an edge-count map.
fn boundary_of(counts: &BTreeMap<(u32, u32), u32>) -> Vec<(u32, u32)> {
    counts
        .iter()
        .filter(|(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .collect()
}

/// Assert the given meshlet set is a watertight, complete surface whose outer
/// boundary equals `expected_boundary`: every edge is used once (only on the
/// boundary) or twice (interior manifold), with no single-used interior edge
/// (a crack) and no missing boundary edge (a hole).
fn assert_watertight(mesh: &VgeomMesh, meshlets: &[usize], expected_boundary: &[(u32, u32)]) {
    let counts = edge_counts(mesh, meshlets.iter().copied());
    for (&e, &c) in &counts {
        assert!(
            c == 1 || c == 2,
            "edge {e:?} used {c} times — non-manifold selection"
        );
    }
    assert_eq!(
        boundary_of(&counts),
        expected_boundary,
        "selected surface boundary differs from the mesh rim (crack or hole)"
    );
}

fn selected_indices(mesh: &VgeomMesh, threshold: f32) -> Vec<usize> {
    mesh.select(threshold).map(|(i, _)| i).collect()
}

fn lod0_indices(mesh: &VgeomMesh) -> Vec<usize> {
    mesh.meshlets
        .iter()
        .enumerate()
        .filter(|(_, m)| m.lod_level == 0)
        .map(|(i, _)| i)
        .collect()
}

/// Triangle counts per LOD level, indexed by level number (0 = finest).
fn triangles_per_level(mesh: &VgeomMesh) -> BTreeMap<u8, usize> {
    let mut per: BTreeMap<u8, usize> = BTreeMap::new();
    for m in &mesh.meshlets {
        *per.entry(m.lod_level).or_insert(0) += m.triangle_count as usize;
    }
    per
}

// ─────────────────────────── tests ─────────────────────────────────────────

#[test]
fn builds_a_multi_level_dag() {
    let (p, n, uv, idx) = wavy_grid(100);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    assert!(mesh.level_count() >= 3, "levels: {}", mesh.level_count());
    // LOD 0 keeps the full input triangle count (welding merges only vertices).
    let per = triangles_per_level(&mesh);
    assert_eq!(per[&0], idx.len() / 3, "LOD 0 keeps every input triangle");
    // Many meshlets at the finest level; fewer at the coarsest.
    let finest = mesh.levels.last().unwrap();
    let coarsest = mesh.levels.first().unwrap();
    assert!(finest.lod_level == 0, "levels are laid out coarsest-first");
    assert!(coarsest.lod_level as usize == mesh.level_count() - 1);
    assert!(finest.meshlet_count > coarsest.meshlet_count);
}

#[test]
fn levels_are_laid_out_coarsest_first_for_streaming() {
    let (p, n, uv, idx) = wavy_grid(80);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    // levels[] descends in lod_level (coarse → fine) and the meshlet ranges are
    // contiguous and cover the whole meshlet array in order.
    let mut expected_start = 0u32;
    let mut last_lod = u8::MAX;
    for lr in &mesh.levels {
        assert!(lr.lod_level < last_lod, "levels descend in lod_level");
        last_lod = lr.lod_level;
        assert_eq!(lr.meshlet_start, expected_start, "levels are contiguous");
        expected_start += lr.meshlet_count;
    }
    assert_eq!(expected_start as usize, mesh.meshlet_count());
    // Every meshlet's stored lod_level matches the range it falls in.
    for lr in &mesh.levels {
        for i in lr.meshlet_start..lr.meshlet_start + lr.meshlet_count {
            assert_eq!(mesh.meshlets[i as usize].lod_level, lr.lod_level);
        }
    }
}

#[test]
fn within_a_level_triangles_do_not_overlap() {
    let (p, n, uv, idx) = wavy_grid(90);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    // For each level, no triangle (as an unordered vertex triple) appears twice —
    // build_meshlets partitions the level's index buffer.
    for lr in &mesh.levels {
        let mut seen: BTreeMap<[u32; 3], u32> = BTreeMap::new();
        let mut total = 0usize;
        for mi in lr.meshlet_start..lr.meshlet_start + lr.meshlet_count {
            let m = &mesh.meshlets[mi as usize];
            for t in 0..m.triangle_count as usize {
                let mut tri = mesh.triangle(mi as usize, t);
                tri.sort_unstable();
                *seen.entry(tri).or_insert(0) += 1;
                total += 1;
            }
        }
        assert_eq!(
            seen.len(),
            total,
            "level {} has overlapping triangles",
            lr.lod_level
        );
    }
}

#[test]
fn dag_errors_are_strictly_monotonic() {
    let (p, n, uv, idx) = wavy_grid(90);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    for m in &mesh.meshlets {
        // The half-open interval [error, parent_error) is always non-empty.
        assert!(
            m.error < m.parent_error,
            "error {} !< parent_error {}",
            m.error,
            m.parent_error
        );
        if m.is_root() {
            assert_eq!(
                m.parent_error,
                f32::INFINITY,
                "roots have +inf parent error"
            );
        } else {
            // A non-root's parent_error equals its group's error, and that group's
            // error strictly exceeds every input meshlet's error.
            let g = &mesh.groups[m.group as usize];
            assert_eq!(m.parent_error, g.error);
            assert!(m.error < g.error);
        }
    }
    // Group errors strictly increase along parent-group edges.
    for (gi, g) in mesh.groups.iter().enumerate() {
        for &pg in &g.parents {
            assert!(
                g.error < mesh.groups[pg as usize].error,
                "group {gi} error {} !< parent group {pg} error {}",
                g.error,
                mesh.groups[pg as usize].error
            );
        }
    }
}

#[test]
fn cut_invariant_holds_at_every_threshold() {
    let (p, n, uv, idx) = wavy_grid(100);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    // The mesh rim: single-used edges of the finest (LOD-0) full surface.
    let lod0 = lod0_indices(&mesh);
    let lod0_counts = edge_counts(&mesh, lod0.iter().copied());
    let rim = boundary_of(&lod0_counts);
    assert!(!rim.is_empty(), "the grid has an open boundary");

    // At threshold 0 the finest meshlets are selected and reconstruct exactly the
    // full input surface.
    let at0 = selected_indices(&mesh, 0.0);
    assert_eq!(at0, lod0, "threshold 0 selects exactly the LOD-0 meshlets");
    assert_watertight(&mesh, &at0, &rim);

    // Sweep thresholds densely across the whole error range, including exact group
    // errors and their immediate neighborhood (interval endpoints are where a cut
    // could go wrong).
    let max_err = mesh
        .meshlets
        .iter()
        .map(|m| m.error)
        .filter(|e| e.is_finite())
        .fold(0.0f32, f32::max);

    let mut thresholds: Vec<f32> = Vec::new();
    for k in 0..=400 {
        thresholds.push(max_err * 1.2 * (k as f32) / 400.0);
    }
    for g in &mesh.groups {
        thresholds.push(g.error);
        thresholds.push(g.error - 1e-5);
        thresholds.push(g.error + 1e-5);
    }
    thresholds.push(max_err * 10.0); // beyond the top → the root set

    for &t in &thresholds {
        let sel = selected_indices(&mesh, t);
        assert!(!sel.is_empty(), "threshold {t} selected nothing");
        assert_watertight(&mesh, &sel, &rim);
    }
}

#[test]
fn simplification_reduces_triangle_count_per_level() {
    let (p, n, uv, idx) = wavy_grid(100);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    let per = triangles_per_level(&mesh);
    // From fine (level 0) to coarse (level max), triangle totals strictly decrease.
    let max_lod = *per.keys().max().unwrap();
    for lod in 0..max_lod {
        let fine = per[&lod];
        let coarse = per[&(lod + 1)];
        assert!(
            coarse < fine,
            "level {} ({coarse} tris) did not reduce below level {} ({fine} tris)",
            lod + 1,
            lod
        );
    }
}

#[test]
fn meshlet_bounds_contain_their_triangles() {
    let (p, n, uv, idx) = wavy_grid(80);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    for (mi, m) in mesh.meshlets.iter().enumerate() {
        for t in 0..m.triangle_count as usize {
            for v in mesh.triangle(mi, t) {
                let pos = mesh.vertices[v as usize].position;
                let d = [
                    pos[0] - m.center[0],
                    pos[1] - m.center[1],
                    pos[2] - m.center[2],
                ];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                // Small epsilon for the bounding-sphere fit's float slack.
                assert!(
                    dist <= m.radius + 1e-3 * (1.0 + m.radius),
                    "vertex dist {dist} exceeds meshlet radius {}",
                    m.radius
                );
            }
        }
    }
}

#[test]
fn build_is_deterministic_byte_identical() {
    let (p, n, uv, idx) = wavy_grid(90);
    let a = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());
    let b = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    let ba = inf_asset::encode(&a).unwrap();
    let bb = inf_asset::encode(&b).unwrap();
    assert_eq!(ba, bb, "two builds must be byte-identical");
}

#[test]
fn tiny_mesh_is_a_single_root_meshlet_with_no_dag() {
    let (p, n, uv, idx) = cube();
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());

    assert_eq!(mesh.meshlet_count(), 1, "a cube fits in one meshlet");
    assert_eq!(mesh.level_count(), 1, "no coarser levels");
    assert!(
        mesh.groups.is_empty(),
        "no groups (nothing to simplify down)"
    );
    let m = &mesh.meshlets[0];
    assert!(m.is_root(), "the lone meshlet is a root");
    assert_eq!(m.parent_error, f32::INFINITY);
    assert_eq!(m.error, 0.0);
    assert_eq!(m.triangle_count, 12, "all 12 cube triangles");
    assert_eq!(m.lod_level, 0);
}

#[test]
fn degenerate_empty_input_yields_no_meshlets() {
    let mesh = build_vgeom(&[], &[], &[], &[], BuildParams::default());
    assert_eq!(mesh.meshlet_count(), 0);
    assert_eq!(mesh.level_count(), 0);
    assert!(mesh.groups.is_empty());
}

#[test]
fn round_trips_through_the_asset_payload() {
    let (p, n, uv, idx) = wavy_grid(60);
    let mesh = build_vgeom(&p, &n, &uv, &idx, BuildParams::default());
    let bytes = inf_asset::encode(&mesh).unwrap();
    let back: VgeomMesh = inf_asset::decode(&bytes).unwrap();
    assert_eq!(mesh, back, "encode → decode round-trips");
    assert_eq!(back.schema_version, VgeomMesh::CURRENT_VERSION);
    // Sanity: the sentinel is preserved for roots.
    assert!(back.meshlets.iter().any(|m| m.group == Meshlet::NO_GROUP));
}
