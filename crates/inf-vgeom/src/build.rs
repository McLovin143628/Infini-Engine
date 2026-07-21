//! The offline meshlet DAG builder (P13.1 deliverable 2).
//!
//! [`build_vgeom`] is an **original** implementation of the well-published
//! Nanite-class build technique (no Epic code):
//!
//! 1. **Clusterize** the welded mesh into meshlets (~64 v / ~124 t) via
//!    `meshopt::build_meshlets` — these are the finest, LOD-0 meshlets.
//! 2. **Group** adjacent meshlets (~4–8) by shared-edge count (a deterministic
//!    greedy agglomeration over a sorted adjacency graph).
//! 3. **Simplify** each group's merged triangle soup ~50% with its *outer
//!    boundary vertices locked* (`meshopt::simplify_with_locks`), so adjacent
//!    groups keep matching boundaries and cracks are impossible between them.
//! 4. **Re-cluster** each simplified group into new, coarser meshlets (the next
//!    LOD level) and **link** them into the DAG (child `parent_error` == parent
//!    `error` == the group's monotone error).
//! 5. **Repeat** on the coarser level until nothing coarsens further (a level
//!    makes no progress, collapses to a single meshlet, or `max_levels` is hit);
//!    the top meshlets become roots (`parent_error = +∞`).
//!
//! ## Error metric & the cut invariant
//!
//! A group's error is `max(child errors) + max(simplify error, ε)` — a
//! *cumulative, strictly increasing* object-space error. Strictness (the `ε`
//! bump) guarantees every meshlet has `error < parent_error`, so the runtime cut
//! `error ≤ t < parent_error` never sees an empty interval. Combined with the
//! shared-boundary construction (see [`crate::model`]) this makes the cut select
//! exactly one meshlet per root-to-leaf path — a complete, non-overlapping
//! surface at every threshold.
//!
//! ## Determinism
//!
//! The build is a deterministic function of its input: `meshopt` is deterministic
//! for fixed input; grouping iterates in meshlet-index order over `BTreeMap`
//! adjacency (no `HashMap`); and the per-group simplification — the named
//! parallel hot loop (§2.5) — runs through [`inf_core::parallel_map`], whose
//! **in-order collect** makes the result independent of pool size. Two builds of
//! the same mesh are byte-identical.

use std::collections::BTreeMap;

use meshopt::{
    build_meshlets, compute_meshlet_bounds, generate_vertex_remap, optimize_vertex_cache,
    remap_index_buffer, remap_vertex_buffer, simplify_with_locks, SimplifyOptions,
    VertexDataAdapter,
};

use crate::model::{Group, LevelRange, Meshlet, VgeomMesh, VgeomVertex};

/// Bytes per [`VgeomVertex`] (position in the first 12) — the `meshopt` vertex
/// stride.
const VERTEX_STRIDE: usize = std::mem::size_of::<VgeomVertex>();

/// The minimum per-level error increment, guaranteeing strict monotonicity of the
/// DAG errors even when a simplification step incurs (near-)zero measured error.
const ERROR_EPS: f32 = 1.0e-6;

/// Tunable parameters for [`build_vgeom`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildParams {
    /// Max vertices per meshlet (`meshopt` limit: ≤ 255).
    pub max_vertices: usize,
    /// Max triangles per meshlet (`meshopt` limit: ≤ 512, divisible by 4).
    pub max_triangles: usize,
    /// Cone weight passed to `build_meshlets` (0 = pure spatial clustering).
    pub cone_weight: f32,
    /// Target group size for the greedy grouping (~4–8 meshlets per group).
    pub max_group_size: usize,
    /// Fraction of triangles to keep when simplifying a group (~0.5 = halve).
    pub target_ratio: f32,
    /// Safety cap on the number of LOD levels built.
    pub max_levels: usize,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124, // ≤ 512 and divisible by 4
            cone_weight: 0.0,
            max_group_size: 8,
            target_ratio: 0.5,
            max_levels: 16,
        }
    }
}

impl BuildParams {
    /// Clamp to `meshopt`'s hard limits (max_vertices ≤ 255; max_triangles ≤ 512
    /// and divisible by 4; group size ≥ 2; ratio in `(0, 1)`).
    fn validated(mut self) -> Self {
        self.max_vertices = self.max_vertices.clamp(3, 255);
        self.max_triangles = (self.max_triangles.clamp(4, 512) / 4) * 4;
        self.max_group_size = self.max_group_size.max(2);
        if !(self.target_ratio > 0.0 && self.target_ratio < 1.0) {
            self.target_ratio = 0.5;
        }
        self.max_levels = self.max_levels.max(1);
        self
    }
}

/// A meshlet under construction: owns its (global) vertex indices and local
/// triangle indices so it survives re-clustering of later levels.
struct BuiltMeshlet {
    verts: Vec<u32>,
    tris: Vec<u8>,
    center: [f32; 3],
    radius: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    lod_level: u8,
    error: f32,
    /// `+∞` until linked to a parent group (stays `+∞` for roots).
    parent_error: f32,
    /// Group index this meshlet feeds into, or [`Meshlet::NO_GROUP`].
    group: u32,
}

/// A group under construction; assembled into a [`Group`] at the end.
struct BuiltGroup {
    input_level: u8,
    error: f32,
    produced_level: u8,
    /// Start index of the produced meshlets *within their level vector*.
    produced_start_within: usize,
    produced_count: usize,
}

/// The per-group simplification job (one unit of the parallel hot loop).
struct GroupJob {
    /// The group's merged triangle soup as global vertex indices.
    combined_index: Vec<u32>,
    /// Per-(whole-buffer-)vertex lock flags: the group's outer boundary is locked.
    lock: Vec<bool>,
    /// Simplification target index count (`3 × target triangles`).
    target_index_count: usize,
    /// Max error among the group's input meshlets (for cumulative error).
    max_child_error: f32,
}

/// The result of one [`GroupJob`].
struct GroupResult {
    simplified_index: Vec<u32>,
    group_error: f32,
    progressed: bool,
}

/// Build a meshlet LOD DAG from a mesh's vertex streams + index buffer.
///
/// `normals` / `uvs` may be shorter than `positions` (missing entries default);
/// `indices` is a triangle list into `positions`. Returns a [`VgeomMesh`] with
/// meshlets laid out coarsest-first (streaming order).
pub fn build_vgeom(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    params: BuildParams,
) -> VgeomMesh {
    // Meshlet-DAG build span (P15.1): the heaviest single cook stage. Free at the
    // default filter; a Tracy capture shows the clusterize→group→simplify loop.
    let _span = tracing::info_span!(
        "build_vgeom",
        verts = positions.len(),
        tris = indices.len() / 3
    )
    .entered();
    let params = params.validated();

    // Interleave the raw vertex streams.
    let raw: Vec<VgeomVertex> = (0..positions.len())
        .map(|i| VgeomVertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
        })
        .collect();

    // Degenerate: nothing to cluster.
    if raw.is_empty() || indices.len() < 3 {
        let (center, radius) = bounding_sphere(&raw);
        return VgeomMesh {
            schema_version: VgeomMesh::CURRENT_VERSION,
            vertices: raw,
            meshlets: Vec::new(),
            meshlet_vertices: Vec::new(),
            meshlet_triangles: Vec::new(),
            groups: Vec::new(),
            levels: Vec::new(),
            center,
            radius,
        };
    }

    // Weld duplicate vertices, then optimize the index order for clustering.
    let (unique, remap) = generate_vertex_remap(&raw, Some(indices));
    let vertices = remap_vertex_buffer(&raw, unique, &remap);
    let index = remap_index_buffer(Some(indices), indices.len(), &remap);
    let index = optimize_vertex_cache(&index, vertices.len());

    // ── LOD 0 ───────────────────────────────────────────────────────────────
    let mut levels: Vec<Vec<BuiltMeshlet>> = vec![clusterize(&vertices, &index, &params, 0, 0.0)];
    let mut groups: Vec<BuiltGroup> = Vec::new();
    // Which group each BuiltGroup ended up at (index into `groups`).
    let mut level = 0usize;

    while level + 1 < params.max_levels {
        let count = levels[level].len();
        if count <= 1 {
            break; // a single meshlet is already a root
        }

        // Partition this level's meshlets into groups (deterministic greedy).
        let group_members = greedy_group(&levels[level], params.max_group_size);

        // Prepare per-group simplification jobs (serial, cheap).
        let jobs: Vec<GroupJob> = group_members
            .iter()
            .map(|members| {
                build_group_job(&levels[level], members, vertices.len(), params.target_ratio)
            })
            .collect();

        // The named parallel hot loop (§2.5): simplify every group. `parallel_map`
        // collects in input order, so the result is pool-size-invariant.
        let results: Vec<GroupResult> =
            inf_core::parallel_map(jobs, |job| simplify_group(&vertices, job));

        // Re-cluster progressed groups into the next level; link the DAG.
        let mut next: Vec<BuiltMeshlet> = Vec::new();
        let mut any_progress = false;

        for (members, res) in group_members.iter().zip(results) {
            if !res.progressed {
                // No coarser replacement for this region: members stay roots.
                continue;
            }
            any_progress = true;
            let new_meshlets = clusterize(
                &vertices,
                &res.simplified_index,
                &params,
                (level + 1) as u8,
                res.group_error,
            );
            let produced_start_within = next.len();
            let produced_count = new_meshlets.len();
            next.extend(new_meshlets);

            let gidx = groups.len() as u32;
            groups.push(BuiltGroup {
                input_level: level as u8,
                error: res.group_error,
                produced_level: (level + 1) as u8,
                produced_start_within,
                produced_count,
            });
            // Link the input meshlets to this group.
            for &mi in members {
                let m = &mut levels[level][mi];
                m.group = gidx;
                m.parent_error = res.group_error;
            }
        }

        if !any_progress || next.is_empty() {
            break; // nothing coarsened further; current level meshlets are roots
        }
        levels.push(next);
        level += 1;
    }

    assemble(vertices, levels, &groups)
}

/// Clusterize an index buffer into [`BuiltMeshlet`]s tagged with `lod`/`error`.
fn clusterize(
    vertices: &[VgeomVertex],
    index: &[u32],
    params: &BuildParams,
    lod: u8,
    error: f32,
) -> Vec<BuiltMeshlet> {
    if index.len() < 3 {
        return Vec::new();
    }
    let vbytes: &[u8] = bytemuck::cast_slice(vertices);
    let adapter = VertexDataAdapter::new(vbytes, VERTEX_STRIDE, 0)
        .expect("vertex adapter: stride divides buffer, offset 0 < stride");
    let ms = build_meshlets(
        index,
        &adapter,
        params.max_vertices,
        params.max_triangles,
        params.cone_weight,
    );
    (0..ms.len())
        .map(|i| {
            let m = ms.get(i);
            let bounds = compute_meshlet_bounds(m, &adapter);
            BuiltMeshlet {
                verts: m.vertices.to_vec(),
                tris: m.triangles.to_vec(),
                center: bounds.center,
                radius: bounds.radius,
                cone_axis: bounds.cone_axis,
                cone_cutoff: bounds.cone_cutoff,
                lod_level: lod,
                error,
                parent_error: f32::INFINITY,
                group: Meshlet::NO_GROUP,
            }
        })
        .collect()
}

/// Greedy shared-edge grouping. Returns a partition of `0..meshlets.len()` into
/// groups of up to `max_group_size`, each grown by repeatedly annexing the
/// unassigned neighbor that shares the most edges (ties broken by lowest index).
/// Deterministic: seeds and iteration are in index order over `BTreeMap`s.
fn greedy_group(meshlets: &[BuiltMeshlet], max_group_size: usize) -> Vec<Vec<usize>> {
    let n = meshlets.len();
    // adjacency[i] = { neighbor -> shared edge count }, sorted.
    let adjacency = build_adjacency(meshlets);

    let mut assigned = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for seed in 0..n {
        if assigned[seed] {
            continue;
        }
        assigned[seed] = true;
        let mut members = vec![seed];

        while members.len() < max_group_size {
            // Best unassigned neighbor of any current member.
            let mut best: Option<(u32, usize)> = None; // (shared_count, neighbor)
            for &mem in &members {
                if let Some(nbrs) = adjacency.get(&mem) {
                    for (&nbr, &cnt) in nbrs {
                        if assigned[nbr] {
                            continue;
                        }
                        // Maximize shared count, then minimize index.
                        let better = match best {
                            None => true,
                            Some((bc, bn)) => cnt > bc || (cnt == bc && nbr < bn),
                        };
                        if better {
                            best = Some((cnt, nbr));
                        }
                    }
                }
            }
            match best {
                Some((_, nbr)) => {
                    assigned[nbr] = true;
                    members.push(nbr);
                }
                None => break,
            }
        }
        members.sort_unstable();
        groups.push(members);
    }
    groups
}

/// Build meshlet adjacency keyed by shared edges (undirected global-vertex edge
/// → the meshlets that use it; a pair sharing an edge is adjacent).
fn build_adjacency(meshlets: &[BuiltMeshlet]) -> BTreeMap<usize, BTreeMap<usize, u32>> {
    // edge -> set of meshlets touching it.
    let mut edge_owners: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (mi, m) in meshlets.iter().enumerate() {
        for t in m.tris.chunks_exact(3) {
            let gv = [
                m.verts[t[0] as usize],
                m.verts[t[1] as usize],
                m.verts[t[2] as usize],
            ];
            for &(a, b) in &[(gv[0], gv[1]), (gv[1], gv[2]), (gv[2], gv[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                let owners = edge_owners.entry(key).or_default();
                if owners.last() != Some(&mi) {
                    owners.push(mi);
                }
            }
        }
    }

    let mut adjacency: BTreeMap<usize, BTreeMap<usize, u32>> = BTreeMap::new();
    for owners in edge_owners.values() {
        // Every unordered pair of distinct owners shares this edge.
        for i in 0..owners.len() {
            for j in (i + 1)..owners.len() {
                let (a, b) = (owners[i], owners[j]);
                if a == b {
                    continue;
                }
                *adjacency.entry(a).or_default().entry(b).or_insert(0) += 1;
                *adjacency.entry(b).or_default().entry(a).or_insert(0) += 1;
            }
        }
    }
    adjacency
}

/// Assemble one group's simplification job: gather its triangle soup, compute the
/// outer-boundary lock, and the simplification target.
fn build_group_job(
    meshlets: &[BuiltMeshlet],
    members: &[usize],
    vertex_count: usize,
    target_ratio: f32,
) -> GroupJob {
    let mut combined_index: Vec<u32> = Vec::new();
    let mut max_child_error = 0.0f32;
    for &mi in members {
        let m = &meshlets[mi];
        max_child_error = max_child_error.max(m.error);
        for t in m.tris.chunks_exact(3) {
            combined_index.push(m.verts[t[0] as usize]);
            combined_index.push(m.verts[t[1] as usize]);
            combined_index.push(m.verts[t[2] as usize]);
        }
    }
    let lock = boundary_lock(&combined_index, vertex_count);
    let tri_count = combined_index.len() / 3;
    let target_tri = ((tri_count as f32 * target_ratio).round() as usize).max(1);
    GroupJob {
        combined_index,
        lock,
        target_index_count: target_tri * 3,
        max_child_error,
    }
}

/// Lock the vertices on the *outer boundary* of a triangle soup: an edge used by
/// exactly one triangle is a boundary edge, and both its endpoints are locked.
/// Locking a group's boundary keeps it fixed across simplification so neighboring
/// groups (which share those vertices) stay crack-free.
fn boundary_lock(index: &[u32], vertex_count: usize) -> Vec<bool> {
    let mut edge_count: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for t in index.chunks_exact(3) {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let mut lock = vec![false; vertex_count];
    for ((a, b), c) in edge_count {
        if c == 1 {
            lock[a as usize] = true;
            lock[b as usize] = true;
        }
    }
    lock
}

/// Simplify one group with its boundary locked. Returns the simplified index
/// buffer, the cumulative (strictly monotone) group error, and whether it made
/// progress (fewer triangles than the input).
fn simplify_group(vertices: &[VgeomVertex], job: GroupJob) -> GroupResult {
    let vbytes: &[u8] = bytemuck::cast_slice(vertices);
    let adapter = VertexDataAdapter::new(vbytes, VERTEX_STRIDE, 0)
        .expect("vertex adapter: stride divides buffer, offset 0 < stride");

    let mut simplify_error = 0.0f32;
    let simplified_index = simplify_with_locks(
        &job.combined_index,
        &adapter,
        &job.lock,
        job.target_index_count,
        1.0, // large target error: reach the target count (metric is relative)
        SimplifyOptions::None,
        Some(&mut simplify_error),
    );

    let progressed =
        simplified_index.len() >= 3 && simplified_index.len() < job.combined_index.len();
    // Cumulative, strictly increasing object-space error (the ε keeps intervals
    // non-empty even when meshopt reports ~0 error).
    let group_error = job.max_child_error + simplify_error.max(ERROR_EPS);

    GroupResult {
        simplified_index,
        group_error,
        progressed,
    }
}

/// Emit the final [`VgeomMesh`]: reorder meshlets **coarsest-level first**
/// (streaming order), concatenate their micro index buffers, and remap all group
/// references to the new indices.
fn assemble(
    vertices: Vec<VgeomVertex>,
    levels: Vec<Vec<BuiltMeshlet>>,
    groups: &[BuiltGroup],
) -> VgeomMesh {
    // New base meshlet index per level, laid out coarsest (highest lod) first.
    // Within a level, creation order is preserved, so each group's produced range
    // (contiguous within its level) stays contiguous.
    let mut new_base = vec![0u32; levels.len()];
    let mut acc = 0u32;
    for lod in (0..levels.len()).rev() {
        new_base[lod] = acc;
        acc += levels[lod].len() as u32;
    }
    let total = acc as usize;

    let mut meshlets: Vec<Meshlet> = Vec::with_capacity(total);
    let mut meshlet_vertices: Vec<u32> = Vec::new();
    let mut meshlet_triangles: Vec<u8> = Vec::new();
    let mut level_ranges: Vec<LevelRange> = Vec::new();

    for lod in (0..levels.len()).rev() {
        let start = meshlets.len() as u32;
        for bm in &levels[lod] {
            let vertex_offset = meshlet_vertices.len() as u32;
            meshlet_vertices.extend_from_slice(&bm.verts);
            let triangle_offset = meshlet_triangles.len() as u32;
            meshlet_triangles.extend_from_slice(&bm.tris);
            meshlets.push(Meshlet {
                vertex_offset,
                vertex_count: bm.verts.len() as u32,
                triangle_offset,
                triangle_count: (bm.tris.len() / 3) as u32,
                center: bm.center,
                radius: bm.radius,
                cone_axis: bm.cone_axis,
                cone_cutoff: bm.cone_cutoff,
                group: bm.group,
                lod_level: bm.lod_level,
                error: bm.error,
                parent_error: bm.parent_error,
            });
        }
        level_ranges.push(LevelRange {
            lod_level: lod as u8,
            meshlet_start: start,
            meshlet_count: levels[lod].len() as u32,
        });
    }

    // Finalize groups: produced range into final indices + parent-group edges.
    let out_groups: Vec<Group> = groups
        .iter()
        .map(|g| {
            let produced_start =
                new_base[g.produced_level as usize] + g.produced_start_within as u32;
            let mut parents: Vec<u32> = Vec::new();
            for k in 0..g.produced_count {
                let bm = &levels[g.produced_level as usize][g.produced_start_within + k];
                if bm.group != Meshlet::NO_GROUP {
                    parents.push(bm.group);
                }
            }
            parents.sort_unstable();
            parents.dedup();
            Group {
                input_level: g.input_level,
                error: g.error,
                produced_start,
                produced_count: g.produced_count as u32,
                parents,
            }
        })
        .collect();

    let (center, radius) = bounding_sphere(&vertices);

    VgeomMesh {
        schema_version: VgeomMesh::CURRENT_VERSION,
        vertices,
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        groups: out_groups,
        levels: level_ranges,
        center,
        radius,
    }
}

/// A bounding sphere (AABB center + farthest-corner radius) over vertex positions.
fn bounding_sphere(vertices: &[VgeomVertex]) -> ([f32; 3], f32) {
    if vertices.is_empty() {
        return ([0.0; 3], 0.0);
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for k in 0..3 {
            min[k] = min[k].min(v.position[k]);
            max[k] = max[k].max(v.position[k]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let mut r2 = 0.0f32;
    for v in vertices {
        let d = [
            v.position[0] - center[0],
            v.position[1] - center[1],
            v.position[2] - center[2],
        ];
        r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
    (center, r2.sqrt())
}
