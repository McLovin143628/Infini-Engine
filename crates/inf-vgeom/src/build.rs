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
//! A group's error is `max(child errors) + simplify error`, forced **strictly
//! greater** than every child error (see [`monotone_group_error`]: when the
//! increment is too small to change the `f32` — a near-zero measured error, or
//! any increment absorbed by rounding at a coarse magnitude — it steps to the
//! next representable float instead). This *cumulative, strictly increasing*
//! object-space error guarantees every meshlet has `error < parent_error`, so
//! the runtime cut `error ≤ t < parent_error` never sees an empty interval.
//! Combined with the shared-boundary construction (see [`crate::model`]) this
//! makes the cut select exactly one meshlet per root-to-leaf path — a complete,
//! non-overlapping surface at every threshold.
//!
//! ## Why locking the seam vertices is not enough (the seam check)
//!
//! Step 3's boundary locking is what makes neighbouring groups agree, and it is
//! *almost* the whole argument — but `meshopt` (like any QEM simplifier) locks
//! **vertices**, not **edges**, and that gap is a real hole the invariant fell
//! through. A locked vertex cannot move, yet an *unlocked interior* vertex `u` is
//! free to collapse **into** a locked seam vertex `s`, and every triangle
//! `(u, s', x)` it belonged to becomes `(s, s', x)` — **inventing** an edge
//! `(s, s')` between two seam vertices that the group never had. The neighbouring
//! group shares exactly those seam vertices, so it can invent the very same edge
//! (or already own it as an interior edge of its own region), and the union of the
//! two coarse patches is then non-manifold: one edge carrying **four** triangles,
//! sometimes the same triangle twice with opposite winding. The per-group border
//! stays intact throughout, so it is not a crack — it is an *overlap*, and the cut
//! invariant forbids both.
//!
//! Nothing about that is input-specific, which is why it stayed hidden: it needs a
//! clusterization that hands one group an interior vertex adjacent to two seam
//! vertices whose chord the neighbour also draws. `meshopt`'s x86_64 clusterizer
//! happened not to produce one for the default parameters on the P13.1 fixture;
//! its arm64 clusterizer did, and so does x86_64 at a dozen other parameter
//! settings (`tests/dag.rs::cut_invariant_holds_under_many_clusterizations`).
//!
//! The fix is [`seam_safe`]: after simplifying a group, its output is *accepted*
//! only if it is manifold, its border is preserved exactly, and no edge between
//! two seam vertices is used more often than the group's own input used it. Note
//! what that does **not** say — inventing a chord is not banned outright. It
//! cannot be: a coarse patch has to retriangulate its interior, and a build that
//! refuses every invented chord flattens the whole DAG to a single level (measured:
//! every fixture drops to `levels = 1`). What is banned is inventing an edge
//! *somebody else may also draw* — one that already exists anywhere in the mesh at
//! any level (tracked in `seen_edges`), or one a sibling group invents in the same
//! round (resolved in the level loop, the only place both outputs are visible).
//!
//! Since the group input soups partition the level, every seam edge then stays at
//! its original ≤ 2 uses and an invented edge is claimed once for the whole build,
//! so the union is manifold **for any clusterization**. A group that fails simply
//! does not coarsen (its meshlets stay roots), exactly as if `meshopt` had made no
//! progress — the DAG loses a little depth in that region and stays correct
//! everywhere. Measured on the fixtures: the default-parameter builds every other
//! suite depends on are **unchanged**, and the pathological parameter settings that
//! used to break lose one group each.
//!
//! ## Determinism
//!
//! The build is a deterministic function of its input: `meshopt` is deterministic
//! for fixed input; grouping iterates in meshlet-index order over `BTreeMap`
//! adjacency (no `HashMap`); and the per-group simplification — the named
//! parallel hot loop (§2.5) — runs through [`inf_core::parallel_map`], whose
//! **in-order collect** makes the result independent of pool size. Two builds of
//! the same mesh are byte-identical.

use std::collections::{BTreeMap, BTreeSet};

use meshopt::{
    build_meshlets, compute_meshlet_bounds, generate_vertex_remap, optimize_vertex_cache,
    remap_index_buffer, remap_vertex_buffer, simplify_with_locks, SimplifyOptions,
    VertexDataAdapter,
};

use crate::model::{Group, LevelRange, Meshlet, VgeomMesh, VgeomVertex};

/// Bytes per [`VgeomVertex`] (position in the first 12) — the `meshopt` vertex
/// stride.
const VERTEX_STRIDE: usize = std::mem::size_of::<VgeomVertex>();

/// Combine a group's max child error with its measured simplify error into a
/// **strictly increasing** cumulative error. Returns `max_child_error +
/// simplify_error`, except that when the sum does not actually exceed
/// `max_child_error` — the increment is zero, or is absorbed by `f32` rounding at
/// a coarse magnitude (where the old absolute `1e-6` ε vanished entirely) — it
/// steps to the next representable float above `max_child_error`. This guarantees
/// `result > max_child_error`, so the DAG cut interval `[error, parent_error)` is
/// never empty even at the coarsest LOD levels.
fn monotone_group_error(max_child_error: f32, simplify_error: f32) -> f32 {
    let raw = max_child_error + simplify_error;
    if raw > max_child_error {
        raw
    } else {
        // `next_up` is stable since Rust 1.86 (workspace toolchain is newer). It
        // handles the finite, non-negative errors we produce; the guard above
        // means we only reach here when `raw == max_child_error`.
        f32::next_up(max_child_error)
    }
}

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

impl BuiltMeshlet {
    /// This meshlet's triangles as global vertex triples.
    fn global_triangles(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.tris.chunks_exact(3).map(|t| {
            [
                self.verts[t[0] as usize],
                self.verts[t[1] as usize],
                self.verts[t[2] as usize],
            ]
        })
    }

    /// This meshlet's undirected edges as global vertex pairs (with repeats).
    fn global_edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.global_triangles().flat_map(|t| {
            [
                edge_key(t[0], t[1]),
                edge_key(t[1], t[2]),
                edge_key(t[2], t[0]),
            ]
        })
    }
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
    /// Per-(whole-buffer-)vertex lock flags: this group's **seam** vertices — the
    /// ones it shares with another group of the same level, plus the ones on the
    /// level's own open rim. Doubles as the "may be shared" predicate
    /// [`seam_safe`] tests edges against.
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
    /// Whether this group actually coarsened **and** its output is safe to
    /// substitute for its input (see [`seam_safe`]). `false` leaves the group's
    /// meshlets as roots.
    progressed: bool,
    /// Brand-new seam edges this group's output introduced (empty unless
    /// `progressed`). The level loop rejects a second group claiming any of them.
    invented: Vec<(u32, u32)>,
}

/// Build a meshlet LOD DAG from a mesh's vertex streams + index buffer.
///
/// `normals` / `uvs` / `tangents` may be shorter than `positions` (missing
/// entries default, and a missing tangent is [`crate::NO_TANGENT`] rather than a
/// guess); `indices` is a triangle list into `positions`. Returns a
/// [`VgeomMesh`] with meshlets laid out coarsest-first (streaming order).
///
/// # The tangent stream does not reach the clusterizer (P28.2)
///
/// `meshopt` sees positions through a [`VertexDataAdapter`] at
/// [`VERTEX_STRIDE`], and the stride is the only thing the tangent word changes
/// about what it sees. Nothing in `build_meshlets`, `simplify_with_locks` or the
/// bounds/cone computation reads past the first twelve bytes of a record, so the
/// DAG is a function of positions and indices exactly as it was before this
/// channel existed — asserted, not assumed, by
/// `tests::the_tangent_stream_does_not_move_the_dag`, which builds the same
/// geometry with two different tangent streams and compares every meshlet.
/// (The P18 law: `meshopt` output is not cross-platform, so a claim about it is
/// only ever checked by mutating a clone and comparing *within* one run.)
pub fn build_vgeom(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    tangents: &[[f32; 4]],
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
            tangent: tangents.get(i).map_or(crate::model::NO_TANGENT, |t| {
                crate::model::pack_tangent([t[0], t[1], t[2]], t[3])
            }),
        })
        .collect();

    // **The second entrance to the import door** (round-2 finding B2).
    //
    // `generate_vertex_remap` below is a raw `unsafe` call into a C library
    // that sizes its remap table from `vertices.len()` and writes
    // `remap[index]` with the `assert` compiled out under `-DNDEBUG`, so one
    // index past the end is an out-of-bounds heap **write**. `inf_mesh`'s
    // `optimize()` grew this backstop in Hardening Wave B; this door did not,
    // and it is reached from exactly the same place — a `.inf_mesh` decoded
    // off disk, by `inf_editor_core::assets::vmesh` and by the cook — where
    // the importer's validator (`inf_mesh::validate`) never ran.
    //
    // `MeshAsset::migrate` now refuses such a payload at the decode, which is
    // where it can name the asset. This is the check that stands between that
    // door and the `unsafe` call for every OTHER producer of raw streams (the
    // DCC exporter, photogrammetry finish, the grammar bake), which build
    // their own index buffers and cross no door at all. Returning the
    // degenerate mesh is the only honest answer at a signature with no error
    // channel — `optimize()`'s own words — and it is loud, because unlike an
    // unoptimized mesh an unclusterized one is visible.
    //
    // No `debug_assert!(false, ..)` here, unlike `optimize()`: this branch has
    // an arm over it (`tests::an_index_outside_the_vertex_buffer_never_reaches_the_ffi`),
    // and an assertion that aborts the test process is not a thing a test can
    // drive. The `tracing::error!` is the louder channel anyway — it survives
    // into the cook log, where the author of the mesh will see it.
    let addressable = indices.iter().all(|&i| (i as usize) < raw.len());
    if !addressable {
        tracing::error!(
            vertices = raw.len(),
            indices = indices.len(),
            "meshlet build refused: an index addresses outside the vertex buffer; \
             this mesh will not virtualize"
        );
    }
    // **The same door, the other half of the same C assert.** `meshopt`'s every
    // entry point opens with `assert(index_count % 3 == 0)` and the vendored
    // build compiles its asserts out, so a partial triangle is not diagnosed —
    // it is read past the end of the buffer (`meshopt_buildMeshletsScan` and
    // `meshopt_computeClusterBounds` both index `i + 1` / `i + 2` at `i += 3`)
    // or floored away, silently, with `optimize_vertex_cache`'s wrapper still
    // returning a buffer of the original length whose tail is a fabricated
    // `[0, 0, 0]`. `MeshAsset::migrate` refuses such a payload at the decode;
    // this is the backstop for every OTHER producer of raw streams, exactly as
    // `addressable` above is.
    let whole_triangles = indices.len().is_multiple_of(3);
    if !whole_triangles {
        tracing::error!(
            vertices = raw.len(),
            indices = indices.len(),
            "meshlet build refused: the index buffer is not a whole number of \
             triangles; this mesh will not virtualize"
        );
    }

    // Degenerate: nothing to cluster.
    if raw.is_empty() || indices.len() < 3 || !addressable || !whole_triangles {
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
    let level0 = clusterize(&vertices, &index, &params, 0, 0.0);
    build_dag(vertices, level0, &params)
}

/// Everything after clusterization: group → simplify → recluster → link, level by
/// level, then [`assemble`].
///
/// Split out of [`build_vgeom`] because this half is **pure Rust** and its
/// guarantee is supposed to hold for *any* clusterization of the mesh, not just
/// the one `meshopt`'s native clusterizer produces on the machine that happens to
/// be cooking. That is exactly what
/// `tests::cut_invariant_survives_adversarial_clusterings` fuzzes through this
/// entry point — `meshopt`'s clusterizer is not the same code on arm64 and
/// x86_64, and a builder hole reachable only through *its* output is still a
/// builder hole.
fn build_dag(
    vertices: Vec<VgeomVertex>,
    level0: Vec<BuiltMeshlet>,
    params: &BuildParams,
) -> VgeomMesh {
    let mut levels: Vec<Vec<BuiltMeshlet>> = vec![level0];
    let mut groups: Vec<BuiltGroup> = Vec::new();
    // Every undirected edge that has appeared at *any* level built so far. A group
    // may invent an edge between two seam vertices (see [`seam_safe`]) only if the
    // edge is new to the whole mesh, because any edge that already exists somewhere
    // belongs to a region a mixed-level cut can select alongside this group's
    // output — which is exactly how the level-4 collision in the P13.1 gate arose.
    let mut seen_edges: BTreeSet<(u32, u32)> =
        levels[0].iter().flat_map(|m| m.global_edges()).collect();
    let mut level = 0usize;

    while level + 1 < params.max_levels {
        let count = levels[level].len();
        if count <= 1 {
            break; // a single meshlet is already a root
        }

        // Partition this level's meshlets into groups (deterministic greedy).
        let group_members = greedy_group(&levels[level], params.max_group_size);

        // The level's seam vertices: shared between two groups, or on the level's
        // own open rim. Computed once for the whole level (it is a property of the
        // partition, not of any one group) and masked per group below.
        let seam = level_seam(&levels[level], &group_members, vertices.len());

        // Prepare per-group simplification jobs (serial, cheap).
        let jobs: Vec<GroupJob> = group_members
            .iter()
            .map(|members| {
                build_group_job(
                    &levels[level],
                    members,
                    &seam,
                    vertices.len(),
                    params.target_ratio,
                )
            })
            .collect();

        // The named parallel hot loop (§2.5): simplify every group. `parallel_map`
        // collects in input order, so the result is pool-size-invariant.
        let mut results: Vec<GroupResult> =
            inf_core::parallel_map(jobs, |job| simplify_group(&vertices, &seen_edges, job));

        // Two groups may each invent the *same* brand-new seam chord — neither can
        // see the other's output, so neither `seam_safe` call can rule it out, and
        // the union would put four triangles on that edge. Resolve it here, where
        // the whole level is in hand: first group (in index order — deterministic)
        // keeps the edge, the rest do not coarsen this round.
        let mut claimed: BTreeSet<(u32, u32)> = BTreeSet::new();
        for res in results.iter_mut() {
            if !res.progressed {
                continue;
            }
            if res.invented.iter().any(|e| claimed.contains(e)) {
                res.progressed = false;
                continue;
            }
            claimed.extend(res.invented.iter().copied());
        }

        // Re-cluster progressed groups into the next level; link the DAG.
        let mut next: Vec<BuiltMeshlet> = Vec::new();
        let mut any_progress = false;

        for (members, res) in group_members.iter().zip(results) {
            if !res.progressed {
                // No coarser replacement for this region: members stay roots.
                continue;
            }
            let new_meshlets = clusterize(
                &vertices,
                &res.simplified_index,
                params,
                (level + 1) as u8,
                res.group_error,
            );
            // A progressed group that clusterizes to *nothing* (e.g. the simplified
            // soup degenerated) must be treated as NOT progressed: creating a group
            // with `produced_count == 0` and linking the members to it would set
            // their `parent_error` finite while no parent meshlet exists, orphaning
            // them (a member with `error ≤ t < parent_error` and no covering coarser
            // meshlet leaves a hole in the cut). Skipping keeps the members roots
            // (`parent_error` stays `+∞`), honoring the model.rs `produced_count ≥ 1`
            // invariant.
            if new_meshlets.is_empty() {
                debug_assert!(
                    false,
                    "progressed group produced zero meshlets (simplified_index len = {})",
                    res.simplified_index.len()
                );
                continue;
            }
            any_progress = true;
            seen_edges.extend(edge_use_counts(&res.simplified_index).into_keys());
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
        for key in m.global_edges() {
            let owners = edge_owners.entry(key).or_default();
            if owners.last() != Some(&mi) {
                owners.push(mi);
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

/// The **seam vertices** of one level's grouping: every vertex used by two or more
/// groups (their shared border), plus every vertex on the level's own open rim.
///
/// These are exactly the vertices whose geometry another group — or the mesh's
/// silhouette — also depends on, so they are the ones simplification must keep
/// fixed, and (see [`seam_safe`]) the ones an invented edge between is forbidden.
///
/// This replaces the older per-group "endpoints of a once-used edge of my soup"
/// rule. On a manifold level the two sets are identical (a vertex shared with a
/// neighbour always ends a once-used edge of each side's soup); they differ only
/// at a *bow-tie* vertex, where two groups meet at a single point with no
/// once-used edge between them — which the old rule left unlocked and free to
/// move, i.e. free to crack.
fn level_seam(
    meshlets: &[BuiltMeshlet],
    group_members: &[Vec<usize>],
    vertex_count: usize,
) -> Vec<bool> {
    let mut seam = vec![false; vertex_count];

    // (a) vertices claimed by more than one group.
    let mut owner: Vec<u32> = vec![u32::MAX; vertex_count];
    // (b) the level's open rim: edges used exactly once across the whole level.
    let mut edge_count: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    for (gi, members) in group_members.iter().enumerate() {
        let gi = gi as u32;
        for &mi in members {
            let m = &meshlets[mi];
            for gv in m.global_triangles() {
                for v in gv {
                    let v = v as usize;
                    if owner[v] == u32::MAX {
                        owner[v] = gi;
                    } else if owner[v] != gi {
                        seam[v] = true;
                    }
                }
                for (a, b) in [(gv[0], gv[1]), (gv[1], gv[2]), (gv[2], gv[0])] {
                    *edge_count.entry(edge_key(a, b)).or_insert(0) += 1;
                }
            }
        }
    }

    for ((a, b), c) in edge_count {
        if c == 1 {
            seam[a as usize] = true;
            seam[b as usize] = true;
        }
    }
    seam
}

/// Assemble one group's simplification job: gather its triangle soup, mask the
/// level's seam flags down to the vertices this group actually uses, and compute
/// the simplification target.
fn build_group_job(
    meshlets: &[BuiltMeshlet],
    members: &[usize],
    seam: &[bool],
    vertex_count: usize,
    target_ratio: f32,
) -> GroupJob {
    let mut combined_index: Vec<u32> = Vec::new();
    let mut max_child_error = 0.0f32;
    for &mi in members {
        let m = &meshlets[mi];
        max_child_error = max_child_error.max(m.error);
        for gv in m.global_triangles() {
            combined_index.extend_from_slice(&gv);
        }
    }
    // Masked to this group's own vertices: a lock flag on a vertex this group does
    // not reference is meaningless to `meshopt` and would only widen what
    // `seam_safe` calls "shared".
    let mut lock = vec![false; vertex_count];
    for &v in &combined_index {
        if seam[v as usize] {
            lock[v as usize] = true;
        }
    }
    let tri_count = combined_index.len() / 3;
    let target_tri = ((tri_count as f32 * target_ratio).round() as usize).max(1);
    GroupJob {
        combined_index,
        lock,
        target_index_count: target_tri * 3,
        max_child_error,
    }
}

/// Undirected edge key.
#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Undirected edge → incident-triangle count over a triangle-list index buffer.
fn edge_use_counts(index: &[u32]) -> BTreeMap<(u32, u32), u32> {
    let mut counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for t in index.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *counts.entry(edge_key(a, b)).or_insert(0) += 1;
        }
    }
    counts
}

/// Whether a group's simplified soup may be **substituted** for its input soup
/// without breaking the union — and if so, which brand-new seam edges it brought
/// with it (the level loop de-duplicates those across groups).
///
/// The conditions:
///
/// 1. **Manifold, non-degenerate output** — no edge carries more than two of the
///    group's own triangles, and no triangle is degenerate.
/// 2. **Border preserved exactly** — the once-used edges of the output are the
///    once-used edges of the input. A lost border edge is a hole, a gained one is
///    a crack; either way the neighbouring group no longer matches.
/// 3. **No seam edge used more than the input used it** — for an edge whose
///    *both* endpoints are seam vertices, `out ≤ in`, and the `in == 0` case
///    (an **invented** chord) is allowed only when the edge has never existed
///    anywhere in the mesh at any level built so far.
///
/// Condition 3 is the one the old builder was missing. `meshopt` locks vertices,
/// not edges: collapsing an interior vertex *into* a seam vertex rewrites every
/// triangle it belonged to and can invent a chord between two seam vertices that
/// the group never had. Inventing is not itself wrong — the coarse patch has to be
/// allowed to retriangulate its interior, and forbidding it outright flattens the
/// whole DAG to one level. It is wrong exactly when *somebody else* also draws
/// that edge, and the two ways that happens are (a) the edge already exists in
/// another region, at this or any finer level, which a mixed-level cut can select
/// beside this group's output — ruled out here by `seen`; and (b) a sibling group
/// invents the identical chord in the same round — ruled out by the level loop,
/// which is the only place both outputs are visible.
///
/// Soundness: the group input soups partition the level, and an edge with a
/// non-seam endpoint is private to one group (a non-seam vertex is used by no other
/// group, by construction of [`level_seam`]). So for pre-existing seam edges
/// `Σ_groups out ≤ Σ_groups in ≤ 2`; an invented edge is claimed once for the whole
/// build; and private edges are bounded by condition 1. The union of the accepted
/// coarse patches stays manifold **whatever the clusterization** — which is the
/// property the P13.1 cut invariant needs, and the reason this check is not tuned
/// to any particular mesh.
fn seam_safe(
    input: &[u32],
    output: &[u32],
    lock: &[bool],
    seen: &BTreeSet<(u32, u32)>,
) -> Option<Vec<(u32, u32)>> {
    if output
        .chunks_exact(3)
        .any(|t| t[0] == t[1] || t[1] == t[2] || t[2] == t[0])
    {
        return None;
    }
    let inc = edge_use_counts(input);
    let outc = edge_use_counts(output);

    let mut invented: Vec<(u32, u32)> = Vec::new();
    for (&e, &c) in &outc {
        if c > 2 {
            return None;
        }
        let (a, b) = e;
        if !(lock[a as usize] && lock[b as usize]) {
            continue; // private to this group
        }
        let had = inc.get(&e).copied().unwrap_or(0);
        if c > had {
            if had > 0 || seen.contains(&e) {
                return None;
            }
            invented.push(e);
        }
    }

    // Borders (once-used edges) must match exactly. `BTreeMap` iteration is
    // ordered, so this compares the two sorted edge sequences without allocating.
    let borders_match = inc
        .iter()
        .filter(|(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .eq(outc.iter().filter(|(_, &c)| c == 1).map(|(&e, _)| e));
    borders_match.then_some(invented)
}

/// Simplify one group with its seam vertices locked. Returns the simplified index
/// buffer, the cumulative (strictly monotone) group error, and whether the result
/// may replace the input — it made progress (fewer triangles) **and** it is
/// [`seam_safe`].
fn simplify_group(
    vertices: &[VgeomVertex],
    seen: &BTreeSet<(u32, u32)>,
    job: GroupJob,
) -> GroupResult {
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

    let coarsened =
        simplified_index.len() >= 3 && simplified_index.len() < job.combined_index.len();
    let invented = if coarsened {
        seam_safe(&job.combined_index, &simplified_index, &job.lock, seen)
    } else {
        None
    };
    // Cumulative, strictly increasing object-space error (kept non-empty even when
    // meshopt reports ~0 error, or the increment rounds away at coarse magnitudes).
    let group_error = monotone_group_error(job.max_child_error, simplify_error);

    GroupResult {
        simplified_index,
        group_error,
        progressed: invented.is_some(),
        invented: invented.unwrap_or_default(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_error_strictly_exceeds_children_at_coarse_magnitudes() {
        // At a large child error, a zero (or sub-ulp) simplify error is absorbed by
        // f32 addition — `16.0 + 1e-6 == 16.0` — which the old absolute-ε bump did
        // NOT survive, collapsing the cut interval `[error, parent_error)`. The
        // monotone construction must still yield a strictly greater value.
        let e = monotone_group_error(16.0, 0.0);
        assert!(e > 16.0, "zero simplify error must still advance: got {e}");
        assert_eq!(e, f32::next_up(16.0));

        // A sub-ulp increment (below one ulp at 16.0 ≈ 1.9e-6) is also absorbed and
        // must be bumped to the next representable float.
        let tiny = monotone_group_error(16.0, 1e-9);
        assert!(tiny > 16.0, "absorbed increment must advance: got {tiny}");

        // A measurable simplify error passes straight through (no bump needed).
        assert_eq!(monotone_group_error(16.0, 0.5), 16.5);

        // The interval is non-empty even at LOD 0 (child error 0).
        let z = monotone_group_error(0.0, 0.0);
        assert!(
            z > 0.0,
            "zero child + zero simplify must still be > 0: got {z}"
        );
    }

    // ── the clusterization fuzz (P13.1 watertightness, arm64 regression) ──────
    //
    // `build_meshlets` is `meshopt`'s native C++, and it is **not the same code**
    // on arm64 and x86_64 — fed byte-identical vertices it returns a different
    // partition, which is how a builder hole that x86_64's clusterization walked
    // straight past showed up as a macOS-only failure of
    // `tests/dag.rs::cut_invariant_holds_at_every_threshold`. Everything the
    // builder guarantees is supposed to hold for **any** partition of the
    // triangles into meshlets, so the honest gate is to drive [`build_dag`] — the
    // whole pure-Rust half — with partitions no clusterizer would ever emit and
    // assert the cut invariant on what comes back. It reproduces the arm64 class
    // of failure on any host.

    /// xorshift64\* — a deterministic PRNG so a fuzz failure is a fixed seed, not
    /// a story. (`rand` is not a dependency of this crate and does not need to be.)
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        /// Uniform-ish in `1..=n`.
        fn range1(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize + 1
        }
    }

    /// The `n × n` wavy grid of `tests/dag.rs` — the fixture the P13.1 gate
    /// actually broke on. Its displacement oscillates (several extrema across the
    /// sheet), so simplification has real choices to make; a monotone bump is too
    /// tidy to provoke the seam collision. `psin64`/`pcos64` keep it bit-portable
    /// (the P14 LAW), and every position is distinct, so the input is manifold and
    /// **The tangent stream is invisible to the clusterizer** (P28.2). Two builds
    /// of one geometry with *different* tangents must produce the same DAG down
    /// to every micro-index — `meshopt` reads the first twelve bytes of a vertex
    /// record and the stride, and the stride is equal here, so what is being
    /// falsified is that the new channel leaks into clusterization or
    /// simplification. Mutating a clone within one run, because `meshopt`'s output
    /// is not comparable across platforms (the P18 law).
    #[test]
    fn the_tangent_stream_does_not_move_the_dag() {
        let n = 20;
        let (p, nrm, uv, idx) =
            crate::test_support::displaced_grid(n, 0.3, crate::test_support::GridNormals::Analytic);
        let ones: Vec<[f32; 4]> = (0..p.len()).map(|_| [1.0, 0.0, 0.0, 1.0]).collect();
        let varied: Vec<[f32; 4]> = (0..p.len())
            .map(|i| {
                let t = i as f32 / p.len() as f32;
                [t, 1.0 - t, 0.25, if i % 2 == 0 { 1.0 } else { -1.0 }]
            })
            .collect();
        let a = build_vgeom(&p, &nrm, &uv, &ones, &idx, BuildParams::default());
        let b = build_vgeom(&p, &nrm, &uv, &varied, &idx, BuildParams::default());
        let c = build_vgeom(&p, &nrm, &uv, &[], &idx, BuildParams::default());
        assert!(a.levels.len() >= 3, "a fixture with a real DAG");
        for (name, other) in [("varied", &b), ("absent", &c)] {
            assert_eq!(a.meshlets, other.meshlets, "{name} moved the meshlets");
            assert_eq!(a.meshlet_vertices, other.meshlet_vertices, "{name}");
            assert_eq!(a.meshlet_triangles, other.meshlet_triangles, "{name}");
            assert_eq!(a.levels, other.levels, "{name}");
            assert_eq!(a.groups, other.groups, "{name}");
        }
        // And the channel really did travel — otherwise the equalities above are
        // satisfied by a build that dropped it.
        assert!(a.vertices.iter().all(|v| v.tangent != crate::NO_TANGENT));
        assert!(c.vertices.iter().all(|v| v.tangent == crate::NO_TANGENT));
        assert_ne!(
            a.vertices.iter().map(|v| v.tangent).collect::<Vec<_>>(),
            b.vertices.iter().map(|v| v.tangent).collect::<Vec<_>>(),
            "two different tangent streams packed to the same words"
        );
    }

    /// nothing welds.
    fn fuzz_grid(n: usize) -> (Vec<VgeomVertex>, Vec<u32>) {
        let mut vertices = Vec::with_capacity(n * n);
        for z in 0..n {
            for x in 0..n {
                let (fx, fz) = (x as f32, z as f32);
                let y = (0.6
                    * inf_math::psin64(fx as f64 * 0.5)
                    * inf_math::pcos64(fz as f64 * 0.5)) as f32;
                vertices.push(VgeomVertex {
                    position: [fx, y, fz],
                    normal: [0.0, 1.0, 0.0],
                    uv: [fx / n as f32, fz / n as f32],
                    tangent: crate::model::NO_TANGENT,
                });
            }
        }
        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
        let idx = |x: usize, z: usize| (z * n + x) as u32;
        for z in 0..n - 1 {
            for x in 0..n - 1 {
                let (a, b, c, d) = (idx(x, z), idx(x + 1, z), idx(x, z + 1), idx(x + 1, z + 1));
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        (vertices, indices)
    }

    /// How a fuzz run partitions the triangles — the stand-in for
    /// `meshopt::build_meshlets`.
    #[derive(Clone, Copy, Debug)]
    enum FuzzMode {
        /// Randomly-sized **compact tiles** with ragged edges: the shape a real
        /// clusterizer produces, and the one that actually provokes the seam
        /// collision — a cluster needs a fat interior for a vertex to have anywhere
        /// to collapse *from*. Strips and scatter lock nearly everything and are
        /// therefore the easy cases, not the hard ones.
        Tiles { w: usize, h: usize, ragged: bool },
        /// Random-length runs of the row-major triangle order (long thin strips).
        Strips,
        /// Random-length runs of a shuffled triangle order (spatially incoherent
        /// clusters no clusterizer would emit).
        Scatter,
    }

    /// Partition `index`'s triangles into [`BuiltMeshlet`]s per `mode`. `quads` is
    /// the grid's quad-per-row count, so a triangle can be located on the sheet.
    fn fuzz_clusters(
        vertices: &[VgeomVertex],
        index: &[u32],
        rng: &mut Rng,
        mode: FuzzMode,
        quads: usize,
    ) -> Vec<BuiltMeshlet> {
        let tri_count = index.len() / 3;
        // Triangle -> cluster key. Materialization below preserves key order and
        // then triangle order, so the whole thing stays deterministic.
        let mut keyed: Vec<(u64, usize)> = Vec::with_capacity(tri_count);
        match mode {
            FuzzMode::Tiles { w, h, ragged } => {
                let cols = quads.div_ceil(w) as u64;
                for t in 0..tri_count {
                    let q = t / 2;
                    let (mut cx, mut cz) = ((q % quads) / w, (q / quads) / h);
                    // Ragged edges: nudge some triangles into a neighbouring tile,
                    // which is what makes the seam curve interesting.
                    if ragged && rng.next_u64().is_multiple_of(8) {
                        match rng.next_u64() % 4 {
                            0 => cx = cx.saturating_sub(1),
                            1 => cx += 1,
                            2 => cz = cz.saturating_sub(1),
                            _ => cz += 1,
                        }
                    }
                    keyed.push((cz as u64 * (cols + 2) + cx as u64, t));
                }
            }
            FuzzMode::Strips | FuzzMode::Scatter => {
                let mut order: Vec<usize> = (0..tri_count).collect();
                if matches!(mode, FuzzMode::Scatter) {
                    for i in (1..order.len()).rev() {
                        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                        order.swap(i, j);
                    }
                }
                let mut key = 0u64;
                let mut left = rng.range1(124);
                for t in order {
                    if left == 0 {
                        key += 1;
                        left = rng.range1(124);
                    }
                    left -= 1;
                    keyed.push((key, t));
                }
            }
        }
        keyed.sort_by_key(|&(k, t)| (k, t));

        let mut out: Vec<BuiltMeshlet> = Vec::new();
        let mut verts: Vec<u32> = Vec::new();
        let mut tris: Vec<u8> = Vec::new();
        let mut current = u64::MAX;
        let flush = |verts: &mut Vec<u32>, tris: &mut Vec<u8>, out: &mut Vec<BuiltMeshlet>| {
            if tris.is_empty() {
                return;
            }
            let (center, radius) = fuzz_bounds(vertices, verts);
            out.push(BuiltMeshlet {
                verts: std::mem::take(verts),
                tris: std::mem::take(tris),
                center,
                radius,
                cone_axis: [0.0, 1.0, 0.0],
                cone_cutoff: 1.0,
                lod_level: 0,
                error: 0.0,
                parent_error: f32::INFINITY,
                group: Meshlet::NO_GROUP,
            });
        };

        for (key, t) in keyed {
            let gv = [index[t * 3], index[t * 3 + 1], index[t * 3 + 2]];
            // A meshlet's triangle indices are `u8` locals, so at most 256 vertices;
            // `meshopt`'s own cap is 512 triangles.
            let fresh = gv.iter().filter(|g| !verts.contains(g)).count();
            if key != current || verts.len() + fresh > 256 || tris.len() / 3 >= 512 {
                flush(&mut verts, &mut tris, &mut out);
                current = key;
            }
            for g in gv {
                let local = match verts.iter().position(|v| *v == g) {
                    Some(p) => p,
                    None => {
                        verts.push(g);
                        verts.len() - 1
                    }
                };
                tris.push(local as u8);
            }
        }
        flush(&mut verts, &mut tris, &mut out);
        out
    }

    fn fuzz_bounds(vertices: &[VgeomVertex], verts: &[u32]) -> ([f32; 3], f32) {
        let mut c = [0.0f32; 3];
        for &v in verts {
            let p = vertices[v as usize].position;
            for k in 0..3 {
                c[k] += p[k] / verts.len() as f32;
            }
        }
        let mut r2 = 0.0f32;
        for &v in verts {
            let p = vertices[v as usize].position;
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        (c, r2.sqrt())
    }

    /// The cut invariant, checked at **every** distinct cut: the selection is
    /// constant on `[b, b')` between consecutive distinct meshlet errors (every
    /// `parent_error` is some meshlet's `error`, or `+∞`), so testing each of those
    /// breakpoints tests every selection the runtime can ever make.
    fn assert_cut_invariant(mesh: &VgeomMesh, what: &str) {
        let lod0: Vec<usize> = mesh
            .meshlets
            .iter()
            .enumerate()
            .filter(|(_, m)| m.lod_level == 0)
            .map(|(i, _)| i)
            .collect();
        let rim: Vec<(u32, u32)> = selection_edges(mesh, &lod0)
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(e, _)| e)
            .collect();

        let mut thresholds: Vec<f32> = mesh.meshlets.iter().map(|m| m.error).collect();
        thresholds.push(0.0);
        thresholds.sort_by(|a, b| a.partial_cmp(b).expect("errors are finite"));
        thresholds.dedup();

        for t in thresholds {
            let sel: Vec<usize> = mesh.select(t).map(|(i, _)| i).collect();
            assert!(!sel.is_empty(), "{what}: threshold {t} selected nothing");
            let counts = selection_edges(mesh, &sel);
            for (e, c) in &counts {
                assert!(
                    *c == 1 || *c == 2,
                    "{what}: t={t} edge {e:?} used {c} times — non-manifold cut"
                );
            }
            let boundary: Vec<(u32, u32)> = counts
                .into_iter()
                .filter(|(_, c)| *c == 1)
                .map(|(e, _)| e)
                .collect();
            assert_eq!(
                boundary, rim,
                "{what}: t={t} boundary drifted (crack or hole)"
            );
        }
    }

    fn selection_edges(mesh: &VgeomMesh, sel: &[usize]) -> BTreeMap<(u32, u32), u32> {
        let mut counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for &mi in sel {
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

    #[test]
    fn cut_invariant_survives_adversarial_clusterings() {
        let n = 48;
        let (vertices, index) = fuzz_grid(n);
        let mut multi_level = 0usize;
        let mut runs = 0usize;

        let mut modes: Vec<FuzzMode> = vec![FuzzMode::Strips, FuzzMode::Scatter];
        for w in [3usize, 5, 8, 11] {
            for h in [3usize, 6, 9] {
                for ragged in [false, true] {
                    modes.push(FuzzMode::Tiles { w, h, ragged });
                }
            }
        }

        for seed in 0..3u64 {
            for &mode in &modes {
                // The grouping and the reduction target shape the seams as much as
                // the clusterization does, so they are part of the fuzz.
                for &(max_group_size, target_ratio) in &[(2usize, 0.5f32), (8, 0.5), (8, 0.7)] {
                    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
                    let level0 = fuzz_clusters(&vertices, &index, &mut rng, mode, n - 1);
                    // The synthetic partition must be a partition — every input
                    // triangle exactly once — or the fuzz proves nothing.
                    let tris: usize = level0.iter().map(|m| m.tris.len() / 3).sum();
                    assert_eq!(tris, index.len() / 3, "the fuzz clustering lost triangles");

                    let params = BuildParams {
                        max_group_size,
                        target_ratio,
                        ..BuildParams::default()
                    }
                    .validated();
                    let mesh = build_dag(vertices.clone(), level0, &params);
                    let what = format!(
                        "seed={seed} mode={mode:?} gs={max_group_size} ratio={target_ratio}"
                    );
                    assert_cut_invariant(&mesh, &what);
                    runs += 1;
                    if mesh.level_count() > 1 {
                        multi_level += 1;
                    }
                }
            }
        }

        // Anti-vacuity: a run where nothing ever coarsens would satisfy the cut
        // invariant trivially and gate nothing.
        assert!(
            multi_level * 2 >= runs,
            "only {multi_level} of {runs} fuzz clusterings built a multi-level DAG — \
             the fuzz is not exercising the group/simplify/recluster path"
        );
    }

    /// A quad `0-1-2-3` fanned around one interior vertex `4` — the smallest soup
    /// that shows the bug. `0..=3` are seam (shared with the neighbouring group),
    /// `4` is this group's own.
    fn fan_soup() -> ([bool; 5], [u32; 12]) {
        (
            [true, true, true, true, false],
            [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
        )
    }

    /// The seam rule itself, on a hand-built soup so it reads without a mesh.
    #[test]
    fn seam_safe_rejects_an_invented_chord_between_seam_vertices() {
        let (lock, input) = fan_soup();
        let seen: BTreeSet<(u32, u32)> = edge_use_counts(&input).into_keys().collect();

        // Same soup back: safe, nothing invented.
        assert_eq!(
            seam_safe(&input, &input, &lock, &seen),
            Some(Vec::new()),
            "an unchanged soup must be accepted"
        );

        // Collapse the interior vertex 4 into the seam vertex 0 — precisely what
        // `meshopt` is free to do, since locking 0 stops 0 from *moving* and says
        // nothing about 4 moving onto it. The two triangles that degenerate drop
        // out and the border survives intact, but the diagonal (0,2) — a chord
        // between two seam vertices that this group never had — is now interior.
        let collapsed = [1u32, 2, 0, 2, 3, 0];
        let invented = seam_safe(&input, &collapsed, &lock, &seen)
            .expect("a brand-new chord between seam vertices is allowed, and reported");
        assert_eq!(
            invented,
            vec![(0, 2)],
            "the invented chord must be reported so the level loop can stop a \
             sibling group claiming the same one"
        );

        // If that chord already exists anywhere in the mesh, it belongs to another
        // region that a mixed-level cut can select beside this output — reject.
        let mut seen_more = seen.clone();
        seen_more.insert((0, 2));
        assert_eq!(
            seam_safe(&input, &collapsed, &lock, &seen_more),
            None,
            "a chord that already exists somewhere must not be invented"
        );
    }

    /// The border is still sacred: losing one is a hole, gaining one is a crack.
    #[test]
    fn seam_safe_rejects_a_changed_border() {
        let (lock, input) = fan_soup();
        let seen: BTreeSet<(u32, u32)> = edge_use_counts(&input).into_keys().collect();
        // Dropping one triangle uses no edge more than the input did, but opens the
        // border along (0,4) and (1,4).
        let holed = [0u32, 1, 4, 1, 2, 4, 2, 3, 4];
        assert_eq!(seam_safe(&input, &holed, &lock, &seen), None);
        // A degenerate triangle is never acceptable either.
        let degenerate = [0u32, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 0];
        assert_eq!(seam_safe(&input, &degenerate, &lock, &seen), None);
    }

    /// **Round-2 finding B2**: an index outside the vertex buffer must never
    /// reach `meshopt::generate_vertex_remap`.
    ///
    /// The FFI sizes its remap table from `vertices.len()` and writes
    /// `remap[index]` with the `assert` compiled out under `-DNDEBUG`, so one
    /// index past the end is an out-of-bounds heap **write**. `inf_mesh`'s
    /// `optimize()` grew a backstop in Wave B; this door reaches the same call
    /// from a `.inf_mesh` decoded off disk (the editor's `.inf_vmesh` derive
    /// and the cook) and had none.
    ///
    /// The control is the same geometry with the index repaired, so "no
    /// meshlets" cannot pass for a fixture that was never clusterizable.
    ///
    /// **Mutation note:** the guard was NOT removed and re-run. Removing it
    /// makes this test perform the out-of-bounds heap write it exists to
    /// prevent, which is the finding, not a verification of it. What is
    /// verified instead is that the control clusterizes and the hostile input
    /// does not — so the arm can only pass while the branch is taken.
    #[test]
    fn an_index_outside_the_vertex_buffer_never_reaches_the_ffi() {
        // Two triangles over four vertices.
        let p = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let n = vec![[0.0f32, 0.0, 1.0]; 4];
        let uv = vec![[0.0f32, 0.0]; 4];
        let good = [0u32, 1, 2, 0, 2, 3];

        let control = build_vgeom(&p, &n, &uv, &[], &good, BuildParams::default());
        assert!(
            !control.meshlets.is_empty(),
            "the control never clusterized, so the hostile case below is vacuous"
        );

        // One index past the end — the exact C4-1 shape, at the other door.
        let hostile = [0u32, 1, 2, 0, 2, 4];
        let out = build_vgeom(&p, &n, &uv, &[], &hostile, BuildParams::default());
        assert!(
            out.meshlets.is_empty() && out.meshlet_vertices.is_empty(),
            "an out-of-range index was clusterized — it went through the raw FFI"
        );
        assert_eq!(
            out.vertices.len(),
            p.len(),
            "the degenerate answer must still be the mesh's own vertices"
        );

        // `u32::MAX` is the same refusal, not a different one.
        let far = [0u32, 1, 2, 0, 2, u32::MAX];
        assert!(build_vgeom(&p, &n, &uv, &[], &far, BuildParams::default())
            .meshlets
            .is_empty());
    }

    /// **The other half of the same compiled-out assert** (round-3): an index
    /// buffer that is not a whole number of triangles.
    ///
    /// Every `meshopt` entry point opens with `assert(index_count % 3 == 0)`,
    /// compiled out under `-DNDEBUG`. The consequence here is not a heap write
    /// but silent geometry loss — `meshopt_buildMeshlets` floors
    /// `index_count / 3`, so the trailing indices are dropped from the DAG
    /// while every count derived from `indices.len()` still includes them.
    /// Refusing says so; flooring does not.
    ///
    /// Every index is in range, so this cannot pass by way of the
    /// `addressable` branch beside it — asserted, not assumed.
    #[test]
    fn a_partial_triangle_never_reaches_the_ffi() {
        let p = vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let n = vec![[0.0f32, 0.0, 1.0]; 4];
        let uv = vec![[0.0f32, 0.0]; 4];

        let control = build_vgeom(
            &p,
            &n,
            &uv,
            &[],
            &[0u32, 1, 2, 0, 2, 3],
            BuildParams::default(),
        );
        assert!(
            !control.meshlets.is_empty(),
            "the control never clusterized, so the hostile cases below are vacuous"
        );

        for hostile in [
            vec![0u32, 1, 2, 0],
            vec![0u32, 1, 2, 0, 2],
            // Four indices, all in range: `< 3` does not catch it and
            // `addressable` answers true.
            vec![0u32, 1, 2, 3],
        ] {
            assert!(
                hostile.iter().all(|&i| (i as usize) < p.len()),
                "the fixture must be in range or it is testing the other guard"
            );
            let out = build_vgeom(&p, &n, &uv, &[], &hostile, BuildParams::default());
            assert!(
                out.meshlets.is_empty() && out.meshlet_triangles.is_empty(),
                "{} indices were clusterized as {} triangles",
                hostile.len(),
                hostile.len() / 3
            );
        }
    }
}
