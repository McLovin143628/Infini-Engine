//! **Deterministic Surface Nets** over the chunk store, and the watertight-seam
//! policy that is this batch's required gate.
//!
//! # The algorithm, in one paragraph
//!
//! Every cell of the global voxel grid whose eight corner samples do not share a
//! sign gets **one** vertex, placed at the average of that cell's sign-change edge
//! crossings; every grid edge whose two samples differ in sign gets **one** quad,
//! joining the vertices of the four cells around it. That is Surface Nets (the
//! dual-contouring family without the QEF solve): it produces a manifold surface
//! with far fewer, better-shaped triangles than marching cubes and — crucially for
//! P21.3's excavation tools — a vertex that moves *continuously* as the field is
//! carved, rather than snapping between marching-cubes cases.
//!
//! # THE SEAM POLICY, and why it is the gate
//!
//! P18 shipped a meshlet builder whose seam chords overlapped, and roughly one
//! mesh in ten went out with a visible crack. A voxel volume is *made* of seams —
//! one every [`CHUNK_DIM`] samples in three axes — so the policy is stated here
//! and gated by a test rather than hoped for.
//!
//! **Cells are partitioned; vertices are duplicated and must be bit-identical.**
//!
//! * A chunk **owns** the `CHUNK_DIM³` cells whose minimum corner is one of its
//!   own samples. Every cell in the world is owned by exactly one chunk, so every
//!   quad is emitted exactly once and no triangle is ever drawn twice.
//! * To emit those quads a chunk needs the vertices of cells **one below it** on
//!   each axis (the four cells around an edge include the cell one step back in
//!   the two off-axis directions), so it also computes vertices for a one-cell
//!   apron. Those apron vertices are *duplicates* of vertices its neighbour also
//!   computes — and the gate is that the two copies are **bit-identical**, not
//!   merely close.
//! * They are bit-identical **by construction**: a vertex is a pure function of
//!   its cell's eight corner samples, both chunks read those same samples from the
//!   same [`VoxelData`], and the position is accumulated in the cell's own
//!   `[0, 1]³` frame and then added to the cell's **integer global coordinate** —
//!   never to a chunk-relative offset that would round differently on the two
//!   sides. [`seam_vertices_are_bit_identical_from_both_sides`] asserts exact
//!   `f64` equality, which is the only assertion worth making here.
//!
//! The alternative — storing a one-voxel apron *inside* each chunk — was rejected:
//! it gives every boundary sample two authorities, so a carve has to write both or
//! the seam silently disagrees, and that is a strictly worse bug than the one it
//! avoids. Reading the neighbour is free here because residency is already a
//! `BTreeMap` the mesher can index.
//!
//! # Determinism
//!
//! The whole pass is a **gather**: one vertex per cell of a fixed integer range,
//! one quad per edge of a fixed integer range, accumulated in a fixed order, with
//! no scatter into shared words and no hash map anywhere. Two runs produce
//! byte-identical buffers, and so do two runs whose chunks were inserted in
//! different orders — both asserted below.
//!
//! # Coordinates
//!
//! [`VoxelMesh::positions`] are **global voxel-grid units in `f64`**. That is what
//! makes the seam claim testable at all (two chunks' copies are directly
//! comparable), and it keeps the mesher free of both the volume's metre scale and
//! the renderer's floating origin. The two projectors rebase to chunk-local `f32`
//! metres through the one shared helper, [`VoxelMesh::local_positions_m`], so they
//! cannot diverge on the conversion.

use std::collections::{BTreeMap, BTreeSet};

use crate::chunk::{ChunkKey, CHUNK_DIM, DEFAULT_MATERIAL, EMPTY_SDF};
use crate::data::{local_of, VoxelData};

/// Samples per axis in the padded gather: the chunk's own `CHUNK_DIM`, plus one
/// apron sample below (for the apron cells) and one above (for the last owned
/// cell's far corner).
const PAD_SAMPLES: usize = CHUNK_DIM as usize + 2;

/// Cells per axis a chunk computes vertices for: its `CHUNK_DIM` owned cells plus
/// the one-cell apron below.
const PAD_CELLS: usize = CHUNK_DIM as usize + 1;

/// Sentinel for "this cell produced no vertex".
const NO_VERTEX: u32 = u32::MAX;

/// One chunk's meshed surface.
///
/// The vertex streams are parallel: `positions[i]`, `normals[i]`, `materials[i]`
/// and `cells[i]` all describe vertex `i`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VoxelMesh {
    /// The chunk that emitted this mesh (and therefore owns its quads).
    pub key: ChunkKey,
    /// Vertex positions in **global voxel-grid units** (`f64`) — see the module
    /// docs on why not chunk-local metres.
    pub positions: Vec<[f64; 3]>,
    /// Unit outward normals, from the gradient of the cell's trilinear
    /// interpolant.
    pub normals: Vec<[f32; 3]>,
    /// Splat material index (`0..MATERIAL_COUNT`) per vertex.
    pub materials: Vec<u8>,
    /// The **global cell coordinate** each vertex came from — the seam-identity
    /// key. Two chunks' copies of a shared vertex carry the same cell, which is
    /// what lets a test compare them without guessing at correspondence, and what
    /// a future stitcher would weld on.
    pub cells: Vec<[i32; 3]>,
    /// Triangle indices into the vertex streams (CCW when viewed from outside, so
    /// back-face culling keeps the cave interior visible from inside it and the
    /// rock face visible from outside).
    pub indices: Vec<u32>,
}

impl VoxelMesh {
    /// An empty mesh for `key` (no surface crosses this chunk's owned cells).
    pub fn empty(key: ChunkKey) -> Self {
        Self {
            key,
            ..Default::default()
        }
    }

    /// `true` when this chunk emitted no triangles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Number of vertices.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Vertex positions rebased into **chunk-local metres** (`f32`) — the form the
    /// GPU wants, paired with the chunk's own `f64` world origin
    /// ([`VoxelData::chunk_origin_world`]) as the floating-origin anchor.
    ///
    /// Lives here rather than in either projector precisely so the editor viewport
    /// and the shipped player cannot disagree about it: a host that subtracted a
    /// different base, or scaled in a different order, would draw a cave a
    /// fraction of a voxel away from where the other host drew it, and nothing but
    /// a screenshot comparison would ever notice.
    pub fn local_positions_m(&self, voxel_size_m: f64) -> Vec<[f32; 3]> {
        let b = self.key.base_sample();
        self.positions
            .iter()
            .map(|p| {
                [
                    ((p[0] - b[0] as f64) * voxel_size_m) as f32,
                    ((p[1] - b[1] as f64) * voxel_size_m) as f32,
                    ((p[2] - b[2] as f64) * voxel_size_m) as f32,
                ]
            })
            .collect()
    }

    /// Inclusive `(min, max)` chunk-local AABB in metres — the per-chunk frustum
    /// cull bound. `([0;3], [0;3])` for an empty mesh.
    pub fn local_bounds_m(&self, voxel_size_m: f64) -> ([f32; 3], [f32; 3]) {
        let local = self.local_positions_m(voxel_size_m);
        let Some(first) = local.first() else {
            return ([0.0; 3], [0.0; 3]);
        };
        let mut lo = *first;
        let mut hi = *first;
        for p in &local {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }
}

/// Mesh one chunk's **owned** cells, reading its neighbours for the padded gather.
///
/// The result is a pure function of the samples in
/// `[base − 1, base + CHUNK_DIM]³` — nothing else about `data` can reach it — so
/// two calls with the same field produce byte-identical buffers regardless of what
/// else is resident or in what order it arrived.
pub fn mesh_chunk(data: &VoxelData, key: ChunkKey) -> VoxelMesh {
    let base = key.base_sample();
    let lo = [base[0] - 1, base[1] - 1, base[2] - 1];

    // ── the padded gather ────────────────────────────────────────────────────
    // Two `BTreeMap` probes per sample (one for the distance, one for the
    // material) would be ~11.7k lookups. The samples are walked x-fastest and a
    // row of `CHUNK_DIM + 2` crosses a chunk boundary at most twice, so a
    // **one-entry memo** of the chunk currently under the cursor collapses that to
    // at most three probes per row — and to exactly one per row in the interior,
    // which is where the cost is. The memo is keyed by [`ChunkKey`], so a miss is
    // one comparison; it is deliberately written out rather than delegated to
    // `sample_global`, which cannot hold state across calls.
    //
    // The memo caches **absence too** (`Some((key, None))` = "probed, not there"),
    // and that is load-bearing rather than tidy. `mesh_keys_for` adds up to eight
    // downward-closure keys per resident chunk and most of them are not resident,
    // so the *majority* of gathers walk mostly-absent space; a memo that only
    // remembered hits would re-probe every single one of those samples and the
    // "three probes per row" claim above would be false exactly where it matters
    // most. Remembering the miss costs one `Option` and makes the claim true for
    // absent chunks and present ones alike.
    let mut sdf = vec![0.0f32; PAD_SAMPLES * PAD_SAMPLES * PAD_SAMPLES];
    let mut mat = vec![DEFAULT_MATERIAL; PAD_SAMPLES * PAD_SAMPLES * PAD_SAMPLES];
    let mut any_solid = false;
    let mut any_empty = false;
    let mut memo: Option<(ChunkKey, Option<&crate::chunk::VoxelChunk>)> = None;
    for sz in 0..PAD_SAMPLES {
        for sy in 0..PAD_SAMPLES {
            for sx in 0..PAD_SAMPLES {
                let g = [lo[0] + sx as i32, lo[1] + sy as i32, lo[2] + sz as i32];
                let ck = ChunkKey::of_sample(g[0], g[1], g[2]);
                if memo.map(|(k, _)| k) != Some(ck) {
                    // Absent ⇒ empty space, and the *miss* is memoized too, so the
                    // next sample in this chunk does not probe again either.
                    memo = Some((ck, data.get_chunk(ck)));
                }
                let hit = match memo {
                    Some((k, c)) if k == ck => c,
                    _ => None,
                };
                let (v, m) = match hit {
                    Some(c) => {
                        let (i, j, k) = local_of(g[0], g[1], g[2]);
                        (c.sample(i, j, k), c.material(i, j, k))
                    }
                    None => (EMPTY_SDF, DEFAULT_MATERIAL),
                };
                let i = block_index(sx, sy, sz);
                sdf[i] = v;
                mat[i] = m;
                if v < 0.0 {
                    any_solid = true;
                } else {
                    any_empty = true;
                }
            }
        }
    }
    // A block of one sign has no surface anywhere in it — the common case for
    // solid rock and for open air alike.
    if !(any_solid && any_empty) {
        return VoxelMesh::empty(key);
    }

    // ── one vertex per active cell (owned cells plus the one-cell apron) ──────
    let mut vert_of_cell = vec![NO_VERTEX; PAD_CELLS * PAD_CELLS * PAD_CELLS];
    let mut mesh = VoxelMesh::empty(key);
    for cz in 0..PAD_CELLS {
        for cy in 0..PAD_CELLS {
            for cx in 0..PAD_CELLS {
                let corners = corner_indices(cx, cy, cz);
                let s: [f64; 8] = std::array::from_fn(|m| sdf[corners[m]] as f64);
                if !crosses(&s) {
                    continue;
                }
                // The global cell coordinate. Integer arithmetic, so both sides of
                // a seam name the same cell with the same number.
                let cell = [lo[0] + cx as i32, lo[1] + cy as i32, lo[2] + cz as i32];
                let offset = cell_vertex_offset(&s);
                mesh.positions.push([
                    cell[0] as f64 + offset[0],
                    cell[1] as f64 + offset[1],
                    cell[2] as f64 + offset[2],
                ]);
                mesh.normals.push(cell_normal(&s));
                mesh.materials.push(cell_material(&s, &corners, &mat));
                mesh.cells.push(cell);
                vert_of_cell[cell_index(cx, cy, cz)] = (mesh.positions.len() - 1) as u32;
            }
        }
    }

    // ── one quad per owned sign-change edge ──────────────────────────────────
    // "Owned" = the edge's minimum corner is one of this chunk's own samples,
    // i.e. the cell at that corner is one of the CHUNK_DIM³ this chunk owns. That
    // partitions the world's quads across chunks exactly once each.
    for cz in 1..PAD_CELLS {
        for cy in 1..PAD_CELLS {
            for cx in 1..PAD_CELLS {
                let m0 = block_index(cx, cy, cz);
                let s0 = sdf[m0];
                for axis in 0..3usize {
                    let mut step = [0usize; 3];
                    step[axis] = 1;
                    let s1 = sdf[block_index(cx + step[0], cy + step[1], cz + step[2])];
                    if (s0 < 0.0) == (s1 < 0.0) {
                        continue;
                    }
                    // The four cells around this edge: the cell at the edge's
                    // minimum corner, and its neighbours one step back in the two
                    // off-axis directions.
                    let b = (axis + 1) % 3;
                    let c = (axis + 2) % 3;
                    let cell = [cx, cy, cz];
                    let at = |db: usize, dc: usize| -> u32 {
                        let mut q = cell;
                        q[b] -= db;
                        q[c] -= dc;
                        vert_of_cell[cell_index(q[0], q[1], q[2])]
                    };
                    let (v00, v10, v11, v01) = (at(0, 0), at(1, 0), at(1, 1), at(0, 1));
                    if v00 == NO_VERTEX || v10 == NO_VERTEX || v11 == NO_VERTEX || v01 == NO_VERTEX
                    {
                        // Unreachable for a well-formed field — an edge that
                        // changes sign makes all four surrounding cells active —
                        // but a missing vertex must drop the quad rather than
                        // index a sentinel into the vertex buffer.
                        debug_assert!(false, "a sign-change edge had an inactive neighbour cell");
                        continue;
                    }
                    // Traversing v00 → v10 → v11 → v01 winds CCW about +axis (the
                    // right-hand rule over the two off-axis steps, which is why the
                    // off-axis pair is taken cyclically). Solid at the low corner
                    // ⇒ the surface faces +axis and that order is already outward;
                    // otherwise it is reversed.
                    if s0 < 0.0 {
                        push_quad(&mut mesh.indices, v00, v10, v11, v01);
                    } else {
                        push_quad(&mut mesh.indices, v00, v01, v11, v10);
                    }
                }
            }
        }
    }
    mesh
}

/// Push a quad as two triangles, `(q0, q1, q2)` and `(q0, q2, q3)`.
#[inline]
fn push_quad(indices: &mut Vec<u32>, q0: u32, q1: u32, q2: u32, q3: u32) {
    indices.extend_from_slice(&[q0, q1, q2, q0, q2, q3]);
}

/// Index of padded-block sample `(sx, sy, sz)`.
#[inline]
fn block_index(sx: usize, sy: usize, sz: usize) -> usize {
    (sz * PAD_SAMPLES + sy) * PAD_SAMPLES + sx
}

/// Index of padded-block cell `(cx, cy, cz)`.
#[inline]
fn cell_index(cx: usize, cy: usize, cz: usize) -> usize {
    (cz * PAD_CELLS + cy) * PAD_CELLS + cx
}

/// Block-sample indices of a cell's eight corners, indexed
/// `m = dx + 2·dy + 4·dz`.
#[inline]
fn corner_indices(cx: usize, cy: usize, cz: usize) -> [usize; 8] {
    std::array::from_fn(|m| block_index(cx + (m & 1), cy + ((m >> 1) & 1), cz + ((m >> 2) & 1)))
}

/// The twelve cube edges as `(low corner, high corner, axis)`, in a fixed order
/// (axis-major, then corner index) so the crossing average is accumulated the same
/// way every time.
const EDGES: [(usize, usize, usize); 12] = [
    (0, 1, 0),
    (2, 3, 0),
    (4, 5, 0),
    (6, 7, 0),
    (0, 2, 1),
    (1, 3, 1),
    (4, 6, 1),
    (5, 7, 1),
    (0, 4, 2),
    (1, 5, 2),
    (2, 6, 2),
    (3, 7, 2),
];

/// Whether a cell's corners straddle the surface.
#[inline]
fn crosses(s: &[f64; 8]) -> bool {
    let first = s[0] < 0.0;
    s.iter().any(|&v| (v < 0.0) != first)
}

/// The vertex offset inside a cell's own `[0, 1]³` frame: the mean of the
/// sign-change edge crossings.
///
/// Accumulated over [`EDGES`] in order and divided once, so the value is a pure,
/// order-fixed function of the eight corner distances — which is what makes two
/// chunks' copies of a seam vertex bit-identical.
fn cell_vertex_offset(s: &[f64; 8]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut count = 0.0f64;
    for &(a, b, axis) in &EDGES {
        let (sa, sb) = (s[a], s[b]);
        if (sa < 0.0) == (sb < 0.0) {
            continue;
        }
        // The crossing parameter along the edge. `sa - sb` cannot be zero here:
        // the two have different signs, and a zero is classified as "not solid",
        // so `sa == sb` would put them on the same side.
        let t = sa / (sa - sb);
        for (axis_i, out) in sum.iter_mut().enumerate() {
            let da = ((a >> axis_i) & 1) as f64;
            *out += if axis_i == axis {
                da + t * (1.0 - 2.0 * da)
            } else {
                da
            };
        }
        count += 1.0;
    }
    if count == 0.0 {
        return [0.5, 0.5, 0.5];
    }
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

/// The outward unit normal at a cell's vertex.
///
/// The **central difference over the cell's own eight corners** — which is exactly
/// the gradient of the cell's trilinear interpolant, averaged over the cell. A
/// six-tap central difference at the vertex position would need a two-voxel apron
/// (a `±½`-voxel probe reads one sample further out in each direction) and buys
/// nothing at this resolution: the field is a clamped band, so the extra reach
/// samples saturated values more often than it samples signal.
///
/// The SDF is negative inside solid, so the gradient already points *out* of the
/// surface — no negation. A degenerate (zero) gradient falls back to `+Y`, which
/// is a visible-but-harmless choice rather than a `NaN` that would poison every
/// lighting term downstream.
fn cell_normal(s: &[f64; 8]) -> [f32; 3] {
    let mut g = [0.0f64; 3];
    for (axis, out) in g.iter_mut().enumerate() {
        let bit = 1usize << axis;
        let mut acc = 0.0;
        for m in 0..8usize {
            if m & bit == 0 {
                acc += s[m | bit] - s[m];
            }
        }
        *out = acc / 4.0;
    }
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if !(len.is_finite() && len > 0.0) {
        return [0.0, 1.0, 0.0];
    }
    [
        (g[0] / len) as f32,
        (g[1] / len) as f32,
        (g[2] / len) as f32,
    ]
}

/// The material a cell's vertex carries: the material of its **most solid**
/// corner (smallest signed distance), ties broken by the lowest corner index.
///
/// "Most solid" rather than "majority" because a vertex sits on the surface of the
/// material that is actually there — a single voxel of ore poking into a wall
/// should read as ore at the point it pokes through, not be outvoted by the rock
/// around it. Deterministic either way, which is the requirement.
fn cell_material(s: &[f64; 8], corners: &[usize; 8], mat: &[u8]) -> u8 {
    let mut best: Option<(f64, u8)> = None;
    for m in 0..8usize {
        if s[m] >= 0.0 {
            continue;
        }
        let material = mat[corners[m]];
        match best {
            Some((d, _)) if s[m] >= d => {}
            _ => best = Some((s[m], material)),
        }
    }
    best.map(|(_, m)| m).unwrap_or(DEFAULT_MATERIAL)
}

// ── volume-level meshing ────────────────────────────────────────────────────

/// Every chunk key that must be meshed to cover a volume's surface: the resident
/// set **closed downward by one chunk on each axis**.
///
/// The closure is not defensive padding, it is a correctness requirement. A cell
/// owned by chunk `K` has corners in `K` *and* in `K + 1`, so a cell whose only
/// solid corner lies in `K + 1` still belongs to `K` — and if `K` is not resident
/// (an author carved a tunnel that stops exactly on a boundary, so the air-side
/// chunk was never materialized) that surface would simply be missing. Adding the
/// eight `K − {0,1}³` keys of every resident chunk covers it; the extra keys
/// almost always mesh to nothing and cost one uniform-sign early-out each.
pub fn mesh_keys_for(data: &VoxelData) -> BTreeSet<ChunkKey> {
    let mut out = BTreeSet::new();
    for &key in data.chunks().map(|(k, _)| k) {
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    out.insert(key.offset(-dx, -dy, -dz));
                }
            }
        }
    }
    out
}

/// Everything about a volume's residency that a chunk's **mesh** is a function
/// of: the greatest change stamp in the 3×3×3 chunk neighbourhood its padded
/// gather can read, **and which of those 27 neighbours are resident at all**.
///
/// ## Why the max alone is not enough (and the bug it hid)
///
/// A mesh is a function of its neighbours as well as itself, so gating a re-mesh
/// on the chunk's own stamp would leave a stale seam every time the neighbour was
/// the one that was carved. That much a max over the neighbourhood fixes. What a
/// max **cannot** see is an *eviction* of a non-maximal neighbour: with stamps
/// `A = 1`, `B = 2`, `C = 3`, evicting `B` leaves the max at `3`, the cache reports
/// "unchanged", and the mesh keeps a seam against samples that are no longer
/// resident — forever, because nothing will ever move that number again. (A
/// non-resident chunk reads as empty space, so this is a *visible* wall where
/// there is now air.)
///
/// The residency mask closes it: one bit per neighbour, set when that neighbour is
/// resident, in the fixed `dz, dy, dx` walk order below. An eviction clears a bit,
/// the key differs, the chunk re-meshes. **Cost: nothing.** The 27 probes were
/// already being made for the max, and the mask rides in a `u32` beside the `u64`
/// that was already stored — 4 bytes per cached mesh, no second walk, no extra
/// state on `VoxelData` to keep in sync.
///
/// (The considered alternative — a per-volume "residency generation" bumped on
/// every evict — was rejected: it is coarser, so *any* eviction anywhere in the
/// volume would re-mesh *every* chunk, which is precisely the thundering re-mesh
/// P21.2's streaming exists to avoid.)
///
/// A key of `MeshSourceKey::default()` means nothing in the neighbourhood is
/// resident.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshSourceKey {
    /// The greatest [`VoxelData::chunk_version`] in the 3×3×3 neighbourhood.
    pub max_chunk_version: u64,
    /// Bit `i` set ⇔ neighbour `i` of the fixed walk order is resident.
    pub resident: u32,
}

/// The [`MeshSourceKey`] of `key` in `data` — see the type docs.
pub fn source_key(data: &VoxelData, key: ChunkKey) -> MeshSourceKey {
    let mut out = MeshSourceKey::default();
    let mut bit = 0u32;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let k = key.offset(dx, dy, dz);
                out.max_chunk_version = out.max_chunk_version.max(data.chunk_version(k));
                if data.is_resident(k) {
                    out.resident |= 1 << bit;
                }
                bit += 1;
            }
        }
    }
    out
}

/// What one [`VoxelMeshCache::sync`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshSyncReport {
    /// Chunks re-meshed because something in their neighbourhood changed.
    pub remeshed: Vec<ChunkKey>,
    /// Cached meshes dropped because their chunk left the volume.
    pub dropped: Vec<ChunkKey>,
    /// Cached meshes that were already current.
    pub unchanged: usize,
}

impl MeshSyncReport {
    /// `true` when nothing was re-meshed or dropped.
    pub fn is_noop(&self) -> bool {
        self.remeshed.is_empty() && self.dropped.is_empty()
    }
}

/// A volume's meshed surface, re-meshed only where the field actually moved.
///
/// The CPU half of what a renderer needs. Both projectors hold one of these per
/// volume, [`sync`](Self::sync) it against the resident chunks, and hand the
/// non-empty meshes over — so a static cave costs one mesh pass and then nothing,
/// and a carve costs exactly the chunks whose neighbourhood the carve reached.
#[derive(Debug, Clone, Default)]
pub struct VoxelMeshCache {
    meshes: BTreeMap<ChunkKey, CachedMesh>,
}

/// One cached mesh: what it was built from, what stamp it carries, and the mesh.
#[derive(Debug, Clone)]
struct CachedMesh {
    /// The residency state this mesh is a function of — the *invalidation* key.
    source: MeshSourceKey,
    /// A **process-globally monotone** stamp minted each time the mesh is built —
    /// the *identity* a GPU buffer cache gates its upload on.
    ///
    /// Deliberately not the source key folded down to a number, and the honest
    /// reason is **width**, not history. A [`MeshSourceKey`] is a `u64` chunk
    /// version plus a 32-bit residency bitmask — 96 bits — while the DTO carries
    /// the identity as a single `u64`, so *any* fold is 96 bits into 64 and two
    /// different keys can collide by counting alone. A cache that answers "are
    /// these the bytes I already hold?" from a colliding number keeps showing the
    /// old triangles, forever, with nothing to notice it by.
    ///
    /// An earlier draft argued from a *sequence* instead — evict a neighbour,
    /// restore it, and both the mask and the max are back where they were while
    /// the mesh in between differed. That claim is false as stated, and it was
    /// falsified rather than argued away: a brute force over thousands of random
    /// evict/restore/edit steps
    /// (`a_folded_source_key_never_repeats_with_a_different_mesh`) found **zero**
    /// repeats-with-a-different-mesh, because restoring a chunk mints a fresh
    /// version and the max moves with it. Pigeonhole survives the test;
    /// storytelling did not. Same counter discipline, and the same failure it
    /// prevents, as [`VoxelData::chunk_version`].
    version: u64,
    mesh: VoxelMesh,
}

/// The single monotone source every **mesh** stamp in the process is drawn from.
/// Global for the reason [`NEXT_CHUNK_VERSION`](crate::data) is: two volumes must
/// never mint the same stamp for the same chunk key, or a key-addressed GPU cache
/// serves one volume's walls as another's after a level switch.
static NEXT_MESH_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl VoxelMeshCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Re-mesh whatever `data` says has changed, drop whatever left.
    ///
    /// Empty meshes are **kept in the cache** (they cost a key and two empty
    /// vectors) so an air chunk beside a wall is not re-meshed every frame just to
    /// discover it is still empty; [`meshes`](Self::meshes) skips them, so nothing
    /// downstream ever sees one.
    pub fn sync(&mut self, data: &VoxelData) -> MeshSyncReport {
        let wants = mesh_keys_for(data);
        let mut report = MeshSyncReport::default();

        for key in self.meshes.keys().copied().collect::<Vec<_>>() {
            if !wants.contains(&key) {
                self.meshes.remove(&key);
                report.dropped.push(key);
            }
        }
        for key in wants {
            let source = source_key(data, key);
            if self.meshes.get(&key).map(|c| c.source) == Some(source) {
                report.unchanged += 1;
                continue;
            }
            let version = NEXT_MESH_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            self.meshes.insert(
                key,
                CachedMesh {
                    source,
                    version,
                    mesh: mesh_chunk(data, key),
                },
            );
            report.remeshed.push(key);
        }
        report
    }

    /// The **non-empty** meshes, ascending by key (a deterministic draw order).
    pub fn meshes(&self) -> impl Iterator<Item = (&ChunkKey, &VoxelMesh)> {
        self.meshes
            .iter()
            .filter(|(_, c)| !c.mesh.is_empty())
            .map(|(k, c)| (k, &c.mesh))
    }

    /// The **monotone stamp** a cached mesh was built at (`0` when not cached) —
    /// the value a GPU buffer cache gates its upload on.
    ///
    /// Strictly increases on every rebuild and is unique across every cache in
    /// the process, so "same stamp" always means "same triangles" (see
    /// [`CachedMesh::version`]).
    pub fn version(&self, key: ChunkKey) -> u64 {
        self.meshes.get(&key).map(|c| c.version).unwrap_or(0)
    }

    /// The residency state a cached mesh is a function of (`None` when not
    /// cached) — the *invalidation* key, exposed so a streamer can reason about
    /// what a sync would do without running one.
    pub fn source(&self, key: ChunkKey) -> Option<MeshSourceKey> {
        self.meshes.get(&key).map(|c| c.source)
    }

    /// A cached mesh, empty ones included.
    pub fn get(&self, key: ChunkKey) -> Option<&VoxelMesh> {
        self.meshes.get(&key).map(|c| &c.mesh)
    }

    /// Total triangles across the non-empty meshes.
    pub fn triangle_count(&self) -> usize {
        self.meshes().map(|(_, m)| m.triangle_count()).sum()
    }

    /// Every cached chunk key, ascending — empty meshes included.
    ///
    /// The companion to [`meshes`](Self::meshes) for consumers that care about
    /// *bookkeeping* rather than triangles (a residency audit, or the no-op
    /// tripwire that pins every stamp across a settled sync).
    pub fn cached_keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.meshes.keys().copied()
    }

    /// Cached entries, empty meshes included.
    pub fn len(&self) -> usize {
        self.meshes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.meshes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::VoxelChunk;
    use crate::residency::chunk_range;
    use glam::DVec3;

    /// A ball of radius `r` voxels centred on global sample `c`, materialized
    /// over the chunk box `[min, max]`.
    fn ball_volume(min: ChunkKey, max: ChunkKey, c: DVec3, r: f64) -> VoxelData {
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(min, max) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    )
                    .distance(c)
                        - r
                }),
            );
        }
        v.clear_dirty();
        v
    }

    /// The batch's flagship shape: a solid slab with a horizontal tunnel bored
    /// through it — an overhang no heightfield can represent.
    fn tunnelled_slab() -> VoxelData {
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(2, 0, 0)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    let g = DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    );
                    // Solid everywhere …
                    let solid = -4.0f64;
                    // … minus a cylinder along x at (y, z) = (8, 8), radius 4.
                    let tube = 4.0 - ((g.y - 8.0).powi(2) + (g.z - 8.0).powi(2)).sqrt();
                    solid.max(tube)
                }),
            );
        }
        v.clear_dirty();
        v
    }

    #[test]
    fn a_uniform_block_meshes_to_nothing() {
        let mut air = VoxelData::new(0.5);
        air.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::empty());
        let m = mesh_chunk(&air, ChunkKey::new(0, 0, 0));
        assert!(m.is_empty());
        assert_eq!(m.vertex_count(), 0);
        assert_eq!(m.key, ChunkKey::new(0, 0, 0));

        // Solid rock surrounded by solid rock likewise has no surface in it.
        let mut rock = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(-1, -1, -1), ChunkKey::new(1, 1, 1)) {
            rock.insert_chunk(key, VoxelChunk::solid(0));
        }
        assert!(mesh_chunk(&rock, ChunkKey::new(0, 0, 0)).is_empty());
    }

    /// **Determinism, twice over**: meshing the same field twice is
    /// byte-identical, and so is meshing a volume whose chunks were inserted in
    /// the opposite order. The second is the one that catches an accidental
    /// `HashMap` or an iteration that leaked insertion order into the output.
    #[test]
    fn meshing_is_bit_identical_across_runs_and_insertion_orders() {
        let v = tunnelled_slab();
        let key = ChunkKey::new(1, 0, 0);
        let a = mesh_chunk(&v, key);
        let b = mesh_chunk(&v, key);
        assert!(!a.is_empty(), "the fixture must actually produce a surface");
        assert_eq!(a, b, "two runs over one field must be identical");

        let mut reversed = VoxelData::new(v.voxel_size_m());
        for (&k, c) in v.chunks().collect::<Vec<_>>().into_iter().rev() {
            reversed.insert_chunk(k, c.clone());
        }
        assert_eq!(
            mesh_chunk(&reversed, key),
            a,
            "chunk insertion order reached the mesh — something is iterating a hash"
        );

        // And the whole-volume walk agrees, key for key.
        let mut c1 = VoxelMeshCache::new();
        let mut c2 = VoxelMeshCache::new();
        c1.sync(&v);
        c2.sync(&reversed);
        let m1: Vec<_> = c1.meshes().map(|(k, m)| (*k, m.clone())).collect();
        let m2: Vec<_> = c2.meshes().map(|(k, m)| (*k, m.clone())).collect();
        assert_eq!(m1, m2);
        assert!(!m1.is_empty());
    }

    /// **THE WATERTIGHT GATE.** Two adjacent chunks both compute the vertices of
    /// the cells in their shared apron. Those copies must be **bit-identical** —
    /// not close, identical — or the seam is a crack, which is the exact bug ~10%
    /// of P18's meshes shipped with.
    ///
    /// Checked on all three axes, and on a field that genuinely crosses every
    /// boundary (a ball big enough to span the whole 3×3×3 box).
    #[test]
    fn seam_vertices_are_bit_identical_from_both_sides() {
        // Radius 10 about the centre chunk's middle: the sphere's SURFACE crosses
        // the x = 15, y = 15 and z = 15 boundary planes (cross-section radius
        // √(100 − 49) ≈ 7.1 there) as well as the x = y = 15 edge, so every
        // neighbour below shares a genuinely *active* apron. A radius that merely
        // swallowed the chunk would leave it entirely interior and the gate
        // vacuous — which is exactly what the `shared > 0` assertions catch.
        let v = ball_volume(
            ChunkKey::new(-1, -1, -1),
            ChunkKey::new(1, 1, 1),
            DVec3::splat(8.0),
            10.0,
        );
        let center = ChunkKey::new(0, 0, 0);
        let mine = mesh_chunk(&v, center);
        assert!(!mine.is_empty());

        let by_cell = |m: &VoxelMesh| -> BTreeMap<[i32; 3], ([f64; 3], [f32; 3], u8)> {
            m.cells
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, (m.positions[i], m.normals[i], m.materials[i])))
                .collect()
        };
        let ours = by_cell(&mine);

        let mut shared_total = 0usize;
        // The three face neighbours (a whole plane of shared apron cells) and one
        // edge neighbour (a line of them). The corner neighbour (1,1,1) shares
        // exactly ONE cell, whose activity is a property of the fixture rather
        // than of the seam policy, so it is deliberately not asserted here.
        for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)] {
            let neighbour = center.offset(dx, dy, dz);
            let theirs = by_cell(&mesh_chunk(&v, neighbour));
            let mut shared = 0usize;
            for (cell, mine_v) in &ours {
                let Some(their_v) = theirs.get(cell) else {
                    continue;
                };
                shared += 1;
                assert_eq!(
                    mine_v.0, their_v.0,
                    "cell {cell:?} got position {:?} from {center:?} and {:?} from \
                     {neighbour:?} — the seam is a crack",
                    mine_v.0, their_v.0
                );
                assert_eq!(mine_v.1, their_v.1, "cell {cell:?} normals disagree");
                assert_eq!(mine_v.2, their_v.2, "cell {cell:?} materials disagree");
            }
            assert!(
                shared > 0,
                "{center:?} and {neighbour:?} shared no apron cells — the gate is vacuous"
            );
            shared_total += shared;
        }
        assert!(shared_total > 40, "only {shared_total} shared cells");
    }

    /// The other half of watertightness: **every surface edge is claimed by
    /// exactly one chunk — no more, and NO FEWER.**
    ///
    /// Cells are partitioned across chunks, so a grid edge claimed twice
    /// double-draws and z-fights, and one claimed by nobody is a hole. The second
    /// failure is the one that matters and the one an "is anything doubled?" check
    /// cannot see, so the count is compared against an **independently derived
    /// reference**: walk the global sample grid, find every edge whose two samples
    /// differ in sign, attribute it to the chunk that owns the cell at its minimum
    /// corner, and demand the mesher's per-chunk quad counts match — chunk for
    /// chunk, not merely in total (a drop in one chunk and a double in another
    /// would balance a global sum).
    #[test]
    fn every_surface_edge_is_claimed_by_exactly_one_chunk() {
        let v = ball_volume(
            ChunkKey::new(-1, -1, -1),
            ChunkKey::new(1, 1, 1),
            DVec3::splat(8.0),
            14.0,
        );

        // ── the reference: sign-change edges, attributed by owning cell ──
        let n = CHUNK_DIM as i32;
        let (lo, hi) = (-2 * n, 2 * n);
        let mut want: BTreeMap<ChunkKey, usize> = BTreeMap::new();
        for gz in lo..hi {
            for gy in lo..hi {
                for gx in lo..hi {
                    let s0 = v.sample_global(gx, gy, gz) < 0.0;
                    for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        if (v.sample_global(gx + dx, gy + dy, gz + dz) < 0.0) != s0 {
                            // The edge belongs to the cell at its minimum corner,
                            // and that cell belongs to exactly one chunk.
                            *want.entry(ChunkKey::of_sample(gx, gy, gz)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let total_want: usize = want.values().sum();
        assert!(total_want > 2_000, "only {total_want} reference edges");

        // ── what the mesher actually emitted, per chunk ──
        let mut got: BTreeMap<ChunkKey, usize> = BTreeMap::new();
        for key in mesh_keys_for(&v) {
            let m = mesh_chunk(&v, key);
            assert_eq!(m.indices.len() % 6, 0, "{key:?} emitted a half quad");
            if !m.is_empty() {
                got.insert(key, m.indices.len() / 6);
            }
        }

        // Chunk for chunk. A drop reads as a missing (or short) entry; a double
        // reads as a long one; either fails, and the message says which chunk.
        let keys: BTreeSet<ChunkKey> = want.keys().chain(got.keys()).copied().collect();
        for key in keys {
            let (w, g) = (
                want.get(&key).copied().unwrap_or(0),
                got.get(&key).copied().unwrap_or(0),
            );
            assert_eq!(
                g, w,
                "{key:?} claimed {g} surface edges but owns {w} — a chunk that \
                 claims too few leaves a HOLE, and one that claims too many \
                 z-fights with its neighbour"
            );
        }
        assert_eq!(got.values().sum::<usize>(), total_want);
    }

    /// Every index is in range and every triangle is non-degenerate — the
    /// "nothing indexes the `NO_VERTEX` sentinel" gate.
    #[test]
    fn indices_are_in_range_and_triangles_are_real() {
        let v = tunnelled_slab();
        for key in mesh_keys_for(&v) {
            let m = mesh_chunk(&v, key);
            assert_eq!(m.indices.len() % 3, 0, "{key:?}");
            assert_eq!(m.positions.len(), m.normals.len());
            assert_eq!(m.positions.len(), m.materials.len());
            assert_eq!(m.positions.len(), m.cells.len());
            for &i in &m.indices {
                assert!(
                    (i as usize) < m.positions.len(),
                    "{key:?} index {i} past {} vertices",
                    m.positions.len()
                );
            }
            for tri in m.indices.chunks_exact(3) {
                assert!(
                    tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2],
                    "{key:?} emitted a degenerate triangle {tri:?}"
                );
            }
            for n in &m.normals {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-3, "{key:?} normal {n:?} is not unit");
            }
        }
    }

    /// Winding is **outward**: a triangle's geometric normal must agree with the
    /// SDF gradient its vertices carry, or every face of every cave would be
    /// back-face culled and the world would look inside out.
    #[test]
    fn triangle_winding_faces_outward() {
        let v = ball_volume(
            ChunkKey::new(-1, -1, -1),
            ChunkKey::new(1, 1, 1),
            DVec3::splat(8.0),
            10.0,
        );
        let mut checked = 0usize;
        for key in mesh_keys_for(&v) {
            let m = mesh_chunk(&v, key);
            for tri in m.indices.chunks_exact(3) {
                let p: Vec<DVec3> = tri
                    .iter()
                    .map(|&i| DVec3::from_array(m.positions[i as usize]))
                    .collect();
                let geo = (p[1] - p[0]).cross(p[2] - p[0]);
                if geo.length() < 1e-9 {
                    continue;
                }
                let shading: DVec3 = tri
                    .iter()
                    .map(|&i| {
                        let n = m.normals[i as usize];
                        DVec3::new(n[0] as f64, n[1] as f64, n[2] as f64)
                    })
                    .sum();
                assert!(
                    geo.normalize().dot(shading.normalize()) > 0.0,
                    "{key:?} wound a triangle inward: geometric {:?} vs shading {:?}",
                    geo.normalize(),
                    shading.normalize()
                );
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} triangles checked");
    }

    /// A tunnel bored through a slab produces surface with normals pointing in
    /// **every** direction — the property a heightfield cannot have, and the one
    /// this whole phase exists for.
    #[test]
    fn a_tunnel_produces_an_overhang_a_heightfield_cannot() {
        let v = tunnelled_slab();
        let mut cache = VoxelMeshCache::new();
        let r = cache.sync(&v);
        assert!(!r.remeshed.is_empty());
        assert!(cache.triangle_count() > 500, "{}", cache.triangle_count());

        let mut up = 0usize;
        let mut down = 0usize;
        for (_, m) in cache.meshes() {
            for n in &m.normals {
                if n[1] > 0.7 {
                    up += 1;
                }
                if n[1] < -0.7 {
                    down += 1;
                }
            }
        }
        assert!(
            up > 20 && down > 20,
            "a bored tunnel must have both a floor ({up} up-facing) and a ROOF \
             ({down} down-facing) — a heightfield has only the former"
        );
    }

    /// Materials come out of the field, per vertex, from the most-solid corner.
    #[test]
    fn vertex_materials_follow_the_most_solid_corner() {
        let mut v = ball_volume(
            ChunkKey::new(0, 0, 0),
            ChunkKey::new(0, 0, 0),
            DVec3::splat(8.0),
            5.0,
        );
        // Paint the whole ball chunk material 2.
        {
            let c = v.get_chunk_mut(ChunkKey::new(0, 0, 0)).unwrap();
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        c.set_material(i, j, k, 2);
                    }
                }
            }
        }
        let m = mesh_chunk(&v, ChunkKey::new(0, 0, 0));
        assert!(!m.is_empty());
        assert!(
            m.materials.iter().all(|&x| x == 2),
            "a uniformly-painted ball must mesh to one material"
        );
    }

    /// The mesh set is the resident set **closed downward**, because a cell owned
    /// by a non-resident chunk can still be the only place a surface appears.
    #[test]
    fn the_mesh_set_closes_the_resident_set_downward() {
        let mut v = VoxelData::new(0.5);
        v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(0));
        let keys = mesh_keys_for(&v);
        assert_eq!(keys.len(), 8);
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    assert!(keys.contains(&ChunkKey::new(-dx, -dy, -dz)));
                }
            }
        }
        // …and the closure earns its keep. The −x face of a lone solid chunk lies
        // on the edges whose minimum corner is global sample (−1, y, z), so the
        // cells that own those edges are (−1, y, z) — owned by the **absent**
        // chunk (−1, 0, 0). Without the downward closure that whole face would
        // simply not be meshed by anyone.
        assert!(
            !mesh_chunk(&v, ChunkKey::new(-1, 0, 0)).is_empty(),
            "the −x face would be missing without the downward closure"
        );
        assert!(
            !mesh_chunk(&v, ChunkKey::new(0, -1, 0)).is_empty(),
            "−y face"
        );
        assert!(
            !mesh_chunk(&v, ChunkKey::new(0, 0, -1)).is_empty(),
            "−z face"
        );
        // The chunk itself owns the three + faces, so it is not empty either.
        assert!(!mesh_chunk(&v, ChunkKey::new(0, 0, 0)).is_empty());
    }

    /// The cache re-meshes exactly what moved, and nothing else.
    #[test]
    fn the_cache_remeshes_only_what_the_carve_reached() {
        let mut v = tunnelled_slab();
        let mut cache = VoxelMeshCache::new();
        let first = cache.sync(&v);
        assert!(first.remeshed.len() >= v.chunk_count());
        assert_eq!(first.unchanged, 0);
        let tris = cache.triangle_count();
        assert!(tris > 0);

        // A second sync with nothing changed is a no-op …
        let again = cache.sync(&v);
        assert!(again.is_noop(), "{again:?}");
        assert_eq!(again.unchanged, cache.len());

        // … and, said in the currency a GPU cache actually spends: **no mesh
        // stamp moves**. `is_noop` is a report-level claim; a rebuild that
        // produced identical triangles would still mint a new stamp and still
        // report zero re-meshes under a sloppier implementation, and every
        // consumer would re-upload the whole volume every frame. Pinned across
        // several consecutive syncs so a stamp that only moves every other pass
        // cannot hide either.
        let settled: Vec<(ChunkKey, u64)> =
            cache.cached_keys().map(|k| (k, cache.version(k))).collect();
        assert!(!settled.is_empty());
        for pass in 0..4 {
            assert!(cache.sync(&v).is_noop(), "pass {pass}");
            for &(key, stamp) in &settled {
                assert_eq!(
                    cache.version(key),
                    stamp,
                    "{key:?} re-stamped on no-op sync pass {pass}"
                );
            }
        }
        assert_eq!(cache.triangle_count(), tris);

        // … and touching ONE chunk re-meshes only its 3×3×3 neighbourhood.
        let touched = ChunkKey::new(1, 0, 0);
        v.get_chunk_mut(touched).unwrap().set_sample(8, 8, 8, -4.0);
        let after = cache.sync(&v);
        assert!(!after.remeshed.is_empty());
        for &k in &after.remeshed {
            let d = [
                (k.x - touched.x).abs(),
                (k.y - touched.y).abs(),
                (k.z - touched.z).abs(),
            ];
            assert!(
                d[0] <= 1 && d[1] <= 1 && d[2] <= 1,
                "{k:?} re-meshed but cannot see {touched:?}"
            );
        }

        // Removing a chunk drops the meshes that are no longer wanted.
        let before_len = cache.len();
        for k in v.resident_keys() {
            v.remove_chunk(k);
        }
        let empty = cache.sync(&v);
        assert_eq!(empty.dropped.len(), before_len);
        assert!(cache.is_empty());
        assert_eq!(cache.triangle_count(), 0);
        cache.clear();
        assert_eq!(cache.version(touched), 0);
        assert!(cache.get(touched).is_none());
    }

    /// **THE EVICTION REGRESSION.** A max over the neighbourhood cannot see a
    /// non-maximal neighbour leaving, and the first cut of this cache used exactly
    /// that as its whole key.
    ///
    /// Stamps `A < B < C` in one neighbourhood: evict `B` and the max is still
    /// `C`, so the old key reported "unchanged" and the chunk kept a seam meshed
    /// against samples that are no longer resident — permanently, since nothing
    /// will ever move that number again, and visibly, since a non-resident chunk
    /// reads as empty space. The residency mask is what closes it.
    ///
    /// Asserted three ways, because each catches a different way of getting the
    /// fix wrong: the **key** must differ, the **sync** must report the re-mesh,
    /// and the exposed **version** (what a GPU buffer cache gates on) must move.
    #[test]
    fn evicting_a_non_maximal_neighbour_remeshes_the_chunks_that_could_see_it() {
        let mut v = VoxelData::new(0.5);
        // Three chunks in one neighbourhood, stamped in ascending order so the
        // middle one is provably NOT the max.
        let (a, b, c) = (
            ChunkKey::new(0, 0, 0),
            ChunkKey::new(1, 0, 0),
            ChunkKey::new(2, 0, 0),
        );
        for key in [a, b, c] {
            let base = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, _, _| ((base[0] + i as i32) as f64) - 20.0),
            );
        }
        v.clear_dirty();
        assert!(
            v.chunk_version(a) < v.chunk_version(b) && v.chunk_version(b) < v.chunk_version(c),
            "the fixture must stamp them in ascending order"
        );

        let mut cache = VoxelMeshCache::new();
        cache.sync(&v);
        // The OBSERVER is the middle chunk `b`: its 3×3×3 neighbourhood is the
        // only one that contains all three, so `a` is in it and is provably not
        // its max — `c` is.
        let before_key = cache.source(b).expect("b is cached");
        let before_version = cache.version(b);
        assert_eq!(before_key.max_chunk_version, v.chunk_version(c));
        assert!(
            cache.sync(&v).is_noop(),
            "a settled volume re-meshes nothing"
        );

        // Evict the NON-MAXIMAL neighbour. The max does not move.
        assert!(v.evict_chunk(a));
        let after_key = source_key(&v, b);
        assert_eq!(
            after_key.max_chunk_version, before_key.max_chunk_version,
            "the fixture is pointless unless the max really is unchanged"
        );
        assert_ne!(
            after_key, before_key,
            "the source key did not notice a neighbour leaving — a max alone              cannot, which is the entire reason for the residency mask"
        );

        let report = cache.sync(&v);
        assert!(
            report.remeshed.contains(&b),
            "chunk {b:?} kept a seam against a chunk that is no longer resident: {report:?}"
        );
        assert!(
            cache.version(b) > before_version,
            "the mesh stamp did not move, so a GPU cache would keep the stale buffers"
        );
        // The observer must actually HAVE a surface, before and after. A fixture
        // that drifted to a uniform field would still satisfy every assertion
        // above — empty meshes re-stamp just fine — and the test would silently
        // degrade from "the seam is re-meshed" to "the bookkeeping is re-stamped",
        // which is not what it is here to prove.
        assert!(
            !cache.get(b).map(VoxelMesh::is_empty).unwrap_or(true),
            "the observer's mesh is empty, so this test proves nothing about seams"
        );

        // …and a chunk that could NOT see the eviction is left alone: `c`'s
        // neighbourhood spans x ∈ [1, 3], which never contained `a`.
        assert!(
            !report.remeshed.contains(&c),
            "{c:?} re-meshed over an eviction it cannot see — the invalidation is              not supposed to be a thundering herd: {report:?}"
        );
    }

    /// **The falsification.** The monotone stamp's doc used to justify itself with
    /// a *sequence* story: evict a neighbour, restore it, and both the residency
    /// mask and the max version are back where they were while the mesh in
    /// between differed — so a folded key would alias. This brute-forces that
    /// claim over thousands of random evict / restore / carve steps and counts the
    /// counterexamples.
    ///
    /// The count is **zero**, and structurally so: restoring a chunk draws a fresh
    /// stamp from the process-global counter, which can only raise a
    /// neighbourhood's max, so a key that has once moved never comes back. The
    /// doc now argues from **width** instead (96 key bits into a 64-bit DTO field
    /// cannot be injective), which needs no sequence at all.
    ///
    /// The test is kept, and kept asserting zero, for the reason the correction
    /// existed: if this ever fires, the old story was right after all and the doc
    /// owes it a line — and until then, nobody re-derives a justification that was
    /// already measured false.
    #[test]
    fn a_folded_source_key_never_repeats_with_a_different_mesh() {
        use std::collections::BTreeMap;

        let mut v = tunnelled_slab();
        let observer = ChunkKey::new(1, 0, 0);
        let neighbours = [ChunkKey::new(0, 0, 0), ChunkKey::new(2, 0, 0)];
        // Every key we have ever seen → the mesh that was live when we saw it.
        let mut seen: BTreeMap<(u64, u32), VoxelMesh> = BTreeMap::new();
        let mut evicted: BTreeMap<ChunkKey, VoxelChunk> = BTreeMap::new();
        let mut aliases = 0usize;
        let mut revisits = 0usize;

        // A tiny deterministic LCG — no dev-dependency, and the sequence is part
        // of the record: a run that fails is a run that can be replayed.
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for step in 0..4000u32 {
            match next() % 3 {
                0 => {
                    // Evict a resident neighbour.
                    let k = neighbours[(next() % 2) as usize];
                    if let Some(c) = v.get_chunk(k).cloned() {
                        if v.evict_chunk(k) {
                            evicted.insert(k, c);
                        }
                    }
                }
                1 => {
                    // Restore an evicted neighbour, byte-identically.
                    let k = neighbours[(next() % 2) as usize];
                    if let Some(c) = evicted.remove(&k) {
                        v.insert_chunk(k, c);
                    }
                }
                _ => {
                    // Change the observer's own contents: flip one interior
                    // sample's sign, which moves the surface for real.
                    let b = observer.base_sample();
                    let d = if step % 2 == 0 { 3.0 } else { -3.0 };
                    v.set_sample_global(b[0] + 8, b[1] + 8, b[2] + 8, d as f32);
                }
            }
            let key = source_key(&v, observer);
            let mesh = mesh_chunk(&v, observer);
            let folded = (key.max_chunk_version, key.resident);
            if let Some(prev) = seen.get(&folded) {
                revisits += 1;
                if *prev != mesh {
                    aliases += 1;
                }
            } else {
                seen.insert(folded, mesh);
            }
        }

        assert_eq!(
            aliases, 0,
            "the sequence story is true after all: a source key repeated while the              mesh differed. Put the eviction/restore justification back on              CachedMesh::version, beside the width one."
        );
        // The brute force has to actually revisit keys, or "zero aliases" is
        // vacuous — the vacuous-checks lesson from P19, applied to a negative
        // result.
        assert!(
            revisits > 0,
            "no key was ever seen twice, so this proves nothing"
        );
    }

    /// The mesh stamp is **monotone and unique across caches**, so "same stamp"
    /// always means "same triangles" — the property a GPU buffer cache keyed by
    /// `(volume, chunk)` actually needs, and one a folded source key could not
    /// provide: the key is 96 bits (a `u64` version + a 32-bit residency mask) and
    /// a DTO carries 64, so folding is lossy by counting. (The *sequence* argument
    /// this doc used to make instead — "the key repeats while the mesh differs" —
    /// is empirically false; `a_folded_source_key_never_repeats_with_a_different_mesh`
    /// is kept below precisely so the stated reason stays the true one.)
    #[test]
    fn mesh_stamps_are_monotone_and_unique_across_caches() {
        let v = tunnelled_slab();
        let key = ChunkKey::new(1, 0, 0);

        let mut a = VoxelMeshCache::new();
        let mut b = VoxelMeshCache::new();
        a.sync(&v);
        b.sync(&v);
        assert!(a.version(key) > 0 && b.version(key) > 0);
        assert_ne!(
            a.version(key),
            b.version(key),
            "two caches minted the same stamp for one chunk key — a key-addressed              GPU cache would serve one volume's walls as the other's"
        );
        assert!(b.version(key) > a.version(key));
        assert_eq!(
            a.version(ChunkKey::new(99, 99, 99)),
            0,
            "uncached ⇒ no stamp"
        );

        // A round trip through the same residency state re-mints rather than
        // reusing: the mesh was rebuilt, so the identity must be new.
        let mut v2 = v.clone();
        let neighbour = ChunkKey::new(0, 0, 0);
        let held = v2.get_chunk(neighbour).unwrap().clone();
        let first = a.version(key);
        assert!(v2.evict_chunk(neighbour));
        a.sync(&v2);
        let mid = a.version(key);
        assert!(mid > first);
        v2.insert_resident_chunk(neighbour, held);
        a.sync(&v2);
        assert!(
            a.version(key) > mid,
            "restoring a neighbour must mint a NEW stamp, not resurrect the old one"
        );
    }

    /// The shared rebase helper both projectors use: chunk-local metres, with the
    /// chunk's `f64` world origin carrying the rest.
    #[test]
    fn local_positions_rebase_into_chunk_local_metres() {
        let v = tunnelled_slab();
        let key = ChunkKey::new(1, 0, 0);
        let m = mesh_chunk(&v, key);
        assert!(!m.is_empty());
        let local = m.local_positions_m(0.5);
        assert_eq!(local.len(), m.vertex_count());
        let base = key.base_sample();
        for (i, l) in local.iter().enumerate() {
            for a in 0..3 {
                let want = ((m.positions[i][a] - base[a] as f64) * 0.5) as f32;
                assert_eq!(l[a], want);
            }
        }
        let (lo, hi) = m.local_bounds_m(0.5);
        for a in 0..3 {
            assert!(lo[a] <= hi[a]);
            for l in &local {
                assert!(l[a] >= lo[a] && l[a] <= hi[a]);
            }
        }
        // The apron reaches at most one voxel outside the chunk's own span.
        let span = CHUNK_DIM as f32 * 0.5;
        assert!(
            lo[0] >= -0.5 - 1e-4 && hi[0] <= span + 0.5 + 1e-4,
            "{lo:?} {hi:?}"
        );
        assert_eq!(
            VoxelMesh::empty(key).local_bounds_m(0.5),
            ([0.0; 3], [0.0; 3])
        );
    }
}
