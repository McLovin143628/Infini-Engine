//! The `.inf_vmesh` data model: a meshlet DAG for Nanite-class virtualized
//! geometry.
//!
//! A [`VgeomMesh`] is the **cook-derived** render form of an authoring
//! [`MeshAsset`](../../inf_mesh/struct.MeshAsset.html): the source mesh stays
//! authoring-clean, and the cook derives this optimized, clusterized,
//! level-of-detail DAG beside it (see the crate docs for the decision).
//!
//! # The cut invariant (the load-bearing property)
//!
//! Every meshlet stores two errors:
//!
//! * [`Meshlet::error`] — the object-space error *of this meshlet's geometry*
//!   (0 for the finest, LOD-0 meshlets; the error of the group that produced it
//!   otherwise);
//! * [`Meshlet::parent_error`] — the error of the group this meshlet feeds into
//!   (i.e. the error you would incur by drawing its coarser replacement).
//!   `+∞` for a root meshlet that has no coarser replacement.
//!
//! The runtime LOD selection is then a per-meshlet screen-space-error cut with
//! **no pointer chasing**:
//!
//! ```text
//!     draw meshlet  iff  error ≤ threshold < parent_error
//! ```
//!
//! This works because of two construction guarantees:
//!
//! 1. **Monotonicity** — for every meshlet `error < parent_error` *strictly*
//!    (group errors increase up the DAG; ties are broken by a positive epsilon
//!    bump so an interval is never empty).
//! 2. **Shared boundaries** — a group `G` is *both* the group its child meshlets
//!    feed into (so each child's `parent_error == error(G)`) *and* the producer
//!    of the coarser meshlets that replace them (so each parent's
//!    `error == error(G)`). The half-open interval `[error, parent_error)` of the
//!    meshlets along any root-to-leaf path therefore tiles `[0, +∞)` with no gap
//!    and no overlap — so for *any* threshold the cut selects **exactly one**
//!    meshlet per path, yielding a complete, non-overlapping surface.
//!
//! [`VgeomMesh::select`] evaluates the cut; the
//! `cut_invariant_holds_at_every_threshold` test pins the property by asserting
//! the selected surface stays watertight (crack-free, hole-free) at every
//! threshold.
//!
//! # Streaming shape (next wave)
//!
//! Meshlets are laid out **level-major, coarsest first** in [`VgeomMesh::meshlets`]
//! (and [`VgeomMesh::levels`] lists the ranges coarse→fine). A future range-request
//! streamer can therefore load the coarse roots first and refine by appending
//! finer levels — a monotone forward read through the payload. Actual paging is
//! P13.1 deliverable 2 (next wave); this module only fixes the *order*.

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// One vertex of a [`VgeomMesh`]. `#[repr(C)]` + `Pod` so it uploads straight to
/// a GPU vertex buffer and feeds `meshopt` (position in the first 12 bytes)
/// without a copy. 32 bytes, naturally aligned.
///
/// v1 stores full-precision `f32` position/normal/uv. Quantized positions
/// (Nanite packs positions to a per-cluster grid) are a documented follow-up —
/// the schema version gates the upgrade.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct VgeomVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Default for VgeomVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
        }
    }
}

/// One meshlet: a small cluster (~64 vertices / ~124 triangles) with a micro
/// index buffer into the shared vertex list, plus culling bounds and its place
/// in the LOD DAG.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Meshlet {
    /// Offset into [`VgeomMesh::meshlet_vertices`] where this meshlet's vertex
    /// index list begins.
    pub vertex_offset: u32,
    /// Number of vertex indices this meshlet references.
    pub vertex_count: u32,
    /// Offset into [`VgeomMesh::meshlet_triangles`] where this meshlet's local
    /// triangle indices begin (`3 × triangle_count` bytes follow).
    pub triangle_offset: u32,
    /// Number of triangles in this meshlet.
    pub triangle_count: u32,

    /// Bounding-sphere center (local space).
    pub center: [f32; 3],
    /// Bounding-sphere radius.
    pub radius: f32,
    /// Normal-cone axis for backface culling (`dot(view, axis) ≥ cone_cutoff`
    /// rejects the meshlet — see `meshopt::compute_meshlet_bounds`).
    pub cone_axis: [f32; 3],
    /// Normal-cone cutoff. `1` means a degenerate (unusable) cone.
    pub cone_cutoff: f32,

    /// Index into [`VgeomMesh::groups`] of the group this meshlet **feeds into**
    /// (its parent group), or [`Meshlet::NO_GROUP`] for a root meshlet.
    pub group: u32,
    /// LOD level: 0 = finest (original geometry), increasing = coarser.
    pub lod_level: u8,
    /// Object-space error of *this* meshlet's geometry. 0 at LOD 0.
    pub error: f32,
    /// Object-space error of the coarser replacement (the group's error), or
    /// [`f32::INFINITY`] for a root meshlet with no replacement.
    pub parent_error: f32,
}

impl Meshlet {
    /// Sentinel [`Meshlet::group`] value for a root meshlet (no parent group).
    pub const NO_GROUP: u32 = u32::MAX;

    /// A root meshlet has no coarser replacement (`parent_error == +∞`).
    pub fn is_root(&self) -> bool {
        self.group == Self::NO_GROUP
    }

    /// The cut test: is this meshlet the one to draw at `threshold`?
    /// `error ≤ threshold < parent_error`.
    pub fn selected_at(&self, threshold: f32) -> bool {
        self.error <= threshold && threshold < self.parent_error
    }
}

/// A group of adjacent meshlets simplified together into the next coarser level.
///
/// A group is the unit of simplification: its member meshlets (those whose
/// [`Meshlet::group`] points here) are merged, simplified ~50% with their shared
/// outer boundary locked, and re-clustered into the produced meshlets
/// (`produced_start .. produced_start + produced_count`) one level coarser. The
/// group's [`Group::error`] is the shared cut boundary (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Group {
    /// LOD level of this group's **input** meshlets (it produces level
    /// `input_level + 1`).
    pub input_level: u8,
    /// Monotone object-space group error — the shared boundary value between the
    /// child meshlets' `parent_error` and the produced meshlets' `error`.
    pub error: f32,
    /// Range of meshlets this group produced (its coarser output), as
    /// `[start, start + count)` into [`VgeomMesh::meshlets`]. A group exists only
    /// when its simplification made progress, so `count ≥ 1`; regions that could
    /// not coarsen create no group and their meshlets are roots instead.
    pub produced_start: u32,
    /// Number of produced meshlets (`≥ 1`).
    pub produced_count: u32,
    /// Parent **group** indices: the coarser groups this group's produced
    /// meshlets feed into (the DAG edges at group granularity), sorted &
    /// deduplicated. Empty when the produced meshlets are roots.
    ///
    /// (v1: a plain `Vec`; the roadmap's "SmallVec-ish" is a later allocation
    /// tweak — most groups have a handful of parents.)
    pub parents: Vec<u32>,
}

/// One discrete **classic LOD level** extracted from the meshlet DAG (P13.4):
/// a self-contained triangle-list index buffer into [`VgeomMesh::vertices`]
/// gathering every meshlet at one LOD level, plus that level's representative
/// object-space error.
///
/// This is the **classic-LOD fallback** the roadmap requires for GPUs without
/// the virtualized-geometry (meshlet compute) path: the *same* geometry the
/// meshlet cut would select at that level, but drawn as an ordinary indexed
/// mesh through the standard PBR pipeline. The vertex buffer is shared with the
/// meshlet path (no duplication) — only the per-level index buffers differ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassicLod {
    /// LOD level number (0 = finest / original geometry).
    pub lod_level: u8,
    /// Representative object-space error of drawing this **whole** level — the
    /// max meshlet error across the level (0 at LOD 0). Increases with
    /// coarseness, so the [`pick`](VgeomMesh::pick_classic_lod) is monotone.
    pub error: f32,
    /// Triangle-list indices into [`VgeomMesh::vertices`] (3 per triangle).
    pub indices: Vec<u32>,
}

impl ClassicLod {
    /// Triangles in this level's index buffer.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// The meshlet index range of one LOD level, in the payload's coarse→fine order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelRange {
    /// The LOD level number (0 = finest). Levels appear in [`VgeomMesh::levels`]
    /// coarsest-first, so `lod_level` **descends** down the vector.
    pub lod_level: u8,
    /// First meshlet of this level in [`VgeomMesh::meshlets`].
    pub meshlet_start: u32,
    /// Number of meshlets at this level.
    pub meshlet_count: u32,
}

/// The `.inf_vmesh` payload: a full meshlet LOD DAG for one mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VgeomMesh {
    pub schema_version: u32,
    /// Shared, welded vertex buffer. Meshlets index into this via
    /// [`meshlet_vertices`](Self::meshlet_vertices).
    pub vertices: Vec<VgeomVertex>,
    /// All meshlets, **coarsest level first** (streaming order — see module docs).
    pub meshlets: Vec<Meshlet>,
    /// Concatenated per-meshlet vertex index lists (into [`vertices`](Self::vertices)).
    pub meshlet_vertices: Vec<u32>,
    /// Concatenated per-meshlet local triangle indices (each in
    /// `0..vertex_count`), 3 per triangle.
    pub meshlet_triangles: Vec<u8>,
    /// All groups (simplification units) forming the DAG.
    pub groups: Vec<Group>,
    /// LOD level ranges into [`meshlets`](Self::meshlets), coarsest first.
    pub levels: Vec<LevelRange>,
    /// Whole-mesh bounding-sphere center (local space).
    pub center: [f32; 3],
    /// Whole-mesh bounding-sphere radius.
    pub radius: f32,
}

impl VgeomMesh {
    /// Current schema version. v1 = f32 vertices, per-group greedy adjacency
    /// grouping, monotone accumulated error.
    pub const CURRENT_VERSION: u32 = 1;

    /// Number of meshlets across all levels.
    pub fn meshlet_count(&self) -> usize {
        self.meshlets.len()
    }

    /// Number of LOD levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Total triangles across every level (sum over meshlets).
    pub fn total_triangles(&self) -> usize {
        self.meshlets
            .iter()
            .map(|m| m.triangle_count as usize)
            .sum()
    }

    /// The global (position) vertex indices of meshlet `i`'s triangle `t` as a
    /// `[u32; 3]`, resolving the local micro-indices through
    /// [`meshlet_vertices`](Self::meshlet_vertices). Panics on out-of-range
    /// indices (a corrupt payload).
    pub fn triangle(&self, meshlet: usize, tri: usize) -> [u32; 3] {
        let m = &self.meshlets[meshlet];
        let vbase = m.vertex_offset as usize;
        let tbase = m.triangle_offset as usize + tri * 3;
        let v = |k: usize| {
            let local = self.meshlet_triangles[tbase + k] as usize;
            self.meshlet_vertices[vbase + local]
        };
        [v(0), v(1), v(2)]
    }

    /// The meshlets selected by the cut at `threshold` — the LOD-selected draw
    /// set (`error ≤ threshold < parent_error`). This is exactly the operation
    /// the next-wave GPU LOD-selection compute pass performs per meshlet; it is
    /// exposed here so the offline builder's invariant can be tested against the
    /// same rule the runtime will apply.
    pub fn select(&self, threshold: f32) -> impl Iterator<Item = (usize, &Meshlet)> {
        self.meshlets
            .iter()
            .enumerate()
            .filter(move |(_, m)| m.selected_at(threshold))
    }

    /// Extract the discrete **classic-LOD chain** from the meshlet levels
    /// (P13.4), **finest first** (LOD 0 at index 0). Each [`ClassicLod`] is the
    /// union of one level's meshlet triangles resolved to global vertex indices —
    /// a plain indexed draw that reproduces that LOD level exactly. Shares
    /// [`vertices`](Self::vertices) (no vertex duplication). Deterministic.
    ///
    /// This is the **classic-LOD fallback** for GPUs without the virtualized-
    /// geometry (meshlet compute) path: the same geometry the meshlet cut selects
    /// at each level, drawn through the ordinary PBR mesh pipeline. Because both
    /// paths key off the level errors, a whole-mesh classic pick
    /// ([`pick_classic_lod`](Self::pick_classic_lod)) tracks the same
    /// screen-error metric as the meshlet cut — true classic-LOD parity.
    pub fn classic_lods(&self) -> Vec<ClassicLod> {
        let mut out: Vec<ClassicLod> = self
            .levels
            .iter()
            .map(|lv| {
                let start = lv.meshlet_start as usize;
                let end = start + lv.meshlet_count as usize;
                let mut indices = Vec::new();
                let mut error = 0.0f32;
                for mi in start..end {
                    let m = &self.meshlets[mi];
                    error = error.max(m.error);
                    for t in 0..m.triangle_count as usize {
                        indices.extend_from_slice(&self.triangle(mi, t));
                    }
                }
                ClassicLod {
                    lod_level: lv.lod_level,
                    error,
                    indices,
                }
            })
            .collect();
        // `levels` is coarsest-first; emit finest (LOD 0) first for the pick.
        out.sort_by_key(|l| l.lod_level);
        out
    }

    /// The per-level representative errors, **finest first** — the cheap input to
    /// [`pick_classic_lod`](Self::pick_classic_lod) (avoids rebuilding the full
    /// index buffers just to select a level). `errors[i]` is the max meshlet
    /// error of the `i`-th finest level (`errors[0] == 0`, ascending).
    pub fn classic_lod_errors(&self) -> Vec<f32> {
        let mut per_level: Vec<(u8, f32)> = self
            .levels
            .iter()
            .map(|lv| {
                let start = lv.meshlet_start as usize;
                let end = start + lv.meshlet_count as usize;
                let error = self.meshlets[start..end]
                    .iter()
                    .map(|m| m.error)
                    .fold(0.0f32, f32::max);
                (lv.lod_level, error)
            })
            .collect();
        per_level.sort_by_key(|(l, _)| *l);
        per_level.into_iter().map(|(_, e)| e).collect()
    }

    /// Pick the classic-LOD **index** (finest-first, into
    /// [`classic_lods`](Self::classic_lods) / [`classic_lod_errors`](Self::classic_lod_errors))
    /// for an object-space screen-error `threshold`: the **coarsest** level whose
    /// representative error stays within tolerance (fewest triangles). LOD 0
    /// (error 0) always qualifies, so the result is defined for every
    /// `threshold ≥ 0`. Monotone — a larger threshold (a farther/smaller
    /// instance) never picks a finer level. The classic twin of the meshlet cut
    /// `error ≤ t < parent_error`.
    pub fn pick_classic_lod(&self, threshold: f32) -> usize {
        pick_classic_level(&self.classic_lod_errors(), threshold)
    }
}

/// Pick the coarsest classic-LOD index whose representative `error ≤ threshold`,
/// given the ascending, finest-first per-level `errors`. Returns 0 (finest) when
/// `errors` is empty. Shared by [`VgeomMesh::pick_classic_lod`] and the renderer's
/// per-instance classic pick so both select identically.
pub fn pick_classic_level(errors: &[f32], threshold: f32) -> usize {
    let mut pick = 0;
    for (i, &e) in errors.iter().enumerate() {
        if e <= threshold {
            pick = i;
        }
    }
    pick
}

impl AssetPayload for VgeomMesh {
    const KIND: AssetKind = AssetKind::MeshletMesh;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod classic_lod_tests {
    use crate::build::{build_vgeom, BuildParams};
    use crate::model::pick_classic_level;

    /// A dense displaced grid → a multi-level meshlet DAG (mirrors the cook input).
    fn dense(n: usize) -> crate::model::VgeomMesh {
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
                indices.extend_from_slice(&[
                    a,
                    a + stride,
                    a + 1,
                    a + 1,
                    a + stride,
                    a + stride + 1,
                ]);
            }
        }
        build_vgeom(&positions, &normals, &uvs, &indices, BuildParams::default())
    }

    #[test]
    fn classic_lods_cover_each_level_finest_first() {
        let m = dense(40);
        let lods = m.classic_lods();
        assert!(lods.len() >= 2, "expected a multi-level DAG");
        // Finest first, ascending LOD number + ascending error.
        for w in lods.windows(2) {
            assert!(w[0].lod_level < w[1].lod_level);
            assert!(w[0].error <= w[1].error);
        }
        assert_eq!(lods[0].lod_level, 0);
        assert_eq!(lods[0].error, 0.0, "LOD 0 has zero error");
        // Every LOD-0 triangle count equals the sum of LOD-0 meshlet triangles.
        let lod0_meshlet_tris: usize = m
            .levels
            .iter()
            .find(|lv| lv.lod_level == 0)
            .map(|lv| {
                let s = lv.meshlet_start as usize;
                let e = s + lv.meshlet_count as usize;
                m.meshlets[s..e]
                    .iter()
                    .map(|ml| ml.triangle_count as usize)
                    .sum()
            })
            .unwrap();
        assert_eq!(lods[0].triangle_count(), lod0_meshlet_tris);
        // Coarser levels are cheaper than the finest.
        assert!(lods.last().unwrap().triangle_count() < lods[0].triangle_count());
        // Every index is in range.
        let nverts = m.vertices.len() as u32;
        for l in &lods {
            assert!(l.indices.iter().all(|&i| i < nverts));
        }
    }

    #[test]
    fn pick_is_monotone_far_picks_coarser() {
        let m = dense(40);
        let errors = m.classic_lod_errors();
        // Near (tiny threshold) → finest.
        assert_eq!(m.pick_classic_lod(0.0), 0);
        // Far (huge threshold) → coarsest.
        assert_eq!(m.pick_classic_lod(f32::MAX), errors.len() - 1);
        // Monotone non-decreasing in threshold.
        let mut prev = 0usize;
        let mut t = 0.0f32;
        while t < errors.last().copied().unwrap_or(1.0) * 2.0 + 1.0 {
            let p = m.pick_classic_lod(t);
            assert!(p >= prev, "pick must not go finer as threshold grows");
            prev = p;
            t += 0.001;
        }
        // A far camera picks a strictly coarser level than a near one.
        assert!(m.pick_classic_lod(f32::MAX) > m.pick_classic_lod(0.0));
    }

    #[test]
    fn pick_classic_level_edge_cases() {
        assert_eq!(pick_classic_level(&[], 5.0), 0);
        assert_eq!(pick_classic_level(&[0.0, 1.0, 2.0], 0.0), 0);
        assert_eq!(pick_classic_level(&[0.0, 1.0, 2.0], 1.5), 1);
        assert_eq!(pick_classic_level(&[0.0, 1.0, 2.0], 100.0), 2);
    }
}
