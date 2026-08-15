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
//! # Streaming shape (P18.2 — landed)
//!
//! Meshlets are laid out **level-major, coarsest first** in [`VgeomMesh::meshlets`]
//! (and [`VgeomMesh::levels`] lists the ranges coarse→fine). This module fixes the
//! *order*; [`crate::asset`] turns it into byte ranges (a paged `.inf_vmesh`
//! container sliced straight out of an mmap'd pack) and [`crate::stream`] decides
//! which pages are in VRAM.
//!
//! Two rules of that design reach back into this module:
//!
//! * the paging unit is a **page**, not a LOD level, because roots live at many
//!   levels (a group that fails to simplify leaves its members roots wherever
//!   they are). Page 0 is every root and is never evicted, which is what makes
//!   "never a hole" structural — see [`crate::asset`];
//! * the cut clamps to residency through
//!   [`VgeomMesh::select_with_residency`] / [`Meshlet::selected_at_clamped`],
//!   which reduce to [`VgeomMesh::select`] / [`Meshlet::selected_at`] exactly at
//!   full residency.

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// One vertex of a [`VgeomMesh`]. `#[repr(C)]` + `Pod` so it uploads straight to
/// a GPU vertex buffer and feeds `meshopt` (position in the first 12 bytes)
/// without a copy. **36 bytes** since P28.2, naturally aligned.
///
/// v1 stores full-precision `f32` position/normal/uv. Quantized positions
/// (Nanite packs positions to a per-cluster grid) are a documented follow-up —
/// the schema version gates the upgrade.
///
/// # The tangent channel (P28.2), and why it is one packed word
///
/// P26.5 routed the tangent here and gave the reason: `VgeomVertex` had no
/// tangent to give, so neither the meshlet stream nor the rigid stream could
/// carry one until the container moved. It moves in this batch (`.inf_vmesh`
/// v3), so the channel lands here — as **one `u32`**, not four `f32`, and the
/// measurement is what decided it. Over 16², 48² and 96² displaced grids the
/// vertex records are **54.5 – 56.7 %** of a cooked asset's meshlet-pool bytes,
/// so a `[f32; 4]` tangent costs **+27.3 – 28.4 % of the whole pool** — better
/// than a quarter of the streaming budget spent on a second-order quality
/// feature — while one packed word costs **+6.8 – 7.1 %**.
///
/// See [`pack_tangent`] for the bit layout and for the reason its exponent
/// field is pinned rather than left to carry payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct VgeomVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Octahedral tangent + handedness in one word — [`pack_tangent`]'s output.
    /// [`NO_TANGENT`] (zero) means *this vertex has no authored tangent*, and
    /// every consumer falls back to the per-fragment cotangent frame it used
    /// before this channel existed.
    pub tangent: u32,
}

impl Default for VgeomVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            tangent: NO_TANGENT,
        }
    }
}

// ── the tangent word ────────────────────────────────────────────────────────

/// "This vertex has no authored tangent." Structurally unreachable from
/// [`pack_tangent`], because every packed word carries the pinned exponent
/// field below — so the sentinel needs no reserved direction.
pub const NO_TANGENT: u32 = 0;

/// The pinned exponent field: bits 23..=30 set to 127.
///
/// **Why a packed word has to look like a float.** The meshlet vertex pool is
/// bound as `array<f32>` in four shaders (`vgeom_mesh`, `visbuffer`,
/// `vis_resolve`, `vis_feedback`) and there is no room for a second, `u32`-typed
/// view of it: `vis_resolve` is a FRAGMENT pass and P28.1 measured the default
/// limit of 8 fragment storage bindings **fully spent**. So the tangent word is
/// loaded as an `f32` and `bitcast` back — and an arbitrary 32-bit payload read
/// that way has two hazards, both of which this pin removes:
///
/// * **subnormals.** Every `u32` below `2^23` has an all-zero exponent field, so
///   as a float it is subnormal, and WGSL permits an implementation to flush
///   subnormals to zero (the hazard `docs/memos/p28-1-visbuffer.md` §3 names).
/// * **NaN.** An all-ones exponent field is a NaN, whose payload bits an
///   implementation may canonicalize.
///
/// With bits 23..=30 fixed at 127 the word is always a normal float in
/// ±[1, 2) — never subnormal, never NaN — and carries no arithmetic anywhere.
/// The cost is nine bits of payload, which buys 11 bits per octahedral axis and
/// a handedness bit, and the angular error that leaves is measured in
/// [`tangent_tests::the_packed_tangent_round_trips_within_its_quantization`].
pub const TANGENT_EXP: u32 = 127 << 23;

/// Quantization scale of one octahedral axis: 11 signed bits, so ±1 maps to
/// ±1023 (the 11-bit two's-complement range is −1024..=1023).
const TANGENT_Q: f32 = 1023.0;

/// Pack a unit tangent + handedness into one [`VgeomVertex::tangent`] word.
///
/// `handedness` is the `w` of `inf_mesh::MeshVertex::tangent` — the sign that
/// turns `cross(normal, tangent)` into the bitangent. A non-finite or
/// zero-length tangent packs to [`NO_TANGENT`], which is how "the importer had
/// nothing to give" travels all the way to the shader as one comparison.
///
/// Layout (LSB first): `qx` 11 bits · `qy` 11 bits · handedness 1 bit ·
/// exponent 8 bits ([`TANGENT_EXP`]) · sign 1 bit (always 0).
pub fn pack_tangent(tangent: [f32; 3], handedness: f32) -> u32 {
    let [x, y, z] = tangent;
    let len = (x * x + y * y + z * z).sqrt();
    if !len.is_finite() || len <= 0.0 {
        return NO_TANGENT;
    }
    let (x, y, z) = (x / len, y / len, z / len);
    // Octahedral projection: the L1-normalized point, folded across the |x|+|y|=1
    // diamond for the lower hemisphere.
    let l1 = x.abs() + y.abs() + z.abs();
    let (mut ox, mut oy) = (x / l1, y / l1);
    if z < 0.0 {
        let (sx, sy) = (sign_not_zero(ox), sign_not_zero(oy));
        let (ax, ay) = (ox.abs(), oy.abs());
        ox = (1.0 - ay) * sx;
        oy = (1.0 - ax) * sy;
    }
    let q = |v: f32| -> u32 {
        let c = (v.clamp(-1.0, 1.0) * TANGENT_Q).round() as i32;
        (c as u32) & 0x7ff
    };
    let h = u32::from(handedness < 0.0);
    q(ox) | (q(oy) << 11) | (h << 22) | TANGENT_EXP
}

/// Invert [`pack_tangent`] — the CPU twin of `vgeom_unpack_tangent` in
/// `vt_sample.wgsl`, and the reference the shader mirror is pinned against.
///
/// Returns `(tangent, handedness)`, or `None` for [`NO_TANGENT`].
pub fn unpack_tangent(word: u32) -> Option<([f32; 3], f32)> {
    if word == NO_TANGENT {
        return None;
    }
    // Sign-extend the two 11-bit fields by shifting them to the top and back.
    let ex = ((word << 21) as i32 >> 21) as f32 / TANGENT_Q;
    let ey = (((word << 10) as i32) >> 21) as f32 / TANGENT_Q;
    let mut x = ex;
    let mut y = ey;
    let z = 1.0 - ex.abs() - ey.abs();
    if z < 0.0 {
        let (sx, sy) = (sign_not_zero(x), sign_not_zero(y));
        let (ax, ay) = (x.abs(), y.abs());
        x = (1.0 - ay) * sx;
        y = (1.0 - ax) * sy;
    }
    let inv = 1.0 / (x * x + y * y + z * z).sqrt();
    let h = if (word >> 22) & 1 == 1 { -1.0 } else { 1.0 };
    Some(([x * inv, y * inv, z * inv], h))
}

/// `+1` for zero, so the octahedral fold has no branch at the axes — the same
/// convention both halves of the mirror use.
#[inline]
fn sign_not_zero(v: f32) -> f32 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
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

    /// Whether this meshlet's data is **resident** when the streamer's finest
    /// resident LOD level is `floor_lod` (P18.2).
    ///
    /// Residency is a prefix of the `.inf_vmesh` page order: page 0 is every
    /// *root*, from every level, and is never evicted; page `p ≥ 1` is the
    /// non-roots at one level, coarse to fine. So a meshlet is resident iff it is
    /// a root **or** sits at or above the floor. See
    /// [`crate::asset`] for why the paging unit cannot be the LOD level alone.
    #[inline]
    pub fn resident_at(&self, floor_lod: u8) -> bool {
        self.is_root() || self.lod_level >= floor_lod
    }

    /// The cut test **clamped to residency** (P18.2): the same rule as
    /// [`selected_at`](Self::selected_at), except a meshlet at or below the
    /// residency floor has its error treated as **0** — so it covers the whole
    /// interval below its parent's error instead of yielding to the finer meshlets
    /// that are not paged in.
    ///
    /// `selected_at_clamped(t, 0) == selected_at(t)` for every meshlet at every
    /// threshold (LOD 0's error is 0 by construction, and a root below the floor
    /// has `parent_error == +∞`), which is what makes full residency
    /// bit-identical to the pre-streaming path.
    #[inline]
    pub fn selected_at_clamped(&self, threshold: f32, floor_lod: u8) -> bool {
        if !self.resident_at(floor_lod) {
            return false;
        }
        let error = if self.lod_level <= floor_lod {
            0.0
        } else {
            self.error
        };
        error <= threshold && threshold < self.parent_error
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
    /// Current schema version. v1 = f32 position/normal/uv vertices, per-group
    /// greedy adjacency grouping, monotone accumulated error. **v2 (P28.2)** adds
    /// [`VgeomVertex::tangent`], which bincode encodes positionally, so the
    /// version has to move for the two shapes to be told apart at all — v1 blobs
    /// decode through the frozen shadow record in
    /// [`crate::asset`](crate::asset) and are converted. Nothing in the tree
    /// *writes* this bincode form any more (the cook and the editor both emit the
    /// paged container), so no committed payload is downgraded by the bump;
    /// what it buys is that the frozen v1 fixture keeps loading.
    pub const CURRENT_VERSION: u32 = 2;

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

    /// The cut at `threshold` **clamped to residency** (P18.2) — the CPU reference
    /// for the streamed GPU cut, exactly as `vgeom_cull.wgsl` applies it.
    ///
    /// `floor_lod` is the streamer's finest resident LOD level (the finest
    /// resident page's [`floor_lod`](crate::asset::VgeomPageEntry::floor_lod)).
    /// The selected surface stays **complete and non-overlapping at every
    /// threshold and every floor** — never a hole, only softer detail:
    ///
    /// * along a root-to-leaf path the intervals `[error, parent_error)` tile
    ///   `[0, ∞)` in level order;
    /// * zeroing the error at and below the floor makes the floor's meshlet cover
    ///   `[0, e_floor+1)`, so the resident meshlets' intervals still tile `[0, ∞)`;
    /// * a path whose root sits *below* the floor keeps exactly that root (page 0
    ///   is never evicted), whose clamped interval is `[0, ∞)`.
    ///
    /// `select_with_residency(t, 0) == select(t)`.
    pub fn select_with_residency(
        &self,
        threshold: f32,
        floor_lod: u8,
    ) -> impl Iterator<Item = (usize, &Meshlet)> {
        self.meshlets
            .iter()
            .enumerate()
            .filter(move |(_, m)| m.selected_at_clamped(threshold, floor_lod))
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

impl VgeomMesh {
    /// Reject a mesh whose index ranges do not fit its own buffers, so every
    /// later slice is unchecked-safe (round-2 finding **B2**'s
    /// default-`migrate` sweep).
    ///
    /// [`VgeomMesh::triangle`] says in its own doc that it *"panics on
    /// out-of-range indices (a corrupt payload)"*, and
    /// [`VgeomMesh::classic_lods`] walks `LevelRange` just as bluntly. The
    /// engine is safe today because `inf_vgeom::asset`'s two validators stand
    /// in front of every production path — but that safety is a **convention
    /// about which door you use**, not a property of the type, and this type
    /// implements `AssetPayload`, whose whole point is that a caller can decode
    /// it generically. `asset::validate` now delegates here, so there is one
    /// rule with two entrances rather than two rules.
    pub fn validate(&self) -> Result<(), String> {
        for (i, m) in self.meshlets.iter().enumerate() {
            let v_end = m.vertex_offset as usize + m.vertex_count as usize;
            let t_end = m.triangle_offset as usize + m.triangle_count as usize * 3;
            if v_end > self.meshlet_vertices.len() || t_end > self.meshlet_triangles.len() {
                return Err(format!("meshlet {i} micro-index range is out of bounds"));
            }
            if self.meshlet_vertices[m.vertex_offset as usize..v_end]
                .iter()
                .any(|&v| v as usize >= self.vertices.len())
            {
                return Err(format!("meshlet {i} references a vertex past the buffer"));
            }
        }
        // The level table, which `classic_lods` indexes with equally bare
        // `[]`. `asset::validate` never looked at it.
        for (i, l) in self.levels.iter().enumerate() {
            let end = l.meshlet_start as usize + l.meshlet_count as usize;
            if end > self.meshlets.len() {
                return Err(format!("lod level {i} names meshlets past the buffer"));
            }
        }
        for (i, g) in self.groups.iter().enumerate() {
            let end = g.produced_start as usize + g.produced_count as usize;
            if end > self.meshlets.len() {
                return Err(format!("group {i} produced meshlets past the buffer"));
            }
        }
        Ok(())
    }
}

impl AssetPayload for VgeomMesh {
    const KIND: AssetKind = AssetKind::MeshletMesh;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current, then [`VgeomMesh::validate`].
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid meshlet mesh: {e}")))?;
        Ok(self)
    }
}

#[cfg(test)]
mod classic_lod_tests {
    use crate::model::pick_classic_level;
    // The shared displaced grid (portable trig — see `test_support`); this module
    // used to carry its own copy of the generator.
    use crate::test_support::dense_mesh as dense;

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

#[cfg(test)]
mod tangent_tests {
    use super::*;

    /// A sweep over the sphere: every packed word is a **normal** float — never
    /// subnormal, never NaN — so the `bitcast` four shaders do on it takes no
    /// arithmetic and cannot be flushed or canonicalized. This is the hazard
    /// `docs/memos/p28-1-visbuffer.md` §3 measured on the instance table, closed
    /// here by construction rather than by an on-device `assert_ne!`.
    #[test]
    fn every_packed_tangent_is_a_normal_float() {
        let mut n = 0usize;
        for i in 0..64 {
            for j in 0..64 {
                // A deterministic spread over the sphere (no trig: the P14 law,
                // and this needs no angles anyway).
                let a = (i as f32 / 63.0) * 2.0 - 1.0;
                let b = (j as f32 / 63.0) * 2.0 - 1.0;
                let c = 1.0 - a.abs() - b.abs();
                for h in [1.0f32, -1.0] {
                    let w = pack_tangent([a, b, c], h);
                    assert_ne!(w, NO_TANGENT, "a real direction packs to the sentinel");
                    let f = f32::from_bits(w);
                    assert!(f.is_normal(), "{w:#010x} is not a normal float");
                    assert_eq!(w & 0x7f80_0000, TANGENT_EXP, "the exponent is pinned");
                    assert_eq!(w >> 31, 0, "the sign bit is reserved and zero");
                    n += 1;
                }
            }
        }
        assert_eq!(n, 64 * 64 * 2);
    }

    /// The round trip is exact to its quantization, and the quantization is
    /// **measured** rather than asserted loosely: 11 bits an axis over the
    /// octahedral map. The bound below is the worst case observed over the sweep;
    /// a coarser packing fails it, which is what makes it a bound and not a
    /// restatement.
    #[test]
    fn the_packed_tangent_round_trips_within_its_quantization() {
        let mut worst = 0.0f32;
        for i in 0..96 {
            for j in 0..96 {
                let a = (i as f32 / 95.0) * 2.0 - 1.0;
                let b = (j as f32 / 95.0) * 2.0 - 1.0;
                for cz in [1.0f32 - a.abs() - b.abs(), a.abs() + b.abs() - 1.0] {
                    let len = (a * a + b * b + cz * cz).sqrt();
                    if len <= 1e-6 {
                        continue;
                    }
                    let t = [a / len, b / len, cz / len];
                    for h in [1.0f32, -1.0] {
                        let (back, hb) = unpack_tangent(pack_tangent(t, h)).expect("packed");
                        assert_eq!(hb, h, "handedness survives");
                        let dot =
                            (t[0] * back[0] + t[1] * back[1] + t[2] * back[2]).clamp(-1.0, 1.0);
                        worst = worst.max(1.0 - dot);
                    }
                }
            }
        }
        // Measured worst `1 - cos(theta)` over this sweep: **1.79e-6**. The bound
        // sits above the real worst case below so dropping an axis to 10 bits —
        // which roughly quadruples the term — fails here rather than showing up
        // as a normal map that leans.
        assert!(worst < 2.5e-6, "worst 1-cos was {worst}");
    }

    /// **The worst case, searched rather than sampled** (P28.2 audit).
    ///
    /// The sweep above walks a grid of *directions* and reports the worst it
    /// happens to land on. That is a sample, and the memo quoted its number —
    /// `2.03e-6`, `0.115°` — as though it were the bound. It is not: a two-
    /// million-direction random sweep already beats it (`2.21e-6`), because the
    /// error is not a function of direction, it is a function of where a
    /// direction falls **inside a quantization cell**, and a 96 × 96 grid of
    /// directions lands near cell centres only by luck.
    ///
    /// So this searches the cell centres directly — every one of the 2 × 2048²
    /// of them — which is where the maximum provably lives for a nearest-value
    /// quantizer. **2.33e-6, i.e. 0.124°**, at oct `(-0.292, -0.378)` in the
    /// lower hemisphere. The channel's real angular bound, and 21 % above the
    /// number that was written down.
    #[test]
    fn the_quantization_bound_is_the_worst_cell_centre_not_the_luckiest_sample() {
        let mut worst = 0.0f64;
        for i in 0..2048i32 {
            for j in 0..2048i32 {
                // The centre of quantization cell (i, j), in octahedral space.
                let ox = (i as f32 - 1023.5) / TANGENT_Q;
                let oy = (j as f32 - 1023.5) / TANGENT_Q;
                if !(-1.0..=1.0).contains(&ox) || !(-1.0..=1.0).contains(&oy) {
                    continue;
                }
                let (mut x, mut y) = (ox, oy);
                let z = 1.0 - x.abs() - y.abs();
                if z < 0.0 {
                    let (sx, sy) = (sign_not_zero(x), sign_not_zero(y));
                    let (ax, ay) = (x.abs(), y.abs());
                    x = (1.0 - ay) * sx;
                    y = (1.0 - ax) * sy;
                }
                let inv = 1.0 / (x * x + y * y + z * z).sqrt();
                for hemi in [1.0f32, -1.0] {
                    let t = [x * inv, y * inv, z * inv * hemi];
                    let Some((b, _)) = unpack_tangent(pack_tangent(t, 1.0)) else {
                        continue;
                    };
                    let dot = (f64::from(t[0]) * f64::from(b[0])
                        + f64::from(t[1]) * f64::from(b[1])
                        + f64::from(t[2]) * f64::from(b[2]))
                    .min(1.0);
                    worst = worst.max(1.0 - dot);
                }
            }
        }
        // Above the sampled sweep's 1.79e-6 — which is the point — and below the
        // ceiling a 10-bit axis would blow through.
        assert!(
            (2.0e-6..3.0e-6).contains(&worst),
            "the searched worst 1-cos was {worst:e}; the channel's bound moved"
        );
    }

    /// The sentinel is *reachable only by asking for it* — a degenerate tangent —
    /// and it survives the round trip as "no tangent" rather than as a direction.
    #[test]
    fn a_degenerate_tangent_is_the_sentinel_and_not_a_direction() {
        for t in [
            [0.0, 0.0, 0.0],
            [f32::NAN, 0.0, 0.0],
            [f32::INFINITY, 0.0, 0.0],
        ] {
            assert_eq!(pack_tangent(t, 1.0), NO_TANGENT, "{t:?}");
        }
        assert!(unpack_tangent(NO_TANGENT).is_none());
        assert_eq!(VgeomVertex::default().tangent, NO_TANGENT);
    }
}

#[cfg(test)]
mod payload_door_tests {
    use super::*;
    use crate::test_support::dense_mesh as dense;

    /// **Round-2 finding B2's sweep**: `VgeomMesh` implements `AssetPayload`,
    /// so a caller can decode one generically without ever reaching
    /// `inf_vgeom::asset` — and `triangle()`'s own doc says it *"panics on
    /// out-of-range indices (a corrupt payload)"*. The engine is safe today
    /// because every production path happens to go through `VgeomSource`, but
    /// that is a convention about which door you use, not a property of the
    /// type. The rule now lives on the type and both entrances call it.
    #[test]
    fn a_meshlet_mesh_with_ranges_past_its_buffers_is_refused_at_decode() {
        let good = dense(24);
        assert!(!good.meshlets.is_empty(), "the fixture is vacuous");
        let bytes = inf_asset::encode(&good).unwrap();
        inf_asset::decode::<VgeomMesh>(&bytes).expect("a healthy DAG must still decode");

        // A micro-index range past the end of `meshlet_vertices`.
        let mut m = good.clone();
        m.meshlets[0].vertex_offset = m.meshlet_vertices.len() as u32;
        m.meshlets[0].vertex_count = 4;
        let bytes = inf_asset::encode(&m).unwrap();
        assert!(
            inf_asset::decode::<VgeomMesh>(&bytes).is_err(),
            "a meshlet naming micro-indices past the buffer decoded as valid"
        );

        // A vertex index past the end of `vertices`.
        let mut m = good.clone();
        let last = m.meshlet_vertices.len() - 1;
        m.meshlet_vertices[last] = m.vertices.len() as u32;
        let bytes = inf_asset::encode(&m).unwrap();
        assert!(inf_asset::decode::<VgeomMesh>(&bytes).is_err());

        // A LOD level naming meshlets past the buffer — `classic_lods` walks
        // this with equally bare `[]`, and `asset::validate` never looked at it.
        let mut m = good.clone();
        m.levels[0].meshlet_count = m.meshlets.len() as u32 + 1;
        let bytes = inf_asset::encode(&m).unwrap();
        assert!(
            inf_asset::decode::<VgeomMesh>(&bytes).is_err(),
            "a level range past the meshlet buffer decoded as valid"
        );

        // And a group producing meshlets past the buffer.
        if !good.groups.is_empty() {
            let mut m = good.clone();
            m.groups[0].produced_start = m.meshlets.len() as u32;
            m.groups[0].produced_count = 1;
            let bytes = inf_asset::encode(&m).unwrap();
            assert!(inf_asset::decode::<VgeomMesh>(&bytes).is_err());
        }
    }
}
