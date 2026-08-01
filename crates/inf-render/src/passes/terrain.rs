//! Heightfield terrain pass (P10.1): geometry-clipmap LOD over a **streamed**
//! page set (P16.3b1).
//!
//! Each resident terrain tile becomes one draw of a unit grid patch, displaced in
//! the vertex shader by that tile's cached R32Float height texture. The patch
//! mesh density falls with the tile's clipmap ring (a power-of-two vertex
//! decimation chosen by camera distance → concentric rings around the camera);
//! LOD morphing blends toward the coarser grid to kill popping; a skirt ring on
//! each patch drops its boundary down to seal cracks between neighbouring LODs.
//!
//! The CPU-side ring/patch assembly, source selection, morph, and cull are pure
//! functions ([`assemble_patches`] & friends) so they unit-test without a GPU.
//! Height textures are cached per [`TerrainTileKey`] in a `BTreeMap`
//! (deterministic) and uploaded under a **per-tile** version gate
//! ([`plan_tile_cache`]). When `scene.terrain` is `None` the node encodes nothing
//! — pre-P10.1 scenes stay byte-identical.
//!
//! # Streaming (P16.3b1)
//!
//! The pass makes **no** assumption that the terrain is fully resident:
//!
//! * **Per-tile upload gating.** Every projected tile carries its own monotone
//!   change stamp; the GPU cache re-uploads a page when — and only when — that
//!   stamp differs from the one its cached copy holds, and drops the textures of
//!   any tile that left the projection. Sculpting one tile therefore re-uploads
//!   exactly that tile. There is no whole-terrain counter left to desync.
//! * **Coarse rings.** Tiles arrive keyed by *asset LOD*: level 0 is the authored
//!   heightfield, level `n` is the pyramid's 2:1-decimated page covering `2ⁿ ×`
//!   the world span. Ring `r` samples asset LOD `r` (both double
//!   metres-per-sample per step — see [`ring_source_lod`]), so far terrain draws
//!   off coarse pages and never needs full-resolution residency. Where a fine and
//!   a coarse page could both serve the same ground, the **fine one wins**
//!   ([`superseded`]).
//! * **Holes are legal.** A tile that is simply not in the projection produces no
//!   patch — exactly as an unauthored tile always did — and its neighbours are
//!   unaffected.
//!
//! ## What the streamer above must deliver (the B2 contract)
//!
//! Source selection is hole-free under *any* residency set, but it is only
//! *artefact*-free under a **quadtree cut**: for every patch of ground exactly one
//! resident page covers it, i.e. a coarse page is resident precisely where its
//! four children are not. The streamer must therefore page children in and out as
//! **complete sets of four** — never publish a residency set where three of four
//! children are resident beside their parent as a steady state.
//!
//! When that invariant is momentarily broken (mid-load, or a partial eviction), the
//! renderer keeps the coarse page *and* the fine ones rather than dropping
//! geometry, so the fallback is visible-but-wrong instead of missing. That
//! fallback is **not free**: the coarse surface is a chord between decimated
//! samples while the fine surface follows the full profile, so the two
//! **interpenetrate** — the coarse page sits below the fine one over convex ground
//! and above it over concave ground, and along the curves where they cross, depth
//! ties produce z-fighting. The artefact is bounded (by the same relief bound as
//! the seam analysis below) and, under a well-behaved streamer, transient — which
//! is why it is preferred to a hole, not why it is acceptable to live in.
//!
//! ## Cracks at fine ↔ coarse boundaries (verified, not re-invented)
//!
//! No new stitching scheme is introduced here: the existing **skirt** machinery
//! covers the fine↔coarse seam, and the bound is worth writing down.
//!
//! The pyramid is built by *decimation*, not filtering, so a coarse tile's
//! samples are bit-identical to the level-0 samples they were taken from, and its
//! edge samples coincide exactly with every other fine edge sample. Along a
//! shared boundary the coarse surface is therefore a **chord** between two fine
//! samples: the maximum vertical disagreement between the two patches is the
//! largest deviation of the fine profile from that chord, which is bounded by the
//! height relief `hmax − hmin` of the tiles involved.
//!
//! Each patch's skirt drops its boundary ring by
//! `skirt = max(|hmax − hmin|, 0.05 · span, 1 m)` — at least the tile's own full
//! relief. So the skirt wall of *either* side already spans the worst-case gap,
//! and both sides skirt, so the seam is sealed twice over. The `0.05 · span`
//! floor matters for the coarse side specifically: a coarse tile's own relief is
//! measured over its decimated samples and can under-report the fine relief
//! *between* two of them, but its span is `2ⁿ ×` larger, so the floor grows with
//! the page's footprint and the fine side's skirt (measured on the full-resolution
//! profile) covers the remainder. Skirts are drawn with `cull_mode: None` into the
//! depth-tested opaque pass, so a wall is never culled by winding.
//!
//! The consequence is a documented one: seams are *sealed*, not *stitched* — a
//! grazing view can see a sliver of skirt wall where two LODs meet. A real
//! stitching scheme (edge-matched index buffers) is deliberately out of scope for
//! this batch.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec3, Vec3};
use inf_math::FloatingOrigin;
use inf_render_2d::aabb_visible;

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{RenderTerrain, RenderTerrainTile, TerrainTileKey};

/// Number of discrete LOD levels (finest = 0).
pub const TERRAIN_LOD_COUNT: u32 = 4;
/// Grid cells per patch side at LOD 0 (halved each coarser level, min 1). The
/// patch mesh density — decoupled from the (usually finer) height-texture
/// resolution, which is sampled for the actual displacement.
pub const TERRAIN_BASE_CELLS: u32 = 32;
/// LOD-0 ring radius as a multiple of the tile span; each coarser ring doubles.
pub const TERRAIN_LOD_SCALE: f64 = 1.5;
/// Fraction of a LOD band (nearest the far edge) over which the morph ramps 0→1.
pub const TERRAIN_MORPH_REGION: f64 = 0.35;

/// Grid cells per side for a patch at `lod` (`TERRAIN_BASE_CELLS >> lod`, min 1).
pub fn cells_at_lod(lod: u32) -> u32 {
    (TERRAIN_BASE_CELLS >> lod.min(TERRAIN_LOD_COUNT - 1)).max(1)
}

/// Camera-distance thresholds between LOD bands (`TERRAIN_LOD_COUNT − 1` of them),
/// scaled by the tile span. A tile centre nearer than `thresholds[0]` is LOD 0.
pub fn lod_thresholds(span: f64) -> [f64; (TERRAIN_LOD_COUNT - 1) as usize] {
    let mut t = [0.0; (TERRAIN_LOD_COUNT - 1) as usize];
    for (k, tk) in t.iter_mut().enumerate() {
        *tk = span * TERRAIN_LOD_SCALE * (1u64 << k) as f64;
    }
    t
}

/// LOD level for a tile centre at horizontal distance `dist` from the eye: the
/// number of thresholds it has passed, capped at the coarsest level.
pub fn lod_for_distance(dist: f64, thresholds: &[f64]) -> u32 {
    let n = thresholds.iter().filter(|&&t| dist >= t).count() as u32;
    n.min(TERRAIN_LOD_COUNT - 1)
}

/// Morph factor (0 → use the fine grid, 1 → snapped to the coarser grid) for a
/// tile at `dist` in LOD `lod`: ramps up over the last [`TERRAIN_MORPH_REGION`]
/// of the band before the next threshold. The coarsest LOD never morphs (0).
pub fn morph_factor(dist: f64, lod: u32, thresholds: &[f64]) -> f32 {
    if lod as usize >= thresholds.len() {
        return 0.0; // coarsest level: nothing coarser to morph toward
    }
    let upper = thresholds[lod as usize];
    let lower = if lod == 0 {
        0.0
    } else {
        thresholds[lod as usize - 1]
    };
    let start = upper - TERRAIN_MORPH_REGION * (upper - lower);
    if dist <= start {
        0.0
    } else if dist >= upper {
        1.0
    } else {
        let t = (dist - start) / (upper - start).max(f64::MIN_POSITIVE);
        // smoothstep
        (t * t * (3.0 - 2.0 * t)) as f32
    }
}

/// The **asset LOD a clipmap ring samples from** (P16.3b1).
///
/// The two ladders are the same ladder. Ring `r` renders a patch with
/// `TERRAIN_BASE_CELLS >> r` grid cells across a tile, i.e. its mesh cell size
/// doubles every ring; an asset LOD `n` page has `mps · 2ⁿ` metres per sample,
/// i.e. its texel size doubles every level. Matching them one-for-one gives
/// `ring r ↔ asset LOD r` — the outer rings read progressively coarser pyramid
/// pages, and the number of height texels per patch cell stays constant across
/// the whole clipmap.
///
/// The result is clamped to `max_lod`, the coarsest level the projection actually
/// carries: a level-0-only (inline, non-streamed) terrain has `max_lod == 0`, so
/// every ring wants level 0 and the mapping collapses to the pre-P16.3 behaviour
/// exactly.
#[inline]
pub fn ring_source_lod(ring: u32, max_lod: u32) -> u32 {
    ring.min(max_lod)
}

/// The **patch grid-mesh density index** for an asset-LOD-`lod` tile drawn in
/// clipmap ring `ring`: `ring − lod`, clamped into `[0, TERRAIN_LOD_COUNT)`.
///
/// The asset LOD already supplies `2^lod` of the decimation (a coarse page covers
/// `2^lod ×` the span at the same sample count), so the mesh only has to supply
/// the rest. Ring 2 sampling a level-2 page therefore draws the *full-density*
/// 32-cell grid over its 4× footprint — the same screen-space triangle size as
/// ring 0 over a level-0 page — instead of a 4-cell grid that would throw the
/// page's detail away. For a level-0 tile this is just `ring`, unchanged.
#[inline]
pub fn patch_mesh_lod(ring: u32, lod: u32) -> u32 {
    ring.saturating_sub(lod).min(TERRAIN_LOD_COUNT - 1)
}

/// Whether `key` is **fully subdivided**: a coarse tile whose four finer children
/// are all resident, so the finer pages already cover its whole footprint.
/// Always `false` at level 0 (nothing is finer).
fn fully_subdivided(key: TerrainTileKey, resident: &BTreeSet<TerrainTileKey>) -> bool {
    key.lod >= 1 && key.children().iter().all(|c| resident.contains(c))
}

/// Whether a resident tile is **served by another resident tile** this frame, and
/// so must not draw (P16.3b1). The source-selection rule, in two clauses:
///
/// 1. **Fine wins.** A coarse tile whose four children are all resident never
///    draws: the finer pages cover exactly its footprint, and where both could
///    serve, the fine one is the better source — regardless of ring.
/// 2. **Coarse serves the outer rings.** A tile finer than its ring wants
///    ([`ring_source_lod`]) steps aside for a resident ancestor at or below that
///    level — but *only* if that ancestor is not itself fully subdivided, i.e.
///    only if the ancestor will really draw.
///
/// That second qualifier is what makes the swap **hole-free**: a tile is only
/// suppressed in favour of a coarser page that is guaranteed to be drawn, and by
/// induction up the parent chain some ancestor always is. The renderer never
/// punches a hole in geometry it was handed. Under an inconsistent
/// (non-quadtree-cut) residency set a coarse page and its partial fine coverage
/// can both draw; the two surfaces **interpenetrate** and z-fight along the curves
/// where they cross — bounded and, under a well-behaved streamer, transient (see
/// the module docs), and strictly preferable to seeing sky through the ground.
///
/// `resident` must be the set of tiles that **survived the frustum cull**, not
/// every projected tile — see the second pass of [`assemble_patches`] for why a
/// culled ancestor must not suppress a visible child.
///
/// For a level-0-only projection `max_lod == 0`, so clause 1 cannot fire (no
/// coarse tiles) and clause 2's loop never runs (`want == 0`) — the output is
/// bit-identical to the pre-P16.3 assembly.
pub fn superseded(
    key: TerrainTileKey,
    ring: u32,
    max_lod: u32,
    resident: &BTreeSet<TerrainTileKey>,
) -> bool {
    if fully_subdivided(key, resident) {
        return true;
    }
    let want = ring_source_lod(ring, max_lod);
    let mut ancestor = key;
    while ancestor.lod < want {
        ancestor = ancestor.parent();
        if resident.contains(&ancestor) && !fully_subdivided(ancestor, resident) {
            return true;
        }
    }
    false
}

/// Render-local AABB (`min`, `max`) of a terrain tile (planar span + its height
/// bounds), rebased through the floating origin. Used for frustum culling.
/// `span` is the tile's **own** world edge length — `tile_span · 2^lod` for a
/// coarse page (see [`RenderTerrain::tile_span_at`]).
pub fn tile_aabb_local(
    tile: &RenderTerrainTile,
    span: f64,
    origin: &FloatingOrigin,
) -> (Vec3, Vec3) {
    let (hmin, hmax) = tile.height_bounds;
    let wmin = DVec3::new(tile.origin.x, tile.origin.y + hmin as f64, tile.origin.z);
    let wmax = DVec3::new(
        tile.origin.x + span,
        tile.origin.y + hmax as f64,
        tile.origin.z + span,
    );
    (origin.to_render(wmin), origin.to_render(wmax))
}

/// One assembled patch: which resident tile sources it, which clipmap ring it
/// fell in, which grid mesh draws it, and how much it morphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainPatch {
    /// The source tile's key (asset LOD + grid coordinate at that level).
    pub key: TerrainTileKey,
    /// Index of the source tile in `terrain.tiles` — so the draw loop reaches the
    /// tile's origin/bounds without a lookup, and the instance-buffer index can
    /// never drift out of step with the patch list.
    ///
    /// **Valid by convention only**: it indexes the exact `RenderTerrain` this
    /// patch was assembled from, in the frame it was assembled. Indexing a
    /// different (or re-projected) terrain with it reads the wrong tile, or panics
    /// if that terrain is shorter. Always pair a patch list with the terrain that
    /// produced it; use [`key`](Self::key) for anything that must survive across
    /// terrains.
    pub tile: usize,
    /// Clipmap ring (camera-distance band, finest = 0) the patch fell in.
    pub ring: u32,
    /// Grid-mesh density index: [`patch_mesh_lod(ring, key.lod)`](patch_mesh_lod).
    pub mesh_lod: u32,
    /// LOD morph factor (0 = this ring's grid, 1 = snapped to the coarser grid).
    pub morph: f32,
}

/// Assemble the visible clipmap patches for `terrain` under `view`.
///
/// For every **resident** tile: cull it against the frustum by its own span, pick
/// its clipmap ring from the camera's horizontal distance to the tile centre,
/// drop it if a better-suited resident page serves the same ground
/// ([`superseded`]), then record its mesh density + morph.
///
/// Deterministic: tiles arrive key-sorted and the per-tile decision is a pure
/// function of `(tile, residency set, view)`, so the output order and content are
/// too (stable draw order). A tile that is not resident simply yields no patch —
/// the renderer draws the set it is handed and never invents coverage.
pub fn assemble_patches(
    terrain: &RenderTerrain,
    view: &RenderView,
    origin: &FloatingOrigin,
) -> Vec<TerrainPatch> {
    let span0 = terrain.tile_span();
    let thresholds = lod_thresholds(span0);
    let view_proj = view.view_proj();
    let eye = view.eye_world;
    let max_lod = terrain.max_lod();

    // Pass 1 — cull, and assign each survivor its clipmap ring.
    let mut out = Vec::with_capacity(terrain.tiles.len());
    for (index, tile) in terrain.tiles.iter().enumerate() {
        let span = terrain.tile_span_at(tile.key.lod);
        let (lo, hi) = tile_aabb_local(tile, span, origin);
        // A small margin so a tile straddling a screen edge isn't popped early.
        let margin = Vec3::splat((span * 0.02) as f32);
        if !aabb_visible(lo - margin, hi + margin, &view_proj) {
            continue;
        }
        let cx = tile.origin.x + span * 0.5;
        let cz = tile.origin.z + span * 0.5;
        let (dx, dz) = (cx - eye.x, cz - eye.z);
        let dist = (dx * dx + dz * dz).sqrt();
        let ring = lod_for_distance(dist, &thresholds);
        out.push(TerrainPatch {
            key: tile.key,
            tile: index,
            ring,
            mesh_lod: patch_mesh_lod(ring, tile.key.lod),
            morph: morph_factor(dist, ring, &thresholds),
        });
    }

    // Pass 2 — source selection, over the tiles that SURVIVED the cull.
    //
    // The index must be built from the visible set, not from every projected
    // tile: a coarse page's AABB can be *vertically narrower* than its children's
    // (decimation samples every other height, so it can miss a spike a child
    // keeps), so an ancestor can be frustum-culled while its child is not.
    // Suppressing that child in favour of a page nobody draws is a hole — sky
    // through the ground. Restricting the index to visible tiles is safe in both
    // directions: a coarse page whose children were culled simply stops counting
    // as fully subdivided and draws, which is at worst overdraw off-screen.
    //
    // A level-0-only (inline) terrain has `max_lod == 0`, so this whole pass is
    // skipped and it pays exactly what it paid before P16.3.
    if max_lod > 0 {
        let visible: BTreeSet<TerrainTileKey> = out.iter().map(|p| p.key).collect();
        out.retain(|p| !superseded(p.key, p.ring, max_lod, &visible));
    }
    out
}

// ── GPU tile cache planning (P16.3b1) ───────────────────────────────────────

/// What a cached GPU tile currently holds: the tile stamp its texels were written
/// from, and the resolution its textures were created at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedTile {
    /// The [`RenderTerrainTile::version`] the cached texels reflect.
    pub version: u64,
    /// The texture dimension the cached pair was created at.
    pub resolution: u32,
}

/// What one texture-cache sync must do: which tiles to (re)upload, which cached
/// entries to drop. Both lists are in a deterministic order — `upload` follows the
/// projection's tile order, `evict` is ascending by key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileCachePlan {
    /// Tiles whose GPU copy is stale (or absent) and must be written.
    pub upload: Vec<TerrainTileKey>,
    /// Cached tiles that left the projection (or went malformed) and must be
    /// dropped, freeing their textures.
    pub evict: Vec<TerrainTileKey>,
}

/// Plan the per-tile GPU texture cache against a projection — a pure function, so
/// the upload/eviction gate unit-tests without a GPU.
///
/// A tile is uploaded when its cached stamp differs from its projected stamp, the
/// cached textures were built at a different resolution, or it is not cached at
/// all. A projected stamp of `0` means "no stamp" and never counts as a hit, so a
/// source that cannot version its tiles degrades to the pre-P16.3 behaviour
/// (re-upload) rather than to a stale frame. Malformed tiles (a height buffer that
/// is not `resolution²`) are neither uploaded nor kept — a bad page never becomes
/// silent geometry.
///
/// `state` projects the cache's value type down to the identity the planner
/// reasons about, so the node can hold GPU resources in the same map the test
/// fills with plain [`CachedTile`]s.
pub fn plan_tile_cache<T>(
    cached: &BTreeMap<TerrainTileKey, T>,
    state: impl Fn(&T) -> CachedTile,
    terrain: &RenderTerrain,
) -> TileCachePlan {
    let res = terrain.tile_resolution.max(2);
    let expect = (res * res) as usize;
    let mut upload = Vec::new();
    let mut live: BTreeSet<TerrainTileKey> = BTreeSet::new();
    for tile in &terrain.tiles {
        if tile.heights.len() != expect {
            continue; // malformed — skip rather than upload garbage
        }
        live.insert(tile.key);
        let fresh = tile.version != 0
            && cached.get(&tile.key).map(&state)
                == Some(CachedTile {
                    version: tile.version,
                    resolution: res,
                });
        if !fresh {
            upload.push(tile.key);
        }
    }
    let evict = cached
        .keys()
        .copied()
        .filter(|k| !live.contains(k))
        .collect();
    TileCachePlan { upload, evict }
}

// ── GPU node ────────────────────────────────────────────────────────────────

/// Per-patch instance data (matches `@location(1..=2)` in terrain.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PatchRaw {
    /// origin_local.xyz, world tile span.
    o_span: [f32; 4],
    /// morph, cells-at-lod, resolution, skirt depth (m).
    params: [f32; 4],
}

/// A unit grid patch mesh (positions in `[0,1]²`, `z` = skirt flag) + indices for
/// one LOD, including a skirt ring.
struct LodMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

/// Build a `cells × cells` unit grid in `[0,1]²` (vertex = `[u, v, skirt]`) plus a
/// skirt ring on all four edges (duplicate boundary verts flagged `skirt = 1`,
/// stitched into vertical quads). Returns `(vertices, indices)`.
fn build_lod_geometry(cells: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n = cells.max(1);
    let stride = n + 1;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let inv = 1.0 / n as f32;
    for j in 0..=n {
        for i in 0..=n {
            verts.push([i as f32 * inv, j as f32 * inv, 0.0]);
        }
    }
    let vid = |i: u32, j: u32| j * stride + i;
    for j in 0..n {
        for i in 0..n {
            let (a, b, c, d) = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1));
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Skirts: one vertical quad strip per boundary edge.
    let edges: [Vec<u32>; 4] = [
        (0..=n).map(|i| vid(i, 0)).collect(), // bottom (v = 0)
        (0..=n).map(|i| vid(i, n)).collect(), // top    (v = 1)
        (0..=n).map(|j| vid(0, j)).collect(), // left   (u = 0)
        (0..=n).map(|j| vid(n, j)).collect(), // right  (u = 1)
    ];
    for edge in edges {
        let mut sk = Vec::with_capacity(edge.len());
        for &bi in &edge {
            let p = verts[bi as usize];
            sk.push(verts.len() as u32);
            verts.push([p[0], p[1], 1.0]);
        }
        for w in 0..edge.len() - 1 {
            let (t0, t1, s0, s1) = (edge[w], edge[w + 1], sk[w], sk[w + 1]);
            indices.extend_from_slice(&[t0, t1, s1, t0, s1, s0]);
        }
    }
    (verts, indices)
}

/// A cached per-tile pair of textures — R32Float height + Rgba8Unorm splat
/// weights — sharing one bind group (`@group(1)`), plus the [`CachedTile`]
/// identity the upload gate compares against.
struct TileTextures {
    _height: wgpu::Texture,
    _weights: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    cached: CachedTile,
}

/// The terrain splat material uniform (`@group(2)`), std140-friendly: four
/// layers' albedo + params (`x` = roughness, `y` = tex_scale) and the macro
/// variation amplitude. Mirrors `TerrainMaterial` in terrain.wgsl.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialRaw {
    albedo: [[f32; 4]; 4],
    params: [[f32; 4]; 4],
    macro_amp: [f32; 4],
}

impl MaterialRaw {
    fn from_terrain(terrain: &RenderTerrain) -> Self {
        let mut albedo = [[0.0f32; 4]; 4];
        let mut params = [[0.0f32; 4]; 4];
        for (k, layer) in terrain.layers.iter().enumerate() {
            albedo[k] = layer.albedo;
            params[k] = [layer.roughness, layer.tex_scale.max(1e-3), 0.0, 0.0];
        }
        MaterialRaw {
            albedo,
            params,
            macro_amp: [terrain.macro_variation, 0.0, 0.0, 0.0],
        }
    }
}

pub struct TerrainNode {
    pipeline: wgpu::RenderPipeline,
    tile_bgl: wgpu::BindGroupLayout,
    /// One grid mesh per LOD (index 0 = finest).
    lod_meshes: Vec<LodMesh>,
    /// Per-tile height + weight textures, keyed by [`TerrainTileKey`] (asset LOD +
    /// coord), so level-0 and coarse pages of the same ground cache side by side.
    /// `BTreeMap` ⇒ deterministic iteration.
    textures: BTreeMap<TerrainTileKey, TileTextures>,
    /// Terrain splat material uniform (`@group(2)`) + its bind group.
    material_buffer: wgpu::Buffer,
    material_bg: wgpu::BindGroup,
    /// The packed material the uniform buffer currently holds — the gate for the
    /// one terrain-wide upload left (a value compare, so it cannot desync).
    uploaded_material: Option<MaterialRaw>,
    /// AO + shadows + GI env bind at `@group(3)` (P13.3b).
    env: super::EnvBinding,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
}

/// Reference (CPU) mirror of the shader's triplanar axis weights: normalized
/// `pow(|n|, sharpness)` over the three world planes (YZ→x, XZ→y, XY→z). Kept in
/// sync with `terrain.wgsl`; unit-tested so the blend law is pinned even though
/// the shader itself needs a GPU. Returns `(w_x, w_y, w_z)` summing to 1.
pub fn triplanar_axis_weights(normal: Vec3, sharpness: f32) -> [f32; 3] {
    let mut w = [
        normal.x.abs().powf(sharpness.max(1e-3)),
        normal.y.abs().powf(sharpness.max(1e-3)),
        normal.z.abs().powf(sharpness.max(1e-3)),
    ];
    let sum = w[0] + w[1] + w[2];
    if sum > 0.0 {
        w[0] /= sum;
        w[1] /= sum;
        w[2] /= sum;
    } else {
        w = [0.0, 1.0, 0.0]; // degenerate normal → treat as flat (top plane)
    }
    w
}

/// Reference (CPU) mirror of the shader's macro albedo modulation:
/// `albedo · (1 + amp · fbm)` where `fbm ∈ [-1, 1]`. Kept in sync with
/// `terrain.wgsl`; unit-tested.
pub fn macro_modulate(albedo: Vec3, fbm: f32, amp: f32) -> Vec3 {
    albedo * (1.0 + amp * fbm)
}

impl TerrainNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("terrain").into()),
            });

        // Per-tile bind group (@group(1)): the R32Float height texture (binding 0)
        // + the Rgba8Unorm splat-weight texture (binding 1), both unfilterable-
        // float and sampled via textureLoad (no sampler, no FLOAT32_FILTERABLE).
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let tile_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain-tile"),
                entries: &[tex_entry(0), tex_entry(1)],
            });

        // Splat material uniform bind group (@group(2)).
        let material_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain-material"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let material_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-material"),
            size: std::mem::size_of::<MaterialRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-material"),
            layout: &material_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });

        let env = super::EnvBinding::new(gpu);
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&tile_bgl),
                    Some(&material_bgl),
                    Some(&env.bgl),
                ],
                immediate_size: 0,
            });

        let vertex_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }];
        let instance_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
        ];

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &vertex_attrs,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<PatchRaw>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &instance_attrs,
                        },
                    ]
                    .map(Some),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    // Skirts are vertical walls of either winding; drop back-face
                    // culling so they always seal (terrain is opaque + depth-tested).
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: SCENE_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let lod_meshes = (0..TERRAIN_LOD_COUNT)
            .map(|lod| {
                let (verts, idx) = build_lod_geometry(cells_at_lod(lod));
                let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain-lod-verts"),
                    size: std::mem::size_of_val(verts.as_slice()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
                let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain-lod-indices"),
                    size: std::mem::size_of_val(idx.as_slice()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));
                LodMesh {
                    vertices,
                    indices,
                    index_count: idx.len() as u32,
                }
            })
            .collect();

        Self {
            pipeline,
            tile_bgl,
            lod_meshes,
            textures: BTreeMap::new(),
            material_buffer,
            material_bg,
            uploaded_material: None,
            env,
            instances: None,
            instance_capacity: 0,
        }
    }

    /// Reconcile the per-tile GPU texture cache with the projection (P16.3b1).
    ///
    /// Per-tile version-gated: [`plan_tile_cache`] names exactly the tiles whose
    /// stamp changed (or that are new / resized) and exactly the cached tiles that
    /// left the projection. Sculpting one tile re-uploads that tile and nothing
    /// else; evicting a page frees its textures the frame the projection drops it.
    /// The terrain-wide splat material rides a value compare beside it.
    fn sync_textures(&mut self, gpu: &GpuContext, terrain: &RenderTerrain) {
        // Splat material uniform (terrain-wide): write only on a real change.
        let material = MaterialRaw::from_terrain(terrain);
        if self.uploaded_material != Some(material) {
            gpu.queue
                .write_buffer(&self.material_buffer, 0, bytemuck::bytes_of(&material));
            self.uploaded_material = Some(material);
        }

        let plan = plan_tile_cache(&self.textures, |t| t.cached, terrain);
        for key in &plan.evict {
            self.textures.remove(key);
        }
        if plan.upload.is_empty() {
            return;
        }

        let res = terrain.tile_resolution.max(2);
        let expect = (res * res) as usize;

        // `plan.upload` is built by walking `terrain.tiles` in order, so it is a
        // subsequence of that walk — one shared cursor pairs each planned key back
        // to its tile with no lookup and no ordering assumption about the vec.
        let mut wanted = plan.upload.iter().peekable();
        for tile in &terrain.tiles {
            if wanted.peek() != Some(&&tile.key) {
                continue;
            }
            wanted.next();

            let entry = self
                .textures
                .entry(tile.key)
                .or_insert_with(|| create_tile_textures(gpu, &self.tile_bgl, res, tile.version));
            if entry.cached.resolution != res {
                *entry = create_tile_textures(gpu, &self.tile_bgl, res, tile.version);
            }
            entry.cached.version = tile.version;
            let tex = &*entry;
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex._height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&tile.heights),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 4),
                    rows_per_image: Some(res),
                },
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
            );

            // Splat weights: fall back to a uniform default when a tile projected
            // no (or a malformed) weight buffer, so shading stays layer 0. Coarse
            // pyramid pages always take this path — the pyramid is heights-only.
            // Uniform default = 100% layer 0 (mirrors inf_terrain::DEFAULT_WEIGHT;
            // kept local so the renderer stays independent of inf-terrain).
            const DEFAULT_WEIGHT: [u8; 4] = [255, 0, 0, 0];
            let default_weights;
            let weights: &[[u8; 4]] = if tile.weights.len() == expect {
                &tile.weights
            } else {
                default_weights = vec![DEFAULT_WEIGHT; expect];
                &default_weights
            };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex._weights,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(weights),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 4),
                    rows_per_image: Some(res),
                },
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
            );
        }
        debug_assert!(
            wanted.next().is_none(),
            "every planned upload must pair with a projected tile"
        );
    }
}

/// Create an empty `res × res` R32Float height texture + Rgba8Unorm weight
/// texture, sharing one bind group. `version` records the stamp the caller is
/// about to write into them.
fn create_tile_textures(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    res: u32,
    version: u64,
) -> TileTextures {
    let size = wgpu::Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: 1,
    };
    let make = |label: &str, format: wgpu::TextureFormat| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let height = make("terrain-height", wgpu::TextureFormat::R32Float);
    let weights = make("terrain-weights", wgpu::TextureFormat::Rgba8Unorm);
    let hview = height.create_view(&wgpu::TextureViewDescriptor::default());
    let wview = weights.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain-tile"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hview),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&wview),
            },
        ],
    });
    TileTextures {
        _height: height,
        _weights: weights,
        bind_group,
        cached: CachedTile {
            version,
            resolution: res,
        },
    }
}

impl RenderNode for TerrainNode {
    fn name(&self) -> &'static str {
        "terrain"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let Some(terrain) = frame.scene.terrain.as_ref() else {
            return; // no terrain → byte-identical to pre-P10.1
        };
        if terrain.tiles.is_empty() {
            return;
        }
        self.sync_textures(gpu, terrain);

        let patches = assemble_patches(terrain, frame.view, &frame.view.origin);
        if patches.is_empty() {
            return;
        }

        // Pack per-patch instance data (one per visible patch, in patch order —
        // `patch.tile` indexes the source tile directly, so instance index i is
        // always patch i and the two can never drift apart).
        let res = terrain.tile_resolution.max(2) as f32;
        let origin = &frame.view.origin;
        let mut raw: Vec<PatchRaw> = Vec::with_capacity(patches.len());
        for p in &patches {
            let tile = &terrain.tiles[p.tile];
            // A coarse page covers 2^lod × the level-0 span; the skirt depth (and
            // the cull margin above) scale with it, which is what keeps the
            // fine↔coarse seam sealed (see the module docs).
            let span = terrain.tile_span_at(p.key.lod) as f32;
            let o = origin.to_render(tile.origin);
            let (hmin, hmax) = tile.height_bounds;
            let skirt = (hmax - hmin).abs().max(span * 0.05).max(1.0);
            raw.push(PatchRaw {
                o_span: [o.x, o.y, o.z, span],
                params: [p.morph, cells_at_lod(p.mesh_lod) as f32, res, skirt],
            });
        }
        if raw.is_empty() {
            return;
        }

        if self.instances.is_none() || self.instance_capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(16);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain-instances"),
                size: (capacity * std::mem::size_of::<PatchRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = capacity;
        }
        let inst_buf = self.instances.as_ref().unwrap();
        gpu.queue
            .write_buffer(inst_buf, 0, bytemuck::cast_slice(&raw));
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(2, &self.material_bg, &[]);
        pass.set_bind_group(3, &env_bg, &[]);
        pass.set_vertex_buffer(1, inst_buf.slice(..));

        for (idx, patch) in patches.iter().enumerate() {
            let Some(tex) = self.textures.get(&patch.key) else {
                continue;
            };
            let mesh = &self.lod_meshes[patch.mesh_lod.min(TERRAIN_LOD_COUNT - 1) as usize];
            pass.set_bind_group(1, &tex.bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            let i = idx as u32;
            pass.draw_indexed(0..mesh.index_count, 0, i..i + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    const RES: u32 = 16;
    const MPS: f64 = 1.0;
    /// Level-0 tile span for the fixtures below.
    const SPAN0: f64 = (RES - 1) as f64 * MPS;

    /// One flat page at `key`, placed at its level's grid pitch (`SPAN0 · 2^lod`).
    fn tile_at(key: TerrainTileKey, version: u64) -> RenderTerrainTile {
        let span = SPAN0 * (1u64 << key.lod) as f64;
        RenderTerrainTile {
            key,
            origin: DVec3::new(key.coord.0 as f64 * span, 0.0, key.coord.1 as f64 * span),
            heights: vec![0.0; (RES * RES) as usize],
            weights: Vec::new(),
            height_bounds: (0.0, 0.0),
            version,
        }
    }

    /// A terrain over an explicit residency set (key order preserved), stamped
    /// `1` per tile.
    fn terrain_of(keys: impl IntoIterator<Item = TerrainTileKey>) -> RenderTerrain {
        RenderTerrain {
            tile_resolution: RES,
            meters_per_sample: MPS,
            tiles: keys.into_iter().map(|k| tile_at(k, 1)).collect(),
            layers: Default::default(),
            macro_variation: 0.0,
        }
    }

    fn flat_terrain(ntiles: i32) -> RenderTerrain {
        // Tuple-sorted order (tx outer, tz inner) — matches the BTreeMap the host
        // projects from, so the assembled patch order is `(i32,i32)`-sorted.
        terrain_of(
            (0..ntiles).flat_map(|tx| (0..ntiles).map(move |tz| TerrainTileKey::lod0((tx, tz)))),
        )
    }

    #[test]
    fn cells_halve_per_lod() {
        assert_eq!(cells_at_lod(0), 32);
        assert_eq!(cells_at_lod(1), 16);
        assert_eq!(cells_at_lod(2), 8);
        assert_eq!(cells_at_lod(3), 4);
        // Beyond the coarsest level clamps.
        assert_eq!(cells_at_lod(99), 4);
    }

    #[test]
    fn lod_rises_with_distance() {
        let span = 100.0;
        let t = lod_thresholds(span); // [150, 300, 600]
        assert_eq!(lod_for_distance(0.0, &t), 0);
        assert_eq!(lod_for_distance(149.0, &t), 0);
        assert_eq!(lod_for_distance(151.0, &t), 1);
        assert_eq!(lod_for_distance(301.0, &t), 2);
        assert_eq!(lod_for_distance(10_000.0, &t), TERRAIN_LOD_COUNT - 1);
    }

    #[test]
    fn morph_ramps_within_band_and_zero_at_coarsest() {
        let span = 100.0;
        let t = lod_thresholds(span); // [150, 300, 600]
                                      // Deep inside LOD-0 band: no morph.
        assert_eq!(morph_factor(10.0, 0, &t), 0.0);
        // Just below the LOD-0→1 threshold: fully morphed.
        assert!(morph_factor(149.9, 0, &t) > 0.99);
        // Mid-band monotonic 0→1.
        let a = morph_factor(120.0, 0, &t);
        let b = morph_factor(140.0, 0, &t);
        assert!((0.0..=1.0).contains(&a) && a <= b);
        // Coarsest LOD never morphs.
        assert_eq!(morph_factor(10_000.0, TERRAIN_LOD_COUNT - 1, &t), 0.0);
    }

    #[test]
    fn assembly_is_deterministic_and_sorted() {
        let terrain = flat_terrain(3);
        let view = top_view();
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let a = assemble_patches(&terrain, &view, &origin);
        let b = assemble_patches(&terrain, &view, &origin);
        assert_eq!(a, b, "assembly must be deterministic");
        // Keys are non-decreasing (tiles arrive key-sorted).
        for w in a.windows(2) {
            assert!(w[0].key <= w[1].key);
        }
        assert!(!a.is_empty());
    }

    #[test]
    fn nearest_tile_is_finest_lod() {
        let terrain = flat_terrain(3);
        let view = top_view(); // eye over tile (0,0)
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let patches = assemble_patches(&terrain, &view, &origin);
        let near = patches
            .iter()
            .find(|p| p.key == TerrainTileKey::lod0((0, 0)))
            .unwrap();
        assert_eq!(near.ring, 0, "the tile under the camera should be ring 0");
        assert_eq!(near.mesh_lod, 0, "ring 0 over a level-0 page → finest grid");
    }

    #[test]
    fn offscreen_tiles_are_culled() {
        // A single tile far behind the camera is culled out.
        let mut terrain = terrain_of([TerrainTileKey::lod0((0, 0))]);
        terrain.tiles[0].origin = DVec3::new(0.0, 0.0, 1000.0); // behind a -Z camera
        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(7.0, 30.0, -20.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        };
        assert!(assemble_patches(&terrain, &view, &FloatingOrigin::new(DVec3::ZERO)).is_empty());
    }

    #[test]
    fn triplanar_weights_normalize_and_favour_dominant_axis() {
        // A flat-top normal (+Y) → the XZ (y) plane dominates and the weights sum
        // to 1.
        let w = triplanar_axis_weights(Vec3::Y, 4.0);
        let sum = w[0] + w[1] + w[2];
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1: {w:?}");
        assert!(
            w[1] > w[0] && w[1] > w[2],
            "top plane should dominate: {w:?}"
        );

        // A 45° X/Y normal with higher sharpness biases harder toward the larger
        // component than a low sharpness does.
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let soft = triplanar_axis_weights(n, 1.0);
        let hard = triplanar_axis_weights(n, 8.0);
        assert!((soft[0] - soft[1]).abs() < 1e-5, "equal axes stay equal");
        assert!((hard[0] - hard[1]).abs() < 1e-5);
        // A steep (mostly-X) normal: sharpness pushes weight onto the YZ (x) plane.
        let steep = Vec3::new(0.9, 0.2, 0.386).normalize();
        let s1 = triplanar_axis_weights(steep, 1.0);
        let s8 = triplanar_axis_weights(steep, 8.0);
        assert!(
            s8[0] > s1[0],
            "sharper blend concentrates on the dominant axis"
        );
    }

    #[test]
    fn triplanar_degenerate_normal_falls_back_to_top() {
        let w = triplanar_axis_weights(Vec3::ZERO, 4.0);
        assert_eq!(w, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn macro_modulation_scales_albedo() {
        let a = Vec3::new(0.4, 0.5, 0.6);
        // No fBm → unchanged.
        assert_eq!(macro_modulate(a, 0.0, 0.3), a);
        // Positive fBm brightens, negative darkens, symmetrically.
        let up = macro_modulate(a, 1.0, 0.2);
        let dn = macro_modulate(a, -1.0, 0.2);
        assert!(up.x > a.x && dn.x < a.x);
        assert!(
            (up + dn - a * 2.0).length() < 1e-6,
            "modulation is symmetric"
        );
    }

    #[test]
    fn material_raw_packs_layers_and_macro() {
        let mut terrain = flat_terrain(1);
        terrain.layers[2].albedo = [0.1, 0.2, 0.3, 1.0];
        terrain.layers[2].roughness = 0.25;
        terrain.layers[2].tex_scale = 3.0;
        terrain.macro_variation = 0.7;
        let raw = MaterialRaw::from_terrain(&terrain);
        assert_eq!(raw.albedo[2], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(raw.params[2][0], 0.25);
        assert_eq!(raw.params[2][1], 3.0);
        assert_eq!(raw.macro_amp[0], 0.7);
    }

    #[test]
    fn lod_geometry_has_skirts() {
        // A 2-cell patch: 9 interior verts + 4 skirt strips of (3 verts each).
        let (verts, indices) = build_lod_geometry(2);
        // Interior 3×3 = 9, plus 4 edges × 3 skirt copies = 12 → 21.
        assert_eq!(verts.len(), 9 + 12);
        // Interior tris: 2×2 cells × 2 = 8 tris (24 idx). Skirts: each edge has 2
        // segments → 2 quads → 4 tris → 24 idx; ×4 edges = 48. Total 72.
        assert_eq!(indices.len(), 24 + 48);
        // Skirt verts carry the flag; interior verts don't.
        assert!(verts[..9].iter().all(|v| v[2] == 0.0));
        assert!(verts[9..].iter().all(|v| v[2] == 1.0));
    }

    // ── P16.3b1: ring → asset LOD, source selection, upload gating ───────────

    #[test]
    fn ring_maps_one_for_one_onto_asset_lod() {
        // Both ladders double per step, so ring r ↔ asset LOD r …
        for ring in 0..TERRAIN_LOD_COUNT {
            assert_eq!(ring_source_lod(ring, 8), ring);
        }
        // … clamped to the coarsest level actually projected.
        assert_eq!(ring_source_lod(3, 1), 1);
        assert_eq!(
            ring_source_lod(3, 0),
            0,
            "level-0-only ⇒ every ring wants 0"
        );
        assert_eq!(ring_source_lod(0, 5), 0);

        // The mesh supplies whatever decimation the asset LOD didn't: a level-2
        // page in ring 2 draws the FULL-density grid over its 4× footprint — the
        // same world cell size as a level-0 page at ring 0.
        assert_eq!(patch_mesh_lod(2, 2), 0);
        assert_eq!(cells_at_lod(patch_mesh_lod(2, 2)), cells_at_lod(0));
        // A level-0 page is unchanged from pre-P16.3 (mesh lod == ring).
        for ring in 0..TERRAIN_LOD_COUNT {
            assert_eq!(patch_mesh_lod(ring, 0), ring);
        }
        // A page coarser than its ring wants can't go below the finest grid, and
        // the index never escapes the mesh table.
        assert_eq!(patch_mesh_lod(1, 3), 0);
        assert_eq!(patch_mesh_lod(99, 0), TERRAIN_LOD_COUNT - 1);
    }

    #[test]
    fn tile_span_doubles_per_asset_lod() {
        let terrain = flat_terrain(1);
        assert_eq!(terrain.tile_span_at(0), terrain.tile_span());
        assert_eq!(terrain.tile_span_at(1), terrain.tile_span() * 2.0);
        assert_eq!(terrain.tile_span_at(3), terrain.tile_span() * 8.0);
        assert_eq!(terrain.max_lod(), 0);
    }

    #[test]
    fn key_parent_child_round_trip_over_negative_coords() {
        // FLOOR semantics, not truncation: tile −1 belongs to block −1.
        assert_eq!(
            TerrainTileKey::lod0((-1, -1)).parent(),
            TerrainTileKey::new(1, (-1, -1))
        );
        assert_eq!(
            TerrainTileKey::lod0((-2, 3)).parent(),
            TerrainTileKey::new(1, (-1, 1))
        );
        // Every child's parent is the tile it came from, on both signs.
        for coord in [(0, 0), (-1, -1), (-3, 2), (5, -7)] {
            let parent = TerrainTileKey::new(2, coord);
            for child in parent.children() {
                assert_eq!(child.lod, 1);
                assert_eq!(child.parent(), parent, "child {child:?} of {parent:?}");
            }
        }
        assert_eq!(
            TerrainTileKey::new(1, (-1, -1)).children(),
            [
                TerrainTileKey::lod0((-2, -2)),
                TerrainTileKey::lod0((-1, -2)),
                TerrainTileKey::lod0((-2, -1)),
                TerrainTileKey::lod0((-1, -1)),
            ]
        );
    }

    /// The residency set of a 2 × 2 level-0 block plus the level-1 page over it.
    fn block_and_parent(coord: (i32, i32)) -> BTreeSet<TerrainTileKey> {
        let parent = TerrainTileKey::new(1, coord);
        parent.children().into_iter().chain([parent]).collect()
    }

    #[test]
    fn fine_wins_when_both_could_serve() {
        // A coarse page whose four children are all resident never draws — in ANY
        // ring — and its children always do.
        let resident = block_and_parent((0, 0));
        let parent = TerrainTileKey::new(1, (0, 0));
        for ring in 0..TERRAIN_LOD_COUNT {
            assert!(
                superseded(parent, ring, 1, &resident),
                "fully-subdivided coarse page must yield in ring {ring}"
            );
            for child in parent.children() {
                assert!(
                    !superseded(child, ring, 1, &resident),
                    "fine page must draw in ring {ring} (fine wins)"
                );
            }
        }
        // Same on negative coordinates.
        let resident = block_and_parent((-1, -2));
        assert!(superseded(
            TerrainTileKey::new(1, (-1, -2)),
            2,
            1,
            &resident
        ));
        assert!(!superseded(TerrainTileKey::lod0((-2, -4)), 2, 1, &resident));
    }

    #[test]
    fn coarse_serves_the_outer_rings_without_punching_holes() {
        // Partial fine coverage: 3 of 4 children resident, so the level-1 page is
        // NOT fully subdivided and stays available as the outer-ring source.
        let parent = TerrainTileKey::new(1, (0, 0));
        let kids = parent.children();
        let resident: BTreeSet<_> = kids[..3].iter().copied().chain([parent]).collect();

        // Outer ring (wants level 1): the coarse page draws, the fine ones stand
        // down — one patch over the whole footprint, no double draw.
        assert!(!superseded(parent, 1, 1, &resident));
        for child in &kids[..3] {
            assert!(superseded(*child, 1, 1, &resident));
        }

        // Innermost ring (wants level 0): the fine pages draw. The coarse page is
        // not suppressed — it still covers the missing quadrant, so the frame has
        // overdraw rather than a hole (the documented preference).
        for child in &kids[..3] {
            assert!(!superseded(*child, 0, 1, &resident));
        }
        assert!(!superseded(parent, 0, 1, &resident));
    }

    /// Recursively: is `key`'s footprint fully covered by drawn pages — either
    /// because `key` itself draws, or because every descendant branch under it
    /// resolves to drawn pages?
    fn footprint_drawn(
        key: TerrainTileKey,
        ring: u32,
        max_lod: u32,
        resident: &BTreeSet<TerrainTileKey>,
    ) -> bool {
        if resident.contains(&key) && !superseded(key, ring, max_lod, resident) {
            return true;
        }
        key.lod >= 1
            && key
                .children()
                .iter()
                .all(|c| resident.contains(c) && footprint_drawn(*c, ring, max_lod, resident))
    }

    #[test]
    fn every_footprint_keeps_a_drawer_across_a_three_level_pyramid() {
        // The hole-free invariant, brute-forced over EVERY residency shape a
        // three-level pyramid can take: all 2^4 subsets of the level-1 keys × all
        // 2^4 subsets of one level-1 parent's children, in every ring. That covers
        // the shapes a fully-stocked set can't reach — 3-of-4 children, an orphan
        // level-0 branch whose parent is absent, a bare level-2 page — which are
        // exactly the mid-stream states a streamer passes through.
        //
        // The claim: for every resident key, its ground is drawn by *something* —
        // the key itself, a drawn ancestor above it, or its descendants all the
        // way down. Nothing is ever suppressed into a hole.
        let l2 = TerrainTileKey::new(2, (0, 0));
        let l1 = l2.children();
        let l0 = l1[0].children(); // one parent's children — the subdivided branch

        for m1 in 0u32..16 {
            for m0 in 0u32..16 {
                fn pick(
                    keys: [TerrainTileKey; 4],
                    mask: u32,
                ) -> impl Iterator<Item = TerrainTileKey> {
                    (0..4)
                        .filter(move |i| mask >> i & 1 == 1)
                        .map(move |i| keys[i])
                }
                let resident: BTreeSet<_> = std::iter::once(l2)
                    .chain(pick(l1, m1))
                    .chain(pick(l0, m0))
                    .collect();

                for ring in 0..TERRAIN_LOD_COUNT {
                    for key in &resident {
                        let by_ancestor =
                            std::iter::successors(Some(*key), |k| (k.lod < 2).then(|| k.parent()))
                                .skip(1)
                                .any(|a| {
                                    resident.contains(&a) && !superseded(a, ring, 2, &resident)
                                });
                        assert!(
                            by_ancestor || footprint_drawn(*key, ring, 2, &resident),
                            "l1 mask {m1:04b}, l0 mask {m0:04b}, ring {ring}: {key:?} was \
                             suppressed with nothing left to cover it"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_culled_ancestor_never_suppresses_a_visible_child() {
        // A coarse page's AABB can be VERTICALLY NARROWER than its child's:
        // decimation samples every other height, so a spike a child keeps can be
        // missing from the parent's bounds. A camera can therefore cull the parent
        // while the child is still on screen — and suppressing the child in favour
        // of a page nobody draws would be sky through the ground.
        let parent = TerrainTileKey::new(1, (0, 0));
        let child = TerrainTileKey::lod0((1, 1)); // one of the four → parent not subdivided
        let mut terrain = terrain_of([child, parent]);
        // The child keeps a tall spike; the coarse page decimated it away and is a
        // flat slab at y = 0.
        terrain.tiles[0].height_bounds = (0.0, 250.0);
        terrain.tiles[1].height_bounds = (0.0, 0.0);

        // Eye at y = 200 looking level down +X: at horizontal distance d the
        // frustum spans y ∈ [200 − d, 200 + d]. The tiles sit ~100–130 m ahead, so
        // the flat parent (y = 0) falls below the bottom plane while the child's
        // box (up to y = 250) stays inside.
        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(-100.0, 200.0, SPAN0 * 1.5),
            forward: Vec3::X,
            up: Vec3::Y,
            fov_y: 90f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        };
        let patches = assemble_patches(&terrain, &view, &view.origin);
        let drawn: Vec<_> = patches.iter().map(|p| p.key).collect();
        assert!(
            !drawn.contains(&parent),
            "fixture is wrong: the coarse page must be culled for this test to bite"
        );
        assert!(
            drawn.contains(&child),
            "a frustum-culled ancestor suppressed a visible child — that is a hole"
        );
        // The child's ring must be far enough out that the ancestor clause would
        // actually have fired had the parent been in the index.
        let ring = patches.iter().find(|p| p.key == child).unwrap().ring;
        assert!(
            ring >= 1,
            "child landed in ring {ring}; clause 2 never applies"
        );
        let all: BTreeSet<_> = terrain.tiles.iter().map(|t| t.key).collect();
        assert!(
            superseded(child, ring, 1, &all),
            "…and it WOULD have been suppressed against the unculled set"
        );
    }

    #[test]
    fn coarse_pages_assemble_into_the_outer_rings() {
        // A quadtree cut: level-0 pages under the camera, one level-1 page far
        // down +X. Both make it into the patch list, with the coarse page carrying
        // its own (doubled) span and the mesh density its ring implies.
        let mut keys: Vec<_> = (0..2)
            .flat_map(|tx| (0..2).map(move |tz| TerrainTileKey::lod0((tx, tz))))
            .collect();
        keys.push(TerrainTileKey::new(1, (4, 0)));
        let terrain = terrain_of(keys);
        assert_eq!(terrain.max_lod(), 1);

        // Straight down over the middle of the whole strip, wide enough that
        // nothing is culled — the selection, not the frustum, is under test.
        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(SPAN0 * 5.0, 200.0, SPAN0),
            forward: Vec3::new(0.0, -1.0, 0.001).normalize(),
            up: Vec3::Z,
            fov_y: 90f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        };
        let patches = assemble_patches(&terrain, &view, &view.origin);
        let coarse = patches
            .iter()
            .find(|p| p.key.lod == 1)
            .expect("the coarse page must render the outer ring");
        assert!(
            coarse.ring >= 1,
            "coarse page landed in ring {}",
            coarse.ring
        );
        assert_eq!(coarse.mesh_lod, patch_mesh_lod(coarse.ring, 1));
        assert!(
            patches.iter().any(|p| p.key.lod == 0),
            "the near level-0 pages must still render"
        );
        // Deterministic + every patch indexes its own source tile.
        assert_eq!(patches, assemble_patches(&terrain, &view, &view.origin));
        for p in &patches {
            assert_eq!(terrain.tiles[p.tile].key, p.key);
        }
    }

    #[test]
    fn a_punched_out_residency_set_drops_only_that_patch() {
        // Level-0-only residency with an interior tile missing: no patch for it,
        // no panic, and every neighbour is assembled exactly as if it were there.
        let full = flat_terrain(3);
        let hole = TerrainTileKey::lod0((1, 1));
        let punched = terrain_of(
            full.tiles
                .iter()
                .map(|t| t.key)
                .filter(|&k| k != hole)
                .collect::<Vec<_>>(),
        );
        let view = top_view();
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let before = assemble_patches(&full, &view, &origin);
        let after = assemble_patches(&punched, &view, &origin);

        assert!(before.iter().any(|p| p.key == hole));
        assert!(!after.iter().any(|p| p.key == hole), "no patch for a hole");
        // Neighbours: same ring, same mesh, same morph as in the full set.
        for p in &after {
            let same = before.iter().find(|q| q.key == p.key).unwrap();
            assert_eq!(
                (p.key, p.ring, p.mesh_lod, p.morph),
                (same.key, same.ring, same.mesh_lod, same.morph),
                "a neighbour changed when {hole:?} was punched out"
            );
        }
        assert_eq!(after.len(), before.len() - 1);
    }

    /// The cache state a plan is computed against, from `(key, version)` pairs at
    /// the fixture resolution.
    fn cache_of(
        entries: impl IntoIterator<Item = (TerrainTileKey, u64)>,
    ) -> BTreeMap<TerrainTileKey, CachedTile> {
        entries
            .into_iter()
            .map(|(k, version)| {
                (
                    k,
                    CachedTile {
                        version,
                        resolution: RES,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn upload_set_is_exactly_the_changed_stamps() {
        let mut terrain = flat_terrain(2); // (0,0) (0,1) (1,0) (1,1)
        for (i, tile) in terrain.tiles.iter_mut().enumerate() {
            tile.version = 10 + i as u64;
        }
        let keys: Vec<_> = terrain.tiles.iter().map(|t| t.key).collect();

        // Cold cache ⇒ everything uploads.
        let plan = plan_tile_cache(&BTreeMap::new(), |c: &CachedTile| *c, &terrain);
        assert_eq!(plan.upload, keys);
        assert!(plan.evict.is_empty());

        // Warm, in sync ⇒ nothing at all moves.
        let warm = cache_of(terrain.tile_versions());
        let plan = plan_tile_cache(&warm, |c| *c, &terrain);
        assert_eq!(plan, TileCachePlan::default());

        // Sculpt ONE tile (its stamp advances) ⇒ exactly that tile re-uploads.
        terrain.tiles[2].version = 99;
        let plan = plan_tile_cache(&warm, |c| *c, &terrain);
        assert_eq!(plan.upload, vec![keys[2]]);
        assert!(plan.evict.is_empty());

        // A projected stamp of 0 is "no stamp" — never a cache hit, even when the
        // cached entry also reads 0.
        let mut unstamped = flat_terrain(1);
        unstamped.tiles[0].version = 0;
        let plan = plan_tile_cache(&cache_of([(keys[0], 0)]), |c| *c, &unstamped);
        assert_eq!(plan.upload, vec![keys[0]]);

        // A resolution change invalidates the cached textures on its own (this
        // cache is rebuilt from the *current* stamps, so nothing else is stale).
        let mut resized = cache_of(terrain.tile_versions());
        resized.get_mut(&keys[1]).unwrap().resolution = RES * 2;
        let plan = plan_tile_cache(&resized, |c| *c, &terrain);
        assert_eq!(plan.upload, vec![keys[1]]);
    }

    #[test]
    fn cache_evicts_tiles_that_left_the_projection() {
        let full = flat_terrain(2);
        let warm = cache_of(full.tile_versions());

        // Shrink the residency set to one tile: the other three are evicted, and
        // the survivor neither uploads nor moves.
        let kept = TerrainTileKey::lod0((1, 1));
        let shrunk = terrain_of([kept]);
        let plan = plan_tile_cache(&warm, |c| *c, &shrunk);
        assert!(plan.upload.is_empty(), "{plan:?}");
        assert_eq!(
            plan.evict,
            vec![
                TerrainTileKey::lod0((0, 0)),
                TerrainTileKey::lod0((0, 1)),
                TerrainTileKey::lod0((1, 0)),
            ],
            "eviction list is ascending by key"
        );

        // Level-0 and coarse pages of the same ground cache independently.
        let mixed = terrain_of([TerrainTileKey::lod0((0, 0)), TerrainTileKey::new(1, (0, 0))]);
        let plan = plan_tile_cache(&BTreeMap::new(), |c: &CachedTile| *c, &mixed);
        assert_eq!(
            plan.upload,
            vec![TerrainTileKey::lod0((0, 0)), TerrainTileKey::new(1, (0, 0))]
        );

        // A malformed page is never uploaded, and never kept.
        let mut broken = flat_terrain(1);
        broken.tiles[0].heights.truncate(3);
        let plan = plan_tile_cache(&warm, |c| *c, &broken);
        assert!(plan.upload.is_empty());
        assert_eq!(plan.evict.len(), 4, "including the malformed tile's cache");
    }

    /// Switching levels must invalidate the cache even though the tile *keys* are
    /// identical — the cache is addressed by key alone, so the stamps are the only
    /// thing that can tell terrain B's tile (0,0) from terrain A's.
    ///
    /// This is the renderer half of the contract `inf_terrain`'s process-global
    /// stamp counter provides (`distinct_terrains_mint_disjoint_stamps`): were the
    /// counter per-terrain, both levels would project 1, 2, 3, … over the same
    /// coordinates, this plan would come back empty, and the frame would render
    /// the previous level's height texels.
    #[test]
    fn a_different_terrain_with_the_same_keys_must_re_upload() {
        let mut level_a = flat_terrain(2);
        let mut level_b = flat_terrain(2);
        for (i, tile) in level_a.tiles.iter_mut().enumerate() {
            tile.version = 1 + i as u64; // as a fresh load stamps them
        }
        for (i, tile) in level_b.tiles.iter_mut().enumerate() {
            tile.version = 5 + i as u64; // strictly later, disjoint stamps
        }
        assert_eq!(
            level_a.tiles.iter().map(|t| t.key).collect::<Vec<_>>(),
            level_b.tiles.iter().map(|t| t.key).collect::<Vec<_>>(),
            "the fixture only bites while both levels share every tile key"
        );

        let warm = cache_of(level_a.tile_versions());
        assert_eq!(
            plan_tile_cache(&warm, |c| *c, &level_a),
            TileCachePlan::default(),
            "the level it was filled from is a clean cache hit"
        );
        let plan = plan_tile_cache(&warm, |c| *c, &level_b);
        assert_eq!(
            plan.upload,
            level_b.tiles.iter().map(|t| t.key).collect::<Vec<_>>(),
            "every tile of the new level must re-upload"
        );
        assert!(
            plan.evict.is_empty(),
            "the keys are shared — nothing to free"
        );
    }

    /// A camera looking down over the origin tile.
    fn top_view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(7.5, 25.0, 7.5),
            forward: Vec3::new(0.0, -1.0, 0.001).normalize(),
            up: Vec3::Z,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        }
    }
}
