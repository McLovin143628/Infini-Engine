//! GPU-driven virtualized-geometry (meshlet) render path (P13.1b).
//!
//! The portable, mesh-shader-free Nanite-class runtime: a **compute cull+LOD**
//! pass selects the visible meshlets of every placed [`VgeomInstance`], then a
//! single **vertex-pulled `draw_indirect`** rasterizes them.
//!
//! # Pipeline (what happens each frame, per referenced vmesh asset)
//!
//! 1. **Stream / page** (P18.2) — the referenced assets' meshlet pages are paged
//!    into four **shared, suballocated** storage pools (vertices, meshlet
//!    descriptors, meshlet→vertex indices, packed meshlet→triangle bytes) by the
//!    node's [`inf_vgeom::VgeomStreamer`], against a VRAM budget and the camera's
//!    own screen-error wants. Per-instance transforms are re-packed each frame
//!    (the LOD threshold depends on the live camera distance).
//! 2. **Cull compute** (`vgeom_cull.wgsl`) — one thread per (instance, meshlet):
//!    the LOD cut, frustum-sphere, backface-cone, and optional HZB tests; each
//!    survivor is appended to a `visible` list via an atomic that **is** the
//!    indirect draw's `instance_count`.
//! 3. **Raster** (`vgeom_mesh.wgsl`) — one non-indexed `draw_indirect` of
//!    `visible_count` instances, each drawing `max_tri*3` vertices pulled from the
//!    storage buffers (unused triangles collapse to a degenerate off-screen point).
//!    Shading is the same metallic-roughness PBR as the rigid mesh pass (shared
//!    lights `@group(1)` + AO `@group(2)`), into the same MSAA scene targets.
//!
//! # The vertex-pulling draw (exact buffers / indices)
//!
//! There are **no vertex buffers**. The draw is `draw_indirect` with args
//! `[vertex_count = max_tri*3, instance_count = visible, first_vertex = 0,
//! first_instance = 0]` (non-indexed — equivalent to the classic "shared
//! `0..max_tri*3` identity index buffer + `draw_indexed_indirect`" technique but
//! without the redundant identity index buffer). In the vertex shader:
//! `instance_index → visible[i] → (global_instance, meshlet_id)`;
//! `vertex_index → triangle = vertex_index/3, corner = vertex_index%3`; the
//! triangle's local index is read from the packed `meshlet_triangles` bytes,
//! resolved through `meshlet_vertices` to a global vertex, and transformed by the
//! instance model matrix. Cost of the fixed vertex count: up to
//! `(max_tri − triangle_count)*3` degenerate vertices per meshlet (a well-known
//! trade for a single indirect draw with no per-meshlet draw args).
//!
//! # Screen-space error projection (the LOD cut)
//!
//! We project the requested **pixel tolerance** to a *single per-instance
//! object-space scalar threshold* `t`, then apply the branchless cut
//! `error ≤ t < parent_error` per meshlet — which is **bit-identical** to
//! [`VgeomMesh::select(t)`](inf_vgeom::VgeomMesh::select). Using one `t` per mesh
//! instance (not per meshlet) is what preserves the DAG *cut invariant* (the
//! half-open `[error, parent_error)` intervals tile `[0,∞)` only against a single
//! threshold). For a perspective view of an instance whose bounding sphere is at
//! distance `d` (to the sphere surface) with instance scale `s`:
//!
//! ```text
//!   focal_px = screen_height / (2·tan(fov_y/2))
//!   screen_error_px(object_error) = object_error · s · focal_px / d
//!   ⇒  t = pixel_tol · d / (focal_px · s)          // solve screen_error == tol
//! ```
//!
//! `t` is monotonic (a positive scaling) in `object_error`, so the cut interval
//! test survives the projection. Farther instances ⇒ larger `d` ⇒ larger `t` ⇒
//! coarser meshlets selected ⇒ fewer visible meshlets (the LOD proof).
//!
//! # Determinism
//!
//! The atomic append order varies (so the *draw order* varies), but the visible
//! **set** is a pure function of (scene, view, settings). Meshlet raster is opaque
//! and depth-tested, so the resulting image is order-independent — byte-stable
//! across renders of the same frame ([`crate::golden`] asserts it).
//!
//! # Two-pass HZB occlusion (P18.1)
//!
//! [`VgeomSettings::occlusion`] adds an HZB depth test to the cull compute, and
//! [`VgeomSettings::two_pass`] makes it the real **two-pass** technique: vgeom
//! geometry occludes vgeom geometry. Per frame, this node records
//!
//! ```text
//!   for each asset:  early cull  →  early draw          (last-frame-visible)
//!   build HZB from the live MSAA scene depth            (classic + early vgeom)
//!   for each asset:  late cull   →  late draw           (newly disoccluded)
//! ```
//!
//! Both draws are the same vertex-pulled `draw_indirect` into the same MSAA
//! targets; the late pass owns a second visible list + args buffer (a non-zero
//! `first_instance` would need the non-portable `INDIRECT_FIRST_INSTANCE`
//! feature, so two buffers it is).
//!
//! The HZB seeds from `targets.depth` — the live 4× MSAA scene depth — rather
//! than the single-sample prepass. See `vgeom_hzb.wgsl` for why (a different
//! rasterization cannot prove anything about the one the meshlets test against;
//! min-over-samples can). Consequences: the classic mesh pass still occludes
//! meshlets exactly as in v1 (it runs earlier and writes that same target), the
//! early vgeom draw *adds* to it for free, no extra depth write or resolve is
//! paid, and [`RenderSettings::needs_depth_prepass`] no longer has to force a
//! full-res prepass just to enable occlusion. Passes that run *after* this node
//! (terrain, skinned, translucent) are not in the pyramid — they cannot occlude
//! meshlets, which costs culls, never correctness.
//!
//! ## The determinism contract
//!
//! Two-pass occlusion is temporal: frame N's *early set* is frame N−1's visible
//! set (`prev_visible`, a GPU-resident u32 per (instance, meshlet) pair, ping-
//! ponged — never read back). The house gates assume a frame is a pure function
//! of (scene, view, settings), so the split is designed so that it **is**:
//!
//! 1. **Occlusion is purely subtractive.** The HZB test only ever removes
//!    meshlets it *proves* contribute zero fragments — the bound is computed from
//!    the meshlet sphere's world-AABB corners, the mip is chosen so a 2×2 gather
//!    provably covers the screen rect, and mip 0 is the min over MSAA subsamples.
//!    The full argument (and where each approximation is rounded to stay
//!    conservative) is in `vgeom_cull.wgsl::occluded`. So for any frame,
//!    *whatever* the temporal state:
//!
//!    ```text
//!    image(occlusion on) == image(occlusion off)      — pixel-identical
//!    ```
//!
//!    Temporal state chooses WHEN a meshlet draws (early vs late pass), never
//!    WHETHER the union covers it. A hole is not reachable, including on the
//!    first frame after a teleport.
//! 2. **No usable state ⇒ conservative.** When there is nothing to inherit — the
//!    first frame, a `scene.version` bump, an instance/meshlet count change, a
//!    frame-target reallocation, or a **camera cut** ([`is_camera_cut`]) — the
//!    early set is the *whole* base cut, so that frame's drawn set is exactly the
//!    single-pass, occlusion-off set: bit-identical to the pre-P18.1 path. This is
//!    a quality/cost heuristic (a stale early set draws the wrong depth and makes
//!    the late pass do all the work), **not** a correctness dependency — point 1
//!    already rules holes out. Nothing downstream may start relying on it.
//! 3. **The CPU-parity reference is untouched.** [`cull_visible`] always runs
//!    `MODE_SINGLE` with occlusion forced off, so the GPU cut it reads back is the
//!    LOD+frustum+cone cut the CPU reference mirrors. Occlusion is a filter
//!    *after* that cut, never part of it.
//!
//! Because a fresh [`crate::EngineRenderer`] has no state, every golden — which
//! renders one frame from cold — takes the conservative branch and is byte-
//! identical to its pre-P18.1 self even with occlusion on by default. The
//! *converged* behaviour is gated separately (`tests/vgeom_occlusion.rs`).
//!
//! Set [`VgeomSettings::two_pass`] to `false` to fall back to single-pass v1
//! (one cull, one draw, HZB from whatever the scene depth holds before this node).
//!
//! # Meshlet streaming (P18.2)
//!
//! Nothing uploads a whole vmesh any more. A [`VgeomAsset`](crate::VgeomAsset)
//! carries a lazily-indexed [`VgeomSource`] — header and page directory only —
//! and this node owns an [`inf_vgeom::VgeomStreamer`] that pages the DAG in by
//! **page**: page 0 is every root of the DAG (always resident), pages 1.. are the
//! non-root meshlets of one LOD level each, coarse to fine. Residency is always a
//! prefix of that order, so a root-to-leaf path always has a resident meshlet:
//! partial residency is **softer detail, never a hole**.
//!
//! Per frame, before any culling:
//!
//! ```text
//!   wants  = per asset, the SMALLEST per-instance LOD threshold t
//!   plan   = streamer.plan(wants)        // deterministic, budgeted, bounded
//!   upload = write plan.uploads into the four shared pools
//!   cull   = the base cut, clamped to residency, through the per-asset remap
//! ```
//!
//! The four pools (vertices, meshlet records, micro vertex indices, micro triangle
//! indices) are suballocated across **every** asset, so a meshlet id resolves
//! through a per-asset `remap` table both the cull compute and the raster read.
//! `remap[i] == NOT_RESIDENT` is how a non-resident page disappears from the cut.
//!
//! The clamp itself is one scalar: `floor_lod`, the finest resident LOD level.
//! The cut becomes `eff_error ≤ t < parent_error` with
//! `eff_error = (lod_level ≤ floor_lod) ? 0 : error`, which is
//! [`VgeomMesh::select_with_residency`](inf_vgeom::VgeomMesh::select_with_residency)
//! — and at full residency (`floor_lod == 0`) it is **identical** to the
//! pre-P18.2 cut, term for term.
//!
//! The stronger statement, and the one the goldens actually rest on: the cut is
//! identical at the streamer's *wanted* floor too, not merely at full residency.
//! [`ideal_page_count`](inf_vgeom::ideal_page_count) grants exactly the pages
//! whose `max_parent_error` still exceeds the instance threshold, and a page below
//! that bound holds only meshlets the cut rejects anyway — so a streamed frame
//! draws the same meshlets while holding a fraction of the asset. Only a **budget**
//! clamp shorter than that want actually coarsens anything, which is the intended
//! degradation.
//!
//! Determinism: the want is derived from the same `t` the GPU cut uses (never from
//! a GPU readback, which would be a frame latent and make residency depend on
//! frame history), loads are `read_ref` slices of an mmap staged through
//! `parallel_map_ref`, and the whole plan is a pure function of
//! `(wants, residency, budget)`. See `inf_vgeom::stream` for the full argument.

use std::collections::BTreeMap;

use glam::{Mat3, Mat4, Vec3, Vec4};
use inf_math::FloatingOrigin;
use inf_vgeom::VgeomMesh;

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::LightsUniform;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::VgeomInstance;
use crate::settings::VgeomSettings;

// ── GPU-side layouts (must match the WGSL structs) ───────────────────────────

/// The meshlet descriptor the cull/raster shaders read is
/// [`inf_vgeom::MeshletRec`] — the **on-disk record of the `.inf_vmesh` container
/// itself** (P18.2). Storing the GPU layout on disk is what makes paging a page in
/// a memcpy plus two `u32` rebases instead of a decode; this assertion is the wire
/// contract between `MeshletRec`, `struct Meshlet` in `vgeom_cull.wgsl` and
/// `vgeom_mesh.wgsl`.
const _: () = assert!(std::mem::size_of::<inf_vgeom::MeshletRec>() == 64);

/// One placed instance as the shaders read it (176 bytes). Mirrors
/// `struct Instance`. `threshold` is the precomputed per-instance LOD scalar.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VgeomInstanceGpu {
    model: [f32; 16],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
    color: [f32; 4],
    emissive: [f32; 4],
    threshold: f32,
    metallic: f32,
    roughness: f32,
    max_scale: f32,
    pick_id: u32,
    /// P26.3: the virtual-texture set (albedo, normal, ORM), each a
    /// `VtTextureHandle + 1`. These were `_p` — three words of alignment padding
    /// uploaded as zero since P13.1b — so a meshlet instance that samples
    /// nothing packs to the bytes it always did and the record stays 176 B.
    vt: [u32; 3],
}

/// The record's size and the offset of the P26.3 words, pinned the way
/// [`inf_vgeom::MeshletRec`] above is — and for a sharper reason.
///
/// `v_instances` is a **storage buffer** (`@group(3) @binding(4)`), not a vertex
/// buffer: `wgpu` validates nothing about a storage struct's field layout, so a
/// Rust record that stops matching `struct Instance` in `vgeom_mesh.wgsl` is
/// silent corruption rather than a pipeline error. P26.3 claimed "the record
/// stays 176 B, pinned by its existing layout arm"; `InstanceRaw` has one and
/// this had none. 64 (mat4) + 5 × 16 (vec4) + 4 × 4 (scalars) + 4 × 4 (pick_id +
/// the three VT words) = 176, and WGSL's std430 lays `struct Instance` out the
/// same way.
const _: () = assert!(std::mem::size_of::<VgeomInstanceGpu>() == 176);
const _: () = assert!(std::mem::offset_of!(VgeomInstanceGpu, pick_id) == 160);
const _: () = assert!(std::mem::offset_of!(VgeomInstanceGpu, vt) == 164);

/// Cull uniform block. Mirrors `struct CullParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParamsGpu {
    view_proj: [f32; 16],
    frustum: [[f32; 4]; 6],
    eye: [f32; 4],
    /// x = total threads, y = meshlet count, z = cull flags, w = [`CullMode`].
    counts: [u32; 4],
    hzb: [f32; 4],
    /// x = conservative (early set == whole base cut), y = audit counters on.
    misc: [u32; 4],
}

/// Which of the three cull-compute modes a dispatch runs (`params.counts.w`;
/// mirrors the `MODE_*` constants in `vgeom_cull.wgsl`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
enum CullMode {
    /// v1 / fallback: base cut + optional HZB test + append, one dispatch.
    Single = 0,
    /// Append the base-cut pairs that were visible last frame (no HZB test).
    Early = 1,
    /// Publish this frame's visibility + append the newly-disoccluded remainder.
    Late = 2,
}

/// Per-meshlet debug flag uniform (`struct VgeomFlags`, `@group(3) @binding(6)`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlagsGpu {
    flags: [u32; 4],
}

const FLAG_FRUSTUM: u32 = 1;
const FLAG_CONE: u32 = 2;
const FLAG_OCCLUSION: u32 = 4;

// ── Pure math (unit-tested without a GPU) ────────────────────────────────────

/// The per-instance object-space LOD threshold `t` for a bounding sphere at
/// render-local `center` with surface distance to `eye`, instance scale `max_scale`
/// (largest axis), under `view`, targeting `pixel_error` px. See the module docs.
pub fn lod_threshold(
    eye: Vec3,
    center: Vec3,
    radius_world: f32,
    max_scale: f32,
    view: &RenderView,
    pixel_error: f32,
) -> f32 {
    let s = max_scale.max(1e-6);
    match view.ortho {
        None => {
            let d = ((eye - center).length() - radius_world).max(1e-3);
            let focal = view.height.max(1) as f32 / (2.0 * (view.fov_y * 0.5).tan()).max(1e-6);
            pixel_error * d / (focal * s)
        }
        Some(o) => {
            // Parallel projection: world units per pixel is distance-independent.
            let wpp = 2.0 * o.half_height / view.height.max(1) as f32;
            pixel_error * wpp / s
        }
    }
}

/// Six normalized frustum planes (render-local clip space, wgpu NDC z ∈ [0,1])
/// extracted from a render-local `view_proj` (Gribb–Hartmann). Order: left,
/// right, bottom, top, near, far. A plane with a (near-)degenerate normal (the far
/// plane of a reverse-infinite projection) is returned as all-zeros → the cull
/// treats it as "always inside".
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let m = view_proj.to_cols_array(); // column-major: element (row r, col c) = m[c*4+r]
    let row = |r: usize| Vec4::new(m[r], m[4 + r], m[8 + r], m[12 + r]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let raw = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near  (wgpu/D3D: z ≥ 0)
        r3 - r2, // far
    ];
    raw.map(|p| {
        let n = Vec3::new(p.x, p.y, p.z).length();
        if n > 1e-6 {
            p / n
        } else {
            Vec4::ZERO
        }
    })
}

/// Whether a world sphere is entirely outside any frustum plane.
fn outside_frustum(center: Vec3, radius: f32, planes: &[Vec4; 6]) -> bool {
    planes.iter().any(|pl| {
        let n = Vec3::new(pl.x, pl.y, pl.z);
        n != Vec3::ZERO && n.dot(center) + pl.w < -radius
    })
}

/// Bit flags matching the cull compute (frustum | cone | occlusion).
pub fn cull_flags(settings: &VgeomSettings) -> u32 {
    let mut f = 0;
    if settings.frustum_cull {
        f |= FLAG_FRUSTUM;
    }
    if settings.cone_cull {
        f |= FLAG_CONE;
    }
    if settings.occlusion {
        f |= FLAG_OCCLUSION;
    }
    f
}

/// The **CPU reference** visible meshlet set for a single instance: the identical
/// residency-clamped LOD cut + frustum + cone filters the GPU cull compute
/// applies, mirrored on the CPU for the parity gate. Returns sorted meshlet
/// indices. Occlusion is never part of the reference (the parity scene is
/// occlusion-free).
///
/// `floor_lod` is the streamer's finest resident LOD level; `0` is full residency
/// and reduces the cut to exactly `VgeomMesh::select(threshold)` — so the parity
/// gate covers both the fully-paged and the punched-out cases with one function.
#[allow(clippy::too_many_arguments)]
pub fn cpu_visible_set(
    mesh: &VgeomMesh,
    model: Mat4,
    normal_mat: Mat3,
    eye: Vec3,
    threshold: f32,
    max_scale: f32,
    planes: &[Vec4; 6],
    flags: u32,
    floor_lod: u8,
) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, m) in mesh.meshlets.iter().enumerate() {
        // 1. LOD cut (== VgeomMesh::select_with_residency semantics).
        if !m.selected_at_clamped(threshold, floor_lod) {
            continue;
        }
        let center = model.transform_point3(Vec3::from(m.center));
        let radius = m.radius * max_scale;
        // 2. Frustum.
        if (flags & FLAG_FRUSTUM) != 0 && outside_frustum(center, radius, planes) {
            continue;
        }
        // 3. Backface cone.
        if (flags & FLAG_CONE) != 0 && m.cone_cutoff < 1.0 {
            let axis = (normal_mat * Vec3::from(m.cone_axis)).normalize_or_zero();
            let vd = (center - eye).normalize_or_zero();
            if vd.dot(axis) >= m.cone_cutoff {
                continue;
            }
        }
        out.push(i as u32);
    }
    out
}

// ── Per-instance packing (shared by the node + the standalone readback) ──────

fn max_scale_of(scale: Vec3) -> f32 {
    scale.abs().max_element().max(1e-6)
}

/// Pack one instance's GPU record (transform, normal matrix, material, and the
/// precomputed LOD threshold from the live view).
fn pack_instance(
    origin: &FloatingOrigin,
    view: &RenderView,
    bounds: ([f32; 3], f32),
    inst: &VgeomInstance,
    pixel_error: f32,
) -> VgeomInstanceGpu {
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = max_scale_of(inst.scale);
    let inv_scale = inst.scale.max(Vec3::splat(1e-6)).recip();
    let nrm = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let c = nrm.to_cols_array_2d();

    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(bounds.0));
    let radius_world = bounds.1 * max_scale;
    let threshold = lod_threshold(
        eye,
        center_world,
        radius_world,
        max_scale,
        view,
        pixel_error,
    );

    VgeomInstanceGpu {
        model: model.to_cols_array(),
        n0: [c[0][0], c[0][1], c[0][2], 0.0],
        n1: [c[1][0], c[1][1], c[1][2], 0.0],
        n2: [c[2][0], c[2][1], c[2][2], 0.0],
        color: inst.color,
        emissive: [inst.emissive[0], inst.emissive[1], inst.emissive[2], 0.0],
        threshold,
        metallic: inst.metallic,
        roughness: inst.roughness,
        max_scale,
        pick_id: inst.id,
        vt: inst.vt.slots(),
    }
}

#[allow(clippy::too_many_arguments)]
fn cull_params(
    view: &RenderView,
    meshlet_count: u32,
    instance_count: u32,
    settings: &VgeomSettings,
    hzb: [f32; 4],
    mode: CullMode,
    conservative: bool,
    audit: bool,
    floor_lod: u32,
) -> CullParamsGpu {
    let vp = view.view_proj();
    let planes = frustum_planes(vp);
    CullParamsGpu {
        view_proj: vp.to_cols_array(),
        frustum: planes.map(|p| p.to_array()),
        eye: view.eye_local().extend(0.0).to_array(),
        counts: [
            instance_count * meshlet_count,
            meshlet_count,
            cull_flags(settings),
            mode as u32,
        ],
        hzb,
        misc: [conservative as u32, audit as u32, floor_lod, 0],
    }
}

/// Whether the view moved discontinuously enough that last frame's visible set is
/// worthless as an early set — a teleport, a level-load camera placement, a
/// cut between cameras, an fov/aspect change. Pure (a function of the two views),
/// so it is unit-tested without a GPU.
///
/// **Not load-bearing for correctness**: the occlusion test is subtractive under
/// *any* early set (see the module docs), so a miss here costs a frame of extra
/// late-pass work, never a hole. The thresholds are therefore chosen for cost,
/// not safety: 50 m in one frame is ~3 km/s, and a 60° snap is not a pan.
pub fn is_camera_cut(prev: &RenderView, cur: &RenderView) -> bool {
    const CUT_METRES: f64 = 50.0;
    const CUT_COS: f32 = 0.5; // 60°
    let ortho = |v: &RenderView| v.ortho.map(|o| (o.half_height, o.near, o.far));
    if prev.width != cur.width
        || prev.height != cur.height
        || prev.fov_y != cur.fov_y
        || ortho(prev) != ortho(cur)
    {
        return true;
    }
    if (prev.eye_world - cur.eye_world).length() > CUT_METRES {
        return true;
    }
    prev.forward.dot(cur.forward) < CUT_COS
}

// ── The shared, suballocated meshlet pools (P18.2) ───────────────────────────

/// A storage buffer holding `data` (padded to ≥16 bytes so a small/empty payload
/// still creates a valid buffer).
fn storage_buffer(gpu: &GpuContext, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)).next_multiple_of(4) as u64;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        gpu.queue.write_buffer(&buf, 0, data);
    }
    buf
}

/// The four GPU buffers the [`inf_vgeom::VgeomStreamer`]'s pools live in.
///
/// One set for the **whole scene**, not one per asset: a page's four sections are
/// suballocated out of these, and a meshlet id is resolved through the asset's
/// remap table. Sizes track the allocators' capacities, which only ever grow (and
/// only ever by appending, so a live block never moves).
///
/// A growth reallocates the buffer, and the streamer answers by re-staging every
/// resident page in the same plan — deliberately, rather than copying the old
/// buffer into the new one inside this frame's encoder: `queue.write_buffer` is
/// ordered *before* an encoder's commands in a submit, so such a copy would
/// clobber the very uploads it was meant to preserve.
struct VgeomPoolBuffers {
    vertices: wgpu::Buffer,
    meshlets: wgpu::Buffer,
    mlverts: wgpu::Buffer,
    mltris: wgpu::Buffer,
    /// Byte sizes, mirroring the allocators' capacities.
    sizes: [u64; 4],
}

impl VgeomPoolBuffers {
    fn new(gpu: &GpuContext) -> Self {
        let mk = |label| storage_buffer(gpu, label, &[]);
        Self {
            vertices: mk("vgeom-pool-vertices"),
            meshlets: mk("vgeom-pool-meshlets"),
            mlverts: mk("vgeom-pool-mlverts"),
            mltris: mk("vgeom-pool-mltris"),
            sizes: [0; 4],
        }
    }

    /// Resize any buffer whose allocator capacity outgrew it. Returns whether
    /// anything was recreated (⇒ every bind group holding one is stale).
    fn ensure(&mut self, gpu: &GpuContext, pools: &inf_vgeom::VgeomPools) -> bool {
        let want = [
            pools.vertices.capacity_bytes(),
            pools.meshlets.capacity_bytes(),
            pools.mlverts.capacity_bytes(),
            pools.mltris.capacity_bytes(),
        ];
        if want == self.sizes {
            return false;
        }
        let mk = |label, bytes: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16).next_multiple_of(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        self.vertices = mk("vgeom-pool-vertices", want[0]);
        self.meshlets = mk("vgeom-pool-meshlets", want[1]);
        self.mlverts = mk("vgeom-pool-mlverts", want[2]);
        self.mltris = mk("vgeom-pool-mltris", want[3]);
        self.sizes = want;
        true
    }

    /// Write one staged page into the pools.
    fn write_page(&self, gpu: &GpuContext, up: &inf_vgeom::PageUpload) {
        let b = &up.blocks;
        let put = |buf: &wgpu::Buffer, byte_off: u64, data: &[u8]| {
            if !data.is_empty() {
                gpu.queue.write_buffer(buf, byte_off, data);
            }
        };
        put(
            &self.meshlets,
            b.meshlets.offset * inf_vgeom::asset::MESHLET_REC_LEN as u64,
            &up.meshlets,
        );
        put(&self.mlverts, b.mlverts.offset * 4, &up.mlverts);
        put(&self.mltris, b.mltris.offset * 4, &up.mltris);
        put(
            &self.vertices,
            b.vertices.offset * inf_vgeom::asset::VERTEX_REC_LEN as u64,
            &up.vertices,
        );
    }
}

// ── The cull compute pipeline (shared by node + standalone readback) ─────────

struct CullPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl CullPipeline {
    fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-cull"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("vgeom_cull").into()),
            });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |ro| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-cull"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(1, storage(true)),
                    entry(2, storage(true)),
                    entry(3, storage(false)),
                    entry(4, storage(false)),
                    entry(
                        5,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    ),
                    // P18.1 two-pass: last/this frame's per-pair visibility +
                    // the occlusion-audit counters. Seven storage buffers in one
                    // compute stage — inside the portable 8 the High tier already
                    // demands (`caps::VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE`).
                    entry(6, storage(true)),
                    entry(7, storage(false)),
                    entry(8, storage(false)),
                    // P18.2 streaming: the per-asset meshlet -> pool-slot remap.
                    // EIGHT storage buffers in one compute stage — exactly the
                    // portable floor the High tier already demands
                    // (`caps::VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE`), with no
                    // headroom left; a ninth needs a capability bump or a merge.
                    entry(9, storage(true)),
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vgeom-cull"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vgeom-cull"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cs_cull"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline, bgl }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_group(
        &self,
        gpu: &GpuContext,
        params: &wgpu::Buffer,
        pools: &VgeomPoolBuffers,
        remap: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        visible: &wgpu::Buffer,
        draw_args: &wgpu::Buffer,
        hzb: &wgpu::TextureView,
        prev_visible: &wgpu::Buffer,
        cur_visible: &wgpu::Buffer,
        stats: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vgeom-cull"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pools.meshlets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: draw_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(hzb),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: prev_visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cur_visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: stats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: remap.as_entire_binding(),
                },
            ],
        })
    }
}

/// A 1×1 dummy HZB texture (bound when occlusion is off so the cull bind group is
/// always complete).
pub(crate) fn dummy_hzb(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("vgeom-hzb-dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Standalone cull readback (tests + player activation check) ───────────────

/// What [`cull_visible_streamed`] read back: the GPU's visible set plus the
/// residency it was culled under.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CullReadback {
    /// Visible `(instance, asset-local meshlet)` pairs, **sorted** (the atomic
    /// append order is nondeterministic; the set is not).
    pub pairs: Vec<[u32; 2]>,
    /// The finest resident LOD level the streamer settled on — the parameter
    /// [`cpu_visible_set`] must be given to reproduce `pairs`.
    pub floor_lod: u8,
    /// Resident pages out of the asset's total.
    pub resident_pages: usize,
    pub total_pages: usize,
}

impl CullReadback {
    /// Nothing to cull (an empty mesh, no instances, or an unreadable payload).
    const EMPTY: Self = Self {
        pairs: Vec::new(),
        floor_lod: 0,
        resident_pages: 0,
        total_pages: 0,
    };
}

/// Run **only** the cull compute for `instances` of `mesh` under `view`+`settings`
/// and read back the visible `(instance, meshlet)` pairs, **sorted**. Always runs
/// [`CullMode::Single`] with occlusion forced off (there is no depth context) — so
/// what it reads back is exactly the residency-clamped LOD+frustum+cone **base
/// cut** the CPU reference mirrors, unchanged by P18.1. This is the exact
/// machinery the render node uses, exposed for the CPU-vs-GPU parity gate and the
/// player's activation test.
///
/// The default budget is unlimited enough to hold any test fixture, so this is
/// the *fully resident* cut; [`cull_visible_streamed`] is the same call with the
/// budget exposed, which is how the parity gate reaches punched-out residency.
pub fn cull_visible(
    gpu: &GpuContext,
    mesh: &VgeomMesh,
    instances: &[VgeomInstance],
    view: &RenderView,
    settings: &VgeomSettings,
) -> Vec<[u32; 2]> {
    cull_visible_streamed(gpu, mesh, instances, view, settings).pairs
}

/// [`cull_visible`], reporting the residency it culled under.
///
/// The streaming budget comes from `settings.stream`, so a caller that shrinks it
/// gets a genuinely partially-resident cut through the identical code path the
/// render node uses — which is what makes the CPU/GPU parity gate meaningful under
/// streaming rather than only at full residency.
pub fn cull_visible_streamed(
    gpu: &GpuContext,
    mesh: &VgeomMesh,
    instances: &[VgeomInstance],
    view: &RenderView,
    settings: &VgeomSettings,
) -> CullReadback {
    match inf_vgeom::VgeomSource::from_mesh(mesh) {
        Ok(source) => cull_visible_source(gpu, &source, instances, view, settings),
        Err(_) => CullReadback::EMPTY,
    }
}

/// [`cull_visible_streamed`] over an already-indexed `.inf_vmesh` — the shape the
/// runtime actually holds (the player's registry hands out
/// [`VgeomSource`](inf_vgeom::VgeomSource)s, never decoded DAGs), so an activation
/// check costs a page-in of what the camera wants rather than a full decode.
pub fn cull_visible_source(
    gpu: &GpuContext,
    source: &inf_vgeom::VgeomSource,
    instances: &[VgeomInstance],
    view: &RenderView,
    settings: &VgeomSettings,
) -> CullReadback {
    if source.meshlet_count() == 0 || instances.is_empty() {
        return CullReadback::EMPTY;
    }
    let mut settings = *settings;
    settings.occlusion = false;

    let meshlet_count = source.meshlet_count();
    let max_tri = source.max_tri();
    let bounds = source.bounds();
    let origin = view.origin;
    let packed: Vec<VgeomInstanceGpu> = instances
        .iter()
        .map(|i| pack_instance(&origin, view, bounds, i, settings.pixel_error))
        .collect();

    // Page the mesh in through the real streamer, then read what it settled on.
    let mut streamer = inf_vgeom::VgeomStreamer::new(settings.stream);
    let threshold = packed
        .iter()
        .map(|p| p.threshold)
        .fold(f32::INFINITY, f32::min);
    let mut pools = VgeomPoolBuffers::new(gpu);
    let plan = streamer.plan(&[inf_vgeom::VgeomWant {
        asset: 0,
        source,
        threshold,
    }]);
    pools.ensure(gpu, streamer.pools());
    for up in &plan.uploads {
        pools.write_page(gpu, up);
    }
    let Some(res) = streamer.residency(0) else {
        return CullReadback::EMPTY;
    };
    let floor_lod = res.floor_lod().min(u8::MAX as u32) as u8;
    let resident_pages = res.resident_pages();
    let remap = storage_buffer(gpu, "vgeom-remap", bytemuck::cast_slice(res.remap()));

    let inst_buf = storage_buffer(gpu, "vgeom-inst", bytemuck::cast_slice(&packed));

    let total = packed.len() as u32 * meshlet_count;
    let visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-visible"),
        size: (total as u64 * 8).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let draw_args = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-args"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(
        &draw_args,
        0,
        bytemuck::cast_slice(&[max_tri * 3, 0u32, 0u32, 0u32]),
    );

    let params = cull_params(
        view,
        meshlet_count,
        packed.len() as u32,
        &settings,
        [1.0, 1.0, 1.0, 0.0],
        CullMode::Single,
        false,
        false,
        floor_lod as u32,
    );
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-cull-params"),
        size: std::mem::size_of::<CullParamsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let hzb = dummy_hzb(gpu);
    // Single-pass mode reads neither visibility buffer and writes no stats, but
    // the bind group must still be complete.
    let vis_flags = storage_buffer(
        gpu,
        "vgeom-vis-flags",
        &vec![0u8; (total as usize * 4).max(16)],
    );
    let vis_flags_cur = storage_buffer(
        gpu,
        "vgeom-vis-flags",
        &vec![0u8; (total as usize * 4).max(16)],
    );
    let stats = storage_buffer(gpu, "vgeom-stats", &[0u8; AUDIT_BYTES as usize]);
    let cull = CullPipeline::new(gpu);
    let bg = cull.bind_group(
        gpu,
        &params_buf,
        &pools,
        &remap,
        &inst_buf,
        &visible,
        &draw_args,
        &hzb,
        &vis_flags,
        &vis_flags_cur,
        &stats,
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vgeom-cull-readback"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vgeom-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cull.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(total.div_ceil(64).max(1), 1, 1);
    }
    // Copy args + visible to a mappable buffer.
    let args_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-args-rb"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let vis_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-visible-rb"),
        size: (total as u64 * 8).max(16),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&draw_args, 0, &args_rb, 0, 16);
    encoder.copy_buffer_to_buffer(&visible, 0, &vis_rb, 0, (total as u64 * 8).max(16));
    gpu.queue.submit([encoder.finish()]);

    let count = map_u32(gpu, &args_rb)[1] as usize;
    let vis = map_u32(gpu, &vis_rb);
    let mut pairs: Vec<[u32; 2]> = (0..count.min(total as usize))
        .map(|i| [vis[i * 2], vis[i * 2 + 1]])
        .collect();
    pairs.sort_unstable();
    CullReadback {
        pairs,
        floor_lod,
        resident_pages,
        total_pages: source.pages().len(),
    }
}

/// Blocking map of a `MAP_READ` buffer into a `Vec<u32>`.
pub(crate) fn map_u32(gpu: &GpuContext, buf: &wgpu::Buffer) -> Vec<u32> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map vgeom readback buffer");
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    buf.unmap();
    out
}

// ── Per-asset dynamic draw state (owned by the node) ─────────────────────────

/// What the persisted per-pair visibility (`prev_visible`) is only meaningful
/// against. Any change invalidates it — slot `i` would name a different
/// (instance, meshlet) pair, or the frame targets the HZB samples were
/// reallocated — and the next frame runs conservative. See the module docs:
/// this is a cost heuristic, not a correctness dependency.
#[derive(Clone, Copy, PartialEq, Eq)]
struct TemporalKey {
    scene_version: u64,
    instance_count: u32,
    meshlet_count: u32,
    targets_generation: u64,
    /// P18.2: the streamer's residency generation. A page arriving or leaving
    /// changes which meshlets the base cut selects, so an early set inherited
    /// across it names a different drawn set — the same class of staleness a
    /// scene-version bump is, and handled the same conservative way.
    residency_generation: u64,
}

struct AssetDraw {
    instances: wgpu::Buffer,
    instance_cap: u32,
    /// Pairs appended by the early cull (and, in single-pass mode, the only list).
    visible: wgpu::Buffer,
    /// Pairs appended by the late cull — the newly-disoccluded remainder. A second
    /// buffer rather than an offset into `visible` because a non-zero
    /// `first_instance` in an indirect draw needs `INDIRECT_FIRST_INSTANCE`, which
    /// is not portable.
    visible_late: wgpu::Buffer,
    visible_cap: u32,
    /// Per-pair "visible last frame" flags, ping-ponged (`prev` is read by both
    /// dispatches, `cur` is written by the late one; they swap at end of frame).
    vis_prev: wgpu::Buffer,
    vis_cur: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    draw_args_late: wgpu::Buffer,
    /// Cull uniforms — one per dispatch, because both are recorded into the same
    /// encoder before submit (a single buffer written twice would apply both
    /// writes before either dispatch ran).
    params: wgpu::Buffer,
    params_late: wgpu::Buffer,
    debug_flags: wgpu::Buffer,
    /// Asset-local meshlet id → slot in the shared meshlet pool, or
    /// `inf_vgeom::NOT_RESIDENT` (P18.2). Read by BOTH the cull compute and the
    /// raster; re-uploaded only when the streamer's residency generation moves.
    remap: wgpu::Buffer,
    remap_cap: u32,
    remap_generation: u64,
    /// `Some` iff `vis_prev` holds a usable early set for this key.
    state: Option<TemporalKey>,
}

impl AssetDraw {
    fn new(gpu: &GpuContext) -> Self {
        let mk = |label, size, usage| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let args = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let params_size = std::mem::size_of::<CullParamsGpu>() as u64;
        Self {
            instances: mk(
                "vgeom-instances",
                256,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            instance_cap: 0,
            visible: mk("vgeom-visible", 256, wgpu::BufferUsages::STORAGE),
            visible_late: mk("vgeom-visible-late", 256, wgpu::BufferUsages::STORAGE),
            visible_cap: 0,
            vis_prev: mk("vgeom-vis-prev", 256, wgpu::BufferUsages::STORAGE),
            vis_cur: mk("vgeom-vis-cur", 256, wgpu::BufferUsages::STORAGE),
            draw_args: mk("vgeom-args", 16, args),
            draw_args_late: mk("vgeom-args-late", 16, args),
            params: mk("vgeom-cull-params", params_size, uniform),
            params_late: mk("vgeom-cull-params-late", params_size, uniform),
            debug_flags: mk(
                "vgeom-flags",
                std::mem::size_of::<FlagsGpu>() as u64,
                uniform,
            ),
            remap: mk(
                "vgeom-remap",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            remap_cap: 0,
            // 0 is never a live residency stamp (they come from a process-global
            // counter that starts at 1), so the first frame always uploads.
            remap_generation: 0,
            state: None,
        }
    }

    /// Push `residency`'s remap table if it moved since the last upload. Returns
    /// whether the buffer was recreated (⇒ bind groups holding it are stale).
    fn sync_remap(&mut self, gpu: &GpuContext, res: &inf_vgeom::AssetResidency) -> bool {
        let table = res.remap();
        let mut rebuilt = false;
        if table.len() as u32 > self.remap_cap {
            let cap = (table.len() as u32).next_power_of_two().max(4);
            self.remap = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-remap"),
                size: cap as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.remap_cap = cap;
            self.remap_generation = 0;
            rebuilt = true;
        }
        if self.remap_generation != res.generation() && !table.is_empty() {
            gpu.queue
                .write_buffer(&self.remap, 0, bytemuck::cast_slice(table));
            self.remap_generation = res.generation();
        }
        rebuilt
    }

    /// Ensure the instance + visible + visibility buffers hold `instance_count`
    /// instances of a `meshlet_count`-meshlet asset. Returns whether any buffer was
    /// recreated (⇒ bind groups must be rebuilt **and** the temporal state is gone).
    fn ensure(&mut self, gpu: &GpuContext, instance_count: u32, meshlet_count: u32) -> bool {
        let mut rebuilt = false;
        if instance_count > self.instance_cap {
            let cap = instance_count.next_power_of_two().max(4);
            self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-instances"),
                size: cap as u64 * std::mem::size_of::<VgeomInstanceGpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_cap = cap;
            rebuilt = true;
        }
        let total = instance_count * meshlet_count;
        if total > self.visible_cap {
            let cap = total.next_power_of_two().max(4);
            let mk = |label, size| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            self.visible = mk("vgeom-visible", cap as u64 * 8);
            self.visible_late = mk("vgeom-visible-late", cap as u64 * 8);
            self.vis_prev = mk("vgeom-vis-prev", cap as u64 * 4);
            self.vis_cur = mk("vgeom-vis-cur", cap as u64 * 4);
            self.visible_cap = cap;
            rebuilt = true;
        }
        rebuilt
    }
}

// ── Occlusion audit (P18.1 tests + tools) ────────────────────────────────────

/// The four counters the occlusion-audit readback exposes, aggregated over every
/// vgeom asset in the frame. Meaningful only for a frame [`VgeomNode`] actually
/// ran (vgeom enabled + the scene carries meshlet content).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VgeomAudit {
    /// (instance, meshlet) pairs that passed the LOD+frustum+cone **base cut**.
    /// Counted by whichever dispatch sees every pair — the late one in two-pass
    /// mode, the only one in single-pass mode — so it means the same in both.
    pub base_cut: u32,
    /// Base-cut pairs the HZB proved invisible this frame — the *subtracted* set.
    pub occluded: u32,
    /// Pairs the early pass drew (last-frame-visible, or all when conservative).
    /// In single-pass mode this is that one pass's entire drawn set.
    pub early_drawn: u32,
    /// Pairs the late pass drew (newly disoccluded). `0` on a conservative frame,
    /// and always `0` in single-pass mode (there is no late pass).
    pub late_drawn: u32,
    /// **P18.2 streaming**: base-cut pairs drawn *coarser* than the threshold
    /// asked for, because the finer page is not resident — the
    /// "requested-but-missing" signal.
    ///
    /// Audit **only**. Nothing reads it back to decide what to load: a GPU
    /// readback is one frame latent, so letting it steer residency would make the
    /// resident set a function of frame history and break the render-trace gates.
    /// The streamer derives the identical want CPU-side from the same threshold
    /// (see `inf_vgeom::stream`). `0` on a fully-resident frame.
    pub clamped: u32,
}

/// Bytes of the audit counter buffer. Five live counters, rounded to eight slots
/// so a sixth costs no layout change.
const AUDIT_BYTES: u64 = 32;

/// GPU-side storage for [`VgeomAudit`]: a storage quartet the cull compute
/// atomically increments plus its mappable mirror. Owned by
/// [`crate::EngineRenderer`] (like the shadow/GI/atmosphere resources) and reached
/// through [`crate::renderer::FrameData`], so the node stays stateless about it.
///
/// **Off by default and free when off** — the shader only touches the counters
/// when `params.misc.y != 0`, and the readback copy is only recorded when
/// enabled. The buffer is always bound, because a bind group must be complete.
pub struct VgeomAuditResources {
    pub(crate) enabled: bool,
    pub(crate) stats: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl VgeomAuditResources {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        Self {
            enabled: false,
            stats: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-audit-stats"),
                size: AUDIT_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-audit-readback"),
                size: AUDIT_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    /// Map the counters recorded by the **last submitted** frame. Blocking — a
    /// test/tools path, never the hot path.
    pub(crate) fn read(&self, gpu: &GpuContext) -> VgeomAudit {
        let v = map_u32(gpu, &self.readback);
        VgeomAudit {
            base_cut: v[0],
            occluded: v[1],
            early_drawn: v[2],
            late_drawn: v[3],
            clamped: v[4],
        }
    }
}

/// What the streamer did this frame, published for hosts and gates.
///
/// The counters live behind a `Mutex` the [`EngineRenderer`](crate::EngineRenderer)
/// and the node share, because the node lives inside the render graph and a graph
/// node is otherwise unreachable from outside. Written once per frame at the end
/// of the node's run; read by `EngineRenderer::vgeom_stream_report`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VgeomStreamReport {
    /// The streamer's counters (residency, backlog, budget clamping).
    pub stats: inf_vgeom::VgeomStreamStats,
    /// Per asset, the finest resident LOD level — the parameter
    /// [`VgeomMesh::select_with_residency`](inf_vgeom::VgeomMesh::select_with_residency)
    /// needs to reproduce what the GPU drew.
    pub floor_lod: BTreeMap<u128, u32>,
    /// Per asset, `(resident pages, total pages)`.
    pub pages: BTreeMap<u128, (usize, usize)>,
}

/// The shared handle the renderer hands the node.
pub type SharedStreamReport = std::sync::Arc<std::sync::Mutex<VgeomStreamReport>>;

// ── The render node ──────────────────────────────────────────────────────────

/// The virtualized-geometry render node (P13.1b, two-pass P18.1): cull compute +
/// vertex-pulled indirect raster. A no-op unless [`VgeomSettings::enabled`] and
/// the scene carries vmesh assets + instances — so the classic path is byte-stable.
pub struct VgeomNode {
    cull: CullPipeline,
    raster: wgpu::RenderPipeline,
    raster_bgl: wgpu::BindGroupLayout,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    /// AO + shadows + GI + atmosphere env bind at `@group(2)` (P17.2; was the
    /// AO-only bind, so aerial perspective now reaches meshlet geometry too).
    env: super::EnvBinding,
    dummy_hzb: wgpu::TextureView,
    /// The four shared, suballocated GPU pools every asset's pages live in.
    pools: VgeomPoolBuffers,
    /// The residency state machine that fills them (P18.2).
    streamer: inf_vgeom::VgeomStreamer,
    draws: BTreeMap<u128, AssetDraw>,
    hzb: HzbChain,
    /// Last frame's view, for the [`is_camera_cut`] conservative trigger.
    prev_view: Option<RenderView>,
    /// Published streaming state (shared with the renderer).
    report: SharedStreamReport,
    /// The view layout, kept so the P28.1 visibility path can build its own
    /// pipelines the first time it is asked for one.
    view_bgl: wgpu::BindGroupLayout,
    /// The lights layout, kept for the same reason.
    lights_bgl: wgpu::BindGroupLayout,
    /// P28.1's visibility-buffer resources, built **lazily**: three pipelines and
    /// two viewport-sized targets are not free, and the mode is off by default on
    /// every tier. A renderer that never turns it on pays exactly what it paid
    /// before this batch — which is also what keeps every existing headless test's
    /// construction cost, and its command stream, unchanged.
    vis: Option<super::visbuffer::VisState>,
    /// The per-asset instance bases the last visibility frame assigned, published
    /// so a test can name the flat index a pixel's id decodes to without
    /// re-deriving the asset order.
    vis_bases: Vec<(u128, u32)>,
}

impl VgeomNode {
    pub fn new(
        gpu: &GpuContext,
        view_bgl: &wgpu::BindGroupLayout,
        report: SharedStreamReport,
    ) -> Self {
        let cull = CullPipeline::new(gpu);

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-mesh"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("vgeom_mesh").into()),
            });

        // Lights (@group(1)) — shared model with the rigid mesh pass.
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-lights"),
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
        let lights_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vgeom-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vgeom-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        let env = super::EnvBinding::new(gpu);
        // Kept (P28.1) so the resolve pipeline's layout can name the SAME lights
        // layout `lights_bg` was built against, rather than a structurally equal
        // twin — "compatible" is a wgpu-internal judgement and a pipeline layout
        // is not the place to rely on one.
        let lights_bgl_kept = lights_bgl.clone();

        // Group 3: the vgeom storage buffers (vertex-visible) + flags uniform.
        let vs = wgpu::ShaderStages::VERTEX;
        let ro_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let raster_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-raster"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // P18.2: the meshlet -> pool-slot remap, read by the vertex
                    // stage exactly as the cull compute reads it. Seven storage
                    // buffers in the vertex stage, inside the portable eight.
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vgeom-raster"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&lights_bgl),
                    Some(&env.bgl),
                    Some(&raster_bgl),
                ],
                immediate_size: 0,
            });
        let raster = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vgeom-raster"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[], // pure vertex pulling
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
                    cull_mode: Some(wgpu::Face::Back),
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

        Self {
            cull,
            raster,
            raster_bgl,
            lights_buf,
            lights_bg,
            env,
            dummy_hzb: dummy_hzb(gpu),
            pools: VgeomPoolBuffers::new(gpu),
            streamer: inf_vgeom::VgeomStreamer::new(inf_vgeom::VgeomStreamBudget::default()),
            draws: BTreeMap::new(),
            hzb: HzbChain::new(gpu),
            prev_view: None,
            report,
            view_bgl: view_bgl.clone(),
            lights_bgl: lights_bgl_kept,
            vis: None,
            vis_bases: Vec::new(),
        }
    }

    /// The P28.1 visibility path's audit counters — zero until a frame turns the
    /// mode on, and the *reason* a frame did not take it once one has.
    pub fn vis_audit(&self) -> super::visbuffer::VisAudit {
        self.vis.as_ref().map(|v| v.audit).unwrap_or_default()
    }

    /// The per-asset bases the last visibility frame assigned into the flat
    /// instance table, in the deterministic asset order the raster walked.
    pub fn vis_instance_bases(&self) -> &[(u128, u32)] {
        &self.vis_bases
    }

    // The streamer's state is published through `report` each frame rather than
    // exposed here: a node lives inside the render graph and is not reachable from
    // outside it, so an accessor on `VgeomNode` would be dead API. See
    // `EngineRenderer::vgeom_stream_report`.
}

impl RenderNode for VgeomNode {
    fn name(&self) -> &'static str {
        "vgeom"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let settings = &frame.settings.vgeom;
        if !settings.enabled
            || frame.scene.vgeom_instances.is_empty()
            || frame.scene.vgeom_assets.is_empty()
        {
            return;
        }

        // Group instances by asset (deterministic asset order).
        let mut by_asset: BTreeMap<u128, Vec<&VgeomInstance>> = BTreeMap::new();
        for inst in &frame.scene.vgeom_instances {
            by_asset.entry(inst.asset).or_default().push(inst);
        }
        let source_of: BTreeMap<u128, &inf_vgeom::VgeomSource> = frame
            .scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.source.as_ref()))
            .collect();

        // ── The streaming sync point (P18.2) ────────────────────────────────
        //
        // ONE call, before any culling, whose result is a pure function of
        // (wants, residency, budget). The want per asset is the SMALLEST
        // per-instance threshold — its closest/largest instance decides how much
        // detail the asset needs — and it is the *same* scalar the cut compares
        // meshlet errors against, so the two can never disagree about what
        // "finer" means. Assets in the scene list with no instances are not
        // wanted, so they are evicted rather than held.
        self.streamer.set_budget(settings.stream);
        let origin = frame.view.origin;
        let wants: Vec<inf_vgeom::VgeomWant<'_>> = by_asset
            .iter()
            .filter_map(|(asset, insts)| {
                let source = *source_of.get(asset)?;
                let bounds = source.bounds();
                let threshold = insts
                    .iter()
                    .map(|i| {
                        pack_instance(&origin, frame.view, bounds, i, settings.pixel_error)
                            .threshold
                    })
                    .fold(f32::INFINITY, f32::min);
                Some(inf_vgeom::VgeomWant {
                    asset: *asset,
                    source,
                    threshold,
                })
            })
            .collect();
        let plan = self.streamer.plan(&wants);
        // A pool that grew was reallocated, so every bind group holding one is
        // stale; the plan re-stages every resident page to refill it.
        let pools_rebuilt = self.pools.ensure(gpu, self.streamer.pools());
        for up in &plan.uploads {
            self.pools.write_page(gpu, up);
        }
        for asset in &plan.dropped {
            self.draws.remove(asset);
        }
        if let Ok(mut r) = self.report.lock() {
            r.stats = *self.streamer.stats();
            r.floor_lod.clear();
            r.pages.clear();
            for (id, res) in self.streamer.assets() {
                r.floor_lod.insert(id, res.floor_lod());
                r.pages.insert(id, (res.resident_pages(), res.page_count()));
            }
        }

        // ── P28.1: the FLAT instance table ──────────────────────────────────
        //
        // Every drawn asset's packed instances, concatenated in the same
        // deterministic asset order the raster loops walk, so `base + local` is
        // the global index the visibility packing stores. It is built here — once
        // — and the per-asset buffers are written from SLICES of it, which is what
        // makes "the flat table and the per-asset table agree" true by
        // construction rather than by a second call to `pack_instance`. The
        // forward path's bytes are unchanged: it writes the same records it always
        // did, from a different place.
        //
        // Built unconditionally, because the alternative is two derivations that
        // agree only while nobody edits one of them (the P21 one-door law).
        let mut flat: Vec<VgeomInstanceGpu> = Vec::new();
        let mut flat_at: BTreeMap<u128, (u32, u32)> = BTreeMap::new();
        let mut max_tri_all = 0u32;
        for (asset_id, insts) in &by_asset {
            let Some(source) = source_of.get(asset_id) else {
                continue;
            };
            let Some(residency) = self.streamer.residency(*asset_id) else {
                continue;
            };
            if source.meshlet_count() == 0
                || source.max_tri() == 0
                || residency.resident_pages() == 0
            {
                continue;
            }
            let bounds = source.bounds();
            let base = flat.len() as u32;
            flat.extend(
                insts
                    .iter()
                    .map(|i| pack_instance(&origin, frame.view, bounds, i, settings.pixel_error)),
            );
            flat_at.insert(*asset_id, (base, insts.len() as u32));
            max_tri_all = max_tri_all.max(source.max_tri());
        }

        // ── P28.1: the admission door ───────────────────────────────────────
        //
        // A pure function of the committed scene and the streamer's residency, so
        // two runs of one scripted path refuse on exactly the same frames. A
        // refusal is not a failure: the frame renders through the forward meshlet
        // raster, which P28.1 keeps precisely so there is somewhere to fall back
        // to, and the counter says which ceiling.
        let mut vis_on = false;
        if settings.visbuffer {
            if self.vis.is_none() {
                self.vis = Some(super::visbuffer::VisState::new(
                    gpu,
                    &self.view_bgl,
                    &self.lights_bgl,
                    &self.env.bgl,
                ));
            }
            let pool_bytes = self.pools.sizes[1];
            let vis = self.vis.as_mut().expect("built above");
            vis.ensure_targets(gpu, frame);
            vis.ensure_instances(
                gpu,
                flat.len() as u32,
                std::mem::size_of::<VgeomInstanceGpu>() as u64,
            );
            vis.ensure_flags(gpu, flat_at.len());
            if vis
                .admit(flat.len() as u32, pool_bytes, max_tri_all)
                .is_ok()
            {
                vis.audit.frames += 1;
                vis_on = true;
                if !flat.is_empty() {
                    gpu.queue
                        .write_buffer(&vis.instances, 0, bytemuck::cast_slice(&flat));
                }
            }
        }

        // Disjoint field borrows: the HZB build below needs `&mut self.hzb` while
        // the asset loops hold `&self.pools` / `&mut self.draws`.
        let Self {
            cull,
            raster,
            raster_bgl,
            lights_buf,
            lights_bg,
            env,
            dummy_hzb,
            pools,
            streamer,
            draws,
            hzb: hzb_chain,
            prev_view,
            report: _,
            view_bgl: _,
            lights_bgl: _,
            vis,
            vis_bases,
        } = self;
        vis_bases.clear();

        // Lights (shared with the rigid pass).
        let lights =
            LightsUniform::from_scene(frame.scene, &frame.view.origin, frame.vsm_light_slots);
        gpu.queue
            .write_buffer(lights_buf, 0, bytemuck::bytes_of(&lights));

        let occlusion = settings.occlusion;
        let two_pass = occlusion && settings.two_pass;
        let audit = frame.vgeom_audit.enabled;

        // The camera-cut trigger is a whole-frame decision (see `is_camera_cut`):
        // a discontinuous view makes last frame's visible set worthless as an
        // early set for EVERY asset.
        let cut = prev_view.is_none_or(|p| is_camera_cut(&p, frame.view));
        *prev_view = Some(*frame.view);

        if audit {
            gpu.queue.write_buffer(
                &frame.vgeom_audit.stats,
                0,
                bytemuck::cast_slice(&[0u32; (AUDIT_BYTES / 4) as usize]),
            );
        }

        if occlusion {
            hzb_chain.ensure(gpu, frame.targets.size, frame.targets.generation);
            // Single-pass v1: the pyramid is whatever the scene depth holds when
            // this node starts (the classic mesh pass), built once up front.
            if !two_pass {
                hzb_chain.build(gpu, encoder, frame);
            }
        }
        let hzb_dims = if occlusion {
            hzb_chain
                .dims()
                .map(|(w, h, mips)| [mips as f32, w as f32, h as f32, 0.0])
                .unwrap_or([1.0, 1.0, 1.0, 0.0])
        } else {
            [1.0, 1.0, 1.0, 0.0]
        };

        let env_bg = env.bind_group(gpu, frame).clone();
        let debug = FlagsGpu {
            flags: [settings.debug_meshlets as u32, 0, 0, 0],
        };

        // A vertex-pulled indirect draw of `visible`/`args` into the MSAA targets.
        // Both passes are identical apart from which pair of buffers they read.
        let raster_draw = |encoder: &mut wgpu::CommandEncoder,
                           label: &str,
                           pools: &VgeomPoolBuffers,
                           draw: &AssetDraw,
                           visible: &wgpu::Buffer,
                           args: &wgpu::Buffer| {
            let raster_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vgeom-raster"),
                layout: raster_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pools.vertices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pools.meshlets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pools.mlverts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pools.mltris.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: draw.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: draw.debug_flags.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: draw.remap.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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
            pass.set_pipeline(raster);
            pass.set_bind_group(0, frame.view_bg, &[]);
            pass.set_bind_group(1, &*lights_bg, &[]);
            pass.set_bind_group(2, &env_bg, &[]);
            pass.set_bind_group(3, &raster_bg, &[]);
            pass.draw_indirect(args, 0);
        };

        // ── Pass 1: (early | single) cull + draw ─────────────────────────────
        //
        // In two-pass mode this draws last frame's visible set — which is what
        // gives the HZB below real vgeom depth. On a conservative frame the early
        // set is the WHOLE base cut, so this draw alone is the occlusion-off
        // result and pass 2 adds nothing (see the module docs).
        // Per asset: (id, conservative, meshlet_count, floor_lod, residency generation).
        let mut planned: Vec<(u128, bool, u32, u32, u64)> = Vec::new();
        // The visibility buffer and its depth are cleared by the FIRST draw of
        // the frame and loaded by every one after — the same shape the forward
        // path's `LoadOp::Load` into the shared scene targets has, one level down.
        let mut vis_cleared = false;
        for (asset_id, insts) in &by_asset {
            let Some(source) = source_of.get(asset_id) else {
                continue;
            };
            let Some(residency) = streamer.residency(*asset_id) else {
                continue;
            };
            let meshlet_count = source.meshlet_count();
            let max_tri = source.max_tri();
            let floor_lod = residency.floor_lod();
            if meshlet_count == 0 || max_tri == 0 || residency.resident_pages() == 0 {
                continue;
            }
            let instance_count = insts.len() as u32;
            let draw = draws
                .entry(*asset_id)
                .or_insert_with(|| AssetDraw::new(gpu));
            // A grow reallocates the visible/visibility buffers, so every bind
            // group is rebuilt from `draw` below (they are per-frame) AND the
            // persisted early set is gone — hence the `state` reset.
            if draw.ensure(gpu, instance_count, meshlet_count) {
                draw.state = None;
            }
            // Residency changes what the base cut selects, so an early set
            // inherited across one is stale in exactly the way a scene-version
            // bump is. `sync_remap` reporting a rebuild IS that signal — and a
            // reallocated pool means every buffer the bind groups named is gone.
            if draw.sync_remap(gpu, residency) || pools_rebuilt {
                draw.state = None;
            }
            // Only the late dispatch publishes `vis_cur`, so a frame that skips it
            // leaves the persisted early set frozen at whatever the last two-pass
            // frame wrote. Toggling two-pass off and back on would then inherit a
            // set from an arbitrary earlier frame — still correct (occlusion is
            // subtractive regardless) but not the documented "no usable state ⇒
            // conservative" contract, so drop it explicitly.
            if !two_pass {
                draw.state = None;
            }
            let key = TemporalKey {
                scene_version: frame.scene.version,
                instance_count,
                meshlet_count,
                targets_generation: frame.targets.generation,
                residency_generation: residency.generation(),
            };
            let conservative = cut || draw.state != Some(key);
            planned.push((
                *asset_id,
                conservative,
                meshlet_count,
                floor_lod,
                key.residency_generation,
            ));

            // The per-asset instance buffer is a SLICE of the flat table built
            // above (P28.1) — one `pack_instance` call per instance per frame, as
            // before, from one place instead of two.
            let (flat_base, flat_count) = flat_at[asset_id];
            let packed = &flat[flat_base as usize..(flat_base + flat_count) as usize];
            gpu.queue
                .write_buffer(&draw.instances, 0, bytemuck::cast_slice(packed));
            let vis_slot = match (vis_on, vis.as_ref()) {
                (true, Some(v)) => {
                    let slot = vis_bases.len();
                    gpu.queue.write_buffer(
                        &v.flags[slot],
                        0,
                        bytemuck::bytes_of(&super::visbuffer::VisFlagsGpu {
                            flags: [flat_base, 0, 0, 0],
                        }),
                    );
                    vis_bases.push((*asset_id, flat_base));
                    Some(slot)
                }
                _ => None,
            };

            // Reset draw args: vertex_count = max_tri*3, instance_count = 0.
            // `max_tri` is the header's whole-mesh maximum, so the draw shape is a
            // constant of the asset and does not move as pages come and go.
            let reset = [max_tri * 3, 0u32, 0u32, 0u32];
            gpu.queue
                .write_buffer(&draw.draw_args, 0, bytemuck::cast_slice(&reset));
            gpu.queue
                .write_buffer(&draw.draw_args_late, 0, bytemuck::cast_slice(&reset));
            gpu.queue
                .write_buffer(&draw.debug_flags, 0, bytemuck::bytes_of(&debug));

            let mode = if two_pass {
                CullMode::Early
            } else {
                CullMode::Single
            };
            let params = cull_params(
                frame.view,
                meshlet_count,
                instance_count,
                settings,
                hzb_dims,
                mode,
                conservative,
                audit,
                floor_lod,
            );
            gpu.queue
                .write_buffer(&draw.params, 0, bytemuck::bytes_of(&params));

            // The early dispatch runs no occlusion test, so it binds the dummy.
            let hzb_view = if occlusion && !two_pass {
                hzb_chain.full_view().unwrap_or(&*dummy_hzb)
            } else {
                &*dummy_hzb
            };
            let cull_bg = cull.bind_group(
                gpu,
                &draw.params,
                pools,
                &draw.remap,
                &draw.instances,
                &draw.visible,
                &draw.draw_args,
                hzb_view,
                &draw.vis_prev,
                &draw.vis_cur,
                &frame.vgeom_audit.stats,
            );
            let total = instance_count * meshlet_count;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vgeom-cull-early"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cull.pipeline);
                pass.set_bind_group(0, &cull_bg, &[]);
                pass.dispatch_workgroups(total.div_ceil(64).max(1), 1, 1);
            }
            match (vis_slot, vis.as_ref()) {
                (Some(slot), Some(v)) => {
                    vis_raster_draw(
                        gpu,
                        encoder,
                        "visbuffer-early",
                        v,
                        pools,
                        &v.flags[slot],
                        &draw.visible,
                        &draw.remap,
                        &draw.draw_args,
                        frame.view_bg,
                        !vis_cleared,
                    );
                    vis_cleared = true;
                }
                _ => raster_draw(
                    encoder,
                    "vgeom-raster-early",
                    pools,
                    draw,
                    &draw.visible,
                    &draw.draw_args,
                ),
            }
        }

        if !two_pass {
            // …and so does the resolve: this is a whole frame's visibility, not
            // half of one. A `return` that skipped it would leave the meshlet
            // geometry rasterized into an id buffer nothing ever shaded — a black
            // hole in the frame that no assertion about the buffer could see.
            if vis_cleared {
                if let Some(v) = vis.as_ref() {
                    vis_resolve_and_feedback(gpu, encoder, v, pools, frame, lights_bg, &env_bg);
                }
            }
            // Single-pass still fills the counters (its one dispatch sees every
            // pair), so the readback has to be recorded on this exit too.
            if audit {
                encoder.copy_buffer_to_buffer(
                    &frame.vgeom_audit.stats,
                    0,
                    &frame.vgeom_audit.readback,
                    0,
                    AUDIT_BYTES,
                );
            }
            return;
        }

        // ── HZB from the depth the early draw just wrote ─────────────────────
        hzb_chain.build(gpu, encoder, frame);
        let hzb_view = hzb_chain.full_view().unwrap_or(&*dummy_hzb);

        // ── Pass 2: late cull (publish this frame's visibility) + late draw ──
        for (asset_id, conservative, meshlet_count, floor_lod, residency_generation) in planned {
            let Some(draw) = draws.get_mut(&asset_id) else {
                continue;
            };
            let Some(insts) = by_asset.get(&asset_id) else {
                continue;
            };
            let instance_count = insts.len() as u32;
            let params = cull_params(
                frame.view,
                meshlet_count,
                instance_count,
                settings,
                hzb_dims,
                CullMode::Late,
                conservative,
                audit,
                floor_lod,
            );
            gpu.queue
                .write_buffer(&draw.params_late, 0, bytemuck::bytes_of(&params));
            let cull_bg = cull.bind_group(
                gpu,
                &draw.params_late,
                pools,
                &draw.remap,
                &draw.instances,
                &draw.visible_late,
                &draw.draw_args_late,
                hzb_view,
                &draw.vis_prev,
                &draw.vis_cur,
                &frame.vgeom_audit.stats,
            );
            let total = instance_count * meshlet_count;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vgeom-cull-late"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cull.pipeline);
                pass.set_bind_group(0, &cull_bg, &[]);
                pass.dispatch_workgroups(total.div_ceil(64).max(1), 1, 1);
            }
            // Issued unconditionally: on a conservative frame the shader appends
            // nothing, so this is a zero-instance indirect draw. Deliberately NOT
            // skipped on the CPU — the drawn set lives entirely on the GPU, and a
            // CPU-side assumption about it is exactly the kind of shortcut that
            // turns into missing geometry.
            let vis_slot = vis_bases.iter().position(|(a, _)| *a == asset_id);
            match (vis_slot, vis.as_ref()) {
                (Some(slot), Some(v)) if vis_on => {
                    vis_raster_draw(
                        gpu,
                        encoder,
                        "visbuffer-late",
                        v,
                        pools,
                        &v.flags[slot],
                        &draw.visible_late,
                        &draw.remap,
                        &draw.draw_args_late,
                        frame.view_bg,
                        !vis_cleared,
                    );
                    vis_cleared = true;
                }
                _ => raster_draw(
                    encoder,
                    "vgeom-raster-late",
                    pools,
                    draw,
                    &draw.visible_late,
                    &draw.draw_args_late,
                ),
            }

            // Ping-pong: what the late dispatch just published becomes next
            // frame's early set. The bind groups above hold their own references,
            // so swapping the handles now is safe.
            std::mem::swap(&mut draw.vis_prev, &mut draw.vis_cur);
            draw.state = Some(TemporalKey {
                scene_version: frame.scene.version,
                instance_count,
                meshlet_count,
                targets_generation: frame.targets.generation,
                // The SAME value pass 1 keyed on, so a frame that publishes an
                // early set cannot disagree with the frame that consumes it.
                residency_generation,
            });
        }

        if vis_cleared {
            if let Some(v) = vis.as_ref() {
                vis_resolve_and_feedback(gpu, encoder, v, pools, frame, lights_bg, &env_bg);
            }
        }

        if audit {
            encoder.copy_buffer_to_buffer(
                &frame.vgeom_audit.stats,
                0,
                &frame.vgeom_audit.readback,
                0,
                AUDIT_BYTES,
            );
        }
    }
}

// ── P28.1: the three visibility passes ───────────────────────────────────────

/// One asset's indirect draw into the visibility buffer.
///
/// A free function rather than a closure because the closure the forward path
/// uses captures `env_bg`/`lights_bg` by reference, and this one has to be
/// callable while `vis` is borrowed for its per-asset flags uniform. Every
/// argument is spelled out for the P27.3 reason: a struct is where a field
/// becomes easy to forget to fill.
#[allow(clippy::too_many_arguments)]
fn vis_raster_draw(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    vis: &super::visbuffer::VisState,
    pools: &VgeomPoolBuffers,
    flags: &wgpu::Buffer,
    visible: &wgpu::Buffer,
    remap: &wgpu::Buffer,
    args: &wgpu::Buffer,
    view_bg: &wgpu::BindGroup,
    clear: bool,
) {
    let Some(targets) = vis.targets.as_ref() else {
        return;
    };
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("visbuffer-raster"),
        layout: &vis.raster_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pools.vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pools.meshlets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pools.mlverts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pools.mltris.as_entire_binding(),
            },
            // The FLAT table, not this asset's slice: the id the raster writes is
            // a global index, so the buffer it indexes has to be the global one.
            wgpu::BindGroupEntry {
                binding: 4,
                resource: vis.instances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: visible.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: flags.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: remap.as_entire_binding(),
            },
        ],
    });
    let color_ops = wgpu::Operations {
        // `VIS_EMPTY` is zero, which the packing's `instance + 1` bias makes
        // unreachable from a real fragment — so "clear to zero" and "no geometry
        // here" are the same statement rather than two that have to be kept in
        // step.
        load: if clear {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        },
        store: wgpu::StoreOp::Store,
    };
    let depth_ops = wgpu::Operations {
        load: if clear {
            wgpu::LoadOp::Clear(crate::camera::DEPTH_CLEAR)
        } else {
            wgpu::LoadOp::Load
        },
        store: wgpu::StoreOp::Store,
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.color,
            resolve_target: None,
            depth_slice: None,
            ops: color_ops,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth,
            depth_ops: Some(depth_ops),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&vis.raster);
    pass.set_bind_group(0, view_bg, &[]);
    pass.set_bind_group(1, &bg, &[]);
    pass.draw_indirect(args, 0);
}

/// The material-resolve pass and the per-fragment feedback, in that order.
///
/// Recorded once per frame after every visibility draw, and only when the frame
/// actually rasterized one — a resolve over a buffer nothing wrote is a
/// fullscreen pass that discards every pixel, which costs a dispatch and proves
/// nothing.
fn vis_resolve_and_feedback(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    vis: &super::visbuffer::VisState,
    pools: &VgeomPoolBuffers,
    frame: &FrameData,
    lights_bg: &wgpu::BindGroup,
    env_bg: &wgpu::BindGroup,
) {
    let Some(targets) = vis.targets.as_ref() else {
        return;
    };
    let pool_entries = |first: u32| {
        [
            wgpu::BindGroupEntry {
                binding: first,
                resource: pools.vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 1,
                resource: pools.meshlets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 2,
                resource: pools.mlverts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 3,
                resource: pools.mltris.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 4,
                resource: vis.instances.as_entire_binding(),
            },
        ]
    };

    // ── the resolve ─────────────────────────────────────────────────────────
    let resolve_bg = {
        let pool = pool_entries(1);
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visbuffer-resolve"),
            layout: &vis.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.color),
                },
                pool[0].clone(),
                pool[1].clone(),
                pool[2].clone(),
                pool[3].clone(),
                pool[4].clone(),
            ],
        })
    };
    {
        // Into the MSAA scene colour AND the MSAA scene depth, both LOADED: the
        // resolve is one more opaque pass as far as everything downstream is
        // concerned, and `@builtin(frag_depth)` is what puts the meshlet depth
        // where translucency, water and the shadow marking expect to find it.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visbuffer-resolve"),
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
        pass.set_pipeline(&vis.resolve);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, lights_bg, &[]);
        pass.set_bind_group(2, env_bg, &[]);
        pass.set_bind_group(3, &resolve_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    // ── the per-fragment virtual-texture feedback ───────────────────────────
    let Some(fb) = frame.vt_feedback else {
        return;
    };
    gpu.queue.write_buffer(
        &vis.feedback_params,
        0,
        bytemuck::bytes_of(&super::visbuffer::VisFeedbackParamsGpu {
            // The JITTERED matrix — the one the buffer above was rasterized
            // with (the P27.1 law).
            view_proj: frame.view_proj,
            counts: [
                targets.size.0,
                targets.size.1,
                fb.words,
                fb.bases.len() as u32,
            ],
        }),
    );
    if !fb.bases.is_empty() {
        gpu.queue
            .write_buffer(&vis.feedback_bases, 0, bytemuck::cast_slice(fb.bases));
    }
    let pool = pool_entries(2);
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("visbuffer-feedback"),
        layout: &vis.feedback_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vis.feedback_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.color),
            },
            pool[0].clone(),
            pool[1].clone(),
            pool[2].clone(),
            pool[3].clone(),
            pool[4].clone(),
            wgpu::BindGroupEntry {
                binding: 7,
                resource: fb.table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: vis.feedback_bases.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: fb.mask.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("visbuffer-feedback"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&vis.feedback);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(
        targets.size.0.div_ceil(8).max(1),
        targets.size.1.div_ceil(8).max(1),
        1,
    );
}

// ── HZB: reverse-Z min-depth pyramid from the MSAA scene depth ───────────────

/// A hierarchical depth pyramid built by compute from the **live 4× MSAA scene
/// depth** (min over subsamples — see `vgeom_hzb.wgsl` for why that, and not the
/// single-sample prepass, is what makes occlusion provably subtractive).
/// Allocated only when [`VgeomSettings::occlusion`] is on, so the occlusion-off
/// path never touches it.
///
/// **Resizable-resource discipline (P17.2):** the pyramid is viewport-sized, so
/// its own recreation bumps `generation`, and both bind-group sets embed a
/// [`FrameTargets`](crate::renderer::FrameTargets) view. They are therefore
/// [`GenCache`](super::GenCache)d on `(targets.generation, hzb.generation)` —
/// either moving alone must invalidate, exactly like `EnvBinding`.
///
/// A key of its **own** rather than [`super::ResourceKey`] (P18.4): the HZB embeds
/// no atmosphere and no GI resource, so borrowing the shared key would rebuild
/// these bind groups on every GI-quality clamp — a lie about what invalidates them,
/// and precisely the kind of drift the `GenCache` extraction exists to prevent.
pub(crate) type HzbKey = (u64, u64);

pub(crate) struct HzbChain {
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    down_pipeline: wgpu::ComputePipeline,
    down_bgl: wgpu::BindGroupLayout,
    texture: Option<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    full_view: Option<wgpu::TextureView>,
    size: (u32, u32),
    /// Monotonic, bumped whenever the pyramid texture is (re)created.
    generation: u64,
    copy_bg: super::GenCache<HzbKey, wgpu::BindGroup>,
    down_bgs: super::GenCache<HzbKey, Vec<wgpu::BindGroup>>,
}

impl HzbChain {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-hzb"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vgeom_hzb.wgsl").into()),
            });
        let copy_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-hzb-copy"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            // P18.1: the MSAA scene depth, loaded per sample.
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let down_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-hzb-down"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let mk = |label, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(bgl)],
                    immediate_size: 0,
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let copy_pipeline = mk("vgeom-hzb-copy", &copy_bgl, "cs_copy");
        let down_pipeline = mk("vgeom-hzb-down", &down_bgl, "cs_down");
        Self {
            copy_pipeline,
            copy_bgl,
            down_pipeline,
            down_bgl,
            texture: None,
            mip_views: Vec::new(),
            full_view: None,
            size: (0, 0),
            generation: 0,
            copy_bg: super::GenCache::default(),
            down_bgs: super::GenCache::default(),
        }
    }

    /// Monotonic stamp bumped whenever the pyramid texture is (re)created — the
    /// second half of every `HzbKey`, so a bind group holding a view into it is
    /// dropped when the view stops being valid.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn dims(&self) -> Option<(u32, u32, u32)> {
        self.texture
            .as_ref()
            .map(|_| (self.size.0, self.size.1, self.mip_views.len() as u32))
    }

    /// Allocate (or reallocate) the pyramid for `size`. `targets_generation` only
    /// participates through the caller's [`HzbKey`]; the texture
    /// itself is keyed on its size, and recreating it bumps `generation` so the
    /// cached bind groups — which hold views into it — are dropped.
    fn ensure(&mut self, gpu: &GpuContext, size: (u32, u32), _targets_generation: u64) {
        if self.size == size && self.texture.is_some() {
            return;
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        let mips = (32 - (w.max(h)).leading_zeros()).max(1);
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vgeom-hzb"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.mip_views = (0..mips)
            .map(|m| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("vgeom-hzb-mip"),
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.full_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.texture = Some(tex);
        self.size = (w, h);
        self.generation = self.generation.wrapping_add(1);
    }

    /// The full-chain sampled view for the cull compute (valid after [`build`]).
    ///
    /// [`build`]: HzbChain::build
    pub(crate) fn full_view(&self) -> Option<&wgpu::TextureView> {
        self.full_view.as_ref()
    }

    /// Build the pyramid from the frame's **MSAA scene depth** into [`full_view`].
    /// Call it at the point in the frame whose depth you want to occlude against:
    /// single-pass builds once before the cull, two-pass builds after the early
    /// draw so meshlets occlude meshlets.
    ///
    /// [`full_view`]: HzbChain::full_view
    pub(crate) fn build(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameData,
    ) {
        self.ensure(gpu, frame.targets.size, frame.targets.generation);
        let (w, h) = self.size;
        let key: HzbKey = (frame.targets.generation, self.generation);
        let (copy_bgl, down_bgl, mip_views) = (&self.copy_bgl, &self.down_bgl, &self.mip_views);

        // Pass 0: min-over-samples of the MSAA scene depth → mip 0.
        let bg0 = self.copy_bg.get_or_build(key, || {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vgeom-hzb-copy"),
                layout: copy_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_views[0]),
                    },
                ],
            })
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vgeom-hzb-copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }

        // Passes N: min-downsample.
        let bgs = self.down_bgs.get_or_build(key, || {
            (1..mip_views.len())
                .map(|m| {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vgeom-hzb-down"),
                        layout: down_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&mip_views[m - 1]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&mip_views[m]),
                            },
                        ],
                    })
                })
                .collect()
        });
        for (i, bg) in bgs.iter().enumerate() {
            let m = i + 1;
            let mw = (w >> m).max(1);
            let mh = (h >> m).max(1);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vgeom-hzb-down"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(mw.div_ceil(8), mh.div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 0.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        }
    }

    #[test]
    fn threshold_grows_with_distance() {
        let v = view();
        let eye = v.eye_local();
        let near = lod_threshold(eye, Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, &v, 1.0);
        let far = lod_threshold(eye, Vec3::new(0.0, 0.0, -50.0), 1.0, 1.0, &v, 1.0);
        assert!(far > near, "far {far} should exceed near {near}");
        assert!(near > 0.0);
    }

    #[test]
    fn threshold_scales_with_pixel_error() {
        let v = view();
        let eye = v.eye_local();
        let t1 = lod_threshold(eye, Vec3::new(0.0, 0.0, -10.0), 1.0, 1.0, &v, 1.0);
        let t2 = lod_threshold(eye, Vec3::new(0.0, 0.0, -10.0), 1.0, 1.0, &v, 2.0);
        assert!((t2 - 2.0 * t1).abs() < 1e-4, "t2 {t2} ≈ 2·t1 {t1}");
    }

    #[test]
    fn frustum_planes_contain_center_reject_far_offscreen() {
        let v = view();
        let planes = frustum_planes(v.view_proj());
        // A point right in front of the camera is inside every plane.
        let inside = Vec3::new(0.0, 0.0, 0.0);
        assert!(!outside_frustum(inside, 0.1, &planes), "center culled");
        // A point far to the side (way outside the horizontal FOV) is rejected.
        let side = Vec3::new(1000.0, 0.0, 0.0);
        assert!(outside_frustum(side, 0.1, &planes), "side not culled");
        // Behind the camera.
        let behind = Vec3::new(0.0, 0.0, 100.0);
        assert!(outside_frustum(behind, 0.1, &planes), "behind not culled");
    }

    /// The `MODE_*` / `AUDIT_*` constants in `vgeom_cull.wgsl` are a wire contract
    /// with [`CullMode`] and [`VgeomAudit`]'s field order. A silent renumbering on
    /// either side would make the early dispatch run the late dispatch's branch,
    /// or report the occluded count as the draw count — both of which produce a
    /// plausible-looking frame. Pin both ends against the shader source. GPU-free,
    /// so it runs on every CI leg.
    ///
    /// The same argument covers `NOT_RESIDENT` (P18.2): it is the sentinel the
    /// remap table and BOTH shaders must agree on, and a mismatch would silently
    /// read meshlet slot 0 for every non-resident page — real geometry, at a
    /// plausible-looking position, from the wrong LOD of the wrong asset.
    #[test]
    fn shader_constants_match_the_rust_side() {
        let src = include_str!("../shaders/vgeom_cull.wgsl");
        for (name, value) in [
            ("MODE_SINGLE", CullMode::Single as u32),
            ("MODE_EARLY", CullMode::Early as u32),
            ("MODE_LATE", CullMode::Late as u32),
            ("AUDIT_BASE", 0),
            ("AUDIT_OCCLUDED", 1),
            ("AUDIT_EARLY", 2),
            ("AUDIT_LATE", 3),
            ("AUDIT_CLAMPED", 4),
        ] {
            let want = format!("const {name}: u32 = {value}u;");
            assert!(src.contains(&want), "vgeom_cull.wgsl must declare `{want}`");
        }
        assert_eq!(
            inf_vgeom::NOT_RESIDENT,
            u32::MAX,
            "the remap sentinel is 0xFFFFFFFF on both sides"
        );
        let sentinel = "const NOT_RESIDENT: u32 = 0xFFFFFFFFu;";
        assert!(
            src.contains(sentinel),
            "vgeom_cull.wgsl must declare `{sentinel}`"
        );
        // The raster resolves through the same table, so it needs the same value.
        let raster = include_str!("../shaders/vgeom_mesh.wgsl");
        assert!(
            raster.contains(sentinel),
            "vgeom_mesh.wgsl must declare `{sentinel}`"
        );
        // The audit buffer's slot count is a layout contract too.
        assert_eq!(AUDIT_BYTES, 32);
        assert!(src.contains("array<atomic<u32>, 8>"));
    }

    /// A pan is not a cut (otherwise the early set would reset every frame in a
    /// moving game and two-pass would buy nothing); a teleport, a snap turn and a
    /// viewport resize are.
    #[test]
    fn camera_cut_fires_on_discontinuities_only() {
        let a = view();
        assert!(!is_camera_cut(&a, &a));

        let mut step = a;
        step.eye_world.x += 0.4;
        assert!(!is_camera_cut(&a, &step), "40 cm is a walk, not a cut");

        let mut teleport = a;
        teleport.eye_world.z += 500.0;
        assert!(is_camera_cut(&a, &teleport));

        let mut snap = a;
        snap.forward = Vec3::Z;
        assert!(is_camera_cut(&a, &snap), "a 180° turn is a cut");

        let mut resized = a;
        resized.width += 1;
        assert!(is_camera_cut(&a, &resized));

        let mut zoomed = a;
        zoomed.fov_y *= 0.5;
        assert!(is_camera_cut(&a, &zoomed));
    }

    #[test]
    fn cull_flags_bits() {
        let mut s = VgeomSettings {
            frustum_cull: true,
            cone_cull: false,
            occlusion: false,
            ..VgeomSettings::default()
        };
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM);
        s.cone_cull = true;
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM | FLAG_CONE);
        s.occlusion = true;
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM | FLAG_CONE | FLAG_OCCLUSION);
    }
}
