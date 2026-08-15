//! GPU-instanced scatter (P18.5) — the pass PCG volumes and painted foliage
//! render through.
//!
//! # What this replaces
//!
//! Since P10.5 a `PcgVolume`'s evaluated cache and a `Foliage`'s placed instances
//! were expanded, **one [`MeshInstance`](crate::MeshInstance) each**, into
//! `RenderScene::instances` by both projectors. A 100k-instance scatter therefore
//! cost 100k CPU structs per projection, 100k 176-byte `InstanceRaw`s per pack,
//! and a ~17 MB vertex-buffer upload — before the GPU had culled a single one of
//! them. Both hosts also carried the same warning: *">50k instances — instanced-
//! draw perf path is a follow-up"*. This is that follow-up, and it is what makes
//! P19's biome populations reachable: the payload is uploaded once per content
//! change and everything else happens on the GPU.
//!
//! # The frame
//!
//! Per batch, three compute dispatches and up to two indirect draws:
//!
//! 1. `cs_classify` — one thread per instance: distance band, frustum, HZB
//!    occlusion; then an in-workgroup exclusive prefix sum over the two "joins
//!    this list" flags.
//! 2. `cs_scan` — one thread, total: exclusive-scan the per-workgroup partials
//!    and publish both indirect `instance_count`s.
//! 3. `cs_compact` — one thread per instance: write its index into its dense slot.
//! 4. `draw_indirect` the full-mesh list, then `draw_indirect` the impostor list.
//!
//! The compaction is a prefix sum rather than an atomic append **on purpose**;
//! `scatter_cull.wgsl`'s header states the argument, and the short version is that
//! the LOD cross-fade is a dithered discard, so unlike the meshlet path the draw
//! order here really can reach the image.
//!
//! # The HZB is this node's own, and that is a cost decision
//!
//! `VgeomNode` builds a pyramid mid-frame (after its early draw) and keeps it
//! private. This node builds a second one from `targets.depth` at the point it
//! runs — which is **after** the rigid mesh pass, both vgeom paths, the skinned
//! pass and terrain — so scatter is occluded by every opaque surface the engine
//! draws, the richest occluder set available. Sharing the meshlet pyramid would
//! have been cheaper and strictly worse: it is built before the late vgeom draw
//! and before terrain, and it would couple scatter's culling to
//! `VgeomSettings::two_pass`, a setting about a different subsystem. A pyramid is
//! a pure function of the depth target at the moment it is built, so a second one
//! is a cost, never a correctness question — and it costs nothing at all on a
//! scene with no scatter, because the node returns before building it.
//!
//! Correctness is inherited, not re-argued: the occlusion test is
//! `hzb_occlusion.wgsl`'s, the same provably-*subtractive* rule P18.1 proved, so a
//! frame with scatter occlusion on is pixel-identical to one with it off.
//!
//! # The CPU fallback
//!
//! `ScatterSettings::gpu` selects the *mechanism*. With it off — `RenderTier`
//! Medium/Low, `clamp_mobile`, or an adapter without compute/indirect — the same
//! batches draw through the rigid mesh pipeline with `InstanceRaw`, CPU-culled
//! against a **bucketed** camera position (the P17.2 sky-view-LUT precedent: the
//! eye is snapped to a 8 m lattice and the snapped value is part of the re-pack
//! key, so a walking camera does not re-pack every frame and the packed set stays
//! a pure function of its key). The tier decides how the foliage is drawn; it
//! never decides whether there is any.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{DVec3, Mat3, Mat4, Vec3};
use inf_math::FloatingOrigin;

use super::mesh::{vertex_layouts, InstanceRaw, LightsUniform};
use super::vgeom::{dummy_hzb, frustum_planes, HzbChain, HzbKey};
use super::GenCache;
use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::primitives::{PrimMesh, PrimStorage};
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{ScatterBatch, ScatterInstanceRaw};

/// Threads per cull workgroup. Must equal `WG` in `scatter_cull.wgsl` (pinned by
/// [`tests::shader_constants_match_the_rust_side`]).
const WORKGROUP: u32 = 256;

/// Audit counter slots, in the order `scatter_cull.wgsl` writes them.
const AUDIT_SLOTS: usize = 8;
const AUDIT_BYTES: u64 = (AUDIT_SLOTS * 4) as u64;

/// The 8 m lattice the CPU fallback snaps the camera onto before culling.
const FALLBACK_EYE_BUCKET_M: f64 = 8.0;

// ── audit ────────────────────────────────────────────────────────────────────

/// GPU instance-cull counters (P18.5), aggregated over every batch in the frame.
///
/// **Off by default and free when off** — the shader skips the atomics and the
/// node records no readback copy — exactly like `VgeomAudit`. It exists so the
/// gates can prove the culling is *real* rather than a no-op that trivially
/// satisfies a pixel comparison.
///
/// Every counter is a SUM, which is why atomic increments are fine here while the
/// compaction is not: addition is order-independent, list construction is not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScatterAudit {
    /// Instances the cull looked at.
    pub candidates: u32,
    /// Rejected by the frustum test.
    pub frustum_culled: u32,
    /// Rejected by the HZB occlusion test.
    pub occluded: u32,
    /// Rejected as past the cull distance.
    pub distance_culled: u32,
    /// Joined the full-mesh list.
    ///
    /// **These six do not partition the candidate set, and the arithmetic is worth
    /// stating.** An instance in the *pure* impostor band — past `mesh_distance_m`
    /// but inside `cull_distance_m` — is counted only in
    /// [`impostor`](Self::impostor), and one in the cross-fade band is counted in
    /// **both** outcome slots. So the only exact relation is a sandwich:
    /// `rejected + max(mesh, impostor) <= candidates <= rejected + mesh + impostor`,
    /// where `rejected = frustum_culled + occluded + distance_culled`. A gate that
    /// assumed a partition would fail on any scene wide enough to have a far field
    /// — which is every scene the impostor band exists for.
    pub mesh: u32,
    /// Joined the impostor list. See [`mesh`](Self::mesh) for how the six counters
    /// relate to `candidates`.
    pub impostor: u32,
    /// Scatter instances packed as **cascaded-shadow casters** this frame.
    ///
    /// The odd one out: a CPU counter, and the only figure here describing work the
    /// cull compute never sees. Scatter casters are packed by the shadow node under
    /// [`shadow_caster_settings`], so this is what says whether the tier clamps
    /// actually bit — the question nobody was asking back when the shadow path
    /// synthesized its own settings and escaped all of them. Bounded by
    /// [`MAX_CPU_SCATTER_INSTANCES`]; unlike the GPU counters it is published even
    /// when the audit is off, because it costs one relaxed store.
    pub shadow_casters: u32,
    /// Distinct instance **payloads** resident on the GPU this frame.
    ///
    /// The observable half of content addressing, and the only way to tell "two
    /// batches share one upload" from "two batches each got one" — a distinction no
    /// pixel comparison can make, and the one the anchor-collision defect hid
    /// behind. Also a CPU counter, published on the same terms.
    pub uploads: u32,
}

pub struct ScatterAuditResources {
    pub(crate) enabled: bool,
    pub(crate) stats: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Scatter instances packed as shadow casters on the last frame — a **CPU**
    /// counter, published through the same struct as the GPU ones because it
    /// answers the same question ("is the culling real?") about the one part of
    /// the scatter path that never reaches the cull compute. `AtomicU32` because
    /// `FrameData` hands every node a `&` to this.
    shadow_casters: std::sync::atomic::AtomicU32,
    /// Distinct instance payloads resident after the last sync — the observable
    /// half of content addressing. Same terms as `shadow_casters`.
    uploads: std::sync::atomic::AtomicU32,
}

impl ScatterAuditResources {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        Self {
            enabled: false,
            stats: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter-audit"),
                size: AUDIT_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter-audit-readback"),
                size: AUDIT_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            shadow_casters: std::sync::atomic::AtomicU32::new(0),
            uploads: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Publish the shadow-caster count for this frame. Called by the shadow node,
    /// the only pass that packs them, and free: one relaxed store.
    pub(crate) fn record_shadow_casters(&self, n: u32) {
        self.shadow_casters
            .store(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shadow_casters(&self) -> u32 {
        self.shadow_casters
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Publish the resident payload count (called by the scatter node's sync).
    pub(crate) fn record_uploads(&self, n: u32) {
        self.uploads.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn uploads(&self) -> u32 {
        self.uploads.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn read(&self, gpu: &GpuContext) -> ScatterAudit {
        let v = super::vgeom::map_u32(gpu, &self.readback);
        ScatterAudit {
            candidates: v.first().copied().unwrap_or(0),
            frustum_culled: v.get(1).copied().unwrap_or(0),
            occluded: v.get(2).copied().unwrap_or(0),
            distance_culled: v.get(3).copied().unwrap_or(0),
            mesh: v.get(4).copied().unwrap_or(0),
            impostor: v.get(5).copied().unwrap_or(0),
            shadow_casters: self
                .shadow_casters
                .load(std::sync::atomic::Ordering::Relaxed),
            uploads: self.uploads.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

// ── uniforms ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParamsGpu {
    view_proj: [f32; 16],
    frustum: [[f32; 4]; 6],
    eye: [f32; 4],
    anchor: [f32; 4],
    counts: [u32; 4],
    bands: [f32; 4],
    hzb: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RasterParamsGpu {
    anchor: [f32; 4],
    material: [f32; 4],
    emissive: [f32; 4],
    bands: [f32; 4],
    geom: [u32; 4],
}

const FLAG_FRUSTUM: u32 = 1;
const FLAG_OCCLUSION: u32 = 4;

/// The effective distance bands for one batch: `(mesh_end, cull, fade,
/// impostors_on)`, in metres.
///
/// Two clamps, both one-directional. The authored `draw_distance` (the content
/// knob, `PcgVolume::draw_distance`) can only pull the cull distance **in** —
/// content may ask for less detail than the tier allows, never for more. And with
/// impostors off the mesh band is stretched to the cull distance, so the same
/// `scatter_mesh.wgsl` weight formula gives it a dithered fade-out at the edge
/// instead of a hard pop; there is no second code path for the no-impostor case.
pub fn effective_bands(
    settings: &crate::settings::ScatterSettings,
    draw_distance: f64,
) -> (f32, f32, f32, bool) {
    let mut cull = settings.cull_distance_m.max(0.0);
    if draw_distance > 0.0 {
        cull = cull.min(draw_distance as f32);
    }
    let fade = settings.fade_band_m.max(0.0);
    let impostors = settings.impostors;
    let mesh_end = if impostors {
        settings.mesh_distance_m.max(0.0).min(cull)
    } else {
        cull
    };
    (mesh_end, cull, fade, impostors)
}

// ── per-batch GPU state ──────────────────────────────────────────────────────

/// A content-addressed instance **upload** — the payload every batch that shares
/// a `ScatterData::key` reads from.
///
/// Deliberately split from the per-batch scratch below. Content addressing is
/// about the *bytes*: two foliage entities painted from the same stroke, or one
/// mesh scattered at two anchors, genuinely are one upload. Everything else a
/// batch needs is **per-frame state for one draw** — its compacted list, its
/// indirect args, its uniforms — and keying that by content too was a defect the
/// audit reproduced: two same-content batches at different anchors overwrote each
/// other's uniforms and only the last one drew (615 instances against a control's
/// 615 + 608). Sharing the expensive thing and separating the cheap ones is the
/// whole fix.
struct InstanceUpload {
    buffer: wgpu::Buffer,
    /// Instances actually uploaded.
    count: u32,
    /// Instance capacity the buffer was sized for (a power of two).
    capacity: u32,
    /// Which primitive the payload draws. Part of `ScatterData::key`, so it is a
    /// property of the upload rather than of the batch.
    mesh: PrimMesh,
}

impl InstanceUpload {
    fn new(gpu: &GpuContext, mesh: PrimMesh, payload: &[ScatterInstanceRaw]) -> Self {
        let count = payload.len() as u32;
        let capacity = count.max(1).next_power_of_two();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-instances"),
            size: (capacity as u64 * std::mem::size_of::<ScatterInstanceRaw>() as u64).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(payload));
        Self {
            buffer,
            count,
            capacity,
            mesh,
        }
    }
}

/// Per-batch, per-frame GPU scratch: the compaction's working set, the indirect
/// args, the two uniforms and the bind groups that tie them to a shared
/// [`InstanceUpload`].
///
/// Keyed by `(content key, batch pick id)`, not by content alone. The pair is
/// unique by construction for everything the projectors emit: a foliage entity's
/// several batches share a pick id but differ in *mesh kind*, which is part of the
/// content key, and two entities have different pick ids. A duplicate pair would
/// mean one object drawn twice with identical geometry, which is a caller bug
/// rather than a case to accommodate.
struct BatchScratch {
    /// The shared payload this scratch was built against — held so the buffer the
    /// raster bind group references cannot be freed under a live entry, and so a
    /// re-upload under the same key is detectable by pointer.
    upload: Arc<InstanceUpload>,
    slots: wgpu::Buffer,
    partials: wgpu::Buffer,
    visible: wgpu::Buffer,
    args: wgpu::Buffer,
    cull_params: wgpu::Buffer,
    raster_params: wgpu::Buffer,
    raster_bg: wgpu::BindGroup,
    cull_bg: GenCache<HzbKey, wgpu::BindGroup>,
}

impl BatchScratch {
    fn new(
        gpu: &GpuContext,
        raster_bgl: &wgpu::BindGroupLayout,
        deform: &crate::deform::DeformResources,
        prim: &PrimStorage,
        upload: Arc<InstanceUpload>,
    ) -> Self {
        // The CULL bind group is not built here: it embeds the HZB pyramid, which
        // is a resizable resource, so it lives behind a `GenCache` keyed on
        // `(targets.generation, hzb.generation)` and is built per frame in `run`.
        // The RASTER group embeds nothing resizable and is built once.
        let capacity = upload.capacity;
        let nwg = capacity.div_ceil(WORKGROUP);
        let mk = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let slots = mk(
            "scatter-slots",
            capacity as u64 * 8,
            wgpu::BufferUsages::empty(),
        );
        // `nwg + 1`: the scan pass parks the two totals one past the last
        // workgroup's base, which is where the compaction reads the capacity clamp
        // from and where a readback would find the visible counts.
        let partials = mk(
            "scatter-partials",
            (nwg as u64 + 1) * 8,
            wgpu::BufferUsages::empty(),
        );
        // Two regions of `capacity`: full-mesh at [0, cap), impostor at [cap, 2cap).
        // Sized for the worst case in both, which is what makes the compaction's
        // bounds check a formality rather than a silent truncation.
        let visible = mk(
            "scatter-visible",
            capacity as u64 * 2 * 4,
            wgpu::BufferUsages::empty(),
        );
        let args = mk("scatter-args", 32, wgpu::BufferUsages::INDIRECT);
        let cull_params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-cull-params"),
            size: std::mem::size_of::<CullParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-raster-params"),
            size: std::mem::size_of::<RasterParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter-raster"),
            layout: raster_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raster_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: upload.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prim.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: prim.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&deform.view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: deform.uniform.as_entire_binding(),
                },
            ],
        });
        Self {
            upload,
            slots,
            partials,
            visible,
            args,
            cull_params,
            raster_params,
            raster_bg,
            cull_bg: GenCache::default(),
        }
    }
}

// ── the node ─────────────────────────────────────────────────────────────────

pub struct ScatterNode {
    cull_pipelines: [wgpu::ComputePipeline; 3],
    cull_bgl: wgpu::BindGroupLayout,
    raster_bgl: wgpu::BindGroupLayout,
    mesh_pipeline: wgpu::RenderPipeline,
    impostor_pipeline: wgpu::RenderPipeline,
    prim_storage: PrimStorage,
    hzb: HzbChain,
    dummy_hzb: wgpu::TextureView,
    /// Instance payloads, keyed by `ScatterData::key` — **content-addressed**, so a
    /// changed scatter is a *different* entry rather than a stale one, and two
    /// batches with identical geometry share one upload. Retained to the frame's
    /// live set (the `ClassicVgeomNode` eviction rule).
    uploads: HashMap<u128, Arc<InstanceUpload>>,
    /// Per-batch draw state, keyed by `(content key, batch pick id)`. Content alone
    /// is **not** enough: two batches of the same payload at different anchors are
    /// two different draws, and sharing their uniforms means the second overwrites
    /// the first and only one of the two fields appears. See [`BatchScratch`].
    scratch: HashMap<(u128, u32), BatchScratch>,

    // The CPU fallback (`ScatterSettings::gpu == false`).
    fallback_pipeline: wgpu::RenderPipeline,
    fallback_prim: crate::primitives::PrimGpu,
    fallback_instances: Option<wgpu::Buffer>,
    fallback_capacity: usize,
    fallback_ranges: [std::ops::Range<u32>; 5],
    fallback_key: Option<(u64, DVec3, [i64; 3], u32)>,

    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    env: super::EnvBinding,
}

impl ScatterNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        // ── cull compute ──
        let cull_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scatter-cull"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("scatter_cull").into()),
            });
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scatter-cull"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let cull_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scatter-cull"),
                bind_group_layouts: &[Some(&cull_bgl)],
                immediate_size: 0,
            });
        let mk_compute = |entry: &str| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&cull_layout),
                    module: &cull_shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let cull_pipelines = [
            mk_compute("cs_classify"),
            mk_compute("cs_scan"),
            mk_compute("cs_compact"),
        ];

        // ── raster ──
        let raster_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scatter-mesh"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("scatter_mesh").into()),
            });
        let vs_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let raster_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scatter-raster"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // The fragment stage reads material/emissive/band weights
                        // out of the same block the vertex stage reads geometry
                        // bases from, so it is visible to both.
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    vs_storage(1),
                    vs_storage(2),
                    vs_storage(3),
                    vs_storage(4),
                    // P22.1 surface deformation: the window texture (5) and its
                    // uniform (6). VERTEX-only — a scatter instance BENDS, it
                    // does not shade differently, so the fragment stage never
                    // reads either. Both are created once in
                    // `EngineRenderer::new` and never resized, so a batch's
                    // raster bind group (built once, outside the `GenCache` the
                    // cull group needs) can hold them for the renderer's life.
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scatter-lights"),
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
            label: Some("scatter-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });
        let env = super::EnvBinding::new(gpu);

        let raster_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scatter-raster"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&lights_bgl),
                    Some(&env.bgl),
                    Some(&raster_bgl),
                ],
                immediate_size: 0,
            });
        let mk_raster = |label: &str, vs: &str, cull: Option<wgpu::Face>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&raster_layout),
                    vertex: wgpu::VertexState {
                        module: &raster_shader,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &[], // pure vertex pulling
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &raster_shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: SCENE_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        cull_mode: cull,
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
                })
        };
        let mesh_pipeline = mk_raster("scatter-mesh", "vs_mesh", Some(wgpu::Face::Back));
        // A billboard has one winding and is viewed from one side; culling it would
        // make it vanish for half the camera orientations that produce it.
        let impostor_pipeline = mk_raster("scatter-impostor", "vs_impostor", None);

        // ── CPU fallback: the rigid mesh pipeline, unmodified ──
        let fallback_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scatter-fallback"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mesh").into()),
            });
        let fallback_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scatter-fallback"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl), Some(&env.bgl)],
                immediate_size: 0,
            });
        let fallback_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("scatter-fallback"),
                    layout: Some(&fallback_layout),
                    vertex: wgpu::VertexState {
                        module: &fallback_shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &vertex_layouts(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fallback_shader,
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
            cull_pipelines,
            cull_bgl,
            raster_bgl,
            mesh_pipeline,
            impostor_pipeline,
            prim_storage: PrimStorage::new(gpu, "scatter"),
            hzb: HzbChain::new(gpu),
            dummy_hzb: dummy_hzb(gpu),
            uploads: HashMap::new(),
            scratch: HashMap::new(),
            fallback_pipeline,
            fallback_prim: crate::primitives::PrimGpu::new(gpu, "scatter-fallback"),
            fallback_instances: None,
            fallback_capacity: 0,
            fallback_ranges: super::mesh::EMPTY_RANGES,
            fallback_key: None,
            lights_buf,
            lights_bg,
            env,
        }
    }

    /// Upload any payload whose content key is not already resident, give every
    /// live batch its own scratch, and drop what this frame does not reference.
    ///
    /// **Two levels of retention on two different keys**, for the reason
    /// [`BatchScratch`] states: an upload is shared *by content*, a draw is not.
    fn sync_batches(&mut self, gpu: &GpuContext, frame: &FrameData) {
        for b in &frame.scene.scatter {
            if b.data.is_empty() {
                continue;
            }
            let key = b.data.key();
            let upload = self
                .uploads
                .entry(key)
                .or_insert_with(|| {
                    Arc::new(InstanceUpload::new(gpu, b.data.mesh, &b.data.instances))
                })
                .clone();
            self.scratch.entry((key, b.id)).or_insert_with(|| {
                BatchScratch::new(
                    gpu,
                    &self.raster_bgl,
                    frame.deform,
                    &self.prim_storage,
                    upload,
                )
            });
        }
        let live_content: std::collections::BTreeSet<u128> = frame
            .scene
            .scatter
            .iter()
            .filter(|b| !b.data.is_empty())
            .map(|b| b.data.key())
            .collect();
        let live_draws: std::collections::BTreeSet<(u128, u32)> = frame
            .scene
            .scatter
            .iter()
            .filter(|b| !b.data.is_empty())
            .map(|b| (b.data.key(), b.id))
            .collect();
        // Scratch first: it holds an `Arc` into the uploads, so dropping the draws
        // before the payloads keeps the eviction order obvious rather than merely
        // correct-by-refcount.
        self.scratch.retain(|k, _| live_draws.contains(k));
        self.uploads.retain(|k, _| live_content.contains(k));
        frame
            .scatter_audit
            .record_uploads(self.uploads.len() as u32);
    }
}

/// The per-frame cull flag bitset — mirrors `FLAG_*` in `scatter_cull.wgsl`.
fn cull_flags(settings: &crate::settings::ScatterSettings) -> u32 {
    let mut f = 0;
    if settings.frustum_cull {
        f |= FLAG_FRUSTUM;
    }
    if settings.occlusion {
        f |= FLAG_OCCLUSION;
    }
    f
}

/// How many scattered instances the CPU paths will pack in one go.
///
/// A ceiling, not a quality knob: the distance clamps already bound the set
/// geometrically, and this is what stops a pathological scene (a million-instance
/// batch with the cull band wound open) from turning into a 176 MB `InstanceRaw`
/// upload on a machine that has already been judged unable to run the compute
/// path. Overflow degrades **nearest-first**, so what is lost is the far field —
/// the P18.4 `priority_order` precedent, and the same total order:
/// `f64::total_cmp` on the squared distance, tie-broken by pack index, so a
/// degenerate transform still yields *an* order.
pub const MAX_CPU_SCATTER_INSTANCES: usize = 65_536;

/// How far past the shadow range a scatter caster is still packed (P18.5). A
/// caster *outside* the last cascade can still cast *into* it when the sun is low,
/// so the clip is deliberately generous; 1.5x costs a handful of instances and
/// removes the class of bug where grass shadows stop at a circle around the camera.
pub const SHADOW_CASTER_MARGIN: f32 = 1.5;

/// What one [`pack_fallback`] produced: the bucketed instances, their per-kind
/// ranges, how many passed the distance clamp, and whether the ceiling bit.
pub struct ScatterPack {
    pub instances: Vec<InstanceRaw>,
    pub ranges: [std::ops::Range<u32>; 5],
    /// Instances that passed the distance clamp — what would have been packed with
    /// no ceiling.
    pub considered: usize,
    /// Whether the caller's `limit` truncated the set.
    pub clamped: bool,
}

/// Pack a scene's scatter batches into `InstanceRaw`s — for the CPU fallback and
/// for the shadow caster set — bucketed by primitive kind exactly as
/// `mesh::pack_bucketed` does.
///
/// Pure: a function of `(origin, batches, eye, settings, limit)` alone. The caller
/// snaps `eye` onto [`FALLBACK_EYE_BUCKET_M`] first and keys its cache on the
/// snapped value, which is what stops a walking camera from re-packing every
/// frame; the cull radius is widened by the bucket's own half-diagonal so a snapped
/// eye can only ever keep MORE instances than the true one would, never fewer.
///
/// **A band of zero packs nothing, and that is a fix rather than a convention.**
/// The first cut read a zero cull distance as "unlimited", which disagreed with
/// `scatter_cull.wgsl` — the shader's `d >= params.bands.y` culls *everything* at
/// `bands.y == 0`. So a host that wound the band to zero got an empty frame on the
/// GPU path and every instance in the world packed on the CPU one. The shader's
/// reading wins: a cull distance of zero means nothing draws.
pub fn pack_fallback(
    origin: &FloatingOrigin,
    batches: &[ScatterBatch],
    eye_world: DVec3,
    settings: &crate::settings::ScatterSettings,
    limit: usize,
) -> ScatterPack {
    let slack = FALLBACK_EYE_BUCKET_M * 0.5 * 3f64.sqrt();
    // (distance^2, kind, raw) — the distance rides along so an over-limit pack can
    // degrade nearest-first without recomputing it.
    let mut kept: Vec<(f64, usize, InstanceRaw)> = Vec::new();
    for b in batches {
        let (mesh_end, cull, _, _) = effective_bands(settings, b.draw_distance);
        // Neither CPU consumer draws impostors, so the band ends where the mesh
        // band does whenever impostors are notionally on.
        let band = if settings.impostors { mesh_end } else { cull };
        if band <= 0.0 {
            continue;
        }
        let far = band as f64 + slack;
        let kind = b.data.mesh.index();
        for inst in &b.data.instances {
            let world = b.anchor
                + DVec3::new(
                    inst.offset[0] as f64,
                    inst.offset[1] as f64,
                    inst.offset[2] as f64,
                );
            let d2 = (world - eye_world).length_squared();
            if d2 > far * far {
                continue;
            }
            let rot = glam::Quat::from_array(inst.rotation);
            let model = origin.model_matrix(world, rot, Vec3::splat(inst.scale));
            let n = (Mat3::from_quat(rot)
                * Mat3::from_diagonal(Vec3::splat(1.0 / inst.scale.abs().max(1e-6))))
            .to_cols_array_2d();
            kept.push((
                d2,
                kind,
                InstanceRaw {
                    model: model.to_cols_array(),
                    normal_mat: [
                        n[0][0], n[0][1], n[0][2], 0.0, //
                        n[1][0], n[1][1], n[1][2], 0.0, //
                        n[2][0], n[2][1], n[2][2], 0.0,
                    ],
                    color: inst.color,
                    misc: [b.id, 0, 0, 0],
                    pbr: [b.metallic, b.roughness, 0.5, 0.0],
                    emissive: [b.emissive[0], b.emissive[1], b.emissive[2], 0.0],
                },
            ));
        }
    }

    let considered = kept.len();
    let clamped = considered > limit;
    if clamped {
        // Nearest-first, deterministically — and only when the ceiling bites, so
        // the ordinary path stays an O(n) walk.
        let mut order: Vec<usize> = (0..kept.len()).collect();
        order.sort_by(|&a, &b| kept[a].0.total_cmp(&kept[b].0).then(a.cmp(&b)));
        order.truncate(limit);
        // Back into pack order, so the surviving set is bucketed in the same
        // authored sequence a sub-limit pack would have produced.
        order.sort_unstable();
        kept = order.into_iter().map(|i| kept[i]).collect();
    }

    let mut buckets: [Vec<InstanceRaw>; 5] = Default::default();
    for (_, kind, raw) in kept {
        buckets[kind].push(raw);
    }
    let mut instances = Vec::new();
    let ranges = std::array::from_fn(|k| {
        let start = instances.len() as u32;
        instances.append(&mut buckets[k]);
        start..instances.len() as u32
    });
    ScatterPack {
        instances,
        ranges,
        considered,
        clamped,
    }
}

/// The scatter settings the **shadow** caster pack runs under (P18.5).
///
/// The first cut synthesized these by *overwriting* `cull_distance_m`, which made
/// the shadow path escape every clamp the renderer has: the tier's band ceilings,
/// `clamp_scatter` and `mesh_distance_m` all became inert, so a Medium-tier
/// machine that had just been told to draw 240 m of foliage still rasterized full
/// primitive meshes for 600 m of it into three cascades. Every clamp here is a
/// `min` against the settings the host already handed the renderer.
///
/// Two rules beyond that:
///
/// * **Only the full-mesh band casts.** An impostor is a camera-facing card; from
///   the sun's point of view it is a sliver or a disc depending on the angle and
///   never the object's silhouette, so rasterizing one into a shadow map is
///   geometrically wrong rather than merely approximate — and casting the *full
///   mesh* for something the camera draws as a disc is precisely the cost the LOD
///   band exists to avoid. The caster band is therefore exactly
///   **`min(mesh_distance_m, cull_distance_m, shadow range × SHADOW_CASTER_MARGIN)`**,
///   and pulling `mesh_distance_m` in below the shadow range legitimately stops the
///   far half of a field casting: a bounded softening the tier explicitly asked for.
///   Note the third clamp in the body — turning impostors off makes
///   [`pack_fallback`] read `cull_distance_m` as the band, so the mesh clamp has to
///   be pushed onto `cull_distance_m` or it does nothing at all.
/// * **A zero shadow range casts nothing.** `ShadowSettings::max_distance == 0` has
///   no cascade to receive anything, and with the sentinel fixed in
///   [`pack_fallback`] it now means what it says instead of packing the world.
pub fn shadow_caster_settings(
    scatter: &crate::settings::ScatterSettings,
    shadow_max_distance: f32,
) -> crate::settings::ScatterSettings {
    let range = shadow_max_distance.max(0.0) * SHADOW_CASTER_MARGIN;
    let mut s = *scatter;
    s.impostors = false;
    s.mesh_distance_m = s.mesh_distance_m.min(range);
    s.cull_distance_m = s.cull_distance_m.min(range);
    // …and the CULL band collapses onto the mesh band, which is the line that makes
    // the rule above true instead of merely documented. `impostors = false` routes
    // `pack_fallback`'s band to `cull_distance_m` (there is no impostor band to end
    // the mesh band at), so clamping `mesh_distance_m` alone was **dead code**: at
    // any shadow range past `mesh_distance_m / SHADOW_CASTER_MARGIN` — 80 m at the
    // defaults — full primitive meshes rasterized into all three cascades out to the
    // cull distance, 2.5× the intended radius and 6.25× the area at 200 m.
    s.cull_distance_m = s.cull_distance_m.min(s.mesh_distance_m);
    s
}

/// Merge two bucketed instance packs into one, keeping the result bucketed by
/// primitive kind so a single [`PrimGpu::draw`] still issues at most five calls.
///
/// Used by the shadow pass, which has to raster the rigid instances **and** the
/// scatter into the same cascade. Concatenating the two `Vec`s would not do: each
/// pack's ranges are contiguous per kind, and appending would interleave them.
///
/// [`PrimGpu::draw`]: crate::primitives::PrimGpu::draw
pub fn merge_bucketed(
    a: (Vec<InstanceRaw>, [std::ops::Range<u32>; 5]),
    b: (Vec<InstanceRaw>, [std::ops::Range<u32>; 5]),
) -> (Vec<InstanceRaw>, [std::ops::Range<u32>; 5]) {
    let (av, ar) = a;
    let (bv, br) = b;
    if bv.is_empty() {
        return (av, ar);
    }
    let mut out = Vec::with_capacity(av.len() + bv.len());
    let ranges = std::array::from_fn(|k| {
        let start = out.len() as u32;
        out.extend_from_slice(&av[ar[k].start as usize..ar[k].end as usize]);
        out.extend_from_slice(&bv[br[k].start as usize..br[k].end as usize]);
        start..out.len() as u32
    });
    (out, ranges)
}

/// Snap a world position onto the fallback's re-pack lattice.
pub fn eye_bucket(eye: DVec3) -> [i64; 3] {
    [
        (eye.x / FALLBACK_EYE_BUCKET_M).round() as i64,
        (eye.y / FALLBACK_EYE_BUCKET_M).round() as i64,
        (eye.z / FALLBACK_EYE_BUCKET_M).round() as i64,
    ]
}

/// The world position at the centre of a re-pack lattice cell.
pub fn bucket_center(eye: DVec3) -> DVec3 {
    bucket_to_world(eye_bucket(eye))
}

fn bucket_to_world(b: [i64; 3]) -> DVec3 {
    DVec3::new(
        b[0] as f64 * FALLBACK_EYE_BUCKET_M,
        b[1] as f64 * FALLBACK_EYE_BUCKET_M,
        b[2] as f64 * FALLBACK_EYE_BUCKET_M,
    )
}

impl ScatterNode {
    fn run_fallback(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameData,
    ) {
        let eye_world = frame.view.origin.to_world(frame.view.eye_local());
        let bucket = eye_bucket(eye_world);
        let settings_stamp = fallback_settings_stamp(&frame.settings.scatter);
        let key = (
            frame.scene.version,
            frame.view.origin.origin(),
            bucket,
            settings_stamp,
        );
        if self.fallback_key != Some(key) {
            let pack = pack_fallback(
                &frame.view.origin,
                &frame.scene.scatter,
                bucket_to_world(bucket),
                &frame.settings.scatter,
                MAX_CPU_SCATTER_INSTANCES,
            );
            if pack.clamped {
                tracing::warn!(
                    "inf-render: CPU scatter fallback clamped to {} of {} instances \
                     inside the band — distant instances stop drawing (nearest-first)",
                    MAX_CPU_SCATTER_INSTANCES,
                    pack.considered,
                );
            }
            let raw = pack.instances;
            self.fallback_ranges = pack.ranges;
            if !raw.is_empty() {
                if self.fallback_instances.is_none() || self.fallback_capacity < raw.len() {
                    let capacity = raw.len().next_power_of_two().max(64);
                    self.fallback_instances =
                        Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("scatter-fallback-instances"),
                            size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    self.fallback_capacity = capacity;
                }
                gpu.queue.write_buffer(
                    self.fallback_instances.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&raw),
                );
            }
            self.fallback_key = Some(key);
        }
        let total: u32 = self.fallback_ranges.iter().map(|r| r.end - r.start).sum();
        let Some(instances) = self.fallback_instances.as_ref() else {
            return;
        };
        if total == 0 {
            return;
        }
        gpu.queue.write_buffer(
            &self.lights_buf,
            0,
            bytemuck::bytes_of(&LightsUniform::from_scene(
                frame.scene,
                &frame.view.origin,
                frame.vsm_light_slots,
            )),
        );
        let env_bg = self.env.bind_group(gpu, frame).clone();
        let mut pass = scene_pass(encoder, frame, "scatter-fallback");
        pass.set_pipeline(&self.fallback_pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &self.lights_bg, &[]);
        pass.set_bind_group(2, &env_bg, &[]);
        self.fallback_prim
            .draw(&mut pass, instances, &self.fallback_ranges);
    }
}

/// A stamp over the fallback-relevant settings, so a tier change re-packs.
fn fallback_settings_stamp(s: &crate::settings::ScatterSettings) -> u32 {
    s.cull_distance_m.to_bits()
}

fn scene_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    frame: &'e FrameData,
    label: &'static str,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    })
}

impl RenderNode for ScatterNode {
    fn name(&self) -> &'static str {
        "scatter"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let live = frame.scene.scatter.iter().any(|b| !b.data.is_empty());
        if !live {
            // Release everything the previous frame held, *before* the early-out,
            // so switching levels or turning scatter off frees its VRAM instead of
            // parking it until device-lost (the P18.3 `ClassicVgeomNode` lesson).
            self.uploads.clear();
            self.scratch.clear();
            self.fallback_key = None;
            // Hardening D: the HZB is a full-resolution mip pyramid (~44 MiB at
            // 4K) and it was the one thing this release path did not name.
            self.hzb.release();
            return;
        }
        if !frame.settings.scatter.gpu {
            self.uploads.clear();
            self.scratch.clear();
            self.run_fallback(gpu, encoder, frame);
            return;
        }
        self.fallback_key = None;
        self.fallback_instances = None;
        self.fallback_capacity = 0;

        self.sync_batches(gpu, frame);

        let audit = frame.scatter_audit.enabled;
        if audit {
            gpu.queue.write_buffer(
                &frame.scatter_audit.stats,
                0,
                bytemuck::cast_slice(&[0u32; AUDIT_SLOTS]),
            );
        }

        let occlusion = frame.settings.scatter.occlusion;
        if occlusion {
            self.hzb.build(gpu, encoder, frame);
        } else {
            // Nothing samples the pyramid this frame; see `HzbChain::release`.
            self.hzb.release();
        }
        let hzb_dims = if occlusion {
            self.hzb
                .dims()
                .map(|(w, h, m)| [m as f32, w as f32, h as f32, 0.0])
                .unwrap_or([1.0, 1.0, 1.0, 0.0])
        } else {
            [1.0, 1.0, 1.0, 0.0]
        };
        let hzb_view = if occlusion {
            self.hzb.full_view().unwrap_or(&self.dummy_hzb)
        } else {
            &self.dummy_hzb
        };
        let hzb_key: HzbKey = (frame.targets.generation, self.hzb.generation());

        let vp = frame.view.view_proj();
        let planes = frustum_planes(vp);
        let eye = frame.view.eye_local();

        gpu.queue.write_buffer(
            &self.lights_buf,
            0,
            bytemuck::bytes_of(&LightsUniform::from_scene(
                frame.scene,
                &frame.view.origin,
                frame.vsm_light_slots,
            )),
        );

        // ── 1. per-batch uniforms + the three cull dispatches ──
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter-cull"),
                timestamp_writes: None,
            });
            for b in &frame.scene.scatter {
                if b.data.is_empty() {
                    continue;
                }
                let Some(g) = self.scratch.get_mut(&(b.data.key(), b.id)) else {
                    continue;
                };
                let anchor = frame.view.origin.to_render(b.anchor);
                let radius = b.data.mesh.bounding_radius() * b.data.max_scale();
                let (mesh_end, cull, fade, impostors) =
                    effective_bands(&frame.settings.scatter, b.draw_distance);
                let range = self.prim_storage.range(g.upload.mesh);
                gpu.queue.write_buffer(
                    &g.cull_params,
                    0,
                    bytemuck::bytes_of(&CullParamsGpu {
                        view_proj: vp.to_cols_array(),
                        frustum: planes.map(|p| p.to_array()),
                        eye: eye.extend(0.0).to_array(),
                        anchor: anchor.extend(radius).to_array(),
                        counts: [
                            g.upload.count,
                            cull_flags(&frame.settings.scatter),
                            g.upload.capacity,
                            audit as u32,
                        ],
                        bands: [mesh_end, cull, fade, if impostors { 1.0 } else { 0.0 }],
                        hzb: hzb_dims,
                    }),
                );
                gpu.queue.write_buffer(
                    &g.raster_params,
                    0,
                    bytemuck::bytes_of(&RasterParamsGpu {
                        anchor: anchor.extend(b.data.mesh.bounding_radius()).to_array(),
                        material: [b.metallic, b.roughness, 0.0, 0.0],
                        emissive: [b.emissive[0], b.emissive[1], b.emissive[2], 0.0],
                        bands: [mesh_end, cull, fade, if impostors { 1.0 } else { 0.0 }],
                        geom: [
                            range.index_start,
                            range.base_vertex as u32,
                            b.id,
                            g.upload.capacity,
                        ],
                    }),
                );
                // vertex_count is a property of the primitive kind, so the CPU owns
                // it; the scan pass writes only the two instance counts.
                gpu.queue.write_buffer(
                    &g.args,
                    0,
                    bytemuck::cast_slice(&[range.index_count, 0u32, 0, 0, 6u32, 0, 0, 0]),
                );

                let bg = g.cull_bg.get_or_build(hzb_key, || {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scatter-cull"),
                        layout: &self.cull_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: g.cull_params.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: g.upload.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: g.slots.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: g.partials.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: g.visible.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: g.args.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: frame.scatter_audit.stats.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(hzb_view),
                            },
                        ],
                    })
                });
                pass.set_bind_group(0, bg, &[]);
                let groups = g.upload.count.div_ceil(WORKGROUP).max(1);
                pass.set_pipeline(&self.cull_pipelines[0]);
                pass.dispatch_workgroups(groups, 1, 1);
                pass.set_pipeline(&self.cull_pipelines[1]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.cull_pipelines[2]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }

        // ── 2. the two indirect draws, per batch ──
        let env_bg = self.env.bind_group(gpu, frame).clone();
        {
            let mut pass = scene_pass(encoder, frame, "scatter");
            pass.set_bind_group(0, frame.view_bg, &[]);
            pass.set_bind_group(1, &self.lights_bg, &[]);
            pass.set_bind_group(2, &env_bg, &[]);
            for b in &frame.scene.scatter {
                if b.data.is_empty() {
                    continue;
                }
                let Some(g) = self.scratch.get(&(b.data.key(), b.id)) else {
                    continue;
                };
                pass.set_bind_group(3, &g.raster_bg, &[]);
                pass.set_pipeline(&self.mesh_pipeline);
                pass.draw_indirect(&g.args, 0);
                if frame.settings.scatter.impostors {
                    pass.set_pipeline(&self.impostor_pipeline);
                    pass.draw_indirect(&g.args, 16);
                }
            }
        }

        if audit {
            encoder.copy_buffer_to_buffer(
                &frame.scatter_audit.stats,
                0,
                &frame.scatter_audit.readback,
                0,
                AUDIT_BYTES,
            );
        }
    }
}

/// A conservative render-local model matrix for one scattered instance — the CPU
/// twin of `scatter_mesh.wgsl`'s vertex transform, used by the fallback and by the
/// tests.
pub fn instance_model(origin: &FloatingOrigin, anchor: DVec3, inst: &ScatterInstanceRaw) -> Mat4 {
    let world = anchor
        + DVec3::new(
            inst.offset[0] as f64,
            inst.offset[1] as f64,
            inst.offset[2] as f64,
        );
    origin.model_matrix(
        world,
        glam::Quat::from_array(inst.rotation),
        Vec3::splat(inst.scale),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::ScatterSettings;

    /// The WGSL constants this module mirrors are a wire contract: the workgroup
    /// size the dispatch divides by, the audit slot indices the readback decodes,
    /// and the flag bits the uniform packs. A drift here is a silent miscount, not
    /// a compile error — the `shader_constants_match_the_rust_side` discipline
    /// P18.2 introduced for the meshlet cull.
    #[test]
    fn shader_constants_match_the_rust_side() {
        let src = include_str!("../shaders/scatter_cull.wgsl");
        for (name, value) in [
            ("WG", WORKGROUP),
            ("FLAG_FRUSTUM", FLAG_FRUSTUM),
            ("FLAG_OCCLUSION", FLAG_OCCLUSION),
            ("AUDIT_CANDIDATES", 0),
            ("AUDIT_FRUSTUM", 1),
            ("AUDIT_OCCLUDED", 2),
            ("AUDIT_DISTANCE", 3),
            ("AUDIT_MESH", 4),
            ("AUDIT_IMPOSTOR", 5),
        ] {
            let want = format!("const {name}: u32 = {value}u;");
            assert!(
                src.contains(&want),
                "scatter_cull.wgsl must declare `{want}`"
            );
        }
        assert!(
            src.contains(&format!("array<atomic<u32>, {AUDIT_SLOTS}>")),
            "the audit buffer must hold {AUDIT_SLOTS} slots"
        );
        // The compaction MUST stay a prefix sum: an `atomicAdd` into the visible
        // list is precisely the regression this batch exists to avoid, and it
        // would still pass every count-based assertion.
        //
        // Checked over the **compaction function bodies**, and against *any* atomic
        // on *any* buffer but the audit counters. Naming two buffers (`draw_args`,
        // `partials`) was the first cut of this guard and it was too narrow in the
        // way guards usually are: the obvious atomic append writes
        // `visible[atomicAdd(&counter, 1u)]`, i.e. touches neither of the two names
        // it was watching. The rule the file actually needs is "only `stats` is
        // atomic", so that is what is asserted.
        for entry in ["cs_classify", "cs_scan", "cs_compact"] {
            // Comments stripped first, or the guard is checking prose — including
            // this module's own header, which quotes the atomic append it forbids.
            let body: String = fn_body(src, entry)
                .lines()
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n");
            for (call, _) in body.match_indices("atomic") {
                let tail = &body[call..];
                assert!(
                    tail.starts_with("atomicAdd(&stats[")
                        || tail.starts_with("atomicLoad(&stats[")
                        || tail.starts_with("atomicStore(&stats["),
                    "{entry} performs an atomic on something other than the audit \
                     counters — the compaction must stay a prefix sum (see the \
                     module header: the dithered cross-fade makes draw order reach \
                     the image). Offending text: {:?}",
                    &tail[..tail.len().min(48)]
                );
            }
            // …and no list may be indexed by an atomic result, however it is spelt.
            for list in ["visible[", "partials[", "slots["] {
                assert!(
                    !body.contains(&format!("{list}atomic")),
                    "{entry} fills `{list}` with an atomic append"
                );
            }
        }
    }

    /// The text of a WGSL entry point's body, for the source guards below. Brace
    /// counting rather than a regex, so a nested block cannot end it early.
    fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
        let sig = format!("fn {name}(");
        let start = src
            .find(&sig)
            .unwrap_or_else(|| panic!("{name} must exist in the shader"));
        let open = start + src[start..].find('{').expect("a body");
        let mut depth = 0usize;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &src[open..open + i];
                    }
                }
                _ => {}
            }
        }
        panic!("{name}'s body is unterminated");
    }

    /// The dither hash is duplicated here in Rust so the WGSL one can be pinned
    /// bit-for-bit without a GPU — the cloud-noise precedent.
    fn wang(px: u32, py: u32) -> f32 {
        let mut h = (px & 0xFFFF) | ((py & 0xFFFF) << 16);
        h = (h ^ 61) ^ (h >> 16);
        h = h.wrapping_add(h << 3);
        h ^= h >> 4;
        h = h.wrapping_mul(0x27d4_eb2d);
        h ^= h >> 15;
        (h >> 8) as f32 / 16_777_216.0
    }

    #[test]
    fn dither_hash_matches_the_rust_side() {
        let src = include_str!("../shaders/scatter_mesh.wgsl");
        for frag in [
            "h = (h ^ 61u) ^ (h >> 16u);",
            "h = h + (h << 3u);",
            "h = h ^ (h >> 4u);",
            "h = h * 0x27d4eb2du;",
            "h = h ^ (h >> 15u);",
            "return f32(h >> 8u) * (1.0 / 16777216.0);",
        ] {
            assert!(
                src.contains(frag),
                "scatter_mesh.wgsl's dither must be the pinned Wang avalanche (missing `{frag}`)"
            );
        }
        // No temporal input anywhere near the fade: a frame index in the hash would
        // make a cross-fade a function of history, which no golden could survive.
        // Checked against the FUNCTION BODY rather than the file, so the header's
        // prose about not jittering cannot satisfy — or trip — its own gate.
        let body = src
            .split_once("fn scatter_dither(px: vec2<f32>) -> f32 {")
            .expect("scatter_dither must exist")
            .1
            .split_once(
                "
}",
            )
            .expect("scatter_dither must close")
            .0;
        for forbidden in ["frame", "time", "jitter", "history"] {
            assert!(
                !body.contains(forbidden),
                "the scatter dither must be a pure function of the pixel (found `{forbidden}`)"
            );
        }
        // Uniform-ish over the screen: a hash that clusters would band the fade.
        let mut buckets = [0u32; 4];
        for y in 0..64u32 {
            for x in 0..64u32 {
                buckets[(wang(x, y) * 4.0) as usize % 4] += 1;
            }
        }
        for b in buckets {
            assert!(
                (700..1400).contains(&b),
                "dither buckets are lopsided: {buckets:?}"
            );
        }
        // The value is strictly BELOW 1.0 for every input, which is the property
        // the 24-bit fold exists for: a full-weight band tests `h < 1.0`, and a
        // hash that can reach exactly 1.0 punches a deterministic, permanently
        // located hole in geometry that is nowhere near a fade. Checked at the
        // saturating input rather than by sampling, because one pixel in 2^32 is
        // not something a sweep would find.
        let mut worst = 0.0f32;
        for y in 0..256u32 {
            for x in 0..256u32 {
                worst = worst.max(wang(x, y));
            }
        }
        assert!(worst < 1.0, "the dither reached {worst}");
        assert!(
            (u32::MAX >> 8) as f32 / 16_777_216.0 < 1.0,
            "the saturating hash value must stay below 1.0"
        );
    }

    #[test]
    fn bands_clamp_down_and_never_up() {
        let s = ScatterSettings::default();
        let (mesh, cull, _, imp) = effective_bands(&s, 0.0);
        assert_eq!(cull, s.cull_distance_m);
        assert_eq!(mesh, s.mesh_distance_m);
        assert!(imp);

        // An authored draw distance only ever pulls the cull IN.
        let (_, near, _, _) = effective_bands(&s, 50.0);
        assert_eq!(near, 50.0);
        let (_, far, _, _) = effective_bands(&s, 10_000.0);
        assert_eq!(
            far, s.cull_distance_m,
            "content must not extend the tier's band"
        );

        // The mesh band can never outrun the cull distance.
        let (mesh_near, cull_near, _, _) = effective_bands(&s, 20.0);
        assert!(mesh_near <= cull_near);

        // With impostors off the mesh band IS the cull band, so the same weight
        // formula fades the mesh out instead of popping it.
        let off = ScatterSettings {
            impostors: false,
            ..s
        };
        let (mesh_off, cull_off, _, imp_off) = effective_bands(&off, 0.0);
        assert!(!imp_off);
        assert_eq!(mesh_off, cull_off);
    }

    #[test]
    fn tier_and_mobile_clamps_only_pull_scatter_in() {
        use crate::caps::RenderTier;
        let base = crate::RenderSettings::default();
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            let c = tier.apply(base);
            assert!(c.scatter.cull_distance_m <= base.scatter.cull_distance_m);
            assert!(c.scatter.mesh_distance_m <= base.scatter.mesh_distance_m);
            assert!(!c.scatter.gpu || base.scatter.gpu);
            assert!(!c.scatter.occlusion || base.scatter.occlusion);
            assert!(!c.scatter.impostors || base.scatter.impostors);
        }
        assert!(RenderTier::High.apply(base).scatter.gpu, "High is a no-op");
        assert!(!RenderTier::Medium.apply(base).scatter.gpu);
        let m = RenderTier::clamp_mobile(base);
        assert!(!m.scatter.gpu && !m.scatter.impostors && !m.scatter.occlusion);
    }

    /// The shadow caster band is a `min` against every knob the host already set —
    /// it may not synthesize its own settings and escape them.
    ///
    /// That is exactly what the first cut did: it *overwrote* `cull_distance_m`
    /// with the shadow range, and since `cull_distance_m` was the only field the
    /// packer read, the tier ceilings, `clamp_scatter` and `mesh_distance_m` all
    /// became inert for shadows. A Medium-tier machine told to draw 240 m of
    /// foliage still rasterized full primitive meshes for 600 m of it into three
    /// cascades.
    #[test]
    fn shadow_casters_cannot_escape_the_clamps() {
        use crate::caps::RenderTier;
        let high = crate::RenderSettings::default().scatter;
        let medium = RenderTier::Medium
            .apply(crate::RenderSettings::default())
            .scatter;

        // A generous shadow range: the SCATTER settings must still bind.
        let s = shadow_caster_settings(&medium, 10_000.0);
        assert!(
            s.mesh_distance_m <= medium.mesh_distance_m
                && s.cull_distance_m <= medium.cull_distance_m,
            "the caster band escaped the tier clamp: {s:?} against {medium:?}"
        );
        assert!(
            s.cull_distance_m < high.cull_distance_m,
            "the Medium ceiling must still be visible in the caster band"
        );

        // A tight shadow range: the RANGE must bind, times the low-sun margin.
        let tight = shadow_caster_settings(&high, 20.0);
        assert_eq!(tight.cull_distance_m, 20.0 * SHADOW_CASTER_MARGIN);
        assert_eq!(tight.mesh_distance_m, 20.0 * SHADOW_CASTER_MARGIN);

        // Impostors never cast: a camera-facing card is not a silhouette from the
        // sun's point of view.
        assert!(!tight.impostors && !s.impostors);

        // Whichever is smaller wins, always — the property, not two examples. The
        // ranges deliberately straddle `mesh_distance_m / SHADOW_CASTER_MARGIN`
        // (80 m at the defaults): below it the range binds, above it the mesh band
        // must, and the earlier sweep only ever sampled below.
        for range in [0.0f32, 1.0, 17.0, 60.0, 79.0, 81.0, 200.0, 400.0, 1e6] {
            for base in [high, medium] {
                let c = shadow_caster_settings(&base, range);
                let want = base
                    .mesh_distance_m
                    .min(base.cull_distance_m)
                    .min(range * SHADOW_CASTER_MARGIN);
                // `pack_fallback` reads `cull_distance_m` as the band when impostors
                // are off, so THAT is the number the rule has to land on.
                assert!(
                    (c.cull_distance_m - want).abs() < 1e-3,
                    "caster band {} != min(mesh {}, cull {}, range×margin {}) at \
                     range {range}",
                    c.cull_distance_m,
                    base.mesh_distance_m,
                    base.cull_distance_m,
                    range * SHADOW_CASTER_MARGIN
                );
                assert!(c.mesh_distance_m <= base.mesh_distance_m);
                assert!(c.cull_distance_m <= base.cull_distance_m);
            }
        }
    }

    /// **The caster band really stops at `mesh_distance_m`** — asserted on the
    /// *packed set*, at a shadow range comfortably ABOVE it.
    ///
    /// The settings-level sweep above cannot catch this on its own: turning
    /// impostors off makes `effective_bands` report `mesh_end == cull`, so a
    /// `shadow_caster_settings` that clamped only `mesh_distance_m` produced settings
    /// that *read* correct and packed to the cull distance anyway. Nothing crossed
    /// that line until this test — at the defaults it takes an 80 m shadow range to
    /// reach it, and every other fixture here sits at 60 m or below.
    #[test]
    fn the_packed_caster_band_stops_at_the_mesh_distance() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        // 400 instances at 1 m spacing along +X, eye at the origin.
        let batches = [line_batch(400, 1.0)];
        let base = crate::RenderSettings::default().scatter;
        // 200 m of shadow range → 300 m of margin-widened range, well past the
        // 120 m mesh band. Before the fix this packed to 300 m.
        let s = shadow_caster_settings(&base, 200.0);
        assert!(
            300.0 > base.mesh_distance_m,
            "the fixture must cross the mesh_distance line, or it proves nothing"
        );

        let pack = pack_fallback(
            &origin,
            &batches,
            DVec3::ZERO,
            &s,
            MAX_CPU_SCATTER_INSTANCES,
        );
        // The band plus the eye-lattice slack, and nothing beyond it.
        let slack = FALLBACK_EYE_BUCKET_M * 0.5 * 3f64.sqrt();
        let ceiling = (base.mesh_distance_m as f64 + slack).ceil() as usize + 1;
        assert!(
            pack.instances.len() <= ceiling,
            "the caster band ran past mesh_distance_m: {} instances packed, at most \
             {ceiling} fit inside {} m",
            pack.instances.len(),
            base.mesh_distance_m
        );
        // …and it is not vacuously empty: the whole mesh band is still packed.
        assert!(
            pack.instances.len() >= base.mesh_distance_m as usize,
            "the caster band is shorter than mesh_distance_m: {} instances",
            pack.instances.len()
        );
    }

    /// A batch to pack against, at a known spacing so distance claims are exact.
    fn line_batch(n: usize, spacing: f64) -> ScatterBatch {
        let insts = (0..n).map(|i| crate::scene::ScatterInstance {
            position: DVec3::new(i as f64 * spacing, 0.0, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: 1.0,
            color: [1.0; 4],
        });
        crate::scene::ScatterBatch::lit(
            std::sync::Arc::new(crate::scene::ScatterData::build(
                PrimMesh::Cube,
                DVec3::ZERO,
                insts,
            )),
            DVec3::ZERO,
            0.8,
            1,
        )
    }

    /// A **zero** shadow range packs no casters, and a zero band packs nothing at
    /// all — the sentinel the first cut got backwards.
    ///
    /// `pack_fallback` read `cull <= 0` as "unlimited", which disagreed with
    /// `scatter_cull.wgsl` (whose `d >= bands.y` culls everything at `bands.y == 0`).
    /// So `shadows.max_distance = 0` — a legal setting with no cascade to receive
    /// anything — packed **every instance in the world** as a shadow caster.
    #[test]
    fn a_zero_band_packs_nothing() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let batches = [line_batch(4_000, 1.0)];
        let base = crate::RenderSettings::default().scatter;

        let none = shadow_caster_settings(&base, 0.0);
        assert_eq!(none.cull_distance_m, 0.0);
        let pack = pack_fallback(
            &origin,
            &batches,
            DVec3::ZERO,
            &none,
            MAX_CPU_SCATTER_INSTANCES,
        );
        assert_eq!(
            pack.instances.len(),
            0,
            "a zero shadow range must cast nothing, not everything"
        );
        assert_eq!(pack.considered, 0);
        assert!(!pack.clamped);

        // Anti-vacuity: the same fixture at a real range does pack.
        let real = shadow_caster_settings(&base, 200.0);
        let pack = pack_fallback(
            &origin,
            &batches,
            DVec3::ZERO,
            &real,
            MAX_CPU_SCATTER_INSTANCES,
        );
        assert!(
            pack.instances.len() > 100,
            "the fixture packs nothing at all"
        );
    }

    /// The pack ceiling is real, degrades **nearest-first**, and is deterministic.
    #[test]
    fn the_pack_ceiling_degrades_nearest_first() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        // 4000 instances on a 1 m line, band wide enough to keep them all.
        let batches = [line_batch(4_000, 1.0)];
        let mut s = crate::RenderSettings::default().scatter;
        s.impostors = false;
        s.cull_distance_m = 10_000.0;

        let all = pack_fallback(&origin, &batches, DVec3::ZERO, &s, usize::MAX);
        assert_eq!(all.instances.len(), 4_000);
        assert!(!all.clamped);

        let capped = pack_fallback(&origin, &batches, DVec3::ZERO, &s, 100);
        assert!(capped.clamped);
        assert_eq!(capped.instances.len(), 100);
        assert_eq!(capped.considered, 4_000);
        // Nearest-first: with the eye at the origin and instances marching along
        // +X, the survivors are exactly the first hundred — and they arrive in the
        // same order a sub-limit pack would have produced them.
        // `InstanceRaw` is `Pod` but not `Debug`/`PartialEq`, so the comparison is
        // over its bytes — which is the stronger claim anyway.
        let bytes = |v: &[InstanceRaw]| bytemuck::cast_slice::<_, u8>(v).to_vec();
        assert_eq!(
            bytes(&capped.instances),
            bytes(&all.instances[..100]),
            "the ceiling did not keep the NEAREST hundred in pack order"
        );
        // Deterministic across calls.
        let again = pack_fallback(&origin, &batches, DVec3::ZERO, &s, 100);
        assert_eq!(bytes(&capped.instances), bytes(&again.instances));
        assert_eq!(capped.ranges, again.ranges);
    }

    #[test]
    fn eye_bucket_is_a_lattice_and_the_fallback_cull_is_conservative() {
        // Two eyes inside the same 8 m cell snap to one key ⇒ one re-pack.
        assert_eq!(
            eye_bucket(DVec3::new(1.0, 0.0, 0.0)),
            eye_bucket(DVec3::new(3.0, 0.0, 0.0))
        );
        assert_ne!(
            eye_bucket(DVec3::new(1.0, 0.0, 0.0)),
            eye_bucket(DVec3::new(9.0, 0.0, 0.0))
        );
        // The snapped eye is never more than half a cell diagonal from the true
        // one, which is exactly the slack the pack widens its cull radius by.
        let e = DVec3::new(3.9, -3.9, 3.9);
        let snapped = bucket_to_world(eye_bucket(e));
        assert!((e - snapped).length() <= FALLBACK_EYE_BUCKET_M * 0.5 * 3f64.sqrt() + 1e-9);
    }
}
