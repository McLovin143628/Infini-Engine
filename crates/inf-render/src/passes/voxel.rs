//! Voxel-surface (volumetric terrain) forward pass — P21.1.
//!
//! Draws the meshed isosurface of the scene's [`RenderVoxelVolume`]s: caves,
//! excavations and overhangs that **locally extend** the heightfield rather than
//! replacing it. The renderer never sees an SDF — the projector meshes each chunk
//! and hands over triangle soup ([`RenderVoxelVertex`] + a `u32` index list), the
//! same arrangement [`RenderTerrain`](crate::scene::RenderTerrain) has with the
//! paged heightfield.
//!
//! # What it draws with
//!
//! One cached GPU vertex/index buffer pair **per chunk**, gated by that chunk's own
//! monotone change stamp (the [`plan_chunk_cache`] planner below, modelled on the
//! terrain pass's [`plan_tile_cache`](crate::passes::terrain::plan_tile_cache)), and
//! one shared per-chunk instance buffer rebuilt each frame carrying the chunk's
//! floating-origin model matrix plus its volume's four-layer splat palette. Shading
//! is `shaders/voxel.wgsl` — Cook-Torrance GGX + a hemispheric ambient, lit by the
//! shared `@group(1)` lights uniform and **no env group at all**. That shader's
//! header comment is the ledger of what P21.1 deliberately does not do (seam
//! blending, shadows, GI, the depth prepass); read it before adding anything here.
//!
//! # No-op on scenes without volumes
//!
//! [`RenderNode::run`] returns before touching the encoder — before any buffer
//! write — when `scene.voxels` is empty, which is the byte-stability guarantee for
//! the 47 pre-P21.1 goldens.
//!
//! That claim is asserted with [`VoxelReport`] — the node's own engagement
//! counter — and **not** with a pixel comparison, because a pixel comparison
//! cannot make it. Two renders of a scene that has no volumes are identical
//! whether the node returned early or opened a render pass, bound a pipeline and
//! drew nothing; the first cut of this gate compared exactly that and would have
//! passed on a node with no early-out at all. The house precedent is
//! `UnderwaterReport` (P20.3).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Mat4;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::LightsUniform;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{RenderVoxelChunk, RenderVoxelVertex, RenderVoxelVolume, VoxelChunkKey};

/// How many frames the voxel pass has **engaged** on — i.e. actually recorded a
/// render pass and at least one draw.
///
/// The house off-path instrument (`UnderwaterReport`, `SharedStreamReport`, the
/// vgeom/scatter audits). It exists because "the node contributed nothing to the
/// command stream" is a statement *about the command stream*, and pixels cannot
/// make it: a pass that engaged and drew zero triangles — or drew them and had
/// them all depth-rejected — is byte-identical from outside. Bumped at the point
/// in [`RenderNode::run`] past which the encoder *will* be touched, so an
/// unchanged count is exactly the property the module docs claim. One relaxed
/// increment on the frames that engage and nothing at all on the frames that do
/// not, so the off path stays as free as it says it is.
pub type VoxelReport = std::sync::Arc<AtomicU64>;

// ── GPU chunk cache planning ────────────────────────────────────────────────

/// What a cached GPU chunk currently holds: the chunk stamp its buffers were
/// written from, and the index count they were sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedChunk {
    /// The [`RenderVoxelChunk::version`] the cached triangles reflect.
    pub version: u64,
    /// The index count the cached buffer pair was written with.
    pub index_count: u32,
}

/// Identity of one cached chunk **across the scene's volumes**: the owning
/// [`RenderVoxelVolume::id`] plus the chunk's own key.
///
/// Two volumes routinely share chunk coordinates — each grid is anchored in its
/// own entity's frame — so the key carries a volume half from the start, which is
/// the lesson the terrain cache's [`TileCacheKey`](crate::passes::terrain::TileCacheKey)
/// learned the expensive way in P16.6. Ordering is `(id, key)`, so a plan is
/// grouped by volume and ascending within each.
pub type ChunkCacheKey = (u64, VoxelChunkKey);

/// What one chunk-cache sync must do: which chunks to (re)upload, which cached
/// entries to drop. Both lists are in a deterministic order — `upload` follows the
/// projection's (volume, chunk) walk, `evict` is ascending by [`ChunkCacheKey`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkCachePlan {
    /// Chunks whose GPU copy is stale (or absent) and must be written.
    pub upload: Vec<ChunkCacheKey>,
    /// Cached chunks that left the projection (or went malformed) and must be
    /// dropped, freeing their buffers.
    pub evict: Vec<ChunkCacheKey>,
}

/// Whether a chunk's triangle soup is well-formed enough to become geometry: a
/// non-empty index list, a whole number of triangles, and every index inside
/// `vertices`.
///
/// The max index is computed over the (already known non-empty) list rather than
/// assumed from its length — a mesher that emits a sparse index list is perfectly
/// legal, and `indices.len()` says nothing about the largest value in it.
fn well_formed(chunk: &RenderVoxelChunk) -> bool {
    if chunk.indices.is_empty() || !chunk.indices.len().is_multiple_of(3) {
        return false;
    }
    let max = chunk.indices.iter().copied().max().unwrap_or(0);
    (max as usize) < chunk.vertices.len()
}

/// Plan the per-chunk GPU buffer cache against a projection — a pure function, so
/// the upload/eviction gate unit-tests without a GPU.
///
/// A chunk is uploaded when its cached stamp differs from its projected stamp, the
/// cached buffers were written at a different index count, or it is not cached at
/// all. A projected stamp of `0` means "no stamp" and never counts as a hit, so a
/// source that cannot version its chunks degrades to *re-uploading* rather than to
/// a stale frame — the same conservative choice
/// [`plan_tile_cache`](crate::passes::terrain::plan_tile_cache) makes, and for the
/// same reason: a wrong cave is worse than a re-uploaded one.
///
/// The index count is part of the identity as well as the stamp because it is the
/// number the draw call is issued with: a cached pair whose count no longer matches
/// the projection would either read past the uploaded triangles or silently drop
/// some, and neither is distinguishable from correct output by looking at the
/// stamp alone.
///
/// **Malformed chunks are neither uploaded nor kept.** A chunk with no indices, an
/// index count that is not a multiple of 3, or an index past the end of its own
/// vertex list is skipped entirely — so it is not uploaded, and (because it never
/// enters the live set) a previously-cached copy of it is *evicted*. A bad page
/// never becomes silent geometry, and a page that goes bad does not keep drawing
/// the last good version of itself.
///
/// `volumes` is the scene's whole list, walked in order, so a scene mixing several
/// volumes plans all of them and a volume leaving the scene evicts exactly its own
/// chunks.
///
/// `state` projects the cache's value type down to the identity the planner reasons
/// about, so the node can hold GPU resources in the same map the test fills with
/// plain [`CachedChunk`]s.
pub fn plan_chunk_cache<T>(
    cached: &BTreeMap<ChunkCacheKey, T>,
    state: impl Fn(&T) -> CachedChunk,
    volumes: &[RenderVoxelVolume],
) -> ChunkCachePlan {
    let mut upload = Vec::new();
    let mut live: BTreeSet<ChunkCacheKey> = BTreeSet::new();
    for volume in volumes {
        for chunk in &volume.chunks {
            if !well_formed(chunk) {
                continue; // malformed — skip rather than upload garbage
            }
            let key = (volume.id, chunk.key);
            live.insert(key);
            let fresh = chunk.version != 0
                && cached.get(&key).map(&state)
                    == Some(CachedChunk {
                        version: chunk.version,
                        index_count: chunk.indices.len() as u32,
                    });
            if !fresh {
                upload.push(key);
            }
        }
    }
    let evict = cached
        .keys()
        .copied()
        .filter(|k| !live.contains(k))
        .collect();
    ChunkCachePlan { upload, evict }
}

// ── GPU node ────────────────────────────────────────────────────────────────

/// One chunk's cached GPU geometry plus the [`CachedChunk`] identity the upload
/// gate compares against.
struct GpuChunk {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    version: u64,
    /// The chunk-local `(min, max)` AABB the cached copy was uploaded with —
    /// readable through [`VoxelNode::cached_bounds`]. P21.1 draws every resident
    /// chunk; this is the bound the P21.2 frustum cull tests, cached here because
    /// the projection that carries it is gone by the time the cull runs.
    bounds: ([f32; 3], [f32; 3]),
}

/// Per-chunk GPU instance data. Matches the `@location(3..=12)` attributes in
/// voxel.wgsl.
///
/// There is deliberately **no normal matrix**: chunks are axis-aligned and
/// unrotated, so the model matrix is a pure translation and a normal survives it
/// unchanged. (The mesh pass's [`InstanceRaw`](crate::passes::mesh::InstanceRaw)
/// carries one because its instances rotate and scale; here it would be a constant
/// identity uploaded once per chunk per frame.)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkInstanceRaw {
    /// Chunk origin as a render-local translation matrix.
    model: [f32; 16],
    /// The volume's four layer albedos.
    albedo: [[f32; 4]; 4],
    /// The four layers' perceptual roughness in x,y,z,w.
    rough: [f32; 4],
    /// x = pick id.
    misc: [u32; 4],
    /// P21.2 seam: x = blend-band width in metres (`0` disables). Per chunk
    /// rather than per vertex because a band is a property of the volume, and
    /// paying four bytes per vertex for a number every vertex of a volume agrees
    /// on would be a strange way to spend bandwidth.
    seam_params: [f32; 4],
}

impl ChunkInstanceRaw {
    /// Pack one chunk of `volume` against the frame's floating origin.
    ///
    /// The model matrix is a plain translation of the chunk's `f64` world anchor
    /// into render-local f32 — architecture rule 3 at the seam the DTO documents.
    /// A chunk's vertices are already chunk-local metres, so this is all the
    /// transform there is.
    fn pack(origin: &FloatingOrigin, volume: &RenderVoxelVolume, chunk: &RenderVoxelChunk) -> Self {
        let local = (chunk.origin - origin.origin()).as_vec3();
        let mut albedo = [[0.0f32; 4]; 4];
        let mut rough = [0.0f32; 4];
        for (k, layer) in volume.layers.iter().enumerate() {
            albedo[k] = layer.albedo;
            rough[k] = layer.roughness;
        }
        Self {
            model: Mat4::from_translation(local).to_cols_array(),
            albedo,
            rough,
            // Voxel surfaces are not pickable in P21.1: the ID-buffer pass does not
            // draw them, so every chunk writes the reserved "nothing" id. Wiring
            // the pick pass (and the click→chunk→brush path that wants it) is a
            // P21.2 follow-up.
            misc: [0; 4],
            seam_params: [volume.seam_band_m.max(0.0), 0.0, 0.0, 0.0],
        }
    }
}

const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 11] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 80,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 96,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 128,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 144,
        shader_location: 12,
    },
    // P21.2 seam band (metres) — location 15, past the per-vertex seam pair.
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 160,
        shader_location: 15,
    },
];

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 24,
        shader_location: 2,
    },
    // P21.2 seam, per vertex: the heightfield normal + surface height, then the
    // terrain's splat-blended albedo + roughness. Locations 13 and 14 leave the
    // 3..12 block whole for the instance stream, which is what keeps the two
    // attribute tables independently extendable.
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 28,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 44,
        shader_location: 14,
    },
];

/// The two vertex buffers the pipeline takes: the per-chunk geometry (slot 0,
/// stepping per vertex) and the shared per-chunk instance data (slot 1, stepping
/// per instance).
fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RenderVoxelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ChunkInstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
    ]
}

/// The voxel-surface render node (P21.1).
pub struct VoxelNode {
    pipeline: wgpu::RenderPipeline,
    /// Per-chunk vertex/index buffers, keyed by [`ChunkCacheKey`] so two volumes
    /// sharing a chunk coordinate cache side by side. `BTreeMap` ⇒ deterministic
    /// iteration.
    chunks: BTreeMap<ChunkCacheKey, GpuChunk>,
    /// One shared instance buffer rebuilt each frame (chunks are cheap to pack and
    /// the floating origin can rebase under any of them).
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    /// The engagement counter (see [`VoxelReport`]).
    report: VoxelReport,
}

impl VoxelNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout, report: VoxelReport) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("voxel"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("voxel").into()),
            });

        // Lights uniform block (@group(1)) — the same `LightsUniform` packing the
        // mesh pass publishes, bound exactly as `classic_vgeom` binds it.
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-lights"),
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
            label: Some("voxel-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        // Two bind groups only: view + lights. There is deliberately no env group
        // — see the header of shaders/voxel.wgsl for what that costs and why P21.1
        // pays it.
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("voxel"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
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
            pipeline,
            chunks: BTreeMap::new(),
            instances: None,
            instance_capacity: 0,
            lights_buf,
            lights_bg,
            report,
        }
    }

    /// The chunk-local `(min, max)` AABB the cached copy of `key` was uploaded
    /// with, if it is resident.
    ///
    /// The bound travels with the projection, which is dropped once the upload is
    /// done, so caching it here is what lets a later consumer ask about a resident
    /// chunk at all — the P21.2 frustum cull is the consumer it exists for.
    pub fn cached_bounds(&self, key: &ChunkCacheKey) -> Option<([f32; 3], [f32; 3])> {
        self.chunks.get(key).map(|c| c.bounds)
    }

    /// Apply a [`ChunkCachePlan`] to the GPU cache: drop the evicted entries
    /// (freeing their buffers) and write the uploads.
    fn sync_chunks(&mut self, gpu: &GpuContext, volumes: &[RenderVoxelVolume]) {
        let plan = plan_chunk_cache(
            &self.chunks,
            |c: &GpuChunk| CachedChunk {
                version: c.version,
                index_count: c.index_count,
            },
            volumes,
        );
        for key in &plan.evict {
            self.chunks.remove(key);
        }
        if plan.upload.is_empty() {
            return;
        }
        let wanted: BTreeSet<ChunkCacheKey> = plan.upload.iter().copied().collect();
        // Walk the projection again rather than indexing it: the plan carries keys,
        // and the payload lives in the scene. The walk order is the plan's order.
        for volume in volumes {
            for chunk in &volume.chunks {
                let key = (volume.id, chunk.key);
                if !wanted.contains(&key) {
                    continue;
                }
                let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-chunk-verts"),
                    size: std::mem::size_of_val(&chunk.vertices[..]) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&chunk.vertices));
                let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-chunk-indices"),
                    size: std::mem::size_of_val(&chunk.indices[..]) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&indices, 0, bytemuck::cast_slice(&chunk.indices));
                self.chunks.insert(
                    key,
                    GpuChunk {
                        vertices,
                        indices,
                        index_count: chunk.indices.len() as u32,
                        version: chunk.version,
                        bounds: chunk.bounds,
                    },
                );
            }
        }
    }

    /// Pack this frame's per-chunk instances (projection walk order, resident
    /// chunks only) and upload them into the shared instance buffer. Returns the
    /// draw list, parallel to the buffer's slots.
    fn sync_instances(&mut self, gpu: &GpuContext, frame: &FrameData) -> Vec<ChunkCacheKey> {
        let mut raw: Vec<ChunkInstanceRaw> = Vec::new();
        let mut draws: Vec<ChunkCacheKey> = Vec::new();
        for volume in &frame.scene.voxels {
            for chunk in &volume.chunks {
                let key = (volume.id, chunk.key);
                if !self.chunks.contains_key(&key) {
                    continue; // malformed / not resident — nothing to draw
                }
                raw.push(ChunkInstanceRaw::pack(&frame.view.origin, volume, chunk));
                draws.push(key);
            }
        }
        if raw.is_empty() {
            return draws;
        }

        let lights = LightsUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        if self.instances.is_none() || self.instance_capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(16);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-instances"),
                size: (capacity * std::mem::size_of::<ChunkInstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = capacity;
        }
        gpu.queue.write_buffer(
            self.instances.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&raw),
        );
        draws
    }
}

impl RenderNode for VoxelNode {
    fn name(&self) -> &'static str {
        "voxel"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if frame.scene.voxels.is_empty() {
            // The byte-stability guarantee for the 47 committed goldens: no render
            // pass, no buffer write, no command of any kind on a scene with no
            // volumes. Releasing the cache here (a plain map drop — not an encoder
            // touch) is the same eviction `plan_chunk_cache` would produce against
            // an empty projection, applied on the one transition the early-out
            // would otherwise hide: the last volume leaving the scene.
            self.chunks.clear();
            self.instances = None;
            self.instance_capacity = 0;
            return;
        }
        self.sync_chunks(gpu, &frame.scene.voxels);
        let draws = self.sync_instances(gpu, frame);
        if draws.is_empty() {
            // Volumes present but nothing drawable (every chunk malformed or
            // empty). Still no encoder touch.
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        let (chunks, pipeline, lights_bg) = (&self.chunks, &self.pipeline, &self.lights_bg);

        // Past this line the encoder WILL be touched — the one place the counter
        // may move, so an unchanged count is a real statement about the command
        // stream rather than about the pixels.
        self.report.fetch_add(1, Ordering::Relaxed);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel"),
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, lights_bg, &[]);
        pass.set_vertex_buffer(1, instances.slice(..));
        // One draw per resident chunk, in the projection's walk order (the order
        // `draws` was built in, and the order the instance slots were packed in) —
        // deterministic per side, which is what the determinism gate needs.
        for (slot, key) in draws.iter().enumerate() {
            let Some(gpu_chunk) = chunks.get(key) else {
                continue;
            };
            pass.set_vertex_buffer(0, gpu_chunk.vertices.slice(..));
            pass.set_index_buffer(gpu_chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            let slot = slot as u32;
            pass.draw_indexed(0..gpu_chunk.index_count, 0, slot..slot + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::RenderTerrainLayer;
    use glam::DVec3;

    /// A well-formed chunk with `tris` triangles (one fresh vertex per index, so
    /// every index is trivially in range).
    fn chunk(key: VoxelChunkKey, version: u64, tris: u32) -> RenderVoxelChunk {
        let n = tris * 3;
        RenderVoxelChunk {
            key,
            origin: DVec3::ZERO,
            vertices: (0..n)
                .map(|i| RenderVoxelVertex {
                    pos: [i as f32, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    material: 0,
                    seam_nh: RenderVoxelVertex::NO_SEAM,
                    seam_albedo: [0.0; 4],
                })
                .collect(),
            indices: (0..n).collect(),
            bounds: ([0.0; 3], [n as f32, 0.0, 0.0]),
            version,
        }
    }

    fn volume(id: u64, chunks: Vec<RenderVoxelChunk>) -> RenderVoxelVolume {
        RenderVoxelVolume {
            id,
            chunks,
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }
    }

    /// The cache map the tests plan against: plain [`CachedChunk`]s, since the
    /// planner is deliberately generic over the value type precisely so the rule
    /// can be exercised without an adapter.
    type Cache = BTreeMap<ChunkCacheKey, CachedChunk>;

    fn plan(cached: &Cache, volumes: &[RenderVoxelVolume]) -> ChunkCachePlan {
        plan_chunk_cache(cached, |c: &CachedChunk| *c, volumes)
    }

    /// The cache entry a successful upload of `chunk` would leave behind.
    fn cached(chunk: &RenderVoxelChunk) -> CachedChunk {
        CachedChunk {
            version: chunk.version,
            index_count: chunk.indices.len() as u32,
        }
    }

    #[test]
    fn cold_cache_uploads_every_chunk_in_walk_order() {
        let a = chunk(VoxelChunkKey::new(0, 0, 0), 1, 2);
        let b = chunk(VoxelChunkKey::new(1, 0, 0), 1, 2);
        let vols = vec![volume(7, vec![a, b])];
        let p = plan(&Cache::new(), &vols);
        assert_eq!(
            p.upload,
            vec![
                (7, VoxelChunkKey::new(0, 0, 0)),
                (7, VoxelChunkKey::new(1, 0, 0)),
            ]
        );
        assert!(p.evict.is_empty());
    }

    #[test]
    fn an_unchanged_stamp_is_a_cache_hit() {
        let c = chunk(VoxelChunkKey::new(2, -1, 3), 42, 4);
        let vols = vec![volume(1, vec![c.clone()])];
        let mut cache = Cache::new();
        cache.insert((1, c.key), cached(&c));
        assert_eq!(plan(&cache, &vols), ChunkCachePlan::default());
    }

    #[test]
    fn a_bumped_stamp_re_uploads() {
        let c = chunk(VoxelChunkKey::new(0, 0, 0), 42, 4);
        let mut cache = Cache::new();
        cache.insert((1, c.key), cached(&c));
        // Same geometry size, newer stamp: the sculpt that moved the surface
        // without changing its triangle count is exactly the case a size-only
        // gate would miss.
        let bumped = RenderVoxelChunk {
            version: 43,
            ..c.clone()
        };
        let vols = vec![volume(1, vec![bumped])];
        assert_eq!(plan(&cache, &vols).upload, vec![(1, c.key)]);
    }

    #[test]
    fn a_zero_stamp_always_re_uploads() {
        let c = chunk(VoxelChunkKey::new(5, 5, 5), 0, 3);
        let vols = vec![volume(1, vec![c.clone()])];
        let mut cache = Cache::new();
        // Cached at the SAME (zero) stamp and the same size — a stamp compare
        // alone would call this fresh. "No stamp" must degrade to re-upload, never
        // to a stale frame.
        cache.insert((1, c.key), cached(&c));
        let p = plan(&cache, &vols);
        assert_eq!(p.upload, vec![(1, c.key)]);
        assert!(p.evict.is_empty(), "a re-upload is not an eviction");
    }

    #[test]
    fn a_changed_index_count_re_uploads() {
        let c = chunk(VoxelChunkKey::new(0, 0, 0), 9, 4);
        let mut cache = Cache::new();
        cache.insert((1, c.key), cached(&c));
        // Same stamp, more triangles — a projector that resizes a chunk without
        // bumping its stamp would otherwise keep drawing the old index count.
        let grown = chunk(VoxelChunkKey::new(0, 0, 0), 9, 6);
        let vols = vec![volume(1, vec![grown])];
        assert_eq!(plan(&cache, &vols).upload, vec![(1, c.key)]);
    }

    #[test]
    fn a_chunk_leaving_the_projection_evicts() {
        let stay = chunk(VoxelChunkKey::new(0, 0, 0), 1, 2);
        let gone = chunk(VoxelChunkKey::new(9, 9, 9), 1, 2);
        let mut cache = Cache::new();
        cache.insert((1, stay.key), cached(&stay));
        cache.insert((1, gone.key), cached(&gone));
        let vols = vec![volume(1, vec![stay.clone()])];
        let p = plan(&cache, &vols);
        assert!(p.upload.is_empty(), "the surviving chunk was a hit");
        assert_eq!(p.evict, vec![(1, gone.key)]);

        // ...and an entire volume leaving evicts exactly its own chunks.
        let p = plan(&cache, &[]);
        assert!(p.upload.is_empty());
        assert_eq!(p.evict, vec![(1, stay.key), (1, gone.key)]);
    }

    #[test]
    fn malformed_chunks_are_neither_uploaded_nor_kept() {
        let key = VoxelChunkKey::new(0, 0, 0);
        let good = chunk(key, 1, 2);

        // Three ways to be malformed, each independently disqualifying.
        let empty = RenderVoxelChunk {
            indices: Vec::new(),
            ..good.clone()
        };
        let odd = RenderVoxelChunk {
            indices: vec![0, 1, 2, 3],
            ..good.clone()
        };
        let out_of_range = RenderVoxelChunk {
            // 6 vertices exist (2 triangles), so index 6 is one past the end.
            indices: vec![0, 1, 6],
            ..good.clone()
        };

        for bad in [empty, odd, out_of_range] {
            let vols = vec![volume(1, vec![bad])];
            // Never uploaded...
            assert!(
                plan(&Cache::new(), &vols).upload.is_empty(),
                "a malformed chunk was uploaded"
            );
            // ...and, if it was cached while it was still good, evicted rather
            // than left drawing the last version of itself.
            let mut cache = Cache::new();
            cache.insert((1, key), cached(&good));
            let p = plan(&cache, &vols);
            assert!(p.upload.is_empty());
            assert_eq!(p.evict, vec![(1, key)], "a chunk that went bad stayed live");
        }
    }

    #[test]
    fn two_volumes_sharing_a_chunk_key_do_not_collide() {
        let key = VoxelChunkKey::new(3, 0, -2);
        let a = chunk(key, 1, 2);
        let b = chunk(key, 1, 4);
        let vols = vec![volume(10, vec![a.clone()]), volume(20, vec![b.clone()])];

        // Cold: both plan independently, in volume-then-chunk walk order.
        assert_eq!(
            plan(&Cache::new(), &vols).upload,
            vec![(10, key), (20, key)]
        );

        // Volume 10 cached, volume 20 not: only the uncached one uploads — the
        // failure a key-without-a-volume-half would produce is exactly the other
        // volume reading as fresh.
        let mut cache = Cache::new();
        cache.insert((10, key), cached(&a));
        let p = plan(&cache, &vols);
        assert_eq!(p.upload, vec![(20, key)]);
        assert!(p.evict.is_empty());
    }

    #[test]
    fn chunk_instance_raw_matches_attribute_offsets() {
        let v = ChunkInstanceRaw::pack(
            &FloatingOrigin::new(DVec3::ZERO),
            &volume(1, Vec::new()),
            &chunk(VoxelChunkKey::default(), 1, 1),
        );
        let base = &v as *const ChunkInstanceRaw as usize;
        assert_eq!(std::mem::size_of::<ChunkInstanceRaw>(), 176);
        assert_eq!(&v.model as *const _ as usize - base, 0);
        assert_eq!(&v.albedo as *const _ as usize - base, 64);
        assert_eq!(&v.rough as *const _ as usize - base, 128);
        assert_eq!(&v.misc as *const _ as usize - base, 144);
        assert_eq!(&v.seam_params as *const _ as usize - base, 160);
        // Every attribute offset the pipeline declares must land inside the struct
        // at the field it names.
        let offsets: Vec<u64> = INSTANCE_ATTRIBUTES.iter().map(|a| a.offset).collect();
        assert_eq!(
            offsets,
            vec![0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
            "instance attribute offsets drifted from ChunkInstanceRaw"
        );
    }

    #[test]
    fn voxel_vertex_matches_attribute_offsets() {
        let v = RenderVoxelVertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let base = &v as *const RenderVoxelVertex as usize;
        assert_eq!(std::mem::size_of::<RenderVoxelVertex>(), 60);
        assert_eq!(&v.pos as *const _ as usize - base, 0);
        assert_eq!(&v.normal as *const _ as usize - base, 12);
        assert_eq!(&v.material as *const _ as usize - base, 24);
        assert_eq!(&v.seam_nh as *const _ as usize - base, 28);
        assert_eq!(&v.seam_albedo as *const _ as usize - base, 44);
        let offsets: Vec<u64> = VERTEX_ATTRIBUTES.iter().map(|a| a.offset).collect();
        assert_eq!(offsets, vec![0, 12, 24, 28, 44]);
    }

    #[test]
    fn packing_is_a_plain_translation_against_the_floating_origin() {
        let origin = FloatingOrigin::new(DVec3::new(1000.0, 0.0, 0.0));
        let mut vol = volume(1, Vec::new());
        vol.layers[2].albedo = [0.1, 0.2, 0.3, 1.0];
        vol.layers[2].roughness = 0.55;
        let mut c = chunk(VoxelChunkKey::default(), 1, 1);
        c.origin = DVec3::new(1001.0, 2.0, 3.0);
        let raw = ChunkInstanceRaw::pack(&origin, &vol, &c);
        // Translation column is origin-relative; the rotation/scale block is
        // identity (chunks are axis-aligned and unrotated).
        assert_eq!(&raw.model[12..15], &[1.0, 2.0, 3.0]);
        assert_eq!(&raw.model[0..4], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&raw.model[5..7], &[1.0, 0.0]);
        // The palette rides along per chunk, in layer order.
        assert_eq!(raw.albedo[2], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(raw.rough[2], 0.55);
        // Pick ids are P21.2 (the ID pass does not draw voxels).
        assert_eq!(raw.misc, [0; 4]);
    }
}
