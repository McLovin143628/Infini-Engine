//! **Fracture debris (P22.3): drawing what a destructible broke into.**
//!
//! # Why this is a pass and not an addition to `mesh`
//!
//! [`MeshNode`](super::mesh::MeshNode) draws *primitives*: one static vertex
//! buffer holding a cube, a sphere, a plane, a cylinder and a cone, and every
//! instance selects one of the five. There is no door in it for custom geometry,
//! and a fracture chunk is nothing but custom geometry — a Voronoi cell of an
//! authored mesh, one of up to sixty-four per actor, each with a solver-owned
//! pose.
//!
//! # …and why it is not a second shader
//!
//! It reuses **`mesh.wgsl` unchanged**. The vertex layout
//! ([`RenderFractureVertex`] is byte-identical to
//! [`MeshVertex`](super::mesh::MeshVertex)), the instance layout
//! ([`InstanceRaw`]), the three bind groups (view,
//! lights, environment) and both pipeline states are the mesh pass's own. What
//! differs is where the *vertices* come from: a per-chunk buffer instead of the
//! shared primitive one. Sharing the shader is what keeps debris lit exactly like
//! the wall it came off — a second PBR shader would be a second thing to keep in
//! step, and the first frame it drifted the rubble would stop matching the
//! building.
//!
//! # The cache
//!
//! [`plan_fracture_cache`] is the [`plan_chunk_cache`](super::voxel::plan_chunk_cache)
//! rule, keyed `(entity, chunk index)`. The stamp is the owning actor's
//! **fracture generation**, which moves when a chunk detaches or is reclaimed and
//! **not** when one merely tumbles — so a collapsing building uploads its geometry
//! once per break and then only writes instance matrices, sixty times a second,
//! for as long as the rubble is settling. Keying on the pose instead would
//! re-upload every chunk every frame, which is the whole cost this exists to
//! avoid.
//!
//! The planner is a **pure function** over the projection, so the whole eviction
//! and re-upload story is unit-tested with no GPU — the pattern `terrain` started
//! and `voxel` copied.

use std::collections::{BTreeMap, BTreeSet};

use glam::Mat3;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{vertex_layouts, InstanceRaw, LightsUniform};
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{RenderFractureChunk, RenderFractureVertex};
use crate::settings::RenderSettings;

/// What identifies one cached chunk buffer: `(actor id, chunk index)`.
///
/// The actor half is present **from the start**, on the lesson
/// [`super::voxel::ChunkCacheKey`] records: two destructibles both have a chunk
/// `0`, and a cache keyed on the index alone would serve one wall's rubble as
/// another's.
pub type FractureCacheKey = (u64, u32);

/// What a cached chunk buffer was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedFracture {
    /// The owning actor's fracture generation at upload time.
    pub version: u64,
    /// Indices uploaded — part of the identity, so a chunk whose geometry
    /// changed length is re-uploaded even if a stamp somehow did not move.
    pub index_count: u32,
}

/// One frame's cache plan: what to upload, what to evict. Both in ascending key
/// order, so two runs issue the same commands in the same sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FractureCachePlan {
    pub upload: Vec<FractureCacheKey>,
    pub evict: Vec<FractureCacheKey>,
}

/// Is this chunk's geometry usable? Non-empty, a whole number of triangles, and
/// every index in range — the maximum is **computed**, not assumed.
///
/// A malformed chunk is neither uploaded nor kept: it does not enter `live`, so
/// the plan evicts whatever was cached for it. Bad geometry never becomes silent
/// triangles.
pub fn well_formed(chunk: &RenderFractureChunk) -> bool {
    if chunk.indices.is_empty()
        || !chunk.indices.len().is_multiple_of(3)
        || chunk.vertices.is_empty()
    {
        return false;
    }
    let max = chunk.indices.iter().copied().max().unwrap_or(0) as usize;
    max < chunk.vertices.len()
}

/// Decide what a chunk cache must do this frame — a **pure function** of the
/// cached state and the projection.
///
/// A chunk is a cache **hit** only when its stamp is non-zero, matches, and the
/// index count matches too. `version == 0` never hits: a source that cannot
/// version its chunks must degrade to re-uploading, not to a stale frame.
pub fn plan_fracture_cache<T>(
    cached: &BTreeMap<FractureCacheKey, T>,
    state: impl Fn(&T) -> CachedFracture,
    chunks: &[RenderFractureChunk],
) -> FractureCachePlan {
    let mut plan = FractureCachePlan::default();
    let mut live: BTreeSet<FractureCacheKey> = BTreeSet::new();
    for chunk in chunks {
        if !well_formed(chunk) {
            continue;
        }
        let key = (chunk.entity, chunk.chunk);
        // A duplicate key in one projection is a projector bug; take the first
        // and ignore the rest rather than uploading twice to one slot.
        if !live.insert(key) {
            continue;
        }
        let fresh = chunk.version != 0
            && cached.get(&key).map(&state)
                == Some(CachedFracture {
                    version: chunk.version,
                    index_count: chunk.indices.len() as u32,
                });
        if !fresh {
            plan.upload.push(key);
        }
    }
    plan.evict = cached
        .keys()
        .filter(|k| !live.contains(k))
        .copied()
        .collect();
    plan.upload.sort_unstable();
    plan
}

/// One chunk's GPU buffers.
struct GpuFracture {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    version: u64,
}

/// Pack one chunk's per-instance data, reusing the mesh pass's own
/// [`InstanceRaw`] so `mesh.wgsl` reads it unchanged.
///
/// Scale is **always 1**: the actor's scale was baked into the chunk's vertices
/// at seed time (see `inf_physics::d3::fracture`), because a detached chunk's
/// pose is solver-owned and the solver knows nothing about the actor's transform.
/// The normal matrix is therefore a pure rotation.
fn pack(origin: &FloatingOrigin, chunk: &RenderFractureChunk) -> InstanceRaw {
    let model = origin.model_matrix(chunk.translation, chunk.rotation, glam::Vec3::ONE);
    let n = Mat3::from_quat(chunk.rotation).to_cols_array_2d();
    InstanceRaw {
        model: model.to_cols_array(),
        normal_mat: [
            n[0][0], n[0][1], n[0][2], 0.0, n[1][0], n[1][1], n[1][2], 0.0, n[2][0], n[2][1],
            n[2][2], 0.0,
        ],
        color: chunk.color,
        // Pick id 0: debris is not selectable. A chunk is not an entity, so there
        // is nothing for a click to select — the same reason the voxel pass wires
        // its pick id to zero.
        misc: [0, 0, 0, 0],
        pbr: [chunk.metallic, chunk.roughness, 0.5, 0.0],
        emissive: [chunk.emissive[0], chunk.emissive[1], chunk.emissive[2], 0.0],
    }
}

/// Draws every live fracture chunk.
pub struct FractureNode {
    pipeline: wgpu::RenderPipeline,
    /// The single-sample, fragment-less depth-prepass twin (wave VIS1a).
    pipeline_depth: wgpu::RenderPipeline,
    /// The frame [`Self::draw_order`] and the instance buffer were built for; the
    /// prepass and the colour pass share one preparation (wave VIS1a).
    prepared_frame: Option<u64>,
    chunks: BTreeMap<FractureCacheKey, GpuFracture>,
    /// One shared per-frame instance buffer: chunk `i`'s instance sits at index
    /// `i` of the draw order, and each chunk is drawn with `instance_range i..i+1`
    /// against its own vertex buffer.
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    draw_order: Vec<FractureCacheKey>,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    env: super::EnvBinding,
}

impl FractureNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        // `mesh.wgsl`, unchanged — see the module docs.
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fracture"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mesh").into()),
            });
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fracture-lights"),
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
            label: Some("fracture-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fracture-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });
        let env = super::EnvBinding::new(gpu);
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fracture"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl), Some(&env.bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fracture"),
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
                    // NO back-face culling. A fracture chunk's interior faces come
                    // from clipping the source mesh against half-spaces, and their
                    // winding is whatever the cut produced — culling would punch
                    // holes in the freshly-exposed inside of a broken wall, which
                    // is the one surface a player is looking at.
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
        // The depth-prepass pipeline (wave VIS1a): the same `mesh.wgsl` `vs`, view
        // group only, no fragment stage. Debris is opaque — the R-P5 alpha test
        // lives in `fs` and a fracture chunk never carries blend code 1 — so the
        // silhouette a fragment-less pipeline writes is the one `fs` would.
        let depth_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fracture-depth"),
                bind_group_layouts: &[Some(view_bgl)],
                immediate_size: 0,
            });
        let pipeline_depth = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fracture-depth"),
                layout: Some(&depth_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
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
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self {
            pipeline,
            pipeline_depth,
            prepared_frame: None,
            chunks: BTreeMap::new(),
            instances: None,
            instance_capacity: 0,
            draw_order: Vec::new(),
            lights_buf,
            lights_bg,
            env,
        }
    }

    /// **Everything both draws share**, done once per frame (wave VIS1a). Called
    /// from [`RenderNode::depth_prepass`] and again from [`RenderNode::run`]; the
    /// second call is a stamp comparison. Returns whether there is anything to
    /// draw.
    fn prepare(&mut self, gpu: &GpuContext, frame: &FrameData) -> bool {
        if self.prepared_frame == Some(frame.frame_index) {
            return !self.draw_order.is_empty();
        }
        self.prepared_frame = Some(frame.frame_index);
        // **The off path touches no encoder at all.** Every level that has broken
        // nothing — which is every committed golden — takes this branch, so the
        // command stream is byte-identical to its pre-P22.3 self. (The
        // `VoxelNode` rule, for the same reason.)
        if frame.scene.fracture_chunks.is_empty() {
            self.chunks.clear();
            self.draw_order.clear();
            return false;
        }
        self.sync_chunks(gpu, &frame.scene.fracture_chunks);
        self.sync_instances(gpu, frame);
        !self.draw_order.is_empty()
    }

    /// How many chunk buffer sets are cached — the engagement counter a gate
    /// reads to tell "the pass ran" from "the pass had nothing to do".
    pub fn cached_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Apply this frame's cache plan: evict, then upload.
    fn sync_chunks(&mut self, gpu: &GpuContext, chunks: &[RenderFractureChunk]) {
        let plan = plan_fracture_cache(
            &self.chunks,
            |c| CachedFracture {
                version: c.version,
                index_count: c.index_count,
            },
            chunks,
        );
        for key in &plan.evict {
            self.chunks.remove(key);
        }
        if plan.upload.is_empty() {
            return;
        }
        let want: BTreeSet<FractureCacheKey> = plan.upload.iter().copied().collect();
        for chunk in chunks {
            let key = (chunk.entity, chunk.chunk);
            if !want.contains(&key) || !well_formed(chunk) {
                continue;
            }
            let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fracture-chunk-verts"),
                size: (chunk.vertices.len() * std::mem::size_of::<RenderFractureVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&vertices, 0, bytemuck::cast_slice(&chunk.vertices));
            let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fracture-chunk-indices"),
                size: (chunk.indices.len() * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&indices, 0, bytemuck::cast_slice(&chunk.indices));
            self.chunks.insert(
                key,
                GpuFracture {
                    vertices,
                    indices,
                    index_count: chunk.indices.len() as u32,
                    version: chunk.version,
                },
            );
        }
    }

    /// Re-pack + upload this frame's instance matrices, and return the draw
    /// order.
    fn sync_instances(&mut self, gpu: &GpuContext, frame: &FrameData) {
        self.draw_order.clear();
        let mut raw: Vec<InstanceRaw> = Vec::new();
        for chunk in &frame.scene.fracture_chunks {
            let key = (chunk.entity, chunk.chunk);
            if !self.chunks.contains_key(&key) {
                continue;
            }
            self.draw_order.push(key);
            raw.push(pack(&frame.view.origin, chunk));
        }
        if raw.is_empty() {
            return;
        }
        if self.instances.is_none() || self.instance_capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(32);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fracture-instances"),
                size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
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
    }
}

impl RenderNode for FractureNode {
    fn name(&self) -> &'static str {
        "fracture"
    }

    /// **Debris in the prepass** (wave VIS1a) — a collapsed wall's chunks are
    /// exactly the contact geometry AO exists to darken.
    fn depth_prepass(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameData,
    ) {
        if !RenderSettings::needs_depth_prepass(frame.settings) {
            return;
        }
        if !self.prepare(gpu, frame) {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fracture-depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth_prepass,
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
        pass.set_pipeline(&self.pipeline_depth);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_vertex_buffer(1, instances.slice(..));
        for (i, key) in self.draw_order.iter().enumerate() {
            let Some(gpu_chunk) = self.chunks.get(key) else {
                continue;
            };
            pass.set_vertex_buffer(0, gpu_chunk.vertices.slice(..));
            pass.set_index_buffer(gpu_chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            let i = i as u32;
            pass.draw_indexed(0..gpu_chunk.index_count, 0, i..i + 1);
        }
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !self.prepare(gpu, frame) {
            return;
        }
        let lights =
            LightsUniform::from_scene(frame.scene, &frame.view.origin, frame.vsm_light_slots);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));
        let env_bg = self.env.bind_group(gpu, frame).clone();
        let Some(instances) = self.instances.as_ref() else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fracture"),
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
        pass.set_bind_group(1, &self.lights_bg, &[]);
        pass.set_bind_group(2, &env_bg, &[]);
        pass.set_vertex_buffer(1, instances.slice(..));
        for (i, key) in self.draw_order.iter().enumerate() {
            let Some(gpu_chunk) = self.chunks.get(key) else {
                continue;
            };
            pass.set_vertex_buffer(0, gpu_chunk.vertices.slice(..));
            pass.set_index_buffer(gpu_chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            let i = i as u32;
            pass.draw_indexed(0..gpu_chunk.index_count, 0, i..i + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, Quat};

    type Cache = BTreeMap<FractureCacheKey, CachedFracture>;

    fn chunk(entity: u64, index: u32, version: u64, tris: u32) -> RenderFractureChunk {
        let verts = (tris * 3).max(3) as usize;
        RenderFractureChunk {
            entity,
            chunk: index,
            translation: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            vertices: vec![
                RenderFractureVertex {
                    pos: [0.0; 3],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.0; 2],
                };
                verts
            ],
            indices: (0..tris * 3).map(|i| i % verts as u32).collect(),
            color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            version,
        }
    }

    fn plan(cached: &Cache, chunks: &[RenderFractureChunk]) -> FractureCachePlan {
        plan_fracture_cache(cached, |c| *c, chunks)
    }

    fn cached(c: &RenderFractureChunk) -> CachedFracture {
        CachedFracture {
            version: c.version,
            index_count: c.indices.len() as u32,
        }
    }

    #[test]
    fn cold_cache_uploads_every_chunk_in_key_order() {
        let chunks = vec![chunk(2, 1, 5, 4), chunk(1, 0, 5, 4), chunk(1, 3, 5, 4)];
        let p = plan(&Cache::new(), &chunks);
        assert_eq!(p.upload, vec![(1, 0), (1, 3), (2, 1)]);
        assert!(p.evict.is_empty());
    }

    #[test]
    fn an_unchanged_generation_is_a_cache_hit() {
        let chunks = [chunk(1, 0, 7, 4)];
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunks[0]));
        assert_eq!(plan(&c, &chunks), FractureCachePlan::default());
    }

    /// **The point of the stamp.** A chunk that only MOVED keeps its geometry:
    /// the generation does not move for a pose, so a collapsing building uploads
    /// once per break rather than sixty times a second.
    #[test]
    fn a_moved_chunk_is_still_a_cache_hit() {
        let mut chunks = vec![chunk(1, 0, 7, 4)];
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunks[0]));
        chunks[0].translation = DVec3::new(3.0, -2.0, 1.0);
        chunks[0].rotation = Quat::from_rotation_y(0.7);
        assert_eq!(
            plan(&c, &chunks),
            FractureCachePlan::default(),
            "a tumbling chunk re-uploaded its geometry"
        );
    }

    #[test]
    fn a_bumped_generation_re_uploads() {
        let chunks = [chunk(1, 0, 7, 4)];
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunks[0]));
        let bumped = vec![chunk(1, 0, 8, 4)];
        assert_eq!(plan(&c, &bumped).upload, vec![(1, 0)]);
    }

    #[test]
    fn a_zero_generation_always_re_uploads() {
        let chunks = [chunk(1, 0, 0, 4)];
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunks[0]));
        assert_eq!(
            plan(&c, &chunks).upload,
            vec![(1, 0)],
            "a stamp of 0 must never be a hit"
        );
    }

    #[test]
    fn a_changed_index_count_re_uploads() {
        let chunks = [chunk(1, 0, 7, 4)];
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunks[0]));
        let regrown = vec![chunk(1, 0, 7, 9)];
        assert_eq!(plan(&c, &regrown).upload, vec![(1, 0)]);
    }

    /// A reclaimed chunk leaves the projection, and the cache follows it out —
    /// the debris budget's despawn, on the GPU side.
    #[test]
    fn a_chunk_leaving_the_projection_evicts() {
        let mut c = Cache::new();
        c.insert((1, 0), cached(&chunk(1, 0, 7, 4)));
        c.insert((1, 1), cached(&chunk(1, 1, 7, 4)));
        let p = plan(&c, &[chunk(1, 0, 7, 4)]);
        assert_eq!(p.evict, vec![(1, 1)]);
        assert!(p.upload.is_empty());
    }

    #[test]
    fn malformed_chunks_are_neither_uploaded_nor_kept() {
        let mut bad = chunk(1, 0, 7, 4);
        bad.indices = vec![0, 1]; // not a whole triangle
        let mut oob = chunk(1, 1, 7, 4);
        oob.indices = vec![0, 1, 999]; // out of range
        let mut empty = chunk(1, 2, 7, 4);
        empty.indices.clear();
        let mut c = Cache::new();
        for k in [(1, 0), (1, 1), (1, 2)] {
            c.insert(k, cached(&chunk(k.0, k.1, 7, 4)));
        }
        let p = plan(&c, &[bad, oob, empty]);
        assert!(p.upload.is_empty(), "a malformed chunk was uploaded");
        assert_eq!(
            p.evict,
            vec![(1, 0), (1, 1), (1, 2)],
            "a malformed chunk was kept"
        );
    }

    /// Two actors' chunk `0` are two different chunks — the whole reason the key
    /// carries the actor.
    #[test]
    fn two_actors_sharing_a_chunk_index_do_not_collide() {
        let chunks = vec![chunk(1, 0, 7, 4), chunk(2, 0, 7, 4)];
        let p = plan(&Cache::new(), &chunks);
        assert_eq!(p.upload, vec![(1, 0), (2, 0)]);
    }

    /// The vertex layout really is the mesh pass's, byte for byte — the claim the
    /// shader reuse rests on.
    ///
    /// **Read off `mesh::VERTEX_ATTRIBUTES`, not off three literals** (P26.5
    /// audit). This buffer is bound against `vertex_layouts()`, whose stride is
    /// `size_of::<MeshVertex>()` and whose offsets are that constant's: a
    /// hand-copied `24` agrees with the pipeline only until someone moves an
    /// attribute, and a fracture chunk drawn from the wrong bytes is what the
    /// P26.5 commit body calls "every chunk drawn from the wrong bytes".
    #[test]
    fn the_vertex_layout_matches_the_mesh_pass() {
        use crate::passes::mesh::VERTEX_ATTRIBUTES;
        assert_eq!(
            std::mem::size_of::<RenderFractureVertex>(),
            std::mem::size_of::<super::super::mesh::MeshVertex>()
        );
        let v = RenderFractureVertex {
            pos: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
        };
        let base = &v as *const RenderFractureVertex as usize;
        // `@location(0)` position, `@location(1)` normal, `@location(2)` uv —
        // each at the offset the pipeline declares for that location.
        for (field, loc) in [
            (&v.pos as *const _ as usize - base, 0u32),
            (&v.normal as *const _ as usize - base, 1),
            (&v.uv as *const _ as usize - base, 2),
        ] {
            let want = VERTEX_ATTRIBUTES
                .iter()
                .find(|a| a.shader_location == loc)
                .unwrap_or_else(|| panic!("the mesh pass declares no @location({loc})"))
                .offset as usize;
            assert_eq!(
                field, want,
                "a fracture chunk's @location({loc}) sits at byte {field} and the \
                 mesh pipeline reads it at {want}"
            );
        }
    }

    /// Debris carries no pick id: a chunk is not an entity, so a click has
    /// nothing to select.
    #[test]
    fn packing_gives_debris_no_pick_id() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let raw = pack(&origin, &chunk(1, 0, 7, 4));
        assert_eq!(raw.misc[0], 0);
    }
}
