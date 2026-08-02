//! Cascaded shadow-map pass (P13.3b): renders the first directional light's three
//! view-frustum cascades into a `Depth32Float` array, and publishes the per-cascade
//! matrices + bias constants in the shared [`ShadowResources`] uniform the lit
//! passes sample (see [`crate::passes::EnvBinding`] + `shaders/mesh.wgsl`).
//!
//! The cascade split scheme, sphere fit, and texel snapping are pure functions in
//! [`crate::csm`]. This node owns the caster pipeline (a depth-only forward-Z
//! render, mirroring [`crate::passes::depth_prepass`]) and re-packs its own cube
//! instance buffer, so it is self-contained.
//!
//! ## Scope (v1, documented)
//!
//! * **Casters:** rigid [`MeshInstance`] geometry only (the golden is boxes on a
//!   plane). Terrain-patch + skinned casters are a follow-up.
//! * **Receivers:** mesh / skinned / terrain all *sample* the cascades (shadows
//!   land on all of them); only casting is scoped.
//! * **Off path:** when [`ShadowSettings::enabled`] is false the node still writes
//!   the shared uniform (with `enabled = 0`) so receivers read a valid flag, then
//!   renders nothing — receivers take the byte-stable un-shadowed instruction path.

use crate::camera::DEPTH_FORMAT;
use crate::csm::{
    bounding_sphere, cascade_matrix, cascade_splits, frustum_slice_corners, SHADOW_CASCADES,
    SHADOW_RESOLUTION,
};
use std::ops::Range;

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{pack_bucketed, vertex_layouts, InstanceRaw, EMPTY_RANGES};
use crate::primitives::PrimGpu;
use crate::renderer::FrameData;
use crate::scene::LightKind;

/// Forward-Z shadow depth: nearest caster wins (clear to 1.0 = far, keep smaller).
const SHADOW_DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::LessEqual;
const SHADOW_DEPTH_CLEAR: f32 = 1.0;

/// The shared shadow uniform block (`std140`), written by [`ShadowNode`] and read
/// by every lit pass through [`crate::passes::EnvBinding`]. Mirrors `struct
/// ShadowData` in the receiver shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowDataGpu {
    /// Per-cascade forward-Z light `view_proj`.
    pub cascade_vp: [[f32; 16]; SHADOW_CASCADES],
    /// Cascade far distances (x,y,z); **w = the P18.4 cascade blend fraction**
    /// ([`crate::ShadowSettings::cascade_blend`]) — the slot that was spare.
    pub splits: [f32; 4],
    /// Per-cascade world texel size (x,y,z); w unused (drives normal-offset bias).
    pub texel_world: [f32; 4],
    /// x = enabled, y = depth_bias (NDC), z = normal_bias (texels),
    /// w = cascade count.
    pub params: [f32; 4],
}

impl ShadowDataGpu {
    fn disabled() -> Self {
        Self {
            cascade_vp: [[0.0; 16]; SHADOW_CASCADES],
            splits: [0.0; 4],
            texel_world: [0.0; 4],
            params: [0.0, 0.0, 0.0, SHADOW_CASCADES as f32],
        }
    }
}

/// Renderer-owned shadow GPU resources, created once (independent of viewport size)
/// and shared with the lit passes via [`FrameData`]. The [`ShadowNode`] renders the
/// cascades into [`layer_views`](Self::layer_views) and writes [`uniform`](Self::uniform);
/// receivers sample [`array_view`](Self::array_view) + read the uniform.
pub struct ShadowResources {
    _map: wgpu::Texture,
    /// Full depth array view (receiver sampling).
    pub array_view: wgpu::TextureView,
    /// Per-cascade single-layer views (caster rendering).
    pub layer_views: Vec<wgpu::TextureView>,
    /// Shared `ShadowData` uniform (written by the node, read by receivers).
    pub uniform: wgpu::Buffer,
    /// Stable generation for receiver bind-group caching (resources never resize).
    pub generation: u64,
}

impl ShadowResources {
    pub fn new(gpu: &GpuContext) -> Self {
        let map = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
                depth_or_array_layers: SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-map-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..SHADOW_CASCADES as u32)
            .map(|layer| {
                map.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow-cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-data"),
            size: std::mem::size_of::<ShadowDataGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed with the disabled state so a receiver bound before the first
        // ShadowNode run still reads enabled = 0.
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&ShadowDataGpu::disabled()));
        Self {
            _map: map,
            array_view,
            layer_views,
            uniform,
            generation: 0,
        }
    }
}

/// Per-cascade caster uniform (`struct Cascade` in `shadow_depth.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CascadeGpu {
    view_proj: [f32; 16],
}

pub struct ShadowNode {
    pipeline: wgpu::RenderPipeline,
    /// R-P5 masked caster variant (`vs_masked` + `fs_masked`, alpha-test discard)
    /// so a masked object casts a cut-out shadow. Used only when the scene carries
    /// masked instances; otherwise the fragment-less `pipeline` draws
    /// (byte-identical to the pre-R-P5 shadow path).
    pipeline_masked: wgpu::RenderPipeline,
    prim: PrimGpu,
    /// One tiny uniform + bind group per cascade (distinct buffers so the
    /// per-cascade writes don't collide before the passes run).
    cascade_bufs: Vec<wgpu::Buffer>,
    cascade_bgs: Vec<wgpu::BindGroup>,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    instance_count: u32,
    ranges: [Range<u32>; 5],
    uploaded_version: Option<(u64, glam::DVec3)>,
    /// Whether the disabled-shadows uniform is already published to the (stable,
    /// created-once) `frame.shadow.uniform`. Gates the constant re-write while
    /// shadows stay off; the enabled path clears it so a later disable re-publishes.
    published_disabled: bool,
}

impl ShadowNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow-depth"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/shadow_depth.wgsl").into(),
                ),
            });

        let cascade_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow-cascade"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let prim = PrimGpu::new(gpu, "shadow");

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow-depth"),
                bind_group_layouts: &[Some(&cascade_bgl)],
                immediate_size: 0,
            });
        let primitive = wgpu::PrimitiveState {
            // Front-face culling reduces peter-panning/acne on the classic box
            // casters (shadow depth from back faces).
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        };
        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(SHADOW_DEPTH_COMPARE),
            stencil: Default::default(),
            // A slope-scaled hardware depth bias further reduces acne.
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };
        let make = |label: &str, vs: &str, fs: Option<wgpu::FragmentState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &vertex_layouts(),
                    },
                    fragment: fs,
                    primitive,
                    depth_stencil: Some(depth_stencil.clone()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let pipeline = make("shadow-depth", "vs", None);
        let pipeline_masked = make(
            "shadow-depth-masked",
            "vs_masked",
            Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_masked"),
                compilation_options: Default::default(),
                targets: &[], // depth-only discard; no colour target
            }),
        );

        let cascade_bufs: Vec<wgpu::Buffer> = (0..SHADOW_CASCADES)
            .map(|c| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow-cascade-{c}")),
                    size: std::mem::size_of::<CascadeGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let cascade_bgs = cascade_bufs
            .iter()
            .map(|buf| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow-cascade"),
                    layout: &cascade_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect();

        Self {
            pipeline,
            pipeline_masked,
            prim,
            cascade_bufs,
            cascade_bgs,
            instances: None,
            instance_capacity: 0,
            instance_count: 0,
            ranges: EMPTY_RANGES,
            uploaded_version: None,
            published_disabled: false,
        }
    }

    /// Re-pack the caster instance buffer when the scene changed or the origin
    /// rebased (mirrors [`crate::passes::depth_prepass`]).
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }
        // Opaque+masked casters only (translucent geometry doesn't cast; folding
        // translucent shadows in is a documented follow-up).
        let (raw, ranges, _translucent) = pack_bucketed(&frame.view.origin, &frame.scene.instances);
        self.ranges = ranges;
        self.instance_count = raw.len() as u32;
        if !raw.is_empty() {
            if self.instances.is_none() || self.instance_capacity < raw.len() {
                let capacity = raw.len().next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shadow-instances"),
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
        self.uploaded_version = Some(key);
    }
}

impl RenderNode for ShadowNode {
    fn name(&self) -> &'static str {
        "shadow"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let s = &frame.settings.shadows;
        if !s.enabled {
            // Publish the disabled uniform (valid enabled=0 flag) once and render
            // nothing; the buffer is created once and only this node writes it, so
            // re-publishing the same constant every frame is redundant.
            if !self.published_disabled {
                gpu.queue.write_buffer(
                    &frame.shadow.uniform,
                    0,
                    bytemuck::bytes_of(&ShadowDataGpu::disabled()),
                );
                self.published_disabled = true;
            }
            return;
        }
        // The enabled path overwrites the uniform below, so a later disable must
        // re-publish the disabled block.
        self.published_disabled = false;

        // Shadow caster: the first directional light that **casts shadows** (R-P3
        // honours `cast_shadows` for directional CSM; point/spot shadow maps are
        // deferred, so their flag is inert). Fallbacks:
        //  * no directional light at all → the scene's **projected** sun (P17.1;
        //    a point/spot-only scene still gets grounded shadows, matching the
        //    receiver shaders' `view.sun_dir` fallback — and both now follow the
        //    time of day);
        //  * directional lights exist but none casts → no CSM this frame (publish
        //    the disabled uniform so receivers read a valid `enabled = 0`, render
        //    nothing).
        let caster = frame
            .scene
            .lights
            .iter()
            .find(|l| l.kind == LightKind::Directional && l.cast_shadows)
            .map(|l| l.direction.normalize_or_zero())
            .filter(|d| d.length_squared() > 1e-6);
        let any_directional = frame
            .scene
            .lights
            .iter()
            .any(|l| l.kind == LightKind::Directional);
        let light_dir_to = match caster {
            Some(d) => d,
            None if !any_directional => frame.scene.sun.unit_direction(),
            None => {
                gpu.queue.write_buffer(
                    &frame.shadow.uniform,
                    0,
                    bytemuck::bytes_of(&ShadowDataGpu::disabled()),
                );
                self.published_disabled = true;
                return;
            }
        };

        self.sync(gpu, frame);

        // Only pay for the alpha-test fragment when the scene has masked casters.
        let caster_pipeline = if frame.scene.instances.iter().any(|i| i.blend == 1) {
            &self.pipeline_masked
        } else {
            &self.pipeline
        };

        // Cascade splits across the shadow range.
        let near = frame.view.near.max(0.05);
        let far = s.max_distance.max(near + 1.0);
        let splits = cascade_splits(near, far, s.lambda);

        let eye = frame.view.eye_local();
        let fwd = frame.view.forward;
        let up = frame.view.up;
        let fov = frame.view.fov_y;
        let aspect = frame.view.aspect();

        let mut data = ShadowDataGpu::disabled();
        data.params = [1.0, s.depth_bias, s.normal_bias, SHADOW_CASCADES as f32];
        for c in 0..SHADOW_CASCADES {
            let d0 = if c == 0 { near } else { splits[c - 1] };
            let d1 = splits[c];
            let corners = frustum_slice_corners(eye, fwd, up, fov, aspect, d0, d1);
            let (center, radius) = bounding_sphere(&corners);
            let (vp, texel) = cascade_matrix(light_dir_to, center, radius, SHADOW_RESOLUTION);
            data.cascade_vp[c] = vp.to_cols_array();
            data.splits[c] = d1;
            data.texel_world[c] = texel;
            gpu.queue.write_buffer(
                &self.cascade_bufs[c],
                0,
                bytemuck::bytes_of(&CascadeGpu {
                    view_proj: vp.to_cols_array(),
                }),
            );
        }
        // P18.4 cascade blending: the receiver lerps into the next cascade across
        // the last `blend × range` of this one. `0` restores the hard switch
        // exactly (the shader branch is not taken and the second PCF never issues).
        data.splits[3] = s.cascade_blend.clamp(0.0, 0.5);
        gpu.queue
            .write_buffer(&frame.shadow.uniform, 0, bytemuck::bytes_of(&data));

        // One depth-only render per cascade (always run so the layer is cleared to
        // far even with no casters → everything reads as lit).
        for c in 0..SHADOW_CASCADES {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &frame.shadow.layer_views[c],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SHADOW_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count == 0 {
                continue;
            }
            let Some(instances) = self.instances.as_ref() else {
                continue;
            };
            pass.set_pipeline(caster_pipeline);
            pass.set_bind_group(0, &self.cascade_bgs[c], &[]);
            self.prim.draw(&mut pass, instances, &self.ranges);
        }
    }
}
