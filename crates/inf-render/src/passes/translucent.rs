//! Translucent forward pass (R-P5). Draws the scene's translucent
//! ([`MeshInstance::blend`] == 2) instances **after** the opaque + terrain passes
//! and **before** the grid, into the same MSAA HDR scene target, with:
//!
//! * **alpha blending** (`src_alpha`, `one_minus_src_alpha`) — the fragment alpha
//!   is the instance's base-color alpha;
//! * **depth test on, depth write OFF** — translucent fragments are occluded by
//!   opaque geometry but don't occlude each other (correct order comes from the
//!   CPU sort, not the depth buffer);
//! * a **per-frame back-to-front CPU sort** by view-space depth (deterministic:
//!   depth descending, ties broken by original index) so overlapping panes
//!   composite correctly.
//!
//! It **reuses the `mesh` shader module** (entry points `vs` + `fs`), so it needs
//! no new [`SHADER_TABLE`](super::SHADER_TABLE) row — the same PBR lighting +
//! unlit-view-mode flag apply. Because the sort mixes primitive kinds, the pass
//! re-packs its own instance buffer each frame and draws one instance at a time
//! (translucent counts are small — a documented v1 cost; a kind-run batching /
//! camera-delta gate is a follow-up).
//!
//! ## Scope (v1, documented)
//!
//! * Single-sided (cull `Back`), like the opaque mesh pass — two-sided translucent
//!   is a follow-up.
//! * Classic [`MeshInstance`] geometry only; vgeom / skinned translucency and
//!   texture-sampled opacity are **deferred** (R-P5 scope).

use std::cmp::Ordering;

use bytemuck::Zeroable;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{vertex_layouts, InstanceRaw, LightsUniform};
use crate::primitives::PrimGpu;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};

/// Back-to-front order (farthest first) of translucent instances by their
/// view-space `depths`. Deterministic: primary key = depth **descending**, ties
/// broken by original index **ascending** (a stable total order — `total_cmp`
/// keeps it well-defined even with NaN/±0 depths).
pub(crate) fn sorted_back_to_front(depths: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..depths.len()).collect();
    idx.sort_by(|&a, &b| match depths[b].total_cmp(&depths[a]) {
        Ordering::Equal => a.cmp(&b),
        other => other,
    });
    idx
}

pub struct TranslucentNode {
    pipeline: wgpu::RenderPipeline,
    prim: PrimGpu,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    /// Sorted per-instance primitive kinds (parallel to the packed buffer).
    kinds: Vec<usize>,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    env: super::EnvBinding,
}

impl TranslucentNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        // Reuse the mesh module (no new SHADER_TABLE row).
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("translucent"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mesh").into()),
            });

        let prim = PrimGpu::new(gpu, "translucent");

        // Lights uniform (@group(1)) — same layout the mesh pass builds.
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("translucent-lights"),
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
            label: Some("translucent-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("translucent-lights"),
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
                label: Some("translucent"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl), Some(&env.bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("translucent"),
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Test against the opaque depth, but never write — translucent
                    // order comes from the CPU sort, not the depth buffer.
                    depth_write_enabled: Some(false),
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
            prim,
            instances: None,
            instance_capacity: 0,
            kinds: Vec::new(),
            lights_buf,
            lights_bg,
            env,
        }
    }

    /// Gather the translucent instances, sort them back-to-front, and re-pack +
    /// upload the sorted instance buffer. Re-run **every frame** (the sort depends
    /// on the live camera, which moves without bumping `scene.version`) — cheap
    /// because translucent counts are small.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        // Refresh the lights uniform (mirrors the mesh pass; render-local point
        // positions depend on the floating origin).
        let lights =
            LightsUniform::from_scene(frame.scene, &frame.view.origin, frame.vsm_light_slots);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        let origin = &frame.view.origin;
        let eye = frame.view.eye_local();
        let fwd = frame.view.forward;

        // Translucent instances + their view-space depth (distance along the view
        // axis; larger = farther = drawn first).
        let mut translucent = Vec::new();
        let mut depths = Vec::new();
        for inst in frame.scene.instances.iter().filter(|i| i.blend == 2) {
            let rp = origin.to_render(inst.translation);
            depths.push((rp - eye).dot(fwd));
            translucent.push(inst);
        }
        self.kinds.clear();
        if translucent.is_empty() {
            return;
        }

        let order = sorted_back_to_front(&depths);
        let mut raw = vec![InstanceRaw::zeroed(); order.len()];
        self.kinds.reserve(order.len());
        for (slot, &src) in order.iter().enumerate() {
            let inst = translucent[src];
            raw[slot] = InstanceRaw::pack(origin, inst);
            self.kinds.push(inst.mesh.index());
        }

        let needed = raw.len();
        if self.instances.is_none() || self.instance_capacity < needed {
            let capacity = needed.next_power_of_two().max(16);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("translucent-instances"),
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

impl RenderNode for TranslucentNode {
    fn name(&self) -> &'static str {
        "translucent"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync(gpu, frame);
        if self.kinds.is_empty() {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("translucent"),
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
                    // No write (depth_write off) — keep the opaque depth intact.
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
        self.prim.draw_sorted(&mut pass, instances, &self.kinds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_farthest_first_stable_on_ties() {
        // depths: farthest (10) first, nearest (1) last; the two 5.0 ties keep
        // their original index order (3 before 4).
        let depths = [5.0f32, 1.0, 10.0, 5.0, 5.0];
        assert_eq!(sorted_back_to_front(&depths), vec![2, 0, 3, 4, 1]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(sorted_back_to_front(&[]).is_empty());
    }

    #[test]
    fn nan_depths_are_ordered_deterministically() {
        // total_cmp gives a total order, so NaN doesn't panic and the result is
        // stable/deterministic (NaN sorts as the largest under total_cmp).
        let depths = [1.0f32, f32::NAN, 2.0];
        let a = sorted_back_to_front(&depths);
        let b = sorted_back_to_front(&depths);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
    }
}
