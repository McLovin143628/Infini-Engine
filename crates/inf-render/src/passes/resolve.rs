//! Explicit MSAA resolve node: an empty pass over the MSAA color target with
//! `resolve_target` set — resolution happens at pass end. Kept separate so
//! later passes (gizmos) can keep drawing MSAA without every earlier node
//! caring which one is "last".

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;

pub struct ResolveNode;

impl RenderNode for ResolveNode {
    fn name(&self) -> &'static str {
        "resolve"
    }

    fn run(&mut self, _gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: Some(&frame.targets.scene_color),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    // The MSAA contents are dead after the resolve.
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
}
