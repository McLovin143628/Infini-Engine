//! Render-graph skeleton: an ordered list of nodes recording into one command
//! encoder against shared frame data. Deliberately minimal for Phase 2 — no
//! automatic resource lifetimes or pass reordering yet — but every pass added
//! from here on is a node, so the growth path (dependency-declared resources,
//! culling, async compute) never requires rewriting passes.

use crate::gpu::GpuContext;
use crate::renderer::FrameData;

pub trait RenderNode {
    fn name(&self) -> &'static str;
    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData);
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraph {
    pub fn add(&mut self, node: impl RenderNode + 'static) {
        self.nodes.push(Box::new(node));
    }

    pub fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        for node in &mut self.nodes {
            let _span = tracing::trace_span!("render_node", name = node.name()).entered();
            node.run(gpu, encoder, frame);
        }
    }
}
