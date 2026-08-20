//! Render-graph skeleton: an ordered list of nodes recording into one command
//! encoder against shared frame data. Deliberately minimal for Phase 2 — no
//! automatic resource lifetimes or pass reordering yet — but every pass added
//! from here on is a node, so the growth path (dependency-declared resources,
//! culling, async compute) never requires rewriting passes.

use crate::gpu::GpuContext;
use crate::renderer::FrameData;

/// A pass in the graph.
///
/// `: Any` is what lets [`RenderGraph::node_mut`] hand a caller one node back by
/// type. P28.2 needs it for exactly one reason: the **cluster page-in** has to
/// happen at the frame's sync point, beside the virtual texture's, because the
/// two halves of one transaction cannot be seated at two different depths of the
/// frame — and the meshlet streamer lives inside a node. The alternative was a
/// shared `Mutex` around the streamer, which is a lock over a value only one
/// thread ever touches, standing in for a borrow the compiler can already prove.
pub trait RenderNode: std::any::Any {
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

    /// The first node of type `T`, mutably.
    ///
    /// There is exactly one of each node type in the graph and the registration
    /// order in `EngineRenderer::new` is the pass order, so "the first" is "the
    /// one". `None` when the graph was built without it, which is a legitimate
    /// configuration and not an error.
    pub fn node_mut<T: RenderNode>(&mut self) -> Option<&mut T> {
        self.nodes
            .iter_mut()
            .find_map(|n| (n.as_mut() as &mut dyn std::any::Any).downcast_mut::<T>())
    }

    /// The first node of type `T`, immutably — [`node_mut`](Self::node_mut)'s
    /// read-only twin, for the accessors that publish a node's counters.
    pub fn node<T: RenderNode>(&self) -> Option<&T> {
        self.nodes
            .iter()
            .find_map(|n| (n.as_ref() as &dyn std::any::Any).downcast_ref::<T>())
    }

    /// Run every node in registration order.
    ///
    /// `timer` is the island-wave-I4 GPU stopwatch, and this is the **one seam**
    /// where per-pass timing enters the engine: a timestamp is written after each
    /// node, beside the `tracing` span that has bracketed them since Phase 2, so
    /// no pass ever learns that it is being measured. `None` — which is every
    /// shipped frame — makes this identical to the loop it replaced, command for
    /// command.
    pub fn run(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameData,
        mut timer: Option<&mut crate::timing::FrameTimer>,
    ) {
        for node in &mut self.nodes {
            let _span = tracing::trace_span!("render_node", name = node.name()).entered();
            node.run(gpu, encoder, frame);
            if let Some(t) = timer.as_deref_mut() {
                t.mark(encoder, node.name());
            }
        }
    }

    /// Every node's name, in run order — what a gate compares a per-pass report
    /// against so a renamed or dropped pass is a red test rather than a line
    /// missing from a diagnostic.
    pub fn names(&self) -> Vec<&'static str> {
        self.nodes.iter().map(|n| n.name()).collect()
    }
}
