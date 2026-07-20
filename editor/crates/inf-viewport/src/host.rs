//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use glam::Vec3;
use inf_math::FloatingOrigin;
use inf_render::{EngineRenderer, GpuContext, RenderScene, RenderView, SurfaceChain};

use crate::camera::EditorCamera;
use crate::SurfaceTarget;

pub struct EngineHost {
    target: SurfaceTarget,
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    pub scene: RenderScene,
    pub origin: FloatingOrigin,
}

impl EngineHost {
    pub fn new(target: SurfaceTarget, width: u32, height: u32) -> Result<Self, String> {
        let (gpu, chain, renderer) = Self::build_gpu_stack(target, width, height)?;
        Ok(Self {
            target,
            gpu,
            chain,
            renderer,
            scene: crate::scene_demo::build(),
            origin: FloatingOrigin::default(),
        })
    }

    fn build_gpu_stack(
        target: SurfaceTarget,
        width: u32,
        height: u32,
    ) -> Result<(GpuContext, SurfaceChain, EngineRenderer), String> {
        let instance = inf_render::create_instance();
        // SAFETY: the native handle outlives the host (the platform module
        // destroys the host before the window/layer).
        let surface = unsafe { target.create_surface(&instance) }?;
        let gpu = GpuContext::for_surface(instance, &surface)?;
        let chain = SurfaceChain::new(&gpu, surface, width, height)?;
        let renderer = EngineRenderer::new(&gpu, chain.target_format());
        Ok((gpu, chain, renderer))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.chain.request_resize(width, height);
    }

    /// Render one frame from `camera`'s point of view. Handles floating-origin
    /// rebases and crash-safe device-lost recovery internally; only errors
    /// that survive a full stack rebuild are returned.
    pub fn render_frame(&mut self, camera: &EditorCamera) -> Result<(), String> {
        if self.gpu.is_lost() {
            tracing::warn!("inf-viewport: device lost — rebuilding GPU stack");
            let (w, h) = self.chain.requested_size();
            let (gpu, chain, renderer) = Self::build_gpu_stack(self.target, w, h)?;
            self.gpu = gpu;
            self.chain = chain;
            self.renderer = renderer;
        }

        self.origin.maybe_rebase(camera.pos);

        // Per-frame debug primitives: world-origin axes tripod.
        self.scene.debug.clear();
        self.scene
            .debug
            .axes(self.origin.to_render(glam::DVec3::ZERO), 1.0);

        let (width, height) = self.chain.requested_size();
        let view = RenderView {
            origin: self.origin,
            eye_world: camera.pos,
            forward: camera.forward(),
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width,
            height,
        };

        let Some(frame) = self.chain.acquire(&self.gpu) else {
            return Ok(()); // transient (occluded/timeout) — skip the frame
        };
        let out_view = self.chain.target_view(&frame);
        self.renderer.render(
            &self.gpu,
            &self.scene,
            &view,
            &out_view,
            self.chain.configured_size(),
        );
        self.gpu.queue.present(frame);
        Ok(())
    }
}
