//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use glam::{DVec3, Vec2, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    gizmo, EngineRenderer, GizmoDelta, GizmoDrag, GizmoMode, GpuContext, Picker, RenderScene,
    RenderView, SurfaceChain,
};

use crate::camera::EditorCamera;
use crate::SurfaceTarget;

pub struct EngineHost {
    target: SurfaceTarget,
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    // Only the Windows input layer picks today; macOS input lands with its
    // hardware pass (kept constructed so the field is ready when it does).
    #[cfg_attr(not(windows), allow(dead_code))]
    picker: Picker,
    pub scene: RenderScene,
    pub origin: FloatingOrigin,
    /// Active transform-gizmo mode; the gizmo shows only with a selection.
    pub gizmo_mode: GizmoMode,
    gizmo_drag: Option<GizmoDrag>,
    fov_y: f32,
}

impl EngineHost {
    pub fn new(target: SurfaceTarget, width: u32, height: u32) -> Result<Self, String> {
        let (gpu, chain, renderer) = Self::build_gpu_stack(target, width, height)?;
        let picker = Picker::new(&gpu);
        Ok(Self {
            target,
            gpu,
            chain,
            renderer,
            picker,
            scene: crate::scene_demo::build(),
            origin: FloatingOrigin::default(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_drag: None,
            fov_y: 60f32.to_radians(),
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

    /// The render view for `camera` at the current surface size.
    fn view_for(&self, camera: &EditorCamera) -> RenderView {
        let (width, height) = self.chain.requested_size();
        RenderView {
            origin: self.origin,
            eye_world: camera.pos,
            forward: camera.forward(),
            up: Vec3::Y,
            fov_y: self.fov_y,
            near: 0.05,
            width,
            height,
        }
    }

    /// World-space center of the current selection, if any.
    fn selection_center(&self) -> Option<DVec3> {
        let sel = &self.scene.selected;
        if sel.is_empty() {
            return None;
        }
        let mut sum = DVec3::ZERO;
        let mut n = 0.0;
        for id in sel {
            if let Some(inst) = self.scene.instances.iter().find(|i| i.id == *id) {
                sum += inst.translation;
                n += 1.0;
            }
        }
        (n > 0.0).then(|| sum / n)
    }
}

/// The pointer-driven interaction API (select, hover, gizmo drag). Currently
/// only the Windows input layer (`win32.rs`) calls into it; macOS input is not
/// wired yet (the camera holds its default pose), so on non-Windows these are
/// legitimately unused until the macOS hardware pass drives them.
#[cfg_attr(not(windows), allow(dead_code))]
impl EngineHost {
    /// Set the hovered instance (drives the weak outline). `None` clears it.
    pub fn set_hover(&mut self, camera: &EditorCamera, px: u32, py: u32) {
        // Don't recompute hover mid-drag (keeps the outline stable).
        if self.gizmo_drag.is_some() {
            return;
        }
        let view = self.view_for(camera);
        self.scene.hovered = self.picker.pick(&self.gpu, &self.scene, &view, px, py);
    }

    /// Click-select: pick the object under the cursor. `additive` extends the
    /// selection (Ctrl-click); otherwise it replaces. A click on empty space
    /// clears the selection (non-additive).
    pub fn select_at(&mut self, camera: &EditorCamera, px: u32, py: u32, additive: bool) {
        let view = self.view_for(camera);
        let hit = self.picker.pick(&self.gpu, &self.scene, &view, px, py);
        match hit {
            Some(id) => {
                if additive {
                    if let Some(pos) = self.scene.selected.iter().position(|s| *s == id) {
                        self.scene.selected.remove(pos);
                    } else {
                        self.scene.selected.push(id);
                    }
                } else {
                    self.scene.selected = vec![id];
                }
            }
            None if !additive => self.scene.selected.clear(),
            None => {}
        }
    }

    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    /// Focus target for the current selection: its center and a radius that
    /// bounds every selected object. `None` when nothing is selected.
    pub fn selection_focus(&self) -> Option<(DVec3, f64)> {
        let center = self.selection_center()?;
        let mut radius: f64 = 1.0;
        for id in &self.scene.selected {
            if let Some(inst) = self.scene.instances.iter().find(|i| i.id == *id) {
                let extent = inst.scale.abs().max_element() as f64;
                radius = radius.max((inst.translation - center).length() + extent);
            }
        }
        Some((center, radius))
    }

    /// If the cursor is over a gizmo handle, begin a drag and return true.
    pub fn try_begin_gizmo(&mut self, camera: &EditorCamera, px: u32, py: u32) -> bool {
        let Some(center) = self.selection_center() else {
            return false;
        };
        let view = self.view_for(camera);
        let origin_local = self.origin.to_render(center);
        let size = gizmo::gizmo_world_size(origin_local, view.eye_local(), self.fov_y);
        let cursor = Vec2::new(px as f32, py as f32);
        let Some(axis) = gizmo::pick_axis(
            self.gizmo_mode,
            origin_local,
            size,
            view.view_proj(),
            cursor,
            view.width as f32,
            view.height as f32,
        ) else {
            return false;
        };
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        self.gizmo_drag = Some(GizmoDrag::begin(
            self.gizmo_mode,
            axis,
            origin_local,
            ro,
            rd,
        ));
        true
    }

    pub fn is_dragging_gizmo(&self) -> bool {
        self.gizmo_drag.is_some()
    }

    /// Apply a gizmo drag update from the cursor. `snap` > 0 quantizes.
    pub fn update_gizmo(&mut self, camera: &EditorCamera, px: u32, py: u32, snap: f32) {
        let Some(drag) = self.gizmo_drag else {
            return;
        };
        let view = self.view_for(camera);
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let delta = drag.update(ro, rd, snap);
        self.apply_delta(delta, drag.origin);
        // Re-anchor the drag so deltas are incremental frame-to-frame.
        if let Some(d) = self.gizmo_drag.as_mut() {
            *d = GizmoDrag::begin(d.mode, d.axis, d.origin, ro, rd);
        }
    }

    fn apply_delta(&mut self, delta: GizmoDelta, pivot_local: Vec3) {
        let pivot = self.origin.to_world(pivot_local);
        let selected = self.scene.selected.clone();
        for id in &selected {
            if let Some(inst) = self.scene.instances.iter_mut().find(|i| i.id == *id) {
                match delta {
                    GizmoDelta::Translate(t) => inst.translation += t,
                    GizmoDelta::Rotate { axis, radians } => {
                        let q = glam::Quat::from_axis_angle(axis, radians);
                        inst.rotation = q * inst.rotation;
                        // Orbit the translation about the pivot too.
                        let rel = (inst.translation - pivot).as_vec3();
                        inst.translation = pivot + (q * rel).as_dvec3();
                    }
                    GizmoDelta::Scale(s) => inst.scale *= s,
                }
            }
        }
        self.scene.mark_dirty();
    }

    pub fn end_gizmo(&mut self) {
        self.gizmo_drag = None;
    }
}

impl EngineHost {
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

        let view = self.view_for(camera);

        // Per-frame debug primitives: world-origin axes tripod, plus the
        // transform gizmo at the selection center (screen-constant size).
        self.scene.debug.clear();
        self.scene
            .debug
            .axes(self.origin.to_render(glam::DVec3::ZERO), 1.0);
        if let Some(center) = self.selection_center() {
            let origin_local = self.origin.to_render(center);
            let size = gizmo::gizmo_world_size(origin_local, view.eye_local(), self.fov_y);
            let active = self.gizmo_drag.map(|d| d.axis);
            gizmo::build_geometry(
                &mut self.scene.debug,
                self.gizmo_mode,
                origin_local,
                size,
                active,
            );
        }

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
