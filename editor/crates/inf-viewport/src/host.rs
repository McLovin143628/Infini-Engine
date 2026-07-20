//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::HashMap;

use glam::{DVec3, Vec2, Vec3};
use inf_ecs::components::{ComputedVisibility, GlobalTransform, Material, MeshRef};
use inf_ecs::{Transform as EcsTransform, Vec3d};
use inf_editor_core::scene::SceneDoc;
use inf_math::FloatingOrigin;
use inf_render::{
    gizmo, EngineRenderer, GizmoDelta, GizmoDrag, GizmoMode, GpuContext, MeshInstance, Picker,
    RenderScene, RenderView, SurfaceChain,
};
use uuid::Uuid;

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
    /// Render-instance id → entity GUID, rebuilt each projection (P3.2). Lets a
    /// pick resolve to a scene entity and a gizmo write back to the document.
    id_to_guid: HashMap<u32, Uuid>,
    guid_to_id: HashMap<Uuid, u32>,
    /// Document version the current projection reflects (skip redundant rebuilds).
    synced_version: Option<u64>,
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
            scene: RenderScene {
                grid_enabled: true,
                ..Default::default()
            },
            origin: FloatingOrigin::default(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_drag: None,
            fov_y: 60f32.to_radians(),
            id_to_guid: HashMap::new(),
            guid_to_id: HashMap::new(),
            synced_version: None,
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

    /// Rebuild the render projection from the shared document when it changed
    /// (P3.2). Renderable entities (those with a `MeshRef`) become instances;
    /// the id↔GUID maps let picks and gizmo writeback cross back to the world.
    /// Skipped mid-drag so an in-flight gizmo edit isn't clobbered.
    pub fn sync_from_doc(&mut self, doc: &SceneDoc) {
        if self.gizmo_drag.is_some() {
            return;
        }
        let version = doc.version();
        if self.synced_version == Some(version) {
            return;
        }
        self.synced_version = Some(version);
        self.rebuild_scene(doc);
    }

    fn rebuild_scene(&mut self, doc: &SceneDoc) {
        self.scene.instances.clear();
        self.id_to_guid.clear();
        self.guid_to_id.clear();

        let world = doc.world();
        let w = world.world();
        let mut next_id: u32 = 1;
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            if w.get::<MeshRef>(entity).is_none() {
                continue; // only meshes render (lights/cameras: Phase 4 icons)
            }
            let visible = w
                .get::<ComputedVisibility>(entity)
                .map(|c| c.0)
                .unwrap_or(true);
            if !visible {
                continue;
            }
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, translation) = affine.to_scale_rotation_translation();
            let color = w
                .get::<Material>(entity)
                .map(|m| m.base_color.to_array())
                .unwrap_or([0.8, 0.8, 0.8, 1.0]);
            let id = next_id;
            next_id += 1;
            self.scene.instances.push(MeshInstance {
                translation,
                rotation: rot.as_quat(),
                scale: scale.as_vec3(),
                color,
                id,
            });
            self.id_to_guid.insert(id, guid);
            self.guid_to_id.insert(guid, id);
        }

        // Selection outline mirrors the document's selection.
        self.scene.selected = doc
            .selection()
            .iter()
            .filter_map(|g| self.guid_to_id.get(g).copied())
            .collect();
        self.scene.hovered = None;
        self.scene.mark_dirty();
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

    /// Pick the entity GUID under the cursor (`None` = empty space). Selection
    /// itself lives in the document — the caller applies the pick to it.
    pub fn pick_guid(&mut self, camera: &EditorCamera, px: u32, py: u32) -> Option<Uuid> {
        let view = self.view_for(camera);
        let id = self.picker.pick(&self.gpu, &self.scene, &view, px, py)?;
        self.id_to_guid.get(&id).copied()
    }

    /// World-space transforms of the current selection after a gizmo drag, keyed
    /// by GUID — the caller writes them back to the document as one undo entry.
    /// (Local == world for the roots/identity-parent objects the gizmo edits;
    /// full parent-relative solve lands with nested transforms.)
    pub fn selected_world_transforms(&self) -> Vec<(Uuid, EcsTransform)> {
        self.scene
            .selected
            .iter()
            .filter_map(|id| {
                let guid = self.id_to_guid.get(id)?;
                let inst = self.scene.instances.iter().find(|i| i.id == *id)?;
                let mut t = EcsTransform::from_translation(inst.translation);
                t.set_quat(inst.rotation.as_dquat());
                t.scale = Vec3d::from_dvec3(inst.scale.as_dvec3());
                Some((*guid, t))
            })
            .collect()
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
