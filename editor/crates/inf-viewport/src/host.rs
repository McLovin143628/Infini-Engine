//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::HashMap;

use glam::{DVec3, Vec2, Vec3};
use inf_ecs::components::{
    ComputedVisibility, GlobalTransform, Light, Light2D, LightKind as EcsLightKind, Material,
    MeshRef, NineSlice, Sprite, Text2D, TextAlign, Tilemap,
};
use inf_ecs::{Transform as EcsTransform, Vec3d};
use inf_editor_core::scene::SceneDoc;
use inf_math::FloatingOrigin;
use inf_render::{
    expand_nine_slice, expand_text, gizmo, handle_from_guid, EngineRenderer, GizmoDelta, GizmoDrag,
    GizmoMode, GpuContext, HAlign, LightKind, MeshInstance, NineSliceParams, Picker, PrebatchedRun,
    RenderChunk, RenderLight, RenderLight2D, RenderScene, RenderTilemap, RenderView,
    SpriteInstance, SurfaceChain, TextParams, TilemapParams, BUILTIN_FONT_TEXTURE,
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
        self.scene.lights.clear();
        self.scene.sprites.clear();
        self.scene.tilemaps.clear();
        self.scene.prebatched.clear();
        self.scene.lights_2d.clear();
        self.id_to_guid.clear();
        self.guid_to_id.clear();

        let world = doc.world();
        let w = world.world();
        let mut next_id: u32 = 1;
        for &guid in doc.order() {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let visible = w
                .get::<ComputedVisibility>(entity)
                .map(|c| c.0)
                .unwrap_or(true);

            // Lights project into the renderer's light list (P7.1).
            if let Some(light) = w.get::<Light>(entity) {
                if visible {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    self.scene.lights.push(project_light(light, &affine));
                }
            }

            // Sprites project into the 2D sprite list (P8.1a). A sprite entity
            // usually has no MeshRef, so this happens before the mesh gate.
            if let Some(sprite) = w.get::<Sprite>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene.sprites.push(project_sprite(sprite, translation));
                }
            }

            // 2D lights project into the sprite pass's light list (P8.1c).
            if let Some(light2d) = w.get::<Light2D>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .lights_2d
                        .push(project_light2d(light2d, translation));
                }
            }

            // 9-slices expand to nine quads (P8.1c), pushed as one prebatched run.
            if let Some(nine) = w.get::<NineSlice>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .prebatched
                        .push(project_nine_slice(nine, translation));
                }
            }

            // Text blocks expand to one quad per glyph (P8.1c), one prebatched run.
            if let Some(text) = w.get::<Text2D>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    if let Some(run) = project_text(text, translation) {
                        self.scene.prebatched.push(run);
                    }
                }
            }

            // Tilemaps project into the 2D tilemap list (P8.1b); the sprite pass
            // culls + expands their chunks each frame.
            if let Some(tilemap) = w.get::<Tilemap>(entity) {
                if visible && !tilemap.is_empty() {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    self.scene
                        .tilemaps
                        .push(project_tilemap(tilemap, translation));
                }
            }

            if w.get::<MeshRef>(entity).is_none() {
                continue; // only meshes become draw instances
            }
            if !visible {
                continue;
            }
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, translation) = affine.to_scale_rotation_translation();
            let (color, metallic, roughness, emissive) = w
                .get::<Material>(entity)
                .map(|m| {
                    let e = m.emissive.to_array();
                    (
                        m.base_color.to_array(),
                        m.metallic,
                        m.roughness,
                        [e[0], e[1], e[2]],
                    )
                })
                .unwrap_or(([0.8, 0.8, 0.8, 1.0], 0.0, 0.5, [0.0; 3]));
            let id = next_id;
            next_id += 1;
            self.scene.instances.push(MeshInstance {
                translation,
                rotation: rot.as_quat(),
                scale: scale.as_vec3(),
                color,
                metallic,
                roughness,
                emissive,
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

/// Project an ECS `Light` (+ its world transform) into a renderer light. Spot is
/// approximated as point until P11 adds cones.
fn project_light(light: &Light, affine: &glam::DAffine3) -> RenderLight {
    let (_, rot, translation) = affine.to_scale_rotation_translation();
    let c = light.color.to_array();
    let color = [c[0], c[1], c[2]];
    match light.kind {
        EcsLightKind::Directional => RenderLight {
            kind: LightKind::Directional,
            color,
            intensity: light.intensity,
            // Direction *toward* the light: the transform's +Z (emission is −Z).
            direction: (rot * DVec3::Z).as_vec3(),
            position: DVec3::ZERO,
            range: 0.0,
        },
        EcsLightKind::Point | EcsLightKind::Spot => RenderLight {
            kind: LightKind::Point,
            color,
            intensity: light.intensity,
            direction: Vec3::ZERO,
            position: translation,
            range: 0.0,
        },
    }
}

/// Project an ECS [`Sprite`] (+ its world position) into a renderer sprite.
///
/// The texture GUID maps to a `TextureHandle`, but the viewport thread has no
/// asset-DB access yet, so no RGBA bytes are pushed to
/// `RenderScene::pending_texture_uploads` — referenced sprites render as the
/// renderer's white fallback tinted by `color` (a colored quad). Resolving the
/// texture bytes in the viewport is the same documented follow-up as rendering
/// imported mesh geometry (both need the asset DB threaded into the viewport;
/// the headless golden test exercises the full textured path). Rotation is left
/// at 0 for P8.1a (2D rotation tooling arrives in P8.2).
fn project_sprite(sprite: &Sprite, translation: DVec3) -> SpriteInstance {
    SpriteInstance {
        position: translation,
        size: Vec2::new(sprite.size.x as f32, sprite.size.y as f32),
        pivot: Vec2::new(sprite.pivot.x as f32, sprite.pivot.y as f32),
        rotation: 0.0,
        uv_min: Vec2::new(
            sprite.atlas_rect.min.x as f32,
            sprite.atlas_rect.min.y as f32,
        ),
        uv_max: Vec2::new(
            sprite.atlas_rect.max.x as f32,
            sprite.atlas_rect.max.y as f32,
        ),
        color: sprite.color.to_array(),
        texture: sprite
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        sorting_layer: sprite.sorting_layer,
        order: sprite.order,
        flip_x: sprite.flip_x,
        flip_y: sprite.flip_y,
    }
}

/// Project an ECS [`Tilemap`] (+ its world position) into a [`RenderTilemap`].
///
/// The atlas texture GUID maps to a `TextureHandle`, but — like [`project_sprite`]
/// — the viewport thread has no asset-DB access yet, so no RGBA bytes are pushed:
/// referenced tilemaps render as the white fallback tinted by `tint` (colored
/// cells). The headless golden test exercises the full textured path. The chunk
/// data is copied out of the sparse ECS store once per document version; the
/// sprite pass culls + expands it per frame.
fn project_tilemap(tilemap: &Tilemap, translation: DVec3) -> RenderTilemap {
    let params = TilemapParams {
        origin: translation,
        tile_size: Vec2::new(tilemap.tile_size.x as f32, tilemap.tile_size.y as f32),
        atlas_cols: tilemap.atlas_cols,
        atlas_rows: tilemap.atlas_rows,
        texture: tilemap
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        color: tilemap.tint.to_array(),
        sorting_layer: tilemap.sorting_layer,
        order: tilemap.order,
    };
    let chunks = tilemap
        .occupied_chunks()
        .map(|(&coord, chunk)| RenderChunk {
            coord,
            tiles: chunk.tiles().to_vec(),
        })
        .collect();
    RenderTilemap { params, chunks }
}

/// Project an ECS [`Light2D`] (+ world position) into a renderer 2D light.
fn project_light2d(light: &Light2D, translation: DVec3) -> RenderLight2D {
    let c = light.color.to_array();
    RenderLight2D {
        color: [c[0], c[1], c[2]],
        intensity: light.intensity,
        radius: light.radius,
        position: translation,
    }
}

/// Project an ECS [`NineSlice`] (+ world position) into a prebatched run of nine
/// cell quads centered on the entity. Like [`project_sprite`], the texture GUID
/// maps to a handle but no bytes are uploaded from the viewport thread yet
/// (referenced panels render as the tinted white fallback; the headless golden
/// exercises the textured path).
fn project_nine_slice(nine: &NineSlice, translation: DVec3) -> PrebatchedRun {
    let params = NineSliceParams {
        position: translation,
        pivot: Vec2::splat(0.5),
        size: Vec2::new(nine.size.x as f32, nine.size.y as f32),
        border_uv: [
            nine.border_uv[0] as f32,
            nine.border_uv[1] as f32,
            nine.border_uv[2] as f32,
            nine.border_uv[3] as f32,
        ],
        border_world: Vec2::new(nine.border_world.x as f32, nine.border_world.y as f32),
        color: nine.tint.to_array(),
        texture: nine
            .texture
            .map(|u| handle_from_guid(u.as_u128()))
            .unwrap_or(inf_render::WHITE_TEXTURE),
        sorting_layer: nine.sorting_layer,
        order: nine.order,
    };
    let instances = expand_nine_slice(&params).to_vec();
    PrebatchedRun {
        texture: params.texture,
        sorting_layer: params.sorting_layer,
        order: params.order,
        instances,
    }
}

/// Project an ECS [`Text2D`] (+ world position) into a prebatched run of glyph
/// quads. A `None` font asset resolves to the renderer's built-in 8×8 bitmap
/// font ([`BUILTIN_FONT_TEXTURE`], always uploaded by the sprite pass). Returns
/// `None` when the string produces no glyphs (nothing to draw).
fn project_text(text: &Text2D, translation: DVec3) -> Option<PrebatchedRun> {
    let texture = text
        .font_texture
        .map(|u| handle_from_guid(u.as_u128()))
        .unwrap_or(BUILTIN_FONT_TEXTURE);
    let halign = match text.halign {
        TextAlign::Left => HAlign::Left,
        TextAlign::Center => HAlign::Center,
        TextAlign::Right => HAlign::Right,
    };
    let params = TextParams {
        position: translation,
        text: &text.text,
        glyph_cols: text.glyph_cols,
        glyph_rows: text.glyph_rows,
        first_codepoint: text.first_codepoint,
        glyph_size: Vec2::new(text.glyph_size.x as f32, text.glyph_size.y as f32),
        tracking: text.tracking as f32,
        color: text.tint.to_array(),
        texture,
        sorting_layer: text.sorting_layer,
        order: text.order,
        halign,
    };
    let instances = expand_text(&params);
    if instances.is_empty() {
        return None;
    }
    Some(PrebatchedRun {
        texture,
        sorting_layer: text.sorting_layer,
        order: text.order,
        instances,
    })
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
