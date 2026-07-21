//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::HashMap;

use glam::{DQuat, DVec2, DVec3, Vec2, Vec3};
use inf_ecs::components::{
    Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind, ComputedVisibility,
    GlobalTransform, Light, Light2D, LightKind as EcsLightKind, Material, MeshRef, NineSlice,
    PcgVolume, Sprite, Terrain, Text2D, TextAlign, Tilemap,
};
use inf_ecs::{Transform as EcsTransform, Vec3d};
use inf_editor_core::scene::SceneDoc;
use inf_math::FloatingOrigin;
use inf_render::{
    collider_outline_2d, collider_outline_3d, expand_nine_slice, expand_text, gizmo,
    handle_from_guid, ColliderOutline2D, ColliderOutline3D, DebugDraw, EngineRenderer, GizmoDelta,
    GizmoDrag, GizmoMode, GpuContext, HAlign, LightKind, MeshInstance, NineSliceParams,
    OrthoParams, Picker, PrebatchedRun, RenderChunk, RenderLight, RenderLight2D, RenderScene,
    RenderTerrain, RenderTerrainTile, RenderTilemap, RenderView, SpriteInstance, SurfaceChain,
    TextParams, TilemapParams, BUILTIN_FONT_TEXTURE,
};
use uuid::Uuid;

use inf_terrain::{
    dab_positions, raycast_terrain, BrushOp, BrushParams, Falloff, FlattenTarget, Stroke,
    TerrainData,
};

use crate::camera::{
    Camera2D, EditorCamera, SculptFalloff, SculptOp, SculptSettings, Snap2DSettings, ToolMode,
    ViewportMode, TWO_D_FAR, TWO_D_NEAR,
};
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
    /// World-space 2D collider outlines by GUID (P8.3b), rebuilt each projection.
    /// Rendered as debug lines for the current selection only.
    collider_outlines: HashMap<Uuid, ColliderDebug>,
    /// World-space 3D collider wireframes by GUID (P9.1), rebuilt each projection.
    /// Rendered as debug lines for the current selection only.
    collider_outlines_3d: HashMap<Uuid, ColliderDebug3D>,
    /// GUIDs of the document's current selection (for collider debug draw —
    /// covers every selected entity, not just the mesh instances).
    selected_guids: Vec<Uuid>,
    /// Document version the current projection reflects (skip redundant rebuilds).
    synced_version: Option<u64>,
    /// Active projection: perspective (3D) or orthographic (2D editor). Drives
    /// the gizmo handle set and the grid plane; the camera itself lives in the
    /// platform loop (which keeps a separate pose per mode). (P8.2c)
    pub mode: ViewportMode,
    /// 2D-mode snapping config pushed from the toolbar (P8.2c). Only the Windows
    /// input layer reads it (via [`EngineHost::snap_2d_translate`]).
    #[cfg_attr(not(windows), allow(dead_code))]
    snap_2d: Snap2DSettings,
    /// Working transforms of selected **non-mesh** entities (sprites, text, …)
    /// for the gizmo: captured from the document on each projection, mutated
    /// during a drag, written back on release. Mesh entities use the render
    /// instances instead (`scene.instances`). (P8.2c)
    selected_2d: HashMap<Uuid, Sel2D>,
    /// Active tool: pick/gizmo (`Select`) or terrain sculpt (`Sculpt`). (P10.2b)
    tool_mode: ToolMode,
    /// Sculpt brush configuration pushed from the toolbar (P10.2b).
    sculpt: SculptSettings,
    /// GUID of the terrain entity the sculpt tool targets (the first visible,
    /// non-empty terrain — matches `scene.terrain`). Set each projection.
    terrain_guid: Option<Uuid>,
    /// In-flight sculpt stroke: the accumulating brush gesture (`None` = idle).
    sculpt_drag: Option<SculptDrag>,
    /// World-space brush-ring loop points (following terrain height), rebuilt as
    /// the cursor hovers/sculpts terrain; drawn as debug lines in Sculpt mode.
    sculpt_ring: Vec<DVec3>,
    /// Colour of the brush ring (encodes the active op).
    sculpt_ring_color: [f32; 4],
    /// Camera eye captured on the last rendered frame (P10.5b). PCG scatter
    /// instances are draw-distance-culled against it at projection time; because
    /// projection is doc-version-gated (not per-frame), the cull set refreshes
    /// whenever the document changes (a `pcg_evaluate` bumps the version) rather
    /// than continuously as the camera moves — a documented v1 simplification.
    last_eye_world: DVec3,
}

/// An in-flight sculpt gesture (P10.2b): the mouse-down→up stroke accumulating
/// dabs into one [`Stroke`], plus the state to resample the drag path and, on
/// release, commit one [`inf_terrain::HeightDelta`] undo step.
struct SculptDrag {
    /// Target terrain entity.
    guid: Uuid,
    /// The accumulating stroke (merged into one delta at commit).
    stroke: Stroke,
    /// The effective op (Ctrl may flip Raise↔Lower).
    op: SculptOp,
    /// Last dab centre in terrain-local XZ (for even path resampling).
    last_local: DVec2,
    /// Local surface height under the stroke's first touch — the Flatten target.
    flatten_height: f64,
}

/// A selected 2D (non-mesh) entity's working transform for the gizmo. World
/// space; mirrors what a mesh instance carries so the writeback path is uniform.
/// Only `translation` is read off Windows (the selection center); the rest feed
/// the gizmo writeback, which is Windows-input-only for now.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
struct Sel2D {
    translation: DVec3,
    rotation: DQuat,
    scale: DVec3,
    /// Half-size estimate (world units) for the focus radius.
    extent: f64,
}

/// A selected entity's collider, resolved to world space for debug outlining.
struct ColliderDebug {
    shape: ColliderOutline2D,
    /// Collider offset in the body frame (world units, XY).
    offset: Vec2,
    /// Entity world translation (Z kept so the outline sits in the sprite plane).
    world_pos: DVec3,
    /// Z rotation of the body (radians).
    z_rot: f64,
}

/// A selected entity's 3D collider, resolved to world space for debug outlining.
struct ColliderDebug3D {
    shape: ColliderOutline3D,
    /// Collider offset in the body frame (world units).
    offset: DVec3,
    /// Entity world translation.
    world_pos: DVec3,
    /// Full world orientation of the body.
    rotation: DQuat,
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
            collider_outlines: HashMap::new(),
            collider_outlines_3d: HashMap::new(),
            selected_guids: Vec::new(),
            synced_version: None,
            mode: ViewportMode::Perspective,
            snap_2d: Snap2DSettings::default(),
            selected_2d: HashMap::new(),
            tool_mode: ToolMode::Select,
            sculpt: SculptSettings::default(),
            terrain_guid: None,
            sculpt_drag: None,
            sculpt_ring: Vec::new(),
            sculpt_ring_color: [1.0; 4],
            last_eye_world: DVec3::ZERO,
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

    /// The perspective render view for `camera` at the current surface size.
    pub fn view_for(&self, camera: &EditorCamera) -> RenderView {
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
            ortho: None,
        }
    }

    /// The orthographic render view for the 2D camera at the current surface
    /// size: eye above the XY plane looking down -Z, up = +Y, reverse-Z ortho.
    pub fn view_2d(&self, cam: &Camera2D) -> RenderView {
        let (width, height) = self.chain.requested_size();
        RenderView {
            origin: self.origin,
            eye_world: cam.eye(),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: self.fov_y,
            near: 0.05,
            width,
            height,
            ortho: Some(OrthoParams {
                half_height: cam.half_height as f32,
                near: TWO_D_NEAR,
                far: TWO_D_FAR,
            }),
        }
    }

    /// Current surface size in physical pixels (for camera/gizmo math). Only the
    /// Windows input layer drives the cameras today.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn surface_size(&self) -> (u32, u32) {
        self.chain.requested_size()
    }

    /// Switch the active projection (perspective ↔ 2D ortho). The platform loop
    /// keeps a separate camera pose per mode, so switching preserves both.
    pub fn set_mode(&mut self, mode: ViewportMode) {
        self.mode = mode;
    }

    /// Replace the 2D-mode snapping configuration (from the toolbar).
    pub fn set_snap_2d(&mut self, snap: Snap2DSettings) {
        self.snap_2d = snap;
    }

    /// Switch the active tool (Select ↔ Sculpt) from the toolbar (P10.2b).
    /// Leaving Sculpt drops any hovered brush ring.
    pub fn set_tool_mode(&mut self, mode: ToolMode) {
        self.tool_mode = mode;
        if mode != ToolMode::Sculpt {
            self.sculpt_ring.clear();
        }
    }

    /// Replace the sculpt brush configuration (from the toolbar).
    pub fn set_sculpt(&mut self, sculpt: SculptSettings) {
        self.sculpt = sculpt;
    }

    /// Translate snap increment (world units) for 2D mode, `0.0` ⇒ none. Only
    /// the Windows input layer applies it during a gizmo drag.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn snap_2d_translate(&self) -> f32 {
        self.snap_2d.translate_snap()
    }

    /// World-space center of the current selection, if any. Reads LIVE working
    /// positions (mesh render instances + selected 2D entities) so it tracks a
    /// gizmo drag in progress.
    fn selection_center(&self) -> Option<DVec3> {
        let mut sum = DVec3::ZERO;
        let mut n = 0.0;
        for id in &self.scene.selected {
            if let Some(inst) = self.scene.instances.iter().find(|i| i.id == *id) {
                sum += inst.translation;
                n += 1.0;
            }
        }
        for s in self.selected_2d.values() {
            sum += s.translation;
            n += 1.0;
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
        self.scene.terrain = None;
        self.terrain_guid = None;
        self.id_to_guid.clear();
        self.guid_to_id.clear();
        self.collider_outlines.clear();
        self.collider_outlines_3d.clear();

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

            // Heightfield terrain (P10.1) projects into the render scene's single
            // terrain slot; the terrain pass assembles clipmap LOD rings each
            // frame. First visible, non-empty terrain wins (multi-terrain merge is
            // a follow-up). Version = doc version so height textures re-upload on
            // any document change.
            if self.scene.terrain.is_none() {
                if let Some(terrain) = w.get::<Terrain>(entity) {
                    if visible && !terrain.data.is_empty() {
                        let translation = w
                            .get::<GlobalTransform>(entity)
                            .map(|g| g.translation())
                            .unwrap_or(DVec3::ZERO);
                        self.scene.terrain =
                            Some(project_terrain(terrain, translation, doc.version()));
                        self.terrain_guid = Some(guid);
                    }
                }
            }

            // PCG scatter volumes (P10.5b): project the cached evaluated
            // instances (refreshed on demand by `pcg_evaluate`) into the existing
            // mesh-instance path as placeholder cubes — kind→GUID→real-mesh upload
            // is the same documented viewport gap as sprites/tilemaps, so PCG
            // proves placement/density/orientation with primitives. Draw-distance
            // culled against the last camera eye. A pick on a scattered cube
            // resolves to the volume entity (id→guid), so the volume is selectable
            // by clicking its content.
            if let Some(vol) = w.get::<PcgVolume>(entity) {
                if visible && !vol.evaluated.is_empty() {
                    let dd = vol.draw_distance;
                    for si in &vol.evaluated {
                        if dd > 0.0 && (si.position - self.last_eye_world).length() > dd {
                            continue;
                        }
                        let id = next_id;
                        next_id += 1;
                        self.scene.instances.push(MeshInstance {
                            translation: si.position,
                            rotation: si.rotation.as_quat(),
                            scale: Vec3::splat(si.scale as f32),
                            color: pcg_kind_color(si.kind),
                            metallic: 0.0,
                            roughness: 0.75,
                            emissive: [0.0; 3],
                            id,
                        });
                        // Pick a scattered cube → select the owning volume.
                        self.id_to_guid.insert(id, guid);
                    }
                }
            }

            // 2D colliders cache a world-space outline (P8.3b); drawn as debug
            // lines for the selection only (in `render_frame`). Independent of
            // MeshRef — a collider often sits on a sprite or bare entity.
            if let Some(col) = w.get::<Collider2D>(entity) {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rot, translation) = affine.to_scale_rotation_translation();
                let (_, _, z_rot) = rot.to_euler(glam::EulerRot::YXZ);
                self.collider_outlines
                    .insert(guid, project_collider(col, translation, z_rot));
            }

            // 3D colliders cache a world-space wireframe (P9.1); drawn as debug
            // lines for the selection only, with full body rotation + offset.
            if let Some(col) = w.get::<Collider3D>(entity) {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rotation, translation) = affine.to_scale_rotation_translation();
                self.collider_outlines_3d
                    .insert(guid, project_collider_3d(col, translation, rotation));
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
        // Full selection (any entity) drives the collider debug outlines.
        self.selected_guids = doc.selection().to_vec();

        // Capture working transforms for selected NON-mesh entities (sprites,
        // text, tilemaps, …) so the gizmo can move them in 2D. Mesh entities are
        // covered by their render instances instead.
        self.selected_2d.clear();
        for &guid in doc.selection() {
            if self.guid_to_id.contains_key(&guid) {
                continue;
            }
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rotation, translation) = affine.to_scale_rotation_translation();
            let extent = w
                .get::<Sprite>(entity)
                .map(|s| (s.size.x.max(s.size.y) * 0.5).max(0.25))
                .unwrap_or(0.5);
            self.selected_2d.insert(
                guid,
                Sel2D {
                    translation,
                    rotation,
                    scale,
                    extent,
                },
            );
        }

        self.scene.hovered = None;
        self.scene.mark_dirty();
    }
}

/// Project a [`Collider2D`] (+ its world pose) into a world-space debug outline.
fn project_collider(col: &Collider2D, world_pos: DVec3, z_rot: f64) -> ColliderDebug {
    let shape = match col.shape_kind {
        ColliderShape2DKind::Box => ColliderOutline2D::Box {
            half: Vec2::new(col.half_extents.x as f32, col.half_extents.y as f32),
        },
        ColliderShape2DKind::Circle => ColliderOutline2D::Circle {
            radius: col.radius as f32,
        },
        ColliderShape2DKind::Capsule => ColliderOutline2D::Capsule {
            half_height: col.half_extents.y as f32,
            radius: col.radius as f32,
        },
    };
    ColliderDebug {
        shape,
        offset: Vec2::new(col.offset.x as f32, col.offset.y as f32),
        world_pos,
        z_rot,
    }
}

/// Stroke a collider outline into the debug-line layer, rebasing through the
/// floating origin. Points are generated in the collider's local XY frame,
/// rotated by the body's Z rotation, offset, and lifted onto the entity's world
/// position (Z preserved so the outline sits in the sprite plane).
fn draw_collider_outline(debug: &mut DebugDraw, origin: &FloatingOrigin, cd: &ColliderDebug) {
    const COLLIDER_COLOR: [f32; 4] = [0.30, 0.95, 0.55, 1.0];
    /// Circle/capsule tessellation for the debug outline.
    const CIRCLE_SEGMENTS: u32 = 32;

    let (sin, cos) = (cd.z_rot.sin() as f32, cd.z_rot.cos() as f32);
    let rotate = |p: Vec2| Vec2::new(cos * p.x - sin * p.y, sin * p.x + cos * p.y);
    let offset = rotate(cd.offset);

    let pts = collider_outline_2d(cd.shape, CIRCLE_SEGMENTS);
    if pts.is_empty() {
        return;
    }
    // World-space (then render-local) point for a local outline vertex.
    let to_local = |p: Vec2| {
        let r = rotate(p) + offset;
        let world = cd.world_pos + DVec3::new(r.x as f64, r.y as f64, 0.0);
        origin.to_render(world)
    };
    for i in 0..pts.len() {
        let a = to_local(pts[i]);
        let b = to_local(pts[(i + 1) % pts.len()]);
        debug.line(a, b, COLLIDER_COLOR);
    }
}

/// Project a [`Collider3D`] (+ its world pose) into a world-space debug wireframe.
fn project_collider_3d(col: &Collider3D, world_pos: DVec3, rotation: DQuat) -> ColliderDebug3D {
    let shape = match col.shape_kind {
        ColliderShape3DKind::Box => ColliderOutline3D::Box {
            half: Vec3::new(
                col.half_extents.x as f32,
                col.half_extents.y as f32,
                col.half_extents.z as f32,
            ),
        },
        ColliderShape3DKind::Sphere => ColliderOutline3D::Sphere {
            radius: col.radius as f32,
        },
        ColliderShape3DKind::Capsule => ColliderOutline3D::Capsule {
            half_height: col.half_extents.y as f32,
            radius: col.radius as f32,
        },
    };
    ColliderDebug3D {
        shape,
        offset: col.offset.to_dvec3(),
        world_pos,
        rotation,
    }
}

/// Stroke a 3D collider wireframe into the debug-line layer, rebasing through the
/// floating origin. Segments are generated in the collider's local frame, offset
/// in the body frame, rotated by the body's world orientation, and lifted onto
/// the entity's world position.
fn draw_collider_outline_3d(debug: &mut DebugDraw, origin: &FloatingOrigin, cd: &ColliderDebug3D) {
    const COLLIDER_COLOR: [f32; 4] = [0.30, 0.95, 0.55, 1.0];
    /// Ring/arc tessellation for the debug wireframe.
    const CIRCLE_SEGMENTS: u32 = 32;

    // Local frame point → render-local: offset in body frame, rotate, translate.
    let to_local = |p: Vec3| {
        let local = DVec3::new(p.x as f64, p.y as f64, p.z as f64) + cd.offset;
        let world = cd.world_pos + cd.rotation * local;
        origin.to_render(world)
    };
    for [a, b] in collider_outline_3d(cd.shape, CIRCLE_SEGMENTS) {
        debug.line(to_local(a), to_local(b), COLLIDER_COLOR);
    }
}

/// A distinct placeholder colour per PCG kind index, so a multi-kind scatter
/// reads as varied content even before real meshes upload (P10.5b). Cycles
/// through a small foliage/rock palette.
fn pcg_kind_color(kind: u32) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 5] = [
        [0.28, 0.52, 0.24, 1.0], // foliage green
        [0.55, 0.40, 0.22, 1.0], // trunk brown
        [0.62, 0.60, 0.55, 1.0], // rock grey
        [0.75, 0.68, 0.35, 1.0], // dry grass
        [0.35, 0.58, 0.45, 1.0], // shrub teal
    ];
    PALETTE[(kind as usize) % PALETTE.len()]
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
        billboard: billboard_mode(sprite.billboard),
    }
}

/// Map the ECS [`BillboardMode`] enum onto the renderer's `u8` billboard flag
/// (P8.4a) — the sprite pass orients the quad by the camera basis for the
/// non-planar modes.
fn billboard_mode(mode: inf_ecs::BillboardMode) -> u8 {
    match mode {
        inf_ecs::BillboardMode::None => inf_render::BILLBOARD_NONE,
        inf_ecs::BillboardMode::Spherical => inf_render::BILLBOARD_SPHERICAL,
        inf_ecs::BillboardMode::Cylindrical => inf_render::BILLBOARD_CYLINDRICAL,
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

/// Project an ECS [`Terrain`] (+ its world translation) into a [`RenderTerrain`].
///
/// Each authored tile becomes a [`RenderTerrainTile`] with its `f64` origin
/// offset by the entity's world translation (so the terrain follows its
/// transform), its `f32` height buffer copied out of the paged data, and its
/// height bounds precomputed for the terrain pass's per-tile frustum cull. Tiles
/// arrive in the paged data's `BTreeMap` order → deterministic upload/draw order.
fn project_terrain(terrain: &Terrain, translation: DVec3, version: u64) -> RenderTerrain {
    let data = &terrain.data;
    let res = data.tile_resolution();
    let tiles = data
        .tiles()
        .map(|(&coord, tile)| RenderTerrainTile {
            coord,
            origin: tile.origin + translation,
            heights: tile.heights().to_vec(),
            height_bounds: tile.height_bounds(),
        })
        .collect();
    RenderTerrain {
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        version,
    }
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
    pub fn set_hover(&mut self, view: &RenderView, px: u32, py: u32) {
        // Don't recompute hover mid-drag (keeps the outline stable).
        if self.gizmo_drag.is_some() {
            return;
        }
        self.scene.hovered = self.picker.pick(&self.gpu, &self.scene, view, px, py);
    }

    /// Pick the entity GUID under the cursor (`None` = empty space). Selection
    /// itself lives in the document — the caller applies the pick to it.
    pub fn pick_guid(&mut self, view: &RenderView, px: u32, py: u32) -> Option<Uuid> {
        let id = self.picker.pick(&self.gpu, &self.scene, view, px, py)?;
        self.id_to_guid.get(&id).copied()
    }

    /// Screen-constant gizmo world size for the current view (perspective uses
    /// distance × fov; ortho uses the zoom half-height).
    fn gizmo_size(&self, view: &RenderView, origin_local: Vec3) -> f32 {
        match view.ortho {
            Some(o) => gizmo::gizmo_world_size_ortho(o.half_height),
            None => gizmo::gizmo_world_size(origin_local, view.eye_local(), self.fov_y),
        }
    }

    /// World-space transforms of the current selection after a gizmo drag, keyed
    /// by GUID — the caller writes them back to the document as one undo entry.
    /// (Local == world for the roots/identity-parent objects the gizmo edits;
    /// full parent-relative solve lands with nested transforms.)
    pub fn selected_world_transforms(&self) -> Vec<(Uuid, EcsTransform)> {
        let mut out: Vec<(Uuid, EcsTransform)> = self
            .scene
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
            .collect();
        // Selected 2D (non-mesh) entities the gizmo moved (P8.2c).
        for (guid, s) in &self.selected_2d {
            let mut t = EcsTransform::from_translation(s.translation);
            t.set_quat(s.rotation);
            t.scale = Vec3d::from_dvec3(s.scale);
            out.push((*guid, t));
        }
        out
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
        for s in self.selected_2d.values() {
            radius = radius.max((s.translation - center).length() + s.extent);
        }
        Some((center, radius))
    }

    /// If the cursor is over a gizmo handle, begin a drag and return true. The
    /// handle set is constrained to the sprite plane in 2D (ortho `view`).
    pub fn try_begin_gizmo(&mut self, view: &RenderView, px: u32, py: u32) -> bool {
        let Some(center) = self.selection_center() else {
            return false;
        };
        let origin_local = self.origin.to_render(center);
        let size = self.gizmo_size(view, origin_local);
        let two_d = view.ortho.is_some();
        let cursor = Vec2::new(px as f32, py as f32);
        let Some(axis) = gizmo::pick_axis(
            self.gizmo_mode,
            origin_local,
            size,
            view.view_proj(),
            cursor,
            view.width as f32,
            view.height as f32,
            two_d,
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
    pub fn update_gizmo(&mut self, view: &RenderView, px: u32, py: u32, snap: f32) {
        let Some(drag) = self.gizmo_drag else {
            return;
        };
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
        // Selected 2D (non-mesh) entities move the same way, in f64 (P8.2c).
        for s in self.selected_2d.values_mut() {
            match delta {
                GizmoDelta::Translate(t) => s.translation += t,
                GizmoDelta::Rotate { axis, radians } => {
                    let q = DQuat::from_axis_angle(axis.as_dvec3(), radians as f64);
                    s.rotation = q * s.rotation;
                    let rel = s.translation - pivot;
                    s.translation = pivot + q * rel;
                }
                GizmoDelta::Scale(sc) => s.scale *= sc.as_dvec3(),
            }
        }
        self.scene.mark_dirty();
    }

    pub fn end_gizmo(&mut self) {
        self.gizmo_drag = None;
    }

    // ── terrain sculpting (P10.2b) ────────────────────────────────────────

    /// The active tool (pick/gizmo vs terrain sculpt).
    pub fn tool_mode(&self) -> ToolMode {
        self.tool_mode
    }

    /// `true` while a sculpt stroke is in progress.
    pub fn is_sculpting(&self) -> bool {
        self.sculpt_drag.is_some()
    }

    /// Raycast the cursor against the target terrain, returning the hit centre in
    /// terrain-local XZ and the local surface height there. Reuses the same
    /// screen→world ray as picking/gizmo drags, rebased through the floating
    /// origin and shifted into the terrain entity's local frame.
    fn sculpt_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> Option<(Uuid, DVec2, f64)> {
        let guid = self.terrain_guid?;
        let (data, translation) = doc.terrain_data_and_origin(guid)?;
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        // Render-local ray → world → terrain-local (render axes == world axes).
        let local_origin = self.origin.to_world(ro) - translation;
        let hit = raycast_terrain(data, local_origin, rd.as_dvec3(), 1.0e6)?;
        Some((guid, DVec2::new(hit.point.x, hit.point.z), hit.point.y))
    }

    /// Rebuild the brush-ring loop points from the current terrain around
    /// `center` (terrain-local XZ), coloured by the active op. Clears the ring if
    /// the terrain vanished.
    fn refresh_ring(&mut self, doc: &SceneDoc, center: DVec2) {
        let op = self
            .sculpt_drag
            .as_ref()
            .map(|d| d.op)
            .unwrap_or(self.sculpt.op);
        let color = op_color(op);
        self.sculpt_ring_color = color;
        if let Some(guid) = self.terrain_guid {
            if let Some((data, translation)) = doc.terrain_data_and_origin(guid) {
                self.sculpt_ring = build_ring(data, translation, center, self.sculpt.radius);
                return;
            }
        }
        self.sculpt_ring.clear();
    }

    /// Update the hovered brush ring (idle Sculpt mode): raycast the cursor and
    /// rebuild the ring, or clear it off-terrain.
    pub fn update_sculpt_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        match self.sculpt_pick(doc, view, px, py) {
            Some((_, center, _)) => self.refresh_ring(doc, center),
            None => self.sculpt_ring.clear(),
        }
    }

    /// Begin a sculpt stroke under the cursor. Raycasts the terrain; on a hit,
    /// opens a [`Stroke`], lays the first dab, and returns `true`. `ctrl` flips
    /// Raise↔Lower for a temporary inverse brush (UE convention).
    pub fn begin_sculpt(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        ctrl: bool,
    ) -> bool {
        let Some((guid, center, height)) = self.sculpt_pick(doc, view, px, py) else {
            return false;
        };
        let op = effective_op(self.sculpt.op, ctrl);
        let settings = self.sculpt;
        let mut stroke = Stroke::begin();
        let (brush, params) = brush_of(op, &settings, center, height);
        doc.sculpt_apply_dab(guid, &mut stroke, brush, params);
        self.sculpt_drag = Some(SculptDrag {
            guid,
            stroke,
            op,
            last_local: center,
            flatten_height: height,
        });
        self.refresh_ring(doc, center);
        true
    }

    /// Continue the stroke: resample the path from the last dab to the cursor at
    /// even spacing (~⅓ radius) and lay a dab at each, mutating the live terrain
    /// (which re-uploads next frame via the version bump).
    pub fn update_sculpt(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) {
        let Some(drag) = self.sculpt_drag.as_ref() else {
            return;
        };
        let (guid, last, op, flatten_h) =
            (drag.guid, drag.last_local, drag.op, drag.flatten_height);
        let Some((_, cur, _)) = self.sculpt_pick(doc, view, px, py) else {
            return; // cursor slid off the terrain — hold the stroke, add nothing
        };
        let settings = self.sculpt;
        let spacing = (0.35 * settings.radius).max(0.05);
        // `dab_positions` re-emits the start (`last`); skip it — already placed.
        let dabs = dab_positions(&[last, cur], spacing);
        let mut new_last = last;
        for &c in dabs.iter().skip(1) {
            let (brush, params) = brush_of(op, &settings, c, flatten_h);
            if let Some(d) = self.sculpt_drag.as_mut() {
                doc.sculpt_apply_dab(guid, &mut d.stroke, brush, params);
            }
            new_last = c;
        }
        if let Some(d) = self.sculpt_drag.as_mut() {
            d.last_local = new_last;
        }
        self.refresh_ring(doc, cur);
    }

    /// Finish the stroke: commit the merged [`inf_terrain::HeightDelta`] as one
    /// undo step. Returns `true` if a non-empty stroke was recorded.
    pub fn finish_sculpt(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.sculpt_drag.take() else {
            return false;
        };
        doc.edit_commit_sculpt(drag.guid, drag.stroke)
    }
}

/// Effective op after a Ctrl modifier: Ctrl temporarily inverts Raise↔Lower (UE
/// convention); other ops are unaffected.
fn effective_op(op: SculptOp, ctrl: bool) -> SculptOp {
    match (op, ctrl) {
        (SculptOp::Raise, true) => SculptOp::Lower,
        (SculptOp::Lower, true) => SculptOp::Raise,
        (op, _) => op,
    }
}

/// Build the `inf_terrain` brush op + params for one dab from the toolbar
/// settings, filling in the op-specific parameters the flat UI enum omits.
fn brush_of(
    op: SculptOp,
    s: &SculptSettings,
    center: DVec2,
    flatten_height: f64,
) -> (BrushOp, BrushParams) {
    let falloff = match s.falloff {
        SculptFalloff::Smooth => Falloff::Smooth,
        SculptFalloff::Linear => Falloff::Linear,
        SculptFalloff::Sphere => Falloff::Sphere,
        SculptFalloff::Sharp => Falloff::Sharp,
    };
    let params = BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff,
    };
    let brush = match op {
        SculptOp::Raise => BrushOp::Raise,
        SculptOp::Lower => BrushOp::Lower,
        SculptOp::Smooth => BrushOp::Smooth { iterations: 1 },
        SculptOp::Flatten => BrushOp::Flatten {
            target: FlattenTarget::PickedHeight(flatten_height),
        },
        SculptOp::Noise => BrushOp::Noise {
            seed: 0x5EED_1234,
            frequency: 0.05,
            octaves: 4,
            amplitude: s.strength,
        },
    };
    (brush, params)
}

/// The brush-ring colour for an op (green raise / red lower / blue smooth /
/// yellow flatten / violet noise).
fn op_color(op: SculptOp) -> [f32; 4] {
    match op {
        SculptOp::Raise => [0.35, 0.90, 0.45, 1.0],
        SculptOp::Lower => [0.95, 0.45, 0.35, 1.0],
        SculptOp::Smooth => [0.40, 0.70, 1.00, 1.0],
        SculptOp::Flatten => [0.95, 0.85, 0.35, 1.0],
        SculptOp::Noise => [0.75, 0.50, 0.95, 1.0],
    }
}

/// Sample a closed ring of world-space points around `center` (terrain-local XZ)
/// at `radius`, each lifted to the terrain surface height there (falling back to
/// the centre height over holes), then shifted by the terrain's world
/// translation. Connect consecutive points (and last→first) to stroke the ring.
fn build_ring(data: &TerrainData, translation: DVec3, center: DVec2, radius: f64) -> Vec<DVec3> {
    const SEGMENTS: u32 = 32;
    let base_h = data.height_at(center).unwrap_or(0.0);
    (0..SEGMENTS)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / (SEGMENTS as f64);
            let p = center + DVec2::new(radius * a.cos(), radius * a.sin());
            let h = data.height_at(p).unwrap_or(base_h);
            translation + DVec3::new(p.x, h, p.y)
        })
        .collect()
}

impl EngineHost {
    /// Render one frame for the resolved [`RenderView`] (the platform loop
    /// builds it from whichever camera is active and rebases the floating origin
    /// first). Handles crash-safe device-lost recovery internally; only errors
    /// that survive a full stack rebuild are returned.
    pub fn render_frame(&mut self, view: &RenderView) -> Result<(), String> {
        // Remember the camera eye for PCG draw-distance culling on the next
        // projection (see `last_eye_world`).
        self.last_eye_world = view.eye_world;
        if self.gpu.is_lost() {
            tracing::warn!("inf-viewport: device lost — rebuilding GPU stack");
            let (w, h) = self.chain.requested_size();
            let (gpu, chain, renderer) = Self::build_gpu_stack(self.target, w, h)?;
            self.gpu = gpu;
            self.chain = chain;
            self.renderer = renderer;
        }

        // Per-frame debug primitives: world-origin axes tripod, plus the
        // transform gizmo at the selection center (screen-constant size). In 2D
        // the gizmo shows only the sprite-plane handles (X/Y, Z ring).
        self.scene.debug.clear();
        self.scene
            .debug
            .axes(self.origin.to_render(glam::DVec3::ZERO), 1.0);
        if let Some(center) = self.selection_center() {
            let origin_local = self.origin.to_render(center);
            let size = self.gizmo_size(view, origin_local);
            let active = self.gizmo_drag.map(|d| d.axis);
            gizmo::build_geometry(
                &mut self.scene.debug,
                self.gizmo_mode,
                origin_local,
                size,
                active,
                view.ortho.is_some(),
            );
        }
        // 2D + 3D collider outlines for the current selection (P8.3b / P9.1).
        for guid in &self.selected_guids {
            if let Some(cd) = self.collider_outlines.get(guid) {
                draw_collider_outline(&mut self.scene.debug, &self.origin, cd);
            }
            if let Some(cd) = self.collider_outlines_3d.get(guid) {
                draw_collider_outline_3d(&mut self.scene.debug, &self.origin, cd);
            }
        }

        // Sculpt brush ring (P10.2b): a closed loop following the terrain height
        // under the cursor, coloured by the active op. Only in Sculpt mode.
        if self.tool_mode == ToolMode::Sculpt && self.sculpt_ring.len() >= 2 {
            let n = self.sculpt_ring.len();
            for i in 0..n {
                let a = self.origin.to_render(self.sculpt_ring[i]);
                let b = self.origin.to_render(self.sculpt_ring[(i + 1) % n]);
                self.scene.debug.line(a, b, self.sculpt_ring_color);
            }
        }

        let Some(frame) = self.chain.acquire(&self.gpu) else {
            return Ok(()); // transient (occluded/timeout) — skip the frame
        };
        let out_view = self.chain.target_view(&frame);
        self.renderer.render(
            &self.gpu,
            &self.scene,
            view,
            &out_view,
            self.chain.configured_size(),
        );
        self.gpu.queue.present(frame);
        Ok(())
    }
}
