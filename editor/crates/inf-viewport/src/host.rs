//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::HashMap;

use glam::{DQuat, DVec2, DVec3, Vec2, Vec3};
use inf_ecs::components::{
    BlendMode, Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind,
    ComputedVisibility, GlobalTransform, Joint2D, Joint3D, Light, Light2D,
    LightKind as EcsLightKind, Material, MeshRef, NineSlice, PcgVolume, Primitive, SkeletalMesh,
    Spline, SplineInterp as EcsSplineInterp, Sprite, Terrain, Text2D, TextAlign, Tilemap, Volume,
};
use inf_ecs::{Transform as EcsTransform, Vec3d};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::serialize::RenderSettingsRecord;
use inf_editor_core::scene::SceneDoc;
use inf_math::{FloatingOrigin, SplineInterp};
// R-P4: scene-persisted post/exposure/lighting settings applied to the live
// renderer (see `apply_record` + `sync_from_doc`).
use inf_render::{
    collider_outline_2d, collider_outline_3d, expand_nine_slice, expand_text, gizmo,
    handle_from_guid, ColliderOutline2D, ColliderOutline3D, DebugDraw, EngineRenderer, GizmoDelta,
    GizmoDrag, GizmoMode, GpuContext, HAlign, LightKind, MeshInstance, NineSliceParams,
    OrthoParams, Picker, PrebatchedRun, PrimMesh, RenderChunk, RenderLight, RenderLight2D,
    RenderScene, RenderTerrain, RenderTerrainLayer, RenderTerrainTile, RenderTilemap, RenderView,
    SpriteInstance, SurfaceChain, TextParams, TilemapParams, BUILTIN_FONT_TEXTURE,
};
use inf_render::{
    detect_tier, BloomSettings, GiSettings, RenderSettings, RenderTier, ShadowSettings,
    SsaoSettings,
};
use uuid::Uuid;

use inf_terrain::{
    dab_positions, raycast_terrain, BrushOp, BrushParams, Falloff, FlattenTarget, SplatStroke,
    Stroke, TerrainData,
};

use crate::camera::{
    Camera2D, EditorCamera, GizmoSpace, SculptFalloff, SculptOp, SculptSettings, Snap2DSettings,
    SnapSettings, ToolMode, ViewportMode, TWO_D_FAR, TWO_D_NEAR,
};
use crate::SurfaceTarget;

/// Map an ECS [`Primitive`] to the renderer's [`PrimMesh`] (R-P1).
///
/// MIRROR: keep identical to `inf_player::render::prim_mesh` (the player's
/// ECS→RenderScene projection). Both seams must agree so the editor viewport and
/// the shipped player draw the same geometry for a given primitive.
fn prim_mesh(p: Primitive) -> PrimMesh {
    match p {
        Primitive::Cube => PrimMesh::Cube,
        Primitive::Sphere => PrimMesh::Sphere,
        Primitive::Plane => PrimMesh::Plane,
        Primitive::Cylinder => PrimMesh::Cylinder,
        Primitive::Cone => PrimMesh::Cone,
    }
}

/// Project the ECS [`BlendMode`] into the renderer's packed `blend` code (R-P5):
/// 0 opaque, 1 masked, 2 translucent. Mirrored in the player's `render.rs`.
fn blend_code(b: BlendMode) -> u8 {
    match b {
        BlendMode::Opaque => 0,
        BlendMode::Masked => 1,
        BlendMode::Translucent => 2,
    }
}

/// Parse a [`SpawnKind`] from its snake_case wire string (the tail of a
/// `"spawn:<kind>"` drop payload). Mirrors the `serde(rename_all = "snake_case")`
/// on the DTO — kept as an explicit match so the drop path stays serde-free.
fn spawn_kind_from_str(s: &str) -> Option<SpawnKind> {
    Some(match s {
        "empty" => SpawnKind::Empty,
        "cube" => SpawnKind::Cube,
        "sphere" => SpawnKind::Sphere,
        "plane" => SpawnKind::Plane,
        "cylinder" => SpawnKind::Cylinder,
        "cone" => SpawnKind::Cone,
        "directional_light" => SpawnKind::DirectionalLight,
        "point_light" => SpawnKind::PointLight,
        "spot_light" => SpawnKind::SpotLight,
        "camera" => SpawnKind::Camera,
        "sprite" => SpawnKind::Sprite,
        "tilemap" => SpawnKind::Tilemap,
        "text2d" => SpawnKind::Text2d,
        "nine_slice" => SpawnKind::NineSlice,
        "light2d" => SpawnKind::Light2d,
        "terrain" => SpawnKind::Terrain,
        _ => return None,
    })
}

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
    /// Gizmo orientation frame (Wave 2): world-aligned handles or local
    /// (selection-rotation) handles. 2D mode always draws/edits in World.
    gizmo_space: GizmoSpace,
    /// 3D transform-gizmo snap increments pushed from the toolbar (Wave 2),
    /// replacing the previously-hardcoded 1 m / 15° / 0.1 constants. Only the
    /// Windows input layer applies it (via [`EngineHost::snap_3d`]).
    #[cfg_attr(not(windows), allow(dead_code))]
    snap_3d: SnapSettings,
    gizmo_drag: Option<GizmoDrag>,
    /// Mesh-instance transforms captured at gizmo-drag start, keyed by instance
    /// id. The cumulative gizmo delta (measured from the ORIGINAL grab anchor,
    /// see [`EngineHost::update_gizmo`]) is applied to THESE each frame — the
    /// live instances are never accumulated frame-to-frame — so snapping
    /// quantizes total displacement, not per-frame deltas (M2).
    gizmo_initial: HashMap<u32, InstanceXform>,
    /// Same as [`Self::gizmo_initial`] for selected 2D (non-mesh) entities.
    gizmo_initial_2d: HashMap<Uuid, Sel2D>,
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
    /// World-space joint debug segments by GUID (P12.1), rebuilt each projection.
    /// Rendered as debug lines for the current selection only (2D + 3D joints).
    joint_lines: HashMap<Uuid, JointDebug>,
    /// World-space `Volume` wireframes by GUID (E-P4), rebuilt each projection.
    /// Unlike collider outlines these are drawn ALWAYS (not selection-gated), in
    /// the volume's tint, so trigger/blocking regions stay visible while editing.
    volume_outlines: HashMap<Uuid, VolumeDebug>,
    /// Spot-light cone gizmos by GUID (R-P3), rebuilt each projection. Drawn as
    /// debug lines for the current selection only.
    spot_lights: HashMap<Uuid, SpotDebug>,
    /// World-space `Spline` polylines by GUID (E-P5), rebuilt each projection.
    /// The sampled curve is drawn ALWAYS (neutral cyan); a selected spline
    /// additionally shows a 3-axis cross at each control point.
    spline_polylines: HashMap<Uuid, SplineDebug>,
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
    /// The GPU capability tier detected once from the adapter (R-P4). `None` until
    /// the first `sync_from_doc` probes it; it clamps the scene-persisted render
    /// settings down (never up) via [`RenderTier::apply`], exactly like the player.
    render_tier: Option<RenderTier>,
    /// The last [`RenderSettings`] pushed to the renderer (R-P4), so a redundant
    /// `set_settings` (which would reset TAA history) is skipped when the mapped
    /// value is unchanged.
    applied_render: Option<RenderSettings>,
}

/// Map the scene-persisted [`RenderSettingsRecord`] onto a live
/// [`RenderSettings`] (R-P4). The record carries the authorable subset; every
/// other field (hdr, vgeom, tier_override, and the shadow/GI tuning knobs the
/// panel doesn't expose) stays at `RenderSettings::default()`, so
/// `apply_record(&RenderSettingsRecord::default()) == RenderSettings::default()`
/// — the mapping is pinned by a unit test on both sides.
///
/// MIRROR: keep identical to `inf_player::render::apply_record` (the player's
/// copy over `inf_scene::RenderSettingsRecord`). Both seams must agree so the
/// editor viewport and the shipped player apply a level's render block the same.
fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {
    let d = RenderSettings::default();
    RenderSettings {
        exposure: r.exposure,
        dither: r.dither,
        bloom: BloomSettings {
            enabled: r.bloom_enabled,
            threshold: r.bloom_threshold,
            knee: r.bloom_knee,
            intensity: r.bloom_intensity,
        },
        ssao: SsaoSettings {
            enabled: r.ssao_enabled,
            radius: r.ssao_radius,
            intensity: r.ssao_intensity,
            bias: r.ssao_bias,
        },
        taa: r.taa,
        shadows: ShadowSettings {
            enabled: r.shadows_enabled,
            max_distance: r.shadows_max_distance,
            ..d.shadows
        },
        gi: GiSettings {
            enabled: r.gi_enabled,
            intensity: r.gi_intensity,
            ..d.gi
        },
        ..d
    }
}

/// An in-flight sculpt gesture (P10.2b): the mouse-down→up stroke accumulating
/// dabs into one [`Stroke`], plus the state to resample the drag path and, on
/// release, commit one [`inf_terrain::HeightDelta`] undo step.
struct SculptDrag {
    /// Target terrain entity.
    guid: Uuid,
    /// The accumulating stroke (merged into one delta at commit) — a height
    /// [`Stroke`] for the sculpt ops, or a [`SplatStroke`] for the Paint sub-mode.
    kind: DragStroke,
    /// The effective op (Ctrl may flip Raise↔Lower).
    op: SculptOp,
    /// Last dab centre in terrain-local XZ (for even path resampling).
    last_local: DVec2,
    /// Local surface height under the stroke's first touch — the Flatten target.
    flatten_height: f64,
}

/// The in-flight stroke of a [`SculptDrag`]: a height sculpt or a splat paint.
enum DragStroke {
    Height(Stroke),
    Splat(SplatStroke),
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

/// A mesh instance's transform captured at gizmo-drag start (M2). The cumulative
/// gizmo delta is applied to this snapshot each frame so snapping is exact.
#[derive(Debug, Clone, Copy)]
struct InstanceXform {
    translation: DVec3,
    rotation: glam::Quat,
    scale: Vec3,
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

/// A selected entity's joint (P12.1), resolved to world-space debug segments: the
/// anchor-to-anchor link plus a small cross marking each anchor.
struct JointDebug {
    segments: Vec<[DVec3; 2]>,
}

/// A [`Volume`]'s editor wireframe (E-P4): the entity's box collider resolved to
/// world space plus the volume's tint. Drawn ALWAYS (not selection-gated) so
/// trigger/blocking regions read while editing; the selection just brightens it.
struct VolumeDebug {
    collider: ColliderDebug3D,
    tint: [f32; 4],
}

/// A [`Spline`]'s editor visualization (E-P5): the sampled curve as a world-space
/// polyline plus the world-space control points (for the selected-only markers).
/// Points are cached in world space (the entity transform already applied) and
/// rebased through the floating origin at draw time.
struct SplineDebug {
    /// Sampled curve vertices in world space (consecutive pairs form segments).
    line: Vec<DVec3>,
    /// Control points in world space (a 3-axis cross is drawn at each when the
    /// spline is selected).
    control: Vec<DVec3>,
}

/// A spot [`Light`]'s editor cone gizmo (R-P3): the beam apex, its world-space
/// emission axis, the outer half-angle, an effective draw distance, and the
/// light's colour. Drawn for the current selection only (cheap, per-selection).
struct SpotDebug {
    /// Beam apex (the light's world position).
    apex: DVec3,
    /// Normalized world emission direction (`rot · −Z`).
    axis: DVec3,
    /// Outer-cone half-angle (radians) — the drawn rim.
    outer_rad: f64,
    /// Draw distance: the light's `range`, or 5 m when unbounded (`range == 0`).
    dist: f64,
    /// The light's colour (rgb, opaque).
    color: [f32; 4],
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
            gizmo_space: GizmoSpace::World,
            snap_3d: SnapSettings::default(),
            gizmo_drag: None,
            gizmo_initial: HashMap::new(),
            gizmo_initial_2d: HashMap::new(),
            fov_y: 60f32.to_radians(),
            id_to_guid: HashMap::new(),
            guid_to_id: HashMap::new(),
            collider_outlines: HashMap::new(),
            collider_outlines_3d: HashMap::new(),
            joint_lines: HashMap::new(),
            volume_outlines: HashMap::new(),
            spot_lights: HashMap::new(),
            spline_polylines: HashMap::new(),
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
            render_tier: None,
            applied_render: None,
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
        // Interactive viewport: DEGRADE on GPU validation/OOM errors (log +
        // count, keep rendering) instead of aborting the whole editor process,
        // which is wgpu's default. The headless golden/thumbnail paths keep that
        // fatal default so CI still fails hard on validation bugs (M1).
        gpu.install_lenient_error_handler();
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

    /// Set the renderer's shading view mode (Lit / Unlit / Wireframe, R-P2). Pure
    /// passthrough to the renderer, which clamps Wireframe→Unlit if the adapter
    /// lacks `POLYGON_MODE_LINE`. Editor-transient (never persisted).
    pub fn set_view_mode(&mut self, mode: inf_render::ViewMode) {
        self.renderer.set_view_mode(mode);
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
        self.apply_render_settings(doc);
    }

    /// Apply the scene-persisted render block (post/exposure/lighting) to the live
    /// renderer (R-P4). The tier is probed once from the adapter and clamps the
    /// mapped settings down (never up), mirroring the player. A redundant push is
    /// skipped (cached in `applied_render`) so an unrelated document edit doesn't
    /// needlessly reset TAA history. Runs from `sync_from_doc` (version-gated), so
    /// an `edit_settings` — which bumps the version — flows straight through.
    fn apply_render_settings(&mut self, doc: &SceneDoc) {
        let tier = *self
            .render_tier
            .get_or_insert_with(|| detect_tier(&self.gpu, &RenderSettings::default()));
        let mapped = tier.apply(apply_record(&doc.settings().render));
        if self.applied_render != Some(mapped) {
            self.renderer.set_settings(mapped);
            self.applied_render = Some(mapped);
        }
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
        self.joint_lines.clear();
        self.volume_outlines.clear();
        self.spot_lights.clear();
        self.spline_polylines.clear();

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
                    // Cache a cone gizmo for spot lights (R-P3), drawn for the
                    // selection only in `render_frame`.
                    if light.kind == EcsLightKind::Spot {
                        let (_, rot, translation) = affine.to_scale_rotation_translation();
                        let c = light.color.to_array();
                        self.spot_lights.insert(
                            guid,
                            SpotDebug {
                                apex: translation,
                                axis: (rot * -DVec3::Z).normalize(),
                                outer_rad: light.outer_cone_deg.to_radians() as f64,
                                dist: if light.range > 0.0 {
                                    light.range as f64
                                } else {
                                    5.0
                                },
                                color: [c[0], c[1], c[2], 1.0],
                            },
                        );
                    }
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
                            // PCG scatter stays a placeholder cube (same documented
                            // gap as mesh-asset viewport rendering).
                            mesh: PrimMesh::Cube,
                            // R-P5: PCG scatter placeholders are opaque.
                            blend: 0,
                            cutoff: 0.5,
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

            // Volumes (E-P4) cache a tinted box wireframe drawn ALWAYS (not
            // selection-gated) so trigger/blocking regions stay visible while
            // editing. Reuses the entity's Collider3D projection; skipped when the
            // entity is hidden (respect the visibility flag).
            if visible {
                if let (Some(vol), Some(col)) =
                    (w.get::<Volume>(entity), w.get::<Collider3D>(entity))
                {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    let (_, rotation, translation) = affine.to_scale_rotation_translation();
                    self.volume_outlines.insert(
                        guid,
                        VolumeDebug {
                            collider: project_collider_3d(col, translation, rotation),
                            tint: vol.tint.to_array(),
                        },
                    );
                }
            }

            // Splines (E-P5) cache a world-space polyline sampled from the
            // control points, drawn ALWAYS (the curve is the only editor cue) so
            // long as the entity is visible. Points are entity-local, so they are
            // lifted through the entity's world transform first; Catmull-Rom /
            // linear are both affine combinations, so transforming the control
            // points then sampling is identical to sampling then transforming (and
            // cheaper). 16 samples per segment. The selected-only control markers
            // reuse the same world control points.
            if visible {
                if let Some(spline) = w.get::<Spline>(entity) {
                    let n = spline.points.len();
                    if n >= 2 {
                        let affine = w
                            .get::<GlobalTransform>(entity)
                            .map(|g| g.0)
                            .unwrap_or(glam::DAffine3::IDENTITY);
                        let control: Vec<DVec3> = spline
                            .points
                            .iter()
                            .map(|p| affine.transform_point3(p.to_dvec3()))
                            .collect();
                        let interp = match spline.interp {
                            EcsSplineInterp::Linear => SplineInterp::Linear,
                            EcsSplineInterp::CatmullRom => SplineInterp::CatmullRom,
                        };
                        let seg_count = if spline.closed { n } else { n - 1 };
                        let steps = seg_count * 16;
                        let mut line = Vec::with_capacity(steps + 1);
                        for i in 0..=steps {
                            let t = i as f64 / steps as f64;
                            line.push(inf_math::eval_spline(&control, spline.closed, interp, t));
                        }
                        self.spline_polylines
                            .insert(guid, SplineDebug { line, control });
                    }
                }
            }

            // Joints cache world-space debug segments (P12.1): the anchor→anchor
            // link + a cross at each anchor. Resolves the OTHER body's world pose
            // via the doc's guid index. Drawn for the selection only.
            let self_pose = || {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (_, rot, tr) = affine.to_scale_rotation_translation();
                (tr, rot)
            };
            let other_pose = |other: Uuid| -> Option<(DVec3, DQuat)> {
                let oe = world.entity_of(other)?;
                let affine = w.get::<GlobalTransform>(oe).map(|g| g.0)?;
                let (_, rot, tr) = affine.to_scale_rotation_translation();
                Some((tr, rot))
            };
            if let Some(j) = w.get::<Joint3D>(entity) {
                if let Some(other) = j.other.get() {
                    if let Some((op, orot)) = other_pose(other) {
                        let (sp, srot) = self_pose();
                        let a = sp + srot * j.local_anchor.to_dvec3();
                        let b = op + orot * j.other_anchor.to_dvec3();
                        self.joint_lines.insert(guid, project_joint(a, b));
                    }
                }
            } else if let Some(j) = w.get::<Joint2D>(entity) {
                if let Some(other) = j.other.get() {
                    if let Some((op, orot)) = other_pose(other) {
                        let (sp, srot) = self_pose();
                        let a = sp + srot * DVec3::new(j.local_anchor.x, j.local_anchor.y, 0.0);
                        let b = op + orot * DVec3::new(j.other_anchor.x, j.other_anchor.y, 0.0);
                        self.joint_lines.insert(guid, project_joint(a, b));
                    }
                }
            }

            // Skeletal meshes (P11.1): the interactive viewport can't upload asset
            // geometry yet (the same documented gap as MeshRef→asset / sprites),
            // and GPU skinning is proven headlessly by the `golden_skinned_mesh`
            // golden. A `SkeletalMesh` entity (without a primitive `MeshRef`)
            // therefore projects as a **selectable placeholder cube** so it is
            // authorable in the scene; driving a real `RenderScene::skinned`
            // instance from an uploaded `.inf_mesh` + `.inf_skel` + the entity's
            // `AnimPlayer` pose is the documented viewport follow-up (v1 skinned
            // rendering in the editor is placeholder-only, headless-golden-proven).
            if w.get::<MeshRef>(entity).is_none() {
                if visible && w.get::<SkeletalMesh>(entity).is_some() {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    let (scale, rot, translation) = affine.to_scale_rotation_translation();
                    let id = next_id;
                    next_id += 1;
                    self.scene.instances.push(MeshInstance {
                        translation,
                        rotation: rot.as_quat(),
                        scale: scale.as_vec3(),
                        color: [0.55, 0.60, 0.72, 1.0],
                        metallic: 0.0,
                        roughness: 0.6,
                        emissive: [0.0; 3],
                        id,
                        // Skeletal placeholder is always a cube (no primitive kind).
                        mesh: PrimMesh::Cube,
                        // R-P5: skeletal placeholders are opaque.
                        blend: 0,
                        cutoff: 0.5,
                    });
                    self.id_to_guid.insert(id, guid);
                    self.guid_to_id.insert(guid, id);
                }
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
            // MIRROR: this Material→MeshInstance projection is duplicated in the
            // player's `render.rs` (inf-player) — keep the two in sync, R-P5 blend
            // + cutoff included.
            let (color, metallic, roughness, emissive, blend, cutoff) = w
                .get::<Material>(entity)
                .map(|m| {
                    let e = m.emissive.to_array();
                    (
                        m.base_color.to_array(),
                        m.metallic,
                        m.roughness,
                        [e[0], e[1], e[2]],
                        blend_code(m.blend),
                        m.alpha_cutoff,
                    )
                })
                .unwrap_or(([0.8, 0.8, 0.8, 1.0], 0.0, 0.5, [0.0; 3], 0, 0.5));
            // R-P1: project the MeshRef's built-in primitive kind so Sphere/Plane/
            // Cylinder/Cone render as real geometry (not everything as a cube).
            let mesh = w
                .get::<MeshRef>(entity)
                .map(|r| prim_mesh(r.primitive))
                .unwrap_or(PrimMesh::Cube);
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
                mesh,
                blend,
                cutoff,
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

/// Build joint debug segments (world space): the anchor-to-anchor link plus a
/// small axis cross at each anchor so the joint reads even when the anchors
/// coincide with the body origins.
fn project_joint(anchor_a: DVec3, anchor_b: DVec3) -> JointDebug {
    const CROSS: f64 = 0.12;
    let mut segments = vec![[anchor_a, anchor_b]];
    for anchor in [anchor_a, anchor_b] {
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            segments.push([anchor - axis * CROSS, anchor + axis * CROSS]);
        }
    }
    JointDebug { segments }
}

/// Stroke joint debug segments into the debug-line layer, rebasing each endpoint
/// through the floating origin.
fn draw_joint_lines(debug: &mut DebugDraw, origin: &FloatingOrigin, jd: &JointDebug) {
    const JOINT_COLOR: [f32; 4] = [0.95, 0.75, 0.20, 1.0];
    for [a, b] in &jd.segments {
        debug.line(origin.to_render(*a), origin.to_render(*b), JOINT_COLOR);
    }
}

/// Stroke a spot-light cone gizmo into the debug-line layer (R-P3): an 8-segment
/// rim circle at the beam's outer-cone radius, plus four apex→rim spokes, in the
/// light's colour. The rim sits at distance `dist` down the emission `axis`, with
/// radius `dist · tan(outer_rad)`. Rebased through the floating origin.
fn draw_spot_cone(debug: &mut DebugDraw, origin: &FloatingOrigin, sd: &SpotDebug) {
    const SEGMENTS: usize = 8;
    let axis = sd.axis;
    // Two axis-perpendicular basis vectors for the rim plane.
    let seed = if axis.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let t1 = axis.cross(seed).normalize();
    let t2 = axis.cross(t1); // already unit (axis ⟂ t1, both unit)
    let center = sd.apex + axis * sd.dist;
    let radius = sd.dist * sd.outer_rad.tan();

    let rim = |i: usize| -> DVec3 {
        let a = std::f64::consts::TAU * i as f64 / SEGMENTS as f64;
        center + (t1 * a.cos() + t2 * a.sin()) * radius
    };
    let apex_local = origin.to_render(sd.apex);
    for i in 0..SEGMENTS {
        let a = origin.to_render(rim(i));
        let b = origin.to_render(rim((i + 1) % SEGMENTS));
        debug.line(a, b, sd.color); // rim
        if i % (SEGMENTS / 4) == 0 {
            debug.line(apex_local, a, sd.color); // apex → rim spoke (×4)
        }
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

/// Stroke a [`Volume`]'s box wireframe into the debug-line layer in its tint,
/// rebasing through the floating origin. Drawn unconditionally (the region is
/// invisible in PIE, so the editor outline is the only cue). The debug-line API
/// has no width, so a selected volume gets a second inset ring in a brightened
/// tint to read as "thicker/highlighted".
fn draw_volume_outline(
    debug: &mut DebugDraw,
    origin: &FloatingOrigin,
    vd: &VolumeDebug,
    selected: bool,
) {
    const CIRCLE_SEGMENTS: u32 = 32;
    let cd = &vd.collider;
    // Local (optionally scaled) collider point → render-local: offset in the body
    // frame, rotate by the body orientation, translate onto the entity.
    let stroke = |debug: &mut DebugDraw, scale: f64, color: [f32; 4]| {
        let to_local = |p: Vec3| {
            let local = DVec3::new(p.x as f64, p.y as f64, p.z as f64) * scale + cd.offset;
            origin.to_render(cd.world_pos + cd.rotation * local)
        };
        for [a, b] in collider_outline_3d(cd.shape, CIRCLE_SEGMENTS) {
            debug.line(to_local(a), to_local(b), color);
        }
    };
    stroke(debug, 1.0, vd.tint);
    if selected {
        let brighten = |c: f32| (c * 1.5).min(1.0);
        let bright = [
            brighten(vd.tint[0]),
            brighten(vd.tint[1]),
            brighten(vd.tint[2]),
            vd.tint[3],
        ];
        stroke(debug, 0.9, bright);
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

/// Project an ECS `Light` (+ its world transform) into a renderer light (R-P3).
///
/// Direction conventions (**mirrored byte-for-byte** in the player's
/// `inf_player::render::project_light` — the parity tests in both crates pin
/// them so the classic mirror bug can never drift):
///  * Directional/spot store the vector *toward* the light = `rot * +Z` (an
///    entity's forward is `-Z`, so this is the anti-emission direction);
///  * the renderer derives a spot's beam emission as `-direction = rot * -Z`.
///
/// Cone half-angles convert to cosines CPU-side (std trig is fine — this is not
/// committed content). `range` and `cast_shadows` pass through for all kinds
/// (fixing the earlier point-range-hardcoded-0 bug); `cast_shadows` is inert for
/// point/spot (shadow maps deferred).
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
            cast_shadows: light.cast_shadows,
            ..RenderLight::default()
        },
        EcsLightKind::Point => RenderLight {
            kind: LightKind::Point,
            color,
            intensity: light.intensity,
            direction: Vec3::ZERO,
            position: translation,
            range: light.range,
            cast_shadows: light.cast_shadows,
            ..RenderLight::default()
        },
        EcsLightKind::Spot => RenderLight {
            kind: LightKind::Spot,
            color,
            intensity: light.intensity,
            // Toward-the-light (like directional); emission = -direction = rot·−Z.
            direction: (rot * DVec3::Z).as_vec3(),
            position: translation,
            range: light.range,
            inner_cos: light.inner_cone_deg.to_radians().cos(),
            outer_cos: light.outer_cone_deg.to_radians().cos(),
            cast_shadows: light.cast_shadows,
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
    let n = (res * res) as usize;
    let tiles = data
        .tiles()
        .map(|(&coord, tile)| {
            // Resolve the sparse weight store into a full res² buffer for upload
            // (an unpainted tile → uniform default layer 0).
            let weights: Vec<[u8; 4]> = if tile.weights_are_default() {
                vec![inf_terrain::DEFAULT_WEIGHT; n]
            } else {
                (0..res)
                    .flat_map(|j| (0..res).map(move |i| (i, j)))
                    .map(|(i, j)| tile.weight_sample(res, i, j))
                    .collect()
            };
            RenderTerrainTile {
                coord,
                origin: tile.origin + translation,
                heights: tile.heights().to_vec(),
                weights,
                height_bounds: tile.height_bounds(),
            }
        })
        .collect();
    let layers = std::array::from_fn(|k| RenderTerrainLayer {
        albedo: terrain.layers[k].albedo.to_array(),
        roughness: terrain.layers[k].roughness as f32,
        tex_scale: terrain.layers[k].tex_scale as f32,
    });
    RenderTerrain {
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        layers,
        macro_variation: terrain.macro_variation as f32,
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

    /// Switch the gizmo orientation frame (World ↔ Local) from the toolbar
    /// (Wave 2).
    pub fn set_gizmo_space(&mut self, space: GizmoSpace) {
        self.gizmo_space = space;
    }

    /// Replace the 3D transform-gizmo snap increments (from the toolbar, Wave 2).
    pub fn set_snap_3d(&mut self, snap: SnapSettings) {
        self.snap_3d = snap;
    }

    /// The active 3D snap settings (read by the Windows input layer during a
    /// gizmo drag).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn snap_3d(&self) -> SnapSettings {
        self.snap_3d
    }

    /// The gizmo's orientation basis for the current selection: `IDENTITY` in
    /// World space (or 2D, which is always world-aligned), otherwise the primary
    /// selection's world rotation for Local space. The "primary" is the first
    /// selected mesh instance, else the first selected 2D entity (Wave 2).
    fn gizmo_basis(&self) -> glam::Quat {
        if self.mode == ViewportMode::TwoD || self.gizmo_space == GizmoSpace::World {
            return glam::Quat::IDENTITY;
        }
        if let Some(id) = self.scene.selected.first() {
            if let Some(inst) = self.scene.instances.iter().find(|i| i.id == *id) {
                return inst.rotation;
            }
        }
        if let Some(s) = self.selected_2d.values().next() {
            return s.rotation.as_quat();
        }
        glam::Quat::IDENTITY
    }

    /// World-space point under a viewport pixel for drag-spawn (Wave 2, feature
    /// A). UE-like precedence: the terrain surface under the cursor (if a terrain
    /// exists), else the ground plane `y = 0`, else — looking at the sky /
    /// near-parallel — a fixed 10 m down the ray from the eye. In 2D mode the
    /// point lands on the `z = 0` sprite plane. Deterministic (no randomness).
    pub fn pick_world_point(&self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) -> DVec3 {
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        let rd = rd.as_dvec3();
        // 2D editor: intersect the sprite plane z = 0.
        if view.ortho.is_some() {
            if rd.z.abs() > 1e-9 {
                let t = -ro_w.z / rd.z;
                if t.is_finite() {
                    return ro_w + rd * t;
                }
            }
            return ro_w;
        }
        // Terrain surface under the cursor (reuses the sculpt raycast pattern).
        if let Some(guid) = self.terrain_guid {
            if let Some((data, translation)) = doc.terrain_data_and_origin(guid) {
                let local_origin = ro_w - translation;
                if let Some(hit) = raycast_terrain(data, local_origin, rd, 1.0e6) {
                    return translation + hit.point;
                }
            }
        }
        // Ground plane y = 0 (in front of the eye).
        if rd.y.abs() > 1e-6 {
            let t = -ro_w.y / rd.y;
            if (0.0..1.0e6).contains(&t) {
                return ro_w + rd * t;
            }
        }
        // Sky / near-parallel: place 10 m down the ray.
        ro_w + rd * 10.0
    }

    /// Handle a drag-drop that ended over the viewport (Wave 2, feature A): pick
    /// the world point under the cursor and spawn there as ONE undo step, then
    /// select the new entity. Returns `true` when something was spawned (the
    /// caller emits `WorldChanged`).
    ///
    /// Payload convention: `"spawn:<snake_case SpawnKind>"` (Place Actors drag)
    /// spawns that primitive/light/etc.; any other payload is treated as a
    /// Content-Drawer asset drop and spawns a placeholder cube (the viewport
    /// thread has no asset DB, mirroring `scene_spawn_asset`'s placeholder — an
    /// optional `"asset:"` prefix is accepted).
    pub fn spawn_drop(
        &self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        payload: &str,
    ) -> bool {
        let mut name = "";
        let kind = if let Some(rest) = payload.strip_prefix("spawn:") {
            match spawn_kind_from_str(rest) {
                Some(k) => k,
                None => {
                    tracing::warn!("inf-viewport: unknown drop spawn kind '{rest}'");
                    return false;
                }
            }
        } else {
            // Asset drop (or a bare/legacy payload) → placeholder cube. An
            // `asset:<id>:<name>` payload carries the display name so the
            // placeholder is named like `scene_spawn_asset`'s would be.
            if let Some(rest) = payload.strip_prefix("asset:") {
                if let Some((_id, n)) = rest.split_once(':') {
                    name = n;
                }
            }
            SpawnKind::Cube
        };
        let point = self.pick_world_point(doc, view, px, py);
        doc.begin_transaction("Spawn");
        let guid = doc.edit_create(kind, name, None);
        doc.edit_set_transform(guid, EcsTransform::from_translation(point));
        doc.select(&[guid], false);
        doc.commit_transaction();
        true
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
        let basis = self.gizmo_basis();
        let cursor = Vec2::new(px as f32, py as f32);
        let Some(axis) = gizmo::pick_axis(
            self.gizmo_mode,
            origin_local,
            basis,
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
            basis,
            origin_local,
            ro,
            rd,
        ));
        // Snapshot the selection's transforms at drag start (M2): every frame's
        // cumulative delta is applied to these, not accumulated onto the live
        // instances, so snapping quantizes total displacement.
        self.gizmo_initial.clear();
        for id in &self.scene.selected {
            if let Some(inst) = self.scene.instances.iter().find(|i| i.id == *id) {
                self.gizmo_initial.insert(
                    *id,
                    InstanceXform {
                        translation: inst.translation,
                        rotation: inst.rotation,
                        scale: inst.scale,
                    },
                );
            }
        }
        self.gizmo_initial_2d = self.selected_2d.clone();
        true
    }

    pub fn is_dragging_gizmo(&self) -> bool {
        self.gizmo_drag.is_some()
    }

    /// Apply a gizmo drag update from the cursor. `snap` > 0 quantizes.
    ///
    /// The drag is NOT re-anchored between frames: [`GizmoDrag::update`] measures
    /// the delta from the original grab point, so `delta` is the CUMULATIVE
    /// motion since the gesture began. Snapping therefore quantizes the total
    /// displacement (a slow sub-snap drag holds still until it crosses a snap
    /// boundary, then jumps exactly one step; total motion is always a multiple
    /// of the step). The cumulative delta is applied to the drag-start snapshot
    /// (`gizmo_initial`), never accumulated onto the live instances (M2).
    pub fn update_gizmo(&mut self, view: &RenderView, px: u32, py: u32, snap: f32) {
        let Some(drag) = self.gizmo_drag else {
            return;
        };
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let delta = drag.update(ro, rd, snap);
        self.apply_delta(delta, drag.origin);
    }

    /// Apply the cumulative gizmo `delta` to the drag-start snapshot, writing the
    /// result onto the live selection. `pivot_local` is the gizmo origin at drag
    /// start (render-local) — fixed for the whole gesture so cumulative rotation
    /// orbits about a stable point.
    fn apply_delta(&mut self, delta: GizmoDelta, pivot_local: Vec3) {
        let pivot = self.origin.to_world(pivot_local);
        let selected = self.scene.selected.clone();
        for id in &selected {
            let Some(init) = self.gizmo_initial.get(id).copied() else {
                continue;
            };
            if let Some(inst) = self.scene.instances.iter_mut().find(|i| i.id == *id) {
                match delta {
                    GizmoDelta::Translate(t) => inst.translation = init.translation + t,
                    GizmoDelta::Rotate { axis, radians } => {
                        let q = glam::Quat::from_axis_angle(axis, radians);
                        inst.rotation = q * init.rotation;
                        // Orbit the translation about the pivot too.
                        let rel = (init.translation - pivot).as_vec3();
                        inst.translation = pivot + (q * rel).as_dvec3();
                    }
                    GizmoDelta::Scale(s) => inst.scale = init.scale * s,
                }
            }
        }
        // Selected 2D (non-mesh) entities move the same way, in f64 (P8.2c).
        for (guid, s) in self.selected_2d.iter_mut() {
            let Some(init) = self.gizmo_initial_2d.get(guid).copied() else {
                continue;
            };
            match delta {
                GizmoDelta::Translate(t) => s.translation = init.translation + t,
                GizmoDelta::Rotate { axis, radians } => {
                    let q = DQuat::from_axis_angle(axis.as_dvec3(), radians as f64);
                    s.rotation = q * init.rotation;
                    let rel = init.translation - pivot;
                    s.translation = pivot + q * rel;
                }
                GizmoDelta::Scale(sc) => s.scale = init.scale * sc.as_dvec3(),
            }
        }
        self.scene.mark_dirty();
    }

    pub fn end_gizmo(&mut self) {
        self.gizmo_drag = None;
        self.gizmo_initial.clear();
        self.gizmo_initial_2d.clear();
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
        // Paint recolours the ring by the target layer's albedo (so the swatch
        // under the cursor reads as the layer being painted); sculpt ops use
        // their fixed op colour.
        let color = if op == SculptOp::Paint {
            self.terrain_guid
                .and_then(|g| doc.terrain_layer_albedo(g, self.sculpt.paint_layer))
                .unwrap_or_else(|| op_color(op))
        } else {
            op_color(op)
        };
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
        let kind = if op == SculptOp::Paint {
            let mut stroke = SplatStroke::begin(settings.paint_layer);
            doc.paint_apply_dab(guid, &mut stroke, paint_params(&settings, center));
            DragStroke::Splat(stroke)
        } else {
            let mut stroke = Stroke::begin();
            let (brush, params) = brush_of(op, &settings, center, height);
            doc.sculpt_apply_dab(guid, &mut stroke, brush, params);
            DragStroke::Height(stroke)
        };
        self.sculpt_drag = Some(SculptDrag {
            guid,
            kind,
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
            if let Some(d) = self.sculpt_drag.as_mut() {
                match &mut d.kind {
                    DragStroke::Height(stroke) => {
                        let (brush, params) = brush_of(op, &settings, c, flatten_h);
                        doc.sculpt_apply_dab(guid, stroke, brush, params);
                    }
                    DragStroke::Splat(stroke) => {
                        doc.paint_apply_dab(guid, stroke, paint_params(&settings, c));
                    }
                }
            }
            new_last = c;
        }
        if let Some(d) = self.sculpt_drag.as_mut() {
            d.last_local = new_last;
        }
        self.refresh_ring(doc, cur);
    }

    /// Finish the stroke: commit the merged height [`inf_terrain::HeightDelta`] or
    /// splat [`inf_terrain::SplatDelta`] as one undo step. Returns `true` if a
    /// non-empty stroke was recorded.
    pub fn finish_sculpt(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.sculpt_drag.take() else {
            return false;
        };
        match drag.kind {
            DragStroke::Height(stroke) => doc.edit_commit_sculpt(drag.guid, stroke),
            DragStroke::Splat(stroke) => doc.edit_commit_paint(drag.guid, stroke),
        }
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
fn falloff_of(f: SculptFalloff) -> Falloff {
    match f {
        SculptFalloff::Smooth => Falloff::Smooth,
        SculptFalloff::Linear => Falloff::Linear,
        SculptFalloff::Sphere => Falloff::Sphere,
        SculptFalloff::Sharp => Falloff::Sharp,
    }
}

/// Brush params for a splat-paint dab (P10.4): `strength` is the per-dab flow
/// rate toward the target layer, `falloff` shapes it across the radius.
fn paint_params(s: &SculptSettings, center: DVec2) -> BrushParams {
    BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff: falloff_of(s.falloff),
    }
}

fn brush_of(
    op: SculptOp,
    s: &SculptSettings,
    center: DVec2,
    flatten_height: f64,
) -> (BrushOp, BrushParams) {
    let params = BrushParams {
        center,
        radius: s.radius,
        strength: s.strength,
        falloff: falloff_of(s.falloff),
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
        // Paint is routed to the splat path before `brush_of` is reached; map it
        // to a no-op-ish Raise for totality (never actually applied).
        SculptOp::Paint => BrushOp::Raise,
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
        // Fallback only — the ring is normally recoloured to the target layer's
        // albedo (see `refresh_ring`).
        SculptOp::Paint => [0.90, 0.90, 0.90, 1.0],
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
    ///
    /// Returns `Ok(true)` when a frame was presented and `Ok(false)` when the
    /// swapchain had no image to acquire (surface occluded/minimized/hidden) so
    /// nothing was drawn. The caller uses this to pace itself: a presented FIFO
    /// frame blocks at vsync, but a non-present must be throttled by the loop or
    /// it busy-spins the CPU (M3).
    pub fn render_frame(&mut self, view: &RenderView) -> Result<bool, String> {
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
            // The picker holds its own device-scoped GPU resources (pipeline,
            // ID-buffer target, readback buffer) created against the OLD device.
            // Rebuild it on the fresh device too — otherwise the next pick (which
            // runs in the interaction block, OUTSIDE the render catch_unwind)
            // hits a device-mismatch validation error and kills the thread with
            // the scene mutex poisoned (H1). The `renderer` was already the only
            // other GPU-resource field; every remaining field on `self` is plain
            // CPU/scene data and survives a device loss untouched.
            self.picker = Picker::new(&self.gpu);
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
            // Draw with the in-flight drag's fixed basis, else recompute from the
            // current selection (so idle Local-mode handles track selection).
            let basis = self
                .gizmo_drag
                .map(|d| d.basis)
                .unwrap_or_else(|| self.gizmo_basis());
            gizmo::build_geometry(
                &mut self.scene.debug,
                self.gizmo_mode,
                origin_local,
                basis,
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
            if let Some(jd) = self.joint_lines.get(guid) {
                draw_joint_lines(&mut self.scene.debug, &self.origin, jd);
            }
            if let Some(sd) = self.spot_lights.get(guid) {
                draw_spot_cone(&mut self.scene.debug, &self.origin, sd);
            }
        }

        // Volume wireframes (E-P4) draw ALWAYS in the volume's tint (invisible in
        // PIE — this is the only editor cue); a selected volume brightens.
        for (guid, vd) in &self.volume_outlines {
            let selected = self.selected_guids.contains(guid);
            draw_volume_outline(&mut self.scene.debug, &self.origin, vd, selected);
        }

        // Spline polylines (E-P5) draw ALWAYS in a neutral cyan; a selected
        // spline additionally shows a brighter 3-axis cross at each control point.
        for (guid, sd) in &self.spline_polylines {
            const SPLINE_COLOR: [f32; 4] = [0.25, 0.85, 0.95, 1.0];
            const SPLINE_MARKER: [f32; 4] = [0.6, 1.0, 1.0, 1.0];
            const MARKER_ARM: f64 = 0.15; // world-metre half-length of a cross arm
            for pair in sd.line.windows(2) {
                let a = self.origin.to_render(pair[0]);
                let b = self.origin.to_render(pair[1]);
                self.scene.debug.line(a, b, SPLINE_COLOR);
            }
            if self.selected_guids.contains(guid) {
                for &p in &sd.control {
                    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                        let a = self.origin.to_render(p - axis * MARKER_ARM);
                        let b = self.origin.to_render(p + axis * MARKER_ARM);
                        self.scene.debug.line(a, b, SPLINE_MARKER);
                    }
                }
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
            return Ok(false); // transient (occluded/timeout) — nothing presented
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
        Ok(true)
    }
}

#[cfg(test)]
mod render_settings_tests {
    use super::{apply_record, RenderSettings, RenderSettingsRecord};

    /// The default record maps to the byte-stable renderer default — this pins the
    /// mapping so the editor viewport starts identical to today's pure defaults
    /// (and identical to the player's mirror).
    #[test]
    fn default_record_maps_to_default_settings() {
        assert_eq!(
            apply_record(&RenderSettingsRecord::default()),
            RenderSettings::default()
        );
    }

    /// A non-default record flows each authored field onto the live settings.
    #[test]
    fn non_default_fields_map_through() {
        let rec = RenderSettingsRecord {
            exposure: 2.0,
            dither: false,
            bloom_enabled: true,
            bloom_intensity: 0.3,
            ssao_enabled: true,
            taa: true,
            shadows_enabled: true,
            shadows_max_distance: 120.0,
            gi_enabled: true,
            gi_intensity: 1.5,
            ..RenderSettingsRecord::default()
        };
        let s = apply_record(&rec);
        assert_eq!(s.exposure, 2.0);
        assert!(!s.dither);
        assert!(s.bloom.enabled && (s.bloom.intensity - 0.3).abs() < 1e-6);
        assert!(s.ssao.enabled);
        assert!(s.taa);
        assert!(s.shadows.enabled && (s.shadows.max_distance - 120.0).abs() < 1e-6);
        assert!(s.gi.enabled && (s.gi.intensity - 1.5).abs() < 1e-6);
        // Untouched tuning knobs stay at their defaults.
        assert_eq!(s.shadows.lambda, RenderSettings::default().shadows.lambda);
        assert_eq!(s.gi.extent, RenderSettings::default().gi.extent);
    }
}

/// Spot-light seam parity (R-P3). The **identical** fixture + hardcoded
/// expectations live in `inf_player::render`'s mirror test; both must agree so
/// the toward-the-light / emission direction convention can never drift between
/// the editor viewport and the player.
#[cfg(test)]
mod project_light_parity {
    use super::{project_light, EcsLightKind, Light, LightKind};
    use glam::{DAffine3, DQuat, DVec3};

    #[test]
    fn spot_projects_with_shared_convention() {
        let light = Light {
            kind: EcsLightKind::Spot,
            intensity: 2.0,
            range: 12.0,
            inner_cone_deg: 20.0,
            outer_cone_deg: 35.0,
            cast_shadows: false,
            ..Light::default()
        };
        // Rotate 30° about X at (1, 2, 3).
        let affine = DAffine3::from_rotation_translation(
            DQuat::from_rotation_x(30f64.to_radians()),
            DVec3::new(1.0, 2.0, 3.0),
        );
        let rl = project_light(&light, &affine);

        assert!(matches!(rl.kind, LightKind::Spot));
        // Direction is *toward* the light = rot · +Z = (0, -sin30, cos30).
        let d = rl.direction;
        assert!((d.x - 0.0).abs() < 1e-5, "dir.x {}", d.x);
        assert!((d.y - (-0.5)).abs() < 1e-5, "dir.y {}", d.y);
        assert!((d.z - 0.866_025_4).abs() < 1e-5, "dir.z {}", d.z);
        assert!((rl.position.x - 1.0).abs() < 1e-9);
        assert!((rl.position.y - 2.0).abs() < 1e-9);
        assert!((rl.position.z - 3.0).abs() < 1e-9);
        assert!((rl.range - 12.0).abs() < 1e-6);
        assert!(
            (rl.inner_cos - 0.939_692_6).abs() < 1e-5,
            "inner {}",
            rl.inner_cos
        );
        assert!(
            (rl.outer_cos - 0.819_152).abs() < 1e-5,
            "outer {}",
            rl.outer_cos
        );
        assert!(!rl.cast_shadows);
    }
}
