//! Platform-shared engine host: owns the GPU stack (context, swapchain,
//! renderer), the render scene, and the floating origin. The per-OS modules
//! (win32, macos) own the native window/layer + input and drive this.

use std::collections::{BTreeSet, HashMap};

use glam::{DQuat, DVec2, DVec3, Vec2, Vec3};
use inf_ecs::components::{
    BlendMode, Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind,
    ComputedVisibility, Foliage, FoliageInstance, GlobalTransform, Joint2D, Joint3D, Light,
    Light2D, LightKind as EcsLightKind, Material, MeshRef, NineSlice, PcgVolume, Primitive,
    SkeletalMesh, Spline, SplineInterp as EcsSplineInterp, Sprite, Terrain, Text2D, TextAlign,
    Tilemap, Volume,
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
    handle_from_guid, AtmosphereParams, CloudParams, ColliderOutline2D, ColliderOutline3D,
    DebugDraw, EngineRenderer, GizmoDelta, GizmoDrag, GizmoMode, GpuContext, HAlign, HeightFog,
    LightKind, MeshInstance, NineSliceParams, OrthoParams, Picker, PrebatchedRun, PrecipParams,
    PrimMesh, RenderChunk, RenderLight, RenderLight2D, RenderScene, RenderTerrain,
    RenderTerrainLayer, RenderTerrainTile, RenderTilemap, RenderView, SkyParams, SpriteInstance,
    SunParams, SurfaceChain, TerrainTileKey, TextParams, TilemapParams, BUILTIN_FONT_TEXTURE,
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
    Camera2D, EditorCamera, FoliageSettings, GizmoSpace, SculptFalloff, SculptOp, SculptSettings,
    Snap2DSettings, SnapSettings, ToolMode, ViewportMode, TWO_D_FAR, TWO_D_NEAR,
};
use crate::SurfaceTarget;

/// Frames of *page movement* between terrain-streaming diagnostics lines
/// (P16.3b2). Roughly 5 s at 60 fps of continuous paging; a settled camera never
/// reaches it because the counter only ticks when the cut actually changed.
const STREAM_LOG_INTERVAL_FRAMES: u32 = 300;

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
        "spline" => SpawnKind::Spline,
        "foliage" => SpawnKind::Foliage,
        "trigger_volume" => SpawnKind::TriggerVolume,
        "blocking_volume" => SpawnKind::BlockingVolume,
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
    /// GUID of the terrain entity the terrain tools currently target (P16.6).
    ///
    /// Set to the **first** projected terrain on each projection — so a
    /// single-terrain document behaves exactly as it did before — and then moved
    /// to whichever terrain the cursor is actually over by
    /// [`sculpt_pick`](Self::sculpt_pick)'s nearest-hit resolution, which the
    /// hover and stroke paths both run through. `None` ⇒ no terrain is projected.
    terrain_guid: Option<Uuid>,
    /// Every visible, non-empty terrain in the current projection, in the
    /// document's order — **index-aligned with `scene.terrains`** (P16.6), which
    /// is what lets a re-projection of one terrain (a streamed cut advancing, a
    /// dab landing) write back into the right slot instead of rebuilding the list.
    terrain_slots: Vec<TerrainSlot>,
    /// Camera-driven streaming for asset-backed terrains (P16.3b2). The policy
    /// lives in `inf_editor_core::terrain_stream` (Ring 1, so Linux CI compiles
    /// and tests it); the host only calls it — at projection time
    /// ([`rebuild_scene`](Self::rebuild_scene)) and at the render-sync point
    /// ([`render_frame`](Self::render_frame)).
    ///
    /// **The determinism seam.** Its *camera-driven* pages land in the streamer's
    /// own working set, never in the document's `Terrain.data`, so moving the
    /// editor camera cannot dirty the document, change a `height_at` answer, or
    /// desync a Simulate session from a shipped run. (An **edit** does page into
    /// the document — synchronously, footprint-shaped — which is a different
    /// thing entirely; see `inf_editor_core::terrain_edit`.) Disabled until a
    /// content root is set, which makes inline terrain behaviour bit-identical to
    /// before.
    terrain_streams: inf_editor_core::terrain_stream::EditorTerrainStreams,
    /// The loose-file render-asset store (P18.3) — the editor's answer to the
    /// player's `VmeshRegistry`. Resolves a `MeshRef.asset` to its derived
    /// `.inf_vmesh` and a `SkeletalMesh` to bind-space geometry + a posed skinning
    /// palette, both from the project's content root.
    ///
    /// Owned here, and released here: the projection is the only thing that knows
    /// which mesh assets a document actually references, so it is the only thing
    /// that can free the rest ([`EditorRenderAssets::retain_only`]). The policy
    /// itself lives in Ring 1 for the same reason terrain streaming does — Linux CI
    /// compiles and tests it, this file it does not.
    ///
    /// [`EditorRenderAssets::retain_only`]: inf_editor_core::render_assets::EditorRenderAssets::retain_only
    render_assets: inf_editor_core::render_assets::EditorRenderAssets,
    /// The last tool-rejection message, for a Ring-2 caller to surface. Drained by
    /// [`take_tool_status`](Self::take_tool_status).
    tool_status: Option<String>,
    /// Frames until the next terrain-streaming diagnostics line. Counts down only
    /// while pages are actually moving, so a settled camera logs nothing.
    stream_log_countdown: u32,
    /// In-flight sculpt stroke: the accumulating brush gesture (`None` = idle).
    sculpt_drag: Option<SculptDrag>,
    /// World-space brush-ring loop points (following terrain height), rebuilt as
    /// the cursor hovers/sculpts terrain; drawn as debug lines in Sculpt mode.
    /// Shared with the Foliage brush (same hover-ring buffer, different colour).
    sculpt_ring: Vec<DVec3>,
    /// Colour of the brush ring (encodes the active op / foliage brush).
    sculpt_ring_color: [f32; 4],
    /// Foliage-brush configuration pushed from the toolbar (E-P6).
    foliage: FoliageSettings,
    /// In-flight foliage scatter stroke (`None` = idle).
    foliage_drag: Option<FoliageDrag>,
    /// Monotonic per-session stroke counter: folded into the scatter RNG so each
    /// stroke is independent yet the same input sequence reproduces identical
    /// instances (determinism law — no wall-clock / thread-rng).
    foliage_stroke_seq: u32,
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
    /// The adapter capabilities probed once alongside [`Self::render_tier`]
    /// (P18.3). Needed because the editor now *requests* the meshlet path, so the
    /// occlusion capability floor (`clamp_occlusion`) has to be applied on top of
    /// the tier — the pair that [`inf_render::detect_and_clamp`] is. Cached rather
    /// than re-probed because `apply_render_settings` runs on every document
    /// version, which during a gizmo drag is every frame.
    render_caps: Option<inf_render::AdapterCaps>,
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

/// Nearest positive ray/sphere intersection distance, or `None` when the ray
/// misses (P18.3 analytic pick fallback).
///
/// A ray starting *inside* the sphere counts as a hit at `t = 0` — clicking while
/// the camera is inside an object must select it, not fall through it. Pure, so
/// the rule is unit-testable without a GPU.
fn ray_sphere_t(ro: DVec3, rd: DVec3, center: DVec3, radius: f64) -> Option<f64> {
    let m = ro - center;
    let b = m.dot(rd);
    let c = m.dot(m) - radius * radius;
    if c > 0.0 && b > 0.0 {
        return None; // pointing away from a sphere we are outside of
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    Some((-b - disc.sqrt()).max(0.0))
}

/// The settings the editor viewport **requests** before any tier clamp (P18.3).
///
/// MIRROR of the player's request in `PlayerRenderHost::new`: take the level's
/// authored block and turn the meshlet path **on**, then let the tier and the
/// adapter capability floor clamp it down. Without this the editor's real-mesh
/// content would always fall through to the classic discrete-LOD node — the same
/// geometry, but none of P18.2's streaming, budget or eviction, and a claim of
/// preview-==-shipping that is not true.
///
/// A free function on purpose: it is the whole editor-side render-settings
/// decision, so it unit-tests without a GPU (below), and
/// `tests/projector_mirror.rs` pins the opt-in against the player's copy.
///
/// The editor deliberately does **not** apply `clamp_mobile`: there is no mobile
/// editor, and the player's own mobile branch is `cfg`-gated to targets this crate
/// does not build for.
fn requested_render_settings(record: &RenderSettingsRecord) -> RenderSettings {
    let base = apply_record(record);
    RenderSettings {
        vgeom: inf_render::VgeomSettings {
            enabled: true,
            ..base.vgeom
        },
        ..base
    }
}

/// One projected terrain (P16.6) — the per-terrain state the old single-terrain
/// `terrain_streamed` / `terrain_editable` / `terrain_unsaved_edits` fields held,
/// now one record per terrain and index-aligned with `scene.terrains`.
struct TerrainSlot {
    /// The terrain entity.
    guid: Uuid,
    /// Asset-backed (streamed from a `.inf_terrain`) rather than inline.
    streamed: bool,
    /// Streamed **and** its asset is a writable file the save path can fold edits
    /// into. Always `false` for an inline terrain, which needs no asset at all —
    /// read together with `streamed`: *streamed && !editable* is the one case the
    /// terrain tools refuse.
    editable: bool,
    /// Carries tiles not yet written back to its asset.
    unsaved: bool,
}

/// One terrain the cursor-resolution helpers below consider (P16.6): the entity,
/// the heightfield **actually under the cursor**, and the terrain's world
/// translation.
///
/// The middle field is the load-bearing one. For an inline terrain it is the
/// document's own `TerrainData`; for a **streamed** one the document's set is
/// empty by design (its tiles live in the `.inf_terrain`, and only what the
/// streamer has paged in is real), so it must be the streamer's render working
/// set. Everything that resolves a cursor against terrain — sculpt, paint,
/// drag-drop, foliage — funnels through [`EngineHost::terrain_probes`], which is
/// the one place that choice is made.
struct TerrainProbe<'a> {
    guid: Uuid,
    data: &'a inf_terrain::TerrainData,
    translation: DVec3,
}

/// Where a ray met a terrain: the entity, the hit in that terrain's local XZ, the
/// local surface height, and the world-space point.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TerrainRayHit {
    guid: Uuid,
    local_xz: DVec2,
    local_height: f64,
    world: DVec3,
}

/// Build a [`TerrainProbe`] per slot, in order, resolving each one's heightfield
/// through `resolve` and honouring `restrict` (P16.6).
///
/// Free + generic over the resolver so the **restrict** rule — the thing that
/// pins a stroke to the terrain it started on — unit-tests without a GPU, which
/// an `EngineHost` method could not.
fn terrain_probes_of<'a>(
    slots: &[TerrainSlot],
    restrict: Option<Uuid>,
    mut resolve: impl FnMut(Uuid) -> Option<(&'a inf_terrain::TerrainData, DVec3)>,
) -> Vec<TerrainProbe<'a>> {
    slots
        .iter()
        .filter(|s| !restrict.is_some_and(|g| g != s.guid))
        .filter_map(|s| {
            resolve(s.guid).map(|(data, translation)| TerrainProbe {
                guid: s.guid,
                data,
                translation,
            })
        })
        .collect()
}

/// The **nearest** terrain hit along a world-space ray (P16.6).
///
/// Nearest-along-the-ray is the only defensible rule once terrains can overlap or
/// nest: the surface you can see is the surface a brush must write, and
/// "whichever terrain the document happens to list first" is neither. Ties (two
/// coincident surfaces) resolve to the earlier probe, i.e. document order, so the
/// choice is deterministic rather than dependent on iteration luck.
///
/// Pure, so the rule unit-tests without a GPU.
fn nearest_terrain_hit(
    probes: &[TerrainProbe<'_>],
    ro_w: DVec3,
    rd: DVec3,
) -> Option<TerrainRayHit> {
    let mut best: Option<(f64, TerrainRayHit)> = None;
    for probe in probes {
        let Some(hit) = raycast_terrain(probe.data, ro_w - probe.translation, rd, 1.0e6) else {
            continue;
        };
        let world = probe.translation + hit.point;
        let d = (world - ro_w).length();
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((
                d,
                TerrainRayHit {
                    guid: probe.guid,
                    local_xz: DVec2::new(hit.point.x, hit.point.z),
                    local_height: hit.point.y,
                    world,
                },
            ));
        }
    }
    best.map(|(_, hit)| hit)
}

/// The **topmost** terrain surface at world XZ `p`, as `(entity, world height)`.
///
/// Topmost rather than nearest, because this answers "what ground is here?" for
/// things scattered from above (foliage) rather than "what did the cursor hit?".
/// Ties resolve to the earlier probe. Pure, so it unit-tests without a GPU.
fn topmost_surface(probes: &[TerrainProbe<'_>], p: DVec2) -> Option<(Uuid, f64)> {
    let mut best: Option<(f64, Uuid)> = None;
    for probe in probes {
        let local = DVec2::new(p.x - probe.translation.x, p.y - probe.translation.z);
        let Some(h) = probe.data.height_at(local) else {
            continue;
        };
        let y = probe.translation.y + h;
        if best.as_ref().is_none_or(|(by, _)| y > *by) {
            best = Some((y, probe.guid));
        }
    }
    best.map(|(y, g)| (g, y))
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

/// An in-flight foliage scatter gesture (E-P6): the mouse-down→up stroke that
/// live-mutates the target [`Foliage`] component per tick and, on release,
/// commits ONE `PaintFoliage` undo step. Either adds (`added`) or erases
/// (`removed`) — never both in one stroke.
struct FoliageDrag {
    /// Target foliage entity (selected, or auto-created at stroke start).
    guid: Uuid,
    /// Erase (remove within radius) vs place.
    erase: bool,
    /// This stroke's index, folded into the scatter RNG for determinism.
    stroke_seq: u32,
    /// Running scatter-sample index (monotonic across the whole stroke) so every
    /// candidate draws a distinct deterministic RNG draw.
    next_sample: u64,
    /// Entity world translation captured at stroke start — foliage instances are
    /// entity-local, so world hit points convert through this.
    origin: DVec3,
    /// Local XZ of every instance known this stroke (pre-existing + added), for
    /// O(n) min-spacing rejection (add mode). A v1 simplification — fine at brush
    /// scale; a spatial hash is the follow-up for very dense components.
    positions: Vec<DVec2>,
    /// Instances placed this stroke, in push order (append-only; the undo record
    /// pops exactly these off the end on revert).
    added: Vec<FoliageInstance>,
    /// Snapshot of the component's instances at stroke start (erase mode only).
    original: Vec<FoliageInstance>,
    /// Original-vector indices removed so far this stroke (erase mode).
    removed: BTreeSet<usize>,
}

/// One deterministic scatter candidate produced by [`foliage_samples`]: a world
/// XZ position within the brush disk plus a yaw + uniform scale. The host lifts it
/// to the terrain (or ground) height and converts to entity-local before placing.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FoliageCandidate {
    /// World-space XZ (the `y`/height is resolved by the host against the terrain).
    pos_xz: DVec2,
    /// Yaw about +Y, degrees (euler-deg YXZ, the `Transform` convention).
    yaw_deg: f64,
    /// Uniform scale (`1 ± scale_jitter`).
    scale: f64,
    /// Palette slot.
    kind: u32,
}

/// Hard cap on candidate samples placed per brush tick (keeps a huge-radius
/// high-density brush from stalling the interaction thread).
const FOLIAGE_MAX_PER_TICK: u32 = 64;

/// Deterministic disk sampler for one foliage brush tick (E-P6). **Pure** — the
/// output is a function of the inputs alone (no wall-clock / thread-rng), so the
/// same stroke input sequence reproduces identical instances (unit-tested). Each
/// candidate `i` derives its uniforms from `xxh3_64(seed, stroke_seq,
/// base_index + i)`; the disk sample is area-uniform (`r = R·√u`).
#[allow(clippy::too_many_arguments)] // brush params are a flat list; a struct here would just shuffle them
fn foliage_samples(
    center_xz: DVec2,
    radius: f64,
    count: u32,
    seed: u32,
    stroke_seq: u32,
    base_index: u64,
    scale_jitter: f64,
    kind: u32,
) -> Vec<FoliageCandidate> {
    let jitter = scale_jitter.max(0.0);
    (0..count)
        .map(|i| {
            let h = foliage_hash(seed, stroke_seq, base_index + i as u64);
            // Split the 64-bit hash into four independent [0,1) uniforms.
            let u0 = unit_from_bits(h);
            let u1 = unit_from_bits(h.rotate_left(16).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let u2 = unit_from_bits(h.rotate_left(32).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            let u3 = unit_from_bits(h.rotate_left(48).wrapping_mul(0x1656_67B1_9E37_79F9));
            let r = radius * u0.sqrt();
            let theta = std::f64::consts::TAU * u1;
            let pos_xz = center_xz + DVec2::new(r * theta.cos(), r * theta.sin());
            let yaw_deg = 360.0 * u2;
            let scale = (1.0 + (u3 * 2.0 - 1.0) * jitter).max(0.01);
            FoliageCandidate {
                pos_xz,
                yaw_deg,
                scale,
                kind,
            }
        })
        .collect()
}

/// xxh3-64 of the three-word RNG key `(seed, stroke_seq, sample_index)`, packed
/// little-endian. Shared hash family with `inf-graph`/`inf-asset`.
fn foliage_hash(seed: u32, stroke_seq: u32, sample_index: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&seed.to_le_bytes());
    bytes[4..8].copy_from_slice(&stroke_seq.to_le_bytes());
    bytes[8..16].copy_from_slice(&sample_index.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

/// Map 64 hash bits to a `[0, 1)` uniform (53-bit mantissa, exactly like the
/// canonical `u64 → f64` construction).
fn unit_from_bits(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Min-spacing (world metres) below which a candidate is rejected against an
/// existing instance: derived from the target density so a denser brush packs
/// tighter. `area/instance = 1/density`, so nominal spacing ≈ √(1/density); a
/// 0.7 factor lets the disk fill without a hard grid look. Clamped to a small
/// floor so `density → ∞` can't reject everything.
fn foliage_min_spacing(density: f64) -> f64 {
    if density <= 0.0 {
        return 0.05;
    }
    (0.7 * (1.0 / density).sqrt()).max(0.05)
}

/// Euler-degrees (YXZ) → quaternion, matching `inf_ecs::Transform::quat` exactly
/// so a foliage instance's stored rotation reads the same everywhere.
fn foliage_rot_quat(rot: Vec3d) -> glam::Quat {
    DQuat::from_euler(
        glam::EulerRot::YXZ,
        rot.y.to_radians(),
        rot.x.to_radians(),
        rot.z.to_radians(),
    )
    .as_quat()
}

/// Sample a flat brush ring at `y = 0` around a world XZ centre (the foliage
/// brush's ground-plane fallback when there's no terrain under the cursor).
fn ground_ring(center_xz: DVec2, radius: f64) -> Vec<DVec3> {
    const SEGMENTS: u32 = 32;
    (0..SEGMENTS)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / (SEGMENTS as f64);
            DVec3::new(
                center_xz.x + radius * a.cos(),
                0.0,
                center_xz.y + radius * a.sin(),
            )
        })
        .collect()
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
            terrain_slots: Vec::new(),
            terrain_streams: inf_editor_core::terrain_stream::EditorTerrainStreams::new(),
            render_assets: inf_editor_core::render_assets::EditorRenderAssets::new(),
            tool_status: None,
            stream_log_countdown: STREAM_LOG_INTERVAL_FRAMES,
            sculpt_drag: None,
            sculpt_ring: Vec::new(),
            sculpt_ring_color: [1.0; 4],
            foliage: FoliageSettings::default(),
            foliage_drag: None,
            foliage_stroke_seq: 0,
            last_eye_world: DVec3::ZERO,
            render_tier: None,
            render_caps: None,
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

    /// Switch the active tool (Select / Sculpt / Foliage) from the toolbar.
    /// Leaving the brush tools drops any hovered brush ring.
    pub fn set_tool_mode(&mut self, mode: ToolMode) {
        self.tool_mode = mode;
        if mode != ToolMode::Sculpt && mode != ToolMode::Foliage {
            self.sculpt_ring.clear();
        }
    }

    /// Replace the sculpt brush configuration (from the toolbar).
    pub fn set_sculpt(&mut self, sculpt: SculptSettings) {
        self.sculpt = sculpt;
    }

    /// Replace the foliage brush configuration (from the toolbar, E-P6).
    pub fn set_foliage(&mut self, foliage: FoliageSettings) {
        self.foliage = foliage;
    }

    /// Translate snap increment (world units) for 2D mode, `0.0` ⇒ none. Only
    /// the Windows input layer applies it during a gizmo drag.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn snap_2d_translate(&self) -> f32 {
        self.snap_2d.translate_snap()
    }

    /// The world transform of the render instance carrying pick id `id`, wherever
    /// it lives (P18.3).
    ///
    /// Until this batch every renderable entity was a [`MeshInstance`], so the
    /// selection-driven affordances — the gizmo snapshot, focus framing, the
    /// Local-space basis, the transform write-back — all searched one list. A
    /// `MeshRef.asset` is now a [`VgeomInstance`](inf_render::VgeomInstance) and a
    /// bound `SkeletalMesh` a [`SkinnedInstance`](inf_render::SkinnedInstance), and
    /// **an imported mesh must be exactly as manipulable as a cube**. Every one of
    /// those call sites reads through here instead, so that is true by construction
    /// rather than by remembering to add a third branch each time.
    ///
    /// Ids are unique across the three lists (one `next_id` feeds them all), so
    /// the first hit is the only hit.
    fn instance_xform(&self, id: u32) -> Option<InstanceXform> {
        if let Some(i) = self.scene.instances.iter().find(|i| i.id == id) {
            return Some(InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            });
        }
        if let Some(i) = self.scene.vgeom_instances.iter().find(|i| i.id == id) {
            return Some(InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            });
        }
        self.scene
            .skinned
            .iter()
            .find(|i| i.id == id)
            .map(|i| InstanceXform {
                translation: i.translation,
                rotation: i.rotation,
                scale: i.scale,
            })
    }

    /// Write a transform back onto whichever render list carries `id` — the
    /// mutable twin of [`instance_xform`](Self::instance_xform).
    fn set_instance_xform(&mut self, id: u32, x: InstanceXform) {
        if let Some(i) = self.scene.instances.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
            return;
        }
        if let Some(i) = self.scene.vgeom_instances.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
            return;
        }
        if let Some(i) = self.scene.skinned.iter_mut().find(|i| i.id == id) {
            i.translation = x.translation;
            i.rotation = x.rotation;
            i.scale = x.scale;
        }
    }

    /// World-space center of the current selection, if any. Reads LIVE working
    /// positions (render instances + selected 2D entities) so it tracks a gizmo
    /// drag in progress.
    fn selection_center(&self) -> Option<DVec3> {
        let mut sum = DVec3::ZERO;
        let mut n = 0.0;
        for id in &self.scene.selected {
            if let Some(inst) = self.instance_xform(*id) {
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
    ///
    /// **P18.3 — the editor asks for the meshlet path.** It never did, which meant
    /// every vgeom asset the viewport now carries would have drawn through
    /// `ClassicVgeomNode`'s discrete-LOD fallback: correct pixels, but not the
    /// streamed, budgeted, evicting P18.2 path the player uses, and therefore not
    /// "the editor streams meshlets exactly as the player does". The request is the
    /// player's, character for character (see `requested_render_settings`); the
    /// clamps below are what decide whether it is granted.
    ///
    /// Both clamps apply, from **cached** probes: `RenderTier::apply` (no meshlet
    /// path below High) and `AdapterCaps::clamp_occlusion` (the storage-texture
    /// floor two-pass occlusion needs). That pair is exactly
    /// [`inf_render::detect_and_clamp`], inlined only so the adapter is probed once
    /// per host rather than on every document version — and it closes P18.1's
    /// honest remainder (1) for the editor, which noted that this host applied the
    /// tier without the occlusion floor.
    fn apply_render_settings(&mut self, doc: &SceneDoc) {
        let tier = *self
            .render_tier
            .get_or_insert_with(|| detect_tier(&self.gpu, &RenderSettings::default()));
        let caps = *self
            .render_caps
            .get_or_insert_with(|| inf_render::AdapterCaps::probe(&self.gpu));
        let requested = requested_render_settings(&doc.settings().render);
        let mapped = caps.clamp_occlusion(tier.apply(requested));
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
        self.scene.terrains.clear();
        // P18.3: real geometry. `MeshRef.asset` entities project as virtualized
        // geometry (meshlet path, or the classic discrete-LOD fallback — the tier
        // decides which node draws it), and `SkeletalMesh` entities project as
        // real skinned draws. Both lists are rebuilt from scratch every projection,
        // exactly like `instances`.
        self.scene.vgeom_assets.clear();
        self.scene.vgeom_instances.clear();
        self.scene.skinned_meshes.clear();
        self.scene.skinned.clear();
        self.terrain_slots.clear();
        // `terrain_guid` (the tool target) is deliberately NOT cleared here — it
        // is re-validated against the new slot list at the end of the projection,
        // so a sculpt stroke (which bumps the document version on every dab, and
        // therefore re-projects) keeps aiming at the terrain it is editing.
        self.id_to_guid.clear();
        self.guid_to_id.clear();
        self.collider_outlines.clear();
        self.collider_outlines_3d.clear();
        self.joint_lines.clear();
        self.volume_outlines.clear();
        self.spot_lights.clear();
        self.spline_polylines.clear();

        let world = doc.world();
        // The sky authority first (P17.1): it writes `scene.sun` / `scene.sky` and,
        // when a clock is present, pushes the sun/moon directional light as
        // `lights[0]` — a stable index on both projector sides.
        project_sky(&mut self.scene, world);
        let w = world.world();
        let mut next_id: u32 = 1;
        // Which vgeom assets this projection has already listed (the render node
        // caches GPU geometry by id, but the asset list must not duplicate), and
        // which `(mesh, skeleton)` pairs already own a `skinned_meshes` slot.
        // MIRROR: `vgeom_seen` is the player's `project_scene` local of the same
        // name and the same purpose.
        let mut vgeom_seen: BTreeSet<u128> = BTreeSet::new();
        let mut skinned_slots: HashMap<(Uuid, Uuid), usize> = HashMap::new();
        // Every render asset this projection actually referenced (meshes,
        // skeletons, clips) — the input to the end-of-projection `retain_only`
        // audit (P16.4b's lesson in mesh form).
        let mut live_render_assets: BTreeSet<Uuid> = BTreeSet::new();
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

            // Heightfield terrain (P10.1) projects into the render scene's terrain
            // list; the terrain pass assembles clipmap LOD rings around the camera
            // for each one, every frame. Each tile carries its own change stamp
            // (P16.3b1), so re-projecting on a document change re-uploads only the
            // tiles a sculpt/paint stroke actually touched.
            //
            // P16.6 — MULTI-TERRAIN: "first visible terrain wins" is gone. EVERY
            // visible, non-empty terrain projects, in **document order**, and the
            // parallel `terrain_slots` records which of them is streamed/editable/
            // dirty.
            //
            // MIRROR, precisely: `inf_player::render::project_scene` runs the same
            // projection but walks its world in `Guid` order — the editor has a
            // document and the player does not. Both orders are deterministic for
            // their own side; what makes a PIE-vs-shipping comparison meaningful is
            // that both stamp the SAME `RenderTerrain::id` from the entity `Guid`,
            // so the two lists match up by identity rather than by index.
            //
            // P16.3b2 — THE SIM/RENDER SPLIT: an asset-backed terrain draws the
            // **streamer's** camera-driven working set; the document's `data` stays
            // exactly as authored (empty, for a streamed terrain) and is never
            // written by the camera. An inline terrain has no stream and projects
            // its own data, unchanged.
            if let Some(terrain) = w.get::<Terrain>(entity) {
                if visible {
                    let translation = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    let streamed = self.terrain_streams.ensure(
                        guid,
                        terrain,
                        translation,
                        self.last_eye_world,
                    );
                    // P16.4b — the document's authored tiles are the authority
                    // for a streamed terrain, so mirror them into the render
                    // set (pinned) before projecting. Copies nothing when
                    // nothing was edited.
                    if streamed {
                        self.terrain_streams.overlay_document_edits(guid, doc);
                    }
                    let projected = if streamed {
                        self.terrain_streams
                            .render_data(guid)
                            .filter(|d| d.tile_count() + d.coarse_tile_count() > 0)
                            .map(|d| project_terrain(guid, terrain, d, translation))
                    } else if !terrain.data.is_empty() {
                        Some(project_terrain(guid, terrain, &terrain.data, translation))
                    } else {
                        None
                    };
                    if let Some(rt) = projected {
                        self.scene.terrains.push(rt);
                        self.terrain_slots.push(TerrainSlot {
                            guid,
                            streamed,
                            editable: streamed && self.terrain_streams.is_editable(guid),
                            unsaved: streamed && terrain.data.has_dirty_tiles(),
                        });
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

            // Foliage scatter (E-P6): project every placed instance into the
            // mesh-instance path, mesh + tint taken from the referenced palette
            // slot. Instances are entity-LOCAL, so lift them by the entity's world
            // translation (the auto-created container sits at the origin; applying
            // the container's rotation/scale to instances is a documented v1
            // follow-up). A pick on any instance selects the owning Foliage entity
            // (id→guid), so the scatter is selectable by clicking its content.
            // MIRROR: the player's `render.rs` runs the same projection (no pick).
            if let Some(fol) = w.get::<Foliage>(entity) {
                if visible && !fol.instances.is_empty() {
                    if fol.instances.len() > 50_000 {
                        tracing::warn!(
                            "inf-viewport: Foliage entity has {} instances (>50k) — \
                             instanced-draw perf path is a follow-up",
                            fol.instances.len()
                        );
                    }
                    let base = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    for fi in &fol.instances {
                        let (mesh, color) = fol
                            .palette
                            .get(fi.kind as usize)
                            .map(|p| (prim_mesh(p.primitive), p.tint.to_array()))
                            .unwrap_or((PrimMesh::Cube, [0.28, 0.52, 0.24, 1.0]));
                        let id = next_id;
                        next_id += 1;
                        self.scene.instances.push(MeshInstance {
                            translation: base + fi.position.to_dvec3(),
                            rotation: foliage_rot_quat(fi.rotation),
                            scale: Vec3::splat(fi.scale as f32),
                            color,
                            metallic: 0.0,
                            roughness: 0.85,
                            emissive: [0.0; 3],
                            id,
                            mesh,
                            blend: 0,
                            cutoff: 0.5,
                        });
                        // Pick a foliage instance → select the owning entity.
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

            // Skeletal meshes (P11.1 → **P18.3**): a `SkeletalMesh` entity now
            // draws its REAL skinned geometry. The bind-space mesh comes from the
            // referenced `.inf_mesh`'s skin streams, the palette from the
            // `.inf_skel` posed by the entity's `AnimPlayer` — rest pose when there
            // is no player, no clip, or an unresolvable one, so a freshly dropped
            // character is visible immediately rather than only once it plays.
            // Both the resolution and the pose rule live in Ring 1
            // (`inf_editor_core::render_assets`), which is the only part of this
            // that Linux CI can see.
            //
            // The **placeholder cube survives** as the honest fallback: a
            // `SkeletalMesh` with no assets bound (or with a mesh carrying no skin
            // stream) is still authorable content and must stay selectable.
            //
            // NOT a mirror: the shipped player has no `SkeletalMesh` branch at all,
            // so there is nothing to keep in sync — giving it one is the matching
            // follow-up, and until then this is editor-only rendering, not a
            // divergence from a projection that exists.
            if w.get::<MeshRef>(entity).is_none() {
                if let (true, Some(sm)) = (visible, w.get::<SkeletalMesh>(entity).copied()) {
                    let affine = w
                        .get::<GlobalTransform>(entity)
                        .map(|g| g.0)
                        .unwrap_or(glam::DAffine3::IDENTITY);
                    let (scale, rot, translation) = affine.to_scale_rotation_translation();
                    let id = next_id;
                    next_id += 1;
                    live_render_assets.extend(sm.mesh);
                    live_render_assets.extend(sm.skeleton);
                    let player = w.get::<inf_ecs::components::AnimPlayer>(entity).copied();
                    live_render_assets.extend(player.and_then(|p| p.clip));
                    match self.render_assets.resolve_skinned(&sm, player.as_ref()) {
                        Some(draw) => {
                            // Real skinned geometry. PBR params come from the
                            // entity's `Material` exactly as they do on the rigid
                            // path (`Material` is what the Details panel edits);
                            // an unmaterialed character gets the renderer's neutral.
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
                            // One `skinned_meshes` entry per (mesh, skeleton)
                            // pair, and the entry is the store's own `Arc` — no
                            // copy here, and the pass keys its GPU upload on that
                            // pointer, so re-projecting an unchanged character
                            // costs neither a memcpy nor a re-upload (P18.3).
                            let slot = *skinned_slots.entry(draw.key).or_insert_with(|| {
                                self.scene.skinned_meshes.push(draw.mesh);
                                self.scene.skinned_meshes.len() - 1
                            });
                            self.scene.skinned.push(inf_render::SkinnedInstance {
                                translation,
                                rotation: rot.as_quat(),
                                scale: scale.as_vec3(),
                                color,
                                metallic,
                                roughness,
                                emissive,
                                id,
                                mesh: slot,
                                palette: draw.palette,
                            });
                        }
                        // Unbound (or unskinned) — the pre-P18.3 placeholder,
                        // unchanged down to its slate tint, so authoring a skeletal
                        // entity before its assets exist looks exactly as it did.
                        None => self.scene.instances.push(MeshInstance {
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
                        }),
                    }
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
            let mesh_ref = w.get::<MeshRef>(entity).copied().unwrap_or_default();
            let id = next_id;
            next_id += 1;
            live_render_assets.extend(mesh_ref.asset);
            // P18.3 — THE OLDEST DOCUMENTED GAP, CLOSED. A `MeshRef.asset` with a
            // derived vmesh renders REAL geometry: the GPU meshlet path (vgeom on)
            // or the classic discrete-LOD fallback (vgeom off), both driven by the
            // same vgeom scene content, with the tier deciding which node draws it.
            // An unresolved asset (or a primitive-only `MeshRef`) falls back to the
            // built-in primitive — which stays *legitimate content*, not a
            // placeholder, for the Cube/Sphere/Plane/Cylinder/Cone kinds.
            //
            // MIRROR of `inf_player::render::project_scene`'s `MeshRef` branch,
            // field for field. The one deliberate difference is where the asset id
            // comes from — the player uses the derived GUID (a pack is immutable),
            // the editor uses the derived payload's content hash (a content root is
            // not) — and the reasoning lives in `inf_editor_core::render_assets`,
            // once, rather than in two comments that could disagree.
            let vgeom = mesh_ref
                .asset
                .and_then(|mesh_id| self.render_assets.resolve_vgeom(mesh_id));
            match vgeom {
                Some(loaded) => {
                    if vgeom_seen.insert(loaded.id) {
                        // The scene carries the PAGED source, not a decoded DAG
                        // (P18.2): the render node's streamer decides what of it is
                        // resident from the camera's own screen-error wants.
                        self.scene
                            .vgeom_assets
                            .push(inf_render::VgeomAsset::new(loaded.id, loaded.source));
                    }
                    self.scene.vgeom_instances.push(inf_render::VgeomInstance {
                        asset: loaded.id,
                        translation,
                        rotation: rot.as_quat(),
                        scale: scale.as_vec3(),
                        color,
                        metallic,
                        roughness,
                        emissive,
                        id,
                    });
                }
                // R-P1: an unresolved / primitive-only MeshRef draws its built-in
                // primitive kind (Sphere/Plane/Cylinder/Cone), not always a cube.
                None => self.scene.instances.push(MeshInstance {
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id,
                    mesh: prim_mesh(mesh_ref.primitive),
                    blend,
                    cutoff,
                }),
            }
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

        // Release every terrain stream the projection did not just use (P16.4b
        // audit; P16.6: **all** live terrains, not just the first). A stream keyed
        // on an entity the projection did not touch — a terrain that became
        // invisible, was deleted, or belonged to a document that has since been
        // replaced — is dead memory holding a whole `.inf_terrain` payload, plus
        // any tile it pinned for an unsaved edit (which nothing would ever unpin).
        // This is the only place that knows which terrains are live, so it is the
        // only place that can do it.
        self.terrain_streams
            .retain_only(self.terrain_slots.iter().map(|s| s.guid));

        // The same audit for mesh assets (P18.3). A `.inf_vmesh` mapping plus its
        // decoded skinned geometry is real memory held on behalf of entities that
        // may no longer exist: a mesh unbound in the Details panel, an entity
        // deleted, or — the case P16.4b was written about — a whole document
        // replaced by File ▸ Open. The projection is the only place that knows the
        // live set, so this is the only place that can release the rest.
        self.render_assets.retain_only(live_render_assets);

        // Re-validate the tool target: keep it if that terrain is still projected
        // (so a stroke's status stays about the terrain being sculpted), else fall
        // back to the first projected terrain — which is exactly the pre-P16.6
        // behaviour for a single-terrain document.
        if !self
            .terrain_guid
            .is_some_and(|g| self.terrain_slots.iter().any(|s| s.guid == g))
        {
            self.terrain_guid = self.terrain_slots.first().map(|s| s.guid);
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

/// Project the level's **sky authority** into the renderer's sun + sky blocks
/// (P17.1) — the seam that retired `inf_render::camera::SUN_DIR`.
///
/// **MIRROR**: byte-for-byte identical in `inf_viewport::host::project_sky` and
/// `inf_player::render::project_sky`, and pinned by a parity test in each crate.
/// The one thing that *could* silently diverge — *which* entity is the authority,
/// since the editor walks document order and the player walks `Guid` order —
/// deliberately does not live here: [`inf_ecs::sky::resolve_sky`] answers it once,
/// in Ring 0, by lowest `Guid`.
///
/// With no authority the renderer's own defaults stand: the retired constant's
/// direction and the historic three-colour gradient, so every level that has not
/// opted into time of day renders exactly the pixels it always did.
///
/// When a clock is present the sun (or, once it has set, the moon) is also pushed
/// as a **directional light**, so shadows, GI and the PBR loop all follow the
/// clock without any of those passes knowing time of day exists. It goes in
/// first, before the entity loop, so its index is stable on both sides. A level
/// that would rather author its own suns sets `SkyAtmosphere::enabled = false`,
/// which keeps the clock and the tint but projects no light.
fn project_sky(scene: &mut RenderScene, world: &inf_ecs::EcsWorld) {
    let Some(sky) = inf_ecs::sky::resolve_sky(world) else {
        scene.sun = SunParams::default();
        scene.sky = SkyParams::default();
        scene.atmosphere = AtmosphereParams::default();
        return;
    };
    let a = &sky.atmosphere;
    let phase = sky.moon_phase as f32;
    scene.sun = SunParams {
        direction: sky.sun.as_vec3(),
        color: [a.sun_color.r, a.sun_color.g, a.sun_color.b],
        intensity: a.sun_intensity,
        moon_direction: sky.moon.as_vec3(),
        moon_color: [a.moon_color.r, a.moon_color.g, a.moon_color.b],
        moon_intensity: a.moon_intensity,
        moon_phase: phase,
    };
    let [zenith, horizon, ground] = sky.sky_gradient();
    scene.sky = SkyParams {
        zenith,
        horizon,
        ground,
    };
    // The **weather in force** (P17.4), resolved once in Ring 0: when the
    // weather block is enabled it *drives* cloud coverage/type, the wind and the
    // fog density; when it is not, those come from the authored fields exactly
    // as they did in v13. Which of the two applies is decided by
    // `ResolvedSky::weather`, not here — it is precisely the kind of one-line
    // derivation two byte-identical MIRROR bodies would eventually stop agreeing
    // about, which is the same reasoning that put `cloud_time_s` in Ring 0.
    let w = sky.weather();
    // The physical atmosphere (P17.2). Only the *multipliers* come from the
    // level: the Rayleigh / Mie / ozone coefficients themselves are physical
    // constants of Earth's air and stay at `AtmosphereParams::default()`, so
    // "atmosphere" cannot be mis-authored into something that is not one.
    scene.atmosphere = AtmosphereParams {
        enabled: a.physical,
        turbidity: a.turbidity,
        mie_g: a.mie_anisotropy,
        sky_intensity: a.sky_intensity,
        aerial_perspective: a.aerial_perspective,
        tint_strength: a.tint_strength,
        sun_disc_deg: a.sun_disc_deg,
        moon_disc_deg: a.moon_disc_deg,
        moon_phase: phase,
        star_intensity: a.star_intensity,
        fog: HeightFog {
            density: w.fog_density,
            falloff: a.fog_falloff,
            height: a.fog_height,
            color: [a.fog_color.r, a.fog_color.g, a.fog_color.b],
        },
        // Volumetric clouds (P17.3). `time_s` is the one field here that is
        // *derived* rather than authored: the wind drifts with the level's clock
        // (`ResolvedSky::cloud_time_s`, defined once in Ring 0) and with nothing
        // else, so two runs at the same time of day see the same sky.
        clouds: CloudParams {
            enabled: a.clouds_enabled,
            coverage: w.cloud_coverage,
            cloud_type: w.cloud_type,
            bottom: a.cloud_bottom,
            top: a.cloud_top,
            density: a.cloud_density,
            detail: a.cloud_detail,
            seed: a.cloud_seed,
            wind_x: w.wind_x,
            wind_z: w.wind_z,
            time_s: sky.cloud_time_s(),
            phase_g: a.cloud_phase_g,
            shadow_strength: a.cloud_shadow,
            ambient: a.cloud_ambient,
            color: [a.cloud_color.r, a.cloud_color.g, a.cloud_color.b],
        },
        // Precipitation (P17.4). Entirely derived: the weather block decides
        // whether it falls, how hard and how frozen, and the same wind that
        // drifts the clouds slants it. `time_s` is the level's clock again, so
        // the rain is a function of the document and of nothing else. The tint
        // is the cloud droplet colour on purpose — rain and the cloud it fell
        // out of are the same water, and a second colour field would be one more
        // thing to keep consistent for a stylised sky.
        precip: PrecipParams {
            enabled: w.precipitation > 0.0,
            intensity: w.precipitation,
            snowiness: w.snowiness,
            wind_x: w.wind_x,
            wind_z: w.wind_z,
            time_s: sky.cloud_time_s(),
            color: [a.cloud_color.r, a.cloud_color.g, a.cloud_color.b],
        },
        ..AtmosphereParams::default()
    };
    if let Some((direction, color, intensity)) = sky.key_light() {
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            color,
            intensity,
            direction: direction.as_vec3(),
            position: DVec3::ZERO,
            range: 0.0,
            cast_shadows: true,
            ..RenderLight::default()
        });
    }
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
/// `data` is the working set to draw and is passed **explicitly** (P16.3b2): for
/// an inline terrain it is `terrain.data`, for a streamed one it is the
/// streamer's camera-driven set. `terrain` still supplies the layers and macro
/// variation, which are authored, not streamed. Making the choice a parameter is
/// what keeps "which residency am I drawing?" a decision at the call site rather
/// than an assumption buried here.
///
/// Each **resident** tile becomes a [`RenderTerrainTile`] with its `f64` origin
/// offset by the entity's world translation (so the terrain follows its
/// transform), its `f32` height buffer copied out of the paged data, its height
/// bounds precomputed for the terrain pass's per-tile frustum cull, and its
/// monotone change stamp so the GPU cache re-uploads only what actually moved
/// (P16.3b1).
///
/// Level 0 (the authored heightfield) is emitted first, then the resident coarse
/// pyramid pages in ascending key order — both from `BTreeMap`s, so the tile list
/// is globally `TileKey`-ascending and the upload/draw order is deterministic. An
/// inline (non-asset) terrain holds no coarse pages, so it projects exactly the
/// level-0 list it always did.
///
/// **MIRROR** of `inf_player::render::project_terrain` — keep the two in sync.
fn project_terrain(
    guid: Uuid,
    terrain: &Terrain,
    data: &TerrainData,
    translation: DVec3,
) -> RenderTerrain {
    let res = data.tile_resolution();
    let n = (res * res) as usize;
    let project_tile = |key: inf_terrain::TileKey, tile: &inf_terrain::TerrainTile| {
        // Resolve the sparse weight store into a full res² buffer for upload
        // (an unpainted tile → uniform default layer 0; a coarse pyramid page is
        // always unpainted — the pyramid is heights-only).
        let weights: Vec<[u8; 4]> = if tile.weights_are_default() {
            vec![inf_terrain::DEFAULT_WEIGHT; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.weight_sample(res, i, j))
                .collect()
        };
        RenderTerrainTile {
            key: TerrainTileKey::new(key.lod, key.coord),
            origin: tile.origin + translation,
            heights: tile.heights().to_vec(),
            weights,
            height_bounds: tile.height_bounds(),
            version: data.tile_version(key),
        }
    };
    let tiles = data
        .tiles()
        .map(|(&coord, tile)| project_tile(inf_terrain::TileKey::lod0(coord), tile))
        .chain(
            data.coarse_tiles()
                .map(|(&key, tile)| project_tile(key, tile)),
        )
        .collect();
    let layers = std::array::from_fn(|k| RenderTerrainLayer {
        albedo: terrain.layers[k].albedo.to_array(),
        roughness: terrain.layers[k].roughness as f32,
        tex_scale: terrain.layers[k].tex_scale as f32,
    });
    RenderTerrain {
        // P16.6: the terrain entity's identity, folded to 64 bits exactly as the
        // player's mirror folds it — what keeps two terrains' GPU tile caches and
        // splat uniforms apart when their grids share coordinates.
        id: inf_render::terrain_id_from_guid(guid.as_u128()),
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        layers,
        macro_variation: terrain.macro_variation as f32,
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
        self.scene.hovered = self.pick_id(view, px, py);
    }

    /// Pick the entity GUID under the cursor (`None` = empty space). Selection
    /// itself lives in the document — the caller applies the pick to it.
    pub fn pick_guid(&mut self, view: &RenderView, px: u32, py: u32) -> Option<Uuid> {
        let id = self.pick_id(view, px, py)?;
        self.id_to_guid.get(&id).copied()
    }

    /// The render-instance id under a viewport pixel.
    ///
    /// The GPU id-buffer pass rasterizes [`RenderScene::instances`] only — the
    /// rigid primitive path — so P18.3's real geometry (virtualized meshes and
    /// skinned characters, which live in their own scene lists) would be
    /// **unclickable**: the whole point of the batch is that an imported mesh is
    /// as much an object as a cube, and an object you cannot click is not one.
    ///
    /// Extending the ID pass to a vertex-pulled indirect meshlet draw is a
    /// renderer change and belongs with the selection-outline work (see the
    /// remainder recorded in ROADMAP §12 P18.3). The stopgap is the technique the
    /// gizmo already uses and this codebase already trusts — **analytic
    /// picking**: on an id-buffer miss, ray-test the cursor against each vgeom /
    /// skinned instance's world bounding sphere and take the nearest hit.
    ///
    /// It is deliberately a *fallback*, not a first choice: whenever the id buffer
    /// answers, that answer wins, so nothing about picking a primitive changes.
    /// Ties are resolved by distance along the ray and then by id, so the result
    /// is a deterministic function of the scene and the pixel. Its honest
    /// limitation is that a bounding sphere is coarser than the silhouette: a
    /// click just outside a concave mesh can select it.
    fn pick_id(&mut self, view: &RenderView, px: u32, py: u32) -> Option<u32> {
        if let Some(id) = self.picker.pick(&self.gpu, &self.scene, view, px, py) {
            return Some(id);
        }
        if self.scene.vgeom_instances.is_empty() && self.scene.skinned.is_empty() {
            return None;
        }
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_world = self.origin.to_world(ro);
        let rd = rd.as_dvec3();

        let bounds_of: std::collections::BTreeMap<u128, ([f32; 3], f32)> = self
            .scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.bounds()))
            .collect();
        let mut best: Option<(f64, u32)> = None;
        let mut consider = |center: DVec3, radius: f64, id: u32| {
            if let Some(t) = ray_sphere_t(ro_world, rd, center, radius) {
                if best.is_none_or(|(bt, bid)| t < bt || (t == bt && id < bid)) {
                    best = Some((t, id));
                }
            }
        };
        for inst in &self.scene.vgeom_instances {
            let Some((c, r)) = bounds_of.get(&inst.asset).copied() else {
                continue;
            };
            let local = Vec3::from_array(c);
            let center = inst.translation + (inst.rotation * (local * inst.scale)).as_dvec3();
            consider(center, (r * inst.scale.abs().max_element()) as f64, inst.id);
        }
        for inst in &self.scene.skinned {
            // Skinned geometry has no cached bounding sphere on the scene DTO, so
            // the bind-space vertex extent stands in. It is computed from the same
            // buffer the pass draws, so it can never disagree with what is on
            // screen — only with where the *pose* moved it, which is the same
            // approximation the rest of this fallback makes.
            let Some(mesh) = self.scene.skinned_meshes.get(inst.mesh) else {
                continue;
            };
            let r = mesh
                .vertices
                .iter()
                .map(|v| Vec3::from_array(v.pos).length())
                .fold(0.0f32, f32::max);
            consider(
                inst.translation,
                (r * inst.scale.abs().max_element()).max(0.05) as f64,
                inst.id,
            );
        }
        best.map(|(_, id)| id)
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
                let inst = self.instance_xform(*id)?;
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
            if let Some(inst) = self.instance_xform(*id) {
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
        // Terrain surface under the cursor — the NEAREST hit across every
        // projected terrain (P16.6), resolved through `terrain_probes` so a
        // STREAMED terrain answers from the pages the streamer has actually paged
        // in. (Reading the document's own set here was the bug: it is empty for a
        // streamed terrain, so every drop fell through to the ground plane.)
        if let Some(hit) = nearest_terrain_hit(&self.terrain_probes(doc, None), ro_w, rd) {
            return hit.world;
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
            if let Some(inst) = self.instance_xform(*id) {
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
        for id in self.scene.selected.clone() {
            if let Some(inst) = self.instance_xform(id) {
                self.gizmo_initial.insert(id, inst);
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
            let mut next = init;
            match delta {
                GizmoDelta::Translate(t) => next.translation = init.translation + t,
                GizmoDelta::Rotate { axis, radians } => {
                    let q = glam::Quat::from_axis_angle(axis, radians);
                    next.rotation = q * init.rotation;
                    // Orbit the translation about the pivot too.
                    let rel = (init.translation - pivot).as_vec3();
                    next.translation = pivot + (q * rel).as_dvec3();
                }
                GizmoDelta::Scale(s) => next.scale = init.scale * s,
            }
            self.set_instance_xform(*id, next);
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

    // ── streamed content: terrain (P16.3b2) + meshes (P18.3) ──────────────

    /// Point the viewport's loose-asset streaming at a project's content root (or
    /// `None` to disable it). Rescans the `.inf_terrain` **and** render-asset
    /// indexes and drops every live stream and opened payload, so a project switch
    /// can never serve the previous project's pages or geometry.
    ///
    /// Until a root is set nothing streams, an asset-backed terrain draws its
    /// (empty) inline data and a `MeshRef.asset` draws its primitive placeholder —
    /// so an editor that never calls this behaves exactly as it did before P16.3b2
    /// / P18.3.
    ///
    /// Pushed from Ring 2's `project://changed` flow: the open project's `Content`
    /// directory, so a `Terrain.asset` authored by the import wizard resolves to a
    /// loose `.inf_terrain` and starts paging (P16.4a) and a `MeshRef.asset`
    /// resolves to its derived `.inf_vmesh` (P18.3). Both *policies* are
    /// unit-tested on all three OSes in `inf_editor_core`; this is only the call
    /// site.
    pub fn set_content_root(&mut self, root: Option<std::path::PathBuf>) {
        self.terrain_streams.set_content_root(root.clone());
        self.render_assets.set_content_root(root);
        self.terrain_slots.clear();
        self.synced_version = None; // force a re-projection
    }

    /// Rebuild the loose-asset indexes after the content database changed.
    ///
    /// Pushed from Ring 2 when an import finishes or the watcher sees an external
    /// edit: an index built when the project opened does not contain assets
    /// written after it, so a freshly imported terrain or mesh would resolve to
    /// nothing and the entity the user just spawned would draw empty (P16.4a).
    ///
    /// The two halves treat live state differently, deliberately. Terrain streams
    /// are **kept** (re-pointing the root would re-page terrain the user is flying
    /// over, and a terrain's tiles are re-read per page anyway). Opened
    /// `.inf_vmesh` payloads are **dropped**: a vmesh is opened once and then only
    /// sliced, so a payload rewritten under the same GUID would otherwise be served
    /// from the stale mapping forever. Re-opening costs a header + page-directory
    /// parse each.
    pub fn refresh_asset_index(&mut self) {
        self.terrain_streams.refresh_index();
        self.render_assets.refresh_index();
        self.synced_version = None; // force a re-projection
    }

    /// The slot for a projected terrain, if it is one (P16.6).
    fn terrain_slot(&self, guid: Uuid) -> Option<&TerrainSlot> {
        self.terrain_slots.iter().find(|s| s.guid == guid)
    }

    /// The slot the terrain tools currently target — the terrain under the cursor
    /// at the last pick, else the first projected one (P16.6).
    fn active_terrain_slot(&self) -> Option<&TerrainSlot> {
        self.terrain_guid
            .and_then(|g| self.terrain_slot(g))
            .or(self.terrain_slots.first())
    }

    /// Whether the terrain the tools are aimed at streams from a `.inf_terrain`
    /// asset.
    ///
    /// Polled each frame by the platform loop and published on
    /// `viewport://tool-status`, where the shell uses it to label the terrain.
    /// As of P16.4b it no longer greys the brush tools out — see
    /// [`terrain_is_editable`](Self::terrain_is_editable), which does.
    ///
    /// P16.6: with several terrains projected this describes the **targeted** one
    /// (the cursor's, else the first) rather than "the" terrain — which is what
    /// the status bar has to say, since it is the terrain a stroke would land on.
    pub fn terrain_is_streamed(&self) -> bool {
        self.active_terrain_slot().is_some_and(|s| s.streamed)
    }

    /// Whether the targeted **streamed** terrain can be sculpted/painted, i.e.
    /// its `.inf_terrain` is a writable file the save path can fold edits into.
    ///
    /// `false` for an inline terrain (which is always editable and needs no
    /// asset) — read it together with [`terrain_is_streamed`](Self::terrain_is_streamed):
    /// *streamed && !editable* is the one case the tools refuse.
    pub fn terrain_is_editable(&self) -> bool {
        self.active_terrain_slot().is_some_and(|s| s.editable)
    }

    /// Whether **any** projected terrain carries tiles not yet written back to its
    /// `.inf_terrain` — the toolbar's "unsaved terrain edits" chip.
    ///
    /// Deliberately the aggregate, not the targeted terrain's: the chip warns that
    /// Ctrl+S has work to do, and a stroke on terrain A must not stop reading as
    /// unsaved because the cursor has since drifted over terrain B.
    pub fn terrain_has_unsaved_edits(&self) -> bool {
        self.terrain_slots.iter().any(|s| s.unsaved)
    }

    /// Release every terrain stream — its resident pages, its edit pins, and its
    /// `.inf_terrain` payload — **and** every opened render asset (P18.3): the
    /// `.inf_vmesh` mappings and decoded skinned geometry the previous level
    /// referenced.
    ///
    /// Pushed by `File ▸ Open` / `File ▸ New` (P16.4b audit): those replace the
    /// document wholesale, so every terrain stream is keyed on entity GUIDs that no
    /// longer exist. Without this the old document's payload and any tile it pinned
    /// for an unsaved edit stay alive for the life of the process. Render assets
    /// are keyed on *asset* GUIDs rather than entity ones, so nothing there is
    /// invalidated by the swap — but everything it holds belongs to the outgoing
    /// level's working set, and the incoming projection's `retain_only` would only
    /// free it after the first frame that already paid to keep it.
    pub fn clear_streams(&mut self) {
        self.terrain_streams.clear();
        self.render_assets.clear();
        self.terrain_slots.clear();
        self.terrain_guid = None;
        self.synced_version = None; // force a re-projection against the new doc
    }

    /// Refresh the cold store of every live terrain stream **in place** — called
    /// after a save rewrote the `.inf_terrain` files.
    ///
    /// Live streams keep their resident pages and their published cut, so saving
    /// does not blink the terrain the user is looking at; only the bytes behind
    /// them, the catalog, and the edit pins change. See
    /// `EditorTerrainStreams::reload_store`.
    pub fn reload_terrain_stores(&mut self) {
        self.terrain_streams.reload_stores();
        for slot in &mut self.terrain_slots {
            slot.unsaved = false;
        }
        self.synced_version = None; // re-project from the refreshed store
    }

    /// Terrain-streaming counters, for the diagnostics path.
    pub fn terrain_stream_stats(&self) -> &inf_terrain::TerrainStreamStats {
        self.terrain_streams.stats()
    }

    /// Take the last tool-rejection message (e.g. a sculpt stroke refused on a
    /// streamed terrain), leaving none.
    ///
    /// The **status seam**: drained once per frame by the platform loop and
    /// emitted as [`ViewportEvent::ToolStatus`], which Ring 2 forwards on
    /// `viewport://tool-status` and the shell shows in the status bar. It is also
    /// still in the Output Log via `tracing`, where every other host-side
    /// diagnostic surfaces.
    pub fn take_tool_status(&mut self) -> Option<String> {
        self.tool_status.take()
    }

    /// Record a tool rejection: remember it for the caller and log it once.
    fn reject_tool(&mut self, message: &str) {
        if self.tool_status.as_deref() != Some(message) {
            tracing::warn!("inf-viewport: {message}");
        }
        self.tool_status = Some(message.to_string());
    }

    /// Advance the streamed terrain's camera-driven cut and, when it changed,
    /// re-project the render terrain from the streamer's working set.
    ///
    /// Re-projecting here (rather than waiting for a document change) is what lets
    /// pages appear as the camera flies: `sync_from_doc` is version-gated and the
    /// camera does not bump the document version — nor should it.
    fn sync_streamed_terrain(&mut self) {
        if !self.terrain_streams.sync_render(self.last_eye_world) {
            return;
        }
        // Streaming diagnostics on the existing debug path (`tracing` → the Output
        // Log / log file), throttled so a flying camera doesn't flood it. No new
        // panel, no new IPC channel.
        self.stream_log_countdown = self.stream_log_countdown.saturating_sub(1);
        if self.stream_log_countdown == 0 {
            self.stream_log_countdown = STREAM_LOG_INTERVAL_FRAMES;
            tracing::info!("inf-viewport: {}", self.terrain_stream_stats().summary());
        }
        // P16.6: every streamed terrain advances its own cut, so re-project each
        // of them into its own slot (`terrain_slots` is index-aligned with
        // `scene.terrains`, so no lookup and no reordering).
        for i in 0..self.terrain_slots.len() {
            let slot_guid = self.terrain_slots[i].guid;
            if !self.terrain_slots[i].streamed {
                continue;
            }
            if let Some((component, data, translation)) =
                self.terrain_streams.projection_inputs(slot_guid)
            {
                if data.tile_count() + data.coarse_tile_count() > 0 {
                    let rt = project_terrain(slot_guid, component, data, translation);
                    if let Some(dst) = self.scene.terrains.get_mut(i) {
                        *dst = rt;
                    }
                }
            }
        }
    }

    /// `true` while a sculpt stroke is in progress.
    pub fn is_sculpting(&self) -> bool {
        self.sculpt_drag.is_some()
    }

    /// The heightfield the cursor is actually looking at, plus the terrain's world
    /// translation.
    ///
    /// For an **inline** terrain that is the document's own `TerrainData`. For a
    /// **streamed** one (P16.4b) it is the streamer's render working set — the
    /// surface being drawn, with the document's unsaved edits already mirrored in
    /// (`overlay_document_edits`). Raycasting the document's set instead would
    /// find nothing until something had already been sculpted, which is
    /// unusable: you cannot click ground the document has not paged in yet, and
    /// paging is what a click is *for*.
    ///
    /// The consequence, stated plainly: a stroke can only start where a level-0
    /// page is resident — i.e. within the render cut's fine ring around the
    /// camera. Aiming at distant terrain that is only covered by a coarse page
    /// finds no hit and starts no stroke. Fly closer; the ring follows.
    fn terrain_probe<'a>(
        &'a self,
        doc: &'a SceneDoc,
        guid: Uuid,
    ) -> Option<(&'a inf_terrain::TerrainData, DVec3)> {
        if self.terrain_slot(guid).is_some_and(|s| s.streamed) {
            if let Some((_, data, translation)) = self.terrain_streams.projection_inputs(guid) {
                return Some((data, translation));
            }
        }
        doc.terrain_data_and_origin(guid)
    }

    /// A [`TerrainProbe`] per projected terrain, in document order — **the one
    /// place** the "which heightfield is under the cursor?" choice is made
    /// (P16.6).
    ///
    /// Every terrain-resolving path (sculpt, paint, drag-drop spawn, foliage) goes
    /// through here, so none of them can drift back to reading the document's own
    /// `TerrainData` — which is *empty by design* for a streamed terrain and would
    /// silently drop every cursor onto the `y = 0` ground plane.
    ///
    /// `restrict` narrows to a single terrain (a stroke in progress; see
    /// [`terrain_pick`](Self::terrain_pick)).
    fn terrain_probes<'a>(
        &'a self,
        doc: &'a SceneDoc,
        restrict: Option<Uuid>,
    ) -> Vec<TerrainProbe<'a>> {
        terrain_probes_of(&self.terrain_slots, restrict, |guid| {
            self.terrain_probe(doc, guid)
        })
    }

    /// Raycast the cursor against **every** projected terrain and return the
    /// NEAREST hit (P16.6): the terrain entity, the hit centre in that terrain's
    /// local XZ, and the local surface height there.
    ///
    /// Reuses the same screen→world ray as picking/gizmo drags, rebased through
    /// the floating origin and shifted into each terrain entity's local frame by
    /// [`nearest_terrain_hit`], which is where the rule (and its tie-break) lives.
    fn sculpt_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> Option<(Uuid, DVec2, f64)> {
        self.terrain_pick(doc, view, px, py, None)
    }

    /// [`sculpt_pick`](Self::sculpt_pick), optionally **restricted to one
    /// terrain**.
    ///
    /// A stroke in progress restricts to the terrain it started on: dragging the
    /// cursor over a neighbouring terrain must not silently move the brush onto
    /// it (the dabs would land in a different document entity, and the single
    /// `HeightDelta` the stroke commits belongs to exactly one terrain).
    fn terrain_pick(
        &self,
        doc: &SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
        restrict: Option<Uuid>,
    ) -> Option<(Uuid, DVec2, f64)> {
        let (ro, rd) = view.pixel_ray(px as f32, py as f32);
        let ro_w = self.origin.to_world(ro);
        let probes = self.terrain_probes(doc, restrict);
        nearest_terrain_hit(&probes, ro_w, rd.as_dvec3())
            .map(|h| (h.guid, h.local_xz, h.local_height))
    }

    /// Page the level-0 tiles a dab at `center` needs into the **document's**
    /// working set, synchronously, before the dab runs (P16.4b).
    ///
    /// A no-op for an inline terrain, whose tiles are all already there. See
    /// `inf_editor_core::terrain_edit` for why the document — and not this
    /// host's streamer — owns the tiles a brush writes.
    fn page_brush_footprint(&mut self, doc: &mut SceneDoc, guid: Uuid, center: DVec2) {
        if !self.terrain_slot(guid).is_some_and(|s| s.streamed) {
            return;
        }
        self.terrain_streams
            .page_brush_footprint(guid, doc, center, self.sculpt.radius);
    }

    /// Mirror the document's edited tiles into the render set and refresh the
    /// unsaved-edits flag — run after every dab so the stroke is visible as it is
    /// made and the status chip lights up on the first sample changed.
    fn after_terrain_edit(&mut self, doc: &SceneDoc, guid: Uuid) {
        let Some(index) = self.terrain_slots.iter().position(|s| s.guid == guid) else {
            return;
        };
        if !self.terrain_slots[index].streamed {
            return;
        }
        self.terrain_streams.overlay_document_edits(guid, doc);
        self.terrain_slots[index].unsaved = !doc.terrain_dirty_tiles(guid).is_empty();
        if let Some((component, data, translation)) = self.terrain_streams.projection_inputs(guid) {
            if data.tile_count() + data.coarse_tile_count() > 0 {
                let rt = project_terrain(guid, component, data, translation);
                if let Some(dst) = self.scene.terrains.get_mut(index) {
                    *dst = rt;
                }
            }
        }
    }

    /// Rebuild the brush-ring loop points around `center` (terrain-local XZ on
    /// `guid`), coloured by the active op. Clears the ring if the terrain vanished.
    ///
    /// P16.6: the terrain is passed in rather than read off "the" terrain field —
    /// the ring must follow the surface the cursor actually resolved to.
    fn refresh_ring(&mut self, doc: &SceneDoc, guid: Option<Uuid>, center: DVec2) {
        let op = self
            .sculpt_drag
            .as_ref()
            .map(|d| d.op)
            .unwrap_or(self.sculpt.op);
        // Paint recolours the ring by the target layer's albedo (so the swatch
        // under the cursor reads as the layer being painted); sculpt ops use
        // their fixed op colour.
        let color = if op == SculptOp::Paint {
            guid.and_then(|g| doc.terrain_layer_albedo(g, self.sculpt.paint_layer))
                .unwrap_or_else(|| op_color(op))
        } else {
            op_color(op)
        };
        self.sculpt_ring_color = color;
        if let Some(guid) = guid {
            if let Some((data, translation)) = self.terrain_probe(doc, guid) {
                let ring = build_ring(data, translation, center, self.sculpt.radius);
                self.sculpt_ring = ring;
                return;
            }
        }
        self.sculpt_ring.clear();
    }

    /// Update the hovered brush ring (idle Sculpt mode): raycast the cursor and
    /// rebuild the ring, or clear it off-terrain.
    ///
    /// P16.6: the pick resolves which terrain is under the cursor, so hovering
    /// also **retargets the tools** — the editable/read-only decision below is
    /// then made against that terrain, not against whichever one came first.
    pub fn update_sculpt_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        let hit = self.sculpt_pick(doc, view, px, py);
        if let Some((guid, _, _)) = hit {
            self.terrain_guid = Some(guid);
        }
        // A streamed terrain whose asset cannot be written has nowhere to save a
        // stroke to, so showing an inviting ring would be a lie (P16.4b — an
        // editable streamed terrain rings exactly like an inline one).
        if self.terrain_is_streamed() && !self.terrain_is_editable() {
            self.sculpt_ring.clear();
            return;
        }
        match hit {
            Some((guid, center, _)) => self.refresh_ring(doc, Some(guid), center),
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
        // P16.6: resolve which terrain the cursor is on FIRST, then judge that
        // terrain. Refusing on "the" terrain's writability while the click lands on
        // a different one is exactly the class of bug multi-terrain introduces.
        let hit = self.sculpt_pick(doc, view, px, py);
        if let Some((guid, _, _)) = hit {
            self.terrain_guid = Some(guid);
        }
        // P16.4b: a streamed terrain IS editable — its tiles page into the
        // document on demand and the save path writes them back. The only refusal
        // left is the honest one: an asset the editor cannot write, where a stroke
        // would be lost at Ctrl+S rather than saved.
        if self.terrain_is_streamed() && !self.terrain_is_editable() {
            self.reject_tool(inf_editor_core::terrain_stream::STREAMED_TERRAIN_READONLY_REJECTION);
            return false;
        }
        let Some((guid, center, height)) = hit else {
            // A miss has two very different causes on a streamed terrain. If some
            // streamed asset really has ground under the cursor, it is simply
            // paged at coarse detail and there is no level-0 page to sculpt — say
            // so (P16.4b audit: a silent no-op reads as a broken tool). Clicking
            // past the edge of every terrain is not a problem and stays silent.
            let p = self.pick_world_point(doc, view, px, py);
            for i in 0..self.terrain_slots.len() {
                let g = self.terrain_slots[i].guid;
                if !self.terrain_slots[i].streamed {
                    continue;
                }
                let local = self
                    .terrain_streams
                    .projection_inputs(g)
                    .map(|(_, _, t)| DVec2::new(p.x - t.x, p.z - t.z))
                    .unwrap_or(DVec2::new(p.x, p.z));
                if self.terrain_streams.covers_level0(g, local) {
                    self.reject_tool(
                        inf_editor_core::terrain_stream::STREAMED_TERRAIN_COARSE_REJECTION,
                    );
                    break;
                }
            }
            return false;
        };
        // Make the footprint resident in the DOCUMENT before the first dab — the
        // brush must never author over ground it has not actually read.
        self.page_brush_footprint(doc, guid, center);
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
        self.after_terrain_edit(doc, guid);
        self.sculpt_drag = Some(SculptDrag {
            guid,
            kind,
            op,
            last_local: center,
            flatten_height: height,
        });
        self.refresh_ring(doc, Some(guid), center);
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
        // Restricted to the stroke's own terrain (P16.6): sliding over a
        // neighbouring terrain holds the stroke exactly as sliding off the world
        // does, rather than teleporting the brush onto different ground.
        let Some((_, cur, _)) = self.terrain_pick(doc, view, px, py, Some(guid)) else {
            return; // cursor slid off the terrain — hold the stroke, add nothing
        };
        let settings = self.sculpt;
        let spacing = (0.35 * settings.radius).max(0.05);
        // `dab_positions` re-emits the start (`last`); skip it — already placed.
        let dabs = dab_positions(&[last, cur], spacing);
        let mut new_last = last;
        for &c in dabs.iter().skip(1) {
            // Every dab pages its own footprint: a drag walks across tiles, and a
            // dab must never write ground it has not read (P16.4b).
            self.page_brush_footprint(doc, guid, c);
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
        self.after_terrain_edit(doc, guid);
        self.refresh_ring(doc, Some(guid), cur);
    }

    /// Finish the stroke: commit the merged height [`inf_terrain::HeightDelta`] or
    /// splat [`inf_terrain::SplatDelta`] as one undo step. Returns `true` if a
    /// non-empty stroke was recorded.
    pub fn finish_sculpt(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.sculpt_drag.take() else {
            return false;
        };
        let recorded = match drag.kind {
            DragStroke::Height(stroke) => doc.edit_commit_sculpt(drag.guid, stroke),
            DragStroke::Splat(stroke) => doc.edit_commit_paint(drag.guid, stroke),
        };
        self.after_terrain_edit(doc, drag.guid);
        recorded
    }

    // ── foliage painting (E-P6) ───────────────────────────────────────────

    /// `true` while a foliage scatter stroke is in progress.
    pub fn is_painting_foliage(&self) -> bool {
        self.foliage_drag.is_some()
    }

    /// The world point under the cursor for the foliage brush centre: the terrain
    /// surface (reusing [`Self::pick_world_point`]'s terrain-then-ground rule).
    fn foliage_center(&self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) -> DVec3 {
        self.pick_world_point(doc, view, px, py)
    }

    /// Rebuild the foliage brush ring around a world-space cursor point (terrain
    /// height when over terrain, else a flat ground-plane ring), coloured green.
    fn refresh_foliage_ring(&mut self, doc: &SceneDoc, center: DVec3) {
        const FOLIAGE_RING: [f32; 4] = [0.35, 0.85, 0.40, 1.0];
        self.sculpt_ring_color = FOLIAGE_RING;
        let center_xz = DVec2::new(center.x, center.z);
        // P16.6: the ring follows the TOPMOST terrain covering the brush centre —
        // the same surface `foliage_surface_height` lifts instances onto — and it
        // is resolved through `terrain_probes`, so a streamed terrain rings on the
        // pages it has actually paged in rather than not at all.
        let probes = self.terrain_probes(doc, None);
        if let Some((guid, _)) = topmost_surface(&probes, center_xz) {
            if let Some(probe) = probes.iter().find(|p| p.guid == guid) {
                let local = DVec2::new(
                    center_xz.x - probe.translation.x,
                    center_xz.y - probe.translation.z,
                );
                self.sculpt_ring =
                    build_ring(probe.data, probe.translation, local, self.foliage.radius);
                return;
            }
        }
        self.sculpt_ring = ground_ring(center_xz, self.foliage.radius);
    }

    /// The world height foliage lands on at world XZ `p`: the topmost terrain
    /// surface covering it, else `0.0` (the ground plane) — the pre-P16.6 answer
    /// for a world with no terrain, unchanged.
    fn foliage_surface_height(&self, doc: &SceneDoc, p: DVec2) -> f64 {
        topmost_surface(&self.terrain_probes(doc, None), p)
            .map(|(_, y)| y)
            .unwrap_or(0.0)
    }

    /// Hover update (idle Foliage mode): move the brush ring to the cursor.
    pub fn update_foliage_hover(&mut self, doc: &SceneDoc, view: &RenderView, px: u32, py: u32) {
        let center = self.foliage_center(doc, view, px, py);
        self.refresh_foliage_ring(doc, center);
    }

    /// Begin a foliage scatter stroke. Resolves the target Foliage entity — the
    /// first SELECTED foliage entity, or a new one auto-created at the origin and
    /// selected — inside one undo transaction, then lays the first tick. Returns
    /// `true` (a stroke always starts; an empty result just records nothing).
    pub fn begin_foliage(
        &mut self,
        doc: &mut SceneDoc,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> bool {
        let target = doc
            .selection()
            .iter()
            .copied()
            .find(|g| doc.has_foliage(*g));
        // One undo entry for the whole stroke (auto-create + scatter, or scatter).
        doc.begin_transaction("Paint Foliage");
        let guid = match target {
            Some(g) => g,
            None => {
                let g = doc.edit_create(SpawnKind::Foliage, "Foliage", None);
                doc.select(&[g], false);
                g
            }
        };
        let origin = doc.foliage_origin(guid).unwrap_or(DVec3::ZERO);
        let original = doc.foliage_instances(guid).unwrap_or_default();
        let stroke_seq = self.foliage_stroke_seq;
        self.foliage_stroke_seq = self.foliage_stroke_seq.wrapping_add(1);
        let positions = original
            .iter()
            .map(|i| DVec2::new(i.position.x, i.position.z))
            .collect();
        self.foliage_drag = Some(FoliageDrag {
            guid,
            erase: self.foliage.erase,
            stroke_seq,
            next_sample: 0,
            origin,
            positions,
            added: Vec::new(),
            original,
            removed: BTreeSet::new(),
        });
        let center = self.foliage_center(doc, view, px, py);
        self.foliage_dab(doc, center);
        self.refresh_foliage_ring(doc, center);
        true
    }

    /// Continue the stroke: lay a tick at the current cursor. (Per-tick placement;
    /// path resampling for very fast strokes is a documented follow-up — min-
    /// spacing rejection keeps a held cursor from stacking.)
    pub fn update_foliage(&mut self, doc: &mut SceneDoc, view: &RenderView, px: u32, py: u32) {
        if self.foliage_drag.is_none() {
            return;
        }
        let center = self.foliage_center(doc, view, px, py);
        self.foliage_dab(doc, center);
        self.refresh_foliage_ring(doc, center);
    }

    /// One brush tick: place (or erase) instances around `center` (world). Live-
    /// mutates the target component so the scatter renders immediately.
    fn foliage_dab(&mut self, doc: &mut SceneDoc, center: DVec3) {
        let Some(drag) = self.foliage_drag.as_ref() else {
            return;
        };
        let (guid, erase, origin, stroke_seq, base) = (
            drag.guid,
            drag.erase,
            drag.origin,
            drag.stroke_seq,
            drag.next_sample,
        );
        let s = self.foliage;
        let center_xz = DVec2::new(center.x, center.z);

        if erase {
            let r2 = s.radius * s.radius;
            let mut newly: Vec<usize> = Vec::new();
            for (i, inst) in drag.original.iter().enumerate() {
                if drag.removed.contains(&i) {
                    continue;
                }
                let wx = origin.x + inst.position.x;
                let wz = origin.z + inst.position.z;
                let d2 = (wx - center_xz.x).powi(2) + (wz - center_xz.y).powi(2);
                if d2 <= r2 {
                    newly.push(i);
                }
            }
            if newly.is_empty() {
                return;
            }
            let kept = {
                let d = self.foliage_drag.as_mut().unwrap();
                for i in newly {
                    d.removed.insert(i);
                }
                d.original
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !d.removed.contains(i))
                    .map(|(_, x)| *x)
                    .collect::<Vec<_>>()
            };
            doc.foliage_set_instances(guid, kept);
            return;
        }

        // Place: target count = density × brush area, capped per tick.
        let area = std::f64::consts::PI * s.radius * s.radius;
        let target =
            ((s.density * area).round() as i64).clamp(0, FOLIAGE_MAX_PER_TICK as i64) as u32;
        if target == 0 {
            return;
        }
        let cands = foliage_samples(
            center_xz,
            s.radius,
            target,
            s.seed,
            stroke_seq,
            base,
            s.scale_jitter,
            s.kind,
        );
        if let Some(d) = self.foliage_drag.as_mut() {
            d.next_sample += target as u64;
        }
        // Lift each candidate onto the topmost terrain covering it (P16.6), else
        // the ground plane (scoped immutable doc borrow).
        let heights: Vec<f64> = cands
            .iter()
            .map(|c| self.foliage_surface_height(doc, c.pos_xz))
            .collect();
        let ms2 = foliage_min_spacing(s.density).powi(2);
        let mut accepted: Vec<FoliageInstance> = Vec::new();
        {
            let d = self.foliage_drag.as_mut().unwrap();
            for (c, y) in cands.iter().zip(heights) {
                let local = DVec3::new(c.pos_xz.x - origin.x, y - origin.y, c.pos_xz.y - origin.z);
                let lxz = DVec2::new(local.x, local.z);
                if d.positions
                    .iter()
                    .any(|p| (p.x - lxz.x).powi(2) + (p.y - lxz.y).powi(2) < ms2)
                {
                    continue;
                }
                let inst = FoliageInstance {
                    position: Vec3d::new(local.x, local.y, local.z),
                    rotation: Vec3d::new(0.0, c.yaw_deg, 0.0),
                    scale: c.scale,
                    kind: c.kind,
                };
                d.positions.push(lxz);
                d.added.push(inst);
                accepted.push(inst);
            }
        }
        if !accepted.is_empty() {
            doc.foliage_append(guid, &accepted);
        }
    }

    /// Finish the stroke: commit ONE `PaintFoliage` undo step (added or removed)
    /// and close the transaction opened in [`Self::begin_foliage`]. Returns `true`
    /// if the stroke changed anything (so the caller emits `WorldChanged`).
    pub fn finish_foliage(&mut self, doc: &mut SceneDoc) -> bool {
        let Some(drag) = self.foliage_drag.take() else {
            return false;
        };
        let changed = if drag.erase {
            let removed: Vec<(usize, FoliageInstance)> = drag
                .removed
                .iter()
                .map(|&i| (i, drag.original[i]))
                .collect();
            doc.edit_commit_foliage(drag.guid, Vec::new(), removed)
        } else {
            doc.edit_commit_foliage(drag.guid, drag.added, Vec::new())
        };
        // Always close the transaction begin_foliage opened (a Create may have
        // been recorded even when the scatter itself was empty).
        doc.commit_transaction();
        changed
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
        // THE RENDER-SYNC POINT (P16.3b2): advance every streamed terrain's
        // camera-driven cut exactly once per frame, here. Unlike the document
        // projection this is *not* version-gated — the cut follows the camera, and
        // the camera moves without the document changing. Nothing it does is
        // visible to the document, which is the whole point.
        self.sync_streamed_terrain();
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

        // Sculpt / foliage brush ring: a closed loop following the terrain height
        // under the cursor, coloured by the active op (Sculpt) or green (Foliage).
        if matches!(self.tool_mode, ToolMode::Sculpt | ToolMode::Foliage)
            && self.sculpt_ring.len() >= 2
        {
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

/// The deterministic foliage scatter sampler (E-P6): a pure function of its
/// inputs, so the same stroke input sequence reproduces identical instances (the
/// determinism law — no wall-clock / thread-rng).
#[cfg(test)]
mod foliage_sampler {
    use super::{foliage_min_spacing, foliage_samples, FOLIAGE_MAX_PER_TICK};
    use glam::DVec2;

    #[test]
    fn sampler_is_pure_and_reproducible() {
        let c = DVec2::new(5.0, -3.0);
        let a = foliage_samples(c, 3.0, 32, 1, 7, 0, 0.2, 2);
        let b = foliage_samples(c, 3.0, 32, 1, 7, 0, 0.2, 2);
        assert_eq!(a, b, "identical inputs must reproduce identical candidates");
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn candidates_stay_in_disk_and_carry_kind_and_jitter() {
        let c = DVec2::new(0.0, 0.0);
        let radius = 4.0;
        let jitter = 0.25;
        let cs = foliage_samples(c, radius, FOLIAGE_MAX_PER_TICK, 42, 3, 100, jitter, 5);
        for s in &cs {
            let d = (s.pos_xz - c).length();
            assert!(d <= radius + 1e-9, "sample outside brush disk: {d}");
            assert!((0.0..=360.0).contains(&s.yaw_deg));
            assert!((1.0 - jitter - 1e-9..=1.0 + jitter + 1e-9).contains(&s.scale));
            assert_eq!(s.kind, 5);
        }
    }

    #[test]
    fn different_strokes_and_indices_diverge() {
        let c = DVec2::new(1.0, 1.0);
        let s0 = foliage_samples(c, 2.0, 8, 1, 0, 0, 0.2, 0);
        let s_stroke = foliage_samples(c, 2.0, 8, 1, 1, 0, 0.2, 0);
        let s_index = foliage_samples(c, 2.0, 8, 1, 0, 8, 0.2, 0);
        assert_ne!(s0, s_stroke, "a new stroke re-seeds the scatter");
        assert_ne!(s0, s_index, "advancing the sample index draws fresh values");
    }

    #[test]
    fn min_spacing_tightens_with_density_and_has_a_floor() {
        assert!(foliage_min_spacing(0.1) > foliage_min_spacing(4.0));
        assert!(foliage_min_spacing(0.0) >= 0.05);
        assert!(foliage_min_spacing(1e9) >= 0.05);
    }
}

/// **P16.6 — how the cursor resolves against N terrains.**
///
/// These pin the two rules the multi-terrain tool paths are built on, as pure
/// functions: nearest-along-the-ray for a pick, topmost for a scatter, plus the
/// `restrict` filter that keeps a stroke on the terrain it started on. An
/// `EngineHost` needs a GPU, so the rules live in free functions and the host is
/// a one-line caller of each — which is what makes them testable at all.
#[cfg(test)]
mod terrain_resolution {
    use super::{
        nearest_terrain_hit, terrain_probes_of, topmost_surface, TerrainProbe, TerrainSlot,
    };
    use glam::{DVec2, DVec3};
    use uuid::Uuid;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// A flat 4 × 4-tile heightfield at local height `h` (9² samples @ 2 m ⇒ a
    /// 64 m square).
    fn flat(h: f64) -> inf_terrain::TerrainData {
        let mut t = inf_terrain::TerrainData::new(9, 2.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), |_, _| h);
            }
        }
        t
    }

    fn slot(g: Uuid) -> TerrainSlot {
        TerrainSlot {
            guid: g,
            streamed: false,
            editable: false,
            unsaved: false,
        }
    }

    fn probe<'a>(g: Uuid, data: &'a inf_terrain::TerrainData, at: DVec3) -> TerrainProbe<'a> {
        TerrainProbe {
            guid: g,
            data,
            translation: at,
        }
    }

    /// A straight-down ray from high above `(x, z)`.
    fn down(x: f64, z: f64) -> (DVec3, DVec3) {
        (DVec3::new(x, 500.0, z), DVec3::new(0.0, -1.0, 0.0))
    }

    /// **Nearest wins.** Two overlapping terrains under one cursor: the pick lands
    /// on the surface you can actually see, not on whichever the document lists
    /// first. Ties resolve to document order, so the answer is deterministic.
    #[test]
    fn a_pick_takes_the_nearest_of_overlapping_terrains() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        // A sits on the ground (surface y = 10); B is a raised platform over the
        // SAME footprint (surface y = 50) — so B is nearer to a camera above.
        let probes = vec![
            probe(a, &low, DVec3::ZERO),
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
        ];
        let (ro, rd) = down(20.0, 20.0);
        let hit = nearest_terrain_hit(&probes, ro, rd).expect("the ray must hit something");
        assert_eq!(hit.guid, b, "the pick fell through the nearer terrain");
        assert!((hit.world.y - 50.0).abs() < 1e-6, "{:?}", hit.world);
        assert!((hit.local_height - 0.0).abs() < 1e-6);

        // Listing order must not change the answer.
        let reversed = vec![
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
            probe(a, &low, DVec3::ZERO),
        ];
        assert_eq!(nearest_terrain_hit(&reversed, ro, rd).unwrap().guid, b);

        // Two coincident surfaces tie — resolved to the EARLIER probe (document
        // order), never to iteration luck.
        let same = flat(10.0);
        let tied = vec![probe(a, &low, DVec3::ZERO), probe(b, &same, DVec3::ZERO)];
        assert_eq!(nearest_terrain_hit(&tied, ro, rd).unwrap().guid, a);

        // A ray that misses every terrain footprint reports nothing (the caller
        // then falls back to the ground plane).
        let (miss_ro, miss_rd) = down(9_000.0, 9_000.0);
        assert!(nearest_terrain_hit(&probes, miss_ro, miss_rd).is_none());
        assert!(nearest_terrain_hit(&[], ro, rd).is_none());
    }

    /// **Restrict pins a stroke.** Mid-stroke, a nearer terrain appearing under
    /// the cursor must not move the brush: the dabs belong to one document entity
    /// and the single `HeightDelta` the stroke commits belongs to one terrain.
    #[test]
    fn a_restricted_pick_stays_on_the_terrain_the_stroke_started_on() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        let slots = [slot(a), slot(b)];
        let resolve = |g: Uuid| {
            if g == a {
                Some((&low, DVec3::ZERO))
            } else {
                Some((&high, DVec3::new(0.0, 50.0, 0.0)))
            }
        };
        let (ro, rd) = down(20.0, 20.0);

        // Unrestricted, the nearer terrain B wins (as the test above pins).
        let free = terrain_probes_of(&slots, None, resolve);
        assert_eq!(free.len(), 2);
        assert_eq!(nearest_terrain_hit(&free, ro, rd).unwrap().guid, b);

        // Restricted to A — the terrain a stroke started on — B is not even a
        // candidate, and the hit stays on A's surface.
        let pinned = terrain_probes_of(&slots, Some(a), resolve);
        assert_eq!(pinned.len(), 1);
        let hit = nearest_terrain_hit(&pinned, ro, rd).expect("A is still under the cursor");
        assert_eq!(hit.guid, a);
        assert!((hit.world.y - 10.0).abs() < 1e-6);

        // Restricting to a terrain that is not projected yields no candidates at
        // all — the stroke holds rather than jumping (`update_sculpt` returns).
        assert!(terrain_probes_of(&slots, Some(guid(99)), resolve).is_empty());
    }

    /// **Topmost wins for a scatter.** Foliage falls from above, so the ground at
    /// a point is the highest surface covering it — not the first listed, and not
    /// the nearest to a camera that may be underneath.
    #[test]
    fn a_scatter_takes_the_topmost_surface() {
        let (low, high) = (flat(10.0), flat(0.0));
        let (a, b) = (guid(1), guid(2));
        let probes = vec![
            probe(a, &low, DVec3::ZERO),
            probe(b, &high, DVec3::new(0.0, 50.0, 0.0)),
        ];
        let p = DVec2::new(20.0, 20.0);
        let (g, y) = topmost_surface(&probes, p).expect("covered");
        assert_eq!(g, b);
        assert!((y - 50.0).abs() < 1e-6);

        // Off both footprints ⇒ nothing (the caller uses the y = 0 ground plane).
        assert!(topmost_surface(&probes, DVec2::new(9_000.0, 9_000.0)).is_none());
        assert!(topmost_surface(&[], p).is_none());

        // A terrain's own world translation lifts its surface.
        let lifted = vec![probe(a, &low, DVec3::new(0.0, 7.0, 0.0))];
        assert!((topmost_surface(&lifted, p).unwrap().1 - 17.0).abs() < 1e-6);
    }

    /// **Why every terrain path must resolve through `terrain_probe`** (the P16.6
    /// audit fix): a streamed terrain's *document* heightfield is EMPTY by design —
    /// its tiles live in the `.inf_terrain` — so resolving a cursor against it
    /// finds nothing and drops silently to the `y = 0` ground plane. Only the
    /// streamer's render working set has real ground in it.
    ///
    /// This is the fixture the sculpt path already used and the drag-drop/foliage
    /// paths did not; it fails loudly if the document's set ever becomes the
    /// answer again.
    #[test]
    fn a_streamed_terrains_document_set_is_empty_but_its_streamer_has_ground() {
        use inf_editor_core::samples::{
            streamed_terrain_scene, write_streamed_terrain_asset, STREAMED_TERRAIN_TERRAIN_GUID,
        };
        use inf_editor_core::terrain_stream::EditorTerrainStreams;

        let dir = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(dir.path()).unwrap();
        let doc = streamed_terrain_scene();
        let terrain = {
            let world = doc.world();
            let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
            world
                .world()
                .get::<inf_ecs::components::Terrain>(e)
                .unwrap()
                .clone()
        };

        // (1) The DOCUMENT's set — what the buggy call sites read — is empty, so
        //     no probe over it resolves any surface anywhere.
        let (doc_data, doc_origin) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("the entity exists");
        assert!(doc_data.is_empty(), "a streamed terrain ships no tiles");
        let doc_probes = vec![probe(STREAMED_TERRAIN_TERRAIN_GUID, doc_data, doc_origin)];
        let p = DVec2::new(64.0, 64.0);
        assert!(
            topmost_surface(&doc_probes, p).is_none(),
            "the document's set must have no ground — that is the bug's cause"
        );
        let (ro, rd) = down(p.x, p.y);
        assert!(nearest_terrain_hit(&doc_probes, ro, rd).is_none());

        // (2) The STREAMER's render set does have ground there, once the camera
        //     has paged it in — which is what `terrain_probe` hands back.
        let mut streams = EditorTerrainStreams::new();
        streams.set_content_root(Some(dir.path().to_path_buf()));
        let eye = DVec3::new(p.x, 40.0, p.y);
        assert!(
            streams.ensure(STREAMED_TERRAIN_TERRAIN_GUID, &terrain, DVec3::ZERO, eye),
            "the fixture terrain must stream"
        );
        for _ in 0..32 {
            streams.sync_render(eye);
        }
        let (_, live, translation) = streams
            .projection_inputs(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("the stream is live");
        assert!(!live.is_empty(), "the camera paged nothing in");
        let live_probes = vec![probe(STREAMED_TERRAIN_TERRAIN_GUID, live, translation)];

        let (_, y) = topmost_surface(&live_probes, p).expect("streamed ground under the cursor");
        assert!(
            y.abs() > 1e-9,
            "the streamed surface read as flat zero — the generator has relief here"
        );
        let hit = nearest_terrain_hit(&live_probes, ro, rd).expect("the ray must hit the ground");
        // The two agree to within the raycaster's marching step: `height_at`
        // bilinearly interpolates the samples while `raycast_terrain` walks the
        // ray and interpolates at the crossing, so they differ by the sub-cell
        // residual — not by "one of them found nothing", which is the claim here.
        assert!((hit.world.y - y).abs() < 1e-3, "{hit:?} vs {y}");
    }
}

#[cfg(test)]
mod sky_projection_tests {
    use super::{project_sky, RenderScene, SkyParams, SunParams};
    use inf_ecs::components::{SkyAtmosphere, TimeOfDay};
    use inf_ecs::EcsWorld;
    use uuid::Uuid;

    fn world_with(tod: TimeOfDay, atmos: SkyAtmosphere) -> EcsWorld {
        let mut w = EcsWorld::new();
        let e = w.spawn_with_guid(Uuid::from_u128(1), "Sky", None);
        w.world_mut().entity_mut(e).insert(tod);
        w.world_mut().entity_mut(e).insert(atmos);
        w
    }

    // NOTE: the **MIRROR gate** that compares this crate's `project_sky` against
    // the shipped player's, character for character, deliberately does NOT live
    // here: this module is `#[cfg(any(windows, target_os = "macos"))]`, so a test
    // inside it is invisible to the Linux CI leg. It lives in
    // `inf-editor-core/tests/projector_mirror.rs`, which compiles on all three
    // platforms and reads both files as source text.

    /// No time-of-day authority ⇒ the renderer's own defaults stand, which are
    /// bit-for-bit the retired `SUN_DIR` and the historic gradient. This is the
    /// promise that keeps every pre-P17.1 level and golden byte-identical.
    #[test]
    fn a_clockless_world_projects_the_retired_defaults() {
        let mut scene = RenderScene::default();
        scene.sun.intensity = 999.0; // deliberately dirty, to prove it is reset
        scene.lights.clear();
        project_sky(&mut scene, &EcsWorld::new());
        assert_eq!(scene.sun, SunParams::default());
        assert_eq!(scene.sky, SkyParams::default());
        assert!(scene.lights.is_empty(), "no clock ⇒ no sun light");
    }

    /// A daytime clock publishes the sun as `lights[0]` and leaves the authored
    /// gradient untouched (the sky only dims once the sun is near the horizon).
    #[test]
    fn a_daytime_clock_publishes_the_sun() {
        let mut scene = RenderScene::default();
        let w = world_with(TimeOfDay::default(), SkyAtmosphere::default());
        project_sky(&mut scene, &w);
        assert!(scene.sun.direction.y > 0.5, "{:?}", scene.sun);
        assert_eq!(scene.sun.intensity, 3.0);
        assert_eq!(scene.sky, SkyParams::default(), "daytime tint is untouched");
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.lights[0].kind, inf_render::LightKind::Directional);
        assert!(scene.lights[0].cast_shadows);
        assert_eq!(
            scene.lights[0].direction, scene.sun.direction,
            "the key light must be the projected sun"
        );
    }

    /// At night the moon takes over as the key light and the gradient darkens.
    #[test]
    fn a_night_clock_publishes_the_moon_and_darkens_the_sky() {
        let mut scene = RenderScene::default();
        let w = world_with(
            TimeOfDay {
                seconds: 0.0,
                ..TimeOfDay::default()
            },
            SkyAtmosphere::default(),
        );
        project_sky(&mut scene, &w);
        assert!(scene.sun.direction.y < 0.0, "the sun has set");
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.lights[0].intensity, 0.15);
        assert_eq!(scene.lights[0].direction, scene.sun.moon_direction);
        assert!(
            scene.sky.zenith[2] < SkyParams::default().zenith[2],
            "the night sky must darken: {:?}",
            scene.sky
        );
    }

    /// `enabled: false` keeps the clock and the tint but authors no light.
    #[test]
    fn a_disabled_atmosphere_projects_no_light() {
        let mut scene = RenderScene::default();
        let w = world_with(
            TimeOfDay::default(),
            SkyAtmosphere {
                enabled: false,
                ..SkyAtmosphere::default()
            },
        );
        project_sky(&mut scene, &w);
        assert!(scene.lights.is_empty());
        assert!(scene.sun.direction.y > 0.5, "the sun is still projected");
    }
}

/// The analytic pick fallback that keeps real geometry clickable (P18.3).
///
/// The rule is pure, so it is testable here without a GPU — which matters,
/// because the ID-buffer pass it backs up cannot be exercised headlessly at all.
#[cfg(test)]
mod analytic_pick {
    use super::ray_sphere_t;
    use glam::DVec3;

    const FWD: DVec3 = DVec3::new(0.0, 0.0, -1.0);

    #[test]
    fn a_ray_through_a_sphere_hits_at_its_near_surface() {
        // Sphere of radius 1 at z = -10, looking down -Z from the origin.
        let t = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -10.0), 1.0)
            .expect("a ray straight at a sphere hits it");
        assert!((t - 9.0).abs() < 1e-9, "near surface, got {t}");
    }

    #[test]
    fn a_ray_beside_a_sphere_misses() {
        assert!(ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(3.0, 0.0, -10.0), 1.0).is_none());
    }

    /// Pointing away from a sphere is a miss — otherwise clicking the sky would
    /// select whatever happens to be behind the camera.
    #[test]
    fn a_sphere_behind_the_eye_misses() {
        assert!(ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, 10.0), 1.0).is_none());
    }

    /// Standing inside an object and clicking must select it, not fall through.
    #[test]
    fn a_ray_starting_inside_hits_at_zero() {
        let t = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -0.5), 5.0).unwrap();
        assert_eq!(t, 0.0);
    }

    /// Nearest-along-the-ray is the rule the fallback resolves overlaps with, so
    /// the ordering it depends on has to be the real one.
    #[test]
    fn nearer_spheres_report_smaller_t() {
        let near = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -5.0), 1.0).unwrap();
        let far = ray_sphere_t(DVec3::ZERO, FWD, DVec3::new(0.0, 0.0, -50.0), 1.0).unwrap();
        assert!(near < far, "{near} !< {far}");
    }
}

/// The editor's render-settings request (P18.3). Pure, so the decision that puts
/// the viewport on the streamed meshlet path is testable without an adapter —
/// which matters, because the bug this pins was invisible: the classic fallback
/// draws the *same geometry*, so nothing looked wrong while the editor silently
/// skipped every part of P18.2.
#[cfg(test)]
mod requested_settings {
    use super::{apply_record, requested_render_settings, RenderSettings, RenderSettingsRecord};
    use inf_render::RenderTier;

    #[test]
    fn the_editor_asks_for_the_meshlet_path() {
        let req = requested_render_settings(&RenderSettingsRecord::default());
        assert!(
            req.vgeom.enabled,
            "the editor must REQUEST vgeom — `VgeomSettings::default()` is off, so \
             without this every imported mesh draws through the classic fallback"
        );
    }

    /// The request changes **only** the vgeom master switch: the level's authored
    /// block still decides everything else, exactly as before P18.3.
    #[test]
    fn nothing_else_moves() {
        let rec = RenderSettingsRecord {
            exposure: 1.75,
            taa: true,
            gi_enabled: true,
            ..RenderSettingsRecord::default()
        };
        let base = apply_record(&rec);
        let req = requested_render_settings(&rec);
        assert_eq!(
            RenderSettings {
                vgeom: base.vgeom,
                ..req
            },
            base,
            "the opt-in must touch nothing but `vgeom.enabled`"
        );
        // …and within vgeom, only `enabled`.
        assert_eq!(
            inf_render::VgeomSettings {
                enabled: base.vgeom.enabled,
                ..req.vgeom
            },
            base.vgeom
        );
    }

    /// The tier still has the last word: a machine without the meshlet path gets
    /// the classic fallback, exactly as the player does. Requesting is not forcing.
    #[test]
    fn the_tier_still_clamps_it_away() {
        let req = requested_render_settings(&RenderSettingsRecord::default());
        assert!(!RenderTier::Medium.apply(req).vgeom.enabled);
        assert!(!RenderTier::Low.apply(req).vgeom.enabled);
        assert!(RenderTier::High.apply(req).vgeom.enabled);
    }
}
