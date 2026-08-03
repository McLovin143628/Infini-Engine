//! The player's render host + its own ECS→[`RenderScene`] projection (P9.3
//! item 1). Ring-2 engine code with **no editor deps**: it uses `inf-render`
//! exactly as `inf-viewport`'s `EngineHost` does (floating origin, reverse-Z, all
//! existing passes) but reads a plain `EcsWorld` instead of the editor's
//! `SceneDoc`.
//!
//! The projection (`project_scene`) mirrors the *shape* of
//! `inf_viewport::host::rebuild_scene` — meshes, lights, sprites, tilemaps, text,
//! 9-slices, 2D lights, billboards — but is duplicated here rather than shared,
//! because the editor version depends on `inf-editor-core`. A shared Ring-0
//! projection crate is a documented follow-up (both would then read `&EcsWorld`).
//!
//! Textures behave as in the editor viewport: the player has no asset-DB in this
//! thread yet, so referenced sprites/tilemaps render as the renderer's white
//! fallback tinted by their color (a colored quad). Uploading real texture bytes
//! is the same follow-up the editor documents.

use std::sync::Arc;

use glam::{DVec3, Vec2, Vec3};
use uuid::Uuid;

use inf_ecs::components::{
    BlendMode, ComputedVisibility, Foliage, GlobalTransform, Light, Light2D,
    LightKind as EcsLightKind, Material, MeshRef, NineSlice, PcgVolume, Primitive,
    ScatteredInstance, SkeletalMesh, Spline, Sprite, Terrain, Text2D, TextAlign, Tilemap,
    WaterBody, WaterKind,
};
use inf_ecs::{Guid, Vec3d};
use inf_math::FloatingOrigin;
use inf_render::{
    detect_tier, expand_nine_slice, expand_text, handle_from_guid, AtmosphereParams, BloomSettings,
    CloudParams, EngineRenderer, GiSettings, GpuContext, HAlign, HeightFog, LightKind,
    MeshInstance, NineSliceParams, PrebatchedRun, PrecipParams, PrimMesh, RenderChunk, RenderLight,
    RenderLight2D, RenderScene, RenderSettings, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, RenderTilemap, RenderView, RenderWater, ScatterBatch, ScatterData,
    ScatterInstance, ShadowSettings, SkinnedInstance, SkyParams, SsaoSettings, SunParams,
    SurfaceChain, TerrainTileKey, TextParams, TilemapParams, VgeomAsset, VgeomInstance,
    BUILTIN_FONT_TEXTURE,
};
use inf_scene::RenderSettingsRecord;

use crate::runtime_sim::RuntimeSim;
use crate::skinned::SkinnedRegistry;
use crate::vmesh::VmeshRegistry;

/// Owns the GPU stack + the render scene the player draws each frame.
pub struct PlayerRenderHost {
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    scene: RenderScene,
    origin: FloatingOrigin,
    /// The cook-derived `.inf_vmesh` DAGs a `MeshRef.asset` resolves to (P13.4);
    /// empty for the `--demo` / primitive-only worlds. Set via [`set_vmeshes`].
    ///
    /// [`set_vmeshes`]: PlayerRenderHost::set_vmeshes
    vmeshes: Arc<VmeshRegistry>,
    /// The skeletal render assets a `SkeletalMesh` resolves to (bind-space
    /// geometry + skeletons + clips), from the loaded pack / dev-dir; inert for
    /// the `--demo` / PIE / browser worlds. Set via [`set_skinned`].
    ///
    /// [`set_skinned`]: PlayerRenderHost::set_skinned
    skinned: Arc<SkinnedRegistry>,
    /// Whether the auto-picked [`RenderTier`](inf_render::RenderTier) enables the
    /// GPU meshlet path (High). Off → the classic discrete-LOD fallback renders the
    /// same vgeom content (the renderer's `ClassicVgeomNode`).
    vgeom_enabled: bool,
}

impl PlayerRenderHost {
    /// Build the render host over an already-created surface + GPU context (the
    /// window module owns the winit window and makes the surface from it). `record`
    /// is the loaded level's scene-persisted render block (R-P4) — post / exposure
    /// / lighting — mapped onto the base [`RenderSettings`]; pass
    /// [`RenderSettingsRecord::default`] for content with no authored block.
    pub fn new(
        gpu: GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        record: RenderSettingsRecord,
    ) -> Result<Self, String> {
        let chain = SurfaceChain::new(&gpu, surface, width, height)?;
        // Record the GPU adapter for the crash report (P15.2) as a first-class
        // field (it also already appears in the tracing log tail).
        crate::log::set_adapter_info(format!("{:?}", gpu.adapter.get_info()));
        let mut renderer = EngineRenderer::new(&gpu, chain.target_format());

        // R-P4: start from the level's persisted render block (exposure / dither /
        // bloom / ssao / taa / shadows / gi) instead of pure defaults, so the
        // shipped player looks like the authored scene — the mirror of the editor
        // viewport's `apply_render_settings`.
        //
        // Auto-tier (P13.4.2): probe the adapter, pick a render tier, and apply it
        // to the renderer's settings. High enables the GPU meshlet path; Medium/Low
        // fall back to the classic discrete-LOD path (and Low drops the expensive
        // post effects). The decision is logged by `detect_tier`.
        let base = apply_record(&record);
        let tier = detect_tier(&gpu, &base);
        // Desktop requests the meshlet path (the tier clamps it on Medium/Low);
        // mobile/web (P14.1) clamps the level's block down to the mobile ceiling —
        // no vgeom, no SSAO/GI/TAA/bloom/shadows — then the live-adapter tier
        // applies on top (Low still drops what little remains).
        #[cfg(any(target_arch = "wasm32", target_os = "android"))]
        let requested = inf_render::RenderTier::clamp_mobile(base);
        #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
        let requested = RenderSettings {
            vgeom: inf_render::VgeomSettings {
                enabled: true,
                ..base.vgeom
            },
            ..base
        };
        let settings = tier.apply(requested);
        let vgeom_enabled = settings.vgeom.enabled;
        renderer.set_settings(settings);

        Ok(Self {
            gpu,
            chain,
            renderer,
            scene: RenderScene {
                grid_enabled: false,
                ..Default::default()
            },
            origin: FloatingOrigin::default(),
            vmeshes: Arc::new(VmeshRegistry::new()),
            skinned: Arc::new(SkinnedRegistry::new()),
            vgeom_enabled,
        })
    }

    /// Attach the cook-derived vmesh registry (from the loaded pack / dev-dir) so
    /// `MeshRef.asset` entities render their real geometry — through the GPU meshlet
    /// path (High tier) or the classic discrete-LOD fallback (otherwise). Empty for
    /// primitive-only worlds.
    pub fn set_vmeshes(&mut self, vmeshes: Arc<VmeshRegistry>) {
        self.vmeshes = vmeshes;
    }

    /// Attach the skeletal render-asset store (from the loaded pack / dev-dir) so
    /// a bound `SkeletalMesh` renders its real, posed skinned geometry instead of
    /// a placeholder cube — the shipped half of the editor viewport's P18.3
    /// projection. Inert for primitive-only worlds.
    ///
    /// `Arc`-shared with nothing else today, but `Arc` on purpose: the registry
    /// owns one `Arc<SkinnedMeshData>` per mesh asset, and a device-loss rebuild
    /// has to hand the *same* store to the new host or every skinned upload would
    /// be cold again for no reason.
    pub fn set_skinned(&mut self, skinned: Arc<SkinnedRegistry>) {
        self.skinned = skinned;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.chain.request_resize(width, height);
    }

    /// Current requested surface size (physical px).
    pub fn size(&self) -> (u32, u32) {
        self.chain.requested_size()
    }

    /// The floating origin (the camera rebases against it before rendering).
    pub fn origin(&self) -> FloatingOrigin {
        self.origin
    }

    /// Rebuild the render scene from the sim's world, interpolated by `alpha`.
    pub fn project(&mut self, sim: &RuntimeSim, alpha: f64) {
        project_scene_with_skinned(&mut self.scene, sim, alpha, &self.vmeshes, &self.skinned);
    }

    /// Stroke the **world-partition cell overlay** into the debug-line layer
    /// (P16.5) — one wireframe box per streamed cell, coloured by its state.
    ///
    /// A deliberately separate, opt-in step *after* [`project`](Self::project)
    /// rather than a branch inside it: this is engine debug geometry behind
    /// `--debug-cells`, and keeping it out of the projection makes it obvious at
    /// the call site that a shipped player draws none of it. It reads only
    /// [`CellStreaming`](crate::cell_stream::CellStreaming) state and writes only
    /// into `scene.debug`, so there is no path from here back into the sim.
    ///
    /// The boxes are 1 m tall slabs sitting on the cell footprint at `y = 0` —
    /// enough to read the grid from a ground camera without occluding the world.
    pub fn draw_cell_overlay(&mut self, sim: &RuntimeSim) {
        use crate::cell_stream::CellState;
        let cells = sim.cell_streaming();
        if cells.is_empty() {
            return;
        }
        self.scene.debug.clear();
        for coord in cells.available() {
            let color = match cells.cell_state(coord) {
                CellState::Active => [0.25, 0.95, 0.35, 1.0],
                CellState::Loaded => [0.95, 0.80, 0.20, 1.0],
                CellState::Cold => [0.35, 0.38, 0.45, 1.0],
                CellState::Failed => [0.95, 0.25, 0.25, 1.0],
            };
            let (min, max) = cells.cell_bounds(coord);
            let center = DVec3::new(
                (min[0] + max[0]) * 0.5,
                CELL_OVERLAY_HALF_HEIGHT_M,
                (min[1] + max[1]) * 0.5,
            );
            let half = Vec3::new(
                ((max[0] - min[0]) * 0.5) as f32,
                CELL_OVERLAY_HALF_HEIGHT_M as f32,
                ((max[1] - min[1]) * 0.5) as f32,
            );
            self.scene.debug.wire_box(
                self.origin.to_render(center),
                half,
                glam::Quat::IDENTITY,
                color,
            );
        }
    }

    /// Whether the GPU meshlet path is active (the auto-picked tier is High).
    pub fn vgeom_enabled(&self) -> bool {
        self.vgeom_enabled
    }

    /// Render one frame for `view`. Handles device-lost recovery like the editor
    /// host (rebuilds nothing here — the caller rebuilds the whole stack on loss;
    /// transient acquire failures skip the frame).
    pub fn render(&mut self, view: &RenderView) {
        let Some(frame) = self.chain.acquire(&self.gpu) else {
            return; // transient (occluded/timeout) — skip
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
    }

    /// Whether the GPU device was lost (the caller rebuilds the stack).
    pub fn is_lost(&self) -> bool {
        self.gpu.is_lost()
    }
}

/// Half-height (metres) of a cell-overlay wireframe slab. Low enough to read the
/// grid from a ground camera without boxing the world in.
const CELL_OVERLAY_HALF_HEIGHT_M: f64 = 0.5;

/// Fill `scene` from `sim`'s world, blending actor positions by `alpha`.
/// Deterministic `Guid` iteration order. `vmeshes` resolves a `MeshRef.asset` to
/// its cook-derived meshlet DAG (P13.4) — a resolved mesh renders real geometry
/// (meshlet path or classic fallback), an unresolved one falls back to a placeholder
/// cube instance (as before).
///
/// `pub` for the same reason [`project_terrain`] is: this DTO is the **entire**
/// input the renderer consumes, so a gate can assert what a frame would draw —
/// two terrains, their ids, their resident pages — without a GPU, and compare it
/// between a cooked run and an editor-document run.
pub fn project_scene(
    scene: &mut RenderScene,
    sim: &RuntimeSim,
    alpha: f64,
    vmeshes: &VmeshRegistry,
) {
    project_scene_with_skinned(scene, sim, alpha, vmeshes, &SkinnedRegistry::new());
}

/// [`project_scene`] plus the skeletal store a bound `SkeletalMesh` resolves
/// against — the **whole** projection, and what the shipped render host calls.
///
/// The four-argument [`project_scene`] is kept as the narrower door because a
/// dozen existing gates drive it and none of them carry a character; it projects
/// with an inert registry, which makes every `SkeletalMesh` fall back to its
/// placeholder exactly as it did before this batch. A host that has skeletal
/// content must call **this**.
pub fn project_scene_with_skinned(
    scene: &mut RenderScene,
    sim: &RuntimeSim,
    alpha: f64,
    vmeshes: &VmeshRegistry,
    skinned: &SkinnedRegistry,
) {
    scene.instances.clear();
    scene.lights.clear();
    scene.sprites.clear();
    scene.tilemaps.clear();
    scene.prebatched.clear();
    scene.lights_2d.clear();
    scene.vgeom_assets.clear();
    scene.vgeom_instances.clear();
    // P18.3's follow-up: real skinned geometry. Both lists are rebuilt from
    // scratch every projection, exactly like `instances` — the bind-space payload
    // behind each entry is `Arc`-shared with the store, so re-projecting an
    // unchanged character re-uses the GPU upload even though the list is rebuilt.
    scene.skinned_meshes.clear();
    scene.skinned.clear();
    scene.terrains.clear();
    // P18.5: GPU-instanced scatter (PCG volumes + painted foliage) is rebuilt from
    // scratch every projection, exactly like `instances`. The payload behind each
    // batch is content-keyed, so re-projecting an unchanged scatter re-uses the
    // GPU buffers even though the list itself is rebuilt.
    scene.scatter.clear();
    // P20.1: water bodies are rebuilt from scratch every projection, like
    // `scatter` — a body's whole state is a pure function of its component, its
    // spline and the level clock, so there is nothing to carry over.
    scene.waters.clear();
    // Track which vmesh assets are already listed this frame (dedup — the render
    // node caches GPU geometry by id, but the asset list must not duplicate), and
    // which `(mesh, skeleton)` pairs already own a `skinned_meshes` slot.
    // MIRROR: both are `inf_viewport::host::rebuild_scene`'s locals of the same
    // names and the same purpose.
    let mut vgeom_seen: std::collections::HashSet<u128> = std::collections::HashSet::new();
    let mut skinned_slots: std::collections::HashMap<(Uuid, Uuid), usize> =
        std::collections::HashMap::new();

    let world = sim.world();
    // The sky authority first (P17.1): it writes `scene.sun` / `scene.sky` and,
    // when a clock is present, pushes the sun/moon directional light as
    // `lights[0]` — a stable index on both projector sides.
    project_sky(scene, world);
    // The clock and wind every water body responds to, resolved ONCE per
    // projection in Ring 0 (`inf_ecs::sky`) so the two MIRROR projectors cannot
    // disagree about what "now" and "the wind" mean — the same reasoning that put
    // `ResolvedSky::cloud_time_s` there.
    let water_env = inf_ecs::sky::water_environment(world);
    // P20.4: the level's terrains, borrowed once, so a river's foam can read the
    // P19.1 flow map. MIRROR of the editor host's line.
    let water_flow = inf_ecs::hydro::terrain_flow(world);
    let w = world.world();

    // Guid-sorted entity list (mirrors doc.order()'s determinism without a doc).
    let mut ents: Vec<(Uuid, inf_ecs::Entity)> = w
        .iter_entities()
        .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
        .collect();
    ents.sort_by_key(|(g, _)| *g);

    let mut next_id: u32 = 1;
    for (guid, entity) in ents {
        let visible = w
            .get::<ComputedVisibility>(entity)
            .map(|c| c.0)
            .unwrap_or(true);
        if !visible {
            continue;
        }

        // Interpolated translation for actors; static geometry uses its global.
        let base = w
            .get::<GlobalTransform>(entity)
            .map(|g| g.translation())
            .unwrap_or(DVec3::ZERO);
        let translation = sim.interp_translation(guid, alpha).unwrap_or(base);

        if let Some(light) = w.get::<Light>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            scene.lights.push(project_light(light, &affine));
        }
        if let Some(sprite) = w.get::<Sprite>(entity) {
            scene.sprites.push(project_sprite(sprite, translation));
        }
        if let Some(light2d) = w.get::<Light2D>(entity) {
            scene.lights_2d.push(project_light2d(light2d, translation));
        }
        if let Some(nine) = w.get::<NineSlice>(entity) {
            scene.prebatched.push(project_nine_slice(nine, translation));
        }
        if let Some(text) = w.get::<Text2D>(entity) {
            if let Some(run) = project_text(text, translation) {
                scene.prebatched.push(run);
            }
        }
        if let Some(tilemap) = w.get::<Tilemap>(entity) {
            if !tilemap.is_empty() {
                scene.tilemaps.push(project_tilemap(tilemap, translation));
            }
        }
        // Heightfield terrain (P10.6): the player projects **every** visible,
        // non-empty terrain into the render scene's terrain list, exactly like the
        // editor viewport host (`inf_viewport::host::project_terrain`), so
        // cooked/PIE terrain renders. Per-tile change stamps ride along (P16.3b1),
        // so the terrain pass re-uploads a height texture only when that tile
        // really changed.
        //
        // P16.6 — MULTI-TERRAIN: the old "first visible terrain wins" rule is
        // gone. Terrains arrive in `Guid` order (this loop's order), and each
        // carries `terrain_id_from_guid(guid)` so the renderer's per-tile texture
        // cache and per-terrain splat uniform stay separate — two terrains
        // routinely share tile coordinates.
        //
        // MIRROR, precisely: the editor viewport emits terrains in the DOCUMENT's
        // entity order, not `Guid` order. Both are deterministic for their own
        // side; what makes a PIE-vs-shipping comparison of the projected scene
        // meaningful is that both stamp the SAME `id` from the entity `Guid`, so
        // the two lists match up by identity rather than by index.
        //
        // P16.3b2 — THE SIM/RENDER SPLIT: an asset-backed terrain draws the
        // **streamer's** camera-driven working set, not the component's. The
        // component's set is the sim's (level-0 pages around the sim's entities);
        // projecting it would put the camera's cut and the sim's residency in the
        // same container, which is exactly the coupling the doctrine forbids.
        // An inline terrain has no streamer and projects its own data, unchanged.
        if let Some(terrain) = w.get::<Terrain>(entity) {
            let data = sim
                .terrain_streaming()
                .render_data(guid)
                .unwrap_or(&terrain.data);
            if !data.is_empty() || data.coarse_tile_count() > 0 {
                let mut rt = project_terrain(terrain, data, translation);
                rt.id = inf_render::terrain_id_from_guid(guid.as_u128());
                scene.terrains.push(rt);
            }
        }
        // PCG scatter volumes (P18.5): the volume's evaluated instance cache
        // (populated on load by the level builder) projects as ONE GPU-instanced
        // scatter batch instead of one `MeshInstance` per instance. The payload
        // uploads once per content change and the cull compute does frustum + HZB
        // + distance banding per instance.
        //
        // The volume's authored `draw_distance` now RIDES ON THE BATCH rather than
        // being culled by the host, and that is what finally makes the two hosts
        // agree about it: the editor used to cull against its own camera eye on the
        // CPU while the player ignored the field entirely, so a shipped build drew
        // strictly more scatter than its preview.
        //
        // MIRROR: `push_pcg_scatter` matches `inf_viewport::host`'s PCG projection
        // (minus its pick-id map).
        if let Some(vol) = w.get::<PcgVolume>(entity) {
            if !vol.evaluated.is_empty() {
                let id = next_id;
                next_id += 1;
                push_pcg_scatter(scene, vol, translation, id);
            }
        }
        // P19.3 — THE TERRAIN'S BIOME POPULATION: the terrain-level sibling of
        // the volume branch above. Each painted biome's `.inf_pcg` graph is
        // evaluated over the region its id owns and the merged result lives in
        // the derived, never-persisted `Terrain::biome_population` (rebuilt by
        // the editor's evaluate command, and by the player on level load —
        // which is exactly what makes the two paths comparable).
        //
        // It goes through the SAME `push_scatter` body as a volume, so a
        // population cannot be packed, shaded, culled or picked differently
        // from a volume's scatter.
        //
        // MIRROR: `inf_viewport::host` runs the same branch (plus the visibility
        // gate and the pick-id map, both host-local).
        if let Some(terrain) = w.get::<Terrain>(entity) {
            if !terrain.biome_population.is_empty() {
                let id = next_id;
                next_id += 1;
                push_biome_population(scene, terrain, translation, id);
            }
        }
        // Water surfaces (P20.1): an ocean, a lake or a spline river. A river
        // reads the `Spline` on THIS SAME ENTITY for its centreline — component
        // composition, not a reference, so there is nothing to resolve and
        // nothing to dangle.
        // MIRROR: `inf_viewport::host` runs the same branch (minus the pick-id
        // map, which is host-local), through the same `project_water` body.
        if let Some(water) = w.get::<WaterBody>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let id = next_id;
            next_id += 1;
            let body = project_water(
                water,
                w.get::<Spline>(entity),
                &affine,
                water_env,
                &water_flow,
                id,
            );
            if body.drawable() {
                scene.waters.push(body);
            }
        }
        // Foliage scatter (P18.5): painted instances project as GPU-instanced
        // scatter batches, one per primitive kind the palette resolves.
        // MIRROR: `push_foliage_scatter` matches `inf_viewport::host`'s Foliage
        // projection so the shipped player and the editor viewport draw the same
        // scatter.
        if let Some(fol) = w.get::<Foliage>(entity) {
            if !fol.instances.is_empty() {
                let id = next_id;
                next_id += 1;
                push_foliage_scatter(scene, fol, translation, id);
            }
        }
        // Skeletal meshes (P11.1 → the P18.3 follow-up): a `SkeletalMesh` entity
        // draws its REAL skinned geometry. The bind-space mesh comes from the
        // referenced `.inf_mesh`'s skin streams, the palette from the `.inf_skel`
        // posed by the entity's `AnimPlayer` — rest pose when there is no player,
        // no clip, or an unresolvable one, so a character in a freshly loaded
        // level is visible immediately rather than only once it plays. Both the
        // resolution and the pose rule live in [`crate::skinned`], which is where
        // the editor's Ring-1 store is mirrored character for character.
        //
        // The **placeholder cube survives** as the honest fallback: a
        // `SkeletalMesh` with no assets bound (or with a mesh carrying no skin
        // stream) is still authorable content that must draw as *something*.
        //
        // MIRROR of `inf_viewport::host::rebuild_scene`'s skeletal branch, pinned
        // field for field by `inf-editor-core`'s `tests/projector_mirror.rs`.
        // Until this batch the editor had this branch and the shipped player had
        // none at all, so a level with a character previewed in PIE and shipped as
        // nothing — a live PIE-vs-shipping divergence, not a missing feature.
        //
        // Host-local, as on the vgeom path: `translation` is the sim's
        // **interpolated** actor position here and the raw affine's in the editor
        // (the editor has no fixed-step interpolation to do), and `id` comes from
        // this host's own counter over `Guid` order.
        if w.get::<MeshRef>(entity).is_none() {
            if let Some(sm) = w.get::<SkeletalMesh>(entity).copied() {
                let affine = w
                    .get::<GlobalTransform>(entity)
                    .map(|g| g.0)
                    .unwrap_or(glam::DAffine3::IDENTITY);
                let (scale, rot, _t) = affine.to_scale_rotation_translation();
                let id = next_id;
                next_id += 1;
                let player = w.get::<inf_ecs::components::AnimPlayer>(entity).copied();
                match skinned.resolve_skinned(&sm, player.as_ref()) {
                    Some(draw) => {
                        // Real skinned geometry. PBR params come from the entity's
                        // `Material` exactly as they do on the rigid path; an
                        // unmaterialed character gets the renderer's neutral.
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
                        // One `skinned_meshes` entry per (mesh, skeleton) pair,
                        // and the entry is the store's own `Arc` — no copy here,
                        // and the pass keys its GPU upload on that pointer, so
                        // re-projecting an unchanged character costs neither a
                        // memcpy nor a re-upload (P18.3). **This is the sharing
                        // convention the projector has to follow**, and it is the
                        // reason `skinned_meshes` is a `Vec<Arc<_>>` at all.
                        let slot = *skinned_slots.entry(draw.key).or_insert_with(|| {
                            scene.skinned_meshes.push(draw.mesh);
                            scene.skinned_meshes.len() - 1
                        });
                        scene.skinned.push(SkinnedInstance {
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
                    // Unbound (or unskinned) — the editor's placeholder, down to
                    // its slate tint, so the two hosts also agree about content
                    // whose assets are missing.
                    None => scene.instances.push(MeshInstance {
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
            }
        }
        if let Some(mesh_ref) = w.get::<MeshRef>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, _t) = affine.to_scale_rotation_translation();
            // MIRROR: this Material→MeshInstance projection is duplicated in the
            // editor viewport's `host.rs` (inf-viewport) — keep the two in sync,
            // R-P5 blend + cutoff included. (The vgeom path below is opaque-only —
            // vgeom translucency is deferred.)
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

            // P13.4: a MeshRef.asset with a cook-derived vmesh renders REAL geometry
            // — the GPU meshlet path (vgeom on) or the classic discrete-LOD fallback
            // (vgeom off), both driven by the same vgeom scene content. The tier the
            // renderer settings carry picks which node draws it. An unresolved asset
            // (or a primitive-only MeshRef) falls back to a placeholder cube.
            let vgeom = mesh_ref.asset.and_then(|mesh_id| vmeshes.resolve(mesh_id));
            if let Some((asset_id, source)) = vgeom {
                if vgeom_seen.insert(asset_id) {
                    // The scene carries the PAGED source, not a decoded DAG
                    // (P18.2): the render node's streamer decides what of it is
                    // resident from the camera's own screen-error wants.
                    scene.vgeom_assets.push(VgeomAsset::new(asset_id, source));
                }
                scene.vgeom_instances.push(VgeomInstance {
                    asset: asset_id,
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id: next_id,
                });
            } else {
                // R-P1: an unresolved / primitive-only MeshRef draws its built-in
                // primitive kind (Sphere/Plane/Cylinder/Cone), not always a cube.
                scene.instances.push(MeshInstance {
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id: next_id,
                    mesh: prim_mesh(mesh_ref.primitive),
                    blend,
                    cutoff,
                });
            }
            next_id += 1;
        }
    }

    scene.mark_dirty();
}

/// Map an ECS [`Primitive`] to the renderer's [`PrimMesh`] (R-P1).
///
/// MIRROR: keep identical to `inf_viewport::host::prim_mesh` (the editor
/// viewport's ECS→RenderScene projection). Both seams must agree so the shipped
/// player and the editor viewport draw the same geometry for a given primitive.
fn prim_mesh(p: Primitive) -> PrimMesh {
    match p {
        Primitive::Cube => PrimMesh::Cube,
        Primitive::Sphere => PrimMesh::Sphere,
        Primitive::Plane => PrimMesh::Plane,
        Primitive::Cylinder => PrimMesh::Cylinder,
        Primitive::Cone => PrimMesh::Cone,
    }
}

/// ONE `ScatterBatch` from a list of [`ScatteredInstance`]s anchored at
/// `translation` (P18.5) — the whole body of every scatter path that speaks in
/// scattered instances: a [`PcgVolume`]'s evaluated cache and P19.3's terrain
/// **biome population**. Written once so the two cannot drift, which is the same
/// argument that pins the two hosts against each other.
///
/// PCG scatter has always drawn as a placeholder cube — kind→real-mesh upload is
/// the same documented viewport gap as sprites/tilemaps — and that does not change
/// here; only how the cubes reach the GPU does. The payload replaces one
/// `MeshInstance` per scattered instance with a content-keyed buffer uploaded once
/// per content change and culled per-instance on the GPU.
///
/// **`draw_distance` rides on the batch now.** The editor used to cull it against
/// its own camera eye on the CPU and the player ignored the field entirely, so a
/// shipped build drew strictly more scatter than its preview. The cull compute
/// honours it for both hosts, which is what finally makes them agree. `0` means
/// unlimited — the renderer's own bands then have sole charge.
///
/// The whole batch takes ONE pick `id`: a scatter is authored, moved and deleted
/// as a whole, so it is one object as far as selection is concerned.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_scatter(
    scene: &mut RenderScene,
    instances: &[ScatteredInstance],
    translation: DVec3,
    draw_distance: f64,
    id: u32,
) {
    if instances.is_empty() {
        return;
    }
    let data = ScatterData::build(
        PrimMesh::Cube,
        translation,
        instances.iter().map(|si| ScatterInstance {
            position: si.position,
            rotation: si.rotation.as_quat(),
            scale: si.scale as f32,
            color: pcg_kind_color(si.kind),
        }),
    );
    scene.scatter.push(ScatterBatch {
        data: Arc::new(data),
        anchor: translation,
        metallic: 0.0,
        roughness: 0.75,
        emissive: [0.0; 3],
        id,
        draw_distance,
    });
}

/// Project a [`PcgVolume`]'s evaluated cache into ONE GPU-instanced scatter batch
/// (P18.5), anchored at the volume entity's world `translation`, carrying the
/// volume's authored content draw distance. Body: [`push_scatter`].
fn push_pcg_scatter(scene: &mut RenderScene, vol: &PcgVolume, translation: DVec3, id: u32) {
    push_scatter(scene, &vol.evaluated, translation, vol.draw_distance, id)
}

/// Project a [`Terrain`]'s **biome population** — P19.3's biome→PCG binding, i.e.
/// each painted biome's `.inf_pcg` graph evaluated over the region its id owns —
/// into ONE GPU-instanced scatter batch. Body: [`push_scatter`], so a population
/// and a volume are packed, shaded, culled and picked by the very same code.
///
/// Instance positions are already ABSOLUTE world positions (the binding evaluates
/// against the terrain's world heightfield), so the batch anchors at the terrain's
/// own origin exactly as a volume anchors at its centre.
///
/// **Draw distance `0` = UNLIMITED, deliberately.** A `PcgVolume` has an authored
/// per-volume knob and it can only clamp the renderer's bands DOWN
/// (`inf_render::ScatterSettings`); a terrain population has no such authored
/// field, so `0` leaves the global `ScatterSettings` — the host's own tier-clamped
/// budget — in sole charge, rather than inventing a content limit nobody authored.
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_biome_population(scene: &mut RenderScene, terrain: &Terrain, translation: DVec3, id: u32) {
    push_scatter(scene, &terrain.biome_population, translation, 0.0, id)
}

/// Project a [`Foliage`] component's painted instances into GPU-instanced scatter
/// batches (P18.5): mesh + tint from the referenced palette slot.
///
/// Instances are entity-LOCAL, so the batch anchor is the entity `translation` and
/// the packed offsets are the local positions with **no conversion**. That is what
/// makes the payload a pure function of the paint stroke: the same stroke placed
/// twice content-hashes to one GPU upload however far apart the two entities sit
/// (the anchor is deliberately not part of `ScatterData::key`).
///
/// The palette resolves a primitive kind PER INSTANCE and one batch draws exactly
/// one kind, so instances bucket by resolved kind in authored order and the buckets
/// emit in [`PrimMesh::ALL`] order — deterministic, and independent of which kinds
/// the palette happens to use.
///
/// Every batch of one entity shares ONE pick `id` (see [`push_pcg_scatter`]).
///
/// MIRROR: identical in `inf_viewport::host` and `inf_player::render`, pinned by
/// `inf-editor-core`'s `tests/projector_mirror.rs`.
fn push_foliage_scatter(scene: &mut RenderScene, fol: &Foliage, translation: DVec3, id: u32) {
    if fol.instances.is_empty() {
        return;
    }
    let mut buckets: [Vec<ScatterInstance>; PrimMesh::ALL.len()] = Default::default();
    for fi in &fol.instances {
        let (mesh, color) = fol
            .palette
            .get(fi.kind as usize)
            .map(|p| (prim_mesh(p.primitive), p.tint.to_array()))
            .unwrap_or((PrimMesh::Cube, [0.28, 0.52, 0.24, 1.0]));
        buckets[mesh.index()].push(ScatterInstance {
            // Entity-LOCAL, paired with the ZERO build-anchor below.
            position: fi.position.to_dvec3(),
            rotation: foliage_rot_quat(fi.rotation),
            scale: fi.scale as f32,
            color,
        });
    }
    for (k, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let data = ScatterData::build(PrimMesh::ALL[k], DVec3::ZERO, bucket);
        scene.scatter.push(ScatterBatch {
            data: Arc::new(data),
            anchor: translation,
            metallic: 0.0,
            roughness: 0.85,
            emissive: [0.0; 3],
            id,
            draw_distance: 0.0,
        });
    }
}

/// Euler-degrees (YXZ) → quaternion for a foliage instance's stored rotation,
/// matching `inf_ecs::Transform::quat` (and the editor viewport's mirror) exactly.
fn foliage_rot_quat(rot: Vec3d) -> glam::Quat {
    glam::DQuat::from_euler(
        glam::EulerRot::YXZ,
        rot.y.to_radians(),
        rot.x.to_radians(),
        rot.z.to_radians(),
    )
    .as_quat()
}

/// Project the ECS [`BlendMode`] into the renderer's packed `blend` code (R-P5):
/// 0 opaque, 1 masked, 2 translucent. Mirrored in the editor viewport's `host.rs`.
fn blend_code(b: BlendMode) -> u8 {
    match b {
        BlendMode::Opaque => 0,
        BlendMode::Masked => 1,
        BlendMode::Translucent => 2,
    }
}

/// Map the level's scene-persisted [`RenderSettingsRecord`] onto a live
/// [`RenderSettings`] (R-P4). The record carries the authorable subset; every
/// other field (hdr, vgeom, tier_override, and the shadow/GI tuning knobs) stays
/// at `RenderSettings::default()`, so
/// `apply_record(&RenderSettingsRecord::default()) == RenderSettings::default()`
/// — pinned by the unit test below.
///
/// MIRROR: keep identical to `inf_viewport::host::apply_record` (the editor
/// viewport's copy over the editor-core `RenderSettingsRecord`). Both seams must
/// agree so the shipped player and the editor viewport apply a level's render
/// block the same way (preview == shipping).
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

/// Project an ECS [`Terrain`] (+ world translation) into a [`RenderTerrain`],
/// mirroring `inf_viewport::host::project_terrain`: each **resident** tile becomes
/// a [`RenderTerrainTile`] (heights + resolved RGBA8 splat weights + resolved
/// biome ids + precomputed height bounds + its monotone change stamp), plus the
/// four material layers + macro variation.
///
/// `data` is the working set to draw and is passed **explicitly** (P16.3b2): for
/// an inline terrain it is `terrain.data`, for a streamed one it is the
/// streamer's camera-driven set. `terrain` still supplies the layers and macro
/// variation, which are authored, not streamed. Making the choice a parameter is
/// what keeps "which residency am I drawing?" a decision at the call site rather
/// than an assumption buried here.
///
/// Level 0 (the authored heightfield) is emitted first, then the resident coarse
/// pyramid pages in ascending key order — both from `BTreeMap`s, so the tile list
/// is globally `TileKey`-ascending and the upload/draw order is deterministic.
///
/// The stamps are what keep the GPU cache hot: `project_scene` rebuilds this DTO
/// every frame, but a tile's stamp only advances when the tile is actually
/// mutated, so a static (or streamed-but-settled) terrain re-uploads nothing.
/// That replaces the old constant `TERRAIN_VERSION` — which was correct only
/// while terrain could never change (P16.3b1).
///
/// **MIRROR** of `inf_viewport::host::project_terrain` — keep the two in sync.
///
/// `pub` so the streaming gate can assert **rendered-frame determinism** without a
/// GPU: the DTO this returns is the entire input to the terrain pass, so hashing
/// it across two runs is exactly "the same frame was drawn". (Excluding
/// `version`, which is a process-global cache stamp and deliberately not
/// reproducible — see `TerrainData::tile_version`.)
pub fn project_terrain(
    terrain: &Terrain,
    data: &inf_terrain::TerrainData,
    translation: DVec3,
) -> RenderTerrain {
    let res = data.tile_resolution();
    let n = (res * res) as usize;
    let project_tile = |key: inf_terrain::TileKey, tile: &inf_terrain::TerrainTile| {
        // A coarse pyramid page is always unpainted (the pyramid is heights-only),
        // so it resolves to the uniform default like any unpainted tile.
        let weights: Vec<[u8; 4]> = if tile.weights_are_default() {
            vec![inf_terrain::DEFAULT_WEIGHT; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.weight_sample(res, i, j))
                .collect()
        };
        // Biome ids (P19.2) resolve exactly like the weights: the sparse default is
        // expanded here so the DTO the renderer sees is always dense, and a coarse
        // pyramid page — never painted — is all-`UNASSIGNED_BIOME`.
        let biomes: Vec<u8> = if tile.biomes_are_default() {
            vec![inf_terrain::UNASSIGNED_BIOME; n]
        } else {
            (0..res)
                .flat_map(|j| (0..res).map(move |i| (i, j)))
                .map(|(i, j)| tile.biome_sample(res, i, j))
                .collect()
        };
        RenderTerrainTile {
            key: TerrainTileKey::new(key.lod, key.coord),
            origin: tile.origin + translation,
            heights: tile.heights().to_vec(),
            weights,
            biomes,
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
        // The caller stamps the terrain entity's identity (P16.6); a bare
        // projection is "unkeyed", which is exactly right for the single-terrain
        // callers (the gates' DTO fingerprints) that never reach a GPU cache.
        id: 0,
        tile_resolution: res,
        meters_per_sample: data.meters_per_sample(),
        tiles,
        layers,
        macro_variation: terrain.macro_variation as f32,
        // EMPTY on purpose (P19.2). The palette is a property of the level's
        // `BiomeSet` asset, and `Terrain::biome_set` is a GUID: resolving it needs
        // an asset DB, which this projection deliberately does not have (it takes
        // an ECS component + a heightfield and nothing else — the same reason
        // layer *textures* never reached it either). The renderer pads every slot
        // with the unassigned colour, so a shipped build that somehow lands in the
        // Biomes view draws uniform neutral rather than reading a stale palette.
        // The mode is an EDITOR view mode — the viewport host, which does hold the
        // DB, is where a real palette is projected from.
        biome_palette: Vec::new(),
    }
}

/// A distinct placeholder colour per PCG kind index (mirrors the viewport host's
/// `pcg_kind_color`), so a multi-kind scatter reads as varied content before real
/// meshes upload.
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
/// Project a [`WaterBody`] (+ the [`Spline`] on the **same entity**, for a river)
/// into a [`RenderWater`] (P20.1).
///
/// MIRROR: this body is byte-identical in `inf_viewport::host` and
/// `inf_player::render`, and `projector_mirror.rs` compares it character for
/// character — like `project_sky`, and for the same reason: neither Ring-0 crate
/// can host it (`inf-render` does not depend on `inf-ecs`, and `inf-ecs` must not
/// depend on `inf-render`), so it is written twice on purpose and gated.
///
/// The two things that *could* silently diverge live in Ring 0 instead:
/// [`inf_ecs::sky::water_environment`] decides what clock and wind a body sees,
/// and [`WaterBody::effective_wind`] decides whether this body follows them. A
/// host that inlined either would be exactly the drift this gate exists to stop.
///
/// `env` is `(level clock in seconds, weather wind (m/s))` — resolved once per
/// projection, never per body, and never from a wall clock.
fn project_water(
    water: &WaterBody,
    spline: Option<&Spline>,
    affine: &glam::DAffine3,
    env: (f64, (f64, f64)),
    flow: &inf_ecs::hydro::TerrainFlow<'_>,
    id: u32,
) -> RenderWater {
    let (time_s, weather_wind) = env;
    let (wind_x, wind_z) = water.effective_wind(weather_wind);
    // A river's ripple travels DOWNSTREAM: its wave frame is (arc length,
    // lateral), so the "wind" is +1 along the river rather than a world
    // direction. Everything else responds to the level's wind.
    let river = water.kind == WaterKind::River;
    let spec = inf_render::WaveSpec {
        amplitude_m: water.wave_amplitude_m,
        wavelength_m: water.wave_length_m,
        steepness: water.wave_steepness,
        wind_x: if river { 1.0 } else { wind_x },
        wind_z: if river { 0.0 } else { wind_z },
        // Degrees at the component boundary, radians below it (the units
        // doctrine); the conversion is a multiply, so it stays bit-portable.
        spread_rad: water.wave_spread_deg.to_radians(),
        seed: water.wave_seed,
        count: water.wave_count,
    };
    let mut out = RenderWater {
        id,
        kind: match water.kind {
            WaterKind::Ocean => inf_render::WaterKindGpu::Ocean,
            WaterKind::Lake => inf_render::WaterKindGpu::Lake,
            WaterKind::River => inf_render::WaterKindGpu::River,
        },
        level_m: water.level_m,
        center: glam::DVec2::new(affine.translation.x, affine.translation.z),
        half_extent: glam::DVec2::new(water.extent.x.max(0.0), water.extent.y.max(0.0)),
        frames: Vec::new(),
        // Forwarded, not dropped (P20.3): `RenderWater::surface` hands it back to
        // the Ring-0 `RiverPath` so the renderer's reconstruction is the path the
        // projector built, flag and all.
        spline_closed: spline.is_some_and(|sp| sp.closed),
        waves: inf_render::WaveField::from_spec(&spec),
        time_s,
        flow_speed_m_s: 0.0,
        shallow_color: [
            water.shallow_color.r,
            water.shallow_color.g,
            water.shallow_color.b,
        ],
        deep_color: [water.deep_color.r, water.deep_color.g, water.deep_color.b],
        absorption: [
            water.absorption.x.max(0.0) as f32,
            water.absorption.y.max(0.0) as f32,
            water.absorption.z.max(0.0) as f32,
        ],
        roughness: water.roughness.clamp(0.0, 1.0) as f32,
        refraction_m: water.refraction_m.max(0.0) as f32,
        shore_fade_m: water.shore_fade_m.max(0.0) as f32,
        opacity: water.opacity.clamp(0.0, 1.0) as f32,
        foam_color: [water.foam_color.r, water.foam_color.g, water.foam_color.b],
        foam_crest_threshold: water.foam_crest_threshold.clamp(0.0, 1.0) as f32,
        foam_shore_m: water.foam_shore_m.max(0.0) as f32,
        foam_flow_m_s: water.foam_flow_m_s.max(0.0) as f32,
    };
    // A river's centreline is the spline on this same entity, in world space.
    // No spline ⇒ no ribbon, and `RenderWater::drawable` skips it: an authoring
    // state, not an error.
    if river {
        if let Some(sp) = spline {
            let points: Vec<DVec3> = sp
                .points
                .iter()
                .map(|p| affine.transform_point3(p.to_dvec3()))
                .collect();
            let interp = match sp.interp {
                inf_ecs::components::SplineInterp::Linear => inf_math::spline::SplineInterp::Linear,
                inf_ecs::components::SplineInterp::CatmullRom => {
                    inf_math::spline::SplineInterp::CatmullRom
                }
            };
            // ONE sanitizer, in Ring 0 (P20.4): the cook, the fixed step and both
            // projectors all build their profile here, so a negative authored
            // depth cannot taper one of them differently from the others.
            let profile = inf_render::RiverProfile::authored(
                water.river_width_start_m,
                water.river_width_end_m,
                water.river_depth_start_m,
                water.river_depth_end_m,
                water.river_flow_m_s,
            );
            let path = inf_render::RiverPath::from_points(&points, sp.closed, interp, &profile);
            out.flow_speed_m_s = path.flow_speed_m_s;
            out.level_m = path
                .frames
                .first()
                .map(|f| f.center.y)
                .unwrap_or(water.level_m);
            // P20.4: the P19.1 flow map modulates each frame's foam. The gain
            // is `1.0` wherever the terrain was never eroded, so this loop is
            // the identity on every level that has no bake — and the whole query
            // is skipped when the level has none, which is the common case.
            let mapped = flow.is_mapped();
            out.frames = path
                .frames
                .iter()
                .map(|f| {
                    let mut wf = inf_render::WaterFrame::from(f);
                    if mapped {
                        wf.flow_gain = flow.foam_gain_at(glam::DVec2::new(f.center.x, f.center.z));
                    }
                    wf
                })
                .collect();
        }
    }
    out
}

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
/// **MIRROR** of `inf_viewport::host::project_light` — kept byte-for-byte
/// identical (the parity tests in both crates pin the shared conventions):
///  * directional/spot store the vector *toward* the light = `rot · +Z` (forward
///    is `−Z`, so this is the anti-emission direction); the renderer derives a
///    spot's beam emission as `−direction = rot · −Z`;
///  * cone half-angles → cosines CPU-side; `range`/`cast_shadows` pass through
///    for all kinds (`cast_shadows` inert for point/spot — shadow maps deferred).
fn project_light(light: &Light, affine: &glam::DAffine3) -> RenderLight {
    let (_, rot, translation) = affine.to_scale_rotation_translation();
    let c = light.color.to_array();
    let color = [c[0], c[1], c[2]];
    match light.kind {
        EcsLightKind::Directional => RenderLight {
            kind: LightKind::Directional,
            color,
            intensity: light.intensity,
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

fn project_sprite(sprite: &Sprite, translation: DVec3) -> inf_render::SpriteInstance {
    inf_render::SpriteInstance {
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

fn billboard_mode(mode: inf_ecs::BillboardMode) -> u8 {
    match mode {
        inf_ecs::BillboardMode::None => inf_render::BILLBOARD_NONE,
        inf_ecs::BillboardMode::Spherical => inf_render::BILLBOARD_SPHERICAL,
        inf_ecs::BillboardMode::Cylindrical => inf_render::BILLBOARD_CYLINDRICAL,
    }
}

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

fn project_light2d(light: &Light2D, translation: DVec3) -> RenderLight2D {
    let c = light.color.to_array();
    RenderLight2D {
        color: [c[0], c[1], c[2]],
        intensity: light.intensity,
        radius: light.radius,
        position: translation,
    }
}

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

#[cfg(test)]
mod render_settings_tests {
    use super::{apply_record, RenderSettings, RenderSettingsRecord};

    /// The default record maps to the byte-stable renderer default — this pins the
    /// mapping so a settings-less level renders exactly as today's defaults (and
    /// identical to the editor viewport's mirror `apply_record`).
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

/// Spot-light seam parity (R-P3). This is the **byte-identical mirror** of
/// `inf_viewport::host`'s `project_light_parity` test — same fixture, same
/// hardcoded expectations — so the toward-the-light / emission direction
/// convention can never drift between the player and the editor viewport.
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

/// Foliage projection mirror (E-P6, reshaped by P18.5): a `Foliage` component's
/// instances survive a serde round-trip and project — with no instance lost or
/// invented — into GPU-instanced scatter batches, one per primitive kind the
/// palette resolves. The shipped player relies on this so a level with painted
/// foliage draws in PIE == shipping (the editor viewport runs the identical
/// projection).
#[cfg(test)]
mod foliage_projection {
    use super::{push_foliage_scatter, PrimMesh, RenderScene};
    use glam::DVec3;
    use inf_ecs::components::{Foliage, FoliageInstance, FoliagePaletteEntry, Primitive};
    use inf_ecs::{Color, Vec3d};

    fn demo_foliage() -> Foliage {
        Foliage {
            palette: vec![
                FoliagePaletteEntry {
                    primitive: Primitive::Cone,
                    tint: Color::new(0.3, 0.6, 0.28, 1.0),
                },
                FoliagePaletteEntry {
                    primitive: Primitive::Sphere,
                    tint: Color::new(0.6, 0.5, 0.2, 1.0),
                },
            ],
            instances: (0..7)
                .map(|i| FoliageInstance {
                    position: Vec3d::new(i as f64, 0.0, (i % 3) as f64),
                    rotation: Vec3d::new(0.0, 20.0 * i as f64, 0.0),
                    scale: 1.0 + 0.05 * i as f64,
                    kind: (i % 2) as u32,
                })
                .collect(),
        }
    }

    #[test]
    fn foliage_round_trips_and_projects_to_matching_instance_count() {
        let fol = demo_foliage();
        // Round-trip the whole component (instances are serde-persisted).
        let bytes = serde_json::to_string(&fol).unwrap();
        let back: Foliage = serde_json::from_str(&bytes).unwrap();
        assert_eq!(
            back.instances, fol.instances,
            "instances survive round-trip"
        );

        // Project into a fresh scene. P18.5: the instances no longer expand into
        // `RenderScene::instances` one by one — they pack into scatter batches, so
        // the count that must be preserved is the SUM across batches.
        let mut scene = RenderScene::default();
        let anchor = DVec3::new(10.0, 0.0, -5.0);
        push_foliage_scatter(&mut scene, &back, anchor, 7);
        let total: usize = scene.scatter.iter().map(|b| b.data.len()).sum();
        assert_eq!(total, back.instances.len(), "no instance lost or invented");
        assert!(
            scene.instances.is_empty(),
            "scatter must not also expand into the per-instance mesh path"
        );

        // Two kinds are painted (alternating), so exactly two batches — emitted in
        // `PrimMesh::ALL` order (Sphere before Cone), NOT in first-use order.
        assert_eq!(scene.scatter.len(), 2);
        assert!(matches!(scene.scatter[0].data.mesh, PrimMesh::Sphere));
        assert!(matches!(scene.scatter[1].data.mesh, PrimMesh::Cone));

        // Kind 1 → the second palette slot (Sphere): it lands in that primitive's
        // batch, tinted by that slot. Kinds alternate over 7 instances, so kind 0
        // (Cone) has 4 and kind 1 (Sphere) has 3.
        assert_eq!(scene.scatter[0].data.len(), 3);
        assert_eq!(scene.scatter[1].data.len(), 4);
        assert_eq!(
            scene.scatter[0].data.instances[0].color,
            [0.6, 0.5, 0.2, 1.0]
        );
        assert_eq!(
            scene.scatter[1].data.instances[0].color,
            [0.3, 0.6, 0.28, 1.0]
        );

        for b in &scene.scatter {
            // The entity translation is the ANCHOR, not baked into the offsets…
            assert_eq!(b.anchor, anchor);
            // …and every batch of one entity carries the one pick id it was given.
            assert_eq!(b.id, 7);
            assert_eq!(b.draw_distance, 0.0);
        }
        // Instance 0 is kind 0 (Cone) at the local origin — offsets are the LOCAL
        // positions with no conversion.
        assert_eq!(scene.scatter[1].data.instances[0].offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_foliage_projects_nothing() {
        let mut scene = RenderScene::default();
        push_foliage_scatter(&mut scene, &Foliage::default(), DVec3::ZERO, 1);
        assert!(scene.scatter.is_empty());
        assert!(scene.instances.is_empty());
    }

    /// **Content addressing** (P18.5): two foliage entities painted with the same
    /// stroke share one GPU upload, however far apart they sit. That only holds
    /// because the packed offsets are the entity-LOCAL positions and the world
    /// anchor is deliberately *not* part of `ScatterData::key` — pack against the
    /// world position instead and every duplicated prop becomes a second upload.
    #[test]
    fn identical_foliage_content_hashes_to_the_same_key() {
        let a_fol = demo_foliage();
        let b_fol = demo_foliage();
        let mut a = RenderScene::default();
        let mut b = RenderScene::default();
        push_foliage_scatter(&mut a, &a_fol, DVec3::new(10.0, 0.0, -5.0), 1);
        push_foliage_scatter(&mut b, &b_fol, DVec3::new(-4000.0, 12.0, 900.0), 2);

        assert_eq!(a.scatter.len(), b.scatter.len());
        assert!(!a.scatter.is_empty(), "the fixture must actually scatter");
        for (x, y) in a.scatter.iter().zip(&b.scatter) {
            assert_eq!(
                x.data.key(),
                y.data.key(),
                "the same stroke must content-key the same at a different anchor"
            );
        }
        assert_ne!(a.scatter[0].anchor, b.scatter[0].anchor);
    }
}
