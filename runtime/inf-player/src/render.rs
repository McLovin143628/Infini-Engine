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
    ComputedVisibility, GlobalTransform, Light, Light2D, LightKind as EcsLightKind, Material,
    MeshRef, NineSlice, PcgVolume, Sprite, Terrain, Text2D, TextAlign, Tilemap,
};
use inf_ecs::Guid;
use inf_math::FloatingOrigin;
use inf_render::{
    detect_tier, expand_nine_slice, expand_text, handle_from_guid, EngineRenderer, GpuContext,
    HAlign, LightKind, MeshInstance, NineSliceParams, PrebatchedRun, RenderChunk, RenderLight,
    RenderLight2D, RenderScene, RenderSettings, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, RenderTilemap, RenderView, SurfaceChain, TextParams, TilemapParams,
    VgeomAsset, VgeomInstance, BUILTIN_FONT_TEXTURE,
};

use crate::runtime_sim::RuntimeSim;
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
    /// Whether the auto-picked [`RenderTier`](inf_render::RenderTier) enables the
    /// GPU meshlet path (High). Off → the classic discrete-LOD fallback renders the
    /// same vgeom content (the renderer's `ClassicVgeomNode`).
    vgeom_enabled: bool,
}

impl PlayerRenderHost {
    /// Build the render host over an already-created surface + GPU context (the
    /// window module owns the winit window and makes the surface from it).
    pub fn new(
        gpu: GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let chain = SurfaceChain::new(&gpu, surface, width, height)?;
        let mut renderer = EngineRenderer::new(&gpu, chain.target_format());

        // Auto-tier (P13.4.2): probe the adapter, pick a render tier, and apply it
        // to the renderer's settings. High enables the GPU meshlet path; Medium/Low
        // fall back to the classic discrete-LOD path (and Low drops the expensive
        // post effects). The decision is logged by `detect_tier`.
        let base = RenderSettings::default();
        let tier = detect_tier(&gpu, &base);
        let settings = tier.apply(RenderSettings {
            // Request the meshlet path; the tier clamps it down on Medium/Low.
            vgeom: inf_render::VgeomSettings {
                enabled: true,
                ..base.vgeom
            },
            ..base
        });
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
        project_scene(&mut self.scene, sim, alpha, &self.vmeshes);
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

/// Fill `scene` from `sim`'s world, blending actor positions by `alpha`.
/// Deterministic `Guid` iteration order. `vmeshes` resolves a `MeshRef.asset` to
/// its cook-derived meshlet DAG (P13.4) — a resolved mesh renders real geometry
/// (meshlet path or classic fallback), an unresolved one falls back to a placeholder
/// cube instance (as before).
fn project_scene(scene: &mut RenderScene, sim: &RuntimeSim, alpha: f64, vmeshes: &VmeshRegistry) {
    scene.instances.clear();
    scene.lights.clear();
    scene.sprites.clear();
    scene.tilemaps.clear();
    scene.prebatched.clear();
    scene.lights_2d.clear();
    scene.vgeom_assets.clear();
    scene.vgeom_instances.clear();
    scene.terrain = None;
    // Track which vmesh assets are already listed this frame (dedup — the render
    // node caches GPU geometry by id, but the asset list must not duplicate).
    let mut vgeom_seen: std::collections::HashSet<u128> = std::collections::HashSet::new();

    let world = sim.world();
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
        // Heightfield terrain (P10.6): the player projects it into the render
        // scene's single terrain slot exactly like the editor viewport host
        // (`inf_viewport::host::project_terrain`), so cooked/PIE terrain renders.
        // First visible, non-empty terrain wins (multi-terrain merge is a
        // follow-up). Terrain is **static in sim v1**, so a constant version keeps
        // the terrain pass from re-uploading height textures every frame.
        if scene.terrain.is_none() {
            if let Some(terrain) = w.get::<Terrain>(entity) {
                if !terrain.data.is_empty() {
                    scene.terrain = Some(project_terrain(terrain, translation, TERRAIN_VERSION));
                }
            }
        }
        // PCG scatter volumes (P10.6): project the volume's evaluated instance
        // cache (populated on load by the level builder) into the existing
        // mesh-instance path as placeholder cubes — the same viewport-parity gap
        // as sprites/tilemaps (kind→real-mesh upload is a follow-up). Draw-distance
        // is not culled here (the player has no persistent camera-eye seam yet; a
        // documented follow-up mirroring the viewport's `last_eye_world` cull).
        if let Some(vol) = w.get::<PcgVolume>(entity) {
            for si in &vol.evaluated {
                scene.instances.push(MeshInstance {
                    translation: si.position,
                    rotation: si.rotation.as_quat(),
                    scale: Vec3::splat(si.scale as f32),
                    color: pcg_kind_color(si.kind),
                    metallic: 0.0,
                    roughness: 0.75,
                    emissive: [0.0; 3],
                    id: next_id,
                });
                next_id += 1;
            }
        }
        if let Some(mesh_ref) = w.get::<MeshRef>(entity) {
            let affine = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or(glam::DAffine3::IDENTITY);
            let (scale, rot, _t) = affine.to_scale_rotation_translation();
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

            // P13.4: a MeshRef.asset with a cook-derived vmesh renders REAL geometry
            // — the GPU meshlet path (vgeom on) or the classic discrete-LOD fallback
            // (vgeom off), both driven by the same vgeom scene content. The tier the
            // renderer settings carry picks which node draws it. An unresolved asset
            // (or a primitive-only MeshRef) falls back to a placeholder cube.
            let vgeom = mesh_ref.asset.and_then(|mesh_id| vmeshes.resolve(mesh_id));
            if let Some((asset_id, mesh)) = vgeom {
                if vgeom_seen.insert(asset_id) {
                    scene.vgeom_assets.push(VgeomAsset { id: asset_id, mesh });
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
                scene.instances.push(MeshInstance {
                    translation,
                    rotation: rot.as_quat(),
                    scale: scale.as_vec3(),
                    color,
                    metallic,
                    roughness,
                    emissive,
                    id: next_id,
                });
            }
            next_id += 1;
        }
    }

    scene.mark_dirty();
}

/// Terrain is static in the player's sim v1, so a fixed version keeps the terrain
/// pass from re-uploading the height/weight textures every frame.
const TERRAIN_VERSION: u64 = 1;

/// Project an ECS [`Terrain`] (+ world translation) into a [`RenderTerrain`],
/// mirroring `inf_viewport::host::project_terrain`: each authored tile becomes a
/// [`RenderTerrainTile`] (heights + resolved RGBA8 splat weights + precomputed
/// height bounds), plus the four material layers + macro variation.
fn project_terrain(terrain: &Terrain, translation: DVec3, version: u64) -> RenderTerrain {
    let data = &terrain.data;
    let res = data.tile_resolution();
    let n = (res * res) as usize;
    let tiles = data
        .tiles()
        .map(|(&coord, tile)| {
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
