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

use glam::{DVec3, Vec2, Vec3};
use uuid::Uuid;

use inf_ecs::components::{
    ComputedVisibility, GlobalTransform, Light, Light2D, LightKind as EcsLightKind, Material,
    MeshRef, NineSlice, Sprite, Text2D, TextAlign, Tilemap,
};
use inf_ecs::Guid;
use inf_math::FloatingOrigin;
use inf_render::{
    expand_nine_slice, expand_text, handle_from_guid, EngineRenderer, GpuContext, HAlign,
    LightKind, MeshInstance, NineSliceParams, PrebatchedRun, RenderChunk, RenderLight,
    RenderLight2D, RenderScene, RenderTilemap, RenderView, SurfaceChain, TextParams, TilemapParams,
    BUILTIN_FONT_TEXTURE,
};

use crate::runtime_sim::RuntimeSim;

/// Owns the GPU stack + the render scene the player draws each frame.
pub struct PlayerRenderHost {
    gpu: GpuContext,
    chain: SurfaceChain,
    renderer: EngineRenderer,
    scene: RenderScene,
    origin: FloatingOrigin,
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
        let renderer = EngineRenderer::new(&gpu, chain.target_format());
        Ok(Self {
            gpu,
            chain,
            renderer,
            scene: RenderScene {
                grid_enabled: false,
                ..Default::default()
            },
            origin: FloatingOrigin::default(),
        })
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
        project_scene(&mut self.scene, sim, alpha);
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
/// Deterministic `Guid` iteration order.
fn project_scene(scene: &mut RenderScene, sim: &RuntimeSim, alpha: f64) {
    scene.instances.clear();
    scene.lights.clear();
    scene.sprites.clear();
    scene.tilemaps.clear();
    scene.prebatched.clear();
    scene.lights_2d.clear();

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
        if w.get::<MeshRef>(entity).is_some() {
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
            next_id += 1;
        }
    }

    scene.mark_dirty();
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
