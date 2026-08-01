//! Renderer: wgpu device/surface, render graph, WGSL pipeline cache,
//! GPU-driven draws, ID picking.
//!
//! Ring 0 — no editor or Tauri concepts. Hosts (the editor's `inf-viewport`,
//! headless tests, the future thumbnailer) provide a [`GpuContext`], describe
//! a [`RenderScene`] + [`RenderView`] each frame, and get pixels.
//!
//! Coordinate contract (architecture rule 3): scene/instance positions are
//! f64 world space; the [`RenderView`]'s floating origin converts to f32
//! render-local at upload. Depth is reverse-infinite Z.

pub mod atmosphere;
pub mod camera;
pub mod caps;
pub mod clouds;
pub mod csm;
pub mod debug_draw;
pub mod gi;
pub mod gizmo;
pub mod golden;
pub mod gpu;
pub mod graph;
pub mod headless;
pub mod passes;
pub mod pick;
pub mod pipeline;
pub mod precip;
pub mod primitives;
pub mod renderer;
pub mod scene;
pub mod settings;
pub mod surface;

pub use atmosphere::{
    camera_radius_km, extinction, height_fog_optical_depth, height_fog_transmittance,
    transmittance_to_top, AtmosphereParams, AtmosphereQuality, HeightFog,
};
pub use camera::{
    ortho_reverse_z, OrthoParams, RenderView, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use caps::{choose_tier, detect_tier, AdapterCaps, RenderTier};
pub use clouds::{
    detail_texel, shape_texel, wind_offset, CloudParams, CloudQuality, CloudVolumes,
    CPU_GPU_EXACT_FRACTION, CPU_GPU_SHADOW_TOLERANCE, CPU_GPU_TEXEL_TOLERANCE,
};
pub use debug_draw::{
    collider_outline_2d, collider_outline_3d, ColliderOutline2D, ColliderOutline3D, DebugDraw,
    DebugVertex,
};
pub use gizmo::{GizmoAxis, GizmoDelta, GizmoDrag, GizmoMode};
pub use gpu::{create_instance, GpuContext};
pub use headless::{HeadlessTarget, HEADLESS_FORMAT};
pub use passes::composite::BlitMode;
pub use passes::terrain::{
    assemble_patches, cells_at_lod, lod_for_distance, lod_thresholds, morph_factor, patch_mesh_lod,
    plan_tile_cache, ring_source_lod, superseded, CachedTile, TerrainPatch, TileCacheKey,
    TileCachePlan, TERRAIN_BASE_CELLS, TERRAIN_LOD_COUNT,
};
pub use pick::Picker;
pub use precip::{
    particle_offset, precip_base, wrap_signed, PrecipParams, PrecipQuality, PRECIP_BOX_XZ_M,
    PRECIP_BOX_Y_M, RAIN_FALL_SPEED, SNOW_FALL_SPEED,
};
pub use primitives::{PrimGpu, PrimMesh, PrimRange};
pub use renderer::{
    EngineRenderer, ViewMode, AO_FORMAT, HDR_FORMAT, LDR_FORMAT, MASK_FORMAT, SCENE_FORMAT,
    SCENE_SAMPLES,
};
pub use scene::{
    terrain_id_from_guid, Ambient2D, LightKind, MeshInstance, PrebatchedRun, RenderChunk,
    RenderLight, RenderLight2D, RenderScene, RenderTerrain, RenderTerrainLayer, RenderTerrainTile,
    RenderTilemap, SkinnedInstance, SkinnedMeshData, SkinnedVertex, SkyParams, SpriteInstance,
    SpriteTextureUpload, SunParams, TerrainTileKey, TextureHandle, TilemapParams, VgeomAsset,
    VgeomInstance, VgeomMesh, DEFAULT_SUN_DIR, ID_GIZMO_BASE, ID_NONE,
};
pub use settings::{
    halton, halton_jitter, mip_chain_sizes, soft_knee_factor, ssao_hemisphere_kernel,
    BloomSettings, GiSettings, RenderSettings, ShadowSettings, SsaoSettings, VgeomSettings,
};
// The GPU meshlet cull readback (P13.1b) — the CPU-vs-GPU parity gate + the
// player's vgeom-activation check drive it.
pub use passes::vgeom::cull_visible;
// The classic-LOD fallback selection (P13.4) — the CI-provable probe of what the
// classic path draws when vgeom is off (the meshlet path's complement).
pub use passes::classic_vgeom::{classic_lod_selection, ClassicSelection};
// 2D batcher API surfaced through the renderer for hosts.
pub use inf_render_2d::{
    aabb_visible, atlas_uv, batch_scene, batch_sprites, billboard_basis, builtin_font_rgba8,
    chunk_world_aabb, corner_offset_billboard, expand_chunk, expand_nine_slice, expand_text,
    handle_from_guid, BatchedSprites, HAlign, NineSliceParams, SpriteBatch, TextParams,
    BILLBOARD_CYLINDRICAL, BILLBOARD_NONE, BILLBOARD_SPHERICAL, BUILTIN_FONT_COLS,
    BUILTIN_FONT_FIRST_CP, BUILTIN_FONT_ROWS, BUILTIN_FONT_TEXTURE, TILE_CHUNK_DIM, WHITE_TEXTURE,
};
pub use surface::{SurfaceChain, RECONFIGURE_DEBOUNCE};
