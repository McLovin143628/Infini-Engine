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
pub mod debris;
pub mod debug_draw;
pub mod deform;
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
/// The non-blocking GPU→CPU readback ring at a pinned frame latency (P26.4).
/// Not a virtual-texturing detail: P27's shadow-page marking and P28's unified
/// streamer read through this same primitive.
pub mod readback;
pub mod renderer;
pub mod scene;
pub mod settings;
pub mod surface;
/// Virtual-shadow-map geometry (P27.1): the per-light projections, the
/// clipmap's centring rule, the depth-convention ruling and the one level rule
/// the marking pass mirrors.
pub mod vsm;
/// The P27.1 GPU mirror: one `Depth32Float` page atlas and one indirection
/// buffer, executing an `inf_vsm::VsmTransaction`.
pub mod vsm_atlas;
/// The P27.1 marking loop: the screen-driven page-marking compute pass, the
/// readback ring that reads it at a pinned latency, and the residency step
/// that turns its bits into allocations.
pub mod vsm_mark;
/// The P27.2 caster pass: the per-page GPU cull, the one render pass that owns
/// the whole atlas, and the viewport/scissor pair that pins each page to its
/// slot.
pub mod vsm_raster;
pub mod vt;
/// The P26.3 registration door: `.inf_tex` v2 payloads become virtual
/// textures here, for both hosts, through one rule.
pub mod vt_library;
/// The P26.4 streaming loop: the analytic want floor, the GPU coverage feedback
/// that refines it, and the pop-in instruments.
pub mod vt_stream;
pub mod water;
pub mod wetness;

pub use atmosphere::{
    camera_radius_km, extinction, height_fog_optical_depth, height_fog_transmittance,
    transmittance_to_top, AtmosphereParams, AtmosphereQuality, HeightFog,
};
pub use camera::{
    ortho_reverse_z, OrthoParams, RenderView, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT,
};
pub use caps::{
    choose_tier, detect_and_clamp, detect_tier, hair_detail_for, AdapterCaps, HairDetailSpec,
    RenderTier,
};
pub use clouds::{
    detail_texel, shape_texel, wind_offset, CloudParams, CloudQuality, CloudVolumes,
    CPU_GPU_EXACT_FRACTION, CPU_GPU_SHADOW_TOLERANCE, CPU_GPU_TEXEL_TOLERANCE,
};
pub use debug_draw::{
    collider_outline_2d, collider_outline_3d, ColliderOutline2D, ColliderOutline3D, DebugDraw,
    DebugVertex,
};
// The P18.4 GI v2 surface: the cost tier, the amortization schedule, the
// voxelization audit, and the pure SH/terrain math the shaders mirror.
pub use gi::{
    bin_macro_cells, env_brdf_ab, intersects_volume, priority_order, sample_terrain_column,
    sh_dominant_direction, sh_radiance, sun_bucket, voxelization_tiles, GiAudit, GiBounds,
    GiQuality, ProbeSchedule, TerrainColumn, EMISSIVE_MAX, GI_DIM, MACRO_DIM, PROBE_DIMS,
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
// The P21.1 voxel-surface cache gate — the pure planner the volumetric-terrain
// pass drives its per-chunk uploads/evictions from, exported like `plan_tile_cache`
// so hosts and gates can reason about residency without a GPU.
pub use passes::voxel::{
    plan_chunk_cache, CachedChunk, ChunkCacheKey, ChunkCachePlan, VoxelReport,
};
pub use pick::Picker;
pub use precip::{
    particle_offset, precip_base, wrap_signed, PrecipParams, PrecipQuality, PRECIP_BOX_XZ_M,
    PRECIP_BOX_Y_M, RAIN_FALL_SPEED, SNOW_FALL_SPEED,
};
pub use primitives::{PrimGpu, PrimMesh, PrimRange};
pub use readback::{ReadbackRing, READBACK_LATENCY_FRAMES};
pub use renderer::{
    EngineRenderer, ViewMode, AO_FORMAT, HDR_FORMAT, LDR_FORMAT, MASK_FORMAT, SCENE_FORMAT,
    SCENE_SAMPLES,
};
pub use scene::{
    apply_seam, box_uv, deformed_skinned_mesh, RenderFractureChunk, RenderFractureVertex,
    RenderTilemap, RenderVoxelChunk, RenderVoxelVertex, RenderVoxelVolume, ScatterBatch,
    ScatterData, ScatterInstance, ScatterInstanceRaw, SkinnedInstance, SkinnedMeshData,
    SkinnedVertex, SkyParams, SpriteInstance, SpriteTextureUpload, SunParams, TerrainTileKey,
    TextureHandle, TilemapParams, VgeomAsset, VgeomInstance, VgeomMesh, VoxelChunkKey, CLOTH_TINT,
    DEFAULT_SUN_DIR, HAIR_TINT, ID_GIZMO_BASE, ID_NONE,
};
pub use scene::{
    terrain_id_from_guid, Ambient2D, LightKind, MeshInstance, PrebatchedRun, RenderChunk,
    RenderLight, RenderLight2D, RenderScene, RenderTerrain, RenderTerrainLayer, RenderTerrainTile,
    SeamSample, VtTextureSet, DEFAULT_SEAM_BAND_M,
};
pub use settings::{
    halton, halton_jitter, mip_chain_sizes, soft_knee_factor, ssao_hemisphere_kernel,
    BloomSettings, GiSettings, RenderSettings, ScatterSettings, ShadowSettings, SsaoSettings,
    VgeomSettings, VirtualTextureSettings, VsmSettings, VsmSettingsError, VSM_BUDGET_LOW_BYTES,
    VSM_BUDGET_MEDIUM_BYTES, VSM_CLIPMAP_PAGES_MEDIUM,
};
// P27.1 virtual shadow maps: the projections and the level rules (pure,
// unit-tested with no adapter), the mirror, and the marking loop's counters.
pub use vsm::{
    clipmap_centre, clipmap_containing_level, clipmap_matrix, clipmap_page_world, cube_face_matrix,
    mark_page_for, page_clip_planes, spot_fov_y, spot_matrix, vsm_justified_level, vsm_light_trees,
    vsm_page_matrix, vsm_page_sees_sphere, vsm_projections, VsmMarkParams, VsmProjection,
    CUBE_FACE_BASES, VSM_DEPTH_CLEAR, VSM_DEPTH_COMPARE, VSM_MAX_PROJECTIONS, VSM_PROJ_ORTHO,
    VSM_PROJ_PERSPECTIVE,
};
pub use vsm_atlas::{VsmApplyReport, VsmPools, VSM_PAGE_FORMAT};
pub use vsm_mark::{VsmMarker, VsmStreamStats, VsmSystem, VSM_PROJECTION_CAP};
pub use vsm_raster::{
    VsmCasterRaw, VsmRaster, VsmRasterStats, SKINNED_POSE_MARGIN, VSM_ARG_WORDS, VSM_MAX_CASTERS,
    VSM_MAX_RASTER_PAGES, VSM_PAGE_DRAW_STRIDE, VSM_RIGID_GROUPS, VSM_TERRAIN_CASTER_CELLS,
};
// The shadow page space's own vocabulary, re-exported on `inf_vt`'s precedent so
// a host or a gate that reads a residency does not have to add the GPU-free
// crate to its own manifest to spell a page address.
pub use inf_vsm::{
    VsmAtlasConfig, VsmLightDesc, VsmLightHandle, VsmMarkLayout, VsmPage, VsmResidency,
    VsmTreeKind, VsmWant, DEFAULT_VSM_BUDGET_BYTES, VSM_PAGE_SIZE,
};
// The P26.3 registration door + P26.4's one registration ORDER: both projectors
// build a level's virtual textures through exactly these, so "PIE == shipping"
// for texture residency is a property of the code rather than of two hosts
// agreeing by inspection.
pub use vt_library::{
    build_vt_level, registration_order, vt_set_for, VtLevelReport, VtMaterialMaps, VtRefusal,
    VtTextures, VtTileSource, VT_FLOOR_LEVELS,
};
// The P26.4 streaming loop: the floor's rules (pure, unit-tested with no
// adapter), the GPU feedback pass, and the pop-in counters a gate asserts on.
pub use vt_stream::{
    analytic_floor, feedback_requests, justified_mip, ndc_margin, on_screen, projection_scale,
    scene_coverage, screen_diameter_px, VtCoverage, VtFeedback, VtPopIn, VT_FEEDBACK_MAX_TILES,
    VT_FLOOR_MAX_TILES,
};
// The pool's own vocabulary, re-exported so a host that builds a VT level does
// not have to depend on `inf-vt` itself: `VtTextures::new` takes a
// `VtPoolConfig`, so a caller that cannot name one cannot call it, and adding
// the crate to two hosts' manifests to spell four constants would put the
// GPU-free half in dependency graphs that have no other use for it.
pub use inf_vt::{
    PageFormat, VtPoolConfig, DEFAULT_MAX_TEXTURE_DIM, DEFAULT_VT_BUDGET_BYTES, STORED_TILE_SIZE,
};
pub use water::{
    camera_underwater, RenderWater, RiverFrame, RiverPath, RiverProfile, Underwater, WaterFrame,
    WaterKindGpu, WaterQuality, WaterSettings, WaterSurface, Wave, WaveField, WaveSpec, MAX_WAVES,
    OCEAN_EXTENT_M, OCEAN_SNAP_M, SHAFT_DECAY, SHAFT_GLOW_POWER, SHAFT_INTENSITY, SHAFT_REACH,
    SHAFT_TINT_DEPTH_M, UNDERWATER_FAR_M, UNDERWATER_RAMP_M,
};
// P22.4 small-debris instancing + the per-tier debris budget: the deterministic
// sub-chunk rubble both hosts lay through the P18.5 scatter path, and the one
// place `RenderTier` is mapped onto a budget (physics stays tier-blind).
pub use debris::{
    debris_batch, debris_budget_for, debris_budget_for_session, debris_instances, DebrisBudgetSpec,
    DebrisCache, DebrisSite, DEBRIS_BUDGET_HIGH, DEBRIS_MAX_SCALE, DEBRIS_MIN_SCALE,
    DEBRIS_RUBBLE_PER_CHUNK,
};
// P22.1 surface deformation: the projected field, the camera-following window's
// packing, and the engine constants argued in `deform.rs` rather than authored.
pub use deform::{
    deform_depth_reference, pack_deform_window, window_origin_texels, DeformResources,
    DeformUniform, RenderDeform, RenderDeformCell, DEFORM_BEND_GAIN, DEFORM_MAX_DEPTH_M,
    DEFORM_TEXEL_M, DEFORM_WINDOW_M, DEFORM_WINDOW_TEXELS, WIND_SWAY, WIND_WAVELENGTH_M,
};
// P20.3 shoreline wetness: the packing the renderer feeds the lit passes, and the
// engine constants whose values are argued in `wetness.rs` rather than authored.
pub use wetness::{
    pack_wetness, WetnessResources, WetnessUniform, MAX_WET_BODIES, MAX_WET_SEGMENTS,
    WET_ALBEDO_SCALE, WET_BAND_M, WET_ROUGHNESS_SCALE, WET_SHORE_MARGIN_M,
};
// The P18.5 scatter instruments: the GPU instance-cull counters (off by default,
// free when off) and the pure band rule both the compute pass and the CPU
// fallback derive their distances from.
pub use passes::scatter::{
    effective_bands, shadow_caster_settings, ScatterAudit, MAX_CPU_SCATTER_INSTANCES,
    SHADOW_CASTER_MARGIN,
};
// The GPU meshlet cull readback (P13.1b) — the CPU-vs-GPU parity gate + the
// player's vgeom-activation check drive it. `VgeomAudit` + `is_camera_cut` are
// the P18.1 two-pass occlusion instruments.
// `cull_visible_streamed` is the same call with the residency it culled under, so
// the parity gate can drive a PUNCHED-OUT resident set (P18.2) rather than only
// the fully-paged case.
pub use passes::vgeom::{
    cull_visible, cull_visible_source, cull_visible_streamed, is_camera_cut, CullReadback,
    VgeomAudit, VgeomStreamReport,
};
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
