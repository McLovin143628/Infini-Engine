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

pub mod camera;
pub mod debug_draw;
pub mod gizmo;
pub mod golden;
pub mod gpu;
pub mod graph;
pub mod headless;
pub mod passes;
pub mod pick;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod surface;

pub use camera::{RenderView, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT};
pub use debug_draw::{DebugDraw, DebugVertex};
pub use gizmo::{GizmoAxis, GizmoDelta, GizmoDrag, GizmoMode};
pub use gpu::{create_instance, GpuContext};
pub use headless::{HeadlessTarget, HEADLESS_FORMAT};
pub use passes::composite::BlitMode;
pub use pick::Picker;
pub use renderer::{EngineRenderer, MASK_FORMAT, SCENE_FORMAT, SCENE_SAMPLES};
pub use scene::{
    Ambient2D, LightKind, MeshInstance, PrebatchedRun, RenderChunk, RenderLight, RenderLight2D,
    RenderScene, RenderTilemap, SkyParams, SpriteInstance, SpriteTextureUpload, TextureHandle,
    TilemapParams, ID_GIZMO_BASE, ID_NONE,
};
// 2D batcher API surfaced through the renderer for hosts.
pub use inf_render_2d::{
    aabb_visible, atlas_uv, batch_scene, batch_sprites, builtin_font_rgba8, chunk_world_aabb,
    expand_chunk, expand_nine_slice, expand_text, handle_from_guid, BatchedSprites, HAlign,
    NineSliceParams, SpriteBatch, TextParams, BUILTIN_FONT_COLS, BUILTIN_FONT_FIRST_CP,
    BUILTIN_FONT_ROWS, BUILTIN_FONT_TEXTURE, TILE_CHUNK_DIM, WHITE_TEXTURE,
};
pub use surface::{SurfaceChain, RECONFIGURE_DEBOUNCE};
