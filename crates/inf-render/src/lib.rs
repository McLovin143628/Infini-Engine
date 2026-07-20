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
pub mod gpu;
pub mod graph;
pub mod headless;
pub mod passes;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod surface;

pub use camera::{RenderView, DEPTH_CLEAR, DEPTH_COMPARE, DEPTH_FORMAT};
pub use debug_draw::{DebugDraw, DebugVertex};
pub use gpu::{create_instance, GpuContext};
pub use headless::{HeadlessTarget, HEADLESS_FORMAT};
pub use passes::composite::BlitMode;
pub use renderer::{EngineRenderer, MASK_FORMAT, SCENE_FORMAT, SCENE_SAMPLES};
pub use scene::{MeshInstance, RenderScene, SkyParams, ID_GIZMO_BASE, ID_NONE};
pub use surface::{SurfaceChain, RECONFIGURE_DEBOUNCE};
