//! Editor core (Tauri-free): project model, asset queries, undo/redo,
//! thumbnailer, build orchestration — PIE session management ([`pie`]) and
//! the shared editor↔frontend IPC types ([`ipc`]).

pub mod assets;
pub mod collision_layers;
pub mod diagnostics;
pub mod erosion_gpu;
pub mod ipc;
pub mod layouts;
pub mod mods;
pub mod pie;
pub mod project_settings;
pub mod samples;
pub mod scene;
pub mod sequencer;
pub mod simulate;
pub mod sorting;
/// Editor-side camera-driven terrain streaming (P16.3b2).
pub mod terrain_stream;
pub mod thumbnail;
