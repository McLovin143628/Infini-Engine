//! Editor core (Tauri-free): project model, asset queries, undo/redo,
//! thumbnailer, build orchestration — PIE session management ([`pie`]) and
//! the shared editor↔frontend IPC types ([`ipc`]).

pub mod assets;
pub mod erosion_gpu;
pub mod ipc;
pub mod layouts;
pub mod pie;
pub mod project_settings;
pub mod samples;
pub mod scene;
pub mod sequencer;
pub mod simulate;
pub mod sorting;
pub mod thumbnail;
