//! Editor core (Tauri-free): project model, asset queries, undo/redo,
//! thumbnailer, build orchestration — PIE session management ([`pie`]) and
//! the shared editor↔frontend IPC types ([`ipc`]).

pub mod assets;
pub mod ipc;
pub mod layouts;
pub mod pie;
pub mod scene;
pub mod sorting;
pub mod thumbnail;
