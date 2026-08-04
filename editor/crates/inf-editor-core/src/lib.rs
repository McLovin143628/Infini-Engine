//! Editor core (Tauri-free): project model, asset queries, undo/redo,
//! thumbnailer, build orchestration — PIE session management ([`pie`]) and
//! the shared editor↔frontend IPC types ([`ipc`]).

pub mod assets;
pub mod collision_layers;
pub mod diagnostics;
pub mod erosion_gpu;
pub mod hydro;
pub mod ipc;
pub mod layouts;
pub mod mods;
pub mod pie;
pub mod project_settings;
/// The interactive viewport's loose-file render-asset store (P18.3): real
/// `MeshRef.asset` geometry + skinned `SkeletalMesh` draws.
pub mod render_assets;
pub mod samples;
pub mod scene;
pub mod sequencer;
pub mod simulate;
pub mod sorting;
/// Editing an asset-backed terrain + the save write-back (P16.4b).
pub mod terrain_edit;
/// Editor-side camera-driven terrain streaming (P16.3b2).
pub mod terrain_stream;
pub mod thumbnail;
/// Carving a voxel volume + the save write-back (P21.2), `terrain_edit`'s twin.
pub mod voxel_edit;
pub mod voxel_store;
/// The dig tools' pure shape policy (P21.3) — the M11 Ring-1 move.
pub mod voxel_tool;
