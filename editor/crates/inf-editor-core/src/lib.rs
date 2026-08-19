//! Editor core (Tauri-free): project model, asset queries, undo/redo,
//! thumbnailer, build orchestration — PIE session management ([`pie`]) and
//! the shared editor↔frontend IPC types ([`ipc`]).

pub mod assets;
// The grammar → mesh bake (P23.6). Docs live in `bake.rs`, for the same
// intra-doc-link reason `dcc` states below.
pub mod bake;
/// Where a blueprint class's generated Rust lives on disk (Wave D).
pub mod blueprint_source;
// P24.5 "New Character from Template": the wizard's Ring-1 half. Docs live in
// `character.rs`, for the intra-doc-link reason `dcc` states below.
pub mod character;
// P25.4 the capture wizard: the Ring-1 session that drives photographs through
// SfM, MVS and the finish into a project, with progress and diagnostics. Docs
// live in `capture.rs`, for the intra-doc-link reason `dcc` states below.
pub mod capture;
pub mod collision_layers;
// The Model Editor's Ring-1 half (P23.4). Its docs live in `dcc.rs`: a `///`
// here is resolved in THIS module's scope, so every intra-doc link into the
// module would dangle (seven of the nine doc warnings the audit counted).
pub mod dcc;
// P24.4 cloth & hair authoring: the Model Editor's two payload doors.
pub mod groom;
// P24.3: the Skeleton Editor's Ring-1 half — an editing session over a
// `SkeletonAsset`, snapshot-undone (see the module docs for why not a journal).
pub mod diagnostics;
// Wave E batch A: the app-level (per-user) editor preferences file behind
// Edit ▸ Editor Preferences… — the sibling of `project_settings`, same
// absent-default / corrupt-refused / atomic-write doctrine, directory injected.
pub mod editor_settings;
pub mod erosion_gpu;
// Wave G: GIS vector layers become scene entities through the doors that
// already exist — a stream becomes the same WaterBody+Spline the hydrology tool
// creates, so the river validator and the cook advisory apply to imported water
// on the day it lands.
pub mod gis;
pub mod hydro;
pub mod ipc;
pub mod layouts;
pub mod mods;
/// The P25.3 finish pipeline: a dense photogrammetric reconstruction in, a
/// standard textured `.inf_mesh` + `.inf_tex` + `.inf_mat` out. Ring 1 because
/// it needs the modelling kernel's unwrapper and `AssetProject`'s writer.
pub mod photogrammetry;
pub mod pie;
pub mod project_settings;
/// The interactive viewport's loose-file render-asset store (P18.3): real
/// `MeshRef.asset` geometry + skinned `SkeletalMesh` draws.
pub mod render_assets;
pub mod samples;
pub mod scene;
pub mod sequencer;
pub mod simulate;
pub mod skel;
// P29.6 pillar S1: the `.inf_sm` text sidecar, as a save door.
pub mod sm_text;
pub mod sorting;
/// Editing an asset-backed terrain + the save write-back (P16.4b).
pub mod terrain_edit;
/// Editor-side camera-driven terrain streaming (P16.3b2).
pub mod terrain_stream;
pub mod thumbnail;
// P29.5 pillar S4: the queue a live tuning edit lands on, drained at the top of
// the next fixed step. Ring 1 ONLY -- the shipped player has no such door.
pub mod tuning;
// Wave E: the drag-to-viewport payload contract, parsed here rather than in
// `inf_viewport::host` so the Linux CI leg exercises it (the `render_assets`
// reason).
pub mod viewport_drop;
/// Carving a voxel volume + the save write-back (P21.2), `terrain_edit`'s twin.
pub mod voxel_edit;
pub mod voxel_store;
/// The dig tools' pure shape policy (P21.3) — the M11 Ring-1 move.
pub mod voxel_tool;
