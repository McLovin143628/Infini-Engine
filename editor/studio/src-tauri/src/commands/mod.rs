//! Per-domain command modules. One module per domain; every command is
//! `#[tauri::command] async fn … -> Result<T, String>`. Register everything
//! here, in one place.

mod app;
mod assets;
mod audio;
mod collections;
mod collision_layers;
mod diagnostics;
mod files;
mod git;
mod graph;
mod layout;
mod lsp;
mod material;
mod package;
mod pcg;
mod pie;
mod project;
mod scene;
mod search;
mod sequencer;
mod settings;
mod shell;
mod sim;
mod sm;
mod sorting;
mod terminal;
mod terrain;
mod viewport;

pub use assets::{init_assets_on_boot, AssetState};
pub use diagnostics::{install_crash_hook, set_crash_dir};
pub use graph::GraphState;
pub use lsp::LspState;
pub use material::MaterialState;
pub use pcg::PcgState;
pub use pie::PieState;
pub use project::ProjectState;
pub use scene::{recover_scene_on_boot, SceneState};
pub use sequencer::SequencerState;
pub use sim::SimState;
pub use sm::SmEditorState;
pub use terminal::PtyState;
pub use terrain::ErosionState;
pub use viewport::ViewportState;

pub fn invoke_handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        app::app_version,
        app::app_build_info,
        graph::graph_registry,
        graph::graph_list,
        graph::graph_create,
        graph::graph_get,
        graph::graph_close,
        graph::graph_apply,
        graph::graph_undo,
        graph::graph_redo,
        graph::graph_run,
        graph::graph_debug_run,
        graph::graph_generate,
        material::material_registry,
        material::material_list,
        material::material_create,
        material::material_get,
        material::material_close,
        material::material_apply,
        material::material_undo,
        material::material_redo,
        material::material_compile,
        material::material_bake,
        terrain::terrain_erode,
        terrain::terrain_export_data_map,
        terrain::terrain_probe_heightmap,
        terrain::terrain_import_plan,
        terrain::terrain_import,
        terrain::terrain_import_cancel,
        terrain::terrain_asset_info,
        terrain::terrain_spawn_streamed,
        pcg::pcg_registry,
        pcg::pcg_list,
        pcg::pcg_create,
        pcg::pcg_get,
        pcg::pcg_close,
        pcg::pcg_apply,
        pcg::pcg_undo,
        pcg::pcg_redo,
        pcg::pcg_compile,
        pcg::pcg_save,
        pcg::pcg_evaluate,
        sm::sm_list,
        sm::sm_create,
        sm::sm_get,
        sm::sm_close,
        sm::sm_save,
        sm::sm_list_clips,
        layout::layout_save,
        layout::layout_load,
        layout::layout_list,
        layout::layout_delete,
        viewport::viewport_attach,
        viewport::viewport_set_rect,
        viewport::viewport_set_visible,
        viewport::viewport_drop,
        viewport::viewport_set_mode,
        viewport::viewport_set_snap2d,
        viewport::viewport_set_tool_mode,
        viewport::viewport_set_sculpt,
        viewport::viewport_set_foliage,
        viewport::viewport_set_gizmo_mode,
        viewport::viewport_set_gizmo_space,
        viewport::viewport_set_snap3d,
        viewport::viewport_set_view_mode,
        scene::scene_snapshot,
        scene::scene_details,
        scene::scene_create,
        scene::scene_spawn_asset,
        scene::scene_apply_material,
        scene::scene_apply_actor,
        scene::scene_apply_sprite_slice,
        scene::tilemap_get,
        scene::tilemap_paint,
        scene::scene_delete,
        scene::scene_duplicate,
        scene::scene_copy,
        scene::scene_cut,
        scene::scene_paste,
        scene::scene_rename,
        scene::scene_reparent,
        scene::scene_set_visible,
        scene::scene_select,
        scene::scene_set_property,
        scene::scene_reset_property,
        scene::scene_list_addable_components,
        scene::scene_add_component,
        scene::scene_remove_component,
        scene::scene_list_default,
        scene::scene_undo,
        scene::scene_redo,
        scene::scene_save,
        scene::scene_open,
        scene::scene_new,
        scene::scene_autosave,
        scene::scene_current_path,
        scene::scene_get_settings,
        scene::scene_set_settings,
        shell::shell_reveal,
        sim::sim_start,
        sim::sim_tick,
        sim::sim_step_fixed,
        sim::sim_set_debug,
        sim::sim_stop,
        sim::sim_is_running,
        sequencer::seq_list,
        sequencer::seq_get,
        sequencer::seq_save,
        sequencer::seq_delete,
        sequencer::seq_key_set,
        sequencer::seq_capture_key,
        sequencer::seq_scrub,
        sequencer::seq_scrub_stop,
        pie::pie_start,
        pie::pie_pause,
        pie::pie_resume,
        pie::pie_step,
        pie::pie_eject,
        pie::pie_stop,
        pie::pie_is_running,
        assets::assets_snapshot,
        assets::asset_references,
        assets::asset_thumbnail,
        assets::asset_import,
        assets::asset_create,
        assets::asset_create_material_instance,
        assets::asset_get_material_instance,
        assets::asset_save_material_instance,
        assets::asset_delete,
        assets::asset_rename,
        assets::asset_duplicate,
        assets::asset_set_tags,
        assets::asset_data,
        assets::asset_data_save,
        assets::asset_table_import,
        assets::asset_rust_source,
        assets::texture_get_slices,
        assets::texture_set_slices,
        audio::mixer_get,
        audio::mixer_save,
        collections::collections_list,
        collections::collections_create,
        collections::collections_rename,
        collections::collections_delete,
        collections::collections_add,
        collections::collections_remove,
        sorting::layers_get,
        sorting::layers_set,
        collision_layers::collision_layers_get,
        collision_layers::collision_layers_set,
        settings::project_settings_get,
        settings::project_settings_set,
        project::project_templates,
        project::project_recent,
        project::project_current,
        project::project_new,
        project::project_open,
        project::project_close,
        package::project_package,
        terminal::pty_create,
        terminal::pty_write,
        terminal::pty_resize,
        terminal::pty_close,
        files::file_read,
        files::file_write,
        files::list_project_files,
        git::git_status,
        git::git_stage,
        git::git_unstage,
        git::git_discard,
        git::git_commit,
        git::git_file_diff,
        git::git_branches,
        git::git_init,
        search::search_workspace,
        lsp::lsp_start,
        lsp::lsp_stop,
        lsp::lsp_did_open,
        lsp::lsp_did_change,
        lsp::lsp_did_close,
        lsp::lsp_request,
        diagnostics::editor_diagnostics,
    ]
}
