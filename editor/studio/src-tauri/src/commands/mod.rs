//! Per-domain command modules. One module per domain; every command is
//! `#[tauri::command] async fn … -> Result<T, String>`. Register everything
//! here, in one place.

mod app;
mod assets;
mod layout;
mod scene;
mod viewport;

pub use assets::{init_assets_on_boot, AssetState};
pub use scene::{recover_scene_on_boot, SceneState};
pub use viewport::ViewportState;

pub fn invoke_handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        app::app_version,
        layout::layout_save,
        layout::layout_load,
        layout::layout_list,
        layout::layout_delete,
        viewport::viewport_attach,
        viewport::viewport_set_rect,
        viewport::viewport_set_visible,
        viewport::viewport_drop,
        scene::scene_snapshot,
        scene::scene_details,
        scene::scene_create,
        scene::scene_delete,
        scene::scene_rename,
        scene::scene_reparent,
        scene::scene_set_visible,
        scene::scene_select,
        scene::scene_set_property,
        scene::scene_reset_property,
        scene::scene_undo,
        scene::scene_redo,
        scene::scene_save,
        scene::scene_open,
        scene::scene_new,
        scene::scene_autosave,
        assets::assets_snapshot,
        assets::asset_references,
        assets::asset_thumbnail,
        assets::asset_import,
        assets::asset_create,
        assets::asset_delete,
        assets::asset_rename,
        assets::asset_duplicate,
        assets::asset_set_tags,
    ]
}
