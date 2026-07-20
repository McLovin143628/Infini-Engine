//! Per-domain command modules. One module per domain; every command is
//! `#[tauri::command] async fn … -> Result<T, String>`. Register everything
//! here, in one place.

mod app;
mod viewport;

pub use viewport::ViewportState;

pub fn invoke_handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        app::app_version,
        viewport::viewport_attach,
        viewport::viewport_set_rect,
    ]
}
