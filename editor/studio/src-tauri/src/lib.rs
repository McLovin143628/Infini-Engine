//! Infinity Engine backend: Tauri v2 app shell.
//!
//! Convention (adopted engine-wide): every backend capability is a
//! `#[tauri::command] async fn … -> Result<T, String>` in a per-domain module
//! under `commands/`, registered once in [`invoke_handler`]. The frontend
//! calls only typed wrappers in `src/lib/ipc.ts`. Events use namespaced
//! channels (`log://line`, `viewport://rect`, `assets://changed/{id}`, …).

mod commands;
mod logging;

use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

pub fn run() {
    // Console output + the Output Log bridge share one EnvFilter; events
    // fired before the webview exists buffer inside the bridge layer.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .with(logging::LogBridgeLayer)
        .init();

    tauri::Builder::default()
        // Native file-open dialog for asset import (P4.4). The only plugin;
        // everything else routes through audited commands.
        .plugin(tauri_plugin_dialog::init())
        .manage(commands::ViewportState::default())
        .manage(commands::SceneState::default())
        .manage(commands::SimState::default())
        .manage(commands::AssetState::default())
        .manage(commands::ProjectState::default())
        .manage(commands::PtyState::default())
        .manage(commands::LspState::default())
        .manage(commands::GraphState::default())
        .manage(commands::MaterialState::default())
        .invoke_handler(commands::invoke_handler())
        .setup(|app| {
            logging::attach_app(app.handle().clone());
            commands::recover_scene_on_boot(app.handle());
            commands::init_assets_on_boot(app.handle());
            tracing::info!("Infinity Engine starting");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Infinity Engine");
}
