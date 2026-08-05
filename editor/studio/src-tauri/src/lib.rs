//! Infinity Engine backend: Tauri v2 app shell.
//!
//! Convention (adopted engine-wide): every backend capability is a
//! `#[tauri::command] async fn … -> Result<T, String>` in a per-domain module
//! under `commands/`, registered once in [`invoke_handler`]. The frontend
//! calls only typed wrappers in `src/lib/ipc.ts`. Events use namespaced
//! channels (`log://line`, `viewport://rect`, `assets://changed/{id}`, …).

mod commands;
mod logging;

use tauri::Manager as _;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

pub fn run() {
    // Console output + the Output Log bridge share one EnvFilter; events
    // fired before the webview exists buffer inside the bridge layer.
    let subscriber = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .with(logging::LogBridgeLayer);
    // Tracy profiler layer (P15.1), behind the off-by-default `tracy` feature.
    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());
    subscriber.init();

    // Editor-side crash reporter (P15.2): a Rust panic hook that writes a
    // structured, timestamped crash file (engine version, OS, panic location +
    // message) into the crash dir, then chains to the default hook. No telemetry
    // upload (opt-in only; see `docs/profiling.md`).
    commands::install_crash_hook();

    tauri::Builder::default()
        // Native file-open dialog for asset import (P4.4). The only plugin;
        // everything else routes through audited commands.
        .plugin(tauri_plugin_dialog::init())
        // **The one carve store and the one fracture map, per process** (P23.2a
        // — the hoist). Created HERE, once, and handed to every
        // `inf_viewport::spawn`; the save path and the Simulate publishes read
        // them straight out of this state, with no viewport in hand.
        .manage(commands::SharedStores::default())
        .manage(commands::ViewportState::default())
        .manage(commands::SceneState::default())
        .manage(commands::SimState::default())
        .manage(commands::PieState::default())
        .manage(commands::AssetState::default())
        .manage(commands::ProjectState::default())
        .manage(commands::PtyState::default())
        .manage(commands::LspState::default())
        .manage(commands::GraphState::default())
        .manage(commands::MaterialState::default())
        .manage(commands::DccState::default())
        .manage(commands::PcgState::default())
        .manage(commands::SmEditorState::default())
        .manage(commands::ErosionState::default())
        .manage(commands::SequencerState::default())
        .invoke_handler(commands::invoke_handler())
        .setup(|app| {
            logging::attach_app(app.handle().clone());
            // Point the crash reporter at the app-data crash dir now that it
            // resolves (before this a very-early panic falls back to a temp dir).
            if let Ok(dir) = app.path().app_data_dir() {
                commands::set_crash_dir(dir.join("crashes"));
            }
            commands::recover_scene_on_boot(app.handle());
            commands::init_assets_on_boot(app.handle());
            tracing::info!("Infinity Engine starting");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Infinity Engine");
}
