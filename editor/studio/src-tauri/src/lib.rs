//! Infinity Engine backend: Tauri v2 app shell.
//!
//! Convention (adopted engine-wide): every backend capability is a
//! `#[tauri::command] async fn … -> Result<T, String>` in a per-domain module
//! under `commands/`, registered once in [`invoke_handler`]. The frontend
//! calls only typed wrappers in `src/lib/ipc.ts`. Events use namespaced
//! channels (`log://line`, `viewport://rect`, `assets://changed/{id}`, …).

mod commands;

pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    tauri::Builder::default()
        .manage(commands::ViewportState::default())
        .invoke_handler(commands::invoke_handler())
        .setup(|_app| {
            tracing::info!("Infinity Engine starting");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Infinity Engine");
}
