//! Infini Engine backend: Tauri v2 app shell.
//!
//! Convention (adopted engine-wide): every backend capability is a
//! `#[tauri::command] async fn … -> Result<T, String>` in a per-domain module
//! under `commands/`, registered once in [`invoke_handler`]. The frontend
//! calls only typed wrappers in `src/lib/ipc.ts`. Events use namespaced
//! channels (`log://line`, `viewport://rect`, `assets://changed/{id}`, …).

mod commands;
mod logging;
// L7.M7: the drift gate for the four hand-written TypeScript wire mirrors that
// sit outside the ts-rs bindings check. **Test-only**, and gated as such — its
// helpers have no non-test caller, so a plain `mod` is three `dead_code`
// warnings in the shipped build (and three clippy errors under `-D warnings`).
#[cfg(test)]
mod wire_mirror;

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
        .manage(commands::SkelState::default())
        .manage(commands::PcgState::default())
        .manage(commands::SmEditorState::default())
        .manage(commands::ErosionState::default())
        .manage(commands::SequencerState::default())
        .manage(commands::PhotogrammetryState::default())
        .manage(commands::CharacterWizardState::default())
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
            tracing::info!("Infini Engine starting");
            Ok(())
        })
        .run(debuggable_context())
        .expect("error while running Infini Engine");
}

/// **The environment variable that lets a script drive this editor** (wave FIX1).
///
/// Set it to a port and the WebView2 that hosts the shell listens for a Chrome
/// DevTools Protocol client there, which is how `tools/demo` presses the Play
/// button by NAME instead of by screen coordinate.
pub const DEBUG_PORT_ENV: &str = "INF_WEBVIEW_DEBUG_PORT";

/// The app context, with remote debugging switched on when [`DEBUG_PORT_ENV`]
/// asks for it.
///
/// # Why it is here and not in an environment variable WebView2 already reads
///
/// It was tried first, and it does not work, which is worth writing down: WebView2
/// reads `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` only when the embedder passes no
/// arguments of its own, and Tauri always passes some
/// (`--disable-features=msWebOOUI,…`). So the variable is silently ignored and a
/// script waits for a debugger port that never opens. The one place the embedder's
/// own string can be reached for a window declared in `tauri.conf.json` is the
/// generated context, before `run` consumes it.
///
/// **Off unless asked for, and asked for by a whole port number.** A shipped
/// editor that listened on a local port would be a shipped editor anything on the
/// machine could drive.
fn debuggable_context() -> tauri::Context {
    let mut context = tauri::generate_context!();
    let Some(port) = std::env::var(DEBUG_PORT_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .filter(|p| *p != 0)
    else {
        return context;
    };
    // Tauri's own default, kept: dropping it would change how the shell renders
    // in the very session a script is watching, which is the opposite of what a
    // demo instrument is for.
    let args = format!(
        "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection          --remote-debugging-port={port}"
    );
    for window in &mut context.config_mut().app.windows {
        window.additional_browser_args = Some(args.clone());
    }
    tracing::warn!("inf-studio: WebView2 remote debugging is ON, port {port}");
    context
}
