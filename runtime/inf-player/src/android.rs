//! Android NativeActivity entry point (P14.1).
//!
//! `android_main` is the symbol the `android-activity` NativeActivity glue (pulled
//! by winit's `android-native-activity` feature) calls once the activity is
//! created. It boots a world and runs the shared windowed [`PlayerApp`](crate::window)
//! on an event loop built from the `AndroidApp` handle — the same fixed-step
//! loop, renderer, and (touch) input path as the desktop/web player, with the
//! mobile render tier ([`RenderTier::mobile_default`](inf_render::RenderTier::mobile_default))
//! and on-screen touch controls cfg'd on for `target_os = "android"`.
//!
//! # Honest status
//!
//! This module builds **only with the Android NDK** (via `cargo-ndk`; a plain
//! `cargo check` needs `aarch64-linux-android-clang` for the C deps `zstd-sys`/
//! `meshopt`), and runs **only on a device/emulator** — neither exists in this
//! repo's CI, so it is structured for compilation and device-verified, not
//! CI-gated. See `docs/android-player.md` for the `cargo-ndk` build + APK steps.
//!
//! v1 runs the bundled `--demo` world; loading a cooked pack from the APK's
//! `assets/` (via `AndroidApp::asset_manager`) is a documented follow-up.

use winit::platform::android::activity::AndroidApp;

/// The NativeActivity entry the android-activity glue invokes.
#[no_mangle]
fn android_main(app: AndroidApp) {
    // Log to logcat via the tracing subscriber (crash file lands in the app's cwd).
    crate::log::init(None, std::path::PathBuf::from("crash.txt"));
    tracing::info!("inf-player(android): starting");

    let built = crate::demo::build();
    let sim = crate::sim_from_built(built);
    if let Err(e) = crate::window::run_android(
        app,
        "Infini Engine".into(),
        sim,
        crate::input::default_map(),
    ) {
        tracing::error!("inf-player(android): {e}");
    }
}
