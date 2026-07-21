# Android player — P14.1

The standalone player (`runtime/inf-player`) has `#[cfg(target_os = "android")]`
paths that build it as an Android NativeActivity: a `winit` android-activity
entry, on-screen **touch controls**, and the **mobile render tier**. The engine
itself is unchanged — Android runs the same fixed-step loop, `inf-render` forward
renderer, and blueprint runtime as desktop.

## Honest status (what is and isn't verified)

- **Code paths exist and are structured for compilation**: the `android_main`
  entry (`src/android.rs`), the winit `android-native-activity` feature +
  `run_android` event loop (`src/window.rs`), winit `Touch` → `TouchControls`
  routing, and `RenderTier::mobile_default()` applied on `target_os = "android"`.
- **Compilation is NDK-gated.** A plain `cargo check --target
  aarch64-linux-android` **cannot** build the C dependencies (`zstd-sys`,
  `meshopt`) without the NDK's `aarch64-linux-android-clang`. Unlike the web
  target (where those C deps are cfg'd out and swapped for pure Rust), Android is
  a full native platform and keeps them — so it is built through **`cargo-ndk`**,
  which wires the NDK toolchain.
- **CI**: the `wasm-check` job includes a **non-blocking, best-effort** `cargo-ndk`
  Android check (GitHub's Ubuntu runners preinstall the NDK). It never gates the
  build — Android's real proof is a device/emulator run, which this repo cannot
  perform (no devkit, no signing).
- **No stub APK is ever produced.** `inf export --target android` cooks the pack
  and writes the exact build steps; it does not fake an APK.

## Prerequisites

```sh
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
cargo install cargo-ndk
# Android Studio → SDK Manager → install the NDK (side-by-side).
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/<version>
```

## Compile-check the player for Android

```sh
cargo ndk -t arm64-v8a check -p inf-player
```

This is what CI attempts (best-effort). It compiles the `cfg(target_os =
"android")` paths (including `android_main`) with the NDK clang.

## Build the player shared object

The APK loads the player as a `.so`. Build one per ABI you ship:

```sh
cargo ndk -t arm64-v8a -t armeabi-v7a -o app/src/main/jniLibs \
  build --release -p inf-player
```

`cargo-ndk` places `libinf_player.so` under `app/src/main/jniLibs/<abi>/`.

## Assemble the APK (Gradle)

There is no committed Gradle project (it needs the SDK, which is out of this
repo). The minimal shape:

1. **`AndroidManifest.xml`** — a `NativeActivity` (or `GameActivity`) pointing at
   `android.app.NativeActivity` with `meta-data android.app.lib_name` = `inf_player`.
2. **`assets/content.inf_pack`** — the cooked pack (from `inf export --target
   android`, or `inf cook`). Loading it from the APK's asset manager (via
   `AndroidApp::asset_manager()`) is the documented follow-up; v1's `android_main`
   runs the bundled `--demo` world.
3. **`jniLibs/<abi>/libinf_player.so`** — from the `cargo-ndk build` above.
4. `./gradlew assembleDebug` → an installable APK.

## Input, rendering, perf

- **Touch**: winit `Touch` events route through `inf_input::TouchControls`
  (`crate::input::default_touch_controls()`): a left **virtual stick** →
  `move_x`/`move_y` and a right **jump button** → the South face button. The
  controls emit the same gamepad events a physical pad would, so touch reuses the
  whole `InputMap` pipeline (see `crates/inf-input/src/touch.rs`). A game with a
  different scheme builds its own `TouchControls` (e.g. `TouchButton`s bound to
  the D-pad for the 2D sample's digital `left`/`right`).
- **Render tier**: `RenderTier::mobile_default()` is applied on Android — no
  virtualized geometry, no SSAO/GI/TAA/bloom, shadows off — then the live adapter
  tier clamps further on weak GPUs. Honest note: CSM shadow *resolution* and MSAA
  sample count are compile-time constants today, so the preset turns shadows
  **off** rather than shrinking the map; a runtime shadow-resolution/MSAA knob is a
  follow-up (`crates/inf-render/src/caps.rs`).

## iOS (Metal) — status

iOS shares the mobile tier + touch design. wgpu's Metal backend + winit's iOS
support are the target; an Xcode project + signing profile are required and, like
the Android SDK, are **not in this repo**. iOS export (`xcodeproj` generation +
signing docs) is the P14.1 follow-up — the render/input foundations here are
platform-neutral and already apply.
