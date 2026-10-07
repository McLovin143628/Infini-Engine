use std::process::Command;

/// Embed build-time version metadata (P15.4): the git short hash becomes a
/// compile-time env var the `app_build_info` command surfaces in the About
/// dialog / status bar. It degrades to "unknown" when git is unavailable
/// (e.g. a source-tarball build), so the build never fails on its account.
fn main() {
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=INF_GIT_HASH={git_hash}");

    // Re-embed when HEAD moves so a fresh commit updates the hash.
    println!("cargo:rerun-if-changed=../../../.git/HEAD");

    // **A RELEASE EDITOR THAT LOADS THE DEV URL IS REFUSED** (the PERF1b
    // audit). A plain `cargo build --release -p inf-studio` builds the tauri
    // crate without its `custom-protocol` feature, so the binary does not embed
    // the built frontend and opens `devUrl` (Vite on :1440) -- a blank window,
    // or a stale dev server's UI, in an editor every perf session then measures.
    // `npx tauri build --no-bundle` (from editor/studio) is the release build:
    // it builds the frontend and turns the feature on. `INF_STUDIO_ALLOW_DEV_URL=1`
    // lets a release build that knowingly wants the dev server through.
    println!("cargo:rerun-if-env-changed=INF_STUDIO_ALLOW_DEV_URL");
    let release = std::env::var("PROFILE").is_ok_and(|p| p == "release");
    let allowed = std::env::var_os("INF_STUDIO_ALLOW_DEV_URL").is_some_and(|v| v == "1");
    if release && tauri_build::is_dev() && !allowed {
        panic!(
            "inf-studio: this RELEASE build would load the dev URL (http://localhost:1440) instead of embedding the frontend -- `cargo build --release -p inf-studio` does not enable tauri's `custom-protocol` feature. Build the release editor with `npx tauri build --no-bundle` from editor/studio (it builds the frontend and embeds it), or set INF_STUDIO_ALLOW_DEV_URL=1 if a release binary against the dev server is what you want."
        );
    }

    tauri_build::build();
}
