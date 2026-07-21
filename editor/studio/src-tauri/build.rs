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

    tauri_build::build();
}
