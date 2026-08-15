//! Mobile + web export targets (P14.1 / P14.2).
//!
//! [`export_web`] assembles a **browser-runnable bundle skeleton**: it cooks the
//! project's `content.inf_pack`, writes an `index.html` + JS glue that calls the
//! wasm player's `start_player(canvas_id, pack_url)` entry, and — honestly — runs
//! the **two-step wasm build** (`cargo build --target wasm32-unknown-unknown` then
//! `wasm-bindgen`) *when the tools are present*, otherwise leaving exact
//! instructions. [`export_android`] cooks the pack and writes the `cargo-ndk` +
//! APK build steps (an APK cannot be assembled without the Android NDK/SDK, which
//! are not in this repo — see `docs/android-player.md`).
//!
//! Neither fakes a device build: the web `.wasm` is emitted only if the real
//! toolchain runs; the Android path is instructions + cooked assets, never a
//! stubbed "APK".

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_project::Project;

use crate::cook::{cook, CookOptions, CookReport, DEFAULT_PACK_NAME};
use crate::error::{CookError, Result};

/// Options for [`export_web`].
#[derive(Debug, Clone, Default)]
pub struct WebExportOptions {
    /// Output directory (the bundle is written directly into it). `None` →
    /// `<project>/Export/web`.
    pub out_dir: Option<PathBuf>,
    /// Attempt the two-step wasm build (`cargo build` + `wasm-bindgen`) when the
    /// tools are on `PATH`. When `false` (skeleton-only; the default the tests
    /// use) it writes the loader + instructions without invoking any toolchain.
    pub run_toolchain: bool,
}

/// The outcome of a [`export_web`].
#[derive(Debug, Clone)]
pub struct WebExportReport {
    pub bundle_dir: PathBuf,
    pub pack_path: PathBuf,
    pub index_html: PathBuf,
    pub instructions: PathBuf,
    /// Whether the `.wasm` + JS bindings were actually produced (the real
    /// toolchain ran). `false` ⇒ the bundle is skeleton-only + instructions.
    pub wasm_built: bool,
    /// **Why** the wasm step ended as it did (C4-40).
    ///
    /// `wasm_built` alone cannot separate "the toolchain is not installed" —
    /// which is the honest skeleton-only path — from "the player failed to
    /// compile", which ships an `index.html` importing an `inf_player.js` that
    /// was never produced. Both used to render as "wasm built: no" and exit 0.
    pub wasm: WasmOutcome,
    pub cook: CookReport,
}

/// How the optional two-step wasm build ended (C4-40).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmOutcome {
    /// Not attempted (`run_toolchain: false`).
    NotRequested,
    /// A required tool is absent. The bundle is skeleton + instructions, which
    /// is exactly what `WEB_BUILD.txt` is for — **not** a failure.
    ToolAbsent(String),
    /// The tools ran and the build **failed**. The bundle is broken: its page
    /// imports JS that does not exist.
    Failed(String),
    /// The `.wasm` and its JS glue were produced.
    Built,
}

impl WebExportReport {
    /// Whether this export must not ship (C4-40): a blocking cook, or a wasm
    /// build that was attempted and failed.
    pub fn has_blocking(&self) -> bool {
        self.cook.has_blocking() || matches!(self.wasm, WasmOutcome::Failed(_))
    }

    /// A human-readable CLI summary.
    pub fn render(&self) -> String {
        let mut s = self.cook.render();
        s.push_str(&format!(
            "Web export → {}\n  index: {}\n  wasm built: {}\n",
            self.bundle_dir.display(),
            self.index_html
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            match &self.wasm {
                WasmOutcome::Built => "yes".to_string(),
                WasmOutcome::NotRequested =>
                    "no — not requested; see WEB_BUILD.txt for the cargo/wasm-bindgen steps"
                        .to_string(),
                WasmOutcome::ToolAbsent(why) =>
                    format!("no — {why}; see WEB_BUILD.txt for the cargo/wasm-bindgen steps"),
                WasmOutcome::Failed(why) => format!(
                    "NO — the build FAILED ({why}). \
                     This bundle's page imports inf_player.js, which was never produced"
                ),
            },
        ));
        s
    }
}

/// Export a browser-runnable web bundle skeleton (P14.2).
pub fn export_web(project_root: &Path, opts: &WebExportOptions) -> Result<WebExportReport> {
    let project = Project::open(project_root)?;
    let name = project.manifest.name.clone();

    let out = opts
        .out_dir
        .clone()
        .unwrap_or_else(|| project_root.join("Export").join("web"));
    std::fs::create_dir_all(&out)?;

    // 1. cook the pack straight into the bundle (the wasm player fetches it).
    let cook = cook(project_root, &out, &CookOptions::default())?;

    // 2. the page + loader glue.
    let index_html = out.join("index.html");
    std::fs::write(&index_html, index_html_contents(&name))?;

    // 3. honest two-step build instructions (always written).
    let instructions = out.join("WEB_BUILD.txt");
    std::fs::write(&instructions, web_build_note(&name))?;

    // 4. run the real toolchain if asked + available (never faked).
    //
    // C4-40: the outcome used to be `try_build_wasm(..).unwrap_or(false)`, which
    // discarded `CookError::Export` *and* collapsed a compile failure into the
    // same `false` a missing toolchain produces.
    let wasm = if opts.run_toolchain {
        match try_build_wasm(&out) {
            Ok(o) => o,
            Err(e) => WasmOutcome::Failed(e.to_string()),
        }
    } else {
        WasmOutcome::NotRequested
    };

    Ok(WebExportReport {
        bundle_dir: out,
        pack_path: cook.pack_path.clone(),
        index_html,
        instructions,
        wasm_built: wasm == WasmOutcome::Built,
        wasm,
        cook,
    })
}

/// Options for [`export_android`].
#[derive(Debug, Clone, Default)]
pub struct AndroidExportOptions {
    /// Output directory. `None` → `<project>/Export/android`.
    pub out_dir: Option<PathBuf>,
}

/// The outcome of a [`export_android`].
#[derive(Debug, Clone)]
pub struct AndroidExportReport {
    pub bundle_dir: PathBuf,
    pub pack_path: PathBuf,
    pub instructions: PathBuf,
    pub cook: CookReport,
}

impl AndroidExportReport {
    /// Whether this export must not ship (C4-40) — here, entirely the cook's
    /// verdict: no toolchain step runs.
    pub fn has_blocking(&self) -> bool {
        self.cook.has_blocking()
    }

    /// A human-readable CLI summary.
    pub fn render(&self) -> String {
        let mut s = self.cook.render();
        s.push_str(&format!(
            "Android export (assets + build steps) → {}\n  next: follow {} (needs the Android NDK/SDK)\n",
            self.bundle_dir.display(),
            self.instructions
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
        ));
        s
    }
}

/// Export the Android build inputs (P14.1): the cooked pack + the `cargo-ndk` /
/// APK build steps. An actual APK requires the NDK/SDK (absent here), so this is
/// honestly instructions + assets, not a stubbed APK.
pub fn export_android(
    project_root: &Path,
    opts: &AndroidExportOptions,
) -> Result<AndroidExportReport> {
    let project = Project::open(project_root)?;
    let name = project.manifest.name.clone();

    let out = opts
        .out_dir
        .clone()
        .unwrap_or_else(|| project_root.join("Export").join("android"));
    std::fs::create_dir_all(&out)?;

    let cook = cook(project_root, &out, &CookOptions::default())?;

    let instructions = out.join("ANDROID_BUILD.txt");
    std::fs::write(&instructions, android_build_note(&name))?;

    Ok(AndroidExportReport {
        bundle_dir: out,
        pack_path: cook.pack_path.clone(),
        instructions,
        cook,
    })
}

/// Run the two-step wasm build if the tools are present.
///
/// Returns [`WasmOutcome::ToolAbsent`] when a tool is missing (the caller leaves
/// the instructions in place — a legitimate skeleton export) and
/// [`WasmOutcome::Failed`] when the tools ran and the build did not succeed. The
/// two used to be the same `Ok(false)`, which is why `inf export --target web`
/// shipped a page importing JS that was never produced, exit 0 (C4-40).
fn try_build_wasm(out_dir: &Path) -> Result<WasmOutcome> {
    // **The target itself is a tool** (round-2 finding B14). This probed
    // `wasm-bindgen` and never `wasm32-unknown-unknown`, even though the
    // sibling `mods.rs` grew exactly that question in the same wave. Without
    // the target, step 1 exits non-zero -> `Failed` -> `has_blocking()` ->
    // `FAILURE`, so `inf export --target web` **hard-fails** where it used to
    // ship a skeleton with instructions — contradicting `ToolAbsent`'s own doc
    // that a missing tool is not a failure. `run_toolchain: true` is the
    // default path, so this is what an author with a stock rustup hits.
    //
    // Only a definite `Absent` short-circuits: an `Unknown` is not evidence
    // (B14b), so the build is attempted and cargo gets to answer.
    if crate::mods::wasm_target() == crate::mods::WasmTarget::Absent {
        return Ok(WasmOutcome::ToolAbsent(
            "the wasm32-unknown-unknown target is not installed \
             (rustup target add wasm32-unknown-unknown)"
                .into(),
        ));
    }
    // wasm-bindgen-cli must be installed for the second step.
    if Command::new("wasm-bindgen")
        .arg("--version")
        .output()
        .is_err()
    {
        return Ok(WasmOutcome::ToolAbsent(
            "wasm-bindgen is not installed".into(),
        ));
    }
    let Some(ws) = workspace_root() else {
        return Ok(WasmOutcome::ToolAbsent(
            "no cargo workspace root above the current directory".into(),
        ));
    };

    // Step 1: build the player for wasm (getrandom needs the browser backend).
    let status = Command::new(env!("CARGO"))
        .current_dir(&ws)
        .env("RUSTFLAGS", "--cfg getrandom_backend=\"wasm_js\"")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "inf-player",
        ])
        .status()
        .map_err(|e| CookError::Export(format!("cargo build wasm: {e}")))?;
    if !status.success() {
        // B14, the other side: the target can be uninstalled between the probe
        // and the build, and a toolchain override inside the workspace can put
        // the build on a rustc the probe never asked. Classify by asking again
        // rather than by assuming the probe above still holds.
        if crate::mods::wasm_target() == crate::mods::WasmTarget::Absent {
            return Ok(WasmOutcome::ToolAbsent(
                "the wasm32-unknown-unknown target is not installed \
                 (rustup target add wasm32-unknown-unknown)"
                    .into(),
            ));
        }
        return Ok(WasmOutcome::Failed(format!(
            "cargo build --target wasm32-unknown-unknown -p inf-player exited {status}"
        )));
    }

    // Step 2: wasm-bindgen → JS glue + trimmed wasm into the bundle.
    let wasm = ws
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("inf_player.wasm");
    if !wasm.is_file() {
        return Ok(WasmOutcome::Failed(format!(
            "cargo reported success but {} does not exist",
            wasm.display()
        )));
    }
    let status = Command::new("wasm-bindgen")
        .args([
            "--target",
            "web",
            "--no-typescript",
            "--out-name",
            "inf_player",
            "--out-dir",
        ])
        .arg(out_dir)
        .arg(&wasm)
        .status()
        .map_err(|e| CookError::Export(format!("wasm-bindgen: {e}")))?;
    if status.success() {
        Ok(WasmOutcome::Built)
    } else {
        Ok(WasmOutcome::Failed(format!("wasm-bindgen exited {status}")))
    }
}

/// Walk up for the workspace root (`Cargo.toml` with a `[workspace]` table).
fn workspace_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file() {
            if let Ok(text) = std::fs::read_to_string(&manifest) {
                if text.contains("[workspace]") {
                    return Some(dir);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// The bundle's `index.html`: a full-viewport `<canvas>` that boots the player.
fn index_html_contents(name: &str) -> String {
    format!(
        "<!doctype html>\n\
         <html lang=\"en\">\n\
         <head>\n  \
             <meta charset=\"utf-8\" />\n  \
             <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n  \
             <title>{name}</title>\n  \
             <style>\n    \
                 html, body {{ margin: 0; height: 100%; background: #101014; overflow: hidden; }}\n    \
                 canvas {{ display: block; width: 100vw; height: 100vh; touch-action: none; }}\n  \
             </style>\n\
         </head>\n\
         <body>\n  \
             <canvas id=\"game\"></canvas>\n  \
             <script type=\"module\">\n    \
                 // Produced by `wasm-bindgen --target web` (see WEB_BUILD.txt).\n    \
                 import init, {{ start_player }} from './inf_player.js';\n    \
                 await init();\n    \
                 start_player('game', './{pack}');\n  \
             </script>\n\
         </body>\n\
         </html>\n",
        name = name,
        pack = DEFAULT_PACK_NAME,
    )
}

/// The honest two-step web-build note (WebGPU + toolchain caveats).
fn web_build_note(name: &str) -> String {
    format!(
        "Infinity Engine — web export: {name}\n\
         ================================================\n\n\
         This folder is a browser bundle SKELETON. It already contains:\n\
         * index.html          — a full-viewport <canvas id=\"game\"> that boots the game\n\
         * content.inf_pack     — the cooked asset pack (fetched at runtime)\n\
         * manifest.toml        — the cook manifest\n\n\
         To produce the runnable player you need the wasm module + JS glue. If\n\
         `inf export --target web` did not already emit inf_player.js + inf_player_bg.wasm\n\
         (because the toolchain was not installed), run the TWO-STEP build yourself:\n\n\
         1. Install the tools (once):\n\
              rustup target add wasm32-unknown-unknown\n\
              cargo install wasm-bindgen-cli\n\n\
         2. Build + bind:\n\
              # getrandom needs the browser backend on wasm:\n\
              RUSTFLAGS='--cfg getrandom_backend=\"wasm_js\"' \\\n\
                cargo build --release --target wasm32-unknown-unknown -p inf-player\n\
              wasm-bindgen --target web --no-typescript --out-name inf_player \\\n\
                --out-dir <this folder> \\\n\
                target/wasm32-unknown-unknown/release/inf_player.wasm\n\n\
         3. Serve over HTTP (WebGPU + module scripts need a real origin, not file://):\n\
              python -m http.server 8080     # then open http://localhost:8080\n\n\
         Requirements (honest):\n\
         * A WebGPU browser (Chrome/Edge 113+, or Firefox/Safari with WebGPU on).\n\
         * WebGPU adapter acquisition is async; the standalone player's GPU init\n\
           currently uses a blocking path (fine on desktop). A live in-browser run\n\
           needs inf-render's async-adapter seam — see docs/web-player.md. Until\n\
           then this bundle compiles + loads but the GPU surface init is the known\n\
           remaining runtime step.\n"
    )
}

/// The honest Android build note (cargo-ndk + APK, NDK required).
fn android_build_note(name: &str) -> String {
    format!(
        "Infinity Engine — Android export: {name}\n\
         ================================================\n\n\
         This folder contains the cooked assets (content.inf_pack + manifest.toml).\n\
         Building the APK requires the Android SDK + NDK, which are NOT in this repo.\n\n\
         Build the player shared object with cargo-ndk (once tools are installed):\n\n\
         1. Install:\n\
              rustup target add aarch64-linux-android armv7-linux-androideabi \\\n\
                x86_64-linux-android\n\
              cargo install cargo-ndk\n\
              # + the Android SDK/NDK; set ANDROID_NDK_HOME.\n\n\
         2. Build the player .so (per ABI):\n\
              cargo ndk -t arm64-v8a -o app/src/main/jniLibs \\\n\
                build --release -p inf-player\n\n\
         3. Package the APK: drop content.inf_pack into the app's assets/, point a\n\
            minimal NativeActivity manifest at the built libinf_player.so, and\n\
            assemble with Gradle. See docs/android-player.md for the full walkthrough.\n\n\
         Honest status: the player has the cfg(target_os = \"android\") entry\n\
         (android_main) + winit android-activity feature + touch input + the mobile\n\
         render tier. Compilation is NDK-gated (a plain `cargo check` needs\n\
         aarch64-linux-android-clang); the run is device-verified. No stub APK is\n\
         produced here.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_html_wires_canvas_and_pack() {
        let html = index_html_contents("My Game");
        assert!(html.contains("<canvas id=\"game\">"));
        assert!(html.contains("start_player('game', './content.inf_pack')"));
        assert!(html.contains("<title>My Game</title>"));
    }

    #[test]
    fn web_note_has_the_two_step_build() {
        let note = web_build_note("G");
        assert!(note.contains("wasm32-unknown-unknown"));
        assert!(note.contains("wasm-bindgen --target web"));
        assert!(note.contains("getrandom_backend"));
    }

    #[test]
    fn android_note_has_cargo_ndk() {
        let note = android_build_note("G");
        assert!(note.contains("cargo ndk"));
        assert!(note.contains("ANDROID_NDK_HOME"));
    }

    /// **Round-2 finding B14**: `inf export --target web` probes the tool and
    /// not the target.
    ///
    /// `try_build_wasm` checked `wasm-bindgen --version` and never
    /// `wasm32-unknown-unknown`, though the sibling `mods.rs` grew exactly that
    /// question in the same wave. Without the target, step 1 exits non-zero ->
    /// `WasmOutcome::Failed` -> `has_blocking()` -> `FAILURE`, so the export
    /// **hard-fails** where it used to ship a skeleton with instructions —
    /// contradicting `ToolAbsent`'s own doc that a missing tool is not a
    /// failure. `run_toolchain: true` is the default path.
    ///
    /// A source pin: reaching the branch needs a machine without the target and
    /// a full release wasm build, neither of which a test can arrange.
    #[test]
    fn the_web_export_probes_the_target_and_not_only_the_tool() {
        let src = include_str!("targets.rs").replace("\r\n", "\n");
        let at = src
            .find("fn try_build_wasm(")
            .expect("`try_build_wasm` occurs nowhere — was it renamed?");
        let rest = &src[at..];
        let body = &rest[..rest.find("\n}\n").unwrap_or(rest.len())];

        let probe = body
            .find("wasm_target()")
            .expect("`try_build_wasm` never asks whether the wasm target is installed");
        let build = body
            .find("wasm32-unknown-unknown\",")
            .expect("`try_build_wasm` no longer runs the cargo build this pin is about");
        assert!(
            probe < build,
            "the target probe happens AFTER the build it is supposed to make \
             unnecessary"
        );
        assert!(
            body.contains("WasmTarget::Absent"),
            "the probe no longer requires a DEFINITE absence, so an unanswerable \
             one would report a missing tool (B14b)"
        );
        assert_eq!(
            body.matches("WasmOutcome::ToolAbsent").count(),
            4,
            "this pin is calibrated against three ToolAbsent sites: the target \
             probe, the wasm-bindgen probe, and the re-classification of a failed \
             step 1"
        );
    }
}
