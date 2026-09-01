//! Desktop export: assemble a cook into a runnable, double-clickable folder
//! (ROADMAP P9.5, deliverable 1).
//!
//! [`export`] layers on top of [`cook`](crate::cook): it cooks the project, then
//! places the release `inf-player` executable (renamed to the project) beside the
//! `content.ipack` + `manifest.toml` the cook produced, and drops a
//! `player.toml` launch config the player reads on boot (see
//! `inf_player::config`). The result is a self-contained bundle:
//!
//! ```text
//! <out>/<ProjectName>/
//!   ├── <ProjectName>[.exe]     # the renamed release player
//!   ├── content.ipack        # cooked assets
//!   ├── manifest.toml           # cook manifest (root level, pack list, …)
//!   ├── player.toml             # launch config (pack, window title/size)
//!   └── docs/PACKAGING.txt       # honest per-OS packaging status
//! ```
//!
//! # Scope (honest)
//!
//! This is **Windows-first correctness** with an OS-agnostic layout: the same
//! folder runs on macOS/Linux (the player binary is just copied and renamed).
//! Real per-OS *installers/containers* — a signed `.app`/`.dmg`, an AppImage, a
//! Windows installer — plus **icon/branding injection** are documented follow-ups
//! (see `docs/PACKAGING.txt`, written into every bundle). We deliberately do **not**
//! fake notarization or generate installer stubs that would misrepresent signing.
//!
//! # The player binary
//!
//! Building `inf-player` in release is expensive, so [`export`] resolves it in
//! order: (1) an explicit [`ExportOptions::player_bin`]; (2) a prebuilt
//! `inf-player` **beside the running CLI**; (3) a `cargo build --release -p
//! inf-player` when invoked from a source checkout. Tests pass an explicit
//! (debug) binary so they never pay the release-build cost.

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_project::Project;

use crate::cook::{cook, CookOptions, CookReport, DEFAULT_PACK_NAME};
use crate::error::{CookError, Result};

/// Which platform the bundle targets. Only [`ExportTarget::Current`] (the host)
/// is implemented; cross-target export is a follow-up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExportTarget {
    /// Bundle for the machine running the export.
    #[default]
    Current,
}

/// Window defaults written into the bundle's `player.toml`.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: Option<String>,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: None,
            width: 1280,
            height: 720,
        }
    }
}

/// Options controlling a desktop export.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Output directory. The bundle folder `<ProjectName>/` is created inside it.
    /// `None` → `<project>/Export`.
    pub out_dir: Option<PathBuf>,
    /// Explicit prebuilt `inf-player` binary to bundle (tests pass the debug one).
    pub player_bin: Option<PathBuf>,
    /// Target platform (host only for now).
    pub target: ExportTarget,
    /// Window defaults for the launch config.
    pub window: WindowConfig,
}

/// The outcome of a successful [`export`].
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// The bundle folder (`<out>/<ProjectName>/`).
    pub bundle_dir: PathBuf,
    /// The bundled (renamed) executable.
    pub exe_path: PathBuf,
    /// The player binary that was copied in.
    pub player_source: PathBuf,
    /// The bundled pack.
    pub pack_path: PathBuf,
    /// The bundled manifest.
    pub manifest_path: PathBuf,
    /// The generated launch config.
    pub config_path: PathBuf,
    /// The per-OS packaging note.
    pub docs_path: PathBuf,
    /// The underlying cook report.
    pub cook: CookReport,
}

impl ExportReport {
    /// Whether this export must not ship (C4-40) — the cook's verdict, carried
    /// through the bundle so `inf export` can refuse the same builds `inf cook`
    /// does.
    pub fn has_blocking(&self) -> bool {
        self.cook.has_blocking()
    }

    /// A human-readable summary for CLI output.
    pub fn render(&self) -> String {
        let mut s = self.cook.render();
        s.push_str(&format!(
            "Exported \"{}\" → {}\n  exe: {}\n  config: {}\n",
            self.cook.project_name,
            self.bundle_dir.display(),
            self.exe_path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            self.config_path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
        ));
        s
    }
}

/// Export `project_root` into a runnable bundle folder.
pub fn export(project_root: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ExportTarget::Current = opts.target;

    // Open the project up front for its name (the bundle folder + exe name).
    let project = Project::open(project_root)?;
    let name = project.manifest.name.clone();

    let out = opts
        .out_dir
        .clone()
        .unwrap_or_else(|| project_root.join("Export"));
    let bundle_dir = out.join(&name);
    std::fs::create_dir_all(&bundle_dir)?;

    // 1. cook straight into the bundle (writes content.ipack + manifest.toml).
    let cook = cook(project_root, &bundle_dir, &CookOptions::default())?;

    // 2. resolve + copy the player binary, renamed to the project.
    let player_source = resolve_player_bin(opts.player_bin.as_deref())?;
    let exe_name = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let exe_path = bundle_dir.join(&exe_name);
    std::fs::copy(&player_source, &exe_path).map_err(|e| {
        CookError::Export(format!(
            "copy player {} → {}: {e}",
            player_source.display(),
            exe_path.display()
        ))
    })?;

    // 3. write the launch config the player reads on boot.
    let config = LaunchConfig {
        // **The constant, not a literal** (round-2, the MED cluster). This was
        // `1` in two places while `CONFIG_SCHEMA_VERSION` had zero external
        // consumers — and the player's loader now REFUSES a config newer than
        // its own, so bumping the constant alone would brick every exported
        // bundle with nothing in the tree relating the pair.
        schema_version: LAUNCH_CONFIG_SCHEMA_VERSION,
        pack: DEFAULT_PACK_NAME.to_string(),
        title: Some(opts.window.title.clone().unwrap_or_else(|| name.clone())),
        width: Some(opts.window.width),
        height: Some(opts.window.height),
    };
    let config_path = bundle_dir.join("player.toml");
    let config_toml = toml::to_string_pretty(&config)
        .map_err(|e| CookError::Export(format!("serialize player.toml: {e}")))?;
    std::fs::write(&config_path, config_toml)?;

    // 4. honest per-OS packaging note.
    let docs_dir = bundle_dir.join("docs");
    std::fs::create_dir_all(&docs_dir)?;
    let docs_path = docs_dir.join("PACKAGING.txt");
    std::fs::write(&docs_path, packaging_note(&name))?;

    Ok(ExportReport {
        bundle_dir,
        exe_path,
        player_source,
        pack_path: cook.pack_path.clone(),
        manifest_path: cook.manifest_path.clone(),
        config_path,
        docs_path,
        cook,
    })
}

/// The `player.toml` schema version this exporter writes.
///
/// Named once (round-2, the MED cluster): it was the literal `1` at two sites
/// while `inf_player::config::CONFIG_SCHEMA_VERSION` had **zero external
/// consumers**, and the player's loader refuses a config newer than its own —
/// so bumping the player's constant alone would brick every exported bundle,
/// with nothing in the tree relating the pair. `inf-packager` does not depend
/// on `inf-player`, so the tie is a source pin
/// (`tests::the_launch_config_schema_matches_the_player_s`) rather than a
/// shared item — the Wave-C law for two copies of one value across a boundary.
pub const LAUNCH_CONFIG_SCHEMA_VERSION: u32 = 1;

/// The bundle's `player.toml` (mirrors `inf_player::config::PlayerConfigFile`;
/// duplicated so the cook pipeline does not depend on the player crate).
#[derive(Debug, serde::Serialize)]
struct LaunchConfig {
    schema_version: u32,
    pack: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
}

/// Resolve the `inf-player` binary to bundle (see the module docs for the order).
fn resolve_player_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return if p.is_file() {
            Ok(p.to_path_buf())
        } else {
            Err(CookError::Export(format!(
                "--player-bin {} not found",
                p.display()
            )))
        };
    }

    // A prebuilt player beside the running CLI (the shipped-tools layout).
    let bin_name = format!("inf-player{}", std::env::consts::EXE_SUFFIX);
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(&bin_name);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }

    // A source checkout: release-build the player and use target/release/.
    if let Some(ws) = workspace_root() {
        let status = Command::new(env!("CARGO"))
            .current_dir(&ws)
            .args(["build", "--release", "-p", "inf-player"])
            .status()
            .map_err(|e| CookError::Export(format!("cargo build inf-player: {e}")))?;
        if !status.success() {
            return Err(CookError::Export(
                "cargo build --release -p inf-player failed".to_string(),
            ));
        }
        let built = ws.join("target").join("release").join(&bin_name);
        if built.is_file() {
            return Ok(built);
        }
    }

    Err(CookError::Export(
        "could not find an inf-player binary — pass --player-bin, place one beside \
         the CLI, or run from a source checkout"
            .to_string(),
    ))
}

/// Walk up from the current dir looking for the workspace root (a `Cargo.toml`
/// containing a `[workspace]` table).
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

fn packaging_note(name: &str) -> String {
    format!(
        "Infini Engine — desktop export: {name}\n\
         ================================================\n\n\
         This folder is a self-contained, runnable build:\n\n\
         * {name}{exe}  — the game executable (the Infini Engine player,\n\
           renamed). It reads player.toml beside it and boots content.ipack.\n\
         * content.ipack       — the cooked, content-addressed asset pack.\n\
         * manifest.toml          — the cook manifest (root level, pack list).\n\
         * player.toml            — launch config (pack path, window title/size).\n\n\
         Running\n\
         -------\n\
         Windows : double-click {name}.exe (or run it from a terminal).\n\
         Linux   : ./{name}\n\
         macOS   : ./{name}\n\
         Headless smoke: {name}{exe} --headless --run-frames 300 --assert-exit\n\n\
         Per-OS packaging status (honest)\n\
         --------------------------------\n\
         Implemented : a portable, OS-agnostic runnable folder (this layout).\n\
         Follow-ups  : platform installers/containers and icon/branding injection\n\
                       are NOT yet produced —\n\
                         * Windows: an installer (MSI/NSIS) + exe icon embedding.\n\
                         * macOS  : a .app bundle + codesign + notarization. This\n\
                                    build is NOT signed or notarized; Gatekeeper\n\
                                    will warn. No notarization is faked here.\n\
                         * Linux  : an AppImage/.deb.\n\
         Until then, distribute this folder as-is (e.g. zipped).\n",
        exe = std::env::consts::EXE_SUFFIX,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_config_serializes_pack_and_window() {
        let c = LaunchConfig {
            schema_version: LAUNCH_CONFIG_SCHEMA_VERSION,
            pack: DEFAULT_PACK_NAME.to_string(),
            title: Some("Game".into()),
            width: Some(800),
            height: Some(600),
        };
        let t = toml::to_string_pretty(&c).unwrap();
        assert!(t.contains("pack = \"content.ipack\""));
        assert!(t.contains("title = \"Game\""));
        assert!(t.contains("width = 800"));
    }

    #[test]
    fn missing_explicit_player_bin_errors() {
        let err = resolve_player_bin(Some(Path::new("/nope/does/not/exist"))).unwrap_err();
        assert!(matches!(err, CookError::Export(_)));
    }

    /// **Round-2, the MED cluster**: the exporter's `player.toml` schema and
    /// the player's must be the same number.
    ///
    /// `inf-packager` does not depend on `inf-player`, so this reads the
    /// player's source rather than its constant — the Wave-C law: *two copies
    /// of one expression across a boundary are a contract nobody is measuring*,
    /// and a source pin is the enforcement available.
    #[test]
    fn the_launch_config_schema_matches_the_player_s() {
        // **Round 3: this call did nothing** — the same eaten escapes as
        // `inf_ecs::components`'s census, and invisible to clippy for the same
        // reason (`no_effect_replace` did not fire on a literal spelled as a
        // real newline, in a tree that was clippy-clean).
        let src = include_str!("../../inf-player/src/config.rs").replace("\r\n", "\n");
        let decl = "pub const CONFIG_SCHEMA_VERSION: u32 = ";
        let at = src
            .find(decl)
            .expect("the player's CONFIG_SCHEMA_VERSION occurs nowhere — was it renamed?");
        let value: u32 = src[at + decl.len()..]
            .split(';')
            .next()
            .unwrap()
            .trim()
            .parse()
            .expect("the player's schema version is a literal");
        assert_eq!(
            LAUNCH_CONFIG_SCHEMA_VERSION, value,
            "the exporter writes player.toml schema {LAUNCH_CONFIG_SCHEMA_VERSION} and the player speaks {value}; the loader refuses a config newer than its own, so every exported bundle would refuse to boot"
        );
    }
}
