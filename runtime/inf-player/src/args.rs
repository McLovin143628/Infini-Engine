//! Command-line parsing for `inf-player` (P9.3).
//!
//! The player has four dispatch modes, chosen by flags (see [`Mode`]):
//!
//! * **windowed** (default) — open a winit window + wgpu surface and play a
//!   world: a cooked `--pack`, a loose `--level`, the `--demo` world, or — with
//!   no flag — the pack named by a `player.toml` beside the exe (exported game).
//! * **headless** (`--headless`) — no window/GPU: run `--run-frames N` fixed
//!   steps of the world and print the determinism hash. The CI smoke path
//!   (`--headless --run-frames 300 --assert-exit`).
//! * **pie** (`--pie`) — the play-in-editor subprocess (Spike D / P9.4): speak
//!   the length-prefixed bincode protocol on stdin/stdout. Unchanged here.
//! * **embed-probe** (`--embed-probe`, Windows) — the Spike D window-embedding
//!   probe. Unchanged here.

use std::path::PathBuf;

/// Which top-level path the process takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Open a window and play (human-verified, like all GPU paths).
    Windowed,
    /// No window: run N fixed steps and print the determinism hash (CI).
    Headless,
    /// Play-in-editor subprocess (bincode protocol over stdio).
    Pie,
    /// Windows-only window-embedding probe.
    EmbedProbe,
}

/// Which world the player builds and runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldChoice {
    /// The programmatic platformer world built in `demo.rs` (mirrors
    /// `samples/platformer-2d`). Fully runnable today.
    Demo,
    /// A loose `.inf_lvl` file on disk (the `--level` dev-dir path). Decoded by
    /// the `inf-scene` reader; `.inf_act` actor classes are read from the level's
    /// directory (or `--content`).
    ///
    /// [`LevelSource`]: crate::level::LevelSource
    /// [`WorldBuilder`]: crate::level::WorldBuilder
    Level(PathBuf),
    /// A cooked pack — either the directory holding `content.inf_pack` +
    /// `manifest.toml`, or the pack file itself (the `--pack` / exported-game
    /// path). The root level GUID comes from the manifest.
    Pack(PathBuf),
}

/// Fully-parsed player arguments.
#[derive(Debug, Clone)]
pub struct Args {
    pub mode: Mode,
    pub world: WorldChoice,
    /// Whether the world was chosen explicitly on the command line
    /// (`--demo`/`--level`/`--pack`). When false the player may boot from a
    /// `player.toml` beside its executable (the exported-game path).
    pub world_explicit: bool,
    /// Window-title override (from `player.toml`); falls back to the world label.
    pub title_override: Option<String>,
    /// Fixed steps to run in headless mode.
    pub run_frames: u64,
    /// Optional content directory for `--level` mode: where `.inf_act` actor
    /// classes are discovered. Defaults to the level file's own directory.
    pub content: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    /// PIE tick rate (only meaningful in `--pie`).
    pub tick_hz: u32,
    /// Optional log-file sink (in addition to stdout).
    pub log_file: Option<PathBuf>,
    /// Where the panic hook writes the crash report (default `crash.txt` in cwd).
    pub crash_file: PathBuf,
    /// Test/diagnostic hook: panic after this many fixed steps (headless), to
    /// exercise the crash-capture + nonzero-exit path. `None` = never.
    pub panic_after: Option<u64>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Windowed,
            world: WorldChoice::Demo,
            world_explicit: false,
            title_override: None,
            run_frames: 0,
            content: None,
            width: 1280,
            height: 720,
            tick_hz: inf_runtime::TICK_HZ,
            log_file: None,
            crash_file: PathBuf::from("crash.txt"),
            panic_after: None,
        }
    }
}

impl Args {
    /// Parse process arguments (skips argv[0]).
    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    /// Parse an argument iterator. Split out for unit tests.
    pub fn parse<I: IntoIterator<Item = String>>(iter: I) -> Result<Self, String> {
        let mut args = Args::default();
        let mut iter = iter.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--headless" => args.mode = Mode::Headless,
                "--pie" => args.mode = Mode::Pie,
                "--embed-probe" => args.mode = Mode::EmbedProbe,
                "--demo" => {
                    args.world = WorldChoice::Demo;
                    args.world_explicit = true;
                }
                // Accepted for CI-invocation compatibility; the exit code already
                // reflects success/failure.
                "--assert-exit" => {}
                "--level" => {
                    let v = iter.next().ok_or("--level needs a path")?;
                    args.world = WorldChoice::Level(PathBuf::from(v));
                    args.world_explicit = true;
                }
                "--pack" => {
                    let v = iter.next().ok_or("--pack needs a path")?;
                    args.world = WorldChoice::Pack(PathBuf::from(v));
                    args.world_explicit = true;
                }
                "--content" => {
                    let v = iter.next().ok_or("--content needs a directory")?;
                    args.content = Some(PathBuf::from(v));
                }
                "--run-frames" => {
                    let v = iter.next().ok_or("--run-frames needs a value")?;
                    args.run_frames = v
                        .parse()
                        .map_err(|_| format!("bad --run-frames value '{v}'"))?;
                }
                "--width" => {
                    let v = iter.next().ok_or("--width needs a value")?;
                    args.width = v.parse().map_err(|_| format!("bad --width '{v}'"))?;
                }
                "--height" => {
                    let v = iter.next().ok_or("--height needs a value")?;
                    args.height = v.parse().map_err(|_| format!("bad --height '{v}'"))?;
                }
                "--tick-hz" => {
                    let v = iter.next().ok_or("--tick-hz needs a value")?;
                    args.tick_hz = v.parse().map_err(|_| format!("bad --tick-hz '{v}'"))?;
                }
                "--log-file" => {
                    let v = iter.next().ok_or("--log-file needs a path")?;
                    args.log_file = Some(PathBuf::from(v));
                }
                "--crash-file" => {
                    let v = iter.next().ok_or("--crash-file needs a path")?;
                    args.crash_file = PathBuf::from(v);
                }
                "--panic-after" => {
                    let v = iter.next().ok_or("--panic-after needs a value")?;
                    args.panic_after =
                        Some(v.parse().map_err(|_| format!("bad --panic-after '{v}'"))?);
                }
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(a: &[&str]) -> Result<Args, String> {
        Args::parse(a.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_to_windowed_demo() {
        let a = parse(&[]).unwrap();
        assert_eq!(a.mode, Mode::Windowed);
        assert_eq!(a.world, WorldChoice::Demo);
    }

    #[test]
    fn ci_smoke_line_parses() {
        let a = parse(&["--headless", "--run-frames", "300", "--assert-exit"]).unwrap();
        assert_eq!(a.mode, Mode::Headless);
        assert_eq!(a.run_frames, 300);
        assert_eq!(a.world, WorldChoice::Demo);
    }

    #[test]
    fn level_and_content_capture_paths() {
        let a = parse(&["--level", "L.inf_lvl", "--content", "Content"]).unwrap();
        assert_eq!(a.world, WorldChoice::Level(PathBuf::from("L.inf_lvl")));
        assert_eq!(a.content, Some(PathBuf::from("Content")));
        assert!(a.world_explicit);
    }

    #[test]
    fn pack_flag_sets_pack_world_and_is_explicit() {
        let a = parse(&["--pack", "Build"]).unwrap();
        assert_eq!(a.world, WorldChoice::Pack(PathBuf::from("Build")));
        assert!(a.world_explicit);
    }

    #[test]
    fn default_world_is_not_explicit() {
        // No world flag → the exported-game boot-config path may apply.
        assert!(!parse(&["--headless"]).unwrap().world_explicit);
    }

    #[test]
    fn unknown_arg_errors() {
        assert!(parse(&["--nope"]).is_err());
    }
}
