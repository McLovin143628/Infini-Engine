//! [`GameSettings`]: **the shipped game's user settings**, and the doctrine they
//! obey.
//!
//! # The doctrine, restated where a player can reach it
//!
//! The editor has had a settings file since Wave E — absent is the default,
//! corrupt is an **error** with the bytes left untouched, a newer schema is
//! **refused** rather than downgraded, every write is atomic, and every numeric
//! is guarded at the door. A shipped game had **none of that**: `player.toml` is
//! a boot config nothing writes at runtime, and `input.toml` is a best-effort
//! read beside a level that a cooked pack never even looks at.
//!
//! So the same three clauses live here, for the same three reasons, and the
//! sentences are deliberately the ones the editor's door uses:
//!
//! * **Absent is the default.** A first run has no file and must not be an
//!   error.
//! * **Corrupt is an error, and the bytes are left alone.** Replacing a file
//!   somebody hand-edited with defaults destroys the thing they were editing;
//!   telling them it cannot be read lets them fix it.
//! * **Newer is refused.** A build that meets a schema it does not understand
//!   says so instead of writing its own shape over it.
//!
//! # Why the file is name-keyed TOML and not a wire schema
//!
//! Every value here is a *preference*, not sim state: nothing in a `.inf_lvl`,
//! a `.inf_pack` or a replay depends on it, and two players with different
//! settings must produce the same simulation from the same inputs. That is what
//! lets the file be a named table with no positional decode and no version
//! ladder past the one guard above — the `blend_mode` precedent, and the reason
//! this whole wave moved no schema.
//!
//! # The tier is a NAME
//!
//! `render_tier` is a string rather than an `inf_render::RenderTier` because
//! this crate must not depend on `inf-render` — the render node that draws the
//! menu lives *in* `inf-render`, so the edge would be a cycle. The host maps the
//! name; an unknown one reads as `auto`, which is what a settings file written
//! by a newer build with a fourth tier means to a build with three.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The schema this build writes and understands.
pub const GAME_SETTINGS_VERSION: u32 = 1;

/// The file's name inside whatever directory the host resolves.
pub const GAME_SETTINGS_FILE: &str = "game-settings.toml";

/// The window modes the video page offers, in the order it lists them.
pub const WINDOW_MODES: [&str; 3] = ["windowed", "borderless", "fullscreen"];
/// The render tiers the video page offers. `auto` is the default and means
/// "whatever `detect_tier` decides on this machine".
pub const RENDER_TIERS: [&str; 4] = ["auto", "high", "medium", "low"];
/// The resolutions the video page offers, widest first.
pub const RESOLUTIONS: [(u32, u32); 6] = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (1024, 576),
];

/// The band a look sensitivity is clamped into — the editor camera's own
/// (`LOOK_SENSITIVITY_RANGE`), because it is the same gesture on the same mouse
/// and two different bands would be two answers to one question.
pub const LOOK_SENSITIVITY_RANGE: (f32, f32) = (0.05, 10.0);
/// The band a mixer gain is clamped into. The top is `1.0` and not more: a bus
/// gain above unity is a clip, and a settings slider is not the place to author
/// one.
pub const VOLUME_RANGE: (f32, f32) = (0.0, 1.0);
/// The band the click/long-press threshold is clamped into, milliseconds — the
/// integer face of `inf_ecs::movement::PRESS_THRESHOLD_RANGE_S`, which is where
/// the argument for the two ends is written down.
pub const PRESS_THRESHOLD_RANGE_MS: (u32, u32) = (50, 2000);
/// The smallest and largest resolution the file may carry, pixels. Wide enough
/// for anything [`RESOLUTIONS`] offers and for a hand-edited oddity, bounded so
/// a hostile file cannot ask for a surface no device can allocate.
pub const RESOLUTION_RANGE: (u32, u32) = (320, 16384);

/// A player's settings for a shipped game.
///
/// **Field order is load-bearing**: every scalar comes before the one table,
/// because `toml` refuses to emit a value after a table. The editor's own
/// settings type carries the same note for the same reason.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GameSettings {
    /// The schema this file was written with. See [`migrate`](Self::migrate).
    pub schema_version: u32,
    // ── video ──
    /// One of [`WINDOW_MODES`]; anything else reads as the default.
    pub window_mode: String,
    /// Backbuffer width in pixels.
    pub resolution_w: u32,
    /// Backbuffer height in pixels.
    pub resolution_h: u32,
    /// One of [`RENDER_TIERS`]; anything else reads as `auto`.
    pub render_tier: String,
    // ── audio ──
    /// The master bus gain, `[0, 1]`.
    pub master_volume: f32,
    /// The effects bus gain, `[0, 1]`.
    pub sfx_volume: f32,
    /// The music bus gain, `[0, 1]`.
    pub music_volume: f32,
    // ── controls ──
    /// A multiplier on the look axes' authored sensitivity.
    pub look_sensitivity: f32,
    /// Whether pushing the mouse forward looks **down**.
    pub invert_look_y: bool,
    /// How long a press must be held to count as a long press, milliseconds.
    pub press_threshold_ms: u32,
    // ── bindings (the one table; it must stay last) ──
    /// **Only what differs from the shipped table**, keyed by row id (see
    /// [`crate::bindings`]).
    ///
    /// Storing overrides rather than the whole map is the editor keybinding
    /// door's rule and it earns its keep the same way: a table that shipped the
    /// whole map would freeze a player's bindings at the build they first ran,
    /// so a control added later would arrive unbound and a default corrected
    /// later would never reach them. An **empty value means "unbound"** — the
    /// difference between a row a player cleared and a row they never touched.
    pub bindings: BTreeMap<String, String>,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            schema_version: GAME_SETTINGS_VERSION,
            window_mode: WINDOW_MODES[0].to_string(),
            resolution_w: 1920,
            resolution_h: 1080,
            render_tier: RENDER_TIERS[0].to_string(),
            master_volume: 1.0,
            sfx_volume: 1.0,
            music_volume: 1.0,
            look_sensitivity: 1.0,
            invert_look_y: false,
            press_threshold_ms: 250,
            bindings: BTreeMap::new(),
        }
    }
}

/// Non-finite falls back to `fallback`; finite clamps into `range`.
///
/// **The two halves are different on purpose**, and it is the editor door's
/// rule verbatim: an infinity is not "a very large sensitivity", it is the
/// absence of one, so it takes the default rather than the top of the band.
fn guard_f32(v: f32, range: (f32, f32), fallback: f32) -> f32 {
    if !v.is_finite() {
        return fallback;
    }
    v.clamp(range.0, range.1)
}

/// The settings file's path inside `dir`.
pub fn settings_path(dir: &Path) -> PathBuf {
    dir.join(GAME_SETTINGS_FILE)
}

impl GameSettings {
    /// Refuse a file a newer build wrote; accept anything this build can read.
    ///
    /// There is no ladder yet because there is one version. What there is, from
    /// the first version, is the **guard**: a build that meets a schema it does
    /// not understand says so and leaves the bytes alone, rather than writing
    /// its own shape over a file that means something else.
    pub fn migrate(&mut self) -> Result<(), String> {
        if self.schema_version > GAME_SETTINGS_VERSION {
            return Err(format!(
                "these settings were written by a newer build (schema v{}, this build understands v{GAME_SETTINGS_VERSION}); the file is left untouched",
                self.schema_version
            ));
        }
        self.schema_version = GAME_SETTINGS_VERSION;
        Ok(())
    }

    /// Load from `dir`, or the defaults when there is no file yet.
    ///
    /// **Absent is `Ok(default)`; anything else is an error and the bytes on
    /// disk are never rewritten.** A player who hand-edited the file and got a
    /// comma wrong is told so; they are not silently given a fresh one.
    pub fn load_or_default(dir: &Path) -> Result<Self, String> {
        let path = settings_path(dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => {
                return Err(format!(
                    "{} exists but cannot be read ({e}); it is left untouched rather than replaced by defaults -- repair or delete it",
                    path.display()
                ))
            }
        };
        let mut s: Self = toml::from_str(&text).map_err(|e| {
            format!(
                "{} exists but cannot be read ({e}); it is left untouched rather than replaced by defaults -- repair or delete it",
                path.display()
            )
        })?;
        s.migrate()?;
        s.normalize();
        Ok(s)
    }

    /// **Total, idempotent and NaN-refusing.** Every numeric door in one place,
    /// so a value that arrived from a file, from a slider or from a mod is
    /// guarded the same way.
    pub fn normalize(&mut self) {
        self.schema_version = GAME_SETTINGS_VERSION;
        if !WINDOW_MODES.contains(&self.window_mode.as_str()) {
            self.window_mode = WINDOW_MODES[0].to_string();
        }
        if !RENDER_TIERS.contains(&self.render_tier.as_str()) {
            // A tier this build has never heard of is what a file written by a
            // newer one carries. `auto` is the honest reading: let this build
            // decide for itself rather than pick a tier at random.
            self.render_tier = RENDER_TIERS[0].to_string();
        }
        self.resolution_w = self
            .resolution_w
            .clamp(RESOLUTION_RANGE.0, RESOLUTION_RANGE.1);
        self.resolution_h = self
            .resolution_h
            .clamp(RESOLUTION_RANGE.0, RESOLUTION_RANGE.1);
        let d = Self::default();
        self.master_volume = guard_f32(self.master_volume, VOLUME_RANGE, d.master_volume);
        self.sfx_volume = guard_f32(self.sfx_volume, VOLUME_RANGE, d.sfx_volume);
        self.music_volume = guard_f32(self.music_volume, VOLUME_RANGE, d.music_volume);
        self.look_sensitivity = guard_f32(
            self.look_sensitivity,
            LOOK_SENSITIVITY_RANGE,
            d.look_sensitivity,
        );
        self.press_threshold_ms = self
            .press_threshold_ms
            .clamp(PRESS_THRESHOLD_RANGE_MS.0, PRESS_THRESHOLD_RANGE_MS.1);
        // A blank row id names nothing; it would render as a row with no
        // control and could never be matched against the map.
        self.bindings.retain(|row, _| !row.trim().is_empty());
    }

    /// The threshold as seconds, which is the unit the movement intent reads.
    pub fn press_threshold_s(&self) -> f64 {
        f64::from(self.press_threshold_ms) / 1000.0
    }

    /// Serialize, normalizing a **clone** first so writing cannot be the thing
    /// that changes a value.
    pub fn to_toml(&self) -> Result<String, String> {
        let mut c = self.clone();
        c.normalize();
        toml::to_string_pretty(&c).map_err(|e| format!("encode settings: {e}"))
    }

    /// Write atomically into `dir`, creating it if needed.
    ///
    /// `inf_asset::write_atomically` stages beside the target and replaces, so a
    /// game killed mid-save leaves either the old file or the new one and never
    /// a half of either — the same door every asset write in this engine uses.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let text = self.to_toml()?;
        inf_asset::write_atomically(&settings_path(dir), text.as_bytes())
            .map_err(|e| format!("write settings: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "inf-ui-settings-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// **The three clauses, each measured.**
    #[test]
    fn absent_is_the_default_corrupt_is_an_error_and_newer_is_refused() {
        let dir = tmp();
        // Absent.
        assert_eq!(
            GameSettings::load_or_default(&dir).unwrap(),
            GameSettings::default()
        );

        // Corrupt: an error, and the bytes are still there afterwards.
        let path = settings_path(&dir);
        std::fs::write(&path, "this = = not toml").unwrap();
        let err = GameSettings::load_or_default(&dir).unwrap_err();
        assert!(err.contains("left untouched"), "{err}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "this = = not toml",
            "a corrupt file was rewritten, which destroys whatever the player was editing"
        );

        // Newer: refused by name, not downgraded.
        std::fs::write(&path, "schema_version = 99\n").unwrap();
        let err = GameSettings::load_or_default(&dir).unwrap_err();
        assert!(err.contains("newer build"), "{err}");
        assert!(std::fs::read_to_string(&path).unwrap().contains("99"));

        // …and a good file round-trips.
        let mut s = GameSettings {
            master_volume: 0.4,
            ..GameSettings::default()
        };
        s.bindings.insert("crouch".into(), "KeyB".into());
        s.save(&dir).unwrap();
        assert_eq!(GameSettings::load_or_default(&dir).unwrap(), s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **Every numeric door, hostile input first** — and idempotence, because a
    /// normalize that moved a value the second time would make "what is saved"
    /// depend on how many times it was saved.
    #[test]
    fn nan_and_out_of_range_numbers_are_guarded_at_the_door() {
        let d = GameSettings::default();
        let mut s = GameSettings {
            window_mode: "hologram".into(),
            render_tier: "ultra".into(),
            resolution_w: 1,
            resolution_h: 999_999,
            master_volume: f32::NAN,
            sfx_volume: 12.0,
            music_volume: -3.0,
            look_sensitivity: f32::INFINITY,
            press_threshold_ms: 0,
            ..GameSettings::default()
        };
        s.bindings.insert("  ".into(), "KeyQ".into());
        s.normalize();

        // A NAME this build does not know reads as the default, not as a crash
        // and not as a blank.
        assert_eq!(s.window_mode, WINDOW_MODES[0]);
        assert_eq!(s.render_tier, "auto");
        // Finite-out-of-range CLAMPS.
        assert_eq!(s.resolution_w, RESOLUTION_RANGE.0);
        assert_eq!(s.resolution_h, RESOLUTION_RANGE.1);
        assert_eq!(s.sfx_volume, VOLUME_RANGE.1);
        assert_eq!(s.music_volume, VOLUME_RANGE.0);
        assert_eq!(s.press_threshold_ms, PRESS_THRESHOLD_RANGE_MS.0);
        // Non-finite takes the DEFAULT: an infinity is not "very sensitive".
        assert_eq!(s.master_volume, d.master_volume);
        assert_eq!(s.look_sensitivity, d.look_sensitivity);
        assert!(!s.bindings.contains_key("  "));

        // Idempotent.
        let once = s.clone();
        s.normalize();
        assert_eq!(s, once);
    }

    /// A hostile file is refused rather than crashing the boot.
    #[test]
    fn hostile_settings_files_are_refused_and_left_alone() {
        let dir = tmp();
        let path = settings_path(&dir);
        let hostile = [
            "[".to_string(),
            "schema_version = \"one\"".to_string(),
            "resolution_w = -5".to_string(),
            "bindings = 3".to_string(),
            format!("window_mode = \"{}\"", "x".repeat(1_000_000)),
            "[bindings]\ncrouch = 12".to_string(),
        ];
        for text in hostile {
            std::fs::write(&path, &text).unwrap();
            // Either it decodes (and is then guarded) or it is refused; what it
            // may never do is panic or rewrite the file.
            let before = std::fs::read(&path).unwrap();
            let _ = GameSettings::load_or_default(&dir);
            assert_eq!(std::fs::read(&path).unwrap(), before);
        }
        // The one that DOES decode is guarded rather than trusted: a million-'x'
        // window mode is a name this build does not know.
        std::fs::write(
            &path,
            format!("window_mode = \"{}\"", "x".repeat(1_000_000)),
        )
        .unwrap();
        let s = GameSettings::load_or_default(&dir).unwrap();
        assert_eq!(s.window_mode, WINDOW_MODES[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The threshold's two faces agree, and the band is the movement model's.
    #[test]
    fn the_press_threshold_reads_as_seconds_inside_the_models_own_band() {
        let s = GameSettings::default();
        assert!((s.press_threshold_s() - 0.25).abs() < 1e-12);
        // The two ends are the millisecond face of the model's band. Restated
        // rather than shared because `inf-ui` must not depend on `inf-ecs`; the
        // one crate that names both holds them together with an arm.
        assert_eq!(PRESS_THRESHOLD_RANGE_MS, (50, 2000));
    }
}
