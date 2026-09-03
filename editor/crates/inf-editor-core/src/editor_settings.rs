//! Application-level editor settings — the file behind Edit ▸ Editor
//! Preferences… (Wave E, batch A).
//!
//! The sibling of [`crate::project_settings`], one ring up in scope: those are
//! *per project* and live under the content root, these are *per user* and live
//! beside the layout presets in the app config directory. The doctrine is the
//! same one and is copied deliberately rather than invented:
//!
//! * **absent is the default, corrupt is an error** (C4-38) — a settings file
//!   that cannot be parsed is never silently replaced by defaults, and the bytes
//!   on disk are left untouched so the user can repair them;
//! * **atomic write** (C4-24) via `inf_asset::write_atomically`;
//! * **the caller passes the directory** — Ring 1 never names `app_config_dir`
//!   (the `LayoutStore` precedent, `layouts.rs`);
//! * **every numeric field is guarded at the door** — `normalize()` is total and
//!   NaN-refusing, so no non-finite value can reach a camera, a snap step or a
//!   timer. A settings file is a text file a human can edit.
//!
//! Before this module the editor's preferences were five ad-hoc `localStorage`
//! keys with no schema, no version and no corruption law. Four of them are
//! folded in here (`theme_id`, `tour_seen`, `snap_3d`, `foliage`); the fifth,
//! `bp:debug:name:*`, is deliberately NOT a preference — it is per-document
//! breakpoint state keyed by a blueprint's name and belongs with the document.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::ipc::{FoliageSettingsDto, Snap3DDto};

/// Current on-disk schema version. Bump + extend [`EditorSettings::migrate`].
pub const EDITOR_SETTINGS_VERSION: u32 = 1;

/// Default theme id (mirrors `lib/theme.ts`'s `DEFAULT_THEME_ID`).
pub const DEFAULT_THEME_ID: &str = "infinity-dark";
/// Default crash-recovery autosave period, seconds (the P3.5.4 hard-coded 5 s).
pub const DEFAULT_AUTOSAVE_INTERVAL_S: f32 = 5.0;
/// Autosave period bounds, seconds.
pub const AUTOSAVE_INTERVAL_RANGE_S: (f32, f32) = (1.0, 3600.0);
/// Default editor flycam speed, m/s (mirrors `EditorCamera::default`).
pub const DEFAULT_FLY_SPEED_MPS: f32 = 8.0;
/// Flycam speed bounds, m/s.
///
/// **These are the CAMERA's own bounds, not a wider range of our own** (Wave E
/// audit, A2). `EditorCamera` clamps `fly_speed` to
/// `inf_viewport::camera::{FLY_SPEED_MIN, FLY_SPEED_MAX}` on every write, so a
/// file allowed to hold 1 000 m/s showed 1 000 in Preferences and flew at 250:
/// the setting and the thing it sets, silently disagreeing, with the round trip
/// through `editor_settings_set` "confirming" the number that would never take
/// effect. Ring 1 cannot name `inf-viewport` (that crate depends on this one),
/// so the pair is repeated here and PINNED from the other side by
/// `inf_viewport::camera`'s `the_preference_range_is_the_cameras_range`.
pub const FLY_SPEED_RANGE_MPS: (f32, f32) = (0.2, 250.0);
/// Default mouse-look sensitivity **multiplier** over the viewport's built-in
/// radians-per-pixel constants (1.0 = the shipped feel, unchanged).
pub const DEFAULT_LOOK_SENSITIVITY: f32 = 1.0;
/// Look-sensitivity multiplier bounds.
pub const LOOK_SENSITIVITY_RANGE: (f32, f32) = (0.05, 10.0);
/// Default right-button click-vs-drag travel threshold, physical pixels.
pub const DEFAULT_RMB_CLICK_TRAVEL_PX: f32 = 4.0;
/// Right-button travel-threshold bounds, physical pixels.
pub const RMB_CLICK_TRAVEL_RANGE_PX: (f32, f32) = (0.0, 64.0);
/// Default right-button click-vs-drag time threshold, milliseconds.
pub const DEFAULT_RMB_CLICK_MS: u32 = 250;
/// Right-button time-threshold bounds, milliseconds.
pub const RMB_CLICK_MS_RANGE: (u32, u32) = (16, 2000);

fn default_theme_id() -> String {
    DEFAULT_THEME_ID.to_string()
}
fn default_version() -> u32 {
    EDITOR_SETTINGS_VERSION
}
fn default_autosave() -> f32 {
    DEFAULT_AUTOSAVE_INTERVAL_S
}
fn default_fly_speed() -> f32 {
    DEFAULT_FLY_SPEED_MPS
}
fn default_look_sensitivity() -> f32 {
    DEFAULT_LOOK_SENSITIVITY
}
fn default_rmb_travel() -> f32 {
    DEFAULT_RMB_CLICK_TRAVEL_PX
}
fn default_rmb_ms() -> u32 {
    DEFAULT_RMB_CLICK_MS
}
fn default_gis_max_entities() -> u32 {
    inf_gis::DEFAULT_MAX_ENTITIES as u32
}
fn default_snap_3d() -> Snap3DDto {
    Snap3DDto {
        translate: 1.0,
        rotate_deg: 15.0,
        scale: 0.1,
        always_on: false,
    }
}
fn default_foliage() -> FoliageSettingsDto {
    FoliageSettingsDto {
        radius: 3.0,
        density: 0.4,
        erase: false,
        kind: 0,
        scale_jitter: 0.2,
        align_to_normal: false,
        seed: 1,
    }
}

/// One row of the **game's** binding table, as the preferences panel renders it
/// (island wave I5).
///
/// A projection of `inf_ui::bindings::table`, not a second table: the in-game
/// dialog and this panel show the same rows in the same order from the same
/// source. A table restated in TypeScript is the "two copies across a language
/// boundary" defect `inf_input::default_map`'s own note records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct GameBindingRowDto {
    /// The stable row id a settings file stores.
    pub id: String,
    /// What a player reads.
    pub label: String,
    /// What it is bound to; empty means unbound.
    pub token: String,
    /// Whether this row has been moved off the shipped table.
    pub overridden: bool,
    /// Whether the control's consumer exists yet.
    pub wired: bool,
}

/// What an edit to the game's bindings decided (island wave I5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct GameBindingApplyDto {
    /// The override set after the edit — **unchanged** when a conflict blocked
    /// it, so a caller that ignores `conflicts` cannot silently steal a key.
    pub overrides: BTreeMap<String, String>,
    /// The row labels that already own the token, when one did. Labels rather
    /// than ids, because this is what a dialog puts in a sentence.
    pub conflicts: Vec<String>,
}

/// **The game's binding table**, resolved against `overrides`.
pub fn game_binding_table(overrides: &BTreeMap<String, String>) -> Vec<GameBindingRowDto> {
    inf_ui::bindings::table(&inf_input::default_map(), overrides)
        .into_iter()
        .map(|r| GameBindingRowDto {
            id: r.id,
            label: r.label,
            token: r.token,
            overridden: r.overridden,
            wired: r.wired,
        })
        .collect()
}

/// **Bind `token` to `row_id`**, through `inf_ui::bindings::apply_row` — the same
/// rule the in-game dialog's capture uses.
///
/// `swap` is the player's answer to the conflict the first call reports.
pub fn game_binding_apply(
    overrides: &BTreeMap<String, String>,
    row_id: &str,
    token: &str,
    swap: bool,
) -> GameBindingApplyDto {
    let base = inf_input::default_map();
    let out = inf_ui::bindings::apply_row(&base, overrides, row_id, token, swap);
    // Ids are what the map stores and labels are what a person reads; the panel
    // asks a question in words.
    let rows = inf_ui::bindings::rows();
    let label_of = |id: &String| {
        rows.iter()
            .find(|r| &r.id == id)
            .map(|r| r.label.to_string())
            .unwrap_or_else(|| id.clone())
    };
    GameBindingApplyDto {
        conflicts: out.conflicts.iter().map(label_of).collect(),
        overrides: out.overrides,
    }
}

/// Application-level (per-user) editor settings.
///
/// This type is **also the wire type**: `ipc.rs` re-exports it and the ts-rs
/// harness roots it, so there is exactly one definition and a Dto mirror cannot
/// drift out of step with the file format.
///
/// Field order is load-bearing for TOML: every scalar comes before every table,
/// because `toml` refuses to emit a value after a table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct EditorSettings {
    /// On-disk schema version. Stamped by [`Self::normalize`]; a file written by
    /// a NEWER editor is refused rather than downgraded.
    #[serde(default = "default_version")]
    pub schema_version: u32,
    /// Selected theme id (`lib/theme.ts`). Unknown ids fall back in the shell.
    #[serde(default = "default_theme_id")]
    pub theme_id: String,
    /// Whether the first-run tour has been seen/dismissed.
    #[serde(default)]
    pub tour_seen: bool,
    /// Crash-recovery autosave period, seconds.
    #[serde(default = "default_autosave")]
    pub autosave_interval_s: f32,
    /// Editor flycam speed at session start, m/s (the wheel still scales it).
    #[serde(default = "default_fly_speed")]
    pub camera_fly_speed_mps: f32,
    /// Mouse-look sensitivity multiplier (1.0 = the shipped feel).
    #[serde(default = "default_look_sensitivity")]
    pub camera_look_sensitivity: f32,
    /// A right-button gesture that travels less than this (physical px) **and**
    /// lasts less than [`Self::rmb_click_ms`] is a click (context menu); longer
    /// or further is a flycam drag.
    #[serde(default = "default_rmb_travel")]
    pub rmb_click_travel_px: f32,
    /// The time half of the same discrimination, milliseconds.
    #[serde(default = "default_rmb_ms")]
    pub rmb_click_ms: u32,
    /// **The GIS vector-import entity cap** (IB-14), remembered per user.
    ///
    /// The certification's complaint about this number was that it was
    /// "reported and never silent, which is the right design — but with no
    /// wizard there is nowhere for an author to raise it". This is the
    /// somewhere: the wizard opens on it, `inf_gis::DEFAULT_MAX_ENTITIES` is
    /// what it starts at, and an author who raises it once does not raise it
    /// again on the next layer.
    ///
    /// Clamped to `1..=`[`inf_gis::ISLAND_MAX_ENTITIES`] by
    /// [`Self::normalize`], because a settings file is a text file a human can
    /// edit and a cap of zero imports nothing while looking like a preference.
    #[serde(default = "default_gis_max_entities")]
    pub gis_max_entities: u32,
    /// **The project the application opens when it is launched with none**
    /// (wave CERT1) — rung 2 of `inf_project::boot::resolve`.
    ///
    /// An absolute project root, or **empty for "not pinned"**. A `String`
    /// rather than an `Option<String>` because this file is TOML and TOML has no
    /// `None` to write; the alternative is a `skip_serializing_if`, which is the
    /// attribute this repository has been bitten by three times and which would
    /// make an absent key and an empty key two different states for a reader to
    /// hold.
    ///
    /// **Written by every successful project open**, so the plain meaning is
    /// *the last project you opened*. Setting it by hand pins one deliberately,
    /// and a pin that names a project the author has since deleted is skipped
    /// rather than fatal — `inf_project::boot` falls through to the showcase.
    #[serde(default)]
    pub boot_project: String,
    /// 3D gizmo snap increments (was `inf.viewport.snap3d` in localStorage).
    #[serde(default = "default_snap_3d")]
    pub snap_3d: Snap3DDto,
    /// Foliage brush configuration (was `inf.viewport.foliage`).
    #[serde(default = "default_foliage")]
    pub foliage: FoliageSettingsDto,
    /// Keybinding OVERRIDES, chord → command id. An empty command id means
    /// "this default chord is unbound". Defaults are not written here; only what
    /// the user changed, so a new default chord reaches an existing user.
    #[serde(default)]
    pub keybindings: BTreeMap<String, String>,
    /// **The GAME's binding overrides**, row id → source token (island wave I5).
    ///
    /// The same map, in the same format, that a shipped player writes into its
    /// own settings file — `inf_ui::bindings`' row ids and name tokens, only
    /// what differs from the shipped table. It lives here so an author can set a
    /// project's controls from the editor's preferences and have Simulate and a
    /// windowed PIE preview honour them, which is the second half of "rebinding
    /// like UE5, in-game AND in the editor".
    ///
    /// **A different map from [`keybindings`](Self::keybindings)**, and
    /// deliberately: that one is the *editor's* chords (Ctrl+Shift+P opens the
    /// palette) and this one is the *game's* controls (C crouches). Folding them
    /// together would put a chord and a key code in one namespace and make
    /// "unbind" mean two things.
    #[serde(default)]
    pub game_bindings: BTreeMap<String, String>,
}

impl Default for EditorSettings {
    fn default() -> Self {
        Self {
            schema_version: EDITOR_SETTINGS_VERSION,
            theme_id: default_theme_id(),
            tour_seen: false,
            autosave_interval_s: DEFAULT_AUTOSAVE_INTERVAL_S,
            camera_fly_speed_mps: DEFAULT_FLY_SPEED_MPS,
            camera_look_sensitivity: DEFAULT_LOOK_SENSITIVITY,
            rmb_click_travel_px: DEFAULT_RMB_CLICK_TRAVEL_PX,
            rmb_click_ms: DEFAULT_RMB_CLICK_MS,
            gis_max_entities: default_gis_max_entities(),
            boot_project: String::new(),
            snap_3d: default_snap_3d(),
            foliage: default_foliage(),
            keybindings: BTreeMap::new(),
            game_bindings: BTreeMap::new(),
        }
    }
}

/// `<dir>/editor-settings.toml`.
fn settings_path(dir: &Path) -> PathBuf {
    dir.join("editor-settings.toml")
}

/// Clamp a scalar into `range`, substituting `fallback` for a non-finite value.
///
/// **NaN is not "out of range", it is "not a number"** — every comparison
/// against it is false, so a plain `clamp` would let it through untouched. This
/// is the door for every f32 in the file.
fn guard_f32(v: f32, range: (f32, f32), fallback: f32) -> f32 {
    if !v.is_finite() {
        return fallback;
    }
    v.clamp(range.0, range.1)
}

/// The f64 twin (the foliage DTO is f64).
fn guard_f64(v: f64, range: (f64, f64), fallback: f64) -> f64 {
    if !v.is_finite() {
        return fallback;
    }
    v.clamp(range.0, range.1)
}

impl EditorSettings {
    /// Migrate a decoded document to the current schema.
    ///
    /// Older versions are upgraded field-by-field (there is only v1 today, so
    /// this is the shape and not yet the work). A **newer** version is refused:
    /// a downgrade would write a lossy file back over a richer one, and the
    /// user's own preferences are exactly the thing we must not quietly eat.
    pub fn migrate(&mut self) -> Result<(), String> {
        if self.schema_version > EDITOR_SETTINGS_VERSION {
            return Err(format!(
                "editor settings were written by a newer Infini Engine \
                 (schema v{}, this build understands v{EDITOR_SETTINGS_VERSION}); \
                 the file is left untouched — update the editor or move it aside",
                self.schema_version
            ));
        }
        self.schema_version = EDITOR_SETTINGS_VERSION;
        Ok(())
    }

    /// Load the settings from `dir`, or the default when none has been saved.
    ///
    /// **Absent is the default; unreadable is an error** (C4-38) — and the
    /// unreadable file is left byte-identical on disk.
    pub fn load_or_default(dir: &Path) -> Result<Self, String> {
        let path = settings_path(dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("read {}: {e}", path.display())),
        };
        let mut settings: Self = toml::from_str(&text).map_err(|e| {
            format!(
                "{} exists but cannot be read ({e}); it is left untouched rather than \
                 replaced by defaults — repair or delete it",
                path.display()
            )
        })?;
        settings.migrate()?;
        settings.normalize();
        Ok(settings)
    }

    /// Total, idempotent guard: every scalar finite and in range, the version
    /// stamped, and empty keybinding chords dropped.
    pub fn normalize(&mut self) {
        self.schema_version = EDITOR_SETTINGS_VERSION;
        if self.theme_id.trim().is_empty() {
            self.theme_id = default_theme_id();
        }
        // A hand-edited path with stray whitespace around it is the same pin as
        // one without, and `inf_project::boot::resolve` trims for the same
        // reason — but normalizing here is what makes the file itself agree with
        // what the resolution does, rather than leaving two places that trim.
        if self.boot_project.trim() != self.boot_project {
            self.boot_project = self.boot_project.trim().to_string();
        }
        self.autosave_interval_s = guard_f32(
            self.autosave_interval_s,
            AUTOSAVE_INTERVAL_RANGE_S,
            DEFAULT_AUTOSAVE_INTERVAL_S,
        );
        self.camera_fly_speed_mps = guard_f32(
            self.camera_fly_speed_mps,
            FLY_SPEED_RANGE_MPS,
            DEFAULT_FLY_SPEED_MPS,
        );
        self.camera_look_sensitivity = guard_f32(
            self.camera_look_sensitivity,
            LOOK_SENSITIVITY_RANGE,
            DEFAULT_LOOK_SENSITIVITY,
        );
        self.rmb_click_travel_px = guard_f32(
            self.rmb_click_travel_px,
            RMB_CLICK_TRAVEL_RANGE_PX,
            DEFAULT_RMB_CLICK_TRAVEL_PX,
        );
        self.rmb_click_ms = self
            .rmb_click_ms
            .clamp(RMB_CLICK_MS_RANGE.0, RMB_CLICK_MS_RANGE.1);
        // A cap of zero imports nothing while looking like a preference, and a
        // cap past the island ceiling is a promise the document cannot keep.
        self.gis_max_entities = self
            .gis_max_entities
            .clamp(1, inf_gis::ISLAND_MAX_ENTITIES as u32);

        // Snap steps divide, so zero and NaN are both fatal downstream.
        let d = default_snap_3d();
        self.snap_3d.translate = guard_f32(self.snap_3d.translate, (1e-4, 1e6), d.translate);
        self.snap_3d.rotate_deg = guard_f32(self.snap_3d.rotate_deg, (1e-3, 360.0), d.rotate_deg);
        self.snap_3d.scale = guard_f32(self.snap_3d.scale, (1e-4, 1e3), d.scale);

        let f = default_foliage();
        self.foliage.radius = guard_f64(self.foliage.radius, (0.05, 4096.0), f.radius);
        self.foliage.density = guard_f64(self.foliage.density, (0.0, 1000.0), f.density);
        self.foliage.scale_jitter =
            guard_f64(self.foliage.scale_jitter, (0.0, 1.0), f.scale_jitter);

        self.keybindings.retain(|chord, _| !chord.trim().is_empty());
        // The same guard, for the same reason: a blank row id names nothing and
        // could never be matched against the game's binding table.
        self.game_bindings.retain(|row, _| !row.trim().is_empty());
    }

    /// Deterministic TOML for the (normalized) settings.
    pub fn to_toml(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.normalize();
        toml::to_string_pretty(&copy).map_err(|e| e.to_string())
    }

    /// Write the settings into `dir` (creating it if needed), atomically.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = settings_path(dir);
        inf_asset::write_atomically(&path, self.to_toml()?).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_shipped_feel() {
        let s = EditorSettings::default();
        assert_eq!(s.theme_id, "infinity-dark");
        assert_eq!(s.autosave_interval_s, 5.0);
        assert_eq!(s.camera_fly_speed_mps, 8.0);
        assert_eq!(s.camera_look_sensitivity, 1.0);
        assert_eq!(s.rmb_click_travel_px, 4.0);
        assert_eq!(s.rmb_click_ms, 250);
        assert!(!s.tour_seen);
        assert!(s.keybindings.is_empty());
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            EditorSettings::load_or_default(dir.path()).unwrap(),
            EditorSettings::default()
        );
    }

    #[test]
    fn save_load_round_trips_every_field_and_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = EditorSettings {
            schema_version: EDITOR_SETTINGS_VERSION,
            theme_id: "graphite".into(),
            tour_seen: true,
            autosave_interval_s: 12.5,
            camera_fly_speed_mps: 42.0,
            camera_look_sensitivity: 2.5,
            rmb_click_travel_px: 7.0,
            rmb_click_ms: 300,
            gis_max_entities: 60_000,
            boot_project: "C:/somewhere/island-build/project".into(),
            snap_3d: Snap3DDto {
                translate: 0.25,
                rotate_deg: 5.0,
                scale: 0.05,
                always_on: true,
            },
            foliage: FoliageSettingsDto {
                radius: 9.0,
                density: 0.75,
                erase: true,
                kind: 3,
                scale_jitter: 0.4,
                align_to_normal: true,
                seed: 77,
            },
            keybindings: BTreeMap::from([("Ctrl+Alt+P".to_string(), "tools.profiler".to_string())]),
            game_bindings: BTreeMap::from([("crouch".to_string(), "KeyB".to_string())]),
        };
        s.normalize();
        s.save(dir.path()).unwrap();
        assert_eq!(s.to_toml().unwrap(), s.to_toml().unwrap());
        // Field-by-field: a mirror that dropped a field would still be `==` if
        // the dropped field happened to hold its default, so every value above
        // is deliberately non-default.
        assert_eq!(EditorSettings::load_or_default(dir.path()).unwrap(), s);
    }

    /// **Corrupt is refused; absent is the default** (C4-38), and the corrupt
    /// file survives byte-identical.
    ///
    /// Mutation-verified: replacing `load_or_default`'s `?` on the `toml::from_str`
    /// with `.unwrap_or_default()` — the exact "be forgiving" change someone will
    /// propose — turns this test red on both assertions.
    #[test]
    fn corrupt_editor_settings_are_refused_while_absent_ones_are_the_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            EditorSettings::load_or_default(dir.path()).unwrap(),
            EditorSettings::default(),
            "absent must read as the default"
        );

        EditorSettings::default().save(dir.path()).unwrap();
        let path = settings_path(dir.path());
        let damaged = b"autosave_interval_s = \"every five seconds\"\n".to_vec();
        std::fs::write(&path, &damaged).unwrap();

        let err = EditorSettings::load_or_default(dir.path())
            .expect_err("damaged settings must not read as the default ones");
        assert!(err.contains("cannot be read"), "{err}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            damaged,
            "the damaged file must be left untouched"
        );
    }

    /// A file from a newer editor is refused by name, not silently downgraded.
    #[test]
    fn a_newer_schema_is_refused_rather_than_downgraded() {
        let dir = tempfile::tempdir().unwrap();
        let path = settings_path(dir.path());
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(
            &path,
            format!("schema_version = {}\n", EDITOR_SETTINGS_VERSION + 1),
        )
        .unwrap();
        let err = EditorSettings::load_or_default(dir.path()).expect_err("newer must be refused");
        assert!(err.contains("newer Infini Engine"), "{err}");
    }

    /// Every numeric door: NaN and infinity fall back to the default, and
    /// out-of-range values clamp. A NaN autosave interval would arm a timer with
    /// `NaN` ms; a NaN snap step divides the gizmo into nothing.
    #[test]
    fn nan_and_out_of_range_numbers_are_guarded_at_the_door() {
        let mut s = EditorSettings {
            autosave_interval_s: f32::NAN,
            camera_fly_speed_mps: f32::INFINITY,
            camera_look_sensitivity: -1.0,
            rmb_click_travel_px: f32::NEG_INFINITY,
            rmb_click_ms: 0,
            snap_3d: Snap3DDto {
                translate: f32::NAN,
                rotate_deg: 0.0,
                scale: f32::NAN,
                always_on: false,
            },
            foliage: FoliageSettingsDto {
                radius: f64::NAN,
                density: -5.0,
                scale_jitter: f64::INFINITY,
                ..default_foliage()
            },
            ..Default::default()
        };
        s.normalize();
        // NON-FINITE falls back to the DEFAULT (an infinity is not "very large",
        // it is "not a number I can trust"); a FINITE out-of-range value clamps.
        assert_eq!(s.autosave_interval_s, DEFAULT_AUTOSAVE_INTERVAL_S);
        assert_eq!(s.camera_fly_speed_mps, DEFAULT_FLY_SPEED_MPS);
        assert_eq!(s.camera_look_sensitivity, LOOK_SENSITIVITY_RANGE.0);
        assert_eq!(s.rmb_click_travel_px, DEFAULT_RMB_CLICK_TRAVEL_PX);
        assert_eq!(s.rmb_click_ms, RMB_CLICK_MS_RANGE.0);
        assert_eq!(s.snap_3d.translate, default_snap_3d().translate);
        assert_eq!(s.snap_3d.rotate_deg, 1e-3);
        assert_eq!(s.snap_3d.scale, default_snap_3d().scale);
        assert_eq!(s.foliage.radius, default_foliage().radius);
        assert_eq!(s.foliage.density, 0.0);
        assert_eq!(s.foliage.scale_jitter, default_foliage().scale_jitter);
        // Idempotent: normalizing a normalized document changes nothing.
        let once = s.clone();
        s.normalize();
        assert_eq!(s, once);

        // The finite-but-absurd half of the same door: these CLAMP, and the
        // clamp is what keeps a hand-edited "fly at 1e9 m/s" usable.
        let mut big = EditorSettings {
            autosave_interval_s: 1e9,
            camera_fly_speed_mps: 1e9,
            camera_look_sensitivity: 1e9,
            rmb_click_travel_px: 1e9,
            rmb_click_ms: u32::MAX,
            ..Default::default()
        };
        big.normalize();
        assert_eq!(big.autosave_interval_s, AUTOSAVE_INTERVAL_RANGE_S.1);
        assert_eq!(big.camera_fly_speed_mps, FLY_SPEED_RANGE_MPS.1);
        assert_eq!(big.camera_look_sensitivity, LOOK_SENSITIVITY_RANGE.1);
        assert_eq!(big.rmb_click_travel_px, RMB_CLICK_TRAVEL_RANGE_PX.1);
        assert_eq!(big.rmb_click_ms, RMB_CLICK_MS_RANGE.1);
    }

    /// A partial file (the realistic hand-edited case, and the shape every future
    /// schema addition takes) fills the rest from defaults rather than refusing.
    #[test]
    fn a_partial_file_keeps_what_it_says_and_defaults_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(settings_path(dir.path()), "theme_id = \"midnight\"\n").unwrap();
        let s = EditorSettings::load_or_default(dir.path()).unwrap();
        assert_eq!(s.theme_id, "midnight");
        assert_eq!(s.autosave_interval_s, DEFAULT_AUTOSAVE_INTERVAL_S);
        assert_eq!(s.snap_3d, default_snap_3d());
    }

    /// **A settings file is a text file a human — or something else — can
    /// write** (Wave E audit). It is the wave's one new attack surface, so the
    /// hostile shapes are asserted rather than assumed: each must be REFUSED as
    /// a value, must not panic or hang, and must leave the bytes on disk exactly
    /// as they were. "Refused" is the load-bearing half — the alternative is
    /// that a malformed file silently becomes the defaults and eats the user's
    /// theme and keybindings.
    #[test]
    fn hostile_settings_files_are_refused_and_left_alone() {
        let cases: Vec<(&str, String)> = vec![
            // Wrong type per field, on each of the three kinds of field.
            (
                "string where a number goes",
                "autosave_interval_s = \"soon\"\n".into(),
            ),
            ("number where a bool goes", "tour_seen = 3\n".into()),
            ("table where a scalar goes", "theme_id = { a = 1 }\n".into()),
            ("scalar where a table goes", "snap_3d = 7\n".into()),
            ("array where a map goes", "keybindings = [1, 2]\n".into()),
            // Duplicate keys — TOML forbids them, and "last one wins" would make
            // the file's meaning depend on the parser.
            (
                "duplicate keys",
                "theme_id = \"a\"\ntheme_id = \"b\"\n".into(),
            ),
            // Deep nesting: the parser must return an error, not recurse until
            // the stack ends. 512 levels is far past anything legitimate.
            (
                "deeply nested inline arrays",
                format!("theme_id = {}{}\n", "[".repeat(512), "]".repeat(512)),
            ),
            // A huge scalar in a numeric field: refused for its TYPE, before
            // anything tries to size it.
            (
                "a megabyte-long string in a numeric field",
                format!("autosave_interval_s = \"{}\"\n", "x".repeat(1 << 20)),
            ),
            ("not TOML at all", "\u{0}\u{1}\u{2}not toml".into()),
        ];

        for (label, body) in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = settings_path(dir.path());
            std::fs::write(&path, body.as_bytes()).unwrap();
            let err = EditorSettings::load_or_default(dir.path())
                .err()
                .unwrap_or_else(|| panic!("{label}: must be refused, not read as defaults"));
            assert!(!err.trim().is_empty(), "{label}: an empty refusal");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                body.as_bytes(),
                "{label}: the file must be left byte-identical"
            );
        }

        // The one hostile shape that is NOT an error, and must not be: a huge
        // but well-typed string in the theme field. It is refused later, by the
        // theme lookup falling back — losing the file over it would be worse.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            settings_path(dir.path()),
            format!("theme_id = \"{}\"\n", "z".repeat(1 << 16)),
        )
        .unwrap();
        let s = EditorSettings::load_or_default(dir.path()).expect("a long theme id is legal TOML");
        assert_eq!(s.theme_id.len(), 1 << 16);
        assert_eq!(s.autosave_interval_s, DEFAULT_AUTOSAVE_INTERVAL_S);
    }

    #[test]
    fn empty_theme_and_blank_chords_are_dropped() {
        let mut s = EditorSettings {
            theme_id: "   ".into(),
            keybindings: BTreeMap::from([
                ("".to_string(), "edit.undo".to_string()),
                ("Ctrl+Q".to_string(), "app.exit".to_string()),
            ]),
            ..Default::default()
        };
        s.normalize();
        assert_eq!(s.theme_id, DEFAULT_THEME_ID);
        assert_eq!(s.keybindings.len(), 1);
        assert_eq!(s.keybindings.get("Ctrl+Q").unwrap(), "app.exit");
    }
}
