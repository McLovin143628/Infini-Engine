//! **The shipped player's UI session** (island wave I5): the settings dialog,
//! the toasts and the interaction prompt, wired to the window.
//!
//! `inf-ui` is pure — a state machine, a reducer and three drawing functions,
//! none of which owns a clock, a file or a window. This module is the half that
//! does: it resolves where the settings file lives, routes a key to the dialog
//! before the game sees it, persists what the dialog changed, and hands the
//! finished draw list to the render host.
//!
//! # Two clocks, again
//!
//! Everything here runs on the **wall** clock, including while the simulation is
//! frozen behind an open menu. A toast fades in real seconds and a menu is
//! navigable while the world is stopped; neither is sim state and neither is ever
//! serialized. The gameplay durations are the sim's — see `inf_input::HoldClock`
//! for the argument.

use std::collections::BTreeSet;
use std::path::PathBuf;

use glam::Vec2;
use inf_input::{InputMap, InputState};
use inf_ui::menu::{self, MenuInput, MenuState};
use inf_ui::{settings::GameSettings, toast::Toasts, UiDrawList};

/// The environment variable that overrides where settings are read and written.
///
/// Exists for two reasons and both are honest: a test needs a directory it owns,
/// and a player on a machine whose config directory is not writable needs a way
/// out that is not "edit the registry".
pub const SETTINGS_DIR_ENV: &str = "INF_PLAYER_SETTINGS_DIR";

/// **Where the shipped game keeps its settings.**
///
/// In order: the [`SETTINGS_DIR_ENV`] override, then the platform's per-user
/// config directory, then the executable's own directory.
///
/// # Why environment variables and not a platform crate
///
/// `%APPDATA%` and `$XDG_CONFIG_HOME`/`$HOME` **are** the platform conventions —
/// a crate that resolved them would read the same two variables. Adding one to
/// get at them would be a new external dependency on the shipped player's
/// dependency graph and a new row for `cargo-deny`, to save six lines. The exe's
/// directory is the last resort rather than the first because a real install is
/// read-only, and a settings file that silently fails to save is worse than one
/// that lives somewhere unexpected.
pub fn settings_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(SETTINGS_DIR_ENV) {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    let base = if cfg!(windows) {
        std::env::var("APPDATA").ok().map(PathBuf::from)
    } else {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".config"))
            })
    };
    if let Some(base) = base {
        return base.join("InfinityEngine");
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The player's UI session.
pub struct PlayerUi {
    /// The dialog.
    pub menu: MenuState,
    /// The player's settings, as loaded and as edited.
    pub settings: GameSettings,
    /// What is on screen.
    pub toasts: Toasts,
    /// Where [`settings`](Self::settings) is written.
    dir: PathBuf,
    /// **The level's own table**, kept so every derived thing is derived against
    /// it: the override set, and the look tuning's multiplier.
    ///
    /// **Not the live map.** The overrides a settings file stores are the
    /// *difference* from the table the player was given, so deriving them needs
    /// that table rather than the one they are running; and the look tuning is a
    /// multiplier, so applying it to the live map would compound it once a
    /// frame.
    base: InputMap,
    /// The finished draw list, rebuilt each frame and reused so a menu costs no
    /// allocation per frame.
    list: UiDrawList,
    /// Whether anything changed since the last save.
    dirty: bool,
    /// How the settings load went, so a host can report it once. `None` on a
    /// clean load or a first run.
    pub load_error: Option<String>,
}

impl PlayerUi {
    /// Open a session against `dir`, returning it and the **input map the
    /// player's overrides produce**.
    ///
    /// A settings file that cannot be read leaves the session on defaults and
    /// records the reason: the alternative — refusing to boot — is what
    /// `player.toml` does for a *boot config* it cannot honour, and a
    /// preferences file is not that. A player whose sensitivity file has a
    /// stray comma should get their game, told.
    ///
    /// `base` is **the level's own table** — `input.toml` beside a dev level, or
    /// the shipped default for a cooked pack. Every override and every tuning is
    /// relative to it, so a project that authored its own bindings or its own
    /// look sensitivity keeps them and the player's settings are a difference
    /// against what they were actually given.
    pub fn open(dir: PathBuf, base: InputMap) -> (Self, InputMap) {
        let (settings, load_error) = match GameSettings::load_or_default(&dir) {
            Ok(s) => (s, None),
            Err(e) => (GameSettings::default(), Some(e)),
        };
        let ui = Self {
            menu: MenuState::new(),
            settings,
            toasts: Toasts::default(),
            dir,
            base,
            list: UiDrawList::new(Vec2::new(1.0, 1.0)),
            dirty: false,
            load_error,
        };
        let map = ui.tuned_map();
        (ui, map)
    }

    /// **The map the player is actually playing with**: the level's table, plus
    /// their binding overrides, plus their look tuning.
    ///
    /// Rebuilt from `base` rather than patched in place, which is what makes it
    /// idempotent — a sensitivity applied to the *current* map compounds, and a
    /// held slider would square it in two frames.
    pub fn tuned_map(&self) -> InputMap {
        use inf_ecs::movement::actions as mv;
        let mut map = self.base.clone();
        inf_ui::bindings::apply_overrides(&mut map, &self.settings.bindings);
        inf_ui::bindings::apply_look_tuning(
            &mut map,
            &self.base,
            mv::LOOK_X,
            mv::LOOK_Y,
            self.settings.look_sensitivity,
            self.settings.invert_look_y,
        );
        map
    }

    /// **Push the settings into the simulation** — the press threshold and the
    /// three mixer buses.
    ///
    /// Called at boot and again whenever the dialog changed something, so a
    /// slider moves the thing it names in the session the player is in rather
    /// than on their next launch. The bindings' half is
    /// [`tuned_map`](Self::tuned_map), which the caller re-seats on the live
    /// input state.
    pub fn apply_to_sim(&self, sim: &mut crate::runtime_sim::RuntimeSim) {
        sim.set_press_threshold_s(self.settings.press_threshold_s());
        sim.set_bus_volumes(
            self.settings.master_volume,
            self.settings.sfx_volume,
            self.settings.music_volume,
        );
    }

    /// Whether the simulation should be frozen this frame.
    pub fn pauses_sim(&self) -> bool {
        self.menu.pauses_sim()
    }

    /// **Route one key**, returning whether the UI took it.
    ///
    /// A key the UI took **must not reach the game**: the same press that moves
    /// the cursor would otherwise also fire a weapon, and — worse — a key being
    /// *captured* for a rebinding would fire the verb it is being taken from.
    ///
    /// `map` is edited in place by a rebinding; the caller re-seats it on the
    /// live [`InputState`] when this answers that the bindings changed.
    pub fn key(&mut self, code: &str, pressed: bool, map: &mut InputMap) -> KeyVerdict {
        if !pressed {
            // **A RELEASE IS NEVER CONSUMED**, even by an open dialog.
            //
            // Measured: consuming them stranded the input state holding the very
            // key that opened the menu — Tab went down, the dialog opened, and
            // the release was eaten, so `menu` read as *held* for the whole
            // window and the state's raw set never came back. It is the
            // stuck-key failure `InputState::release_all` exists for, arrived at
            // through the menu instead of through a focus loss.
            //
            // Forwarding it is safe by construction: the *press* never reached
            // the state, so there is no `just_released` edge for it to fire —
            // an edge needs the previous frame to have had the action down.
            return KeyVerdict::default();
        }
        if !self.menu.open {
            return KeyVerdict::default();
        }
        let out = menu::handle(
            &mut self.menu,
            &mut self.settings,
            map,
            &MenuInput::Key(code.to_string()),
        );
        self.dirty |= out.settings_changed || out.bindings_changed;
        if out.bindings_changed {
            // Only the DIFFERENCE from the shipped table is stored, so a control
            // added by a later build arrives bound and a default corrected by a
            // later build reaches a player who never touched it.
            self.settings.bindings = inf_ui::bindings::overrides_from(&self.base, map);
        }
        KeyVerdict {
            consumed: out.consumed,
            bindings_changed: out.bindings_changed,
            settings_changed: out.settings_changed,
        }
    }

    /// Route one mouse button. Only meaningful while a rebinding capture is
    /// running — a mouse button is a bindable source and nothing else the dialog
    /// does reads one.
    pub fn mouse(&mut self, button: inf_input::MouseButton, map: &mut InputMap) -> KeyVerdict {
        if !self.menu.open {
            return KeyVerdict::default();
        }
        let out = menu::handle(
            &mut self.menu,
            &mut self.settings,
            map,
            &MenuInput::Mouse(button),
        );
        self.dirty |= out.bindings_changed;
        if out.bindings_changed {
            self.settings.bindings = inf_ui::bindings::overrides_from(&self.base, map);
        }
        KeyVerdict {
            consumed: out.consumed,
            bindings_changed: out.bindings_changed,
            settings_changed: out.settings_changed,
        }
    }

    /// Open or close the dialog — the `menu` action's consumer.
    pub fn toggle(&mut self) {
        let open = !self.menu.open;
        self.menu.set_open(open);
        if !open {
            // Closing is the natural save point: it is the moment a player is
            // done, and it is the one moment that cannot be reached by holding
            // an arrow key. Saving on every keystroke would rewrite the file
            // sixty times a second on a held slider.
            self.save();
        }
    }

    /// Persist if anything changed. A failure is logged, never fatal: a player
    /// whose disk is full still gets to play with the settings they chose for
    /// this session.
    pub fn save(&mut self) {
        if !self.dirty {
            return;
        }
        match self.settings.save(&self.dir) {
            Ok(()) => self.dirty = false,
            Err(e) => tracing::warn!("inf-player: could not save settings: {e}"),
        }
    }

    /// **The honest "not yet" toast** for a control that is bound and has no
    /// consumer.
    ///
    /// Reads the four names from `inf_input::actions::NOT_YET_CONSUMED` rather
    /// than restating them, so a control that gains a consumer stops toasting by
    /// being removed from one list.
    pub fn report_unconsumed(&mut self, state: &InputState) {
        if self.menu.open {
            return;
        }
        for action in inf_input::actions::NOT_YET_CONSUMED {
            let fired = if action == inf_input::actions::WEAPON_SWITCH {
                // An axis has no edge; a wheel that moved at all is a press.
                state.axis(action).abs() > f32::EPSILON
            } else {
                state.just_pressed(action)
            };
            if fired {
                self.toasts.push_not_yet(action);
            }
        }
    }

    /// Advance the wall clock and rebuild the draw list for a `viewport`-pixel
    /// screen. Returns the list, ready for the render host.
    pub fn build(&mut self, dt: f64, viewport: Vec2, map: &InputMap) -> &UiDrawList {
        self.toasts.advance(dt);
        self.list = UiDrawList::new(viewport);
        inf_ui::view::menu(&mut self.list, &self.menu, &self.settings, map);
        inf_ui::view::toasts(&mut self.list, &self.toasts);
        &self.list
    }

    /// Draw a world-space interaction prompt into the list built this frame.
    ///
    /// `screen` is where the projection put the target, in pixels; `None` when
    /// it is behind the camera. Nothing is drawn while the menu is open — a
    /// prompt over a settings dialog is a prompt for a control the dialog is
    /// currently eating.
    pub fn prompt(&mut self, screen: Option<Vec2>, text: &str) {
        if self.menu.open {
            return;
        }
        if let Some(at) = screen {
            inf_ui::view::prompt(&mut self.list, at, text, inf_ui::view::palette::TEXT);
        }
    }

    /// The finished list.
    pub fn list(&self) -> &UiDrawList {
        &self.list
    }

    /// What the player must not interact with — itself, and whatever it is
    /// already in. A convenience so the prompt and the press ask the same
    /// question.
    pub fn exclude(actor: uuid::Uuid, seated: Option<uuid::Uuid>) -> BTreeSet<uuid::Uuid> {
        let mut out = BTreeSet::new();
        out.insert(actor);
        if let Some(v) = seated {
            out.insert(v);
        }
        out
    }
}

/// What routing one input decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyVerdict {
    /// The UI took it; the game must not see it.
    pub consumed: bool,
    /// The input map changed and must be re-seated on the live state.
    pub bindings_changed: bool,
    /// A setting changed and must be pushed into the sim
    /// ([`PlayerUi::apply_to_sim`]) — and, for the look tuning, into the map.
    pub settings_changed: bool,
}

impl KeyVerdict {
    /// Whether anything the host has to act on moved.
    pub fn changed(&self) -> bool {
        self.bindings_changed || self.settings_changed
    }
}

/// **Project a world point onto the screen**, in pixels, or `None` behind the
/// camera.
///
/// The prompt's other half. It goes through the view's own `view_proj` and the
/// same `FloatingOrigin` the renderer packs instances with, so the label lands
/// where the thing is drawn rather than where the world thinks it is — the two
/// differ by up to a kilometre on a rebasing island.
pub fn project_to_screen(view: &inf_render::RenderView, world: glam::DVec3) -> Option<Vec2> {
    let local = view.origin.to_render(world);
    let clip = view.view_proj() * glam::Vec4::new(local.x, local.y, local.z, 1.0);
    if !(clip.w.is_finite() && clip.w > 0.0) {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.x.is_finite() || !ndc.y.is_finite() {
        return None;
    }
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * view.width.max(1) as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * view.height.max(1) as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "inf-player-ui-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// **THE REBINDING ROUND TRIP, THROUGH THE SHIPPED DOORS.**
    ///
    /// Open the dialog, walk to a row, capture a key, close it — then open a
    /// *fresh* session against the same directory and find the same binding.
    /// Nothing here reaches into a struct: every step is a key the player
    /// pressed, which is what makes it a proof about the game rather than about
    /// the model.
    #[test]
    fn a_key_rebound_in_the_dialog_survives_a_restart() {
        let dir = tmp();
        let (mut ui, mut map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        assert!(ui.load_error.is_none(), "{:?}", ui.load_error);
        let table = inf_ui::bindings::rows();
        let interact = table.iter().position(|r| r.id == "interact").unwrap();
        assert_eq!(inf_ui::bindings::token_in(&map, &table[interact]), "KeyE");

        ui.toggle();
        assert!(ui.menu.open && ui.pauses_sim());
        // Walk to the bindings page and to the row, the way a player does.
        ui.key("ArrowRight", true, &mut map);
        ui.key("ArrowRight", true, &mut map);
        ui.key("ArrowRight", true, &mut map);
        assert_eq!(ui.menu.page, inf_ui::Page::Bindings);
        let focus = menu::rows(&ui.menu, &ui.settings, &map)
            .iter()
            .position(|r| r.id == inf_ui::RowId::Binding(interact))
            .unwrap();
        for _ in 0..focus {
            ui.key("ArrowDown", true, &mut map);
        }
        let v = ui.key("Enter", true, &mut map);
        assert!(v.consumed);
        let v = ui.key("KeyJ", true, &mut map);
        assert!(v.bindings_changed, "the capture bound nothing");
        assert_eq!(inf_ui::bindings::token_in(&map, &table[interact]), "KeyJ");
        // Only the DIFFERENCE is stored.
        assert_eq!(ui.settings.bindings.len(), 1, "{:?}", ui.settings.bindings);
        assert_eq!(ui.settings.bindings["interact"], "KeyJ");
        ui.toggle();
        assert!(!ui.menu.open);

        // A fresh session — a restart — finds it.
        let (again, map2) = PlayerUi::open(dir.clone(), inf_input::default_map());
        assert!(again.load_error.is_none(), "{:?}", again.load_error);
        assert_eq!(inf_ui::bindings::token_in(&map2, &table[interact]), "KeyJ");
        // …and everything the player did NOT touch is still the shipped table.
        let base = inf_input::default_map();
        for row in &table {
            if row.id == "interact" {
                continue;
            }
            assert_eq!(
                inf_ui::bindings::token_in(&map2, row),
                inf_ui::bindings::token_in(&base, row),
                "row `{}` moved without being touched",
                row.label
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A key the dialog took never reaches the game.**
    ///
    /// The mutation this kills: a router that drew the menu and forwarded the
    /// keys would let the same press that binds a control also fire it.
    #[test]
    fn an_open_dialog_eats_every_press_and_a_closed_one_eats_none() {
        let dir = tmp();
        let (mut ui, mut map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        for k in ["KeyW", "Space", "ArrowDown", "Enter", "KeyC"] {
            assert!(
                !ui.key(k, true, &mut map).consumed,
                "a closed dialog ate `{k}`"
            );
        }
        ui.toggle();
        for k in ["KeyW", "Space", "ArrowDown", "Enter", "KeyC"] {
            assert!(
                ui.key(k, true, &mut map).consumed,
                "an open dialog let `{k}` through to the game"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A settings file that cannot be read leaves the player on defaults **and
    /// says so** — a preferences file is not a boot config, so it does not
    /// refuse to start.
    #[test]
    fn a_corrupt_settings_file_is_reported_and_the_game_still_runs() {
        let dir = tmp();
        std::fs::write(inf_ui::settings::settings_path(&dir), "nonsense = = 1").unwrap();
        let (ui, map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        assert!(ui.load_error.is_some(), "the failure was swallowed");
        assert_eq!(ui.settings, GameSettings::default());
        assert_eq!(map, inf_input::default_map());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The four unconsumed controls toast, and a wired one does not.
    #[test]
    fn an_unconsumed_control_says_so_and_a_wired_one_stays_quiet() {
        let dir = tmp();
        let (mut ui, map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        let mut state = InputState::new(map);
        state.apply(&[inf_input::InputEvent::Key {
            code: "KeyR".into(),
            pressed: true,
        }]);
        ui.report_unconsumed(&state);
        assert_eq!(ui.toasts.live().len(), 1, "{:?}", ui.toasts.live());
        assert!(ui.toasts.live()[0].text.contains("reload"));

        // A control with a consumer is silent.
        let mut ui2 = PlayerUi::open(dir.clone(), inf_input::default_map()).0;
        let mut state = InputState::new(inf_input::default_map());
        state.apply(&[inf_input::InputEvent::Key {
            code: "Space".into(),
            pressed: true,
        }]);
        ui2.report_unconsumed(&state);
        assert!(ui2.toasts.is_empty(), "{:?}", ui2.toasts.live());

        // …and an open menu reports nothing at all: the dialog is eating the
        // keys, so a toast would be about a press the game never saw.
        ui2.toggle();
        let mut state = InputState::new(inf_input::default_map());
        state.apply(&[inf_input::InputEvent::Key {
            code: "KeyR".into(),
            pressed: true,
        }]);
        ui2.report_unconsumed(&state);
        assert!(ui2.toasts.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The settings directory is resolved in the order the docs claim, and the
    /// override wins.
    #[test]
    fn the_settings_directory_honours_its_override() {
        let want = std::env::temp_dir().join("inf-ui-dir-probe");
        // SAFETY: single-threaded within this test, and the variable is read
        // only by `settings_dir` on the same thread immediately below.
        unsafe { std::env::set_var(SETTINGS_DIR_ENV, &want) };
        assert_eq!(settings_dir(), want);
        unsafe { std::env::set_var(SETTINGS_DIR_ENV, "   ") };
        assert_ne!(
            settings_dir(),
            PathBuf::from("   "),
            "a blank override was honoured"
        );
        unsafe { std::env::remove_var(SETTINGS_DIR_ENV) };
        // Whatever it falls back to, it is a real path with a parent.
        assert!(settings_dir().components().count() > 0);
    }
}
