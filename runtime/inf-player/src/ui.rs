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
use inf_ui::inventory::{self, InventoryState, InventoryVerb, InventoryView};
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
    /// **The inventory panel** (I6). Beside the dialog rather than inside it
    /// because they are different kinds of surface: the dialog is modal and
    /// pauses, the panel is a HUD and does not.
    pub inventory: InventoryState,
    /// What the panel is showing — the host's projection of the character's own
    /// bag, refreshed once a frame.
    pub bag: InventoryView,
    /// Verbs the panel produced and the sim has not applied yet.
    ///
    /// A queue rather than a direct edit, and the reason is the wave's own:
    /// gameplay happens on the **fixed** step. A panel that reached into the
    /// world would move a player's things on the frame clock, which is the one
    /// thing a PIE-versus-shipping trace cannot survive.
    pending: Vec<InventoryVerb>,
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
            inventory: InventoryState::default(),
            bag: InventoryView::default(),
            pending: Vec::new(),
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
        // **A settings file cannot lock the player out of the settings** (I5
        // audit, A1). The edit doors refuse to produce this state; the file is
        // the case they do not cover — it outlives the build that wrote it, it
        // can be hand-edited, and a project's own `input.toml` may simply not
        // bind the menu at all. Restoring the shipped key is the only answer
        // that leaves the player a way to change their mind.
        inf_ui::bindings::restore_menu_if_unreachable(&mut map);
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
        // **The inventory panel is deliberately NOT here.** See
        // `inf_ui::inventory`'s header: a bag that froze the world would be a
        // safe place to stand, which is the opposite of what an inventory in a
        // game of this kind is for.
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
        // **The dialog outranks the panel.** Both can be open — the panel does
        // not pause, so a player can open the menu over it — and a key can only
        // belong to one of them. The modal one wins, which is what modal means.
        if !self.menu.open {
            if self.inventory.open {
                // **THE KEY THAT OPENED IT CLOSES IT** — the owner's table says
                // `I = inventory open/close`, and an open panel takes every key
                // it is given (that is `inf_ui::inventory`'s own rule), so the
                // press would never reach the host's `just_pressed` edge and the
                // panel could only be left through `Escape`. The I6 audit
                // measured that: `I` opened the bag and `I` did nothing.
                //
                // Asked of the **live map** rather than by literal, unlike the
                // dialog's own `"Escape" | "Tab"` arm: the panel is opened by a
                // binding a player can change, and a close key frozen at `KeyI`
                // would strand anyone who moved it.
                if map
                    .owners_of_key(code)
                    .iter()
                    .any(|(name, axis, _)| !axis && *name == inf_input::actions::INVENTORY)
                {
                    self.inventory.set_open(false);
                    return KeyVerdict {
                        consumed: true,
                        ..Default::default()
                    };
                }
                let out = inventory::handle(
                    &mut self.inventory,
                    &self.bag,
                    &inf_ui::inventory::InventoryInput::Key(code.to_string()),
                );
                if let Some(v) = out.verb {
                    self.pending.push(v);
                }
                return KeyVerdict {
                    consumed: out.consumed,
                    ..Default::default()
                };
            }
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

    /// **Open or close the inventory panel** — the `inventory` action's
    /// consumer, and the last of the four controls I5 bound against nothing.
    ///
    /// Refused while the settings dialog is open, because the dialog is modal:
    /// a panel that opened behind it would take keys the dialog had already
    /// claimed.
    pub fn toggle_inventory(&mut self) {
        if self.menu.open {
            return;
        }
        let open = !self.inventory.open;
        self.inventory.set_open(open);
    }

    /// What the panel produced since the last drain — the host applies these on
    /// the sim's own step.
    pub fn take_inventory_verbs(&mut self) -> Vec<InventoryVerb> {
        std::mem::take(&mut self.pending)
    }

    /// Refresh what the panel is showing. Called once a frame with the
    /// character's own bag, projected out of the sim.
    pub fn set_bag(&mut self, bag: InventoryView) {
        self.bag = bag;
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
            // An axis has no edge, so a source that moved at all counts as a
            // press — asked of every name rather than of one by identity,
            // because the list is what says which controls are unwired and a
            // second list of "which of them are axes" would be a second thing
            // to keep in step. (It was one name until I6, when the last of the
            // four gained a consumer and the list emptied.)
            let fired = state.just_pressed(action) || state.axis(action).abs() > f32::EPSILON;
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
        // The panel draws UNDER the toasts and under the dialog: a modal is on
        // top of everything, and a toast is what tells a player why something
        // did not happen.
        inf_ui::inventory::draw(&mut self.list, &self.inventory, &self.bag);
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

    /// **Draw the driver's readout** (wave VEH2b) — speed and gear, bottom
    /// centre.
    ///
    /// The same box `prompt` draws, at a fixed place instead of a projected
    /// one, because a HUD element and a world prompt are the same thing to a
    /// draw list and a second style would be a second thing to keep in step.
    /// Nothing is drawn while the menu is open, for `prompt`'s reason.
    pub fn readout(&mut self, text: &str) {
        if self.menu.open || text.is_empty() {
            return;
        }
        let vp = self.list.viewport;
        inf_ui::view::prompt(
            &mut self.list,
            Vec2::new(vp.x * 0.5, vp.y),
            text,
            inf_ui::view::palette::TEXT,
        );
    }

    /// **Draw the aiming reticle** (wave WPN1) — the centre of the screen, and
    /// only while the character is pointing a weapon.
    ///
    /// The caller decides *whether*, this decides *how*, and the split is
    /// [`readout`](Self::readout)'s: the condition is a question about the world
    /// (is the camera subject in `RotationMode::Aiming` with something loaded)
    /// and the drawing is a question about a viewport.
    ///
    /// Nothing is drawn while the menu is open, for `prompt`'s reason: a
    /// crosshair over a settings dialog points at a control the dialog is
    /// currently eating.
    pub fn reticle(&mut self) {
        if self.menu.open {
            return;
        }
        inf_ui::view::reticle(&mut self.list, inf_ui::view::palette::TEXT);
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

    /// **A SETTINGS FILE THAT UNBINDS THE MENU DOES NOT SHIP A GAME WITH NO
    /// SETTINGS** (I5 audit, A1) — measured at the boot door rather than at the
    /// rule.
    ///
    /// `inf_ui::bindings` refuses to *make* this state and has its own arms for
    /// that; the thing this one holds is the **wiring**, which is the I1 law: a
    /// gate that calls the rule measures the rule, and only a gate that goes
    /// through the boot path measures the fix. Stubbing the call in
    /// [`PlayerUi::tuned_map`] leaves every other arm in this tree green.
    ///
    /// The file is written by hand on purpose: it is the case the edit doors
    /// cannot cover — a build that had no guard, a hand edit, a project whose own
    /// `input.toml` simply never bound the menu.
    #[test]
    fn a_settings_file_with_no_menu_key_still_boots_a_game_with_a_menu() {
        let dir = tmp();
        let mut on_disk = GameSettings::default();
        on_disk.bindings.insert("menu".into(), String::new());
        on_disk.save(&dir).unwrap();
        let (ui, map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        assert!(ui.load_error.is_none(), "{:?}", ui.load_error);
        assert_eq!(
            ui.settings.bindings.get("menu").map(String::as_str),
            Some(""),
            "the fixture did not reach the settings, so nothing below proves anything"
        );
        assert!(
            !inf_ui::bindings::menu_is_unreachable(&map),
            "the shipped player booted with no way to open its own settings dialog"
        );
        let table = inf_ui::bindings::rows();
        let menu = table.iter().find(|r| r.id == "menu").unwrap();
        assert_eq!(inf_ui::bindings::token_in(&map, menu), "Tab");
        // …and the heal is the MENU's alone: an unrelated override survives it.
        let dir = tmp();
        let mut on_disk = GameSettings::default();
        on_disk.bindings.insert("menu".into(), String::new());
        on_disk.bindings.insert("crouch".into(), "KeyB".into());
        on_disk.save(&dir).unwrap();
        let (_, map) = PlayerUi::open(dir, inf_input::default_map());
        let crouch = table.iter().find(|r| r.id == "crouch").unwrap();
        assert_eq!(inf_ui::bindings::token_in(&map, crouch), "KeyB");
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

    /// **Every control in the shipped table is wired now** (I6), so the honest
    /// "not yet" toast has nothing left to say.
    ///
    /// The arm changed shape rather than being deleted, and the change is the
    /// finding: it used to press `KeyR` and demand a toast, because `reload` was
    /// one of the four the owner's table bound ahead of its consumer. All four —
    /// reload, attack, inventory, the weapon wheel — have one as of this wave,
    /// so pressing any of them must now be **silent**, and the mechanism is
    /// exercised on its own rather than through a control that no longer needs
    /// it. An arm that still expected a toast would be asserting that the wave
    /// did not happen.
    #[test]
    fn every_bound_control_is_wired_and_the_not_yet_toast_still_works() {
        let dir = tmp();
        // Every key the four ex-unwired controls are bound to is silent.
        for code in ["KeyR", "KeyI", "Space"] {
            let mut ui = PlayerUi::open(dir.clone(), inf_input::default_map()).0;
            let mut state = InputState::new(inf_input::default_map());
            state.apply(&[inf_input::InputEvent::Key {
                code: code.into(),
                pressed: true,
            }]);
            ui.report_unconsumed(&state);
            assert!(
                ui.toasts.is_empty(),
                "`{code}` still toasts as unwired: {:?}",
                ui.toasts.live()
            );
        }
        // …and so is the mouse wheel, which is the axis half of the same rule.
        let mut ui = PlayerUi::open(dir.clone(), inf_input::default_map()).0;
        let mut state = InputState::new(inf_input::default_map());
        state.apply(&[inf_input::InputEvent::MouseWheel { delta: [0.0, 1.0] }]);
        ui.report_unconsumed(&state);
        assert!(ui.toasts.is_empty(), "{:?}", ui.toasts.live());

        // **The mechanism is not retired**, only unemployed: the day a later
        // wave binds a control ahead of its consumer, this is the sentence a
        // player reads. Exercised directly, because the list it walks is a
        // const and is now empty — an arm driven through an empty list is an
        // arm that cannot fail.
        ui.toasts.push_not_yet("grapple");
        assert_eq!(ui.toasts.live().len(), 1);
        assert!(ui.toasts.live()[0].text.contains("grapple"));

        // …and an open menu reports nothing at all: the dialog is eating the
        // keys, so a toast would be about a press the game never saw.
        let mut ui2 = PlayerUi::open(dir.clone(), inf_input::default_map()).0;
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

    /// **`I` OPENS THE BAG AND `I` CLOSES IT** — the owner's control table's own
    /// words, armed by the I6 audit.
    ///
    /// It could not, and the reason is structural rather than a typo: an open
    /// panel consumes **every** key (`inf_ui::inventory`'s own rule, and the
    /// dialog's), so the press never reached `InputState` and the host's
    /// `just_pressed(INVENTORY)` edge — the thing that opened it — could never
    /// fire again. `Escape` was the only way out.
    ///
    /// The close is decided against the **live map**, so the third block below
    /// is the half a literal `"KeyI"` would fail.
    #[test]
    fn the_inventory_key_closes_the_panel_it_opened_and_a_rebound_one_does_too() {
        let dir = tmp();
        let (mut ui, mut map) = PlayerUi::open(dir.clone(), inf_input::default_map());
        // Closed: the key belongs to the game, and the HOST's edge opens it.
        assert!(!ui.key("KeyI", true, &mut map).consumed);
        ui.toggle_inventory();
        assert!(ui.inventory.open);

        // Open: the same key closes it, and is taken rather than forwarded — a
        // key that reached the game would move the character behind the panel.
        let v = ui.key("KeyI", true, &mut map);
        assert!(v.consumed, "the panel let its own key through to the game");
        assert!(
            !ui.inventory.open,
            "`I` opened the bag and `I` could not close it"
        );

        // Rebound: the map is asked, so the new key closes it and the old one is
        // just another key the open panel eats.
        let mut rebound = inf_input::default_map();
        rebound.remove_action_source(
            inf_input::actions::INVENTORY,
            &inf_input::ActionSource::Key("KeyI".into()),
        );
        rebound.bind_key(inf_input::actions::INVENTORY, "KeyB");
        ui.toggle_inventory();
        assert!(ui.inventory.open);
        assert!(ui.key("KeyI", true, &mut rebound).consumed);
        assert!(
            ui.inventory.open,
            "a key that is no longer bound to the inventory closed the panel"
        );
        assert!(ui.key("KeyB", true, &mut rebound).consumed);
        assert!(
            !ui.inventory.open,
            "the rebound key did not close the panel"
        );

        // …and `Escape` still does, because a player must always have a way out
        // (the I5 rule that `Escape` cannot be bound away).
        ui.toggle_inventory();
        assert!(ui.key("Escape", true, &mut map).consumed);
        assert!(!ui.inventory.open);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
