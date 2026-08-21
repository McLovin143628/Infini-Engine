//! [`InputMap`]: the serde-able binding of named actions/axes to input sources.
//! Pure data — resolution against live device state lives in
//! [`InputState`](crate::InputState).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{ActionSource, AxisSource, GamepadAxis, GamepadButton, MouseAxis, MouseButton};

/// A named binding of digital **actions** and analog **axes** to input sources,
/// shared by the editor and the runtime. It serializes deterministically
/// (`BTreeMap`) so it can live in project settings and diff cleanly (P9).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputMap {
    /// Action name → the sources that trigger it (any source active ⇒ action
    /// active).
    #[serde(default)]
    actions: BTreeMap<String, Vec<ActionSource>>,
    /// Axis name → the sources that contribute to it (contributions summed).
    #[serde(default)]
    axes: BTreeMap<String, Vec<AxisSource>>,
    /// Radial deadzone applied to each gamepad-**axis** source before scaling,
    /// in `[0, 1)`. Keyboard/button contributions are exact and unaffected.
    #[serde(default = "default_deadzone")]
    deadzone: f32,
}

fn default_deadzone() -> f32 {
    0.1
}

impl Default for InputMap {
    fn default() -> Self {
        Self {
            actions: BTreeMap::new(),
            axes: BTreeMap::new(),
            deadzone: default_deadzone(),
        }
    }
}

impl InputMap {
    /// A new, empty map with the default deadzone.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the gamepad-axis deadzone (clamped to `[0, 1)`).
    pub fn with_deadzone(mut self, deadzone: f32) -> Self {
        self.deadzone = deadzone.clamp(0.0, 0.999);
        self
    }

    /// The current deadzone.
    pub fn deadzone(&self) -> f32 {
        self.deadzone
    }

    /// Bind a source to an action (creating the action if new).
    pub fn bind_action(&mut self, action: impl Into<String>, source: ActionSource) -> &mut Self {
        self.actions.entry(action.into()).or_default().push(source);
        self
    }

    /// Bind a keyboard key (a `KeyboardEvent.code`) to an action — the common case.
    pub fn bind_key(&mut self, action: impl Into<String>, code: impl Into<String>) -> &mut Self {
        self.bind_action(action, ActionSource::Key(code.into()))
    }

    /// Bind a gamepad button to an action.
    pub fn bind_button(&mut self, action: impl Into<String>, button: GamepadButton) -> &mut Self {
        self.bind_action(action, ActionSource::GamepadButton(button))
    }

    /// Bind a source to an axis (creating the axis if new).
    pub fn bind_axis(&mut self, axis: impl Into<String>, source: AxisSource) -> &mut Self {
        self.axes.entry(axis.into()).or_default().push(source);
        self
    }

    /// Bind a keyboard key contributing `scale` to an axis while held.
    pub fn bind_axis_key(
        &mut self,
        axis: impl Into<String>,
        code: impl Into<String>,
        scale: f32,
    ) -> &mut Self {
        self.bind_axis(
            axis,
            AxisSource::Key {
                code: code.into(),
                scale,
            },
        )
    }

    /// Bind a gamepad analog axis (scaled) to an axis.
    pub fn bind_axis_stick(
        &mut self,
        axis: impl Into<String>,
        stick: GamepadAxis,
        scale: f32,
    ) -> &mut Self {
        self.bind_axis(axis, AxisSource::GamepadAxis { axis: stick, scale })
    }

    /// Bind `axis` to a mouse motion/wheel channel at `scale` — a *sensitivity*
    /// (P29.3). Convenience for [`bind_axis`](Self::bind_axis) with an
    /// [`AxisSource::MouseAxis`].
    pub fn bind_axis_mouse(
        &mut self,
        axis: impl Into<String>,
        mouse: MouseAxis,
        scale: f32,
    ) -> &mut Self {
        self.bind_axis(axis, AxisSource::MouseAxis { axis: mouse, scale })
    }

    /// Bind `action` to a mouse button (P29.3).
    pub fn bind_mouse(&mut self, action: impl Into<String>, button: MouseButton) -> &mut Self {
        self.bind_action(action, ActionSource::MouseButton(button))
    }

    /// The sources bound to `action`, if any.
    pub fn action_sources(&self, action: &str) -> Option<&[ActionSource]> {
        self.actions.get(action).map(Vec::as_slice)
    }

    /// The sources bound to `axis`, if any.
    pub fn axis_sources(&self, axis: &str) -> Option<&[AxisSource]> {
        self.axes.get(axis).map(Vec::as_slice)
    }

    /// Every bound action name, sorted.
    pub fn action_names(&self) -> impl Iterator<Item = &str> {
        self.actions.keys().map(String::as_str)
    }

    /// Every bound axis name, sorted.
    pub fn axis_names(&self) -> impl Iterator<Item = &str> {
        self.axes.keys().map(String::as_str)
    }

    /// Iterate `(action, sources)` in sorted action order — used by
    /// [`InputState`](crate::InputState) to resolve every action each frame.
    pub fn actions_iter(&self) -> impl Iterator<Item = (&str, &[ActionSource])> {
        self.actions.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Iterate `(axis, sources)` in sorted axis order.
    pub fn axes_iter(&self) -> impl Iterator<Item = (&str, &[AxisSource])> {
        self.axes.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    // ── rebinding (I5) ──────────────────────────────────────────────────────
    //
    // A rebinding UI edits **one source of one name**, not a whole list: a pad
    // binding is not something a player rebinds with a keyboard, and replacing
    // the list would silently unbind the controller. So every door below names
    // the *desk* source — the keyboard or mouse one — and leaves the gamepad
    // sources exactly where the table put them.

    /// Whether `source` is one a desk rebinding may touch (a key or a mouse
    /// button) — as opposed to a gamepad binding, which the same row is not
    /// about.
    fn is_desk(source: &ActionSource) -> bool {
        matches!(source, ActionSource::Key(_) | ActionSource::MouseButton(_))
    }

    /// The keyboard/mouse source bound to `action`, if any — what a bindings
    /// table shows in the row's cell.
    pub fn desk_source(&self, action: &str) -> Option<&ActionSource> {
        self.actions.get(action)?.iter().find(|s| Self::is_desk(s))
    }

    /// Replace `action`'s keyboard/mouse source with `source`, returning the one
    /// it displaced. An action with no desk source yet gets one at the **front**,
    /// so the table's cell and the map agree from the next read on.
    pub fn set_desk_source(&mut self, action: &str, source: ActionSource) -> Option<ActionSource> {
        let list = self.actions.entry(action.to_string()).or_default();
        match list.iter().position(|s| Self::is_desk(s)) {
            Some(i) => Some(std::mem::replace(&mut list[i], source)),
            None => {
                list.insert(0, source);
                None
            }
        }
    }

    /// Remove `action`'s keyboard/mouse source, returning it. The gamepad
    /// sources stay; an action left with none at all keeps its (empty) entry so
    /// the bindings table can still show the row.
    pub fn clear_desk_source(&mut self, action: &str) -> Option<ActionSource> {
        let list = self.actions.get_mut(action)?;
        let i = list.iter().position(|s| Self::is_desk(s))?;
        Some(list.remove(i))
    }

    /// **Remove one exact source from `action`**, returning whether it was
    /// there.
    ///
    /// [`clear_desk_source`](Self::clear_desk_source) removes whichever desk
    /// source is *first*, which is what a row's "clear" button means. This
    /// removes the one a conflict is **about** — and the two differ whenever an
    /// action has more than one key, which the shipped table's `jump` (Space and
    /// a pad button) and `move_y` (W and Up) both do. A conflict resolved with
    /// the first-source door would take the wrong key away.
    pub fn remove_action_source(&mut self, action: &str, source: &ActionSource) -> bool {
        let Some(list) = self.actions.get_mut(action) else {
            return false;
        };
        let before = list.len();
        list.retain(|s| s != source);
        before != list.len()
    }

    /// **Remove one exact key from `axis`**, whichever sign it contributes on,
    /// returning whether it was there. The twin of
    /// [`remove_action_source`](Self::remove_action_source), and it exists for
    /// the same reason: `move_y` is bound to W *and* Up on its positive half.
    pub fn remove_axis_key(&mut self, axis: &str, code: &str) -> bool {
        let Some(list) = self.axes.get_mut(axis) else {
            return false;
        };
        let before = list.len();
        list.retain(|s| !matches!(s, AxisSource::Key { code: c, .. } if c == code));
        before != list.len()
    }

    /// Every key bound to `axis` on the given sign, in binding order — what a
    /// conflict check has to walk, because a row shows its *first* key and can
    /// answer a press with any of them.
    pub fn axis_keys(&self, axis: &str, positive: bool) -> Vec<&str> {
        self.axes
            .get(axis)
            .map(|list| {
                list.iter()
                    .filter_map(|s| match s {
                        AxisSource::Key { code, scale }
                            if (*scale > 0.0) == positive && *scale != 0.0 =>
                        {
                            Some(code.as_str())
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every keyboard/mouse source bound to `action`, in binding order — the
    /// action twin of [`axis_keys`](Self::axis_keys).
    pub fn desk_sources(&self, action: &str) -> Vec<&ActionSource> {
        self.actions
            .get(action)
            .map(|list| list.iter().filter(|s| Self::is_desk(s)).collect())
            .unwrap_or_default()
    }

    /// The key bound to `axis` on the given **sign** — the half of an axis a
    /// bindings table shows as its own row ("Move Forward" is `move_y` at `+1`).
    ///
    /// `positive` picks the sign; a source whose scale is exactly zero belongs
    /// to neither half and is skipped, because a zero-scale binding contributes
    /// nothing and naming it in a row would offer the player a control that
    /// cannot move.
    pub fn axis_key(&self, axis: &str, positive: bool) -> Option<&str> {
        self.axes.get(axis)?.iter().find_map(|s| match s {
            AxisSource::Key { code, scale } if (*scale > 0.0) == positive && *scale != 0.0 => {
                Some(code.as_str())
            }
            _ => None,
        })
    }

    /// Replace the key bound to `axis` on `positive`'s side with `code`,
    /// returning the one it displaced. The **scale is preserved** when a source
    /// is replaced and defaults to `±1` when one is inserted: a rebinding
    /// changes which key, never how far the key deflects.
    pub fn set_axis_key(
        &mut self,
        axis: &str,
        positive: bool,
        code: impl Into<String>,
    ) -> Option<String> {
        let code = code.into();
        let list = self.axes.entry(axis.to_string()).or_default();
        let at = list.iter().position(|s| {
            matches!(s, AxisSource::Key { scale, .. } if (*scale > 0.0) == positive && *scale != 0.0)
        });
        match at {
            Some(i) => {
                let AxisSource::Key { code: old, .. } = &mut list[i] else {
                    unreachable!("the position above matched an AxisSource::Key")
                };
                Some(std::mem::replace(old, code))
            }
            None => {
                list.insert(
                    0,
                    AxisSource::Key {
                        code,
                        scale: if positive { 1.0 } else { -1.0 },
                    },
                );
                None
            }
        }
    }

    /// Remove the key bound to `axis` on `positive`'s side, returning it.
    pub fn clear_axis_key(&mut self, axis: &str, positive: bool) -> Option<String> {
        let list = self.axes.get_mut(axis)?;
        let i = list.iter().position(|s| {
            matches!(s, AxisSource::Key { scale, .. } if (*scale > 0.0) == positive && *scale != 0.0)
        })?;
        match list.remove(i) {
            AxisSource::Key { code, .. } => Some(code),
            _ => unreachable!("the position above matched an AxisSource::Key"),
        }
    }

    /// **Who already owns this key**, as `(name, is_axis, positive)` — every
    /// name in sorted order, so a conflict dialog can say *which* control it is
    /// about to take the key from.
    ///
    /// A key legitimately means one thing to an axis and another to an action
    /// (Space is `jump`, `handbrake` and `move_up`), so this answers with **all**
    /// of them and the caller decides which collisions matter. Answering with
    /// the first would make a conflict dialog name whichever row sorted lowest
    /// and quietly steal the rest.
    pub fn owners_of_key(&self, code: &str) -> Vec<(&str, bool, bool)> {
        let mut out = Vec::new();
        for (name, sources) in self.actions_iter() {
            if sources
                .iter()
                .any(|s| matches!(s, ActionSource::Key(c) if c == code))
            {
                out.push((name, false, true));
            }
        }
        for (name, sources) in self.axes_iter() {
            for s in sources {
                if let AxisSource::Key { code: c, scale } = s {
                    if c == code && *scale != 0.0 {
                        out.push((name, true, *scale > 0.0));
                    }
                }
            }
        }
        out
    }

    /// [`owners_of_key`](Self::owners_of_key) for a mouse button. An axis cannot
    /// be bound to one in the default table, but [`AxisSource::MouseButton`]
    /// exists, so both halves are walked rather than one assumed.
    pub fn owners_of_mouse(&self, button: MouseButton) -> Vec<(&str, bool, bool)> {
        let mut out = Vec::new();
        for (name, sources) in self.actions_iter() {
            if sources
                .iter()
                .any(|s| matches!(s, ActionSource::MouseButton(b) if *b == button))
            {
                out.push((name, false, true));
            }
        }
        for (name, sources) in self.axes_iter() {
            for s in sources {
                if let AxisSource::MouseButton { button: b, scale } = s {
                    if *b == button && *scale != 0.0 {
                        out.push((name, true, *scale > 0.0));
                    }
                }
            }
        }
        out
    }
}

/// **The action names the default table ships that are not movement verbs**
/// (I5).
///
/// `inf_ecs::movement::actions` is the *movement* vocabulary and lives in
/// `inf-ecs` because the movement intent reads it. These are the rest of the
/// shipped control scheme — the menu, the interaction verb's siblings, and the
/// four the owner's table binds now against consumers that arrive with the
/// weapons and inventory work. They live **here**, beside the table that binds
/// them, for exactly the reason the movement half lives beside the intent that
/// reads it: a name and the thing that reads it belong in one crate, and the
/// one crate that names both vocabularies (`inf-player`) holds them together
/// with an arm rather than a comment.
///
/// The two sets are **disjoint by construction** and
/// `the_two_vocabularies_do_not_overlap` is what says so.
pub mod actions {
    /// Open / close the in-game menu. Tab. Read by the host, not by the sim's
    /// movement step — see `inf_ui::menu`.
    pub const MENU: &str = "menu";
    /// Reload the held weapon. Bound now; its consumer is the weapons work.
    pub const RELOAD: &str = "reload";
    /// Primary fire. Bound now; its consumer is the weapons work.
    pub const ATTACK: &str = "attack";
    /// Open the inventory. Bound now; its consumer is the inventory work.
    pub const INVENTORY: &str = "inventory";
    /// Change weapon. An **axis**, because a wheel has a sign and no button:
    /// its consumer reads the sign. Bound now; its consumer is the weapons
    /// work.
    pub const WEAPON_SWITCH: &str = "weapon_switch";

    /// The four whose consumers do not exist yet, in one place, so a host can
    /// answer "is this control wired to anything" without a second list.
    ///
    /// **A bound key with no consumer is a dead key**, and a dead key is
    /// indistinguishable from a broken one. The shipped player says so out
    /// loud instead (`inf_ui::Toasts`), which is the refusals-are-values law
    /// applied to a control scheme.
    pub const NOT_YET_CONSUMED: [&str; 4] = [RELOAD, ATTACK, INVENTORY, WEAPON_SWITCH];
}

/// **The engine's default bindings** (P29.6 — moved here from the shipped
/// player).
///
/// It lived in `inf_player::input` from P9.3, which was survivable while the
/// editor's Simulate took a set of *action names* straight off the frontend and
/// could not carry an axis at all. P29.6's camera needs the mouse in the editor
/// too, and a second copy of a binding table is the "two copies across a language
/// boundary" defect the campaign's Wave I found at this very seam. So the table
/// is Ring 0, both hosts read it, and a project still overrides the whole thing
/// with an `input.toml` beside its level.
///
/// The vocabulary is `inf_ecs::movement::actions`' by construction — the two are
/// held together by `inf-player`'s own arm, because `inf-input` must not depend
/// on `inf-ecs` to say so.
///
/// `look_x`/`look_y` are **degrees per raw device unit**; the delta reaches the
/// sim as degrees per SECOND ([`crate::InputState::axis_snapshot`]), which is
/// exactly ALS's `AimYawRate`. 0.15 deg/count is a middle-of-the-road desktop
/// sensitivity. `look_y` inverts because the platform reports `+y` down and a
/// look control wants `+pitch` up — the binding says so rather than the engine
/// guessing.
///
/// # The shipped table (I5)
///
/// | control | action / axis |
/// |---|---|
/// | W / S / A / D (and the arrows) | `move_y` / `move_x` |
/// | mouse | `look_x` / `look_y` |
/// | **Tab** | `menu` — the in-game settings dialog |
/// | **Shift** | `sprint` |
/// | **Ctrl** | `walk` (the gait default is RUN; this is the *slow* modifier) |
/// | **E** | `interact` |
/// | **R** | `reload` |
/// | **C** | `crouch` — click crouches or slides, a long press goes prone or dives |
/// | **Space** | `jump`, and `move_up` while swimming or flying, and `handbrake` while driving |
/// | **LMB** | `attack` |
/// | **RMB** | `aim` — `RotationMode::Aiming` |
/// | **wheel** | `weapon_switch` |
/// | **I** | `inventory` |
/// | X · Z · F · V | `prone` · `roll` · `dive` · `fly`, the direct controls the table above folds |
///
/// Four of those — `reload`, `attack`, `inventory`, `weapon_switch` — are bound
/// against consumers that do not exist yet, and the shipped player *says so*
/// when one is pressed. See [`actions::NOT_YET_CONSUMED`].
pub fn default_map() -> InputMap {
    use crate::types::{GamepadAxis, GamepadButton, MouseAxis, MouseButton};
    let mut m = InputMap::new();
    m.bind_key("left", "KeyA")
        .bind_key("left", "ArrowLeft")
        .bind_key("right", "KeyD")
        .bind_key("right", "ArrowRight")
        .bind_key("up", "KeyW")
        .bind_key("up", "ArrowUp")
        .bind_key("down", "KeyS")
        .bind_key("down", "ArrowDown")
        // **Space alone jumps** (I5). `KeyW` and `ArrowUp` were bound here for
        // the 2D platformer, where up *is* jump — and on a 3D character that
        // made every step forward a jump, because `move_y`'s forward key and
        // the jump action were the same key and the intent reads both. Nothing
        // in the tree could have said so: every scripted trace presses action
        // NAMES. The 2D vocabulary keeps `up`, which the Coyote blueprint
        // queries; a 2D project that wants W to jump binds it in its own
        // `input.toml`.
        .bind_key("jump", "Space")
        .bind_button("jump", GamepadButton::South)
        .bind_axis_key("move_x", "KeyD", 1.0)
        .bind_axis_key("move_x", "ArrowRight", 1.0)
        .bind_axis_key("move_x", "KeyA", -1.0)
        .bind_axis_key("move_x", "ArrowLeft", -1.0)
        .bind_axis_stick("move_x", GamepadAxis::LeftStickX, 1.0)
        // Screen/stick y is +down; invert so "up = forward/positive".
        .bind_axis_stick("move_y", GamepadAxis::LeftStickY, -1.0)
        .bind_axis_mouse("look_x", MouseAxis::X, 0.15)
        .bind_axis_mouse("look_y", MouseAxis::Y, -0.15)
        .bind_axis_stick("look_x", GamepadAxis::RightStickX, 180.0)
        .bind_axis_stick("look_y", GamepadAxis::RightStickY, -180.0)
        .bind_key("sprint", "Shift")
        .bind_button("sprint", GamepadButton::LeftThumb)
        // **Ctrl walks** (I5, the owner's table). It was `AltLeft`, which no
        // shipped game uses and which the window manager eats on two of three
        // platforms. `keycode_to_code` already folded both control keys onto
        // one name, so the binding was one string away the whole time.
        .bind_key("walk", "Control")
        .bind_key("crouch", "KeyC")
        .bind_button("crouch", GamepadButton::East)
        // `prone` and `dive` keep a **direct** key each beside the C-key's
        // click/long-press discrimination (I5). Two reasons, and neither is
        // nostalgia: the intent still reads both as edges — a Blueprint, an AI
        // or a rebind can raise one — so an action bound to nothing would be a
        // control that silently does nothing, which is what
        // `every_movement_action_the_intent_reads_is_bound` exists to refuse;
        // and a player who wants a dedicated prone key can now rebind one
        // without the C key's timing.
        .bind_key("prone", "KeyX")
        // ── P29.6: the four the catalogue names and P29.3 left unbound ──
        //
        // `move_y` had no KEYBOARD binding at all — only the left stick — so a
        // character on a keyboard could strafe but not walk forward. `move_up`,
        // `roll` and `dive` had no binding on any device. Found by the showcase
        // course, which is the first content to drive a character.
        .bind_axis_key("move_y", "KeyW", 1.0)
        .bind_axis_key("move_y", "ArrowUp", 1.0)
        .bind_axis_key("move_y", "KeyS", -1.0)
        .bind_axis_key("move_y", "ArrowDown", -1.0)
        // **The vertical axis moved off Q/E onto Space/Ctrl** (I5). `KeyE` was
        // the ascend key *and* the interact key, so a swimmer who pressed E to
        // open a hatch also rose; and the owner's table gives E one job. Space
        // and Ctrl already mean "up" and "down" to every swimmer and pilot
        // alive, and neither means anything to a grounded step — a key meaning
        // one thing to an axis and another to an action is what a binding table
        // is for, and only the modes that read each can tell.
        .bind_axis_key("move_up", "Space", 1.0)
        .bind_axis_key("move_up", "Control", -1.0)
        // **R is reload** (I5, the owner's table), so the roll moved to Z.
        .bind_key("roll", "KeyZ")
        .bind_key("dive", "KeyF")
        // ── P29.7: the three the vehicle and flight seams read ──
        //
        // `interact` is E, which is every game's enter-a-vehicle key; it shares
        // that key with the `move_up` AXIS, which is a binding table doing
        // exactly what a binding table is for (a key means one thing to an axis
        // and another to an action, and only the modes that read each can tell).
        // `handbrake` shares Space with `jump` for the same reason: a driver has
        // no jump and a pedestrian has no handbrake.
        .bind_key("interact", "KeyE")
        .bind_button("interact", GamepadButton::North)
        .bind_key("fly", "KeyV")
        .bind_button("fly", GamepadButton::West)
        .bind_key("handbrake", "Space")
        .bind_button("handbrake", GamepadButton::South)
        .bind_mouse("aim", MouseButton::Right)
        .bind_button("aim", GamepadButton::LeftTrigger)
        // ── I5: the owner's table, the rest of it ──
        //
        // `menu` is read by the HOST, not by the movement intent: it opens the
        // in-game settings dialog and pauses a single-player sim, and both of
        // those are decisions about the session rather than about the
        // character. The other four are bound against consumers that arrive
        // with the weapons and inventory work — see [`actions::NOT_YET_CONSUMED`]
        // and the toast the player raises for them, which is the difference
        // between a control that is not built yet and a control that is broken.
        .bind_key(actions::MENU, "Tab")
        .bind_button(actions::MENU, GamepadButton::Start)
        .bind_key(actions::RELOAD, "KeyR")
        .bind_mouse(actions::ATTACK, MouseButton::Left)
        .bind_button(actions::ATTACK, GamepadButton::RightTrigger)
        .bind_key(actions::INVENTORY, "KeyI")
        .bind_button(actions::INVENTORY, GamepadButton::Select)
        // The wheel is a DELTA source, so this axis is unclamped and reaches a
        // consumer as a RATE (see `InputState::axis_snapshot`). Its consumer
        // reads the sign; a notch count would need the wheel to be a button,
        // which it is not on any platform this engine speaks to.
        .bind_axis_mouse(actions::WEAPON_SWITCH, MouseAxis::WheelY, 1.0);
    m
}
