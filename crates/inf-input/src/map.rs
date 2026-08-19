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
        .bind_key("jump", "Space")
        .bind_key("jump", "KeyW")
        .bind_key("jump", "ArrowUp")
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
        .bind_key("walk", "AltLeft")
        .bind_key("crouch", "KeyC")
        .bind_button("crouch", GamepadButton::East)
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
        .bind_axis_key("move_up", "KeyE", 1.0)
        .bind_axis_key("move_up", "KeyQ", -1.0)
        .bind_key("roll", "KeyR")
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
        .bind_mouse("aim", MouseButton::Right);
    m
}
