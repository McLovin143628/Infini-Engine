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
