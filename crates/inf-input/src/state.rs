//! [`InputState`]: live device state + per-frame resolution of an [`InputMap`]
//! into action edges and axis values. This is the **pure, tested core** — it owns
//! no device and reads no clock; [`apply`](InputState::apply) is fed a frame's raw
//! events (from any source, including the optional gamepad poller) and everything
//! else is a query over the resolved snapshot.

use std::collections::{BTreeMap, BTreeSet};

use crate::map::InputMap;
use crate::types::{ActionSource, AxisSource, GamepadAxis, GamepadButton, InputEvent};

/// Radial (1-D) deadzone with rescale: values inside `dz` read as 0; values
/// outside are remapped so the response starts at 0 exactly at the deadzone edge
/// (no discontinuity). `dz` is clamped to `[0, 1)`.
fn apply_deadzone(v: f32, dz: f32) -> f32 {
    let dz = dz.clamp(0.0, 0.999);
    let a = v.abs();
    if a <= dz {
        0.0
    } else {
        v.signum() * ((a - dz) / (1.0 - dz)).min(1.0)
    }
}

/// Live input state resolved through an [`InputMap`].
///
/// Drive it once per frame with [`apply`](Self::apply); then query
/// [`pressed`](Self::pressed) / [`just_pressed`](Self::just_pressed) /
/// [`just_released`](Self::just_released) for actions and [`axis`](Self::axis) for
/// analog values. Edge detection compares this frame's resolved actions against
/// the previous frame's, so exactly one `apply` per frame yields correct edges.
#[derive(Clone, Debug)]
pub struct InputState {
    map: InputMap,
    // ── raw device state (accumulated across events) ──
    keys_down: BTreeSet<String>,
    buttons_down: BTreeSet<GamepadButton>,
    axes_raw: BTreeMap<GamepadAxis, f32>,
    // ── resolved snapshot (recomputed each frame) ──
    actions_prev: BTreeSet<String>,
    actions_cur: BTreeSet<String>,
    axis_values: BTreeMap<String, f32>,
}

impl InputState {
    /// A state driven by `map`, with no keys/buttons held.
    pub fn new(map: InputMap) -> Self {
        Self {
            map,
            keys_down: BTreeSet::new(),
            buttons_down: BTreeSet::new(),
            axes_raw: BTreeMap::new(),
            actions_prev: BTreeSet::new(),
            actions_cur: BTreeSet::new(),
            axis_values: BTreeMap::new(),
        }
    }

    /// **Release everything the device is holding** (round-2 finding R2-9).
    ///
    /// The raw sets — `keys_down`, `buttons_down`, `axes_raw` — accumulate
    /// across events and are only ever cleared by the matching *release*. So
    /// they outlive the device that made them: a gamepad unplugged mid-sprint,
    /// a window that loses focus with W held, an app suspended with a stick
    /// pushed forward. The OS sends no release for any of those, and the
    /// character keeps running for the rest of the session — through PIE,
    /// through a level change, and into the replay trace a parity gate
    /// compares.
    ///
    /// One frame is committed as part of the release, so
    /// [`just_released`](Self::just_released) fires for everything that was
    /// down: gameplay that ends an ability on the release edge must see it,
    /// or "release everything" becomes "the ability never ends".
    ///
    /// The map is kept — this is a device event, not a rebind.
    pub fn release_all(&mut self) {
        self.keys_down.clear();
        self.buttons_down.clear();
        self.axes_raw.clear();
        self.commit_frame();
    }

    /// Whether any raw device input is currently held.
    ///
    /// Exists so [`release_all`](Self::release_all) can be falsified: every
    /// public query answers from the RESOLVED snapshot, which a bound-to-nothing
    /// key never reaches, so a release that cleared the actions and left the raw
    /// sets standing would look identical through all of them.
    pub fn anything_held(&self) -> bool {
        !self.keys_down.is_empty() || !self.buttons_down.is_empty() || !self.axes_raw.is_empty()
    }

    /// The map in use.
    pub fn map(&self) -> &InputMap {
        &self.map
    }

    /// Replace the map (e.g. the user rebound a key). Raw device state is kept;
    /// the next [`apply`](Self::apply) re-resolves against the new bindings.
    pub fn set_map(&mut self, map: InputMap) {
        self.map = map;
    }

    /// Advance one frame: fold in `events`, then re-resolve actions + axes and
    /// roll the edge-detection snapshot forward. Call once per frame with that
    /// frame's events (an empty slice is fine — it still commits the frame, so a
    /// `just_pressed` from last frame clears).
    pub fn apply(&mut self, events: &[InputEvent]) {
        for e in events {
            self.apply_raw(e);
        }
        self.commit_frame();
    }

    fn apply_raw(&mut self, event: &InputEvent) {
        match event {
            InputEvent::Key { code, pressed } => {
                if *pressed {
                    self.keys_down.insert(code.clone());
                } else {
                    self.keys_down.remove(code);
                }
            }
            InputEvent::GamepadButton { button, pressed } => {
                if *pressed {
                    self.buttons_down.insert(*button);
                } else {
                    self.buttons_down.remove(button);
                }
            }
            InputEvent::GamepadAxis { axis, value } => {
                self.axes_raw.insert(*axis, *value);
            }
            // Raw touch is not a map source on its own — on-screen controls
            // (`crate::touch::TouchControls`) translate it into the gamepad
            // events above before it reaches the resolver, so here it is ignored.
            InputEvent::Touch { .. } => {}
        }
    }

    fn action_active(&self, sources: &[ActionSource]) -> bool {
        sources.iter().any(|s| match s {
            ActionSource::Key(code) => self.keys_down.contains(code),
            ActionSource::GamepadButton(b) => self.buttons_down.contains(b),
        })
    }

    fn axis_value(&self, sources: &[AxisSource]) -> f32 {
        let dz = self.map.deadzone();
        let mut sum = 0.0;
        for s in sources {
            sum += match s {
                AxisSource::Key { code, scale } => {
                    if self.keys_down.contains(code) {
                        *scale
                    } else {
                        0.0
                    }
                }
                AxisSource::GamepadButton { button, scale } => {
                    if self.buttons_down.contains(button) {
                        *scale
                    } else {
                        0.0
                    }
                }
                AxisSource::GamepadAxis { axis, scale } => {
                    let raw = self.axes_raw.get(axis).copied().unwrap_or(0.0);
                    apply_deadzone(raw, dz) * *scale
                }
            };
        }
        sum.clamp(-1.0, 1.0)
    }

    fn commit_frame(&mut self) {
        std::mem::swap(&mut self.actions_prev, &mut self.actions_cur);
        self.actions_cur.clear();
        for (name, sources) in self.map.actions_iter() {
            if self.action_active(sources) {
                self.actions_cur.insert(name.to_string());
            }
        }
        self.axis_values.clear();
        for (name, sources) in self.map.axes_iter() {
            self.axis_values
                .insert(name.to_string(), self.axis_value(sources));
        }
    }

    // ── queries ────────────────────────────────────────────────────────────────

    /// Whether `action` is currently active (any bound source held).
    pub fn pressed(&self, action: &str) -> bool {
        self.actions_cur.contains(action)
    }

    /// Whether `action` became active this frame (rising edge).
    pub fn just_pressed(&self, action: &str) -> bool {
        self.actions_cur.contains(action) && !self.actions_prev.contains(action)
    }

    /// Whether `action` became inactive this frame (falling edge).
    pub fn just_released(&self, action: &str) -> bool {
        !self.actions_cur.contains(action) && self.actions_prev.contains(action)
    }

    /// The resolved value of `axis` in `[-1, 1]` (0 if unbound).
    pub fn axis(&self, axis: &str) -> f32 {
        self.axis_values.get(axis).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActionSource, AxisSource, GamepadAxis, GamepadButton, InputEvent};
    use crate::InputMap;

    /// **Round-2 finding R2-9**: raw device state outlives the device.
    ///
    /// `keys_down`/`buttons_down`/`axes_raw` accumulate across events and are
    /// cleared only by the matching release. The OS sends none when a window
    /// loses focus with W held, when an app suspends with a stick pushed, or
    /// when a pad is unplugged mid-sprint — so the character kept running for
    /// the rest of the session.
    #[test]
    fn releasing_everything_clears_the_raw_state_and_fires_the_edges() {
        let mut map = InputMap::default();
        map.bind_action("sprint", ActionSource::Key("KeyW".into()));
        map.bind_axis(
            "move_x",
            AxisSource::GamepadAxis {
                axis: GamepadAxis::LeftStickX,
                scale: 1.0,
            },
        );
        let mut st = InputState::new(map);

        st.apply(&[
            InputEvent::Key {
                code: "KeyW".into(),
                pressed: true,
            },
            InputEvent::GamepadAxis {
                axis: GamepadAxis::LeftStickX,
                value: 1.0,
            },
            InputEvent::GamepadButton {
                button: GamepadButton::South,
                pressed: true,
            },
        ]);
        assert!(st.pressed("sprint"), "the fixture never held anything");
        assert_eq!(st.axis("move_x"), 1.0);
        assert!(st.anything_held());

        // The window loses focus. No release arrives from anywhere.
        st.release_all();

        assert!(!st.pressed("sprint"), "the action is still held");
        assert_eq!(st.axis("move_x"), 0.0, "the stick is still pushed");
        assert!(
            !st.anything_held(),
            "the RAW sets survived — every public query answers from the resolved              snapshot, so a release that cleared only the actions would look              identical through all of them"
        );
        assert!(
            st.just_released("sprint"),
            "the release edge did not fire, so gameplay that ends an ability on it              never ends the ability"
        );

        // …and the state is live again: a fresh press works.
        st.apply(&[InputEvent::Key {
            code: "KeyW".into(),
            pressed: true,
        }]);
        assert!(st.pressed("sprint") && st.just_pressed("sprint"));
    }
}
