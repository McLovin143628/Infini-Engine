//! [`InputState`]: live device state + per-frame resolution of an [`InputMap`]
//! into action edges and axis values. This is the **pure, tested core** — it owns
//! no device and reads no clock; [`apply`](InputState::apply) is fed a frame's raw
//! events (from any source, including the optional gamepad poller) and everything
//! else is a query over the resolved snapshot.

use std::collections::{BTreeMap, BTreeSet};

use crate::map::InputMap;
use crate::types::{
    ActionSource, AxisSource, GamepadAxis, GamepadButton, InputEvent, MouseAxis, MouseButton,
};

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
    // ── mouse (P29.3) ──
    //
    // The buttons are level state like the keys above. The axes are NOT: a
    // mouse delta is a *quantity of motion that happened*, so it accumulates
    // across the events of one frame and is cleared by `commit_frame` once the
    // axes have read it. A stick that is not touched keeps reporting its
    // position; a mouse that is not moved reports nothing, and the difference is
    // the whole reason `AimYawRate` is derivable at all.
    mouse_down: BTreeSet<MouseButton>,
    mouse_axes: BTreeMap<MouseAxis, f32>,
    // ── resolved snapshot (recomputed each frame) ──
    actions_prev: BTreeSet<String>,
    actions_cur: BTreeSet<String>,
    // The two halves each axis resolved into this frame, kept because
    // `axis_snapshot` must divide only the delta one by `dt` and the raw mouse
    // deltas are cleared by `commit_frame` before anyone can ask again.
    axis_positions: BTreeMap<String, f32>,
    axis_deltas: BTreeMap<String, f32>,
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
            mouse_down: BTreeSet::new(),
            mouse_axes: BTreeMap::new(),
            actions_prev: BTreeSet::new(),
            actions_cur: BTreeSet::new(),
            axis_positions: BTreeMap::new(),
            axis_deltas: BTreeMap::new(),
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
        self.mouse_down.clear();
        self.mouse_axes.clear();
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
            InputEvent::MouseButton { button, pressed } => {
                if *pressed {
                    self.mouse_down.insert(*button);
                } else {
                    self.mouse_down.remove(button);
                }
            }
            // Motion ACCUMULATES. Two 4-pixel events in one frame are an 8-pixel
            // frame, not a 4-pixel one: a backend is free to deliver a frame's
            // motion in as many events as the OS gave it, and the resolved axis
            // must not depend on that packetization.
            InputEvent::MouseMotion { delta } => {
                *self.mouse_axes.entry(MouseAxis::X).or_insert(0.0) += delta[0];
                *self.mouse_axes.entry(MouseAxis::Y).or_insert(0.0) += delta[1];
            }
            InputEvent::MouseWheel { delta } => {
                *self.mouse_axes.entry(MouseAxis::WheelX).or_insert(0.0) += delta[0];
                *self.mouse_axes.entry(MouseAxis::WheelY).or_insert(0.0) += delta[1];
            }
        }
    }

    fn action_active(&self, sources: &[ActionSource]) -> bool {
        sources.iter().any(|s| match s {
            ActionSource::Key(code) => self.keys_down.contains(code),
            ActionSource::GamepadButton(b) => self.buttons_down.contains(b),
            ActionSource::MouseButton(b) => self.mouse_down.contains(b),
        })
    }

    /// Whether any source bound to this axis reports a **delta** rather than a
    /// position — the one question both the clamp rule and the rate conversion
    /// turn on.
    fn has_delta_source(sources: &[AxisSource]) -> bool {
        sources
            .iter()
            .any(|s| matches!(s, AxisSource::MouseAxis { .. }))
    }

    /// One axis resolved into its **two physically different halves**: the
    /// sources that report a POSITION (a key, a button, a stick — read the same
    /// at any frame rate) and the sources that report an AMOUNT THAT HAPPENED
    /// this frame (a mouse delta).
    ///
    /// They are split rather than summed because only the second half is a
    /// per-frame quantity, so only the second half may be divided by `dt` to
    /// become a rate ([`axis_snapshot`](Self::axis_snapshot)). Summing first and
    /// dividing the total is the shape that made a right stick bound to the same
    /// axis as the mouse read sixty times its own scale (P29.3 audit, A2) — the
    /// rate conversion is a property of the SOURCE, exactly as the clamp is.
    fn axis_parts(&self, sources: &[AxisSource]) -> (f32, f32) {
        let dz = self.map.deadzone();
        let mut position = 0.0;
        let mut delta = 0.0;
        for s in sources {
            match s {
                AxisSource::Key { code, scale } => {
                    if self.keys_down.contains(code) {
                        position += *scale;
                    }
                }
                AxisSource::GamepadButton { button, scale } => {
                    if self.buttons_down.contains(button) {
                        position += *scale;
                    }
                }
                AxisSource::GamepadAxis { axis, scale } => {
                    let raw = self.axes_raw.get(axis).copied().unwrap_or(0.0);
                    position += apply_deadzone(raw, dz) * *scale;
                }
                AxisSource::MouseAxis { axis, scale } => {
                    // No deadzone: a deadzone rescales a *position*, and there
                    // is no position here. A 1-pixel move is a 1-pixel move.
                    delta += self.mouse_axes.get(axis).copied().unwrap_or(0.0) * *scale;
                }
                AxisSource::MouseButton { button, scale } => {
                    if self.mouse_down.contains(button) {
                        position += *scale;
                    }
                }
            }
        }
        (position, delta)
    }

    /// The two halves recombined under the axis's clamp rule.
    ///
    /// An axis that names a mouse delta is **not clamped**. Every other source is
    /// bounded by construction — a key is 0 or 1, a stick is `[-1, 1]` — so the
    /// clamp is free for them and a guard against a map that binds five keys to
    /// one axis. A mouse delta has no such range: a fast flick is fifty pixels
    /// and a slow drag is two, and clamping both to 1 makes them the same
    /// gesture, which is exactly the look control every engine gets wrong once.
    /// The same reasoning makes such an axis a *rate* axis, so a stick bound to
    /// it may legitimately carry a scale far outside `[-1, 1]`.
    fn combine(sum: f32, unbounded: bool) -> f32 {
        if unbounded {
            sum
        } else {
            sum.clamp(-1.0, 1.0)
        }
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
        self.axis_positions.clear();
        self.axis_deltas.clear();
        for (name, sources) in self.map.axes_iter() {
            let (position, delta) = self.axis_parts(sources);
            self.axis_values.insert(
                name.to_string(),
                Self::combine(position + delta, Self::has_delta_source(sources)),
            );
            self.axis_positions.insert(name.to_string(), position);
            self.axis_deltas.insert(name.to_string(), delta);
        }
        // The mouse deltas are consumed BY the resolution above and cleared
        // after it: they describe motion that belongs to the frame just
        // committed. Leaving them would make a single flick drive the look for
        // every frame after it — the mouse-look equivalent of the stuck-key
        // failure `release_all` exists for.
        self.mouse_axes.clear();
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

    /// Every resolved axis for this frame, with the **delta axes converted to
    /// rates** (P29.3). `dt` is the wall-clock seconds the frame covered.
    ///
    /// # Why the conversion has to happen here and exactly once
    ///
    /// The two kinds of axis this state resolves are not the same physical
    /// quantity. A stick or a key is a **position**: it reads the same at 30 fps
    /// and at 240 fps, and a fixed step that samples it N times gets the right
    /// answer N times. A mouse delta is an **amount that happened**: the same
    /// gesture is one big number at 30 fps and eight small ones at 240, and a
    /// fixed step that consumed it once per step would turn the same flick into
    /// eight times the rotation on a fast machine.
    ///
    /// Dividing a delta axis by the frame's own `dt` makes it a rate, and a
    /// fixed step that integrates `rate × fixed_dt` then reproduces the gesture
    /// exactly, whatever the frame rate. It also happens to be the *definition*
    /// of ALS's `AimYawRate` — degrees of camera yaw per second — which three
    /// ported systems read.
    ///
    /// Bounded axes pass through untouched: dividing a stick position by `dt`
    /// would be meaningless. `dt <= 0` (or non-finite) leaves everything alone,
    /// because the alternative is publishing an infinity into the sim.
    ///
    /// # The conversion is per SOURCE, not per axis name (P29.3 audit, A2)
    ///
    /// The first version asked "does this axis name any mouse source" and, if so,
    /// divided the axis's whole resolved value. The shipped `default_map` binds
    /// `look_x` to the mouse **and** to the right stick — a look axis is a rate
    /// axis, so the stick's scale is 180 deg/s rather than 1 — and the stick's
    /// contribution was divided by `dt` along with the mouse's: a full deflection
    /// read **10 800 deg/s at 60 fps**, and read differently at every other frame
    /// rate. Only the delta half is a per-frame quantity, so only the delta half
    /// becomes a rate; see [`axis_parts`](Self::axis_parts).
    pub fn axis_snapshot(&self, dt: f64) -> BTreeMap<String, f32> {
        let k = if dt.is_finite() && dt > 0.0 {
            (1.0 / dt) as f32
        } else {
            1.0
        };
        self.map
            .axes_iter()
            .map(|(name, sources)| {
                let position = self.axis_positions.get(name).copied().unwrap_or(0.0);
                let delta = self.axis_deltas.get(name).copied().unwrap_or(0.0);
                (
                    name.to_string(),
                    Self::combine(position + delta * k, Self::has_delta_source(sources)),
                )
            })
            .collect()
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
            "the RAW sets survived — every public query answers from the resolved snapshot, so a release that cleared only the actions would look identical through all of them"
        );
        assert!(
            st.just_released("sprint"),
            "the release edge did not fire, so gameplay that ends an ability on it never ends the ability"
        );

        // …and the state is live again: a fresh press works.
        st.apply(&[InputEvent::Key {
            code: "KeyW".into(),
            pressed: true,
        }]);
        assert!(st.pressed("sprint") && st.just_pressed("sprint"));
    }
}
