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
