//! On-screen touch controls (P14.1): a **virtual gamepad** for touch platforms
//! (Android / iOS / web).
//!
//! Touch does not bind to actions directly. Instead a [`TouchControls`] overlay
//! of [`VirtualStick`]s and [`TouchButton`]s consumes the raw
//! [`InputEvent::Touch`] stream and **translates** it into the gamepad
//! axis/button [`InputEvent`]s that the [`InputMap`](crate::InputMap) already
//! resolves — so a touch build reuses the whole action/axis/deadzone pipeline
//! with no new mapping model, and the same game logic runs on desktop, gamepad,
//! and touch. A game that wants touch just binds its actions/axes to the
//! [`GamepadButton`]/[`GamepadAxis`] the controls emit (the player wires a
//! default: left stick → `move_x`/`move_y`, the South button → `jump`).
//!
//! # Pure core, tested
//!
//! Everything here is pure geometry + a touch-id → grabbed-control bookkeeping
//! map: [`TouchControls::process`] takes a frame's events and returns the
//! synthetic gamepad events, with no device, clock, or platform dependency. The
//! player feeds it winit `Touch` events (cfg'd to touch platforms) and applies
//! the result to its [`InputState`](crate::InputState) alongside the real events;
//! desktop never constructs it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{GamepadAxis, GamepadButton, InputEvent, TouchPhase};

/// An axis-aligned rectangle in the control coordinate space (the player authors
/// controls in physical pixels, origin top-left). `min` is the top-left corner,
/// `max` the bottom-right.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Rect {
    /// A rectangle from its two corners (each component of `min` should be ≤ the
    /// matching `max`).
    pub fn new(min: [f32; 2], max: [f32; 2]) -> Self {
        Self { min, max }
    }

    /// A rectangle centred at `center` with the given full `size`.
    pub fn from_center_size(center: [f32; 2], size: [f32; 2]) -> Self {
        let hx = size[0] * 0.5;
        let hy = size[1] * 0.5;
        Self {
            min: [center[0] - hx, center[1] - hy],
            max: [center[0] + hx, center[1] + hy],
        }
    }

    /// Whether `p` lies inside the rectangle (min edges inclusive, max exclusive).
    pub fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.min[0] && p[0] < self.max[0] && p[1] >= self.min[1] && p[1] < self.max[1]
    }
}

/// A virtual analog stick: a touch inside `radius` of `center` yields an axis
/// pair in `[-1, 1]` proportional to the offset from the centre (clamped to the
/// unit disc). Screen space is y-down, so the raw y component is `+` downward —
/// invert it with a negative axis scale in the [`InputMap`](crate::InputMap) when
/// binding "up = forward".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VirtualStick {
    pub center: [f32; 2],
    pub radius: f32,
}

impl VirtualStick {
    /// A stick centred at `center` with activation `radius` (clamped to a small
    /// positive minimum so `axis` never divides by zero).
    pub fn new(center: [f32; 2], radius: f32) -> Self {
        Self {
            center,
            radius: radius.max(f32::EPSILON),
        }
    }

    /// Whether `pos` is within the stick's activation radius (so a touch-down
    /// there grabs the stick).
    pub fn contains(&self, pos: [f32; 2]) -> bool {
        let dx = pos[0] - self.center[0];
        let dy = pos[1] - self.center[1];
        dx * dx + dy * dy <= self.radius * self.radius
    }

    /// Map an absolute touch position to a `[-1, 1]` axis pair: the offset from
    /// the centre divided by the radius, clamped to magnitude 1 (the unit disc,
    /// so diagonals never exceed full deflection).
    pub fn axis(&self, pos: [f32; 2]) -> [f32; 2] {
        let mut x = (pos[0] - self.center[0]) / self.radius;
        let mut y = (pos[1] - self.center[1]) / self.radius;
        let mag = (x * x + y * y).sqrt();
        if mag > 1.0 {
            x /= mag;
            y /= mag;
        }
        [x, y]
    }
}

/// A rectangular touch button that reads as a held [`GamepadButton`] (→ an action
/// through the [`InputMap`](crate::InputMap)) while a finger is inside it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TouchButton {
    pub rect: Rect,
}

impl TouchButton {
    /// A button covering `rect`.
    pub fn new(rect: Rect) -> Self {
        Self { rect }
    }

    /// Whether `pos` is inside the button.
    pub fn contains(&self, pos: [f32; 2]) -> bool {
        self.rect.contains(pos)
    }
}

/// One stick binding: the two gamepad axes it drives + its geometry.
#[derive(Clone, Copy, Debug)]
struct StickBinding {
    x: GamepadAxis,
    y: GamepadAxis,
    geom: VirtualStick,
}

/// One button binding: the gamepad button it emits + its geometry.
#[derive(Clone, Copy, Debug)]
struct ButtonBinding {
    button: GamepadButton,
    geom: TouchButton,
}

/// Which control a live touch grabbed on touch-down (so `Moved`/`Ended` route to
/// the same control regardless of where the finger drifts).
#[derive(Clone, Copy, Debug)]
enum Grab {
    Stick(usize),
    Button(usize),
}

/// The on-screen control set. Register sticks/buttons once, then call
/// [`process`](TouchControls::process) each frame with that frame's raw events;
/// it returns synthetic [`GamepadAxis`](InputEvent::GamepadAxis) /
/// [`GamepadButton`](InputEvent::GamepadButton) events to feed
/// [`InputState::apply`](crate::InputState::apply) — mixed in with any real
/// keyboard/gamepad events.
#[derive(Clone, Debug, Default)]
pub struct TouchControls {
    sticks: Vec<StickBinding>,
    buttons: Vec<ButtonBinding>,
    active: BTreeMap<u64, Grab>,
}

impl TouchControls {
    /// An empty control set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a virtual stick driving the `x`/`y` gamepad axes.
    pub fn add_stick(&mut self, x: GamepadAxis, y: GamepadAxis, geom: VirtualStick) -> &mut Self {
        self.sticks.push(StickBinding { x, y, geom });
        self
    }

    /// Add a touch button emitting `button` while pressed.
    pub fn add_button(&mut self, button: GamepadButton, geom: TouchButton) -> &mut Self {
        self.buttons.push(ButtonBinding { button, geom });
        self
    }

    /// Number of live (grabbed) touches — for debug/HUD.
    pub fn active_touches(&self) -> usize {
        self.active.len()
    }

    /// Translate this frame's events into synthetic gamepad events. Only
    /// [`InputEvent::Touch`] events are consumed; anything else is ignored (the
    /// caller applies those directly). The returned events set gamepad axes/
    /// buttons that the [`InputMap`](crate::InputMap) resolves to actions/axes.
    pub fn process(&mut self, events: &[InputEvent]) -> Vec<InputEvent> {
        let mut out = Vec::new();
        for ev in events {
            let InputEvent::Touch {
                id,
                phase,
                position,
            } = ev
            else {
                continue;
            };
            match phase {
                TouchPhase::Started => self.on_down(*id, *position, &mut out),
                TouchPhase::Moved => self.on_move(*id, *position, &mut out),
                TouchPhase::Ended | TouchPhase::Cancelled => self.on_up(*id, &mut out),
            }
        }
        out
    }

    fn on_down(&mut self, id: u64, pos: [f32; 2], out: &mut Vec<InputEvent>) {
        // Buttons sit above sticks, so test them first.
        if let Some(i) = self.buttons.iter().position(|b| b.geom.contains(pos)) {
            self.active.insert(id, Grab::Button(i));
            out.push(InputEvent::GamepadButton {
                button: self.buttons[i].button,
                pressed: true,
            });
            return;
        }
        if let Some(i) = self.sticks.iter().position(|s| s.geom.contains(pos)) {
            self.active.insert(id, Grab::Stick(i));
            self.emit_stick(i, pos, out);
        }
    }

    fn on_move(&mut self, id: u64, pos: [f32; 2], out: &mut Vec<InputEvent>) {
        if let Some(Grab::Stick(i)) = self.active.get(&id).copied() {
            self.emit_stick(i, pos, out);
        }
    }

    fn on_up(&mut self, id: u64, out: &mut Vec<InputEvent>) {
        match self.active.remove(&id) {
            Some(Grab::Button(i)) => out.push(InputEvent::GamepadButton {
                button: self.buttons[i].button,
                pressed: false,
            }),
            Some(Grab::Stick(i)) => {
                // Recentre the stick to zero on release.
                let s = &self.sticks[i];
                out.push(InputEvent::GamepadAxis {
                    axis: s.x,
                    value: 0.0,
                });
                out.push(InputEvent::GamepadAxis {
                    axis: s.y,
                    value: 0.0,
                });
            }
            None => {}
        }
    }

    fn emit_stick(&self, i: usize, pos: [f32; 2], out: &mut Vec<InputEvent>) {
        let s = &self.sticks[i];
        let [x, y] = s.geom.axis(pos);
        out.push(InputEvent::GamepadAxis {
            axis: s.x,
            value: x,
        });
        out.push(InputEvent::GamepadAxis {
            axis: s.y,
            value: y,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputMap, InputState};

    #[test]
    fn rect_contains_is_min_inclusive_max_exclusive() {
        let r = Rect::new([0.0, 0.0], [10.0, 10.0]);
        assert!(r.contains([0.0, 0.0]));
        assert!(r.contains([9.9, 9.9]));
        assert!(!r.contains([10.0, 5.0]));
        assert!(!r.contains([-0.1, 5.0]));
    }

    #[test]
    fn rect_from_center_size() {
        let r = Rect::from_center_size([50.0, 50.0], [20.0, 10.0]);
        assert_eq!(r.min, [40.0, 45.0]);
        assert_eq!(r.max, [60.0, 55.0]);
    }

    #[test]
    fn stick_axis_is_offset_over_radius_clamped_to_unit_disc() {
        let s = VirtualStick::new([100.0, 100.0], 50.0);
        // Centre → no deflection.
        assert_eq!(s.axis([100.0, 100.0]), [0.0, 0.0]);
        // Full right at the radius → +1 on x.
        assert_eq!(s.axis([150.0, 100.0]), [1.0, 0.0]);
        // Screen-down at the radius → +1 on y (y-down convention).
        assert_eq!(s.axis([100.0, 150.0]), [0.0, 1.0]);
        // Way beyond the radius diagonally → clamped to magnitude 1.
        let [x, y] = s.axis([1000.0, 1000.0]);
        let mag = (x * x + y * y).sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-5,
            "magnitude clamped to 1, got {mag}"
        );
    }

    #[test]
    fn stick_contains_within_radius() {
        let s = VirtualStick::new([0.0, 0.0], 10.0);
        assert!(s.contains([6.0, 8.0])); // exactly on the edge (magnitude 10)
        assert!(!s.contains([8.0, 8.0])); // magnitude ~11.3 > 10
    }

    #[test]
    fn button_down_and_up_emit_press_and_release() {
        let mut c = TouchControls::new();
        c.add_button(
            GamepadButton::South,
            TouchButton::new(Rect::new([0.0, 0.0], [100.0, 100.0])),
        );
        let down = c.process(&[InputEvent::Touch {
            id: 1,
            phase: TouchPhase::Started,
            position: [50.0, 50.0],
        }]);
        assert_eq!(
            down,
            vec![InputEvent::GamepadButton {
                button: GamepadButton::South,
                pressed: true
            }]
        );
        assert_eq!(c.active_touches(), 1);
        let up = c.process(&[InputEvent::Touch {
            id: 1,
            phase: TouchPhase::Ended,
            position: [50.0, 50.0],
        }]);
        assert_eq!(
            up,
            vec![InputEvent::GamepadButton {
                button: GamepadButton::South,
                pressed: false
            }]
        );
        assert_eq!(c.active_touches(), 0);
    }

    #[test]
    fn touch_outside_any_control_grabs_nothing() {
        let mut c = TouchControls::new();
        c.add_button(
            GamepadButton::South,
            TouchButton::new(Rect::new([0.0, 0.0], [10.0, 10.0])),
        );
        let out = c.process(&[InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Started,
            position: [500.0, 500.0],
        }]);
        assert!(out.is_empty());
        assert_eq!(c.active_touches(), 0);
    }

    #[test]
    fn stick_move_updates_axis_and_release_recenters() {
        let mut c = TouchControls::new();
        c.add_stick(
            GamepadAxis::LeftStickX,
            GamepadAxis::LeftStickY,
            VirtualStick::new([100.0, 100.0], 50.0),
        );
        // Down inside the stick.
        c.process(&[InputEvent::Touch {
            id: 3,
            phase: TouchPhase::Started,
            position: [100.0, 100.0],
        }]);
        // Move fully right.
        let mv = c.process(&[InputEvent::Touch {
            id: 3,
            phase: TouchPhase::Moved,
            position: [150.0, 100.0],
        }]);
        assert!(mv.contains(&InputEvent::GamepadAxis {
            axis: GamepadAxis::LeftStickX,
            value: 1.0
        }));
        // Release recentres both axes to zero.
        let up = c.process(&[InputEvent::Touch {
            id: 3,
            phase: TouchPhase::Ended,
            position: [150.0, 100.0],
        }]);
        assert!(up.contains(&InputEvent::GamepadAxis {
            axis: GamepadAxis::LeftStickX,
            value: 0.0
        }));
        assert!(up.contains(&InputEvent::GamepadAxis {
            axis: GamepadAxis::LeftStickY,
            value: 0.0
        }));
    }

    #[test]
    fn touch_drives_actions_through_the_input_map() {
        // The end-to-end proof: touch → synthetic gamepad events → InputMap →
        // resolved actions/axes, identical to a physical pad.
        let mut map = InputMap::new();
        map.bind_button("jump", GamepadButton::South)
            .bind_axis_stick("move_x", GamepadAxis::LeftStickX, 1.0);
        let mut state = InputState::new(map);

        let mut controls = TouchControls::new();
        controls
            .add_stick(
                GamepadAxis::LeftStickX,
                GamepadAxis::LeftStickY,
                VirtualStick::new([100.0, 100.0], 50.0),
            )
            .add_button(
                GamepadButton::South,
                TouchButton::new(Rect::new([300.0, 0.0], [400.0, 100.0])),
            );

        // Finger 1 presses the jump button; finger 2 pushes the stick right.
        let frame = [
            InputEvent::Touch {
                id: 1,
                phase: TouchPhase::Started,
                position: [350.0, 50.0],
            },
            InputEvent::Touch {
                id: 2,
                phase: TouchPhase::Started,
                position: [150.0, 100.0], // fully right at the radius (offset == radius)
            },
        ];
        let synth = controls.process(&frame);
        state.apply(&synth);
        assert!(state.just_pressed("jump"), "touch button → jump action");
        assert!(
            (state.axis("move_x") - 1.0).abs() < 1e-5,
            "stick → move_x axis, got {}",
            state.axis("move_x")
        );

        // Lift both fingers → action releases, axis recentres.
        let release = controls.process(&[
            InputEvent::Touch {
                id: 1,
                phase: TouchPhase::Ended,
                position: [350.0, 50.0],
            },
            InputEvent::Touch {
                id: 2,
                phase: TouchPhase::Ended,
                position: [140.0, 100.0],
            },
        ]);
        state.apply(&release);
        assert!(!state.pressed("jump"));
        assert_eq!(state.axis("move_x"), 0.0);
    }
}
