//! Gamepad poller over `gilrs` (behind the `gamepad` feature).
//!
//! This is the **device half** — deliberately thin and kept out of the tested
//! core. It translates `gilrs` events into [`InputEvent`]s that
//! [`InputState::apply`](crate::InputState::apply) consumes, so the mapping/edge/
//! deadzone logic stays device-free and fully unit-tested. Like the audio device
//! backend, the poller is not exercised in CI beyond construction fallback — a CI
//! runner has no gamepad, and `Gilrs` init may legitimately fail there, which
//! [`GamepadPoller::new`] reports as `None` rather than panicking.

use gilrs::{Axis, Button, EventType, Gilrs};

use crate::types::{GamepadAxis, GamepadButton, InputEvent};

/// Owns a `gilrs` context and drains it into engine [`InputEvent`]s.
pub struct GamepadPoller {
    gilrs: Gilrs,
}

impl GamepadPoller {
    /// Initialize the gamepad backend, or `None` if it is unavailable (no backend,
    /// a headless/CI environment). Never panics — a missing backend is a normal
    /// "no gamepads" state, not an error.
    pub fn new() -> Option<Self> {
        Gilrs::new().ok().map(|gilrs| Self { gilrs })
    }

    /// Drain all pending gamepad events, translated into [`InputEvent`]s. Buttons/
    /// axes `gilrs` does not model in our vocabulary are dropped. Call once per
    /// frame and feed the result to `InputState::apply`.
    pub fn poll(&mut self) -> Vec<InputEvent> {
        let mut out = Vec::new();
        while let Some(ev) = self.gilrs.next_event() {
            match ev.event {
                EventType::ButtonPressed(b, _) => {
                    if let Some(button) = map_button(b) {
                        out.push(InputEvent::GamepadButton {
                            button,
                            pressed: true,
                        });
                    }
                }
                EventType::ButtonReleased(b, _) => {
                    if let Some(button) = map_button(b) {
                        out.push(InputEvent::GamepadButton {
                            button,
                            pressed: false,
                        });
                    }
                }
                EventType::AxisChanged(a, value, _) => {
                    if let Some(axis) = map_axis(a) {
                        out.push(InputEvent::GamepadAxis { axis, value });
                    }
                }
                _ => {}
            }
        }
        out
    }
}

fn map_button(b: Button) -> Option<GamepadButton> {
    Some(match b {
        Button::South => GamepadButton::South,
        Button::East => GamepadButton::East,
        Button::North => GamepadButton::North,
        Button::West => GamepadButton::West,
        Button::LeftTrigger => GamepadButton::LeftBumper,
        Button::LeftTrigger2 => GamepadButton::LeftTrigger,
        Button::RightTrigger => GamepadButton::RightBumper,
        Button::RightTrigger2 => GamepadButton::RightTrigger,
        Button::Select => GamepadButton::Select,
        Button::Start => GamepadButton::Start,
        Button::Mode => GamepadButton::Mode,
        Button::LeftThumb => GamepadButton::LeftThumb,
        Button::RightThumb => GamepadButton::RightThumb,
        Button::DPadUp => GamepadButton::DPadUp,
        Button::DPadDown => GamepadButton::DPadDown,
        Button::DPadLeft => GamepadButton::DPadLeft,
        Button::DPadRight => GamepadButton::DPadRight,
        _ => return None,
    })
}

fn map_axis(a: Axis) -> Option<GamepadAxis> {
    Some(match a {
        Axis::LeftStickX => GamepadAxis::LeftStickX,
        Axis::LeftStickY => GamepadAxis::LeftStickY,
        Axis::RightStickX => GamepadAxis::RightStickX,
        Axis::RightStickY => GamepadAxis::RightStickY,
        Axis::LeftZ => GamepadAxis::LeftZ,
        Axis::RightZ => GamepadAxis::RightZ,
        _ => return None,
    })
}
