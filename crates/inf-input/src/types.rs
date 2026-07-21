//! Device-agnostic input vocabulary: gamepad buttons/axes, raw events, and the
//! map's source kinds. Kept free of `gilrs` so the mapping core and its serde
//! surface never depend on the (optional) gamepad backend — the poller
//! (`crate::poller`, behind the `gamepad` feature) translates `gilrs` types into
//! these.

use serde::{Deserialize, Serialize};

/// A gamepad face/shoulder/dpad button, in the layout-neutral naming `gilrs`
/// uses (South = the bottom face button, "A"/"✕"). Serialized by name so an
/// `InputMap` in project settings is readable and stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GamepadButton {
    South,
    East,
    North,
    West,
    LeftBumper,
    RightBumper,
    LeftTrigger,
    RightTrigger,
    Select,
    Start,
    Mode,
    LeftThumb,
    RightThumb,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

/// A gamepad analog axis. Trigger axes (`LeftZ`/`RightZ`) are included so a
/// trigger can drive an analog axis as well as act as a button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftZ,
    RightZ,
}

/// A raw input event fed to [`InputState::apply`](crate::InputState::apply). The
/// editor's Simulate loop and the runtime both translate their platform events
/// into this shape; the gamepad poller produces the gamepad variants.
///
/// Keyboard keys use the **`KeyboardEvent.code`** convention (physical, layout-
/// independent — `"KeyW"`, `"Space"`, `"ArrowLeft"`), the same strings the editor
/// Simulate mapping already speaks (`editor/studio/src/stores/simStore.ts`).
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    /// A keyboard key changed state. `code` is a `KeyboardEvent.code` string.
    Key { code: String, pressed: bool },
    /// A gamepad button changed state.
    GamepadButton {
        button: GamepadButton,
        pressed: bool,
    },
    /// A gamepad analog axis moved. `value` is roughly `[-1, 1]`.
    GamepadAxis { axis: GamepadAxis, value: f32 },
}

/// A source that can trigger a digital **action**.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSource {
    /// A keyboard key (a `KeyboardEvent.code` string).
    Key(String),
    /// A gamepad button.
    GamepadButton(GamepadButton),
}

/// A source that contributes to an analog **axis**, each with its own scale (a
/// negative scale inverts the source — e.g. bind `KeyA` at `-1` and `KeyD` at
/// `+1` to make a keyboard "move_x" axis).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AxisSource {
    /// A keyboard key: contributes `scale` while held, `0` otherwise.
    Key { code: String, scale: f32 },
    /// A gamepad analog axis: contributes `raw × scale` (after deadzone).
    GamepadAxis { axis: GamepadAxis, scale: f32 },
    /// A gamepad button as a digital axis contribution (`scale` while held).
    GamepadButton { button: GamepadButton, scale: f32 },
}
