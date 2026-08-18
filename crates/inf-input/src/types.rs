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

/// The lifecycle phase of a touch point (P14.1). Mirrors the winit / W3C
/// `TouchPhase` set so a platform's raw touch stream maps 1:1 onto
/// [`InputEvent::Touch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TouchPhase {
    /// A finger touched down.
    Started,
    /// A tracked finger moved.
    Moved,
    /// A finger lifted cleanly.
    Ended,
    /// The touch was cancelled by the system (e.g. a gesture took over).
    Cancelled,
}

/// A mouse button (P29.3). The five buttons every platform agrees on, in the
/// W3C `MouseEvent.button` order (0 = Left, 1 = Middle, 2 = Right) so a web or
/// winit backend maps onto it without a lookup table, plus the two side buttons.
///
/// # Frozen, append-only, with reserved slots
///
/// This reaches an [`InputMap`](crate::InputMap) and therefore a project's saved
/// settings, so the wire law applies: **append only**. The two reserved slots
/// exist because a mouse with more buttons is a real thing and a saved map that
/// mentions one must be *refused by name* rather than renumber `Back` into
/// `Forward`. They are unit variants like their neighbours, so a map naming one
/// decodes and is rejected by the reader that reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MouseButton {
    /// The primary button (`MouseEvent.button == 0`).
    Left,
    /// The wheel button (`MouseEvent.button == 1`).
    Middle,
    /// The secondary button (`MouseEvent.button == 2`).
    Right,
    /// The first side button ("back").
    Back,
    /// The second side button ("forward").
    Forward,
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved5,
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved6,
}

impl MouseButton {
    /// `Some(index)` if this is a reserved slot a newer build wrote, else `None`.
    ///
    /// A reserved slot that no reader ever asks about is a comment rather than a
    /// slot, which is why this accessor exists and why the pin exercises it.
    pub fn reserved_slot(self) -> Option<u8> {
        match self {
            MouseButton::Reserved5 => Some(5),
            MouseButton::Reserved6 => Some(6),
            _ => None,
        }
    }
}

/// A mouse analog axis (P29.3) — the **relative** motion channels.
///
/// Unlike a stick, every one of these is a *delta accumulated over one frame*,
/// not a position: it is zero when the mouse is still and it is cleared when the
/// frame commits. That difference is why [`InputState`](crate::InputState) does
/// not clamp an axis that names one (see `axis`), and it is the whole reason
/// `AimYawRate` — degrees of camera yaw per second — can be derived at all.
///
/// Frozen and append-only for the same reason as [`MouseButton`], with the same
/// reserved-slot argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MouseAxis {
    /// Horizontal motion this frame, in the platform's raw units (pixels or
    /// counts). Positive is right.
    X,
    /// Vertical motion this frame. **Positive is down**, matching the screen
    /// convention every platform reports; a look binding inverts with a negative
    /// `scale` rather than by the engine guessing.
    Y,
    /// Horizontal wheel/tilt motion this frame.
    WheelX,
    /// Vertical wheel motion this frame. Positive is "scroll up / away".
    WheelY,
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved4,
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved5,
}

impl MouseAxis {
    /// `Some(index)` if this is a reserved slot a newer build wrote, else `None`.
    pub fn reserved_slot(self) -> Option<u8> {
        match self {
            MouseAxis::Reserved4 => Some(4),
            MouseAxis::Reserved5 => Some(5),
            _ => None,
        }
    }
}

/// A raw input event fed to [`InputState::apply`](crate::InputState::apply). The
/// editor's Simulate loop and the runtime both translate their platform events
/// into this shape; the gamepad poller produces the gamepad variants; the touch
/// platforms (Android / iOS / web) produce [`Touch`](InputEvent::Touch).
///
/// Keyboard keys use the **`KeyboardEvent.code`** convention (physical, layout-
/// independent — `"KeyW"`, `"Space"`, `"ArrowLeft"`), the same strings the editor
/// Simulate mapping already speaks (`editor/studio/src/stores/simStore.ts`).
///
/// [`Touch`](InputEvent::Touch) events are **not** map sources on their own —
/// [`InputState`](crate::InputState) ignores them. On-screen controls
/// ([`TouchControls`](crate::touch::TouchControls)) translate them into the
/// gamepad axis/button variants that the [`InputMap`](crate::InputMap) already
/// resolves, so touch reuses the whole action/axis pipeline.
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
    /// A touch point changed. `id` identifies the finger across its lifetime;
    /// `position` is in the same coordinate space the on-screen controls were
    /// authored in (physical pixels, origin top-left, in the player).
    Touch {
        id: u64,
        phase: TouchPhase,
        position: [f32; 2],
    },
    /// A mouse button changed state (P29.3). Level, like a key.
    MouseButton { button: MouseButton, pressed: bool },
    /// The mouse **moved**, by `delta` in the platform's raw units (pixels or
    /// device counts), `+x` right and `+y` down (P29.3).
    ///
    /// Relative, not absolute: several of these in one frame accumulate, and the
    /// total is consumed and cleared when the frame commits. A backend should
    /// prefer the *unaccelerated device* delta (winit's `DeviceEvent::MouseMotion`)
    /// over a window-cursor delta, because the cursor is clipped by the screen
    /// edge and a look control that stops at the edge of the monitor is the
    /// oldest bug in first-person games.
    MouseMotion { delta: [f32; 2] },
    /// The wheel moved, by `delta` in lines/notches (P29.3). Accumulated and
    /// cleared exactly like [`MouseMotion`](InputEvent::MouseMotion).
    MouseWheel { delta: [f32; 2] },
}

/// A source that can trigger a digital **action**.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSource {
    /// A keyboard key (a `KeyboardEvent.code` string).
    Key(String),
    /// A gamepad button.
    GamepadButton(GamepadButton),
    /// A mouse button (P29.3) — appended, because this enum is saved in a
    /// project's `input.toml`.
    MouseButton(MouseButton),
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
    /// A mouse motion/wheel delta (P29.3): contributes `delta × scale`, where
    /// `delta` is **this frame's accumulated motion** in raw device units.
    ///
    /// `scale` is therefore a *sensitivity*, and it is the one place the raw unit
    /// is converted: bind `look_x` to `MouseAxis::X` at `0.15` and the axis reads
    /// degrees of yaw per frame. An axis that names one of these is **not
    /// clamped** to `[-1, 1]` — see [`InputState::axis`](crate::InputState::axis).
    MouseAxis { axis: MouseAxis, scale: f32 },
    /// A mouse button as a digital axis contribution (`scale` while held).
    MouseButton { button: MouseButton, scale: f32 },
}
