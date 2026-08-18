//! Infinity Engine input facade (Ring 0).
//!
//! An **action/axis mapping** layer shared by the editor and the runtime (roadmap
//! P9.1): the game speaks in named actions (`"jump"`) and axes (`"move_x"`), and
//! an [`InputMap`] binds those names to concrete sources — keyboard keys, gamepad
//! buttons, and gamepad analog axes. [`InputState`] resolves the map against live
//! device state each frame, giving edge-detected action queries
//! ([`pressed`](InputState::pressed) / [`just_pressed`](InputState::just_pressed)
//! / [`just_released`](InputState::just_released)) and deadzoned, scaled axis
//! values.
//!
//! # Pure core, device edge
//!
//! The whole mapping/edge/deadzone engine is pure: [`InputState::apply`] takes a
//! frame's [`InputEvent`]s and updates the resolved snapshot — no device, no
//! clock, no threads — so it is fully unit-tested and deterministic. Producing
//! those events is the device half's job. Keyboard events are already produced by
//! the editor Simulate loop / runtime windowing in the **`KeyboardEvent.code`**
//! convention (`"KeyW"`, `"Space"`, …) this crate's keyboard sources use. Gamepad
//! events come from the optional [`poller::GamepadPoller`] (behind the `gamepad`
//! feature, over `gilrs`), which — like the audio device backend — is kept out of
//! the default build and untested in CI beyond construction fallback, so the core
//! stays dependency-light and portable.
//!
//! # Serde
//!
//! [`InputMap`] is `Serialize`/`Deserialize` (deterministic `BTreeMap` ordering)
//! because it will live in project settings (P9); a round-trip is byte-stable.

mod map;
mod state;
pub mod touch;
mod types;

#[cfg(feature = "gamepad")]
pub mod poller;

pub use map::InputMap;
pub use state::InputState;
pub use touch::{Rect, TouchButton, TouchControls, VirtualStick};
pub use types::{
    ActionSource, AxisSource, GamepadAxis, GamepadButton, InputEvent, MouseAxis, MouseButton,
    TouchPhase,
};
