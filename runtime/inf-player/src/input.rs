//! Input wiring for the windowed player (P9.3 item 1).
//!
//! Provides the default platformer [`InputMap`] (WASD/arrows/space → the
//! `left`/`right`/`jump`/`up`/`down` vocabulary the sample's Coyote blueprint
//! queries), TOML loading of a map beside the level, translation of winit key
//! events into `inf-input`'s `KeyboardEvent.code` convention, and the reduction
//! of a resolved [`InputState`] into the held-action set
//! [`runtime_sim`](crate::runtime_sim) ticks against.

use std::path::Path;

use winit::keyboard::KeyCode;

use inf_input::{InputMap, InputState};

use crate::runtime_sim::RuntimeInput;

/// The default platformer bindings. Actions match the sample Coyote graph
/// (`left`, `right`, `jump`); `up`/`down` and a `move_x`/`move_y` axis round out
/// the vocabulary for other content.
///
/// Gamepad + touch sources are bound alongside the keyboard: the South face
/// button → `jump`, the left stick → `move_x`/`move_y`. On touch platforms the
/// on-screen [`default_touch_controls`] emits exactly these gamepad events, so
/// touch reuses the same bindings with no separate mapping.
pub fn default_map() -> InputMap {
    use inf_input::{GamepadAxis, GamepadButton, MouseAxis, MouseButton};
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
        // ── P29.3: look, sprint, walk, crouch ──
        //
        // `look_x`/`look_y` are DEGREES per raw device unit; the delta reaches
        // the sim as degrees per SECOND (`InputState::axis_snapshot`), which is
        // exactly ALS's `AimYawRate`. 0.15 deg/count is a middle-of-the-road
        // desktop sensitivity; a project overrides the whole map in `input.toml`.
        // `look_y` inverts because the platform reports +y down and a look
        // control wants +pitch up — the binding says so rather than the engine
        // guessing (see `MouseAxis::Y`).
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
        .bind_mouse("aim", MouseButton::Right);
    m
}

/// The default on-screen touch layout (P14.1) for touch platforms (web /
/// Android): a **left virtual stick** driving `move_x`/`move_y` (via the left
/// gamepad stick) and a **right jump button** (the South face button). Both
/// primitives are demonstrated; a game with a different control scheme (e.g. the
/// 2D sample's digital `left`/`right`) builds its own [`TouchControls`] with
/// `TouchButton`s bound to the D-pad.
///
/// **Honest layout note:** rects/centres are in physical pixels at a landscape
/// ~1280×720 reference; a resolution-/safe-area-aware layout is a follow-up.
#[cfg(any(target_arch = "wasm32", target_os = "android"))]
pub fn default_touch_controls() -> inf_input::TouchControls {
    use inf_input::{GamepadAxis, GamepadButton, Rect, TouchButton, TouchControls, VirtualStick};
    let mut c = TouchControls::new();
    c.add_stick(
        GamepadAxis::LeftStickX,
        GamepadAxis::LeftStickY,
        VirtualStick::new([200.0, 520.0], 140.0),
    );
    c.add_button(
        GamepadButton::South,
        TouchButton::new(Rect::new([1080.0, 460.0], [1240.0, 620.0])),
    );
    c
}

/// Load an [`InputMap`] from `input.toml` beside `level_path` if present, else
/// the [`default_map`]. A malformed file falls back to the default (logged).
pub fn load_map_beside(level_path: &Path) -> InputMap {
    let toml_path = level_path.with_file_name("input.toml");
    match std::fs::read_to_string(&toml_path) {
        Ok(text) => match toml::from_str::<InputMap>(&text) {
            Ok(map) => {
                tracing::info!("inf-player: loaded input map from {}", toml_path.display());
                map
            }
            Err(e) => {
                tracing::warn!(
                    "inf-player: bad input map {}: {e}; using defaults",
                    toml_path.display()
                );
                default_map()
            }
        },
        Err(_) => default_map(),
    }
}

/// The actions currently held **and this frame's resolved axes**, as
/// [`RuntimeInput`] — what the blueprint host queries via `input.is_down` /
/// `input.just_pressed`, and what P29.3's movement component reads its intent
/// from.
///
/// `frame_dt` is the wall-clock seconds this frame covered; it is used for one
/// thing only, and it is used inside `inf-input` rather than here:
/// [`InputState::axis_snapshot`] converts the **delta** axes (mouse) into rates
/// so a fixed step can integrate them, and leaves the bounded ones alone. Doing
/// it there rather than at this call site is what keeps the rule from being
/// spelled once per host.
pub fn held_actions(state: &InputState, frame_dt: f64) -> RuntimeInput {
    let held: Vec<String> = state
        .map()
        .action_names()
        .filter(|name| state.pressed(name))
        .map(str::to_string)
        .collect();
    RuntimeInput::with_down(held).with_axes(state.axis_snapshot(frame_dt))
}

/// Map a winit [`KeyCode`] to the `KeyboardEvent.code` string `inf-input` uses.
/// Only the keys the desktop player cares about are mapped; anything else is
/// ignored (returns `None`).
pub fn keycode_to_code(code: KeyCode) -> Option<&'static str> {
    Some(match code {
        KeyCode::KeyA => "KeyA",
        KeyCode::KeyB => "KeyB",
        KeyCode::KeyC => "KeyC",
        KeyCode::KeyD => "KeyD",
        KeyCode::KeyE => "KeyE",
        KeyCode::KeyQ => "KeyQ",
        KeyCode::KeyR => "KeyR",
        KeyCode::KeyS => "KeyS",
        KeyCode::KeyW => "KeyW",
        KeyCode::ArrowLeft => "ArrowLeft",
        KeyCode::ArrowRight => "ArrowRight",
        KeyCode::ArrowUp => "ArrowUp",
        KeyCode::ArrowDown => "ArrowDown",
        KeyCode::Space => "Space",
        KeyCode::ShiftLeft | KeyCode::ShiftRight => "Shift",
        KeyCode::Enter => "Enter",
        KeyCode::Escape => "Escape",
        // ── P29.3: the keys the default movement bindings name ──
        KeyCode::KeyX => "KeyX",
        KeyCode::AltLeft => "AltLeft",
        KeyCode::ControlLeft | KeyCode::ControlRight => "Control",
        _ => return None,
    })
}

/// Map a winit mouse button onto the engine's [`inf_input::MouseButton`] (P29.3). Anything
/// past the two side buttons is ignored — a reserved slot is a wire promise, not
/// a device-mapping one, and a 12-button gaming mouse's extra buttons have no
/// stable meaning to bind to.
pub fn mouse_button(button: winit::event::MouseButton) -> Option<inf_input::MouseButton> {
    use winit::event::MouseButton as W;
    Some(match button {
        W::Left => inf_input::MouseButton::Left,
        W::Middle => inf_input::MouseButton::Middle,
        W::Right => inf_input::MouseButton::Right,
        W::Back => inf_input::MouseButton::Back,
        W::Forward => inf_input::MouseButton::Forward,
        W::Other(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_input::InputEvent;

    #[test]
    fn default_map_binds_the_coyote_vocabulary() {
        let m = default_map();
        assert!(m.action_sources("jump").is_some());
        assert!(m.action_sources("left").is_some());
        assert!(m.action_sources("right").is_some());
    }

    #[test]
    fn held_actions_reflects_pressed_keys() {
        let mut state = InputState::new(default_map());
        state.apply(&[InputEvent::Key {
            code: "KeyD".to_string(),
            pressed: true,
        }]);
        let held = held_actions(&state, 1.0 / 60.0);
        assert!(held.is_down("right"));
        assert!(!held.is_down("left"));
    }

    #[test]
    fn keycode_maps_movement_keys() {
        assert_eq!(keycode_to_code(KeyCode::Space), Some("Space"));
        assert_eq!(keycode_to_code(KeyCode::KeyW), Some("KeyW"));
        // P29.3 added three, and a binding whose key never maps is a binding
        // that silently does nothing.
        for key in ["Shift", "KeyC", "KeyX", "AltLeft"] {
            assert!(
                default_map()
                    .actions_iter()
                    .any(|(_, srcs)| srcs.iter().any(|s| matches!(
                        s,
                        inf_input::ActionSource::Key(c) if c == key
                    ))),
                "{key} is bound in the default map"
            );
        }
        assert_eq!(keycode_to_code(KeyCode::KeyX), Some("KeyX"));
        assert_eq!(keycode_to_code(KeyCode::AltLeft), Some("AltLeft"));
        assert_eq!(keycode_to_code(KeyCode::ShiftLeft), Some("Shift"));
        assert_eq!(keycode_to_code(KeyCode::KeyC), Some("KeyC"));
    }

    /// **The wiring, not the model** (P29.3). `inf-input`'s own arms prove that a
    /// mouse delta resolves and that a delta axis snapshots as a rate; this one
    /// proves the runtime *carries* it into the thing the fixed step reads.
    /// Before this wave `RuntimeInput` was a set of action names and a mouse
    /// delta had nowhere to arrive at all.
    #[test]
    fn a_mouse_delta_reaches_the_fixed_step_as_degrees_per_second() {
        let mut state = InputState::new(default_map());
        state.apply(&[
            InputEvent::MouseMotion {
                delta: [100.0, 0.0],
            },
            InputEvent::MouseButton {
                button: inf_input::MouseButton::Right,
                pressed: true,
            },
        ]);
        let held = held_actions(&state, 0.5);
        // 100 counts x 0.15 deg = 15 deg, over half a second = 30 deg/s.
        assert!(
            (held.axis("look_x") - 30.0).abs() < 1e-3,
            "look_x = {}",
            held.axis("look_x")
        );
        assert!(held.is_down("aim"), "the right button is bound to aim");

        // The control: a held key is a position and is NOT divided by dt, so the
        // rate conversion is a property of the source and not of the sampler.
        state.apply(&[InputEvent::Key {
            code: "KeyD".to_string(),
            pressed: true,
        }]);
        let held = held_actions(&state, 0.5);
        assert_eq!(held.axis("move_x"), 1.0, "a key axis is full deflection");
        assert_eq!(
            held.axis("look_x"),
            0.0,
            "and last frame's motion is not re-delivered"
        );
    }
}
