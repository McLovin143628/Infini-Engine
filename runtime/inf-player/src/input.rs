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
/// (`left`, `right`, `jump`); `up`/`down` and a `move_x` axis round out the
/// vocabulary for other content.
pub fn default_map() -> InputMap {
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
        .bind_axis_key("move_x", "KeyD", 1.0)
        .bind_axis_key("move_x", "ArrowRight", 1.0)
        .bind_axis_key("move_x", "KeyA", -1.0)
        .bind_axis_key("move_x", "ArrowLeft", -1.0);
    m
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

/// The set of actions currently held, as [`RuntimeInput`] — what the blueprint
/// host queries via `input.is_down` / `input.just_pressed`.
pub fn held_actions(state: &InputState) -> RuntimeInput {
    let held: Vec<String> = state
        .map()
        .action_names()
        .filter(|name| state.pressed(name))
        .map(str::to_string)
        .collect();
    RuntimeInput::with_down(held)
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
        _ => return None,
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
        let held = held_actions(&state);
        assert!(held.is_down("right"));
        assert!(!held.is_down("left"));
    }

    #[test]
    fn keycode_maps_movement_keys() {
        assert_eq!(keycode_to_code(KeyCode::Space), Some("Space"));
        assert_eq!(keycode_to_code(KeyCode::KeyW), Some("KeyW"));
    }
}
