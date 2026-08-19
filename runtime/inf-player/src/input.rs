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

/// The default bindings — **re-exported from Ring 0** (P29.6).
///
/// The table used to live here, which was survivable only while the editor's
/// Simulate could not carry an axis. Both hosts read the same one now; see
/// [`inf_input::default_map`] for the table and for why the mouse scales are
/// degrees per count.
pub use inf_input::default_map;

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

/// Load the locomotion camera's tunables from `camera.toml` beside `level_path`
/// if present, else the ported ALS table. A malformed file falls back to the
/// defaults (logged) — the same shape, and the same rationale, as
/// [`load_map_beside`] directly above.
///
/// **Here rather than in `inf-ecs`** (P29.6 audit, A9). The first cut put this
/// on `CameraTuning` itself, which gave `inf-ecs` — the world model — the only
/// `std::fs` call in the crate, to read a host's file, for a single caller that
/// lives in this file's own crate. The table's *parsing* is Ring 0 (it is a
/// property of the type); reading a path is a host's job, and the neighbour it
/// claims to mirror was always here.
pub fn load_camera_beside(level_path: &Path) -> inf_ecs::camera::CameraTuning {
    let path = level_path.with_file_name("camera.toml");
    match std::fs::read_to_string(&path) {
        Ok(text) => match inf_ecs::camera::CameraTuning::from_toml(&text) {
            Ok(t) => {
                tracing::info!("inf-player: loaded camera table from {}", path.display());
                t
            }
            Err(e) => {
                tracing::warn!(
                    "inf-player: bad camera table {}: {e}; using defaults",
                    path.display()
                );
                inf_ecs::camera::CameraTuning::default()
            }
        },
        Err(_) => inf_ecs::camera::CameraTuning::default(),
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
    let (held, axes) = state.resolved(frame_dt);
    RuntimeInput::with_down(held).with_axes(axes)
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
        KeyCode::KeyF => "KeyF",
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

    /// **Every action the movement intent reads is BOUND** (P29.6).
    ///
    /// `inf_ecs::movement::actions` is the vocabulary and `inf_input::default_map`
    /// is the table, and they live in two crates that cannot name each other —
    /// `inf-input` must not depend on `inf-ecs`. So the claim that they agree is
    /// held here, in the one crate that names both, and it is the arm rather than
    /// a comment because a binding that is missing is a control that **silently
    /// does nothing**: exactly what `move_y`, `move_up`, `roll` and `dive` were
    /// doing until this wave's course tried to drive a character with them.
    #[test]
    fn every_movement_action_the_intent_reads_is_bound() {
        use inf_ecs::movement::actions as a;
        let m = default_map();
        // **The list is EXTRACTED, not restated** (P29.6 audit, A14). The first
        // cut wrote the thirteen names out by hand, which catches a binding that
        // is deleted and cannot catch the thing its own doc comment names: a
        // *fourteenth* action added to `MovementIntent::from_actions` and left
        // unbound. That is precisely how `move_y`, `move_up`, `roll` and `dive`
        // came to do nothing — they were read and never bound — so the arm has
        // to read the reader.
        //
        // The scope is `from_actions`' body: it is the one function that turns
        // named input into an intent, and `commands/sim.rs`'s own tunable arm
        // sets the precedent for extracting a list from the source it guards.
        const MODEL: &str = include_str!("../../../crates/inf-ecs/src/movement.rs");
        let src = MODEL.replace("\r\n", "\n");
        let start = src
            .find("    pub fn from_actions(")
            .expect("the intent still has a `from_actions`");
        let body = &src[start..];
        let end = body.find("\n    }\n").expect("its closing brace");
        let body = &body[..end];
        let read: std::collections::BTreeSet<&str> = body
            .split("actions::")
            .skip(1)
            .filter_map(|rest| rest.split(')').next())
            .filter(|n| n.chars().all(|c| c.is_ascii_uppercase() || c == '_'))
            .collect();
        assert!(
            read.len() >= 13,
            "only {} action constants were extracted from `from_actions` — the \
             extraction broke, and an arm that greps nothing passes everything: \
             {read:?}",
            read.len()
        );
        // Each extracted CONSTANT name resolves to its value through the
        // vocabulary, so a rename fails to compile rather than silently
        // shrinking the set.
        let value_of = |c: &str| -> &'static str {
            match c {
                "MOVE_X" => a::MOVE_X,
                "MOVE_Y" => a::MOVE_Y,
                "LOOK_X" => a::LOOK_X,
                "LOOK_Y" => a::LOOK_Y,
                "MOVE_UP" => a::MOVE_UP,
                "SPRINT" => a::SPRINT,
                "WALK" => a::WALK,
                "AIM" => a::AIM,
                "JUMP" => a::JUMP,
                "CROUCH" => a::CROUCH,
                "PRONE" => a::PRONE,
                "ROLL" => a::ROLL,
                "DIVE" => a::DIVE,
                other => panic!(
                    "`MovementIntent::from_actions` reads `actions::{other}`, which \
                     this arm has never heard of — add it to `default_map` and to \
                     the line below, or the control does nothing and nothing else \
                     in the tree would say so"
                ),
            }
        };
        for c in &read {
            let name = value_of(c);
            assert!(
                m.axis_names().any(|n| n == name) || m.action_names().any(|n| n == name),
                "`{name}` is read by the movement intent and bound to nothing — \
                 the control does nothing and nothing else in the tree would say so"
            );
        }
        // The five analog ones are AXES and the eight discrete ones are ACTIONS:
        // a `jump` bound as an axis would satisfy the sweep above and never
        // reach `pressed`.
        for axis in [a::MOVE_X, a::MOVE_Y, a::LOOK_X, a::LOOK_Y, a::MOVE_UP] {
            assert!(
                m.axis_names().any(|n| n == axis),
                "the `{axis}` axis is unbound — the control does nothing and \
                 nothing else in the tree would say so"
            );
        }
        for action in [
            a::SPRINT,
            a::WALK,
            a::AIM,
            a::JUMP,
            a::CROUCH,
            a::PRONE,
            a::ROLL,
            a::DIVE,
        ] {
            assert!(
                m.action_names().any(|n| n == action),
                "the `{action}` action is unbound"
            );
        }
        // …and every KEY the table names really maps from a winit keycode, or the
        // binding is bound to a string the player can never produce.
        let mut keys: Vec<String> = Vec::new();
        for (_, sources) in m.actions_iter() {
            for s in sources {
                if let inf_input::ActionSource::Key(c) = s {
                    keys.push(c.clone());
                }
            }
        }
        for (_, sources) in m.axes_iter() {
            for s in sources {
                if let inf_input::AxisSource::Key { code, .. } = s {
                    keys.push(code.clone());
                }
            }
        }
        keys.sort();
        keys.dedup();
        let reachable: std::collections::BTreeSet<&str> = [
            KeyCode::KeyA,
            KeyCode::KeyB,
            KeyCode::KeyC,
            KeyCode::KeyD,
            KeyCode::KeyE,
            KeyCode::KeyF,
            KeyCode::KeyQ,
            KeyCode::KeyR,
            KeyCode::KeyS,
            KeyCode::KeyW,
            KeyCode::KeyX,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::Space,
            KeyCode::ShiftLeft,
            KeyCode::Enter,
            KeyCode::Escape,
            KeyCode::AltLeft,
            KeyCode::ControlLeft,
        ]
        .into_iter()
        .filter_map(keycode_to_code)
        .collect();
        for k in &keys {
            assert!(
                reachable.contains(k.as_str()),
                "`{k}` is bound but no winit keycode maps to it — the binding \
                 is unreachable from a keyboard"
            );
        }
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
