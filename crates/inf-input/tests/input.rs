//! Tests for the pure input core: action resolution, edges, deadzone/scale math,
//! and `InputMap` serde round-trip. No device is involved.

use inf_input::{AxisSource, GamepadAxis, GamepadButton, InputEvent, InputMap, InputState};

fn key(code: &str, pressed: bool) -> InputEvent {
    InputEvent::Key {
        code: code.to_string(),
        pressed,
    }
}

/// A map mirroring the editor's Simulate controls plus a gamepad binding.
fn platformer_map() -> InputMap {
    let mut m = InputMap::new();
    m.bind_key("jump", "Space")
        .bind_key("jump", "KeyW")
        .bind_button("jump", GamepadButton::South);
    // move_x: A/D on the keyboard, plus the left stick X on a gamepad.
    m.bind_axis_key("move_x", "KeyA", -1.0)
        .bind_axis_key("move_x", "KeyD", 1.0)
        .bind_axis_stick("move_x", GamepadAxis::LeftStickX, 1.0);
    m
}

#[test]
fn action_resolves_from_any_bound_source() {
    let mut state = InputState::new(platformer_map());
    assert!(!state.pressed("jump"));

    state.apply(&[key("Space", true)]);
    assert!(state.pressed("jump"), "Space should trigger jump");

    state.apply(&[key("Space", false)]);
    assert!(!state.pressed("jump"));

    // The alternate binding (KeyW) works too.
    state.apply(&[key("KeyW", true)]);
    assert!(state.pressed("jump"));

    // And the gamepad button.
    state.apply(&[
        key("KeyW", false),
        InputEvent::GamepadButton {
            button: GamepadButton::South,
            pressed: true,
        },
    ]);
    assert!(state.pressed("jump"), "gamepad South should trigger jump");
}

#[test]
fn edge_detection_rising_and_falling() {
    let mut state = InputState::new(platformer_map());

    // Frame 1: press → just_pressed this frame, and held.
    state.apply(&[key("Space", true)]);
    assert!(state.just_pressed("jump"));
    assert!(state.pressed("jump"));
    assert!(!state.just_released("jump"));

    // Frame 2: still held, no event → no longer "just" pressed.
    state.apply(&[]);
    assert!(!state.just_pressed("jump"));
    assert!(state.pressed("jump"));

    // Frame 3: release → just_released this frame.
    state.apply(&[key("Space", false)]);
    assert!(state.just_released("jump"));
    assert!(!state.pressed("jump"));

    // Frame 4: nothing → edge cleared.
    state.apply(&[]);
    assert!(!state.just_released("jump"));
}

#[test]
fn keyboard_axis_scales_and_sums() {
    let mut state = InputState::new(platformer_map());
    assert_eq!(state.axis("move_x"), 0.0);

    state.apply(&[key("KeyD", true)]);
    assert_eq!(state.axis("move_x"), 1.0);

    state.apply(&[key("KeyA", true)]);
    // Both held: +1 and -1 cancel.
    assert_eq!(state.axis("move_x"), 0.0);

    state.apply(&[key("KeyD", false)]);
    assert_eq!(state.axis("move_x"), -1.0);

    // Unbound axis reads zero.
    assert_eq!(state.axis("nope"), 0.0);
}

#[test]
fn gamepad_axis_deadzone_and_clamp() {
    // Deadzone 0.25 on the left-stick-driven move_x.
    let map = InputMap::new().with_deadzone(0.25).tap(|m| {
        m.bind_axis_stick("move_x", GamepadAxis::LeftStickX, 1.0);
    });
    let mut state = InputState::new(map);

    // Inside the deadzone → 0.
    state.apply(&[InputEvent::GamepadAxis {
        axis: GamepadAxis::LeftStickX,
        value: 0.2,
    }]);
    assert_eq!(state.axis("move_x"), 0.0);

    // Just past the deadzone edge → starts from 0 (rescaled), not a jump to 0.25.
    state.apply(&[InputEvent::GamepadAxis {
        axis: GamepadAxis::LeftStickX,
        value: 0.25,
    }]);
    assert!(state.axis("move_x").abs() < 1e-6);

    // Full deflection → 1.0.
    state.apply(&[InputEvent::GamepadAxis {
        axis: GamepadAxis::LeftStickX,
        value: 1.0,
    }]);
    assert!((state.axis("move_x") - 1.0).abs() < 1e-6);

    // Halfway between deadzone (0.25) and 1.0 → 0.5 after rescale.
    state.apply(&[InputEvent::GamepadAxis {
        axis: GamepadAxis::LeftStickX,
        value: 0.625,
    }]);
    assert!((state.axis("move_x") - 0.5).abs() < 1e-6);
}

#[test]
fn axis_negative_scale_inverts_and_clamps() {
    // Two full-deflection sources with the same sign clamp to the [-1,1] range.
    let map = InputMap::new().with_deadzone(0.0).tap(|m| {
        m.bind_axis(
            "throttle",
            AxisSource::GamepadButton {
                button: GamepadButton::RightTrigger,
                scale: 0.8,
            },
        );
        m.bind_axis(
            "throttle",
            AxisSource::GamepadButton {
                button: GamepadButton::South,
                scale: 0.8,
            },
        );
    });
    let mut state = InputState::new(map);
    state.apply(&[
        InputEvent::GamepadButton {
            button: GamepadButton::RightTrigger,
            pressed: true,
        },
        InputEvent::GamepadButton {
            button: GamepadButton::South,
            pressed: true,
        },
    ]);
    // 0.8 + 0.8 = 1.6 → clamped to 1.0.
    assert_eq!(state.axis("throttle"), 1.0);
}

#[test]
fn input_map_serde_round_trip_is_stable() {
    let map = platformer_map();
    let json = serde_json::to_string_pretty(&map).unwrap();
    let back: InputMap = serde_json::from_str(&json).unwrap();
    assert_eq!(map, back, "InputMap did not survive a serde round-trip");
    // Deterministic emit: re-serializing the deserialized map is byte-identical.
    let json2 = serde_json::to_string_pretty(&back).unwrap();
    assert_eq!(json, json2);
}

#[test]
fn input_map_deserializes_from_minimal_json() {
    // Missing fields fall back to defaults (empty maps, default deadzone).
    let m: InputMap = serde_json::from_str("{}").unwrap();
    assert_eq!(m.deadzone(), 0.1);
    assert_eq!(m.action_names().count(), 0);
    assert_eq!(m.axis_names().count(), 0);
}

/// Tiny helper so the binding-builder tests read as one expression.
trait Tap: Sized {
    fn tap(mut self, f: impl FnOnce(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}
impl Tap for InputMap {}
