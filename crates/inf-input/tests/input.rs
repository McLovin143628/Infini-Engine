//! Tests for the pure input core: action resolution, edges, deadzone/scale math,
//! and `InputMap` serde round-trip. No device is involved.

use inf_input::{
    AxisSource, GamepadAxis, GamepadButton, InputEvent, InputMap, InputState, MouseAxis,
    MouseButton,
};

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

// ── mouse (P29.3) ───────────────────────────────────────────────────────────
//
// Ruling 3's second unscheduled blocker. Before this wave `InputEvent` was
// `Key | GamepadButton | GamepadAxis | Touch`: there was no mouse source and no
// mouse-delta event anywhere in the engine, so mouse look did not exist and
// `AimYawRate` — degrees of camera yaw per second, an input to three ported ALS
// systems — had nothing to be derived from.

/// A map whose `look_x`/`look_y` read the mouse at a sensitivity, the shape
/// every consumer of `AimYawRate` will bind.
fn look_map() -> InputMap {
    let mut m = InputMap::new();
    m.bind_axis_mouse("look_x", MouseAxis::X, 0.1)
        // Negative scale: the platform reports +y DOWN and a look control wants
        // +pitch UP. The engine does not guess this; the binding states it.
        .bind_axis_mouse("look_y", MouseAxis::Y, -0.1)
        .bind_mouse("fire", MouseButton::Left)
        .bind_mouse("aim", MouseButton::Right);
    m
}

#[test]
fn a_mouse_button_resolves_an_action_and_reports_both_edges() {
    let mut st = InputState::new(look_map());
    assert!(!st.pressed("fire"));

    st.apply(&[InputEvent::MouseButton {
        button: MouseButton::Left,
        pressed: true,
    }]);
    assert!(st.pressed("fire"));
    assert!(st.just_pressed("fire"));
    assert!(!st.pressed("aim"), "the right button is a different action");

    st.apply(&[]);
    assert!(st.pressed("fire"), "a button is level state, like a key");
    assert!(!st.just_pressed("fire"), "the edge is one frame");

    st.apply(&[InputEvent::MouseButton {
        button: MouseButton::Left,
        pressed: false,
    }]);
    assert!(!st.pressed("fire"));
    assert!(st.just_released("fire"));
}

#[test]
fn mouse_motion_accumulates_within_a_frame_and_is_gone_by_the_next() {
    let mut st = InputState::new(look_map());

    // A frame's motion may arrive as any number of OS events; the resolved axis
    // must not depend on that packetization.
    st.apply(&[
        InputEvent::MouseMotion { delta: [4.0, 0.0] },
        InputEvent::MouseMotion { delta: [4.0, 0.0] },
        InputEvent::MouseMotion { delta: [2.0, 6.0] },
    ]);
    assert!(
        (st.axis("look_x") - 1.0).abs() < 1e-6,
        "10 px at 0.1 = 1.0, not 0.4: {}",
        st.axis("look_x")
    );
    assert!(
        (st.axis("look_y") + 0.6).abs() < 1e-6,
        "+y is down and the binding inverts it: {}",
        st.axis("look_y")
    );

    // The delta belongs to the frame it happened in. Leaving it would make one
    // flick drive the look forever — the mouse-look form of the stuck key.
    st.apply(&[]);
    assert_eq!(st.axis("look_x"), 0.0, "a still mouse contributes nothing");
    assert_eq!(st.axis("look_y"), 0.0);
}

#[test]
fn a_mouse_axis_is_not_clamped_and_a_bounded_one_still_is() {
    // The load-bearing difference between a delta and a position. A fast flick
    // is fifty pixels and a slow drag is two; clamping both to 1 makes them the
    // same gesture, which is the look bug every engine ships once.
    let mut st = InputState::new(look_map());
    st.apply(&[InputEvent::MouseMotion {
        delta: [500.0, 0.0],
    }]);
    assert!(
        (st.axis("look_x") - 50.0).abs() < 1e-4,
        "a 500 px flick is 50 degrees at 0.1, not 1: {}",
        st.axis("look_x")
    );

    // The control, and the reason this is a property of the SOURCE and not of
    // the axis pipeline: a keyboard axis with five contributions still clamps.
    let mut m = InputMap::new();
    m.bind_axis_key("move_x", "KeyD", 1.0)
        .bind_axis_key("move_x", "KeyE", 1.0)
        .bind_axis_key("move_x", "KeyR", 1.0);
    let mut bounded = InputState::new(m);
    bounded.apply(&[
        InputEvent::Key {
            code: "KeyD".into(),
            pressed: true,
        },
        InputEvent::Key {
            code: "KeyE".into(),
            pressed: true,
        },
        InputEvent::Key {
            code: "KeyR".into(),
            pressed: true,
        },
    ]);
    assert_eq!(
        bounded.axis("move_x"),
        1.0,
        "three keys at +1 is still full deflection"
    );
}

#[test]
fn releasing_everything_forgets_the_mouse_too() {
    // The R2-9 failure met from the mouse's side: a window that loses focus with
    // the right button held leaves the character aiming for the session.
    let mut st = InputState::new(look_map());
    st.apply(&[
        InputEvent::MouseButton {
            button: MouseButton::Right,
            pressed: true,
        },
        InputEvent::MouseMotion { delta: [9.0, 9.0] },
    ]);
    assert!(st.pressed("aim"));

    st.release_all();
    assert!(!st.pressed("aim"), "the button is forgotten");
    assert!(st.just_released("aim"), "and the release edge fires");
    assert_eq!(st.axis("look_x"), 0.0, "and the pending motion is dropped");
}

#[test]
fn the_wheel_is_its_own_pair_of_axes() {
    let mut m = InputMap::new();
    m.bind_axis_mouse("zoom", MouseAxis::WheelY, 1.0);
    let mut st = InputState::new(m);
    st.apply(&[
        InputEvent::MouseWheel { delta: [0.0, 1.0] },
        InputEvent::MouseWheel { delta: [0.0, 2.0] },
    ]);
    assert_eq!(st.axis("zoom"), 3.0, "notches accumulate like motion");
    st.apply(&[]);
    assert_eq!(st.axis("zoom"), 0.0);
}

#[test]
fn the_mouse_source_tokens_are_frozen_and_append_only() {
    // `ActionSource`/`AxisSource` are saved in a project's `input.toml`, so the
    // wire law reaches them: a variant may be appended and never inserted, and
    // the *spelling* is the contract because this format is name-tagged.
    //
    // Written as a literal table rather than derived from the variants, because
    // a table derived from the type agrees with any rename by construction —
    // the shape of gate this repository has had to repair before.
    const FROZEN_BUTTONS: [(MouseButton, &str); 7] = [
        (MouseButton::Left, "Left"),
        (MouseButton::Middle, "Middle"),
        (MouseButton::Right, "Right"),
        (MouseButton::Back, "Back"),
        (MouseButton::Forward, "Forward"),
        // ── reserved (P29.3) ──
        (MouseButton::Reserved5, "Reserved5"),
        (MouseButton::Reserved6, "Reserved6"),
    ];
    const FROZEN_AXES: [(MouseAxis, &str); 6] = [
        (MouseAxis::X, "X"),
        (MouseAxis::Y, "Y"),
        (MouseAxis::WheelX, "WheelX"),
        (MouseAxis::WheelY, "WheelY"),
        // ── reserved (P29.3) ──
        (MouseAxis::Reserved4, "Reserved4"),
        (MouseAxis::Reserved5, "Reserved5"),
    ];

    for (b, token) in FROZEN_BUTTONS {
        let json = serde_json::to_string(&b).expect("a unit variant serializes");
        assert_eq!(json, format!("\"{token}\""), "{b:?} changed its spelling");
        let back: MouseButton = serde_json::from_str(&json).expect("and reads back");
        assert_eq!(back, b);
    }
    for (a, token) in FROZEN_AXES {
        let json = serde_json::to_string(&a).expect("a unit variant serializes");
        assert_eq!(json, format!("\"{token}\""), "{a:?} changed its spelling");
        let back: MouseAxis = serde_json::from_str(&json).expect("and reads back");
        assert_eq!(back, a);
    }

    // A reserved slot no reader ever asks about is a comment, not a slot. Both
    // directions, so the accessor cannot answer `Some` for everything.
    assert_eq!(MouseButton::Reserved5.reserved_slot(), Some(5));
    assert_eq!(MouseButton::Reserved6.reserved_slot(), Some(6));
    assert_eq!(MouseButton::Left.reserved_slot(), None);
    assert_eq!(MouseButton::Forward.reserved_slot(), None);
    assert_eq!(MouseAxis::Reserved4.reserved_slot(), Some(4));
    assert_eq!(MouseAxis::Reserved5.reserved_slot(), Some(5));
    assert_eq!(MouseAxis::X.reserved_slot(), None);
    assert_eq!(MouseAxis::WheelY.reserved_slot(), None);

    // And the whole map round-trips through the format it actually lives in.
    let toml_text = toml::to_string(&look_map()).expect("an InputMap serializes to TOML");
    let back: InputMap = toml::from_str(&toml_text).expect("and reads back");
    assert_eq!(back, look_map(), "the mouse bindings survive `input.toml`");
    assert!(
        toml_text.contains("MouseAxis") && toml_text.contains("MouseButton"),
        "the round-trip must have exercised the new variants, not an empty map:\n{toml_text}"
    );
}

/// **A look axis bound to BOTH a mouse and a stick converts only the mouse
/// half** (P29.3 audit, A2).
///
/// This is the shape the shipped `default_map` actually has: `look_x` names the
/// mouse (degrees per raw count) *and* the right stick (degrees per second,
/// hence a scale of 180 rather than 1, which is legal precisely because a delta
/// source makes the axis unclamped). The first `axis_snapshot` asked its
/// question of the axis NAME and divided the resolved total, so the stick's
/// 180 deg/s came out as 10 800 at 60 fps — and as 5 400 at 30, which is a look
/// control whose speed depends on the frame rate.
///
/// The control is the load-bearing half: the mouse-only axis beside it must
/// still convert, or this arm is satisfied by an `axis_snapshot` that converts
/// nothing at all.
#[test]
fn a_stick_bound_to_a_look_axis_is_not_divided_by_the_frame_time() {
    let mut m = look_map();
    m.bind_axis_stick("look_x", GamepadAxis::RightStickX, 180.0);
    let mut st = InputState::new(m);

    // A stick at full deflection and NO mouse motion is 180 deg/s, whatever the
    // frame rate.
    st.apply(&[InputEvent::GamepadAxis {
        axis: GamepadAxis::RightStickX,
        value: 1.0,
    }]);
    for dt in [1.0 / 30.0, 1.0 / 60.0, 1.0 / 240.0] {
        let snap = st.axis_snapshot(dt);
        assert!(
            (snap["look_x"] - 180.0).abs() < 1e-3,
            "dt = {dt}: a stick is a POSITION and must not be divided by it, got {}",
            snap["look_x"]
        );
    }

    // Both together add as rates: 30 counts x 0.1 deg = 3 deg in 1/60 s is
    // 180 deg/s of mouse, on top of the stick's 180.
    st.apply(&[InputEvent::MouseMotion { delta: [30.0, 0.0] }]);
    let snap = st.axis_snapshot(1.0 / 60.0);
    assert!(
        (snap["look_x"] - 360.0).abs() < 1e-2,
        "the mouse half converts and the stick half does not: {}",
        snap["look_x"]
    );
    // The control: the mouse-only axis still becomes a rate, so the assertion
    // above is about the SOURCE split and not about a conversion that stopped.
    st.apply(&[InputEvent::MouseMotion { delta: [0.0, 30.0] }]);
    let snap = st.axis_snapshot(1.0 / 60.0);
    assert!(
        (snap["look_y"] + 180.0).abs() < 1e-2,
        "look_y is mouse-only and must still convert: {}",
        snap["look_y"]
    );
}

#[test]
fn a_delta_axis_snapshots_as_a_rate_and_a_bounded_one_as_a_position() {
    // The frame-rate independence the fixed step rests on: the same gesture
    // delivered in one slow frame and in four fast ones must integrate to the
    // same rotation.
    let mut m = look_map();
    m.bind_axis_key("move_x", "KeyD", 1.0);
    let mut st = InputState::new(m);

    st.apply(&[
        InputEvent::MouseMotion {
            delta: [100.0, 0.0],
        },
        InputEvent::Key {
            code: "KeyD".into(),
            pressed: true,
        },
    ]);
    let slow = st.axis_snapshot(0.1);
    assert!(
        (slow["look_x"] - 100.0).abs() < 1e-3,
        "10 degrees over 0.1 s is 100 deg/s: {}",
        slow["look_x"]
    );
    assert_eq!(
        slow["move_x"], 1.0,
        "a held key is a POSITION and must not be divided by dt"
    );

    // A quarter of the motion in a quarter of the time is the same rate.
    st.apply(&[InputEvent::MouseMotion { delta: [25.0, 0.0] }]);
    let fast = st.axis_snapshot(0.025);
    assert!(
        (fast["look_x"] - slow["look_x"]).abs() < 1e-2,
        "the same gesture at four times the frame rate is the same rate: {} vs {}",
        fast["look_x"],
        slow["look_x"]
    );

    // A degenerate dt publishes the raw value rather than an infinity.
    st.apply(&[InputEvent::MouseMotion { delta: [10.0, 0.0] }]);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let snap = st.axis_snapshot(bad);
        assert!(
            snap["look_x"].is_finite(),
            "dt = {bad} produced {} for look_x",
            snap["look_x"]
        );
    }
}
