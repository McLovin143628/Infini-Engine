//! **The window class must ask Windows for double-clicks** (Wave E).
//!
//! Windows only synthesizes `WM_*BUTTONDBLCLK` for window classes registered
//! with `CS_DBLCLKS`. Without it the viewport's class carried the default style
//! (`WNDCLASS_STYLES(0)`), so a double-click in the 3D view **did not exist at
//! the OS level** — not "was not handled". The handler is invisible in that
//! state: the code looks complete and the gesture is dead.
//!
//! # Why a source gate, and why it reads a SCOPE
//!
//! There is nothing to assert at runtime: CI has no window and no GPU, and this
//! module is `#[cfg(windows)]` besides. So the pin is on the source — and it
//! reads the *registration block*, not a spelling anywhere in the file (the P23
//! law: a byte pin cannot see a semantic change, so ban a MODULE rather than a
//! string). Putting `CS_DBLCLKS` in a comment, or in some other `WNDCLASSW`
//! literal, does not satisfy it.
//!
//! `include_str!` is safe on a Windows checkout because `.rs` carries
//! `text eol=lf` in `.gitattributes` (P22.4) — but the CRLF strip is kept
//! anyway, since a locally-created file has whatever the editor wrote.

const WIN32: &str = include_str!("../src/win32.rs");

/// The `WNDCLASSW { … }` literal inside `create_child_window`, as text.
fn class_literal() -> String {
    let src = WIN32.replace("\r\n", "\n");
    let fn_start = src
        .find("fn create_child_window(")
        .expect("create_child_window is where the viewport's class is registered");
    let body = &src[fn_start..];
    let lit_start = body
        .find("WNDCLASSW {")
        .expect("create_child_window must build a WNDCLASSW literal");
    let rest = &body[lit_start..];
    let end = rest
        .find("};")
        .expect("the WNDCLASSW literal must terminate");
    rest[..end].to_string()
}

#[test]
fn the_window_class_asks_for_double_clicks() {
    let lit = class_literal();
    assert!(
        lit.contains("style: CS_DBLCLKS"),
        "the viewport window class must set `style: CS_DBLCLKS`, or Windows \
         never delivers WM_LBUTTONDBLCLK and double-click-to-open is dead code. \
         The literal was:\n{lit}"
    );
}

/// The other half: the message the flag exists to enable is actually handled.
///
/// Two separate failures — the flag without a handler, and a handler without the
/// flag — produce the same symptom (nothing happens), and only one of them is
/// visible in the code. Both are pinned.
#[test]
fn the_double_click_message_is_handled() {
    let src = WIN32.replace("\r\n", "\n");
    assert!(
        src.contains("WM_LBUTTONDBLCLK => {"),
        "CS_DBLCLKS with no WM_LBUTTONDBLCLK arm is a flag that enables nothing"
    );
    // …and it must reach the frame loop, not be swallowed in the wnd_proc.
    assert!(
        src.contains("left_dbl = Some("),
        "the double-click arm must record the press for the frame loop to pick"
    );
}

/// The right-button gesture must still begin its capture on button-DOWN.
///
/// This is the flycam-feel pin. The obvious way to implement a context menu is
/// to wait and see whether the gesture becomes a drag before grabbing the mouse
/// — which puts a dead zone at the start of every fly, and is exactly the
/// regression the wave was told not to accept. The discrimination belongs on
/// button-UP.
#[test]
fn the_right_button_still_captures_immediately() {
    let src = WIN32.replace("\r\n", "\n");
    let down = src
        .find("WM_RBUTTONDOWN => {")
        .expect("the RMB arm must exist");
    let arm_end = src[down..]
        .find("WM_RBUTTONUP")
        .expect("the RMB down arm must be followed by the up arm");
    let arm = &src[down..down + arm_end];
    assert!(
        arm.contains("begin_capture(hwnd, kind)"),
        "the flycam must still grab the mouse on RMB-down; deferring it until \
         the click thresholds elapse is a 250 ms dead zone on every fly. Arm:\n{arm}"
    );
}
