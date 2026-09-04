//! **The player's window, as the operating system sees it** (wave FIX1).
//!
//! Two questions this crate could not answer before, both of which the author
//! answered for it by pressing Play and reporting what they saw:
//!
//! * *"a secondary powershell window is open the whole time"* — [`console`]
//!   reports whether this process owns a console window at all, so the fix
//!   ([`inf_editor_core::pie`]'s `CREATE_NO_WINDOW` plus this binary's
//!   `windows_subsystem` attribute) is **measured in the real host** off the
//!   player's own stderr rather than asserted at the call site.
//! * *"the movement does not work at all"* — [`take_keyboard_focus`] is both the
//!   instrument and the fix. An embedded PIE session's window is a `WS_CHILD` of
//!   the editor's top-level window **owned by another process's thread**, and
//!   `inf-viewport` adopts it with `SW_SHOWNA` / `SWP_NOACTIVATE` — deliberately,
//!   because activating it would yank the editor's own activation around. The
//!   consequence nobody had measured is that the WebView keeps the keyboard: mouse
//!   messages reach the player because they are routed by hit-test, and key
//!   messages do not because they are routed by **focus**.
//!
//! # Why the focus door escalates rather than always attaching
//!
//! Keyboard focus is a property of an *input queue*, not of a window, and two
//! threads share one only while their queues are attached. Reparenting a window
//! into another thread's window tree attaches them on the Windows versions this
//! engine targets — which is why `inf-viewport`'s own `begin_capture` can call a
//! bare `SetFocus` on its child from the viewport thread and take the keyboard
//! off the WebView. Whether that also holds **across a process boundary** is not
//! something this repository should assume, so the door tries the cheap call
//! first, **measures the result against the foreground thread's own focus
//! record** (`GetGUIThreadInfo`, which is the ground truth for "who receives the
//! next keystroke" — `GetFocus` only ever answers for the calling thread's queue
//! and would report success for a focus nobody's keys reach), and only attaches
//! when that measurement says the cheap call did not land.
//!
//! An unconditional `AttachThreadInput` would have been the worse choice for a
//! reason that is not style: attachment is a **boolean, not a refcount**, so an
//! attach the system had already made and a detach this module owns are the same
//! detach — this module would have silently un-attached queues it did not
//! attach. [`release_keyboard_focus`] therefore undoes exactly what
//! [`take_keyboard_focus`] did and nothing else.

/// What [`take_keyboard_focus`] found and what it did about it — logged once per
/// grab so the Output Log carries the measurement, not a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FocusReport {
    /// Our own window.
    pub hwnd: isize,
    /// The parent we are a `WS_CHILD` of, or `0` when we are top-level.
    pub parent: isize,
    /// The foreground top-level window at the moment of the grab.
    pub foreground: isize,
    /// The foreground thread's focus window **before** the grab — the answer to
    /// "who was getting the keystrokes".
    pub before: isize,
    /// …and after.
    pub after: isize,
    /// This call attached our input queue to the parent's thread.
    pub attached: bool,
    /// `after == hwnd`: the keyboard is ours.
    pub landed: bool,
}

impl std::fmt::Display for FocusReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hwnd={:#x} parent={:#x} fg={:#x} focus {:#x} -> {:#x} attached={} landed={}",
            self.hwnd,
            self.parent,
            self.foreground,
            self.before,
            self.after,
            self.attached,
            self.landed
        )
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Console::GetConsoleWindow;
    use windows::Win32::System::Threading::AttachThreadInput;
    use windows::Win32::UI::Input::KeyboardAndMouse::SetFocus;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetGUIThreadInfo, GetParent, GetWindowLongPtrW,
        GetWindowThreadProcessId, SetForegroundWindow, GUITHREADINFO, GWL_STYLE, WS_CHILD,
    };

    use super::FocusReport;

    /// The thread we attached ours to, or `0`. Paired with [`ATTACHED`] so a
    /// detach can only ever undo an attach this module made.
    static ATTACHED_TO: AtomicU32 = AtomicU32::new(0);
    static ATTACHED: AtomicBool = AtomicBool::new(false);
    /// Our own window thread, recorded at attach time.
    static OUR_THREAD: AtomicU32 = AtomicU32::new(0);

    fn h(hwnd: isize) -> HWND {
        HWND(hwnd as *mut _)
    }

    fn i(hwnd: HWND) -> isize {
        hwnd.0 as isize
    }

    /// The **foreground thread's** focus window: the window the next keystroke
    /// goes to, whichever process owns it. `GetFocus` cannot answer this — it
    /// reports the calling thread's own queue, which for an unattached thread is
    /// a private answer nobody's keys respect.
    fn foreground_focus() -> isize {
        let mut gti = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        // Thread id 0 means "the foreground thread".
        if unsafe { GetGUIThreadInfo(0, &mut gti) }.is_ok() {
            i(gti.hwndFocus)
        } else {
            0
        }
    }

    pub fn console() -> Option<isize> {
        let w = unsafe { GetConsoleWindow() };
        if w.0.is_null() {
            None
        } else {
            Some(i(w))
        }
    }

    fn parent_of(hwnd: HWND) -> Option<HWND> {
        let style = unsafe { GetWindowLongPtrW(hwnd, GWL_STYLE) };
        if style & (WS_CHILD.0 as isize) == 0 {
            return None;
        }
        unsafe { GetParent(hwnd) }.ok().filter(|p| !p.0.is_null())
    }

    pub fn take_keyboard_focus(hwnd: isize) -> FocusReport {
        let win = h(hwnd);
        let mut r = FocusReport {
            hwnd,
            foreground: i(unsafe { GetForegroundWindow() }),
            before: foreground_focus(),
            ..Default::default()
        };
        match parent_of(win) {
            // Embedded PIE: a child of the editor's window, owned by the
            // editor's process. Never call `SetForegroundWindow` here — the
            // foreground window IS the editor and stealing it would flash the
            // taskbar and, on a locked foreground, do nothing at all.
            Some(parent) => {
                r.parent = i(parent);
                let _ = unsafe { SetFocus(Some(win)) };
                r.after = foreground_focus();
                if r.after != hwnd {
                    // The queues are not shared. Attach ours to the parent's and
                    // ask again; recorded so `release_keyboard_focus` undoes
                    // exactly this and nothing the system did for us.
                    let ptid = unsafe { GetWindowThreadProcessId(parent, None) };
                    let mytid = unsafe { GetWindowThreadProcessId(win, None) };
                    if ptid != 0 && mytid != 0 && ptid != mytid {
                        let ok = unsafe { AttachThreadInput(ptid, mytid, true) }.as_bool();
                        if ok {
                            ATTACHED_TO.store(ptid, Ordering::SeqCst);
                            OUR_THREAD.store(mytid, Ordering::SeqCst);
                            ATTACHED.store(true, Ordering::SeqCst);
                            r.attached = true;
                            let _ = unsafe { SetFocus(Some(win)) };
                            r.after = foreground_focus();
                        }
                    }
                }
            }
            // A top-level window: the standalone player, or "Play in New
            // Window". Activation is ours to take and is the whole of it.
            None => {
                let _ = unsafe { SetForegroundWindow(win) };
                let _ = unsafe { SetFocus(Some(win)) };
                r.after = foreground_focus();
            }
        }
        r.landed = r.after == hwnd;
        r
    }

    pub fn release_keyboard_focus() {
        if !ATTACHED.swap(false, Ordering::SeqCst) {
            return;
        }
        let ptid = ATTACHED_TO.swap(0, Ordering::SeqCst);
        let mytid = OUR_THREAD.swap(0, Ordering::SeqCst);
        if ptid != 0 && mytid != 0 {
            let _ = unsafe { AttachThreadInput(ptid, mytid, false) };
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::FocusReport;

    pub fn console() -> Option<isize> {
        None
    }

    pub fn take_keyboard_focus(hwnd: isize) -> FocusReport {
        FocusReport {
            hwnd,
            ..Default::default()
        }
    }

    pub fn release_keyboard_focus() {}
}

/// The console **window** this process owns, if any. `None` is the shipped
/// answer and the one the editor's spawn flag guarantees.
pub fn console() -> Option<isize> {
    imp::console()
}

/// `"none"`, or the console window's handle — the string the `--pie` entry puts
/// on its ready line so the editor's own arm can read the measurement back off a
/// real subprocess.
pub fn console_report() -> String {
    match console() {
        None => "none".to_string(),
        Some(h) => format!("{h:#x}"),
    }
}

/// Take the keyboard for `hwnd`, and report what the operating system said.
///
/// Idempotent in the sense that matters: a call made while we already hold the
/// focus attaches nothing new and returns `landed: true`.
pub fn take_keyboard_focus(hwnd: isize) -> FocusReport {
    imp::take_keyboard_focus(hwnd)
}

/// Undo the input-queue attachment [`take_keyboard_focus`] made, if it made one.
/// Called when the session ends; a leaked attachment ties the editor's input
/// processing to a thread that no longer exists.
pub fn release_keyboard_focus() {
    imp::release_keyboard_focus();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report renders every field a diagnosis needs, and `landed` is derived
    /// from the measurement rather than from the act.
    #[test]
    fn a_focus_report_says_what_was_measured() {
        let r = FocusReport {
            hwnd: 0x1234,
            parent: 0x5678,
            foreground: 0x5678,
            before: 0x9abc,
            after: 0x1234,
            attached: true,
            landed: true,
        };
        let s = r.to_string();
        assert!(s.contains("hwnd=0x1234"), "{s}");
        assert!(s.contains("parent=0x5678"), "{s}");
        assert!(s.contains("0x9abc -> 0x1234"), "{s}");
        assert!(s.contains("attached=true"), "{s}");
        assert!(s.contains("landed=true"), "{s}");
    }

    /// The console report is a *measurement of this process*, so the arm asserts
    /// the two shapes it can produce rather than one machine's answer: a test
    /// binary run from a terminal owns a console and one run by an IDE may not.
    #[test]
    fn the_console_report_is_none_or_a_handle() {
        let s = console_report();
        assert!(
            s == "none" || (s.starts_with("0x") && console().is_some()),
            "unexpected console report {s:?}"
        );
    }

    /// Releasing a focus grab that never attached anything is a no-op, and it
    /// has to be: `release` runs on every session end, including the ones where
    /// the cheap `SetFocus` was enough.
    #[test]
    fn releasing_an_unattached_grab_does_nothing() {
        release_keyboard_focus();
        release_keyboard_focus();
    }
}
