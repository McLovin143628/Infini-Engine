//! [`HoldClock`]: how long each named action has been held (I5).
//!
//! # Why this is its own type and not a field on [`InputState`](crate::InputState)
//!
//! There are **two clocks in this engine and they measure different things**,
//! and the whole click-versus-long-press feature turns on keeping them apart.
//!
//! * The **wall clock** is what the in-game UI runs on. A menu's key repeat, a
//!   toast's fade and a rebinding capture all happen while the simulation is
//!   paused, so a sim clock would stop and the menu would freeze.
//! * The **sim clock** is what gameplay runs on. A press duration that decides
//!   whether the character crouches or goes prone is a *gameplay* fact, and a
//!   gameplay fact measured in wall-clock seconds is one the frame rate can
//!   change: the same 300 ms press is eighteen fixed steps on one machine and
//!   eighteen on another only because both accumulate the same fixed `dt`.
//!   Measured in wall time it would be eighteen steps of one duration on a fast
//!   machine and three of another on a slow one, and PIE would stop being the
//!   shipped game.
//!
//! So the *rule* (`inf_ecs::movement::classify_press`, which is pure arithmetic
//! over two durations and therefore needs nothing from this crate) lives once
//! and the *accumulator* lives once, and each host runs one instance of it per clock it
//! owns. `InputState` runs one on the wall clock for the UI; each sim runs one
//! on its fixed step for the intent. Two instances of one implementation is not
//! the two-copies defect — two implementations would be.

use std::collections::{BTreeMap, BTreeSet};

/// Per-action hold durations, in the seconds of whatever clock `advance` is fed.
///
/// Deterministic by construction: `BTreeMap`/`BTreeSet` throughout, and the only
/// arithmetic is one addition per held action per tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HoldClock {
    /// This tick's durations, keyed by action name.
    cur: BTreeMap<String, f64>,
    /// Last tick's, kept so "the hold crossed a threshold on THIS tick" has a
    /// yes/no answer.
    prev: BTreeMap<String, f64>,
    /// Which actions were active last tick — the thing that tells a *fresh*
    /// press (duration 0) from a continuing one, and a release from an absence.
    down_prev: BTreeSet<String>,
}

impl HoldClock {
    /// An empty clock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance one tick against the set of currently-active actions.
    ///
    /// Three cases, and the third is the one that earns the type:
    ///
    /// 1. **Newly active** → duration `0`. A press has been held for no time at
    ///    all on the tick it arrives.
    /// 2. **Still active** → last tick's duration plus `dt`.
    /// 3. **Released this tick** → last tick's duration, **carried unchanged**.
    ///    The falling edge and the length of the press that produced it are
    ///    readable together, at the one tick where both are still knowable. A
    ///    reader that had to ask before the release would need a clock of its
    ///    own.
    ///
    /// A non-finite or non-positive `dt` contributes **zero** rather than
    /// poisoning every duration in the map with a NaN that no comparison can
    /// escape.
    pub fn advance(&mut self, down: &BTreeSet<String>, dt: f64) {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.clear();
        for name in down {
            let d = if self.down_prev.contains(name) {
                self.prev.get(name).copied().unwrap_or(0.0) + dt
            } else {
                0.0
            };
            self.cur.insert(name.clone(), d);
        }
        for name in self.down_prev.difference(down) {
            let d = self.prev.get(name).copied().unwrap_or(0.0);
            self.cur.insert(name.clone(), d);
        }
        self.down_prev.clear();
        self.down_prev.extend(down.iter().cloned());
    }

    /// `(this tick's duration, last tick's)` for `action`, both `0` if it is
    /// neither active nor released this tick.
    ///
    /// The **pair** is the answer rather than one number, because the rule that
    /// reads it asks whether a threshold was crossed *on this tick* — which is a
    /// question about two durations and cannot be answered by either alone.
    pub fn hold(&self, action: &str) -> (f64, f64) {
        (
            self.cur.get(action).copied().unwrap_or(0.0),
            self.prev.get(action).copied().unwrap_or(0.0),
        )
    }

    /// Forget everything — the device went away, or the sim was reset.
    ///
    /// Deliberately **not** a tick: it does not carry a release duration, so
    /// nothing downstream reads a click out of a window losing focus.
    pub fn clear(&mut self) {
        self.cur.clear();
        self.prev.clear();
        self.down_prev.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// **The three cases, each measured** — a fresh press reads zero, a
    /// continuing one accumulates, and a release carries its final duration.
    #[test]
    fn a_hold_starts_at_zero_accumulates_and_survives_its_own_release() {
        let mut c = HoldClock::new();
        let dt = 1.0 / 60.0;

        c.advance(&set(&["crouch"]), dt);
        assert_eq!(
            c.hold("crouch"),
            (0.0, 0.0),
            "a press has been held for no time"
        );

        c.advance(&set(&["crouch"]), dt);
        assert!((c.hold("crouch").0 - dt).abs() < 1e-12);
        assert_eq!(c.hold("crouch").1, 0.0);

        c.advance(&set(&["crouch"]), dt);
        assert!((c.hold("crouch").0 - 2.0 * dt).abs() < 1e-12);

        // Released: the duration is carried, so the falling edge can be read
        // together with the length of the press that produced it.
        c.advance(&set(&[]), dt);
        assert!(
            (c.hold("crouch").0 - 2.0 * dt).abs() < 1e-12,
            "the release tick dropped the duration, so nothing downstream can tell a tap from a hold"
        );

        // …and one tick later it is gone.
        c.advance(&set(&[]), dt);
        assert_eq!(c.hold("crouch"), (0.0, 2.0 * dt));
    }

    /// A re-press starts from zero rather than resuming the previous one.
    #[test]
    fn a_second_press_does_not_resume_the_first() {
        let mut c = HoldClock::new();
        let dt = 0.1;
        for _ in 0..5 {
            c.advance(&set(&["crouch"]), dt);
        }
        assert!((c.hold("crouch").0 - 0.4).abs() < 1e-12);
        c.advance(&set(&[]), dt);
        c.advance(&set(&[]), dt);
        c.advance(&set(&["crouch"]), dt);
        assert_eq!(
            c.hold("crouch").0,
            0.0,
            "a fresh press resumed the old duration, so a rapid tap would classify as a hold"
        );
    }

    /// **A NaN `dt` cannot poison a duration.** Every comparison a NaN takes
    /// part in is false, so one would make the control it belongs to silently
    /// stop responding — for ever, because the value is fed back into itself.
    #[test]
    fn a_hostile_dt_contributes_nothing() {
        let mut c = HoldClock::new();
        for dt in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
            c.clear();
            c.advance(&set(&["crouch"]), 0.1);
            c.advance(&set(&["crouch"]), dt);
            let (cur, _) = c.hold("crouch");
            assert!(
                cur.is_finite() && cur == 0.0,
                "dt = {dt} produced a hold of {cur}"
            );
        }
    }

    /// `clear` is not a tick: a window that loses focus must not read as a tap.
    #[test]
    fn clearing_does_not_leave_a_release_duration_behind() {
        let mut c = HoldClock::new();
        c.advance(&set(&["crouch"]), 0.1);
        c.advance(&set(&["crouch"]), 0.1);
        c.clear();
        assert_eq!(c.hold("crouch"), (0.0, 0.0));
    }
}
