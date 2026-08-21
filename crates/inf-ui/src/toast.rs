//! [`Toasts`]: the honest half of a control that is bound and not yet wired.
//!
//! # A dead key is indistinguishable from a broken one
//!
//! The owner's table binds `reload`, `attack`, `inventory` and `weapon_switch`
//! now, against consumers that arrive with the weapons and inventory work. A
//! player who presses R and gets **nothing** cannot tell "not built yet" from
//! "broken", and neither can a tester filing the bug. So the shipped player
//! says so, out loud, once per press — which is the refusals-are-values law
//! applied to a control scheme rather than to a movement mode.
//!
//! # It runs on the WALL clock
//!
//! A toast fades in real seconds, including while the simulation is paused
//! behind an open menu. It is not sim state, it is never serialized, and two
//! players with different toasts on screen are running the same simulation —
//! which is what makes it safe for it to have a clock of its own.

/// How long a toast stays up before it starts to fade, seconds.
pub const TOAST_HOLD_S: f64 = 2.2;
/// How long the fade takes, seconds.
pub const TOAST_FADE_S: f64 = 0.6;
/// How many are shown at once. The oldest is dropped rather than the newest:
/// somebody mashing four unbound keys wants to see the last thing they pressed.
pub const TOAST_MAX: usize = 4;

/// One line on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct Toast {
    /// What it says.
    pub text: String,
    /// How long it has been up, seconds.
    pub age_s: f64,
}

impl Toast {
    /// Its opacity in `[0, 1]`: solid for [`TOAST_HOLD_S`], then linear to zero.
    pub fn alpha(&self) -> f32 {
        if !self.age_s.is_finite() {
            return 0.0;
        }
        if self.age_s <= TOAST_HOLD_S {
            return 1.0;
        }
        let t = (self.age_s - TOAST_HOLD_S) / TOAST_FADE_S;
        (1.0 - t).clamp(0.0, 1.0) as f32
    }

    /// Whether it is finished and can be dropped.
    fn expired(&self) -> bool {
        !self.age_s.is_finite() || self.age_s >= TOAST_HOLD_S + TOAST_FADE_S
    }
}

/// The stack, newest first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Toasts {
    live: Vec<Toast>,
}

impl Toasts {
    /// Push a line, dropping the oldest if the stack is full.
    ///
    /// **Repeating a line that is already up restarts it** rather than stacking
    /// a second copy: a player holding an unbound key would otherwise fill the
    /// screen with one sentence.
    pub fn push(&mut self, text: impl Into<String>) {
        let text = text.into();
        if let Some(t) = self.live.iter_mut().find(|t| t.text == text) {
            t.age_s = 0.0;
            return;
        }
        self.live.insert(0, Toast { text, age_s: 0.0 });
        self.live.truncate(TOAST_MAX);
    }

    /// The line the player sees when they press a control nothing consumes yet.
    ///
    /// One sentence, built in one place, so the four controls cannot each grow
    /// their own wording.
    pub fn push_not_yet(&mut self, action: &str) {
        self.push(format!("{action}: not wired up yet"));
    }

    /// Age every toast by `dt` **wall-clock** seconds and drop the finished
    /// ones. A non-finite `dt` ages nothing rather than expiring the stack.
    pub fn advance(&mut self, dt: f64) {
        if dt.is_finite() && dt > 0.0 {
            for t in &mut self.live {
                t.age_s += dt;
            }
        }
        self.live.retain(|t| !t.expired());
    }

    /// What is on screen, newest first.
    pub fn live(&self) -> &[Toast] {
        &self.live
    }

    /// Whether anything is showing.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_toast_holds_then_fades_then_goes() {
        let mut t = Toasts::default();
        t.push_not_yet("reload");
        assert_eq!(t.live()[0].text, "reload: not wired up yet");
        assert_eq!(t.live()[0].alpha(), 1.0);

        t.advance(TOAST_HOLD_S);
        assert_eq!(t.live()[0].alpha(), 1.0, "it fades EARLY");
        t.advance(TOAST_FADE_S * 0.5);
        let a = t.live()[0].alpha();
        assert!(a > 0.4 && a < 0.6, "half way through the fade: {a}");
        t.advance(TOAST_FADE_S);
        assert!(t.is_empty(), "it never went away");
    }

    /// **A held key does not fill the screen with one sentence.**
    #[test]
    fn repeating_a_line_restarts_it_rather_than_stacking_it() {
        let mut t = Toasts::default();
        t.push_not_yet("reload");
        t.advance(1.0);
        t.push_not_yet("reload");
        assert_eq!(t.live().len(), 1, "the same line stacked");
        assert_eq!(t.live()[0].age_s, 0.0, "and it did not restart");
    }

    /// The stack is bounded, and it drops the OLDEST.
    #[test]
    fn the_stack_is_bounded_and_keeps_the_newest() {
        let mut t = Toasts::default();
        for i in 0..(TOAST_MAX + 3) {
            t.push(format!("line {i}"));
        }
        assert_eq!(t.live().len(), TOAST_MAX);
        assert_eq!(t.live()[0].text, format!("line {}", TOAST_MAX + 2));
    }

    /// A hostile `dt` neither ages nor expires anything.
    #[test]
    fn a_hostile_dt_does_not_clear_the_stack() {
        let mut t = Toasts::default();
        t.push("x");
        for bad in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
            t.advance(bad);
            assert_eq!(t.live().len(), 1, "dt = {bad} cleared the stack");
            assert!(t.live()[0].age_s.is_finite());
        }
    }
}
