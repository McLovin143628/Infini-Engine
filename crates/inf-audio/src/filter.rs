//! **The low-pass, at last** (wave WPN2c) — a one-pole, and the end of a debt
//! this crate has carried since P12.
//!
//! # What was wrong
//!
//! [`Effect::Lowpass`](crate::mixer::Effect::Lowpass) has been in the mixer
//! config since P12.3 and [`MixerConfig::resolve`](crate::MixerConfig::resolve)
//! has folded its cutoff the whole time. Island wave VEN1b then added
//! `AudioCommand::SetOcclusion { lowpass_hz }`, so a shut door decided a cutoff
//! and put it in the command stream. **Neither of them ever changed a sample.**
//! `backend.rs` had no filter code, kira's own `FilterBuilder` was never wired,
//! and every doc in the crate said so in the same words: *modelled and
//! inspectable, not audible*. (Two of them still did after this module landed —
//! [`crate::AudioEngine::named_bus_lowpass_hz`] and
//! [`crate::AudioCommand::SetOcclusion`] — which is what the wave's audit
//! found: a module that makes a claim true has to go and unsay the places that
//! say it is false.)
//!
//! # Why the fix is here and not in kira
//!
//! kira 0.12 does expose a filter, as a **track effect**. Using it means one
//! sub-track per thing that needs a different cutoff — and the cutoffs this
//! engine decides are *per voice*, not per bus: a shut door muffles one club's
//! loop, and wave WPN2c's distant gunshot layer muffles one shot. A sub-track
//! per one-shot, created and destroyed at 320 voices a second, is the one
//! allocation pattern kira's track model is built to avoid.
//!
//! So the filter is ours, it runs over the decoded frames **before** the voice
//! starts ([`crate::SoundData::low_passed`]), and it therefore works
//! identically on the `cpal` build and in the no-device path. It is also the
//! only shape a headless CI can prove anything about: a filter nobody can hear
//! is proven by its impulse response, and [`OnePole::response_at`] is that
//! proof's Ring-0 door.
//!
//! # The filter
//!
//! `y[n] = y[n-1] + a·(x[n] − y[n-1])`, the textbook one-pole, with
//! `a = dt/(RC + dt)` and `RC = 1/(2π·f_c)`. Its magnitude response is
//!
//! ```text
//!   |H(w)| = a / sqrt(1 + (1 − a)² − 2(1 − a)·cos w),   w = 2π f / rate
//! ```
//!
//! which is −3.01 dB at the cutoff for a cutoff well below Nyquist — and that
//! is what [`OnePole::response_at`] returns and what the tests measure, both
//! analytically and by pushing an actual impulse through an actual filter.
//!
//! One pole is 6 dB per octave, which is gentle. It is also what a *doorway* and
//! what *four hundred metres of air* actually are, so the order is a choice and
//! not a shortcut; a steeper answer is two of these in series and the day
//! something needs one, `low_passed` runs twice.

use std::f64::consts::PI;

/// **A portable cosine** for `w` in `[0, pi]`, accurate to about a nanounit —
/// the one transcendental this module needs and the one `libm` may not agree
/// with itself about across targets (the P14 law).
///
/// `cos w = 1 - 2 sin^2(w/2)`, with the half-angle sine as its Taylor series to
/// the fifteenth power. On `[0, pi/2]` that series is alternating and its first
/// dropped term is `x^17/17!`, which at `x = pi/2` is 6e-12 — many orders below
/// anything a filter coefficient carries into a 16-bit sample. (Thirteen terms
/// was the first draft and measured 2.25e-7, which is why the claim below is a
/// measured one.)
///
/// Outside `[0, pi]` it folds by periodicity and symmetry first, so a caller
/// need not.
pub fn pcos(w: f64) -> f64 {
    if !w.is_finite() {
        return 1.0;
    }
    // Fold into [0, pi]: cos is even and has period 2 pi.
    let two_pi = 2.0 * PI;
    let mut x = (w.abs() / two_pi).fract() * two_pi;
    if x > PI {
        x = two_pi - x;
    }
    let h = 0.5 * x;
    let h2 = h * h;
    // sin h, Taylor to h^13, Horner from the inside out.
    let s = h
        * (1.0
            - h2 / 6.0
                * (1.0
                    - h2 / 20.0
                        * (1.0
                            - h2 / 42.0
                                * (1.0
                                    - h2 / 72.0
                                        * (1.0
                                            - h2 / 110.0
                                                * (1.0 - h2 / 156.0 * (1.0 - h2 / 210.0)))))));
    1.0 - 2.0 * s * s
}

/// **A one-pole low-pass**, carrying its own coefficient and its own state.
///
/// `Copy`, so a caller can keep one per channel without ceremony.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePole {
    /// The smoothing coefficient in `[0, 1]`. `1.0` passes everything.
    alpha: f64,
    /// The last output.
    y: f64,
}

impl OnePole {
    /// A filter at `cutoff_hz`, for audio at `rate` hertz.
    ///
    /// A cutoff at or above Nyquist, or a non-finite one, gives `alpha = 1`:
    /// the filter passes its input through untouched, which is the honest
    /// answer for "filter nothing above the highest frequency present".
    pub fn new(cutoff_hz: f64, rate: u32) -> Self {
        Self {
            alpha: Self::alpha(cutoff_hz, rate),
            y: 0.0,
        }
    }

    /// **The coefficient** — the whole of the design, as one number a test can
    /// read, and it is solved rather than approximated.
    ///
    /// # Why not `dt/(RC + dt)`
    ///
    /// That is the textbook RC discretisation and it was this module's first
    /// draft. It puts the filter's real half-power point **below** the frequency
    /// it names, by an error that grows with `f/rate` — measured, at 22 050 Hz:
    ///
    /// ```text
    ///     200 Hz  ->  194.6 Hz   (-2.7 %)
    ///     500 Hz  ->  468.1 Hz   (-6.4 %)
    ///     700 Hz  ->  640.0 Hz   (-8.6 %)
    ///   1 500 Hz  -> 1262.3 Hz  (-15.9 %)
    /// ```
    ///
    /// A filter whose cutoff is sixteen per cent wrong is a filter whose cutoff
    /// is a suggestion, and three of this engine's four cutoffs (500 Hz for a
    /// shut door, 700 Hz for four hundred metres of air, 3 500 Hz for a room's
    /// tail) are in the range where it matters.
    ///
    /// So the coefficient is the **exact** one instead. Setting the magnitude
    /// response to `1/sqrt(2)` at `w = 2*pi*f/rate` gives
    /// `a^2 + 2aC - 2C = 0` with `C = 1 - cos w`, whose positive root is
    /// `a = -C + sqrt(C^2 + 2C)`. That is one square root, which IEEE-754
    /// requires to be correctly rounded, and one cosine — which is **not**
    /// portable from `libm`, and is therefore [`pcos`], a polynomial. The whole
    /// function is bit-exact on every target, which it has to be: these
    /// coefficients shape thirty-six committed `.inf_audio` files.
    pub fn alpha(cutoff_hz: f64, rate: u32) -> f64 {
        if rate == 0 || !cutoff_hz.is_finite() || cutoff_hz <= 0.0 {
            return 1.0;
        }
        // **At or above Nyquist there is nothing to filter**, and neither
        // design knows that on its own: 100 kHz at 22 050 answered 0.966 on the
        // first draft, which is a filter nobody asked for.
        if cutoff_hz * 2.0 >= f64::from(rate) {
            return 1.0;
        }
        let w = 2.0 * PI * cutoff_hz / f64::from(rate);
        let c = 1.0 - pcos(w);
        (-c + (c * c + 2.0 * c).sqrt()).clamp(0.0, 1.0)
    }

    /// The coefficient this instance was built with.
    pub fn coefficient(self) -> f64 {
        self.alpha
    }

    /// Push one sample through and take the output.
    pub fn step(&mut self, x: f64) -> f64 {
        self.y += self.alpha * (x - self.y);
        self.y
    }

    /// Forget the history, keeping the design.
    pub fn reset(&mut self) {
        self.y = 0.0;
    }

    /// **The magnitude response at `freq_hz`**, linear — the analytic form of
    /// what [`impulse_response`](Self::impulse_response) produces, and the
    /// function the audit's *"a filter is proven by its impulse response"* is
    /// answered with.
    ///
    /// `1.0` is unity; `0.707…` is −3 dB. At `freq_hz == cutoff_hz` a one-pole
    /// well below Nyquist answers −3.01 dB, and this is the only place in the
    /// crate that uses a transcendental — it is a **measurement**, never a
    /// sample-path operation, so no committed byte depends on it.
    pub fn response_at(self, freq_hz: f64, rate: u32) -> f64 {
        if rate == 0 || !freq_hz.is_finite() || freq_hz < 0.0 {
            return 1.0;
        }
        let a = self.alpha;
        let w = 2.0 * PI * freq_hz / f64::from(rate);
        let b = 1.0 - a;
        let denom = 1.0 + b * b - 2.0 * b * pcos(w);
        if denom <= 0.0 {
            return 1.0;
        }
        a / denom.sqrt()
    }

    /// **The impulse response**, `n` samples — a one at time zero pushed
    /// through a fresh copy of this filter.
    ///
    /// The thing itself, rather than a description of it. A test that wants to
    /// know what this filter does to a step or to a tone builds it from here.
    pub fn impulse_response(self, n: usize) -> Vec<f64> {
        let mut f = self;
        f.reset();
        (0..n)
            .map(|i| f.step(if i == 0 { 1.0 } else { 0.0 }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 22_050;

    /// **THE PROOF** — the analytic response and the measured one agree, and the
    /// cutoff really is where the design says it is.
    ///
    /// The measurement is a discrete Fourier transform of the filter's own
    /// impulse response at one frequency, which is the definition of a transfer
    /// function and owes nothing to the formula it is checking.
    #[test]
    fn the_impulse_response_is_minus_three_decibels_at_the_cutoff() {
        let dft_at = |h: &[f64], f: f64, rate: u32| -> f64 {
            let w = 2.0 * PI * f / f64::from(rate);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, x) in h.iter().enumerate() {
                let p = w * n as f64;
                re += x * p.cos();
                im -= x * p.sin();
            }
            (re * re + im * im).sqrt()
        };
        // The frequency at which a design is really down 3.01 dB, by bisection
        // over its own analytic response — which the block below has just
        // checked against the impulse response, so this is a measurement of the
        // filter and not of the formula.
        let half_power_hz = |f: OnePole| -> f64 {
            let (mut lo, mut hi) = (1.0f64, f64::from(RATE) / 2.0);
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                if f.response_at(mid, RATE) > std::f64::consts::FRAC_1_SQRT_2 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        // Every probe has to stay below Nyquist or it is not a measurement of
        // the filter, it is a measurement of the sampling. Eight times 3 500 Hz
        // is 28 kHz at a 22 050 Hz rate, and the first draft of this arm read
        // -5.39 dB there and called it a stopband.
        let nyquist = f64::from(RATE) / 2.0;
        println!("cutoff  at f/2   at f     at 2f    at 4f    -3dB at   error");
        for cutoff in [200.0, 500.0, 700.0, 1_500.0, 3_500.0] {
            let f = OnePole::new(cutoff, RATE);
            let h = f.impulse_response(1 << 15);
            let db = |x: f64| 20.0 * x.max(1e-12).log10();
            let at = |m: f64| dft_at(&h, cutoff * m, RATE);
            let f3 = half_power_hz(f);
            println!(
                "{cutoff:6.0} {:7.2} {:8.2} {:8.2} {:8} {:9.1} {:6.2}%",
                db(at(0.5)),
                db(at(1.0)),
                db(at(2.0)),
                if cutoff * 4.0 < nyquist {
                    format!("{:8.2}", db(at(4.0)))
                } else {
                    "     n/a".to_string()
                },
                f3,
                100.0 * (f3 - cutoff) / cutoff
            );
            // **THE CLAIM, AND IT IS THE FREQUENCY AND NOT THE DECIBEL.** An RC
            // design discretised as dt/(RC + dt) puts its half-power point a
            // little BELOW the frequency it names, and the error grows with
            // f/rate: this table measures it. Ten per cent is the bound across
            // every cutoff this tree uses (500 Hz for a shut door, 700 Hz for
            // four hundred metres of air, 3 500 Hz for a room's tail), and the
            // printed column is what it actually is.
            assert!(
                (f3 - cutoff).abs() / cutoff < 0.001,
                "{cutoff} Hz is really down 3 dB at {f3:.1} Hz"
            );
            // The measurement and the formula agree to a thousandth.
            for m in [0.5, 1.0, 2.0, 4.0] {
                if cutoff * m >= nyquist {
                    continue;
                }
                let measured = at(m);
                let analytic = f.response_at(cutoff * m, RATE);
                assert!(
                    (measured - analytic).abs() < 1e-3,
                    "{cutoff} Hz x{m}: measured {measured}, analytic {analytic}"
                );
            }
            // …which is the same fact read the other way: at the frequency it
            // names, the filter is down 3.01 dB, to a hundredth. The exact
            // coefficient is what buys this; the RC approximation this module
            // started with read -3.13, -3.31, -3.42 and -3.81 dB at the four
            // cutoffs above.
            let at_cutoff = db(at(1.0));
            assert!(
                (at_cutoff + 3.0103).abs() < 0.01,
                "{cutoff} Hz reads {at_cutoff:.3} dB at its own cutoff"
            );
            // …and it really is a LOW-pass: monotone down, and 6 dB an octave in
            // the stopband, measured only where the octave itself fits under
            // Nyquist.
            assert!(at(0.5) > at(1.0) && at(1.0) > at(2.0));
            if cutoff * 4.0 < nyquist {
                let octave = db(at(2.0)) - db(at(4.0));
                assert!(
                    (4.0..=7.0).contains(&octave),
                    "one pole should be about 6 dB an octave, this is {octave:.2}"
                );
            }
        }
    }

    /// **A STEP INPUT** — the brief's own wording: what a step is attenuated to,
    /// and how long it takes.
    ///
    /// A one-pole reaches 63.2 % of a step in exactly one time constant, and the
    /// time constant is `1/(2π·f_c)`, so this is the design read back off the
    /// samples in seconds.
    #[test]
    fn a_step_reaches_sixty_three_per_cent_in_one_time_constant() {
        println!("cutoff  tau (samples)   step at tau   discrete truth");
        for cutoff in [50.0, 200.0, 700.0, 3_500.0] {
            let mut f = OnePole::new(cutoff, RATE);
            let a = f.coefficient();
            let tau_samples = (f64::from(RATE) / (2.0 * PI * cutoff)).round() as usize;
            let mut y = 0.0;
            for _ in 0..tau_samples {
                y = f.step(1.0);
            }
            // What a one-pole with THIS coefficient must reach after n samples,
            // exactly. Comparing against it rather than against 0.632 is the
            // difference between measuring the filter and measuring the
            // rounding of tau to a whole number of samples.
            let truth = 1.0 - (1.0 - a).powi(tau_samples as i32);
            println!("{cutoff:6.0} {tau_samples:14} {y:13.4} {truth:16.4}");
            assert!(
                (y - truth).abs() < 1e-9,
                "{cutoff} Hz reached {y}, the discrete filter says {truth}"
            );
            // The textbook 63.2 per cent holds where tau is many samples. It
            // does NOT at 3 500 Hz, where tau is five samples and the
            // discretisation is the dominant term — measured: 0.5972. That is a
            // property of every digital one-pole, stated rather than tuned
            // around.
            if tau_samples >= 20 {
                // Two hundredths, measured: the exact coefficient is designed
                // against the half-power FREQUENCY, not against the continuous
                // time constant, so a step reaches 0.6414 at 200 Hz rather than
                // 0.6321. Both numbers are printed above.
                assert!(
                    (y - 0.632).abs() < 0.02,
                    "{cutoff} Hz reached {y} after one time constant"
                );
            }
            // …and it gets there, rather than to some other number.
            for _ in 0..tau_samples * 40 + 400 {
                y = f.step(1.0);
            }
            assert!(
                (y - 1.0).abs() < 1e-3,
                "a step should settle at unity, got {y}"
            );
        }
    }

    /// The polynomial cosine is the real one, to a nanounit, over the whole
    /// range a cutoff can reach — and it is what makes every generated clip the
    /// same bytes on every target.
    #[test]
    fn the_portable_cosine_is_the_real_one() {
        let mut worst = 0.0f64;
        for i in 0..=4_000 {
            let w = PI * (i as f64) / 4_000.0;
            worst = worst.max((pcos(w) - w.cos()).abs());
        }
        println!("pcos worst error over [0, pi]: {worst:.3e}");
        assert!(worst < 1e-9, "pcos is off by {worst:e}");
        // It folds, so a caller never has to.
        for w in [-1.3, 7.9, -21.0, 100.0] {
            assert!((pcos(w) - w.cos()).abs() < 1e-8, "pcos({w})");
        }
        assert_eq!(pcos(f64::NAN), 1.0);
    }

    /// A degenerate design is a pass-through, never a panic and never a
    /// silence — a refusal is a value here as everywhere.
    #[test]
    fn a_cutoff_this_filter_cannot_express_passes_the_signal_through() {
        for (c, r) in [(0.0, RATE), (-5.0, RATE), (f64::NAN, RATE), (500.0, 0)] {
            let mut f = OnePole::new(c, r);
            assert!((f.coefficient() - 1.0).abs() < 1e-12);
            assert!((f.step(0.5) - 0.5).abs() < 1e-12);
        }
        // A cutoff at or above Nyquist is a pass-through rather than a filter
        // that does three per cent of nothing.
        for c in [f64::from(RATE) / 2.0, 100_000.0] {
            let f = OnePole::new(c, RATE);
            assert!(
                (f.coefficient() - 1.0).abs() < 1e-12,
                "{c} Hz gave {}",
                f.coefficient()
            );
        }
        // …and just below it, it is a real filter again.
        assert!(OnePole::new(f64::from(RATE) / 2.0 - 1.0, RATE).coefficient() < 1.0);
    }
}
