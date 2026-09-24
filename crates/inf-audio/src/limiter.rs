//! **The master-bus limiter** (VEH3e audit) — a soft-knee, look-ahead peak
//! limiter every sink this crate plays through sits behind: the device and the
//! render-to-file door build their kira mixer with ONE main track
//! (`backend::master_track`), and this is the only effect on it.
//!
//! # Why
//!
//! The mix is a sum of independently-gained voices, and a sum has no ceiling:
//! the VEH3e course render (a turbo V8's full-throttle wheelspin under the
//! driver's ear) peaked at full scale and clipped 30 samples with the layer
//! gains already set for headroom. Gains cannot fix that without making every
//! quiet moment quieter; a limiter on the bus is what a mixer does instead.
//!
//! # The shape — and why it cannot overshoot
//!
//! Per stereo frame, the frame's peak `p = max(|l|, |r|)` asks for a gain
//! `r(p) = knee(p) / p`, where [`knee`] is the identity up to
//! [`KNEE_START`] and a rational curve above it that approaches
//! [`CEILING`] and never reaches it. The gain APPLIED to a frame is:
//!
//! 1. `m[n]` = the minimum requested gain over the last `W` frames;
//! 2. `b[n]` = the mean of the last `W` values of `m` (a boxcar — the attack
//!    ramp, so the gain glides down instead of stepping);
//! 3. `g[n] = min(b[n], g[n-1] + 1 / (RELEASE_S · rate))` (a linear release);
//!
//! and the frame that leaves is the one `W − 1` frames old. Every `m` the
//! boxcar averages at that moment is a minimum over a window that contains
//! the leaving frame, so `b ≤ r(p_leaving)` and the output peak is at most
//! `knee(p) < CEILING`: **a guarantee, not a tuning.**
//!
//! Plain `f32` arithmetic — no `exp`, no `log`, no trig — so the offline door's
//! bytes are the same on every platform the mixer's own arithmetic is.

/// The output ceiling, linear full scale (−0.35 dBFS). The limiter's output
/// peak is strictly below it.
pub const CEILING: f32 = 0.96;

/// Where the knee starts, linear. A peak below it passes untouched.
pub const KNEE_START: f32 = 0.72;

/// The look-ahead, seconds — also the attack ramp's length.
pub const LOOKAHEAD_S: f64 = 0.0015;

/// How long the gain takes to come back from full reduction to unity, seconds.
pub const RELEASE_S: f64 = 0.120;

/// **The soft knee**: the output peak a frame of input peak `p` is allowed.
/// The identity up to [`KNEE_START`]; above it `k + (c−k)·x / (x + (c−k))`
/// with `x = p − k` — continuous with slope one at the knee, and approaching
/// [`CEILING`] from below as `p` grows.
pub fn knee(p: f32) -> f32 {
    let p = if p.is_finite() { p.abs() } else { 0.0 };
    if p <= KNEE_START {
        return p;
    }
    let span = CEILING - KNEE_START;
    let x = p - KNEE_START;
    KNEE_START + span * x / (x + span)
}

/// The gain a frame of peak `p` asks for.
fn wanted(p: f32) -> f32 {
    if p <= KNEE_START {
        1.0
    } else {
        knee(p) / p
    }
}

/// **The limiter's state** — three rings of `W` and the release envelope.
#[derive(Clone, Debug)]
pub struct MasterLimiter {
    window: usize,
    delay: Vec<(f32, f32)>,
    wants: Vec<f32>,
    mins: Vec<f32>,
    at: usize,
    gain: f32,
    release_step: f32,
    /// How many frames left with any gain reduction applied.
    reduced_frames: u64,
    /// The deepest gain the limiter has applied, linear.
    deepest: f32,
}

impl MasterLimiter {
    /// A limiter for `sample_rate` frames a second.
    pub fn new(sample_rate: u32) -> Self {
        let rate = f64::from(sample_rate.max(1));
        let window = ((LOOKAHEAD_S * rate).round() as usize).max(1);
        Self {
            window,
            delay: vec![(0.0, 0.0); window],
            wants: vec![1.0; window],
            mins: vec![1.0; window],
            at: 0,
            gain: 1.0,
            release_step: (1.0 / (RELEASE_S * rate)) as f32,
            reduced_frames: 0,
            deepest: 1.0,
        }
    }

    /// The look-ahead in frames — the latency the limiter adds.
    pub fn latency_frames(&self) -> usize {
        self.window - 1
    }

    /// How many frames left with the gain below unity.
    pub fn reduced_frames(&self) -> u64 {
        self.reduced_frames
    }

    /// The deepest gain applied so far, linear (`1` = never engaged).
    pub fn deepest_gain(&self) -> f32 {
        self.deepest
    }

    /// **One stereo frame in, the frame `W − 1` frames old out**, limited.
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
        let (l, r) = (clean(l), clean(r));
        let w = self.window;
        let i = self.at;
        self.delay[i] = (l, r);
        self.wants[i] = wanted(l.abs().max(r.abs()));
        let mut m = 1.0f32;
        for v in &self.wants {
            m = m.min(*v);
        }
        self.mins[i] = m;
        let mut sum = 0.0f32;
        for v in &self.mins {
            sum += *v;
        }
        let boxcar = (sum / w as f32).min(1.0);
        // The boxcar is a mean of values each at or below the leaving frame's
        // request — but a sum of f32 rounds, so the ceiling is enforced on the
        // leaving frame's own request as well (a no-op in exact arithmetic).
        self.at = (i + 1) % w;
        let (ol, or) = self.delay[self.at];
        let own = self.wants[self.at];
        self.gain = (self.gain + self.release_step).min(boxcar).min(own);
        if self.gain < 1.0 {
            self.reduced_frames += 1;
            self.deepest = self.deepest.min(self.gain);
        }
        (ol * self.gain, or * self.gain)
    }

    /// [`process`](Self::process) over an interleaved `L R` buffer, in place.
    pub fn process_interleaved(&mut self, buf: &mut [f32]) {
        for f in buf.chunks_exact_mut(2) {
            let (l, r) = self.process(f[0], f[1]);
            f[0] = l;
            f[1] = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knee_is_the_identity_below_and_never_reaches_the_ceiling() {
        assert_eq!(knee(0.5), 0.5);
        assert_eq!(knee(KNEE_START), KNEE_START);
        let mut last = KNEE_START;
        for i in 1..2000 {
            let p = KNEE_START + i as f32 * 0.01;
            let k = knee(p);
            assert!(k > last - 1e-6 && k < CEILING, "knee({p}) = {k}");
            last = k;
        }
        // Slope one at the knee: no corner to hear.
        let d = (knee(KNEE_START + 1e-3) - KNEE_START) / 1e-3;
        assert!((d - 1.0).abs() < 0.01, "slope {d}");
    }

    #[test]
    fn a_full_scale_burst_leaves_below_the_ceiling() {
        let mut lim = MasterLimiter::new(48_000);
        let mut peak = 0.0f32;
        // Quiet, then a 3x-overdriven square burst, then quiet again.
        for n in 0..48_000 {
            let x = if (10_000..20_000).contains(&n) {
                if (n / 40) % 2 == 0 {
                    3.0
                } else {
                    -3.0
                }
            } else {
                0.3 * if (n / 40) % 2 == 0 { 1.0 } else { -1.0 }
            };
            let (l, r) = lim.process(x, -x);
            peak = peak.max(l.abs()).max(r.abs());
        }
        assert!(peak < CEILING, "peak {peak}");
        assert!(lim.deepest_gain() < 0.35, "{}", lim.deepest_gain());
        assert!(lim.reduced_frames() > 10_000);
    }

    #[test]
    fn quiet_material_passes_bit_exact_after_the_latency() {
        let mut lim = MasterLimiter::new(48_000);
        let d = lim.latency_frames();
        let input: Vec<f32> = (0..4800)
            .map(|n| 0.7 * ((n % 97) as f32 / 97.0 - 0.5))
            .collect();
        let out: Vec<f32> = input.iter().map(|x| lim.process(*x, *x).0).collect();
        for n in d..input.len() {
            assert_eq!(out[n], input[n - d]);
        }
        assert_eq!(lim.reduced_frames(), 0);
    }

    #[test]
    fn the_gain_releases_back_to_unity() {
        let mut lim = MasterLimiter::new(48_000);
        for _ in 0..100 {
            lim.process(2.0, 2.0);
        }
        // RELEASE_S of quiet brings the gain all the way home.
        for _ in 0..((RELEASE_S * 48_000.0) as usize + 200) {
            lim.process(0.1, 0.1);
        }
        let (l, _) = lim.process(0.1, 0.1);
        assert_eq!(l, 0.1);
    }
}
