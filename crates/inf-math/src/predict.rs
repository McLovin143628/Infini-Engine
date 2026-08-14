//! **Deterministic dead reckoning** (P28.4, clause 1): where the camera will be
//! in `h` ticks, as a pure function of the camera history the host committed.
//!
//! # Why this is here and not in the streamer
//!
//! `inf-stream` arbitrates and states in its own crate docs that it *"holds no
//! camera, no matrix"*. `inf-render` holds both and also holds a device. The
//! prediction is neither: it is f64 world arithmetic over a short history, the
//! same shape as [`FloatingOrigin`](crate::FloatingOrigin) — which is the other
//! thing in this crate whose whole job is to follow a camera.
//!
//! # The determinism argument, precisely
//!
//! The direction memo's §3 deviation replaced a learned predictor with this one
//! because *"a learned predictor is untestable under house gates"*. That buys
//! nothing unless the analytic one is genuinely a pure function, so it is worth
//! writing down exactly what is guaranteed and what the guarantee rests on.
//!
//! **Guaranteed here, structurally:**
//!
//! * There is no clock, no frame counter, no `Instant`, no adapter and no
//!   interior mutability in this module. [`dead_reckon`] reads its argument and
//!   nothing else.
//! * Every transcendental is a [`portable`](crate::portable) one
//!   (`pacos64`/`psin64`/`pcos64`). `f64::acos` is libm and libm is not
//!   bit-portable — the P14 law — and a prediction reaches a *want set*, which a
//!   gate compares between two hosts. This is the same reasoning P24.2 applied
//!   to `root_motion`'s yaw, one crate over.
//! * The horizon is in **ticks**, never milliseconds. A millisecond is wall
//!   clock; a tick is committed. A host converts once, at its own door, with
//!   [`horizon_ticks`] — and the conversion is a property of the host's fixed
//!   step, which is committed too.
//!
//! **Rests on the caller, and cannot be checked from in here:** that a
//! [`CameraSample`] is *committed* — that its pose is a function of the
//! committed input stream and of nothing else. [`CameraHistory::commit`]
//! enforces the half that is mechanical (a tick index that only ever advances,
//! so a host cannot append twice for one step or replay an old one), and the
//! half that is not is a property of the host:
//!
//! * the shipped player's camera is `RuntimeSim::camera_focus`, a fold of actor
//!   positions, which is a pure function of the committed input — it commits;
//! * a gate's scripted path is a pure function of the step index — it commits;
//! * **the editor viewport's flycam is driven by OS input at render rate and is
//!   not committed at all** — it does not commit, so its history stays empty,
//!   [`dead_reckon`] answers `None`, and the editor streams exactly as it did
//!   before this module existed. That is the honest bound, not an oversight.
//!
//! An empty history is therefore the real enable flag, and it fails safe: a
//! consumer that gets `None` emits no speculative want and lands on exactly the
//! analytic floor.

use glam::DVec3;

use crate::portable::{pacos64, pcos64, psin64};

/// How many committed samples the history holds, and therefore how long the
/// secant that estimates velocity is.
///
/// **Six**, measured (`docs/memos/p28-4-predictive-prefetch.md` §2). The
/// estimate is a difference of two poses over the window, so a longer window
/// divides the pose quantization by more — and lags a *change* in velocity by
/// proportionally more. Six ticks is 100 ms at the shipped 60 Hz step: long
/// enough that the secant is not one step's noise, short enough that a whip's
/// acceleration phase is not averaged away.
pub const PREDICT_HISTORY: usize = 6;

/// The most a prediction may turn the view, in radians.
///
/// A clamp and not a tuning knob. Dead reckoning assumes the rate holds; past
/// half a turn that assumption has stopped being an extrapolation and become a
/// guess, and a guess that wrong does not merely waste bytes — it takes slots in
/// a lane the floor can reclaim, on frames the floor is already contending for
/// them. Clamping keeps the want set inside the cone the motion actually
/// justifies. Reported as [`Prediction::clamped`] so a host can see it bind.
pub const PREDICT_MAX_TURN: f64 = std::f64::consts::PI;

/// Below this the two ends of the window are one direction and there is no arc
/// to extrapolate — including the antiparallel case, which is a 180° flip
/// inside one window and has no axis at all.
const AXIS_EPS: f64 = 1e-9;

/// **The horizon, in ticks, for a horizon in milliseconds at a `hz` fixed step.**
///
/// The one place a millisecond is allowed to touch the predictor, and it is at
/// the host's door rather than inside it: `hz` is the host's *fixed* step rate,
/// which is committed, so the product is committed too. Rounds to nearest and
/// never returns 0 — a zero horizon is a predictor that predicts the present.
#[inline]
pub const fn horizon_ticks(horizon_ms: u32, hz: u32) -> u32 {
    let t = (horizon_ms as u64 * hz as u64 + 500) / 1000;
    if t == 0 {
        1
    } else {
        t as u32
    }
}

/// One committed camera pose: the tick it belongs to, and where the camera was.
///
/// World space and f64, because the history outlives a floating-origin rebase
/// and a render-local sample would jump a kilometre when it happened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraSample {
    /// The **fixed-step** index this pose belongs to. Strictly increasing, and
    /// the only notion of time in this module.
    pub tick: u64,
    /// The eye, in world space (metres).
    pub eye: DVec3,
    /// The unit view direction.
    pub forward: DVec3,
    /// The view's up vector.
    pub up: DVec3,
}

/// The last [`PREDICT_HISTORY`] committed poses, oldest first.
///
/// Fixed capacity and no allocation after the first push, because it is written
/// once per fixed step for the life of a session.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraHistory {
    samples: Vec<CameraSample>,
    refused: u64,
}

impl CameraHistory {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            refused: 0,
        }
    }

    /// **Commit one pose.** `false` — and nothing is stored — when `sample.tick`
    /// does not strictly advance.
    ///
    /// The refusal is the mechanical half of "committed": a host that samples
    /// per *render* frame calls this several times for one step, and a host that
    /// re-enters a step calls it with a tick it has already used. Either would
    /// make the velocity estimate a function of the frame rate, which is the one
    /// thing this whole module exists not to be. Refusing is not a courtesy to
    /// such a host — it is what keeps the *other* hosts' answers reproducible.
    pub fn commit(&mut self, sample: CameraSample) -> bool {
        if self.samples.last().is_some_and(|s| sample.tick <= s.tick) {
            self.refused += 1;
            return false;
        }
        if self.samples.len() == PREDICT_HISTORY {
            self.samples.remove(0);
        }
        self.samples.push(sample);
        true
    }

    /// Samples held.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The most recent committed pose.
    #[inline]
    pub fn latest(&self) -> Option<CameraSample> {
        self.samples.last().copied()
    }

    /// The oldest pose still in the window — the other end of the secant.
    #[inline]
    pub fn oldest(&self) -> Option<CameraSample> {
        self.samples.first().copied()
    }

    /// Pushes refused for not advancing the tick. **A counter and not a
    /// silence**: a host wired to the wrong loop reads as "the predictor does
    /// nothing", and this is the number that says why.
    #[inline]
    pub fn refused(&self) -> u64 {
        self.refused
    }

    /// Forget everything — a level switch, an ejected PIE session, a camera cut.
    ///
    /// A cut is exactly the case a velocity estimate must not span: the secant
    /// across it is a teleport per tick, and the prediction it produces is a
    /// want set for somewhere the camera has never been.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Where the camera is predicted to be, and what it took to say so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Predicted eye, world space.
    pub eye: DVec3,
    /// Predicted unit view direction.
    pub forward: DVec3,
    /// Predicted up.
    pub up: DVec3,
    /// The horizon this answers for, in ticks.
    pub horizon_ticks: u32,
    /// The arc the prediction turned through, radians (after the clamp).
    pub turn: f64,
    /// Whether [`PREDICT_MAX_TURN`] bound the turn.
    pub clamped: bool,
}

/// **Dead-reckon `horizon_ticks` ahead of the newest committed sample.**
///
/// `None` when the window holds fewer than two poses, or when the two ends
/// share a tick — there is no rate to extrapolate and inventing one is worse
/// than declining.
///
/// Two independent extrapolations over the same window, because a camera has
/// two degrees of freedom that fail differently:
///
/// * **linear** — a secant over the window, `v = (eye_new − eye_old) / span`
///   metres per tick, applied for `horizon_ticks`;
/// * **angular** — the *arc* from the old view direction to the new one, its
///   axis and its rate, applied for `horizon_ticks` about the same axis
///   (Rodrigues, portable trig). Not a lerp of the two directions: a lerp is
///   bounded by the endpoints and cannot extrapolate at all, which is precisely
///   what a whip-pan needs.
///
/// The `up` vector rides the same rotation, so the predicted frame is the real
/// frame turned — a predicted `forward` with the committed `up` is a view that
/// rolls as the camera pans.
pub fn dead_reckon(history: &CameraHistory, horizon_ticks: u32) -> Option<Prediction> {
    let old = history.oldest()?;
    let new = history.latest()?;
    if new.tick <= old.tick {
        return None;
    }
    let span = (new.tick - old.tick) as f64;
    let h = f64::from(horizon_ticks);

    let eye = new.eye + (new.eye - old.eye) * (h / span);

    let a = old.forward.normalize_or_zero();
    let b = new.forward.normalize_or_zero();
    let axis = a.cross(b);
    let axis_len = axis.length();
    // Parallel, antiparallel, or a degenerate sample: no arc, so the view is
    // carried forward unturned rather than turned by a made-up axis.
    if axis_len <= AXIS_EPS || a.length_squared() <= 0.0 || b.length_squared() <= 0.0 {
        return Some(Prediction {
            eye,
            forward: new.forward,
            up: new.up,
            horizon_ticks,
            turn: 0.0,
            clamped: false,
        });
    }
    let axis = axis / axis_len;
    let arc = pacos64(a.dot(b).clamp(-1.0, 1.0));
    let want = arc * (h / span);
    let clamped = want > PREDICT_MAX_TURN;
    let turn = if clamped { PREDICT_MAX_TURN } else { want };

    Some(Prediction {
        eye,
        forward: rodrigues(new.forward, axis, turn),
        up: rodrigues(new.up, axis, turn),
        horizon_ticks,
        turn,
        clamped,
    })
}

/// Rotate `v` about the unit `axis` by `angle` radians — Rodrigues, in portable
/// trig so two hosts get the same bits.
fn rodrigues(v: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (s, c) = (psin64(angle), pcos64(angle));
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tick: u64, eye: DVec3, forward: DVec3) -> CameraSample {
        CameraSample {
            tick,
            eye,
            forward,
            up: DVec3::Y,
        }
    }

    /// A camera turning at a constant rate about `up`, sampled per tick.
    fn spin(ticks: impl IntoIterator<Item = u64>, per_tick: f64) -> CameraHistory {
        let mut h = CameraHistory::new();
        for t in ticks {
            let a = per_tick * t as f64;
            h.commit(sample(
                t,
                DVec3::ZERO,
                DVec3::new(psin64(a), 0.0, -pcos64(a)),
            ));
        }
        h
    }

    /// **The whole contract in one arm**: same committed history, same answer,
    /// bit for bit, however many times it is asked.
    #[test]
    fn one_history_predicts_one_pose_bit_for_bit() {
        let h = spin(0..6, 0.05);
        let a = dead_reckon(&h, 18).expect("six samples");
        for _ in 0..8 {
            let b = dead_reckon(&h, 18).expect("six samples");
            assert_eq!(a.eye.to_array(), b.eye.to_array());
            assert_eq!(a.forward.to_array(), b.forward.to_array());
            assert_eq!(a.up.to_array(), b.up.to_array());
        }
        // …and a SECOND history built by a different route (the samples appended
        // in the same order by a different caller) is the same history.
        let mut other = CameraHistory::new();
        for t in 0..6u64 {
            let x = 0.05 * t as f64;
            other.commit(sample(
                t,
                DVec3::ZERO,
                DVec3::new(psin64(x), 0.0, -pcos64(x)),
            ));
        }
        let c = dead_reckon(&other, 18).expect("six samples");
        assert_eq!(a, c);
    }

    /// **A tick that does not advance is refused**, and the refusal is counted.
    ///
    /// The arm that would fail if the history ever became a function of how
    /// often a host happens to call it: pushing the same tick ten times must
    /// leave the prediction exactly where one push left it.
    #[test]
    fn a_repeated_tick_changes_neither_the_window_nor_the_answer() {
        let mut h = spin(0..4, 0.05);
        let before = dead_reckon(&h, 12).expect("four samples");
        let len = h.len();
        for _ in 0..10 {
            assert!(!h.commit(sample(3, DVec3::new(999.0, 0.0, 0.0), DVec3::NEG_Z)));
            assert!(!h.commit(sample(0, DVec3::new(-999.0, 0.0, 0.0), DVec3::NEG_Z)));
        }
        assert_eq!(h.len(), len);
        assert_eq!(h.refused(), 20);
        assert_eq!(dead_reckon(&h, 12), Some(before));
    }

    /// The window is bounded and slides: the oldest sample leaves.
    #[test]
    fn the_window_holds_exactly_the_last_six_committed_poses() {
        let mut h = CameraHistory::new();
        for t in 0..20u64 {
            assert!(h.commit(sample(t, DVec3::new(t as f64, 0.0, 0.0), DVec3::NEG_Z)));
        }
        assert_eq!(h.len(), PREDICT_HISTORY);
        assert_eq!(h.oldest().expect("full").tick, 20 - PREDICT_HISTORY as u64);
        assert_eq!(h.latest().expect("full").tick, 19);
    }

    /// Fewer than two poses is `None`, not a zero prediction — a consumer must
    /// be able to tell "no motion" from "no evidence".
    #[test]
    fn a_window_with_nothing_to_extrapolate_declines() {
        assert_eq!(dead_reckon(&CameraHistory::new(), 18), None);
        let mut one = CameraHistory::new();
        one.commit(sample(4, DVec3::ZERO, DVec3::NEG_Z));
        assert_eq!(dead_reckon(&one, 18), None);
        // …and a still camera predicts itself, which is a different answer.
        let mut still = CameraHistory::new();
        still.commit(sample(4, DVec3::ZERO, DVec3::NEG_Z));
        still.commit(sample(5, DVec3::ZERO, DVec3::NEG_Z));
        let p = dead_reckon(&still, 18).expect("two samples");
        assert_eq!(p.eye, DVec3::ZERO);
        assert_eq!(p.turn, 0.0);
        assert_eq!(p.forward, DVec3::NEG_Z);
    }

    /// **The extrapolation is an extrapolation.** A constant-rate spin sampled
    /// over five ticks, predicted eighteen ahead, must land where the spin
    /// really goes — *outside* the window, which a lerp of the two endpoints
    /// cannot reach at all.
    #[test]
    fn a_constant_spin_lands_where_the_spin_actually_goes() {
        const RATE: f64 = 0.05; // rad/tick — 3 rad/s at 60 Hz
        let h = spin(0..6, RATE);
        let p = dead_reckon(&h, 18).expect("six samples");
        let truth = 0.05 * (5.0 + 18.0);
        let want = DVec3::new(psin64(truth), 0.0, -pcos64(truth));
        assert!(
            (p.forward - want).length() < 1e-6,
            "predicted {:?}, the spin reaches {want:?}",
            p.forward
        );
        // ANTI-VACUITY, read off the fixture: the answer is well outside the
        // window it was derived from, so this cannot be satisfied by returning
        // an endpoint.
        let newest = h.latest().expect("six samples").forward;
        assert!(
            (p.forward - newest).length() > 0.5,
            "the prediction did not leave the window"
        );
        assert!(!p.clamped);
    }

    /// Linear motion, on its own, over a window whose ends are several ticks
    /// apart — the `span` divisor is the half a two-sample estimate gets for
    /// free and this one does not.
    #[test]
    fn a_constant_dolly_extrapolates_at_the_windows_own_rate() {
        let mut h = CameraHistory::new();
        for t in 0..6u64 {
            // Ticks 0,2,4,… — a host stepping twice per commit. The rate is per
            // TICK, so the span divisor has to be the tick difference and not
            // the sample count.
            h.commit(sample(
                t * 2,
                DVec3::new(0.0, 0.0, -0.5 * (t * 2) as f64),
                DVec3::NEG_Z,
            ));
        }
        let p = dead_reckon(&h, 20).expect("six samples");
        assert!(
            (p.eye.z - (-0.5 * (10.0 + 20.0))).abs() < 1e-9,
            "{:?}",
            p.eye
        );
    }

    /// The clamp binds, and says so.
    #[test]
    fn a_turn_past_half_a_revolution_is_clamped_and_reported() {
        let h = spin(0..6, 0.5); // 2.5 rad over the window
        let p = dead_reckon(&h, 60).expect("six samples");
        assert!(p.clamped);
        assert_eq!(p.turn, PREDICT_MAX_TURN);
        // The control: the same history at a horizon the rate does not overrun.
        let q = dead_reckon(&h, 4).expect("six samples");
        assert!(!q.clamped && q.turn < PREDICT_MAX_TURN);
    }

    /// A history cleared at a cut predicts nothing, rather than predicting the
    /// teleport.
    #[test]
    fn a_cut_clears_rather_than_spanning() {
        let mut h = spin(0..6, 0.05);
        h.clear();
        assert_eq!(dead_reckon(&h, 18), None);
        assert!(h.commit(sample(0, DVec3::ZERO, DVec3::NEG_Z)));
    }

    /// The millisecond door: rounds to nearest, never to zero.
    #[test]
    fn the_horizon_converts_at_the_hosts_own_step_rate() {
        assert_eq!(horizon_ticks(200, 60), 12);
        assert_eq!(horizon_ticks(300, 60), 18);
        assert_eq!(horizon_ticks(500, 60), 30);
        assert_eq!(horizon_ticks(300, 30), 9);
        assert_eq!(horizon_ticks(1, 60), 1, "a horizon never rounds to nothing");
        assert_eq!(horizon_ticks(0, 60), 1);
    }

    /// **The non-test half of this file, comment lines dropped** — what both
    /// source pins below read.
    ///
    /// Two exclusions, each for a reason a green-because-it-cannot-see arm would
    /// have: the doc comments above name `f64::acos` and `Instant` in order to
    /// say why neither is used, and the pins themselves carry every banned
    /// spelling as a string literal — a scan over the whole file would be
    /// satisfied by its own fixture, which is the P28.1 audit's finding in
    /// miniature.
    ///
    /// `split` on the attribute rather than on a line shape, and `lines` rather
    /// than a newline search, so the scope it reads is the same one on a CRLF
    /// checkout as on an LF one — the P22 law about a `.rs` file read by a test.
    fn non_test_body() -> String {
        include_str!("predict.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("split always yields one")
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<String>()
    }

    /// **No libm on this path.** The rotation is Rodrigues in portable trig, so
    /// this is a source-level pin on the module that computes a want set two
    /// hosts compare: the P14 law, armed where it is decided.
    #[test]
    fn the_predictor_spells_no_libm_transcendental() {
        let body = non_test_body();
        for banned in [".acos(", ".asin(", ".atan2(", ".sin(", ".cos(", ".cbrt("] {
            assert!(
                !body.contains(banned),
                "the predictor calls {banned} — libm is not bit-portable (P14)"
            );
        }
        // …and the anti-vacuity half: the portable ones ARE called, so this is
        // not a pin on a module that does no trigonometry.
        assert!(body.contains("psin64(") && body.contains("pcos64(") && body.contains("pacos64("));
    }

    /// **No clock, no entropy, no interior mutability** — the *other* half of
    /// the purity claim, which was prose until the P28.4 audit measured what
    /// happens without it.
    ///
    /// The module's own header says "no clock, no frame counter, no `Instant`
    /// and no interior mutability", and the arms above enforce none of it. They
    /// cannot: `one_history_predicts_one_pose_bit_for_bit` re-asks inside **one
    /// process**, so an impurity whose reading is constant for a process and
    /// differs between two of them is invisible to it, and
    /// `a_constant_spin_lands_where_the_spin_actually_goes` compares against a
    /// closed form at `1e-6`, so anything smaller than that sails through.
    /// **Measured**, in the P28.4 audit's mutation matrix: a wall clock scaled
    /// to `1e-9` (`SystemTime::now() … as_secs() % 2`) passed all 59 arms of
    /// this crate. That is precisely the P14 failure — two hosts, two want sets
    /// — and the only thing that can see it is a pin on the source.
    ///
    /// A ban list and not an allowlist, deliberately: the subject is *reaching
    /// outside the argument*, and the spellings for that are few and stable;
    /// the P23 law's allowlist shape is for statement sequences, which this is
    /// not.
    #[test]
    fn the_predictor_reads_no_clock_and_holds_no_state() {
        const BANNED: [&str; 12] = [
            "Instant",
            "SystemTime",
            "std::time",
            "elapsed(",
            "process::id",
            "thread_rng",
            "random(",
            "RefCell",
            "AtomicU",
            "AtomicI",
            "OnceLock",
            "static mut ",
        ];
        let offenders = |src: &str| -> Vec<&'static str> {
            BANNED.iter().copied().filter(|b| src.contains(b)).collect()
        };

        let body = non_test_body();
        assert!(
            offenders(&body).is_empty(),
            "the predictor reaches outside its argument: {:?} — a reading that is              constant for one process and differs between two is exactly what the              bit-for-bit arm cannot see (P14)",
            offenders(&body)
        );

        // **The pin is built to falsify** (the P22 law): the same predicate over
        // a poisoned copy of the real body must reject it. Without this the arm
        // is a statement about a string that could be empty.
        let poisoned = format!("{body} let h = h + std::time::SystemTime::now().elapsed();");
        assert!(
            !offenders(&poisoned).is_empty(),
            "the pin cannot see a clock even when one is put in front of it"
        );
        // …and the scope really is this module's non-test half: the function the
        // pin exists for is in it, and the fixtures that carry the banned
        // spellings as literals are not.
        assert!(
            body.contains("pub fn dead_reckon"),
            "the pin read no source"
        );
        assert!(
            !body.contains("BANNED"),
            "the split let this test's own fixture into the scanned scope"
        );
    }
}
