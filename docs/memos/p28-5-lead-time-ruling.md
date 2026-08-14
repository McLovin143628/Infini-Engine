# P28.5 — the lead-time ruling: a ROADMAP clause its own gate refuted

**Status:** decided 2026-08-14, during P28.5. This memo records a **deviation
from a ROADMAP clause**, which the house rule says requires a memo and not
silence. The clause is P28.4's first:

> deterministic dead-reckoning over committed input history (camera velocity +
> angular momentum, **200–500 ms horizon** — a pure function, the memo's
> neural-predictor deviation)

`inf_render::DEFAULT_PREDICT_HORIZON_TICKS` is **0** as of this batch. The
200–500 ms band is not shipped, and this is the measurement that says why.

---

## 1. WHAT WAS MEASURED, AND BY WHAT

P28.4 shipped 18 ticks (300 ms at the 60 Hz fixed step), the middle of the
band, chosen by a sweep over the band itself. The P28.4 **audit** then built the
control that sweep never had: `dead_reckon` at `h = 0` scales the secant by
nothing and turns by nothing, so it returns the newest committed pose exactly —
same lane, same cap, same rank, **no lead**. It needs no mutation; it is
reachable through the shipped API.

On `crates/inf-render/tests/whip_pan.rs`'s 360° path (still → ramp → constant
120°/s → ramp → still, 260 ticks, 800 pages), against **131** blur frames and
**19 872** blur tiles with the predictor OFF:

| lead (ticks) | **0** | 3 | 6 | 12 | **18 (P28.4's default)** | 24 | 36 |
|---|---|---|---|---|---|---|---|
| blur frames | **105** | 108 | 113 | 115 | **115** | 112 | 124 |
| blur tiles | **18 752** | 18 800 | 18 912 | 18 976 | **18 976** | 18 896 | 19 152 |
| arrival-window blur / 1 728 | **64** | 96 | 144 | 176 | **176** | 128 | 224 |
| arrival frames | **2** | — | — | — | **7** | — | — |

Three readings, and they agree:

* **h = 0 beats the whole band**, and the band is flat — nine frames of 131
  separate its best from its worst, so P28.4's "no worse than either end" could
  not distinguish 18 from 24 and was never going to find this.
* **The aggregate is monotone in the lead** away from the ramps' noise: the
  more lead, the more blur.
* **The arrival window says it loudest** — tiles a surface justifies on the tick
  it *enters view*, which is the only class a lead time can serve at all. 64
  against 176, over 2 frames against 7, where OFF is 384 over 15.

## 2. WHY — and it is structural, not a tuning accident

`VtResidency::apply_wants` seats a miss **the frame it is offered**, out of the
same pool. There is no per-frame admission throttle anywhere in this loop
(`VT_ADMITS_PER_FRAME_CEILING` is a *gate ceiling*, not a governor; P28.3 §8
re-measured the loader and left it alone), and there is **no latency between
admitted and sampleable** — `VtTextures::sync` applies and stages in one
synchronous call from mmap slices.

So *having asked earlier* buys nothing anywhere in this loop, while every want
spent on where the camera **will be** is a slot not spent on where it **is**.

That is exactly P28.4's own refutation, one want class up. The batch found the
**floor** half — *"the analytic floor cannot be prefetched: its fallback is
`max(0, demand − pool)`, two numbers a prediction changes neither of"*, measured
as 30 812 fallbacks byte-identical in both arms on a starved pool — and stopped
one class short of the refinement half. The lane is the win; the lead is a cost.

**The lane, to be clear, is a real and measured win.** A speculative want set at
the refinement's cap, ranked below every proved class, takes 131 blur frames to
105 with no lead at all. What P28.4 built is right; the number it shipped on top
of it is not.

## 3. THE RULING

1. **Ship `horizon_ticks = 0`.** The lane at the committed pose.
2. **Keep the dead-reckoner, the sweep and the clamp**, all of them, behind the
   knob. `PredictSettings::horizon_ticks` is a live setting; a non-zero value
   extrapolates exactly as P28.4 built it, `PREDICT_MAX_TURN` still binds, and
   `inf_math::dead_reckon` is untouched by this batch.
3. **Keep the ROADMAP's number by name**, as
   `inf_render::ROADMAP_PREDICT_HORIZON_TICKS = 18`. It is the *lead* arm of the
   A/B that produced this ruling, so deleting it would delete the falsification.
4. **The tripwires re-open the ruling by test rather than by memory**:
   * `a_lead_time_costs_this_fixture_what_the_lane_earns_it` runs the shipped
     zero against `ROADMAP_PREDICT_HORIZON_TICKS` and fails the day the lead
     wins on the aggregate **or** on the arrival window;
   * `every_horizon_in_the_roadmaps_band_beats_the_predictor_being_off` keeps
     the half of P28.4's claim that survived — every member of the ROADMAP's
     band beats OFF — and now also asserts the shipped horizon beats **every
     row of the band**, not merely both of its ends;
   * `a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so` is unchanged
     and is the same ruling for the class above.
5. **What would reverse it**: a per-frame admission throttle, or a loader with
   real latency between *admitted* and *sampleable* (an async upload path). On
   the day either lands, the arms above go red and the default goes back to
   `ROADMAP_PREDICT_HORIZON_TICKS`.

## 4. THE TRUTH ORACLE — accuracy, which no arm had measured

The P28.4 audit's verdict on `the_prediction_replays_from_the_recorded_history_
alone` is that it is a genuine oracle of **conformance**: a second longhand
implementation of the same specification, so its worst error of 4.04 × 10⁻¹⁶ is
the distance between two spellings of one formula (`arc * (h / span)` against
`(arc * h) / span`) and says nothing about where the camera actually went. The
audit routed the missing half here: *"a truth oracle would compare the
prediction against `whip_view(tick + h)`."*

Built, as `the_prediction_is_measured_against_where_the_camera_actually_went`.
It needs no residency at all — a prediction is a pose, and the fixture already
has a closed form for the camera at any tick. Worst angular error against the
world, over the whole path:

| lead (ticks) | 0 | 3 | 6 | 12 | 18 | 24 | 36 |
|---|---|---|---|---|---|---|---|
| worst error (rad) | **2.1e-8** | 0.0168 | 0.0461 | 0.1424 | 0.2890 | 0.4859 | 0.9556 |
| worst error (°) | ~0 | 0.96 | 2.64 | 8.16 | 16.56 | 27.84 | 54.75 |
| worst over the constant-rate hold | — | 0.0067 | 0.0134 | 0.0268 | 0.0402 | 0.0536 | 0.0804 |
| worst over the acceleration ramps | — | 0.0168 | 0.0461 | 0.1424 | 0.2890 | 0.4859 | 0.9556 |
| worst eye error (m) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Four things it establishes that nothing else did:

* **`h = 0` is the committed pose against the world**, not merely against a
  rearrangement of the same algebra — 2.1 × 10⁻⁸ rad, which is
  `psin64(0)`/`pcos64(0)` residue and eleven orders under a tile boundary. The
  zero-lead control is the control it claims to be.
* **The error grows monotonically with the lead**, every step of the ladder. The
  reckoner is not broken; it is being asked a question whose answer is further
  away. At the ROADMAP's own 18 ticks it is **16.6° wrong** in the worst case,
  which is a want set for a different part of the ring.
* **It is right about what it assumes and wrong about what it does not.** Over
  the constant-rate hold — where "the rate holds" is *true* — 18 ticks costs
  0.040 rad; over the acceleration ramps, **0.289**, seven times worse. A
  dead-reckoner that scored the same in both would not be dead-reckoning, and
  the arm asserts the ratio.
* **The linear half is exact.** This path's drift is linear and a secant
  extrapolates linear motion with no error at all, at every horizon — so every
  number above is *angular*, and the cost of a lead on this fixture is entirely
  the cost of guessing a turn.

## 5. THE LAW THIS IS THE THIRD INSTANCE OF

P20 and P25 both recorded it: **an unmeasured prescription can be backwards.**
The 200–500 ms band was written into the ROADMAP before there was a loop to
measure it against, and it was wrong for *this* loop — not by a little, and not
because the predictor is bad, but because the band presumes a latency this
streamer does not have.

The honest ledger says so rather than shipping the clause and calling it done. A
phase may not close with its headline knob measured backwards and shipped
anyway; the P28.4 audit wrote that sentence, and this is the batch that had to
answer it.
