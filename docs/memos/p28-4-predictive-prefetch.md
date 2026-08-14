# P28.4 — predictive prefetch: the pure function, the lane, and the class it turned out could not be prefetched

**Status:** decided 2026-08-14, during P28.4. The rulings this batch took, the
measurements that took them, the two it reversed on itself, and the debt it did
not close, by name.

The direction memo's §3 deviation replaced a neural motion predictor with
deterministic analytic dead reckoning, on the grounds that *"a learned predictor
is untestable under house gates (training nondeterminism, no falsifiable
bound)"*. That trade only pays if the analytic one is genuinely a pure function
and its win is genuinely measurable, so this memo is mostly about those two
things — and about the fact that measuring the second one refuted the batch's
first design.

---

## 1. THE PREDICTOR IS A PURE FUNCTION — what is guaranteed, and by what

`inf_math::dead_reckon(&CameraHistory, horizon_ticks) -> Option<Prediction>`.

**Structural, in the module:**

* no clock, no frame counter, no `Instant`, no adapter, no interior mutability;
* every transcendental is `inf_math::portable`'s (`pacos64`/`psin64`/`pcos64`),
  pinned by a source arm that reads **the non-test half only** — the test
  carries every banned spelling as a string literal, and a scan over the whole
  file would be satisfied by its own fixture;
* the horizon is in **ticks**, never milliseconds. A millisecond is wall clock;
  a tick is committed. `inf_math::horizon_ticks(ms, hz)` is the host's one door
  and `hz` is the host's *fixed* step, which is committed too.

**Not checkable from inside, so named instead.** The premise is that a
`CameraSample` is committed — that its pose is a function of the committed input
stream and nothing else. `CameraHistory::commit` enforces the mechanical half (a
tick that only ever advances, so a host cannot append twice for one step) and
**counts** the refusals. The other half is a property of the host:

| host | camera | commits? |
|---|---|---|
| shipped windowed player | `RuntimeSim::camera_focus`, a fold of actor positions | **yes**, at `RuntimeSim::steps()` |
| a gate's scripted path | a pure function of the step index | **yes** |
| the editor viewport | a flycam driven by OS input at render rate | **no**, and it must not |

The editor's exclusion is the honest bound, not an oversight: its camera is not
committed input in any sense, so it never commits, the history stays empty,
`dead_reckon` answers `None`, and the editor streams byte-for-byte as it did
before this batch. **An empty history is the real enable flag**, and it fails
safe — a consumer handed `None` emits no speculative want and lands on exactly
the analytic floor.

There is one honest consequence worth recording: this tree has **no committed
input stream** as a type. `RuntimeInput` is a per-tick set that only the previous
tick's copy survives, the PIE wire has no input message, and every trace in the
repo steps with `RuntimeInput::default()`. So "a pure function of committed input
history" is realised here as *a pure function of the committed camera-pose
history*, which is a pure function of the committed input wherever a host samples
at the fixed step. That is the strongest statement the tree currently supports,
and pretending otherwise would be the inference-dressed-as-measurement failure
P22 named.

---

## 2. THE WINDOW: six samples, and why a secant rather than a difference

`PREDICT_HISTORY = 6` — 100 ms at 60 Hz. The velocity estimate is a **secant
between the two ends of the window**, not a difference of the last two samples.

The trade is stated rather than tuned: a pose is quantized (the direction
arrives as an f32 `Vec3`), so a two-sample estimate divides that noise by one
tick and a six-sample estimate divides it by five — and lags a *change* in rate
by proportionally more. Six ticks is long enough that the secant is not one
step's rounding and short enough that a whip's acceleration phase is not averaged
away. The gate's ramp phases are where that trade is paid, and the horizon sweep
in §3 is measured over a path that contains them.

`PREDICT_MAX_TURN = π` is a clamp and not a knob: past half a turn the
extrapolation has stopped being one. It is reported (`Prediction::clamped`) so a
host can see it bind rather than discovering a want set that describes nowhere.

---

## 3. THE HORIZON: the sweep, and what it does and does not say

The ROADMAP names 200–500 ms. Measured on `whip_pan`'s 260-tick path, 800-page
pool, twelve 2 048² surfaces in three clusters — **blur frames**, against **131**
with the predictor off:

| horizon | ticks | blur frames | blur tiles |
|---|---|---|---|
| 200 ms | 12 | 115 | 18 976 |
| 250 ms | 15 | 115 | 18 976 |
| **300 ms** | **18** | **115** | **18 976** |
| 350 ms | 21 | 117 | 18 976 |
| 400 ms | 24 | **112** | 18 896 |
| 450 ms | 27 | 121 | 19 072 |
| 500 ms | 30 | 117 | 18 976 |

**Every member of the band beats OFF**, which is the claim the gate asserts — a
band in which some horizons lose is a band the ROADMAP should not have named.
The shipped default is **18 ticks / 300 ms**, and the honest reading of the
table is that the band is *flat*: the spread between best and worst is 9 frames
of 131, and 400 ms wins this fixture by 3 frames over the middle. The gate
therefore asserts what the table supports — the shipped horizon is no worse than
either **end** of the band — and not "the shipped horizon is optimal", which one
fixture cannot establish. Picking 400 ms off a single scene's 3-frame margin
would be the "average hides a station" failure P25 named.

---

## 4. THE REFUTATION: the floor cannot be prefetched, and the measurement that says so

The batch's first design speculated in the **floor's** language: the analytic
floor's own footprint rule, at the predicted camera, at `VT_FLOOR_MAX_TILES`. It
is wrong, and the gate found it rather than review:

`VtResidency::apply_wants` seats a miss **the frame it is offered**, out of the
same pool, and there is no per-frame admission throttle anywhere in the loop
(`VT_ADMITS_PER_FRAME_CEILING` is a *gate ceiling*, not a governor; the loader
stages a page synchronously from an mmap slice — P28.3 §8 re-measured that and
left it alone). So the floor's fallback count is `max(0, demand − pool)`:

* **under** the pool, nothing misses at all;
* **over** it, the shortfall is the arithmetic difference between two numbers;
* in neither regime does *having asked earlier* change either number.

Measured, over the same 360° path with a deliberately starved 96-page pool:
**30 812 floor fallbacks over all 260 frames, byte-identical in both arms**,
while the predictor offered 137 584 speculative wants. Kept as an arm
(`a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so`) rather than as
this paragraph, so the day an admission throttle appears the ruling is
re-opened by a red test instead of by memory.

**What lags the camera is the refinement.** It is marked off a depth buffer, so
it can only ever ask for surfaces that are *already visible*, and it arrives
`READBACK_LATENCY_FRAMES` after that. That gap is what `VtPopIn`'s own header
calls "what pop-in **is**", it is the only thing in this subsystem a prediction
can close, and closing it requires speaking the refinement's language exactly:
`VT_PREDICT_MAX_TILES = VT_FEEDBACK_MAX_TILES`.

**And the cap has to be exact, because a tile is an address.** `(texture, mip,
x, y)`. A different cap settles on a different *mip* and shares not one tile
with the class it claims to prefetch for. The intuitive middle ground — "a guess
should claim less than a proof" — is precisely the choice that fills the pool
with pages nobody will ask for. Swept over five square pyramids in
`a_prediction_at_a_finer_cap_names_addresses_the_floor_will_never_ask_for`: where
the caps bind at different levels the two address sets are **disjoint**.

A second, smaller correction on the way: the first reading of `analytic_floor`
concluded that its camera-driven half is subsumed by the camera-free
`want_floor` on any square pyramid, which would have made a floor prediction
vacuous for a different reason. It is **false**, and measuring it is what showed
so: `full_pyramid` runs the chain down to one *texel*, not one tile, so a 512²
texture has ten mips and the three pinned coarsest are 1 tile each, while the
camera-driven half lands at mip 0 or 1. The reasoning that produced the wrong
model never checked the shape of a real descriptor.

---

## 5. THE SHADOW LANE: what a predicted camera can say about shadow pages

P28.3 refused shadow pages as `Coupling` members and routed the refusal here by
name:

> which pages a caster reaches is decided by a per-page frustum test that runs
> over the pages that are **already resident**, after the marking mask has been
> read — so producing that membership at the sync point means deriving *next*
> frame's page set from *last* frame's casters, which is a **prediction**, and a
> prediction enters at `LANE_PREDICT`.

`inf_render::speculative_shadow_wants` is that derivation, done deliberately:
the pages last frame's depth buffer **proved** were needed, moved to where the
predicted camera puts the clipmap window. The move is a translation in page-index
space between two `ClipmapLayout::clip_origins`, and nothing else — no matrix, no
basis, no second copy of the snapping rule (the predicted layout comes from
`vsm_projections` itself, the shipped door).

**Its bound is measured rather than implied**, and it is a large one: a camera
that only **rotates** does not scroll a camera-centred clipmap, so a *pure*
whip-pan produces the empty set here. `a_pure_rotation_produces_no_speculative_
shadow_want` asserts exactly that, with a two-page dolly along the light's own
`right` as the control. What a rotating camera changes is which *receivers* are
visible, and predicting that means predicting a depth buffer nobody has drawn.
`VsmStreamStats::speculative_wants` is the counter that tells "the predictor is
off", "the history is empty" and "the camera only turned" apart, all three of
which produce an identical residency.

`VSM_PRIORITY_SPECULATIVE` therefore gets the producer it has been waiting two
phases for, and moves from `LANE_FEEDBACK` to `LANE_PREDICT`. P28.3's argument
for the feedback lane (a shadow page has no refinement class) is still true of
that consumer read alone and is the wrong reading now the lane has a producer:
the invariant this batch has to assert is **one statement over all three
consumers**, and a speculation sitting in the feedback lane makes it mean
something different here than next door. Nothing observable moves — no producer
in that crate has ever emitted `LANE_FEEDBACK`.

---

## 6. STRICTLY LOWER CAME OUT AN EQUALITY

The ROADMAP asks for *"speculative wants enter at strictly lower priority than
the analytic floor and feedback"*, and the world form of it is *residency ⊇
floor ∪ feedback at full speculative pressure*. Measured over the whip-pan's 260
ticks, the proved resident set is **identical** in the two arms at every tick —
not merely a superset.

That is the result rather than a weak arm. Every proved want is offered a slot
before any speculative one, so a pool that can seat it seats it in both arms and
a pool that cannot seats it in neither: a strictly-lower lane is *exactly*
neutral to the classes above it. Under 19 152 deferrals and 137 584 speculative
wants, with a proved set peaking at 708 tiles, **zero** resident floor tiles were
evicted in either arm.

One claim is deliberately **not** made, and the reason is recorded in the arm:
the per-tick count of sharp tiles is not dominated. A tile a *visible* surface
justifies but no class has asked for *this* frame is exactly what a speculative
miss should take first (the P28.3 audit's `reserved` fix), so speculation can
cost the visible set a tile on individual ticks while never touching one any
class has actually asked for. The aggregate is asserted; the per-tick series is
not, and saying so is cheaper than a bound nobody could defend.

---

## 7. THE OWED ITEMS, item by item

**Landed.**

* **`VSM_PRIORITY_SPECULATIVE`'s producer** — §5. Discharged, not retired.
* **Shadow-page membership from a predicted camera cone** — §5, in the lane
  P28.3 named. What is *not* landed is the per-**group** membership (a cluster
  page's shadow pages as `Coupling` members): this producer works in page-index
  space over a light's whole proved set, and a coupling needs to know which
  group each page belongs to, which is still only knowable from the raster's
  per-page frustum verdict. **P28.5**, with P28.3's reason unchanged.
* **The `count_fallbacks` third class** — not owed, found. Two arms and an
  `else` were exhaustive until this batch added a lane; with three, every
  speculative want counted as a floor want, and turning the predictor on would
  have *raised* the two numbers the A/B arm reads. The gate would have measured
  its own instrument.

**Measured and refused.**

* **The clipmap scroll (127 pages against 4 096)** → **P28.5**, and the refusal
  is structural rather than about effort. The residency half landed in P27.3 and
  `VsmResidency::set_clip_origins`' own doc states the rest: *"The pages keep
  their slots. A page whose world cell changed is stale **content**, not a stale
  allocation, and which of them is stale is a question about the raster's
  content stamps rather than about residency."* Every page index is always in
  range under a scroll, so there is no want a speculative lane could emit that
  would help — the work is in `vsm_raster`'s page cache, which this batch did
  not open. The 127-against-4 096 number is a *re-rasterization* saving and it
  belongs with the caster-pack restructure P28.3 already routed to P28.5.
* **Retraction's cost in frames** → **P28.5**, with the measurement that
  disposes of it. P28.3 routed it here as *"your gate's own fixture"*; it is
  not, and the whip-pan gate is what shows why. `retracted` counts cluster pages
  the **texture half refused** — a budget event — and the whip-pan fixture is
  deliberately a *latency* fixture (§4: in the budget regime both arms are
  byte-identical and the predictor is provably inert). At 800 pages the floor
  never falls back at all: `floor 0 over 0 frames`. A retraction-cost fixture is
  a contended-pool fixture with the vgeom coupling in it, which is
  `cluster_pages`' shape and not this one's.
* **The palette-union caster bound** → **P28.5**, with the P27.5 correction
  re-verified rather than repeated. It changes what a *mover* invalidates, which
  is the CPU caster cache's contract, and P27.5's correction stands: the 67 %/30 %
  figures are the margin's own reciprocal on a one-joint fixture, so what has to
  be measured is a real rig where joints far apart can make the union *larger*
  than the inflated bind sphere. Nothing in P28.4 touched a caster bound, and
  the predictor has no bearing on it: a mover's invalidation is not a camera
  question. Re-routing it here was the routing table's error, and it is
  corrected by name rather than silently.

---

## 8. WHAT THIS BATCH DID NOT DO

* **The meshlet streamer gets no speculative lane**, and cannot as things stand:
  `inf-vgeom` is deliberately not a `SlotPool` (P28.3 §1), its residency is a
  *prefix* over a byte auction, and a prefix has no rank to be strictly lower
  in. Asking the auction for a finer cut than the camera justifies would *raise*
  the floor rather than add a class beneath it. Stated, not attempted.
* **No `.wgsl` moved.** The predictor is CPU-side throughout; the only shader
  question it could raise (a predicted depth buffer) is the one §5 refuses.
* **No schema, no container version, no golden.** Goldens stay **54**.
* **The editor viewport gets no prefetch**, by construction (§1).
