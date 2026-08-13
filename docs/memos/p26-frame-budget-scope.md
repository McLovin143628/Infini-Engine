# P26.5 — what the streaming loop's frame budget measures, and where a wall clock may be read

**Status:** ruled 2026-08-13, after `macos-latest` went red on the Phase 26 close
(`7e0ced1`). The house rule: a gate that changes what it claims gets a memo, not
a quieter assertion.

---

## The failure

CI on `7e0ced1`: `windows-latest` green, `ubuntu-latest` green,
`macos-latest` red on exactly one arm —

```
inf-player::phase26_gate::the_streaming_loop_stays_inside_its_budgets
  panicked at runtime/inf-player/tests/phase26_gate.rs:2004:
  a streamed frame cost 49.55 ms, over the 33 ms frame budget
  (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)

  phase26 budgets: level build 0.79 ms (load budget 5000 ms)
  phase26 budgets: 49.55 ms/frame over 12 streamed frames (32 admits, 8 deferred);
                   frame budget 33 ms, streaming ratchet 8 ms
```

Immediately before it, on the same runner, in the same binary,
`the_scripted_paths_residency_trace_is_bit_exact_twice` **passed** — including
its `feedback_misses == READBACK_LATENCY_FRAMES + 1` assertion. So on that
machine the streamer is deterministic, the arrival pattern is pinned, and the
world the phase is about is intact. What failed is a clock.

## Three readings, and what the code says about each

**1. Is it a regression?** No, and this is measured rather than assumed. The
macOS run reports **32 admits and 8 deferred**; this machine reports **32 admits
and 8 deferred** (table below). The residency trace is bit-exact on the runner,
its `feedback_misses` lands exactly on `READBACK_LATENCY_FRAMES + 1`, and the
LOAD-class half of the very same arm measured **0.79 ms** there — comfortably
inside 5 000 ms. Nothing in the streaming loop got slower between the green
Windows leg and the red macOS one; they are different machines.

**2. Is it a real product gap the gate caught — an admit-per-frame cap tuned on
fast adapters?** No, and the code is specific about why. There is no per-frame
admit cap to be mis-tuned: `VtResidency::apply_wants` admits until the slots run
out and reports the remainder as `deferred`, so the deferral in this fixture is
the pool being six times too small, not a work-spreading policy declining to do
more. Sizing that cap by `RenderTier` was considered and rejected on a stronger
ground than tuning: `RenderTier` is **detected from the adapter**
(`inf_render::caps::detect_tier`), so a tier-riding admission rule would make the
residency trace a function of the machine that drew it — which is exactly the
property arms (a) and (b) exist to deny. And it would not have fixed this: the 32
admits are **18 on the cold frame and 14 spread over the other eleven** (measured
below), so the frames that were "over budget" were admitting nothing at all —
frames 2–6, 8, 9 and 11 admit **zero** pages and still cost whatever a frame
costs on that adapter. A cap on a number that is already zero cannot buy back
49 ms.

**3. Is the arm mis-scoped?** **Yes.** This is the finding.

## What the number was actually measuring

`StreamRun::frame_ms` was the mean of

```rust
renderer.render(gpu, &scene, &path_view(step), &target.view, (320, 180));
let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
```

— a whole headless frame through the whole pass stack, **plus a blocking pump to
device idle**, on whatever adapter is present. That is not the streaming loop's
cost; it is the streaming loop's cost buried inside a GPU frame's cost. P26.5's
own audit block says as much and blessed it: *"honestly named a FRAME-class
number that includes the render"*. The name was honest. The **consequence** was
missed: a number that includes a GPU frame is a hardware claim, and
`inf_player::budget`'s own doctrine is that §8 numbers *"are not hardware
claims: they are unbounded-growth tripwires"*.

Two smaller errors sat inside it, and both are worth writing down because they
are the kind that survive review:

* **The runner factor was borrowed from the wrong class.** The constant was sized
  as *"~15× the measurement, ≈4× after the ~4× a shared CI runner costs"*. That
  4× is `LOAD_BUDGET_MS`'s number for **CPU** work on a shared runner. Applied to
  a measurement that includes GPU execution on a paravirtualized adapter it is
  wrong by more than an order of magnitude: the factor measured on that runner is
  **49.55 / 0.53 = 93×**.
* **The mean folded a cold frame into eleven ordinary ones.** Frame 0 admits the
  whole analytic floor into an empty pool — it is the largest frame of the run by
  construction — and the arm ran no warm-up, unlike every other timing harness in
  the tree (`frame_budget.rs` warms 10 frames, `phase18_gate` (e) warms 8,
  `vgeom_streaming` measures its cold frame separately and prints it). Over
  twelve frames a cold frame is a twelfth of the answer.

## The precedent this arm was the only exception to

Every other wall-clock arm in the tree already declines to assert on a
non-representative adapter, and one of them names this exact runner:

| arm | condition |
| --- | --- |
| `inf-render/tests/frame_budget.rs` (5 sites, since P15.1) | `device_type == Cpu \|\| name contains "paravirtual"/"virtualbox"/"vmware"` → print, return |
| `inf-render/tests/vgeom_streaming.rs::streaming_overhead_is_bounded` | same, and its cold frame is printed rather than asserted |
| `inf-player/tests/phase18_gate.rs::the_composed_frame_stays_inside_the_frame_budget` | same |
| `inf-photo-gpu/tests/mvs_gate.rs::parity_is_strict` | discrete-and-not-virtualized, with an env override |

`frame_budget.rs` has carried the comment *"virtualized GPUs (the CI macOS runner
reports 'Apple Paravirtual device') have non-representative, run-to-run-noisy
timing"* since Phase 15. The P26.5 arm was written without it. That is the whole
defect: a new arm did not inherit an old law.

## The ruling

**A budget asserted everywhere must be denominated in a unit the machine cannot
inflate.** The arm now makes three claims in three classes:

* **LOAD** — the level build against `LOAD_BUDGET_MS`. Everywhere, unchanged.
* **WORLD** — what one frame's streaming loop *did*, in two numbers, against two
  new §8 constants. Everywhere, unconditionally.
  * `VT_ADMITS_PER_FRAME_CEILING` (16, measured 6 on a steady frame, 18 on the
    cold one): the pages a **steady** frame admits. One admit is one
    `queue.write_texture` of one page — `VtPools::apply` writes exactly the
    transaction's admits and nothing else — so this bounds the upload work a
    frame asks the queue for, in pages.
  * `VT_WANTS_PER_FRAME_CEILING` (48, measured 36): the size of the **scan**
    that asked. Needed because admits are clamped by the pool, so the regression
    `budget.rs` names — *"a want scan that walks the whole pyramid"* — is nearly
    invisible in them (see the mutation matrix below).

  Both are exact integers and arm (a) proves the sequence they come from is a
  pure function of committed input, so they are bounds and not races.
* **CLOCK** — the steady-state mean against `VT_STREAM_STEP_BUDGET_MS` and
  `FRAME_BUDGET_MS`, on an adapter whose timing represents a frame. Printed on
  every adapter, always, with the cold frame and a **VT-free control** beside it.

The cold frame is separated from the steady state in *both* the clock and the
page claim, and for the same reason in each. `vgeom_streaming` had already ruled
on the clock half — *"a regression that moved work from the cold frame into the
steady state would otherwise look like an improvement"* — and the page half turns
out to need it more sharply: a ceiling high enough to clear the cold frame's
whole-floor admission would be satisfied by a loop that re-admitted that same
full pool on **every** frame, which is precisely the thrash the ceiling exists to
catch.

`VT_STREAM_STEP_BUDGET_MS` keeps its value of 8.0. Rescoping what a number covers
is not a licence to move it, and §8 numbers only fall.

## The measurements

On this developer machine (Windows 11, **NVIDIA GeForce RTX 4070 Ti**,
`DiscreteGpu`, dev profile with optimizations), over the gate's scripted path —
twelve 320×180 frames against a 2 MiB pool holding a sixth of the level:

| | |
| --- | --- |
| level build (LOAD class) | 0.10 ms |
| cold frame (frame 0, the floor into an empty pool) | **6.67 ms** |
| steady mean (frames 1–11) | **0.54 ms** |
| mean over all 12 — *the number the old arm asserted* | **1.05 ms** |
| the same path, same cubes, no virtual texturing at all | 0.34 ms |
| **the streaming loop's share of a steady frame** | **+0.20 ms** |
| admits / deferrals | **32 / 8** |
| admits per frame | `0:18 1:6 2:0 3:0 4:0 5:0 6:0 7:4 8:0 9:0 10:4 11:0` |
| wants per frame | `0:18 1:24 2:24 3:30 4:30 5:30 6:30 7:30 8:30 9:30 10:36 11:36` |

Four things fall out of that table, and each of them is a claim the old arm could
not have made:

1. **The world's totals are the same on both runners.** The macOS failure printed
   *"32 admits, 8 deferred"*; this machine prints **32 admits, 8 deferred**. The
   one number that differs between a green leg and a red one is the wall clock.
   That equality is **measured**, one side from the CI log and one side here.
   What is *not* measured is the per-frame **distribution** on macOS — the CI log
   printed only the totals, and 18+6+4+4 and 18+10+4 both sum to 32. The two new
   ceilings bound per-frame numbers, so they rest on the distribution agreeing;
   the argument that it does is that the want set is CPU-derived and the mask's
   arrival is pinned (`feedback_misses` matched), and the margins are sized with
   that residual risk in mind rather than fitted to the measurement. The next
   macOS run prints the distribution and closes the gap.
2. **More than half of the number that went red was the cold frame.** 6.67 ms
   over twelve frames is 0.56 ms of the 1.05 ms mean.
3. **The streaming loop is 0.20 ms of a 0.54 ms steady frame** — the renderer's
   pass stack is the other 0.34 ms. So even on this machine, ~60 % of what the
   arm called "the streaming sync" was never the streaming sync.
4. **One discrete card drifts about 2× run to run.** This constant was minted at
   0.53 ms/frame; the same code on the same machine measures 1.05 ms/frame
   today. A wall clock here cannot be tight even where it is honest.

Applying (2) and (3) to the runner: if macOS's cold frame is a similar share of
its mean, its steady frame is ~23 ms of a 33 ms budget and most of that is its
pass stack, not paging. That is a prediction, not a measurement — the rescoped
arm prints all four numbers on every adapter, so the next `macos-latest` run
settles it.

## The mutation matrix

A rescoped gate has to be shown to still fail. Four mutations, each applied to
the engine (never to the test), with byte backups and restores:

| mutation | what it models | steady admits | peak wants | deferrals | verdict |
| --- | --- | --- | --- | --- | --- |
| *(none — baseline)* | | 6 | 36 | 8 | green |
| `VT_FLOOR_MAX_TILES` 16 → 64 | the floor claims 4× the tiles | 6 | 36 | 8 | **green — inert** |
| `lru_victim` → `max_by_key` | the evictor picks the newest page | 6 | 36 | 8 | **green — inert** |
| `justified_mip` two levels finer | 4× the tiles per surface — *"a want scan that walks the whole pyramid"* | 10 | **66** | 226 | **RED** on the wants ceiling |
| `acquire_slot` ignores `protected` | the transaction stops protecting what it just admitted | 8 | 36 | **0** | **RED** on the deferral anti-vacuity |

Three things this matrix says, and the first two are findings in their own right:

* **`VT_FLOOR_MAX_TILES` is not exercised by this fixture.** A 512² container is
  4×4 tiles at mip 0 — exactly 16 — so the cap never binds and quadrupling it
  changes nothing. The floor's own bound is therefore untested here; a fixture
  with a 2048²+ texture would test it. (Not fixed in this commit: it is a
  coverage gap in a different clause, and it belongs with P28.1's per-fragment
  feedback work, which is where the floor's sizing is next touched.)
* **The eviction policy is inert on this path.** Wanted pages are protected
  inside a transaction and the want set recurs frame to frame, so a *wrong*
  victim choice costs nothing measurable here. Also worth knowing before someone
  "optimises" the evictor and sees no gate move.
* **The two live signals catch different things.** The pool clamps admits (28
  slots), so a want-scan regression shows up as +4 admits and +218 deferrals —
  which is why the wants ceiling exists. Conversely a protection bug leaves the
  scan alone and empties the deferral count. Neither ceiling alone would have
  caught both.

A fifth was tried and rejected as a mutation: making `apply_wants` treat every
resident tile as a miss (re-upload every page every frame) panics inside
`VtResidency::unseat` — *"a non-root tile has a parent"* — before any gate sees
it. The engine's own invariant catches that one first, which is the right place
for it to be caught, but it means the mutation proves nothing about this arm.

## What is claimed, and what is not

* **Claimed:** on any adapter, in CI, the streaming loop's per-frame page work
  stays inside a ceiling, the residency trace is bit-exact, and the level build
  is inside the load budget.
* **Claimed on real hardware only:** that a streamed frame fits inside 33 ms.
  Since `windows-latest` and `ubuntu-latest` have no usable adapter and skip the
  arm outright, and `macos-latest` is the paravirtual one, this means the clock
  half is a **developer-machine gate** in practice — the same standing
  `frame_budget.rs` has had since P15.1, and the same standing the golden PNGs'
  visual claim has. The ROADMAP's Phase 26 ledger says so now.
* **Not claimed, and never was:** any frame-rate target. §8's 120 fps-class
  figure stays human-verified on real hardware, as `budget.rs` has said since
  P16.6.

## Carried

* **The macOS numbers in this memo are one CI run's, and the prediction above is
  a prediction.** The rescoped arm prints the cold/steady/VT-free decomposition
  on every adapter, so the next `macos-latest` run answers the question this
  failure could not: how much of that 49.55 ms is the paravirtual pass stack and
  how much is paging. If the VT-free control is most of it, the ruling above is
  confirmed by measurement rather than by inference, and the number worth quoting
  in the ledger is the difference. Nothing in this memo was executed on macOS —
  it cannot be, from here.
* **The renderer discards `VtApplyReport`.** `VtPools::apply` already counts
  `pages` and `page_bytes`, and `EngineRenderer::vt_sync` throws the report away,
  so `VtPopIn` cannot distinguish "admitted one page" from "wrote that page four
  times". Today they cannot differ — uploads happen only for a transaction's
  admits — so admits-per-frame is a faithful upload count and the ceiling above
  is sound. A re-upload path that broke that equivalence would be invisible to
  everything but a wall clock. Folding the report into `VtPopIn` belongs with
  **P28.3's unified streamer**, which merges this ring with the shadow one and
  will want one instrument for both.
