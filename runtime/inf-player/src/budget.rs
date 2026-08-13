//! **Streamed-scene budgets** (P16.6) — the §8 ratchet, extended to a world that
//! pages.
//!
//! §8's existing budgets each cover one still frame of one fixed thing:
//! `FRAME_BUDGET_MS` (a reference render), `SIM_STEP_BUDGET_MS` (a fixed world of
//! ~275 entities), [`LOAD_BUDGET_MS`] (a one-shot pack-load-to-first-world). None
//! of them can see the cost this phase introduced: a step that may *page terrain
//! tiles and spawn a partition cell before it does any of its own work*, and a
//! residency that grows with how far the player has walked.
//!
//! So this module adds four **P16.6** constants, asserted headless over the
//! **composed Phase 16 gate scene** (`samples/phase16-world`: a wizard-imported streamed
//! terrain, a partitioned world on top of it, a second inline terrain, and a
//! scripted walk) by `runtime/inf-player/tests/phase16_gate.rs`:
//!
//! * [`STREAMED_STEP_BUDGET_MS`] — mean fixed-step time while both streamers are
//!   live;
//! * [`TERRAIN_RESIDENT_BYTES_CEILING`] — peak terrain page bytes;
//! * [`CELL_RESIDENT_BYTES_CEILING`] / [`CELL_RESIDENT_CEILING`] — peak cell blob
//!   bytes and peak active cells.
//!
//! Three more arrived with **P26.5**, over a different gate scene and named so —
//! [`VT_STREAM_STEP_BUDGET_MS`], [`VT_ADMITS_PER_FRAME_CEILING`] and
//! [`VT_WANTS_PER_FRAME_CEILING`]: what one frame of the virtual-texture
//! streaming loop is allowed to cost, to upload, and to ask for, asserted by
//! `runtime/inf-player/tests/phase26_gate.rs`. They are here rather than in
//! `inf-render` for the same reason every other number in this file is: a
//! ratchet belongs where the gate that reads it lives, and a Ring-0 crate that
//! could read a budget is one edit away from letting the machine decide what a
//! frame contains.
//!
//! # A WORLD number and a CLOCK number are different kinds of budget (2026-08-13)
//!
//! The three above are one budget split three ways, and the split is the
//! sharpest statement in this file of what a §8 number can be. **The two
//! ceilings count pages and wants** — a page is a page on a discrete card, on a
//! CPU rasterizer and on a paravirtualized runner, and both sequences are a pure
//! function of committed input, so they are asserted **everywhere,
//! unconditionally**. **`VT_STREAM_STEP_BUDGET_MS` counts milliseconds**, which
//! is a fact about the machine as much as about the engine, so — like every
//! other wall-clock arm in this tree — it is asserted only where a millisecond
//! represents a frame, and reported everywhere else.
//!
//! That distinction was paid for: P26.5's budget arm asserted a GPU-inclusive
//! millisecond unconditionally, and `macos-latest` went red at **49.55 ms**
//! against a 33 ms frame budget with nothing regressed at all — the runner's
//! adapter is the "Apple Paravirtual device" `inf-render`'s `frame_budget.rs`
//! has skipped by name since P15.1. See
//! `docs/memos/p26-frame-budget-scope.md`. The general rule it leaves behind:
//! **prefer a budget in a unit the machine cannot inflate**, and when only a
//! clock will do, condition it the way the rest of the tree already does.
//!
//! Alongside them lives [`LOAD_BUDGET_MS`], the player's **load-class** ceiling.
//! It is deliberately a different *class* of number from everything above and from
//! `FRAME_BUDGET_MS`: a load is measured once, cold, so it may not be held against
//! a per-frame or per-step budget. Every arm that times a one-shot world build
//! shares that one constant — see its docs for why, and for what a wall clock on a
//! shared runner can and cannot be asked.
//!
//! # THE RATCHET RULE (§8), restated because it is the whole point
//!
//! **Every constant here may only ever DECREASE.** Lower one when the measured
//! floor drops; never raise one to make a red build green. A number that has to go
//! up is a regression report, not a settings change. The gate prints every
//! measured value on each run, so tightening is a matter of reading the line.
//!
//! # What these numbers are, and what they are not
//!
//! They are **tripwires**, and they are deliberately generous — a multiple of the
//! measured value on a developer machine — because they run on shared CI runners
//! of three operating systems under unknown load, where a tight bound produces
//! flakes rather than information. A regression that matters (a per-step
//! whole-world scan, a residency set that leaks, a page that reloads every frame)
//! moves these by an order of magnitude and trips them; a 20% drift does not, and
//! is not what CI is for.
//!
//! **On the "120 fps class" target, honestly.** Phase 16's goal states a
//! 120 fps-class frame budget: 8.3 ms for *everything* in a frame, on real
//! hardware, with a GPU. Nothing in CI can assert that — the render half needs a
//! GPU these runners may not have, and a millisecond on a loaded shared runner is
//! not a millisecond on a target machine. What CI *can* assert is that the
//! CPU-side streaming work stays a small, bounded fraction of that budget on any
//! machine, which is what [`STREAMED_STEP_BUDGET_MS`] does. **The frame-rate claim
//! itself stays human-verified on real hardware**, exactly as the golden PNGs'
//! visual claim does, and the ROADMAP's Phase 16 status block says so.
//!
//! **On what a byte ceiling can and cannot catch.** Bytes are machine-independent,
//! so these are honest bounds — but on a *gate-sized* scene they cannot
//! distinguish "the streamer stopped evicting" from correct behaviour, because the
//! whole asset is only a few megabytes and full residency would sit under any
//! sane ceiling. What they do catch is **unbounded** growth: a set that is never
//! freed, a page counted twice, a working set that duplicates per frame. The
//! *bounded cut* claim itself is asserted structurally in the gate (the residency
//! set is a quadtree cut, it churns, and every resident cell is inside the
//! activation radius), which is the assertion that actually has teeth.

/// Hard ceiling for a **one-shot load**, in milliseconds: opening a cooked pack
/// and building the world it describes, measured once. The player's own boot path
/// (`tests/startup_budget.rs`) and the Phase 19 town build (`tests/phase19_gate.rs`)
/// both assert against this one constant, because they are the same *class* of
/// measurement and a class deserves one number.
///
/// # Why a load may not be held against the frame budget
///
/// `inf_core::FRAME_BUDGET_MS` is 33 ms because that is what a *frame* gets —
/// a thing that must happen thirty times a second, forever. A load happens once.
/// Asserting that an entire furnished town builds in the time one frame gets is
/// not a growth check, it is a **hardware claim**, and §8 budgets are not hardware
/// claims: they are **unbounded-growth tripwires**, deliberately generous, run on
/// shared CI runners of three operating systems under unknown load. Those runners
/// are roughly **4× slower than developer hardware and noisy**, so any gate that
/// reads a wall clock at frame resolution ends up reporting the runner rather than
/// the engine. The Phase 19 town-load arm did exactly that — ~8 ms locally,
/// 34.77 ms on a `windows-latest` runner, red, with nothing regressed but the
/// machine — which is what moved it here.
///
/// # What 5 000 ms is
///
/// The P15.1 precedent, reused rather than re-invented: the startup tripwire has
/// shipped at 5 000 ms against 5.6 ms measured — three orders of headroom, on
/// purpose. It is ~150× the frame budget and ~600× the measured town build (~8 ms
/// on a developer machine). A load that crosses it is no longer linear in its
/// content — an O(n²) resolve, a per-instance re-walk, a cache that stopped hitting
/// — which is the class of regression CI can honestly catch. Drift of tens of
/// percent is invisible here **on purpose**; every arm prints its measured
/// milliseconds, and that printed line is where load-time drift is read.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const LOAD_BUDGET_MS: f64 = 5000.0;

/// Hard **mean** fixed-step budget, in milliseconds, for a step that also runs
/// cell streaming and terrain sim-residency at its top (the composed gate scene).
///
/// Measured at **≈0.18 ms/step** on a developer machine (Windows, dev profile with
/// optimizations), dominated by the two want-set scans over the world; the paging
/// itself amortizes to nearly nothing once the walk settles inside a cell. The
/// budget is ~20 × that so a loaded CI runner cannot flake it, and still half of
/// the 8.3 ms a 120 fps frame has for *everything* — which is the property worth
/// asserting: streaming must never become a visible share of the frame.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const STREAMED_STEP_BUDGET_MS: f64 = 4.0;

/// Hard ceiling on **terrain** page bytes resident at any point of the gate
/// flythrough (`TerrainStreamStats::bytes_resident`, summed over every streamed
/// terrain — the camera's render cut plus the pages the sim pinned).
///
/// The gate scene's terrain is 64 level-0 pages of 129² samples (≈66 KB each) over
/// 8.2 km, plus 20 coarse pages: ≈5.6 MB of tile data in total, of which the
/// measured peak resident is **≈5.65 MiB**. That the peak is close to the whole
/// asset is a property of a *gate-sized* world (the render radius is 2.5 tile
/// spans on an 8-tile world), not of the streamer — which is why 16 MiB is the
/// tripwire: it cannot be reached by any bounded set over this scene, so crossing
/// it means residency grew without bound.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const TERRAIN_RESIDENT_BYTES_CEILING: u64 = 16 * 1024 * 1024;

/// Hard ceiling on **cell blob** bytes resident (`CellStreamStats::bytes_resident`
/// — active cells plus the prefetch buffer) at any point of the gate flythrough.
///
/// Measured at **≈2.8 KiB** (the gate's cells hold one cube each). Cell payloads
/// are small enough that [`CELL_RESIDENT_CEILING`] is the load-bearing bound of
/// the two; this one exists to catch a prefetch buffer that is filled and never
/// drained, which grows without limit rather than to a plausible-looking number.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const CELL_RESIDENT_BYTES_CEILING: u64 = 256 * 1024;

/// Hard ceiling on **active** partition cells at any point of the gate flythrough.
///
/// The gate's 512 m activation radius on a 2048 m grid can touch at most the four
/// cells meeting at a corner; the measured peak is **4**. Eight is the tripwire.
///
/// Distinct from `cell_stream::ACTIVATION_SOFT_CEILING`, which is an advisory
/// runtime warning that is deliberately never enforced (activation is never
/// clamped — a missing cell changes the simulation). This is a *test* assertion
/// about a *known* scene: a value above it means the want set stopped being a
/// neighbourhood, which is a design regression, caught in CI rather than in a
/// shipped build's memory profile.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const CELL_RESIDENT_CEILING: usize = 8;

/// Hard **mean per-frame** budget, in milliseconds, for a frame that also runs
/// the virtual-texture streaming loop (P26.5).
///
/// The P26.4 remainder, discharged: *"the feedback's own budget is a page cap,
/// not a millisecond cap. `VT_FEEDBACK_MAX_TILES` and `VT_FEEDBACK_REQUEST_CAP`
/// bound the work; what the sync costs in LOAD-class milliseconds is not yet
/// ratcheted, and the `phase26_gate` budget arm is where that lands."*
///
/// It is a **FRAME**-class number, and the P20 law is why the distinction is
/// spelled out rather than assumed: the level *build* — registry, pool, floor —
/// happens once and is held against [`LOAD_BUDGET_MS`]; the sync happens thirty
/// times a second forever. `phase26_gate` asserts both, on the same fixture, in
/// the same run.
///
/// Measured at **0.53 ms/frame** on a developer machine (Windows, RTX 4070 Ti,
/// dev profile with optimizations) over the gate's scripted path — a 320×180
/// headless frame including the render, against a pool six times too small so
/// every frame admits and defers.
///
/// # What it covers, and where it is asserted (rescoped 2026-08-13)
///
/// It is a **whole frame** — `render` plus a pump to idle — and therefore
/// includes the renderer's entire pass stack and the GPU's own execution, not
/// just the streaming loop. That was true when it was minted and honestly
/// written down; what was missed is the consequence. A number that includes a
/// GPU frame is a **hardware claim**, and §8 numbers are not hardware claims:
/// the arm asserted it on every adapter and `macos-latest` went red at
/// **49.55 ms** with nothing regressed. The arithmetic that sized this constant
/// — *"~15× the measurement, ≈4× after the ~4× a shared CI runner costs"* — used
/// the CPU-class runner factor from [`LOAD_BUDGET_MS`]'s docs on a GPU-inclusive
/// measurement; the factor actually measured on that runner is **93×**.
///
/// So the assertion is now conditional, exactly as every other wall-clock arm in
/// this tree already is (`inf-render`'s `frame_budget.rs` since P15.1,
/// `vgeom_streaming`, `phase18_gate` (e), `mvs_gate`): reported on every
/// adapter, asserted on one whose timing represents a frame. The portable half
/// of the claim moved to [`VT_ADMITS_PER_FRAME_CEILING`], which is asserted
/// everywhere. `docs/memos/p26-frame-budget-scope.md` carries the ruling.
///
/// It is also measured against the **steady state** now — the mean of every
/// frame after the cold one — rather than a mean that folds frame 0's whole
/// floor admission into eleven ordinary frames. Measured on the same machine on
/// 2026-08-13: cold frame **6.67 ms**, steady mean **0.54 ms**, mean over all
/// twelve **1.05 ms** — so *more than half* of the number this constant used to
/// be compared against was frame 0. The cold frame is printed beside the steady
/// one, `vgeom_streaming::streaming_overhead_is_bounded`'s arrangement.
///
/// The arm also prints the same path rendered with **no virtual texturing at
/// all**: 0.34 ms steady, which makes the streaming loop's own share of a steady
/// frame **0.20 ms** and the renderer's pass stack the other 0.34. That
/// difference, not the frame, is what this constant is morally about — it is not
/// asserted, because a difference of two wall clocks is noisier than either.
///
/// The value is unchanged at 8.0 (a §8 constant may only fall, and rescoping
/// what a number covers is not a licence to move it): ~15× the steady
/// measurement, which is where it already sat, and ~40× the streaming loop's own
/// share. A regression that matters — a want scan that walks the whole pyramid,
/// a page that re-uploads every frame — moves it by an order of magnitude; a
/// 20 % drift does not, and is not what CI is for. Run-to-run drift on one
/// discrete card is itself about 2× (1.05 ms today against the 0.53 ms this
/// constant was minted on), which is a second reason no wall clock here can be
/// tight.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const VT_STREAM_STEP_BUDGET_MS: f64 = 8.0;

/// Hard ceiling on the pages **one steady frame** of the virtual-texture
/// streaming loop may admit (P26.5, 2026-08-13) — the machine-independent half
/// of the frame budget above.
///
/// # Why this is the assertion with teeth
///
/// One admit is one `queue.write_texture` of one page: `VtPools::apply` walks
/// `VtTransaction::admits` and writes exactly those, so this bounds the upload
/// work a frame asks the queue for — in pages, which cost the same number of
/// pages on every adapter in the world. And the sequence is a **pure function of
/// committed input** (`phase26_gate` arm (a) asserts the residency trace it comes
/// from is bit-exact across runs), so a ceiling on it is not a race.
///
/// That is what a millisecond could not be. The wall clock is skipped on
/// software and paravirtualized adapters, and on CI those are the only adapters
/// there are — Linux and Windows runners have no usable ICD and skip the arm
/// entirely, macOS runs it on an "Apple Paravirtual device". So without this
/// constant the gate's frame half would assert **nothing** in CI, which is the
/// state the P26.5 arm was quietly in the moment it was made conditional.
///
/// # The number, and why the cold frame is not in it
///
/// Measured over the gate's scripted path against a pool six times too small.
/// Frame 0 admits **the whole floor the pool can hold** — it is the pool's own
/// slot count by construction — and every frame after it admits a small tail as
/// the camera walks. A single ceiling over both would have to clear the cold
/// frame, and would therefore be satisfied by a loop that re-admitted that same
/// full pool on **every** frame: the thrash this exists to catch. So the cold
/// frame is excluded and printed, exactly as its milliseconds are, and this
/// bounds the steady state.
///
/// Measured on 2026-08-13 over that path: `0:18 1:6 2:0 3:0 4:0 5:0 6:0 7:4 8:0
/// 9:0 10:4 11:0` — **18 pages on the cold frame** and a **peak of 6** on a
/// steady one, 32 in all against 8 deferrals.
///
/// Sixteen is therefore a tripwire at ~2.7× the measurement **and deliberately
/// below the cold frame's own 18**, which is the property that makes it able to
/// see anything at all: a loop that re-admitted the floor on every frame would
/// sit at 18 and trip it, where any ceiling chosen to clear frame 0 would wave
/// it through.
///
/// # It is the coarse half of the pair, on purpose
///
/// This one catches **pool-wide re-admission**; [`VT_WANTS_PER_FRAME_CEILING`]
/// catches a **scan that grew**. The division is not a matter of taste — it was
/// measured. A frame's admits are clamped by the pool (28 slots here), so the
/// two mutations run against this arm move it barely: a `justified_mip` two
/// levels too fine takes the steady peak from 6 to **10**, and an
/// `acquire_slot` that ignores its protected set takes it to **8**. Both are
/// caught, and neither is caught *here* — the first by the wants ceiling
/// (36 → 66) and the second by the arm's own anti-vacuity assertion (deferrals
/// 8 → 0). Tightening this constant toward its measurement to make it the
/// sensitive one would buy a detection the other half already has, at the price
/// of a bound that is 1.3× a number produced by a `log2` whose last bit is not
/// promised to be the same on every backend. See the arm's printed
/// `per frame:` line, which is where drift is read.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const VT_ADMITS_PER_FRAME_CEILING: u64 = 16;

/// Hard ceiling on the **wants one frame offers** — floor plus refinement,
/// before the residency decides what fits (P26.5, 2026-08-13).
///
/// # Why admits alone are not enough, measured
///
/// [`VT_ADMITS_PER_FRAME_CEILING`] bounds what a frame *uploads*, and a pool
/// bounds that anyway: a fixture with 28 slots cannot admit two hundred pages
/// however badly it asks for them. So the regression this file names by name —
/// *"a want scan that walks the whole pyramid"* — is nearly invisible in admits.
/// Mutation-measured on the gate scene: `justified_mip` shifted two levels finer
/// (every surface asking for four times the tiles) moves the peak admits from
/// **6 to 10** — inside any ceiling a 6 could justify — while the peak wants go
/// from **36 to 66** and the deferrals from **8 to 226**. One frame's want set
/// is where that regression lives, so that is where it is caught: the arm goes
/// red on this constant and on nothing else.
///
/// # The number
///
/// The floor is bounded **by construction** — the coarsest levels are
/// camera-free and the camera-driven level adds at most `VT_FLOOR_MAX_TILES` per
/// *visible* surface — while the refinement half is deliberately **uncapped in
/// wants** (the budget, not the scan, decides what is served), which is exactly
/// why the composed number is worth watching rather than deriving.
///
/// Measured peak on the path: **36** (per frame `0:18 1:24 2:24 3:30 … 10:36
/// 11:36` — it rises as the walk brings surfaces in, which is the shape a
/// bounded scan should have). Forty-eight is 1.33× that, and 27 % under the
/// mutation's 66. A tighter margin than the millisecond budgets carry, and
/// legitimately so: this is an exact, deterministic integer, not a noisy
/// measurement — it was byte-identical across every run of the day, and the red
/// macOS run reported the same 32 admits / 8 deferrals as this machine, which is
/// the evidence that the want sets agree across backends.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const VT_WANTS_PER_FRAME_CEILING: u64 = 48;

/// The message every budget assertion fails with — the ratchet rule, at the point
/// where somebody is most tempted to break it.
pub const RATCHET_NOTE: &str =
    "(the §8 budget only ratchets DOWN — investigate the regression, do not raise it)";
