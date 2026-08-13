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
//! A fifth arrived with **P26.5**, over a different gate scene and named so:
//! [`VT_STREAM_STEP_BUDGET_MS`], the per-frame cost of the virtual-texture
//! streaming loop, asserted by `runtime/inf-player/tests/phase26_gate.rs`. It is
//! here rather than in `inf-render` for the same reason every other number in
//! this file is: a ratchet belongs where the gate that reads it lives, and a
//! Ring-0 crate that could read a budget is one edit away from letting the
//! machine decide what a frame contains.
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
/// Measured at **0.53 ms/frame** on a developer machine over the gate's scripted
/// path — a 320×180 headless frame including the render, against a pool six
/// times too small so every frame admits and defers. The ratchet is ~15× that
/// (≈4× after the ~4× a shared CI runner costs) and under a quarter of
/// `inf_core::FRAME_BUDGET_MS`, which is the property worth asserting:
/// streaming must never become a visible share of a frame. A regression that
/// matters — a want scan that walks the whole pyramid, a page that re-uploads
/// every frame — moves this by an order of magnitude; a 20 % drift does not, and
/// is not what CI is for.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const VT_STREAM_STEP_BUDGET_MS: f64 = 8.0;

/// The message every budget assertion fails with — the ratchet rule, at the point
/// where somebody is most tempted to break it.
pub const RATCHET_NOTE: &str =
    "(the §8 budget only ratchets DOWN — investigate the regression, do not raise it)";
