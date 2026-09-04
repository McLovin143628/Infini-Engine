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

/// Hard **mean** budget, in milliseconds, for ONE sim→render **projection** —
/// `inf_player::render::project_scene_full` over a streamed world (Hardening
/// Wave E, 2026-08-14).
///
/// # A fourth class, because the seam had none
///
/// The three classes above cover the render (`inf_core::FRAME_BUDGET_MS`), the
/// fixed step ([`STREAMED_STEP_BUDGET_MS`]) and the one-shot build
/// ([`LOAD_BUDGET_MS`]). Between the second and the first sits the projection:
/// the function that turns an `EcsWorld` into the `RenderScene` the renderer is
/// then measured against. **Nothing in this tree measured it**, which is how
/// three per-frame deep copies of every resident terrain tile, voxel chunk and
/// fracture chunk stood against change stamps their own consumers honour — the
/// producers ignored the stamps, the consumers discarded the payloads, and the
/// entire cost landed in a function no budget could see.
///
/// # It is asserted EVERYWHERE, unlike the frame budget
///
/// A projection touches no GPU: it is CPU work over CPU data, and the whole
/// reason `inf-render`'s `frame_budget.rs` skips software and paravirtualized
/// adapters — that their milliseconds do not represent a frame — simply does not
/// apply. So `runtime/inf-player/tests/projection_budget.rs` asserts this on
/// every runner of every operating system, which makes it one of the few
/// wall-clock arms in the tree that is not conditional. The ~4× shared-runner
/// factor from [`LOAD_BUDGET_MS`]'s docs *is* a CPU factor, and it is carried in
/// the number below rather than in a skip.
///
/// # The number, and the ratchet it has already taken
///
/// Measured on a developer machine (Windows, dev profile with optimizations)
/// over the gate fixture — 36 level-0 tiles at 129², a 55-chunk / 19 623-vertex
/// voxel slab and 200 props, sixty projections into one scene:
///
/// * **6.370 ms** with the three producers rebuilding their payloads every
///   frame against stamps their own consumers honour. That is six and a third
///   milliseconds of a shipped player's frame, every frame, producing bytes the
///   renderer discards on a version match.
/// * **0.041 ms** once terrain tiles and voxel chunks are carried forward from
///   the previous frame's scene instead — **155×**, with byte-identical
///   payloads (the gate's arm (d) asserts exactly that).
///
/// This constant was minted at **48.0** — the *unrepaired* measurement × ~7.5 —
/// before the repair, on purpose: a budget minted after a fix cannot certify
/// that the fix is what moved the number. It ratchets here, in the same wave,
/// against the repaired one. Read the git log of this line, not just its value.
///
/// **1.5 ms** is ~37× the repaired measurement and ~9× after the ~4×
/// shared-runner factor from [`LOAD_BUDGET_MS`]'s docs — a wider margin than
/// [`STREAMED_STEP_BUDGET_MS`] carries, because a projection's cost is
/// dominated by allocation and a loaded runner's allocator is the noisiest
/// thing in this file. What matters more than the margin is that it sits **more
/// than four times below the 6.370 ms this fixture cost before the repair**: a
/// producer that goes back to rebuilding per frame trips it, which is the
/// regression it exists for and the property a number chosen to clear the old
/// cost could not have had.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const PROJECTION_BUDGET_MS: f64 = 1.5;

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

/// Hard **mean** fixed-step budget, in milliseconds, for a step over a **CITY**
/// (island wave I4b) — the §8 number the AAA-readiness certification said was
/// missing, and the one [`STREAMED_STEP_BUDGET_MS`] above cannot be.
///
/// # Why a second step budget
///
/// `STREAMED_STEP_BUDGET_MS` is asserted over the phase-16 gate scene, which is
/// **a walker on a heightfield**: it measures a step whose whole job is two
/// want-set scans, and it measures it at 0.18 ms. Wave I4's instrument then
/// measured a step over the phase-30 city — 100 `PcgVolume` blocks, 1 000
/// grammar buildings, 370 468 solids, a streamed terrain and a skinned character
/// — at **13.0–14.9 ms**, three times the older budget, with nothing regressed
/// at all. One number cannot be both, and stretching the walker's budget to
/// cover the city would have retired the only tripwire the walker has.
///
/// So this is the city's own, asserted by `fps_instrument.rs`'s
/// `the_fixed_steps_own_budget` over the same composed scene the frame numbers
/// come from, and it is a **whole-step total whose phases are printed beside it**
/// — the point of the wave that minted it is that a step which cannot say where
/// its milliseconds went is the CPU twin of the frame that could not say where
/// its GPU milliseconds went.
///
/// # A clock, so: release only, real machine only
///
/// This module's own law (`prefer a budget in a unit the machine cannot
/// inflate`) applies at full strength — the step is a wall clock, `[profile.dev]`
/// is `opt-level = 1` with debug assertions, and a shared CI runner's
/// milliseconds are a fact about the runner. The arm therefore **reports**
/// everywhere and **asserts** under `cargo test --release` off CI, which is the
/// same conditioning [`SHIPPING_FRAME_CEILING_MS`] carries.
///
/// # The number, and the ratchet it has already taken
///
/// Minted at **20.0** against the *unrepaired* 13.0-14.9 ms measurement, on
/// [`PROJECTION_BUDGET_MS`]'s stated precedent: a budget minted after a fix
/// cannot certify that the fix is what moved the number. It ratcheted to **6.0**
/// in the same wave, against a repaired step of **1.222 ms** - about five times
/// the measurement, and a fifth of the older walker-scene budget it sits beside
/// even though it covers three hundred times the content. Read the git log of
/// this line, not just its value.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const CITY_STEP_BUDGET_MS: f64 = 6.0;

/// Hard **mean** budget, in milliseconds, for the fixed step's **crowd** phase
/// (wave NPC1a) — `inf_ecs::crowd::step_crowd` over a thousand NPCs.
///
/// # Why a phase gets its own number
///
/// [`CITY_STEP_BUDGET_MS`] is a whole-step total, and it is deliberately one
/// number with its phases printed beside it. That works while every phase is
/// something the engine has always done; it stops working the moment a phase is
/// *new*, because a new phase's growth is invisible inside a total that has
/// three hundred times the content in it. NPC1a is exactly that case: before it
/// there was no crowd, no `crowd` row and nothing anywhere in this tree that had
/// ever put a thousand NPCs in a world and looked at the clock.
///
/// So the crowd carries its own ceiling, sized against its own measurement, and
/// the whole-step number stays where it is. That is also the shape the wave's
/// "zero cost when absent" claim needs: a level with no population must move
/// neither number, and only a per-phase one can say so.
///
/// # What is IN it, and what is deliberately not
///
/// The phase is the tier decision, the materialize/dematerialize pass and the
/// kinematic route write — `inf_ecs::crowd::step_crowd`, and nothing else. The
/// pose the `Full` and `Near` tiers then pay for lands in `animation`, and the
/// capsules they hand the solver land in `physics3d sync` and `solver`. That
/// split is the point rather than an accident: the crowd system's job is to
/// decide *how many* agents reach those phases, so a budget that folded them in
/// could be met by a tier ladder that decided nothing.
///
/// # The number
///
/// Measured by `runtime/inf-player/tests/crowd_sweep.rs` over the fps
/// instrument's own composed scene (the phase-30 city, a streamed terrain, the
/// phase-29 character) with **1 000** agents standing in a 320 m block on the
/// drive line, three rounds of forty steps after forty discarded, MIN of rounds:
///
/// | N | crowd phase | per agent |
/// |---|---|---|
/// | 0 | **0.0002 ms** | — |
/// | 1 | 0.003 | 3.0 µs |
/// | 10 | 0.004 | 0.4 |
/// | 100 | 0.014 | 0.14 |
/// | **1 000** | **0.103 ms** | **0.103 µs** |
///
/// # The number this was FIRST minted against, and why it moved (NPC1a audit)
///
/// NPC1a minted 2.0 ms as ~7× a measurement of **0.282–0.301 ms** over six
/// readings. That measurement was real and it was not a measurement of the
/// crowd: the phase built a digest of **every** pose in the store on every step
/// to serve the rare demotion, so it scaled with the *level's* posed characters
/// rather than with the population. The wave's own table said so and nobody read
/// it that way — at N = 1 000 the phase charged **0.282 ms banded and 0.759 ms
/// all-`Full`**, the same thousand agents doing the same work, differing only in
/// how many characters were in the pose store.
///
/// With the digest taken per demotion the two configurations now agree —
/// **0.103 and 0.109 ms** — which is what a phase whose work is per *agent* has
/// to look like, and the ratchet takes the budget down with it.
///
/// **1.0 ms is ~10× the 1 000-agent measurement**, near
/// [`PROJECTION_BUDGET_MS`]'s own minting arithmetic — and, like that one, the
/// measurement it is minted against is a `dev`-profile number (`opt-level = 1`
/// with debug assertions), so the release margin is wider still. It is a sixth
/// of [`CITY_STEP_BUDGET_MS`], which is the property that makes it able to see
/// anything: a crowd phase that grew to fill a whole step would be invisible to
/// the total and trips this by an order of magnitude.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons — reported
/// everywhere, asserted under `cargo test --release` off CI.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 2.0
/// (NPC1a) and ratcheted to 1.0 by the NPC1a audit.
pub const NPC_STEP_BUDGET_MS: f64 = 1.0;

/// **What the `society` phase may cost on a settled level**, milliseconds
/// (island wave NPC1d).
///
/// The phase does two different things and only one of them is steady. On a
/// **settled** level — every volume folded, every resident given a day — it is
/// one entity walk that finds no new volume and returns; that is what this
/// budget is about, and it is the number a shipped frame pays sixty times a
/// second for ever.
///
/// The other thing is the **derivation**, which happens once per volume as a
/// settlement streams in: a pavement ring, an interior absorbed, and up to
/// [`inf_ecs::society::SOCIETY_PLANS_PER_STEP`] days planned over the network.
/// That is a bounded transient by construction — the per-step cap is exactly the
/// mechanism that stops it being a cliff — and it is measured and REPORTED
/// rather than budgeted, because a load-time spike asserted against a steady
/// budget is the "a load asserts LOAD_BUDGET_MS, never FRAME_BUDGET_MS" rule
/// (P20) pointed the wrong way.
///
/// **0.5 ms**, half of [`NPC_STEP_BUDGET_MS`]: the settled phase does strictly
/// less work than the crowd phase beside it (a walk, against a walk plus a
/// per-agent plan and a per-agent transform write), so a society phase that grew
/// past half a crowd's has stopped being a walk.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 0.5
/// (NPC1d).
pub const SOCIETY_STEP_BUDGET_MS: f64 = 0.5;

/// The population the [`NPC_STEP_BUDGET_MS`] measurement is taken at.
///
/// Pinned beside the budget because a per-step millisecond without a population
/// beside it is a number about an unnamed world — the `SHIPPING_FRAME_CEILING_MS`
/// rule ("a frame time without an adapter name") one system over. A sweep that
/// quietly measured a hundred agents would meet this budget by a factor of ten
/// and certify nothing.
pub const NPC_BUDGET_AGENTS: usize = 1000;

/// **What the `vehicle` phase may cost**, milliseconds (island wave VEH1a) —
/// `inf_physics::d3::step_vehicles` over [`VEHICLE_BUDGET_CARS`] cars.
///
/// # Why a phase gets its own number, again
///
/// [`NPC_STEP_BUDGET_MS`]'s argument, verbatim, one system along: a whole-step
/// total is one number with its phases printed beside it, and that works while
/// every phase is something the engine has always done. It stops working the
/// moment a phase is *new*, because a new phase's growth is invisible inside a
/// total that has an island in it. Before VEH1a there was no `vehicle` row —
/// P29.7 ran `step_vehicles` inside the last statement of
/// `step_character_movement`, where a car's milliseconds were charged to
/// `character move` and could not be told from a crowd's.
///
/// # What is IN it
///
/// The four wheel rays, the model's `solve`, the force application and the
/// visual wheel write — `step_vehicles`, and nothing else. A driven car also
/// pays `solver` (its chassis is a dynamic body), `physics3d sync` and
/// `character move` (its driver), and those stay where they are: this phase's
/// job is the *rig*, so a budget that folded the solver in could be met by a
/// vehicle door that did nothing.
///
/// # The number
///
/// Measured by `inf-physics`'s `the_vehicle_phase_asks_four_questions_a_car_
/// and_costs_what_it_prints` over cars parked on a four-tile 1 m heightfield —
/// the island's own grid — settled first, then MIN of three rounds of forty
/// steps:
///
/// | cars | dev (VEH1a) | dev (VEH2a) | per car (VEH2a) |
/// |---|---|---|---|
/// | 1 | 0.0015 ms | 0.0018 ms | 1.81 µs |
/// | 4 | 0.0053 | 0.0063 | 1.57 |
/// | 16 | 0.0227 | 0.0271 | 1.70 |
/// | **64** | **0.1101 ms** | **0.1277 ms** | **2.00 µs** |
///
/// Linear in the car count, which is what an `O(vehicles)` walk over four rays
/// each has to look like. VEH1a's release column was 0.0882 ms at 64 cars, or
/// about 0.80 of its `dev` figure.
///
/// # RE-PRICED AT WAVE VEH2a: **+16 %, and the constant does not move**
///
/// The model under this phase went from a spring, a flat drive force and two
/// axis clamps to a wheel with angular state, a torque curve through a gearbox,
/// a differential, a simplified-Pacejka tyre solved stick-or-slide, anti-roll
/// bars, aero and three driver aids. That is **0.1101 → 0.1277 ms at 64 cars**,
/// 1.72 → 2.00 µs a car: sixteen per cent for all of it, because the expensive
/// thing in this phase was always the four ray casts and none of the new work
/// casts anything.
///
/// **The budget stays at 0.5 ms.** Re-minting it at VEH1a's own ~4.5× rule would
/// give 0.575, and §8 says this constant may only ever DECREASE — so the ratio
/// tightens instead, from ~4.5× to **~3.9×** the dev measurement. That is the
/// ratchet doing exactly what it is for: the wave that spends headroom is the
/// wave that has less of it afterwards.
///
/// The comparison to [`NPC_STEP_BUDGET_MS`]'s ~10× still holds and still has its
/// reason: this phase's work is a fixed four casts a car with no tier ladder in
/// front of it, so there is nothing here whose cost is *supposed* to vary. It is
/// a twelfth of [`CITY_STEP_BUDGET_MS`], which is the property that makes it
/// able to see anything: a vehicle phase that grew to a visible share of a 6 ms
/// step trips this by an order of magnitude.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons — reported
/// everywhere, asserted under `cargo test --release` off CI.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 0.5
/// (VEH1a); re-priced but NOT moved at VEH2a (see above).
pub const VEHICLE_STEP_BUDGET_MS: f64 = 0.5;

/// The fleet [`VEHICLE_STEP_BUDGET_MS`] is measured at.
///
/// [`NPC_BUDGET_AGENTS`]'s rule: a per-step millisecond without a population
/// beside it is a number about an unnamed world. Sixty-four is far more than the
/// island spawns (one car at each of seven settlements) and is sized for the
/// traffic wave that follows this one, so the budget does not have to move the
/// first time a level puts cars on its roads.
pub const VEHICLE_BUDGET_CARS: usize = 64;

/// **What the `traffic` phase may cost**, milliseconds (island wave VEH2b) —
/// `inf_physics::d3::traffic::step_traffic` over a settlement's whole car
/// population.
///
/// # Why a phase gets its own number, a third time
///
/// [`NPC_STEP_BUDGET_MS`]'s argument and then
/// [`VEHICLE_STEP_BUDGET_MS`]'s, one system along: a whole-step total works
/// while every phase is something the engine has always done, and stops working
/// the moment a phase is NEW, because a new phase's growth is invisible inside a
/// total that has an island in it.
///
/// # What is IN it
///
/// Everything a car costs that is not the rig: the block-stamp walk, the
/// carriageway derivation when the blocks move, one batch of commuter routes,
/// the band, the tier decision for every record, the body build or take-down,
/// the kinematic transform write for the clock's tier, the obstacle gather and
/// the steering decision for each `Full` car. What it does **not** hold is
/// `step_vehicles` — a steered traffic car pays the `vehicle` row exactly as the
/// hero's car does, at [`VEHICLE_STEP_BUDGET_MS`]'s own 2.00 µs — nor the solver
/// its chassis is a body in, nor `character move` for its driver.
///
/// # The number, and the arithmetic that sizes it
///
/// The population is bounded by GEOMETRY rather than by a setting:
/// `inf_ecs::traffic::KERB_SLOT_M` is 14 m, `KERB_OCCUPANCY` is 0.45, and
/// `TRAFFIC_NEAR_M` is 128 m — so the cars that exist at once are the occupied
/// slots inside a 128 m disc, and the cars that are RIGS are the ones inside
/// 64 m. Measured on a 3x3 city-block town (four 20 m streets, 49 parked cars):
/// **10 `Full` and 21 `Near`** at the crossroads. Ten rigs at VEH2a's own
/// 2.00 µs is **0.020 ms of the 0.5 ms `vehicle` budget — four per cent** — and
/// the whole of that town's traffic phase is under a tenth of this ceiling.
///
/// One millisecond, which is [`NPC_STEP_BUDGET_MS`]'s figure, because the two
/// phases do the same shape of work over populations of the same order: a band,
/// a tier per record and a materialization on transitions. It is a sixth of
/// [`CITY_STEP_BUDGET_MS`], which is the property that makes it able to see
/// anything — a traffic phase that grew to a visible share of a 6 ms step trips
/// this first.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 1.0
/// (VEH2b).
pub const TRAFFIC_STEP_BUDGET_MS: f64 = 1.0;

/// **What the `dispatch` phase may cost**, milliseconds (wave EMS2).
///
/// # Why a phase gets its own number, a fifth time
///
/// [`NPC_STEP_BUDGET_MS`]'s argument, then [`VEHICLE_STEP_BUDGET_MS`]'s, then
/// [`TRAFFIC_STEP_BUDGET_MS`]'s, then [`AUDIO_STEP_BUDGET_MS`]'s: a whole-step
/// total works while every phase is something the engine has always done, and
/// stops working the moment a phase **grows**. `dispatch` is new in this wave,
/// so it starts with a row and a ceiling rather than being discovered inside
/// `CITY_STEP_BUDGET_MS` two waves later.
///
/// # What is in it
///
/// On almost every step: one `block_stamp` walk (which `society` and `traffic`
/// already make in the two phases before it), one walk over the units to steer
/// them through `inf_ecs::traffic::drive_intent`, the siren and light-bar cue
/// lists, and the smoke's rise-and-fade over at most
/// `inf_ecs::dispatch::MAX_PUFFS` sprites. All `O(units + puffs)`, both bounded
/// by constants.
///
/// On the steps something is **assigned** it also pays for a carriageway graph
/// and one `inf_nav` Dijkstra per candidate unit — which is why
/// `inf_physics::d3::dispatch::ASSIGNS_PER_STEP` is **one**: a town that woke to
/// four simultaneous emergencies spreads them over four steps rather than
/// spiking one.
///
/// # Half a millisecond, and where the number comes from
///
/// [`VEHICLE_STEP_BUDGET_MS`]'s figure, because the two phases bound the same
/// population the same way: `inf_ecs::dispatch::MAX_UNITS` is 64 and
/// [`VEHICLE_BUDGET_CARS`] is 64, deliberately — a unit that is responding **is**
/// a vehicle on four rays, and the phase that decides where it goes should cost
/// less than the phase that integrates it. It is a twelfth of
/// [`CITY_STEP_BUDGET_MS`], which is the property that makes it able to see
/// anything: a dispatch phase that grew to a visible share of a 6 ms step trips
/// this first.
///
/// Measured by `ems2_dispatch_gate::the_dispatch_phase_costs_what_it_costs` on
/// three units, three open incidents and a town of traffic — the table is
/// printed on every run, in dev and in CI, and asserted on a real machine in
/// release.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 0.5
/// (EMS2).
pub const DISPATCH_STEP_BUDGET_MS: f64 = 0.5;

/// **What the `audio` phase may cost**, milliseconds (island wave VEN1b audit).
///
/// # Why a phase gets its own number, a fourth time
///
/// [`NPC_STEP_BUDGET_MS`]'s argument, then [`VEHICLE_STEP_BUDGET_MS`]'s, then
/// [`TRAFFIC_STEP_BUDGET_MS`]'s: a whole-step total works while every phase is
/// something the engine has always done, and stops working the moment a phase
/// **grows**, because new growth is invisible inside a total that has an island
/// in it.
///
/// The `audio` row has existed since the step profile did and had never been
/// asserted, because until wave VEN1b it was a walk over the emitters and a
/// listener pose. That wave gave it real per-step work — see below — and the
/// wave's own arms measured `society` and `crowd`, the two phases it barely
/// touched. This constant is the audit's answer to that.
///
/// # What is IN it, and the shape that makes it worth watching
///
/// The autoplay walk over every `AudioSource` in the world, the despawn sweep,
/// the listener pose, the vehicle engine loop's pitch/volume pair per car — and
/// VEN1b's **doorway re-evaluation**: one `inf_physics::d3::audio::portal_gain`
/// for every looping spatial source that opts into occlusion and stands inside
/// its own `max_distance`, every step.
///
/// The last of those is the one with a shape in it. `portal_gain` falls through
/// to `portal_of`, which reads `d3::door::placements` — the **unbanded** list,
/// which builds a `DoorPlacement` (and a label `String`) for every authored door
/// and every grammar `PcgDoorway` in the resident world.
/// `placements_near`'s own doc is the measurement of what that can cost: *"the
/// shipped city plans 19 790 doorways and the band keeps 234"*. The venue's
/// speakers are the first content ever to take that path.
///
/// # The number, and the measurement that sized it
///
/// Measured at the club on the CI fixture — **six speakers, 750 doors resident,
/// 750 `SetOcclusion` over 120 steps** — with the profiler armed, on both hosts:
///
/// ```text
///   society 0.028 ms | crowd 0.178 ms | audio 0.185 ms
/// ```
///
/// The audio row is the dearest of the three, which is the fact this constant
/// exists to keep visible: the wave that grew it measured the other two.
///
/// **0.55 ms before the hoist.** The first reading was `audio 0.549 / 0.556 ms`
/// — three times the crowd phase — because the door list was rebuilt inside
/// `portal_gain` once per source, five times a step over 750 doors.
/// `audio::portal_gain_in` takes the list the step already built; the verdicts,
/// the 120 step digests and the audio command stream are identical either way.
/// What is left scales with `doors + sources × doors` comparisons rather than
/// with `sources × doors` allocations.
///
/// One millisecond, [`NPC_STEP_BUDGET_MS`]'s and [`TRAFFIC_STEP_BUDGET_MS`]'s
/// figure, and a sixth of [`CITY_STEP_BUDGET_MS`] — which is the property that
/// makes it able to see anything at all: an audio phase that grew to a visible
/// share of a 6 ms step trips this first. The fixture sits at 19 % of it, and
/// the headroom is deliberately not spent: a city block's 19 790 doorways is
/// twenty-six times this fixture's 750, and the day a gate stands a listener in
/// one, this is the arm that says so.
///
/// # A clock, so: release only, real machine only
///
/// [`CITY_STEP_BUDGET_MS`]'s conditioning, for its reasons.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Minted at 1.0
/// (VEN1b audit).
pub const AUDIO_STEP_BUDGET_MS: f64 = 1.0;

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

/// **What "≥ 60 fps" means** (island wave I4): the 95th-percentile frame, at a
/// shipping resolution, over composed content, on a reference desktop GPU.
///
/// Every other frame number in this tree is measured against
/// [`inf_core::FRAME_BUDGET_MS`] — **33 ms, a 30 fps floor**, and deliberately
/// so: those gates run on CI machines with software adapters, where *"a budget
/// nobody can meet is a budget everybody disables"*. That number could never
/// carry a 60 fps claim, and until this constant existed no number in the
/// repository could: the AAA-readiness certification's finding was that *"the
/// only GPU frame harness renders 640 × 360"* and *"no test in this repo measures
/// fps at a shipping resolution"*.
///
/// # What this is measured over
///
/// `runtime/inf-player/tests/fps_instrument.rs`: the phase-30 city (1 000
/// grammar buildings, 370 468 solids, 100 banded volumes, a real road mesh), a
/// streamed terrain paging beneath it, the phase-29 wizard character skinned and
/// animating, and the render settings a shipped player builds for that level on
/// the live adapter — driven down a scripted street at 1920 × 1080 and
/// 2560 × 1440, re-projected in full every frame.
///
/// # Why the 95th percentile and not the mean
///
/// A mean hides the frame that stutters, and a stutter is the only frame a
/// player notices. p50 says whether the engine is fast; **p95 says whether it is
/// smooth**, and 60 fps is a claim about smoothness. [`SHIPPING_FRAME_P99_BUDGET_MS`]
/// is the hitch ceiling beside it.
///
/// # This is a TARGET, and it is not met
///
/// It is deliberately **not asserted**, because on 2026-08-20 the engine does not
/// meet it and a constant asserted where it fails is a red build somebody
/// silently raises. What the instrument does with it is print the **distance**,
/// every run, at both resolutions. The tripwire that *is* asserted is
/// [`SHIPPING_FRAME_CEILING_MS`] beside it, which ratchets down toward this
/// number; the day the two meet, this one becomes the assertion and the other is
/// deleted.
///
/// Separating them is the point. A single constant would have had to be either
/// the goal (and permanently red) or the measurement (and silently a claim that
/// 46 ms is what 60 fps means).
///
/// The reference card is this machine's **RTX 4070 Ti**; the instrument prints
/// the adapter it ran on beside every number, because a frame time without an
/// adapter name is a number about an unnamed machine.
pub const SHIPPING_FRAME_BUDGET_MS: f64 = 16.6;

/// The **hitch target** beside [`SHIPPING_FRAME_BUDGET_MS`]: the 99th-percentile
/// frame of the same run.
///
/// Two frames in a hundred are allowed to miss the 60 fps deadline; what they may
/// not do is miss it by a lot. Twice the frame budget is one dropped frame — the
/// player sees a hitch and the next frame is on time. A p99 past this means the
/// hundredth frame is not a jitter but a *stall*: a pipeline compile, a synchronous
/// upload, a residency cliff — something with a name, which is what makes this a
/// separate number rather than a looser version of the first.
///
/// A target, on the same terms as the budget above.
pub const SHIPPING_FRAME_P99_BUDGET_MS: f64 = 33.2;

/// **The ratcheting tripwire** the instrument asserts: the 95th-percentile frame
/// at the worse of the two shipping resolutions, on a real adapter, in a
/// **release** build.
///
/// # The measurement it comes from
///
/// Island wave I4, RTX 4070 Ti, `cargo test --release`, MIN of three rounds of
/// 120 frames after a **discarded pass of 120** (the first write-up said "24
/// warm-up"; the harness's `FRAMES` is one constant and the discarded pass is a
/// whole one of them — corrected by the I4 audit), over the composed instrument
/// scene (1 000 grammar buildings / 370 468 solids, a streamed terrain, a skinned
/// character): see the wave's ROADMAP block for the p50/p95/p99 pair at 1080p and
/// 1440p and the per-pass breakdown. This constant is set with headroom over the
/// measured p95 the way every §8 tripwire is — *"deliberately generous… a
/// regression that matters moves these by an order of magnitude and trips them; a
/// 20 % drift does not, and is not what CI is for."*
///
/// # RE-MINTED FOR A LIT RENDERER (wave CERT1)
///
/// This paragraph used to end *"the day a shipped level turns the stack on, this
/// ceiling is measuring a different renderer and has to be re-minted, not
/// raised."* **That day was wave CERT1**, which authored shadows, GI, bloom,
/// SSAO, TAA and flare into the showcase island and the three 3D starter
/// templates. So the ceiling is discharged as the paragraph required:
///
/// * the instrument now **asserts this constant on the LIT configuration too**,
///   not only on the shipped one — the same run, the same content, the same
///   percentile, behind the same adapter/CI/profile exemptions;
/// * and it is **ratcheted down**, 40.0 -> 38.0, on the measurement that made
///   the re-mint possible.
///
/// The number that made it possible is the one nobody expected. Island wave I4
/// measured the lighting stack at **p95 92.3-92.9 ms against 43.7-44.0**, and
/// wave I4b at 38.1-41.8 against 15.8-19.5. At CERT1, on the same composed city
/// at 1080p on the same RTX 4070 Ti, **over two runs**:
///
/// | run | lit p95 | shipped p95 | delta | lit GPU | shipped GPU |
/// |---|---|---|---|---|---|
/// | whole suite | 16.862 | 16.733 | **+0.129** | 5.506 | 7.575 (**-2.069**) |
/// | this arm alone | 20.499 | 14.209 | **+6.290** | 5.439 | 3.409 (**+2.029**) |
///
/// **Both are quoted because one of them alone would be a claim.** The lit
/// frame's own GPU cost barely moved (5.506 / 5.439); what moved is the SHIPPED
/// frame's (7.575 / 3.409), which is this file's own warning about device state
/// paying out — a card that has been boosting through a six-arm suite is not the
/// card that ran a single test, and the GPU columns are only comparable between
/// runs whose CPU frames are comparable. The honest reading is that the stack
/// costs somewhere between a seventh of a millisecond and six of them on this
/// content, and that it is nowhere near I4's forty-eight.
///
/// So the "different renderer" this ceiling had to be re-minted for is close
/// enough to the one it was minted on that a ceiling covering BOTH is tighter
/// than the one that covered neither. **38.0 is 1.85x the worst lit p95
/// observed** (20.499), which is the same order of headroom the constant has
/// always carried.
///
/// **The island is still reported and never asserted here**, and it is the
/// number that keeps this constant honest rather than one that flatters it. On
/// the same machine, the same run and the same resolution, the shipped island —
/// the 51.38 km² one, with the record its own level now authors — measures
/// **p50 39.672 / p95 42.299 / p99 43.188 ms, 25.2 fps**, and with its town's
/// own thousand-agent society at the rush hour **p50 138.701 / p95 153.814 ms,
/// 7.2 fps**. Those are over this ceiling and they are supposed to be: it is a
/// city's ratchet, asserting it over a different world would re-pin it by
/// accident, and the island's own distance from sixty is a matter for the
/// certification memo rather than for a tripwire.
///
/// # Where it is asserted
///
/// **Locally, in a release build, on a real adapter, and in the I9
/// certification.** Nowhere else, each exemption taken by name in the test:
///
/// * a software or paravirtual adapter reports (the P15.1 rule);
/// * **every CI runner** reports (the P26.5 rule this file's header records);
/// * the **dev profile** reports — `[profile.dev]` is `opt-level = 1` with debug
///   assertions on, so the CPU half of a frame measured there is a fact about a
///   build nobody ships. That is the paravirtual-adapter law one layer down, and
///   it is why the full battery running this test does not assert it.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** It is expected to
/// walk down toward [`SHIPPING_FRAME_BUDGET_MS`] as the frame's named costs are
/// paid off; the instrument prints the distance so the next step is always
/// visible.
pub const SHIPPING_FRAME_CEILING_MS: f64 = 38.0;

/// The 99th-percentile twin of [`SHIPPING_FRAME_CEILING_MS`], asserted on exactly
/// the same terms by the same test.
///
/// **RATCHET RULE (§8): this constant may only ever DECREASE.**
pub const SHIPPING_FRAME_P99_CEILING_MS: f64 = 46.0;

/// The message every budget assertion fails with — the ratchet rule, at the point
/// where somebody is most tempted to break it.
pub const RATCHET_NOTE: &str =
    "(the §8 budget only ratchets DOWN — investigate the regression, do not raise it)";
