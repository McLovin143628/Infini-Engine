//! **THE FIXED STEP'S OWN BREAKDOWN** (island wave I4b) — where the sim
//! milliseconds go.
//!
//! Wave I4's instrument measured the frame and found it **CPU-bound**, with the
//! single dearest thing in it the fixed step: **13.0–14.9 ms** over the phase-30
//! city, against a whole 1080p GPU frame of 14.4–19.8 ms. Of that, the I3
//! collider band accounted for ~2.2 ms **and the other ~11.5 ms was
//! unattributed** — the I4 audit routed "attribute it before prescribing" here
//! as wave I4b's first clause.
//!
//! A frame that is CPU-bound and cannot say *where* is the defect the AAA
//! certification found on the GPU side, and `fps_instrument.rs` closed it there
//! with a query-set clock whose segments tile the frame. This module is that
//! same instrument one processor over: [`RuntimeSim::fixed_step`] marks every
//! phase of its own body into a [`StepProfile`], and the phases **tile the
//! step** by construction — each mark measures from the previous one, so the sum
//! is the step.
//!
//! [`RuntimeSim::fixed_step`]: crate::runtime_sim::RuntimeSim
//!
//! # Off by default, and the cost when it is on
//!
//! [`RuntimeSim::set_step_profiling`](crate::runtime_sim::RuntimeSim::set_step_profiling)
//! arms it. Off, every mark is one predictable branch on a `bool` the step
//! already has in cache. On, it is one `Instant::now()` per phase —
//! `QueryPerformanceCounter` on Windows, ~25 ns — so the whole breakdown costs
//! well under a microsecond against a step measured in milliseconds. It is a
//! diagnostic and never a gate input: nothing in the fixed step reads the
//! profile, so a profiled step and an unprofiled one produce **byte-identical**
//! sim state. `the_profile_does_not_move_the_simulation` is the arm that says so.
//!
//! # Why the shipped host and not both
//!
//! The editor's `SimSession::fixed_step` is a MIRROR of the shipped one, and
//! every *behavioural* line in it is mirrored deliberately. A stopwatch is not
//! behaviour: it reads no sim state, writes none, and changes no ordering. The
//! shipped host is the one the fps instrument drives and the one a shipped game
//! runs, so it is the one that carries the clock — exactly as
//! `inf_render::timing` lives on the renderer and not on a second copy of it.

use std::time::Instant;

/// How many phases one fixed step is split into.
pub const STEP_PHASES: usize = 29;

/// The phases, in the order [`RuntimeSim::fixed_step`] runs them.
///
/// [`RuntimeSim::fixed_step`]: crate::runtime_sim::RuntimeSim
///
/// Two of them are gathered rather than contiguous, and both are gathered
/// because the *question* is about the total: `propagate` runs three times in one
/// step (after write-back, after root motion, after attachments) and a reader
/// asking "what does transform propagation cost this world" wants one number;
/// the same goes for the two dispatch drains folded into their event phases.
/// What each one covers, in order, is documented on the matching constant in
/// this module's `phase` — deliberately there and not as a trailing comment
/// here, because
/// `rustfmt` aligns trailing comments into runs of spaces on lines that contain
/// a string literal, and `inf_packager`'s workspace-wide
/// `no_string_literal_in_the_workspace_carries_an_eaten_continuation` reads such
/// a run as an eaten `\`-continuation. A table that trips the tree's own
/// mangled-whitespace gate is a table that has to be maintained around it.
pub const STEP_PHASE_NAMES: [&str; STEP_PHASES] = [
    "cell stream",
    "terrain stream",
    "biome scatter",
    "sky",
    "society",
    "crowd",
    "traffic",
    "dispatch",
    "physics2d sync",
    "physics3d sync",
    "water forces",
    "input events",
    "blueprint tick",
    "character move",
    "vehicle",
    "gameplay",
    "solver",
    "collision drain",
    "write-back",
    "propagate",
    "deformation",
    "animation",
    "attachments",
    "cloth + hair",
    "mods",
    "destruction",
    "audio",
    "camera",
    "position capture",
];

/// Phase indices, named so a mark cannot drift from its meaning — and the place
/// each phase's *contents* are written down, since [`super::STEP_PHASE_NAMES`]
/// carries only the labels.
pub(crate) mod phase {
    /// P16.5 partition cells.
    pub const CELL_STREAM: usize = 0;
    /// P16.3b2 sim-side terrain paging.
    pub const TERRAIN_STREAM: usize = 1;
    /// I7b — the biome-bound population of the ground that just paged in.
    pub const BIOME_SCATTER: usize = 2;
    /// P17.1 time of day + P17.4 the weather blend.
    pub const SKY: usize = 3;
    /// NPC1d — the level's own society: which of its buildings have been folded
    /// into a walkable network, and the days their residents are given. HERE,
    /// after the sky (a schedule reads the clock) and before the crowd (a record
    /// installed after the tiering would be `Dormant` for one step on one host
    /// and not the other). Its own phase rather than a corner of the crowd's,
    /// because a level's whole population is derived in it and a step that
    /// cannot say where its milliseconds went is the defect wave I4b existed to
    /// remove. Inert — one entity walk that finds nothing — on a level whose
    /// volumes offer no resident.
    pub const SOCIETY: usize = 4;
    /// NPC1a — the sim-LOD tier decision and the crowd's own kinematic step.
    /// HERE, after streaming (so a streamed-in agent is tiered on the step it
    /// arrives) and before the physics sync (so the bridge sees this step's
    /// bodies), the character step and the animation (so both read this step's
    /// tiers). Inert — one `contains_resource` — on a level with no crowd.
    pub const CROWD: usize = 5;
    /// VEH2b — the level's own carriageway, the traffic tier decision, and the
    /// stick each steered car's driver is handed.
    ///
    /// **HERE, immediately after [`CROWD`]**, and for the crowd's own two
    /// reasons one system along: a car that materializes this step has to be
    /// mirrored by the physics sync on the same step (or it is a body that
    /// exists for the renderer and not for the solver), and the driver's
    /// **intent** has to be written before [`CHARACTER_MOVE`] reads it six
    /// phases later — which is the same slot a player's stick is written in,
    /// through the same `VehicleControls::from_intent`.
    ///
    /// Its own row rather than a corner of [`CROWD`]'s, because a settlement's
    /// whole traffic population is derived in it and a step that cannot say
    /// where its milliseconds went is the defect wave I4b existed to remove.
    /// Inert — one `get_resource` and one entity walk that finds nothing — on
    /// a level with no blocks in it.
    pub const TRAFFIC: usize = 6;
    /// EMS2 — what has happened, who is going, and the stick their driver is
    /// handed: `inf_physics::d3::dispatch::step_dispatch`.
    ///
    /// **HERE, immediately after [`TRAFFIC`]**, and for the traffic's own two
    /// reasons one system along: a crew body that materializes this step has to
    /// be mirrored by the physics sync on the same step (or it is a person the
    /// renderer can see and the solver cannot), and a responding unit's driver's
    /// **intent** has to be written before [`CHARACTER_MOVE`] reads it — which
    /// is the same slot a player's stick and a commuter's are written in,
    /// through the same `VehicleControls::from_intent`.
    ///
    /// After the traffic rather than before it, because the yield rule reads the
    /// dispatcher: a civilian car must be told a siren is behind it in the same
    /// step the siren decided it was coming, and running the dispatcher first
    /// would have put every yield one step in the past.
    ///
    /// Its own row rather than a corner of [`TRAFFIC`]'s, because a level's
    /// whole emergency response is decided in it — an `inf_nav` search per
    /// candidate unit, on the steps something happens — and a step that cannot
    /// say where its milliseconds went is the defect wave I4b existed to remove.
    /// Inert — one `block_stamp` walk the two phases before it already make, and
    /// one `get_resource` — on a level with no emergency vehicle in it.
    pub const DISPATCH: usize = 7;
    /// The 2D bridge's ECS → rapier2d walk.
    pub const PHYSICS2D_SYNC: usize = 8;
    /// The 3D bridge's walk — the I3 collider band's gather lives here, and so
    /// does the P22.3 fracture follow that precedes it.
    pub const PHYSICS3D_SYNC: usize = 9;
    /// P20.2 buoyancy + hydrodynamic drag.
    pub const WATER: usize = 10;
    /// Wave 3 input edges and the dispatches they queue.
    pub const INPUT_EVENTS: usize = 11;
    /// The Tick pass over every actor, and the dispatches it queues.
    pub const BLUEPRINT_TICK: usize = 12;
    /// P29.3 — the one Ring-0 movement step, plus the intent that feeds it.
    pub const CHARACTER_MOVE: usize = 13;
    /// VEH1a — `inf_physics::d3::step_vehicles`: the wheel rays, the model's
    /// forces and the visual wheel write.
    ///
    /// **HERE, immediately after [`CHARACTER_MOVE`]**, which is exactly where it
    /// ran before it had a row: a driver's controls are written by the character
    /// step above (from the same intent), and the forces must land before
    /// `bridge.step`. The ordering is unchanged and the trace is unmoved; what
    /// moved is that the milliseconds are now attributed.
    ///
    /// # Why it left the movement door
    ///
    /// P29.7 put `step_vehicles` inside the last statement of
    /// `step_character_movement` on a real argument — *"a sibling both hosts had
    /// to call separately would be a hand-maintained mirror"* — and the price of
    /// that argument is a phase that cannot say where its milliseconds went,
    /// which is the defect wave I4b existed to remove. On the island a car is
    /// not a corner of the character step: it casts four rays into a streamed
    /// heightfield every step of the drive.
    ///
    /// The mirror the argument feared is paid for rather than avoided: the two
    /// call sites are fenced (`MIRROR-BEGIN vehicle_step`) and pinned
    /// character-for-character by `inf-editor-core`'s `fixed_step_mirror`, which
    /// is the same instrument the projectors already use — and it is a stronger
    /// guard than "it is one statement inside another function", because that
    /// guarded the *call* and nothing guarded the two hosts agreeing about
    /// **when**.
    pub const VEHICLE: usize = 14;
    /// I6 — doors, weapons and the health they spend, plus the host's own drain
    /// of the energy they owe the P22 damage door.
    pub const GAMEPLAY: usize = 15;
    /// rapier2d + rapier3d.
    pub const SOLVER: usize = 16;
    /// Wave 3 contacts and overlaps, and the dispatches they queue.
    pub const COLLISION_DRAIN: usize = 17;
    /// rapier → ECS.
    pub const WRITE_BACK: usize = 18;
    /// The transform + visibility DFS. **Three call sites, gathered.**
    pub const PROPAGATE: usize = 19;
    /// P22.1 — the ground remembers what stood on it.
    pub const DEFORMATION: usize = 20;
    /// Play-heads, state machines and root motion.
    pub const ANIMATION: usize = 21;
    /// P11.3 sockets.
    pub const ATTACHMENTS: usize = 22;
    /// P24.4 garments and hair.
    pub const CLOTH_HAIR: usize = 23;
    /// P14.5 WASM mods — which propagate internally, so their propagate is here
    /// rather than in [`PROPAGATE`].
    pub const MODS: usize = 24;
    /// P22.3 fracture write-back, the structural solve and the debris budget.
    pub const DESTRUCTION: usize = 25;
    /// P12.3 — the audio command queue.
    pub const AUDIO: usize = 26;
    /// P29.6 — the locomotion camera.
    pub const CAMERA: usize = 27;
    /// The interpolation history roll and the rising-edge clear.
    pub const POSITION_CAPTURE: usize = 28;
}

/// One fixed step's phase milliseconds — or, after
/// [`accumulate`](Self::accumulate) and [`scale`](Self::scale), a mean over many.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepProfile {
    /// Milliseconds per phase, indexed by [`STEP_PHASE_NAMES`].
    pub ms: [f64; STEP_PHASES],
}

impl StepProfile {
    /// The whole step — the sum of its phases, which is what "the phases tile the
    /// step" means arithmetically.
    pub fn total_ms(&self) -> f64 {
        self.ms.iter().sum()
    }

    /// `(name, milliseconds)` in step order.
    pub fn rows(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        STEP_PHASE_NAMES.iter().copied().zip(self.ms)
    }

    /// `(name, milliseconds)` dearest first — the order a reader wants when the
    /// question is "what do I fix".
    pub fn dearest_first(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<(&'static str, f64)> = self.rows().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Fold another step's phases in (for a mean over a run).
    pub fn accumulate(&mut self, other: &StepProfile) {
        for (slot, v) in self.ms.iter_mut().zip(other.ms) {
            *slot += v;
        }
    }

    /// Divide every phase by `n` — the second half of a mean.
    pub fn scale(&mut self, k: f64) {
        for slot in self.ms.iter_mut() {
            *slot *= k;
        }
    }
}

/// The stopwatch [`RuntimeSim::fixed_step`] marks its phases with.
///
/// [`RuntimeSim::fixed_step`]: crate::runtime_sim::RuntimeSim
///
/// Disabled it holds `None` and every [`mark`](Self::mark) is a branch. Enabled,
/// each mark charges the time since the previous mark to a phase and **adds**
/// rather than assigns, which is what lets `propagate`'s three call sites share
/// one row.
pub(crate) struct StepClock {
    at: Option<Instant>,
    ms: [f64; STEP_PHASES],
}

impl StepClock {
    pub(crate) fn start(on: bool) -> Self {
        StepClock {
            at: on.then(Instant::now),
            ms: [0.0; STEP_PHASES],
        }
    }

    /// Charge everything since the previous mark to `phase`.
    ///
    /// One line of body, because the arithmetic lives in
    /// [`mark_at`](Self::mark_at) — see there for why the split exists.
    #[inline]
    pub(crate) fn mark(&mut self, phase: usize) {
        if self.at.is_some() {
            self.mark_at(phase, Instant::now());
        }
    }

    /// [`mark`](Self::mark) with the clock read **supplied rather than taken**.
    ///
    /// # Why a stopwatch has a seam (the I7 CI-red)
    ///
    /// `mark` does two separable things: it reads a clock, and it charges an
    /// interval to a phase — adding, so `propagate`'s three call sites share one
    /// row. Only the second is a *property*, and it was tested by sleeping for
    /// two milliseconds twice and asserting the second reading was more than
    /// 1.5× the first. On a shared ubuntu runner the first "2 ms" stretch
    /// measured **4.990 ms** and the pair **6.991 ms**, the ratio came out at
    /// 1.40, and CI went red with nothing wrong: the arm was measuring the
    /// runner's scheduler, not this function.
    ///
    /// Splitting the clock read out makes the property testable without one.
    /// The arithmetic below is the *whole* of what `mark` does once it has a
    /// timestamp, so an arm that drives this drives the shipped code — there is
    /// no second copy to drift.
    #[inline]
    fn mark_at(&mut self, phase: usize, now: Instant) {
        if let Some(at) = self.at.as_mut() {
            self.ms[phase] += now.duration_since(*at).as_secs_f64() * 1000.0;
            *at = now;
        }
    }

    /// The finished breakdown, or `None` when profiling was off.
    pub(crate) fn finish(self) -> Option<StepProfile> {
        self.at.map(|_| StepProfile { ms: self.ms })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Slop allowed when comparing charged milliseconds, in milliseconds.
    ///
    /// **Not a noise budget — there is no noise here.** Nothing below reads a
    /// clock, so the only inexactness is the decimal→binary rounding in
    /// `Duration::as_secs_f64`'s divide by 1e9 and the multiply back up by 1e3,
    /// both of which are correctly-rounded IEEE operations and therefore the same
    /// on every target. A picosecond of slop against millisecond-scale stretches
    /// is nine orders of headroom over that and still eleven orders below the
    /// difference a replacing `mark` would make.
    const EXACT_MS: f64 = 1e-9;

    /// A clock started at a chosen instant, so an arm can charge intervals it
    /// *decides* rather than intervals it *measures*.
    ///
    /// The `tests` module is a child of `step_profile`, so it may build a
    /// `StepClock` from its private fields — which is why this seam costs the
    /// shipped clock nothing at all (no `#[cfg(test)]` constructor, no unused
    /// `pub(crate)` fn for the lib target's dead-code pass to find).
    fn clock_at(base: Instant) -> StepClock {
        StepClock {
            at: Some(base),
            ms: [0.0; STEP_PHASES],
        }
    }

    #[test]
    fn a_disabled_clock_measures_nothing_and_answers_none() {
        let mut c = StepClock::start(false);
        c.mark(phase::SOLVER);
        c.mark(phase::PROPAGATE);
        assert!(c.finish().is_none(), "a disabled clock has no profile");
    }

    /// **The gathered row really gathers.** `propagate` is marked three times in
    /// one step and the reader is owed one number, so a mark must ADD. Written
    /// with a mutation in mind: an assigning `mark` would report the last of the
    /// three propagations rather than all three, which is exactly the shape of
    /// under-attribution this module exists to remove.
    ///
    /// # This arm used to sleep, and that is why CI went red (the I7 CI-red)
    ///
    /// "A mark adds rather than assigns" is a statement about arithmetic, and
    /// the first cut of it asked a wall clock: spin 2 ms, mark, spin 2 ms, mark,
    /// assert the pair reads more than 1.5× the first. On a shared
    /// `ubuntu-latest` runner the first stretch measured **4.990 ms** and the
    /// pair **6.991 ms** — the sleep overshot by 2.5×, the ratio fell to 1.40,
    /// and a green tree went red with nothing changed but the machine it ran on.
    /// The I4b clock law, one processor over: **a control must hold what its
    /// subject holds**, and a structural property must not be asked of a clock.
    ///
    /// So the intervals below are *decided*: three exact stretches handed to
    /// [`StepClock::mark_at`], which is the whole of `mark`'s body after the
    /// timestamp. It can still fail — mutate `+=` to `=` and it reads 4 ms
    /// instead of 12 — and it can never redden from a scheduler.
    #[test]
    fn a_phase_marked_twice_sums_rather_than_replaces() {
        let base = Instant::now();
        let mut c = clock_at(base);

        // Three stretches, because `propagate` really is marked three times in
        // one step: 2 ms, then 6 ms, then 4 ms. Unequal on purpose — three equal
        // ones cannot tell "sums" from "reports the largest".
        let mut at = 0u64;
        let mut charged = Vec::new();
        for ms in [2u64, 6, 4] {
            at += ms;
            c.mark_at(phase::PROPAGATE, base + Duration::from_millis(at));
            charged.push(c.ms[phase::PROPAGATE]);
        }

        assert!(
            (charged[0] - 2.0).abs() < EXACT_MS,
            "one 2 ms stretch charged {} ms",
            charged[0]
        );
        assert!(
            (charged[1] - 8.0).abs() < EXACT_MS,
            "2 ms + 6 ms charged to one phase read {} ms — a mark that assigned \
             would read 6, and one that kept the largest would read 6 too",
            charged[1]
        );
        assert!(
            (charged[2] - 12.0).abs() < EXACT_MS,
            "2 ms + 6 ms + 4 ms charged to one phase read {} ms — a mark that \
             assigned would read 4, and one that kept the largest would read 6. \
             The mark is not summing, so `propagate`'s three call sites do not \
             share one row and the breakdown under-attributes exactly the phase \
             this module exists to attribute",
            charged[2]
        );

        // …and only that phase moved: a mark must not spill into its neighbours.
        let p = c.finish().expect("an armed clock has a profile");
        assert!((p.ms[phase::PROPAGATE] - 12.0).abs() < EXACT_MS);
        assert!(
            (p.total_ms() - 12.0).abs() < EXACT_MS,
            "the whole profile reads {} ms against 12 charged to one phase",
            p.total_ms()
        );
    }

    /// **The shipped `mark` really is the seam above with a clock attached.**
    ///
    /// The arm before this one drives `mark_at`, so on its own it would leave
    /// open the reading that `mark` does something else — the house law that a
    /// gate must aim at the thing it names. This aims at `mark`, and it does so
    /// with assertions no runner can move: an elapsed interval is non-negative
    /// and finite whatever the scheduler does, the clock must advance, and the
    /// charge must land on the phase that was named and nowhere else.
    #[test]
    fn the_live_mark_charges_the_phase_it_names_and_advances_the_clock() {
        let mut c = StepClock::start(true);
        let before = c.at.expect("an armed clock holds an instant");
        c.mark(phase::SOLVER);
        let after = c.at.expect("still armed");
        assert!(after >= before, "the mark must carry the clock forward");

        let charged = c.ms[phase::SOLVER];
        assert!(
            charged.is_finite() && charged >= 0.0,
            "a phase was charged {charged} ms"
        );
        for (i, v) in c.ms.iter().enumerate() {
            assert!(
                i == phase::SOLVER || *v == 0.0,
                "marking `solver` also charged {} ({v} ms)",
                STEP_PHASE_NAMES[i]
            );
        }

        // A second mark of the same phase can only ever raise it — the summing
        // property again, stated the one way a wall clock is allowed to state it.
        c.mark(phase::SOLVER);
        assert!(c.ms[phase::SOLVER] >= charged);
    }

    /// The names and the indices are one table — **and the two the ledger's
    /// headline attributions rest on are pinned by name** (the I4b audit).
    ///
    /// The wave's own table reads "solver 9.224 ms (72.5 %)" and "camera 2.258 ms
    /// (17.8 %)". Those are *labels off this array*, so a `phase` constant that
    /// drifted by one would have printed the same milliseconds against the wrong
    /// name, and the whole attribution — and the two repairs it prescribed —
    /// would have been aimed somewhere else. `SOLVER` and `CAMERA` were the two
    /// nothing pinned.
    #[test]
    fn the_names_and_the_indices_are_one_table() {
        assert_eq!(STEP_PHASE_NAMES.len(), STEP_PHASES);
        assert_eq!(STEP_PHASE_NAMES[phase::BIOME_SCATTER], "biome scatter");
        assert_eq!(STEP_PHASE_NAMES[phase::CROWD], "crowd");
        assert_eq!(STEP_PHASE_NAMES[phase::DISPATCH], "dispatch");
        assert_eq!(STEP_PHASE_NAMES[phase::PHYSICS3D_SYNC], "physics3d sync");
        assert_eq!(STEP_PHASE_NAMES[phase::PROPAGATE], "propagate");
        assert_eq!(STEP_PHASE_NAMES[phase::SOLVER], "solver");
        // VEH1a's own two, for the reason this arm exists: `vehicle` was
        // inserted in the MIDDLE of the table, so every constant below it moved
        // by one, and a label that drifted would put the wave's own budget
        // measurement against the wrong row.
        assert_eq!(STEP_PHASE_NAMES[phase::CHARACTER_MOVE], "character move");
        assert_eq!(STEP_PHASE_NAMES[phase::VEHICLE], "vehicle");
        assert_eq!(STEP_PHASE_NAMES[phase::GAMEPLAY], "gameplay");
        assert_eq!(STEP_PHASE_NAMES[phase::CAMERA], "camera");
        assert_eq!(
            STEP_PHASE_NAMES[phase::POSITION_CAPTURE],
            "position capture"
        );
    }

    /// **EVERY PHASE HAS A DISTINCT SLOT** (the I4b audit).
    ///
    /// `phase`'s constants are hand-written indices into a fixed-size array, and two
    /// of them landing on one slot is the shape that reads as "this phase costs
    /// nothing" while another reads double — the exact under-attribution this
    /// module exists to remove, reintroduced one level down. `PROPAGATE` is the
    /// one deliberate collision (three call sites, one row), and it collides with
    /// itself rather than with a neighbour.
    ///
    /// It does restate the constant list, which is the thing this module moved
    /// its index table *away* from — and it earns that because the list here
    /// carries **no content**: Rust cannot enumerate a module's consts, so the
    /// alternative is no check at all, and a phase added without a line here
    /// fails the length assertion on the next run rather than drifting quietly.
    /// A maintained list that goes red when it is not maintained is the
    /// acceptable kind.
    #[test]
    fn the_phase_indices_are_a_permutation_of_the_slots() {
        let all = [
            phase::CELL_STREAM,
            phase::TERRAIN_STREAM,
            phase::BIOME_SCATTER,
            phase::SKY,
            phase::SOCIETY,
            phase::CROWD,
            phase::TRAFFIC,
            phase::DISPATCH,
            phase::PHYSICS2D_SYNC,
            phase::PHYSICS3D_SYNC,
            phase::WATER,
            phase::INPUT_EVENTS,
            phase::BLUEPRINT_TICK,
            phase::CHARACTER_MOVE,
            phase::VEHICLE,
            phase::GAMEPLAY,
            phase::SOLVER,
            phase::COLLISION_DRAIN,
            phase::WRITE_BACK,
            phase::PROPAGATE,
            phase::DEFORMATION,
            phase::ANIMATION,
            phase::ATTACHMENTS,
            phase::CLOTH_HAIR,
            phase::MODS,
            phase::DESTRUCTION,
            phase::AUDIO,
            phase::CAMERA,
            phase::POSITION_CAPTURE,
        ];
        assert_eq!(
            all.len(),
            STEP_PHASES,
            "the constant list and the slot count disagree"
        );
        let mut seen = [false; STEP_PHASES];
        for (i, p) in all.iter().enumerate() {
            assert!(
                *p < STEP_PHASES,
                "`phase` constant {i} is {p}, past the {STEP_PHASES} slots"
            );
            assert!(
                !seen[*p],
                "two `phase` constants share slot {p} — one row of the breakdown \
                 would read double and another would read zero, which is the \
                 under-attribution this module exists to remove"
            );
            seen[*p] = true;
        }
    }

    #[test]
    fn a_mean_is_an_accumulate_and_a_scale() {
        let mut a = StepProfile::default();
        let mut one = StepProfile::default();
        one.ms[phase::SOLVER] = 4.0;
        let mut two = StepProfile::default();
        two.ms[phase::SOLVER] = 6.0;
        a.accumulate(&one);
        a.accumulate(&two);
        a.scale(0.5);
        assert_eq!(a.ms[phase::SOLVER], 5.0);
        assert_eq!(a.total_ms(), 5.0);
        assert_eq!(a.dearest_first()[0], ("solver", 5.0));
    }
}
