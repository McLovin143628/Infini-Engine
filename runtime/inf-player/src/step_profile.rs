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
pub const STEP_PHASES: usize = 23;

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
    "sky",
    "physics2d sync",
    "physics3d sync",
    "water forces",
    "input events",
    "blueprint tick",
    "character move",
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
    /// P17.1 time of day + P17.4 the weather blend.
    pub const SKY: usize = 2;
    /// The 2D bridge's ECS → rapier2d walk.
    pub const PHYSICS2D_SYNC: usize = 3;
    /// The 3D bridge's walk — the I3 collider band's gather lives here, and so
    /// does the P22.3 fracture follow that precedes it.
    pub const PHYSICS3D_SYNC: usize = 4;
    /// P20.2 buoyancy + hydrodynamic drag.
    pub const WATER: usize = 5;
    /// Wave 3 input edges and the dispatches they queue.
    pub const INPUT_EVENTS: usize = 6;
    /// The Tick pass over every actor, and the dispatches it queues.
    pub const BLUEPRINT_TICK: usize = 7;
    /// P29.3 — the one Ring-0 movement step, plus the intent that feeds it.
    pub const CHARACTER_MOVE: usize = 8;
    /// I6 — doors, weapons and the health they spend, plus the host's own drain
    /// of the energy they owe the P22 damage door.
    pub const GAMEPLAY: usize = 9;
    /// rapier2d + rapier3d.
    pub const SOLVER: usize = 10;
    /// Wave 3 contacts and overlaps, and the dispatches they queue.
    pub const COLLISION_DRAIN: usize = 11;
    /// rapier → ECS.
    pub const WRITE_BACK: usize = 12;
    /// The transform + visibility DFS. **Three call sites, gathered.**
    pub const PROPAGATE: usize = 13;
    /// P22.1 — the ground remembers what stood on it.
    pub const DEFORMATION: usize = 14;
    /// Play-heads, state machines and root motion.
    pub const ANIMATION: usize = 15;
    /// P11.3 sockets.
    pub const ATTACHMENTS: usize = 16;
    /// P24.4 garments and hair.
    pub const CLOTH_HAIR: usize = 17;
    /// P14.5 WASM mods — which propagate internally, so their propagate is here
    /// rather than in [`PROPAGATE`].
    pub const MODS: usize = 18;
    /// P22.3 fracture write-back, the structural solve and the debris budget.
    pub const DESTRUCTION: usize = 19;
    /// P12.3 — the audio command queue.
    pub const AUDIO: usize = 20;
    /// P29.6 — the locomotion camera.
    pub const CAMERA: usize = 21;
    /// The interpolation history roll and the rising-edge clear.
    pub const POSITION_CAPTURE: usize = 22;
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
    #[inline]
    pub(crate) fn mark(&mut self, phase: usize) {
        if let Some(at) = self.at.as_mut() {
            let now = Instant::now();
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
    #[test]
    fn a_phase_marked_twice_sums_rather_than_replaces() {
        let mut c = StepClock::start(true);
        // Two separately-marked stretches charged to one phase.
        let spin = |us: u64| {
            let t = Instant::now();
            while t.elapsed().as_micros() < us as u128 {
                std::hint::spin_loop();
            }
        };
        spin(2_000);
        c.mark(phase::PROPAGATE);
        let first = c.ms[phase::PROPAGATE];
        spin(2_000);
        c.mark(phase::PROPAGATE);
        let both = c.ms[phase::PROPAGATE];
        assert!(
            both > first * 1.5,
            "two 2 ms stretches charged to one phase read {both:.3} ms against \
             {first:.3} for the first alone — the mark is replacing, not summing"
        );
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
        assert_eq!(STEP_PHASE_NAMES[phase::PHYSICS3D_SYNC], "physics3d sync");
        assert_eq!(STEP_PHASE_NAMES[phase::PROPAGATE], "propagate");
        assert_eq!(STEP_PHASE_NAMES[phase::SOLVER], "solver");
        assert_eq!(STEP_PHASE_NAMES[phase::CAMERA], "camera");
        assert_eq!(
            STEP_PHASE_NAMES[phase::POSITION_CAPTURE],
            "position capture"
        );
    }

    /// **EVERY PHASE HAS A DISTINCT SLOT** (the I4b audit).
    ///
    /// `phase`'s constants are hand-written indices into a 22-slot array, and two
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
            phase::SKY,
            phase::PHYSICS2D_SYNC,
            phase::PHYSICS3D_SYNC,
            phase::WATER,
            phase::INPUT_EVENTS,
            phase::BLUEPRINT_TICK,
            phase::CHARACTER_MOVE,
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
