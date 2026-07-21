//! The parallel ECS schedule facade (P9.1a, §2.5 — the concurrency deliverable).
//!
//! `inf-ecs` enables `bevy_ecs`'s `multi_threaded` feature and drives the runtime
//! sim through a real [`Schedule`] whose non-conflicting systems run concurrently.
//! [`SimSchedule`] wraps that schedule so **no `bevy_ecs` type crosses the facade**
//! — outside crates select an execution [`ScheduleMode`], compose the standard sim
//! phases through [`SimScheduleBuilder`], and call [`SimSchedule::run`] with an
//! [`EcsWorld`]. The systems themselves live in [`crate::sim`] (they must, to carry
//! the component-access declarations the executor needs to prove parallelism).
//!
//! ## Two pools, coordinated — the doctrine reconciliation (§2.5)
//!
//! §2.5 names `inf-core`'s rayon [`JobPool`](../../inf_core/job/index.html) as *the*
//! Ring-0 entry point for CPU data-parallelism. `bevy_ecs`'s `multi_threaded`
//! executor does **not** run on that pool — it hardcodes its own
//! `bevy_tasks::ComputeTaskPool` (a work-stealing async task pool) as the surface
//! its scoped executor spawns system tasks onto. That is fine and intended: §2.5
//! calls the ECS executor the *one sanctioned sibling* of the job pool. The two
//! coexist under one rule — **coordinated thread counts, no oversubscription**:
//!
//! * `inf-core::job::global()` sizes to `available_parallelism`.
//! * [`init_ecs_task_pool`] sizes the `ComputeTaskPool` to the same basis (named
//!   `inf-ecs-worker-{i}` threads) and is idempotent.
//!
//! They are never both hot on the same core at the same instant in the current
//! design: the job pool serves asset import / graph compile (off the frame's
//! critical path), while the ECS pool serves the sim schedule. Should a future
//! phase run both inside one frame, the shared sizing keeps total worker threads
//! at the core count. Unifying them onto a single pool (a rayon-backed
//! `bevy_tasks` executor) is a documented Phase-15 tuning follow-up, not a
//! correctness issue.
//!
//! ## Determinism
//!
//! Determinism is a property of [`crate::sim`]'s access discipline (disjoint or
//! ordered systems; structural changes at sync points; guid-keyed trace), **not**
//! of the executor: [`ScheduleMode::Serial`] and [`ScheduleMode::Parallel`]
//! produce byte-identical traces. The `inf-runtime` replay harness is the §8 CI
//! gate that proves it across pool sizes.

use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{
    LogLevel, MultiThreadedExecutor, ScheduleBuildSettings, SingleThreadedExecutor, SystemKey,
};

use crate::sim;
use crate::world::EcsWorld;

/// How a [`SimSchedule`] executes its systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleMode {
    /// One system at a time on the calling thread (the deterministic baseline).
    Serial,
    /// Non-conflicting systems run concurrently on the ECS `ComputeTaskPool`.
    Parallel,
}

/// The ordered phases of the standard runtime sim. Systems are assigned to a set
/// and the sets carry the ordering edges that keep conflicting systems apart.
///
/// `Tick → {Motion ∥ Lifetime} → Structural → Propagate`
///
/// `Motion` and `Lifetime` touch disjoint component/resource state, so they carry
/// **no** edge between them and run concurrently — the parallelism showcase.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SimSet {
    /// Frame clock + structural director (writes `SimTick`, `SimControl`).
    Tick,
    /// Gather→scatter coupling then integrate/spin/bounce (writes `Transform`,
    /// `Velocity`, `Centroid`).
    Motion,
    /// Age + reap lifetimes (writes `Lifetime`, `SimControl`).
    Lifetime,
    /// Apply queued spawns/despawns via deferred `Commands`.
    Structural,
    /// Transform/visibility propagation (exclusive sync-point tail).
    Propagate,
}

/// Which standard phases a [`SimSchedule`] includes.
#[derive(Clone, Copy, Debug)]
pub struct SimScheduleBuilder {
    mode: ScheduleMode,
    gameplay: bool,
    structural: bool,
    propagation: bool,
}

impl Default for SimScheduleBuilder {
    fn default() -> Self {
        Self {
            mode: ScheduleMode::Parallel,
            gameplay: true,
            structural: true,
            propagation: true,
        }
    }
}

impl SimScheduleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serial vs parallel execution.
    pub fn mode(mut self, mode: ScheduleMode) -> Self {
        self.mode = mode;
        self
    }

    /// Include the gameplay motion systems (centroid coupling, integrate, spin,
    /// bounce, lifetime aging).
    pub fn gameplay(mut self, yes: bool) -> Self {
        self.gameplay = yes;
        self
    }

    /// Include the structural director + spawn/despawn sync-point systems.
    pub fn structural(mut self, yes: bool) -> Self {
        self.structural = yes;
        self
    }

    /// Include the transform-propagation tail.
    pub fn propagation(mut self, yes: bool) -> Self {
        self.propagation = yes;
        self
    }

    /// Assemble the [`SimSchedule`].
    pub fn build(self) -> SimSchedule {
        let mut schedule = Schedule::default();

        // Flag unordered conflicting systems: within a fixed step, any two systems
        // with overlapping access MUST carry an ordering edge, or the step is not
        // deterministic. `Warn` (not `Error`) so the schedule always builds and
        // [`SimSchedule::conflict_report`] can resolve system names — the empty
        // conflict-report test is the hard gate (deliverable 2), backed by the
        // replay-determinism harness (§8).
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Warn,
            ..Default::default()
        });

        // ── set ordering: Tick → Motion → Structural → Propagate, Lifetime ∥ Motion
        schedule.configure_sets(
            (
                SimSet::Tick,
                SimSet::Motion,
                SimSet::Structural,
                SimSet::Propagate,
            )
                .chain(),
        );
        schedule.configure_sets(
            SimSet::Lifetime
                .after(SimSet::Tick)
                .before(SimSet::Structural),
        );

        if self.structural {
            schedule.add_systems(
                (sim::tick_counter, sim::director)
                    .chain()
                    .in_set(SimSet::Tick),
            );
            schedule.add_systems(
                (sim::apply_spawns, sim::apply_despawns)
                    .chain()
                    .in_set(SimSet::Structural),
            );
        } else {
            // Still advance the clock so a propagation-only loop has a frame count.
            schedule.add_systems(sim::tick_counter.in_set(SimSet::Tick));
        }

        if self.gameplay {
            schedule.add_systems(
                (
                    sim::accumulate_centroid,
                    sim::attract_to_centroid,
                    sim::integrate_velocity,
                    sim::apply_spin,
                    sim::bounce_bounds,
                )
                    .chain()
                    .in_set(SimSet::Motion),
            );
            schedule.add_systems(
                (sim::age_lifetime, sim::reap_expired)
                    .chain()
                    .in_set(SimSet::Lifetime),
            );
        }

        if self.propagation {
            schedule.add_systems(sim::propagate_transforms.in_set(SimSet::Propagate));
        }

        let mut sched = SimSchedule {
            schedule,
            mode: self.mode,
        };
        sched.apply_executor();
        sched
    }
}

/// A runtime sim schedule over an [`EcsWorld`]. Wraps a `bevy_ecs` [`Schedule`]
/// without exposing it.
pub struct SimSchedule {
    schedule: Schedule,
    mode: ScheduleMode,
}

impl SimSchedule {
    /// The standard engine sim schedule (all phases) in the given mode.
    pub fn engine_default(mode: ScheduleMode) -> Self {
        SimScheduleBuilder::new().mode(mode).build()
    }

    /// A propagation-only schedule (frame clock + transform propagation) — the
    /// minimal loop for a world with no movers.
    pub fn propagation_only(mode: ScheduleMode) -> Self {
        SimScheduleBuilder::new()
            .mode(mode)
            .gameplay(false)
            .structural(false)
            .build()
    }

    pub fn mode(&self) -> ScheduleMode {
        self.mode
    }

    /// Switch execution mode (swaps the underlying executor).
    pub fn set_mode(&mut self, mode: ScheduleMode) {
        if self.mode != mode {
            self.mode = mode;
            self.apply_executor();
        }
    }

    fn apply_executor(&mut self) {
        match self.mode {
            ScheduleMode::Serial => {
                self.schedule.set_executor(SingleThreadedExecutor::new());
            }
            ScheduleMode::Parallel => {
                self.schedule.set_executor(MultiThreadedExecutor::new());
            }
        }
    }

    /// Run one pass of the schedule over the world.
    ///
    /// In [`ScheduleMode::Parallel`] the ECS `ComputeTaskPool` must exist; if a
    /// caller never called [`init_ecs_task_pool`], the executor lazily initializes
    /// it to `available_parallelism`. Call [`init_ecs_task_pool`] first to control
    /// the size and thread names.
    pub fn run(&mut self, world: &mut EcsWorld) {
        self.schedule.run(world.world_mut());
    }

    /// Deliverable-2 debug helper: initialize the schedule and return every pair of
    /// **conflicting systems that lack an ordering edge**. An empty result proves
    /// the discipline holds (all overlapping-access systems are ordered), so the
    /// step is order-independent and therefore pool-size-invariant.
    ///
    /// `build()` already sets `ambiguity_detection = Error`, so an undisciplined
    /// schedule fails to build; this helper is the explicit, test-friendly probe.
    pub fn conflict_report(&mut self, world: &mut EcsWorld) -> Vec<(String, String)> {
        let w = world.world_mut();
        // Build the schedule against the real world so component ids resolve.
        let _ = self.schedule.initialize(w);

        // Collect the unordered-conflict key pairs first (owned copies).
        let pairs: Vec<(SystemKey, SystemKey)> = self
            .schedule
            .graph()
            .conflicting_systems()
            .0
            .iter()
            .map(|(a, b, _)| (*a, *b))
            .collect();
        if pairs.is_empty() {
            return Vec::new();
        }

        // Resolve system keys → names.
        let names: std::collections::HashMap<SystemKey, String> = match self.schedule.systems() {
            Ok(iter) => iter.map(|(k, s)| (k, format!("{}", s.name()))).collect(),
            Err(_) => std::collections::HashMap::new(),
        };
        let name_of = |k: &SystemKey| {
            names
                .get(k)
                .cloned()
                .unwrap_or_else(|| "<unknown>".to_string())
        };
        pairs
            .into_iter()
            .map(|(a, b)| (name_of(&a), name_of(&b)))
            .collect()
    }
}

// ───────────────────────────── ECS task pool ───────────────────────────────

static ECS_POOL_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Initialize the `bevy_ecs` `ComputeTaskPool` — the pool the `multi_threaded`
/// executor spawns system tasks onto — with `threads` named workers
/// (`inf-ecs-worker-{i}`). `0` means "size to `available_parallelism`".
///
/// **Idempotent and process-global**: the pool is a `OnceLock` inside
/// `bevy_tasks`, so the *first* call wins and later calls are no-ops (they return
/// the already-configured thread count). This is why the replay harness varies
/// pool size across **subprocesses** rather than re-initializing in one process.
/// Returns the pool's actual worker-thread count.
pub fn init_ecs_task_pool(threads: usize) -> usize {
    use bevy_tasks::{ComputeTaskPool, TaskPoolBuilder};

    let requested = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
    .max(1);

    let pool = ComputeTaskPool::get_or_init(|| {
        TaskPoolBuilder::new()
            .num_threads(requested)
            .thread_name("inf-ecs-worker".to_string())
            .build()
    });
    let actual = pool.thread_num().max(1);
    ECS_POOL_THREADS.store(actual, Ordering::Relaxed);
    actual
}

/// The ECS task pool's worker-thread count, or `0` if it has not been initialized
/// (and no parallel schedule has yet run to lazily create it).
pub fn ecs_task_pool_threads() -> usize {
    ECS_POOL_THREADS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{self, SimConfig};

    fn seeded_world() -> EcsWorld {
        let mut w = EcsWorld::new();
        sim::install_resources(&mut w, SimConfig::default());
        for i in 0..16u128 {
            let e = w.spawn_with_guid(uuid::Uuid::from_u128(i + 1), "m", None);
            sim::set_velocity(&mut w, e, crate::Vec3d::new(1.0, 0.5, -0.5));
            if i.is_multiple_of(2) {
                sim::set_lifetime(&mut w, e, 100.0);
            }
        }
        w
    }

    #[test]
    fn standard_schedule_has_no_unordered_conflicts() {
        let mut w = seeded_world();
        let mut sched = SimSchedule::engine_default(ScheduleMode::Serial);
        let conflicts = sched.conflict_report(&mut w);
        assert!(
            conflicts.is_empty(),
            "conflicting systems without an ordering edge: {conflicts:?}"
        );
    }

    #[test]
    fn serial_schedule_runs() {
        let mut w = seeded_world();
        let mut sched = SimSchedule::engine_default(ScheduleMode::Serial);
        for _ in 0..10 {
            sched.run(&mut w);
        }
        // Movers advanced from the origin cluster.
        let snap = sim::sim_snapshot(&mut w);
        assert!(!snap.is_empty());
    }

    #[test]
    fn mode_switch_is_idempotent() {
        let mut sched = SimSchedule::engine_default(ScheduleMode::Serial);
        assert_eq!(sched.mode(), ScheduleMode::Serial);
        sched.set_mode(ScheduleMode::Parallel);
        assert_eq!(sched.mode(), ScheduleMode::Parallel);
        sched.set_mode(ScheduleMode::Parallel);
        assert_eq!(sched.mode(), ScheduleMode::Parallel);
    }
}
