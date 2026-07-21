//! The game loop: assembles engine systems; consumed by PIE and the player.
//!
//! Spike D scope: a deliberately small but *real* runtime — a deterministic
//! fixed-step world ([`sim`]), the cooked snapshot the editor hands a player
//! process ([`snapshot`]), and the length-prefixed PIE wire protocol
//! ([`pie`]) spoken between the editor (`inf-editor-core::pie`) and the
//! `inf-player` subprocess over stdin/stdout. Later phases replace the toy
//! world with ECS/renderer/physics behind the same seams; the process model
//! and protocol are the part Spike D locks in.
//!
//! **P9.1a (runtime assembly):** the real fixed-step game loop lands in
//! [`game_loop::GameLoop`] — a headless loop owning an `inf-ecs` world advanced by
//! a **parallel** [`SimSchedule`] on a fixed timestep ([`step::FixedStep`]) with
//! interpolation `alpha` for render. [`replay`] is the §8 concurrency-determinism
//! harness: it proves the parallel schedule's trace is byte-identical to the
//! serial baseline across ECS task-pool sizes. The Spike-D toy [`sim::World`]
//! remains until the PIE/player seams are rehosted on `GameLoop` (P9.3/P9.4).

pub mod game_loop;
pub mod pie;
pub mod replay;
pub mod replication;
pub mod sim;
pub mod snapshot;
pub mod step;

pub use game_loop::GameLoop;
pub use replication::{apply_snapshot, snapshot_world, ReplicationSource};
pub use sim::{Actor, ActorState, World, FIXED_DT, TICK_HZ};
pub use snapshot::CookedSnapshot;
pub use step::FixedStep;

// Re-export the ECS scheduling surface the runtime is built on, so downstream
// crates (player, PIE host) can pick a mode without depending on inf-ecs directly.
pub use inf_ecs::{ScheduleMode, SimConfig, SimSchedule};
