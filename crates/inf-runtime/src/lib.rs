//! The game loop: assembles engine systems; consumed by PIE and the player.
//!
//! Spike D scope: a deliberately small but *real* runtime — a deterministic
//! fixed-step world ([`sim`]), the cooked snapshot the editor hands a player
//! process ([`snapshot`]), and the length-prefixed PIE wire protocol
//! ([`pie`]) spoken between the editor (`inf-editor-core::pie`) and the
//! `inf-player` subprocess over stdin/stdout. Later phases replace the toy
//! world with ECS/renderer/physics behind the same seams; the process model
//! and protocol are the part Spike D locks in.

pub mod pie;
pub mod sim;
pub mod snapshot;

pub use sim::{Actor, ActorState, World, FIXED_DT, TICK_HZ};
pub use snapshot::CookedSnapshot;
