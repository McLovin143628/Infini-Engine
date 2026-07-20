//! A minimal deterministic world: named actors integrating velocity under a
//! fixed timestep, bouncing inside a bounding box. Enough to prove state
//! flows editor → cooked snapshot → player → frame reports, and that two
//! runs of the same snapshot are bit-identical (CI asserts the state hash).
//!
//! World coordinates are `f64` per the engine doctrine (f64 world / f32
//! render); nothing here may ever regress to f32.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::snapshot::CookedSnapshot;

pub const TICK_HZ: u32 = 60;
pub const FIXED_DT: f64 = 1.0 / TICK_HZ as f64;
/// Actors bounce off the walls of a ±`WORLD_BOUND` box (per axis).
pub const WORLD_BOUND: f64 = 100.0;

/// Spawn record and live state of one actor (Spike D keeps them the same
/// shape; later phases split spawn data from runtime components).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actor {
    pub name: String,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

/// Wire-friendly actor transform reported to the editor every frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorState {
    pub name: String,
    pub pos: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct World {
    pub frame: u64,
    pub actors: Vec<Actor>,
}

impl World {
    pub fn from_snapshot(snapshot: &CookedSnapshot) -> Self {
        World {
            frame: 0,
            actors: snapshot.actors.clone(),
        }
    }

    /// Advance exactly one fixed step. Pure f64 math on ordered actors —
    /// bit-deterministic across runs and platforms with IEEE-754 doubles.
    pub fn step(&mut self) {
        self.frame += 1;
        for actor in &mut self.actors {
            for axis in 0..3 {
                actor.pos[axis] += actor.vel[axis] * FIXED_DT;
                if actor.pos[axis] > WORLD_BOUND {
                    actor.pos[axis] = WORLD_BOUND;
                    actor.vel[axis] = -actor.vel[axis];
                } else if actor.pos[axis] < -WORLD_BOUND {
                    actor.pos[axis] = -WORLD_BOUND;
                    actor.vel[axis] = -actor.vel[axis];
                }
            }
        }
    }

    /// xxh3 of the canonical (bincode) encoding of the whole world — the
    /// determinism fingerprint printed by `inf-player --headless` and
    /// reported in every PIE frame message.
    pub fn state_hash(&self) -> u64 {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("world state is always encodable");
        xxh3_64(&bytes)
    }

    pub fn actor_states(&self) -> Vec<ActorState> {
        self.actors
            .iter()
            .map(|a| ActorState {
                name: a.name.clone(),
                pos: a.pos,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_snapshot_same_hash() {
        let snapshot = CookedSnapshot::demo();
        let mut a = World::from_snapshot(&snapshot);
        let mut b = World::from_snapshot(&snapshot);
        for _ in 0..300 {
            a.step();
            b.step();
        }
        assert_eq!(a, b);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn stepping_changes_the_hash() {
        let mut world = World::from_snapshot(&CookedSnapshot::demo());
        let before = world.state_hash();
        world.step();
        assert_ne!(before, world.state_hash());
    }

    #[test]
    fn actors_stay_inside_the_bound() {
        let mut world = World::from_snapshot(&CookedSnapshot::demo());
        for _ in 0..100_000 {
            world.step();
        }
        for actor in &world.actors {
            for axis in 0..3 {
                assert!(
                    actor.pos[axis].abs() <= WORLD_BOUND,
                    "{} escaped on axis {axis}: {}",
                    actor.name,
                    actor.pos[axis]
                );
            }
        }
    }
}
