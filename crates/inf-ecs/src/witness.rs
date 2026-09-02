//! **Who saw what** (wave WPN1) — the record a later wave's police, ambulances
//! and reputations are built on, seeded now because the facts it needs are
//! produced by *this* wave and are gone a step later.
//!
//! # This is a SEED, and saying so is the point
//!
//! Nothing in this engine reads it yet. It exists because the reference frames'
//! own "Crime reported" line
//! (`docs/reference_videos/frames/steal-car/0022`) is a claim about something
//! having been *witnessed*, and the only place that can be known is the step the
//! act happens on: a gunshot's position, who fired it, and who was standing
//! where they could see it are all facts the gameplay step holds for one step and
//! then throws away. Rebuilding them later means storing them; storing them later
//! means the wave that needs them has to change this one. So the record is
//! written now and read by nobody, which is a cost of one `Vec` push per act on
//! the levels that have acts.
//!
//! # It is a RESOURCE, and it is not in the trace
//!
//! [`crate::item::ItemDefs`]' shape exactly: derived at run time, nothing can
//! save it, and **no schema moves** — scene v27 and `ScenePayload` v12 stand.
//!
//! It is deliberately **not** folded into `state_bytes`, and that is a ruling
//! rather than an omission. Folding it would put it after the traffic section
//! (the fold order is frozen and append-only), which is legal — but it would make
//! every gate in the tree compare a record that nothing consumes, and the first
//! wave to change what an observer *is* would move every committed trace hash for
//! a reason that has nothing to do with the simulation. What a two-host gate does
//! instead is compare the log directly, which is what `weapon_hands_gate` already
//! does with the engagement counters.
//!
//! The day something reads it — EMS3's dispatcher — folding it becomes the right
//! call, and the position is named here so it is not a decision anybody has to
//! make twice.

use bevy_ecs::prelude::Resource;
use glam::DVec3;
use uuid::Uuid;

use crate::world::EcsWorld;

/// The most acts the log keeps. A ring, not a leak: a firefight produces one
/// act per shooter per step, and a session that ran for an hour would otherwise
/// grow a record nobody reads without bound.
///
/// Two hundred and fifty-six is about four seconds of a four-way firefight at
/// 600 rpm, which is the window a dispatcher would care about.
pub const MAX_WITNESSED_ACTS: usize = 256;

/// The most observers one act records.
///
/// Eight. It is a **cost** bound and not a design one: each observer is one
/// line-of-sight ray, and an act in a crowded square could otherwise cast one
/// per agent in the radius. The nearest eight are the ones a dispatcher would
/// name anyway.
pub const MAX_OBSERVERS: usize = 8;

/// **What happened.**
///
/// **Append-only.** Nothing serializes this today, so a variant inserted in the
/// middle costs nothing *yet* — and the wave that puts a dispatcher on the wire
/// will discover that it does, which is the `WaterKind` freeze-pin's whole
/// lesson. New kinds go at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActKind {
    /// A round left a barrel.
    Shot,
    /// A body stopped working.
    Killed,
}

/// **One thing somebody did, and who could see it.**
#[derive(Clone, Debug, PartialEq)]
pub struct WitnessedAct {
    /// What happened.
    pub kind: ActKind,
    /// Who did it.
    pub actor: Uuid,
    /// Where, world metres — the muzzle for a shot, the body for a death.
    pub at: DVec3,
    /// The fixed step it happened on, so a dispatcher can age it without a
    /// clock of its own.
    pub step: u64,
    /// Who could see it, in `Guid` order, at most [`MAX_OBSERVERS`].
    pub observers: Vec<Uuid>,
    /// **A digest of what the actor looked like**, or `0` for one the crowd
    /// cannot describe.
    ///
    /// The reference's own "they have your description" line
    /// (`frames/police-bike/0030`) is a claim about *this*: what a witness can
    /// say is not who you are but what you were wearing. It is the crowd's own
    /// derived look draw ([`crate::crowd::agent_look`]'s seed), which is cheap,
    /// deterministic and already a pure function of the guid — so recording it
    /// costs one hash and is the same number on both hosts.
    ///
    /// A hero has no crowd look, and answers the same draw against its own guid:
    /// the point is that two different actors describe differently, not that the
    /// number means anything on its own yet.
    pub actor_look: u64,
    /// **The vehicle the actor was in**, if any — the other half of a
    /// description, and the one the reference's wanted system keys on.
    pub actor_vehicle: Option<Uuid>,
}

/// **The log** — a bounded ring of what has been seen.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct WitnessLog {
    /// Oldest first.
    acts: Vec<WitnessedAct>,
    /// How many fell off the front. Non-zero means [`acts`](Self::acts) is a
    /// tail — `BoundedLog::dropped`'s own contract, spelled here because Ring 0
    /// may not depend on `inf-core`'s.
    dropped: u64,
}

impl WitnessLog {
    /// The retained acts, oldest first.
    pub fn acts(&self) -> &[WitnessedAct] {
        &self.acts
    }

    /// How many acts fell off the front of the ring.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Append one, evicting the oldest when the ceiling is reached.
    pub fn push(&mut self, act: WitnessedAct) {
        if self.acts.len() >= MAX_WITNESSED_ACTS {
            self.acts.remove(0);
            self.dropped = self.dropped.saturating_add(1);
        }
        self.acts.push(act);
    }
}

/// **Record one act.** The one door, so a second producer cannot invent a second
/// shape of record.
pub fn record_act(world: &mut EcsWorld, act: WitnessedAct) {
    let mut log = world
        .world_mut()
        .remove_resource::<WitnessLog>()
        .unwrap_or_default();
    log.push(act);
    world.world_mut().insert_resource(log);
}

/// What has been seen, or an empty slice on a level where nothing has happened.
pub fn witnessed(world: &EcsWorld) -> &[WitnessedAct] {
    world
        .world()
        .get_resource::<WitnessLog>()
        .map(|l| l.acts())
        .unwrap_or(&[])
}

/// How many acts have fallen off the front of the log.
pub fn witness_dropped(world: &EcsWorld) -> u64 {
    world
        .world()
        .get_resource::<WitnessLog>()
        .map(|l| l.dropped())
        .unwrap_or(0)
}

/// **Forget everything that was seen** — `clear_crowd`'s twin, for its reason:
/// an editor Simulate session must leave nothing behind in the author's
/// document.
pub fn clear_witness(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<WitnessLog>();
}

/// **What an actor looks like, as one number** — the description half of an act.
///
/// The crowd's own look draw, asked of any guid: a pure function, portable by
/// construction (integer arithmetic), and already the number the crowd tints its
/// own agents with. A hero has no crowd record and still answers, which is what
/// makes "two different actors describe differently" true without a second table.
pub fn look_digest(actor: Uuid) -> u64 {
    crate::crowd::agent_rand(actor, 0, crate::crowd::SALT_LOOK)
}

/// **Which vehicle this actor is in**, or `None`.
///
/// Read off the movement runtime's own seat, which is where the link between a
/// character and the chassis it is riding lives — `d3::camera`'s own sentence.
pub fn actor_vehicle(world: &EcsWorld, actor: Uuid) -> Option<Uuid> {
    let e = world.entity_of(actor)?;
    let cm = world
        .world()
        .get::<crate::components::CharacterMovement>(e)?;
    cm.runtime
        .seat
        .is_seated()
        .then_some(cm.runtime.seat.vehicle)
}

/// **Every crowd agent within `radius_m` of `at`**, nearest first, at most
/// [`MAX_OBSERVERS`] — the candidate observers, before any line of sight.
///
/// Off the population's own records rather than a component query, for
/// [`crate::crowd::blocked_agents`]'s reason: `O(agents)` rather than
/// `O(entities)` over a furnished town, and allocation-free on a level with no
/// crowd. **Every tier**, including `Dormant`: an agent with no entity is still a
/// person standing where they are standing, and refusing them would make what a
/// street saw a function of where the player is looking.
pub fn candidates_near(world: &EcsWorld, at: DVec3, radius_m: f64) -> Vec<(Uuid, DVec3)> {
    let Some(pop) = world
        .world()
        .get_resource::<crate::crowd::CrowdPopulationRes>()
    else {
        return Vec::new();
    };
    let mut near: Vec<(f64, Uuid, DVec3)> = Vec::new();
    for (guid, rec) in &pop.records {
        let d = (rec.last - at).length();
        if !d.is_finite() || d > radius_m {
            continue;
        }
        near.push((d, *guid, rec.last));
    }
    // Nearest first, ties on the `Guid` — the `BTreeMap` walk above is already
    // `Guid`-ordered and `sort_by` is stable, so two agents at one distance keep
    // that order on both hosts.
    // **Nearest first to CHOOSE, `Guid` order to RECORD**, and the two are
    // different questions. The first is the cost bound — an act in a crowded
    // square must cast a bounded number of rays, and the nearest few are the
    // ones a dispatcher would name anyway. The second is the record: "who saw
    // it" is a set, and a set written in an order that depends on where
    // somebody happened to stand is a set two readers can compare wrongly.
    //
    // `sort_by` is stable and the walk above is already `Guid`-ordered, so two
    // agents at one distance keep that order on both hosts.
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    near.truncate(MAX_OBSERVERS);
    let mut out: Vec<(Uuid, DVec3)> = near.into_iter().map(|(_, g, at)| (g, at)).collect();
    out.sort_by_key(|(g, _)| *g);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crowd::{CrowdArchetype, CrowdRecord};
    use std::collections::BTreeMap;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// **The log is a ring and it says when it has dropped something**, and an
    /// empty world costs nothing.
    #[test]
    fn the_witness_log_is_bounded_and_honest_about_it() {
        let mut w = EcsWorld::new();
        assert!(witnessed(&w).is_empty());
        assert_eq!(witness_dropped(&w), 0);
        let act = |n: u64| WitnessedAct {
            kind: ActKind::Shot,
            actor: guid(1),
            at: DVec3::new(n as f64, 0.0, 0.0),
            step: n,
            observers: Vec::new(),
            actor_look: 0,
            actor_vehicle: None,
        };
        for n in 0..MAX_WITNESSED_ACTS as u64 {
            record_act(&mut w, act(n));
        }
        assert_eq!(witnessed(&w).len(), MAX_WITNESSED_ACTS);
        assert_eq!(witness_dropped(&w), 0, "the ring evicted early");
        record_act(&mut w, act(9999));
        assert_eq!(witnessed(&w).len(), MAX_WITNESSED_ACTS);
        assert_eq!(witness_dropped(&w), 1);
        assert_eq!(
            witnessed(&w).last().expect("an act").step,
            9999,
            "the newest act fell off instead of the oldest"
        );
        assert_eq!(
            witnessed(&w).first().expect("an act").step,
            1,
            "the ring dropped the wrong end"
        );
        clear_witness(&mut w);
        assert!(witnessed(&w).is_empty());
        assert_eq!(witness_dropped(&w), 0);
    }

    /// **The observers are the nearest few, in `Guid` order, at every tier.**
    ///
    /// The two claims a distance-ordered list would fail: the *set* is the
    /// nearest [`MAX_OBSERVERS`] (so a crowded square costs a bounded number of
    /// rays), and the *record* is `Guid`-ordered (so what a street saw does not
    /// depend on where anybody happened to stand).
    #[test]
    fn the_observers_are_the_nearest_few_and_are_recorded_in_guid_order() {
        let mut w = EcsWorld::new();
        assert!(candidates_near(&w, DVec3::ZERO, 50.0).is_empty());
        let a = CrowdArchetype::humanoid(None, None, None);
        let mut records = BTreeMap::new();
        // Twenty agents, DESCENDING guid with ASCENDING distance — so a list
        // that came out in guid order and one that came out in distance order
        // are exactly reversed and cannot be confused.
        for i in 0..20u128 {
            records.insert(
                guid(100 - i),
                CrowdRecord::standing(a, DVec3::new(1.0 + i as f64, 0.0, 0.0)),
            );
        }
        crate::crowd::set_population(&mut w, records);
        let near = candidates_near(&w, DVec3::ZERO, 6.5);
        println!(
            "20 agents, 6.5 m radius -> {} observer(s): {:?}",
            near.len(),
            near.iter().map(|(g, _)| g.as_u128()).collect::<Vec<_>>()
        );
        // Six are inside 6.5 m (at 1..6 m) and that is fewer than the cap, so
        // the radius is what bound this one.
        assert_eq!(near.len(), 6);
        let ids: Vec<u128> = near.iter().map(|(g, _)| g.as_u128()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "the observers are not in `Guid` order");
        // …and with a radius that admits everybody, the CAP is what bounds it,
        // and the ones kept are the NEAREST — which in this fixture are the
        // highest guids, so a naive "first eight in guid order" fails here.
        let all = candidates_near(&w, DVec3::ZERO, 1000.0);
        assert_eq!(all.len(), MAX_OBSERVERS);
        let ids: Vec<u128> = all.iter().map(|(g, _)| g.as_u128()).collect();
        println!("the same crowd at 1000 m -> {ids:?}");
        assert_eq!(
            ids,
            (93..=100).collect::<Vec<u128>>(),
            "the cap kept the wrong agents — these are the eight FURTHEST"
        );
    }

    /// **A description is a pure function of who you are**, and two actors
    /// describe differently.
    #[test]
    fn a_look_digest_is_derived_and_discriminates() {
        assert_eq!(look_digest(guid(7)), look_digest(guid(7)));
        assert_ne!(look_digest(guid(7)), look_digest(guid(8)));
        // A world with nobody in it still answers, which is what lets a hero be
        // described without a crowd record.
        let w = EcsWorld::new();
        assert_eq!(actor_vehicle(&w, guid(7)), None);
    }
}
