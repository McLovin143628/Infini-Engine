//! **Who saw what** (wave WPN1) — the record a later wave's police, ambulances
//! and reputations are built on, seeded now because the facts it needs are
//! produced by *this* wave and are gone a step later.
//!
//! # It WAS a seed, and two waves later it is load-bearing
//!
//! WPN1 wrote it with no reader at all. EMS2's dispatcher opens a crime incident
//! off it, and EMS3's [`crate::crime`] builds a criminal profile out of it — so
//! every sentence below about *why* it was written before anybody needed it now
//! reads as the record of a bet that paid. It exists because the reference
//! frames' own "Crime reported" line
//! (`docs/reference_videos/frames/steal-car/0022`) is a claim about something
//! having been *witnessed*, and the only place that can be known is the step the
//! act happens on: a gunshot's position, who fired it, and who was standing
//! where they could see it are all facts the gameplay step holds for one step and
//! then throws away. Rebuilding them later means storing them; storing them later
//! means the wave that needs them has to change this one. So the record is
//! written now and read by nobody, which is a cost of one `Vec` push per act on
//! the levels that have acts.
//!
//! # It is a RESOURCE, and since wave EMS3 it IS in the trace
//!
//! [`crate::item::ItemDefs`]' shape exactly: derived at run time, nothing can
//! save it, and **no schema moves** — scene v27 and `ScenePayload` v12 stand.
//!
//! WPN1 deliberately kept it **out** of `state_bytes`, and wrote the condition
//! for putting it in: *"The day something reads it — EMS3's dispatcher —
//! folding it becomes the right call, and the position is named here so it is
//! not a decision anybody has to make twice."* That day arrived: EMS2's
//! dispatcher opens crime incidents off this log and EMS3's profiles are built
//! from it, so what an observer is now decides where police cars drive.
//!
//! So [`witness_state_bytes`] is folded, **after the dispatcher** — the frozen
//! append-only position the fold's own comments describe — and
//! `projector_mirror`'s `SECTIONS` allowlist was extended in the same commit,
//! which is the discipline the traffic section had to learn the hard way (it
//! went two waves unpinned). A level where nothing has happened produces an
//! empty vec, so every trace committed before this wave is byte-identical.

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
    /// **Somebody was pulled out of their own car** (wave EMS3) — VEH2b's
    /// `Carjack::Ejected`, which until this wave was a thing that happened in
    /// front of a street full of people and left no record at all.
    ///
    /// Appended, because the freeze-pin this enum's own doc predicted came true
    /// one wave later: [`crate::crime::profile_state_bytes`] folds
    /// [`ActKind::as_u8`] into a trace, so a variant inserted in the middle now
    /// costs every committed hash in the tree.
    Carjack,
    /// **A blow landed** — a swing or a kick that connected. The quiet crime,
    /// and the reason it is separate from [`Shot`](Self::Shot): a fist makes no
    /// noise, so the only people who know about it are the ones who *saw* it,
    /// which is exactly what an observer list is for.
    Assault,
}

impl ActKind {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            ActKind::Shot => "shot",
            ActKind::Killed => "killed",
            ActKind::Carjack => "carjack",
            ActKind::Assault => "assault",
        }
    }

    /// The byte this kind folds into a trace. **Frozen, append-only** on
    /// [`crate::dispatch::UnitKind::as_u8`]'s terms.
    pub fn as_u8(self) -> u8 {
        match self {
            ActKind::Shot => 0,
            ActKind::Killed => 1,
            ActKind::Carjack => 2,
            ActKind::Assault => 3,
        }
    }

    /// **How much heat this act adds to whoever did it** — the severity ladder's
    /// input, in the units [`crate::crime::Profile::heat`] counts.
    ///
    /// A death is worth three, a shot two, and a carjack or a punch one each.
    /// The shape is the point rather than the exact numbers: one petty act does
    /// not bring a tactical van, and three of them do
    /// ([`crate::crime::Response::for_heat`]).
    pub fn heat(self) -> u32 {
        match self {
            ActKind::Killed => 3,
            ActKind::Shot => 2,
            ActKind::Carjack | ActKind::Assault => 1,
        }
    }
}

/// **One thing somebody did, and who could see it.**
#[derive(Clone, Debug, PartialEq)]
pub struct WitnessedAct {
    /// What happened.
    pub kind: ActKind,
    /// **Who did it** — and it is *who did it*, not who it happened to.
    ///
    /// Stated twice because wave EMS3's audit found the difference costing a
    /// wave: an [`ActKind::Killed`] act is raised off the list of bodies that
    /// stopped working, so the obvious guid to hand is the **victim's** — and
    /// [`crate::crime::report_act`] keys a criminal profile on this field. The
    /// producer resolves the killer out of the same step's hits; see
    /// `d3::gameplay::step_witness`.
    ///
    /// `Uuid::nil()` when the world genuinely cannot name anybody. A call is
    /// still worth opening at the place it happened, and
    /// [`crate::crime::report_act`] refuses to put it on a file.
    pub actor: Uuid,
    /// Where, world metres — the muzzle for a shot, the body for a death.
    pub at: DVec3,
    /// The fixed step it happened on, so a dispatcher can age it without a
    /// clock of its own.
    pub step: u64,
    /// Who could see it, in `Guid` order, at most [`MAX_OBSERVERS`].
    pub observers: Vec<Uuid>,
    /// **A digest of what the actor was WEARING**, or `0` for one the crowd
    /// cannot describe.
    ///
    /// The reference's own "they have your description" line
    /// (`frames/police-bike/0030`) is a claim about *this*: what a witness can
    /// say is not who you are but what you were wearing.
    ///
    /// # It was an identity hash, and replacing it is wave EMS3's headline
    ///
    /// WPN1 wrote this as `agent_rand(actor, 0, SALT_LOOK)` — the actor's guid,
    /// mixed — and said so plainly: *"two different actors describe
    /// differently"*. That is a true sentence about an identity and a false one
    /// about a description. Under it, two people in the same coat had different
    /// numbers and one person who changed coats kept theirs, so a police force
    /// reading it would have been tracking **who you are** through a channel
    /// that looked like *what you look like* — and no wardrobe in the world
    /// could have defeated it.
    ///
    /// It is now [`look_digest`], which is [`crate::crowd::Appearance::digest`]
    /// of what the actor has on: a value, not an identity. Two people dressed
    /// alike collide **on purpose**, and changing at a wardrobe changes the
    /// number. The collision is a property of the channel and not yet of a
    /// street — see [`crate::crowd::Appearance`] and
    /// `inf_physics::d3::crime::look` for who actually gets scored.
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
    /// **Acts raised earlier in this same fixed step**, waiting for somebody to
    /// work out who could see them — see [`raise_act`].
    ///
    /// Drained to empty by the gameplay phase's witness pass every step, and
    /// therefore **not** folded into [`witness_state_bytes`]: it is a within-step
    /// hand-off and never survives a step boundary, so folding it would put a
    /// value that is always zero into every committed hash.
    pending: Vec<(ActKind, Uuid, DVec3)>,
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

/// **Something happened, and the witness pass has not run yet** (wave EMS3) —
/// the one door for an act raised OUTSIDE the gameplay phase.
///
/// # Why a queue and not a second `record_act`
///
/// A carjack is applied in the `character move` phase, three phases before the
/// gameplay step that works out who could see anything. A producer there has
/// the *fact* (somebody was pulled out of a car, here, by this person) and none
/// of the *observation* (a physics bridge, a candidate walk, a line-of-sight
/// ray, and the step's own ray budget). Recording it on the spot would have
/// been a second answer to "who saw it" — one with no line of sight at all,
/// because the collision world is not reachable from a movement rule — and
/// `MAX_ACTS_PER_STEP`'s ray bound would have had a hole in it the size of
/// every non-gunfire crime in the engine.
///
/// So the producer states the fact and the witness pass observes it, through
/// the same [`candidates_near`] + ray both other kinds of act go through.
/// Drained every step, and deliberately **not** folded into
/// [`witness_state_bytes`]: it is a within-step hand-off that never survives a
/// step boundary, so folding it would put a value that is always zero into every
/// committed hash.
pub fn raise_act(world: &mut EcsWorld, kind: ActKind, actor: Uuid, at: DVec3) {
    if !at.is_finite() {
        return;
    }
    let mut log = world
        .world_mut()
        .remove_resource::<WitnessLog>()
        .unwrap_or_default();
    // Bounded by the same ceiling the pass itself is: a step that raised more
    // acts than can be observed would grow a queue the drain then throws away.
    if log.pending.len() < MAX_WITNESSED_ACTS {
        log.pending.push((kind, actor, at));
    }
    world.world_mut().insert_resource(log);
}

/// **Take everything raised this step**, in the order it was raised — the
/// witness pass's own drain, and the only reader of the queue.
pub fn take_raised(world: &mut EcsWorld) -> Vec<(ActKind, Uuid, DVec3)> {
    let Some(mut log) = world.world_mut().remove_resource::<WitnessLog>() else {
        return Vec::new();
    };
    let out = std::mem::take(&mut log.pending);
    world.world_mut().insert_resource(log);
    out
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

/// **What an actor looks like, as one number** — the description half of an act,
/// and the ONE input the recognition scorer takes from this side.
///
/// [`crate::crowd::appearance_of`] digested: the palette swap the person is
/// *drawn in* right now, which is what a witness on the pavement can actually
/// report. A hero with no crowd record answers the derived draw, exactly as
/// every crowd agent that has never changed does — so this needs no second
/// table and no per-character authoring.
///
/// # It takes a WORLD now, and that is the whole repair
///
/// Wave WPN1's version took only the guid, because the guid was the answer.
/// Reading a *description* means reading what somebody is wearing, and what
/// somebody is wearing is state that a wardrobe can change — so it is in the
/// world or it is not real. `describe_here` in `inf_ecs::crime` is the only
/// caller that matters; everything else compares digests.
pub fn look_digest(world: &EcsWorld, actor: Uuid) -> u64 {
    crate::crowd::appearance_of(world, actor).digest()
}

/// Bytes one act folds into [`witness_state_bytes`].
///
/// `actor (16) | kind (1) | at.x/y/z (24) | step (8) | look (8) | vehicle (16) |
/// observers (1 count + 16 each)`, so an act with no observers is 74.
pub const ACT_TRACE_BYTES: usize = 74;

/// **What the street saw, as bytes** — the section a replay trace folds (wave
/// EMS3).
///
/// The dispatcher's argument verbatim, one system back: an act's **observer
/// list** decides whether a crime is ever reported, and it is produced by a
/// line-of-sight ray that two hosts could answer differently if one of them
/// built a collider the other did not. Without this section that divergence is
/// invisible until a police car drives somewhere — which is many seconds later,
/// through a dispatcher, and by then the trace says the cars disagree rather
/// than that the *sight lines* did, which is the step a reader needs.
///
/// The observer list is folded **in full** rather than as a count: "two people
/// saw it" and "two *different* people saw it" are the same number and different
/// worlds, and [`MAX_OBSERVERS`] bounds the cost at eight guids an act.
/// [`WitnessLog::dropped`] rides along, because a ring that evicted on one host
/// and not the other has diverged in a way the retained tail cannot show.
///
/// Empty on a level where nothing has happened, which is what keeps every trace
/// committed before this wave byte-identical.
pub fn witness_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(log) = world.world().get_resource::<WitnessLog>() else {
        return Vec::new();
    };
    if log.acts.is_empty() && log.dropped == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(8 + log.acts.len() * ACT_TRACE_BYTES);
    out.extend_from_slice(&log.dropped.to_le_bytes());
    for a in &log.acts {
        out.extend_from_slice(a.actor.as_bytes());
        out.push(a.kind.as_u8());
        out.extend_from_slice(&a.at.x.to_le_bytes());
        out.extend_from_slice(&a.at.y.to_le_bytes());
        out.extend_from_slice(&a.at.z.to_le_bytes());
        out.extend_from_slice(&a.step.to_le_bytes());
        out.extend_from_slice(&a.actor_look.to_le_bytes());
        out.extend_from_slice(a.actor_vehicle.unwrap_or(Uuid::nil()).as_bytes());
        // A `u8` count, because the list is bounded at `MAX_OBSERVERS` = 8 and a
        // length that cannot be trusted is a length a reader has to case-split.
        out.push(a.observers.len().min(MAX_OBSERVERS) as u8);
        for o in a.observers.iter().take(MAX_OBSERVERS) {
            out.extend_from_slice(o.as_bytes());
        }
    }
    out
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

    /// **A DESCRIPTION IS NOT AN IDENTITY** — the wave's headline, as an
    /// assertion.
    ///
    /// WPN1's arm here read `assert_ne!(look_digest(guid(7)),
    /// look_digest(guid(8)))` and called it "two actors describe differently".
    /// That assertion is exactly what an identity hash satisfies and what a
    /// description must **not**: the three claims below are the three the old
    /// one got wrong, and reinstating it would fail the first of them.
    #[test]
    fn a_description_is_what_you_wear_and_not_who_you_are() {
        use crate::crowd::{set_appearance, Appearance};
        let mut w = EcsWorld::new();
        // (1) TWO PEOPLE IN THE SAME CLOTHES DESCRIBE THE SAME. This is what an
        //     identity hash can never do. It is what an innocent bystander in
        //     the wrong coat would REST on; whether one can happen is a question
        //     about the recognition pass rather than about the digest, and
        //     `crime_3d::an_innocent_in_the_same_coat_is_never_looked_at` is
        //     where that is measured.
        set_appearance(&mut w, guid(7), Appearance { outfit: 3 });
        set_appearance(&mut w, guid(8), Appearance { outfit: 3 });
        assert_eq!(
            look_digest(&w, guid(7)),
            look_digest(&w, guid(8)),
            "two people in one outfit described differently — this is an identity, not a look"
        );
        // (2) ONE PERSON WHO CHANGES DESCRIBES DIFFERENTLY. This is the whole of
        //     the mandate's evasion route.
        let before = look_digest(&w, guid(7));
        set_appearance(&mut w, guid(7), Appearance { outfit: 5 });
        assert_ne!(
            look_digest(&w, guid(7)),
            before,
            "changing clothes did not change the description"
        );
        // (3) …and it is still a pure function of sim state: same world, same
        //     answer, and no `Guid` reaches the digest at all.
        assert_eq!(look_digest(&w, guid(7)), look_digest(&w, guid(7)));
        assert_eq!(
            Appearance { outfit: 5 }.digest(),
            look_digest(&w, guid(7)),
            "the digest read something other than the outfit"
        );
        // An undressed world answers the DERIVED draw, which is what keeps every
        // level committed before this wave identical.
        let bare = EcsWorld::new();
        assert_eq!(
            look_digest(&bare, guid(7)),
            Appearance {
                outfit: crate::crowd::derived_outfit(guid(7))
            }
            .digest()
        );
        assert_eq!(actor_vehicle(&bare, guid(7)), None);
    }

    /// **The trace carries WHO SAW IT, not how many**, and an empty log folds
    /// nothing.
    #[test]
    fn the_witness_trace_folds_the_observer_list_itself() {
        let mut w = EcsWorld::new();
        assert!(
            witness_state_bytes(&w).is_empty(),
            "a level where nothing happened folds bytes"
        );
        let act = |obs: Vec<Uuid>| WitnessedAct {
            kind: ActKind::Carjack,
            actor: guid(1),
            at: DVec3::new(1.0, 2.0, 3.0),
            step: 40,
            observers: obs,
            actor_look: 0xabcd,
            actor_vehicle: Some(guid(9)),
        };
        record_act(&mut w, act(vec![guid(2), guid(3)]));
        let two = witness_state_bytes(&w);
        assert_eq!(
            two.len(),
            8 + ACT_TRACE_BYTES + 2 * 16,
            "the act's byte width is not what `ACT_TRACE_BYTES` says"
        );
        // **Two DIFFERENT people is a different world from two people.** A count
        // cannot see this; the list can.
        let mut other = EcsWorld::new();
        record_act(&mut other, act(vec![guid(2), guid(4)]));
        assert_ne!(
            witness_state_bytes(&other),
            two,
            "two different observer sets folded the same bytes"
        );
        // …and the eviction counter is in it, so a ring that dropped on one host
        // and not the other diverges here rather than silently.
        for n in 0..MAX_WITNESSED_ACTS as u64 {
            record_act(
                &mut w,
                WitnessedAct {
                    step: n,
                    ..act(Vec::new())
                },
            );
        }
        assert!(witness_dropped(&w) > 0);
        assert_eq!(
            &witness_state_bytes(&w)[..8],
            &witness_dropped(&w).to_le_bytes()
        );
    }
}
