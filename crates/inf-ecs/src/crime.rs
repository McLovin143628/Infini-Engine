//! **CRIMINAL PROFILES** (wave EMS3) — what the police know about somebody, and
//! the one rule that decides whether the person in front of an officer is them.
//!
//! # The mandate, and the sentence this module exists to make true
//!
//! *"There will be 'Criminal Profiles' that will be dynamically built through the
//! game. The police will remember the clothing that a user was wearing when
//! committing a crime or the vehicle they were driving — and the user will
//! realistically have to ditch their car and/or their clothes to evade the
//! police and drop their wanted level."*
//!
//! Every clause of that is a **channel**: a profile is not a tag on a person, it
//! is a description made of things the world can show and a player can change.
//! There are two of them — what you are wearing ([`crate::crowd::Appearance`])
//! and what you are driving ([`vehicle_digest`]) — and evading is the act of
//! making the description stop matching you.
//!
//! # THE POLICE DO NOT CHEAT, and that is a law with a shape
//!
//! A wanted system is trivial to write and almost always wrong: read the
//! player's transform, drive at it. This engine may not, and the refusal is
//! structural rather than a promise:
//!
//! * [`Profile::last_seen`] is a **private field**. Nothing outside this module
//!   can write it, and inside it exactly two functions do —
//!   [`report_act`] (somebody watched a crime happen) and [`sight`] (an officer
//!   recognised somebody). There is no third door, so a search can only ever
//!   converge on a place a witness actually put the suspect.
//! * [`match_score`] — the channel half of recognition — **takes no `Uuid` at
//!   all**. An identity is not in scope, so the scorer cannot accidentally
//!   become a tracker.
//! * The other half (line of sight, distance, the night) lives in
//!   `inf_physics::d3::crime`, because the ray is a physics primitive; it
//!   multiplies this score and can only ever make it smaller.
//!
//! # …and the bound on the other side of that law (EMS3 audit)
//!
//! The applier walks `officers x wanted` and scores each suspect against **their
//! own file**, so an officer never compares the people in front of them against
//! the descriptions they are carrying. Nothing above is weakened by it — no
//! `last_seen` can be written without a ray — but one sentence this module used
//! to make is not yet true of the *world*: two people dressed alike collide in
//! the **channel**, and the collision cannot yet reach an officer's eyes,
//! because somebody with no file is never in a scored pair. The innocent
//! bystander in the wrong coat is a property of the description and not yet a
//! thing that happens on a street; `inf_physics::d3::crime::look` carries the
//! cost argument and `crime_3d::an_innocent_in_the_same_coat_is_never_looked_at`
//! pins it.
//!
//! `ems3_crime_gate::the_police_never_read_the_players_true_position` is the
//! falsifiable form: the suspect walks a long way with no officer able to see
//! them, and the ledger's `last_seen` does not move a millimetre.
//!
//! # It is a RESOURCE
//!
//! [`crate::dispatch::DispatchRes`]' shape exactly: derived at run time, nothing
//! can save it, and **no schema moves** — scene v27 and `ScenePayload` v13
//! stand. A wanted level is not something an author writes into a level, and
//! [`clear_crime`] is its Simulate twin.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use glam::DVec3;
use uuid::Uuid;

use crate::witness::WitnessedAct;
use crate::world::EcsWorld;

// ── what a description is made of ───────────────────────────────────────────

/// **The two things a witness can describe.**
///
/// Two, and no third, because each one has to be a thing the world *shows* and
/// the player can *change*. A face would be neither: this engine draws one mesh
/// for a crowd and a player cannot put on a different nose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    /// What they had on — [`crate::crowd::Appearance::digest`]. Dies the moment
    /// they change at a wardrobe.
    Outfit,
    /// What they were driving — [`vehicle_digest`]. Dies the moment they get
    /// out, and comes back if they get into the same car again.
    Vehicle,
}

impl Channel {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            Channel::Outfit => "outfit",
            Channel::Vehicle => "vehicle",
        }
    }

    /// The byte this channel folds into a trace. **Frozen, append-only** on
    /// [`crate::dispatch::UnitKind::as_u8`]'s terms.
    pub fn as_u8(self) -> u8 {
        match self {
            Channel::Outfit => 0,
            Channel::Vehicle => 1,
        }
    }

    /// **How much of a recognition this channel alone is worth**, in
    /// [`match_score`]'s units.
    ///
    /// A **vehicle** outweighs an **outfit**, and the ordering is the design
    /// rather than a tuning knob: a description of a car is a shape, a size and
    /// a colour, and a description of a person on a pavement is one colour seen
    /// once.
    ///
    /// # The table these numbers make, MEASURED rather than asserted
    ///
    /// The first draft of this doc claimed *"neither reaches
    /// [`RECOGNIZE_SCORE`] on its own at night"*, and
    /// `a_sighting_is_worth_less_far_away_and_less_at_night` printed the
    /// arithmetic and said otherwise — twice, in both directions. This is what
    /// the constants actually do, at full [`freshness`], as
    /// **the range inside which a suspect is recognised**:
    ///
    /// | matching | daylight | night |
    /// |---|---|---|
    /// | outfit only | 16.7 m | 1.1 m |
    /// | vehicle only | 23.5 m | 12.5 m |
    /// | both | 25.1 m | 15.2 m |
    ///
    /// Which is the mandate's *"ditch the car **and/or** the clothes"* as a
    /// consequence rather than as a claim: changing your coat in daylight takes
    /// an officer from seeing you across a street to not seeing you at all,
    /// changing it at night is nearly total, and **the car is the one you cannot
    /// talk your way out of** — it is visible at twice the distance and half of
    /// it survives the dark.
    pub fn weight(self) -> f64 {
        match self {
            Channel::Outfit => 0.60,
            Channel::Vehicle => 0.85,
        }
    }
}

/// **What somebody looks like right now** — the live half of a recognition.
///
/// Deliberately **not** carrying a `Uuid`: this is the input a scorer compares
/// against a record, and the whole police-don't-cheat law is that the comparison
/// has no access to who the person actually is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Description {
    /// The outfit digest — every person has one, so this is not an `Option`.
    pub outfit: u64,
    /// The vehicle digest, or `None` for somebody on foot.
    pub vehicle: Option<u64>,
}

/// **One thing the police remember**, and when they last had it confirmed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evidence {
    /// Which channel it describes.
    pub channel: Channel,
    /// The description itself.
    pub digest: u64,
    /// The fixed step it was last confirmed on — what [`freshness`] ages.
    pub step: u64,
}

// ── how long anything lasts ─────────────────────────────────────────────────

/// How long a piece of evidence stays warm before it is worthless, fixed steps.
///
/// **Three thousand six hundred — a minute at 60 Hz**, and it is the same number
/// [`crate::dispatch::INCIDENT_KEEP_STEPS`] is, on purpose: an incident and the
/// description that opened it should stop mattering together, or a town keeps
/// looking for a coat nobody is still calling about.
///
/// Evidence does not fall off a cliff at the end of it —
/// see [`freshness`] — it fades linearly, so an officer who sees you at
/// fifty-nine seconds is nearly sure and one who sees you at sixty-one has
/// nothing.
pub const EVIDENCE_COLD_STEPS: u64 = 3600;

/// How long one point of heat survives without a fresh sighting, fixed steps.
///
/// Nine hundred — **fifteen seconds a star-ish**, so a two-point carjack is
/// forgotten in half a minute of not being seen and a nine-point spree takes
/// over two minutes to walk off. It is deliberately *shorter* than
/// [`EVIDENCE_COLD_STEPS`]: the heat goes first and the description lingers,
/// which is what makes "they stopped looking for me but they still have my
/// description" a state the game can be in.
pub const HEAT_DECAY_STEPS: u64 = 900;

/// The most profiles one level keeps.
///
/// Thirty-two. A **cost** bound on [`crate::dispatch::MAX_OPEN_INCIDENTS`]'
/// terms: every profile is a candidate the recognition pass scores against
/// everybody an officer can see, and a town with thirty-two separate wanted
/// criminals in it has more going on than a player can follow. The
/// thirty-third **replaces the coldest**, rather than being refused: the newest
/// crime is the one somebody is calling about, and a refusal here would mean the
/// only way to become un-arrestable is to commit thirty-two crimes.
pub const MAX_PROFILES: usize = 32;

/// **How much a piece of evidence is still worth**, `[0, 1]`.
///
/// One when it was just confirmed, zero at [`EVIDENCE_COLD_STEPS`], linear
/// between. Linear rather than exponential because it is a *description* and not
/// a physical quantity: a witness's memory has an end, and a curve with a tail
/// would leave a fifty-year-old coat worth a thousandth of a recognition for
/// ever, which is a number that only ever confuses a reader.
///
/// A pure function of two integers, so both hosts agree by construction.
pub fn freshness(recorded_step: u64, now: u64) -> f64 {
    let age = now.saturating_sub(recorded_step);
    if age >= EVIDENCE_COLD_STEPS {
        return 0.0;
    }
    1.0 - (age as f64 / EVIDENCE_COLD_STEPS as f64)
}

// ── the severity ladder ─────────────────────────────────────────────────────

/// **What the police send**, by how wanted somebody is.
///
/// The ladder the mandate's *"dynamic"* is actually made of: the same act
/// escalates from one car to a street full of them, and **SWAT finally becomes a
/// behaviour** rather than the parked van EMS1 gave the island. (EMS2's ruling
/// stands and is why this is here and not in `UnitKind`: SWAT is a **crew**, not
/// a service — a van full of officers is still the police — so the escalation
/// belongs to the *response* and not to the fleet.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Response {
    /// Nobody is looking.
    Cold,
    /// One car.
    Patrol,
    /// More than one, converging.
    MultiUnit,
    /// Everything the town has.
    Swat,
}

impl Response {
    /// A stable short name.
    pub fn name(self) -> &'static str {
        match self {
            Response::Cold => "cold",
            Response::Patrol => "patrol",
            Response::MultiUnit => "multi-unit",
            Response::Swat => "swat",
        }
    }

    /// The byte this rung folds into a trace. **Frozen, append-only.**
    pub fn as_u8(self) -> u8 {
        match self {
            Response::Cold => 0,
            Response::Patrol => 1,
            Response::MultiUnit => 2,
            Response::Swat => 3,
        }
    }

    /// **Which rung this much heat is on** — the whole ladder, as one function.
    ///
    /// One or two points is a car; three to five is a converging pair; six and
    /// over is everything. The thresholds are the ladder's *shape* rather than
    /// exact truths: the property that matters is that a single petty act
    /// ([`crate::witness::ActKind::heat`] of one) can never reach the top rung, and that a
    /// killing plus anything else does.
    pub fn for_heat(heat: u32) -> Self {
        match heat {
            0 => Response::Cold,
            1..=2 => Response::Patrol,
            3..=5 => Response::MultiUnit,
            _ => Response::Swat,
        }
    }

    /// **How many units this rung wants at the search at once.**
    ///
    /// A *request*, not a guarantee: the dispatcher sends the nearest free unit
    /// of the right service and a town with one cruiser answers a SWAT-grade
    /// spree with one cruiser. That is the honest outcome and it is visible —
    /// `DispatchRes::unanswered` rises — rather than a queue nobody can see.
    pub fn units(self) -> usize {
        match self {
            Response::Cold => 0,
            Response::Patrol => 1,
            Response::MultiUnit => 2,
            Response::Swat => 3,
        }
    }
}

/// **The heat each star costs**, ascending — the HUD's whole scale.
///
/// Five thresholds, so a `heat` of 0 is no stars and 15 or more is five. It is a
/// table rather than a division because the rungs are not evenly spaced: the
/// first star is one act, the fifth is a spree, and a linear scale would either
/// make the first star unreachable or the fifth routine.
pub const WANTED_STARS: [u32; 5] = [1, 3, 6, 10, 15];

/// **How many stars this much heat draws**, `0..=5`.
pub fn stars(heat: u32) -> u8 {
    WANTED_STARS.iter().filter(|t| heat >= **t).count() as u8
}

// ── recognition, the half that has no identity in it ────────────────────────

/// How far away a description can be read at all, metres.
///
/// Forty. It is the distance at which a person is a silhouette with a colour —
/// which is exactly what a description *is* in this engine — and it is
/// deliberately shorter than `crate::traffic::TRAFFIC_FULL_M` (64 m), so
/// everybody an officer can recognise is somebody the simulation is running at
/// full detail.
pub const RECOGNITION_RANGE_M: f64 = 40.0;

/// What a sighting is worth at night, as a multiplier.
///
/// **0.60**, and the number carries the whole of the night's design: at 0.60 a
/// vehicle match alone is still read at 12.5 m and an outfit match alone
/// collapses to 1.1 m — see [`Channel::weight`]'s measured table. Driving a
/// marked car at night gets you seen; walking past in the same coat does not.
/// That is the behaviour the reference's own night chases have and it falls out
/// of one constant.
pub const NIGHT_RECOGNITION: f64 = 0.60;

/// What a match has to score to count as **this is them**.
///
/// 0.35. See [`Channel::weight`] and [`NIGHT_RECOGNITION`] for the table this
/// number sits in the middle of; `ems3_crime_gate` prints the whole thing.
pub const RECOGNIZE_SCORE: f64 = 0.35;

/// **How much a clear view at this distance is worth**, `[0, 1]`.
///
/// Linear to [`RECOGNITION_RANGE_M`] and zero past it. Linear rather than
/// inverse-square because this is not light falling on a retina, it is a
/// person's confidence that the man across the road is the man from the
/// description, and that does go to nothing at a definite distance.
///
/// `night` multiplies it. Line of sight is **not** in here: it is a hard gate on
/// the caller's side (`inf_physics::d3::crime`), because a wall is not a
/// discount.
pub fn sight_factor(distance_m: f64, night: bool) -> f64 {
    if !distance_m.is_finite() || distance_m >= RECOGNITION_RANGE_M {
        return 0.0;
    }
    let near = 1.0 - (distance_m.max(0.0) / RECOGNITION_RANGE_M);
    near * if night { NIGHT_RECOGNITION } else { 1.0 }
}

/// **Is it night on this level?** — the one door, so an officer's eyes and
/// anything else that cares agree about when it is dark.
///
/// `crate::traffic::NIGHT_CIRCUIT_H`'s hours, reused rather than restated: the
/// engine already has an opinion about when a street goes quiet, and a second
/// pair of numbers here would be a second night.
pub fn is_night(hour: f64) -> bool {
    let (from, to) = crate::traffic::NIGHT_CIRCUIT_H;
    if !hour.is_finite() {
        return false;
    }
    let span = (to - from).rem_euclid(24.0);
    (hour - from).rem_euclid(24.0) < span
}

/// **DOES THIS DESCRIPTION MATCH THIS PROFILE?** — the channel half of
/// recognition, `[0, 1]`.
///
/// # There is no `Uuid` in this signature, and that is the law
///
/// A wanted system that reads an identity is a wanted system a player cannot
/// beat. Passing the suspect's guid in here would have been the natural thing to
/// do — the caller has it, the profile is keyed on it — and it would have made
/// every clause of the mandate unfalsifiable, because a scorer *could* have
/// compared the two and no test would show it. So it cannot: the profile's
/// evidence and the live description are all that is in scope.
///
/// # How the channels combine, and why it is not a sum
///
/// `1 - PROD(1 - w_i * f_i)` — "the chance that at least one of these gives you
/// away". The obvious rule is a sum clamped to one, and it was written that way
/// first; `the_scorer_reads_a_description_and_never_an_identity` measured what
/// it does. Two fresh channels sum to **1.45**, so the clamp held the score at a
/// flat 1.0 for the first **31 %** of [`EVIDENCE_COLD_STEPS`] — a description
/// that visibly did not age for the first nineteen seconds, and then fell off a
/// step. This product ages continuously, is monotone in every channel (two
/// pieces of evidence are always worth more than one), and never exceeds one
/// without a clamp to hide behind.
///
/// A channel that does not match contributes **nothing** — not a penalty: a
/// criminal in a stolen car wearing new clothes is not *less* recognisable than
/// a stranger, they are simply not recognisable by that piece of evidence.
pub fn match_score(profile: &Profile, seen: Description, now: u64) -> f64 {
    let mut miss = 1.0;
    for e in profile.evidence.values() {
        let matched = match e.channel {
            Channel::Outfit => seen.outfit == e.digest,
            Channel::Vehicle => seen.vehicle == Some(e.digest),
        };
        if matched {
            miss *= 1.0 - e.channel.weight() * freshness(e.step, now);
        }
    }
    1.0 - miss
}

// ── the profile ─────────────────────────────────────────────────────────────

/// **One person's file**, built from what people saw.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// How wanted they are, in [`crate::witness::ActKind::heat`]'s units. Zero means the file is
    /// open and nobody is looking.
    pub heat: u32,
    /// What the police have, one entry per channel — a later sighting through
    /// the same channel **replaces** the earlier one, because a description is
    /// the latest one anybody gave and not a history.
    pub evidence: BTreeMap<Channel, Evidence>,
    /// The step the file was opened on.
    pub opened_step: u64,
    /// How many times somebody has been recognised against it — the engagement
    /// counter that tells "the search is working" from "nobody has looked".
    pub sightings: u64,
    /// The last step heat was taken off. Not a decay *clock* — the decay is a
    /// pure function of this and the current step, so a host that started
    /// mid-trace computes the same answer.
    pub decayed_step: u64,
    /// **Where the police last had them**, world metres — the search's own
    /// destination.
    ///
    /// PRIVATE, and it is the police-don't-cheat law in its compile-checked
    /// form: nothing outside this module can write it, and inside it only
    /// [`report_act`] and [`sight`] do. A search that converged on the player
    /// would need a third writer, and there is nowhere to put one.
    last_seen: DVec3,
    /// The step [`last_seen`](Self::last_seen) was written on.
    last_seen_step: u64,
}

impl Profile {
    /// **Where the police think they are** — the read side.
    pub fn last_seen(&self) -> DVec3 {
        self.last_seen
    }

    /// The step [`last_seen`](Self::last_seen) was written on.
    pub fn last_seen_step(&self) -> u64 {
        self.last_seen_step
    }

    /// How stale the trail is, in fixed steps.
    pub fn trail_age(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_seen_step)
    }

    /// What the police would send for this file right now.
    pub fn response(&self) -> Response {
        Response::for_heat(self.heat)
    }

    /// How many stars this file draws.
    pub fn stars(&self) -> u8 {
        stars(self.heat)
    }

    /// **Whether any evidence in this file is still worth anything** at `now`.
    ///
    /// A file with heat and no warm evidence is a town that wants somebody it
    /// can no longer describe, which is the state a successful evasion produces
    /// and the state [`decay`] closes.
    pub fn describable(&self, now: u64) -> bool {
        self.evidence.values().any(|e| freshness(e.step, now) > 0.0)
    }
}

/// **The ledger** — every open file, in `Guid` order.
///
/// Keyed on the suspect's guid, and that key is a **case number**: nothing on
/// the recognition path reads it. [`match_score`] cannot (there is no `Uuid` in
/// its signature) and the applier looks a description up rather than a person.
/// It is here because a file has to be attached to *something* so that a second
/// crime by the same person adds to the first, and because the trace needs a
/// stable order.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct CrimeRes {
    /// The open files.
    pub profiles: BTreeMap<Uuid, Profile>,
    /// Files opened over the session — a lifetime counter, so a gate can arm
    /// itself before asserting anything about a table allowed to be empty.
    pub opened: u64,
    /// Recognitions over the session.
    pub sightings: u64,
    /// Files that went cold and were dropped.
    pub cleared: u64,
    /// Acts already turned into evidence, by the step they happened on — so the
    /// witness log is read forward once and one crime cannot be filed twice.
    pub seen_act_step: u64,
}

/// The ledger, or `None` on a level where nothing has happened.
pub fn crime_of(world: &EcsWorld) -> Option<&CrimeRes> {
    world.world().get_resource::<CrimeRes>()
}

/// One person's file, or `None`.
pub fn profile_of(world: &EcsWorld, suspect: Uuid) -> Option<&Profile> {
    crime_of(world).and_then(|c| c.profiles.get(&suspect))
}

/// **How wanted somebody is**, `0` for everybody the police have never heard of
/// — the HUD's own reader and the cheapest possible question.
pub fn heat_of(world: &EcsWorld, suspect: Uuid) -> u32 {
    profile_of(world, suspect).map(|p| p.heat).unwrap_or(0)
}

/// **HOW MANY STARS TO DRAW OVER THIS PERSON**, and how many slots to draw at
/// all — the HUD's one Ring-0 door (wave EMS3).
///
/// `inf_ecs::vehicle::drive_readout` and `inf_ecs::weapon::ammo_readout`'s
/// shape: the *decision* about what a player is told is sim state and lives in
/// Ring 0, and the host's only job is to draw it. A host that computed
/// `heat / 3` for itself would be a second opinion about the ladder, and the
/// two would disagree the first time [`WANTED_STARS`] moved.
///
/// `None` for somebody nobody is looking for, which is what keeps the corner of
/// the screen empty in every game that is not this one.
pub fn wanted_readout(world: &EcsWorld, actor: Uuid) -> Option<(u8, u8)> {
    let heat = heat_of(world, actor);
    (heat > 0).then_some((stars(heat), WANTED_STARS.len() as u8))
}

/// **Forget every open file** — [`crate::dispatch::clear_dispatch`]'s twin, for
/// its reason: an editor Simulate session must leave nothing behind in the
/// author's document, and a wanted level is emphatically not something an author
/// wrote. A profile owns no entity, so unlike the dispatcher's crews there is
/// nothing to despawn.
pub fn clear_crime(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<CrimeRes>();
}

// ── the two doors that may write `last_seen` ────────────────────────────────

/// **FILE A CRIME** — the first of the two `last_seen` writers.
///
/// The act carries its own description ([`WitnessedAct::actor_look`]) and its
/// own vehicle, both recorded on the step it happened by the pass that also
/// worked out who could see it. A crime **nobody saw** files nothing: an empty
/// observer list is a crime with no witness, and a police force that opened a
/// file on it would be reading the simulation rather than the street.
///
/// `last_seen` is the act's own position, which is legitimate for exactly that
/// reason — somebody was standing there watching when it happened.
///
/// **A NIL ACTOR FILES NOTHING** (wave EMS3 audit). `WitnessedAct::actor` is
/// *who did it*, and the producer answers `Uuid::nil()` when the world cannot
/// name anybody — a body that stopped working with no blow in the step's hits to
/// account for it. The scene is still worth a call and
/// `d3::dispatch::open_incidents` still opens one; what nobody may do is put an
/// unattributable death on a person's file, and `Uuid::nil()` is a person.
///
/// Returns the heat the file now carries, or `None` if nothing was filed.
pub fn report_act(world: &mut EcsWorld, act: &WitnessedAct, vehicle: Option<u64>) -> Option<u32> {
    if act.observers.is_empty() || !act.at.is_finite() || act.actor.is_nil() {
        return None;
    }
    let mut res = world
        .world_mut()
        .remove_resource::<CrimeRes>()
        .unwrap_or_default();
    let file = res.profiles.entry(act.actor).or_insert_with(|| Profile {
        heat: 0,
        evidence: BTreeMap::new(),
        opened_step: act.step,
        sightings: 0,
        decayed_step: act.step,
        last_seen: act.at,
        last_seen_step: act.step,
    });
    let fresh = file.heat == 0;
    file.heat = file.heat.saturating_add(act.kind.heat());
    file.last_seen = act.at;
    file.last_seen_step = act.step;
    // **The description, replacing whatever the channel held.** A witness gives
    // the description they just saw; a file that kept both would let a criminal
    // be recognised by clothes they took off an hour ago for ever.
    file.evidence.insert(
        Channel::Outfit,
        Evidence {
            channel: Channel::Outfit,
            digest: act.actor_look,
            step: act.step,
        },
    );
    // **A crime on foot does not CLEAR an earlier car**, which is why this is an
    // `if let` and not an `insert`-or-`remove`: a `None` vehicle leaves whatever
    // the channel held, and it ages out on its own. "He did the first one in a
    // red van" is still worth something, and letting a second crime erase it
    // would make committing one the way to forget the other.
    if let Some(digest) = vehicle {
        file.evidence.insert(
            Channel::Vehicle,
            Evidence {
                channel: Channel::Vehicle,
                digest,
                step: act.step,
            },
        );
    }
    let heat = file.heat;
    if fresh {
        res.opened = res.opened.saturating_add(1);
    }
    evict_coldest(&mut res, act.step);
    world.world_mut().insert_resource(res);
    Some(heat)
}

/// **SOMEBODY WAS RECOGNISED** — the second and last `last_seen` writer.
///
/// Called by `inf_physics::d3::crime` and by nothing else, for a pair whose
/// combined score cleared [`RECOGNIZE_SCORE`]. `at` is where the *officer* saw
/// them, which is a place an officer actually had a line to.
///
/// Refreshes the matching channels' [`Evidence::step`] rather than the whole
/// file: seeing a criminal in the same coat renews the coat's freshness, and
/// does **not** invent evidence about a car nobody saw.
///
/// Returns whether the file existed to be refreshed.
pub fn sight(world: &mut EcsWorld, suspect: Uuid, at: DVec3, seen: Description, step: u64) -> bool {
    if !at.is_finite() {
        return false;
    }
    let Some(mut res) = world.world_mut().remove_resource::<CrimeRes>() else {
        return false;
    };
    let found = match res.profiles.get_mut(&suspect) {
        Some(file) => {
            file.last_seen = at;
            file.last_seen_step = step;
            file.sightings = file.sightings.saturating_add(1);
            for e in file.evidence.values_mut() {
                let still = match e.channel {
                    Channel::Outfit => seen.outfit == e.digest,
                    Channel::Vehicle => seen.vehicle == Some(e.digest),
                };
                if still {
                    e.step = step;
                }
            }
            // **A sighting stops the clock, it does not reset the crime.** The
            // heat is what they did; being seen only means the decay has not
            // been running.
            file.decayed_step = step;
            res.sightings = res.sightings.saturating_add(1);
            true
        }
        None => false,
    };
    world.world_mut().insert_resource(res);
    found
}

/// **THE COOL-DOWN** — heat falls when nobody has seen you, and files that are
/// cold and undescribable are closed.
///
/// A pure function of the ledger and the step: `heat` loses one point per
/// [`HEAT_DECAY_STEPS`] since the last sighting, and a file with no heat and no
/// warm evidence is dropped. Nothing here reads the world, so two hosts that
/// agree about the ledger agree about the decay by construction.
///
/// Returns how many files were closed.
pub fn decay(world: &mut EcsWorld, step: u64) -> usize {
    let Some(mut res) = world.world_mut().remove_resource::<CrimeRes>() else {
        return 0;
    };
    for file in res.profiles.values_mut() {
        if file.heat == 0 {
            continue;
        }
        let elapsed = step.saturating_sub(file.decayed_step);
        let lost = (elapsed / HEAT_DECAY_STEPS) as u32;
        if lost == 0 {
            continue;
        }
        file.heat = file.heat.saturating_sub(lost);
        // Advance by whole periods only, so the remainder is carried and a file
        // does not lose its fraction of a period every time this runs.
        file.decayed_step = file
            .decayed_step
            .saturating_add(u64::from(lost) * HEAT_DECAY_STEPS);
    }
    let before = res.profiles.len();
    res.profiles
        .retain(|_, f| f.heat > 0 || f.describable(step));
    let closed = before - res.profiles.len();
    res.cleared = res.cleared.saturating_add(closed as u64);
    world.world_mut().insert_resource(res);
    closed
}

/// Drop the coldest file when the ledger is full — see [`MAX_PROFILES`].
fn evict_coldest(res: &mut CrimeRes, now: u64) {
    while res.profiles.len() > MAX_PROFILES {
        // Least heat first, then the stalest trail, then the guid — three keys
        // so two hosts drop the same file.
        let Some(victim) = res
            .profiles
            .iter()
            .min_by(|a, b| {
                a.1.heat
                    .cmp(&b.1.heat)
                    .then(b.1.trail_age(now).cmp(&a.1.trail_age(now)))
                    .then(a.0.cmp(b.0))
            })
            .map(|(g, _)| *g)
        else {
            return;
        };
        res.profiles.remove(&victim);
        res.cleared = res.cleared.saturating_add(1);
    }
}

// ── what somebody looks like, right now ─────────────────────────────────────

/// **The description of a vehicle** — its size and its paint, read off the
/// world.
///
/// # The level is read, not the recipe — `dispatch::unit_kind_of`'s ruling
///
/// The obvious source is `crate::traffic::TrafficRecord`, which carries the
/// catalogue row and the paint the car was planned with. It is the wrong one for
/// this crate's reason exactly: a hero's own car, a fleet vehicle EMS1 parked
/// and anything a Blueprint spawned have no traffic record at all, and a
/// recogniser that answered `None` for them would have made *the one car in the
/// level a player actually drives* undescribable.
///
/// So it reads what the vehicle **shows**: the chassis collider's own
/// half-extents and the material it is painted in — a big red box or a small
/// blue one, which is what a witness reports. Quantised to a decimetre and to
/// sixteenths of a channel before it is hashed, so two hosts that computed a
/// paint through different-but-equal arithmetic describe one car.
///
/// `None` for a guid that is not a body with a collider, which is the honest
/// answer to "what car is that" about something that is not a car.
pub fn vehicle_digest(world: &EcsWorld, chassis: Uuid) -> Option<u64> {
    let e = world.entity_of(chassis)?;
    let w = world.world();
    let collider = w.get::<crate::components::Collider3D>(e)?;
    let half = collider.half_extents;
    if !(half.x.is_finite() && half.y.is_finite() && half.z.is_finite()) {
        return None;
    }
    let paint = w
        .get::<crate::components::Material>(e)
        .map(|m| m.base_color.to_array())
        .unwrap_or([0.5, 0.5, 0.5, 1.0]);
    // Decimetres of size and sixteenths of colour: enough to tell a van from a
    // sports car and red from blue, coarse enough that a description is a
    // description rather than a serial number.
    let dm = |v: f64| -> u64 { (v * 10.0).round().clamp(0.0, 4095.0) as u64 };
    let ch = |v: f32| -> u64 { ((v * 16.0).round() as i64).clamp(0, 16) as u64 };
    let bits = dm(half.x) | (dm(half.y) << 12) | (dm(half.z) << 24);
    let colour = ch(paint[0]) | (ch(paint[1]) << 8) | (ch(paint[2]) << 16);
    let mut x = bits ^ colour.rotate_left(40) ^ SALT_VEHICLE_LOOK;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    Some(x ^ (x >> 31))
}

/// Salts a vehicle's description, so a car and a coat cannot collide.
pub const SALT_VEHICLE_LOOK: u64 = 0x5645_4849_434c_0001;

/// **What this person looks like right now** — the ONE producer of a
/// [`Description`], so an officer and the file cannot be comparing two different
/// readings of one street.
pub fn describe(world: &EcsWorld, actor: Uuid) -> Description {
    Description {
        outfit: crate::witness::look_digest(world, actor),
        vehicle: crate::witness::actor_vehicle(world, actor)
            .and_then(|chassis| vehicle_digest(world, chassis)),
    }
}

// ── reading the log forward ─────────────────────────────────────────────────

/// **Turn everything new in the witness log into evidence** — the feed, read
/// forward exactly once.
///
/// `crate::dispatch`'s own crime feed shape and its reason: the log is a ring
/// that evicts, so it is walked by step and never re-scanned, and an act cannot
/// be filed twice.
///
/// Returns how many acts were filed — an engagement counter, because "the pass
/// ran" and "somebody is now wanted" are different facts.
pub fn file_new_acts(world: &mut EcsWorld) -> usize {
    let seen = crime_of(world).map(|c| c.seen_act_step).unwrap_or(0);
    // Cloned, because filing writes the resource and the log borrows the world.
    // Bounded by `MAX_WITNESSED_ACTS`, and empty on every step nothing happened.
    let fresh: Vec<WitnessedAct> = crate::witness::witnessed(world)
        .iter()
        .filter(|a| a.step > seen && !a.observers.is_empty())
        .cloned()
        .collect();
    if fresh.is_empty() {
        return 0;
    }
    let mut newest = seen;
    let mut filed = 0usize;
    for act in fresh {
        newest = newest.max(act.step);
        let vehicle = act
            .actor_vehicle
            .and_then(|chassis| vehicle_digest(world, chassis));
        if report_act(world, &act, vehicle).is_some() {
            filed += 1;
        }
    }
    let mut res = world
        .world_mut()
        .remove_resource::<CrimeRes>()
        .unwrap_or_default();
    res.seen_act_step = newest;
    world.world_mut().insert_resource(res);
    filed
}

/// **Everybody the police are actively looking for**, in `Guid` order — the
/// files with heat on them.
pub fn wanted(world: &EcsWorld) -> Vec<Uuid> {
    crime_of(world)
        .map(|c| {
            c.profiles
                .iter()
                .filter(|(_, p)| p.heat > 0)
                .map(|(g, _)| *g)
                .collect()
        })
        .unwrap_or_default()
}

/// **The suspects an officer should be scoring against**, at most
/// [`MAX_PROFILES`] — the wanted set as a lookup.
pub fn wanted_set(world: &EcsWorld) -> BTreeSet<Uuid> {
    wanted(world).into_iter().collect()
}

// ── the trace ───────────────────────────────────────────────────────────────

/// Bytes one profile folds into [`profile_state_bytes`], before its evidence.
///
/// `suspect (16) | heat (4) | last_seen.x/y/z (24) | last_seen_step (8) |
/// opened (8) | decayed (8) | response (1) | evidence count (1)`.
pub const PROFILE_TRACE_BYTES: usize = 70;

/// Bytes one piece of evidence adds — `channel (1) | digest (8) | step (8)`.
pub const EVIDENCE_TRACE_BYTES: usize = 17;

/// **The ledger, as bytes** — the section a replay trace folds.
///
/// The dispatcher's argument verbatim: a profile's `heat` decides how many cars
/// come and its `last_seen` decides where they drive, so two hosts that filed
/// different evidence would compare equal at every step until one of them
/// happened to send a cruiser the other did not — which is many seconds after
/// they diverged.
///
/// The **response rung** is folded even though it is a pure function of `heat`,
/// for `crowd_state_bytes`' pose-digest reason: it is what the world *does*
/// with the number, and a wave that changes [`Response::for_heat`] should move
/// the traces that depend on it rather than moving nothing until a car happens
/// to be dispatched.
///
/// Empty on a level where nobody is wanted, which is what keeps every trace
/// committed before this wave byte-identical.
pub fn profile_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(res) = crime_of(world) else {
        return Vec::new();
    };
    if res.profiles.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(res.profiles.len() * PROFILE_TRACE_BYTES);
    for (suspect, p) in &res.profiles {
        out.extend_from_slice(suspect.as_bytes());
        out.extend_from_slice(&p.heat.to_le_bytes());
        out.extend_from_slice(&p.last_seen.x.to_le_bytes());
        out.extend_from_slice(&p.last_seen.y.to_le_bytes());
        out.extend_from_slice(&p.last_seen.z.to_le_bytes());
        out.extend_from_slice(&p.last_seen_step.to_le_bytes());
        out.extend_from_slice(&p.opened_step.to_le_bytes());
        out.extend_from_slice(&p.decayed_step.to_le_bytes());
        out.push(p.response().as_u8());
        out.push(p.evidence.len() as u8);
        for e in p.evidence.values() {
            out.push(e.channel.as_u8());
            out.extend_from_slice(&e.digest.to_le_bytes());
            out.extend_from_slice(&e.step.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crowd::{set_appearance, Appearance};
    use crate::witness::ActKind;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn act(kind: ActKind, actor: Uuid, look: u64, step: u64) -> WitnessedAct {
        WitnessedAct {
            kind,
            actor,
            at: DVec3::new(10.0, 0.0, 20.0),
            step,
            observers: vec![guid(0xbeef)],
            actor_look: look,
            actor_vehicle: None,
        }
    }

    /// **A CRIME NOBODY SAW IS NOT A CRIME**, and one somebody saw opens a file
    /// with a description in it.
    #[test]
    fn a_file_opens_only_when_somebody_was_watching() {
        let mut w = EcsWorld::new();
        let unseen = WitnessedAct {
            observers: Vec::new(),
            ..act(ActKind::Carjack, guid(1), 0xaaa, 10)
        };
        assert_eq!(report_act(&mut w, &unseen, None), None);
        assert!(crime_of(&w).is_none(), "an unseen crime opened a ledger");

        assert_eq!(
            report_act(
                &mut w,
                &act(ActKind::Carjack, guid(1), 0xaaa, 10),
                Some(0xccc)
            ),
            Some(1)
        );
        let p = profile_of(&w, guid(1)).expect("a file");
        assert_eq!(p.heat, 1);
        assert_eq!(p.response(), Response::Patrol);
        assert_eq!(p.last_seen(), DVec3::new(10.0, 0.0, 20.0));
        assert_eq!(p.evidence.len(), 2, "both channels were not recorded");
        // …and a second crime stacks.
        report_act(&mut w, &act(ActKind::Killed, guid(1), 0xaaa, 40), None);
        let p = profile_of(&w, guid(1)).expect("a file");
        assert_eq!(p.heat, 4);
        assert_eq!(p.response(), Response::MultiUnit);
        // The car is still on file: committing a crime on foot must not be a way
        // to make the police forget the van.
        assert!(p.evidence.contains_key(&Channel::Vehicle));
        assert_eq!(
            crime_of(&w).expect("ledger").opened,
            1,
            "one person, one file"
        );
    }

    /// **THE SCORER CANNOT SEE WHO YOU ARE** — two different people with one
    /// description score identically, which no identity-keyed rule can do.
    #[test]
    fn the_scorer_reads_a_description_and_never_an_identity() {
        let mut w = EcsWorld::new();
        report_act(
            &mut w,
            &act(ActKind::Carjack, guid(1), 0xaaa, 10),
            Some(0xccc),
        );
        let p = profile_of(&w, guid(1)).expect("a file").clone();
        let same = Description {
            outfit: 0xaaa,
            vehicle: Some(0xccc),
        };
        // The clothes alone, and the car alone — the two rungs the mandate's
        // "and/or" is made of — and both together.
        let clothes = Description {
            outfit: 0xaaa,
            vehicle: None,
        };
        let car = Description {
            outfit: 0x999,
            vehicle: Some(0xccc),
        };
        let (b, c, v) = (
            match_score(&p, same, 10),
            match_score(&p, clothes, 10),
            match_score(&p, car, 10),
        );
        println!("fresh: both {b:.3}, outfit only {c:.3}, vehicle only {v:.3}");
        assert!((c - 0.60).abs() < 1e-9);
        assert!((v - 0.85).abs() < 1e-9);
        // **Two channels beat one and never exceed one.** The clamped-sum rule
        // this replaced answered a flat 1.0 for both of the first two.
        assert!(b > v && b <= 1.0);
        // NEITHER: a swapped coat and a ditched car is a stranger.
        let stranger = Description {
            outfit: 0x999,
            vehicle: None,
        };
        assert_eq!(match_score(&p, stranger, 10), 0.0);
        // …and freshness ages it to nothing, CONTINUOUSLY — which is the
        // property the clamp destroyed: at a third of the window a clamped sum
        // was still answering 1.0.
        let quarter = match_score(&p, same, 10 + EVIDENCE_COLD_STEPS / 4);
        let half = match_score(&p, same, 10 + EVIDENCE_COLD_STEPS / 2);
        println!("both, aged 0/25/50/100 %: {b:.3} {quarter:.3} {half:.3} 0.000");
        assert!(
            b > quarter && quarter > half,
            "a description that does not age between 0 % and 50 % of its window"
        );
        assert_eq!(match_score(&p, same, 10 + EVIDENCE_COLD_STEPS), 0.0);
    }

    /// **HEAT FALLS WHEN NOBODY IS LOOKING, AND A SIGHTING STOPS THE CLOCK.**
    #[test]
    fn heat_decays_unseen_and_a_sighting_holds_it() {
        let mut w = EcsWorld::new();
        report_act(&mut w, &act(ActKind::Killed, guid(1), 0xaaa, 0), None);
        assert_eq!(heat_of(&w, guid(1)), 3);
        // Two whole periods: two points gone, and the remainder carried rather
        // than thrown away.
        decay(&mut w, HEAT_DECAY_STEPS * 2 + 5);
        assert_eq!(heat_of(&w, guid(1)), 1);
        assert_eq!(
            profile_of(&w, guid(1)).expect("a file").decayed_step,
            HEAT_DECAY_STEPS * 2,
            "the decay clock swallowed the remainder"
        );
        // A sighting stops it: the same elapsed time now costs nothing.
        let seen = Description {
            outfit: 0xaaa,
            vehicle: None,
        };
        assert!(sight(
            &mut w,
            guid(1),
            DVec3::new(5.0, 0.0, 5.0),
            seen,
            3000
        ));
        assert_eq!(profile_of(&w, guid(1)).expect("a file").last_seen().x, 5.0);
        decay(&mut w, 3000 + HEAT_DECAY_STEPS - 1);
        assert_eq!(
            heat_of(&w, guid(1)),
            1,
            "a fresh sighting did not hold the heat"
        );
        // …and once it is cold AND undescribable the file closes.
        decay(&mut w, 3000 + HEAT_DECAY_STEPS + EVIDENCE_COLD_STEPS);
        assert!(profile_of(&w, guid(1)).is_none(), "a cold file stayed open");
        assert_eq!(crime_of(&w).expect("ledger").cleared, 1);
        clear_crime(&mut w);
        assert!(crime_of(&w).is_none());
    }

    /// **THE LADDER**, rung by rung, and the stars that draw it.
    #[test]
    fn the_severity_ladder_climbs_and_the_stars_follow() {
        assert_eq!(Response::for_heat(0), Response::Cold);
        assert_eq!(Response::for_heat(1), Response::Patrol);
        assert_eq!(Response::for_heat(2), Response::Patrol);
        assert_eq!(Response::for_heat(3), Response::MultiUnit);
        assert_eq!(Response::for_heat(5), Response::MultiUnit);
        assert_eq!(Response::for_heat(6), Response::Swat);
        assert_eq!(Response::for_heat(99), Response::Swat);
        assert_eq!(Response::Swat.units(), 3);
        // ONE petty act can never reach the top rung — the ladder's whole shape.
        assert_ne!(
            Response::for_heat(ActKind::Carjack.heat()),
            Response::Swat,
            "a single carjack brought a tactical van"
        );
        let drawn: Vec<u8> = [0u32, 1, 2, 3, 6, 10, 15, 40]
            .iter()
            .map(|h| stars(*h))
            .collect();
        println!("heat 0/1/2/3/6/10/15/40 -> stars {drawn:?}");
        assert_eq!(drawn, vec![0, 1, 1, 2, 3, 4, 5, 5]);
    }

    /// **THE NIGHT, THE DISTANCE AND THE TABLE THEY MAKE.**
    #[test]
    fn a_sighting_is_worth_less_far_away_and_less_at_night() {
        assert_eq!(sight_factor(RECOGNITION_RANGE_M, false), 0.0);
        assert_eq!(sight_factor(f64::NAN, false), 0.0);
        assert!((sight_factor(0.0, false) - 1.0).abs() < 1e-9);
        assert!((sight_factor(0.0, true) - NIGHT_RECOGNITION).abs() < 1e-9);
        assert!(sight_factor(20.0, false) > sight_factor(20.0, true));
        // The design in one assertion: a CAR at ten metres at night is still
        // recognised; a COAT at the same place and hour is not.
        let f = sight_factor(10.0, true);
        assert!(Channel::Vehicle.weight() * f >= RECOGNIZE_SCORE);
        assert!(Channel::Outfit.weight() * f < RECOGNIZE_SCORE);
        // **THE TABLE `Channel::weight` CLAIMS, MEASURED.** A prescription in a
        // doc comment that nothing computes is the thing this repository has
        // been wrong about twice; this prints the ranges and pins them.
        let reach = |w: f64, night: bool| -> f64 {
            // The distance at which `w * sight_factor(d, night)` falls to the
            // threshold, solved rather than searched.
            let n = if night { NIGHT_RECOGNITION } else { 1.0 };
            RECOGNITION_RANGE_M * (1.0 - RECOGNIZE_SCORE / (w * n)).max(0.0)
        };
        let both = 1.0 - (1.0 - Channel::Outfit.weight()) * (1.0 - Channel::Vehicle.weight());
        for (name, w) in [
            ("outfit", Channel::Outfit.weight()),
            ("vehicle", Channel::Vehicle.weight()),
            ("both", both),
        ] {
            println!(
                "{name:>8} is recognised inside {:.1} m by day and {:.1} m at night",
                reach(w, false),
                reach(w, true)
            );
        }
        // The three orderings the design rests on.
        assert!(reach(Channel::Vehicle.weight(), false) > reach(Channel::Outfit.weight(), false));
        assert!(
            reach(Channel::Outfit.weight(), true) < 2.0,
            "a coat works at night"
        );
        assert!(
            reach(Channel::Vehicle.weight(), true) > 10.0,
            "a car is invisible at night"
        );
        // …and the engine has ONE night: `NIGHT_CIRCUIT_H`'s.
        assert!(is_night(23.0));
        assert!(is_night(3.0));
        assert!(!is_night(14.0));
        assert!(!is_night(21.0));
    }

    /// **TWO CARS THAT LOOK ALIKE DESCRIBE ALIKE**, and a coat is not a car.
    #[test]
    fn a_vehicle_is_described_by_what_it_shows() {
        use crate::components::{Collider3D, ColliderShape3DKind, Material};
        use crate::math::{Color, Vec3d};
        let mut w = EcsWorld::new();
        let mut spawn = |g: Uuid, half: Vec3d, colour: Color| {
            let e = w.spawn_with_guid(g, "car", None);
            w.world_mut().entity_mut(e).insert((
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: half,
                    ..Default::default()
                },
                Material {
                    base_color: colour,
                    ..Default::default()
                },
            ));
        };
        let red = Color::new(0.8, 0.1, 0.1, 1.0);
        let blue = Color::new(0.1, 0.1, 0.8, 1.0);
        let van = Vec3d::new(1.1, 1.2, 2.8);
        spawn(guid(1), van, red);
        spawn(guid(2), van, red);
        spawn(guid(3), van, blue);
        spawn(guid(4), Vec3d::new(0.9, 0.5, 2.2), red);
        let d = |g: u128| vehicle_digest(&w, guid(g)).expect("a car");
        assert_eq!(d(1), d(2), "two identical vans described differently");
        assert_ne!(d(1), d(3), "colour is not in the description");
        assert_ne!(d(1), d(4), "size is not in the description");
        // Something that is not a body is not a car.
        assert_eq!(vehicle_digest(&w, guid(77)), None);
        // …and `describe` puts the two channels together for somebody on foot.
        set_appearance(&mut w, guid(9), Appearance { outfit: 4 });
        let seen = describe(&w, guid(9));
        assert_eq!(seen.outfit, Appearance { outfit: 4 }.digest());
        assert_eq!(seen.vehicle, None, "a pedestrian was given a car");
    }

    /// **THE TRACE IS EMPTY UNTIL SOMEBODY IS WANTED**, and it carries the
    /// evidence itself.
    #[test]
    fn the_profile_trace_folds_the_evidence_and_nothing_before_it() {
        let mut w = EcsWorld::new();
        assert!(profile_state_bytes(&w).is_empty());
        report_act(
            &mut w,
            &act(ActKind::Carjack, guid(1), 0xaaa, 10),
            Some(0xccc),
        );
        let two = profile_state_bytes(&w);
        assert_eq!(two.len(), PROFILE_TRACE_BYTES + 2 * EVIDENCE_TRACE_BYTES);
        // A DIFFERENT description on the same crime is a different world.
        let mut other = EcsWorld::new();
        report_act(
            &mut other,
            &act(ActKind::Carjack, guid(1), 0xbbb, 10),
            Some(0xccc),
        );
        assert_ne!(profile_state_bytes(&other), two);
        // …and so is a different rung on the ladder.
        let mut hot = EcsWorld::new();
        report_act(
            &mut hot,
            &act(ActKind::Killed, guid(1), 0xaaa, 10),
            Some(0xccc),
        );
        assert_ne!(profile_state_bytes(&hot), two);
    }

    /// **THE LEDGER IS BOUNDED**, and the file it drops is the coldest.
    #[test]
    fn the_thirty_third_file_replaces_the_coldest() {
        let mut w = EcsWorld::new();
        // One serious file first, then `MAX_PROFILES` petty ones.
        report_act(&mut w, &act(ActKind::Killed, guid(1), 0xaaa, 1), None);
        for i in 0..MAX_PROFILES as u128 {
            report_act(
                &mut w,
                &act(ActKind::Carjack, guid(100 + i), 0xaaa, 2 + i as u64),
                None,
            );
        }
        let res = crime_of(&w).expect("ledger");
        assert_eq!(res.profiles.len(), MAX_PROFILES);
        assert!(
            res.profiles.contains_key(&guid(1)),
            "the ledger dropped the killing and kept a carjack"
        );
        println!(
            "{} files kept of {} opened, {} evicted",
            res.profiles.len(),
            res.opened,
            res.cleared
        );
    }
}
