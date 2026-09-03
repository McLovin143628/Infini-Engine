//! **RECOGNITION** (wave EMS3) — the officer looks at somebody and decides
//! whether they are the person on the file.
//!
//! The `inf_physics` half of [`inf_ecs::crime`], and the fifth instance of this
//! crate's own split: everything here touches rapier or the ECS, and nothing
//! here decides anything. The channels, the weights, the freshness curve, the
//! severity ladder and the two `last_seen` writers are all on the other side of
//! that wall and are unit-tested without a world.
//!
//! # Why the ray is on THIS side, and why that is the whole reason for the split
//!
//! A recognition is *"can this officer see that person, and does that person
//! match the description"*. The second half has no identity in it and no world
//! in it and belongs in Ring 0's decider. The first half is
//! `PhysicsBridge3D::cast_ray_excluding` — the same primitive `resolve_shot`,
//! the audio occlusion path and WPN1's own witness pass use, so this is the
//! engine's existing answer to *"is there something between these two points"*
//! rather than a fifth one.
//!
//! # THE POLICE DO NOT CHEAT, and here is the applier's half of the law
//!
//! [`inf_ecs::crime::sight`] is the only door in this file that writes anything
//! into a profile, it is called from exactly one place, and that place is inside
//! the branch where the range gate passed, the ray came back clear and the score
//! cleared [`inf_ecs::crime::RECOGNIZE_SCORE`]. There is no other path from a
//! suspect's transform to the ledger — so a search converges on a place somebody
//! actually saw them, and on nothing else.
//!
//! The subject's position **is** read, to measure a distance and to aim a ray,
//! and that is not the cheat: an officer with a clear line of sight does know
//! where somebody is standing. The cheat would be *remembering* it without one.
//!
//! # What it costs
//!
//! `O(officers × files)` distance tests, both bounded by constants
//! ([`inf_ecs::dispatch::MAX_UNITS`] and [`inf_ecs::crime::MAX_PROFILES`]), and
//! at most [`MAX_RECOGNITION_RAYS`] rays a step — WPN1's own witness budget,
//! reused rather than raised, so the two passes together are bounded at 64.
//! **Zero of everything** on a level where nobody is wanted, which is every
//! level committed before this wave: one `get_resource` and an early return.
//!
//! # The clock, and a bound this wave inherits rather than fixes
//!
//! Every step here is `inf_ecs::traffic::steps` — the same counter
//! `step_witness` stamps an act with, because a freshness is a subtraction and
//! two clocks in one subtraction is a bug with a long fuse. That counter lives
//! on the traffic population, so **a level with no streets has no clock**: every
//! act is stamped `0`, the feed's forward read never advances and the whole
//! wanted system is inert. It is EMS2's crime feed's bound met one wave along,
//! it is stated rather than worked around, and `ems3_crime_gate` asserts the
//! clock actually moves so that no arm below can be certified against a frozen
//! one.

use glam::DVec3;
use std::collections::BTreeSet;
use uuid::Uuid;

use inf_ecs::crime::{self, Description};
use inf_ecs::dispatch;
use inf_ecs::EcsWorld;

use super::PhysicsBridge3D;

/// How many line-of-sight rays one recognition pass may cast.
///
/// **Thirty-two** — `MAX_ACTS_PER_STEP × MAX_OBSERVERS`, which is WPN1's own
/// witness budget, reused by name rather than re-derived. The two passes are the
/// same shape of work (a bounded number of "can A see B" questions) and giving
/// this one a bigger number would have been a second opinion about what a step
/// can afford to look at.
///
/// A pass that runs out simply stops looking, in `Guid` order, which is a
/// **refusal and not a queue**: an officer who did not get a ray this step gets
/// one next step, sixty times a second.
pub const MAX_RECOGNITION_RAYS: usize = 32;

/// How high a pair of eyes is, metres — `super::gameplay::MUZZLE_HEIGHT_M`,
/// which is where this engine already measures a chest from.
const EYE_HEIGHT_M: f64 = 1.4;

/// What one [`step_recognition`] did — the instrument's read, and the gate's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecognitionStats {
    /// Officers who looked.
    pub officers: usize,
    /// Open files they looked for.
    pub files: usize,
    /// `(officer, suspect)` pairs inside [`crime::RECOGNITION_RANGE_M`] — the
    /// pairs that were worth a ray.
    pub in_range: usize,
    /// Rays actually cast.
    pub rays: usize,
    /// Pairs the ray found a wall between. The falsifier for "the line of sight
    /// is doing something": a pass that recognised everybody through every wall
    /// reads zero here.
    pub blocked: usize,
    /// Pairs that had a clear view and **still** did not clear the threshold —
    /// the number the whole evasion clause is about.
    pub unrecognised: usize,
    /// Recognitions. `sight` was called exactly this many times.
    pub recognised: usize,
    /// Acts filed into the ledger this step.
    pub filed: usize,
    /// Files closed by the decay this step.
    pub closed: usize,
}

/// **Advance the wanted system one fixed step** — file what happened, let the
/// officers look, and let the heat fall.
///
/// Called from [`super::dispatch::step_dispatch`] and from nowhere else, so
/// there is one place in the engine where a description becomes a sighting.
///
/// The order is load-bearing:
///
/// 1. **file** the witness log's new acts, so a crime committed this step is on
///    a file before anybody is asked to recognise its author;
/// 2. **look** — every on-duty officer against every open file, gated by range,
///    then by a ray, then by the score;
/// 3. **decay** — heat falls for everybody nobody saw, which must be last or a
///    file refreshed in step 2 would be aged in the same step it was renewed.
pub fn step_recognition(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    step: u64,
) -> RecognitionStats {
    let mut stats = RecognitionStats::default();
    stats.filed = crime::file_new_acts(world);
    let wanted = crime::wanted(world);
    stats.files = wanted.len();
    if !wanted.is_empty() {
        look(world, bridge, &wanted, step, &mut stats);
    }
    stats.closed = crime::decay(world, step);
    stats
}

/// Every officer who is in a position to look, in `Guid` order.
///
/// **The duty roster, not the crowd.** An observer of an *act* is any pedestrian
/// (WPN1's `candidates_near`, which reads the population); an observer of a
/// *suspect* is a police officer, and a police officer in this engine is a crew
/// member `inf_ecs::dispatch::ensure_crew` spawned — deliberately **not** a
/// population record (`clear_dispatch`'s own note). So the two passes read two
/// different sets and neither is the other's.
///
/// Fire and ambulance crews are on the roster too and are filtered out here: a
/// paramedic kneeling at a patient is on duty and is not looking for anybody.
///
/// **The hero can never be in this list**, and it is worth saying rather than
/// leaving to be noticed: a player is not a crew member, is not on
/// `RespondersRes`, and could not be put there by anything short of a new door.
/// A wanted system in which the player's own eyes refresh the police's file
/// would be the omniscience this wave exists to remove, wearing a different hat.
fn officers(world: &EcsWorld) -> Vec<Uuid> {
    let Some(res) = dispatch::dispatch_of(world) else {
        return Vec::new();
    };
    let Some(fleet) = dispatch::fleet_of(world) else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = Vec::new();
    for (chassis, run) in &res.runs {
        // A unit sitting in its bay is not looking at the street. That is a
        // design choice and not an oversight: a station full of parked cruisers
        // that recognised everybody walking past would make the whole search
        // pointless, and a patrol that has been sent somewhere is the thing the
        // player is actually evading.
        if run.state == dispatch::UnitState::InStation {
            continue;
        }
        if fleet.units.get(chassis).map(|u| u.kind) != Some(dispatch::UnitKind::Police) {
            continue;
        }
        out.push(dispatch::crew_guid(*chassis));
    }
    out.sort_unstable();
    out
}

/// The look itself — the one place a score becomes a sighting.
fn look(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    wanted: &[Uuid],
    step: u64,
    stats: &mut RecognitionStats,
) {
    let eyes = officers(world);
    stats.officers = eyes.len();
    if eyes.is_empty() {
        return;
    }
    let night = crime::is_night(inf_ecs::sky::local_hour(world));
    // Gathered before anything is written, so the write-back never overlaps a
    // read of the ledger, and in `Guid` order both ways: two hosts look in the
    // same order and therefore spend the ray budget on the same pairs.
    let mut seen: Vec<(Uuid, DVec3, Description, f64)> = Vec::new();
    for officer in &eyes {
        let Some(eye) = eye_of(world, *officer) else {
            continue;
        };
        for suspect in wanted {
            if suspect == officer {
                continue;
            }
            let Some(at) = eye_of(world, *suspect) else {
                continue;
            };
            let to = at - eye;
            let d = to.length();
            if !d.is_finite() || d >= crime::RECOGNITION_RANGE_M {
                continue;
            }
            stats.in_range += 1;
            // **The channel half FIRST**, because it is arithmetic and a ray is
            // a broad-phase query: a suspect who changed their coat and ditched
            // their car scores zero, and spending a ray to confirm that they are
            // visible costs the budget an officer needs for somebody who is
            // still recognisable.
            let look = crime::describe(world, *suspect);
            let Some(file) = crime::profile_of(world, *suspect) else {
                continue;
            };
            let channels = crime::match_score(file, look, step);
            if channels <= 0.0 {
                stats.unrecognised += 1;
                continue;
            }
            if stats.rays >= MAX_RECOGNITION_RAYS {
                continue;
            }
            stats.rays += 1;
            if d > 1.0e-6 && blocked(world, bridge, *officer, *suspect, eye, to, d) {
                stats.blocked += 1;
                continue;
            }
            let score = channels * crime::sight_factor(d, night);
            if score < crime::RECOGNIZE_SCORE {
                stats.unrecognised += 1;
                continue;
            }
            seen.push((*suspect, at, look, score));
        }
    }
    // **`sight` is called HERE and nowhere else**, which is the applier's half
    // of the police-don't-cheat law: every write into a profile is downstream of
    // a range gate, a ray and a threshold, all three of which are above.
    for (suspect, at, look, score) in seen {
        let _ = score;
        if crime::sight(world, suspect, at, look, step) {
            stats.recognised += 1;
        }
    }
}

/// Whether something is between these two people.
///
/// `super::gameplay::step_witness`' rule verbatim, including its centimetre of
/// tolerance — a ray stopped short of the target has hit a wall, and a wall a
/// shot leaves through is about a centimetre of slack.
///
/// **The two vehicles are excluded**, and that is not a convenience: a suspect
/// sitting in a car is *seen through the windscreen*, and an officer looking out
/// of one is looking out of one. Without it a driver would be permanently
/// invisible behind the chassis they are inside — which is the exact case the
/// mandate's *"the vehicle they were driving"* channel exists for, and it would
/// have made that channel unreachable.
fn blocked(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    officer: Uuid,
    suspect: Uuid,
    eye: DVec3,
    to: DVec3,
    d: f64,
) -> bool {
    let mut exclude = BTreeSet::new();
    for who in [officer, suspect] {
        if let Some(c) = bridge.collider_of(who) {
            exclude.insert(c);
        }
        if let Some(chassis) = inf_ecs::witness::actor_vehicle(world, who) {
            if let Some(c) = bridge.collider_of(chassis) {
                exclude.insert(c);
            }
        }
    }
    bridge
        .world_mut()
        .cast_ray_excluding(eye, to / d, d, &exclude)
        .is_some_and(|h| h.toi < d - 0.01)
}

/// **Where somebody's eyes are**, world metres — the ONE positional door this
/// pass has.
///
/// `super::dispatch::chassis_at`'s shape and its reason: a body the world has
/// answers from its own transform, and one the world does not — a `Dormant`
/// crowd agent, which has a population record and no entity at all — answers
/// from the record. A pass that only read entities would have made every distant
/// pedestrian unrecognisable *and* unable to be looked for, which is a wanted
/// system that stops working when you walk far enough from a camera.
///
/// A character's `Transform` is its capsule **centre**, which is already about
/// chest height, so it is used as-is; a population record is a pair of feet and
/// is lifted by [`EYE_HEIGHT_M`].
fn eye_of(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    if let Some(e) = world.entity_of(guid) {
        if let Some(t) = world
            .world()
            .get::<inf_ecs::components::Transform>(e)
            .map(|t| t.translation.to_dvec3())
        {
            if t.is_finite() {
                return Some(t);
            }
        }
    }
    let pop = world
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()?;
    let rec = pop.records.get(&guid)?;
    rec.last
        .is_finite()
        .then(|| rec.last + DVec3::Y * EYE_HEIGHT_M)
}
