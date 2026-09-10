//! **THE FIRING POLICY, APPLIED** (wave WPN2e) — the pass that turns a criminal
//! profile into an officer with a weapon pointed at somebody, and the **first
//! shipped caller** of [`super::gameplay::npc_aim_at`] and
//! [`super::gameplay::equip_weapon`].
//!
//! The `inf_physics` half of [`inf_ecs::engage`], and the sixth instance of this
//! crate's own split. Everything here touches rapier or the ECS; **nothing here
//! decides anything.** The ladder, the cadence, the cone and the
//! police-don't-cheat gate are all on the other side of that wall and are
//! unit-tested without a world.
//!
//! # THE POLICE DO NOT CHEAT — the applier's half of the law
//!
//! Two facts have to be true before a single `npc_aim_at` call is made, and both
//! of them are arguments to [`inf_ecs::engage::may_engage`]:
//!
//! 1. **the file's own [`last_seen`](inf_ecs::crime::Profile::last_seen)** is
//!    inside [`inf_ecs::engage::ENGAGE_RANGE_M`] of the officer and is no older
//!    than [`inf_ecs::engage::TRAIL_STALE_STEPS`]. That field has exactly two
//!    writers, both behind a witness or a recognition ray, and it is private to
//!    `inf_ecs::crime` — so an officer's *destination* can only ever be a place
//!    somebody actually saw the suspect;
//! 2. **a ray this step came back clear** — [`look_along`], which is
//!    `super::crime::blocked`'s rule with the hit's owner kept.
//!
//! The suspect's real transform is read to AIM the ray, exactly as
//! `super::crime::look` reads it, and `super::crime`'s own header says why that
//! is not the cheat: an officer with a clear line of sight does know where
//! somebody is standing. The cheat would be *remembering* it without one, and
//! there is nowhere in this file to put that.
//!
//! # Where it sits in the step, and why the trigger lands the same step
//!
//! Inside `step_gameplay`, **immediately before `step_weapons`**. That ordering
//! is the whole reason an officer's decision and its round are one step and not
//! two: `npc_aim_at` writes `MovementRuntime::want_attack` as a LEVEL, and
//! `step_weapons` is the very next thing that reads it.
//!
//! It reads the cover state the *movement* step wrote this step (phase 8, seven
//! phases earlier), so the peek it fires through is the lean the capsule is
//! actually in rather than the one the cover pass will ask for at the bottom of
//! this phase.
//!
//! # What it costs, and the zero
//!
//! `O(police units x open files)` distance tests — both bounds constants
//! ([`inf_ecs::dispatch::MAX_UNITS`] 64 and [`inf_ecs::crime::MAX_PROFILES`] 32)
//! — and at most [`inf_ecs::engage::NPC_ENGAGE_RAYS_PER_STEP`] rays a step.
//!
//! **Zero of everything on a level where nobody is wanted**: one `get_resource`
//! for the ledger, an empty `wanted` list, and an early return before the duty
//! roster is even gathered. That is every level committed before this wave, and
//! it is what `wpn2e_gate::a_unit_that_is_not_engaged_runs_no_engage_rays`
//! measures.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, MovementMode};
use inf_ecs::crime;
use inf_ecs::dispatch::{self, UnitKind};
use inf_ecs::engage::{self, EngageRes, Posture, UnitEngagement};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;

use super::PhysicsBridge3D;

/// What one [`step_engage`] did — the instrument's read, and the gate's.
///
/// Every field is an **engagement counter** rather than a "the pass ran" flag
/// (the EMS2 law): a gate that cannot tell "the officers held their fire" from
/// "no officer was ever in range" certifies a no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngageStats {
    /// Armed police units the pass looked at — the cost.
    pub units: usize,
    /// Open files it looked for.
    pub files: usize,
    /// `(unit, file)` pairs whose LAST-SEEN was inside
    /// [`inf_ecs::engage::ENGAGE_RANGE_M`] — the pairs worth a ray.
    pub candidates: usize,
    /// **Engage rays actually cast.** ZERO on a step where nobody is wanted,
    /// which is the budget arm's whole claim.
    pub rays: usize,
    /// Rays a wall (or a car) stopped. **No line of sight, so no aim** — the
    /// law's own falsifier: a pass that aimed through walls reads zero here.
    pub blocked: usize,
    /// Pairs the trail-age rule refused — the file is warm, the officer is in
    /// range, and nobody has seen the suspect for three seconds.
    pub cold_trail: usize,
    /// `npc_aim_at` calls that returned `true` — weapons actually pointed.
    pub aimed: usize,
    /// Of those, the ones taken at [`Posture::Warn`] — a weapon pointed at
    /// somebody with the trigger closed **because that is what the rung does**,
    /// and not because a discipline rule or a cadence refused it. **The patrol
    /// rung's whole behaviour**, counted.
    ///
    /// Deliberately NOT "every aim whose trigger was shut": at
    /// [`Posture::FireOnSight`] most steps of a cadence cycle have the trigger
    /// shut, and counting those as warnings would make a SWAT response read as
    /// a patrol.
    pub warned: usize,
    /// Of those, the ones whose trigger was OPEN.
    pub triggers: usize,
    /// Shots refused because a **responder** was inside the cone, nearer than
    /// the target — a unit behind another unit holds fire.
    pub friendly_holds: usize,
    /// Shots refused because a **civilian** was in the way: inside the cone, or
    /// the first thing the engage ray hit.
    pub civilian_holds: usize,
    /// Shots refused because the posture may not fire yet —
    /// [`Posture::ReturnFire`] that has not been fired upon.
    pub posture_holds: usize,
    /// Shots refused because this unit's cadence window is shut.
    pub cadence_holds: usize,
    /// Units firing BLIND from cover this step — in cover, not leaned out, and
    /// pulling anyway.
    pub blind: usize,
    /// Weapons issued at arrival this step.
    pub equipped: usize,
    /// Units a round came past this step — what [`Posture::ReturnFire`] reads
    /// next step. Written by [`note_incoming`].
    pub incoming: usize,
    /// **Triggers the pass CLOSED on a unit it did not engage** — units that
    /// were firing and have stopped.
    ///
    /// Its own counter because it measures a defect this wave's own gate found:
    /// `npc_aim_at` writes a LEVEL, and a level nobody lowers stays high. An
    /// officer whose suspect walks behind a wall is not visited by the loop
    /// above at all, so without this it would go on firing at nothing for ever
    /// — 258 rounds over a run in which the policy decided to fire zero times.
    pub released: usize,
}

/// **Advance the firing policy one fixed step.**
///
/// The sequence, all of it a pure function of sim state:
///
/// 1. the open files, and the hottest rung any of them is at — the town's
///    posture ([`inf_ecs::engage::posture_for`]);
/// 2. the armed police units, from the same duty roster
///    [`super::crime::officers`] reads, so an officer who is looking and an
///    officer who is shooting are the same set;
/// 3. per pair, the range gate against the file's **last-seen**, then a ray,
///    then [`inf_ecs::engage::may_engage`];
/// 4. the aim, through [`super::gameplay::npc_aim_at`], with the trigger
///    decided by the posture, the cadence and the two discipline rules.
pub fn step_engage(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, step: u64) -> EngageStats {
    let mut stats = EngageStats::default();
    let wanted = crime::wanted(world);
    if wanted.is_empty() {
        // **Nobody is wanted: nothing at all happens** — with ONE exception,
        // which is the whole of `super::gameplay::npc_set_trigger`'s reason.
        //
        // No roster walk, no distance test, no ray. But a unit that WAS firing
        // when the last file closed is holding a trigger nobody is going to
        // lower, so every engagement in the ledger is released before the ledger
        // is dropped. A town that goes cold with an officer mid-burst would
        // otherwise have an officer emptying its magazine into an empty street
        // for the rest of the session.
        if world.world().get_resource::<EngageRes>().is_some() {
            let firing: Vec<Uuid> = engage::engage_of(world)
                .map(|r| r.units.keys().copied().collect())
                .unwrap_or_default();
            for unit in firing {
                if super::gameplay::npc_set_trigger(world, unit, false) {
                    stats.released += 1;
                }
            }
            world.world_mut().insert_resource(EngageRes::default());
        }
        return stats;
    }
    stats.files = wanted.len();
    // **The town's posture, from the hottest open file.** One rung for the whole
    // response, which is `super::cover::hottest_response`'s own rule and the
    // reason SWAT cover and SWAT fire cannot disagree about which rung it is.
    let posture = engage::posture_for(
        wanted
            .iter()
            .filter_map(|s| crime::profile_of(world, *s).map(|p| p.response()))
            .max()
            .unwrap_or(crime::Response::Cold),
    );
    if !posture.aims() {
        return stats;
    }
    // The suspects, gathered before anything is written — `super::crime::look`'s
    // own shape, and its reason: the write-back must never overlap a read of the
    // ledger, and two hosts must walk the same pairs in the same order.
    let subjects: Vec<(Uuid, DVec3, Vec3d, u64)> = wanted
        .iter()
        .filter_map(|suspect| {
            let at = super::crime::eye_of(world, *suspect)?;
            let file = crime::profile_of(world, *suspect)?;
            Some((
                *suspect,
                at,
                Vec3d::from_dvec3(file.last_seen()),
                file.trail_age(step),
            ))
        })
        .collect();
    let officers = armed_officers(world);
    stats.units = officers.len();
    if officers.is_empty() || subjects.is_empty() {
        return stats;
    }
    // Everybody who must not be shot: the duty roster (friendlies) and the
    // crowd's own records (civilians). Two sets because they are two rules and
    // two counters — see `EngageStats::friendly_holds`.
    let friendlies: BTreeSet<Uuid> = dispatch::responders(world).into_iter().collect();
    let mut res = world
        .world_mut()
        .remove_resource::<EngageRes>()
        .unwrap_or_default();
    // A unit that has gone away forgets what it was doing — `NpcCoverRes`' own
    // prune rule, so a guid that comes back does not inherit a stale target.
    let live: BTreeSet<Uuid> = officers.iter().map(|(g, _)| *g).collect();
    res.units.retain(|g, _| live.contains(g));

    let mut decisions: Vec<(Uuid, Uuid, bool, bool)> = Vec::new();
    // Units the pass looked at and decided NOT to engage — their triggers come
    // down below.
    let mut release: Vec<Uuid> = Vec::new();
    for (officer, _class) in &officers {
        let Some(eye) = super::crime::eye_of(world, *officer) else {
            release.push(*officer);
            continue;
        };
        let here = Vec3d::from_dvec3(eye);
        let fired_upon = res
            .units
            .get(officer)
            .is_some_and(|slot| slot.fired_upon(step));
        let mut engaged: Option<(Uuid, DVec3, Sight, Option<DVec3>)> = None;
        for (suspect, at, last_seen, trail_age) in &subjects {
            if suspect == officer {
                continue;
            }
            // ── the RANGE gate, on the FILE and not on the body. The one place
            //    this pass could have started reading a transform nobody looked
            //    at, and the reason it takes `last_seen` rather than `at`.
            //
            //    `worth_a_ray` is `may_engage` WITHOUT the sight argument, and
            //    it exists so this call site never contains the placeholder
            //    `true` the law arm's own mutation would be indistinguishable
            //    from.
            if !engage::worth_a_ray(posture, here, *last_seen, *trail_age) {
                // Distinguish the two refusals so the counters mean something:
                // a cold trail is a suspect the police have lost, and a range
                // failure is one they have not reached.
                if *trail_age > engage::TRAIL_STALE_STEPS {
                    stats.cold_trail += 1;
                }
                continue;
            }
            stats.candidates += 1;
            if stats.rays >= engage::NPC_ENGAGE_RAYS_PER_STEP {
                // A refusal and not a queue: an officer that did not get a ray
                // this step gets one next step, sixty times a second.
                continue;
            }
            stats.rays += 1;
            let sight = look_along(world, bridge, *officer, *suspect, eye, *at);
            // **THE LAW, as one call.** The `line_of_sight` argument is the ray
            // one line up and nothing else: a wall means no sight, so no aim.
            //
            // A PERSON on the line does not break the officer's sight — you can
            // see somebody past a pedestrian — but it absolutely breaks the
            // firing line, and that distinction is the whole of the discipline
            // rule below.
            if !engage::may_engage(posture, here, *last_seen, *trail_age, sight != Sight::Wall) {
                stats.blocked += 1;
                // **...AND BLIND FIRE IS THE ONE THING A UNIT MAY STILL DO.**
                //
                // No line of sight means no AIM -- that is the law, and it holds:
                // nothing below reads the suspect's transform. What a unit in
                // cover may do is put rounds over its OWN WALL, in the direction
                // that wall faces, because it knows a suspect was last seen out
                // there (`last_seen`, in range, fresh) and because a surface it
                // is touching tells it which way "out there" is.
                //
                // It is only available FROM COVER. A unit standing in the open
                // with a wall between it and a suspect is not suppressing
                // anything; it is shooting a wall.
                if let Some(out) = blind_out(world, *officer) {
                    engaged = Some((*suspect, *at, sight, Some(out)));
                    break;
                }
                continue;
            }
            engaged = Some((*suspect, *at, sight, None));
            break;
        }
        let Some((suspect, at, sight, blind_out)) = engaged else {
            // **Seen nobody: the slot forgets its target AND the trigger comes
            // down.** The second half is not tidying — see
            // `super::gameplay::npc_set_trigger`: `want_attack` is a level, this
            // officer will not be visited again while it can see nothing, and a
            // level nobody lowers stays high.
            if let Some(slot) = res.units.get_mut(officer) {
                slot.target = Uuid::nil();
            }
            release.push(*officer);
            continue;
        };
        // ── THE TRIGGER, and every reason it stays shut.
        let mut hold = false;
        if !posture.may_fire(fired_upon) {
            // A `Warn` is not a hold: pointing a weapon and not firing is what
            // that rung DOES, and counting it as a refusal would make the patrol
            // behaviour look like a bug in the discipline rules.
            if posture != Posture::Warn {
                stats.posture_holds += 1;
            }
            hold = true;
        }
        // **WHAT THE DISCIPLINE RULES ARE MEASURED AGAINST.** For an aimed shot
        // that is the suspect; for a blind one it is a point out along the
        // cover's own normal at the engagement range, because that is where the
        // rounds are going. A unit spraying over its wall into a colleague
        // standing in front of it is exactly as wrong as one doing it on
        // purpose.
        let aim_at = match blind_out {
            Some(out) => Vec3d::from_dvec3(eye + out * engage::ENGAGE_RANGE_M),
            None => Vec3d::from_dvec3(at),
        };
        if !hold && friendly_in_cone(world, &friendlies, *officer, here, aim_at).is_some() {
            stats.friendly_holds += 1;
            hold = true;
        }
        // A blind shot's ray already came back stopped, so `sight` says nothing
        // useful about who is in ITS line -- the cone walk is the whole test.
        let line = if blind_out.is_some() {
            Sight::Clear
        } else {
            sight
        };
        if !hold && civilian_in_the_way(world, line, *officer, suspect, eye, here, aim_at) {
            stats.civilian_holds += 1;
            hold = true;
        }
        if !hold && !engage::fire_window(*officer, step) {
            stats.cadence_holds += 1;
            hold = true;
        }
        let blind = blind_out.is_some();
        decisions.push((*officer, suspect, !hold, blind));
        let slot = res.units.entry(*officer).or_default();
        slot.target = suspect;
        slot.seen_step = step;
        if hold {
            slot.holds = slot.holds.saturating_add(1);
            res.holds = res.holds.saturating_add(1);
        } else {
            slot.shots = slot.shots.saturating_add(1);
            res.shots = res.shots.saturating_add(1);
        }
    }
    world.world_mut().insert_resource(res);
    // **`npc_aim_at` is called HERE and nowhere else**, which is the applier's
    // half of the law: every one of these is downstream of a range gate on a
    // remembered position, a ray, and `may_engage`.
    for unit in release {
        if super::gameplay::npc_set_trigger(world, unit, false) {
            stats.released += 1;
        }
    }
    for (officer, suspect, trigger, blind) in decisions {
        // **BLIND FIRE WRITES THE TRIGGER AND NOTHING ELSE.** `npc_aim_at`
        // resolves `strike_point` off the target's transform, and a shooter with
        // no line of sight must not read one -- see
        // `super::gameplay::npc_set_trigger`. Its body is already pointed at the
        // PLACE the gunfire came from by `super::cover::aim_and_lean`, which
        // heard it rather than saw it.
        if blind {
            super::gameplay::npc_set_trigger(world, officer, trigger);
            if trigger {
                stats.triggers += 1;
                stats.blind += 1;
            }
            continue;
        }
        if !super::gameplay::npc_aim_at(world, officer, suspect, trigger) {
            continue;
        }
        stats.aimed += 1;
        if trigger {
            stats.triggers += 1;
        } else if posture == Posture::Warn {
            stats.warned += 1;
        }
    }
    stats
}

/// **Every police unit that is out of its station AND has a weapon**, in `Guid`
/// order.
///
/// The roster is [`super::crime::officers`] — the same set the recognition pass
/// looks with — so an officer who can see you and an officer who can shoot you
/// are the same person by construction, and a paramedic is out of both for the
/// same reason.
///
/// The weapon is [`inf_ecs::weapon::equipped_def`], which is the folded
/// definition an attachment may have changed: a policy that read the base row
/// would be deciding with a different weapon from the one that fires.
fn armed_officers(world: &EcsWorld) -> Vec<(Uuid, inf_ecs::weapon::WeaponClass)> {
    super::crime::officers(world)
        .into_iter()
        .filter_map(|g| {
            let (_, def) = inf_ecs::weapon::equipped_def(world, g)?;
            (!def.is_melee()).then_some((g, def.audio_class()))
        })
        .collect()
}

/// What the engage ray found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sight {
    /// Nothing at all between the two of them.
    Clear,
    /// A wall, a car, a crate — the officer cannot see the suspect.
    Wall,
    /// A person, who is not the suspect. The officer CAN see; it must not fire.
    Person(Uuid),
}

/// **Cast the engage ray** — `super::crime::blocked`'s rule, with the hit's
/// owner kept.
///
/// Two differences from the recognition ray, and both are deliberate:
///
/// * it asks [`super::CastTargets::AllSolid`] rather than `Fixed`, because it is
///   answering *"will the round get there"* and a round has seen dynamic bodies
///   since wave WPN2a. A policy that cleared a shot the bullet stops in a parked
///   car would be a policy that disagrees with the ballistics;
/// * it keeps `hit.collider` and resolves it through
///   [`super::gameplay::hit_owner`], so a person on the line is told from a wall
///   on the line. `blocked` answers a `bool` and cannot.
///
/// The exclusions are the recognition ray's: both bodies and both of their
/// vehicles, so a suspect sitting in a car is seen through the windscreen and an
/// officer looking out of one is looking out of one.
fn look_along(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    officer: Uuid,
    suspect: Uuid,
    eye: DVec3,
    at: DVec3,
) -> Sight {
    let to = at - eye;
    let d = to.length();
    if !d.is_finite() || d <= 1.0e-6 {
        return Sight::Clear;
    }
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
    let hit =
        bridge
            .world_mut()
            .cast_ray_where(eye, to / d, d, &exclude, super::CastTargets::AllSolid);
    // A centimetre of tolerance — the wall a shot leaves through, and
    // `super::crime::blocked`'s own slack.
    let Some(hit) = hit.filter(|h| h.toi < d - 0.01) else {
        return Sight::Clear;
    };
    match super::gameplay::hit_owner(bridge, hit.collider) {
        Some(g) if is_person(world, g) => Sight::Person(g),
        _ => Sight::Wall,
    }
}

/// Whether this guid is a person — a body with a `CharacterMovement`, or a crowd
/// record the world has not built an entity for.
///
/// `super::gameplay::is_flesh`'s question asked without the `Health` half: a
/// pedestrian who has never been hurt has no `Health` component at all (the LAZY
/// health path), and a discipline rule that only protected people who had
/// already been shot would be the wrong way round.
fn is_person(world: &EcsWorld, guid: Uuid) -> bool {
    if world
        .entity_of(guid)
        .is_some_and(|e| world.world().get::<CharacterMovement>(e).is_some())
    {
        return true;
    }
    world
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .is_some_and(|p| p.records.contains_key(&guid))
}

/// **Is one of this officer's own people in the cone** — the friendly-fire rule.
///
/// Over the duty roster and not the crowd: a "friendly" is somebody wearing the
/// same uniform, and the crowd is the other rule. It answers WHO so a failure
/// message can name them.
fn friendly_in_cone(
    world: &EcsWorld,
    friendlies: &BTreeSet<Uuid>,
    officer: Uuid,
    here: Vec3d,
    at: Vec3d,
) -> Option<Uuid> {
    friendlies.iter().copied().find(|other| {
        if *other == officer {
            return false;
        }
        super::crime::eye_of(world, *other).is_some_and(|p| {
            engage::in_cone(here, at, Vec3d::from_dvec3(p), engage::FRIENDLY_CONE_DEG)
        })
    })
}

/// **Is a civilian in the way** — the crowd's half of the discipline rule, both
/// tests.
///
/// 1. **first on the line**: the engage ray's own hit, handed in rather than
///    re-cast, because a second cast here would spend a ray outside
///    [`inf_ecs::engage::NPC_ENGAGE_RAYS_PER_STEP`] and make the budget a lie;
/// 2. **in the cone**, nearer than the target: a pedestrian 0.4 m off the line
///    at twenty metres is not hit by an infinitely thin ray and is very much hit
///    by a shot with a spread cone.
///
/// The crowd walk is bounded by [`inf_ecs::engage::ENGAGE_RANGE_M`] through
/// [`inf_ecs::witness::candidates_near`], which is the same door WPN1's witness
/// pass uses, so there is one answer in this engine to *"who is standing near
/// here"*.
#[allow(clippy::too_many_arguments)]
fn civilian_in_the_way(
    world: &EcsWorld,
    sight: Sight,
    officer: Uuid,
    suspect: Uuid,
    eye: DVec3,
    here: Vec3d,
    aim_at: Vec3d,
) -> bool {
    if let Sight::Person(who) = sight {
        // A colleague on the line is the friendly rule's, already asked; anybody
        // else is a civilian standing in front of a police weapon.
        if who != suspect {
            return true;
        }
    }
    inf_ecs::witness::candidates_near(world, eye, engage::ENGAGE_RANGE_M)
        .into_iter()
        .any(|(guid, feet)| {
            if guid == officer || guid == suspect {
                return false;
            }
            let chest = feet + DVec3::Y * super::gameplay::MUZZLE_HEIGHT_M;
            engage::in_cone(
                here,
                aim_at,
                Vec3d::from_dvec3(chest),
                engage::FRIENDLY_CONE_DEG,
            )
        })
}

/// **Which way this unit would fire BLIND**, or `None` if it cannot.
///
/// In cover, not leaned out of it, and against a surface whose normal is real.
/// `super::cover::step_npc_cover` runs the peek duty cycle and writes
/// `want_aim`; a unit whose trigger comes up in the *shut* half of that cycle
/// does not stand up to take the shot, it puts the weapon over the top.
///
/// The vector is the cover's **outward** planar normal (`CoverState::normal`
/// points back AT the character) and it is what the discipline cone is measured
/// along. The ROUND's own direction is resolved a second time, from the same
/// field, in `super::gameplay::step_weapons`: one rule, two readers, and the
/// second one is the hero's as well as the NPC's.
fn blind_out(world: &EcsWorld, unit: Uuid) -> Option<DVec3> {
    let e = world.entity_of(unit)?;
    let cm = world.world().get::<CharacterMovement>(e)?;
    if cm.mode != MovementMode::Cover || cm.runtime.want_aim || !cm.runtime.cover.active {
        return None;
    }
    let n = cm.runtime.cover.normal;
    let out = DVec3::new(-n.x, 0.0, -n.z);
    let len = out.length();
    (len.is_finite() && len > 1.0e-9).then(|| out / len)
}

/// **Note that a round came past these units** — what [`Posture::ReturnFire`]
/// reads next step.
///
/// Called from `step_gameplay` beside `super::cover::step_npc_cover`, on the
/// **same coalesced source list**, because they are the same fact seen from two
/// sides: the responders that take cover are the responders that have been shot
/// at. Sharing the list is also what makes the cost bound one bound rather than
/// two — `panic_sources` is already capped at `MAX_PANIC_SOURCES`.
///
/// A unit that is under fire but has no engagement slot **gets one**: being shot
/// at is exactly the thing that has to be remembered before the officer has a
/// target, or a `MultiUnit` response would need to already be firing to start
/// firing.
///
/// Inert on every step nothing was fired on: the source list is empty and the
/// function returns before it touches the ledger.
pub fn note_incoming(world: &mut EcsWorld, sources: &[DVec3], radius_m: f64, step: u64) -> usize {
    if sources.is_empty() {
        return 0;
    }
    let responders = dispatch::responders(world);
    if responders.is_empty() {
        return 0;
    }
    let places: Vec<Vec3d> = sources.iter().map(|s| Vec3d::from_dvec3(*s)).collect();
    let mut under: Vec<Uuid> = Vec::new();
    for unit in responders {
        let Some(here) = super::crime::eye_of(world, unit) else {
            continue;
        };
        if inf_ecs::cover::under_fire(Vec3d::from_dvec3(here), &places, radius_m).is_some() {
            under.push(unit);
        }
    }
    if under.is_empty() {
        return 0;
    }
    let mut res = world
        .world_mut()
        .remove_resource::<EngageRes>()
        .unwrap_or_default();
    for unit in &under {
        res.units
            .entry(*unit)
            .or_insert_with(UnitEngagement::default)
            .fired_upon_step = step;
    }
    world.world_mut().insert_resource(res);
    under.len()
}

/// **Issue a weapon to a crew that has just got out of its car** — the first
/// shipped caller of [`super::gameplay::equip_weapon`].
///
/// Called from `super::dispatch::arrive` and from nowhere else, so there is one
/// place in the engine where an officer becomes armed.
///
/// * a **police** crew only. A paramedic kneeling at a patient and a fire crew
///   on a branch are on the duty roster and are not armed, which is EMS1's own
///   distinction and the reason [`UnitKind`] is an argument;
/// * the row comes from [`inf_ecs::engage::crew_weapon_ids`] — a sidearm at
///   `Patrol` and `MultiUnit`, a rifle at `Swat` — walked against the level's
///   own [`inf_ecs::item::ItemDefs`], so the same policy arms an officer on the
///   island (`glock_17`) and in the `phase30-gameplay` fixture (`pistol`) and
///   arms nobody on a level that defines no weapons;
/// * a **`Cold`** town issues nothing, so a level that has never had a crime
///   never equips anything and steps the bytes it stepped before this wave.
///
/// Answers which row was issued, or `None` for every refusal — refusals as
/// values, all the way down.
pub fn arm_crew(world: &mut EcsWorld, crew: Uuid, kind: UnitKind) -> Option<&'static str> {
    if kind != UnitKind::Police {
        return None;
    }
    // The town's own rung, from the hottest open file — `step_engage`'s rule, so
    // the weapon in the hand and the posture behind it cannot disagree.
    let response = crime::wanted(world)
        .into_iter()
        .filter_map(|s| crime::profile_of(world, s).map(|p| p.response()))
        .max()
        .unwrap_or(crime::Response::Cold);
    let ids = engage::crew_weapon_ids(response);
    if ids.is_empty() {
        return None;
    }
    let defs = inf_ecs::item::item_defs(world)?.clone();
    let id = ids
        .iter()
        .copied()
        .find(|id| defs.get(&inf_ecs::item::canonical_id(id)).is_some())?;
    // Already carrying it: do not hand out a second one, and do not re-equip —
    // `equip_weapon` reinstalls a full magazine, which would make an officer's
    // ammunition infinite by arriving twice.
    if inf_ecs::weapon::equipped_def(world, crew).is_some_and(|(have, _)| have == id) {
        return Some(id);
    }
    // `item::give` inserts an `Inventory` for a body that has none — a crew
    // member is `crowd::spawn_body` and carries nothing — and answers what it
    // could NOT fit, so `0` left over is the whole of "it went in".
    if inf_ecs::item::give(world, crew, id, 1) != 0 {
        return None;
    }
    if !super::gameplay::equip_weapon(world, crew, id) {
        return None;
    }
    let mut res = world
        .world_mut()
        .remove_resource::<EngageRes>()
        .unwrap_or_default();
    res.equips = res.equips.saturating_add(1);
    world.world_mut().insert_resource(res);
    Some(id)
}

/// **What every engagement is aimed at right now**, `(unit, target)` in `Guid`
/// order — the read a gate and the demo loop share.
///
/// Reads the ledger and nothing else, so a caller cannot accidentally ask the
/// world where somebody is.
pub fn engagements(world: &EcsWorld) -> BTreeMap<Uuid, Uuid> {
    engage::engage_of(world)
        .map(|r| {
            r.units
                .iter()
                .filter(|(_, u)| !u.target.is_nil())
                .map(|(g, u)| (*g, u.target))
                .collect()
        })
        .unwrap_or_default()
}
