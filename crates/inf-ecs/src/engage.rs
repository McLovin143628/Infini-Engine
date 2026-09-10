//! **THE FIRING POLICY** (wave WPN2e) — *who* a responding unit shoots at, when
//! it pulls, and every reason it does not.
//!
//! The Ring-0 half of the applier `inf_physics::d3::engage`, and the sixth
//! instance of this crate's own split: [`crate::dispatch`] decides and
//! `d3::dispatch` applies; [`crate::crime`] decides and `d3::crime` applies;
//! this decides and `d3::engage` applies. **Nothing here touches a world.**
//! Every function is a pure function of numbers a caller already has, so the
//! ladder, the cadence, the cone and the range gate are unit-tested without a
//! crowd.
//!
//! # THE POLICE DO NOT CHEAT, and this is the decider's half of the law
//!
//! There are exactly two doors from *"where is the suspect"* to *"aim there"*,
//! and both of them are above this module:
//!
//! 1. [`crate::crime::Profile::last_seen`] — **what the police remember**,
//!    written only by [`crate::crime::report_act`] and [`crate::crime::sight`],
//!    both of which are behind a witness or a recognition ray;
//! 2. a line-of-sight ray the applier casts *this step*, which is
//!    `d3::crime::blocked`'s rule verbatim.
//!
//! [`may_engage`] takes **both** as arguments and answers `false` if either is
//! missing, and the applier calls it before it resolves a shooter's aim. A unit
//! with no LOS and a cold file therefore never aims, and the falsifier is
//! written down rather than asserted: replace the `last_seen` argument at the
//! call site with the suspect's real transform and
//! `wpn2e_gate::a_unit_with_no_sight_and_no_trail_never_aims` goes red.
//!
//! # Why the ladder is BEHAVIOUR and not a unit count
//!
//! [`crate::crime::Response`] has meant *how many cars* since EMS2
//! ([`crate::crime::Response::units`]). What a player sees is not a count: it is
//! whether the officer who got out of the car shouts, or shoots back, or shoots
//! first. [`posture_for`] is that second reading of the same rung — one
//! function, four arms, no new state — so a town that escalates escalates in the
//! only way a player can perceive.
//!
//! | rung | units | posture | what a player sees |
//! |---|---|---|---|
//! | `Cold` | 0 | [`Posture::Hold`] | nobody is looking |
//! | `Patrol` | 1 | [`Posture::Warn`] | one officer, weapon out, pointed at you, **no shot** |
//! | `MultiUnit` | 2 | [`Posture::ReturnFire`] | two units that shoot **once you do** |
//! | `Swat` | 3 | [`Posture::FireOnSight`] | they shoot the moment they see you |
//!
//! # What it costs
//!
//! Nothing that allocates. [`EngageRes`] is one `BTreeMap` keyed on the units
//! that are actually engaged — **empty on every level where nobody is wanted**,
//! which is every level committed before this wave — and it is a resource rather
//! than a component for [`crate::cover::NpcCoverRes`]' reason: it is the pass's
//! own memory, it is outside `ScenePersist::Memory` by construction, and it is
//! rebuilt from the world on every step that matters.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;
use uuid::Uuid;

use crate::crime::Response;
use crate::math::Vec3d;

// ── the ladder, as behaviour ────────────────────────────────────────────────

/// **What a responding unit does about the person on the file.**
///
/// Ordered from least to most violent so `max` over a squad is meaningful and so
/// a gate can assert that an escalation went UP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Posture {
    /// Weapon stays away. Nobody is wanted, or nobody can be seen.
    Hold,
    /// **Weapon out, pointed, trigger closed** — the warning.
    ///
    /// A real behaviour and not a null one: the officer's aim goes to the
    /// suspect and its body turns, so a player is looked at down a barrel and
    /// nothing happens. That is what one star buys.
    Warn,
    /// **Fires once fired upon** — the officer holds until a round comes past.
    ReturnFire,
    /// **Fires on sight.**
    FireOnSight,
}

impl Posture {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            Posture::Hold => "hold",
            Posture::Warn => "warn",
            Posture::ReturnFire => "return-fire",
            Posture::FireOnSight => "fire-on-sight",
        }
    }

    /// The byte this posture folds into a diagnostic. **Frozen, append-only** on
    /// [`crate::dispatch::UnitKind::as_u8`]'s terms.
    pub fn as_u8(self) -> u8 {
        match self {
            Posture::Hold => 0,
            Posture::Warn => 1,
            Posture::ReturnFire => 2,
            Posture::FireOnSight => 3,
        }
    }

    /// Whether this posture points a weapon at all.
    ///
    /// [`Posture::Hold`] is the only one that does not, and that is the
    /// difference between a level where the police have noticed you and one
    /// where they have not.
    pub fn aims(self) -> bool {
        !matches!(self, Posture::Hold)
    }

    /// **Whether this posture may pull the trigger**, given whether the unit has
    /// been shot at recently.
    ///
    /// The whole of the ladder's firing rule as one expression, so that "a
    /// patrol never fires" is a property of a pure function rather than of a
    /// branch inside a pass.
    pub fn may_fire(self, fired_upon: bool) -> bool {
        match self {
            Posture::Hold | Posture::Warn => false,
            Posture::ReturnFire => fired_upon,
            Posture::FireOnSight => true,
        }
    }
}

/// **Which posture this rung of the response ladder is** — the ladder made
/// behavioural.
///
/// See the module header's table. It is a total function of the rung and takes
/// nothing else, which is what makes the three behaviours reproducible: a gate
/// sets a heat, reads a posture, and the two cannot disagree.
pub fn posture_for(response: Response) -> Posture {
    match response {
        Response::Cold => Posture::Hold,
        Response::Patrol => Posture::Warn,
        Response::MultiUnit => Posture::ReturnFire,
        Response::Swat => Posture::FireOnSight,
    }
}

// ── the bounds ──────────────────────────────────────────────────────────────

/// **How far a unit will engage from**, metres.
///
/// **Thirty-five.** Deliberately SHORTER than
/// [`crate::crime::RECOGNITION_RANGE_M`] (40 m), and the gap is the design: an
/// officer recognises somebody before it is willing to shoot at them, so there
/// is a band five metres deep in which the file is refreshed and no round is
/// fired. Without the gap a suspect would be recognised and shot on the same
/// step, and there would be no distance at which the police have seen you and
/// have not opened fire.
///
/// It is also inside every shipped weapon's own range, so the bound that stops a
/// shot is this policy and not the ballistics — which is what makes it a
/// *decision* a designer can move.
pub const ENGAGE_RANGE_M: f64 = 35.0;

/// **How close is too close to fire**, metres.
///
/// **Four.** The WPN2d audit's own arithmetic: a launcher's 4 000 J over an 8 m
/// blast radius is lethal to a 2 000 J body inside 2.34 m, so a unit that fired
/// a rocket at somebody standing next to it would kill itself. Rounded up to
/// four so the guard has margin over the exact figure and so it also covers the
/// case a rocket is not needed for — an officer firing a rifle through the
/// colleague standing in front of it.
///
/// A minimum engagement range is **policy, not physics**, which is why it is
/// here and not in [`crate::weapon::WeaponDef`]: the round would happily fly.
pub const MIN_ENGAGE_M: f64 = 4.0;

/// **How stale a trail may be and still be worth aiming at**, fixed steps.
///
/// **One hundred and eighty** — three seconds at 60 Hz. It is the police's
/// MEMORY and it is short on purpose: a suspect who breaks line of sight for
/// three seconds stops being a target and becomes a search, which is EMS3's
/// whole evasion clause continuing to work while somebody is shooting at it.
///
/// It is much shorter than [`crate::crime::EVIDENCE_COLD_STEPS`] (3 600) because
/// the two answer different questions: the file stays warm for a minute, and the
/// *aim* goes off in three seconds.
pub const TRAIL_STALE_STEPS: u64 = 180;

/// **How long a unit remembers being shot at**, fixed steps.
///
/// **Three hundred** — five seconds. The window [`Posture::ReturnFire`] reads,
/// and long enough that a `MultiUnit` response answered by one shot keeps
/// answering for a few bursts rather than going quiet between the hero's trigger
/// pulls.
pub const RETURN_FIRE_MEMORY_STEPS: u64 = 300;

/// **The half-angle of the cone a friendly may not stand in**, degrees.
///
/// **Twelve.** At the engagement range's own 35 m that is a corridor 14.9 m wide
/// at the far end and 3.4 m wide at 8 m, which comfortably covers the distance
/// at which an officer's colleague actually gets in the way. Wider and a unit
/// never fires in a street; narrower and the cone stops catching anybody a
/// spread cone would hit.
///
/// It is a HALF angle, tested with [`in_cone`], and it is deliberately wider
/// than any shipped weapon's own spread: the discipline rule has to refuse a
/// round the ballistics would have allowed, or it is not a rule.
pub const FRIENDLY_CONE_DEG: f64 = 12.0;

/// **How many line-of-sight rays the firing policy may cast in one fixed step.**
///
/// **Sixteen**, and it is its own number rather than a reuse of either of the
/// budgets it sits between, which is the whole reason it is written down:
///
/// * `d3::gameplay::MAX_ACTS_PER_STEP` × [`crate::witness::MAX_OBSERVERS`] —
///   WPN1's **witness** budget, 32 — is about who SAW something happen. (A code
///   span for the first, because it lives in the applying crate and Ring 0
///   cannot link downhill.);
/// * `d3::crime::MAX_RECOGNITION_RAYS` — 32 — is about who is RECOGNISED;
/// * [`crate::ballistics::MAX_SHOT_RAYS_PER_STEP`] is what a step's **bullets**
///   may cost.
///
/// This is none of those. It bounds *"can this officer see the person it is
/// about to shoot at"*, and sixteen is the honest ceiling for it: at
/// [`crate::dispatch::MAX_UNITS`] 64 units and [`crate::crime::MAX_PROFILES`] 32
/// files the pair space is 2 048, and a pass that cast a ray for every pair
/// would spend more on aiming than the whole rest of the step spends on
/// shooting. Sixteen is a unit for every rung of a three-unit `Swat` response,
/// five times over, which is more shooters than the island has ever had on one
/// street.
///
/// A pass that runs out simply **stops looking**, in `Guid` order — the
/// recognition budget's own rule: an officer that did not get a ray this step
/// gets one next step, sixty times a second.
pub const NPC_ENGAGE_RAYS_PER_STEP: usize = 16;

/// **How long one engagement cycle is**, fixed steps.
///
/// **Ninety-six** — 1.6 s. A unit's trigger is open for [`ENGAGE_BURST_STEPS`]
/// out of every one of these and closed for the rest, which is what makes an
/// NPC's fire read as *bursts with pauses* rather than as a hose. The weapon's
/// own `fire_interval_s` decides how many rounds fit in the open window, so a
/// pistol and a rifle burst differently off one cadence.
pub const ENGAGE_PERIOD_STEPS: u64 = 96;

/// **How much of a cycle the trigger is open**, fixed steps.
///
/// **Eighteen** — 0.30 s. A `glock_17` at 450 rpm puts 2 rounds through it and
/// an `m4a1` at 800 rpm puts 4, which is a burst on both and a magazine on
/// neither.
pub const ENGAGE_BURST_STEPS: u64 = 18;

/// The salt the cadence's phase offset is drawn with.
///
/// Its own value so a unit's firing phase cannot collide with its crowd look,
/// its route or its crew guid — [`crate::crowd::agent_rand`]'s own doctrine.
pub const SALT_ENGAGE: u64 = 0x454e_4741_4745_0001;

/// **Whether this unit's trigger is open on this step** — the cadence.
///
/// A **counter hash and not a clock**: the phase is
/// [`crate::crowd::agent_rand`] of the unit's own guid, so two officers arriving
/// together do not fire in lockstep, and the answer is a pure function of
/// `(guid, step)` with **no state to store and nothing to replay**. That is the
/// property that makes the whole policy a pure function of sim state: a host
/// that joined the trace mid-way computes the same window.
///
/// `false` for a period of zero, which is a caller asking for a cadence with no
/// cycle in it.
pub fn fire_window(unit: Uuid, step: u64) -> bool {
    if ENGAGE_PERIOD_STEPS == 0 {
        return false;
    }
    let phase = crate::crowd::agent_rand(unit, 0, SALT_ENGAGE) % ENGAGE_PERIOD_STEPS;
    (step.wrapping_add(phase)) % ENGAGE_PERIOD_STEPS < ENGAGE_BURST_STEPS
}

// ── what a crew carries ─────────────────────────────────────────────────────

/// **What a police crew is issued when it gets out of the car**, best first.
///
/// A LIST and not an id, because two levels in this repository define two
/// different catalogues and a policy that named one of them would arm nobody on
/// the other: the island's own registry opens with `glock_17`
/// (`inf_editor_core::island::ISLAND_SIDEARM_ID`) and the `phase30-gameplay`
/// fixture defines three rows of which `pistol` is one
/// (`GAMEPLAY_ITEMS_TOML`). The applier walks this list and takes the first row
/// the level actually defines, so the same policy arms an officer on both and
/// arms nobody on a level with no weapons at all — which is a refusal and not a
/// crash.
pub const POLICE_SIDEARM_IDS: [&str; 2] = ["glock_17", "pistol"];

/// **What a SWAT-grade response is issued instead** — see
/// [`POLICE_SIDEARM_IDS`] for why it is a list.
///
/// EMS3's carried item — *"Swat is a COUNT not a crew"* — becoming the second
/// behaviour it implies: at the top rung the town sends everything it has AND
/// what gets out of the van is carrying a rifle. It is the same ladder
/// [`prefers_high`](crate::cover::prefers_high) reads for cover, answered for
/// the weapon.
pub const SWAT_RIFLE_IDS: [&str; 2] = ["m4a1", "rifle"];

/// Which list this rung of the ladder issues from.
///
/// `Cold` issues nothing at all: a town where nobody is wanted does not arm its
/// patrols, so a level that never has a crime never equips a weapon and every
/// trace committed before this wave steps the bytes it stepped before.
pub fn crew_weapon_ids(response: Response) -> &'static [&'static str] {
    match response {
        Response::Cold => &[],
        Response::Patrol | Response::MultiUnit => &POLICE_SIDEARM_IDS,
        Response::Swat => &SWAT_RIFLE_IDS,
    }
}

// ── the geometry ────────────────────────────────────────────────────────────

/// **Is `other` inside the cone from `from` toward `at`** — the friendly-fire
/// test, and the coarse half of the civilian-in-the-line test.
///
/// Three things have to be true, and the third is the one a naive angle test
/// forgets: `other` must be *in front*, must be inside `half_deg` of the aim
/// line, **and must be NEARER than the target**. Somebody standing behind the
/// person you are shooting at is not in the way.
///
/// Portable: [`inf_math::pacos64`] and nothing else, because this answer decides
/// whether a round is fired and therefore reaches a trace.
///
/// `false` for degenerate geometry — a zero-length aim, a body on top of the
/// shooter — which is the conservative answer: a test that cannot see anything
/// does not stop a shot, and the LOS ray is the backstop.
pub fn in_cone(from: Vec3d, at: Vec3d, other: Vec3d, half_deg: f64) -> bool {
    let aim = (at.x - from.x, at.y - from.y, at.z - from.z);
    let to = (other.x - from.x, other.y - from.y, other.z - from.z);
    let aim_len = (aim.0 * aim.0 + aim.1 * aim.1 + aim.2 * aim.2).sqrt();
    let to_len = (to.0 * to.0 + to.1 * to.1 + to.2 * to.2).sqrt();
    if !aim_len.is_finite() || !to_len.is_finite() || aim_len <= 1.0e-6 || to_len <= 1.0e-6 {
        return false;
    }
    // Behind the shooter, or further away than what it is aiming at.
    if to_len >= aim_len {
        return false;
    }
    let cos = ((aim.0 * to.0 + aim.1 * to.1 + aim.2 * to.2) / (aim_len * to_len)).clamp(-1.0, 1.0);
    inf_math::pacos64(cos).to_degrees() <= half_deg
}

/// **May this unit aim at this suspect at all** — the police-don't-cheat gate,
/// as one expression.
///
/// Every argument is something the applier had to EARN:
///
/// * `last_seen` — the file's own memory, and `trail_age` its age. A unit whose
///   file has not been refreshed inside [`TRAIL_STALE_STEPS`] is searching, not
///   shooting;
/// * `line_of_sight` — a ray the applier cast **this step**;
/// * `unit_at` — the officer's own position, for the range gate.
///
/// The range is measured against `last_seen`, **not** against the suspect's real
/// position, and that is not pedantry: it is the one place a policy could
/// silently start reading a transform nobody looked at. The two are within a
/// capsule of each other whenever the LOS is clear, which is the only case that
/// gets past the second argument anyway.
///
/// Returns `false` for a suspect closer than [`MIN_ENGAGE_M`] — the launcher
/// guard — and for any non-finite input.
pub fn may_engage(
    posture: Posture,
    unit_at: Vec3d,
    last_seen: Vec3d,
    trail_age: u64,
    line_of_sight: bool,
) -> bool {
    line_of_sight && worth_a_ray(posture, unit_at, last_seen, trail_age)
}

/// **Everything [`may_engage`] asks EXCEPT the ray** — the pre-filter that
/// decides whether a pair is worth spending one of
/// [`NPC_ENGAGE_RAYS_PER_STEP`] on.
///
/// It exists so the applier never has to pass a *placeholder* line-of-sight
/// value into [`may_engage`] before it has cast anything. That sounds like a
/// nicety and is not: a `may_engage(.., true)` at a call site above the ray is
/// exactly the mutation the law arm makes to prove the law, and a call site that
/// contained one by design would make the mutation invisible.
///
/// Everything else is here — the posture, the trail's age, and the range
/// measured against the **remembered** position rather than the real one.
pub fn worth_a_ray(posture: Posture, unit_at: Vec3d, last_seen: Vec3d, trail_age: u64) -> bool {
    if !posture.aims() || trail_age > TRAIL_STALE_STEPS {
        return false;
    }
    let d = ((last_seen.x - unit_at.x).powi(2)
        + (last_seen.y - unit_at.y).powi(2)
        + (last_seen.z - unit_at.z).powi(2))
    .sqrt();
    d.is_finite() && (MIN_ENGAGE_M..=ENGAGE_RANGE_M).contains(&d)
}

// ── the pass's own memory ───────────────────────────────────────────────────

/// **What one unit is doing about one suspect.**
///
/// Plain scalars, `Copy`, on [`crate::cover::UnitCover`]'s terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitEngagement {
    /// Who it is engaging. `Uuid::nil()` when it has a file and no line of
    /// sight — the unit is *looking* and not aiming.
    pub target: Uuid,
    /// **The last step a round came past this unit**, which is what
    /// [`Posture::ReturnFire`] reads. `u64::MAX` means never, so the
    /// subtraction below cannot wrap into "just now".
    pub fired_upon_step: u64,
    /// Steps this unit's trigger has been open over the session — the engagement
    /// counter that tells "the policy fires" from "the policy runs".
    pub shots: u64,
    /// Steps this unit was refused a shot by the discipline rules — the other
    /// half of the same question.
    pub holds: u64,
}

impl UnitEngagement {
    /// **Whether this unit is pointing a weapon at somebody right now.**
    ///
    /// A named predicate rather than `!target.is_nil()` at four call sites, on
    /// [`crate::dispatch::is_responder`]'s terms: it is a fact about the world
    /// and a fact nobody can ask about is a filter.
    pub fn engaged(&self) -> bool {
        !self.target.is_nil()
    }
}

impl Default for UnitEngagement {
    fn default() -> Self {
        Self {
            target: Uuid::nil(),
            // **Never, and it has to be spelled** — a zero here would read as
            // "shot at on step 0", which on a level that opens hot is inside
            // `RETURN_FIRE_MEMORY_STEPS` and would make every `MultiUnit` unit
            // open fire in its first five seconds without a round being fired.
            fired_upon_step: u64::MAX,
            shots: 0,
            holds: 0,
        }
    }
}

impl UnitEngagement {
    /// Whether this unit has been shot at inside [`RETURN_FIRE_MEMORY_STEPS`] of
    /// `step`.
    pub fn fired_upon(&self, step: u64) -> bool {
        self.fired_upon_step != u64::MAX
            && step.saturating_sub(self.fired_upon_step) <= RETURN_FIRE_MEMORY_STEPS
    }
}

/// **Who is engaging whom**, in `Guid` order.
///
/// A resource and not a component, for [`crate::cover::NpcCoverRes`]' reasons
/// verbatim: it is the pass's own memory, it never enters a document, and it is
/// **absent on every level where nobody has ever been wanted**.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct EngageRes {
    /// The engaged units.
    pub units: BTreeMap<Uuid, UnitEngagement>,
    /// Steps the policy has held a trigger open over the session.
    pub shots: u64,
    /// Shots the discipline rules refused over the session.
    pub holds: u64,
    /// Weapons handed out at arrival over the session.
    pub equips: u64,
}

/// The engagements, or `None` on a level where nobody has ever engaged.
pub fn engage_of(world: &crate::EcsWorld) -> Option<&EngageRes> {
    world.world().get_resource::<EngageRes>()
}

/// **How many units are pointing a weapon at somebody right now** — the number
/// the demo loop's `hero.csv` carries and a gate's engagement counter.
///
/// Reads the world rather than a report: a slot with a non-nil target is a unit
/// whose aim was written this step.
pub fn engaged_units(world: &crate::EcsWorld) -> usize {
    engage_of(world)
        .map(|r| r.units.values().filter(|u| u.engaged()).count())
        .unwrap_or(0)
}

/// **Forget every engagement** — [`crate::dispatch::clear_dispatch`]'s twin, for
/// its reason: an editor Simulate session must leave nothing behind in the
/// author's document.
pub fn clear_engage(world: &mut crate::EcsWorld) {
    world.world_mut().remove_resource::<EngageRes>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_is_four_behaviours_and_a_patrol_never_fires() {
        assert_eq!(posture_for(Response::Cold), Posture::Hold);
        assert_eq!(posture_for(Response::Patrol), Posture::Warn);
        assert_eq!(posture_for(Response::MultiUnit), Posture::ReturnFire);
        assert_eq!(posture_for(Response::Swat), Posture::FireOnSight);
        // The whole firing rule, both ways round.
        assert!(!Posture::Hold.may_fire(true));
        assert!(!Posture::Warn.may_fire(true), "a patrol fired");
        assert!(!Posture::ReturnFire.may_fire(false));
        assert!(Posture::ReturnFire.may_fire(true));
        assert!(Posture::FireOnSight.may_fire(false));
        // …and only `Hold` keeps the weapon away.
        assert!(!Posture::Hold.aims());
        for p in [Posture::Warn, Posture::ReturnFire, Posture::FireOnSight] {
            assert!(p.aims(), "{} does not point a weapon", p.name());
        }
        // The bytes are frozen and distinct.
        let mut seen: Vec<u8> = [
            Posture::Hold,
            Posture::Warn,
            Posture::ReturnFire,
            Posture::FireOnSight,
        ]
        .iter()
        .map(|p| p.as_u8())
        .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn no_line_of_sight_and_no_trail_are_each_enough_to_refuse() {
        let unit = Vec3d::new(0.0, 0.0, 0.0);
        let seen = Vec3d::new(0.0, 0.0, 20.0);
        // Everything right.
        assert!(may_engage(Posture::FireOnSight, unit, seen, 0, true));
        // …and each argument alone takes it away.
        assert!(
            !may_engage(Posture::FireOnSight, unit, seen, 0, false),
            "a unit with no line of sight aimed"
        );
        assert!(
            !may_engage(
                Posture::FireOnSight,
                unit,
                seen,
                TRAIL_STALE_STEPS + 1,
                true
            ),
            "a unit aimed at a trail three seconds cold"
        );
        assert!(
            !may_engage(Posture::Hold, unit, seen, 0, true),
            "a cold town aimed at somebody"
        );
        // Out of range, both ways.
        assert!(!may_engage(
            Posture::FireOnSight,
            unit,
            Vec3d::new(0.0, 0.0, ENGAGE_RANGE_M + 0.1),
            0,
            true
        ));
        assert!(
            !may_engage(
                Posture::FireOnSight,
                unit,
                Vec3d::new(0.0, 0.0, MIN_ENGAGE_M - 0.1),
                0,
                true
            ),
            "a unit fired a launcher at somebody inside its own blast"
        );
        // The engagement range is INSIDE the recognition range, deliberately.
        const { assert!(ENGAGE_RANGE_M < crate::crime::RECOGNITION_RANGE_M) };
    }

    #[test]
    fn the_cone_refuses_a_friendly_in_front_and_allows_one_behind() {
        let from = Vec3d::new(0.0, 0.0, 0.0);
        let at = Vec3d::new(0.0, 0.0, 20.0);
        // Dead in the line, halfway.
        assert!(in_cone(
            from,
            at,
            Vec3d::new(0.0, 0.0, 10.0),
            FRIENDLY_CONE_DEG
        ));
        // Behind the target: not in the way.
        assert!(!in_cone(
            from,
            at,
            Vec3d::new(0.0, 0.0, 25.0),
            FRIENDLY_CONE_DEG
        ));
        // Behind the shooter.
        assert!(!in_cone(
            from,
            at,
            Vec3d::new(0.0, 0.0, -5.0),
            FRIENDLY_CONE_DEG
        ));
        // Off to the side by more than the half angle: at 10 m out, 12 deg is
        // 2.126 m, so 3 m is clear and 1 m is not.
        assert!(!in_cone(
            from,
            at,
            Vec3d::new(3.0, 0.0, 10.0),
            FRIENDLY_CONE_DEG
        ));
        assert!(in_cone(
            from,
            at,
            Vec3d::new(1.0, 0.0, 10.0),
            FRIENDLY_CONE_DEG
        ));
        // Degenerate inputs refuse rather than divide.
        assert!(!in_cone(from, from, Vec3d::new(1.0, 0.0, 1.0), 12.0));
        assert!(!in_cone(from, at, from, 12.0));
    }

    #[test]
    fn the_cadence_opens_the_trigger_for_a_burst_and_two_units_do_not_lockstep() {
        let a = Uuid::from_u128(0x2E00_0001);
        let b = Uuid::from_u128(0x2E00_0002);
        let mut open_a = 0u64;
        let mut same = 0u64;
        for step in 0..ENGAGE_PERIOD_STEPS * 10 {
            let wa = fire_window(a, step);
            let wb = fire_window(b, step);
            if wa {
                open_a += 1;
            }
            if wa == wb {
                same += 1;
            }
        }
        assert_eq!(
            open_a,
            ENGAGE_BURST_STEPS * 10,
            "the trigger was open {open_a} steps in ten cycles, not {}",
            ENGAGE_BURST_STEPS * 10
        );
        assert!(
            same < ENGAGE_PERIOD_STEPS * 10,
            "two units fired in lockstep over every one of {} steps",
            ENGAGE_PERIOD_STEPS * 10
        );
        // …and it is a pure function of (guid, step): the same call twice is the
        // same answer, which is what makes it replayable.
        for step in 0..200 {
            assert_eq!(fire_window(a, step), fire_window(a, step));
        }
    }

    #[test]
    fn a_unit_that_was_never_shot_at_does_not_return_fire() {
        let fresh = UnitEngagement::default();
        assert!(!fresh.fired_upon(0), "a fresh slot remembered a shot");
        assert!(
            !fresh.fired_upon(u64::MAX),
            "`never` wrapped into `just now`"
        );
        let hit = UnitEngagement {
            fired_upon_step: 100,
            ..Default::default()
        };
        assert!(hit.fired_upon(100));
        assert!(hit.fired_upon(100 + RETURN_FIRE_MEMORY_STEPS));
        assert!(!hit.fired_upon(101 + RETURN_FIRE_MEMORY_STEPS));
    }
}
