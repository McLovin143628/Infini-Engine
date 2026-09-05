//! **THE DISPATCHER** (wave EMS2) — who is sent to what, and what a person sent
//! to something is allowed to do.
//!
//! The deciding half. Everything here is a pure function of sim state over
//! resources: no schema moves (scene v27 / `ScenePayload` v13 stand), nothing is
//! ever written to a file, and the applying half — the routes, the bodies, the
//! sirens — lives in `inf_physics::d3::dispatch` behind this crate's own split
//! (`inf_ecs::vehicle` decides and `inf_physics::d3::vehicle` applies; the same
//! wall, a third time).
//!
//! # This module opens with the panic exemption, and the order is the point
//!
//! [`crate::crowd::flee_from`] is the one door a frightened person goes through,
//! and until this wave it had no idea who it was frightening. `step_panic` walks
//! the *whole* population on the step a shot goes off, `flee_from` **clears the
//! schedule** of everybody it reaches, and [`crate::crowd::PanickedRes`] is never
//! released — so a gunshot at an incident would have permanently routed the
//! officers standing at it. Every later clause of this wave rests on that not
//! happening, so the exemption is the first thing in the file rather than a
//! guard bolted onto the last.
//!
//! The rule is one sentence and it is a rule rather than a filter: **a responder
//! does not rout**. It lives at the flee door so it holds for every caller —
//! the crowd panic, the carjack, and anything a later wave adds — and
//! `PanicReport::exempt` counts the times it fired so a gate can tell "the
//! officers held" from "no officer was ever in the radius".

use std::collections::BTreeSet;

use bevy_ecs::prelude::Resource;
use glam::DVec3;
use uuid::Uuid;

use crate::world::EcsWorld;

/// **Everybody who is on duty at something** (wave EMS2) — the named responder
/// set the panic exemption reads.
///
/// # A resource, and it is [`crate::crowd::PanickedRes`]' reason exactly
///
/// A responder is a crowd agent, and a [`Dormant`](crate::crowd::CrowdTier::Dormant)
/// one has **no entity at all** — its record is in the population, it still
/// steps, and it is exactly the agent a shot at the far edge of a panic radius
/// reaches. A marker component would have been silently absent on every one of
/// them, which is the tier-dependent-state trap `crowd_state_bytes`' own doc
/// names.
///
/// Derived, never saved, no schema moves — [`crate::item::ItemDefs`]' shape.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct RespondersRes {
    /// Who is on duty, in `Guid` order.
    pub on_duty: BTreeSet<Uuid>,
}

/// **Is this person a responder?** — the named predicate the panic exemption is
/// written in terms of.
///
/// Public and named rather than folded into a `filter` inside the panic pass,
/// because the exemption is a *rule about the world* — an officer under fire
/// does not rout — and a rule nobody can ask about is a filter. The gate arm
/// `an_officer_under_fire_does_not_rout` asks it.
///
/// `O(log n)` on a level that has responders and one `get_resource` on every
/// level that does not, which is every level committed before this wave.
pub fn is_responder(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .world()
        .get_resource::<RespondersRes>()
        .is_some_and(|r| r.on_duty.contains(&guid))
}

/// **Put somebody on duty**, or take them off it.
///
/// The one door, so a second producer cannot invent a second shape of the set.
/// Returns whether the set changed — an engagement counter for a caller that
/// wants to know it did something.
pub fn set_responder(world: &mut EcsWorld, guid: Uuid, on_duty: bool) -> bool {
    let mut res = world
        .world_mut()
        .remove_resource::<RespondersRes>()
        .unwrap_or_default();
    let changed = if on_duty {
        res.on_duty.insert(guid)
    } else {
        res.on_duty.remove(&guid)
    };
    world.world_mut().insert_resource(res);
    changed
}

/// Everybody on duty right now, in `Guid` order — empty on a level with no
/// responders.
pub fn responders(world: &EcsWorld) -> Vec<Uuid> {
    world
        .world()
        .get_resource::<RespondersRes>()
        .map(|r| r.on_duty.iter().copied().collect())
        .unwrap_or_default()
}

/// **Forget who was on duty, what was on fire and who was on the way** —
/// [`crate::crowd::clear_crowd`]'s twin, for its reason: an editor Simulate
/// session must leave nothing behind in the author's document, and a resource is
/// outside the `ScenePersist::Memory` snapshot by construction.
pub fn clear_dispatch(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<RespondersRes>();
    world.world_mut().remove_resource::<FleetRes>();
    // **The puffs and the CREWS are DESPAWNED, not forgotten** — the VEN1b
    // speaker's rule: a body this session spawned and left behind is a row in
    // the author's Outliner that no Outliner row put there. Read before the
    // resource goes, because the resource is the only list of either.
    //
    // # The crew is on this list for the puff's reason exactly (EMS2 audit)
    //
    // A crew member is an [`crate::crowd::spawn_body`], which is deliberately
    // **not** a population record — so `clear_crowd` walks
    // `CrowdPopulationRes` and never sees it, and a session stopped while a
    // unit was out left a person standing in the road that nothing on any
    // clear-path could reach. The first cut of this door despawned the sprites
    // and not the people, which is the same law applied to one of the two
    // things this wave spawns.
    //
    // `crew_guid` is a pure function of the chassis and `entity_of` answers
    // `None` for a unit that never left its bay, so the walk is over every run
    // and costs nothing on a station at rest.
    let (puffs, crews): (Vec<Uuid>, Vec<Uuid>) = world
        .world()
        .get_resource::<DispatchRes>()
        .map(|d| {
            (
                d.puffs.keys().copied().collect(),
                d.runs.keys().copied().map(crew_guid).collect(),
            )
        })
        .unwrap_or_default();
    for guid in puffs.into_iter().chain(crews) {
        if let Some(e) = world.entity_of(guid) {
            world.despawn(e);
        }
    }
    world.world_mut().remove_resource::<DispatchRes>();
}

// ── what a service is ───────────────────────────────────────────────────────

/// **The three services**, and no fourth.
///
/// SWAT is a *crew* and not a service — a van full of officers is still the
/// police — which is the EMS1 audit's own reading and is why there are three
/// here against four rows in the fleet catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnitKind {
    /// A cruiser or a tactical van.
    Police,
    /// An appliance.
    Fire,
    /// An ambulance.
    Ambulance,
}

impl UnitKind {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            UnitKind::Police => "police",
            UnitKind::Fire => "fire",
            UnitKind::Ambulance => "ambulance",
        }
    }

    /// The byte this service folds into a trace. **Frozen, append-only** on
    /// [`crate::components::SlotRole::as_u8`]'s terms: `Police` 0, `Fire` 1,
    /// `Ambulance` 2.
    pub fn as_u8(self) -> u8 {
        match self {
            UnitKind::Police => 0,
            UnitKind::Fire => 1,
            UnitKind::Ambulance => 2,
        }
    }
}

/// **The entity name every emergency light bar carries.**
///
/// EMS1's three light-bar `BodyPart`s (`SEDAN_BAR`, `VAN_BAR`, `TRUCK_BAR`) are
/// all spelled `light_bar`, and `inf_ecs::vehicle::rig_nodes_at` names the
/// entity after the part — so this string is the persisted trace of a livery,
/// and it is the one thing in a *committed document* that says "this vehicle is
/// an emergency vehicle".
pub const LIGHT_BAR_PART: &str = "light_bar";

/// How long a chassis has to be, in half-metres of its own collider, before a
/// red-barred vehicle is an **appliance** rather than an ambulance.
///
/// # The number is a measurement, and the first guess was nearly wrong
///
/// It was written as 3.0 from a remembered table, and
/// `every_livery_is_recognised_as_the_service_it_declares` printed what the
/// island's four rows *actually* build: cruiser **2.32**, SWAT van **2.80**,
/// ambulance **2.95**, appliance **3.90** metres of half-length. Three metres
/// left the ambulance five centimetres from being dispatched to house fires —
/// one editorial tweak to a `.toml` row away from a silent misrouting that no
/// arm in this tree would have named.
///
/// 3.4 is the middle of the measured gap: 45 cm of margin on each side, which is
/// more than any of the four rows has ever moved. It is a *rule* and not a table
/// lookup — a red-barred vehicle seven metres long is a fire appliance whoever
/// authored it — which is what lets a project that adds a fifth row be
/// recognised without touching this crate.
pub const APPLIANCE_HALF_LENGTH_M: f64 = 3.4;

/// **What service a vehicle in the world belongs to**, or `None` for a civilian
/// one — the ONE recogniser.
///
/// # Why the level is read and not the recipe
///
/// The fleet is authored: EMS1's generator parks it into the `.inf_lvl` with a
/// guid salted on `island.ems.{site}.{col}.{row}.{k}.{id}`, and the *recipe*
/// that knows a row is called `"ambulance"` lives in Ring 1 and is not there
/// when a shipped player opens the document. Nothing on a chassis says what it
/// is — `VehicleClass` is forty-seven suspension numbers and a schema this wave
/// may not move — so the only honest answer is to read what the livery **left
/// in the world**:
///
/// * a child entity named [`LIGHT_BAR_PART`] whose material is **bloomed**
///   (emissive over 1, which `PartPaint` requires or the HDR path does not see
///   it) — that is an emergency vehicle and nothing else in this engine has one;
/// * its hue: blue is police, red is fire or medical (`BEACON_BLUE` /
///   `BEACON_RED`);
/// * and, for the red pair, the chassis's own length against
///   [`APPLIANCE_HALF_LENGTH_M`].
///
/// Two observable channels, three services, no ambiguity. `Livery::service`
/// declares the same answer on the authoring side and
/// `every_livery_is_recognised_as_the_service_it_declares` holds the two
/// together, so a fifth livery that disagreed with this rule fails a test rather
/// than dispatching an ambulance to a fire.
///
/// `O(children)` — the shape [`crate::vehicle::rig_of`] already is, and asked
/// once per block stamp rather than once per step.
pub fn unit_kind_of(world: &EcsWorld, chassis: Uuid) -> Option<UnitKind> {
    let entity = world.entity_of(chassis)?;
    let w = world.world();
    // A chassis, so a stray named prop cannot be dispatched.
    let collider = w.get::<crate::components::Collider3D>(entity)?;
    crate::vehicle::chassis_of(
        Some(collider),
        w.get::<crate::components::RigidBody3D>(entity),
    )?;
    let half_length_m = collider.half_extents.z;
    for child in world.children_of(entity) {
        let named = w
            .get::<crate::components::Name>(child)
            .is_some_and(|n| n.0 == LIGHT_BAR_PART);
        if !named {
            continue;
        }
        let Some(m) = w.get::<crate::components::Material>(child) else {
            continue;
        };
        let lin = m.emissive_linear();
        // Not bloomed is not a beacon: an unlit grey box named `light_bar` is a
        // part somebody modelled, not a unit somebody can send.
        if lin[0].max(lin[1]).max(lin[2]) <= 1.0 {
            continue;
        }
        return Some(if lin[2] > lin[0] {
            UnitKind::Police
        } else if half_length_m >= APPLIANCE_HALF_LENGTH_M {
            UnitKind::Fire
        } else {
            UnitKind::Ambulance
        });
    }
    None
}

// ── the fleet, and who owns it ──────────────────────────────────────────────

/// How far from a block a parked vehicle may be and still be **its** vehicle,
/// metres.
///
/// EMS1 parks a fleet on an apron 6 m off the nearest route vertex to the
/// block's centre, at 11 m pitches, at most sixteen shunts along — so the
/// furthest a station's own third appliance can legitimately sit from the block
/// centre is a long way down one street and nowhere near another block. Eighty
/// metres reaches the whole apron of a 52 m lot and stops well short of the
/// island's own settlement pitch.
pub const STATION_CLAIM_M: f64 = 80.0;

/// The most units one level's dispatcher owns.
///
/// The island parks seventeen. Sixty-four is four times that and is the same
/// bound `inf_player::budget::VEHICLE_BUDGET_CARS` is measured at, which is the
/// point: a unit that is responding is a vehicle with a rig on four rays, and
/// the two ceilings should be the same number for the same reason.
pub const MAX_UNITS: usize = 64;

/// **One emergency vehicle, and the institution it belongs to** (wave EMS2) —
/// the ownership edge EMS1 left out.
///
/// EMS1's audit put it plainly: *the fleet is parked, not owned*. Nothing linked
/// a vehicle to a station, so a dispatcher had nothing to send and nowhere to
/// send it back to. This is that edge, derived rather than authored — see
/// [`sync_fleet`] for why it can be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FleetUnit {
    /// What service it is — [`unit_kind_of`]'s answer.
    pub kind: UnitKind,
    /// The block it is parked at. A diagnostic and a *return address*: a unit
    /// goes home to the space it came out of, and the station is what says two
    /// units belong to one institution.
    pub station: Uuid,
    /// The space it was parked in, world metres — where it goes back to.
    pub home: DVec3,
    /// The way that space points, degrees.
    pub home_yaw_deg: f64,
}

/// **The level's own fleet**, derived once and rebuilt when the blocks move.
///
/// A resource on [`crate::traffic::TrafficRes`]' exact terms: no schema moves,
/// absent until something asks, and one `block_stamp` walk to decide whether it
/// is stale — which is the same walk `sync_society` and `sync_carriageway`
/// already make.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct FleetRes {
    /// Every unit, in `Guid` order.
    pub units: std::collections::BTreeMap<Uuid, FleetUnit>,
    /// The block-set fold this was derived from. `0` before the first
    /// derivation.
    pub stamp: u64,
    /// How many times the derivation has actually run — a counter a gate can
    /// assert is **one** over a settled level, which is what says the cache is a
    /// cache.
    pub derivations: u64,
}

/// **Derive the fleet if the level's blocks have moved** — the one door both
/// hosts call, through their applier.
///
/// Returns whether it rebuilt.
///
/// # A unit's station is the block it is parked at, and that is a rule
///
/// The recipe parks a fleet on its own institution's apron and nowhere else, so
/// *the nearest block within [`STATION_CLAIM_M`]* recovers the edge the recipe
/// drew. It is deliberately not "the nearest **institution**": `inf-ecs` is
/// Ring 0 and an archetype is Ring 1 authoring vocabulary, and the question
/// being asked here — **where does this vehicle live** — is answered correctly
/// by the block whether or not this crate can name what kind of building it is.
///
/// A unit parked on no block at all is refused rather than given a station at
/// the origin, because a return address that is wrong is worse than a unit that
/// never leaves.
///
/// `O(vehicles × blocks)` on the step a level's geometry changes and one
/// `block_stamp` walk on every other, which on a settled level is never.
pub fn sync_fleet(world: &mut EcsWorld) -> bool {
    let stamp = crate::traffic::block_stamp(world);
    if let Some(res) = world.world().get_resource::<FleetRes>() {
        if res.stamp == stamp {
            return false;
        }
    }
    // The blocks, once: every `PcgVolume` with a place in the world.
    let mut blocks: Vec<(Uuid, DVec3)> = Vec::new();
    let mut candidates: Vec<Uuid> = Vec::new();
    for e in world.world().iter_entities() {
        let Some(g) = e.get::<crate::components::Guid>() else {
            continue;
        };
        if e.get::<crate::components::PcgVolume>().is_some() {
            if let Some(t) = e.get::<crate::components::GlobalTransform>() {
                let p = t.translation();
                if p.is_finite() {
                    blocks.push((g.0, p));
                }
            }
            continue;
        }
        if e.get::<crate::components::Collider3D>().is_some()
            && e.get::<crate::components::RigidBody3D>().is_some()
        {
            candidates.push(g.0);
        }
    }
    blocks.sort_by_key(|(g, _)| *g);
    candidates.sort_unstable();
    let mut units: std::collections::BTreeMap<Uuid, FleetUnit> = std::collections::BTreeMap::new();
    for chassis in candidates {
        if units.len() >= MAX_UNITS {
            break;
        }
        let Some(kind) = unit_kind_of(world, chassis) else {
            continue;
        };
        let Some(e) = world.entity_of(chassis) else {
            continue;
        };
        let Some(t) = world.world().get::<crate::components::Transform>(e) else {
            continue;
        };
        let home = t.translation.to_dvec3();
        let home_yaw_deg = t.rotation.y;
        if !home.is_finite() {
            continue;
        }
        // The nearest block, ties on the `Guid` — `blocks` is `Guid`-ordered and
        // the comparison is strict, so the first of two equidistant blocks wins
        // on both hosts.
        let mut best: Option<(f64, Uuid)> = None;
        for (g, p) in &blocks {
            let d = (*p - home).length();
            if !d.is_finite() || d > STATION_CLAIM_M {
                continue;
            }
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, *g));
            }
        }
        let Some((_, station)) = best else {
            continue;
        };
        units.insert(
            chassis,
            FleetUnit {
                kind,
                station,
                home,
                home_yaw_deg,
            },
        );
    }
    let derivations = world
        .world()
        .get_resource::<FleetRes>()
        .map(|r| r.derivations)
        .unwrap_or(0);
    world.world_mut().insert_resource(FleetRes {
        units,
        stamp,
        derivations: derivations + 1,
    });
    true
}

/// The level's fleet, or `None` — the read side, for a caller that must not
/// derive.
pub fn fleet_of(world: &EcsWorld) -> Option<&FleetRes> {
    world.world().get_resource::<FleetRes>()
}

// ── what happens, and what is done about it ─────────────────────────────────

/// **Something that needs somebody.**
///
/// Three, each carrying the one fact its own service needs and nothing else:
/// what is burning, who has collapsed, and how bad the crime was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncidentKind {
    /// A building is alight. `intensity` is the fire's own size, unitless in
    /// `[0, 1]`, and it is what a fire crew spends down — see
    /// [`SUPPRESSION_PER_S`].
    Fire { building: Uuid, intensity: f64 },
    /// Somebody is on the ground. `severity` is `1` for a collapse and `2` for a
    /// body that has been shot.
    Medical { npc: Uuid, severity: u8 },
    /// Something was done to somebody. `severity` is `1` for a shot fired and
    /// `2` for a death.
    Crime { severity: u8 },
}

impl IncidentKind {
    /// **Who goes** — the whole of the routing rule.
    pub fn service(self) -> UnitKind {
        match self {
            IncidentKind::Fire { .. } => UnitKind::Fire,
            IncidentKind::Medical { .. } => UnitKind::Ambulance,
            IncidentKind::Crime { .. } => UnitKind::Police,
        }
    }

    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            IncidentKind::Fire { .. } => "fire",
            IncidentKind::Medical { .. } => "medical",
            IncidentKind::Crime { .. } => "crime",
        }
    }

    /// The byte this kind folds into a trace. **Frozen, append-only.**
    pub fn as_u8(self) -> u8 {
        match self {
            IncidentKind::Fire { .. } => 0,
            IncidentKind::Medical { .. } => 1,
            IncidentKind::Crime { .. } => 2,
        }
    }
}

/// **How far through its own life an incident is.**
///
/// A lifecycle and not a flag, because "reported" and "nobody could go" are the
/// same world with different meanings, and a dispatcher that could not tell them
/// apart would look identical to one that had no units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IncidentState {
    /// Known, nobody sent yet.
    Reported,
    /// A unit is on the way.
    Assigned,
    /// A unit is at it.
    OnScene,
    /// Dealt with. Kept for [`INCIDENT_KEEP_STEPS`] so a gate, a HUD and a
    /// later wave's reputation ledger can read what happened.
    Resolved,
}

impl IncidentState {
    /// A stable short name.
    pub fn name(self) -> &'static str {
        match self {
            IncidentState::Reported => "reported",
            IncidentState::Assigned => "assigned",
            IncidentState::OnScene => "on-scene",
            IncidentState::Resolved => "resolved",
        }
    }

    /// The byte this state folds into a trace. **Frozen, append-only.**
    pub fn as_u8(self) -> u8 {
        match self {
            IncidentState::Reported => 0,
            IncidentState::Assigned => 1,
            IncidentState::OnScene => 2,
            IncidentState::Resolved => 3,
        }
    }
}

/// **One thing that has happened and is being dealt with.**
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Incident {
    /// What it is.
    pub kind: IncidentKind,
    /// Where, world metres.
    pub at: DVec3,
    /// How far through its life it is.
    pub state: IncidentState,
    /// The fixed step it was opened on.
    pub opened_step: u64,
    /// The unit sent to it, once one has been.
    pub unit: Option<Uuid>,
    /// The step it was resolved on, or `None`.
    pub resolved_step: Option<u64>,
}

impl Incident {
    /// **How long it took, in fixed steps** — `None` while it is still open.
    ///
    /// The number a gate prints as a response time, and the one thing about a
    /// dispatcher a player actually experiences.
    pub fn response_steps(&self) -> Option<u64> {
        self.resolved_step
            .map(|s| s.saturating_sub(self.opened_step))
    }
}

/// **Where a unit is in its own run.**
///
/// **NOT on [`crate::traffic::TrafficRecord`]** — and the reason is that record's
/// own doc: it is *re-derived on block stamps*, so a unit's state would be
/// silently reset the moment a cell streamed in. It is also the wrong home on
/// principle: an emergency vehicle is authored content on EMS1's Path A and the
/// traffic has never heard of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnitState {
    /// Parked at its station.
    InStation,
    /// Driving to an incident.
    EnRoute,
    /// Stopped at one, with its crew out.
    OnScene,
    /// Driving home.
    Returning,
}

impl UnitState {
    /// A stable short name.
    pub fn name(self) -> &'static str {
        match self {
            UnitState::InStation => "in-station",
            UnitState::EnRoute => "en-route",
            UnitState::OnScene => "on-scene",
            UnitState::Returning => "returning",
        }
    }

    /// The byte this state folds into a trace. **Frozen, append-only.**
    pub fn as_u8(self) -> u8 {
        match self {
            UnitState::InStation => 0,
            UnitState::EnRoute => 1,
            UnitState::OnScene => 2,
            UnitState::Returning => 3,
        }
    }

    /// **Whether a unit in this state is running with its lights and siren on.**
    ///
    /// The one door, because three separate systems ask it — the audio emit, the
    /// projector's flashing bar and the yield rule — and three spellings of "is
    /// this thing responding" is the defect this repository has paid for at four
    /// seams. A returning unit is **not** running hot, which is what an
    /// ambulance that has already dropped its patient does.
    pub fn running_hot(self) -> bool {
        matches!(self, UnitState::EnRoute | UnitState::OnScene)
    }
}

/// **What one unit is doing** — the sim state a fleet derivation must not touch.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitRun {
    /// Where it is in its run.
    pub state: UnitState,
    /// The incident it is on, or `None`.
    pub incident: Option<Uuid>,
    /// The step it entered [`state`](Self::state) on — what an on-scene timer
    /// counts from.
    pub since_step: u64,
    /// The drive it is on: out to the scene, or home again. `None` in station.
    pub path: Option<inf_nav::NavPath>,
}

impl Default for UnitRun {
    fn default() -> Self {
        Self {
            state: UnitState::InStation,
            incident: None,
            since_step: 0,
            path: None,
        }
    }
}

/// The most incidents one level holds open at once.
///
/// Sixteen. It is a **cost** bound: every open incident is a candidate for an
/// assignment search, and an assignment search is a Dijkstra over the whole
/// carriageway. A town with sixteen simultaneous emergencies has more than a
/// player can watch, and the seventeenth is refused rather than queued — a
/// refusal is a value, and a queue nobody can reach is a leak with a deadline.
pub const MAX_OPEN_INCIDENTS: usize = 16;

/// How long a resolved incident is kept before it is forgotten, fixed steps.
///
/// Sixty seconds at 60 Hz. Long enough that a gate which resolves an incident
/// and then measures the response time finds it, short enough that an hour's
/// play does not accumulate a ledger nothing reads.
pub const INCIDENT_KEEP_STEPS: u64 = 3600;

/// **The dispatcher's own state** — what is open, and what every unit is doing.
///
/// Separate from [`FleetRes`] on purpose, and the separation is load-bearing:
/// the fleet is a **derivation** that is thrown away and rebuilt whenever the
/// level's blocks move, and this is **sim state** that must survive that. Fusing
/// them would reset a unit half-way to a fire because a terrain cell paged in —
/// which is the trap `TrafficRecord`'s own *"a re-derivation is not a fresh
/// start"* names, met one system over.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct DispatchRes {
    /// What is open, in `Guid` order.
    pub incidents: std::collections::BTreeMap<Uuid, Incident>,
    /// What every unit is doing, in `Guid` order. Keyed on the chassis.
    pub runs: std::collections::BTreeMap<Uuid, UnitRun>,
    /// How many fixed steps the dispatcher has run.
    pub steps: u64,
    /// The highest witnessed-act step already turned into a crime — so the log
    /// is read forward once and an act cannot be reported twice.
    pub seen_act_step: u64,
    /// Incidents opened over the session. A lifetime counter, so a gate can
    /// arm itself before it asserts anything about a table that is allowed to
    /// be empty at the end.
    pub opened: u64,
    /// …assigned to a unit.
    pub assigned: u64,
    /// …resolved.
    pub resolved: u64,
    /// **Incidents that found no free unit this step.** The falsifier for "the
    /// dispatcher works": a town whose every incident goes unanswered reads zero
    /// `assigned` and a rising number here, which is a different fact from a
    /// town where nothing happened.
    pub unanswered: u64,
    /// **Which units had a siren running at the end of the last step** — what
    /// tells a [`SirenCue::Start`] from a [`SirenCue::Move`].
    ///
    /// Not folded into [`dispatch_state_bytes`], and that is a ruling rather
    /// than an omission: this is the previous step's image of
    /// [`UnitState::running_hot`] over [`runs`](Self::runs), which **is** folded
    /// — so two hosts that agree about the runs on every step of a trace agree
    /// about this one by construction, and folding it would put sixteen bytes a
    /// unit into every committed hash to say a thing the hash already says.
    pub siren_on: BTreeSet<Uuid>,
    /// **What the audio step should do about this level's sirens this step** —
    /// rebuilt every step, drained at the audio phase. See [`siren_cues`].
    pub sirens: Vec<SirenCue>,
    /// **The emissive intensity each flashing bar was authored with**, held
    /// while its unit is running hot and **given back** when it stops.
    ///
    /// A pin with a release, which is P21.4's own law: the first cut multiplied
    /// the live value and a bar that had been out for a minute came home black.
    /// Keyed on the bar entity's guid.
    pub bars: std::collections::BTreeMap<Uuid, f32>,
    /// **What every flashing bar should be set to this step** — rebuilt every
    /// step, drained by the hosts' `light_bar_flash` fence. See [`bar_flashes`].
    pub flashes: Vec<BarFlash>,
    /// **Bodies an ambulance has already been to.**
    ///
    /// # The honest sentence about what "stabilized" means here
    ///
    /// It does not mean carried anywhere. This engine has no stretcher, no
    /// gurney animation and no hospital bed a body can occupy — EMS1's wards
    /// hold `Soft` modules and a `Ward` room's occupancy, not a place a
    /// simulated person lies down in — so a patient who has been attended stays
    /// exactly where they fell. What changes is that **the street has been
    /// attended**: the body is in this set, so no second ambulance is called for
    /// it.
    ///
    /// Without it the loop is visible and was measured: the `Downed` latch is
    /// permanent, `INCIDENT_KEEP_STEPS` forgets the resolved incident after a
    /// minute, and the same body calls another ambulance — for ever.
    /// (`a_paramedic_kneels_at_the_patient_and_stands_up_to_leave` runs past
    /// `INCIDENT_KEEP_STEPS` to measure it, because inside that window the
    /// incidents table's own guard is doing the work and this set is invisible.)
    ///
    /// # It is keyed on a SUBJECT, and an ambient collapse's subject is a BLOCK
    ///
    /// [`ambient_draw`] names the *block* as the casualty — there is no person
    /// there to name — so a block whose ambient collapse has been attended is in
    /// this set and can never draw another one. That is the right answer for the
    /// `Downed` body it was designed for and a **quiet retirement** of the
    /// ambient medical feed, block by block, over a long session: a town's
    /// collapses become fires-only once every block has had one. Sixteen epochs
    /// of a nine-block town is nowhere near it and the island is larger still,
    /// but it is a slope rather than a floor and it is stated rather than left to
    /// be met. EMS3 gives an ambient casualty a person, which retires the whole
    /// question.
    pub treated: BTreeSet<Uuid>,
    /// **Which criminal file a crime scene is about** (wave EMS3), incident guid
    /// to suspect guid.
    ///
    /// # Why the dispatcher holds the link and not the incident
    ///
    /// `IncidentKind::Crime` carries a severity and no subject, and giving it
    /// one would have been a variant field on a frozen enum for the sake of a
    /// map with at most `MAX_OPEN_INCIDENTS` rows in it. It is also the right
    /// owner on principle: *"who is this call about"* is dispatcher state — it
    /// decides where the car drives and how many of them go — and the incident
    /// is a fact about a place.
    ///
    /// It is what makes a search **follow** rather than sit at the address the
    /// first witness gave, and it is folded into
    /// [`dispatch_state_bytes`] because two hosts that were searching for
    /// different people have diverged in a way no position can show.
    pub searches: std::collections::BTreeMap<Uuid, Uuid>,
    /// **The smoke a fire is making**, puff guid to the step it was let go of.
    ///
    /// Real entities this session spawned, so they are cleared with everything
    /// else — see [`clear_dispatch`], which despawns them rather than merely
    /// forgetting them: a puff left behind is a row in the author's Outliner
    /// that no Outliner row put there (the VEN1b speaker's own rule).
    pub puffs: std::collections::BTreeMap<Uuid, u64>,
}

/// The dispatcher's state, or `None`.
pub fn dispatch_of(world: &EcsWorld) -> Option<&DispatchRes> {
    world.world().get_resource::<DispatchRes>()
}

// ── where incidents come from ───────────────────────────────────────────────

/// Salts the ambient-incident draw.
pub const SALT_INCIDENT: u64 = 0x494e_4349_4400_0001;

/// Salts which kind an ambient incident is.
pub const SALT_INCIDENT_KIND: u64 = 0x494e_4349_4400_0002;

/// Salts the guid an incident is minted with.
pub const SALT_INCIDENT_GUID: u64 = 0x494e_4349_4400_0003;

/// Salts a unit's crew guid.
pub const SALT_CREW: u64 = 0x4352_4557_0000_0001;

/// How many fixed steps one ambient draw covers.
///
/// 1 800 — thirty seconds at 60 Hz. The draw is taken on the *epoch*
/// (`step / AMBIENT_PERIOD`) rather than on the step, which is what makes it
/// sparse without a per-step random walk: a block is asked once every thirty
/// seconds whether something has happened to it, and the answer is a pure
/// function of `(block, epoch)` that both hosts compute without exchanging a
/// byte.
pub const AMBIENT_PERIOD: u64 = 1800;

/// The chance one block has an incident in one epoch.
///
/// One in fifty. On a settlement of forty blocks that is a little under one
/// incident every thirty seconds somewhere in town — often enough that a player
/// crossing a city hears sirens, rare enough that a station's two ambulances are
/// not permanently out.
pub const AMBIENT_CHANCE: f64 = 0.02;

/// **Whether something has happened at this block in this epoch**, and what.
///
/// A pure function of `(block, epoch)`: no counter, no stored state, no RNG —
/// the house doctrine (`agent_rand`), so two hosts set the same town on fire at
/// the same moment without exchanging anything.
///
/// A **fire** and a **medical collapse** only. An ambient *crime* is deliberately
/// not drawn: a crime with no criminal is a siren with nothing behind it, and
/// the crimes this wave dispatches to are the ones somebody actually committed —
/// WPN1's witness log. Wave EMS3 gives them profiles.
pub fn ambient_draw(block: Uuid, epoch: u64) -> Option<IncidentKind> {
    if crate::crowd::agent_unit(block, epoch, SALT_INCIDENT) >= AMBIENT_CHANCE {
        return None;
    }
    let which = crate::crowd::agent_unit(block, epoch, SALT_INCIDENT_KIND);
    Some(if which < 0.5 {
        IncidentKind::Fire {
            building: block,
            intensity: 1.0,
        }
    } else {
        // The person is the block: an ambient collapse has no named victim, and
        // naming the building is more honest than minting a guid for somebody
        // who does not exist. The applier reads it only to decide where the
        // paramedic kneels.
        IncidentKind::Medical {
            npc: block,
            severity: 1,
        }
    })
}

/// **The guid an incident is minted with** — content-addressed, so two hosts
/// that opened the same incident agree about its identity without a counter.
///
/// A counter would have been the obvious shape and is the wrong one: a host that
/// started mid-trace, or one that refused an incident the other took, would
/// number every later incident differently for ever. This is `mix64` over what
/// the incident *is* — its kind, where it is and the step it opened on — which
/// is the same argument `inf_ecs::traffic::parked_car_guid` and the P22 debris
/// guids rest on.
pub fn incident_guid(kind: IncidentKind, at: DVec3, step: u64) -> Uuid {
    let q = |v: f64| -> u64 {
        // Quantized to a millimetre before it is hashed, so two hosts that
        // computed a position through different-but-equal arithmetic mint one
        // guid. (They do not — the position is derived — but a guid is an
        // identity and an identity that could depend on a last bit is a bug
        // waiting for a compiler flag.)
        if v.is_finite() {
            (v * 1000.0).round() as i64 as u64
        } else {
            0
        }
    };
    let seed = Uuid::from_u64_pair(
        q(at.x) ^ (u64::from(kind.as_u8()) << 56),
        q(at.z) ^ q(at.y).rotate_left(17),
    );
    let hi = crate::crowd::agent_rand(seed, step, SALT_INCIDENT_GUID);
    let lo = crate::crowd::agent_rand(seed, step ^ 0x5a5a_5a5a, SALT_INCIDENT_GUID);
    Uuid::from_u64_pair(hi, lo)
}

/// **The person who drives a unit and works its scene** — derived from the
/// chassis.
///
/// `inf_ecs::traffic::driver_guid`'s shape and its reason: the crew is not
/// authored, so it needs a stable identity both hosts mint the same way. A
/// **different salt** from the traffic driver's, so a unit that was once a
/// traffic car could not end up with one body claimed by two systems.
pub fn crew_guid(chassis: Uuid) -> Uuid {
    let n = crate::crowd::agent_rand(chassis, 0, SALT_CREW);
    let m = crate::crowd::agent_rand(chassis, 1, SALT_CREW);
    Uuid::from_u64_pair(n, m)
}

// ── how an incident is answered ─────────────────────────────────────────────

/// How close a unit has to get to be **at** an incident, metres.
///
/// Twelve. A fire appliance is 7.8 m long and stops in the road outside the
/// building rather than on top of it, so the distance from the chassis origin to
/// the thing that is burning is most of a vehicle plus a pavement.
pub const ON_SCENE_M: f64 = 12.0;

/// How close a returning unit has to get to its own space to be home, metres.
///
/// Eight — the parking pitch less a car, so a unit that has been shunted one
/// space along still arrives.
pub const HOME_M: f64 = 8.0;

/// How far from **its own vehicle** a crew member stands, metres, on the line
/// toward the scene.
///
/// # Beside the truck, not on top of the incident, and that is the honest rule
///
/// The first cut stood the crew `SCENE_STAND_M` from the *incident*, which is
/// right for a patient lying in the road and wrong for the case the gate
/// actually stages: a building fire is at a block's centre, the nearest
/// carriageway node is forty metres away, and a crew placed at the fire would
/// have **teleported through a wall** to get there.
///
/// So a crew member gets out of its vehicle and stands four metres from it,
/// facing what it came for. Nobody walks into the building: this engine has no
/// path from a road to a room a unit could follow, and pretending otherwise
/// would be a body inside geometry. What reaches the fire is the hose
/// (`extinguish_beams`), which is a line and does not care.
pub const SCENE_STAND_M: f64 = 4.0;

/// How much road a unit may have left and still count as **arrived**, metres.
///
/// Six. [`ON_SCENE_M`] answers "is the incident within reach of where I am",
/// which is the right question for something in the street and the wrong one for
/// something inside a building: a fire at a block's centre is forty metres from
/// the nearest lane, and a unit that only ever tested the distance would sit at
/// the end of its own route for ever with the fire still burning.
///
/// So a unit has arrived when it is near the thing **or** when it has run out of
/// road. Six metres is [`crate::traffic::STANDING_GAP_M`] — one car — which is
/// as close to the end of a lane as a vehicle that stops behind things can get.
pub const PATH_END_M: f64 = 6.0;

/// How much of a fire's intensity one appliance puts out per second.
///
/// A fifth, so a full-intensity fire takes **five seconds of a crew on scene**
/// to put out. That is a game's minute rather than a real one, and the number is
/// a *rate* rather than a duration precisely so a bigger fire takes longer — a
/// half-intensity blaze is two and a half seconds and reads as the smaller
/// emergency it is.
pub const SUPPRESSION_PER_S: f64 = 0.2;

/// How long a paramedic works on a patient before the incident is closed,
/// seconds.
pub const STABILIZE_S: f64 = 6.0;

/// How long the police stand at a crime scene before it is closed, seconds.
///
/// Longer than a stabilisation, because securing a scene is the one of the three
/// that is *waiting* rather than *working*.
pub const SECURE_S: f64 = 10.0;

/// **What it costs each candidate to get there** — one `inf_nav` search per
/// candidate, over the level's own carriageway.
///
/// # Why the search and not the crow's flight
///
/// A straight-line distance is one subtraction and is wrong in exactly the case
/// that matters: an ambulance across a river, or on the other side of a
/// settlement's one bridge, is *near* and cannot get there. `NavRoute::cost_m`
/// is the sum of the **edge** costs, so a builder that made a stair — or a
/// bridge, or a ferry ramp — dear is telling the search something the geometry
/// does not say, and this is the number that hears it.
///
/// Lives here, in the deciding half, rather than in the applier: the applier may
/// not decide anything, and *which* unit goes is the whole decision. It also
/// keeps `inf-physics` from having to name `inf-nav` at all.
///
/// A candidate whose home is off the graph, or which cannot reach `dest`, is
/// **left out** rather than given an infinite cost — the two are the same to
/// [`nearest_unit`] and a `Vec` a caller can count is a better diagnostic than a
/// column of infinities.
pub fn route_costs(
    graph: &inf_nav::NavGraph,
    dest: inf_nav::NavNodeId,
    homes: &[(Uuid, DVec3)],
) -> Vec<(Uuid, f64)> {
    let mut out: Vec<(Uuid, f64)> = Vec::with_capacity(homes.len());
    for (guid, home) in homes {
        let Some(from) = graph.nearest_planar(*home, f64::INFINITY) else {
            continue;
        };
        if let Some(route) = inf_nav::route(graph, from, dest).route() {
            out.push((*guid, route.cost_m));
        }
    }
    out
}

/// **Where on the carriageway an incident is** — the node a route to it ends at.
///
/// `None` for a level with no streets, which is the honest answer to "how do I
/// drive there" for a place with no roads.
pub fn scene_node(graph: &inf_nav::NavGraph, at: DVec3) -> Option<inf_nav::NavNodeId> {
    graph.nearest_planar(at, f64::INFINITY)
}

/// **Which unit goes** — the nearest free one of the right service, by route
/// cost.
///
/// `costs` is `(chassis, cost_m)` for every candidate the caller has already
/// routed; this is the choosing rule alone, hoisted out of the applier so it can
/// be tested without a carriageway.
///
/// **Ties break on the `Guid`**, which is the whole reason this is a function:
/// a station parks two identical ambulances the same distance from the same
/// junction more often than not, and a `min_by` over an unsorted map answers
/// whichever the iterator reached first. `f64::total_cmp` and then the guid, so
/// two hosts send the same ambulance.
pub fn nearest_unit(costs: &[(Uuid, f64)]) -> Option<Uuid> {
    costs
        .iter()
        .filter(|(_, c)| c.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|(g, _)| *g)
}

// ── the scene ───────────────────────────────────────────────────────────────

/// **What a crew member does at a scene**, by service.
///
/// The one place the three services meet the pose pipeline's vocabulary, so a
/// paramedic cannot kneel in one system and stand in another.
///
/// A **paramedic kneels** — the patient is on the ground, and everything else a
/// paramedic does follows from that. A **firefighter stands**: the appliance's
/// line is worked from the feet, and this engine has no two-handed hose pose.
/// An **officer stands**, which is what securing a scene is.
pub fn scene_posture(kind: UnitKind) -> crate::components::SlotPosture {
    match kind {
        UnitKind::Ambulance => crate::components::SlotPosture::Kneel,
        UnitKind::Fire | UnitKind::Police => crate::components::SlotPosture::Stand,
    }
}

/// Salts a smoke puff's guid.
pub const SALT_PUFF: u64 = 0x5055_4646_0000_0001;

/// How often a burning building lets go of a puff, in fixed steps.
///
/// Twenty — three a second. Against [`PUFF_LIFETIME_S`] that is about eleven
/// alive at once per fire, which is a column of smoke rather than a cloud and is
/// eleven entities rather than a particle system this engine does not have.
pub const PUFF_PERIOD: u64 = 20;

/// How long one puff lives, seconds.
pub const PUFF_LIFETIME_S: f64 = 3.5;

/// How fast a puff rises, m/s.
pub const PUFF_RISE_MPS: f64 = 1.6;

/// How wide a puff starts, metres, and it grows with age.
pub const PUFF_SIZE_M: f64 = 2.2;

/// The most puffs one level holds at once.
///
/// Sixty-four — six simultaneous fires' worth. A **cost** bound and a refusal:
/// past it a fire simply stops smoking, which is a visible outcome rather than a
/// level that grows an entity every twenty steps for ever.
pub const MAX_PUFFS: usize = 64;

/// **One smoke puff's identity** — content-addressed on its fire and the step it
/// was let go of, so two hosts spawn the same entity without exchanging a byte.
pub fn puff_guid(incident: Uuid, step: u64) -> Uuid {
    let n = crate::crowd::agent_rand(incident, step, SALT_PUFF);
    let m = crate::crowd::agent_rand(incident, step ^ 0x3c3c_3c3c, SALT_PUFF);
    Uuid::from_u64_pair(n, m)
}

// ── the siren ───────────────────────────────────────────────────────────────

/// How often a running siren's emitter is moved, in fixed steps.
///
/// # SIZE THE RING FIRST — the VEH2a loss, not repeated
///
/// `inf_core::DEFAULT_LOG_CAPACITY` is **8 192** commands, and the shipped
/// host's `audio_command_log` is a ring that *evicts*: `RigSpawn::engine_voice`
/// is `false` for all of traffic precisely because a dozen cars pushing two
/// commands a step lost the one `Play` the island's drive gate exists to count.
/// A siren that pushed a position every step would be the same mistake wearing
/// a different hat.
///
/// So it is pushed every **six** steps — ten hertz at 60 Hz — and the
/// arithmetic is written down rather than hoped for:
///
/// * one hot unit costs one `Play`, one `Stop` and **10 commands a second**;
/// * an authored vehicle's engine voice costs **120 a second** (a `SetPitch`
///   and a `SetVolume` every step), so a siren is a **twelfth** of one engine;
/// * four units hot at once — more than a town of this size ever has — is 40 a
///   second, which is 205 seconds of ring.
///
/// And ten hertz is not a compromise on the *sound*: a unit at 12 m/s moves
/// 1.2 m between updates, which is inside the ear's own localisation blur at
/// any distance a siren is audible from.
///
/// # WHERE THIS RUNS OUT, said rather than left to be discovered (EMS2 audit)
///
/// The table above stops at four because a three-unit town does. The **island**
/// parks **seventeen** (`inf_editor_core::island::station_fleet` over its
/// settlements), and seventeen hot at once is 170 a second — **48 seconds** of
/// ring. `ems2_dispatch_gate` measured 4 253 of 8 192 held with an average of
/// 1.7 hot over 250 seconds, so the shipped arithmetic has about a factor of two
/// in hand *at this fleet size* and none at the island's.
///
/// What that costs is worth being exact about, because it is smaller than it
/// looks: `RuntimeSim::audio_command_log` is a **diagnostic** ring and the
/// commands themselves go to the engine through `audio_cmds`, so an overflow
/// silences nothing. What it breaks is a **gate**: a test that counts `Play`s
/// off the front of that log is reading a tail, which is exactly the VEH2a loss
/// this constant exists to not repeat — and `dropped_audio_commands()` is the
/// door that says so. A wave that puts most of an island's fleet on the road at
/// once raises `SIREN_POSITION_PERIOD` or the log's capacity; it does not
/// discover this from a flaky count.
pub const SIREN_POSITION_PERIOD: u64 = 6;

/// The mixer bus a siren plays on.
pub const SIREN_BUS: &str = "sfx";

/// A siren's base linear volume.
///
/// 0.9 — under unity, because the whole point of the attenuation curve below is
/// that distance decides how loud it is, and a source that starts clipped has
/// nowhere to go.
pub const SIREN_VOLUME: f64 = 0.9;

/// Inside this many metres a siren is at full volume.
pub const SIREN_MIN_DISTANCE_M: f64 = 8.0;

/// Beyond this many metres a siren is silent.
///
/// Two hundred and twenty. A real two-tone carries much further; this is the
/// distance at which a player is *meant* to notice one, and it is deliberately
/// larger than `inf_ecs::traffic::TRAFFIC_FULL_M` (64 m) so a siren is heard
/// well before the vehicle making it is a rig on four rays.
pub const SIREN_MAX_DISTANCE_M: f64 = 220.0;

/// Salts a unit's siren emitter guid.
pub const SALT_SIREN: u64 = 0x5349_5245_4e00_0001;

/// **The emitter a unit's siren plays from** — derived from the chassis.
///
/// A guid of its own rather than the chassis's, because the chassis may already
/// carry an engine voice keyed on exactly that guid, and one source key is one
/// voice: a siren `Play` on the chassis would silence the engine it is sitting
/// on top of.
pub fn siren_guid(chassis: Uuid) -> Uuid {
    let n = crate::crowd::agent_rand(chassis, 0, SALT_SIREN);
    let m = crate::crowd::agent_rand(chassis, 1, SALT_SIREN);
    Uuid::from_u64_pair(n, m)
}

/// **One thing the audio step should do about one siren this step.**
///
/// Decided here, on the sim side, so that the two hosts' fenced audio blocks
/// contain a `match` and no policy at all: what a siren *is*, when it starts,
/// how often it moves and when it stops are all facts about the simulation, and
/// a host that decided any of them would be the second answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SirenCue {
    /// Begin a looping spatial voice at `at`.
    Start { source: Uuid, at: DVec3 },
    /// Move the running voice to `at` — every [`SIREN_POSITION_PERIOD`] steps.
    Move { source: Uuid, at: DVec3 },
    /// The unit has stopped running hot; silence it.
    Stop { source: Uuid },
}

impl SirenCue {
    /// The emitter guid this cue addresses.
    pub fn source(self) -> Uuid {
        match self {
            SirenCue::Start { source, .. }
            | SirenCue::Move { source, .. }
            | SirenCue::Stop { source } => source,
        }
    }
}

/// **What the audio step should do about this level's sirens this step** — the
/// list the two fenced host blocks drain.
///
/// Rebuilt by the dispatch step and read at the audio phase of the **same**
/// step, which is the `GameplayReport::hits` shape: a decision the sim took at
/// phase 7 turned into commands at phase 26, with nothing in between able to
/// change it.
///
/// Empty and allocation-free on a level with no dispatcher.
pub fn siren_cues(world: &EcsWorld) -> &[SirenCue] {
    dispatch_of(world)
        .map(|d| d.sirens.as_slice())
        .unwrap_or(&[])
}

// ── the yield ───────────────────────────────────────────────────────────────

/// How far behind a car a siren is felt, metres.
///
/// Sixty. It is `inf_physics::d3::traffic::LOOK_AHEAD_M`'s number seen from the
/// other end — four seconds at a 50 km/h limit — so a car begins moving over
/// about as far out as it begins braking for a queue. Further than that and a
/// street pulls over for a siren three blocks away, which reads as a bug.
pub const YIELD_RANGE_M: f64 = 60.0;

/// How far to either side of a car's own heading a siren behind it counts,
/// metres.
///
/// Six — a carriageway. A unit on the parallel street one block over is not
/// behind this car in any sense that matters, and a rule with no lateral bound
/// would have a whole junction pulling over for a unit crossing it.
pub const YIELD_CORRIDOR_M: f64 = 6.0;

/// **How fast a yielding car keeps rolling**, m/s, while it moves over.
///
/// # The half of the yield that a steering bias alone cannot do
///
/// `drive_intent` answers **"nothing and the handbrake"** below
/// [`STOPPED_MPS`](crate::traffic::STOPPED_MPS) — which is right for a car
/// waiting at a queue and is a deadlock for a car being asked to get out of the
/// way, because a stationary car with the handbrake on cannot steer anywhere.
/// Measured on this wave's own gate: a responding unit closed on a stopped queue,
/// every car in it was told to pull over, none of them could, and three units sat
/// in the same street for six thousand steps.
///
/// So a yielding car **creeps**: 1.5 m/s is a walking pace, which is what a car
/// edging onto a kerb does, and it is enough that the 2.6 m of
/// [`YIELD_BIAS_M`] is covered in under two seconds. It applies only while a
/// siren is actually behind — the term is guarded on the same non-zero bias the
/// steering term is — so nothing about an ordinary street changes.
///
/// # WHAT IT OVERRIDES, and which half of that is deliberate (EMS2 audit)
///
/// The floor is a `max` taken **after** `drive_intent` has already minimised the
/// limit against the bend, the gap ahead and the end of the road, so for the
/// second or so a siren is behind it, a yielding car ignores all four:
///
/// * the **gap** is the one this is *for* and it cannot be otherwise. The
///   deadlock being closed is precisely a car stopped at `STANDING_GAP_M` behind
///   the car in front of it: leave the gap clamp in and the creep is zero and the
///   car is pinned in the lane, which is the stop-in-lane design §7 rejected.
///   The 6 m of standing gap is the room it creeps into, and it leaves the
///   corridor sideways before it closes it — measured on the gate, where no
///   yielding car ever contacted the one in front;
/// * the **end of the road** is the half that is *not* deliberate. A car asked
///   to yield within a lookahead of its path's end creeps past it at 1.5 m/s
///   instead of stopping on it. Bounded by how long a siren stays behind (a
///   second or two, so a couple of metres) and by the traffic step re-planning
///   the leg, and it is on the carried list rather than fixed here, because
///   guarding it moves the gate's own byte trace for a metre of overshoot.
pub const YIELD_CREEP_MPS: f64 = 1.5;

/// **How far a yielding car pulls over**, metres, and the number is the whole
/// design.
///
/// 2.6 m, and it is chosen against `inf_physics::d3::traffic::CORRIDOR_HALF_M`
/// (2.5 m) — the half-width inside which the following rule counts a body as
/// *in the way*. A car that pulls over by 2.6 m has left the responding unit's
/// corridor, so `gap_ahead` stops seeing it and the unit's own
/// stopping-distance rule stops braking for it.
///
/// **That is why the pull-over was shipped and the stop-in-lane was not.** See
/// `a_car_that_stops_in_lane_stops_the_ambulance_behind_it` for the
/// measurement: the cheaper design costs zero fields and *deadlocks*.
///
/// # WHERE 2.6 m ACTUALLY PUTS THE CAR, measured off this engine's own street
/// (EMS2 audit)
///
/// The number was chosen against one constraint — clear `CORRIDOR_HALF_M` — and
/// the street it moves into was never measured against it. It is:
///
/// | thing                                    | metres from the centreline |
/// |------------------------------------------|---------------------------:|
/// | the forward lane's centre (half of `DEFAULT_LANE_WIDTH_M` = 3.5) | 1.75 |
/// | that car's right flank (+ 0.92 half-width)                       | 2.67 |
/// | the kerb-parked row (`KERB_PARK_OFFSET_M`)                       | 5.00 |
/// | a parked car's left flank (− 0.92)                               | 4.08 |
///
/// So there is **1.41 m** of clear road between a lane car's flank and the
/// parked row, and a 2.6 m bias asks for 2.6 — putting the yielding car's centre
/// at 4.35 m and its flank at 5.27 m, about **1.2 m into the parked cars**. It
/// does *not* go into the opposing lane: [`yield_bias_m`]'s sign is `right_of`'s
/// and the oncoming lane is the other way. And [`YIELD_CREEP_MPS`] removes the
/// gap clamp that would otherwise brake for what it is moving into, so the
/// contact is at a walking pace and rapier resolves it.
///
/// The tension is real and has no cheap resolution: the bias **must** exceed
/// 2.5 m to leave the responder's corridor at all, and only 1.41 m of kerb
/// exists. Closing it means narrowing `CORRIDOR_HALF_M`, widening the street, or
/// giving the responder an overtake — all of which are `inf_physics::d3::traffic`
/// decisions rather than this constant's. It is on the wave's carried list, with
/// this table, rather than discovered from a screenshot of a car in a hedge.
pub const YIELD_BIAS_M: f64 = 2.6;

/// **How far right this car should aim to let a siren past** — the whole yield
/// rule, as a pure function.
///
/// `hot` is [`crate::dispatch`]'s own running-hot list (position only; the guid
/// is the caller's business). `at` and `forward` are the car's.
///
/// A unit counts when it is **behind** this car — `d · forward > 0` where `d`
/// runs from the unit to the car — inside [`YIELD_RANGE_M`] and inside
/// [`YIELD_CORRIDOR_M`] of its heading. A unit *in front* is somebody this car
/// is already going to meet, and moving over for it would be moving into it.
///
/// `0.0` on a street with no sirens on it, which is every street in every level
/// committed before this wave — and `drive_intent`'s term is guarded on exactly
/// that value, so those levels steer the bits they always steered.
///
/// `O(hot)`, and `hot` is at most the units a level owns.
pub fn yield_bias_m(at: DVec3, forward: DVec3, hot: &[(Uuid, DVec3)]) -> f64 {
    if hot.is_empty() || !at.is_finite() {
        return 0.0;
    }
    let len = (forward.x * forward.x + forward.z * forward.z).sqrt();
    if !(len.is_finite() && len > 1.0e-6) {
        return 0.0;
    }
    let f = DVec3::new(forward.x / len, 0.0, forward.z / len);
    for (_, unit) in hot {
        let d = *unit - at;
        if !d.is_finite() {
            continue;
        }
        // Behind: the car is ahead of the unit along its own heading.
        let along = -(d.x * f.x + d.z * f.z);
        if along <= 0.0 || along > YIELD_RANGE_M {
            continue;
        }
        // …and roughly on the same line. `|d x f|` in the ground plane, which is
        // a cross product and no trigonometry (the P14 law).
        let lateral = (d.z * f.x - d.x * f.z).abs();
        if lateral > YIELD_CORRIDOR_M {
            continue;
        }
        return YIELD_BIAS_M;
    }
    0.0
}

// ── the light bar ───────────────────────────────────────────────────────────

/// How fast a responding unit's bar flashes, hertz.
///
/// Two. A real light bar strobes faster than that per head, and this engine has
/// **one** emissive box per vehicle rather than a rotator with two lamps in it —
/// so what is being modelled is the *impression* a bar makes at a distance, not
/// a flash pattern. Two hertz is the rate at which a box that brightens and dims
/// reads as an emergency vehicle rather than as a fault.
pub const SIREN_FLASH_HZ: f32 = 2.0;

/// **One light bar that should be flashing this step, and what it flashes
/// from.**
///
/// The base intensity travels with the cue because it is a *pin*: the authored
/// value is captured the step a unit goes hot and given back the step it stops
/// (see [`DispatchRes::bars`]). A flash that multiplied the live value would
/// dim the bar a little further on every step of a drive and leave it black.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarFlash {
    /// The light-bar entity — [`light_bar_of`]'s answer.
    pub bar: Uuid,
    /// The `Material::emissive_intensity` the level authored.
    pub base_intensity: f32,
    /// The dispatcher's own clock, seconds — what the pulse is a function of.
    ///
    /// Carried rather than read from the sky, so the flash is a pure function of
    /// the *fixed step* and not of a level clock a designer can pause: an
    /// ambulance with its lights on does not stop flashing because somebody
    /// froze the time of day.
    pub clock_s: f64,
}

/// **The light bar on this chassis**, or `None`.
///
/// The child entity named [`LIGHT_BAR_PART`] — the same channel
/// [`unit_kind_of`] recognises a unit by, asked for its guid instead of its
/// colour, so there is one answer to *"which entity is this vehicle's bar"* and
/// not two.
pub fn light_bar_of(world: &EcsWorld, chassis: Uuid) -> Option<Uuid> {
    let entity = world.entity_of(chassis)?;
    for child in world.children_of(entity) {
        let named = world
            .world()
            .get::<crate::components::Name>(child)
            .is_some_and(|n| n.0 == LIGHT_BAR_PART);
        if named {
            return world.guid_of(child);
        }
    }
    None
}

/// **What every flashing bar should be set to this step** — the list the two
/// hosts' fenced blocks drain.
///
/// Empty and allocation-free on a level with no dispatcher.
pub fn bar_flashes(world: &EcsWorld) -> &[BarFlash] {
    dispatch_of(world)
        .map(|d| d.flashes.as_slice())
        .unwrap_or(&[])
}

/// **Write one bar's emissive intensity** — the one door, so the flash and the
/// release cannot spell the write two ways.
///
/// Returns whether anything was written. A bar with no `Material` is a refusal
/// and not a failure: a project that models a light bar without painting one has
/// a vehicle that does not flash, which is a visible outcome rather than a
/// crash.
pub fn set_bar_intensity(world: &mut EcsWorld, bar: Uuid, intensity: f32) -> bool {
    let Some(e) = world.entity_of(bar) else {
        return false;
    };
    let Some(mut m) = world.world_mut().get_mut::<crate::components::Material>(e) else {
        return false;
    };
    if !intensity.is_finite() {
        return false;
    }
    m.emissive_intensity = intensity;
    true
}

// ── the trace ───────────────────────────────────────────────────────────────

/// Bytes one incident folds into [`dispatch_state_bytes`].
///
/// `guid (16) | kind (1) | state (1) | at.x/y/z (24) | opened (8) | unit (16)`.
pub const INCIDENT_TRACE_BYTES: usize = 66;

/// Bytes one unit run folds into [`dispatch_state_bytes`].
///
/// `guid (16) | state (1) | since (8) | incident (16) | path length_m (8)`.
pub const UNIT_TRACE_BYTES: usize = 49;

/// Bytes one search folds into [`dispatch_state_bytes`] — `incident (16) |
/// suspect (16)` (wave EMS3).
pub const SEARCH_TRACE_BYTES: usize = 32;

/// **The dispatcher, as bytes** — the section a replay trace folds.
///
/// The crowd's argument verbatim, one system over: a unit's *state* decides
/// everything the dispatch step does with it and is not a transform anything
/// else folds, so without this section two hosts that assigned different
/// ambulances to one fire would compare equal at every step until one of them
/// happened to solve a chassis the other did not.
///
/// What is folded is the **decision**, not its geometry: a unit's position is
/// its chassis's `Transform` and the sim snapshot already carries it, so the
/// path is folded as its own **length** rather than point by point — one number
/// that changes when a route changes and costs eight bytes instead of a
/// kilometre of them.
///
/// Empty on a level with no dispatcher, which is what keeps every trace
/// committed before this wave byte-identical.
pub fn dispatch_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(res) = dispatch_of(world) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(
        res.incidents.len() * INCIDENT_TRACE_BYTES + res.runs.len() * UNIT_TRACE_BYTES,
    );
    for (guid, inc) in &res.incidents {
        out.extend_from_slice(guid.as_bytes());
        out.push(inc.kind.as_u8());
        out.push(inc.state.as_u8());
        out.extend_from_slice(&inc.at.x.to_le_bytes());
        out.extend_from_slice(&inc.at.y.to_le_bytes());
        out.extend_from_slice(&inc.at.z.to_le_bytes());
        out.extend_from_slice(&inc.opened_step.to_le_bytes());
        out.extend_from_slice(inc.unit.unwrap_or(Uuid::nil()).as_bytes());
    }
    for (guid, run) in &res.runs {
        out.extend_from_slice(guid.as_bytes());
        out.push(run.state.as_u8());
        out.extend_from_slice(&run.since_step.to_le_bytes());
        out.extend_from_slice(run.incident.unwrap_or(Uuid::nil()).as_bytes());
        let len = run.path.as_ref().map(|p| p.length_m()).unwrap_or(0.0);
        out.extend_from_slice(&len.to_le_bytes());
    }
    // EMS3, appended: two hosts searching for different people have diverged in
    // a way no unit state and no position can show — both cars are driving
    // somewhere sensible until one of them arrives at somebody the other has
    // never heard of.
    for (incident, suspect) in &res.searches {
        out.extend_from_slice(incident.as_bytes());
        out.extend_from_slice(suspect.as_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// The set is a set: idempotent, order-free, and absent costs nothing.
    #[test]
    fn a_responder_is_on_duty_until_taken_off_it() {
        let mut w = EcsWorld::new();
        assert!(
            !is_responder(&w, guid(1)),
            "an empty world has no duty roster"
        );
        assert!(responders(&w).is_empty());

        assert!(
            set_responder(&mut w, guid(1), true),
            "the first mark changes it"
        );
        assert!(!set_responder(&mut w, guid(1), true), "the second does not");
        assert!(is_responder(&w, guid(1)));
        assert!(!is_responder(&w, guid(2)));

        set_responder(&mut w, guid(2), true);
        assert_eq!(responders(&w), vec![guid(1), guid(2)], "`Guid` order");

        assert!(set_responder(&mut w, guid(1), false));
        assert!(!is_responder(&w, guid(1)));
        assert!(is_responder(&w, guid(2)));

        clear_dispatch(&mut w);
        assert!(!is_responder(&w, guid(2)));
        assert!(responders(&w).is_empty());
    }

    /// **THE TIE BREAKS ON THE GUID**, which is the whole reason
    /// [`nearest_unit`] is a function.
    ///
    /// A station parks two identical ambulances the same distance from the same
    /// junction more often than not, and a `min_by` over a map answers whichever
    /// the iterator reached first. Two hosts that iterated differently would send
    /// different ambulances to one fire — a divergence no *position* check can
    /// see, because both cars are in the right place until one of them starts
    /// driving.
    #[test]
    fn two_units_at_one_distance_are_chosen_by_guid() {
        // Given in the WRONG order on purpose: a rule that kept the first finite
        // entry would answer `guid(9)`.
        let tied = [(guid(9), 120.0), (guid(2), 120.0), (guid(5), 120.0)];
        assert_eq!(nearest_unit(&tied), Some(guid(2)));
        // …and the cost still wins when there is one.
        let apart = [(guid(2), 120.0), (guid(9), 11.5)];
        assert_eq!(nearest_unit(&apart), Some(guid(9)));
        // A non-finite cost is not a candidate — a unit that cannot get there is
        // not "infinitely far", it is out.
        let broken = [(guid(1), f64::NAN), (guid(3), f64::INFINITY)];
        assert_eq!(nearest_unit(&broken), None);
        assert_eq!(nearest_unit(&[]), None);
    }

    /// **THE AMBIENT DRAW IS SPARSE, DETERMINISTIC AND SPREAD.**
    ///
    /// Three claims, and the third is the one a constant-`None` implementation
    /// passes the other two of: over a town's worth of blocks and two minutes'
    /// worth of epochs the draw fires at about [`AMBIENT_CHANCE`], it fires the
    /// same way twice, and it produces **both** kinds.
    #[test]
    fn the_ambient_draw_is_sparse_and_reproducible() {
        const BLOCKS: u128 = 200;
        const EPOCHS: u64 = 120;
        let mut fires = 0u32;
        let (mut fire, mut medical) = (0u32, 0u32);
        for b in 0..BLOCKS {
            for e in 0..EPOCHS {
                let Some(kind) = ambient_draw(guid(0xB10C_0000 + b), e) else {
                    continue;
                };
                fires += 1;
                match kind {
                    IncidentKind::Fire {
                        building,
                        intensity,
                    } => {
                        assert_eq!(building, guid(0xB10C_0000 + b));
                        assert!((0.0..=1.0).contains(&intensity));
                        fire += 1;
                    }
                    IncidentKind::Medical { .. } => medical += 1,
                    IncidentKind::Crime { .. } => {
                        panic!("the ambient draw produced a crime with no criminal")
                    }
                }
            }
        }
        let trials = f64::from(BLOCKS as u32) * EPOCHS as f64;
        let rate = f64::from(fires) / trials;
        println!(
            "EMS2 ambient: {fires} of {trials:.0} draws ({rate:.4} against \
             {AMBIENT_CHANCE}) — {fire} fire(s), {medical} medical"
        );
        assert!(
            rate > AMBIENT_CHANCE * 0.5 && rate < AMBIENT_CHANCE * 2.0,
            "the draw fired at {rate:.4} against a declared {AMBIENT_CHANCE}"
        );
        assert!(fire > 0 && medical > 0, "the draw only ever makes one kind");
        // …and it is a pure function.
        for b in 0..20u128 {
            for e in 0..5u64 {
                assert_eq!(
                    ambient_draw(guid(b), e),
                    ambient_draw(guid(b), e),
                    "the draw is not a pure function of (block, epoch)"
                );
            }
        }
    }

    /// **AN INCIDENT'S IDENTITY IS WHAT IT IS**, not a counter.
    ///
    /// A counter would number every later incident differently on a host that
    /// refused one the other took — for ever. This is content-addressed, so the
    /// same thing at the same place on the same step is the same incident, and
    /// three different things are three.
    #[test]
    fn an_incidents_guid_is_content_addressed() {
        let at = DVec3::new(120.5, 0.0, -40.25);
        let fire = IncidentKind::Fire {
            building: guid(7),
            intensity: 1.0,
        };
        let a = incident_guid(fire, at, 900);
        assert_eq!(a, incident_guid(fire, at, 900), "not a pure function");
        assert_ne!(a, incident_guid(fire, at, 901), "the step is not in it");
        assert_ne!(
            a,
            incident_guid(fire, at + DVec3::X, 900),
            "the place is not in it"
        );
        assert_ne!(
            a,
            incident_guid(IncidentKind::Crime { severity: 1 }, at, 900),
            "the kind is not in it"
        );
        // **The intensity is NOT an identity.** A fire that has been partly put
        // out is the same fire, and a guid that moved as it burned down would
        // have made every step of a suppression a new incident.
        assert_eq!(
            a,
            incident_guid(
                IncidentKind::Fire {
                    building: guid(7),
                    intensity: 0.25
                },
                at,
                900
            ),
            "a fire changed identity as it was put out"
        );
    }

    /// **THE TRACE IS EMPTY WHEN THERE IS NO DISPATCHER, AND THE RIGHT SIZE WHEN
    /// THERE IS.**
    ///
    /// The first half is what keeps every hash committed before this wave
    /// byte-identical; the second is what says the section carries the *decision*
    /// rather than a length prefix.
    #[test]
    fn the_dispatch_trace_is_empty_until_there_is_something_to_say() {
        let mut w = EcsWorld::new();
        assert!(dispatch_state_bytes(&w).is_empty());

        let mut res = DispatchRes::default();
        res.incidents.insert(
            guid(1),
            Incident {
                kind: IncidentKind::Crime { severity: 2 },
                at: DVec3::new(1.0, 2.0, 3.0),
                state: IncidentState::Assigned,
                opened_step: 12,
                unit: Some(guid(4)),
                resolved_step: None,
            },
        );
        res.runs.insert(
            guid(4),
            UnitRun {
                state: UnitState::EnRoute,
                incident: Some(guid(1)),
                since_step: 12,
                path: None,
            },
        );
        w.world_mut().insert_resource(res.clone());
        let bytes = dispatch_state_bytes(&w);
        assert_eq!(bytes.len(), INCIDENT_TRACE_BYTES + UNIT_TRACE_BYTES);

        // …and it MOVES when the decision does. A trace that folded only the
        // positions would be identical here, which is the whole reason this
        // section exists: two hosts that sent different units to one incident are
        // in the same world with different plans.
        let mut other = res.clone();
        other.runs.get_mut(&guid(4)).expect("the run").state = UnitState::Returning;
        w.world_mut().insert_resource(other);
        assert_ne!(
            bytes,
            dispatch_state_bytes(&w),
            "a unit that turned round left the trace unchanged"
        );
    }
}
