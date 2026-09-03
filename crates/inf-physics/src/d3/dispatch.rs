//! **The dispatch fixed-step door** (wave EMS2): what has happened, who is
//! going, and the stick their driver is handed.
//!
//! The `inf_physics` half of [`inf_ecs::dispatch`], and the fourth instance of
//! this crate's own split — `inf_ecs::vehicle` decides and [`super::vehicle`]
//! applies; `inf_ecs::traffic` decides and [`super::traffic`] applies;
//! `inf_ecs::movement` decides and [`super::movement`] applies. Everything here
//! touches rapier or the ECS; nothing here decides anything. The services, the
//! lifecycles, the recogniser and the choosing rule are on the other side of
//! that wall and are unit-tested without a world.
//!
//! # There is ONE driving rule in this engine and this is not a second one
//!
//! A responding unit is steered by [`inf_ecs::traffic::drive_intent`] — the same
//! pure function a commuter's driver holds — over a path built by
//! [`inf_ecs::traffic::drive_path`], and the intent is written onto the **crew
//! member's** `CharacterMovement` exactly as `super::traffic::steer_car` writes a
//! traffic driver's. It then travels the same road every other stick in this
//! engine does: `step_character_movement` hands it to
//! `VehicleControls::from_intent`, and `step_vehicles` turns that into wheel
//! forces. A dispatcher that wrote controls directly would have been a second
//! answer to *"how does an AI drive"*, and an ambulance would have braked
//! differently from a taxi.
//!
//! # Visibility never filters this
//!
//! An incident opens because something happened, not because somebody was
//! looking — the ambient draw is a pure function of `(block, epoch)` and the
//! crime feed is WPN1's witness log. A unit responds whether or not the hero can
//! see it. That is the P20 law this tree keeps re-proving, and the one thing
//! that makes a town feel alive rather than staged.
//!
//! # What it costs on a level that has no fleet
//!
//! One `block_stamp` walk (which `sync_society` and `sync_carriageway` already
//! make in the two phases before this one) and one `get_resource`. No
//! allocation, no route, no resource inserted — so every level committed before
//! this wave steps exactly the bytes it stepped before.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, MovementMode, SeatState, Transform};
use inf_ecs::dispatch::{
    self, DispatchRes, FleetRes, IncidentKind, IncidentState, UnitKind, UnitRun, UnitState,
};
use inf_ecs::math::Vec3d;
use inf_ecs::traffic::{self, DriveView};
use inf_ecs::EcsWorld;

use super::PhysicsBridge3D;

/// How many incidents may be assigned in one fixed step.
///
/// **One.** An assignment is a Dijkstra over the whole carriageway per candidate
/// unit, and `inf_ecs::traffic::TRAFFIC_PLANS_PER_STEP` is the precedent: a town
/// that woke up to four simultaneous emergencies pays for them over four steps —
/// sixty-seventh of a second — rather than in one spike a frame budget can see.
pub const ASSIGNS_PER_STEP: usize = 1;

/// What one [`step_dispatch`] did — the instrument's read, and the gate's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DispatchStats {
    /// Units the level owns.
    pub units: usize,
    /// Incidents open right now (every state, including `Resolved` ones still
    /// inside `INCIDENT_KEEP_STEPS`).
    pub incidents: usize,
    /// Opened on this step.
    pub opened: usize,
    /// Assigned to a unit on this step.
    pub assigned: usize,
    /// Reached a scene on this step.
    pub arrived: usize,
    /// Resolved on this step.
    pub resolved: usize,
    /// Units steered on this step — the falsifier for the whole clause: a
    /// dispatcher that assigned everything and drove nothing reads zero here.
    pub steered: usize,
    /// Units back in station on this step.
    pub returned: usize,
    /// Incidents that found no free unit on this step.
    pub unanswered: usize,
    /// Units running with lights and siren right now.
    pub running_hot: usize,
}

/// **Advance the dispatcher one fixed step.**
///
/// Called by each host in its own `dispatch` phase, immediately after the
/// traffic and before the physics sync — the traffic's own placement argument,
/// one system along: a crew body that materializes this step has to be mirrored
/// by the sync on the same step, and the driver's **intent** has to be written
/// before `character move` reads it six phases later.
///
/// The sequence, all of it a pure function of sim state:
///
/// 1. derive the fleet if the level's blocks moved
///    ([`inf_ecs::dispatch::sync_fleet`]);
/// 2. open what has happened — the witness log's crimes, the bodies on the
///    ground, and the ambient draw;
/// 3. assign at most [`ASSIGNS_PER_STEP`] of them, nearest free unit by route
///    cost;
/// 4. drive every unit that is going somewhere, through the one driving rule;
/// 5. work every scene a crew is standing at, and send home what is finished.
pub fn step_dispatch(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, dt: f64) -> DispatchStats {
    dispatch::sync_fleet(world);
    let Some(fleet) = world.world().get_resource::<FleetRes>().cloned() else {
        return DispatchStats::default();
    };
    // **Absent costs nothing.** A level with no emergency vehicle in it never
    // gets a `DispatchRes` at all, so `dispatch_state_bytes` stays empty and
    // every trace committed before this wave is byte-identical.
    //
    // **"Never had one" and "no longer has one" are different worlds** (EMS2
    // audit), and the first cut returned on both. A fleet that goes empty after
    // having units — every station in a cell that streamed out — left a live
    // `DispatchRes` frozen mid-response: its `sirens` list never cleared, so the
    // hosts' fenced audio blocks re-pushed the same `Move` **every step for the
    // rest of the session** into a ring that evicts, and its crews were never
    // retired. So the door is the RESOURCE and not the fleet: once a dispatcher
    // exists, the step below runs and retires it properly — every run is dropped,
    // every crew is parked, every siren gets its `Stop`.
    if fleet.units.is_empty() && world.world().get_resource::<DispatchRes>().is_none() {
        return DispatchStats::default();
    }
    let mut res = world
        .world_mut()
        .remove_resource::<DispatchRes>()
        .unwrap_or_default();
    let step = res.steps;
    // The runs follow the fleet: a unit the derivation dropped takes its run
    // with it, and a new one starts in station.
    //
    // **And its CREW goes with it** (EMS2 audit). A run is not only a row in a
    // map: a unit that is out has a body in the world and a name on the duty
    // roster, and both are put there by `ensure_crew` and taken away by `park`
    // alone. Dropping the row silently left the person — standing in the road,
    // `Driving` a chassis that is no longer in the fleet, and **exempt from the
    // panic for the rest of the process**, because `RespondersRes` is only ever
    // released at the station. That is reachable without anything going wrong:
    // a cell whose station streams out, or a block that moves past
    // `STATION_CLAIM_M`, re-derives a fleet that no longer holds the unit.
    //
    // So a dropped run is retired the way an arriving one is, through the same
    // two doors, and only for units that had actually left their bay — a
    // parked unit has no crew and `entity_of` answers `None` for it.
    let dropped: Vec<Uuid> = res
        .runs
        .keys()
        .filter(|g| !fleet.units.contains_key(*g))
        .copied()
        .collect();
    for chassis in dropped {
        park(
            world,
            bridge,
            &mut res,
            chassis,
            dispatch::crew_guid(chassis),
        );
    }
    res.runs.retain(|g, _| fleet.units.contains_key(g));
    for guid in fleet.units.keys() {
        res.runs.entry(*guid).or_default();
    }

    let mut stats = DispatchStats {
        units: fleet.units.len(),
        ..DispatchStats::default()
    };

    open_incidents(world, &mut res, step, &mut stats);
    assign(world, &fleet, &mut res, step, &mut stats);
    run_units(world, bridge, &fleet, &mut res, step, dt, &mut stats);
    forget_old(&mut res, step);
    sound_and_light(world, &mut res, step, dt);
    smoke(world, &mut res, step, dt);

    stats.incidents = res.incidents.len();
    stats.running_hot = res.runs.values().filter(|r| r.state.running_hot()).count();
    // **Only when something actually moved**, which is `step_traffic`'s own rule
    // and its reason: a level with a fleet parked in its bays and nothing
    // happening must not bump the version every step and make both projectors
    // rebuild a scene that did not change. A responding unit's chassis, its
    // flashing bar and a rising smoke puff are all things the renderer has to
    // see again; a station at rest is not.
    let moved = stats.running_hot > 0 || !res.puffs.is_empty();
    res.steps += 1;
    world.world_mut().insert_resource(res);
    if moved {
        world.mark_dirty();
    }
    stats
}

// ── 2. what has happened ────────────────────────────────────────────────────

/// Open the incidents this step's world implies, from the three feeds.
fn open_incidents(
    world: &mut EcsWorld,
    res: &mut DispatchRes,
    step: u64,
    stats: &mut DispatchStats,
) {
    // ── (a) THE CRIME FEED IS WPN1's WITNESS LOG ──
    //
    // Written by `step_witness` on the step an act happens and read here, which
    // is the reader that module's own doc named: *"The day something reads it —
    // EMS3's dispatcher — folding it becomes the right call"*. EMS2 is that day
    // one wave early, because a shot fired is a crime whether or not anybody has
    // a description of who fired it.
    //
    // Read FORWARD by step, so an act cannot be reported twice and a ring that
    // has evicted its front is not re-scanned.
    let mut newest = res.seen_act_step;
    let acts: Vec<(DVec3, u64, u8)> = inf_ecs::witness::witnessed(world)
        .iter()
        .filter(|a| a.step > res.seen_act_step)
        .map(|a| {
            (
                a.at,
                a.step,
                match a.kind {
                    inf_ecs::witness::ActKind::Shot => 1,
                    inf_ecs::witness::ActKind::Killed => 2,
                },
            )
        })
        .collect();
    for (at, act_step, severity) in acts {
        newest = newest.max(act_step);
        // **One crime scene, not one per round.** A burst is many acts in one
        // place, and a dispatcher that opened an incident per shot would empty a
        // station into one street corner. An open crime inside
        // `CRIME_MERGE_M` of this one is this one.
        if res.incidents.values().any(|i| {
            matches!(i.kind, IncidentKind::Crime { .. })
                && i.state != IncidentState::Resolved
                && (i.at - at).length() < CRIME_MERGE_M
        }) {
            continue;
        }
        open(res, IncidentKind::Crime { severity }, at, step, stats);
    }
    res.seen_act_step = newest;

    // ── (b) SOMEBODY IS ON THE GROUND ──
    //
    // I6's `Health::dead` bodies, latched by WPN1's `Downed`. The latch is what
    // makes this cheap and correct at once: a body is handed to the ragdoll
    // once, and it is a medical incident once.
    for npc in downed_bodies(world) {
        // **A body an ambulance has already been to does not call another one.**
        // See `DispatchRes::treated` for the honest sentence about what having
        // been to one means, and for the loop this closes.
        if res.treated.contains(&npc) {
            continue;
        }
        if res
            .incidents
            .values()
            .any(|i| matches!(i.kind, IncidentKind::Medical { npc: n, .. } if n == npc))
        {
            continue;
        }
        let Some(at) = body_at(world, npc) else {
            continue;
        };
        open(
            res,
            IncidentKind::Medical { npc, severity: 2 },
            at,
            step,
            stats,
        );
    }

    // ── (c) THE AMBIENT DRAW ──
    //
    // Taken on the EPOCH and not on the step, so the walk over the level's
    // blocks happens once every `AMBIENT_PERIOD` steps and costs nothing on the
    // other one thousand seven hundred and ninety-nine.
    if !step.is_multiple_of(dispatch::AMBIENT_PERIOD) {
        return;
    }
    let epoch = step / dispatch::AMBIENT_PERIOD;
    for (block, at) in blocks_of(world) {
        let Some(kind) = dispatch::ambient_draw(block, epoch) else {
            continue;
        };
        // A building that is already burning does not catch fire again, and a
        // block that already has an ambulance coming does not call a second one.
        if res.incidents.values().any(|i| {
            i.state != IncidentState::Resolved
                && match (i.kind, kind) {
                    (
                        IncidentKind::Fire { building: a, .. },
                        IncidentKind::Fire { building: b, .. },
                    ) => a == b,
                    (
                        IncidentKind::Medical { npc: a, .. },
                        IncidentKind::Medical { npc: b, .. },
                    ) => a == b,
                    _ => false,
                }
        }) {
            continue;
        }
        open(res, kind, at, step, stats);
    }
}

/// How close two crimes have to be to be one crime scene, metres.
///
/// Forty — a city block. A firefight produces one act per shooter per step and
/// they are all the same emergency; the number is deliberately the crowd's own
/// `FLEE_M`, because the people running from it are running that far.
pub const CRIME_MERGE_M: f64 = 40.0;

/// Mint one incident, if there is room for it.
///
/// Past [`inf_ecs::dispatch::MAX_OPEN_INCIDENTS`] the seventeenth is **dropped**
/// rather than queued — a refusal is a value (P21.4), and a queue nothing can
/// reach is a leak with a deadline. It is counted as unanswered, so a town that
/// is saturated says so rather than looking quiet.
fn open(
    res: &mut DispatchRes,
    kind: IncidentKind,
    at: DVec3,
    step: u64,
    stats: &mut DispatchStats,
) {
    if !at.is_finite() {
        return;
    }
    let open_now = res
        .incidents
        .values()
        .filter(|i| i.state != IncidentState::Resolved)
        .count();
    if open_now >= dispatch::MAX_OPEN_INCIDENTS {
        res.unanswered = res.unanswered.saturating_add(1);
        stats.unanswered += 1;
        return;
    }
    let guid = dispatch::incident_guid(kind, at, step);
    if res.incidents.contains_key(&guid) {
        return;
    }
    res.incidents.insert(
        guid,
        inf_ecs::dispatch::Incident {
            kind,
            at,
            state: IncidentState::Reported,
            opened_step: step,
            unit: None,
            resolved_step: None,
        },
    );
    res.opened = res.opened.saturating_add(1);
    stats.opened += 1;
}

/// Every `PcgVolume` block and where it is, in `Guid` order — the ambient draw's
/// subjects.
fn blocks_of(world: &EcsWorld) -> Vec<(Uuid, DVec3)> {
    let mut out: Vec<(Uuid, DVec3)> = Vec::new();
    for e in world.world().iter_entities() {
        let (Some(g), Some(_)) = (
            e.get::<inf_ecs::components::Guid>(),
            e.get::<inf_ecs::components::PcgVolume>(),
        ) else {
            continue;
        };
        let Some(t) = e.get::<inf_ecs::components::GlobalTransform>() else {
            continue;
        };
        let p = t.translation();
        if p.is_finite() {
            out.push((g.0, p));
        }
    }
    out.sort_by_key(|(g, _)| *g);
    out
}

/// Every body that has stopped working, in `Guid` order.
///
/// `inf_ecs::weapon::downed_bodies`' shape — the `Downed` latch read as a
/// census rather than as an event, because a dispatcher that missed the one step
/// a marker was inserted on would never send an ambulance at all.
fn downed_bodies(world: &EcsWorld) -> Vec<Uuid> {
    inf_ecs::weapon::downed(world)
}

/// Where a body is, world metres.
fn body_at(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let e = world.entity_of(guid)?;
    let t = world.world().get::<Transform>(e)?;
    let p = t.translation.to_dvec3();
    p.is_finite().then_some(p)
}

// ── 3. who goes ─────────────────────────────────────────────────────────────

/// Send the nearest free unit of the right service to the oldest open incident
/// **that some free unit could actually go to**.
///
/// # Both halves of that sentence are audit repairs, and the second is a
/// deadlock
///
/// The first cut took the first pending incident in **guid order** — which is
/// hash order, so it was neither the oldest nor anything a reader could predict
/// — and took exactly [`ASSIGNS_PER_STEP`] of them *whether or not the attempt
/// succeeded*. Put together those two make a **permanent starvation**: an
/// incident nobody can ever answer sits at the front of that order for ever and
/// consumes the step's one assignment every step, so every other incident in the
/// table — with free units parked and waiting — is never even considered.
///
/// It is not a corner. `ambient_draw` produces a **fire** half the time, and
/// `inf_editor_core::island::station_fleet` gives a settlement an appliance only
/// if it has a `FireHall`: a town with a police station and no fire hall draws
/// its first ambient fire within a couple of minutes and the dispatcher stops
/// answering *anything* — including the crimes and collapses its own cruisers
/// and ambulances are sitting idle for.
///
/// So the candidate list is filtered by **which services have a free unit right
/// now**, which is `O(units)` and is the question a Dijkstra would otherwise be
/// paid to answer, and it is ordered by `opened_step` with a guid tiebreak — the
/// order this function's own doc claimed from the start. The cost bound is
/// unchanged: at most [`ASSIGNS_PER_STEP`] route searches per step.
///
/// What is **not** closed, and is on the wave's carried list: an incident whose
/// service *does* have a free unit but which no route reaches (`drive_path`
/// answers `None`) still holds the slot. That needs a per-incident refusal the
/// ledger can show, which is a shape this audit did not invent on its own.
fn assign(
    world: &EcsWorld,
    fleet: &FleetRes,
    res: &mut DispatchRes,
    step: u64,
    stats: &mut DispatchStats,
) {
    // Which services could take a call at all this step. `O(units)`, and it is
    // taken before the carriageway so a town whose every unit is out pays for
    // nothing.
    let free: std::collections::BTreeSet<UnitKind> = fleet
        .units
        .iter()
        .filter(|(chassis, _)| {
            res.runs
                .get(*chassis)
                .is_some_and(|r| r.state == UnitState::InStation)
        })
        .map(|(_, unit)| unit.kind)
        .collect();
    // Oldest first, guid tiebreak — and only the ones somebody could go to.
    let mut pending: Vec<(u64, Uuid)> = res
        .incidents
        .iter()
        .filter(|(_, i)| i.state == IncidentState::Reported && free.contains(&i.kind.service()))
        .map(|(g, i)| (i.opened_step, *g))
        .collect();
    pending.sort_unstable();
    let pending: Vec<Uuid> = pending.into_iter().map(|(_, g)| g).collect();
    if pending.is_empty() {
        // Something may still be waiting — it is waiting on a *unit*, not on
        // this function. Counted once (never once per waiting incident), so the
        // magnitude of `unanswered` keeps meaning "steps on which nobody could
        // be sent" rather than "incidents × steps".
        if res
            .incidents
            .values()
            .any(|i| i.state == IncidentState::Reported)
        {
            res.unanswered = res.unanswered.saturating_add(1);
            stats.unanswered += 1;
        }
        return;
    }
    // **The carriageway, once, and only when something is pending.** Building
    // the graph is `O(streets)` and the search over it is a Dijkstra; a level
    // with nothing happening pays for neither.
    let Some(car) = traffic::carriageway_of(world) else {
        // No streets, no route, no unit. Counted rather than silent: a town
        // whose institutions are not on a road answers this way for ever, and
        // that is a fact somebody has to be able to read.
        for _ in pending.iter().take(ASSIGNS_PER_STEP) {
            res.unanswered = res.unanswered.saturating_add(1);
            stats.unanswered += 1;
        }
        return;
    };
    let graph = traffic::carriageway_graph(&car.streets);
    let lanes = car.lanes.clone();
    for incident_guid in pending.into_iter().take(ASSIGNS_PER_STEP) {
        let Some(incident) = res.incidents.get(&incident_guid).copied() else {
            continue;
        };
        let want = incident.kind.service();
        let Some(dest) = dispatch::scene_node(&graph, incident.at) else {
            res.unanswered = res.unanswered.saturating_add(1);
            stats.unanswered += 1;
            continue;
        };
        // The candidates: every unit of the right service that is in its
        // station. The COST is `inf_nav`'s and is asked for on the other side of
        // the wall — an applier may not decide who goes.
        let homes: Vec<(Uuid, DVec3)> = fleet
            .units
            .iter()
            .filter(|(chassis, unit)| {
                unit.kind == want
                    && res
                        .runs
                        .get(*chassis)
                        .is_some_and(|r| r.state == UnitState::InStation)
            })
            .map(|(chassis, unit)| (*chassis, unit.home))
            .collect();
        let costs = dispatch::route_costs(&graph, dest, &homes);
        let Some(chassis) = dispatch::nearest_unit(&costs) else {
            res.unanswered = res.unanswered.saturating_add(1);
            stats.unanswered += 1;
            continue;
        };
        let unit = fleet.units[&chassis];
        let to_yaw = traffic::yaw_of_dir(incident.at - unit.home);
        let Some(path) = traffic::drive_path(
            &graph,
            &lanes,
            unit.home,
            unit.home_yaw_deg,
            incident.at,
            to_yaw,
        ) else {
            res.unanswered = res.unanswered.saturating_add(1);
            stats.unanswered += 1;
            continue;
        };
        res.runs.insert(
            chassis,
            UnitRun {
                state: UnitState::EnRoute,
                incident: Some(incident_guid),
                since_step: step,
                path: Some(path),
            },
        );
        if let Some(i) = res.incidents.get_mut(&incident_guid) {
            i.state = IncidentState::Assigned;
            i.unit = Some(chassis);
        }
        res.assigned = res.assigned.saturating_add(1);
        stats.assigned += 1;
    }
}

// ── 4 + 5. the drive, the scene and the way home ────────────────────────────

/// Advance every unit that is doing something.
fn run_units(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    fleet: &FleetRes,
    res: &mut DispatchRes,
    step: u64,
    dt: f64,
    stats: &mut DispatchStats,
) {
    let archetype = inf_ecs::society::level_archetype(world);
    let mut obstacles: Option<Vec<(Uuid, DVec3)>> = None;
    let chassis_list: Vec<Uuid> = res.runs.keys().copied().collect();
    for chassis in chassis_list {
        let Some(run) = res.runs.get(&chassis).cloned() else {
            continue;
        };
        let unit = fleet.units[&chassis];
        let crew = dispatch::crew_guid(chassis);
        match run.state {
            UnitState::InStation => {}
            UnitState::EnRoute | UnitState::Returning => {
                let target = match run.state {
                    UnitState::EnRoute => run
                        .incident
                        .and_then(|g| res.incidents.get(&g))
                        .map(|i| i.at),
                    _ => Some(unit.home),
                };
                let Some(target) = target else {
                    // The incident it was going to is gone. Turn round rather
                    // than drive to a place that no longer means anything.
                    send_home(world, res, &unit, chassis, step);
                    continue;
                };
                let here = chassis_at(world, bridge, chassis).unwrap_or(unit.home);
                let seated = ensure_crew(world, bridge, chassis, crew, &archetype, here);
                // ── the obstacles, LAZILY and ONCE — `step_traffic`'s own shape
                //    and its reason: gathering them per unit would be
                //    `O(units x world)`, and gathering them unconditionally
                //    would walk a furnished town on every step of a level where
                //    nothing is happening.
                if obstacles.is_none() {
                    obstacles = Some(obstacles_for_units(world, res));
                }
                let obs = obstacles.as_deref().unwrap_or(&[]);
                if seated && steer(world, bridge, chassis, crew, &run, obs) {
                    stats.steered += 1;
                }
                let reach = if run.state == UnitState::EnRoute {
                    dispatch::ON_SCENE_M
                } else {
                    dispatch::HOME_M
                };
                // **Near the thing, OR out of road.** See `PATH_END_M`: an
                // incident inside a building is forty metres from the nearest
                // lane, and a unit that only tested the distance would sit at
                // the end of its own route for ever.
                let out_of_road = run.path.as_ref().is_some_and(|p| {
                    let left = p.length_m() - p.project(here).s_m;
                    left.is_finite() && left <= dispatch::PATH_END_M
                });
                if (here - target).length() <= reach || out_of_road {
                    if run.state == UnitState::EnRoute {
                        arrive(world, bridge, res, chassis, Some(unit.kind), here, step);
                        stats.arrived += 1;
                    } else {
                        park(world, bridge, res, chassis, crew);
                        stats.returned += 1;
                    }
                }
            }
            UnitState::OnScene => {
                if work_the_scene(world, res, chassis, crew, unit.kind, step, dt) {
                    send_home(world, res, &unit, chassis, step);
                    stats.resolved += 1;
                }
            }
        }
    }
}

/// **Everything a responding unit has to not drive into** — every solid body in
/// the world **except the fleet still parked in its station**.
///
/// # Why the parked fleet is not an obstacle, and why that is a rule
///
/// `super::traffic::obstacles_of` answers every body with a position, which is
/// exactly right for a car on a carriageway. A station's apron is not a
/// carriageway: EMS1 parks a fleet nose-to-tail at `EMS_PARK_PITCH_M`, and the
/// drive out of a bay runs **along** that row by construction — so a unit that
/// applied the following rule to its own station would brake for the appliance
/// parked eleven metres in front of it and never leave. Measured on this wave's
/// own fixtures the moment the following rule was added: three units, three
/// assignments, zero departures.
///
/// So a unit that is `InStation` is not in anybody's lane. Everything else
/// still is — real traffic, the hero standing in the road, and **another unit
/// that is under way**, which is the case that matters: two units converging on
/// one junction queue for each other like anybody else.
///
/// `O(world)` and built at most once per fixed step, only on a step something is
/// actually driving.
fn obstacles_for_units(world: &EcsWorld, res: &DispatchRes) -> Vec<(Uuid, DVec3)> {
    let parked: std::collections::BTreeSet<Uuid> = res
        .runs
        .iter()
        .filter(|(_, r)| r.state == UnitState::InStation)
        .map(|(g, _)| *g)
        .collect();
    let mut out = super::traffic::obstacles_of(world);
    if !parked.is_empty() {
        out.retain(|(g, _)| !parked.contains(g));
    }
    out
}

/// Where a unit's chassis is: the solver's body if the bridge has one, the
/// transform otherwise.
///
/// The fall-back is not a nicety. This phase runs **before** the physics sync,
/// so on the step a cell activates the body does not exist yet — and a unit that
/// answered `None` there would have been treated as having no position at all.
fn chassis_at(world: &EcsWorld, bridge: &PhysicsBridge3D, chassis: Uuid) -> Option<DVec3> {
    if let Some(body) = bridge.body_of(chassis) {
        if let Some(p) = bridge.world().body_translation(body) {
            if p.is_finite() {
                return Some(p);
            }
        }
    }
    let e = world.entity_of(chassis)?;
    let t = world.world().get::<Transform>(e)?;
    let p = t.translation.to_dvec3();
    p.is_finite().then_some(p)
}

/// **Put the crew in the seat**, building the body if it is not there yet.
///
/// `super::traffic::ensure_driver`'s shape and its reasons: the same
/// `inf_ecs::crowd::spawn_body` door, the level's own archetype, the same
/// capsule and mesh every resident wears, and the warp already finished —
/// because a crew that was in the station when the call came did not climb in
/// while anybody was watching.
///
/// Marks the crew a **responder** on the way in, which is what makes clause 1's
/// exemption reach them.
fn ensure_crew(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    chassis: Uuid,
    crew: Uuid,
    archetype: &inf_ecs::crowd::CrowdArchetype,
    at: DVec3,
) -> bool {
    dispatch::set_responder(world, crew, true);
    if let Some(e) = world.entity_of(crew) {
        let seated = world
            .world()
            .get::<CharacterMovement>(e)
            .is_some_and(|cm| cm.runtime.seat.vehicle == chassis);
        if seated {
            return true;
        }
        // Coming back from a scene: the same body goes back in the same seat,
        // and back on its feet — a paramedic who drove home on one knee would be
        // the posture write with no release.
        if let Some(mut a) = world.world_mut().get_mut::<inf_ecs::crowd::CrowdAgent>(e) {
            a.posture = inf_ecs::components::SlotPosture::Stand;
            a.posture_t = 0.0;
        }
        if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.mode = MovementMode::Driving;
            cm.runtime.seat = SeatState {
                vehicle: chassis,
                entering: false,
                time_s: 0.0,
                start: Vec3d::from_dvec3(at),
                start_yaw_deg: 0.0,
            };
        }
        super::vehicle::park_collider(bridge, crew, true);
        return true;
    }
    let e = inf_ecs::crowd::spawn_body(world, crew, archetype, at);
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Driving;
        cm.runtime.seat = SeatState {
            vehicle: chassis,
            entering: false,
            time_s: 0.0,
            start: Vec3d::from_dvec3(at),
            start_yaw_deg: 0.0,
        };
    }
    super::vehicle::park_collider(bridge, crew, true);
    true
}

/// **Write the driving unit's stick** — through the one rule.
///
/// `super::traffic::steer_car` verbatim except for where the path comes from: a
/// traffic car follows its own schedule's leg and a unit follows the drive the
/// dispatcher gave it. Everything after that is shared — the view, the intent,
/// the character's `intent_move`, `from_intent`, the wheels.
///
/// Returns whether a stick was actually written, which is the falsifier for the
/// whole clause: a dispatcher that assigned units and steered none reads zero.
fn steer(
    world: &mut EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    crew: Uuid,
    run: &UnitRun,
    obstacles: &[(Uuid, DVec3)],
) -> bool {
    let Some(path) = run.path.as_ref() else {
        return false;
    };
    let Some(body) = bridge.body_of(chassis) else {
        return false;
    };
    let w = bridge.world();
    let (Some(at), Some(rot)) = (w.body_translation(body), w.body_rotation(body)) else {
        return false;
    };
    let linvel = w.body_linvel(body).unwrap_or(DVec3::ZERO);
    let forward = rot * DVec3::Z;
    let s_m = path.project(at).s_m;
    let view = DriveView {
        at,
        forward,
        forward_mps: linvel.dot(forward),
        path,
        s_m,
        // **A unit under way is not held to the town's own limit.** That is what
        // a siren buys and it is the one number a responding vehicle does
        // differently from a taxi; everything else about the drive is identical.
        speed_limit_mps: traffic::street_speed_mps() * RESPONSE_SPEED_FACTOR,
        // A unit under way never yields to itself.
        lateral_bias_m: 0.0,
        // **A siren does not go through the car in front of it.** The same
        // `gap_ahead` every other driver in this engine reads — and it is what
        // makes the yield rule matter rather than decorate: a civilian that
        // stays in the lane is an obstacle this unit brakes for and never passes
        // (`drive_intent` has no overtake), and one that pulls 2.6 m over has
        // left the corridor `gap_ahead` measures.
        //
        // Written as `None` in the first cut, which is a unit that ignores
        // everything in front of it — and what that produced was not a unit that
        // drove through traffic but one that drove INTO it and stopped there,
        // permanently, half a mile short of a fire. Measured on this wave's own
        // gate.
        gap_m: super::traffic::gap_ahead(path, s_m, chassis, crew, obstacles),
        loops: false,
    };
    let intent = traffic::drive_intent(&view);
    let Some(e) = world.entity_of(crew) else {
        return false;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.intent_move = intent.move_input;
        cm.runtime.want_handbrake = intent.handbrake;
        return true;
    }
    false
}

/// How much faster than the sign a responding unit drives.
///
/// **1.4.** The island's streets are signed at 30 km/h, so a unit under way runs
/// at 42 — quick enough that a player watching one go past knows it is in a
/// hurry, and slow enough that the same `drive_intent` corner rule keeps it on
/// the road. It multiplies the *limit* and nothing else: the bend, the gap and
/// the end of the road are unchanged, so a unit still slows for a corner and
/// still stops behind a queue.
pub const RESPONSE_SPEED_FACTOR: f64 = 1.4;

/// **The unit is at the scene**: stop it, and put its crew out.
fn arrive(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    res: &mut DispatchRes,
    chassis: Uuid,
    kind: Option<UnitKind>,
    here: DVec3,
    step: u64,
) {
    // Derived rather than passed: `crew_guid` is a pure function of the chassis
    // and a seventh argument that can be computed from the third is a parameter
    // list nobody can hold.
    let crew = dispatch::crew_guid(chassis);
    let at = res
        .runs
        .get(&chassis)
        .and_then(|r| r.incident)
        .and_then(|g| res.incidents.get(&g))
        .map(|i| i.at)
        .unwrap_or(here);
    if let Some(run) = res.runs.get_mut(&chassis) {
        run.state = UnitState::OnScene;
        run.since_step = step;
        run.path = None;
    }
    if let Some(g) = res.runs.get(&chassis).and_then(|r| r.incident) {
        if let Some(i) = res.incidents.get_mut(&g) {
            i.state = IncidentState::OnScene;
        }
    }
    handbrake(bridge, chassis);
    // The crew stands `SCENE_STAND_M` from ITS OWN VEHICLE, on the line toward
    // the incident — which is where somebody who has just got out of that
    // vehicle is. See `SCENE_STAND_M` for why it is not measured from the
    // incident: a fire is inside a building and nobody walks in.
    let toward = at - here;
    let len = (toward.x * toward.x + toward.z * toward.z).sqrt();
    let dir = if len > 1.0e-6 {
        DVec3::new(toward.x / len, 0.0, toward.z / len)
    } else {
        DVec3::Z
    };
    let stand = here + dir * dispatch::SCENE_STAND_M;
    unseat_crew(world, bridge, crew, stand, dir);
    // ── what the crew DOES, once it is out. Written onto the body's own
    //    `CrowdAgent`, which is where `step_pose_evaluation` reads a posture
    //    from — and which nothing else writes for this body, because a crew
    //    member is a `spawn_body` and not a population record, so `step_crowd`
    //    has never heard of it. One authority, no overwrite.
    if let Some(kind) = kind {
        let posture = dispatch::scene_posture(kind);
        if let Some(e) = world.entity_of(crew) {
            if let Some(mut a) = world.world_mut().get_mut::<inf_ecs::crowd::CrowdAgent>(e) {
                a.posture = posture;
                a.face = dir;
                a.posture_t = 0.0;
            }
        }
    }
}

/// Take the crew out of the seat and stand it at `stand`, facing `dir`.
fn unseat_crew(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    crew: Uuid,
    stand: DVec3,
    dir: DVec3,
) {
    let Some(e) = world.entity_of(crew) else {
        return;
    };
    let yaw = inf_math::patan2_64(dir.x, dir.z).to_degrees();
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Grounded;
        cm.runtime.seat = SeatState::default();
        cm.runtime.intent_move = inf_ecs::math::Vec2d::ZERO;
        cm.runtime.want_handbrake = false;
        if yaw.is_finite() {
            cm.runtime.body_yaw_deg = yaw;
            cm.runtime.target_yaw_deg = yaw;
        }
    }
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(stand.x, stand.y, stand.z);
        if yaw.is_finite() {
            t.rotation.y = yaw;
        }
    }
    super::vehicle::park_collider(bridge, crew, false);
}

/// Hold a stopped unit where it stopped.
///
/// `super::traffic`'s own sentence: a rig with no controls on a graded street
/// rolls away, and a fire appliance that rolled away from its own fire is a
/// thing a player would notice.
fn handbrake(bridge: &mut PhysicsBridge3D, chassis: Uuid) {
    if let Some(v) = bridge.vehicle_mut(chassis) {
        v.control(inf_ecs::vehicle::VehicleControls {
            handbrake: true,
            ..inf_ecs::vehicle::VehicleControls::default()
        });
    }
}

/// **Work the scene** — returns whether the incident is finished.
///
/// One rule per service, and each is the honest thing that service does:
///
/// * a fire crew spends the fire's **intensity** down at
///   [`inf_ecs::dispatch::SUPPRESSION_PER_S`], so a bigger fire takes longer;
/// * a paramedic works for [`inf_ecs::dispatch::STABILIZE_S`];
/// * the police stand at a crime scene for
///   [`inf_ecs::dispatch::SECURE_S`], which is *waiting* rather than working and
///   is therefore the longest of the three.
fn work_the_scene(
    world: &mut EcsWorld,
    res: &mut DispatchRes,
    chassis: Uuid,
    _crew: Uuid,
    kind: UnitKind,
    step: u64,
    dt: f64,
) -> bool {
    let Some(run) = res.runs.get(&chassis).cloned() else {
        return false;
    };
    let Some(incident_guid) = run.incident else {
        return true;
    };
    let Some(incident) = res.incidents.get_mut(&incident_guid) else {
        return true;
    };
    let elapsed_s = step.saturating_sub(run.since_step) as f64 * dt;
    let done = match (&mut incident.kind, kind) {
        (IncidentKind::Fire { intensity, .. }, UnitKind::Fire) => {
            *intensity -= dispatch::SUPPRESSION_PER_S * dt;
            *intensity <= 0.0
        }
        (IncidentKind::Medical { .. }, UnitKind::Ambulance) => elapsed_s >= dispatch::STABILIZE_S,
        (IncidentKind::Crime { .. }, UnitKind::Police) => elapsed_s >= dispatch::SECURE_S,
        // **The wrong service at a scene does nothing, for ever.** It cannot
        // happen — `assign` matches the service to the kind — and saying so is
        // cheaper than a match arm that guesses. A gate that ever saw this would
        // see a unit parked at a scene that never closes, which is a visible
        // failure rather than a silent one.
        _ => false,
    };
    if done {
        let treated = match incident.kind {
            IncidentKind::Medical { npc, .. } => Some(npc),
            _ => None,
        };
        incident.state = IncidentState::Resolved;
        incident.resolved_step = Some(step);
        res.resolved = res.resolved.saturating_add(1);
        if let Some(npc) = treated {
            res.treated.insert(npc);
        }
        let _ = world;
    }
    done
}

/// Turn a unit round and give it the drive home.
fn send_home(
    world: &EcsWorld,
    res: &mut DispatchRes,
    unit: &inf_ecs::dispatch::FleetUnit,
    chassis: Uuid,
    step: u64,
) {
    let from = res
        .runs
        .get(&chassis)
        .and_then(|r| r.incident)
        .and_then(|g| res.incidents.get(&g))
        .map(|i| i.at)
        .unwrap_or(unit.home);
    let path = traffic::carriageway_of(world).and_then(|car| {
        let graph = traffic::carriageway_graph(&car.streets);
        traffic::drive_path(
            &graph,
            &car.lanes,
            from,
            traffic::yaw_of_dir(unit.home - from),
            unit.home,
            unit.home_yaw_deg,
        )
    });
    if let Some(run) = res.runs.get_mut(&chassis) {
        run.state = UnitState::Returning;
        run.since_step = step;
        run.incident = None;
        run.path = path;
    }
}

/// The unit is home: park it, take the crew off duty and off the street.
fn park(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    res: &mut DispatchRes,
    chassis: Uuid,
    crew: Uuid,
) {
    if let Some(run) = res.runs.get_mut(&chassis) {
        run.state = UnitState::InStation;
        run.incident = None;
        run.path = None;
    }
    handbrake(bridge, chassis);
    dispatch::set_responder(world, crew, false);
    if let Some(e) = world.entity_of(crew) {
        super::vehicle::park_collider(bridge, crew, false);
        world.despawn(e);
    }
}

/// Forget resolved incidents once they are older than the ledger keeps.
fn forget_old(res: &mut DispatchRes, step: u64) {
    res.incidents.retain(|_, i| {
        i.state != IncidentState::Resolved
            || i.resolved_step
                .is_none_or(|s| step.saturating_sub(s) < dispatch::INCIDENT_KEEP_STEPS)
    });
}

/// **Decide what this level's sirens do this step** — the list the two hosts'
/// fenced audio blocks drain.
///
/// Three cues and no policy left over: a unit that has just started running hot
/// gets a `Start`, one that already was gets a `Move` every
/// `SIREN_POSITION_PERIOD` steps, and one that has stopped gets a `Stop`. The
/// walk is in `Guid` order — the `BTreeMap`'s — so a bounded audio ring evicts
/// the same commands on both hosts.
///
/// `O(units)`, and it allocates only on a level that actually has a fleet.
fn sound_and_light(world: &mut EcsWorld, res: &mut DispatchRes, step: u64, dt: f64) {
    res.sirens.clear();
    res.flashes.clear();
    let clock_s = step as f64 * dt;
    let mut now: std::collections::BTreeSet<Uuid> = Default::default();
    let mut hot_bars: std::collections::BTreeSet<Uuid> = Default::default();
    let chassis_list: Vec<Uuid> = res.runs.keys().copied().collect();
    for chassis in chassis_list {
        if !res.runs[&chassis].state.running_hot() {
            continue;
        }
        now.insert(chassis);
        let source = dispatch::siren_guid(chassis);
        let Some(at) = chassis_at_no_bridge(world, chassis) else {
            continue;
        };
        if res.siren_on.contains(&chassis) {
            // **The move is on a CADENCE, not on a change.** See
            // `SIREN_POSITION_PERIOD` for the ring arithmetic that decides it.
            if step.is_multiple_of(dispatch::SIREN_POSITION_PERIOD) {
                res.sirens.push(dispatch::SirenCue::Move { source, at });
            }
        } else {
            res.sirens.push(dispatch::SirenCue::Start { source, at });
        }
        // ── the bar. The PIN is taken once, on the step the unit goes hot, off
        //    the value the level authored — after that the live intensity is the
        //    flash's own output and reading it back would compound.
        let Some(bar) = dispatch::light_bar_of(world, chassis) else {
            continue;
        };
        hot_bars.insert(bar);
        let base = match res.bars.get(&bar) {
            Some(v) => *v,
            None => {
                let authored = authored_intensity(world, bar);
                res.bars.insert(bar, authored);
                authored
            }
        };
        res.flashes.push(dispatch::BarFlash {
            bar,
            base_intensity: base,
            clock_s,
        });
    }
    for chassis in res.siren_on.iter() {
        if !now.contains(chassis) {
            res.sirens.push(dispatch::SirenCue::Stop {
                source: dispatch::siren_guid(*chassis),
            });
        }
    }
    res.siren_on = now;
    // ── THE RELEASE. A pin with no release is a leak with a deadline (P21.4),
    //    and here the leak is visible: a bar left at the bottom of its own pulse
    //    is a unit that came home with a black light on its roof.
    let stale: Vec<(Uuid, f32)> = res
        .bars
        .iter()
        .filter(|(bar, _)| !hot_bars.contains(bar))
        .map(|(bar, base)| (*bar, *base))
        .collect();
    for (bar, base) in stale {
        dispatch::set_bar_intensity(world, bar, base);
        res.bars.remove(&bar);
    }
}

/// The emissive intensity a light bar's material was authored with.
///
/// `0.0` for a bar with no material at all, which flashes nothing — the same
/// refusal `set_bar_intensity` makes, spelled once on each side of the pin.
fn authored_intensity(world: &EcsWorld, bar: Uuid) -> f32 {
    world
        .entity_of(bar)
        .and_then(|e| world.world().get::<inf_ecs::components::Material>(e))
        .map(|m| m.emissive_intensity)
        .unwrap_or(0.0)
}

/// Where a chassis is, off the ECS alone.
///
/// The bridge is not consulted here and does not need to be: the write-back has
/// already put the solver's answer onto the `Transform` by the time anything
/// reads a siren's position, and a siren that needed a physics world would be a
/// decision that could not be taken without one.
fn chassis_at_no_bridge(world: &EcsWorld, chassis: Uuid) -> Option<DVec3> {
    let e = world.entity_of(chassis)?;
    let t = world.world().get::<Transform>(e)?;
    let p = t.translation.to_dvec3();
    p.is_finite().then_some(p)
}

/// **A burning building makes smoke** (wave EMS2) — sprite billboards, spawned
/// and reaped by the sim.
///
/// # There is no particle system, and this is what that means
///
/// The tree says so three times over and this wave did not add one. What it has
/// is `inf_ecs::components::Sprite`, a textured camera-facing quad the 2D
/// batcher already draws over the 3D scene, so a puff is **one entity** with a
/// `Cylindrical` billboard, a size that grows with age and an alpha that falls.
/// A column of them is a fire.
///
/// # They never reach the author's document
///
/// This is P21's own law (*"in the editor the render store IS the save's
/// staging source"*) applied one system over. Two things keep it:
/// [`inf_ecs::dispatch::DispatchRes::puffs`] is the list of what was spawned,
/// and `clear_dispatch` — which the editor calls at **both** ends of a Simulate
/// session — **despawns** them rather than forgetting them. A puff that outlived
/// its session would be a row in the author's Outliner that no Outliner row put
/// there, which is the VEN1b speaker's rule verbatim.
///
/// The identity is content-addressed on `(incident, step)`, so both hosts spawn
/// the same entity and neither needs a counter.
fn smoke(world: &mut EcsWorld, res: &mut DispatchRes, step: u64, dt: f64) {
    // ── reap first, so a level at the ceiling can still make new smoke.
    let dead: Vec<Uuid> = res
        .puffs
        .iter()
        .filter(|(_, born)| (step.saturating_sub(**born) as f64) * dt >= dispatch::PUFF_LIFETIME_S)
        .map(|(g, _)| *g)
        .collect();
    for guid in dead {
        if let Some(e) = world.entity_of(guid) {
            world.despawn(e);
        }
        res.puffs.remove(&guid);
    }
    // ── age what is left: rise, grow, fade. A pure function of `(born, step)`,
    //    so a puff is where it is because of when it was let go of and for no
    //    other reason.
    let live: Vec<(Uuid, u64)> = res.puffs.iter().map(|(g, b)| (*g, *b)).collect();
    for (guid, born) in live {
        let age_s = (step.saturating_sub(born) as f64) * dt;
        let u = (age_s / dispatch::PUFF_LIFETIME_S).clamp(0.0, 1.0);
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
            t.translation.y += dispatch::PUFF_RISE_MPS * dt;
        }
        if let Some(mut sp) = world.world_mut().get_mut::<inf_ecs::components::Sprite>(e) {
            let grow = dispatch::PUFF_SIZE_M * (1.0 + u);
            sp.size = inf_ecs::math::Vec2d::new(grow, grow);
            sp.color.a = (0.55 * (1.0 - u)) as f32;
        }
    }
    // ── and let one go, on the period, from every fire that is still burning.
    if !step.is_multiple_of(dispatch::PUFF_PERIOD) {
        return;
    }
    let fires: Vec<(Uuid, DVec3)> = res
        .incidents
        .iter()
        .filter(|(_, i)| {
            matches!(i.kind, IncidentKind::Fire { .. }) && i.state != IncidentState::Resolved
        })
        .map(|(g, i)| (*g, i.at))
        .collect();
    for (incident, at) in fires {
        if res.puffs.len() >= dispatch::MAX_PUFFS {
            break;
        }
        let guid = dispatch::puff_guid(incident, step);
        if world.entity_of(guid).is_some() {
            continue;
        }
        // A little spread, drawn off the puff's own guid so the column is not a
        // line of quads at one x.
        let jx = inf_ecs::crowd::agent_unit(guid, 0, dispatch::SALT_PUFF) - 0.5;
        let jz = inf_ecs::crowd::agent_unit(guid, 1, dispatch::SALT_PUFF) - 0.5;
        let e = world.spawn_with_guid(guid, "Smoke", None);
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::new(at.x + jx * 3.0, at.y + 2.0, at.z + jz * 3.0),
                ..Transform::IDENTITY
            },
            inf_ecs::components::Sprite {
                size: inf_ecs::math::Vec2d::new(dispatch::PUFF_SIZE_M, dispatch::PUFF_SIZE_M),
                color: inf_ecs::math::Color::new(0.22, 0.21, 0.20, 0.55),
                billboard: inf_ecs::components::BillboardMode::Cylindrical,
                ..Default::default()
            },
        ));
        res.puffs.insert(guid, step);
    }
}

/// **Report an incident by hand** — the staging door.
///
/// The three feeds above are what a *level* produces on its own: a witnessed
/// act, a body on the ground, an ambient draw. This is the fourth way one can
/// exist, and it is deliberately narrow: a gate that wants a fire at a named
/// building at a named moment, a designer's script, a later wave's mission.
///
/// It goes through the **same** `open`: the same ceiling, the same
/// content-addressed guid, the same `Reported` state and the same counters — so
/// a staged incident is answered by exactly the dispatcher a real one is, and a
/// gate that staged one is not testing a second code path. Returns the incident's
/// guid, or `None` if the level has no dispatcher yet (no fleet) or the table is
/// full.
///
/// **It does not create the dispatcher.** A level with no emergency vehicle in
/// it has no `DispatchRes` by construction (the "absent costs nothing" rule), and
/// a staging door that manufactured one would make every trace committed before
/// this wave depend on whether anybody had ever called it.
pub fn report_incident(world: &mut EcsWorld, kind: IncidentKind, at: DVec3) -> Option<Uuid> {
    let mut res = world.world_mut().remove_resource::<DispatchRes>()?;
    let step = res.steps;
    let mut stats = DispatchStats::default();
    open(&mut res, kind, at, step, &mut stats);
    let guid = (stats.opened > 0).then(|| dispatch::incident_guid(kind, at, step));
    world.world_mut().insert_resource(res);
    guid
}

// ── the read side ───────────────────────────────────────────────────────────

/// **Every extinguish line a fire crew is working right now** — `(from, to)` in
/// world metres, in `Guid` order.
///
/// A **debug line**, and that is the muzzle flash's own sentence one wave along:
/// *"there is no particle system, so a muzzle flash is the first twenty
/// centimetres of the same line drawn brighter"*. A hose here is a segment from
/// the crew member's shoulder to the fire, which is the substrate this engine
/// has. It is drawn by the host's render side, out of this list, so there is no
/// path from a beam back into the sim and a frame that drew none and a frame
/// that drew three are the same simulation.
///
/// Empty and allocation-free on a level with no dispatcher, and on every level
/// where nothing is on fire.
pub fn extinguish_beams(world: &EcsWorld) -> Vec<(DVec3, DVec3)> {
    let Some(res) = dispatch::dispatch_of(world) else {
        return Vec::new();
    };
    let Some(fleet) = world.world().get_resource::<FleetRes>() else {
        return Vec::new();
    };
    let mut out: Vec<(DVec3, DVec3)> = Vec::new();
    for (chassis, run) in &res.runs {
        if run.state != UnitState::OnScene {
            continue;
        }
        if fleet.units.get(chassis).map(|u| u.kind) != Some(UnitKind::Fire) {
            continue;
        }
        let Some(incident) = run.incident.and_then(|g| res.incidents.get(&g)) else {
            continue;
        };
        if !matches!(incident.kind, IncidentKind::Fire { .. }) {
            continue;
        }
        let crew = dispatch::crew_guid(*chassis);
        let Some(from) = body_at(world, crew) else {
            continue;
        };
        // From the shoulder rather than from the feet: a line that starts at
        // ground level reads as a crack in the road.
        out.push((from + DVec3::Y * 1.4, incident.at));
    }
    out
}

/// **Every unit running with its lights and siren on, and where it is** — the
/// query the audio emit, the projector and the yield rule all make.
///
/// One door for three readers, because three spellings of *"is this thing
/// responding"* is the defect this repository has paid for at four seams. In
/// `Guid` order, so a bounded audio ring evicts the same commands on both hosts.
///
/// Empty and allocation-free on a level with no dispatcher.
pub fn running_hot(world: &EcsWorld) -> Vec<(Uuid, DVec3)> {
    let Some(res) = dispatch::dispatch_of(world) else {
        return Vec::new();
    };
    let mut out: Vec<(Uuid, DVec3)> = Vec::new();
    for (chassis, run) in &res.runs {
        if !run.state.running_hot() {
            continue;
        }
        let Some(e) = world.entity_of(*chassis) else {
            continue;
        };
        let Some(t) = world.world().get::<Transform>(e) else {
            continue;
        };
        let p = t.translation.to_dvec3();
        if p.is_finite() {
            out.push((*chassis, p));
        }
    }
    out
}
