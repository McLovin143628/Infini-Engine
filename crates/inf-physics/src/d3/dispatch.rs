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
    if fleet.units.is_empty() {
        // **Absent costs nothing.** A level with no emergency vehicle in it
        // never gets a `DispatchRes` at all, so `dispatch_state_bytes` stays
        // empty and every trace committed before this wave is byte-identical.
        return DispatchStats::default();
    }
    let mut res = world
        .world_mut()
        .remove_resource::<DispatchRes>()
        .unwrap_or_default();
    let step = res.steps;
    // The runs follow the fleet: a unit the derivation dropped takes its run
    // with it, and a new one starts in station.
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

    stats.incidents = res.incidents.len();
    stats.running_hot = res.runs.values().filter(|r| r.state.running_hot()).count();
    res.steps += 1;
    world.world_mut().insert_resource(res);
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
    if step % dispatch::AMBIENT_PERIOD != 0 {
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

/// Send the nearest free unit of the right service to the oldest open incident.
fn assign(
    world: &EcsWorld,
    fleet: &FleetRes,
    res: &mut DispatchRes,
    step: u64,
    stats: &mut DispatchStats,
) {
    let pending: Vec<Uuid> = res
        .incidents
        .iter()
        .filter(|(_, i)| i.state == IncidentState::Reported)
        .map(|(g, _)| *g)
        .collect();
    if pending.is_empty() {
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
                if seated && steer(world, bridge, chassis, crew, &run) {
                    stats.steered += 1;
                }
                let reach = if run.state == UnitState::EnRoute {
                    dispatch::ON_SCENE_M
                } else {
                    dispatch::HOME_M
                };
                if (here - target).length() <= reach {
                    if run.state == UnitState::EnRoute {
                        arrive(world, bridge, res, chassis, crew, here, step);
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
        // Coming back from a scene: the same body goes back in the same seat.
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
        gap_m: None,
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
    crew: Uuid,
    here: DVec3,
    step: u64,
) {
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
    // The crew stands `SCENE_STAND_M` from the incident, on the line from the
    // vehicle to it — which is where somebody who has just got out of that
    // vehicle would be.
    let toward = at - here;
    let len = (toward.x * toward.x + toward.z * toward.z).sqrt();
    let dir = if len > 1.0e-6 {
        DVec3::new(toward.x / len, 0.0, toward.z / len)
    } else {
        DVec3::Z
    };
    let stand = at - dir * dispatch::SCENE_STAND_M;
    unseat_crew(world, bridge, crew, stand, dir);
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
        incident.state = IncidentState::Resolved;
        incident.resolved_step = Some(step);
        res.resolved = res.resolved.saturating_add(1);
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

// ── the read side ───────────────────────────────────────────────────────────

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
