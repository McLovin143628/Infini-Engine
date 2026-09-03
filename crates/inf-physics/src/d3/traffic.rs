//! **The traffic fixed-step door** (wave VEH2b): tier every car, build or take
//! down its body, and hand the ones near the hero a stick.
//!
//! The `inf_physics` half of [`inf_ecs::traffic`], and the third instance of
//! this crate's own split — `inf_ecs::vehicle` decides and
//! [`super::vehicle`] applies; `inf_ecs::movement` decides and
//! [`super::movement`] applies. Everything here touches rapier or the ECS;
//! nothing here decides anything. The map, the controller and the tier ladder
//! are on the other side of that wall and are unit-tested without a world.
//!
//! # THE WATCHED-CAR SENTENCE, and it is structural
//!
//! A car outside [`inf_ecs::traffic::TRAFFIC_FULL_M`] **is not simulated**. It
//! has no `VehicleClass`, no wheel sensors and therefore no
//! `inf_ecs::vehicle::VehicleRig` — `rig_of` answers `None`, the bridge never
//! builds a `RaycastVehicle` for it, and `step_vehicles` cannot reach it even
//! by accident. Its transform is `place(clock)`: a pure function of its
//! schedule and the hour, exactly as a `Near` crowd agent's is.
//!
//! So the honest sentence is: **the cars you watch from a distance are not
//! driving, they are being drawn where the clock says a drive would have got
//! to.** They do not collide with each other, they do not queue, and two of
//! them on one lane will pass through one another. Inside 64 m every one of
//! them is a real rig on four rays with a driver in it, and the hand-off is the
//! crowd's own: on the way down the clock is moved onto the body
//! ([`TrafficRecord::rephase_delta_on`]) so the transition moves nothing.
//!
//! # Visibility never filters this
//!
//! The band is [`inf_ecs::crowd::CrowdBand`] over `StreamingSource` entities —
//! **sim state**, not a camera. A car behind the hero is the same tier as a car
//! in front of it. That is the P20 law this tree keeps re-proving, and it is
//! why the tier ladder is keyed on the same anchors the collider band and the
//! crowd are.
//!
//! [`TrafficRecord::rephase_delta_on`]: inf_ecs::traffic::TrafficRecord::rephase_delta_on

use std::collections::BTreeMap;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, MovementMode, RigidBody3D, SeatState, Transform,
};
use inf_ecs::crowd::{CrowdBand, CrowdClock, CrowdTier};
use inf_ecs::math::Vec3d;
use inf_ecs::traffic::{self, DriveView, RigDetail, TrafficRecord, TrafficStats, TRAFFIC_RADII};
use inf_ecs::vehicle::{RigSpawn, VehicleControls};
use inf_ecs::EcsWorld;

use super::PhysicsBridge3D;

/// How far to either side of a lane a body counts as **in the way**, metres.
///
/// Half a lane plus a body's own half width: a car whose centre is inside 2.5 m
/// of the lane is in it, and one 5 m away is parked at the kerb. It is what
/// makes the following rule see the queue in front of it and not the row it is
/// driving past.
pub const CORRIDOR_HALF_M: f64 = 2.5;

/// How far ahead the following rule looks, metres.
///
/// Sixty metres is four seconds at a 50 km/h limit and two at a highway's, so a
/// car has begun braking long before the stopping-distance rule needs it to.
/// Past it the road is treated as clear, which is the honest answer for a
/// controller whose whole job is the car in front.
pub const LOOK_AHEAD_M: f64 = 60.0;

/// **Advance every traffic car one fixed step.**
///
/// Called by each host in its own `traffic` phase, immediately after the crowd
/// and before the physics sync — so a car that materializes this step gets its
/// bodies mirrored on the same step, which is [`inf_ecs::crowd::step_crowd`]'s
/// own placement argument.
///
/// The sequence, all of it a pure function of sim state:
///
/// 1. derive the carriageway if the level's blocks moved, and plan a batch of
///    commuter routes ([`inf_ecs::traffic::sync_traffic`]);
/// 2. read the band off the world's `StreamingSource` entities;
/// 3. walk the records in `Guid` order; for each, the tier of where it *is*;
/// 4. build or take down the rig its tier calls for;
/// 5. for a `Near` car, write `place(clock)` onto the transform; for a `Full`
///    one, write its driver's **intent** and let the movement step turn that
///    into controls.
pub fn step_traffic(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, dt: f64) -> TrafficStats {
    traffic::sync_carriageway(world);
    let planned = traffic::sync_traffic(world);
    if traffic::traffic_of(world).is_none() {
        return TrafficStats::default();
    }
    let band = CrowdBand::from_world(world, TRAFFIC_RADII);
    // **A level with no streaming source has no traffic**, which is the exact
    // opposite of the crowd's rule and has its own reason.
    // `CrowdBand::unbounded` fails toward `Full` because dropping an NPC's tier
    // is the dangerous direction and keeping it is "merely slow". For a car it
    // is not merely slow: every one of up to `MAX_TRAFFIC_CARS` records would
    // become a fourteen-entity rig with a `RaycastVehicle`, a settle ray and an
    // NPC driver, in ONE step. A world with nothing to be near has nothing for
    // traffic to be near, so the honest answer is none of it.
    if band.is_unbounded() {
        let mut pop = match world
            .world_mut()
            .remove_resource::<traffic::TrafficPopulationRes>()
        {
            Some(p) => p,
            None => return TrafficStats::default(),
        };
        let gone: Vec<(Uuid, inf_ecs::vehicle::VehicleDef)> = pop
            .records
            .iter()
            .filter(|(_, r)| r.detail != RigDetail::None && !r.taken)
            .map(|(g, r)| (*g, r.def))
            .collect();
        for (g, def) in &gone {
            inf_ecs::vehicle::despawn_rig(world, *g, def);
            despawn_driver(world, bridge, *g);
        }
        for (_, rec) in pop.records.iter_mut().filter(|(_, r)| !r.taken) {
            rec.detail = RigDetail::None;
            rec.tier = CrowdTier::Dormant;
        }
        let cars = pop.records.len();
        pop.steps += 1;
        world.world_mut().insert_resource(pop);
        return TrafficStats {
            cars,
            removed: gone.len(),
            per_tier: [0, 0, 0, cars],
            band_stamp: 0,
            ..TrafficStats::default()
        };
    }
    let mut pop = match world
        .world_mut()
        .remove_resource::<traffic::TrafficPopulationRes>()
    {
        Some(p) => p,
        None => return TrafficStats::default(),
    };
    let clock = CrowdClock::from_world(world, pop.steps as f64 * dt);
    let mut stats = TrafficStats {
        cars: pop.records.len(),
        planned_now: planned,
        pending: pop.pending.len(),
        band_stamp: band.stamp(),
        ..TrafficStats::default()
    };
    let archetype = inf_ecs::society::level_archetype(world);

    // ── who is in which car, ONCE. The seat check below runs for every record
    //    the traffic still owns, and `occupant_of` is `O(characters)` each
    //    time — which on a settlement with three hundred and twenty-nine
    //    residents is `O(cars x characters)` sixty times a second.
    let occupants = super::carjack::occupants(world);
    // ── the obstacles, LAZILY. A `Full` car's following rule reads every solid
    //    body in the world, and gathering that per car would be
    //    `O(cars x world)`. Gathering it unconditionally would be `O(world)` on
    //    a street where every car is parked, which is most streets at most
    //    hours — so it is built the first time something actually steers and
    //    not at all otherwise.
    let mut obstacles: Option<Vec<(Uuid, DVec3)>> = None;
    // ── and the sirens, on the same terms and for the same reason (EMS2). The
    //    yield rule is `O(hot)` per steered car; gathering the list per car
    //    would be `O(cars x units)`, and gathering it unconditionally would walk
    //    the dispatcher's runs on every street in the world that has no
    //    emergency vehicle within a mile of it.
    let mut hot: Option<Vec<(Uuid, DVec3)>> = None;
    // ── and the colliders a settle ray must look THROUGH, on the same terms:
    //    built the first time a car actually asks what it is standing on, and
    //    not at all on a settled street where every car already knows.
    let mut look_through: Option<std::collections::BTreeSet<super::ColliderId3D>> = None;

    for (guid, rec) in pop.records.iter_mut() {
        let guid = *guid;
        stats.commuters += usize::from(rec.commutes());
        let entity = world.entity_of(guid);
        let here = match entity.and_then(|e| world.world().get::<Transform>(e)) {
            Some(t) => t.translation.to_dvec3(),
            None => rec.last,
        };
        // ── has somebody taken this car? Two ways in, one flag, one rule:
        //    a car the player has touched is no longer traffic's. The seat
        //    check is here rather than in the carjack door because a hero can
        //    also just get into an EMPTY traffic car, which no carjack ever
        //    hears about.
        if !rec.taken
            && occupants
                .get(&guid)
                .is_some_and(|who| *who != traffic::driver_guid(guid))
        {
            rec.taken = true;
        }
        if rec.taken {
            stats.taken += 1;
            if let Some(e) = world.entity_of(guid) {
                if let Some(t) = world.world().get::<Transform>(e) {
                    rec.last = t.translation.to_dvec3();
                    rec.yaw_deg = t.rotation.y;
                }
            }
            stats.per_tier[rec.tier.as_u8() as usize] += 1;
            continue;
        }
        let leg = rec.leg_at(guid, clock);
        if let Some((i, _)) = leg {
            let i = i as u8;
            if i != rec.leg {
                rec.leg = i;
                rec.rephase_m = 0.0;
            }
        }
        let driving = rec.is_driving(clock, leg);
        stats.driving += usize::from(driving);
        let (mut at, mut yaw) = rec.place(guid, clock, leg);
        // **A circuit car outside its hours is not there.** Not parked back at
        // its space — that would be a teleport somebody watched. The tier is
        // forced rather than banded, which is the one place in this step that
        // is not `band.tier`, and it is a fact about the LEVEL CLOCK rather than
        // about a camera, so the "visibility never filters sim" law is
        // untouched.
        let alive = rec.alive(clock.hour);
        let tier = if alive {
            band.tier(here)
        } else {
            CrowdTier::Dormant
        };
        let was = rec.tier;
        if tier != was {
            stats.retiered += 1;
        }
        rec.tier = tier;
        stats.per_tier[tier.as_u8() as usize] += 1;

        // ── the hand-off DOWN. A car leaving the steered tier hands the clock
        //    back the metre its BODY reached, not the metre the clock ran on
        //    to — `CrowdRecord`'s own argument, and without it a car that spent
        //    ten seconds behind a queue teleports the moment it drops to `Near`.
        if was == CrowdTier::Full && tier != CrowdTier::Full && driving {
            if let Some(path) = rec.active_path(clock, leg) {
                let s_body = path.project(here).s_m;
                let delta = rec.rephase_delta(guid, clock, leg, s_body);
                if delta != 0.0 {
                    rec.rephase_m += delta;
                    stats.rephased += 1;
                    let (a2, y2) = rec.place(guid, clock, leg);
                    at = a2;
                    yaw = y2;
                }
            }
        }

        // ── the ground, measured once, and **NO BODY UNTIL IT IS KNOWN**.
        //
        //    Measuring first is not enough: on the step a settlement's cells
        //    activate there is no terrain collider under the slot yet, the ray
        //    finds nothing, and a car built anyway sits at the street's own
        //    derived height. That height is a MEAN over the blocks that bound
        //    the line, and a block whose volume offers no exterior ground-floor
        //    doorway falls back to its entity's `y` — which the island authors
        //    as **zero**. Mixing zeros with a settlement pad at 130 m gives 86,
        //    and the island fixture's cars were built forty-five metres under
        //    the island and fell for the whole window: measured at **-631 m**
        //    after ten seconds, which is free fall to the digit.
        //
        //    So a car whose ground nothing can answer for stays `Dormant` and is
        //    asked again next step. A car that never gets an answer is never
        //    built, which is the right outcome for a slot over a hole.
        let mut want = RigDetail::of(tier);
        if want != RigDetail::None && rec.ground_y.is_none() {
            // …and it asks about the GROUND, which is why it is handed a set to
            // look through. See `not_the_ground`. Built at most once a step, and
            // only on a step something actually settles.
            if look_through.is_none() {
                look_through = Some(not_the_ground(world, bridge));
            }
            let through = look_through.as_ref().expect("just built");
            match settle_on_the_ground(bridge, at, through) {
                Some(g) => {
                    rec.ground_y = Some(g);
                    let (a2, y2) = rec.place(guid, clock, leg);
                    at = a2;
                    yaw = y2;
                    stats.settled += 1;
                }
                None => {
                    want = RigDetail::None;
                    stats.groundless += 1;
                }
            }
        }
        // ── the body its tier calls for.
        if want != rec.detail {
            inf_ecs::vehicle::despawn_rig(world, guid, &rec.def);
            despawn_driver(world, bridge, guid);
            if want != RigDetail::None {
                inf_ecs::vehicle::spawn_rig_at(
                    world,
                    guid,
                    &rec.def,
                    &RigSpawn {
                        name: "Traffic Car".to_string(),
                        at,
                        yaw_deg: yaw,
                        paint: rec.paint,
                        clip: None,
                        // **Silent until the emitter can follow the car** — see
                        // `RigSpawn::engine_voice` for VEH2a's carried item 5,
                        // which is the whole reason.
                        engine_voice: false,
                        // **Traffic is CIVILIAN** (wave EMS1). The emergency
                        // fleet is authored content on Path A and never reaches
                        // this door; a livery here would be a police cruiser at
                        // a kerb slot, which is the trap the fleet borrows
                        // civilian bodies to avoid.
                        livery: None,
                    },
                    want == RigDetail::Full,
                );
                stats.built += 1;
            } else {
                stats.removed += 1;
            }
            rec.detail = want;
        }
        if want == RigDetail::None {
            rec.last = at;
            rec.yaw_deg = yaw;
            continue;
        }

        match tier {
            // ── the steered tier. A driving car gets a driver and a stick; a
            //    parked one gets the handbrake, because a rig with no controls
            //    on a graded street rolls away.
            CrowdTier::Full => {
                rec.last = here;
                if let Some(e) = world.entity_of(guid) {
                    if let Some(t) = world.world().get::<Transform>(e) {
                        rec.yaw_deg = t.rotation.y;
                    }
                }
                if driving {
                    let driver = ensure_driver(world, bridge, guid, &archetype, at);
                    stats.drivers += usize::from(driver);
                    if obstacles.is_none() {
                        obstacles = Some(obstacles_of(world));
                    }
                    if hot.is_none() {
                        hot = Some(super::dispatch::running_hot(world));
                    }
                    let around = Around {
                        obstacles: obstacles.as_deref().unwrap_or(&[]),
                        hot: hot.as_deref().unwrap_or(&[]),
                    };
                    steer_car(world, bridge, guid, rec, clock, leg, around);
                } else {
                    despawn_driver(world, bridge, guid);
                    if let Some(v) = bridge.vehicle_mut(guid) {
                        v.control(VehicleControls {
                            handbrake: true,
                            ..VehicleControls::default()
                        });
                    }
                }
            }
            // ── the clock's tiers. `place(clock)` onto the transform, and
            //    nothing else: no rig, no rays, no controls.
            _ => {
                rec.last = at;
                rec.yaw_deg = yaw;
                if let Some(e) = world.entity_of(guid) {
                    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
                        t.translation = Vec3d::new(at.x, at.y, at.z);
                        t.rotation = Vec3d::new(0.0, yaw, 0.0);
                    }
                }
            }
        }
    }

    pop.steps += 1;
    let moved = stats.cars > 0;
    world.world_mut().insert_resource(pop);
    // Only when there is something to have moved: a level with no streets must
    // not bump the version every step and make both projectors rebuild a scene
    // that did not change.
    if moved {
        world.mark_dirty();
    }
    stats
}

/// Everything a traffic car has to not drive into: every solid body in the
/// world with a position, in `Guid` order.
///
/// Vehicles **and characters**, and the second half is deliberate: a town whose
/// traffic drove through its pedestrians would kill a resident a minute, and
/// the cheapest correct answer is that traffic yields to anything standing in
/// its lane. The visible consequence is stated rather than hidden — **a hero
/// standing in the road stops the street**, which is a thing a player will do
/// and a thing this rule answers honestly.
pub(crate) fn obstacles_of(world: &EcsWorld) -> Vec<(Uuid, DVec3)> {
    let mut out: Vec<(Uuid, DVec3)> = Vec::new();
    for e in world.world().iter_entities() {
        let Some(g) = e.get::<inf_ecs::components::Guid>() else {
            continue;
        };
        let is_body = e.get::<CharacterMovement>().is_some()
            || (e.get::<RigidBody3D>().is_some() && e.get::<Collider3D>().is_some());
        if !is_body {
            continue;
        }
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

/// **The gap in front**, metres along the car's own path — `None` for a clear
/// road.
///
/// Every obstacle is projected onto the path; one that is inside
/// [`CORRIDOR_HALF_M`] of it and ahead of the car by less than
/// [`LOOK_AHEAD_M`] is in the way, and the nearest such is the gap. Parked cars
/// are five metres off the lane and never qualify, which is what
/// [`inf_ecs::traffic::KERB_PARK_OFFSET_M`] is sized for.
pub(crate) fn gap_ahead(
    path: &traffic::LanePath,
    s_m: f64,
    self_guid: Uuid,
    driver: Uuid,
    obstacles: &[(Uuid, DVec3)],
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for (g, p) in obstacles {
        if *g == self_guid || *g == driver {
            continue;
        }
        let proj = path.project(*p);
        if proj.distance_m > CORRIDOR_HALF_M {
            continue;
        }
        let ahead = proj.s_m - s_m;
        if ahead <= 0.0 || ahead > LOOK_AHEAD_M {
            continue;
        }
        if best.is_none_or(|b| ahead < b) {
            best = Some(ahead);
        }
    }
    best
}

/// **The two lists a driver's view is built against**, gathered once a step by
/// the caller and handed down.
///
/// One struct rather than two parameters because both are the same kind of
/// thing — *what else is near this car* — and because a function that took the
/// world, the bridge, the record, the clock, the leg and both of them has more
/// arguments than a reader can hold (and more than `clippy` allows).
#[derive(Clone, Copy)]
pub(crate) struct Around<'a> {
    /// Every solid body with a position, in `Guid` order.
    pub obstacles: &'a [(Uuid, DVec3)],
    /// Every unit running with its lights and siren on.
    pub hot: &'a [(Uuid, DVec3)],
}

/// **What one car's driver can see** — the `DriveView` both the steering and
/// the instrument build, so an arm cannot measure a controller the engine does
/// not run.
fn view_of<'a>(
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    rec: &'a TrafficRecord,
    clock: CrowdClock,
    leg: inf_ecs::crowd::ActiveLeg,
    around: Around<'_>,
) -> Option<DriveView<'a>> {
    let path = rec.active_path(clock, leg)?;
    let body = bridge.body_of(chassis)?;
    let w = bridge.world();
    let at = w.body_translation(body)?;
    let rot = w.body_rotation(body)?;
    let linvel = w.body_linvel(body).unwrap_or(DVec3::ZERO);
    let forward = rot * DVec3::Z;
    let s_m = path.project(at).s_m;
    let driver = traffic::driver_guid(chassis);
    // ── EMS2 the yield. The rule is `inf_ecs::dispatch`'s; what is here is the
    //    list, gathered once a step by the caller and handed down — the
    //    `obstacles` shape one system along.
    let yield_bias = inf_ecs::dispatch::yield_bias_m(at, forward, around.hot);
    Some(DriveView {
        at,
        forward,
        forward_mps: linvel.dot(forward),
        path,
        s_m,
        speed_limit_mps: traffic::street_speed_mps(),
        gap_m: gap_ahead(path, s_m, chassis, driver, around.obstacles),
        lateral_bias_m: yield_bias,
        loops: rec.circuit.is_some(),
    })
}

/// **What one traffic car's driver is asking for right now** — the same view
/// `steer_car` builds, for an instrument that wants the decision rather than
/// its consequence.
///
/// One door for both: an arm that rebuilt the view itself would be measuring a
/// controller the engine does not run, which is this repository's own "a gate
/// must aim at the thing it names".
pub fn probe_intent(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    dt: f64,
) -> Option<inf_ecs::traffic::DriveIntent> {
    let pop = traffic::traffic_of(world)?;
    let rec = pop.records.get(&chassis)?;
    let clock = CrowdClock::from_world(world, pop.steps as f64 * dt);
    let leg = rec.leg_at(chassis, clock);
    let obstacles = obstacles_of(world);
    let hot = super::dispatch::running_hot(world);
    let around = Around {
        obstacles: &obstacles,
        hot: &hot,
    };
    let view = view_of(bridge, chassis, rec, clock, leg, around)?;
    Some(traffic::drive_intent(&view))
}

/// Write a driving car's stick — the whole of "an AI drives".
///
/// The intent goes onto the **driver's** `CharacterMovement`, not onto the car:
/// `step_driving` six phases later reads it, hands it to
/// `VehicleControls::from_intent` and calls `Vehicle::control`, which is
/// exactly what happens when a player holds the same stick. There is no second
/// path into a vehicle's controls in this engine and this wave did not add one.
fn steer_car(
    world: &mut EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    rec: &TrafficRecord,
    clock: CrowdClock,
    leg: inf_ecs::crowd::ActiveLeg,
    around: Around<'_>,
) {
    let driver = traffic::driver_guid(chassis);
    let Some(view) = view_of(bridge, chassis, rec, clock, leg, around) else {
        return;
    };
    let intent = traffic::drive_intent(&view);
    let Some(e) = world.entity_of(driver) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.intent_move = intent.move_input;
        cm.runtime.want_handbrake = intent.handbrake;
    }
}

/// Put a person in the seat if there is not one there already.
///
/// Returns whether the car has a driver afterwards. The body is built through
/// [`inf_ecs::crowd::spawn_body`] — the level's own archetype, the same capsule
/// and mesh every resident wears — and dropped straight into `Driving` with the
/// warp already finished, because a traffic car's driver did not climb in while
/// anybody was watching.
fn ensure_driver(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    chassis: Uuid,
    archetype: &inf_ecs::crowd::CrowdArchetype,
    at: DVec3,
) -> bool {
    let driver = traffic::driver_guid(chassis);
    if let Some(e) = world.entity_of(driver) {
        // Already there — unless something pulled it out, in which case the
        // seat is empty and stays empty. THE CARJACK'S OWN CLAUSE: a car whose
        // driver has been ejected does not grow another one.
        return world
            .world()
            .get::<CharacterMovement>(e)
            .is_some_and(|cm| cm.runtime.seat.vehicle == chassis);
    }
    let e = inf_ecs::crowd::spawn_body(world, driver, archetype, at);
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Driving;
        cm.runtime.seat = SeatState {
            vehicle: chassis,
            // Not `entering`: the warp is a choreography for a character the
            // player watched walk up to a car. A traffic driver was always in
            // it, and starting one would slide a body across the street.
            entering: false,
            time_s: 0.0,
            start: Vec3d::from_dvec3(at),
            start_yaw_deg: 0.0,
        };
    }
    super::vehicle::park_collider(bridge, driver, true);
    true
}

/// Take the driver out with the car, when the car itself goes.
fn despawn_driver(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, chassis: Uuid) {
    let driver = traffic::driver_guid(chassis);
    let Some(e) = world.entity_of(driver) else {
        return;
    };
    // A driver that is no longer IN this car has been pulled out of it and is
    // now an ordinary NPC standing in the street — it is not this car's to
    // despawn. That is the carjack's own clause, enforced here rather than
    // remembered by the caller.
    let in_this_car = world
        .world()
        .get::<CharacterMovement>(e)
        .is_some_and(|cm| cm.runtime.seat.vehicle == chassis);
    if !in_this_car {
        return;
    }
    super::vehicle::park_collider(bridge, driver, false);
    world.despawn(e);
}

/// **`audit:` VEH2b — the colliders a settle ray must look THROUGH**, because a
/// person is not the ground.
///
/// [`settle_on_the_ground`] casts `CastTargets::AllSolid`, which is *every*
/// solid body and not only the level's geometry — and a crowd agent at
/// [`CrowdTier::Full`] carries a **kinematic capsule**
/// (`inf_ecs::crowd::set_tier_components`). A kerb slot is 5 m from the street's
/// centreline and the society links two blocks' pavements straight across the
/// gap, so a resident crossing the road stands on a parking space several times
/// a day; the hero can stand on one deliberately.
///
/// One ray that landed on a head is **permanent**, which is what makes this
/// worth a set rather than a note: `ground_y` is latched on purpose (see its own
/// doc), a `Near` car is kinematic and [`TrafficRecord::place`] writes its
/// transform every step from that number — so the car is parked 1.8 m in the air
/// for the rest of the session. A `Full` car falls the 1.8 m and looks fine
/// until it drops a rung.
///
/// Excluded in the **broad phase** rather than rejected after the cast, which is
/// the P22.3 M4 law one system over: a downstream check would turn "a person is
/// not the ground" into "a person HIDES the ground", and the road underneath —
/// which is what the ray was asking about — would never be seen at all.
///
/// `O(characters)`, built at most once per fixed step and only on a step a car
/// actually settles. **What it deliberately does not exclude**: another vehicle.
/// Kerb slots are [`inf_ecs::traffic::KERB_SLOT_M`] apart so two derived cars
/// cannot share one, and a car the player parked across a space is a case with
/// no arm behind it — named here rather than guessed at.
///
/// [`CrowdTier::Full`]: inf_ecs::crowd::CrowdTier::Full
/// [`TrafficRecord::place`]: inf_ecs::traffic::TrafficRecord::place
fn not_the_ground(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
) -> std::collections::BTreeSet<super::ColliderId3D> {
    inf_ecs::movement::movement_targets(world)
        .into_iter()
        .filter_map(|g| bridge.collider_of(g))
        .collect()
}

/// **What is under this car**, world Y — a ray straight down through
/// everything solid that is not a person.
///
/// `AllSolid` and from well above, so a slot whose derived height is under the
/// terrain still finds the terrain above it: the ray starts
/// [`SETTLE_UP_M`] over the derived point and reaches [`SETTLE_DOWN_M`] past
/// it, which brackets every error a per-street pad mean can make on a graded
/// settlement. `look_through` is `not_the_ground`'s set, and it is a parameter
/// rather than a default so the one call site names what it is asking about.
///
/// `None` when nothing is there — a slot over a hole, or a terrain tile that
/// has not paged in. **The caller does not fall back**: it leaves the car
/// `Dormant` and asks again next step, because the derived height is a median
/// over blocks and has been measured forty-five metres out. See the call site,
/// which is the only one.
pub fn settle_on_the_ground(
    bridge: &mut PhysicsBridge3D,
    at: DVec3,
    look_through: &std::collections::BTreeSet<super::ColliderId3D>,
) -> Option<f64> {
    let from = at + DVec3::Y * SETTLE_UP_M;
    bridge
        .world_mut()
        .cast_ray_where(
            from,
            -DVec3::Y,
            SETTLE_UP_M + SETTLE_DOWN_M,
            look_through,
            super::query::CastTargets::AllSolid,
        )
        .map(|hit| hit.point.y)
}

/// How far above its derived place the settle ray starts, metres.
///
/// Eighty, and the number is the measurement: the worst pad error this wave met
/// on the CI island was **forty-five metres** (a median of blocks whose `y`
/// fell back to an entity origin the island authors as zero, against a
/// settlement pad at 130). A forty-metre ray would have started *below* the
/// terrain on that case and never hit — a safe failure, but one that leaves a
/// street with no cars on it and no way to say why.
pub const SETTLE_UP_M: f64 = 80.0;
/// How far below it the ray reaches, metres.
pub const SETTLE_DOWN_M: f64 = 80.0;

/// **Every traffic car within reach of a point, nearest first** — the query a
/// gate makes and a HUD would.
pub fn cars_near(world: &EcsWorld, p: DVec3, radius_m: f64) -> Vec<(Uuid, f64)> {
    let Some(pop) = traffic::traffic_of(world) else {
        return Vec::new();
    };
    let mut out: Vec<(Uuid, f64)> = pop
        .records
        .iter()
        .filter(|(_, r)| r.detail != RigDetail::None)
        .filter_map(|(g, r)| {
            let d = r.last - p;
            let m = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            (m <= radius_m).then_some((*g, m))
        })
        .collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

/// The traffic records, for an instrument that wants the whole table.
pub fn records(world: &EcsWorld) -> BTreeMap<Uuid, TrafficRecord> {
    traffic::traffic_of(world)
        .map(|p| p.records.clone())
        .unwrap_or_default()
}

/// A bridge-side sanity door: whether a chassis is one the traffic owns.
pub fn is_traffic(world: &EcsWorld, chassis: Uuid) -> bool {
    traffic::traffic_of(world).is_some_and(|p| p.records.contains_key(&chassis))
}

/// Whether a traffic car's body is currently `Kinematic` — a `Near` car, moved
/// by its clock rather than by the solver.
pub fn is_kinematic(world: &EcsWorld, chassis: Uuid) -> bool {
    world
        .entity_of(chassis)
        .and_then(|e| world.world().get::<RigidBody3D>(e))
        .is_some_and(|b| b.kind == BodyKind3D::Kinematic)
}
