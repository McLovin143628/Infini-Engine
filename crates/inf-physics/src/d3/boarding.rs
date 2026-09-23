//! **BOARDING, the half with a world in it** (wave VEH3d).
//!
//! [`inf_ecs::boarding`] is the arithmetic — the eight derived sockets, the
//! phase table, the Hermite, the reach. This module is where that arithmetic
//! meets a physics world: the live chassis pose, the door on its revolute
//! joint, the ground under a stance node, the colliders a body leaving a car
//! must not be put inside, and the hand and foot requests the pose step solves.
//!
//! # The machine, and where each phase runs
//!
//! ```text
//!   E ──► Locked ─► Unlocking ─► OpeningDoor ─► EnteringIK ─► Seated ─► Driving
//!          └─────── on the ground ──────────┘   └──────── in the seat ────────┘
//!   E ──► Exiting ─► ClosingDoor ─► Idle          (the reverse; > 2 m/s is a roll)
//! ```
//!
//! * The GROUND phases (`Locked`, `Unlocking`, `OpeningDoor`, `ClosingDoor`)
//!   are [`step_ground`], which `movement::step_one` hands the body to the way
//!   it hands a mantle to `step_mantle`: the choreography owns the body
//!   outright, the capsule is parked, and no locomotion integrates it.
//! * The SEAT phases (`EnteringIK`, `Seated`, `Driving`, `Exiting`, and the
//!   victim's `Jacked`) are inside `movement::step_driving`, because a body in
//!   them has a seat and the seat step is the one place a seated body is
//!   placed. P29.7's quintic window IS `EnteringIK` now — the same
//!   `seat_warp()` numbers, the same `warp_ease` — with a path that goes round
//!   the door instead of through the B-pillar.
//! * The HANDS AND FEET are [`follow_boarding`], called from the bridge's own
//!   write-back beside `follow_seats`, AFTER the solver — because a hand on a
//!   door handle has to be where the door IS, and the door, like the chassis,
//!   is moved by the solve. A request computed before it would be one step
//!   behind a car doing thirty metres a second, which is half a metre of hand
//!   floating in front of the wheel.
//!
//! # `interact::nearest_seat` is still the one door
//!
//! Nothing here resolves a seat. A press reaches [`begin`] with the chassis the
//! interaction door already chose, and an occupied driver's seat is never
//! begun: it is either the carjack verb (which begins with `carjack = true`
//! and pulls the occupant out on this same pipeline) or a refusal by name.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::boarding::{
    self as board, BoardPhase, BoardingState, PartGeom, SeatIndex, VehicleSockets,
};
use inf_ecs::components::{CharacterMovement, Collider3D, MovementMode, Transform};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement as model;
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;
use super::{CastTargets, ColliderId3D, ColliderShape3D};

/// **What one boarding body may add to a fixed step**, milliseconds, over the
/// same step with that body standing still — the whole step (movement,
/// vehicles, gameplay, the solver, the write-back with its hands and feet, and
/// the pose's IK), worst phase, RELEASE build, off CI. The population is ONE
/// boarding body: a player boards one car at a time. Measured and reported in
/// `veh3d_gate::a_boarding_costs_what_it_costs`; see `docs/profiling.md`.
pub const BOARDING_STEP_BUDGET_MS: f64 = 0.25;

/// **What the post-solve hands-and-feet pass may cost at SIXTY-FOUR seated
/// drivers**, milliseconds, RELEASE, off CI — the population
/// `VEHICLE_STEP_BUDGET_MS` is named at. Measured and reported in
/// `veh3d_gate::sixty_four_seated_drivers_cost_what_they_cost`.
pub const SEATED_POSTURE_BUDGET_MS: f64 = 0.5;

/// **One car, as the boarding sees it this step** — the live pose off the
/// solver and the geometry off the level.
#[derive(Clone, Debug)]
pub struct CarFrame {
    /// The chassis.
    pub chassis: Uuid,
    /// Its body's world position.
    pub pos: DVec3,
    /// Its body's world rotation.
    pub rot: DQuat,
    /// Its linear velocity, m/s.
    pub vel: DVec3,
    /// The collider's half-extents (its bounding box for a sphere or capsule).
    pub half: Vec3d,
    /// The collider's own offset from the body origin.
    pub offset: Vec3d,
    /// The parts still on it, as the socket derivation wants them.
    pub parts: Vec<PartGeom>,
    /// The eight sockets.
    pub sockets: VehicleSockets,
}

impl CarFrame {
    /// A chassis-frame point, in the world.
    pub fn world(&self, local: Vec3d) -> DVec3 {
        self.pos + self.rot * local.to_dvec3()
    }

    /// A chassis-frame direction, in the world.
    pub fn dir(&self, local: DVec3) -> DVec3 {
        self.rot * local
    }

    /// A world point, in the chassis frame.
    pub fn local(&self, world: DVec3) -> Vec3d {
        Vec3d::from_dvec3(self.rot.inverse() * (world - self.pos))
    }

    /// The chassis's compass heading, degrees.
    pub fn yaw_deg(&self) -> f64 {
        let f = self.rot * DVec3::Z;
        model::planar_yaw_deg(Vec2d::new(f.x, f.z))
    }

    /// The foot-well floor under a seat, chassis frame `y`.
    pub fn floor_y(&self) -> f64 {
        self.offset.y + board::SEAT_FLOOR_FRAC_Y * self.half.y.abs()
    }
}

/// **Read one car's frame**, or `None` for a guid that is not a live chassis.
///
/// `with_parts` asks for the parts census (the handles need it; a seated
/// body's wheel and pedals do not, and the census is a `BTreeMap` walk the
/// sixty-four-driver cost row does not need to pay).
pub fn car_frame(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    with_parts: bool,
) -> Option<CarFrame> {
    let body = bridge.body_of(chassis)?;
    let w = bridge.world();
    let pos = w.body_translation(body)?;
    let rot = w.body_rotation(body)?;
    let vel = w.body_linvel(body).unwrap_or(DVec3::ZERO);
    let collider = world
        .entity_of(chassis)
        .and_then(|e| world.world().get::<Collider3D>(e).copied())?;
    let half = inf_ecs::vehicle::chassis_half_extents(&collider);
    let parts = if with_parts {
        board::part_geoms(world, chassis)
    } else {
        Vec::new()
    };
    let sockets = board::sockets_of(half, collider.offset, &parts);
    Some(CarFrame {
        chassis,
        pos,
        rot,
        vel,
        half,
        offset: collider.offset,
        parts,
        sockets,
    })
}

/// **Where a door's handle is RIGHT NOW**, world metres — on the door as the
/// solver left it, so a hand chasing it follows the leaf round its hinge.
///
/// A LIVE door (VEH3c: opened or hit) is a rapier body on a revolute joint, and
/// its handle is that body's pose times the handle's offset in the door's own
/// frame. A LATCHED door has no body at all — it is a drawn child of the
/// chassis — so its handle is the chassis pose times the same offset from the
/// door's authored centre. One offset, two poses, and the arithmetic is
/// [`board::outer_handle_in_door`]'s in both.
///
/// `None` for a part that is not on the car.
pub fn handle_world(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    car: &CarFrame,
    door: Uuid,
    inner: bool,
) -> Option<DVec3> {
    let st = inf_ecs::bodywork::damage_row(world, car.chassis)?
        .parts
        .get(&door)
        .copied()?;
    if !st.latch.attached() {
        return None;
    }
    let side = if st.centre_frac.x >= 0.0 { 1.0 } else { -1.0 };
    let geom = PartGeom {
        kind: st.kind,
        centre_frac: st.centre_frac,
        half_frac: st.half_frac,
    };
    let (centre, half_m) = board::door_metres(car.half, Vec3d::ZERO, &geom);
    let off = if inner {
        board::inner_handle_in_door(half_m, side)
    } else {
        board::outer_handle_in_door(half_m, side)
    };
    if let Some(p) = bridge.part_body(door) {
        let w = bridge.world();
        if let (Some(at), Some(rot)) = (w.body_translation(p.body), w.body_rotation(p.body)) {
            return Some(at + rot * off.to_dvec3());
        }
    }
    Some(car.pos + car.rot * (centre.to_dvec3() + off.to_dvec3()))
}

/// **The hinge angle one door stands at**, degrees, magnitude — VEH3c's joint
/// read back through `part_angle_rad` (portable), or the bodywork row's own
/// `angle_deg` for a door the solver has never been given. `0` for no door.
pub fn door_angle_deg(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    door: Uuid,
) -> f64 {
    if door.is_nil() {
        return 0.0;
    }
    if let Some(a) = bridge.part_angle_rad(door) {
        return a.to_degrees().abs();
    }
    inf_ecs::bodywork::damage_row(world, chassis)
        .and_then(|r| r.parts.get(&door))
        .map(|s| s.angle_deg.abs())
        .unwrap_or(0.0)
}

/// Every collider that belongs to a car — its chassis and its live parts —
/// so a ground ray under a stance node sees the road and not the door.
fn car_colliders(bridge: &PhysicsBridge3D, chassis: Uuid) -> BTreeSet<ColliderId3D> {
    let mut out = BTreeSet::new();
    if let Some(c) = bridge.collider_of(chassis) {
        out.insert(c);
    }
    for g in bridge.parts_of_chassis(chassis) {
        if let Some(c) = bridge.part_body(g).and_then(|p| p.collider) {
            out.insert(c);
        }
    }
    out
}

/// How far above a candidate point the ground ray starts, metres.
const GROUND_PROBE_ABOVE_M: f64 = 1.2;
/// How far it looks down, metres.
const GROUND_PROBE_REACH_M: f64 = 3.5;

/// **The ground under a point**, world `y`, or `None` when there is none in
/// reach — a stance node over a drop is not a stance node.
pub fn ground_under(
    bridge: &mut PhysicsBridge3D,
    at: DVec3,
    exclude: &BTreeSet<ColliderId3D>,
) -> Option<f64> {
    let from = DVec3::new(at.x, at.y + GROUND_PROBE_ABOVE_M, at.z);
    bridge
        .world_mut()
        .cast_ray_where(
            from,
            -DVec3::Y,
            GROUND_PROBE_REACH_M,
            exclude,
            CastTargets::AllSolid,
        )
        .map(|h| h.point.y)
}

/// How far above the ground a placed capsule's feet are, metres — a hair over
/// the mover's own two-centimetre skin, so the first step settles rather than
/// starts inside the floor (P29.6's `settle_on_spawn` measurement).
pub const PLACE_SKIN_M: f64 = 0.03;

/// How much a clearance test shrinks the capsule's radius, metres, so that a
/// body standing flush against a wall is not "inside" it.
const CLEAR_SHRINK_M: f64 = 0.02;

/// **THE POINT-IN-COLLIDER CHECK** — whether a standing capsule with its feet
/// at `feet` overlaps anything solid in the world.
///
/// A capsule the character's own size (less a two-centimetre skin), swept one
/// millimetre: a sweep that STARTS penetrating is a body placed inside
/// geometry, which is the one thing an exit and a pull-out must never do.
/// `exclude` is the caller's — the body's own parked capsule.
pub fn capsule_clear(
    bridge: &mut PhysicsBridge3D,
    feet: DVec3,
    half_height: f64,
    radius: f64,
    exclude: &BTreeSet<ColliderId3D>,
) -> bool {
    let r = (radius - CLEAR_SHRINK_M).max(0.05);
    // Lifted by the shrink as well, so the bottom of the tested capsule is the
    // bottom of the real one plus the skin — the floor it stands on is not an
    // overlap.
    let centre = feet + DVec3::Y * (half_height + radius + CLEAR_SHRINK_M);
    let hit = bridge.world_mut().cast_shape_where(
        &ColliderShape3D::Capsule {
            half_height: half_height.max(0.0),
            radius: r,
        },
        centre,
        DQuat::IDENTITY,
        DVec3::Y,
        1e-3,
        exclude,
        CastTargets::AllSolid,
    );
    !matches!(hit, Some(h) if h.started_penetrating)
}

/// **Every character's capsule** — the set [`clear_exit`] lets step aside.
/// Walked only on an exit or a pull, which is a press and not a step.
pub fn people(world: &EcsWorld, bridge: &PhysicsBridge3D) -> BTreeSet<ColliderId3D> {
    model::movement_targets(world)
        .into_iter()
        .filter_map(|g| bridge.collider_of(g))
        .collect()
}

/// **Where a body may be put down beside a car**, world feet — the first of
/// [`board::exit_candidates`] that has ground under it and a clear capsule.
///
/// `None` when every candidate is inside something or over nothing: the car is
/// wedged, and the caller refuses the exit (or the pull) rather than put a body
/// in a wall. Returns the chosen point's index with it, which is what a gate
/// reads to know the preferred point was NOT the one taken.
///
/// `people` are the capsules that STEP ASIDE — the body's own and every other
/// character's ([`people`]). A person is not a wall: measured in the wave's
/// own demo, the carjacked driver and the passenger it forced out stood at
/// both doors for fourteen seconds, and every press of E in that time was a
/// refused exit from a car the hero had just stolen.
pub fn clear_exit(
    bridge: &mut PhysicsBridge3D,
    car: &CarFrame,
    preferred_local: Vec3d,
    half_height: f64,
    radius: f64,
    people: &BTreeSet<ColliderId3D>,
) -> Option<(DVec3, usize)> {
    let mut ground_ex = car_colliders(bridge, car.chassis);
    ground_ex.extend(people.iter().copied());
    let clear_ex = people.clone();
    // The ground a body may be put down on is the car's OWN ground: not a wall
    // top the probe happened to land on (a slab against the flank is ground
    // at two metres to a ray cast from above it), and not a drop.
    let car_bottom = car.pos.y + car.offset.y - car.half.y.abs();
    for (i, cand) in board::exit_candidates(preferred_local, car.half, car.offset)
        .iter()
        .enumerate()
    {
        let mut at = car.world(*cand);
        at.y = car.pos.y;
        // **No ground at all is WATER (or a drop), and that is an exit** — a
        // boat's occupant steps over the side and swims, which is what P29.7's
        // exit always did. The body is put down level with the car's own
        // underside and falls from there; ground that IS there but at a wall's
        // height is refused.
        let ground = match ground_under(bridge, at, &ground_ex) {
            Some(g) => {
                if g > car_bottom + EXIT_STEP_UP_M || g < car_bottom - EXIT_STEP_DOWN_M {
                    continue;
                }
                g
            }
            None => car_bottom,
        };
        let feet = DVec3::new(at.x, ground + PLACE_SKIN_M, at.z);
        if !capsule_clear(bridge, feet, half_height, radius, &clear_ex) {
            continue;
        }
        // **AND THE WAY THERE** — a point beyond a wall is a point a body would
        // have to pass through the wall to reach. The body's own capsule is
        // swept from inside the car (the car's own colliders excluded) out to
        // the point; anything in between refuses the candidate.
        let side = if cand.x - car.offset.x >= 0.0 { 1.0 } else { -1.0 };
        let inside = Vec3d::new(
            car.offset.x + side * 0.4 * car.half.x.abs(),
            0.0,
            cand.z.clamp(
                car.offset.z - car.half.z.abs(),
                car.offset.z + car.half.z.abs(),
            ),
        );
        let from_w = car.world(inside);
        let from = DVec3::new(from_w.x, feet.y, from_w.z);
        if path_blocked(bridge, from, feet, half_height, radius, &ground_ex) {
            continue;
        }
        return Some((feet, i));
    }
    None
}

/// How far ABOVE the car's own underside a body may be put down, metres — a
/// kerb, not a wall top.
pub const EXIT_STEP_UP_M: f64 = 0.5;

/// How far BELOW it, metres — a gutter, not a drop.
pub const EXIT_STEP_DOWN_M: f64 = 1.5;

/// Whether a standing capsule swept from `from` to `to` (feet points) meets
/// anything solid on the way, the car's own colliders and the body's excluded.
fn path_blocked(
    bridge: &mut PhysicsBridge3D,
    from: DVec3,
    to: DVec3,
    half_height: f64,
    radius: f64,
    exclude: &BTreeSet<ColliderId3D>,
) -> bool {
    let d = to - from;
    let len = d.length();
    if len <= 1e-6 {
        return false;
    }
    let r = (radius - CLEAR_SHRINK_M).max(0.05);
    let centre = from + DVec3::Y * (half_height + radius + CLEAR_SHRINK_M);
    bridge
        .world_mut()
        .cast_shape_where(
            &ColliderShape3D::Capsule {
                half_height: half_height.max(0.0),
                radius: r,
            },
            centre,
            DQuat::IDENTITY,
            d / len,
            len,
            exclude,
            CastTargets::AllSolid,
        )
        .is_some()
}

/// **Who is in one seat of one car**, or `None` — the inverse of
/// `SeatState`, derived from the characters (there is one answer to "is this
/// seat taken" and it lives on the character, `carjack::occupant_of`'s own
/// ruling).
pub fn occupant_in(world: &EcsWorld, chassis: Uuid, seat: SeatIndex) -> Option<Uuid> {
    for guid in model::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if world.world().get::<CharacterMovement>(e).is_some_and(|cm| {
            cm.runtime.seat.vehicle == chassis && cm.runtime.seat.seat == seat.as_u8()
        }) {
            return Some(guid);
        }
    }
    None
}

/// **How many bodies are in each seat of every car** — the census the
/// occupied-seat arm reads: any value over one is two people in one seat.
pub fn seat_census(world: &EcsWorld) -> std::collections::BTreeMap<(Uuid, u8), u32> {
    let mut out = std::collections::BTreeMap::new();
    for guid in model::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(cm) = world.world().get::<CharacterMovement>(e) {
            if cm.runtime.seat.is_seated() {
                *out.entry((cm.runtime.seat.vehicle, cm.runtime.seat.seat))
                    .or_insert(0) += 1;
            }
        }
    }
    out
}

/// What [`begin`] answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Begin {
    /// The machine is running.
    Started,
    /// **The seat is taken** and this was not a carjack — the occupied-seat
    /// rule. The press does nothing and the refusal is counted.
    Occupied,
    /// There is no car there any more, or no ground to stand on at its door.
    NoCar,
}

/// The standing height of a body — its capsule, doubled.
pub fn standing_m(cm: &CharacterMovement, radius: f64) -> f64 {
    2.0 * (cm.stand_half_height_m + radius)
}

/// **Start a boarding** — the press, after the interaction door chose a car.
///
/// Parks the collider (P29.7's door: *"parked at the START of the
/// choreography, not at the end"*), derives the take and the step-back points
/// from the chassis and its door, lays the Hermite from where the body stands
/// to the take, prices the reach dip, and — for a carjack — tells the
/// occupant it is being pulled out (`Jacked`, which brakes the car it is
/// driving).
///
/// **An occupied driver's seat is never begun as an ENTER.** That is the
/// VEH3b audit's inherited finding: every island car the hero could not drive
/// had its own traffic driver in it, and the occupant's silence beat the hero's
/// throttle every step. The interaction door already refuses to offer one; this
/// is the second lock on the same door, for any caller that reaches it by
/// another route.
#[allow(clippy::too_many_arguments)]
pub fn begin(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: Uuid,
    cm: &mut CharacterMovement,
    position: DVec3,
    radius: f64,
    chassis: Uuid,
    carjack: bool,
) -> Begin {
    let seat = SeatIndex::Driver;
    let occupant = occupant_in(world, chassis, seat).filter(|o| *o != guid);
    if occupant.is_some() && !carjack {
        return Begin::Occupied;
    }
    let Some(car) = car_frame(world, bridge, chassis, true) else {
        return Begin::NoCar;
    };
    let side = VehicleSockets::side_sign(seat);
    let door_geom = board::front_door(&car.parts, side).copied();
    let door = board::door_for_seat(world, chassis, seat).unwrap_or(Uuid::nil());
    let (take, back) =
        board::stance_points(&car.sockets, seat, car.half, car.offset, door_geom.as_ref());
    let mut exclude = car_colliders(bridge, chassis);
    if let Some(c) = bridge.collider_of(guid) {
        exclude.insert(c);
    }
    let lift = cm.stand_half_height_m + radius;
    let mut take_w = car.world(take);
    take_w.y = car.pos.y;
    // **A stance over no ground is not walked to** — a boat at a jetty, a
    // helicopter's far side over a drop. The body boards from where it stands,
    // which is exactly P29.7's warp: the approach becomes a turn in place and
    // the door phase has nothing to open.
    let (take, back, ground_y) = match ground_under(bridge, take_w, &exclude) {
        Some(g) => (take, back, g),
        None => {
            let here = car.local(position);
            let stay = Vec3d::new(here.x, 0.0, here.z);
            take_w = car.world(stay);
            (stay, stay, position.y - lift - PLACE_SKIN_M)
        }
    };
    let end = DVec3::new(take_w.x, ground_y + PLACE_SKIN_M + lift, take_w.z);
    // The facing to arrive at: the flank's INWARD normal — the body faces the
    // car, the doc's "lock character facing vector to vehicle side normal".
    let inward = car.dir(DVec3::new(-side, 0.0, 0.0));
    let stance_yaw = model::planar_yaw_deg(Vec2d::new(inward.x, inward.z));
    // The Hermite: leave along the way the body is already facing, arrive
    // along the inward normal. Both tangents are the chord's length, which is
    // the Catmull-Rom scaling — a curve that neither overshoots a short
    // approach nor cuts a long one.
    let chord = (end - position).length().max(1e-3);
    let yaw = cm.runtime.body_yaw_deg.to_radians();
    let facing = DVec3::new(inf_math::psin64(yaw), 0.0, inf_math::pcos64(yaw));
    let m0 = facing * chord;
    let m1 = inward * chord;
    let len = board::hermite_len(
        Vec3d::from_dvec3(position),
        Vec3d::from_dvec3(m0),
        Vec3d::from_dvec3(end),
        Vec3d::from_dvec3(m1),
    );
    // The dip, priced at the take against the handle as it stands.
    let standing = standing_m(cm, radius);
    let handle = if door.is_nil() {
        car.world(car.sockets.handle(seat))
    } else {
        handle_world(world, bridge, &car, door, false)
            .unwrap_or_else(|| car.world(car.sockets.handle(seat)))
    };
    // The NEAR shoulder: the body faces the flank, so its shoulders lie along
    // the car's own length, and the one on the handle's side is half a
    // shoulder span nearer the handle than the body's centre is.
    let plan = {
        let along = car.dir(DVec3::Z);
        let to = DVec3::new(handle.x - end.x, 0.0, handle.z - end.z);
        let toward = if to.dot(along) >= 0.0 { along } else { -along };
        let shoulder = end + toward * (SHOULDER_HALF_SPAN_FRAC * standing);
        DVec3::new(handle.x - shoulder.x, 0.0, handle.z - shoulder.z).length()
    };
    // No door, no handle: a car whose door is in the road (or that never had
    // one) is boarded through the opening and nobody bends to reach nothing.
    let dip = if door.is_nil() {
        0.0
    } else {
        board::reach_dip_m(standing, ground_y, handle.y, plan)
    };

    let mut b = BoardingState {
        vehicle: chassis,
        seat: seat.as_u8(),
        door,
        take_local: take,
        back_local: back,
        ground_y,
        stance_yaw_deg: stance_yaw,
        start: Vec3d::from_dvec3(position),
        start_yaw_deg: cm.runtime.body_yaw_deg,
        tangent_start: Vec3d::from_dvec3(m0),
        tangent_end_len: chord,
        carjack,
        other: occupant.unwrap_or(Uuid::nil()),
        dip_m: dip,
        door_deg: door_angle_deg(world, bridge, chassis, door),
        ..Default::default()
    };
    b.enter(BoardPhase::Locked, board::LOCKED_S);
    // The approach length is decided NOW, from the curve as laid — the clock is
    // what makes two hosts agree about when the body arrives.
    b.phase_len_s = board::LOCKED_S;
    b.hand_err_m = 0.0;
    b.foot_err_m = 0.0;
    b.tangent_end_len = chord;
    // `mark_s` carries the approach's own length into `Unlocking`.
    b.mark_s = board::approach_s(len);
    cm.runtime.boarding = b;
    super::vehicle::park_collider(bridge, guid, true);
    if let Some(victim) = occupant {
        mark_jacked(world, victim, guid, chassis);
    }
    Begin::Started
}

/// Half the shoulder span as a fraction of standing height — `0.20 / 1.75`,
/// `inf_anim::BodyParams`' own default girdle. The near shoulder is this much
/// closer to a handle beside the body than the body's centre is.
pub const SHOULDER_HALF_SPAN_FRAC: f64 = 0.114;

/// **Tell an occupant it is being pulled out.** The car it is driving is
/// braked from this step (its seat step reads `Jacked`) and the pull itself
/// happens when the hero's door is open.
fn mark_jacked(world: &mut EcsWorld, victim: Uuid, by: Uuid, chassis: Uuid) {
    let Some(e) = world.entity_of(victim) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        let seat = cm.runtime.seat.seat;
        let mut b = BoardingState {
            vehicle: chassis,
            seat,
            carjack: true,
            other: by,
            ..Default::default()
        };
        b.enter(BoardPhase::Jacked, board::JACKED_MAX_S);
        cm.runtime.boarding = b;
    }
}

/// **Release a body the boarding was holding** — back on its feet, collider
/// restored, no hand or foot request left behind. Used by a refusal mid-way
/// and by the end of `ClosingDoor`.
pub fn release(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: Uuid,
    cm: &mut CharacterMovement,
) {
    cm.runtime.boarding = BoardingState::default();
    cm.runtime.pelvis_offset = Vec3d::ZERO;
    super::vehicle::park_collider(bridge, guid, false);
    inf_ecs::pose::set_hand_ik(world, guid, inf_ecs::pose::HandIk::default());
    inf_ecs::anim_bridge::set_foot_ik(world, guid, [None, None]);
}

/// **Let a victim go** — a carjack refused after the victim was told: the car
/// is its own again and it drives on.
pub fn unjack(world: &mut EcsWorld, victim: Uuid) {
    let Some(e) = world.entity_of(victim) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        if cm.runtime.boarding.phase == BoardPhase::Jacked {
            cm.runtime.boarding = BoardingState::default();
        }
    }
}

/// **The pull** — the victim's reverse pipeline, FORCED.
///
/// Finds a clear place to put the victim down (the pull-out point first, then
/// the exit candidates — the VEH2b carried collide-check, closed), and starts
/// the victim's own `Exiting` with that point as its destination. `false` when
/// there is nowhere clear: the car is wedged against something on every side,
/// and the carjack is refused rather than a body put through a wall.
pub fn start_pull(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    victim: Uuid,
    car: &CarFrame,
) -> bool {
    let Some(e) = world.entity_of(victim) else {
        return false;
    };
    let Some(cm) = world.world().get::<CharacterMovement>(e).cloned() else {
        return false;
    };
    let radius = world
        .world()
        .get::<Collider3D>(e)
        .map(|c| c.radius)
        .unwrap_or(0.3);
    let seat = SeatIndex::from_u8(cm.runtime.seat.seat);
    let preferred = board::pull_out_point(&car.sockets, seat, car.half, car.offset);
    let mut people_here = people(world, bridge);
    people_here.extend(bridge.collider_of(victim));
    let Some((feet, _)) =
        clear_exit(bridge, car, preferred, cm.stand_half_height_m, radius, &people_here)
    else {
        return false;
    };
    let by = cm.runtime.boarding.other;
    // **A passenger bails out as well** (wave VEH3d): a front-seat rider in a
    // car that is being carjacked leaves through its own door, on its own
    // reverse pipeline, forced — GTA's reference picture, and the honest one: a
    // stranger does not stay in the seat beside the person who pulled the
    // driver out. Best-effort: a rider with nowhere clear to go stays put.
    if seat.drives() {
        if let Some(rider) = occupant_in(world, car.chassis, SeatIndex::Passenger) {
            if rider != victim {
                force_out(world, bridge, rider, car, by);
            }
        }
    }
    let door = board::door_for_seat(world, car.chassis, seat).unwrap_or(Uuid::nil());
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        let mut b = BoardingState {
            vehicle: car.chassis,
            seat: seat.as_u8(),
            door,
            back_local: car.local(feet),
            ground_y: feet.y - PLACE_SKIN_M,
            carjack: true,
            other: by,
            ..Default::default()
        };
        b.enter(BoardPhase::Exiting, 0.0);
        // FORCED: the door is already open (the hero opened it), so the warp
        // starts now rather than after a door phase of its own.
        b.mark_s = 0.0;
        cm.runtime.boarding = b;
    }
    true
}

/// Start a seated body's FORCED exit to the first clear point beside its own
/// seat — [`start_pull`]'s arithmetic for a body nobody is standing at the
/// door of.
fn force_out(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    who: Uuid,
    car: &CarFrame,
    by: Uuid,
) -> bool {
    let Some(e) = world.entity_of(who) else {
        return false;
    };
    let Some(cm) = world.world().get::<CharacterMovement>(e).cloned() else {
        return false;
    };
    let radius = world
        .world()
        .get::<Collider3D>(e)
        .map(|c| c.radius)
        .unwrap_or(0.3);
    let seat = SeatIndex::from_u8(cm.runtime.seat.seat);
    let preferred = board::pull_out_point(&car.sockets, seat, car.half, car.offset);
    let mut people_here = people(world, bridge);
    people_here.extend(bridge.collider_of(who));
    let Some((feet, _)) =
        clear_exit(bridge, car, preferred, cm.stand_half_height_m, radius, &people_here)
    else {
        return false;
    };
    let door = board::door_for_seat(world, car.chassis, seat).unwrap_or(Uuid::nil());
    if !door.is_nil() {
        super::bodywork::set_part_open(world, bridge, car.chassis, door, true);
    }
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        let mut b = BoardingState {
            vehicle: car.chassis,
            seat: seat.as_u8(),
            door,
            back_local: car.local(feet),
            ground_y: feet.y - PLACE_SKIN_M,
            carjack: true,
            other: by,
            ..Default::default()
        };
        b.enter(BoardPhase::Exiting, 0.0);
        cm.runtime.boarding = b;
    }
    true
}

/// **The ground choreography** — `Locked`, `Unlocking`, `OpeningDoor` and
/// `ClosingDoor`. Answers where the capsule centre goes this step and the
/// facing, and moves the machine; the caller writes them back.
///
/// Everything here is a function of the phase clock and the world, so both
/// hosts walk the same body along the same curve.
pub enum GroundStep {
    /// Stay on the ground at this capsule centre, facing this way.
    Stand {
        /// The capsule centre.
        at: DVec3,
        /// The facing, degrees.
        yaw_deg: f64,
    },
    /// The door phase is done: sit down. The seat step takes over this step.
    IntoSeat {
        /// Where the warp starts.
        at: DVec3,
        /// Its facing.
        yaw_deg: f64,
    },
    /// The boarding was refused part-way (no car, a carjack with nowhere to
    /// put the victim, a victim who never came out). Release the body here.
    Refused {
        /// Where it stands.
        at: DVec3,
    },
    /// `ClosingDoor` is done; the body is free.
    Done {
        /// Where it stands.
        at: DVec3,
    },
}

/// Advance one ground phase. `position` is the capsule centre the body had at
/// the top of the step.
pub fn step_ground(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    cm: &mut CharacterMovement,
    position: DVec3,
    radius: f64,
    dt: f64,
) -> GroundStep {
    let mut b = cm.runtime.boarding;
    b.time_s += dt;
    let lift = cm.stand_half_height_m + radius;
    let Some(car) = car_frame(world, bridge, b.vehicle, false) else {
        cm.runtime.boarding = b;
        return GroundStep::Refused { at: position };
    };
    let side = VehicleSockets::side_sign(b.seat_index());
    let inward = car.dir(DVec3::new(-side, 0.0, 0.0));
    let stance_yaw = model::planar_yaw_deg(Vec2d::new(inward.x, inward.z));
    let at_ground = |local: Vec3d, ground_y: f64| {
        let w = car.world(local);
        DVec3::new(w.x, ground_y + PLACE_SKIN_M + lift, w.z)
    };
    let take = at_ground(b.take_local, b.ground_y);
    let back = at_ground(b.back_local, b.ground_y);
    b.door_deg = door_angle_deg(world, bridge, b.vehicle, b.door);
    let out = match b.phase {
        BoardPhase::Locked => {
            if b.time_s >= board::LOCKED_S {
                let approach = b.mark_s.max(board::UNLOCK_MIN_S);
                b.enter(BoardPhase::Unlocking, approach);
            }
            GroundStep::Stand {
                at: position,
                yaw_deg: cm.runtime.body_yaw_deg,
            }
        }
        BoardPhase::Unlocking => {
            // Root motion along the Hermite. The END is the take as the car
            // stands NOW, so a car that is nudged (or a victim's car that is
            // still braking) is boarded at its door.
            let alpha = b.alpha();
            let p = board::hermite(
                b.start,
                b.tangent_start,
                Vec3d::from_dvec3(take),
                Vec3d::from_dvec3(inward * b.tangent_end_len),
                alpha,
            )
            .to_dvec3();
            // The facing turns to the flank as the curve arrives, and is LOCKED
            // to it from the end of the approach on.
            let s = alpha * alpha * (3.0 - 2.0 * alpha);
            let yaw = model::wrap_deg(
                b.start_yaw_deg + model::angle_delta_deg(stance_yaw, b.start_yaw_deg) * s,
            );
            if b.expired() {
                b.enter(BoardPhase::OpeningDoor, 0.0);
            }
            GroundStep::Stand {
                at: p,
                yaw_deg: yaw,
            }
        }
        BoardPhase::OpeningDoor => {
            let pull_at = board::HAND_REACH_S + board::HANDLE_HOLD_S;
            // The latch: once the hand has held the handle, the motor is told.
            if b.mark_s < 0.0 && b.time_s >= pull_at {
                b.mark_s = b.time_s;
                if !b.door.is_nil() {
                    super::bodywork::set_part_open(world, bridge, b.vehicle, b.door, true);
                }
            }
            // The step back, from the moment the door starts to swing.
            let back_a = if b.mark_s >= 0.0 {
                ((b.time_s - b.mark_s) / board::STEP_BACK_S).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let s = back_a * back_a * (3.0 - 2.0 * back_a);
            let p = take + (back - take) * s;
            let door_open = b.door.is_nil() || b.door_deg >= board::DOOR_BOARD_DEG;
            let waited = b.mark_s >= 0.0 && b.time_s - b.mark_s >= board::OPENING_MAX_S;
            let stepped = back_a >= 1.0;
            let mut outcome = GroundStep::Stand {
                at: p,
                yaw_deg: stance_yaw,
            };
            if stepped && (door_open || waited) {
                if b.carjack {
                    match seat_state(world, b.other, b.vehicle) {
                        // The victim is still at the wheel and has not been
                        // pulled yet: pull.
                        Some(BoardPhase::Jacked) => {
                            if !start_pull(world, bridge, b.other, &car) {
                                b.enter(BoardPhase::Idle, 0.0);
                                outcome = GroundStep::Refused { at: p };
                            }
                        }
                        // Out of the seat: in we go.
                        None => {
                            outcome = GroundStep::IntoSeat {
                                at: p,
                                yaw_deg: stance_yaw,
                            }
                        }
                        // Being pulled: wait for it, bounded.
                        Some(_) => {
                            if b.time_s - b.mark_s.max(0.0)
                                >= board::OPENING_MAX_S + board::PULL_WAIT_MAX_S
                            {
                                outcome = GroundStep::Refused { at: p };
                            }
                        }
                    }
                } else if occupant_in(world, b.vehicle, b.seat_index()).is_some() {
                    // Somebody got in while the door was opening. The rule is
                    // the rule: an occupied seat is never entered.
                    outcome = GroundStep::Refused { at: p };
                } else {
                    outcome = GroundStep::IntoSeat {
                        at: p,
                        yaw_deg: stance_yaw,
                    };
                }
            }
            outcome
        }
        BoardPhase::ClosingDoor => {
            let shut = b.door.is_nil() || b.door_deg <= board::DOOR_SHUT_DEG;
            if (shut && b.time_s >= board::HAND_REACH_S) || b.time_s >= board::CLOSING_MAX_S {
                GroundStep::Done { at: position }
            } else {
                GroundStep::Stand {
                    at: position,
                    yaw_deg: cm.runtime.body_yaw_deg,
                }
            }
        }
        _ => GroundStep::Stand {
            at: position,
            yaw_deg: cm.runtime.body_yaw_deg,
        },
    };
    cm.runtime.boarding = b;
    out
}

/// The boarding phase of a body IN a given car, or `None` when it is not in
/// that car at all (pulled out, never there).
fn seat_state(world: &EcsWorld, guid: Uuid, chassis: Uuid) -> Option<BoardPhase> {
    let e = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(e)?;
    (cm.runtime.seat.vehicle == chassis).then_some(cm.runtime.boarding.phase)
}

// ── the hands and the feet, AFTER the solver ────────────────────────────────

/// What one step of [`follow_boarding`] did — the engagement counters a gate
/// and a cost row read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoardingReport {
    /// Bodies whose hands the boarding placed this step.
    pub hands: u32,
    /// Bodies whose feet it placed.
    pub feet: u32,
    /// Seated bodies it walked.
    pub seated: u32,
}

/// Whether a body carries a skeleton the pose step will solve against — a
/// request for a capsule with no rig is a request nothing reads.
fn has_rig(world: &EcsWorld, e: inf_ecs::Entity) -> bool {
    world
        .world()
        .get::<inf_ecs::components::SkeletalMesh>(e)
        .is_some_and(|s| s.skeleton.is_some())
}

/// **Place the hands and the feet of every boarding or seated body** — the
/// post-solve pass, beside `follow_seats`.
///
/// For each body this writes ONE hand request and ONE foot request through the
/// existing doors (`inf_ecs::pose::set_hand_ik`, `inf_ecs::anim_bridge::
/// set_foot_ik`); the gameplay hand pass leaves a body alone while the
/// boarding owns its hands, so there is one producer per body per step, which
/// is SK1c's law.
///
/// It also reads back what the pose step made of LAST step's requests — the
/// hand and foot residuals — into the machine, so `hero.csv`, the HUD and the
/// gate read a measurement of the solve and not a restatement of the goal.
pub fn follow_boarding(bridge: &mut PhysicsBridge3D, world: &mut EcsWorld) -> BoardingReport {
    let mut report = BoardingReport::default();
    if bridge.vehicle_count() == 0 {
        return report;
    }
    for guid in model::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(cm) = world.world().get::<CharacterMovement>(e).cloned() else {
            continue;
        };
        let b = cm.runtime.boarding;
        let seated = cm.runtime.seat.is_seated() && !cm.runtime.seat.entering;
        if b.phase == BoardPhase::Idle && !seated {
            continue;
        }
        if !has_rig(world, e) {
            continue;
        }
        let chassis = if b.phase != BoardPhase::Idle {
            b.vehicle
        } else {
            cm.runtime.seat.vehicle
        };
        let with_parts = !b.door.is_nil();
        let Some(car) = car_frame(world, bridge, chassis, with_parts) else {
            continue;
        };
        if seated {
            report.seated += 1;
        }
        let radius = world
            .world()
            .get::<Collider3D>(e)
            .map(|c| c.radius)
            .unwrap_or(0.3);
        let standing = standing_m(&cm, radius);
        let reach = board::ARM_REACH_FRAC * standing;
        // What the pose made of last step's requests: the residuals, and the
        // shoulders the arms hang from (for choosing which hand, and pricing
        // the reach).
        let last = inf_ecs::pose::hand_ik_report(world, guid).cloned();
        // The rig's own shoulders when the pose published them last step; the
        // capsule's estimate otherwise. The fallback is not a nicety: a hand
        // request that fades to nothing is REMOVED, which removes its report,
        // and a reach priced against "no shoulder" at full weight put the hand
        // straight back -- measured as a hand that flickered between 0.07 and
        // 1.00 of weight every other step while the door swung away from it.
        let estimate = shoulder_estimate(world, e, &cm, radius);
        let shoulders: [Option<DVec3>; 2] = match &last {
            Some(r) => [
                r.shoulder[0].map(|v| v.to_dvec3()).or(Some(estimate[0])),
                r.shoulder[1].map(|v| v.to_dvec3()).or(Some(estimate[1])),
            ],
            None => [Some(estimate[0]), Some(estimate[1])],
        };
        let hand_err = last
            .as_ref()
            .map(|r| {
                r.reach
                    .iter()
                    .flatten()
                    .filter_map(|o| match o {
                        inf_ecs::pose::IkOutcome::Solved(s) => Some(f64::from(s.reach_error)),
                        _ => None,
                    })
                    .fold(0.0f64, f64::max)
            })
            .unwrap_or(0.0);
        let foot_err = inf_ecs::anim_bridge::foot_error(world, guid)
            .map(|e| e.iter().flatten().fold(0.0f64, |a, v| a.max(v.abs())))
            .unwrap_or(0.0);
        // The body's own lateral: which chassis side its RIGHT shoulder is on.
        // A rig built by `inf_anim::build_template` answers `+X`; an import that
        // kept glTF's handedness answers `-X`. Unknown until the first pose.
        let right_is_plus_x = match (shoulders[0], shoulders[1]) {
            (Some(l), Some(r)) => {
                let lr = car.local(r).x - car.local(l).x;
                lr >= 0.0
            }
            _ => true,
        };
        let seat = if b.phase != BoardPhase::Idle {
            b.seat_index()
        } else {
            SeatIndex::from_u8(cm.runtime.seat.seat)
        };
        let mut hands: [Option<(DVec3, f64)>; 2] = [None, None];
        let mut feet: [Option<DVec3>; 2] = [None, None];
        let mut poles: [Option<DVec3>; 2] = [None, None];
        let mut feet_w = 0.0f64;
        let phase = if b.phase == BoardPhase::Idle && seated {
            BoardPhase::Driving
        } else {
            b.phase
        };
        // The hands arrive on a ramp at the start of a phase that puts them
        // somewhere new — and do NOT at the start of `Exiting`, which begins
        // with the hands already on the rim and the pull (a ramp there dropped
        // both hands off the wheel for a fifth of a second on the press).
        let ramp = if matches!(b.phase, BoardPhase::Idle | BoardPhase::Exiting) {
            1.0
        } else {
            board::phase_hand_weight(phase, b.time_s)
        };
        let nearest_hand = |target: DVec3| -> usize {
            match (shoulders[0], shoulders[1]) {
                (Some(l), Some(r)) => usize::from((r - target).length() < (l - target).length()),
                _ => usize::from(b.hand_side != 0),
            }
        };
        let reach_w = |side: usize, target: DVec3| -> f64 {
            match shoulders[side] {
                Some(s) => board::reach_weight((s - target).length(), reach),
                None => 1.0,
            }
        };
        let mut hand_side = b.hand_side;
        let mut redip: Option<f64> = None;
        match phase {
            BoardPhase::OpeningDoor | BoardPhase::ClosingDoor => {
                if let Some(h) = (!b.door.is_nil())
                    .then(|| handle_world(world, bridge, &car, b.door, false))
                    .flatten()
                {
                    // Chosen at the phase's first step — and chosen AGAIN while
                    // the hand is still arriving, once the rig's own shoulders
                    // are on the report: the first step's are the capsule's
                    // estimate, whose "right" is the template's +X, and on the
                    // island's MetaHuman that picked the FAR hand (measured:
                    // the right shoulder 0.44 m from the handle, the left
                    // 0.24 m), which no dip could bring within reach.
                    let from_rig = last
                        .as_ref()
                        .is_some_and(|r| r.shoulder.iter().all(Option::is_some));
                    let arriving = b.mark_s < 0.0 && b.time_s < board::HAND_REACH_S;
                    let side = if b.time_s <= dt_eps() || (from_rig && arriving) {
                        nearest_hand(h)
                    } else {
                        usize::from(b.hand_side != 0)
                    };
                    hand_side = side as u8;
                    // Held at full weight while the latch is pulled; after
                    // that the reach decides, so a door swinging away and a
                    // body stepping back let go of it smoothly.
                    let pulled = b.mark_s >= 0.0 || phase == BoardPhase::ClosingDoor;
                    let w = if pulled {
                        ramp * reach_w(side, h)
                    } else {
                        ramp
                    };
                    hands[side] = Some((h, w));
                    // **The dip, re-priced on the body's OWN arm** while the
                    // hand is still reaching: `begin` priced it off the
                    // capsule's proportions, which are the template
                    // mannequin's. Measured on the island's MetaHuman, that
                    // plan left the hand 75.5 mm short of the handle at weight
                    // 1. The shoulder is last step's pose, so the pelvis offset
                    // it was dipped by is added back to find the undipped one;
                    // the answer converges once the ramp saturates.
                    if phase == BoardPhase::OpeningDoor && b.mark_s < 0.0 {
                        let arm = last.as_ref().and_then(|r| r.arm_len[side]);
                        if let (Some(s), Some(len)) = (shoulders[side], arm) {
                            let plan = DVec3::new(h.x - s.x, 0.0, h.z - s.z).length();
                            let upright = s.y - cm.runtime.pelvis_offset.y;
                            let l = REACH_USE_FRAC * len;
                            let want_dy = (l * l - plan * plan).max(0.0).sqrt();
                            let target = (upright - h.y - want_dy).clamp(0.0, board::MAX_REACH_DIP_M);
                            // HALF the way each step: the shoulder read is one
                            // step old, so a full correction overshoots and the
                            // pelvis was measured bobbing 0.25 <-> 0.30 m on
                            // alternate steps; half a step's error converges.
                            redip = Some(b.dip_m + REDIP_GAIN * (target - b.dip_m));
                        }
                    }
                }
                // The dip needs the feet held on the ground, or the whole
                // leg chain goes down with the pelvis.
                if cm.runtime.pelvis_offset.y < -1e-4 {
                    if let Some(f) = inf_ecs::anim_bridge::feet_of(world, guid) {
                        for (i, s) in f.iter().enumerate() {
                            if let Some(s) = s {
                                feet[i] = Some(DVec3::new(
                                    s.world.x,
                                    b.ground_y + ANKLE_ABOVE_GROUND_M,
                                    s.world.z,
                                ));
                            }
                        }
                        feet_w = 1.0;
                    }
                }
            }
            BoardPhase::EnteringIK
            | BoardPhase::Seated
            | BoardPhase::Driving
            | BoardPhase::Jacked
            | BoardPhase::Exiting => {
                let drives = seat.drives();
                let grips = if drives {
                    let steer = vehicle_steer(world, bridge, chassis);
                    board::wheel_grips(&car.sockets, steer)
                } else {
                    board::passenger_grips(&car.sockets, seat)
                };
                // Grip 0 is on the `+X` side: the RIGHT hand's, for a body
                // whose right is `+X`.
                let (right_grip, left_grip) = if right_is_plus_x {
                    (grips[0], grips[1])
                } else {
                    (grips[1], grips[0])
                };
                let mut rw = ramp;
                let mut lw = ramp;
                // `EnteringIK`: the hands are the climb's, not the wheel's.
                if phase == BoardPhase::EnteringIK {
                    rw = 0.0;
                    lw = 0.0;
                }
                // `Seated`: the door is pulled shut first, by the hand nearer it.
                if phase == BoardPhase::Seated || phase == BoardPhase::Exiting {
                    if let Some(h) = (!b.door.is_nil())
                        .then(|| handle_world(world, bridge, &car, b.door, true))
                        .flatten()
                    {
                        let side = nearest_hand(h);
                        let shut = b.door_deg <= board::DOOR_SHUT_DEG;
                        let warping = phase == BoardPhase::Exiting && b.mark_s >= 0.0;
                        // Seated: pulling it SHUT, so only while it is open.
                        // Exiting: pushing it OPEN, from the press to the warp.
                        let pulling = match phase {
                            BoardPhase::Seated => !shut,
                            _ => !warping,
                        };
                        if pulling {
                            hands[side] = Some((h, ramp * reach_w(side, h)));
                            hand_side = side as u8;
                            if side == 1 {
                                rw = 0.0;
                            } else {
                                lw = 0.0;
                            }
                        }
                    }
                }
                if phase == BoardPhase::Exiting && b.mark_s >= 0.0 {
                    rw = 0.0;
                    lw = 0.0;
                }
                if hands[1].is_none() && rw > 0.0 {
                    hands[1] = Some((car.world(right_grip), rw));
                }
                if hands[0].is_none() && lw > 0.0 {
                    hands[0] = Some((car.world(left_grip), lw));
                }
                // The feet.
                let floor = car.floor_y();
                let (t, bk) = if drives {
                    let (throttle, brake) = pedal_inputs(&cm);
                    let (mut tp, mut bp) = board::pedal_faces(&car.sockets, throttle, brake);
                    if !right_is_plus_x {
                        let sx = car.sockets.seat(seat).x;
                        tp = board::mirror_about_seat(tp, sx);
                        bp = board::mirror_about_seat(bp, sx);
                    }
                    (tp, bp)
                } else {
                    let f = board::floor_feet(&car.sockets, seat, floor);
                    if right_is_plus_x {
                        (f[1], f[0])
                    } else {
                        (f[0], f[1])
                    }
                };
                // Right foot on the throttle, left on the brake.
                feet[1] = Some(car.world(t));
                feet[0] = Some(car.world(bk));
                // **The knees bend UP** -- a pole above each knee, between the
                // cushion and the pedal. See `FootGoal::pole` for the
                // measurement.
                let cushion = car.world(car.sockets.seat(seat));
                let up = car.dir(DVec3::Y);
                for (i, f) in feet.iter().enumerate() {
                    if let Some(f) = f {
                        poles[i] = Some((cushion + *f) * 0.5 + up * SEATED_KNEE_POLE_UP_M);
                    }
                }
                feet_w = match phase {
                    // The legs swing in with the body: the same eased clock the
                    // seat warp runs on.
                    BoardPhase::EnteringIK => inf_anim::warp_ease(b.alpha()),
                    BoardPhase::Exiting if b.mark_s >= 0.0 => {
                        let a = ((b.time_s - b.mark_s) / board::EXIT_WARP_S).clamp(0.0, 1.0);
                        1.0 - inf_anim::warp_ease(a)
                    }
                    _ => 1.0,
                };
            }
            _ => {}
        }
        // ── write ──
        let req = inf_ecs::pose::HandIk {
            reach: [
                hands[0]
                    .filter(|(_, w)| *w > 0.0)
                    .map(|(t, w)| inf_ecs::pose::HandReach {
                        target: Vec3d::from_dvec3(t),
                        weight: w as f32,
                    }),
                hands[1]
                    .filter(|(_, w)| *w > 0.0)
                    .map(|(t, w)| inf_ecs::pose::HandReach {
                        target: Vec3d::from_dvec3(t),
                        weight: w as f32,
                    }),
            ],
            gun: None,
            grip: [
                hands[0]
                    .filter(|(_, w)| *w > 0.0)
                    .map(|(_, w)| inf_ecs::pose::HandGrip {
                        name: inf_anim::GRIP_HANDLE.to_string(),
                        amount: w as f32,
                    }),
                hands[1]
                    .filter(|(_, w)| *w > 0.0)
                    .map(|(_, w)| inf_ecs::pose::HandGrip {
                        name: inf_anim::GRIP_HANDLE.to_string(),
                        amount: w as f32,
                    }),
            ],
        };
        if !req.is_empty() {
            report.hands += 1;
        }
        inf_ecs::pose::set_hand_ik(world, guid, req);
        let goals = if feet_w > 0.0 {
            report.feet += 1;
            [
                feet[0].map(|t| inf_ecs::anim_bridge::FootGoal {
                    target: Vec3d::from_dvec3(t),
                    weight: feet_w as f32,
                    pitch_deg: 0.0,
                    roll_deg: 0.0,
                    pole: poles[0].map(Vec3d::from_dvec3),
                    unlimited: poles[0].is_some(),
                }),
                feet[1].map(|t| inf_ecs::anim_bridge::FootGoal {
                    target: Vec3d::from_dvec3(t),
                    weight: feet_w as f32,
                    pitch_deg: 0.0,
                    roll_deg: 0.0,
                    pole: poles[1].map(Vec3d::from_dvec3),
                    unlimited: poles[1].is_some(),
                }),
            ]
        } else {
            [None, None]
        };
        inf_ecs::anim_bridge::set_foot_ik(world, guid, goals);
        // ── the read-back, into the machine ──
        let hand_weight = hands
            .iter()
            .flatten()
            .map(|(_, w)| *w)
            .fold(0.0f64, f64::max);
        if let Some(mut slot) = world.world_mut().get_mut::<CharacterMovement>(e) {
            let bb = &mut slot.runtime.boarding;
            bb.hand_err_m = hand_err;
            bb.foot_err_m = foot_err;
            if bb.phase != BoardPhase::Idle {
                bb.hand_weight = hand_weight;
                bb.hand_side = hand_side;
                if let Some(d) = redip {
                    bb.dip_m = d;
                }
            }
        }
    }
    report
}

/// The share of the re-priced dip's error taken per step — see the call.
pub const REDIP_GAIN: f64 = 0.5;

/// How much of a measured arm the re-priced reach dip plans to use: a hand
/// asked for at the arm's full length is a straight elbow, which the two-bone
/// solve reaches only in the limit.
pub const REACH_USE_FRAC: f64 = 0.96;

/// **How far above the knee a seated leg's pole is**, metres -- the point the
/// knee bends toward, above the midpoint of cushion and pedal.
pub const SEATED_KNEE_POLE_UP_M: f64 = 0.6;

/// **Where a body's shoulders are, estimated from its capsule** -- `[left,
/// right]`, world metres: the standing shoulder height plus the pelvis offset,
/// half a shoulder span either side along the body's own right axis.
///
/// The right axis is `inf_anim::build_template`'s convention (right is model
/// `+X`); a rig that disagrees publishes its real shoulders through the hand
/// pass's report, which is read first.
fn shoulder_estimate(
    world: &EcsWorld,
    e: inf_ecs::Entity,
    cm: &CharacterMovement,
    radius: f64,
) -> [DVec3; 2] {
    let centre = world
        .world()
        .get::<Transform>(e)
        .map(|t| t.translation.to_dvec3())
        .unwrap_or(DVec3::ZERO);
    let standing = standing_m(cm, radius);
    let feet = centre - DVec3::Y * (cm.stand_half_height_m + radius);
    let h = board::SHOULDER_FRAC * standing + cm.runtime.pelvis_offset.y;
    let yaw = cm.runtime.body_yaw_deg.to_radians();
    let right = DVec3::new(inf_math::pcos64(yaw), 0.0, -inf_math::psin64(yaw));
    let half = SHOULDER_HALF_SPAN_FRAC * standing;
    let mid = feet + DVec3::Y * h;
    [mid - right * half, mid + right * half]
}

/// The first step of a phase, in seconds — a phase whose clock has not passed
/// one fixed step is choosing its hand.
fn dt_eps() -> f64 {
    1.0 / 60.0 + 1e-9
}

/// How far above the ground an ankle joint rests on a standing body, metres —
/// the foot goal a dipped body's feet are held at. The rig's own ankle
/// height is 0.06–0.10 m on every body this engine builds or imports; a goal a
/// few centimetres off it is a knee that bends a few degrees more, never a foot
/// through the road.
pub const ANKLE_ABOVE_GROUND_M: f64 = 0.08;

/// **The rack angle, as the rim shows it** — the mean steer of the wheels that
/// steer, turned into rim degrees by the chassis's own `max_steer_deg`. The
/// number `hero.csv`'s `rim_deg` carries and the hands' grips turn by.
pub fn vehicle_steer(world: &EcsWorld, bridge: &PhysicsBridge3D, chassis: Uuid) -> f64 {
    let Some(v) = bridge.vehicle_of(chassis) else {
        return 0.0;
    };
    let (mut sum, mut n) = (0.0f64, 0usize);
    for w in v.wheels() {
        if w.steer_deg.abs() > 1e-9 {
            sum += w.steer_deg;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    let max = world
        .entity_of(chassis)
        .and_then(|e| world.world().get::<inf_ecs::components::VehicleClass>(e))
        .map(|c| c.max_steer_deg)
        .unwrap_or(inf_ecs::vehicle::VehicleTuning::default().max_steer_deg);
    board::rim_angle_deg(sum / n as f64, max)
}

/// The throttle and the brake this body asked for, `[0, 1]` each — recorded by
/// the seat step from the same `VehicleControls::from_intent` the car was
/// given, so the foot presses exactly what the car heard.
fn pedal_inputs(cm: &CharacterMovement) -> (f64, f64) {
    (
        cm.runtime.boarding.throttle_in.clamp(0.0, 1.0),
        cm.runtime.boarding.brake_in.clamp(0.0, 1.0),
    )
}

/// **Where the capsule centre goes during `EnteringIK`**, world — the Hermite
/// from the step-back point into the seat, round the door rather than through
/// the pillar: it leaves heading FORWARD along the car and arrives heading
/// INWARD, so the body walks into the opening before it turns into the seat.
pub fn enter_path(start: DVec3, target: DVec3, car: &CarFrame, side: f64, eased: f64) -> DVec3 {
    let chord = (target - start).length().max(1e-3);
    let m0 = car.dir(DVec3::Z) * chord;
    let m1 = car.dir(DVec3::new(-side, 0.0, 0.0)) * chord;
    board::hermite(
        Vec3d::from_dvec3(start),
        Vec3d::from_dvec3(m0),
        Vec3d::from_dvec3(target),
        Vec3d::from_dvec3(m1),
        eased,
    )
    .to_dvec3()
}

/// **The exit warp's path**, world — [`enter_path`] backwards: out of the seat
/// heading outward, arriving at the exit point heading aft.
pub fn exit_path(from: DVec3, to: DVec3, car: &CarFrame, side: f64, eased: f64) -> DVec3 {
    let chord = (to - from).length().max(1e-3);
    let m0 = car.dir(DVec3::new(side, 0.0, 0.0)) * chord;
    let m1 = car.dir(-DVec3::Z) * chord;
    board::hermite(
        Vec3d::from_dvec3(from),
        Vec3d::from_dvec3(m0),
        Vec3d::from_dvec3(to),
        Vec3d::from_dvec3(m1),
        eased,
    )
    .to_dvec3()
}

/// The pelvis dip the ground phases ask for this step, metres (negative is
/// down): ramped in with the hand, held while the latch is pulled, given back
/// as the body steps away from the door.
pub fn ground_pelvis_drop(b: &BoardingState) -> f64 {
    match b.phase {
        BoardPhase::OpeningDoor => {
            let ramp = (b.time_s / board::HAND_REACH_S).clamp(0.0, 1.0);
            let back = if b.mark_s >= 0.0 {
                ((b.time_s - b.mark_s) / board::STEP_BACK_S).clamp(0.0, 1.0)
            } else {
                0.0
            };
            -b.dip_m * ramp * (1.0 - back)
        }
        _ => 0.0,
    }
}

/// Where a mover whose placement a gate wants to read stands — the capsule
/// centre off the entity's transform.
pub fn capsule_centre(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    world
        .entity_of(guid)
        .and_then(|e| world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
}

/// The mode a body is in, for a gate or a HUD.
pub fn mode_of(world: &EcsWorld, guid: Uuid) -> Option<MovementMode> {
    world
        .entity_of(guid)
        .and_then(|e| world.world().get::<CharacterMovement>(e))
        .map(|cm| cm.mode)
}
