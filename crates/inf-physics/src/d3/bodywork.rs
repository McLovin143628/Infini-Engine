//! **WHAT HAPPENS TO A CAR'S BODYWORK** (wave VEH3c) — the half of
//! [`inf_ecs::bodywork`] that needs a world: the crash that tears a bumper off,
//! the hinge that swings a door, the pane that goes, the panel that dents, the
//! joules a round spends and the fire at the end of it.
//!
//! The `inf_physics` half of `inf_ecs::bodywork`, and the same split as
//! [`super::vehicle`] / `inf_ecs::vehicle`: the arithmetic — the impulse share,
//! the hinge's motor, the dent's depth — is on the other side of that wall and
//! is unit-tested without a world. Everything here touches rapier or the ECS.
//!
//! # It rides the VEHICLE phase, and therefore needs no fence of its own
//!
//! [`step_bodywork`] is called from the last statement of
//! [`super::vehicle::step_vehicles`], inside the `// MIRROR-BEGIN vehicle_step`
//! fence both hosts already carry and `fixed_step_mirror` already pins
//! character-for-character. That is deliberate and it is the cheapest correct
//! answer: a sibling function each host had to call would be a hand-maintained
//! mirror, which is the defect the vehicle phase's own module doc spends four
//! paragraphs on.
//!
//! # Before the solver, which is where a crash can be SEEN
//!
//! The vehicle phase runs before `bridge.step`, so the chassis velocity this
//! reads is the one the last solve left. A crash is therefore measured as the
//! velocity CHANGE across the previous step — `m·Δv`, less what gravity is
//! owed — which is the impulse the world actually delivered. One step of lag,
//! deterministic on both hosts, and it needs no contact-event drain.
//!
//! And the joint impulses a LIVE part's hinge is watched on are last step's for
//! the same reason, which is exactly what `joint_impulse` answers.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::bodywork::{
    damage_mut, damage_of, part_pose, DamageLimits, Debris, PartLatch, PartState, VehicleDamage,
    GLASS_SHARDS, GLASS_SHARD_LIFETIME_S, LATCH_POP_FRAC, MAX_DENT_M, MAX_GLASS_SHARDS,
    MAX_SHED_PARTS, PART_DEBRIS_LIFETIME_S,
};
use inf_ecs::components::{Collider3D, GlobalTransform, MeshRef, Sprite, Transform, Visibility};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::BodyPartKind;
use inf_ecs::EcsWorld;

use super::PhysicsBridge3D;

/// **The smallest blow this model calls a crash**, newton-seconds.
///
/// Three hundred. A 1 200 kg car changes speed by a quarter of a metre per
/// second in a step under hard braking, which is 300 N.s — so the floor is
/// exactly where "the driver did something" stops and "the car hit something"
/// begins, and a car being driven hard sheds nothing.
pub const CRASH_MIN_NS: f64 = 300.0;

/// **How fast a car has to have been going for a blow to be a crash**, m/s.
///
/// Two metres a second, which is walking pace. Under it a car is parking, being
/// nudged by a kerb, or being placed by something that is not the solver -- and
/// none of those should cost it a bumper. A real shunt at 15 km/h is four.
pub const CRASH_MIN_SPEED_MPS: f64 = 2.0;

/// **How hard a blow has to stop a car for it to be a crash**, m/s^2.
///
/// Twenty-five, which is two and a half g. No tyre delivers it: a road car
/// brakes at about 1.0 g and a racing slick at 1.5, so this floor is a property
/// of the physics rather than of the car's mass -- which an impulse floor alone
/// cannot be, because a five-tonne appliance braking normally puts more than a
/// thousand newton-seconds into a step and a saloon doing the same puts two
/// hundred.
pub const CRASH_MIN_DECEL_MPS2: f64 = 25.0;

/// **How far a chassis may drift from where its own velocity would have put it
/// before this model calls it PLACED**, metres.
///
/// A quarter of a metre. A rapier body integrates its own position from its own
/// velocity, so the discrepancy for a body the solver owns is the half-a-t-
/// squared of one step -- micrometres. Anything at this scale is somebody
/// writing a pose: the dispatcher's escort, a park, a tier respawn.
pub const TELEPORT_M: f64 = 0.25;

/// **The share of a crash's kinetic energy the HULL absorbs.**
///
/// A tenth. The rest goes into the thing it hit, into the tyres, into heat and
/// into the parts that came off. A 60 km/h wall crash carries about 165 kJ, so
/// the hull takes 16.5 kJ of a 36 kJ default — half a car in one big crash,
/// which is the reference footage's own answer and is why two of them burn it.
pub const HULL_CRASH_FRAC: f64 = 0.10;

/// **How much of a crash's energy reaches a PANE**, as a fraction of what the
/// part's own load path carries.
///
/// Glass is brittle and it is the first thing to go in a real shunt: a
/// windscreen in the impact face takes about a thousandth of a 60 km/h crash's
/// energy through its own mounts, which against a 120 J default is ten times
/// what it takes to break it.
pub const GLASS_CRASH_FRAC: f64 = 0.001;

/// **How hard a shed part is thrown**, as a fraction of the chassis's own
/// velocity.
///
/// One — it leaves with the car, and nothing else. There is deliberately no
/// kick: the WPN2d law is *size a blast against the LIGHTEST body in its
/// radius*, and a 2.6 kg glass shard given an impulse sized for a tonne and a
/// half of car is the levitating-ragdoll defect with a bumper in it.
pub const SHED_VELOCITY_FRAC: f64 = 1.0;

/// **How far below a shed part this model looks for the ground**, metres.
///
/// Eight. A panel that leaves a car on a bridge has a long way to fall and a
/// bounded look for it; past this it simply keeps falling until it is reaped,
/// which is what a part thrown off a cliff should do.
pub const DEBRIS_GROUND_REACH_M: f64 = 8.0;

/// **How fast a shed part tumbles**, degrees per second.
pub const DEBRIS_SPIN_DEG_S: f64 = 220.0;

/// The salt a glass shard's `Guid` is carved out of.
const SHARD_SALT: u128 = 0x5645_4833_4347_4c41_5353_5348_4152_4400;

/// **What one step of the bodywork did** — returned so a test can assert on the
/// decisions, while the arms that matter assert on the WORLD.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyworkReport {
    /// How many cars carry a damage row.
    pub cars: usize,
    /// How many took a blow over [`CRASH_MIN_NS`] this step.
    pub crashes: usize,
    /// The biggest blow any of them took, N.s.
    pub peak_impulse_ns: f64,
    /// How many latched parts popped onto a live hinge this step.
    pub popped: usize,
    /// How many parts left a car this step.
    pub shed: usize,
    /// How many panes went this step.
    pub panes: usize,
    /// How many cars caught fire this step.
    pub fires: usize,
    /// How many shed parts and shards were reaped this step.
    pub reaped: usize,
    /// How many shed parts are still lying about.
    pub debris: usize,
}

/// One part, as the WORLD describes it — the facts gathered before any borrow
/// is taken.
struct PartFacts {
    guid: Uuid,
    kind: BodyPartKind,
    centre_frac: Vec3d,
    half_frac: Vec3d,
    /// Whether its entity is a CHILD of the chassis right now. A part that was
    /// re-spawned latched by a tier change wins over whatever this module last
    /// believed about it.
    child: bool,
}

/// One car, as the world and the solver describe it.
struct CarFacts {
    chassis: Uuid,
    half: Vec3d,
    mass: f64,
    vel: DVec3,
    pos: DVec3,
    rot: DQuat,
    /// **Whether this chassis is TOUCHING anything solid** — the other half of
    /// what makes a blow a crash.
    touching: bool,
    /// **The three thresholds this row's own class sets.**
    ///
    /// Read off the chassis's own `VehicleClass` every step, and cheap enough to
    /// be: three field reads, not `to_tuning`'s hundred `set` calls. It was
    /// briefly LAZY and that was a defect with a measurement — a car whose row
    /// was quiet skipped it, so a crash on that step used the Ring-0 default
    /// 4 500 N.s instead of the row's own, and a car tuned UNBREAKABLE shed its
    /// bumper anyway.
    limits: DamageLimits,
    /// **Whether this car's row was quiet at the TOP of the step** --
    /// `VehicleDamage::is_quiet`, read once.
    ///
    /// Read ONCE, and that is a measurement rather than a tidy-up. `is_quiet`
    /// walks the row's whole `BTreeMap` of parts, and the first cut asked it
    /// twice a car a step -- once to decide whether to walk the children and
    /// once to decide whether to run the hinge pass. On a thousand parked
    /// saloons that is twenty-eight thousand B-tree node visits a step for an
    /// answer that cannot change between the two questions, and it cost
    /// **x1.097 against a x1.05 ceiling in release** -- the configuration the
    /// ceiling is actually asserted in. Asked once and carried here it is
    /// **x1.021**.
    ///
    /// A car that goes loud DURING the step is picked up by `loud_now`, not by
    /// this: a crash is the one thing that can falsify this flag after it is
    /// read, and the crash pass knows exactly which cars it touched.
    quiet: bool,
    parts: Vec<PartFacts>,
    wheels: Vec<(Vec3d, f64)>,
}

/// **Advance every car's bodywork one fixed step.**
///
/// Called by [`super::vehicle::step_vehicles`] as its last statement, so it
/// rides the `vehicle_step` MIRROR fence both hosts carry.
pub fn step_bodywork(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> BodyworkReport {
    let mut report = BodyworkReport::default();
    if !dt.is_finite() || dt <= 0.0 {
        return report;
    }
    let guids = bridge.vehicle_guids();
    if guids.is_empty() && damage_of(world).is_none() {
        return report;
    }
    // **THE BODYWORK'S OWN CLOCK.** Not `traffic::steps`, which is advanced by
    // the traffic pass and stands still on every level that has no traffic in
    // it — measured: a fixture's glass shards outlived their own second and a
    // half for ever, because nothing ever aged. A counter on this resource is
    // advanced by this function, which both hosts call once per fixed step.
    let step = {
        let res = damage_mut(world);
        res.steps = res.steps.wrapping_add(1);
        res.steps
    };

    // ── 1. what the world says ──────────────────────────────────────────────
    //
    // **THE CHEAP PATH IS THE POINT.** A car whose row is already known and
    // whose bodywork is untouched needs its velocity, its mass and its limits
    // and NOTHING else: no walk over its children, no string, no kind
    // recognition. That is what a thousand parked cars take, and without it
    // they cost 1.70x what a thousand cars with no parts at all cost.
    let cars: Vec<CarFacts> = guids
        .iter()
        .filter_map(|g| {
            let known = damage_of(world)
                .and_then(|r| r.rows.get(g))
                .map(|row| (!row.parts.is_empty(), row.is_quiet()));
            let walk = match known {
                Some((true, quiet)) => !quiet,
                _ => true,
            };
            let quiet = matches!(known, Some((_, true)));
            car_facts(world, bridge, *g, walk).map(|mut c| {
                c.quiet = quiet;
                c
            })
        })
        .collect();
    report.cars = cars.len();

    // ── 2. reconcile the table against it ───────────────────────────────────
    let live_chassis: BTreeSet<Uuid> = cars.iter().map(|c| c.chassis).collect();
    {
        let res = damage_mut(world);
        for car in &cars {
            let row = res.rows.entry(car.chassis).or_default();
            for p in &car.parts {
                // **WRITTEN ONCE, ON THE STEP THE PART IS FIRST SEEN.**
                //
                // The baseline is what the LEVEL says and it never changes. The
                // first cut re-wrote it from the entity's live `Transform` every
                // step, which is a feedback loop with a measurement: this walk
                // only runs for a car that is not quiet, so the moment a bonnet
                // was dented its "authored" box was re-read from the dented one
                // and dented again — a 39.6 mm dent closed a 186 mm panel to
                // **0.1 mm** in under two seconds, and it took the part's own
                // FACING axis with it.
                row.parts
                    .entry(p.guid)
                    .or_insert_with(|| PartState::authored(p.kind, p.centre_frac, p.half_frac));
                let _ = p.child;
            }
        }
    }

    // ── 3. the crash ────────────────────────────────────────────────────────
    let mut ignite: Vec<(Uuid, DVec3)> = Vec::new();
    let mut shatter: Vec<(Uuid, Uuid)> = Vec::new();
    // **The cars this step made loud.** `CarFacts::quiet` is read at the top of
    // the step and a crash is the one thing that can falsify it before the hinge
    // pass asks -- so the crash pass says so, rather than the hinge pass asking
    // the row a second time. Empty on every level where nobody hit anything,
    // which is the point.
    let mut loud_now: BTreeSet<Uuid> = BTreeSet::new();
    for car in &cars {
        let Some(row) = damage_mut(world).rows.get_mut(&car.chassis) else {
            continue;
        };
        let mass = car.mass.max(1e-6);
        let dv = if row.seen {
            car.vel - row.last_vel.to_dvec3()
        } else {
            DVec3::ZERO
        };
        // **WAS IT PLACED, OR DID IT DRIVE?** A rapier body integrates its own
        // position from its own velocity, so a body whose position did not
        // follow it was written by somebody. That is not a rare case: the
        // dispatcher's `escort` drags a lagging unit along its nav path and
        // zeroes both velocities, `park` puts one back on its apron, and a
        // traffic car crossing a tier boundary is despawned and respawned at the
        // same guid.
        let drift = if row.seen {
            (car.pos - row.last_pos.to_dvec3() - car.vel * dt).length()
        } else {
            0.0
        };
        let placed = drift > TELEPORT_M;
        row.last_vel = Vec3d::from_dvec3(car.vel);
        row.last_pos = Vec3d::from_dvec3(car.pos);
        row.seen = true;
        let j = mass * dv.length();
        if j.is_finite() && j > report.peak_impulse_ns {
            report.peak_impulse_ns = j;
        }
        // **A CRASH IS FOUR FACTS AT ONCE**, and every one of them was paid for.
        //
        // The first cut was `m*dv` less gravity and less the force the model
        // itself asked for, on the reasoning that what is left is what the world
        // delivered. That reasoning has a hole in it: the force a car's
        // suspension asks for is NOT the net force on it when the chassis is
        // also resting on something. The EMS fixture's van sits on its belly
        // (`size_the_suspension`'s own documented failure), so its struts asked
        // for **107 kN** on a 2.6 t body and the ground quietly cancelled it --
        // and the model read the difference as a **1 370 N.s blow every step for
        // four thousand steps**. On top of that the dispatcher zeroes an
        // escorted unit's velocities, so a unit pulling off its apron read
        // **14 020 N.s** from going 0 to 2.6 m/s in a single step. The cruiser
        // shed its bumper, the appliance popped its tailgate, and the joint that
        // then tried to hold a door onto a chassis being teleported along a nav
        // path launched an ambulance to **1 705 metres**.
        //
        // So the applied-force correction is gone and four plain facts stand in
        // its place. A crash is a blow that
        //
        // * is a real DECELERATION -- harder than any tyre can deliver
        //   ([`CRASH_MIN_DECEL_MPS2`]), which is what makes the floor a property
        //   of the physics rather than of the car's mass;
        // * is carried by a body that was MOVING ([`CRASH_MIN_SPEED_MPS`]);
        // * SLOWED IT DOWN -- an impulse that speeds a car up is somebody
        //   writing its state;
        // * and lands on a body that is TOUCHING something. A blow has to have
        //   something on the other end of it.
        //
        // …on a car that was not PLACED this step.
        let was = car.vel - dv;
        let decel = dv.length() / dt;
        if !j.is_finite()
            || placed
            || j < CRASH_MIN_NS
            || decel < CRASH_MIN_DECEL_MPS2
            || was.length() < CRASH_MIN_SPEED_MPS
            || car.vel.length() > was.length()
            || !car.touching
        {
            continue;
        }
        report.crashes += 1;
        loud_now.insert(car.chassis);
        let limits = car.limits;
        // The blow arrives along the direction the car was travelling, which is
        // the OPPOSITE of the velocity change a wall makes — and in the chassis
        // frame, because a part's own offset is.
        let dir_world = -dv.normalize_or_zero();
        let dir_local = car.rot.inverse() * dir_world;
        let dir = Vec3d::new(dir_local.x, dir_local.y, dir_local.z);
        // Energy: J = m|dv| and E = 1/2 m |dv|^2, so E = 1/2 J |dv|.
        let energy = 0.5 * j * dv.length();
        row.hull_j += energy * HULL_CRASH_FRAC;

        let parts: Vec<Uuid> = row.parts.keys().copied().collect();
        for guid in parts {
            let Some(p) = row.parts.get_mut(&guid) else {
                continue;
            };
            if p.latch == PartLatch::Shed || p.latch == PartLatch::Gone {
                continue;
            }
            let kind = kind_of_state(p);
            let face = BodyPartKind::face(p.centre_frac, dir);
            if face <= 0.0 {
                continue;
            }
            let share = j * face * kind.impact_share();
            if kind.dents() {
                p.dent_m =
                    (p.dent_m + share * 1e-3 * inf_ecs::bodywork::DENT_M_PER_KNS).min(MAX_DENT_M);
            }
            if kind == BodyPartKind::Glass {
                p.damage_j += energy * face * GLASS_CRASH_FRAC;
                if p.damage_j >= limits.glass_health_j {
                    shatter.push((car.chassis, guid));
                }
                continue;
            }
            if !kind.sheds() || limits.part_break_impulse_ns <= 0.0 {
                continue;
            }
            if share >= limits.part_break_impulse_ns {
                p.latch = PartLatch::Shed;
                p.shed_step = step;
                report.shed += 1;
            } else if share >= LATCH_POP_FRAC * limits.part_break_impulse_ns
                && kind.hinge().is_some()
                && p.latch == PartLatch::Latched
            {
                p.latch = PartLatch::Live;
                p.target_deg = kind.hinge().map(|h| h.open_deg).unwrap_or(0.0);
                report.popped += 1;
            }
        }
        // The crash pass has just written every part it reached.
        row.refresh_parts();
        if row.hull_j >= limits.hull_capacity_j() && row.fire_step == 0 {
            row.fire_step = step;
            ignite.push((car.chassis, car.pos));
        }
    }

    // ── 4. the hinges, and the drawn pose ───────────────────────────────────
    //
    // Skipped whole for a car nothing has happened to, which is the other half
    // of the cheap path: a latched hinge at zero with a target of zero has
    // nothing to integrate and nothing to write.
    for car in &cars {
        // Read once at the top of the step and corrected by the crash pass --
        // never asked of the row a second time. See `CarFacts::quiet` for the
        // x1.097-against-a-x1.05-ceiling measurement that decided it.
        if car.quiet && !loud_now.contains(&car.chassis) {
            continue;
        }
        let work: Vec<(Uuid, BodyPartKind, PartState)> = damage_of(world)
            .and_then(|r| r.rows.get(&car.chassis))
            .map(|r| {
                r.parts
                    .iter()
                    // **Latched AND Live.** A part that has been knocked open
                    // is still a child of the chassis and is still drawn by the
                    // same hinge — `Live` means *open*, not *elsewhere*.
                    .filter(|(_, s)| s.latch.attached())
                    .map(|(g, s)| (*g, kind_of_state(s), *s))
                    .collect()
            })
            .unwrap_or_default();
        for (guid, kind, state) in work {
            let open = kind.hinge().map(|h| h.open_deg).unwrap_or(0.0);
            let (angle, vel) = if open == 0.0 {
                (0.0, 0.0)
            } else {
                inf_ecs::bodywork::hinge_step(
                    state.angle_deg,
                    state.vel_deg_s,
                    state.target_deg,
                    open,
                    dt,
                )
            };
            if let Some(r) = damage_mut(world).rows.get_mut(&car.chassis) {
                if let Some(s) = r.parts.get_mut(&guid) {
                    s.angle_deg = angle;
                    s.vel_deg_s = vel;
                }
                // A hinge that has come home makes its part quiet again, which
                // is what lets a car whose door was opened and shut go back to
                // costing nothing.
                r.refresh_parts();
            }
            if state.dent_m == 0.0 && angle == 0.0 && state.angle_deg == 0.0 {
                continue;
            }
            let mut drawn = state;
            drawn.angle_deg = angle;
            let (t, r, sc) = part_pose(&drawn, kind, car.half, angle);
            if let Some(e) = world.entity_of(guid) {
                if let Some(mut tr) = world.world_mut().get_mut::<Transform>(e) {
                    tr.translation = t;
                    tr.rotation = r;
                    tr.scale = sc;
                }
            }
        }
    }

    // ── 5. the parts that have LEFT the car ─────────────────────────────────
    //
    // **A DETACHED PART IS DRAWN DEBRIS AND NOT A RAPIER BODY**, and that is a
    // ruling with three measurements behind it.
    //
    // The first cut gave a shed part its own dynamic body and gave a POPPED one
    // a body plus a real rapier revolute back to the chassis, watched for its
    // own break by `d3::joint::BreakWatch3D`. Every one of those is a new
    // physics object attached to, or lying in front of, a car that is still
    // being driven — and the EMS fixture measured what that costs. A bumper that
    // came off in front of a responding ambulance went under its own wheel rays
    // and stopped it getting home; a door on a hinge, held to a chassis the
    // dispatcher teleports along a nav path and zeroes the velocities of, was
    // yanked by its own joint until the **ambulance was at 1 705 metres**. Three
    // EMS gates that predate this wave went red, and the matrix says exactly
    // which half each of them was: two on the debris, one on the joint.
    //
    // **THE JOINT THIRD OF THAT DOES NOT REPRODUCE** (VEH3c audit).
    // `joints3d::a_jointed_rig_survives_being_teleported_as_a_unit` runs
    // `escort_nudge`'s own four writes for 240 steps with a 22 kg door on a real
    // revolute drawn half inside its own hull, six ways -- three placements x
    // the pair's contacts on and off -- and the chassis finishes **1.8 to 6.3 mm**
    // off its own schedule in every one of them, with the door never more than
    // 1.62 m from it. The peak the hinge carries goes 772.8 N.s naive with
    // contacts ON, 15.5 with them OFF, 3.0 moved as a unit, 3.0 with the joint
    // remade -- against a 4 500 N.s bumper mount. What the wave measured was the
    // SPURIOUS crash its own `applied_n` correction manufactured (see section 3
    // above), and that correction was removed in the same wave without the
    // refusal being re-measured. The two DEBRIS measurements stand; the joint
    // one does not, and the recipe for closing it is in that arm.
    //
    // So a part that leaves a car cannot interfere with a car. A SHED part is
    // reparented to the root, keeps the velocity it left with, falls under
    // gravity to the ground it was over and lies there until it is reaped — all
    // of it arithmetic in this function, deterministic on both hosts, and
    // invisible to the solver. A POPPED part stays a CHILD and swings open on
    // the analytic hinge it already had.
    //
    // What that costs is named rather than hidden: **a bumper in the road cannot
    // be run over and an open door cannot be torn off by a lamp post.** The
    // facade door those need — `joint_impulse` and `break_over_threshold`, with
    // their own arms in `joints3d.rs` — is built and is not called from here.
    for car in &cars {
        // **A quiet car has shed nothing**, so it does not pay for the question.
        // Without this the gather ran for every car every step -- a resource
        // lookup, a fourteen-node B-tree walk and a `Vec` allocation apiece --
        // to answer `Shed` fourteen times for a thousand parked saloons. Same
        // flag, same correction, as the hinge pass above.
        if car.quiet && !loud_now.contains(&car.chassis) {
            continue;
        }
        let detaching: Vec<(Uuid, BodyPartKind, PartState)> = damage_of(world)
            .and_then(|r| r.rows.get(&car.chassis))
            .map(|r| {
                r.parts
                    .iter()
                    .filter(|(_, s)| s.latch == PartLatch::Shed)
                    .map(|(g, s)| (*g, kind_of_state(s), *s))
                    .collect()
            })
            .unwrap_or_default();
        for (guid, kind, state) in detaching {
            shed_to_debris(world, bridge, car, guid, kind, &state, step);
        }
    }

    // ── 6. the debris falls ─────────────────────────────────────────────────
    report.debris = step_debris(world, dt);

    // ── 7. the glass ────────────────────────────────────────────────────────
    for (chassis, pane) in shatter {
        if shatter_pane(world, chassis, pane, step) {
            report.panes += 1;
        }
    }

    // ── 8. the fire ─────────────────────────────────────────────────────────
    for (chassis, at) in ignite {
        // The EMS2 door, unchanged: a burning car is an incident of kind `Fire`
        // whose `building` slot carries the CHASSIS' guid. That slot is read for
        // de-duplication and never dereferenced, which is why a car drops into
        // it with no schema move — and the brigade's whole chain (assign, route,
        // arrive, hose, resolve) works on it without knowing it is a car.
        if super::dispatch::report_incident(
            world,
            inf_ecs::dispatch::IncidentKind::Fire {
                building: chassis,
                intensity: 1.0,
            },
            at,
        )
        .is_some()
        {
            report.fires += 1;
        }
    }

    // ── 9. reap, and tell the models what has been done to them ─────────────
    report.reaped += reap(world, step, &live_chassis);
    for car in &cars {
        let Some(row) = damage_of(world).and_then(|r| r.rows.get(&car.chassis)) else {
            continue;
        };
        let (scale, flats) = (row.engine_scale(), row.flats);
        if let Some(v) = bridge.vehicle_mut(car.chassis) {
            v.set_damage(scale, flats);
        }
    }
    report
}

/// Gather one car's facts — the world's answer, before any borrow is taken.
///
/// `walk` decides whether the chassis's CHILDREN are visited. A car whose row
/// already knows its parts and whose bodywork is untouched answers `false` and
/// skips the walk entirely, which is what a thousand parked cars do sixty times
/// a second. The parts of a car do not change unless the car is respawned, and
/// a respawn mints the same content-derived guids.
fn car_facts(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    walk: bool,
) -> Option<CarFacts> {
    let entity = world.entity_of(chassis)?;
    let half = world.world().get::<Collider3D>(entity)?.half_extents;
    if half.x <= 0.0 || half.y <= 0.0 || half.z <= 0.0 {
        return None;
    }
    let body = bridge.body_of(chassis)?;
    let w = bridge.world();
    let pos = w.body_translation(body)?;
    let rot = w.body_rotation(body)?;
    let vel = w.body_linvel(body).unwrap_or(DVec3::ZERO);
    let mass = w.body_mass(body).unwrap_or(0.0);
    let touching = w.body_has_contact(body);
    let mut parts = Vec::new();
    if walk {
        for child in world.children_of(entity) {
            let Some(guid) = world.guid_of(child) else {
                continue;
            };
            // A part is something DRAWN. A wheel sensor and a thruster marker
            // carry no mesh of their own (their drawn child does), so this one
            // test partitions the chassis's children without a second
            // recogniser.
            if world.world().get::<MeshRef>(child).is_none() {
                continue;
            }
            let Some(name) = world.name_of(child) else {
                continue;
            };
            let t = world
                .world()
                .get::<Transform>(child)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            let centre_frac = Vec3d::new(
                t.translation.x / half.x,
                t.translation.y / half.y,
                t.translation.z / half.z,
            );
            let half_frac = Vec3d::new(
                t.scale.x / (2.0 * half.x),
                t.scale.y / (2.0 * half.y),
                t.scale.z / (2.0 * half.z),
            );
            parts.push(PartFacts {
                guid,
                kind: BodyPartKind::of(name, centre_frac, half_frac),
                centre_frac,
                half_frac,
                child: true,
            });
        }
    }
    // **The wheels, only when somebody is going to ask** — `hit_vehicle` needs
    // them to find which tyre a round punctured and nothing else does, and
    // `VehicleRig::wheels` is a `Vec` behind the trait, so gathering it for a
    // parked car is one heap allocation a car a step for an answer nobody wants.
    let wheels: Vec<(Vec3d, f64)> = if walk {
        bridge
            .vehicle_of(chassis)
            .map(|v| {
                v.rig()
                    .wheels
                    .iter()
                    .map(|w| (w.mount_local, w.radius_m))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    Some(CarFacts {
        chassis,
        half,
        mass,
        vel,
        pos,
        rot,
        limits: DamageLimits::of(world, chassis),
        touching,
        // Filled in by the caller, which is the only place that has already
        // asked the row the question -- see `CarFacts::quiet`.
        quiet: false,
        parts,
        wheels,
    })
}

/// The kind a stored state describes — its geometry re-read through the one
/// recogniser, so a detached part cannot acquire a second opinion about what it
/// is.
fn kind_of_state(s: &PartState) -> BodyPartKind {
    let name = match s.kind {
        inf_ecs::vehicle::KIND_DOOR => "door",
        inf_ecs::vehicle::KIND_HOOD => "hood",
        inf_ecs::vehicle::KIND_TRUNK => "trunk",
        inf_ecs::vehicle::KIND_BUMPER => "bumper",
        inf_ecs::vehicle::KIND_GLASS => "glass",
        _ => "panel",
    };
    BodyPartKind::of(name, s.centre_frac, s.half_frac)
}

/// **Take a part off the car and make it debris** (wave VEH3c) — the one door a
/// shed part goes through.
///
/// Idempotent: a part already off the hierarchy is left alone, so this is safe
/// to call every step for as long as the part exists.
fn shed_to_debris(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    car: &CarFacts,
    guid: Uuid,
    kind: BodyPartKind,
    state: &PartState,
    step: u64,
) {
    let Some(entity) = world.entity_of(guid) else {
        return;
    };
    if world.parent_of(entity).is_none() {
        return;
    }
    // Where it is, in the world, at the moment it lets go.
    let (local_t, local_r, scale) = part_pose(state, kind, car.half, state.angle_deg);
    let rot = car.rot
        * Transform {
            translation: Vec3d::ZERO,
            rotation: local_r,
            scale: Vec3d::ONE,
        }
        .quat();
    let half = Vec3d::new(
        (scale.x * 0.5).abs().max(0.01),
        (scale.y * 0.5).abs().max(0.01),
        (scale.z * 0.5).abs().max(0.01),
    );
    // **PUSHED CLEAR OF THE CAR IT CAME OFF**, along the axis it faces — a
    // bumper forwards, a door sideways, a boot lid backwards. The direction is
    // derived from the part's own offset and not authored.
    let (axis, sign) = state.facing();
    let out = [half.x, half.y, half.z][axis] + 0.12;
    let mut local_out = local_t;
    match axis {
        0 => local_out.x += sign * out,
        1 => local_out.y += sign * out,
        _ => local_out.z += sign * out,
    }
    let at = car.pos + car.rot * local_out.to_dvec3();
    // **The ground it will lie on**, one ray, once, at the moment it sheds. A
    // falling panel needs somewhere to stop and this is the cheapest honest
    // answer: everything solid, straight down, from a metre above where it left.
    //
    // **The car it came off is excluded**, and that is not a nicety: the ray
    // starts a metre over a panel that is still touching its own chassis, so
    // without the exclusion it hits the car and a bumper "rests" a metre ABOVE
    // where it let go. Measured: a rest of 1.721 m for a part at 0.621.
    let mut exclude: BTreeSet<super::ColliderId3D> = BTreeSet::new();
    if let Some(c) = bridge.collider_of(car.chassis) {
        exclude.insert(c);
    }
    let rest_y = bridge
        .world_mut()
        .cast_ray_where(
            at + DVec3::Y,
            -DVec3::Y,
            DEBRIS_GROUND_REACH_M,
            &exclude,
            super::CastTargets::AllSolid,
        )
        .map(|h| at.y + 1.0 - h.toi + half.y)
        .unwrap_or(at.y - DEBRIS_GROUND_REACH_M)
        // …and it can never be above where the part let go: a panel falls.
        .min(at.y);
    world.reparent(entity, None);
    let mut t = Transform {
        translation: Vec3d::from_dvec3(at),
        rotation: Vec3d::ZERO,
        scale,
    };
    t.set_quat(rot);
    if let Some(mut tr) = world.world_mut().get_mut::<Transform>(entity) {
        *tr = t;
    }
    damage_mut(world).shed.push(Debris {
        guid,
        born: step,
        at: Vec3d::from_dvec3(at),
        vel: Vec3d::from_dvec3(car.vel * SHED_VELOCITY_FRAC),
        rest_y,
        // A tumble, derived from which way it left rather than drawn from
        // anything: a part that went sideways rolls, one that went forward
        // pitches. No RNG reaches a fixed step.
        spin_deg_s: match axis {
            0 => Vec3d::new(0.0, 0.0, -sign * DEBRIS_SPIN_DEG_S),
            1 => Vec3d::new(DEBRIS_SPIN_DEG_S, 0.0, 0.0),
            _ => Vec3d::new(sign * DEBRIS_SPIN_DEG_S, 0.0, 0.0),
        },
    });
    world.mark_dirty();
}

/// **Advance every piece of debris**, and answer how many are still lying about.
///
/// Ballistic and then still: `at += v·dt`, `v.y -= g·dt`, and when it reaches the
/// ground it was over it stops there and stays. Pure arithmetic on state this
/// module owns, so two hosts agree by construction and the solver never sees it.
fn step_debris(world: &mut EcsWorld, dt: f64) -> usize {
    let Some(res) = damage_of(world) else {
        return 0;
    };
    if res.shed.is_empty() {
        return 0;
    }
    let mut moved: Vec<(Uuid, Vec3d, Vec3d)> = Vec::with_capacity(res.shed.len());
    let mut next: Vec<Debris> = Vec::with_capacity(res.shed.len());
    for d in &res.shed {
        let mut d = *d;
        if d.at.y > d.rest_y {
            d.vel.y -= 9.81 * dt;
            d.at = Vec3d::new(
                d.at.x + d.vel.x * dt,
                d.at.y + d.vel.y * dt,
                d.at.z + d.vel.z * dt,
            );
            if d.at.y <= d.rest_y {
                d.at.y = d.rest_y;
                d.vel = Vec3d::ZERO;
            }
            let spin = Vec3d::new(
                d.spin_deg_s.x * dt,
                d.spin_deg_s.y * dt,
                d.spin_deg_s.z * dt,
            );
            moved.push((d.guid, d.at, spin));
        }
        next.push(d);
    }
    let n = next.len();
    damage_mut(world).shed = next;
    for (guid, at, spin) in moved {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
            t.translation = at;
            t.rotation = Vec3d::new(
                t.rotation.x + spin.x,
                t.rotation.y + spin.y,
                t.rotation.z + spin.z,
            );
        }
    }
    if n > 0 {
        world.mark_dirty();
    }
    n
}

/// **Shatter one pane**: hide it, and throw a handful of shards.
fn shatter_pane(world: &mut EcsWorld, chassis: Uuid, pane: Uuid, step: u64) -> bool {
    {
        let Some(r) = damage_mut(world).rows.get_mut(&chassis) else {
            return false;
        };
        let Some(s) = r.parts.get_mut(&pane) else {
            return false;
        };
        if s.latch == PartLatch::Gone {
            return false;
        }
        s.latch = PartLatch::Gone;
        r.refresh_parts();
    }
    let Some(entity) = world.entity_of(pane) else {
        return false;
    };
    let at = world
        .world()
        .get::<GlobalTransform>(entity)
        .map(|g| g.0.translation)
        .unwrap_or(DVec3::ZERO);
    world
        .world_mut()
        .entity_mut(entity)
        .insert(Visibility { visible: false });
    // **The shipped option, of the two the wave priced**: the pane is hidden and
    // a handful of `Sprite` shards is thrown. See the report for what the P22
    // fracture door would have cost instead.
    for i in 0..GLASS_SHARDS {
        let guid = shard_guid(pane, step, i);
        if world.entity_of(guid).is_some() {
            continue;
        }
        if damage_of(world).map(|r| r.shards.len()).unwrap_or(0) >= MAX_GLASS_SHARDS {
            break;
        }
        // A deterministic fan: no RNG, no clock, a pure function of the index.
        let a = (i as f64) * std::f64::consts::TAU / (GLASS_SHARDS as f64);
        let off = DVec3::new(inf_math::pcos64(a) * 0.18, 0.05, inf_math::psin64(a) * 0.18);
        let e = world.spawn_with_guid(guid, "Glass Shard", None);
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform {
                translation: Vec3d::from_dvec3(at + off),
                ..Transform::IDENTITY
            })
            .insert(Sprite {
                size: inf_ecs::math::Vec2d::new(0.09, 0.09),
                color: inf_ecs::math::Color::new(0.72, 0.82, 0.88, 0.85),
                billboard: inf_ecs::components::BillboardMode::Cylindrical,
                ..Default::default()
            })
            .insert(Visibility::default());
        damage_mut(world).shards.push((guid, step));
    }
    world.mark_dirty();
    true
}

/// A glass shard's `Guid` — content-derived, so two hosts throwing the same
/// shard throw the same entity.
fn shard_guid(pane: Uuid, step: u64, index: usize) -> Uuid {
    let mut x = SHARD_SALT ^ pane.as_u128();
    x = x.rotate_left(13) ^ ((step as u128) << 8 | index as u128);
    x = x.rotate_left(29) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// **Take away what has been lying about too long** — the shed parts past
/// [`PART_DEBRIS_LIFETIME_S`] or over [`MAX_SHED_PARTS`], the shards past their
/// own second and a half, and anything whose car has gone.
fn reap(world: &mut EcsWorld, step: u64, live: &BTreeSet<Uuid>) -> usize {
    let life_steps = (PART_DEBRIS_LIFETIME_S * 60.0) as u64;
    let shard_steps = (GLASS_SHARD_LIFETIME_S * 60.0) as u64;
    let (mut kill_parts, mut kill_shards): (Vec<Uuid>, Vec<Uuid>) = (Vec::new(), Vec::new());
    let mut orphan_rows: Vec<Uuid> = Vec::new();
    {
        let Some(res) = damage_of(world) else {
            return 0;
        };
        let over = res.shed.len().saturating_sub(MAX_SHED_PARTS);
        for (i, d) in res.shed.iter().enumerate() {
            if i < over || step.saturating_sub(d.born) > life_steps {
                kill_parts.push(d.guid);
            }
        }
        for (guid, born) in &res.shards {
            if step.saturating_sub(*born) > shard_steps {
                kill_shards.push(*guid);
            }
        }
        // A car that has left the level takes its parts with it, and a row for a
        // car nobody can reach is a leak with no deadline.
        for chassis in res.rows.keys() {
            if !live.contains(chassis) && world.entity_of(*chassis).is_none() {
                orphan_rows.push(*chassis);
            }
        }
    }
    let mut n = 0;
    for guid in kill_parts.iter().chain(kill_shards.iter()) {
        if let Some(e) = world.entity_of(*guid) {
            world.despawn(e);
            n += 1;
        }
    }
    for chassis in &orphan_rows {
        if let Some(row) = damage_of(world).and_then(|r| r.rows.get(chassis)) {
            let parts: Vec<Uuid> = row.parts.keys().copied().collect();
            for p in parts {
                if let Some(e) = world.entity_of(p) {
                    if world.parent_of(e).is_none() {
                        world.despawn(e);
                        n += 1;
                    }
                }
            }
        }
    }
    {
        let res = damage_mut(world);
        res.shed.retain(|d| !kill_parts.contains(&d.guid));
        res.shards.retain(|(g, _)| !kill_shards.contains(g));
        for chassis in orphan_rows {
            res.rows.remove(&chassis);
        }
    }
    if n > 0 {
        world.mark_dirty();
    }
    n
}

// ── what a round does to a car ──────────────────────────────────────────────

/// **What one hit on a vehicle spent, and on what** — returned so `apply_hit`
/// can report it and a gate can read it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VehicleHit {
    /// The joules that reached the hull.
    pub hull_j: f64,
    /// The joules that reached the engine.
    pub engine_j: f64,
    /// The pane this hit shattered, if it did.
    pub pane: Option<Uuid>,
    /// The wheel this hit flattened, if it did.
    pub flat: Option<usize>,
    /// Whether this hit set the car alight.
    pub ignited: bool,
}

/// **Spend a round's joules on a car** (wave VEH3c) — `is_flesh`'s complement.
///
/// Before this wave a round that stopped in a car spent NOTHING: `apply_hit`
/// gave a body only to something with `CharacterMovement`, a chassis has no
/// `Destructible`, and the WPN2a audit measured the whole of it (*"a car spends
/// nothing -- no `Health`, 0 entries at the P22 door"*). It spends now.
///
/// `at` is the impact point in world metres — the round's own `WeaponHit::to` —
/// and where on the car it landed decides where the joules go: a pane within
/// [`inf_ecs::bodywork::PANE_HIT_M`] takes them as glass, a wheel within
/// [`inf_ecs::bodywork::FLAT_HIT_RADII`] of its own radius takes them as a
/// puncture, the front [`inf_ecs::bodywork::ENGINE_BAY_FRAC`] of the car takes
/// them as engine damage, and everything takes them as hull.
pub fn hit_vehicle(
    world: &mut EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    at: DVec3,
    energy_j: f64,
) -> Option<VehicleHit> {
    if !energy_j.is_finite() || energy_j <= 0.0 {
        return None;
    }
    // **IS IT A CAR AT ALL**, and this is the whole reason `is_vehicle` exists.
    //
    // Without this line `car_facts` answers for anything with a `Collider3D` and
    // a body in the bridge, which is a WALL — and a round into a wall was then
    // spent on a "vehicle" that is a lamp post, so it never reached the P22
    // destructible door below. Measured the hard way: five arms across four
    // files went red at once (`weapon_3d`'s wall and its no-health character,
    // `phase30_gameplay_gate`'s rifle round, `wpn2d_gate`'s blast), every one of
    // them about joules arriving at the destructible door and finding nobody
    // there. The bridge's vehicle map is the ONE answer to "is this a car" —
    // the same door the E-key prompt and WPN2d's lock both ask.
    if !is_vehicle(bridge, chassis) {
        return None;
    }
    let car = car_facts(world, bridge, chassis, true)?;
    let limits = car.limits;
    let step = now(world);
    let local = car.rot.inverse() * (at - car.pos);
    let mut out = VehicleHit::default();

    // A WHEEL first: it is the smallest target and the most specific answer.
    let mut best: Option<(usize, f64)> = None;
    for (i, (mount, radius)) in car.wheels.iter().enumerate() {
        let d = (local - mount.to_dvec3()).length();
        if d <= radius * inf_ecs::bodywork::FLAT_HIT_RADII && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    if let Some((i, _)) = best {
        let row = damage_mut(world).rows.entry(chassis).or_default();
        if row.flatten(i) {
            out.flat = Some(i);
            return Some(out);
        }
    }

    // …then a PANE. The round stopped on the chassis's outer box, so the pane is
    // a few centimetres inside the impact point.
    let mut pane: Option<(Uuid, f64)> = None;
    for p in &car.parts {
        if p.kind != BodyPartKind::Glass {
            continue;
        }
        let centre = DVec3::new(
            p.centre_frac.x * car.half.x,
            p.centre_frac.y * car.half.y,
            p.centre_frac.z * car.half.z,
        );
        let d = (local - centre).length();
        if d <= inf_ecs::bodywork::PANE_HIT_M && pane.is_none_or(|(_, bd)| d < bd) {
            pane = Some((p.guid, d));
        }
    }
    if let Some((guid, _)) = pane {
        let broke = {
            let row = damage_mut(world).rows.entry(chassis).or_default();
            let slot = row.parts.entry(guid).or_default();
            slot.damage_j += energy_j;
            let broke = slot.damage_j >= limits.glass_health_j && slot.latch != PartLatch::Gone;
            row.refresh_parts();
            broke
        };
        if broke && shatter_pane(world, chassis, guid, step) {
            out.pane = Some(guid);
            return Some(out);
        }
        if pane.is_some() {
            return Some(out);
        }
    }

    // …then the ENGINE BAY, and the hull for everything else.
    let nose = car.half.z * (1.0 - 2.0 * inf_ecs::bodywork::ENGINE_BAY_FRAC);
    let in_bay = local.z >= nose;
    let mut ignited = false;
    {
        let row = damage_mut(world).rows.entry(chassis).or_default();
        row.hull_j += energy_j;
        out.hull_j = energy_j;
        if in_bay {
            let share = energy_j / limits.engine_capacity_j().max(1.0);
            row.engine_damage = (row.engine_damage + share).clamp(0.0, 1.0);
            out.engine_j = energy_j;
        }
        if row.hull_j >= limits.hull_capacity_j() && row.fire_step == 0 {
            row.fire_step = step;
            ignited = true;
        }
    }
    if ignited {
        out.ignited = super::dispatch::report_incident(
            world,
            inf_ecs::dispatch::IncidentKind::Fire {
                building: chassis,
                intensity: 1.0,
            },
            car.pos,
        )
        .is_some();
    }
    Some(out)
}

/// **What step it is, by the bodywork's own clock** — never `traffic::steps`,
/// which stands still on a level with no traffic in it.
///
/// Floored at 1, so a fire filed before the bodywork has ever stepped still
/// reads as burning: `fire_step` of `0` means *not on fire*, and a clock that
/// could answer zero would make the two the same.
fn now(world: &EcsWorld) -> u64 {
    damage_of(world).map(|r| r.steps).unwrap_or(0).max(1)
}

/// **Is this guid a vehicle chassis the bodywork can spend on?** — the one test
/// `apply_hit` asks, so a round cannot find a second answer.
pub fn is_vehicle(bridge: &PhysicsBridge3D, guid: Uuid) -> bool {
    bridge.vehicle_of(guid).is_some()
}

/// **Open or shut one part** (wave VEH3c) — the door VEH3d's boarding pipeline
/// will drive.
///
/// **Nothing calls it outside `veh3c_gate`**, and that is deliberate: an input
/// path to a car door is VEH3d's and this wave does not build it early. The
/// first draft of this line said *"and the one the demo's own hero key
/// reaches"*, which is not true of any key in the shipped input map and
/// contradicts the wave's own carried item 6 — caught by the VEH3c audit.
///
/// `open` drives the hinge to its own limit; `!open` drives it shut. Answers
/// whether the part exists and has a hinge at all.
pub fn set_part_open(world: &mut EcsWorld, chassis: Uuid, part: Uuid, open: bool) -> bool {
    let Some(row) = damage_of(world).and_then(|r| r.rows.get(&chassis)) else {
        return false;
    };
    let Some(state) = row.parts.get(&part).copied() else {
        return false;
    };
    let kind = kind_of_state(&state);
    let Some(hinge) = kind.hinge() else {
        return false;
    };
    if let Some(r) = damage_mut(world).rows.get_mut(&chassis) {
        if let Some(s) = r.parts.get_mut(&part) {
            s.target_deg = if open { hinge.open_deg } else { 0.0 };
        }
        r.refresh_parts();
    }
    true
}

/// **Every part of one car, in `Guid` order** — the census a gate, a HUD and
/// VEH3d's handle-finder all read.
pub fn parts_of(world: &EcsWorld, chassis: Uuid) -> Vec<(Uuid, PartState)> {
    damage_of(world)
        .and_then(|r| r.rows.get(&chassis))
        .map(|r| r.parts.iter().map(|(g, s)| (*g, *s)).collect())
        .unwrap_or_default()
}

/// One car's damage, or a whole one.
pub fn damage_or_default(world: &EcsWorld, chassis: Uuid) -> VehicleDamage {
    damage_of(world)
        .and_then(|r| r.rows.get(&chassis))
        .cloned()
        .unwrap_or_default()
}

/// The parts still bolted to a chassis — the count a crash arm reads before and
/// after.
pub fn attached_count(world: &EcsWorld, chassis: Uuid) -> usize {
    parts_of(world, chassis)
        .into_iter()
        .filter(|(_, s)| s.latch.attached())
        .count()
}

/// How many `BTreeMap` rows the table holds — a cheap handle on "does this level
/// have any bodywork state at all".
pub fn row_count(world: &EcsWorld) -> usize {
    damage_of(world).map(|r| r.rows.len()).unwrap_or(0)
}
