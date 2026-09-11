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
    damage_mut, damage_of, part_pose, DamageLimits, PartLatch, PartState, VehicleDamage,
    GLASS_SHARDS, GLASS_SHARD_LIFETIME_S, LATCH_POP_FRAC, MAX_DENT_M, MAX_GLASS_SHARDS,
    MAX_SHED_PARTS, PART_DEBRIS_LIFETIME_S,
};
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, GlobalTransform, Joint3D,
    JointKind3D as SceneJointKind3D, MeshRef, RigidBody3D, Sprite, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::BodyPartKind;
use inf_ecs::EcsWorld;

use super::{BreakWatch3D, PhysicsBridge3D};

/// **The smallest blow this model calls a crash**, newton-seconds.
///
/// Three hundred. A 1 200 kg car changes speed by a quarter of a metre per
/// second in a step under hard braking, which is 300 N.s — so the floor is
/// exactly where "the driver did something" stops and "the car hit something"
/// begins, and a car being driven hard sheds nothing.
pub const CRASH_MIN_NS: f64 = 300.0;

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
    /// How many live hinges are being watched for their own break.
    pub watched: usize,
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
    /// **The three thresholds this row's own class sets.**
    ///
    /// Read off the chassis's own `VehicleClass` every step, and cheap enough to
    /// be: three field reads, not `to_tuning`'s hundred `set` calls. It was
    /// briefly LAZY and that was a defect with a measurement — a car whose row
    /// was quiet skipped it, so a crash on that step used the Ring-0 default
    /// 4 500 N.s instead of the row's own, and a car tuned UNBREAKABLE shed its
    /// bumper anyway.
    limits: DamageLimits,
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
    outcomes: &[super::vehicle::VehicleOutcome],
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
            car_facts(world, bridge, *g, walk)
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
                let slot = row
                    .parts
                    .entry(p.guid)
                    .or_insert_with(|| PartState::authored(p.kind, p.centre_frac, p.half_frac));
                slot.kind = p.kind.as_u8();
                slot.centre_frac = p.centre_frac;
                slot.half_frac = p.half_frac;
                // THE WORLD WINS. A traffic car that crossed a tier boundary was
                // despawned and respawned whole, so a part this table believed
                // was hanging off its hinge is a fresh child again — and a row
                // that went on believing otherwise would try to tear off a door
                // that is already bolted on.
                if p.child && slot.latch == PartLatch::Live {
                    slot.latch = PartLatch::Latched;
                    slot.angle_deg = 0.0;
                    slot.vel_deg_s = 0.0;
                    slot.target_deg = 0.0;
                }
            }
        }
    }

    // ── 3. the crash ────────────────────────────────────────────────────────
    let mut ignite: Vec<(Uuid, DVec3)> = Vec::new();
    let mut shatter: Vec<(Uuid, Uuid)> = Vec::new();
    for car in &cars {
        // What this car's own model asked the solver for LAST step — the force
        // whose work is not a crash. Absent for a car the phase did not step,
        // which is a car with no wheels.
        let applied = outcomes
            .iter()
            .find(|o| o.chassis == car.chassis)
            .map(|o| o.applied_n)
            .unwrap_or(DVec3::ZERO);
        let Some(row) = damage_mut(world).rows.get_mut(&car.chassis) else {
            continue;
        };
        let mass = car.mass.max(1e-6);
        // The velocity change the WORLD delivered: what actually happened, less
        // gravity and less what this car's own suspension, tyres and aero asked
        // for on the step it happened.
        let owed = (row.last_force.to_dvec3() / mass + DVec3::new(0.0, -9.81, 0.0)) * dt;
        let dv = if row.seen {
            car.vel - row.last_vel.to_dvec3() - owed
        } else {
            DVec3::ZERO
        };
        row.last_vel = Vec3d::from_dvec3(car.vel);
        row.last_force = Vec3d::from_dvec3(applied);
        row.seen = true;
        let j = mass * dv.length();
        if j.is_finite() && j > report.peak_impulse_ns {
            report.peak_impulse_ns = j;
        }
        if !j.is_finite() || j < CRASH_MIN_NS {
            continue;
        }
        report.crashes += 1;
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
        let quiet = damage_of(world)
            .and_then(|r| r.rows.get(&car.chassis))
            .map(|r| r.is_quiet())
            .unwrap_or(true);
        if quiet {
            continue;
        }
        let work: Vec<(Uuid, BodyPartKind, PartState)> = damage_of(world)
            .and_then(|r| r.rows.get(&car.chassis))
            .map(|r| {
                r.parts
                    .iter()
                    .filter(|(_, s)| s.latch == PartLatch::Latched)
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
            if let Some(s) = damage_mut(world)
                .rows
                .get_mut(&car.chassis)
                .and_then(|r| r.parts.get_mut(&guid))
            {
                s.angle_deg = angle;
                s.vel_deg_s = vel;
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

    // ── 5. the parts that are no longer children ────────────────────────────
    let mut watches: Vec<BreakWatch3D> = Vec::new();
    let mut breaks: Vec<(Uuid, Uuid)> = Vec::new();
    for car in &cars {
        let limits = car.limits;
        let detaching: Vec<(Uuid, BodyPartKind, PartState)> = damage_of(world)
            .and_then(|r| r.rows.get(&car.chassis))
            .map(|r| {
                r.parts
                    .iter()
                    .filter(|(_, s)| s.latch == PartLatch::Live || s.latch == PartLatch::Shed)
                    .map(|(g, s)| (*g, kind_of_state(s), *s))
                    .collect()
            })
            .unwrap_or_default();
        for (guid, kind, state) in detaching {
            let hinged = state.latch == PartLatch::Live;
            detach(world, car, guid, kind, &state, hinged, step);
            if hinged {
                if let Some(j) = bridge.joint_of(guid) {
                    watches.push(BreakWatch3D {
                        joint: j,
                        threshold_ns: limits.part_break_impulse_ns,
                    });
                    breaks.push((car.chassis, guid));
                }
            } else if state.shed_step + 2 >= step {
                // It leaves with the car and no faster — the WPN2d law.
                if let Some(b) = bridge.body_of(guid) {
                    bridge
                        .world_mut()
                        .set_body_linvel(b, car.vel * SHED_VELOCITY_FRAC);
                }
            }
        }
    }
    report.watched = watches.len();

    // ── 6. the breaks ───────────────────────────────────────────────────────
    if !watches.is_empty() {
        let broken = bridge.world_mut().break_over_threshold(&watches);
        for b in broken {
            let Some(idx) = watches.iter().position(|w| w.joint == b.joint) else {
                continue;
            };
            let (chassis, part) = breaks[idx];
            if let Some(e) = world.entity_of(part) {
                world.world_mut().entity_mut(e).remove::<Joint3D>();
                // A shed part is SOLID: it has to land on the road rather than
                // pass through it, where a swinging one is deliberately not (see
                // `detach`).
                if let Some(mut c) = world.world_mut().get_mut::<Collider3D>(e) {
                    c.sensor = false;
                }
            }
            if let Some(s) = damage_mut(world)
                .rows
                .get_mut(&chassis)
                .and_then(|r| r.parts.get_mut(&part))
            {
                s.latch = PartLatch::Shed;
                s.shed_step = step;
            }
            damage_mut(world).shed.push((part, step));
            report.shed += 1;
        }
    }

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
    let rig = bridge.vehicle_of(chassis).map(|v| v.rig().clone());
    let wheels = rig
        .map(|r| {
            r.wheels
                .iter()
                .map(|w| (w.mount_local, w.radius_m))
                .collect()
        })
        .unwrap_or_default();
    Some(CarFacts {
        chassis,
        half,
        mass,
        vel,
        pos,
        rot,
        limits: DamageLimits::of(world, chassis),
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

/// **Take a part off the hierarchy and give it a body** — the one door both the
/// LIVE and the SHED regimes go through.
///
/// Idempotent: a part that already has a body is left alone, so this is safe to
/// call every step for as long as the part exists.
fn detach(
    world: &mut EcsWorld,
    car: &CarFacts,
    guid: Uuid,
    kind: BodyPartKind,
    state: &PartState,
    hinged: bool,
    step: u64,
) {
    let Some(entity) = world.entity_of(guid) else {
        return;
    };
    if world.world().get::<RigidBody3D>(entity).is_some() {
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
    // **A SHED PART IS SPAWNED CLEAR OF THE CAR IT CAME OFF.**
    //
    // A part is drawn ON the chassis's own outer face, so half its box is
    // INSIDE the chassis collider — and two overlapping dynamic bodies are a
    // depenetration force with nowhere to go. Measured before this offset
    // existed: a 7 kg bumper that came off at 60 km/h **rose** from 0.501 m to
    // 1.328 m in the second after it let go, which is the P29.6 ragdoll
    // launch wearing a bumper.
    //
    // So it is pushed out along the axis it FACES by its own half-thickness and
    // a hand's width more. The direction is derived, not authored: a door goes
    // sideways, a bumper forwards, a boot lid backwards.
    let (axis, sign) = state.facing();
    let out = [half.x, half.y, half.z][axis] + 0.12;
    let mut local_out = local_t;
    match axis {
        0 => local_out.x += sign * out,
        1 => local_out.y += sign * out,
        _ => local_out.z += sign * out,
    }
    let at = car.pos + car.rot * if hinged { local_t } else { local_out }.to_dvec3();
    let mass = inf_ecs::vehicle::part_mass_kg(kind, state.half_frac, car.half).max(0.5);
    let volume = 8.0 * half.x * half.y * half.z;
    world.reparent(entity, None);
    let mut t = Transform {
        translation: Vec3d::from_dvec3(at),
        rotation: Vec3d::ZERO,
        scale,
    };
    t.set_quat(rot);
    let mut em = world.world_mut().entity_mut(entity);
    em.insert(t);
    em.insert(RigidBody3D {
        kind: BodyKind3D::Dynamic,
        angular_damping: 0.6,
        ..Default::default()
    });
    em.insert(Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: half,
        // **A SWINGING part is a SENSOR and a SHED one is solid**, and that is
        // the same depenetration finding from the other end. A door on its
        // hinge is held INSIDE the car's own collider by the joint that holds
        // it, so a solid one would fight the chassis every step for as long as
        // it hung there; the facade's `Joint3D` has no `contacts` flag to turn
        // off (retired at P29, disposition row 12), so the collider is the door
        // that is available. What it costs is named: **an open door does not
        // collide with the world** — it swings through a lamp post — until a
        // wave gives `Joint3D` its flag back. A shed part is solid and lands on
        // the road, which is the half that matters for a frame.
        sensor: hinged,
        // `Collider3D::density` is rapier's own mass-per-volume and not a
        // material density (the P20.2 finding): a 0.4 m wheel at the default
        // weighs 268 grams. So it is derived from the mass the PART is worth,
        // and `a_shed_part_weighs_what_a_part_weighs` is what keeps that honest.
        density: (mass / volume.max(1e-6)).clamp(1.0, 20_000.0),
        friction: 0.7,
        ..Default::default()
    });
    if hinged {
        if let Some(hinge) = kind.hinge() {
            let pivot = Vec3d::new(
                hinge.at.x * car.half.x,
                hinge.at.y * car.half.y,
                hinge.at.z * car.half.z,
            );
            let (lo, hi) = if hinge.open_deg < 0.0 {
                (hinge.open_deg, 0.0)
            } else {
                (0.0, hinge.open_deg)
            };
            em.insert(Joint3D {
                other: inf_ecs::refs::EntityRef::new(car.chassis),
                kind: SceneJointKind3D::Revolute,
                local_anchor: Vec3d::new(
                    pivot.x - local_t.x,
                    pivot.y - local_t.y,
                    pivot.z - local_t.z,
                ),
                other_anchor: pivot,
                axis: hinge.axis,
                limits_enabled: true,
                limit_min: lo.to_radians(),
                limit_max: hi.to_radians(),
                motor_enabled: true,
                motor_target_pos: state.target_deg.to_radians(),
                motor_target_vel: 0.0,
                motor_stiffness: inf_ecs::bodywork::HINGE_STIFFNESS,
                motor_damping: inf_ecs::bodywork::HINGE_DAMPING,
                motor_max_force: 1_200.0,
                ..Default::default()
            });
        }
    } else {
        damage_mut(world).shed.push((guid, step));
    }
    world.mark_dirty();
}

/// **Shatter one pane**: hide it, and throw a handful of shards.
fn shatter_pane(world: &mut EcsWorld, chassis: Uuid, pane: Uuid, step: u64) -> bool {
    {
        let Some(s) = damage_mut(world)
            .rows
            .get_mut(&chassis)
            .and_then(|r| r.parts.get_mut(&pane))
        else {
            return false;
        };
        if s.latch == PartLatch::Gone {
            return false;
        }
        s.latch = PartLatch::Gone;
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
        for (i, (guid, born)) in res.shed.iter().enumerate() {
            if i < over || step.saturating_sub(*born) > life_steps {
                kill_parts.push(*guid);
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
                    if world.world().get::<RigidBody3D>(e).is_some() {
                        world.despawn(e);
                        n += 1;
                    }
                }
            }
        }
    }
    {
        let res = damage_mut(world);
        res.shed.retain(|(g, _)| !kill_parts.contains(g));
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
            slot.damage_j >= limits.glass_health_j && slot.latch != PartLatch::Gone
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
/// will drive, and the one the demo's own hero key reaches.
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
    if let Some(s) = damage_mut(world)
        .rows
        .get_mut(&chassis)
        .and_then(|r| r.parts.get_mut(&part))
    {
        s.target_deg = if open { hinge.open_deg } else { 0.0 };
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
