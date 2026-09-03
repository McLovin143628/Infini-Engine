//! **The raycast vehicle model** (P29.7) — suspension by ray, drive and steer at
//! the controller, as pure functions of numbers.
//!
//! This module is the `inf_ecs::movement` of vehicles: the whole model, with no
//! simulation in it. The fixed-step door that casts the rays, applies the forces
//! and writes the wheels back is `inf_physics::d3::vehicle`, exactly as
//! `inf_physics::d3::movement` is the door for the movement model here and
//! `inf_physics::d3::camera` is for `inf_ecs::camera`. The split is what lets the
//! interesting arithmetic be tested without a physics world and keeps every
//! rapier type on the other side of one wall.
//!
//! # Why owned code and not rapier's `DynamicRayCastVehicleController`
//!
//! rapier ships one, and the workspace pin enables only `enhanced-determinism` —
//! `control` is compiled but the type is a *pattern* here rather than a
//! dependency, for the reason every facade in this repository exists: its
//! vocabulary is rapier handles and nalgebra-flavoured vectors, its tunables are
//! not SI-documented, and a gameplay feature whose behaviour we cannot change is
//! a feature the island phase cannot extend. What is ported is the **shape** —
//! per-wheel ray, spring + damper along the suspension axis, a friction circle at
//! the contact — which is the standard raycast-vehicle design and is older than
//! either engine.
//!
//! # The rig is DERIVED, never authored
//!
//! There is no `Vehicle` component and this wave adds no scene field: the scene
//! schema bumps exactly once in a phase and P29.3 spent Phase 29's. A vehicle is
//! recognised from geometry the wire already carries —
//!
//! * the **chassis** is an entity with a `RigidBody3D { kind: Dynamic }` and a
//!   `Collider3D`;
//! * a **wheel** is a direct child carrying `Collider3D { shape_kind: Sphere,
//!   sensor: true }` and **no** `RigidBody3D` of its own; its local `Transform`
//!   is the mount point and its collider radius is the wheel radius.
//!
//! — and the recogniser is one function ([`wheel_of`]) that both the physics
//! bridge and the sample generator read, so the level and the simulation cannot
//! disagree about what a wheel is. `sensor: true` is the honest spelling of "a
//! shape that describes a volume and exerts no force", and it is also the
//! safety net: the bridge **consumes** a wheel rather than mirroring it into
//! rapier (the same treatment a fractured actor's own collider gets), and if
//! that ever stopped happening a sensor would still push nothing.
//!
//! Everything that is not geometry — spring rates, engine force, grip — is a
//! **tunable** ([`VehicleTuning`]), which lives on the running vehicle and is
//! live-editable during Simulate through the P29.5 door. That is deliberate and
//! not a shortcut: a vehicle *class* is content, and content is the island
//! phase's.

use glam::{DQuat, DVec3};
use uuid::Uuid;

use crate::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform};
use crate::math::Vec3d;
use crate::EcsWorld;

// ── the rig, derived from the scene ─────────────────────────────────────────

/// One wheel's **geometry**, read off the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMount {
    /// The wheel entity's stable identity — the door writes the visual
    /// suspension travel and spin back onto its `Transform`.
    pub guid: Uuid,
    /// Where the suspension is bolted to the chassis, in the chassis frame,
    /// metres. This is the wheel's position at **full extension**, so the ray
    /// starts here.
    pub mount_local: Vec3d,
    /// The wheel's radius, metres — its collider's.
    pub radius_m: f64,
}

impl WheelMount {
    /// Whether this wheel steers: the ones in **front of the chassis origin**.
    ///
    /// `+Z` is forward everywhere in this engine (`movement::rotate_from_frame`
    /// maps a forward stick of `(0, 1)` onto world `+Z` at zero yaw), so the rule
    /// is a sign test and not a table. A rig whose author wants four-wheel steer
    /// is a different `Vehicle` implementation, which is what the trait is for.
    pub fn steered(&self) -> bool {
        self.mount_local.z > 0.0
    }
}

/// **What a vehicle's non-wheel part IS** (wave VEH2c) — the discriminator that
/// lets a wheel-less craft be recognised at all.
///
/// # Why the COLLIDER SHAPE says which, and nothing else does
///
/// [`wheel_of`] has answered "is this a wheel" since P29.7 with one rule: a
/// **sphere** sensor with no body of its own. That rule is derived rather than
/// authored, which is this engine's whole doctrine for rigs — *"the rig is
/// DERIVED never authored"* — and it costs no schema, because a `Collider3D` on
/// a child entity is a component every level has been able to write since v1.
///
/// The authored vocabulary of collider shapes is **exactly three**
/// ([`ColliderShape3DKind`]), the sphere is taken, and so the other two are the
/// whole space of parts a rig can name:
///
/// | shape, as a sensor child with no body | part |
/// |---|---|
/// | sphere | a **wheel** ([`wheel_of`], unchanged since P29.7) |
/// | box | a **thruster** — a hull's screw and its rudder |
/// | capsule | a **rotor** — a rotorcraft's disc |
///
/// That the space is exhausted is worth stating rather than hiding: a fourth
/// part kind is not expressible without a fourth collider shape, and the day one
/// is wanted the honest move is a new shape, not a second recogniser keyed off
/// something else. **One door per rule**, and this is the door.
///
/// # The regression surface, and why it is bounded
///
/// A part is only a part if its parent is a chassis *and that chassis has no
/// wheels* — see [`rig_of`]. So no existing car changes behaviour at all: a
/// trigger volume parented to a car is mirrored into rapier exactly as it always
/// was. What is exposed is the narrow case of a wheel-less dynamic body with a
/// box or capsule sensor child, which is a shape nothing in this repository
/// authors today — `the_islands_only_wheel_less_vehicles_are_the_ones_it_placed`
/// (`inf-editor-core/tests/committed_level_sidecars.rs`) derives a rig for every
/// entity of all **twenty-four** committed levels and measures it: **28 wheeled
/// rigs, none of them carrying a part, and exactly two wheel-less craft, both
/// the island's own.** *That arm was written by wave VEH2c's audit; at
/// `4c69d3b5` this sentence cited a test that did not exist.*
///
/// [`ColliderShape3DKind`]: crate::components::ColliderShape3DKind
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PartKind {
    /// A hull thruster: a screw that pushes along the chassis's forward axis and
    /// a rudder that deflects that push sideways. Recognised as a **box** sensor.
    Thruster,
    /// A lifting rotor: a disc that pushes along the chassis's up axis.
    /// Recognised as a **capsule** sensor.
    Rotor,
}

impl PartKind {
    /// A stable short name, for a gate trace and a diagnostic.
    pub fn name(self) -> &'static str {
        match self {
            PartKind::Thruster => "thruster",
            PartKind::Rotor => "rotor",
        }
    }
}

/// One non-wheel part's **geometry**, read off the scene (wave VEH2c).
///
/// The sibling of [`WheelMount`], and deliberately a separate list on
/// [`VehicleRig`] rather than a `kind` field on that one: a wheel is the only
/// part the fixed-step door **casts a ray for** and the only one carrying
/// suspension state, so folding the two together would give every boat a
/// suspension and every wheel a kind nobody reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartMount {
    /// The part entity's stable identity — the door writes the part's own
    /// rotation back onto its `Transform` (a spinning rotor, a turned rudder).
    pub guid: Uuid,
    /// Which kind of part this is — the collider shape, resolved once.
    pub kind: PartKind,
    /// Where it is bolted to the chassis, in the chassis frame, metres. **The
    /// force is applied here**, so a screw below and behind the centre of mass
    /// makes a boat squat under power exactly as one does.
    pub mount_local: Vec3d,
    /// The part's own size, metres: a box's half-extents, or a capsule's
    /// `(radius, half_height, radius)`.
    ///
    /// Read by the class, not by the door. A rotor's `x` is its **disc
    /// radius**, which is what the drawn blade is scaled to and what a tip-speed
    /// bound would be measured against; a thruster's `y` is how deep the screw
    /// sits, which is the tolerance band on [`ChassisState::water_y`].
    pub size: Vec3d,
}

/// A vehicle's geometry: its chassis, its wheels and its other parts, in `Guid`
/// order.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleRig {
    /// The chassis entity — the dynamic body the forces are applied to.
    pub chassis: Uuid,
    /// Where a driver sits, in the chassis frame, metres. Derived from the
    /// chassis collider rather than authored: the top face's centre, so a
    /// character's **feet** land on the chassis and the seat needs no field.
    pub seat_local: Vec3d,
    /// The wheels, sorted by `Guid` so the order is a function of the level's
    /// contents and not of a bevy archetype walk.
    pub wheels: Vec<WheelMount>,
    /// The **non-wheel** parts — thrusters and rotors — sorted by `Guid` for the
    /// same reason (wave VEH2c). Empty for every wheeled vehicle, and empty is
    /// what every level written before this wave produces.
    pub parts: Vec<PartMount>,
}

impl VehicleRig {
    /// The parts of one kind, in `Guid` order.
    ///
    /// The one place a class asks "which of my parts are rotors", so a class
    /// cannot invent a second filter that disagrees about `Guid` order.
    pub fn parts_of(&self, kind: PartKind) -> impl Iterator<Item = (usize, &PartMount)> {
        self.parts
            .iter()
            .enumerate()
            .filter(move |(_, p)| p.kind == kind)
    }
}

/// Whether `entity`'s components describe a **wheel** — the one recogniser.
///
/// A wheel is a sphere sensor with no body of its own. Returns the radius so a
/// caller that has already asked the question does not ask it twice.
pub fn wheel_of(collider: Option<&Collider3D>, body: Option<&RigidBody3D>) -> Option<f64> {
    let c = collider?;
    if body.is_some() || !c.sensor || c.shape_kind != ColliderShape3DKind::Sphere {
        return None;
    }
    (c.radius.is_finite() && c.radius > 0.0).then_some(c.radius)
}

/// Whether `entity`'s components describe a **non-wheel part** — a thruster or a
/// rotor (wave VEH2c), with its size.
///
/// [`wheel_of`]'s sibling and deliberately the same shape: a sensor with no body
/// of its own, discriminated by [`PartKind`]'s shape table. A sphere is a wheel
/// and answers `None` here, so the two recognisers partition the vocabulary
/// rather than overlapping in it.
pub fn part_of(
    collider: Option<&Collider3D>,
    body: Option<&RigidBody3D>,
) -> Option<(PartKind, Vec3d)> {
    let c = collider?;
    if body.is_some() || !c.sensor {
        return None;
    }
    let (kind, size) = match c.shape_kind {
        // A wheel. `wheel_of` owns the sphere and this function does not.
        ColliderShape3DKind::Sphere => return None,
        ColliderShape3DKind::Box => (PartKind::Thruster, c.half_extents),
        ColliderShape3DKind::Capsule => (
            PartKind::Rotor,
            Vec3d::new(c.radius, c.half_extents.y, c.radius),
        ),
    };
    // A part with no size is a part that can neither push nor be drawn, and a
    // refusal is a value: the entity stays an ordinary sensor.
    (size.x.is_finite() && size.z.is_finite() && size.x > 0.0 && size.z > 0.0)
        .then_some((kind, size))
}

/// Whether `entity`'s components describe a **chassis** — a dynamic body with a
/// collider to hang wheels off.
pub fn chassis_of(collider: Option<&Collider3D>, body: Option<&RigidBody3D>) -> Option<Vec3d> {
    let (c, b) = (collider?, body?);
    if b.kind != BodyKind3D::Dynamic {
        return None;
    }
    // The seat is the top face's centre. A capsule or a sphere chassis answers
    // with its radius, so the rule composes over the whole authored vocabulary.
    let top = match c.shape_kind {
        ColliderShape3DKind::Box => c.half_extents.y,
        ColliderShape3DKind::Sphere => c.radius,
        ColliderShape3DKind::Capsule => c.half_extents.y + c.radius,
    };
    Some(Vec3d::new(c.offset.x, c.offset.y + top, c.offset.z))
}

/// **Derive a vehicle rig from the scene**, or `None` if `chassis` is not one.
///
/// `O(children)`. The physics bridge does not call this per step — it collects
/// the same facts inside the entity walk it already makes — but a test, a sample
/// generator and an editor tool all want the question answered directly, and two
/// spellings of "what is a wheel" is the defect this repository has paid for at
/// four separate seams.
///
/// # WHEELS WIN, and that is the whole compatibility rule (wave VEH2c)
///
/// P29.7 through VEH2b answered `None` for a chassis with no wheels, so a
/// wheel-less craft was **structurally invisible to the recogniser** however
/// completely the [`Vehicle`] trait supported one — the trait's own doc has said
/// *"a tank, a hovercraft and a boat can share one fixed-step door"* since the
/// day it was written, and the derivation did not.
///
/// The seam opens with one rule and one rule only: **a chassis that has wheels
/// is a wheeled vehicle, and its box and capsule sensor children are ordinary
/// sensors exactly as they always were.** Only a chassis with *no* wheels looks
/// for [`part_of`]. So every level ever written keeps the rig it had, byte for
/// byte, and the new surface is bounded to a shape nothing in this repository
/// authored before this wave.
pub fn rig_of(world: &EcsWorld, chassis: Uuid) -> Option<VehicleRig> {
    let entity = world.entity_of(chassis)?;
    let w = world.world();
    let seat_local = chassis_of(w.get::<Collider3D>(entity), w.get::<RigidBody3D>(entity))?;
    let mut wheels: Vec<WheelMount> = Vec::new();
    let mut parts: Vec<PartMount> = Vec::new();
    for child in world.children_of(entity) {
        let Some(guid) = world.guid_of(child) else {
            continue;
        };
        let (col, body) = (w.get::<Collider3D>(child), w.get::<RigidBody3D>(child));
        let mount_local = w
            .get::<Transform>(child)
            .map(|t| t.translation)
            .unwrap_or(Vec3d::ZERO);
        if let Some(radius_m) = wheel_of(col, body) {
            wheels.push(WheelMount {
                guid,
                mount_local,
                radius_m,
            });
        } else if let Some((kind, size)) = part_of(col, body) {
            parts.push(PartMount {
                guid,
                kind,
                mount_local,
                size,
            });
        }
    }
    if !wheels.is_empty() {
        // Wheels win: a car's trigger children stay triggers.
        parts.clear();
    } else if parts.is_empty() {
        return None;
    }
    wheels.sort_unstable_by_key(|wm| wm.guid);
    parts.sort_unstable_by_key(|p| p.guid);
    Some(VehicleRig {
        chassis,
        seat_local,
        wheels,
        parts,
    })
}

// ── the tunables ────────────────────────────────────────────────────────────

/// A vehicle's tunables — **SI throughout**, live-editable during Simulate.
///
/// The defaults describe the test rig this phase ships: a 4 × 1 × 2 m body at
/// 150 kg/m³ (≈ 1 200 kg, a hollow shell — see `Collider3D::density`'s own note
/// on why a car body is far lighter than its material) on four 0.35 m wheels.
/// Every number below is derived from that mass rather than dialled in, and the
/// derivation is in the field's doc so a different vehicle can redo it.
///
/// # The VEH2a window — sixty-two numbers, landed once
///
/// P29.7 shipped fifteen, and island wave VEH1a serialized exactly those fifteen
/// as [`VehicleClass`](crate::components::VehicleClass). Wave **VEH2a** grows the
/// set to the whole Forza-grade surface **in one bump** — a torque curve, a
/// gearbox, a drivetrain, a tyre model, an aero package, a steering rack, three
/// driver aids and the seat warp — because the house law is one scene-schema
/// window per phase and a tunable that arrives a wave late is a tunable that
/// costs a second rung of the ladder.
///
/// Every one of the sixty-two is an `f64` and is reachable through
/// [`set`](Self::set) by the name [`names`](Self::names) advertises, so the
/// component, the live tuner, the catalogue row and the Details grid all read one
/// list. **There is no second door**: an enum-shaped concept (the drivetrain) is
/// spelled as the scalar the physics actually reads
/// ([`front_torque_split`](Self::front_torque_split)) rather than as a parallel
/// typed field with its own setter — see that field's own note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleTuning {
    /// Suspension length at full extension, metres — how far below its mount a
    /// wheel hangs when the vehicle is in the air.
    pub rest_length_m: f64,
    /// How far the suspension may compress from rest, metres.
    pub travel_m: f64,
    /// Spring rate, **newtons per metre, per wheel**.
    ///
    /// Sized so the rig sits at about 60 % of its travel: 1 200 kg on four
    /// wheels is 2 943 N each, and 2 943 / 20 000 = 0.147 m of a 0.25 m travel.
    pub stiffness_n_per_m: f64,
    /// Damping, **newton-seconds per metre, per wheel**. Critical for a 300 kg
    /// quarter-mass on the spring above is 2·√(k·m) ≈ 4 900; this is about 0.6
    /// of it, which settles in under a second without feeling locked.
    pub damping_ns_per_m: f64,
    /// Peak drive force, newtons, summed over the driven wheels. 8 000 N on
    /// 1 200 kg is 6.7 m/s², which is a brisk road car.
    pub max_engine_force_n: f64,
    /// The speed at which the drive force reaches zero, m/s — the far end of the
    /// engine curve, and therefore the vehicle's top speed on the flat.
    pub max_speed_mps: f64,
    /// Peak braking force, newtons, summed over all wheels.
    pub brake_force_n: f64,
    /// Peak handbrake force, newtons, applied to the **rear** wheels only, which
    /// is what makes a handbrake turn a turn rather than a stop.
    pub handbrake_force_n: f64,
    /// Steering angle at a standstill, degrees.
    pub max_steer_deg: f64,
    /// Steering angle at [`max_speed_mps`](Self::max_speed_mps) and above,
    /// degrees — the speed-sensitive half. A car that can turn 35° at 25 m/s is
    /// a car that spins on every input.
    pub min_steer_deg: f64,
    /// Lateral friction coefficient (µ) at the contact. The sideways force is
    /// whatever would cancel the slip in one step, **clamped** to µ × the
    /// suspension load — a friction circle, which is what keeps this stable at
    /// any timestep instead of exploding when the slip is large.
    pub lateral_grip: f64,
    /// Longitudinal friction coefficient (µ) — the same clamp for drive and
    /// brake force, so a wheel cannot push harder than it is pressed down.
    pub longitudinal_grip: f64,
    /// Rolling resistance coefficient — a force of `c × load` opposing motion.
    /// 0.015 is a road tyre on tarmac.
    pub rolling_resistance: f64,
    /// Aerodynamic drag, newtons per (m/s)². `0.5·ρ·Cd·A` for a car is about
    /// 0.4; the engine curve sets the top speed here, so this is the shape of
    /// the approach to it rather than the limit itself.
    pub drag_n_per_mps2: f64,
    /// How long the enter/exit choreography takes, seconds.
    pub enter_time_s: f64,
    /// Clip time the seat warp **opens**, seconds — the first half of what used
    /// to be an `inf_anim::WarpWindow` field.
    ///
    /// # Why two `f64`s and not the `WarpWindow`
    ///
    /// P29.7 carried the window as its own type, and VEH1a's ledger recorded the
    /// consequence as a named bound: *"`VehicleTuning::enter_window` is **not**
    /// [in `VehicleClass`], because it is not nameable through `set`"*. A tunable
    /// that no door can reach is a tunable no author has. The window is the same
    /// window — [`enter_window`](Self::enter_window) rebuilds it — and it is now
    /// two names on the one door, which is what closes that carried item.
    ///
    /// Before it the character is still walking; after it the character is seated
    /// and the last of the clip plays out. See `inf_physics::d3::vehicle`'s
    /// enter/exit section for why a window and not the whole clip.
    pub enter_warp_start: f64,
    /// Clip time the seat warp **closes**, seconds.
    pub enter_warp_end: f64,

    // ── the engine (VEH2a) ──────────────────────────────────────────────────
    //
    // A torque curve through three knots — idle, peak, redline — with ONE shape
    // parameter between them. Torque, not force: `max_engine_force_n` above is
    // now the driveline's CEILING and no longer the curve, because a curve that
    // is a single number cannot be revvy or torquey and those are the two things
    // a driver feels first.
    /// Idle speed, rpm. The engine never turns slower than this: below it a real
    /// clutch is slipping, and modelling the clutch is a state machine this
    /// engine does not need to be a car.
    pub idle_rpm: f64,
    /// Where the torque curve peaks, rpm.
    pub peak_torque_rpm: f64,
    /// Where the limiter cuts, rpm — the top of the curve.
    pub redline_rpm: f64,
    /// Peak crankshaft torque, newton-metres. Multiplied by the gear, the final
    /// drive and divided by the wheel's own radius, which is what makes wheel
    /// size finally matter (P29.7's model never read the radius on the drive
    /// path at all).
    pub peak_torque_nm: f64,
    /// Torque at [`idle_rpm`](Self::idle_rpm), as a fraction of the peak.
    pub idle_torque_frac: f64,
    /// Torque at [`redline_rpm`](Self::redline_rpm), as a fraction of the peak.
    pub redline_torque_frac: f64,
    /// **The one shape knob**, `(0, 1)`, applied to both sides of the peak
    /// through [`curve_bias`].
    ///
    /// `0.5` is straight lines between the three knots. Above it the torque
    /// arrives *early* and dies *hard* — a diesel or a truck. Below it the engine
    /// is soft low down and holds its torque to the limiter — a sports engine.
    /// One number rather than two exponents because the two halves of "where does
    /// this engine make its torque" are the same question asked twice.
    pub torque_curve_bias: f64,
    /// Engine braking at the crank with the throttle shut, newton-metres at the
    /// redline (scaled by the rev fraction below it). What makes lifting off
    /// slow the car without touching the brake.
    pub engine_brake_nm: f64,

    // ── the gearbox (VEH2a) ─────────────────────────────────────────────────
    /// How many forward gears are in use, `1..=`[`MAX_GEARS`]. A count rather
    /// than "the first zero ratio", because a sentinel is a value that means two
    /// things.
    pub gear_count: f64,
    /// First gear's ratio.
    pub gear_1_ratio: f64,
    /// Second gear's ratio.
    pub gear_2_ratio: f64,
    /// Third gear's ratio.
    pub gear_3_ratio: f64,
    /// Fourth gear's ratio.
    pub gear_4_ratio: f64,
    /// Fifth gear's ratio.
    pub gear_5_ratio: f64,
    /// Sixth gear's ratio.
    pub gear_6_ratio: f64,
    /// Seventh gear's ratio — `0` when [`gear_count`](Self::gear_count) is below
    /// seven.
    pub gear_7_ratio: f64,
    /// Eighth gear's ratio.
    pub gear_8_ratio: f64,
    /// Reverse's ratio. **Reverse is a gear**, not a scalar on the drive force:
    /// P29.7 reversed at "a third of the force", which is a car whose reverse
    /// gets *stronger* as its engine gets stronger and never runs out of revs.
    pub reverse_ratio: f64,
    /// The final drive — the differential's own reduction, multiplying every
    /// gear.
    pub final_drive: f64,
    /// How long a shift takes, seconds. **No drive torque crosses the gearbox
    /// during it**, which is the whole of why a shift is felt.
    pub shift_time_s: f64,
    /// Upshift above this, rpm.
    pub shift_up_rpm: f64,
    /// Downshift below this, rpm. Must be enough under
    /// [`shift_up_rpm`](Self::shift_up_rpm) divided by the ratio step, or the box
    /// hunts.
    pub shift_down_rpm: f64,

    // ── the drivetrain (VEH2a) ──────────────────────────────────────────────
    /// The share of drive torque the **front** axle takes, `[0, 1]`.
    ///
    /// # This scalar IS the drivetrain, and that is a ruling
    ///
    /// `0.0` is rear-wheel drive, `1.0` is front, and anything between is
    /// all-wheel drive at that split. A `Drivetrain { Fwd, Rwd, Awd }` enum was
    /// considered and **refused**: the physics reads a split and nothing else, so
    /// an enum beside it would be a second source of truth for one fact, and it
    /// could not travel the `set(name, f64)` door every other tunable travels —
    /// which would mean a second setter on the `Vehicle` trait, i.e. the P29.6
    /// A14 defect (two lists of one thing) bought for no behaviour.
    ///
    /// A catalogue row may still *say* `drivetrain = "awd"`: that is a string key
    /// [`VehicleDef::from_toml_table`] resolves to this number before it reads
    /// any numeric key, so an explicit `front_torque_split` always wins.
    pub front_torque_split: f64,
    /// Front differential lock, `[0, 1]` — `0` open, `1` a spool. A locked diff
    /// pulls its two wheels' speeds together, which is what sends torque to the
    /// wheel that still has grip.
    pub diff_lock_front: f64,
    /// Rear differential lock, `[0, 1]`.
    pub diff_lock_rear: f64,
    /// The share of the brake budget the **front** axle takes, `[0, 1]`. Road
    /// cars run 0.6–0.7 forward, because braking transfers load forward and grip
    /// follows load.
    pub brake_bias: f64,

    // ── the tyres (VEH2a) ───────────────────────────────────────────────────
    //
    // A simplified Pacejka: a rising branch to a peak and a falling branch to a
    // sliding plateau, per axis, coupled through ONE combined-slip magnitude —
    // a true friction circle. P29.7's two independent per-axis clamps could hold
    // µ×load sideways *and* µ×load forwards at the same time, which is 1.41 × µ
    // of grip and is why it could brake out of any corner.
    /// Slip **ratio** at which longitudinal grip peaks — 0.10–0.15 for a road
    /// tyre.
    pub tyre_long_peak_slip: f64,
    /// How stiff the longitudinal rise is, `(0, 1)`, through [`curve_bias`].
    /// `0.5` is a straight line to the peak; above it the tyre bites early (a
    /// stiff sidewall), below it the rise is lazy.
    pub tyre_long_rise_bias: f64,
    /// **Tangent** of the slip angle at which lateral grip peaks — 0.16 is about
    /// 9°. A tangent and not an angle because the model computes
    /// `v_lateral / |v_forward|` directly and an `atan` on the tyre path would be
    /// a transcendental in the sim loop for a number that is immediately undone.
    pub tyre_lat_peak_slip: f64,
    /// How stiff the lateral rise is, `(0, 1)`.
    pub tyre_lat_rise_bias: f64,
    /// Grip once the tyre is fully sliding, as a fraction of the peak. Below 1 by
    /// construction: a sliding tyre grips less than a gripping one, and the gap
    /// is what makes a slide something a driver must correct.
    pub tyre_slide_frac: f64,
    /// **Load sensitivity** — how fast µ falls as vertical load rises, per unit
    /// of load over the static share.
    ///
    /// `0` is the schoolbook tyre whose grip is exactly `µ × Fz`; a real one at
    /// `0.22` loses 22 % of its µ when it carries twice its static load. This is
    /// the number that makes **weight transfer cost grip**, and therefore the
    /// number that makes a soft roll bar and a low centre of gravity worth
    /// having.
    pub tyre_load_sensitivity: f64,
    /// A wheel's rotational inertia, kg·m². A disc of mass `m` and radius `r` is
    /// `½ m r²`, so a 20 kg wheel on a 0.35 m tyre is 1.2.
    pub wheel_inertia_kgm2: f64,

    // ── the chassis (VEH2a) ─────────────────────────────────────────────────
    /// Height of the centre of gravity **above the chassis origin**, metres —
    /// negative for a car whose mass is in its floor, which is every car.
    ///
    /// The solver's centre of mass is the chassis collider's centre and this
    /// engine has no door to move it. So the model moves the *forces* instead:
    /// a horizontal tyre force applied at `contact - up × cog_height_m` produces
    /// exactly the moment the true centre of gravity would have felt, and the
    /// **suspension force is untouched by construction** because it is parallel
    /// to `up` and `up × up` is zero.
    pub cog_height_m: f64,
    /// Front anti-roll bar rate, newtons per metre of **compression difference**
    /// across the axle. It transfers load from the inside wheel to the outside
    /// one without adding any, which — with
    /// [`tyre_load_sensitivity`](Self::tyre_load_sensitivity) above zero — costs
    /// that axle grip. Stiffening the front is how a car is made to understeer.
    pub anti_roll_front_n_per_m: f64,
    /// Rear anti-roll bar rate, same units.
    pub anti_roll_rear_n_per_m: f64,
    /// Downforce, newtons per (m/s)² — pressed **into** the chassis up, so it
    /// adds load (and therefore grip) rather than mass.
    pub downforce_n_per_mps2: f64,
    /// Where that downforce acts along the wheelbase, in fractions of the half
    /// wheelbase. `0` is the centre of gravity and makes no moment; `-1` is over
    /// the rear axle, which is a wing.
    pub downforce_centre_z: f64,
    /// Aerodynamic drag **sideways**, newtons per (m/s)². A car's side area is
    /// two to three times its frontal area, and the difference is what stops a
    /// slide feeling like ice.
    pub drag_lateral_n_per_mps2: f64,

    // ── the steering rack (VEH2a) ───────────────────────────────────────────
    /// How fast the road wheels turn toward the driver's demand, degrees per
    /// second. P29.7's steering was instant, which is a car that changes
    /// direction in one frame.
    pub steer_rate_deg_per_s: f64,
    /// How fast they return to centre with no input, degrees per second. Faster
    /// than the rate above, because a real rack is self-centring.
    pub steer_return_deg_per_s: f64,
    /// Ackermann, `[0, 1]` — how much more the **inside** wheel turns than the
    /// outside one. `0` is parallel steering, `1` is the geometry that puts both
    /// front wheels on the same turn centre.
    pub ackermann: f64,

    // ── the driver aids (VEH2a) ─────────────────────────────────────────────
    //
    // Each is one number that carries both the toggle and the threshold: zero is
    // OFF. A separate `bool` beside a threshold is two fields that can disagree.
    /// ABS: the slip ratio above which brake torque is bled off. `0` disables it.
    pub abs_slip: f64,
    /// Traction control: the drive slip ratio above which engine torque is bled
    /// off. `0` disables it.
    pub traction_control_slip: f64,
    /// Stability control strength, `[0, 1]` — how hard a single wheel is braked
    /// to pull the yaw rate back to what the steering asked for. `0` disables it.
    pub stability_control: f64,
}

/// The most forward gears a gearbox may declare.
///
/// Eight, because eight is what a modern automatic has and a fixed arity is what
/// keeps [`VehicleTuning`] `Copy`, keeps every gear reachable through the one
/// by-name door, and keeps the serialized class a flat run of `f64`s that a
/// bincode wire pin can account for field by field. A `Vec` would cost all three.
pub const MAX_GEARS: usize = 8;

/// How many multiples of the peak slip a tyre takes to reach its sliding
/// plateau.
///
/// One number for the whole engine rather than a per-class knob, on
/// [`TYRE_WIDTH_FRAC`]'s argument: *where* a tyre gives up is
/// [`VehicleTuning::tyre_long_peak_slip`] and *how much* it keeps is
/// [`VehicleTuning::tyre_slide_frac`]; how quickly it gets from one to the other
/// is a property of rubber, not of a car.
pub const TYRE_SLIDE_SLIP_MULT: f64 = 3.0;

impl Default for VehicleTuning {
    fn default() -> Self {
        Self {
            rest_length_m: 0.5,
            travel_m: 0.25,
            stiffness_n_per_m: 20_000.0,
            damping_ns_per_m: 3_000.0,
            // The DRIVELINE CEILING since VEH2a, not the curve: whatever the
            // torque curve and the gearbox ask for, the wheels are never handed
            // more than this in total. 8 000 N on 1 200 kg is 6.7 m/s², which is
            // a launch a road car's clutch and half-shafts would survive.
            max_engine_force_n: 8_000.0,
            // Still the flat curve's own falloff speed until the powertrain
            // clause replaces that curve; it becomes the class's REFERENCE top
            // speed (the steering limit's and the drive camera's) there, and
            // moves with it.
            max_speed_mps: 25.0,
            brake_force_n: 12_000.0,
            handbrake_force_n: 9_000.0,
            max_steer_deg: 35.0,
            min_steer_deg: 8.0,
            lateral_grip: 1.1,
            longitudinal_grip: 1.2,
            rolling_resistance: 0.015,
            drag_n_per_mps2: 0.4,
            enter_time_s: 0.55,
            // 18 % in and 82 % through: the approach and the settle are the
            // clip's, the warp is the middle.
            enter_warp_start: 0.1,
            enter_warp_end: 0.45,

            // A 2.0-litre naturally-aspirated petrol: 260 N·m at 3 800, pulling
            // to 6 500. Straight lines between the knots (`bias` 0.5) is the
            // shape an author can predict, and the two extremes are what the
            // truck and the sports row reach for.
            idle_rpm: 800.0,
            peak_torque_rpm: 3_800.0,
            redline_rpm: 6_500.0,
            peak_torque_nm: 260.0,
            idle_torque_frac: 0.55,
            redline_torque_frac: 0.72,
            torque_curve_bias: 0.5,
            engine_brake_nm: 35.0,

            // Six speeds and a 3.7 final — the ratios step by about 1.45, which
            // is what keeps the engine inside its band across a shift.
            gear_count: 6.0,
            gear_1_ratio: 3.50,
            gear_2_ratio: 2.10,
            gear_3_ratio: 1.45,
            gear_4_ratio: 1.10,
            gear_5_ratio: 0.88,
            gear_6_ratio: 0.72,
            gear_7_ratio: 0.0,
            gear_8_ratio: 0.0,
            reverse_ratio: 3.20,
            final_drive: 3.70,
            shift_time_s: 0.28,
            shift_up_rpm: 6_000.0,
            shift_down_rpm: 2_200.0,

            // An even split is the closest thing to P29.7's "share the drive over
            // the grounded wheels", so the default rig's character survives the
            // wave; a catalogue row says fwd/rwd/awd for itself.
            front_torque_split: 0.5,
            diff_lock_front: 0.0,
            diff_lock_rear: 0.25,
            brake_bias: 0.62,

            tyre_long_peak_slip: 0.12,
            tyre_long_rise_bias: 0.74,
            tyre_lat_peak_slip: 0.16,
            tyre_lat_rise_bias: 0.72,
            tyre_slide_frac: 0.72,
            tyre_load_sensitivity: 0.22,
            // ½ × 20 kg × 0.35 m² — a wheel and a tyre.
            wheel_inertia_kgm2: 1.2,

            // A car's mass is in its floor: a quarter-metre below the middle of a
            // 1.24 m tall body puts the centre of gravity about half a metre off
            // the road, which is a saloon's.
            cog_height_m: -0.25,
            anti_roll_front_n_per_m: 12_000.0,
            anti_roll_rear_n_per_m: 9_000.0,
            downforce_n_per_mps2: 0.15,
            downforce_centre_z: -0.3,
            // Three times the forward drag: a car's flank is about that much
            // bigger than its nose.
            drag_lateral_n_per_mps2: 1.2,

            steer_rate_deg_per_s: 220.0,
            steer_return_deg_per_s: 320.0,
            ackermann: 1.0,

            // The Ring-0 default is a MODERN road car, and a modern road car has
            // its aids on. A row that wants a driver's car turns them down.
            abs_slip: 0.15,
            traction_control_slip: 0.18,
            stability_control: 0.35,
        }
    }
}

impl VehicleTuning {
    /// Set one tunable **by name**, answering whether it took.
    ///
    /// A refusal is a value, exactly as `CameraTuning::set`'s is and for the same
    /// reason: a tuning surface is live over a world that is changing underneath
    /// it, and taking a session down over a stale field name is the wrong trade.
    /// A non-finite value is refused too; a wide value is not, because a tuner
    /// asking "what does 200 000 N/m feel like" is asking a real question.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        let slot: &mut f64 = match name {
            "abs_slip" => &mut self.abs_slip,
            "ackermann" => &mut self.ackermann,
            "anti_roll_front_n_per_m" => &mut self.anti_roll_front_n_per_m,
            "anti_roll_rear_n_per_m" => &mut self.anti_roll_rear_n_per_m,
            "brake_bias" => &mut self.brake_bias,
            "brake_force_n" => &mut self.brake_force_n,
            "cog_height_m" => &mut self.cog_height_m,
            "damping_ns_per_m" => &mut self.damping_ns_per_m,
            "diff_lock_front" => &mut self.diff_lock_front,
            "diff_lock_rear" => &mut self.diff_lock_rear,
            "downforce_centre_z" => &mut self.downforce_centre_z,
            "downforce_n_per_mps2" => &mut self.downforce_n_per_mps2,
            "drag_lateral_n_per_mps2" => &mut self.drag_lateral_n_per_mps2,
            "drag_n_per_mps2" => &mut self.drag_n_per_mps2,
            "engine_brake_nm" => &mut self.engine_brake_nm,
            "enter_time_s" => &mut self.enter_time_s,
            "enter_warp_end" => &mut self.enter_warp_end,
            "enter_warp_start" => &mut self.enter_warp_start,
            "final_drive" => &mut self.final_drive,
            "front_torque_split" => &mut self.front_torque_split,
            "gear_1_ratio" => &mut self.gear_1_ratio,
            "gear_2_ratio" => &mut self.gear_2_ratio,
            "gear_3_ratio" => &mut self.gear_3_ratio,
            "gear_4_ratio" => &mut self.gear_4_ratio,
            "gear_5_ratio" => &mut self.gear_5_ratio,
            "gear_6_ratio" => &mut self.gear_6_ratio,
            "gear_7_ratio" => &mut self.gear_7_ratio,
            "gear_8_ratio" => &mut self.gear_8_ratio,
            "gear_count" => &mut self.gear_count,
            "handbrake_force_n" => &mut self.handbrake_force_n,
            "idle_rpm" => &mut self.idle_rpm,
            "idle_torque_frac" => &mut self.idle_torque_frac,
            "lateral_grip" => &mut self.lateral_grip,
            "longitudinal_grip" => &mut self.longitudinal_grip,
            "max_engine_force_n" => &mut self.max_engine_force_n,
            "max_speed_mps" => &mut self.max_speed_mps,
            "max_steer_deg" => &mut self.max_steer_deg,
            "min_steer_deg" => &mut self.min_steer_deg,
            "peak_torque_nm" => &mut self.peak_torque_nm,
            "peak_torque_rpm" => &mut self.peak_torque_rpm,
            "redline_rpm" => &mut self.redline_rpm,
            "redline_torque_frac" => &mut self.redline_torque_frac,
            "rest_length_m" => &mut self.rest_length_m,
            "reverse_ratio" => &mut self.reverse_ratio,
            "rolling_resistance" => &mut self.rolling_resistance,
            "shift_down_rpm" => &mut self.shift_down_rpm,
            "shift_time_s" => &mut self.shift_time_s,
            "shift_up_rpm" => &mut self.shift_up_rpm,
            "stability_control" => &mut self.stability_control,
            "steer_rate_deg_per_s" => &mut self.steer_rate_deg_per_s,
            "steer_return_deg_per_s" => &mut self.steer_return_deg_per_s,
            "stiffness_n_per_m" => &mut self.stiffness_n_per_m,
            "torque_curve_bias" => &mut self.torque_curve_bias,
            "traction_control_slip" => &mut self.traction_control_slip,
            "travel_m" => &mut self.travel_m,
            "tyre_lat_peak_slip" => &mut self.tyre_lat_peak_slip,
            "tyre_lat_rise_bias" => &mut self.tyre_lat_rise_bias,
            "tyre_load_sensitivity" => &mut self.tyre_load_sensitivity,
            "tyre_long_peak_slip" => &mut self.tyre_long_peak_slip,
            "tyre_long_rise_bias" => &mut self.tyre_long_rise_bias,
            "tyre_slide_frac" => &mut self.tyre_slide_frac,
            "wheel_inertia_kgm2" => &mut self.wheel_inertia_kgm2,
            _ => return false,
        };
        *slot = value;
        true
    }

    /// Every settable name, sorted — so a UI and a test can enumerate the door
    /// rather than restate it. (The restated-list defect is the P29.6 audit's
    /// A14, met at this exact shape.)
    pub fn names() -> &'static [&'static str] {
        &[
            "abs_slip",
            "ackermann",
            "anti_roll_front_n_per_m",
            "anti_roll_rear_n_per_m",
            "brake_bias",
            "brake_force_n",
            "cog_height_m",
            "damping_ns_per_m",
            "diff_lock_front",
            "diff_lock_rear",
            "downforce_centre_z",
            "downforce_n_per_mps2",
            "drag_lateral_n_per_mps2",
            "drag_n_per_mps2",
            "engine_brake_nm",
            "enter_time_s",
            "enter_warp_end",
            "enter_warp_start",
            "final_drive",
            "front_torque_split",
            "gear_1_ratio",
            "gear_2_ratio",
            "gear_3_ratio",
            "gear_4_ratio",
            "gear_5_ratio",
            "gear_6_ratio",
            "gear_7_ratio",
            "gear_8_ratio",
            "gear_count",
            "handbrake_force_n",
            "idle_rpm",
            "idle_torque_frac",
            "lateral_grip",
            "longitudinal_grip",
            "max_engine_force_n",
            "max_speed_mps",
            "max_steer_deg",
            "min_steer_deg",
            "peak_torque_nm",
            "peak_torque_rpm",
            "redline_rpm",
            "redline_torque_frac",
            "rest_length_m",
            "reverse_ratio",
            "rolling_resistance",
            "shift_down_rpm",
            "shift_time_s",
            "shift_up_rpm",
            "stability_control",
            "steer_rate_deg_per_s",
            "steer_return_deg_per_s",
            "stiffness_n_per_m",
            "torque_curve_bias",
            "traction_control_slip",
            "travel_m",
            "tyre_lat_peak_slip",
            "tyre_lat_rise_bias",
            "tyre_load_sensitivity",
            "tyre_long_peak_slip",
            "tyre_long_rise_bias",
            "tyre_slide_frac",
            "wheel_inertia_kgm2",
        ]
    }

    /// The seat warp window these two clip times describe.
    ///
    /// The one place `enter_warp_start`/`enter_warp_end` become an
    /// `inf_anim::WarpWindow`, so the pair can never be read in the wrong order
    /// and the `f32` narrowing happens once.
    pub fn enter_window(&self) -> inf_anim::WarpWindow {
        inf_anim::WarpWindow::new(self.enter_warp_start as f32, self.enter_warp_end as f32)
    }

    /// How many forward gears this box actually has, `1..=`[`MAX_GEARS`].
    ///
    /// A refusal is a value here too: a `gear_count` an author typed as `0`, as
    /// `12` or as a fraction becomes a usable box rather than a division by zero
    /// three call sites down.
    pub fn gears(&self) -> usize {
        if !self.gear_count.is_finite() {
            return 1;
        }
        (self.gear_count.round() as i64).clamp(1, MAX_GEARS as i64) as usize
    }

    /// The **total** reduction from crank to wheel in `gear`, including the final
    /// drive: `-1` is reverse, `0` is neutral, `1..=gears()` are the forward
    /// gears. Zero for neutral and for a gear the box does not have.
    ///
    /// One function, so the torque path, the rev calculation and the shift model
    /// cannot disagree about what gear a car is in.
    pub fn drive_ratio(&self, gear: i32) -> f64 {
        let raw = match gear {
            -1 => self.reverse_ratio,
            1 => self.gear_1_ratio,
            2 => self.gear_2_ratio,
            3 => self.gear_3_ratio,
            4 => self.gear_4_ratio,
            5 => self.gear_5_ratio,
            6 => self.gear_6_ratio,
            7 => self.gear_7_ratio,
            8 => self.gear_8_ratio,
            _ => 0.0,
        };
        if gear > self.gears() as i32 {
            return 0.0;
        }
        raw * self.final_drive
    }
}

// ── the body a vehicle is DRAWN as (island wave VEH1a) ──────────────────────

/// Salt for [`body_part_guid`] — its own constant, so a drawn car part can
/// never alias a door leaf, a PCG doorway, a structure collider, a fracture
/// chunk or a building module.
const VEHICLE_PART_SALT: u128 = 0x7645_4831_4143_4152_424f_4459_5041_5254;

/// One drawn part of a vehicle's body, **in fractions of the chassis
/// half-extents**.
///
/// # Why fractions and not metres
///
/// The I8b module lesson, one content kind over: a family and not a mesh per
/// entry. A sedan and a hatchback are the same silhouette at two sizes, and a
/// table in metres would need a row set per size — so a `VehicleDef` carries the
/// half-extents and this table carries the *proportions*. Every feature is
/// therefore proportional by construction, which is the property that stops a
/// fixed 40 mm sill looking like a kerb on a truck and a pinstripe on a
/// hatchback.
///
/// `centre` and `half` are both in `[-1, 1]`-of-half-extent units, so a part
/// with `centre.y = 0.5` and `half.y = 0.5` fills the chassis's top half exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyPart {
    /// The part's stable name. **The content key** — `body_part_guid` derives
    /// the entity's `Guid` from it, so renaming a part is renaming an entity.
    pub name: &'static str,
    /// Centre, in fractions of the chassis half-extents.
    pub centre: Vec3d,
    /// Half-extents, in fractions of the chassis half-extents.
    pub half: Vec3d,
    /// Which built-in primitive draws it.
    pub primitive: crate::components::Primitive,
}

/// **What one part of a vehicle is painted, when it is not painted the
/// vehicle's own colour** (wave EMS1).
///
/// Three fields and not a whole [`Material`](crate::components::Material),
/// because two of a material's eight are decisions this table has no business
/// re-making: `metallic` and `roughness` are what a car's paint IS, and a livery
/// that could set them would let a police cruiser be made of velvet. The other
/// three are what a livery is: a colour, whether the part glows, and how much.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartPaint {
    /// The part's own base colour, in place of `RigSpawn::paint`.
    pub base_color: crate::math::Color,
    /// Linear emissive colour. `emissive_intensity` scales it; a part that only
    /// differs in paint carries black here and the material is unlit as usual.
    pub emissive: crate::math::Color,
    /// How bright that emission is — `Material::emissive_intensity`.
    ///
    /// **Above 1.0 or a light bar does not bloom**: the HDR path thresholds at
    /// a linear luminance of 1.0, and the brightest colour a `Color` can carry
    /// is exactly 1.0 (`Material::emissive_linear`'s own doc).
    pub emissive_intensity: f32,
}

impl PartPaint {
    /// A part that is only a different colour — no emission.
    pub const fn flat(base_color: crate::math::Color) -> Self {
        Self {
            base_color,
            emissive: crate::math::Color::new(0.0, 0.0, 0.0, 1.0),
            emissive_intensity: 1.0,
        }
    }
}

/// **A vehicle's livery** (wave EMS1) — a per-part paint table, plus the parts
/// the livery adds that the body itself does not have.
///
/// # Why the livery rides the SPAWN and not the parts table
///
/// The obvious place for a per-part override is [`BodyPart`] itself, and it is
/// the wrong one: a `VehicleBody`'s parts table is shared by every catalogue row
/// that names that body, and the emergency fleet **deliberately borrows civilian
/// bodies** (the kerb trap — a sixth `VehicleBody` would put an ambulance at
/// every sixth kerb slot, because `traffic::catalogue_row` draws uniformly over
/// `VehicleBody::ALL`). A white-over-blue entry on `SEDAN_PARTS` would therefore
/// have repainted **every saloon in the town** — the same defect the body
/// variant was avoided to prevent, arriving through the other door.
///
/// So a livery is a property of the vehicle, it is looked up by the part's own
/// NAME, and a part the table does not name keeps `RigSpawn::paint`. That also
/// makes it additive: a body may gain a part and every livery still works.
///
/// # `extra` is what makes a light bar possible at all
///
/// A saloon has a lower body, a greenhouse, a bonnet and a boot; none of them is
/// a roof bar, and adding one to `SEDAN_PARTS` would put a light bar on every
/// taxi. The livery's own parts are appended after the body's, so they take
/// their guids from [`body_part_guid`] like any other and `despawn_rig` takes
/// them with the chassis (the hierarchy walk removes children whatever the
/// recipe says).
///
/// **The bar is STATIC this wave.** A flashing one is a per-step write to the
/// vehicle's `Material` from two fenced hosts, and that is wave EMS2's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Livery {
    /// A stable short name for diagnostics and gate traces.
    pub name: &'static str,
    /// Per-part overrides, keyed on [`BodyPart::name`]. A part absent from this
    /// list keeps the vehicle's own paint.
    pub parts: &'static [(&'static str, PartPaint)],
    /// Parts this livery adds — a light bar, a beacon — in fractions of the
    /// chassis half-extents exactly as [`BodyPart`] is.
    pub extra: &'static [(BodyPart, PartPaint)],
    /// **What service a vehicle wearing this livery belongs to** (wave EMS2), or
    /// `None` for a livery that is only paint.
    ///
    /// # It is a DECLARATION, and the recogniser does not read it
    ///
    /// `inf_ecs::dispatch::unit_kind_of` answers the same question off the
    /// *world* — a bloomed `light_bar` child, its hue, and the chassis's own
    /// length — because that is the only channel that survives being written to
    /// an `.inf_lvl` and opened by a shipped player, which has no livery table at
    /// all. This field is the authoring side of the same fact, and
    /// `every_livery_is_recognised_as_the_service_it_declares` holds the two
    /// together: a fifth livery whose colours disagreed with the rule fails a
    /// test instead of sending an ambulance to a fire.
    ///
    /// It is read rather than decorative: a fixture that spawns a unit and then
    /// asks the dispatcher about it takes its expectation from here, so the
    /// declaration and the recognition are never the same sentence twice.
    pub service: Option<crate::dispatch::UnitKind>,
}

impl Livery {
    /// The paint for a named part, or `None` for one this livery does not
    /// re-colour.
    pub fn part(&self, name: &str) -> Option<PartPaint> {
        self.parts.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
    }
}

/// One **mount** of a vehicle's body — a thruster or a rotor — in the same
/// fractions of the chassis half-extents [`BodyPart`] uses (wave VEH2c).
///
/// [`BodyPart`]'s sibling, and deliberately a separate table rather than a flag
/// on that one: a body part is drawn and nothing else, and a mount is a
/// *recognised* collider that the vehicle door reads. They differ in what they
/// emit, not merely in what they look like.
///
/// # A fraction greater than one is expected here
///
/// A rotor is much wider than the fuselage it lifts — a light helicopter's disc
/// is about as wide as the machine is long — so `half.x` on a
/// [`PartKind::Rotor`] row is normally between two and four. That is the
/// fraction convention doing its job rather than straining it: the disc stays
/// proportional to the airframe across a whole family of sizes, which is the
/// property `BodyPart`'s own doc exists to defend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountPart {
    /// The part's stable name. **The content key** — `body_part_guid` derives
    /// the entity's `Guid` from it, exactly as a `BodyPart`'s does.
    pub name: &'static str,
    /// Which part this is, and therefore which collider shape recognises it.
    pub kind: PartKind,
    /// Centre, in fractions of the chassis half-extents.
    pub centre: Vec3d,
    /// Size, in fractions of the chassis half-extents. A thruster's box
    /// half-extents; a rotor's `(radius, half-height, radius)`.
    pub half: Vec3d,
    /// What draws it, as a child of the marker — the tyre's own arrangement,
    /// for the tyre's own reason: the vehicle door writes the marker's rotation
    /// every step and leaves no slot to lay a primitive down in.
    pub primitive: crate::components::Primitive,
}

/// The launch's drawn parts: a hull, a deck, a wheelhouse and a bow.
const LAUNCH_PARTS: &[BodyPart] = &[
    // The hull below the waterline, inset all round: a boat's forefoot is
    // narrower than its deck, which is what the topsides above sit on.
    BodyPart {
        name: "hull",
        centre: Vec3d::new(0.0, -0.42, 0.0),
        half: Vec3d::new(0.88, 0.58, 0.92),
        primitive: crate::components::Primitive::Cube,
    },
    // The topsides — full beam and full length, and where the gunwale is.
    BodyPart {
        name: "topsides",
        centre: Vec3d::new(0.0, 0.14, -0.05),
        half: Vec3d::new(1.00, 0.22, 0.95),
        primitive: crate::components::Primitive::Cube,
    },
    // The wheelhouse, forward of amidships where a launch's is.
    BodyPart {
        name: "wheelhouse",
        centre: Vec3d::new(0.0, 0.64, 0.18),
        half: Vec3d::new(0.60, 0.36, 0.42),
        primitive: crate::components::Primitive::Cube,
    },
    // The bow, narrower and higher — the flare that keeps water off the deck.
    BodyPart {
        name: "bow",
        centre: Vec3d::new(0.0, 0.10, 0.82),
        half: Vec3d::new(0.56, 0.36, 0.18),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The launch's mounts: one screw, right aft and below the waterline.
const LAUNCH_MOUNTS: &[MountPart] = &[MountPart {
    name: "screw",
    kind: PartKind::Thruster,
    // Aft of the transom's inside face and under the hull, which is where a
    // shaft comes out — and, crucially, deep enough to stay wetted at rest.
    centre: Vec3d::new(0.0, -0.92, -0.86),
    half: Vec3d::new(0.16, 0.16, 0.12),
    primitive: crate::components::Primitive::Cylinder,
}];

/// The rotorcraft's drawn parts: a cabin, a tail boom, a fin and two skids.
const ROTORCRAFT_PARTS: &[BodyPart] = &[
    BodyPart {
        name: "cabin",
        centre: Vec3d::new(0.0, 0.02, 0.42),
        half: Vec3d::new(1.00, 0.84, 0.58),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "boom",
        centre: Vec3d::new(0.0, 0.30, -0.57),
        half: Vec3d::new(0.20, 0.20, 0.43),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "fin",
        centre: Vec3d::new(0.0, 0.60, -0.80),
        half: Vec3d::new(0.07, 0.40, 0.14),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "skid_left",
        centre: Vec3d::new(-0.62, -0.95, 0.05),
        half: Vec3d::new(0.07, 0.05, 0.70),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "skid_right",
        centre: Vec3d::new(0.62, -0.95, 0.05),
        half: Vec3d::new(0.07, 0.05, 0.70),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The rotorcraft's mounts: one main disc on the mast above the cabin.
///
/// A tail rotor is deliberately absent. [`RotorVehicle`] yaws with a torque
/// rather than with a second thrust, so a tail-rotor mount would be a marker the
/// model does not read — and a part nothing reads is exactly the kind of
/// decoration this table is not.
///
/// # THE MAST IS OVER THE CENTRE OF GRAVITY, and that is a measurement
///
/// The first cut put the hub 0.29 m forward of the chassis origin, where a
/// drawing would put it. The thrust acts at the hub and the mass acts at the
/// origin, so that offset is a permanent **nose-up** couple — and the attitude
/// hold is proportional, so it does not remove a steady one, it balances it at
/// an error.
///
/// What was MEASURED is the trajectory: the machine flew **33 m backwards in
/// the seven and a half seconds of its climb** and went on out over the bay,
/// where it descended into water a helicopter has no buoyancy for. The 4.3
/// degrees of standing trim is DERIVED from the offset against the hold gains
/// and the airframe inertia rather than read off a trace, and the two are not
/// the same kind of claim. With the mast over the CG the climb is vertical to
/// the centimetre, which is the measurement that closes it.
///
/// Real helicopters put the mast over the CG for exactly this reason. The
/// alternative — an integral term in the attitude hold — buys a trim the
/// airframe should not have needed.
const ROTORCRAFT_MOUNTS: &[MountPart] = &[MountPart {
    name: "rotor",
    kind: PartKind::Rotor,
    centre: Vec3d::new(0.0, 1.22, 0.0),
    // Wider than the machine — see `MountPart`'s own note.
    half: Vec3d::new(3.60, 0.04, 3.60),
    primitive: crate::components::Primitive::Cylinder,
}];

/// The vehicle silhouettes this engine draws.
///
/// **Axis-aligned boxes only, and that is a decision rather than a limit.**
/// `inf-dcc` refuses `Op::BevelEdges` on edges that share an endpoint (the P23
/// finding: on a right angle both offsets land at one position and the weld
/// fuses them into an edge used twice), so a bevelled car body is not something
/// the modelling kernel can produce today. A union of boxes is what it can, and
/// a union of boxes read at forty metres is a car.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VehicleBody {
    /// Three-box saloon: a lower body, a set-back greenhouse, a bonnet and a
    /// boot lid.
    #[default]
    Sedan,
    /// Two-volume pickup: a tall forward cab and an open rear bed with sides.
    Truck,
    /// Low two-seater: a long bonnet, a cabin set well back, and a spoiler
    /// (island wave VEH2a).
    Sports,
    /// Tall five-door with a full-length greenhouse and roof rails.
    Suv,
    /// Box body over a low chassis with a stepped cab in front — the delivery
    /// van of `docs/reference_videos/frames/driving/0014`.
    Van,
    /// **A motor launch** (wave VEH2c): a hull, a deck, a wheelhouse and a bow,
    /// with one screw right aft. The first body in this table with no wheels.
    Launch,
    /// **A light helicopter** (wave VEH2c): a cabin, a tail boom, a fin, two
    /// skids and a main disc on the mast.
    Rotorcraft,
}

/// The sedan's parts. `+Z` is forward, `+Y` is up.
const SEDAN_PARTS: &[BodyPart] = &[
    // The lower body, full width and length, sitting on the sills.
    BodyPart {
        name: "lower",
        centre: Vec3d::new(0.0, -0.5, 0.0),
        half: Vec3d::new(1.0, 0.5, 1.0),
        primitive: crate::components::Primitive::Cube,
    },
    // The greenhouse: narrower, shorter, set back from the nose.
    BodyPart {
        name: "cabin",
        centre: Vec3d::new(0.0, 0.5, -0.06),
        half: Vec3d::new(0.86, 0.5, 0.42),
        primitive: crate::components::Primitive::Cube,
    },
    // The bonnet, low and forward of the screen. It meets the lower body at
    // `y = 0` rather than floating above it — a gap between two boxes is a slot
    // you can see the road through.
    BodyPart {
        name: "bonnet",
        centre: Vec3d::new(0.0, 0.15, 0.62),
        half: Vec3d::new(0.94, 0.15, 0.36),
        primitive: crate::components::Primitive::Cube,
    },
    // The boot lid, a little higher than the bonnet — which is most of what
    // reads as "saloon" rather than "estate" from behind.
    BodyPart {
        name: "boot",
        centre: Vec3d::new(0.0, 0.18, -0.72),
        half: Vec3d::new(0.94, 0.18, 0.26),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The truck's parts.
const TRUCK_PARTS: &[BodyPart] = &[
    BodyPart {
        name: "lower",
        centre: Vec3d::new(0.0, -0.6, 0.0),
        half: Vec3d::new(1.0, 0.4, 1.0),
        primitive: crate::components::Primitive::Cube,
    },
    // A tall cab over the front axle.
    BodyPart {
        name: "cab",
        centre: Vec3d::new(0.0, 0.4, 0.5),
        half: Vec3d::new(0.94, 0.6, 0.42),
        primitive: crate::components::Primitive::Cube,
    },
    // The bed's floor and its two sides — the open volume is what makes it a
    // pickup rather than a van.
    BodyPart {
        name: "bed",
        centre: Vec3d::new(0.0, -0.1, -0.5),
        half: Vec3d::new(0.96, 0.1, 0.5),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "bed_left",
        centre: Vec3d::new(-0.88, 0.15, -0.5),
        half: Vec3d::new(0.09, 0.35, 0.5),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "bed_right",
        centre: Vec3d::new(0.88, 0.15, -0.5),
        half: Vec3d::new(0.09, 0.35, 0.5),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "headboard",
        centre: Vec3d::new(0.0, 0.2, 0.02),
        half: Vec3d::new(0.94, 0.4, 0.06),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The sports car's parts: a long bonnet, a cabin set well back, a spoiler.
const SPORTS_PARTS: &[BodyPart] = &[
    BodyPart {
        name: "lower",
        centre: Vec3d::new(0.0, -0.5, 0.0),
        half: Vec3d::new(1.0, 0.5, 1.0),
        primitive: crate::components::Primitive::Cube,
    },
    // Small, narrow, and a long way back — which is most of what reads as
    // "sports car" from any angle at all.
    BodyPart {
        name: "cabin",
        centre: Vec3d::new(0.0, 0.5, -0.20),
        half: Vec3d::new(0.80, 0.5, 0.34),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "bonnet",
        centre: Vec3d::new(0.0, 0.05, 0.62),
        half: Vec3d::new(0.96, 0.15, 0.36),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "boot",
        centre: Vec3d::new(0.0, 0.05, -0.72),
        half: Vec3d::new(0.94, 0.13, 0.26),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "spoiler",
        centre: Vec3d::new(0.0, 0.34, -0.92),
        half: Vec3d::new(0.86, 0.05, 0.07),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The SUV's parts: a tall greenhouse over the whole cabin, plus roof rails.
const SUV_PARTS: &[BodyPart] = &[
    BodyPart {
        name: "lower",
        centre: Vec3d::new(0.0, -0.55, 0.0),
        half: Vec3d::new(1.0, 0.45, 1.0),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "cabin",
        centre: Vec3d::new(0.0, 0.42, -0.14),
        half: Vec3d::new(0.92, 0.52, 0.55),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "bonnet",
        centre: Vec3d::new(0.0, 0.12, 0.66),
        half: Vec3d::new(0.94, 0.22, 0.32),
        primitive: crate::components::Primitive::Cube,
    },
    // The rails are what reach the top of the hull, so the topmost part of an
    // SUV is a 16 cm rail rather than its whole roof — which is what keeps the
    // silhouette test's anti-brick clause meaningful on a tall car.
    BodyPart {
        name: "rail_left",
        centre: Vec3d::new(-0.78, 0.95, -0.10),
        half: Vec3d::new(0.08, 0.05, 0.50),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "rail_right",
        centre: Vec3d::new(0.78, 0.95, -0.10),
        half: Vec3d::new(0.08, 0.05, 0.50),
        primitive: crate::components::Primitive::Cube,
    },
];

/// The van's parts: a low chassis, a stepped cab, a box body and an inset roof.
const VAN_PARTS: &[BodyPart] = &[
    BodyPart {
        name: "chassis",
        centre: Vec3d::new(0.0, -0.80, 0.0),
        half: Vec3d::new(1.0, 0.20, 1.0),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "cab",
        centre: Vec3d::new(0.0, 0.05, 0.72),
        half: Vec3d::new(0.92, 0.55, 0.26),
        primitive: crate::components::Primitive::Cube,
    },
    BodyPart {
        name: "box",
        centre: Vec3d::new(0.0, 0.06, -0.30),
        half: Vec3d::new(0.96, 0.62, 0.68),
        primitive: crate::components::Primitive::Cube,
    },
    // Inset on every axis, which is both what a van's roof actually looks like
    // and what keeps the topmost part from being the whole vehicle.
    BodyPart {
        name: "roof",
        centre: Vec3d::new(0.0, 0.84, -0.30),
        half: Vec3d::new(0.88, 0.16, 0.58),
        primitive: crate::components::Primitive::Cube,
    },
];

impl VehicleBody {
    /// Every family, in the canonical order.
    pub const ALL: [VehicleBody; 7] = [
        VehicleBody::Sedan,
        VehicleBody::Truck,
        VehicleBody::Sports,
        VehicleBody::Suv,
        VehicleBody::Van,
        VehicleBody::Launch,
        VehicleBody::Rotorcraft,
    ];

    /// **The bodies that belong at a kerb** — the named civilian sub-list
    /// (wave VEH2c).
    ///
    /// THE KERB TRAP, and it is written down in this repository twice already:
    /// `inf_ecs::traffic::catalogue_row` draws a parked car's silhouette
    /// **uniformly over a list**, so a sixth family in that list puts one at
    /// every sixth kerb slot in every town. Wave EMS1 avoided it by borrowing
    /// civilian bodies for the emergency fleet and left the remedy written down
    /// for the day a family really was new: *"if a body variant is ever added,
    /// `catalogue_row` gains a named CIVILIAN sub-list in the same commit."*
    ///
    /// This is that commit and this is that list. A launch and a helicopter are
    /// not cars, and a town whose kerbs held boats would be the exact defect the
    /// note predicted.
    pub const CIVILIAN: [VehicleBody; 5] = [
        VehicleBody::Sedan,
        VehicleBody::Truck,
        VehicleBody::Sports,
        VehicleBody::Suv,
        VehicleBody::Van,
    ];

    /// The stable name a catalogue row names this family by.
    pub fn name(self) -> &'static str {
        match self {
            VehicleBody::Sedan => "sedan",
            VehicleBody::Truck => "truck",
            VehicleBody::Sports => "sports",
            VehicleBody::Suv => "suv",
            VehicleBody::Van => "van",
            VehicleBody::Launch => "launch",
            VehicleBody::Rotorcraft => "rotorcraft",
        }
    }

    /// Whether this family is drawn on **wheels** (wave VEH2c).
    ///
    /// Derived from the mounts rather than authored as a flag, so a family
    /// cannot claim to be one thing and emit another: a body with mounts is a
    /// craft the parts recogniser will find, and `rig_of`'s WHEELS-WIN rule
    /// means the two can never both be true of one rig.
    pub fn wheeled(self) -> bool {
        self.mounts().is_empty()
    }

    /// This family's **mounts** — its thrusters and rotors (wave VEH2c).
    ///
    /// Empty for every road vehicle, which is every family that existed before
    /// this wave, so nothing already committed changes by a byte.
    pub fn mounts(self) -> &'static [MountPart] {
        match self {
            VehicleBody::Sedan
            | VehicleBody::Truck
            | VehicleBody::Sports
            | VehicleBody::Suv
            | VehicleBody::Van => &[],
            VehicleBody::Launch => LAUNCH_MOUNTS,
            VehicleBody::Rotorcraft => ROTORCRAFT_MOUNTS,
        }
    }

    /// The family a catalogue row's `body = "…"` means, or `None`.
    ///
    /// Exhaustive over [`ALL`](Self::ALL) with no wildcard fallback: a spelling
    /// this does not know is a **refusal**, not a silent sedan.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }

    /// This family's parts.
    pub fn parts(self) -> &'static [BodyPart] {
        match self {
            VehicleBody::Sedan => SEDAN_PARTS,
            VehicleBody::Truck => TRUCK_PARTS,
            VehicleBody::Sports => SPORTS_PARTS,
            VehicleBody::Suv => SUV_PARTS,
            VehicleBody::Van => VAN_PARTS,
            VehicleBody::Launch => LAUNCH_PARTS,
            VehicleBody::Rotorcraft => ROTORCRAFT_PARTS,
        }
    }
}

/// The `Guid` a vehicle's drawn part carries — a pure function of the chassis
/// and the part's own name.
///
/// The synthetic-guid rule this repository already uses for door leaves, PCG
/// doorways, structure colliders, fracture chunks and building modules: the id
/// is derived from *what the thing is*, so the same car authored twice produces
/// the same entity ids and a level's bytes do not depend on the order a
/// generator happened to spawn things in.
pub fn body_part_guid(chassis: Uuid, part: &str) -> Uuid {
    let mut x = VEHICLE_PART_SALT ^ chassis.as_u128();
    for b in part.as_bytes() {
        x = x.rotate_left(11) ^ (*b as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

// ── what a car is made of (wave VEH2b) ──────────────────────────────────────

/// The tyres' colour — dark, unlit-looking rubber, so a wheel reads against the
/// body whatever the body is painted.
pub const TYRE_COLOR: crate::math::Color = crate::math::Color::new(0.07, 0.07, 0.08, 1.0);

/// The chassis's own angular damping.
///
/// A car does not spin on its own axis for want of damping; the suspension
/// supplies the rest of the resistance. (P29.7's number, lifted here with the
/// rest of the recipe.)
pub const CHASSIS_ANGULAR_DAMPING: f64 = 0.5;

/// The chassis collider's friction.
pub const CHASSIS_FRICTION: f64 = 0.5;

/// **Where one vehicle goes and what it looks like** — the half of a rig that is
/// not the [`VehicleDef`].
#[derive(Clone, Debug, PartialEq)]
pub struct RigSpawn {
    /// The entity name in the outliner.
    pub name: String,
    /// The **chassis origin** in world space.
    pub at: DVec3,
    /// Heading about `+Y`, degrees. `0` faces `+Z`, which is forward.
    pub yaw_deg: f64,
    /// The body colour. The tyres are always [`TYRE_COLOR`].
    pub paint: crate::math::Color,
    /// The engine loop's `.inf_audio` clip. `None` is a silent voice, and the
    /// *command stream* — which is what PIE == shipping compares — is the same
    /// either way.
    pub clip: Option<Uuid>,
    /// **Whether this car gets an engine emitter at all** (wave VEH2b).
    ///
    /// `true` for a car somebody authored. **`false` for traffic**, and the
    /// reason is VEH2a's own carried item 5, quoted: *"the engine's emitter
    /// still does not MOVE — no `AudioCommand::SetPosition`, so a driving car's
    /// engine is spatialized where its `Play` was issued."* One car that a
    /// player drives away from its own engine noise is a bug you notice once; a
    /// dozen traffic cars each leaving a stationary engine loop behind at the
    /// kerb they pulled out of is a town that sounds broken.
    ///
    /// So traffic is **silent until the emitter can follow the car**, which is
    /// the honest order to do these two things in, and it is named in the
    /// wave's carried list rather than hidden. It also stops a dozen cars
    /// flooding the shipped player's bounded `audio_command_log` — which the
    /// island's own drive gate caught, by losing the one `Play` it exists to
    /// count off the front of an evicting ring.
    pub engine_voice: bool,
    /// **The livery this vehicle wears** (wave EMS1), or `None` for a car
    /// painted one colour — which is every civilian vehicle and all of traffic.
    ///
    /// See [`Livery`] for why this rides the spawn rather than the parts table.
    pub livery: Option<&'static Livery>,
}

/// **One entity of a vehicle rig, as a DESCRIPTION** — wave VEH2b.
///
/// # Why a description and not a spawn
///
/// Two callers build a car and they build it into two different things. The
/// island's generator writes one into a `SceneDoc`, through
/// `create_with_guid`, which tracks creation order and an undo step; traffic
/// materializes one straight into an `EcsWorld` sixty times a second, where
/// there is no document and nothing to undo. Before this wave only the first
/// existed, and the second would have been a second recipe for what a car is —
/// which is the shape this tree has caught and refused a dozen times (the P22.3
/// "one door for three paths", the P29.6 A14 restated selection rule).
///
/// So the recipe is a value: `rig_nodes` says what entities a car is and what
/// is on each of them, and the two callers differ only in how they create an
/// entity. `inf_editor_core::vehicle::spawn_vehicle` walks it into a document
/// and [`spawn_rig`] walks it into a world, and the day a car grows a wing
/// mirror it grows one in both.
#[derive(Clone, Debug, PartialEq)]
pub struct RigNode {
    /// The entity's `Guid` — [`body_part_guid`]'s, except for the chassis,
    /// which is the caller's.
    pub guid: Uuid,
    /// The outliner name.
    pub name: String,
    /// `None` for the chassis, the chassis for a panel or a wheel, the wheel
    /// for a tyre. Always a guid this same list already introduced.
    pub parent: Option<Uuid>,
    /// Its local transform.
    pub transform: crate::components::Transform,
    pub body: Option<crate::components::RigidBody3D>,
    pub collider: Option<crate::components::Collider3D>,
    pub mesh: Option<crate::components::MeshRef>,
    pub material: Option<crate::components::Material>,
    /// The tuning, on the chassis only (scene v25 onward).
    pub class: Option<crate::components::VehicleClass>,
    /// The engine loop's emitter, on the chassis only.
    pub audio: Option<crate::components::AudioSource>,
    /// **Flotation**, on the chassis only (wave VEH2c) — what makes a hull a
    /// boat rather than a box that sinks.
    ///
    /// On the recipe rather than left to the island's spawner, because a launch
    /// that floats in one caller and sinks in another is two boats and this is
    /// the one place a vehicle's entities are described.
    pub buoyancy: Option<crate::components::Buoyancy>,
}

/// **Every entity one vehicle is made of, in creation order.**
///
/// * the **chassis** — a `Dynamic` `RigidBody3D`, a box `Collider3D` at the
///   def's half-extents and density, the def's
///   [`VehicleClass`](crate::components::VehicleClass) and a looping spatial
///   [`AudioSource`](crate::components::AudioSource) for the engine;
/// * the **body** — one child per [`BodyPart`] of the def's family, drawing its
///   own primitive at its own size;
/// * the **wheels** — four sphere **sensors** with no body of their own, which
///   is what [`wheel_of`] recognises, each with a **tyre** child that draws the
///   cylinder the vehicle door's own rotation write cannot lay down.
///
/// The order is load-bearing: it is the order the island's committed `.inf_lvl`
/// was authored in, and `SceneDoc` writes its entities in creation order. A
/// caller that walked this list backwards would produce a level with different
/// bytes and the same contents.
pub fn rig_nodes(chassis: Uuid, def: &VehicleDef, spawn: &RigSpawn) -> Vec<RigNode> {
    rig_nodes_at(chassis, def, spawn, true)
}

/// [`rig_nodes`] with the **wheels** made optional (wave VEH2b).
///
/// `wheels = false` answers the chassis and its body panels alone, and drops
/// the [`VehicleClass`](crate::components::VehicleClass) with them. That is not
/// a level-of-detail convenience: it is what makes a distant traffic car
/// *structurally* unable to be simulated. [`rig_of`] needs wheel sensors to
/// answer a [`VehicleRig`], the bridge needs a rig to build a `RaycastVehicle`,
/// and `step_vehicles` walks the bridge's vehicles — so a car built without
/// wheels cannot be reached by the vehicle phase even by accident.
///
/// The visible cost is stated: **past
/// [`inf_ecs::traffic::TRAFFIC_FULL_M`](crate::traffic::TRAFFIC_FULL_M) a
/// parked car is drawn without its wheels**, sitting on its floor pan. At 64 m
/// a 0.35 m wheel is a couple of pixels; what it buys is eight entities a car
/// against fourteen, over a population sized by a settlement's kerbs.
pub fn rig_nodes_at(
    chassis: Uuid,
    def: &VehicleDef,
    spawn: &RigSpawn,
    wheels: bool,
) -> Vec<RigNode> {
    use crate::components::{
        AudioSource, BodyKind3D, Collider3D, ColliderShape3DKind, Material, MeshRef, Primitive,
        RigidBody3D, Transform,
    };
    let h = def.half_extents;
    // **What "simulated" means, once** (wave VEH2c). A road car is simulated
    // when it has wheels; a launch and a rotorcraft have none and are simulated
    // whenever their family names mounts, because the parts recogniser will find
    // them and the bridge will build a class over them. The `wheels` flag keeps
    // its exact VEH2b meaning for every family that has wheels — a `Near` car
    // built with `wheels = false` is still a kinematic box that cannot be
    // stepped — and a mounted craft ignores it, having no wheels to drop.
    let simulated = (wheels && def.body.wheeled()) || !def.body.mounts().is_empty();
    let part_guid = |part: &str| body_part_guid(chassis, part);
    let mut out: Vec<RigNode> = Vec::with_capacity(1 + def.body.parts().len() + 8);

    out.push(RigNode {
        guid: chassis,
        name: spawn.name.clone(),
        parent: None,
        transform: Transform {
            translation: Vec3d::from_dvec3(spawn.at),
            rotation: Vec3d::new(0.0, spawn.yaw_deg, 0.0),
            scale: Vec3d::ONE,
        },
        body: Some(RigidBody3D {
            // A car the vehicle phase drives is a DYNAMIC body under four
            // suspension rays; a car its clock draws is a KINEMATIC one that
            // the solver pushes nothing through. The two travel together with
            // the wheels for one reason: they are the same decision about
            // whether this car is being simulated.
            kind: if simulated {
                BodyKind3D::Dynamic
            } else {
                BodyKind3D::Kinematic
            },
            angular_damping: CHASSIS_ANGULAR_DAMPING,
            ..Default::default()
        }),
        collider: Some(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: h,
            density: def.density_kg_m3,
            friction: CHASSIS_FRICTION,
            ..Default::default()
        }),
        mesh: None,
        material: None,
        // The tuning rides with the wheels: a body with no wheels is not a
        // vehicle, and a `VehicleClass` on one would be a class the bridge
        // installs on a rig that does not exist.
        class: simulated.then_some(def.class),
        audio: spawn.engine_voice.then(|| AudioSource {
            clip: spawn.clip,
            looping: true,
            spatial: true,
            // Not autoplay: the VEH1a engine loop emits the `Play` itself, on
            // the step the vehicle first appears in an outcome, and two paths
            // both starting one voice is two `Play`s for one source.
            autoplay: false,
            ..Default::default()
        }),
        // Flotation, if this family floats. `Buoyancy` is opt-in and scene-v18,
        // so a road car emits none and nothing already committed moves.
        buoyancy: (simulated && def.buoyancy_density_kg_m3 > 0.0).then(|| {
            crate::components::Buoyancy {
                density_kg_m3: def.buoyancy_density_kg_m3,
                linear_drag: def.buoyancy_linear_drag.max(0.0),
                ..Default::default()
            }
        }),
    });

    // **The body, and its livery** (wave EMS1). One loop over the family's own
    // parts followed by the livery's own, because the livery's parts are parts:
    // they take their guids from `body_part_guid` and their transform from the
    // same three multiplications. A part the livery does not name keeps
    // `spawn.paint`, which is every part of every civilian vehicle and all of
    // traffic — so nothing that predates this wave moves a byte.
    let body_parts = def.body.parts().iter().map(|p| {
        let paint = spawn.livery.and_then(|l| l.part(p.name));
        (*p, paint)
    });
    let livery_parts = spawn
        .livery
        .map(|l| l.extra)
        .unwrap_or(&[])
        .iter()
        .map(|(p, paint)| (*p, Some(*paint)));
    for (part, paint) in body_parts.chain(livery_parts) {
        // Built from the unliveried material and then overwritten, so a part
        // with no override is BYTE-IDENTICAL to what this loop wrote before the
        // livery existed rather than merely equal to it by inspection.
        let mut material = Material {
            base_color: spawn.paint,
            metallic: 0.35,
            roughness: 0.42,
            ..Default::default()
        };
        if let Some(p) = paint {
            material.base_color = p.base_color;
            material.emissive = p.emissive;
            material.emissive_intensity = p.emissive_intensity;
        }
        out.push(RigNode {
            guid: part_guid(part.name),
            name: part.name.to_string(),
            parent: Some(chassis),
            transform: Transform {
                translation: Vec3d::new(
                    part.centre.x * h.x,
                    part.centre.y * h.y,
                    part.centre.z * h.z,
                ),
                rotation: Vec3d::ZERO,
                // The built-in primitives span ±0.5, so a part's SCALE is its
                // full extent — twice its half-extent.
                scale: Vec3d::new(
                    2.0 * part.half.x * h.x,
                    2.0 * part.half.y * h.y,
                    2.0 * part.half.z * h.z,
                ),
            },
            body: None,
            collider: None,
            mesh: Some(MeshRef {
                primitive: part.primitive,
                asset: None,
            }),
            material: Some(material),
            class: None,
            audio: None,
            buoyancy: None,
        });
    }

    // ── the MOUNTS (wave VEH2c): a thruster or a rotor, each a sensor the
    //    parts recogniser finds, each with a drawn child the way a wheel has a
    //    tyre — and for the wheel's own reason, that the vehicle door writes the
    //    marker's rotation every step and leaves no slot to lay a primitive
    //    down in.
    for mount in def.body.mounts() {
        let marker = part_guid(mount.name);
        let size = Vec3d::new(mount.half.x * h.x, mount.half.y * h.y, mount.half.z * h.z);
        let collider = match mount.kind {
            PartKind::Thruster => Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: size,
                sensor: true,
                ..Default::default()
            },
            PartKind::Rotor => Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                radius: size.x,
                half_extents: Vec3d::new(size.x, size.y, size.z),
                sensor: true,
                ..Default::default()
            },
        };
        out.push(RigNode {
            guid: marker,
            name: mount.name.to_string(),
            parent: Some(chassis),
            transform: Transform::from_translation(
                Vec3d::new(
                    mount.centre.x * h.x,
                    mount.centre.y * h.y,
                    mount.centre.z * h.z,
                )
                .to_dvec3(),
            ),
            body: None,
            collider: Some(collider),
            mesh: None,
            material: None,
            class: None,
            audio: None,
            buoyancy: None,
        });
        out.push(RigNode {
            guid: part_guid(&format!("{}_drawn", mount.name)),
            name: format!("{} Blade", mount.name),
            parent: Some(marker),
            transform: Transform {
                translation: Vec3d::ZERO,
                rotation: Vec3d::ZERO,
                scale: Vec3d::new(2.0 * size.x, 2.0 * size.y, 2.0 * size.z),
            },
            body: None,
            collider: None,
            mesh: Some(MeshRef {
                primitive: mount.primitive,
                asset: None,
            }),
            material: Some(Material {
                base_color: TYRE_COLOR,
                metallic: 0.2,
                roughness: 0.7,
                ..Default::default()
            }),
            class: None,
            audio: None,
            buoyancy: None,
        });
    }

    if !wheels || !def.body.wheeled() {
        return out;
    }
    let r = def.wheel_radius_m;
    for (i, mount) in def.wheel_mounts().into_iter().enumerate() {
        let wheel = part_guid(&format!("wheel{i}"));
        out.push(RigNode {
            guid: wheel,
            name: "Wheel".to_string(),
            parent: Some(chassis),
            transform: Transform::from_translation(mount.to_dvec3()),
            body: None,
            collider: Some(Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: r,
                // A wheel is a SENSOR and has no body: that is the whole of
                // `wheel_of`, and the bridge consumes it rather than mirroring
                // it into rapier.
                sensor: true,
                ..Default::default()
            }),
            mesh: None,
            material: None,
            class: None,
            audio: None,
            buoyancy: None,
        });
        // The tyre is a child of the wheel because `step_vehicles` writes the
        // wheel's rotation every step as euler `(spin, steer, 0)` — there is no
        // roll slot left to lay a `+Y`-axis cylinder on its side with.
        out.push(RigNode {
            guid: part_guid(&format!("tyre{i}")),
            name: "Tyre".to_string(),
            parent: Some(wheel),
            transform: Transform {
                translation: Vec3d::ZERO,
                rotation: Vec3d::new(0.0, 0.0, TYRE_ROLL_DEG),
                scale: Vec3d::new(2.0 * r, 2.0 * r * TYRE_WIDTH_FRAC, 2.0 * r),
            },
            body: None,
            collider: None,
            mesh: Some(MeshRef {
                primitive: Primitive::Cylinder,
                asset: None,
            }),
            material: Some(Material {
                base_color: TYRE_COLOR,
                metallic: 0.0,
                roughness: 0.9,
                ..Default::default()
            }),
            class: None,
            audio: None,
            buoyancy: None,
        });
    }
    out
}

/// **Build a rig straight into a world** — the RUNTIME door (wave VEH2b's
/// traffic), over the same [`rig_nodes`] recipe the authoring door walks.
///
/// Returns the chassis guid. A guid the world already holds is **left alone**
/// rather than overwritten, for `crowd::add_agents`' reason: `spawn_with_guid`
/// does not refuse a key the world already has, and a traffic car that
/// overwrote an authored one would leave the author's entity unreachable while
/// it still existed.
pub fn spawn_rig(world: &mut EcsWorld, chassis: Uuid, def: &VehicleDef, spawn: &RigSpawn) -> Uuid {
    spawn_rig_at(world, chassis, def, spawn, true)
}

/// [`spawn_rig`] over [`rig_nodes_at`] — the door a traffic car's tier uses.
///
/// A `Near` car is built with `wheels = false` and is therefore invisible to
/// `step_vehicles`; a `Full` one is built whole. The transition between the two
/// is a despawn and a respawn rather than a component edit, because the guids
/// are content-derived and a respawn is byte-identical — and because a car
/// crossing a tier boundary is a thing that happens to one car every few
/// hundred steps, which is the budget the crowd takes its own pose digest on.
pub fn spawn_rig_at(
    world: &mut EcsWorld,
    chassis: Uuid,
    def: &VehicleDef,
    spawn: &RigSpawn,
    wheels: bool,
) -> Uuid {
    for node in rig_nodes_at(chassis, def, spawn, wheels) {
        if world.entity_of(node.guid).is_some() {
            continue;
        }
        let parent = node.parent.and_then(|p| world.entity_of(p));
        let e = world.spawn_with_guid(node.guid, &node.name, parent);
        let mut em = world.world_mut().entity_mut(e);
        em.insert(node.transform);
        if let Some(c) = node.body {
            em.insert(c);
        }
        if let Some(c) = node.collider {
            em.insert(c);
        }
        if let Some(c) = node.mesh {
            em.insert(c);
        }
        if let Some(c) = node.material {
            em.insert(c);
        }
        if let Some(c) = node.class {
            em.insert(c);
        }
        if let Some(c) = node.audio {
            em.insert(c);
        }
        if let Some(c) = node.buoyancy {
            em.insert(c);
        }
    }
    world.mark_dirty();
    chassis
}

/// **Take a rig back out of a world** — the other half of [`spawn_rig`], and
/// the door a traffic car's `Dormant` tier goes through.
///
/// Returns how many of the recipe's entities **were in the world**, counted
/// before anything is despawned.
///
/// Counted first on purpose. `EcsWorld::despawn` walks the hierarchy, so taking
/// the chassis takes its panels, its wheels and its tyres with it — a loop that
/// counted as it went would answer **one** for a whole car, which reads as "a
/// rig went out" and is a number nothing can hold against the recipe. The
/// count as written is the one an arm can assert is the recipe's own length,
/// and it goes to zero on a second call.
pub fn despawn_rig(world: &mut EcsWorld, chassis: Uuid, def: &VehicleDef) -> usize {
    let spawn = RigSpawn {
        name: String::new(),
        at: DVec3::ZERO,
        yaw_deg: 0.0,
        paint: crate::math::Color::WHITE,
        clip: None,
        engine_voice: false,
        // A livery adds parts, and this list is only used to COUNT what was
        // present: `EcsWorld::despawn` walks the hierarchy, so taking the
        // chassis takes a light bar with it whatever the recipe says.
        livery: None,
    };
    let nodes = rig_nodes(chassis, def, &spawn);
    let present = nodes
        .iter()
        .filter(|n| world.entity_of(n.guid).is_some())
        .count();
    for node in &nodes {
        if let Some(e) = world.entity_of(node.guid) {
            world.despawn(e);
        }
    }
    present
}

/// Where a vehicle's chassis origin goes if its wheels are to touch `ground`
/// with the suspension at full extension — the placement an author makes.
///
/// Lifted from `inf_editor_core::vehicle` at wave VEH2b so the runtime spawner
/// and the authoring one place a car the same way. "Wheel drop plus wheel
/// radius" written out twice is two chances to write it once with the sign
/// wrong.
///
/// # A craft with no wheels rests on its own hull (wave VEH2c)
///
/// A helicopter stands on its skids and a launch on a trailer, and neither has
/// a wheel drop to measure from. The honest answer is the collider's own
/// half-height, which is the same number `chassis_of` derives the seat from —
/// so the two cannot disagree about how tall the machine is.
pub fn resting_origin_y(def: &VehicleDef, ground_y: f64) -> f64 {
    if def.body.wheeled() {
        ground_y - def.wheel_drop_m + def.wheel_radius_m
    } else {
        ground_y + def.half_extents.y
    }
}

/// Where a hull's chassis origin goes so that it floats in **equilibrium** on a
/// surface at `water_y` — the placement an author makes for a boat (wave VEH2c).
///
/// Archimedes, and nothing else: a hull of density `rho` in fluid of density
/// `rho_f` floats with `rho / rho_f` of its depth under, so its origin sits
/// `half_height * (1 - 2 * rho / rho_f)` above the surface. Spawning a boat
/// anywhere else is spawning a splash, and the first seconds of every arm that
/// followed would be measuring one.
///
/// A craft that does not float (`buoyancy_density_kg_m3 <= 0`) is placed with
/// its hull ON the surface, which is what a box in a puddle does.
pub fn floating_origin_y(def: &VehicleDef, water_y: f64) -> f64 {
    let fluid = crate::components::Buoyancy::default().fluid_density_kg_m3;
    let rho = def.buoyancy_density_kg_m3;
    if !(rho > 0.0 && fluid > 0.0) {
        return water_y + def.half_extents.y;
    }
    water_y + def.half_extents.y * (1.0 - 2.0 * (rho / fluid).clamp(0.0, 1.0))
}

/// Metres per second to knots — the unit a boat's speed is read in.
///
/// SI everywhere is this repository's rule and this does not break it: the
/// number stored, stepped and asserted on is always m/s, and this conversion
/// happens once, in a `format!`, at the boundary where a human reads it. It is
/// `speed_limit_kmh`'s own precedent (VEH2b) with a different destination.
pub const KNOTS_PER_MPS: f64 = 1.943_844_492_440_605;

/// **What a CRAFT's own instruments say** (wave VEH2c) — the one door, chosen
/// by what the rig is made of.
///
/// A car reads a speedometer and a gear, a boat reads knots and a telegraph, an
/// aircraft reads a speed and a height. Dispatched on the rig's own parts rather
/// than on a kind field: the parts are what the scene authored and what the
/// class was chosen from, so a readout cannot disagree with the model driving
/// the machine.
///
/// `height_m` is height above the ground, which only an aircraft draws and only
/// a host can measure — `None` reads as a machine that does not know, and prints
/// the speed alone rather than a lie.
pub fn craft_readout(rig: &VehicleRig, speed_mps: f64, gear: i32, height_m: Option<f64>) -> String {
    let speed = if speed_mps.is_finite() {
        speed_mps.abs()
    } else {
        0.0
    };
    if rig.parts_of(PartKind::Rotor).next().is_some() {
        let kmh = (speed * 3.6).round();
        return match height_m.filter(|h| h.is_finite()) {
            Some(h) => format!("{kmh:.0} km/h    ALT {:.0} m", h.max(0.0).round()),
            None => format!("{kmh:.0} km/h"),
        };
    }
    if rig.parts_of(PartKind::Thruster).next().is_some() {
        let kn = speed * KNOTS_PER_MPS;
        // A telegraph, not a gearbox — see `HullVehicle::gear`.
        let telegraph = if gear > 0 {
            "AHEAD"
        } else if gear < 0 {
            "ASTERN"
        } else {
            "STOP"
        };
        return format!("{kn:.1} kn    {telegraph}");
    }
    drive_readout(speed_mps, gear)
}

/// **What a driver's own instruments say** — speed and gear, as one line.
///
/// The whole of wave VEH2b's HUD, and it is in Ring 0 rather than in the
/// player's window for the reason every other decision in this crate is: the
/// window cannot be tested and this can. What the window does is read the
/// numbers and hand them here.
///
/// Speed is a MAGNITUDE, so a car rolling backwards reads a positive number and
/// an `R` beside it, which is what a speedometer does. Non-finite reads zero
/// rather than `NaN km/h`.
pub fn drive_readout(speed_mps: f64, gear: i32) -> String {
    let kmh = if speed_mps.is_finite() {
        (speed_mps.abs() * 3.6).round()
    } else {
        0.0
    };
    format!("{kmh:.0} km/h    {}", gear_label(gear))
}

/// The letter or number on the gate for one gear.
///
/// `-1` and below is reverse, `0` is neutral, and everything above is its own
/// number up to [`MAX_GEARS`]; past that it is the top gear's number, because a
/// gearbox this engine cannot build is not a case a readout should invent a
/// symbol for.
pub fn gear_label(gear: i32) -> &'static str {
    const FORWARD: [&str; MAX_GEARS] = ["1", "2", "3", "4", "5", "6", "7", "8"];
    if gear <= -1 {
        "R"
    } else if gear == 0 {
        "N"
    } else {
        FORWARD[(gear as usize - 1).min(MAX_GEARS - 1)]
    }
}

/// The share of its target's force an aid actually asks for.
///
/// Two per cent under, and the two per cent is load-bearing: the aid's cap is one
/// term of a resist budget that also carries rolling resistance and engine
/// braking, so a cap set at exactly the achievable force is a wheel one newton
/// over its limit.
pub const AID_CAP_MARGIN: f64 = 0.98;

/// **The torque one wheel may be given, or take, at its aid's target slip.**
///
/// Both driver aids that modulate torque — traction control on the engine's, ABS
/// on the brake's — ask one question: *what is the most this contact patch can
/// transmit if the slip is to stay at `target_slip`?* And that question has an
/// exact answer, because the tyre curve is right here: the normalized force at
/// the target slip, times µ at this load, times the radius.
///
/// # Why FEED-FORWARD, and the measurement that decided it
///
/// The first two cuts were feedback controllers on last step's slip ratio, and
/// both failed for the same reason. A wheel with 1 kg·m² of inertia under 4 kN·m
/// of drive torque changes speed by **66 rad/s in a single 60 Hz step**: a
/// controller that measures slip and then reacts is always exactly one step
/// behind an event that is over inside one step. Measured on the coupe's launch:
/// `tc / slip` applied directly oscillated the crank between **416 N·m and 109**
/// and the revs between 4 138 and 1 827; integrating the error instead settled no
/// better, because the input it integrates alternates between 0 (the wheel stuck)
/// and 5 (the wheel spinning). Both took **8.7 s** to 100 km/h on an axle that
/// can carry the car there in under four.
///
/// A real traction-control system does not work that way either: it knows the
/// wheel load and it knows the surface, and it *limits the torque request*. So
/// does this — one clamp, no state, no lag, settled on the first step.
pub fn aid_torque_cap_nm(
    tuning: &VehicleTuning,
    target_slip: f64,
    load_n: f64,
    static_load_n: f64,
    radius_m: f64,
) -> f64 {
    let peak = if tuning.tyre_long_peak_slip.is_finite() {
        tuning.tyre_long_peak_slip.max(1e-4)
    } else {
        1e-4
    };
    let load = if load_n.is_finite() {
        load_n.max(0.0)
    } else {
        0.0
    };
    let mu = load_sensitive_mu(
        tuning.longitudinal_grip,
        load,
        static_load_n,
        tuning.tyre_load_sensitivity,
    );
    // **The target is clamped to the tyre's own peak, and held just under it.**
    //
    // Past the peak the curve FALLS, so an aid aiming there is standing on an
    // unstable operating point: one newton too much and the wheel gives up a
    // little grip, which lets the brake win, which takes more slip, which gives
    // up more grip. Measured with an `abs_slip` of 0.15 against a peak slip of
    // 0.12 — a target only a quarter past the peak — the wheel still spent
    // **82 % of the stop fully locked**. An aid cannot usefully ask for more slip
    // than its tyre's best, and [`AID_CAP_MARGIN`] is what keeps it on the rising
    // side of it.
    let frac = tyre_curve(
        (target_slip / peak).min(1.0),
        tuning.tyre_long_rise_bias,
        tuning.tyre_slide_frac,
    );
    (AID_CAP_MARGIN * frac * mu * load * radius_m.max(1e-3)).max(0.0)
}

/// How far the yaw rate may stray from what the steering asked for, rad/s,
/// before stability control does anything.
///
/// A dead band, and not an optional one: a car is always a little off its own
/// bicycle-model reference, and an aid with no tolerance would be braking a
/// wheel on a motorway.
pub const ESC_YAW_TOLERANCE_RAD_S: f64 = 0.12;

/// How hard stability control brakes, newtons per rad/s of yaw error past the
/// tolerance, at full [`VehicleTuning::stability_control`].
///
/// Sized against a road car's brake: 9 000 N per rad/s means half a radian a
/// second of error spends about a third of a 12 kN brake budget on one wheel,
/// which is a correction a driver feels as the car tightening rather than as a
/// hand on the wheel.
pub const ESC_GAIN_N_PER_RAD_S: f64 = 9_000.0;

/// The speed difference, rad/s, a differential's lock ramps in over.
///
/// Two radians a second — about 0.7 m/s of wheel speed on a road tyre. A lock
/// that engaged on any difference at all would be thrown between two torque
/// splits by the last bit of a float; a lock that needed a big one would never
/// engage on the surface it exists for.
pub const DIFF_SPEED_BAND: f64 = 2.0;

/// The front share a catalogue row's `drivetrain = "awd"` means.
///
/// Forty per cent, which is the rear-biased split nearly every road-going
/// all-wheel-drive system runs, and a row that wants a different one says
/// `front_torque_split` and gets exactly what it asks for.
pub const AWD_FRONT_SPLIT: f64 = 0.4;

/// A tyre's width as a fraction of its own radius.
///
/// One number for the whole engine rather than a per-class knob, on
/// `GLAZING_GLOW`'s argument: a tyre is a tyre, and a second value would be a
/// second chance for one car's wheels to look wrong for no authored reason.
pub const TYRE_WIDTH_FRAC: f64 = 0.62;

/// The local **roll**, in degrees about `Z`, that lays a `Primitive::Cylinder`
/// on its side so its axis is the wheel's axle.
///
/// # Why the tyre is a child of the wheel and not the wheel itself
///
/// `inf_physics::d3::step_vehicles` writes the wheel entity's rotation every
/// step as euler `(spin, steer, 0)` — pitch about its axle, yaw for the steer,
/// **and a roll of zero**. A cylinder's axis is `+Y`, so a tyre drawn on the
/// wheel entity would stand on end and there is no third euler slot to lay it
/// down with. One more entity, whose local transform is authored once and never
/// written, is the whole fix: the parent supplies the spin and the steer, the
/// child supplies the roll, and the door's write is untouched.
pub const TYRE_ROLL_DEG: f64 = 90.0;

// ── the catalogue (island wave VEH1a) ───────────────────────────────────────

/// **A vehicle class as content** — the geometry a spawner builds and the
/// tuning it installs, in one row.
///
/// # Zero schema, on the `WeaponDef` precedent
///
/// This type derives **no `Serialize`**, exactly as `crate::weapon::WeaponDef`
/// does not, and for the same reason: adding a field to it costs no schema bump
/// anywhere, because nothing ever writes one to a file. A catalogue is TOML an
/// author edits and a *generator* consumes; what reaches a level is ordinary
/// scene content — entities, colliders, and the
/// [`VehicleClass`](crate::components::VehicleClass) the scene has carried since
/// v25.
///
/// So the island's fleet is a table somebody can edit without touching Rust, and
/// the shipped bytes are the same shape they were before the table existed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleDef {
    /// Which silhouette the body is drawn as.
    pub body: VehicleBody,
    /// The chassis collider's half-extents, metres. `+Z` is forward, so `z` is
    /// **half the length** and `x` is half the width.
    pub half_extents: Vec3d,
    /// Chassis density, kg/m³ — a hollow shell rather than a material
    /// (`Collider3D::density`'s own note: rapier's 1.0 placeholder would make a
    /// car weigh eight kilograms).
    pub density_kg_m3: f64,
    /// Wheel radius, metres.
    pub wheel_radius_m: f64,
    /// **How a hull floats**, kg/m3 — the density the water pass weighs this
    /// craft's flotation by, or `0` for a vehicle that does not float
    /// (wave VEH2c).
    ///
    /// A geometry key and therefore free: this type derives no `Serialize`, so
    /// the field costs no schema anywhere. It is deliberately NOT the chassis
    /// collider's own `density_kg_m3`, which is what the craft WEIGHS: a boat
    /// is a light shell that floats high, and one number cannot be both.
    pub buoyancy_density_kg_m3: f64,
    /// The hull's linear drag through the water, per second — P20.2's own
    /// coefficient, authored per craft (wave VEH2c).
    ///
    /// It is what actually sets a boat's top speed. `Buoyancy`'s default is a
    /// blunt body's and held the launch fixture to 5.7 knots; a hull is shaped
    /// to go through water, and this is where that is said.
    pub buoyancy_linear_drag: f64,
    /// Half the track, metres — the wheel centres' `|x|` in the chassis frame.
    pub half_track_m: f64,
    /// Half the wheelbase, metres — the wheel centres' `|z|`.
    pub half_wheelbase_m: f64,
    /// The wheel centres' `y` in the chassis frame, metres, at full extension.
    /// Negative: a wheel hangs below its mount.
    pub wheel_drop_m: f64,
    /// The tuning installed on the chassis at creation.
    pub class: crate::components::VehicleClass,
}

impl Default for VehicleDef {
    /// The P29.7 test rig, **with its length and its width the right way
    /// round**.
    ///
    /// `PHASE29_CAR_HALF` is `(2.0, 0.5, 1.0)` and its own doc calls it
    /// "4 × 1 × 2 m" — which it is, in the order width, height, length. `+Z` is
    /// forward in this engine, so those half-extents describe a car **four
    /// metres wide and two metres long**, standing across a wheelbase of 2.8 m
    /// and a track of 1.8 m. Nothing had ever drawn the body at the size of its
    /// own collider, so nobody saw it (the I8b finding, met on a vehicle).
    fn default() -> Self {
        Self {
            body: VehicleBody::Sedan,
            half_extents: Vec3d::new(0.92, 0.5, 2.2),
            density_kg_m3: 150.0,
            wheel_radius_m: 0.35,
            half_track_m: 0.9,
            half_wheelbase_m: 1.4,
            wheel_drop_m: -0.75,
            // A road car does not float: zero is the refusal, not a density.
            buoyancy_density_kg_m3: 0.0,
            buoyancy_linear_drag: 0.0,
            class: crate::components::VehicleClass::default(),
        }
    }
}

impl VehicleDef {
    /// Set one **geometry** number by name, answering whether it took.
    ///
    /// Tuning names are not here: they are `VehicleTuning::names()` and they are
    /// reached through [`class`](Self::class), so there is one list of tunables
    /// in the engine rather than two. `set` tries its own names first and then
    /// delegates, which is what makes a catalogue row able to say
    /// `max_speed_mps = 30` beside `wheel_radius_m = 0.42`.
    ///
    /// A refusal is a value and a non-finite number is refused —
    /// `VehicleTuning::set`'s rule, met at the layer above it.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        let slot: &mut f64 = match name {
            "half_width_m" => &mut self.half_extents.x,
            "half_height_m" => &mut self.half_extents.y,
            "half_length_m" => &mut self.half_extents.z,
            "density_kg_m3" => &mut self.density_kg_m3,
            "wheel_radius_m" => &mut self.wheel_radius_m,
            "buoyancy_density_kg_m3" => &mut self.buoyancy_density_kg_m3,
            "buoyancy_linear_drag" => &mut self.buoyancy_linear_drag,
            "half_track_m" => &mut self.half_track_m,
            "half_wheelbase_m" => &mut self.half_wheelbase_m,
            "wheel_drop_m" => &mut self.wheel_drop_m,
            _ => return self.class.set(name, value),
        };
        *slot = value;
        true
    }

    /// Every **geometry** name, sorted — the door enumerated rather than
    /// restated (the P29.6 A14 shape).
    pub fn geometry_names() -> &'static [&'static str] {
        &[
            "buoyancy_density_kg_m3",
            "buoyancy_linear_drag",
            "density_kg_m3",
            "half_height_m",
            "half_length_m",
            "half_track_m",
            "half_wheelbase_m",
            "half_width_m",
            "wheel_drop_m",
            "wheel_radius_m",
        ]
    }

    /// The four wheel mounts this def implies, **front pair first** — `+Z` is
    /// forward and [`WheelMount::steered`] is a sign test on exactly this.
    ///
    /// One place, so the generator that spawns a car and any test that checks
    /// one cannot disagree about where its wheels are.
    pub fn wheel_mounts(&self) -> [Vec3d; 4] {
        let (x, y, z) = (self.half_track_m, self.wheel_drop_m, self.half_wheelbase_m);
        [
            Vec3d::new(-x, y, z),
            Vec3d::new(x, y, z),
            Vec3d::new(-x, y, -z),
            Vec3d::new(x, y, -z),
        ]
    }

    /// **Read one catalogue row**, or `None` when the table has no `[vehicle]`
    /// sub-table.
    ///
    /// `WeaponDef::from_toml_table`'s shape exactly: `body` is the one string
    /// key and every other key goes through [`set`](Self::set), so a name the
    /// engine does not know is reported by name rather than ignored.
    pub fn from_toml_table(
        t: &toml::map::Map<String, toml::Value>,
    ) -> Result<Option<Self>, String> {
        let Some(v) = t.get("vehicle") else {
            return Ok(None);
        };
        let table = v
            .as_table()
            .ok_or_else(|| "`vehicle` must be a table".to_string())?;
        let mut def = VehicleDef::default();
        if let Some(b) = table.get("body") {
            let name = b
                .as_str()
                .ok_or_else(|| "`body` must be a string".to_string())?;
            def.body = VehicleBody::from_name(name)
                .ok_or_else(|| format!("unknown vehicle body `{name}`"))?;
        }
        // **`drivetrain` is a spelling, not a field** (island wave VEH2a). The
        // physics reads one number — `front_torque_split` — and an enum beside it
        // would be a second source of truth for one fact; see that tunable's own
        // note. An author still gets to write `drivetrain = "awd"`, and it is
        // resolved HERE, before any numeric key, so an explicit
        // `front_torque_split` in the same table always wins whatever order the
        // TOML map happens to iterate in.
        if let Some(d) = table.get("drivetrain") {
            let name = d
                .as_str()
                .ok_or_else(|| "`drivetrain` must be a string".to_string())?;
            let split = match name {
                "fwd" => 1.0,
                "rwd" => 0.0,
                "awd" => AWD_FRONT_SPLIT,
                _ => {
                    return Err(format!(
                        "unknown drivetrain `{name}` (fwd, rwd or awd; for any \
                         other split say `front_torque_split` directly)"
                    ))
                }
            };
            def.class.front_torque_split = split;
        }
        for (k, v) in table {
            if k == "body" || k == "drivetrain" {
                continue;
            }
            let n = v
                .as_float()
                .or_else(|| v.as_integer().map(|i| i as f64))
                .ok_or_else(|| format!("`{k}` must be a number"))?;
            if !def.set(k, n) {
                return Err(format!("unknown vehicle key `{k}`"));
            }
        }
        Ok(Some(def))
    }
}

/// **The catalogue** — vehicle ids to their definitions, in id order.
///
/// A `BTreeMap` rather than a `HashMap` because a generator walks it to author
/// content, and content whose order depends on a hash seed is content whose
/// bytes are not reproducible.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VehicleDefs(pub std::collections::BTreeMap<String, VehicleDef>);

/// The most rows one catalogue may declare.
///
/// `ItemDefs::MAX_ITEM_DEFS`'s reason, at a fleet's scale: a table is authored
/// text and a bound on authored text is what stops a malformed one from being a
/// memory problem instead of an error message.
pub const MAX_VEHICLE_DEFS: usize = 256;

impl VehicleDefs {
    /// Merge a TOML catalogue in, answering how many rows it declared.
    ///
    /// `ItemDefs::merge_toml`'s shape: every top-level table is a row, its key
    /// is the id, and a row with no `[<id>.vehicle]` sub-table is skipped rather
    /// than refused (so one file can carry rows a different reader cares about).
    pub fn merge_toml(&mut self, text: &str) -> Result<usize, String> {
        let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let table = doc
            .as_table()
            .ok_or_else(|| "a vehicle catalogue is a table of rows".to_string())?;
        let mut n = 0usize;
        for (id, value) in table {
            let Some(row) = value.as_table() else {
                continue;
            };
            let Some(def) =
                VehicleDef::from_toml_table(row).map_err(|e| format!("vehicle `{id}`: {e}"))?
            else {
                continue;
            };
            if self.0.len() >= MAX_VEHICLE_DEFS && !self.0.contains_key(id) {
                return Err(format!(
                    "a vehicle catalogue may declare at most {MAX_VEHICLE_DEFS} rows"
                ));
            }
            self.0.insert(id.clone(), def);
            n += 1;
        }
        Ok(n)
    }

    /// One row by id.
    pub fn get(&self, id: &str) -> Option<&VehicleDef> {
        self.0.get(id)
    }
}

// ── controls, state, and the forces that come out ───────────────────────────

/// What the driver is asking for this step — the **whole** of the input a
/// vehicle sees, and the reason `MovementMode::Driving` needs no vehicle-shaped
/// knowledge in the movement step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VehicleControls {
    /// Drive, `[-1, 1]`. Negative is reverse.
    pub throttle: f64,
    /// Steer, `[-1, 1]`. Positive is right (a right turn is a negative yaw
    /// rate — see the door, where the sign is applied once).
    pub steer: f64,
    /// Foot brake, `[0, 1]`.
    pub brake: f64,
    /// Handbrake, held.
    pub handbrake: bool,
    /// **Vertical demand**, `[-1, 1]` — a rotorcraft's collective (wave VEH2c).
    ///
    /// # It costs no new control and no new intent field
    ///
    /// `MovementIntent::vertical` has existed since P20 as *"vertical intent
    /// while swimming or flying"*, is already written onto the runtime by
    /// `apply_intent`, and is already the axis a player uses to rise and sink.
    /// Routing it to the vehicle is one argument, so a helicopter is flown with
    /// the controls a swimmer already has rather than with a second binding
    /// table that could mean something different in the two hosts.
    ///
    /// A class that cannot climb ignores it, which is what
    /// [`RaycastVehicle`] and [`HullVehicle`] both do.
    pub vertical: f64,
    /// **Whether anybody is commanding this vehicle at all** (wave VEH2c).
    ///
    /// `false` on [`Default`], and that is the whole point: the vehicle door
    /// clears a vehicle's controls after every step, so a machine nobody spoke
    /// to this step hears silence rather than whatever it was last told.
    ///
    /// # Why a boolean, when zero would do for a car
    ///
    /// Because zero does NOT do for a helicopter. A parked car with the throttle
    /// at zero applies no force and sits still; a rotorcraft with the collective
    /// at zero is in a HOVER, which is what the governed collective means — so an
    /// unmanned machine would carry its own weight and drift off its pad for
    /// ever. Read the neutral as "hovering" and you cannot express "switched
    /// off"; the boolean is the difference between a stick at rest and no hand
    /// on it.
    ///
    /// It also closes a defect that predates this wave and was invisible in a
    /// car: `movement`, `traffic` and `dispatch` all command their vehicles
    /// through this struct and none of them cleared it, so a vehicle whose
    /// driver got out kept the last throttle it was given for ever. Nothing
    /// noticed, because the only thing anybody had ever got out of was a car
    /// that was already stopping.
    pub occupied: bool,
}

impl VehicleControls {
    /// Build controls from a character's movement intent: forward/back is the
    /// throttle **and** the brake, left/right is the steer.
    ///
    /// One place, so the editor's Simulate and the shipped player cannot
    /// interpret a stick differently — the `MovementIntent::from_actions`
    /// argument, one layer down.
    ///
    /// Back-into-forward-motion is a **brake**, not reverse: pressing back at
    /// 20 m/s should stop the car, and only once it is nearly stopped should it
    /// reverse. That is one comparison and it is here rather than in the door,
    /// because it is a control decision.
    pub fn from_intent(
        move_input: crate::math::Vec2d,
        forward_speed_mps: f64,
        handbrake: bool,
        vertical: f64,
    ) -> Self {
        let fwd = move_input.y.clamp(-1.0, 1.0);
        let steer = move_input.x.clamp(-1.0, 1.0);
        // 0.5 m/s: below it the car is stopped for a driver's purposes.
        let rolling_forward = forward_speed_mps > 0.5;
        let rolling_back = forward_speed_mps < -0.5;
        let (throttle, brake) = if fwd < 0.0 && rolling_forward {
            (0.0, -fwd)
        } else if fwd > 0.0 && rolling_back {
            (0.0, fwd)
        } else {
            (fwd, 0.0)
        };
        Self {
            throttle,
            steer,
            brake,
            handbrake,
            vertical: vertical.clamp(-1.0, 1.0),
            // An intent is a person's, so a control built from one is commanded
            // by definition. This is the only place that is true.
            occupied: true,
        }
    }
}

/// Where a wheel's ray landed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelContact {
    /// The contact point, world space.
    pub point: DVec3,
    /// The surface normal there, world space.
    ///
    /// # Nothing in the shipped model reads this, and that is the disposition of
    /// the D-14 snap (island wave VEH1a)
    ///
    /// A ray reads the **hit triangle's own** normal. `FIX_INTERNAL_EDGES` is on
    /// every heightfield this engine builds and it fixes *contacts*, not ray
    /// hits, so a wheel crossing a cell diagonal sees this vector change
    /// discontinuously — measured at **0.06°** on a levelled road corridor and
    /// **15.69°** on open ground with a real DTM's relief
    /// (`a_wheel_ray_normal_snaps_at_a_heightfield_cell_diagonal`).
    ///
    /// The obvious repair is to smooth it at the wheel. It was **not taken**,
    /// because [`RaycastVehicle::solve`] pushes the suspension along the
    /// **chassis up** (deliberately — see the force comment there) and takes its
    /// friction basis from the steered wheel's own axes, so no force is a
    /// function of this field. `the_snapped_normal_reaches_no_force_in_the_model`
    /// says so in metres rather than by grep: two rigs driven six hundred steps
    /// across the rough surface, one of them with every contact normal replaced
    /// by a direction the ground could not have, end **bit-identical**.
    ///
    /// It stays on the struct because it is what a ray answers and an island
    /// class may want it (a tracked vehicle's grouser, a tyre model with a
    /// camber term). **The day one reads it, that arm goes red**, and the
    /// smoothing question re-opens with the numbers above already measured.
    pub normal: DVec3,
    /// Distance from the mount to the contact along the suspension axis.
    pub distance_m: f64,
}

/// One wheel's live state — never serialized, like every other runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelState {
    /// This step's ground contact, or `None` in the air.
    pub contact: Option<WheelContact>,
    /// Current suspension length, metres: `rest_length_m` in the air, less when
    /// compressed.
    pub length_m: f64,
    /// The suspension force this wheel carried last step, newtons — the load the
    /// friction circle is sized against, and the number a tuner watches.
    pub load_n: f64,
    /// Steer angle, degrees, after the speed-sensitive limit.
    pub steer_deg: f64,
    /// Rolling angle, degrees, wrapped to `[0, 360)`.
    ///
    /// **Derived from [`omega_rad_s`](Self::omega_rad_s) since wave VEH2a.** It
    /// used to be integrated from the chassis's forward speed over the radius,
    /// which is a wheel that can never be seen to spin up under power or lock
    /// under braking — the visual said "this car is rolling" no matter what the
    /// contact patch was doing.
    pub spin_deg: f64,
    /// **Angular velocity about the axle, rad/s** (island wave VEH2a) — positive
    /// is rolling forward.
    ///
    /// The state P29.7 did not have, and the reason everything else in this model
    /// could be a force: with no `ω` there is no slip, with no slip there is no
    /// tyre curve, and without a tyre curve the only available longitudinal law
    /// is "cancel the slip, clamped", which is a tyre that is infinitely stiff
    /// right up to the moment it gives up completely.
    pub omega_rad_s: f64,
    /// **Longitudinal slip ratio** this step: `(ω·r − v_forward)` over
    /// `max(|v_forward|, `[`SLIP_REF_MPS`]`)`.
    ///
    /// Positive is the wheel turning faster than the road (wheelspin), negative
    /// is slower (lockup). Published because it is what a tuner, the ABS, the
    /// traction control and a tyre-smoke effect all read.
    pub slip_ratio: f64,
    /// **Lateral slip** this step, as the *tangent* of the slip angle:
    /// `v_right / max(|v_forward|, `[`SLIP_REF_MPS`]`)`.
    ///
    /// A tangent rather than an angle because that is the form the curve wants
    /// and computing it needs no `atan` — see
    /// [`VehicleTuning::tyre_lat_peak_slip`].
    pub slip_lat: f64,
    /// **Traction control's share of this wheel's drive torque**, `[0, 1]` —
    /// `1` is "no intervention".
    ///
    /// A **readout**, not a controller: the aid is one clamp on the torque request
    /// ([`aid_torque_cap_nm`]) and this is how much of it survived. What a HUD
    /// would draw, and what a test reads to see the aid engage.
    pub tc_cut: f64,
}

impl WheelState {
    /// A wheel at rest with traction control not intervening.
    ///
    /// `Default` cannot say this — a derived `tc_cut` is `0.0`, which is "cut all
    /// the torque" — so every construction site goes through here.
    fn fresh(rest_length_m: f64) -> Self {
        Self {
            length_m: rest_length_m,
            tc_cut: 1.0,
            ..Default::default()
        }
    }
}

/// A force to apply at a world point — what a [`Vehicle`] answers with, so the
/// model never touches a physics type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelForce {
    /// Where to apply it, world space.
    pub point: DVec3,
    /// The force, newtons, world space.
    pub force: DVec3,
}

/// The chassis body's state this step, as the door read it out of the solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChassisState {
    /// World position of the body origin.
    pub position: DVec3,
    /// World rotation.
    pub rotation: DQuat,
    /// Linear velocity, m/s.
    pub linvel: DVec3,
    /// Angular velocity, rad/s.
    pub angvel: DVec3,
    /// Mass, kg — the friction model's only use of it.
    pub mass_kg: f64,
    /// **The water surface over the chassis origin**, world metres, or `None`
    /// where no water body covers it (wave VEH2c).
    ///
    /// # Why the door samples it, and why once
    ///
    /// A hull's screw pushes only while it is *in the water*, which is the whole
    /// difference between a boat and a car with a rudder — and the honest way to
    /// know is to ask the same water index the buoyancy pass asks, at the same
    /// sim-clock time, so PIE and the shipped player cannot disagree about the
    /// sea state. The door has that seam already
    /// (`PhysicsBridge3D::water_surface_at`), so the class does not need one and
    /// [`Vehicle::solve`] stays a pure function of numbers.
    ///
    /// **One sample per vehicle, not one per part.** The chassis origin is the
    /// reference and a part's own immersion is derived from it through the
    /// chassis pose, which costs a subtraction rather than a second index query
    /// per thruster. The error that buys is a wave's *slope* across the hull, and
    /// on the island's sea (0.6 m amplitude over a 34 m wavelength) a 5 m boat
    /// spans 15 % of a wave and the slope term is under 6 cm — smaller than the
    /// draught band a screw sits in. Stated rather than hidden, and the fix if it
    /// ever matters is a sample per part.
    ///
    /// `None` on every level with no water, which is every level written before
    /// P20 and most of them since — so a wheeled vehicle never pays for this.
    pub water_y: Option<f64>,
}

impl ChassisState {
    /// The chassis basis: forward (`+Z`), right (`+X`) and up (`+Y`), rotated.
    pub fn basis(&self) -> (DVec3, DVec3, DVec3) {
        (
            self.rotation * DVec3::Z,
            self.rotation * DVec3::X,
            self.rotation * DVec3::Y,
        )
    }

    /// The velocity of the point `p` (world space) on this rigid body.
    ///
    /// The **suspension's** input: a damper exists to resist the chassis pitching
    /// and rolling, so it must see the whole rotation.
    pub fn point_velocity(&self, p: DVec3) -> DVec3 {
        self.linvel + self.angvel.cross(p - self.position)
    }

    /// The velocity of `p` with the chassis's **pitch and roll rates removed** —
    /// the *tyre's* input.
    ///
    /// # Why this is not [`point_velocity`](Self::point_velocity)
    ///
    /// A tyre resists slip in the ground plane, and the only rotation that makes
    /// one contact patch slip differently from another is the one about the
    /// vertical: that is what turns a steering angle into a yaw rate and what
    /// makes a handbrake turn possible. A pitch rate does *not* make a tyre slip
    /// — the suspension is what absorbs it.
    ///
    /// Feeding the whole rotation in is an **energy pump**, and it was measured
    /// before this function existed. Nose-down pitch makes the front contact
    /// move backwards and the rear forwards; a friction model that cancels each
    /// wheel's own slip then applies opposite forces at front and rear, which is
    /// a couple, which is more pitch. The rig reached a steady state of
    /// **0.21 rad/s of pitch that angular damping could not remove** and crawled
    /// backwards at 5 cm/s with its brakes full on. With the rates filtered out,
    /// the same brake stops it and holds it.
    pub fn contact_velocity(&self, p: DVec3, up: DVec3) -> DVec3 {
        let spin = up * self.angvel.dot(up);
        self.linvel + spin.cross(p - self.position)
    }
}

// ── the seam the island extends ─────────────────────────────────────────────

/// **A vehicle.** The seam `MovementMode::Driving` routes input through, and the
/// one the island phase implements per vehicle class.
///
/// Object-safe on purpose: a running world holds `Box<dyn Vehicle>` keyed by
/// chassis `Guid`, so a tank, a hovercraft and a boat can share one fixed-step
/// door. Everything the door needs is here and nothing rapier-shaped is:
/// [`solve`](Self::solve) is handed the chassis state and the wheels' contacts
/// (which the door filled by casting the rays this trait described) and answers
/// with forces. An implementation that wants no rays says so by having no
/// wheels.
pub trait Vehicle: Send + Sync + 'static {
    /// The rig this vehicle drives — its chassis, its seat and its wheels.
    fn rig(&self) -> &VehicleRig;

    /// **Replace the rig** because the scene's geometry changed, keeping
    /// everything a tuner has been editing.
    ///
    /// On the trait rather than on the implementation because the bridge
    /// reconciles every vehicle through one loop, and an island class that
    /// wanted to keep its own derived state on a re-derive is exactly the case
    /// the trait exists for.
    fn set_rig(&mut self, rig: VehicleRig);

    /// The wheels' live state, in [`VehicleRig::wheels`] order.
    fn wheels(&self) -> &[WheelState];

    /// The same, for the door to write each ray's answer into before
    /// [`solve`](Self::solve).
    fn wheels_mut(&mut self) -> &mut [WheelState];

    /// **Route one step's driver input in.** The trait's headline: what a
    /// controller asks for is a `VehicleControls`, and what a class does with it
    /// is the class's business.
    fn control(&mut self, controls: VehicleControls);

    /// The tunables, by name — the live-tuning door's target.
    fn tune(&mut self, name: &str, value: f64) -> bool;

    /// **The enter/exit choreography**: how long it takes, seconds, and the
    /// window of it the seat warp occupies.
    ///
    /// On the trait because the seat step reads it and an island class may want
    /// a different one — a motorbike is thrown a leg over and a tank is climbed
    /// into. See `inf_physics::d3::movement::step_driving` for what the window
    /// does, and `VehicleTuning::enter_window` for why it is a window.
    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow);

    /// The suspension's length at full extension, metres.
    ///
    /// On the trait rather than derived from the wheels because the door needs
    /// it to place the **ray anchor**, and deriving it from a wheel's current
    /// length would move that anchor down as the suspension compressed — a
    /// feedback loop that walks a parked car into the floor. (It was written
    /// that way first; this sentence is why it is not.) A class with no
    /// suspension answers `0.0` and gets a ray from the wheel centre.
    fn suspension_rest_m(&self) -> f64;

    /// **The step**: given the chassis state and this step's contacts, append the
    /// forces to apply. Pure — no world, no physics types, no allocation beyond
    /// `out`.
    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>);

    /// The suspension length to draw each wheel at, in
    /// [`VehicleRig::wheels`] order — what the door writes onto the wheel
    /// entities' transforms.
    fn wheel_pose(&self, index: usize) -> Option<(f64, f64, f64)> {
        let w = self.wheels().get(index)?;
        Some((w.length_m, w.steer_deg, w.spin_deg))
    }

    /// The **rotation to draw each non-wheel part at**, euler YXZ degrees, in
    /// [`VehicleRig::parts`] order (wave VEH2c).
    ///
    /// The sibling of [`wheel_pose`](Self::wheel_pose) and deliberately a
    /// rotation only: a wheel travels on its suspension and a rotor does not, so
    /// a part's translation stays the authored mount and there is nothing for a
    /// re-read to feed back in.
    ///
    /// A **visual write only**, exactly as the wheel pose is — nothing reads it
    /// back, so a rig with no drawn blade simulates identically to one with a
    /// blade. The default is `None`, which is what a class with no parts should
    /// answer.
    fn part_pose(&self, index: usize) -> Option<Vec3d> {
        let _ = index;
        None
    }

    /// **What the engine is doing**, as two numbers in `[0, 1]`: how hard the
    /// driver is asking (`load`) and how fast it is turning over (`revs`).
    ///
    /// Island wave VEH1a — the input to [`engine_cue`], and the whole of what a
    /// vehicle tells the audio queue. On the trait rather than derived from the
    /// door's `VehicleOutcome` because *what counts as revs* is a class's business: a
    /// road car's is its speed against its own top speed, an electric one has no
    /// idle, a helicopter's is its collective and has nothing to do with how
    /// fast it is going. The door hands in the forward speed it has already
    /// computed, so this stays a pure function of numbers the class holds and
    /// adds no state.
    ///
    /// **Which gear the box is in** — `-1` reverse, `0` neutral, `1..` forward.
    ///
    /// On the trait rather than only on `RaycastVehicle` because wave VEH2b's
    /// readout is the first thing in this engine that DRAWS it, and it holds a
    /// `&dyn Vehicle`. VEH2a's carried item 8 (*"rpm, gear and the aids'
    /// intervention are all published and nothing draws them"*) is half closed
    /// by that: the gear is drawn, the rest is still carried.
    fn gear(&self) -> i32;

    /// The default is **silence** (`(0.0, 0.0)`), which is what a class that has
    /// not thought about sound should make.
    fn engine_state(&self, forward_mps: f64) -> (f64, f64) {
        let _ = forward_mps;
        (0.0, 0.0)
    }
}

// ── the engine loop (island wave VEH1a) ─────────────────────────────────────

/// Pitch a vehicle's engine loop plays at when it is barely turning over, as a
/// multiplier on the authored [`AudioSource::pitch`].
///
/// [`AudioSource::pitch`]: crate::components::AudioSource::pitch
pub const ENGINE_IDLE_PITCH: f64 = 0.65;
/// Pitch at the class's own top speed, same units.
///
/// 2.05 against 0.65 is a little over an octave and a half across the range,
/// which is what a single-clip engine loop can carry before it reads as a
/// slowed-down tape.
pub const ENGINE_TOP_PITCH: f64 = 2.05;
/// The share of the authored volume an engine makes with the throttle closed —
/// an idling engine is quieter than one under load and is **not silent**, which
/// is the difference between a car and a golf cart.
pub const ENGINE_IDLE_GAIN: f64 = 0.35;

/// What the engine loop should sound like this step.
///
/// Deliberately not an `AudioCommand`: `inf-ecs` does not depend on `inf-audio`,
/// and the decision is the part that has to be identical in both hosts. The
/// mapping onto `SetPitch`/`SetVolume` is six lines and lives in each host's
/// audio step, behind a `MIRROR-BEGIN vehicle_engine_audio` fence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineCue {
    /// Playback-rate factor for the looping clip.
    pub pitch: f64,
    /// Linear volume, before the bus, the master and the spatial fall-off.
    pub volume: f64,
}

/// **The engine loop, as a pure function of sim state** (island wave VEH1a) —
/// the P12 command-queue doctrine, met exactly.
///
/// `revs` and `load` are [`Vehicle::engine_state`]'s two numbers; `base_pitch`
/// and `base_volume` are the emitter's own authored values, so an author who
/// wants a truck to idle lower says so on the `AudioSource` rather than in
/// engine code.
///
/// Every input is clamped and a non-finite one answers **idle** rather than
/// propagating: a NaN pitch reaches a device mixer, and a refusal is a value
/// (the `Tune::set` rule, one system over).
pub fn engine_cue(revs: f64, load: f64, base_pitch: f64, base_volume: f64) -> EngineCue {
    let ok = |v: f64| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let (revs, load) = (ok(revs), ok(load));
    let base_pitch = if base_pitch.is_finite() {
        base_pitch.max(0.0)
    } else {
        1.0
    };
    let base_volume = if base_volume.is_finite() {
        base_volume.max(0.0)
    } else {
        1.0
    };
    EngineCue {
        pitch: base_pitch * (ENGINE_IDLE_PITCH + (ENGINE_TOP_PITCH - ENGINE_IDLE_PITCH) * revs),
        volume: base_volume * (ENGINE_IDLE_GAIN + (1.0 - ENGINE_IDLE_GAIN) * load),
    }
}

/// **Which class drives a wheel-less rig** (wave VEH2c) — the one place the
/// parts a scene authored become the model that reads them.
///
/// In Ring 0, beside the classes it chooses between, rather than as a `match` in
/// the physics bridge: the choice is a pure function of a rig and it is tested
/// as one. `rig_of` has already guaranteed that a rig reaching here has no
/// wheels, so this is not competing with `RaycastVehicle` — it is the other
/// branch of a decision already made.
///
/// **Rotors outrank thrusters**, and it is a real rule rather than an arbitrary
/// order: a craft with both is an amphibious rotorcraft, whose behaviour is
/// dominated by the thing holding it up. A rig with neither cannot occur (a rig
/// with no wheels and no parts is not a rig at all) and answers with the
/// inert wheel-less `RaycastVehicle` rather than with a panic — a refusal is a
/// value here as everywhere.
///
/// A wheel-less [`RaycastVehicle`] is genuinely inert: its solve loops over an
/// empty wheel list, so it applies **no force at all**. That is what a rig whose
/// parts nothing recognises should do.
pub fn class_for_parts(rig: VehicleRig) -> Box<dyn Vehicle> {
    if rig.parts_of(PartKind::Rotor).next().is_some() {
        Box::new(RotorVehicle::new(rig))
    } else if rig.parts_of(PartKind::Thruster).next().is_some() {
        Box::new(HullVehicle::new(rig))
    } else {
        Box::new(RaycastVehicle::new(rig))
    }
}

// ── the implementation this phase ships ─────────────────────────────────────

/// The raycast vehicle: a spring, a damper, a powertrain and a tyre per wheel.
#[derive(Clone, Debug)]
pub struct RaycastVehicle {
    rig: VehicleRig,
    tuning: VehicleTuning,
    controls: VehicleControls,
    wheels: Vec<WheelState>,
    /// The gear the box is in: `-1` reverse, `0` neutral, `1..=gears()`.
    gear: i32,
    /// Seconds left of the shift in progress. **No drive torque crosses the box
    /// while this is positive**, which is the whole of why a shift is felt.
    shift_left_s: f64,
    /// Engine speed, rpm — a *derived* value, held so the audio and the shift
    /// model read the same number the torque came from.
    rpm: f64,
    /// The road wheels' actual steer angle, degrees — the RACK's own state,
    /// which is what makes a steering rate and a return-to-centre possible.
    steer_deg: f64,
    /// Per-wheel drive torque this step, N·m — scratch, in
    /// [`VehicleRig::wheels`] order.
    ///
    /// A field rather than a local because the differential's share depends on
    /// **every** wheel on an axle, so the torques must all be known before the
    /// per-wheel loop takes a mutable borrow of the wheel states — and a `Vec`
    /// allocated inside a fixed step, once per vehicle per step, is exactly the
    /// kind of cost the vehicle phase was given its own budget row to notice.
    drive_nm: Vec<f64>,
}

impl RaycastVehicle {
    /// Build one over a derived rig, with the default tuning.
    pub fn new(rig: VehicleRig) -> Self {
        let tuning = VehicleTuning::default();
        let wheels = vec![WheelState::fresh(tuning.rest_length_m); rig.wheels.len()];
        Self {
            drive_nm: vec![0.0; rig.wheels.len()],
            rig,
            controls: VehicleControls::default(),
            wheels,
            gear: 1,
            shift_left_s: 0.0,
            steer_deg: 0.0,
            rpm: tuning.idle_rpm,
            tuning,
        }
    }

    /// The tuning, for a test or a UI to read.
    pub fn tuning(&self) -> &VehicleTuning {
        &self.tuning
    }

    /// The controls in force this step.
    pub fn controls(&self) -> VehicleControls {
        self.controls
    }

    /// Engine speed, rpm (island wave VEH2a) — the real number the torque curve
    /// was read at, not a speed rescaled to look like revs.
    pub fn rpm(&self) -> f64 {
        self.rpm
    }

    /// The gear the box is in: `-1` reverse, `0` neutral, `1..=`.
    pub fn gear(&self) -> i32 {
        self.gear
    }

    /// The rack's angle, degrees — where the road wheels actually are, which
    /// since VEH2a is not where the driver's stick is.
    pub fn steer_deg(&self) -> f64 {
        self.steer_deg
    }

    /// **The load one axle carries**, newtons — the readout a tuner watches and
    /// the number every weight-transfer claim in this model is made of.
    ///
    /// `front` selects the steered axle. On a rig at rest the two sum to the
    /// car's weight; under braking the front's rises and the rear's falls, and
    /// with `tyre_load_sensitivity` above zero that swap is not free.
    pub fn axle_load_n(&self, front: bool) -> f64 {
        self.rig
            .wheels
            .iter()
            .zip(self.wheels.iter())
            .filter(|(m, _)| m.steered() == front)
            .map(|(_, w)| w.load_n)
            .sum()
    }

    /// Whether a shift is in progress — the window with no drive torque in it.
    pub fn shifting(&self) -> bool {
        self.shift_left_s > 0.0
    }

    /// The static vertical load one wheel carries, newtons — the reference
    /// [`load_sensitive_mu`] measures against.
    ///
    /// Derived from the chassis mass rather than authored, because it is not a
    /// tuning choice: it is what the car weighs divided by how many wheels it
    /// stands on, and a class that authored it could disagree with its own body.
    fn static_load_n(&self, mass_kg: f64) -> f64 {
        // `is_finite() && > 0.0` rather than `!(> 0.0)`: the negated form is
        // NaN-safe but lets an INFINITE mass through, and clippy's
        // `neg_cmp_op_on_partial_ord` is right that it reads badly. This spelling
        // refuses both, which is what a refusal-is-a-value model wants.
        if self.rig.wheels.is_empty() || !(mass_kg.is_finite() && mass_kg > 0.0) {
            return 0.0;
        }
        mass_kg * 9.81 / self.rig.wheels.len() as f64
    }
}

/// **The one shape function** this model bends every curve with (VEH2a) —
/// Schlick's bias, `[0, 1] → [0, 1]`, monotone, with `k = 0.5` the identity.
///
/// `b(t, k) = t / ((1/k − 2)(1 − t) + 1)`. Its slope at the origin is
/// `k / (1 − k)`, so `k` is *exactly* "how much sooner than linear does this
/// arrive": 0.8 rises four times as fast out of zero, 0.2 a quarter as fast, and
/// both still land on 1 at `t = 1`.
///
/// # Why this and not an exponent
///
/// A `powf` would do the same job and is the obvious spelling. It is refused for
/// the reason P14's law exists: `powf` routes through the `libm` crate on
/// `wasm32` and through the platform's own on everything else, and a vehicle's
/// pose is compared **byte for byte between two hosts** by the island drive gate
/// and rides into committed `.inf_lvl` bytes through nothing at all — but the day
/// it does, a shape function built out of a transcendental is a divergence with
/// no gate. This is four arithmetic operations and is bit-identical everywhere.
///
/// `k` is clamped into `[0.05, 0.95]` and a non-finite `k` is the identity, on
/// the standing rule that a refusal is a value.
pub fn curve_bias(t: f64, k: f64) -> f64 {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let k = if k.is_finite() {
        k.clamp(0.05, 0.95)
    } else {
        0.5
    };
    let a = 1.0 / k - 2.0;
    let denom = a * (1.0 - t) + 1.0;
    if denom.abs() < 1e-12 {
        return t;
    }
    (t / denom).clamp(0.0, 1.0)
}

/// The speed the slip ratio's denominator is floored at, m/s.
///
/// Slip ratio is `(ω·r − v) / |v|` and `v` is zero at every traffic light, so
/// the denominator needs a floor or a stationary car reports infinite slip and
/// launches itself. Two metres per second is the standard choice and it is a
/// statement about the model's validity rather than a fudge: below a walking
/// pace a tyre is inside its linear region whatever the ratio says, and the
/// floor is what keeps the curve's argument finite there.
pub const SLIP_REF_MPS: f64 = 2.0;

/// The least of its nominal µ a tyre keeps under load sensitivity, and the most.
///
/// Load sensitivity is a linear fit to a curve that is not linear, so it is
/// bracketed: an unloaded inside wheel does not get 300 % grip and a wheel
/// carrying four times its share does not get none.
pub const MU_MIN_FRAC: f64 = 0.35;
/// The most of its nominal µ a tyre keeps under load sensitivity.
pub const MU_MAX_FRAC: f64 = 1.60;

/// **The torque curve**: crankshaft torque, newton-metres, at `rpm`.
///
/// Three knots — `idle_torque_frac` at [`VehicleTuning::idle_rpm`], `1.0` at
/// `peak_torque_rpm`, `redline_torque_frac` at `redline_rpm` — with
/// [`curve_bias`] bending both halves by the one shape knob. Below idle and above
/// the redline the **argument** is clamped, so the curve is flat outside its own
/// knots.
///
/// # What clamping the argument is, and is NOT (`audit:` VEH2a)
///
/// It is a **plateau**, not a fuel cut: past the redline this answers
/// `redline_torque_frac × peak_torque_nm` for ever rather than falling to zero,
/// and [`engine_rpm`] clamps to the same ceiling, so an over-revving engine is
/// invisible rather than punished. Nothing in this model cuts fuel.
///
/// It costs nothing in the gears the box shifts through, because `shift_up_rpm`
/// is below the redline and [`shift_target`] leaves before the ceiling is
/// reached. In **top** gear it means the ceiling that actually stops the car is
/// [`governor`] — the road-speed limiter — and not the redline. That is the
/// honest reading of the wave's own "a limiter cuts fuel" law: the law is about
/// where a limiter may NOT reach (the contact patch, see the note beside
/// `omega_ceiling`), and the fuel half of it is not built.
///
/// This replaces P29.7's `engine_force_n`, whose whole curve was "a peak force
/// falling linearly to zero at the top speed". That function could not be revvy
/// or torquey, could not be geared, and — because it answered a *force* — never
/// read the wheel's radius at all, so a truck's 0.42 m wheels and a hatchback's
/// 0.30 m ones pushed exactly as hard.
pub fn engine_torque_nm(tuning: &VehicleTuning, rpm: f64) -> f64 {
    let idle = if tuning.idle_rpm.is_finite() {
        tuning.idle_rpm.max(0.0)
    } else {
        0.0
    };
    let peak = if tuning.peak_torque_rpm.is_finite() {
        tuning.peak_torque_rpm.max(idle + 1.0)
    } else {
        idle + 1.0
    };
    let red = if tuning.redline_rpm.is_finite() {
        tuning.redline_rpm.max(peak + 1.0)
    } else {
        peak + 1.0
    };
    let rpm = if rpm.is_finite() {
        rpm.clamp(idle, red)
    } else {
        idle
    };
    let frac = if rpm <= peak {
        let lo = tuning.idle_torque_frac.clamp(0.0, 1.0);
        lo + (1.0 - lo) * curve_bias((rpm - idle) / (peak - idle), tuning.torque_curve_bias)
    } else {
        let hi = tuning.redline_torque_frac.clamp(0.0, 1.0);
        1.0 - (1.0 - hi) * curve_bias((rpm - peak) / (red - peak), tuning.torque_curve_bias)
    };
    let peak_nm = if tuning.peak_torque_nm.is_finite() {
        tuning.peak_torque_nm.max(0.0)
    } else {
        0.0
    };
    peak_nm * frac
}

/// The share of [`VehicleTuning::max_speed_mps`] the speed limiter tapers over.
///
/// Six per cent: full torque below 94 % of the limiter, nothing at it. A taper
/// rather than a cut, because a limiter that switches off is a wall a car bounces
/// against and hunts around, and every driver has felt the difference.
pub const GOVERNOR_BAND: f64 = 0.06;

/// **The speed limiter**, `[0, 1]` on the drive torque.
///
/// `max_speed_mps` was the point where P29.7's flat drive curve reached zero, so
/// it *was* the top speed by construction. With a real torque curve the top speed
/// is an emergent balance against the drag — and for this engine's default rig
/// that balance is **52.7 m/s**, which is a correct answer to a question nobody
/// asked: the committed content, its gates and its camera were all built around a
/// car that tops out at 25.
///
/// So the field keeps its meaning and gets an honest mechanism. Nearly every road
/// car built this century is electronically limited, and this is that limiter: a
/// taper across the last [`GOVERNOR_BAND`] of the authored speed. It is also what
/// makes "top speed" a number a per-class spec row can *bound* rather than a
/// number that falls out of four other numbers.
pub fn governor(tuning: &VehicleTuning, forward_mps: f64) -> f64 {
    let top = tuning.max_speed_mps;
    if !(top.is_finite() && top > 0.0) || !forward_mps.is_finite() {
        return 1.0;
    }
    let over = forward_mps.abs() / top - (1.0 - GOVERNOR_BAND);
    if over <= 0.0 {
        1.0
    } else {
        (1.0 - over / GOVERNOR_BAND).clamp(0.0, 1.0)
    }
}

/// **Engine speed from wheel speed** — a rigid driveline with an idle floor.
///
/// No clutch state machine: below the speed the gearing implies, a real clutch is
/// slipping and the engine is at idle, and `max(idle, …)` is that in one
/// comparison. Neutral (and a gear the box does not have) answers idle.
pub fn engine_rpm(tuning: &VehicleTuning, wheel_omega: f64, gear: i32) -> f64 {
    let idle = if tuning.idle_rpm.is_finite() {
        tuning.idle_rpm.max(0.0)
    } else {
        0.0
    };
    let red = if tuning.redline_rpm.is_finite() {
        tuning.redline_rpm.max(idle + 1.0)
    } else {
        idle + 1.0
    };
    let ratio = tuning.drive_ratio(gear);
    if !wheel_omega.is_finite() || ratio == 0.0 {
        return idle;
    }
    (wheel_omega.abs() * ratio.abs() * 60.0 / std::f64::consts::TAU).clamp(idle, red)
}

/// **The automatic gearbox's decision**: which gear to be in.
///
/// Reverse is a **gear**, not a scalar on a force — P29.7 reversed at "a third of
/// the drive force", which is a reverse that gets stronger as the engine does and
/// never runs out of revs. It is selected the way a real automatic selects it: by
/// the driver asking for backwards while the car is not already going forwards.
///
/// A refusal is a value here as everywhere: the answer is always a gear the box
/// has.
pub fn shift_target(
    tuning: &VehicleTuning,
    gear: i32,
    rpm: f64,
    throttle: f64,
    forward_mps: f64,
) -> i32 {
    let top = tuning.gears() as i32;
    // `VehicleControls::from_intent` only produces a negative throttle once the
    // car is nearly stopped, so this is the whole of the reverse rule.
    if throttle < 0.0 && forward_mps < 0.5 {
        return -1;
    }
    if gear <= 0 {
        return 1;
    }
    if gear == -1 {
        return 1;
    }
    if rpm >= tuning.shift_up_rpm && gear < top {
        return gear + 1;
    }
    if rpm <= tuning.shift_down_rpm && gear > 1 {
        return gear - 1;
    }
    gear.clamp(1, top)
}

/// **The tyre's normalized force curve**, `[0, 1]` of `µ × load`.
///
/// `slip_norm` is the slip in units of *that axis's own peak slip*, so the peak
/// is always at 1. Below it the rise is [`curve_bias`]-shaped by `rise_bias`
/// (which is therefore the tyre's stiffness, independently of where the peak is);
/// above it the force falls linearly to `slide_frac` at
/// [`TYRE_SLIDE_SLIP_MULT`] and holds there.
///
/// The falling branch is the whole point and is what P29.7's model could not
/// have: past the peak a tyre grips **less**, so a slide is something a driver
/// has to correct rather than a state the car settles into comfortably.
pub fn tyre_curve(slip_norm: f64, rise_bias: f64, slide_frac: f64) -> f64 {
    let s = if slip_norm.is_finite() {
        slip_norm.abs()
    } else {
        0.0
    };
    let slide = if slide_frac.is_finite() {
        slide_frac.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if s <= 1.0 {
        curve_bias(s, rise_bias)
    } else if s >= TYRE_SLIDE_SLIP_MULT {
        slide
    } else {
        1.0 + (slide - 1.0) * (s - 1.0) / (TYRE_SLIDE_SLIP_MULT - 1.0)
    }
}

/// **Load sensitivity**: the µ a tyre actually has at `load_n`, given the µ it
/// has at its static share.
///
/// `µ(Fz) = µ · (1 − k·(Fz/Fz₀ − 1))`, bracketed by [`MU_MIN_FRAC`] and
/// [`MU_MAX_FRAC`]. `k = 0` is the schoolbook tyre whose grip is exactly `µ·Fz`;
/// every real one loses grip as it is pressed harder.
///
/// **This is the function that makes weight transfer cost something.** Without
/// it, load moved from the inside wheel to the outside one is grip moved with it
/// and an axle's total is unchanged — so a centre of gravity height, an anti-roll
/// bar and a soft spring would all be free.
pub fn load_sensitive_mu(mu: f64, load_n: f64, static_load_n: f64, sensitivity: f64) -> f64 {
    if !(load_n.is_finite() && sensitivity.is_finite())
        || !(static_load_n.is_finite() && static_load_n > 0.0)
    {
        return mu;
    }
    let k = (1.0 - sensitivity * (load_n / static_load_n - 1.0)).clamp(MU_MIN_FRAC, MU_MAX_FRAC);
    mu * k
}

/// **The friction circle** — one contact's ground force, newtons, in the wheel's
/// own `(forward, right)` frame.
///
/// The two slips are normalized by their own peaks and then **combined into one
/// magnitude**; the curve is evaluated at that magnitude and the result is split
/// back along the slip direction. Pure longitudinal slip therefore reproduces the
/// longitudinal curve exactly, pure lateral the lateral one, and everything
/// between lies on the ellipse the two µ's describe.
///
/// # What this replaces, and why it is the headline of the wave
///
/// P29.7 clamped the two axes **independently**: `lateral.clamp(-µ·N, µ·N)` on
/// one line and `longitudinal.clamp(-µ·N, µ·N)` on another. A tyre under that
/// rule can hold its whole grip sideways *and* its whole grip forwards at the
/// same instant — √2 ≈ 1.41 times what it has — so the car could brake at full
/// force out of a corner it was already sliding through, and no amount of
/// throttle could ever cost it steering. That is not a friction circle; it is two
/// boxes. This is the circle.
pub fn tyre_force_n(
    tuning: &VehicleTuning,
    load_n: f64,
    static_load_n: f64,
    slip_ratio: f64,
    slip_lat: f64,
) -> (f64, f64) {
    if !(load_n.is_finite() && load_n > 0.0) || !slip_ratio.is_finite() || !slip_lat.is_finite() {
        return (0.0, 0.0);
    }
    let px = if tuning.tyre_long_peak_slip.is_finite() {
        tuning.tyre_long_peak_slip.max(1e-4)
    } else {
        1e-4
    };
    let py = if tuning.tyre_lat_peak_slip.is_finite() {
        tuning.tyre_lat_peak_slip.max(1e-4)
    } else {
        1e-4
    };
    let (sx, sy) = (slip_ratio / px, slip_lat / py);
    let s = (sx * sx + sy * sy).sqrt();
    if s < 1e-12 {
        return (0.0, 0.0);
    }
    let mu_x = load_sensitive_mu(
        tuning.longitudinal_grip,
        load_n,
        static_load_n,
        tuning.tyre_load_sensitivity,
    );
    let mu_y = load_sensitive_mu(
        tuning.lateral_grip,
        load_n,
        static_load_n,
        tuning.tyre_load_sensitivity,
    );
    let fx_mag = mu_x * load_n * tyre_curve(s, tuning.tyre_long_rise_bias, tuning.tyre_slide_frac);
    let fy_mag = mu_y * load_n * tyre_curve(s, tuning.tyre_lat_rise_bias, tuning.tyre_slide_frac);
    // A positive slip RATIO is the wheel outrunning the road, which pushes the
    // car FORWARD; a positive lateral slip is the patch sliding right, which the
    // tyre resists to the LEFT. The two conventions differ by a sign and it is
    // spent here, once, rather than at three call sites.
    (fx_mag * (sx / s), -fy_mag * (sy / s))
}

/// **The steering limit**: full lock at a standstill, tightening with speed.
///
/// Linear between the two authored angles over `[0, max_speed_mps]`. ALS's
/// camera and this share a habit worth naming: a curve with two ends and a
/// straight line between them is a curve an author can predict.
pub fn steer_limit_deg(tuning: &VehicleTuning, speed_mps: f64) -> f64 {
    if tuning.max_speed_mps <= 0.0 {
        return tuning.max_steer_deg;
    }
    let t = (speed_mps.abs() / tuning.max_speed_mps).clamp(0.0, 1.0);
    tuning.max_steer_deg + (tuning.min_steer_deg - tuning.max_steer_deg) * t
}

/// **Ackermann steering**: the angle one front wheel takes, given the rack's.
///
/// Both front wheels turn about one centre, so the inside wheel — the one on a
/// tighter radius — must turn **more**. `amount` blends between parallel steering
/// (`0`, both wheels at the rack angle) and the full geometry (`1`).
///
/// # A first-order geometry, and why it is not the textbook one
///
/// The exact relation is `cot δ_outer − cot δ_inner = track / wheelbase`, which
/// needs a tangent and an arctangent. P14's law says std trigonometry is not
/// bit-portable, and this repository has no portable `tan`, so the small-angle
/// form — `R = L/δ`, `δ_inner = δ / (1 − δ·w/L)` — is used instead. It is exact
/// to first order, it has the right sign and the right magnitude everywhere a
/// road car steers, and it is four arithmetic operations. The denominator is
/// floored so a rack angle large enough to put the turn centre inside the track
/// answers a big number rather than a singular one.
pub fn ackermann_deg(
    rack_deg: f64,
    inner: bool,
    half_track_m: f64,
    wheelbase_m: f64,
    amount: f64,
) -> f64 {
    let a = if amount.is_finite() {
        amount.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if a <= 0.0
        || !(wheelbase_m.is_finite() && wheelbase_m > 0.0)
        || !rack_deg.is_finite()
        || !(half_track_m.is_finite() && half_track_m > 0.0)
    {
        return rack_deg;
    }
    let k = rack_deg.to_radians().abs() * half_track_m / wheelbase_m;
    let factor = if inner {
        1.0 / (1.0 - k).max(0.25)
    } else {
        1.0 / (1.0 + k)
    };
    rack_deg * (1.0 + a * (factor - 1.0))
}

/// **The suspension**: spring plus damper, in newtons, never negative.
///
/// `compression_m` is how far the suspension is from full extension and
/// `closing_mps` how fast it is still compressing (positive = compressing). A
/// suspension that pulls the chassis DOWN when it is extending is a suspension
/// that sucks a car onto the road, so the result is floored at zero — the
/// standard, and the reason a raycast vehicle does not need a rebound spring.
pub fn suspension_force_n(tuning: &VehicleTuning, compression_m: f64, closing_mps: f64) -> f64 {
    let x = compression_m.clamp(0.0, tuning.travel_m);
    let f = tuning.stiffness_n_per_m * x + tuning.damping_ns_per_m * closing_mps;
    f.max(0.0)
}

/// Rotate `forward` by `deg` about `up` — the steer, without a quaternion.
///
/// `DQuat::from_axis_angle` would reach `f64::sin_cos`, and P14's law is that
/// std trigonometry is not bit-portable across platforms. Two `inf_math`
/// portable calls and a plane rotation is the whole of it, and it is exact for
/// an orthonormal `(forward, right)` pair, which the chassis basis is.
pub fn steer_direction(forward: DVec3, right: DVec3, deg: f64) -> DVec3 {
    let rad = deg.to_radians();
    (forward * inf_math::pcos64(rad) + right * inf_math::psin64(rad)).normalize_or_zero()
}

impl Vehicle for RaycastVehicle {
    fn gear(&self) -> i32 {
        RaycastVehicle::gear(self)
    }

    fn rig(&self) -> &VehicleRig {
        &self.rig
    }

    fn set_rig(&mut self, rig: VehicleRig) {
        // The wheel STATE survives a re-derive whose wheel count did not change:
        // a suspension that reset to full extension every time the scene changed
        // would make an editor edit a bounce.
        if rig.wheels.len() != self.wheels.len() {
            self.wheels = vec![WheelState::fresh(self.tuning.rest_length_m); rig.wheels.len()];
        }
        self.rig = rig;
    }

    fn wheels(&self) -> &[WheelState] {
        &self.wheels
    }

    fn wheels_mut(&mut self) -> &mut [WheelState] {
        &mut self.wheels
    }

    fn control(&mut self, controls: VehicleControls) {
        self.controls = controls;
    }

    fn tune(&mut self, name: &str, value: f64) -> bool {
        self.tuning.set(name, value)
    }

    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
        (self.tuning.enter_time_s, self.tuning.enter_window())
    }

    fn suspension_rest_m(&self) -> f64 {
        self.tuning.rest_length_m
    }

    /// **Revs are the engine's own rpm** between idle and the redline (island
    /// wave VEH2a); load is the throttle, unsigned, because reversing is not
    /// quieter.
    ///
    /// VEH1a computed this from road speed against the class's top speed, which
    /// is a car with one infinitely long gear: it rose smoothly to the top and
    /// never once fell as a shift went through. It now comes from
    /// [`engine_rpm`] through the gear actually engaged, so the loop **drops at
    /// every upshift and flares on every downshift** — which is the single thing
    /// that makes an engine sound like a car rather than a hair dryer. The
    /// `forward_mps` the door hands in is no longer read, and it stays in the
    /// signature because the trait is what an island class implements and a class
    /// whose revs really are its road speed is a legitimate class.
    ///
    /// The brake is deliberately not in it: a car braking hard from its top
    /// speed still has an engine turning over, and folding the brake in would
    /// make the loudest moment of a drive the one where the pedal comes off.
    fn engine_state(&self, forward_mps: f64) -> (f64, f64) {
        let _ = forward_mps;
        let span = (self.tuning.redline_rpm - self.tuning.idle_rpm).max(1.0);
        let revs = ((self.rpm - self.tuning.idle_rpm) / span).clamp(0.0, 1.0);
        (revs, self.controls.throttle.abs().clamp(0.0, 1.0))
    }

    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let (fwd, right, up) = chassis.basis();
        let forward_mps = chassis.linvel.dot(fwd);
        let speed = chassis.linvel.length();
        // ── the steering rack ────────────────────────────────────────────────
        //
        // P29.7's steering was INSTANT: the road wheels were at the driver's
        // demand on the frame it arrived, which is a car that changes direction
        // in a sixtieth of a second and cannot be caught once it is sideways. The
        // rack now moves at a rate, and returns to centre faster than it turns
        // away from it, which is what a real one's castor does.
        let limit = steer_limit_deg(&self.tuning, forward_mps);
        let demand = self.controls.steer.clamp(-1.0, 1.0) * limit;
        let returning = demand.abs() < self.steer_deg.abs() || demand * self.steer_deg < 0.0;
        let rate = if returning {
            self.tuning.steer_return_deg_per_s
        } else {
            self.tuning.steer_rate_deg_per_s
        };
        let step = if rate.is_finite() {
            rate.max(0.0) * dt
        } else {
            f64::INFINITY
        };
        self.steer_deg += (demand - self.steer_deg).clamp(-step, step);
        // The speed-sensitive limit binds the RESULT too: slowing into a corner
        // must not leave the wheels at an angle the rack could no longer reach.
        self.steer_deg = self.steer_deg.clamp(-limit, limit);
        let steer_deg = self.steer_deg;
        // The rig's own geometry, for the Ackermann. Derived rather than
        // authored: a track and a wheelbase are facts about where the wheels
        // are, and a class that authored them could disagree with its own rig.
        let half_track = self
            .rig
            .wheels
            .iter()
            .map(|w| w.mount_local.x.abs())
            .fold(0.0f64, f64::max);
        let wheelbase = 2.0
            * self
                .rig
                .wheels
                .iter()
                .map(|w| w.mount_local.z.abs())
                .fold(0.0f64, f64::max);
        let static_load = self.static_load_n(chassis.mass_kg);
        let inertia = if self.tuning.wheel_inertia_kgm2.is_finite() {
            self.tuning.wheel_inertia_kgm2.max(1e-3)
        } else {
            1e-3
        };
        let wheels = self.rig.wheels.len();

        // ── the drivetrain ───────────────────────────────────────────────────
        //
        // `front_torque_split` IS the drivetrain: 0 is rear drive, 1 is front,
        // between is all-wheel at that split. An axle with no wheels hands its
        // share to the other one rather than losing it, so a three-wheeler and a
        // rig whose recogniser found only rear wheels are both still driven.
        //
        // This block comes BEFORE the gearbox because `axle_share` is what the
        // engine is connected THROUGH: it decides the drive torque, the engine
        // braking and the revs alike, and reading the revs off wheels the engine
        // cannot turn is how a rear-drive car ends up shifting on its front axle.
        let front_wheels = self.rig.wheels.iter().filter(|w| w.steered()).count();
        let rear_wheels = wheels - front_wheels;
        let split = if front_wheels == 0 {
            0.0
        } else if rear_wheels == 0 {
            1.0
        } else {
            self.tuning.front_torque_split.clamp(0.0, 1.0)
        };
        let axle_share = |front: bool| -> f64 {
            let (share, n) = if front {
                (split, front_wheels)
            } else {
                (1.0 - split, rear_wheels)
            };
            if n == 0 {
                0.0
            } else {
                share / n as f64
            }
        };

        // ── the gearbox ──────────────────────────────────────────────────────
        //
        // Revs come from the wheels the engine is connected to, through the gear
        // it is in — the DRIVEN wheels, weighted by exactly the share of the
        // driveline each of them is on, so a car with one wheel in the air does
        // not scream and a front-drive car does not read its revs off its rears.
        let mean_omega: f64 = self
            .rig
            .wheels
            .iter()
            .zip(self.wheels.iter())
            .map(|(m, w)| axle_share(m.steered()) * w.omega_rad_s)
            .sum();
        self.shift_left_s = (self.shift_left_s - dt).max(0.0);
        self.rpm = engine_rpm(&self.tuning, mean_omega, self.gear);
        if self.shift_left_s <= 0.0 {
            let want = shift_target(
                &self.tuning,
                self.gear,
                self.rpm,
                self.controls.throttle,
                forward_mps,
            );
            if want != self.gear {
                self.gear = want;
                self.shift_left_s = self.tuning.shift_time_s.max(0.0);
                self.rpm = engine_rpm(&self.tuning, mean_omega, self.gear);
            }
        }
        let ratio = self.tuning.drive_ratio(self.gear);
        let throttle = self.controls.throttle.clamp(-1.0, 1.0).abs();
        // **No torque crosses the box during a shift** — the whole of why a shift
        // is something a driver feels rather than a number that changes.
        let crank = if self.shifting() {
            0.0
        } else {
            engine_torque_nm(&self.tuning, self.rpm)
                * throttle
                * governor(&self.tuning, forward_mps)
        };
        // Reverse is a gear, so its ratio is positive and the DIRECTION is the
        // gear's sign. One sign, spent here.
        let direction = if self.gear < 0 { -1.0 } else { 1.0 };

        // ── the differentials ────────────────────────────────────────────────
        //
        // **A diff shares an axle's TORQUE, and the lock decides how much of it
        // the slower wheel takes.** Open (`0`) is an even split whatever the two
        // wheels are doing, which is why an open axle with one wheel in the air
        // delivers only half its torque and wastes the rest spinning. Spooled
        // (`1`) hands the whole axle to the wheel that is turning slowest — the
        // one that still has grip — which is what a locked diff is FOR.
        //
        // A **speed** average was tried first and refused: pulling a grounded
        // wheel's `ω` up toward a spinning partner's puts momentum into it that
        // the ground then has to absorb, so the wheel that had grip breaks away
        // and the spool delivered LESS force than the open diff (3 083 N against
        // 3 853 at matched revs). Torque is what a differential divides.
        //
        // The transfer ramps in over [`DIFF_SPEED_BAND`] of speed difference so
        // an axle whose wheels are simply rolling together is not thrown between
        // two states by numerical noise, and ties are split evenly, so the answer
        // never depends on which wheel a `Guid` sort happened to put first.
        let axle_lock = |front: bool| -> f64 {
            let v = if front {
                self.tuning.diff_lock_front
            } else {
                self.tuning.diff_lock_rear
            };
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        self.drive_nm.clear();
        self.drive_nm.resize(wheels, 0.0);
        for i in 0..wheels {
            let front = self.rig.wheels[i].steered();
            let n = if front { front_wheels } else { rear_wheels };
            let base = axle_share(front);
            let lock = axle_lock(front);
            let weight = if lock <= 0.0 || n < 2 {
                base
            } else {
                let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                for (j, m) in self.rig.wheels.iter().enumerate() {
                    if m.steered() == front {
                        let w = self.wheels[j].omega_rad_s;
                        lo = lo.min(w);
                        hi = hi.max(w);
                    }
                }
                let ramp = ((hi - lo) / DIFF_SPEED_BAND).clamp(0.0, 1.0);
                let slowest = self
                    .rig
                    .wheels
                    .iter()
                    .enumerate()
                    .filter(|(j, m)| m.steered() == front && self.wheels[*j].omega_rad_s <= lo)
                    .count()
                    .max(1);
                let mine = if self.wheels[i].omega_rad_s <= lo {
                    1.0 / slowest as f64
                } else {
                    0.0
                };
                let axle = base * n as f64;
                base * (1.0 - lock * ramp) + axle * lock * ramp * mine
            };
            // **Traction control**, which is one line once a wheel has a slip
            // ratio: if this wheel was spinning last step, hand it less. Last
            // step's, and that is not a shortcut — a real system measures and
            // then modulates, so a one-step lag is the mechanism rather than an
            // approximation of it.
            // **Traction control**: the torque request, clamped to what this
            // contact patch can take at the aid's own target slip. Last step's
            // load, because the suspension pass has not run yet — a load changes
            // over tens of milliseconds where a wheel's speed changes in one
            // step, so a step-old load is a measurement and a step-old slip is
            // not (see `aid_torque_cap_nm`).
            let mut torque = crank * ratio * direction * weight;
            let tc = self.tuning.traction_control_slip;
            if tc.is_finite() && tc > 0.0 {
                let cap = aid_torque_cap_nm(
                    &self.tuning,
                    tc,
                    self.wheels[i].load_n,
                    static_load,
                    self.rig.wheels[i].radius_m,
                );
                let held = torque.clamp(-cap, cap);
                self.wheels[i].tc_cut = if torque.abs() > 1e-9 {
                    held / torque
                } else {
                    1.0
                };
                torque = held;
            } else {
                self.wheels[i].tc_cut = 1.0;
            }
            self.drive_nm[i] = torque;
        }

        // **The driveline ceiling.** `max_engine_force_n` used to be the whole
        // engine curve; since VEH2a it is the bound on what the wheels may be
        // handed in total, which is the half-shaft and the clutch a car actually
        // has. Scaled rather than clipped per wheel, so the SPLIT stays the split.
        let force_sum: f64 = self
            .rig
            .wheels
            .iter()
            .zip(self.drive_nm.iter())
            .map(|(w, t)| t.abs() / w.radius_m.max(1e-3))
            .sum();
        let ceiling = self.tuning.max_engine_force_n.max(0.0);
        if force_sum > ceiling && force_sum > 0.0 {
            let derate = ceiling / force_sum;
            for t in self.drive_nm.iter_mut() {
                *t *= derate;
            }
        }

        // ── the brakes ───────────────────────────────────────────────────────
        let brake_force = self.tuning.brake_force_n.max(0.0) * self.controls.brake.clamp(0.0, 1.0);
        //
        // Brake bias is the share of the budget the FRONT axle takes. Road cars
        // run it forward because braking transfers load forward and grip follows
        // load — and with `tyre_load_sensitivity` above zero, biasing it the
        // wrong way is now something a driver can feel rather than a number with
        // no consequence.
        let bias = if front_wheels == 0 {
            0.0
        } else if rear_wheels == 0 {
            1.0
        } else {
            self.tuning.brake_bias.clamp(0.0, 1.0)
        };
        let brake_share = |front: bool| -> f64 {
            let (share, n) = if front {
                (bias, front_wheels)
            } else {
                (1.0 - bias, rear_wheels)
            };
            if n == 0 {
                0.0
            } else {
                brake_force * share / n as f64
            }
        };
        // Engine braking: the crank's own drag with the throttle shut, rising
        // with the revs, reaching the wheels through the same gear.
        let rev_span = (self.tuning.redline_rpm - self.tuning.idle_rpm).max(1.0);
        let rev_frac = ((self.rpm - self.tuning.idle_rpm) / rev_span).clamp(0.0, 1.0);
        // Distributed through `axle_share`, exactly like drive torque: an axle
        // the engine cannot turn is an axle the engine cannot slow, so on a
        // rear-drive car the front wheels genuinely coast.
        let engine_brake_total = if throttle > 0.0 || self.shifting() {
            0.0
        } else {
            self.tuning.engine_brake_nm.max(0.0) * rev_frac * ratio.abs()
        };
        // ── stability control ────────────────────────────────────────────────
        //
        // The yaw rate the steering asked for, from the bicycle model:
        // `psi = v · delta / L`, negative for a right turn because a positive
        // steer points the wheels at `+X` and yawing toward `+X` is a rotation
        // about `-Y`. Compared against the yaw rate the car actually has, and the
        // difference decides WHICH wheel to brake:
        //
        // * **oversteer** — the car is rotating faster than it was asked to, so
        //   brake the OUTSIDE FRONT wheel, whose drag yaws the car back out of
        //   the turn;
        // * **understeer** — it is rotating slower, so brake the INSIDE REAR,
        //   whose drag yaws it into the turn.
        //
        // Which is the whole of what a stability system is: one wheel, chosen by
        // the sign of an error.
        let esc_strength = if self.tuning.stability_control.is_finite() {
            self.tuning.stability_control.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let yaw_rate = chassis.angvel.dot(up);
        let yaw_ref = if wheelbase > 0.0 {
            -forward_mps * steer_deg.to_radians() / wheelbase
        } else {
            0.0
        };
        let esc_over = yaw_rate.abs() > yaw_ref.abs();
        let esc_force = if esc_strength > 0.0 && steer_deg != 0.0 {
            let error = (yaw_rate - yaw_ref).abs() - ESC_YAW_TOLERANCE_RAD_S;
            (esc_strength * ESC_GAIN_N_PER_RAD_S * error.max(0.0))
                .min(self.tuning.brake_force_n.max(0.0))
        } else {
            0.0
        };

        // The fastest a wheel may turn in this gear — applied **only to a wheel in
        // the air**, where nothing on the ground is there to stop it. A grounded
        // one is held by the contact patch and, at road speed, by `governor`;
        // clamping its speed clamps the force it can transmit, which is the
        // defect the limiter's own note records.
        //
        // `audit:` VEH2a — the first spelling of this said a grounded wheel was
        // "limited by the fuel cut above", and there is no fuel cut: above the
        // redline `engine_torque_nm` plateaus at `redline_torque_frac` rather
        // than falling to zero. See its own doc for what that costs and where.
        let omega_ceiling = if ratio.abs() > 1e-9 {
            self.tuning.redline_rpm.max(0.0) * std::f64::consts::TAU / 60.0 / ratio.abs()
        } else {
            f64::INFINITY
        };

        // ── the suspension, and the bars that tie its two sides together ─────
        //
        // Two passes over the wheels, and the reason is the anti-roll bar: it
        // transfers load ACROSS an axle, so every load on that axle has to exist
        // before any of them is final. One pass would have to look at a
        // neighbour's state before that neighbour had been visited, which is the
        // shape of bug that reads as "the bar only works on one side".
        for (i, mount) in self.rig.wheels.iter().enumerate() {
            let Some(state) = self.wheels.get_mut(i) else {
                break;
            };
            let Some(contact) = state.contact else {
                state.length_m = self.tuning.rest_length_m;
                state.load_n = 0.0;
                continue;
            };
            let length = (contact.distance_m - mount.radius_m).clamp(
                self.tuning.rest_length_m - self.tuning.travel_m,
                self.tuning.rest_length_m,
            );
            state.length_m = length;
            // Closing speed from the CONTACT POINT's velocity rather than from
            // the length difference: a finite difference over one step is a step
            // behind and rings at exactly the frequency the damper exists to
            // kill.
            let closing = -chassis.point_velocity(contact.point).dot(up);
            state.load_n =
                suspension_force_n(&self.tuning, self.tuning.rest_length_m - length, closing);
        }

        // **The anti-roll bars.** A bar resists the DIFFERENCE in compression
        // across its axle, so it moves load from the inside wheel to the outside
        // one and adds none: the adjustment sums to zero over the axle by
        // construction. With `tyre_load_sensitivity` above zero that transfer
        // COSTS the axle grip, which is why stiffening the front is how a car is
        // made to understeer — and why a bar would be free without a load-
        // sensitive tyre under it.
        for front_axle in [true, false] {
            let rate = if front_axle {
                self.tuning.anti_roll_front_n_per_m
            } else {
                self.tuning.anti_roll_rear_n_per_m
            };
            if !rate.is_finite() || rate <= 0.0 {
                continue;
            }
            let (mut sum, mut n) = (0.0f64, 0usize);
            for (i, mount) in self.rig.wheels.iter().enumerate() {
                if mount.steered() == front_axle {
                    if let Some(w) = self.wheels.get(i) {
                        sum += self.tuning.rest_length_m - w.length_m;
                        n += 1;
                    }
                }
            }
            if n < 2 {
                continue;
            }
            let mean = sum / n as f64;
            for (i, mount) in self.rig.wheels.iter().enumerate() {
                if mount.steered() != front_axle {
                    continue;
                }
                let Some(w) = self.wheels.get_mut(i) else {
                    continue;
                };
                if w.contact.is_none() {
                    continue;
                }
                let compression = self.tuning.rest_length_m - w.length_m;
                w.load_n = (w.load_n + rate * (compression - mean)).max(0.0);
            }
        }

        for (i, mount) in self.rig.wheels.iter().enumerate() {
            let Some(state) = self.wheels.get_mut(i) else {
                break;
            };
            let steer = if mount.steered() {
                ackermann_deg(
                    steer_deg,
                    mount.mount_local.x * steer_deg > 0.0,
                    half_track,
                    wheelbase,
                    self.tuning.ackermann,
                )
            } else {
                0.0
            };
            state.steer_deg = steer;
            let wheel_fwd = if steer == 0.0 {
                fwd
            } else {
                steer_direction(fwd, right, steer)
            };
            let wheel_right = if steer == 0.0 {
                right
            } else {
                steer_direction(right, -fwd, steer)
            };
            let radius = mount.radius_m.max(1e-3);

            // ── the wheel's own equation of motion ───────────────────────────
            //
            // **Drive, then the GROUND, then everything that resists.** The order
            // is not cosmetic and it was measured wrong first: with the brake
            // applied before the ground, a fully locked wheel is pulled to zero,
            // pushed back to 9 rad/s by the contact patch inside the same step,
            // and pulled to zero again — a wheel that chatters between locked and
            // rolling at 30 Hz for as long as the pedal is down. Resolving the
            // brake LAST, against the post-contact speed, is what makes a locked
            // wheel stay locked: the no-reverse clamp then has exactly the
            // ground's contribution to remove.
            let mut omega =
                state.omega_rad_s + self.drive_nm.get(i).copied().unwrap_or(0.0) * dt / inertia;

            // Everything that RESISTS is one budget and it may not reverse the
            // wheel — P29.7's law, moved to where it now belongs. It used to be
            // stated against the chassis's own motion, where a resistive force
            // that overshot crept a braked car 5.8 cm backwards per second; on a
            // wheel the same rule is exact, because a brake that stopped the
            // wheel and kept pulling is a brake that drives it in reverse.
            let hand = if self.controls.handbrake && !mount.steered() {
                self.tuning.handbrake_force_n.max(0.0) * radius
            } else {
                0.0
            };
            let rolling = state.load_n.max(0.0) * self.tuning.rolling_resistance.max(0.0) * radius;
            // **ABS**: the brake's torque request, clamped to what this contact
            // patch can take at the aid's own target slip — traction control's
            // rule with the sign turned round, and the same feed-forward reason
            // (`aid_torque_cap_nm`). The HANDBRAKE is deliberately outside it: a
            // handbrake that could not lock a wheel would not be a handbrake.
            let abs = self.tuning.abs_slip;
            let mut modulated = brake_share(mount.steered()) * radius
                + rolling
                + engine_brake_total * axle_share(mount.steered());
            if abs.is_finite() && abs > 0.0 {
                let cap = aid_torque_cap_nm(&self.tuning, abs, state.load_n, static_load, radius);
                // The WHOLE modulated budget, not the foot brake alone. Rolling
                // resistance and engine braking are small, and they were enough
                // to push a capped brake past the peak and lock the wheel anyway.
                modulated = modulated.min(cap);
            }
            // Stability control's own wheel, and only that one.
            let esc = if esc_force > 0.0
                && mount.steered() == esc_over
                && (mount.mount_local.x * steer_deg < 0.0) == esc_over
            {
                esc_force * radius
            } else {
                0.0
            };
            // The handbrake and stability control sit OUTSIDE the modulation: a
            // handbrake that could not lock a wheel would not be a handbrake, and
            // a stability system that an anti-lock system could veto would not be
            // one either.
            let resist = modulated + hand + esc;
            let shed = |w: f64| {
                if resist > 0.0 && w != 0.0 {
                    w - w.signum() * (resist * dt / inertia).min(w.abs())
                } else {
                    w
                }
            };

            let Some(contact) = state.contact else {
                // In the air the wheel turns on its own torques alone — which is
                // how a wheel that left the ground under power is SEEN to be
                // spinning when it lands. Its length and load were already
                // written by the suspension pass above.
                state.slip_ratio = 0.0;
                state.slip_lat = 0.0;
                state.omega_rad_s = shed(omega).clamp(-omega_ceiling, omega_ceiling);
                state.spin_deg = (state.spin_deg
                    + state.omega_rad_s * dt * 180.0 / std::f64::consts::PI)
                    .rem_euclid(360.0);
                continue;
            };

            let load = state.load_n;
            // The spring pushes along the SUSPENSION axis (the chassis up), not
            // along the contact normal: a raycast wheel on a slope is still held
            // up by its own strut, and projecting onto the normal is how a car
            // slides sideways off a ramp it should drive up.
            out.push(WheelForce {
                point: contact.point,
                force: up * load,
            });

            // ── the contact patch ────────────────────────────────────────────
            //
            // On the TYRE's velocity, not the suspension's — see
            // `ChassisState::contact_velocity` for the pitch pump this closes.
            let tyre_vel = chassis.contact_velocity(contact.point, up);
            let along_v = tyre_vel.dot(wheel_fwd);
            let side_v = tyre_vel.dot(wheel_right);
            let reference = along_v.abs().max(SLIP_REF_MPS);
            let slip_lat = side_v / reference;
            let free = along_v / radius;

            // ── STICK OR SLIDE ───────────────────────────────────────────────
            //
            // The wheel and the tyre are a **stiff** pair: a 1.2 kg·m² wheel on a
            // tyre whose force changes by ~2.5 kN per rad/s has a time constant
            // of about 1.4 ms, and the fixed step is 16.7. Integrating that
            // explicitly does not merely lose accuracy, it makes the longitudinal
            // force a function of the timestep — measured before this branch
            // existed: a free-rolling wheel carrying nothing but engine braking
            // reported **1 084 N** of drag where its own equilibrium is 159.
            //
            // So the sticking case is solved **exactly** instead of integrated.
            // If the force needed to end the step rolling with the road fits
            // inside what the tyre has left after the lateral demand, the wheel
            // sticks and takes precisely that force; otherwise it breaks away and
            // the sliding branch integrates — which is safe, because past the
            // peak the curve is flat or falling and the stiffness is gone. This
            // is the standard stick/slip split and it is the reason the model is
            // stable at 60 Hz without a sub-step.
            let (_, fy_stick) = tyre_force_n(&self.tuning, load, static_load, 0.0, slip_lat);
            let mu_x = load_sensitive_mu(
                self.tuning.longitudinal_grip,
                load,
                static_load,
                self.tuning.tyre_load_sensitivity,
            );
            let mu_y = load_sensitive_mu(
                self.tuning.lateral_grip,
                load,
                static_load,
                self.tuning.tyre_load_sensitivity,
            );
            // The ellipse, read the honest way round: how much of the LATERAL
            // capacity is already spent decides how much LONGITUDINAL is left.
            let cap_y = (mu_y * load).max(0.0);
            let spent = if cap_y > 0.0 {
                (fy_stick / cap_y).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let spare = (mu_x * load).max(0.0) * (1.0 - spent * spent).max(0.0).sqrt();
            // The force the ground must supply to hold this wheel at rolling
            // speed **against every other torque on it** — the drive already
            // folded into `omega`, and the resist budget, which is the term the
            // first cut of this branch forgot. Without it a handbraked wheel on a
            // slope tested as "sticking" (the ground only had to spin it up by
            // 1.4 rad/s), took 294 N instead of locking, and let a parked car
            // creep **0.99 m in two seconds** down the audited 0.108 grade.
            //
            // Its resistive half carries P29.7's law forward unchanged: **a
            // resistive force may not reverse the motion it resists.** The most
            // it may do in one step is bring this wheel's share of the car to a
            // stop. Without that cap the branch reads `(0.0f64).signum() == 1.0`
            // — Rust's answer for a positive zero — and applies a full 12 kN of
            // brake to a car that is already stationary, which drove one
            // backwards at **7.3 cm/s**: the exact defect the law was written
            // for, met again in a new place.
            let inertial = inertia * (omega - free) / (dt * radius);
            let cap = chassis.mass_kg.max(0.0) / wheels.max(1) as f64 * along_v.abs() / dt;
            let braking = (-free.signum() * resist / radius).clamp(-cap, cap);
            let stick_fx = inertial + braking;

            let (fx, fy) = if stick_fx.abs() <= spare {
                // Sticking: the ground holds the wheel at rolling speed and the
                // resist is already paid for inside `stick_fx`, so it is NOT shed
                // again below.
                omega = free;
                state.slip_ratio = 0.0;
                (stick_fx, fy_stick)
            } else {
                let slip_ratio = (omega * radius - along_v) / reference;
                state.slip_ratio = slip_ratio;
                let f = tyre_force_n(&self.tuning, load, static_load, slip_ratio, slip_lat);
                // The ground's reaction may still not push the wheel PAST free
                // rolling in one step; that is the sliding branch's own safety
                // net and it is what a breakaway settles onto.
                let next = omega - f.0 * radius * dt / inertia;
                omega = if (omega - free) * (next - free) < 0.0 {
                    free
                } else {
                    next
                };
                // …and only now the brake, against the speed the ground left
                // behind. See the ordering note above.
                omega = shed(omega);
                f
            };
            state.slip_lat = slip_lat;
            // **The centre of gravity, applied where it costs nothing.**
            //
            // The solver's centre of mass is the chassis collider's centre and
            // this engine has no door to move it. But a force applied at
            // `contact − up·cog_height_m` produces exactly the moment the true
            // centre of gravity would have felt about the real one, and the
            // linear force is unchanged — so the whole of "a low car rolls less"
            // is one subtraction. The SUSPENSION force is untouched by
            // construction, because it is parallel to `up` and `up × up` is zero;
            // only the tyre's horizontal force needs the correction.
            out.push(WheelForce {
                point: contact.point - up * self.tuning.cog_height_m,
                force: wheel_fwd * fx + wheel_right * fy,
            });
            state.omega_rad_s = omega;
            state.spin_deg = (state.spin_deg
                + state.omega_rad_s * dt * 180.0 / std::f64::consts::PI)
                .rem_euclid(360.0);
        }

        // ── the air ──────────────────────────────────────────────────────────
        //
        // Once, at the centre of gravity — not per wheel, because a car in the
        // air has no wheels on the ground and still has air on it.
        let cog = chassis.position + up * self.tuning.cog_height_m;
        if speed > 1e-6 {
            // **Anisotropic.** A car's flank is two to three times its nose, and
            // the difference is the whole reason a slide feels like a slide
            // rather than like ice: sideways motion is what the air resists most.
            // P29.7 had one isotropic coefficient, so a car sliding at 20 m/s met
            // exactly as much air as one driving at 20.
            let along = chassis.linvel.dot(fwd);
            let across = chassis.linvel - fwd * along;
            let mut drag = -fwd * along.abs() * along * self.tuning.drag_n_per_mps2.max(0.0);
            let sideways = across.length();
            if sideways > 1e-9 {
                drag -= across / sideways
                    * sideways
                    * sideways
                    * self.tuning.drag_lateral_n_per_mps2.max(0.0);
            }
            if drag != DVec3::ZERO {
                out.push(WheelForce {
                    point: cog,
                    force: drag,
                });
            }
            // **Downforce**, pressed into the chassis up at its own centre of
            // pressure — so a rear wing is a rear wing rather than extra mass.
            // It adds LOAD and not weight, which is why it buys grip a heavier
            // car does not have.
            let df = self.tuning.downforce_n_per_mps2.max(0.0) * speed * speed;
            if df > 0.0 {
                let cp = chassis.position
                    + chassis.rotation
                        * DVec3::new(
                            0.0,
                            self.tuning.cog_height_m,
                            self.tuning.downforce_centre_z * wheelbase * 0.5,
                        );
                out.push(WheelForce {
                    point: cp,
                    force: -up * df,
                });
            }
        }
    }
}

// ── the boat (wave VEH2c) ───────────────────────────────────────────────────

/// How much of its ahead thrust a screw makes going **astern**.
///
/// A propeller is an aerofoil designed to work in one direction; run backwards
/// it stalls, and the usual figure for a fixed-pitch screw is a little under
/// half. It is also why a boat has no brakes and stops by going astern — which
/// is exactly what [`VehicleControls::from_intent`] hands this class when the
/// driver pulls back at speed, so the mapping needs no special case.
pub const HULL_ASTERN_FRACTION: f64 = 0.45;

/// The share of the screw's own thrust a hard-over rudder can turn sideways.
///
/// A rudder sitting in the propeller race is far more effective than its area
/// suggests — the water reaching it is already moving fast, which is why a boat
/// with no way on can still be swung by a burst of throttle. Ahead of the race
/// there is nothing to deflect, and this constant is the ahead figure.
pub const RUDDER_WASH_GAIN: f64 = 0.35;

/// The rudder's own lift from the hull's motion through the water, as a fraction
/// of [`VehicleTuning::drag_lateral_n_per_mps2`]'s force at the same speed.
///
/// Derived from the keel's number rather than authored as a second one: a rudder
/// is a small vertical surface in the same flow, so its lift scales with the
/// same `v²` and the same water. This is what lets a boat that has closed the
/// throttle still steer while it carries its way — without it a drifting boat is
/// uncontrollable, which is wrong and reads as broken.
///
/// # Why it is well under one percent
///
/// Because the number it is a fraction OF is the whole immersed profile. A
/// hull's resistance to sideways motion is the largest force on a boat that is
/// not gravity — it is why boats do not slide — and a rudder is a fraction of a
/// percent of that area. Both terms carry the same square of speed, so their
/// ratio does not move with speed, which is why a boat's turning circle is
/// roughly constant in hull lengths however fast it is going. It is also why
/// this could be sized once, against a measurement: the launch fixture's steady
/// circle is **10.6 m** for a 4 m hull — about two and a half lengths, and the
/// same to the digit at either helm.
pub const RUDDER_FLOW_GAIN: f64 = 0.006;

/// Where a hull's lateral drag acts, as a fraction of the lever arm the rig
/// names — the bow and quarter points.
///
/// **Two points and not one**, on the buoyancy pass's own argument: a hull
/// resists turning because every section of it resists moving sideways, so
/// applying the keel's force at two points separated along the hull turns the
/// same coefficient into yaw damping. One force at the centre of mass would damp
/// sway and leave a boat spinning like a top.
pub const HULL_DRAG_ARM_FRACTION: f64 = 0.6;

/// **The boat**: a screw, a rudder and a hull, over the wheel-less rig
/// (wave VEH2c).
///
/// # What holds it up is NOT here
///
/// Buoyancy is P20.2's and it stays P20.2's. A boat is a chassis with a
/// [`Buoyancy`](crate::components::Buoyancy) component, so the water pass at
/// fixed-step stage 8 does the Archimedes solve over its four mid-plane samples
/// — including the righting moment that makes it heel and pitch on a swell —
/// and the vehicle door at stage 12 **adds** to that rather than resetting it
/// (`PhysicsBridge3D::is_buoyant`, the interlock P20.2 wrote and P29.7 armed).
/// This class contributes thrust, steering and the hull's own resistance, and
/// nothing vertical at all.
///
/// # The draught bound does NOT bite this hull, and that is a measurement
///
/// `inf_physics::d3::water::sample_geometry`'s doc carries a v1 bound — a convex
/// hull's draught is approximated by its AABB, which "over-states the waterplane
/// of anything that is not a box" — and names the fix as belonging "with
/// whatever first needs floating debris". A boat was the obvious candidate. It
/// is not one: the bound is on the `ConvexHull` and `Trimesh` branches, and a
/// boat's chassis is a **`Box`**, whose branch states its own half-extent
/// exactly. The sampled draught of this hull is therefore exact rather than
/// optimistic, and the waterplane-section fix stays where its own doc put it.
///
/// # Which of the sixty-two tunables it reads
///
/// Every one of them by the meaning its own name already has, which is why this
/// class needed no schema window (the VEH2a one is spent):
///
/// | tunable | what it is on a boat |
/// |---|---|
/// | `max_engine_force_n` | peak ahead thrust at the screw, newtons |
/// | `max_speed_mps` | the speed the thrust falls to zero at |
/// | `drag_n_per_mps2` | the hull's resistance along its length |
/// | `drag_lateral_n_per_mps2` | the keel — resistance to sideways motion |
/// | `max_steer_deg` / `min_steer_deg` | rudder angle at rest / at full speed |
/// | `steer_rate_deg_per_s` / `steer_return_deg_per_s` | how fast the wheel turns it |
/// | `enter_time_s`, `enter_warp_start`, `enter_warp_end` | the seat warp, unchanged |
///
/// The other **fifty-one** are accepted by [`tune`](Self::tune) — a catalogue row is
/// one table and refusing half of it would leave an author's file half-read —
/// and are not consulted. A gearbox on a boat is a thing that does not exist.
#[derive(Clone, Debug)]
pub struct HullVehicle {
    rig: VehicleRig,
    tuning: VehicleTuning,
    controls: VehicleControls,
    /// The rudder's actual angle, degrees, positive to starboard — the RACK's
    /// own state, exactly as `RaycastVehicle`'s steer angle is, so a rudder takes
    /// time to come over and comes back faster than it goes across.
    rudder_deg: f64,
    /// How much of the hull was in the water at the last solve, `[0, 1]` —
    /// published for a readout and a test rather than held for the model, which
    /// recomputes it.
    immersion: f64,
    /// Nothing. A hull has no wheels, and the trait's wheel channel is the
    /// suspension's; answering an empty slice is the honest shape.
    no_wheels: Vec<WheelState>,
}

impl HullVehicle {
    /// Build one over a derived rig, with the default tuning.
    pub fn new(rig: VehicleRig) -> Self {
        Self {
            rig,
            tuning: VehicleTuning::default(),
            controls: VehicleControls::default(),
            rudder_deg: 0.0,
            immersion: 0.0,
            no_wheels: Vec::new(),
        }
    }

    /// The tuning, for a test or a UI to read.
    pub fn tuning(&self) -> &VehicleTuning {
        &self.tuning
    }

    /// The rudder's angle this step, degrees, positive to starboard.
    pub fn rudder_deg(&self) -> f64 {
        self.rudder_deg
    }

    /// How much of the hull was under water at the last solve, `[0, 1]`.
    pub fn immersion(&self) -> f64 {
        self.immersion
    }

    /// The hull's half-length, metres — **derived from the rig**, never
    /// authored.
    ///
    /// The aftmost thruster the scene named is where the screw is, and a screw
    /// is at the stern; a class that carried its own length could disagree with
    /// its own geometry, which is the defect `RaycastVehicle`'s derived
    /// wheelbase exists to avoid. A rig whose parts are all amidships answers
    /// the seat height instead, so the lever is never zero.
    fn half_length(&self) -> f64 {
        self.rig
            .parts_of(PartKind::Thruster)
            .map(|(_, p)| p.mount_local.z.abs())
            .fold(0.0f64, f64::max)
            .max(self.rig.seat_local.y.abs())
            .max(0.1)
    }

    /// How much of the hull is under `water_y`, `[0, 1]`.
    ///
    /// The hull's half-height is the seat's own — `chassis_of` derives the seat
    /// as the collider's top face, so the number is the collider's rather than a
    /// second opinion about it.
    fn hull_immersion(&self, chassis: &ChassisState) -> f64 {
        let Some(surface) = chassis.water_y else {
            return 0.0;
        };
        let half = self.rig.seat_local.y.abs().max(1e-6);
        ((surface - (chassis.position.y - half)) / (2.0 * half)).clamp(0.0, 1.0)
    }
}

impl Vehicle for HullVehicle {
    fn rig(&self) -> &VehicleRig {
        &self.rig
    }

    fn set_rig(&mut self, rig: VehicleRig) {
        self.rig = rig;
    }

    fn wheels(&self) -> &[WheelState] {
        &self.no_wheels
    }

    fn wheels_mut(&mut self) -> &mut [WheelState] {
        &mut self.no_wheels
    }

    fn control(&mut self, controls: VehicleControls) {
        self.controls = controls;
    }

    fn tune(&mut self, name: &str, value: f64) -> bool {
        self.tuning.set(name, value)
    }

    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
        (self.tuning.enter_time_s, self.tuning.enter_window())
    }

    /// Zero: a hull has no suspension, so a ray anchor would be the mount
    /// itself. There are no rays either, this rig having no wheels — but the
    /// trait's contract is answered rather than left to a default.
    fn suspension_rest_m(&self) -> f64 {
        0.0
    }

    fn gear(&self) -> i32 {
        // A boat has no gearbox and a telegraph has three positions. `1` ahead,
        // `-1` astern, `0` stopped — which `drive_readout` already draws as
        // "1" / "R" / "N", and which is the truth about a boat rather than an
        // imitation of a car's.
        let demand = self.controls.throttle - self.controls.brake;
        if demand > 0.05 {
            1
        } else if demand < -0.05 {
            -1
        } else {
            0
        }
    }

    fn engine_state(&self, forward_mps: f64) -> (f64, f64) {
        let top = self.tuning.max_speed_mps.max(1e-6);
        let revs = (forward_mps.abs() / top).clamp(0.0, 1.0);
        let load = (self.controls.throttle - self.controls.brake)
            .abs()
            .min(1.0);
        // A screw out of the water races and a screw driving a hull is loaded,
        // so the sound follows the immersion for the same reason the thrust
        // does.
        (revs.max(load * 0.35), load * self.immersion.max(0.15))
    }

    /// The rudder, drawn. Yaw only, in the part's own frame.
    fn part_pose(&self, index: usize) -> Option<Vec3d> {
        let part = self.rig.parts.get(index)?;
        (part.kind == PartKind::Thruster).then(|| Vec3d::new(0.0, self.rudder_deg, 0.0))
    }

    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>) {
        if !dt.is_finite() || dt <= 0.0 || !(chassis.mass_kg.is_finite() && chassis.mass_kg > 0.0) {
            return;
        }
        let (fwd, right, _up) = chassis.basis();
        let forward_mps = chassis.linvel.dot(fwd);
        self.immersion = self.hull_immersion(&chassis);

        // ── the rudder rack. The car's rule, by its own names: a rate to go
        //    across, a faster one to come back, and a limit that tightens with
        //    speed (a rudder hard over at twenty knots is how a boat broaches).
        let limit = steer_limit_deg(&self.tuning, forward_mps);
        let want = self.controls.steer.clamp(-1.0, 1.0) * limit;
        let returning = want.abs() < self.rudder_deg.abs() || want * self.rudder_deg < 0.0;
        let rate = if returning {
            self.tuning.steer_return_deg_per_s
        } else {
            self.tuning.steer_rate_deg_per_s
        };
        let step = if rate.is_finite() {
            rate.max(0.0) * dt
        } else {
            f64::INFINITY
        };
        self.rudder_deg += (want - self.rudder_deg).clamp(-step, step);
        self.rudder_deg = self.rudder_deg.clamp(-limit, limit);

        // ── the screw. Ahead is the throttle and astern is the brake, because
        //    `from_intent` has already decided which of the two a pull-back at
        //    speed is — and a boat stops by going astern either way.
        let ahead = self.controls.throttle.clamp(-1.0, 1.0);
        let astern = self.controls.brake.clamp(0.0, 1.0);
        let demand = if ahead >= 0.0 {
            ahead - astern * HULL_ASTERN_FRACTION
        } else {
            (ahead - astern) * HULL_ASTERN_FRACTION
        };
        let top = self.tuning.max_speed_mps.max(1e-6);
        // The thrust falls off as the hull approaches its own top speed — the
        // same shape `RaycastVehicle` uses and the same reason: a screw at the
        // speed of its own wake is doing no work.
        let falloff = if demand == 0.0 {
            0.0
        } else {
            (1.0 - (forward_mps / top).clamp(-1.0, 1.0) * demand.signum()).clamp(0.0, 1.0)
        };
        let peak = self.tuning.max_engine_force_n.max(0.0);
        let lateral = self.tuning.drag_lateral_n_per_mps2.max(0.0);
        let thrusters: Vec<(DVec3, f64)> = self
            .rig
            .parts_of(PartKind::Thruster)
            .map(|(_, p)| {
                let point = chassis.position + chassis.rotation * p.mount_local.to_dvec3();
                // A screw pushes only while it is IN the water, and its own
                // half-height is the band it fades over: a boat lifted clear by
                // a swell loses its bite and gets it back on the way down.
                let bite = match chassis.water_y {
                    Some(surface) => {
                        let band = (2.0 * p.size.y).max(1e-6);
                        ((surface - point.y) / band).clamp(0.0, 1.0)
                    }
                    None => 0.0,
                };
                (point, bite)
            })
            .collect();
        let share = if thrusters.is_empty() {
            0.0
        } else {
            1.0 / thrusters.len() as f64
        };
        let rad = self.rudder_deg.to_radians();
        for (point, bite) in &thrusters {
            let thrust_n = demand * peak * falloff * bite * share;
            if thrust_n != 0.0 {
                out.push(WheelForce {
                    point: *point,
                    force: fwd * thrust_n,
                });
            }
            // ── the rudder: the screw's race turned sideways, plus the rudder's
            //    own lift from the hull's way through the water. Applied AT THE
            //    STERN, which is what makes it a yaw moment and a little sway —
            //    exactly what a real rudder does. A turn to starboard swings the
            //    stern to port, so the force is along `-right`.
            let wash = thrust_n.abs() * RUDDER_WASH_GAIN;
            let flow = lateral * forward_mps * forward_mps * RUDDER_FLOW_GAIN * share;
            let side = inf_math::psin64(rad) * (wash + flow) * bite;
            if side != 0.0 {
                out.push(WheelForce {
                    point: *point,
                    force: -right * side,
                });
            }
        }

        // ── the hull. Two resistances, both scaled by how much of it is wet: a
        //    boat out of the water is a box, and a box has neither.
        let wet = self.immersion;
        if wet > 0.0 {
            let along = self.tuning.drag_n_per_mps2.max(0.0);
            let drag_n = along * forward_mps * forward_mps.abs() * wet;
            if drag_n != 0.0 {
                out.push(WheelForce {
                    point: chassis.position,
                    force: -fwd * drag_n,
                });
            }
            // The keel, at the bow and the quarter — see `HULL_DRAG_ARM_FRACTION`
            // for why two points and not one.
            let arm = self.half_length() * HULL_DRAG_ARM_FRACTION;
            for sign in [1.0f64, -1.0] {
                let point = chassis.position + fwd * (arm * sign);
                let v = chassis.point_velocity(point).dot(right);
                let f = lateral * v * v.abs() * wet * 0.5;
                if f != 0.0 {
                    out.push(WheelForce {
                        point,
                        force: -right * f,
                    });
                }
            }
        }
    }
}

// ── the helicopter (wave VEH2c) ─────────────────────────────────────────────

/// **A pure torque, as the pair of forces that makes one.**
///
/// [`WheelForce`] is a force at a point, which is the whole force channel this
/// engine's vehicles have — and it is enough, because a couple is two of them.
/// Given a desired world torque this returns two equal and opposite forces
/// `lever` metres either side of `centre` whose moment is exactly that torque
/// and whose sum is exactly zero, so nothing translates.
///
/// The arm is chosen from the world axis **least** aligned with the torque, and
/// that choice is a pure function of the torque, so two runs and two hosts pick
/// the same one. A zero (or non-finite) torque answers `None` rather than
/// dividing by a length.
pub fn torque_pair(centre: DVec3, torque: DVec3, lever: f64) -> Option<[WheelForce; 2]> {
    let len = torque.length();
    if !len.is_finite() || len <= 0.0 || !(lever.is_finite() && lever > 0.0) {
        return None;
    }
    let axis = torque / len;
    // The world axis this torque leans on least — deterministic, and never
    // parallel to the torque, so the cross product below cannot vanish.
    let pick = if axis.x.abs() <= axis.y.abs() && axis.x.abs() <= axis.z.abs() {
        DVec3::X
    } else if axis.y.abs() <= axis.z.abs() {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let arm = axis.cross(pick).normalize_or_zero();
    if arm == DVec3::ZERO {
        return None;
    }
    // tau = 2 * lever * (arm x force), and force = (tau x arm) / (2 * lever)
    // satisfies it exactly when `arm` is perpendicular to `tau`, which it is by
    // construction.
    let force = torque.cross(arm) / (2.0 * lever);
    Some([
        WheelForce {
            point: centre + arm * lever,
            force,
        },
        WheelForce {
            point: centre - arm * lever,
            force: -force,
        },
    ])
}

/// How much thrust full collective adds above the hover setting, as a fraction.
///
/// `MovementMode::Flying`'s 6-DOF flycam is the FEEL reference this wave was
/// told to steer by — it rises at `FLY_ASCEND_MPS`, 8 m/s, with no gravity at
/// all. A rotorcraft cannot answer that with a velocity because it has weight;
/// what it can answer is an excess of thrust, and 35 % of hover against a
/// fuselage's own drag puts the fixture's climb in the same country as the
/// flycam's rise without pretending gravity is off.
pub const HELI_CLIMB_AUTHORITY: f64 = 0.35;

/// The cosine of the bank angle past which the collective governor gives up.
///
/// The governor below holds the *vertical* component of the rotor's thrust, so
/// a coordinated turn does not sink; past about seventy degrees of bank there
/// is so little vertical left that holding it would need a rotor four times its
/// own size, and the honest answer is that the aircraft falls. It is the one
/// edge of the governor and it is a cliff on purpose: an aircraft that could be
/// rolled inverted and still hold height is not one.
pub const HELI_MIN_LIFT_COS: f64 = 0.35;

/// Yaw rate the pedals ask for at full deflection, degrees per second.
///
/// A Ring-0 feel constant rather than a tunable, for the reason
/// `movement::FLY_SPEED_MPS` is one: it is what the control MEANS, not what a
/// particular airframe is. 90 deg/s is four seconds for a full circle on the
/// spot, which is a brisk pedal turn and is about what a light helicopter does.
pub const HELI_YAW_RATE_DEG_PER_S: f64 = 90.0;

/// How far the machine banks into a turn, degrees per (radian per second) of
/// yaw rate per (metre per second) of forward speed.
///
/// **The turn is coordinated and the pilot does not fly it** — see
/// [`RotorVehicle`]'s own note for what that costs. The number is the
/// coordinated-turn relation `tan(bank) = V * omega / g` linearized about
/// zero, which is `bank_rad ~= V * omega / 9.81`; in degrees that is
/// `57.2958 / 9.81`.
pub const HELI_BANK_PER_TURN_DEG: f64 = 5.84;

/// The radius of gyration of a rotorcraft's airframe, as a fraction of its own
/// disc radius.
///
/// A helicopter's fuselage is about as long as its rotor is wide, and that is a
/// fact about helicopters rather than a convenience: the tail boom exists to
/// keep the tail rotor clear of the main disc, so it reaches roughly the disc's
/// edge. Modelled as a uniform rod of length `2R` about its own centre, whose
/// radius of gyration is `R / sqrt(3)`.
///
/// It is what turns [`HELI_ATTITUDE_KP`] into a torque, which is why those gains
/// are a property of the CONTROL and not of the airframe: a heavier or a larger
/// machine gets a proportionally heavier hand and settles in the same time.
pub const ROTOR_GYRADIUS_FRACTION: f64 = 0.577_350_269_189_625_8;

/// The attitude hold's proportional and derivative gains, as an angular
/// acceleration per unit of sine error and per radian per second.
///
/// Sized together rather than separately: the pair is critically damped at the
/// fixture's own inertia, which is what makes the machine settle onto a commanded
/// attitude instead of ringing. A derivative term of zero is the mutation that
/// reds `a_helicopter_settles_on_the_attitude_it_is_asked_for`.
pub const HELI_ATTITUDE_KP: f64 = 6.0;
/// The derivative half of [`HELI_ATTITUDE_KP`].
pub const HELI_ATTITUDE_KD: f64 = 2.2;

/// How fast a drawn rotor turns, degrees per second, at full power.
///
/// A **visual** number and nothing else — no part of the model reads it and no
/// force depends on it, exactly as `WheelState::spin_deg` is visual. It is fast
/// enough to read as a blur at 60 Hz and it is not a real rotor's RPM, which
/// would strobe.
pub const ROTOR_SPIN_DEG_PER_S: f64 = 1_440.0;

/// **The helicopter**: a governed collective, a commanded attitude and a tail
/// rotor, over the wheel-less rig (wave VEH2c).
///
/// # What the pilot commands is an ATTITUDE, not a torque
///
/// The stick's fore-and-aft is a **pitch attitude** and the stick's sideways is
/// a **yaw rate**; the collective is `VehicleControls::vertical`. An attitude
/// hold drives the fuselage onto what the stick asked for and the rotor's thrust
/// — which acts along the fuselage's own up axis, at the hub — does the rest:
/// nose down tilts the thrust vector forward and the machine accelerates. The
/// translation is not commanded anywhere. It emerges, which is how a helicopter
/// actually works and is why this needs no lift model and no aerofoil.
///
/// # Three refusals, stated rather than discovered
///
/// * **The turn is coordinated and the pilot does not fly it** — up to a
///   twentieth of the pedal, and no further. Bank is derived from the yaw rate
///   and the forward speed by the coordinated-turn relation, so the pilot has
///   no way to ASK for a skid; that is a stability augmentation system, which
///   every helicopter of this century has, and it is also why there is no
///   sideways strafe, which is carried rather than hidden.
///
///   **But the derived bank is then clamped by [`steer_limit_deg`], which is
///   the PITCH stick's speed-tapered authority — a road car's steering rack —
///   and that clamp knows nothing about the pedal.** The coordinated bank grows
///   with the turn and the rack shrinks with speed, so they cross early:
///   measured on the 26 kN fixture from the cruise, a 0.05 pedal holds its
///   coordinated 18.4 deg exactly, a 0.10 pedal holds **20.9 deg of a turn
///   wanting 32.1**, and a 0.25 pedal holds **24.2 of 51.0**. *Above a
///   twentieth of a pedal this machine skids, and the refusal above was written
///   without that sentence.*
///   `a_turn_at_speed_saturates_the_bank_on_the_pitch_sticks_limit`
///   (`inf-physics/tests/vehicle_3d.rs`) measures it; giving the bank a limit
///   of its own, derived from the rotor's ceiling rather than borrowed from the
///   pitch stick, is a flight-model change and is carried at that size.
/// * **The collective is governed.** At neutral it holds the *vertical* part of
///   the thrust, so a coordinated turn does not sink. Past
///   [`HELI_MIN_LIFT_COS`] it gives up and the aircraft falls. There is no
///   autorotation and no engine failure: a governed collective cannot be lost.
///
///   **Neither of the governor's two edges is reachable from the stick.** The
///   rotor's own ceiling (about 56 deg of bank on the fixture's 26 kN) and
///   [`HELI_MIN_LIFT_COS`] (about 70 deg) both sit far above the 26 deg the
///   rack allows at rest, so a pilot cannot fly to either; the two arms that
///   measure them get there by rotating the chassis directly, which is a state
///   a collision could produce and a turn cannot.
/// * **The rotor disc is rigid with the mast.** A real cyclic tilts the disc
///   relative to the shaft and the fuselage follows it; here the thrust is along
///   the fuselage's own up and the attitude hold is what moves the fuselage.
///   The difference is a lag of a few tenths of a second in the first instant of
///   a stick input, and buying it would need a disc state with its own flapping
///   dynamics. Carried, with its size named.
///
/// # Which of the sixty-two tunables it reads
///
/// | tunable | what it is on a helicopter |
/// |---|---|
/// | `max_engine_force_n` | the rotor's thrust ceiling, newtons |
/// | `max_speed_mps` | the speed the tilt authority tapers to its minimum by — and **only** that: unlike a car's, this class's thrust does not taper with speed, so the field does not set a top speed here (the fuselage's drag does, at 38.7 m/s against the shipped row's 70) |
/// | `max_steer_deg` / `min_steer_deg` | cyclic tilt at the hover / at that speed |
/// | `steer_rate_deg_per_s` / `steer_return_deg_per_s` | how fast the stick moves the command |
/// | `drag_n_per_mps2` | the fuselage's drag along its length |
/// | `drag_lateral_n_per_mps2` | the fuselage's drag sideways and vertically |
/// | `enter_time_s`, `enter_warp_start`, `enter_warp_end` | the seat warp, unchanged |
#[derive(Clone, Debug)]
pub struct RotorVehicle {
    rig: VehicleRig,
    tuning: VehicleTuning,
    controls: VehicleControls,
    /// The commanded pitch attitude, degrees, nose-up positive — the stick's own
    /// state, rate-limited exactly as a steering rack is.
    pitch_cmd_deg: f64,
    /// The rotor's drawn azimuth, degrees, wrapped to `[0, 360)`.
    azimuth_deg: f64,
    /// The rotor thrust the last solve asked for, newtons — published for a
    /// readout and a test.
    thrust_n: f64,
    /// Nothing: a rotorcraft has no wheels.
    no_wheels: Vec<WheelState>,
}

impl RotorVehicle {
    /// Build one over a derived rig, with the default tuning.
    pub fn new(rig: VehicleRig) -> Self {
        Self {
            rig,
            tuning: VehicleTuning::default(),
            controls: VehicleControls::default(),
            pitch_cmd_deg: 0.0,
            azimuth_deg: 0.0,
            thrust_n: 0.0,
            no_wheels: Vec::new(),
        }
    }

    /// The tuning, for a test or a UI to read.
    pub fn tuning(&self) -> &VehicleTuning {
        &self.tuning
    }

    /// The commanded pitch attitude this step, degrees, nose-up positive.
    pub fn pitch_cmd_deg(&self) -> f64 {
        self.pitch_cmd_deg
    }

    /// The rotor thrust the last solve asked for, newtons.
    pub fn thrust_n(&self) -> f64 {
        self.thrust_n
    }

    /// The rotor hub in the chassis frame and the disc's radius — **derived**
    /// from the rig, so a class cannot disagree with its own geometry.
    ///
    /// The hub is where the thrust acts and the radius is the lever the attitude
    /// hold's couples use, which is why the marker's size is read rather than
    /// decorative. A rig with several rotors uses the largest disc and their
    /// mean position, which is what a tandem's thrust line is.
    /// **The fuselage's own drag** — two coefficients, along and across, both
    /// quadratic (wave VEH2c).
    ///
    /// Its own function because it runs on BOTH paths: a machine nobody is
    /// flying still falls through air rather than through vacuum, and a drag
    /// that only applied to a flown aircraft would make an abandoned one
    /// accelerate for ever.
    fn fuselage_drag(&self, chassis: ChassisState, fwd: DVec3, out: &mut Vec<WheelForce>) {
        let along = self.tuning.drag_n_per_mps2.max(0.0);
        let across = self.tuning.drag_lateral_n_per_mps2.max(0.0);
        let v = chassis.linvel;
        let v_along = v.dot(fwd);
        let rest = v - fwd * v_along;
        let drag = -fwd * (along * v_along * v_along.abs()) - rest * (across * rest.length());
        if drag != DVec3::ZERO {
            out.push(WheelForce {
                point: chassis.position,
                force: drag,
            });
        }
    }

    fn hub(&self) -> (Vec3d, f64) {
        let mut sum = Vec3d::ZERO;
        let mut radius = 0.0f64;
        let mut n = 0.0f64;
        for (_, p) in self.rig.parts_of(PartKind::Rotor) {
            sum = Vec3d::new(
                sum.x + p.mount_local.x,
                sum.y + p.mount_local.y,
                sum.z + p.mount_local.z,
            );
            radius = radius.max(p.size.x.abs());
            n += 1.0;
        }
        if n == 0.0 {
            return (Vec3d::new(0.0, self.rig.seat_local.y, 0.0), 1.0);
        }
        (Vec3d::new(sum.x / n, sum.y / n, sum.z / n), radius.max(0.5))
    }
}

impl Vehicle for RotorVehicle {
    fn rig(&self) -> &VehicleRig {
        &self.rig
    }

    fn set_rig(&mut self, rig: VehicleRig) {
        self.rig = rig;
    }

    fn wheels(&self) -> &[WheelState] {
        &self.no_wheels
    }

    fn wheels_mut(&mut self) -> &mut [WheelState] {
        &mut self.no_wheels
    }

    fn control(&mut self, controls: VehicleControls) {
        self.controls = controls;
    }

    fn tune(&mut self, name: &str, value: f64) -> bool {
        self.tuning.set(name, value)
    }

    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
        (self.tuning.enter_time_s, self.tuning.enter_window())
    }

    fn suspension_rest_m(&self) -> f64 {
        0.0
    }

    fn gear(&self) -> i32 {
        // A helicopter has no gearbox and nothing a telegraph would say. `0`,
        // which `drive_readout` draws as "N" — and which is why wave VEH2c gives
        // an aircraft its own readout instead of borrowing a car's.
        0
    }

    fn engine_state(&self, _forward_mps: f64) -> (f64, f64) {
        let peak = self.tuning.max_engine_force_n.max(1e-6);
        let revs = (self.thrust_n / peak).clamp(0.0, 1.0);
        // A rotor under load is heard, not seen: the note is the collective.
        let load = (0.5 + 0.5 * self.controls.vertical).clamp(0.0, 1.0);
        (revs.max(0.35), load)
    }

    /// The rotor, drawn: an azimuth about the mast, and nothing else.
    fn part_pose(&self, index: usize) -> Option<Vec3d> {
        let part = self.rig.parts.get(index)?;
        (part.kind == PartKind::Rotor).then(|| Vec3d::new(0.0, self.azimuth_deg, 0.0))
    }

    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>) {
        if !dt.is_finite() || dt <= 0.0 || !(chassis.mass_kg.is_finite() && chassis.mass_kg > 0.0) {
            return;
        }
        let (fwd, right, up) = chassis.basis();
        let forward_mps = chassis.linvel.dot(fwd);
        let (hub_local, disc_r) = self.hub();
        let hub = chassis.position + chassis.rotation * hub_local.to_dvec3();
        let weight = chassis.mass_kg * 9.81;

        // ── the collective, GOVERNED. Neutral holds the vertical component of
        //    the thrust, so a coordinated turn does not sink; full up adds
        //    `HELI_CLIMB_AUTHORITY` of hover and full down takes it away. Past
        //    `HELI_MIN_LIFT_COS` of bank the governor gives up and it falls.
        //
        // **A machine nobody is flying has its rotor STOPPED**, and that is not
        // a nicety: the governed collective's neutral is a HOVER, so an unmanned
        // helicopter would carry its own weight and drift off its pad for ever.
        // Measured before this line existed — the harbour gate's own air leg
        // flew the parked machine 476 m and 12 m under the world with nobody
        // aboard. `VehicleControls::occupied` is what a stick at rest and no
        // hand on it differ by.
        if !self.controls.occupied {
            self.thrust_n = 0.0;
            // The stick goes back to centre with the pilot: a command left
            // standing would be applied the instant somebody else climbed in.
            self.pitch_cmd_deg = 0.0;
            // …but the FUSELAGE is still a fuselage. An unmanned machine falls
            // through air rather than through vacuum, so the drag runs HERE
            // before the return rather than being skipped with everything else.
            self.fuselage_drag(chassis, fwd, out);
            return;
        }
        let lift_cos = up.y.max(HELI_MIN_LIFT_COS);
        let collective = self.controls.vertical.clamp(-1.0, 1.0);
        let hover = weight / lift_cos;
        let peak = self.tuning.max_engine_force_n.max(0.0);
        self.thrust_n = (hover * (1.0 + collective * HELI_CLIMB_AUTHORITY)).clamp(0.0, peak);
        if self.thrust_n > 0.0 {
            out.push(WheelForce {
                point: hub,
                force: up * self.thrust_n,
            });
        }

        // ── the stick. Fore-and-aft is a PITCH ATTITUDE, rate-limited exactly
        //    as a steering rack is and tapered with speed by the same law
        //    (`steer_limit_deg`), because a full-authority cyclic at a hundred
        //    knots is how a machine is over-stressed.
        let limit = steer_limit_deg(&self.tuning, forward_mps);
        // Nose DOWN to go forward, so a forward stick is a negative pitch.
        let want = -self.controls.throttle.clamp(-1.0, 1.0) * limit;
        let returning = want.abs() < self.pitch_cmd_deg.abs() || want * self.pitch_cmd_deg < 0.0;
        let rate = if returning {
            self.tuning.steer_return_deg_per_s
        } else {
            self.tuning.steer_rate_deg_per_s
        };
        let step = if rate.is_finite() {
            rate.max(0.0) * dt
        } else {
            f64::INFINITY
        };
        self.pitch_cmd_deg += (want - self.pitch_cmd_deg).clamp(-step, step);
        self.pitch_cmd_deg = self.pitch_cmd_deg.clamp(-limit, limit);

        // ── the pedals, and the bank that goes with them. The yaw rate is the
        //    command; the bank is DERIVED from it and the speed by the
        //    coordinated-turn relation, which is what makes a turn a turn rather
        //    than a skid. See this type's own note for what that refuses.
        //
        // `turn` is positive to the RIGHT, which is what the stick means
        // (`VehicleControls::steer`). About the machine's own up axis that is a
        // NEGATIVE rate — the same sentence `VehicleControls::steer`'s own doc
        // has carried since P29.7 (*"a right turn is a negative yaw rate — see
        // the door, where the sign is applied once"*), and this is that door for
        // an aircraft.
        let turn = self.controls.steer.clamp(-1.0, 1.0) * HELI_YAW_RATE_DEG_PER_S.to_radians();
        let yaw_cmd = -turn;
        // …and a right turn drops the RIGHT wing, which is a positive bank.
        let bank_cmd = (turn * forward_mps * HELI_BANK_PER_TURN_DEG).clamp(-limit, limit);

        // ── the attitude hold. Errors in SINE space, which is where the basis
        //    already is: no inverse trigonometry, and therefore nothing whose
        //    range reduction could differ between two machines.
        let pitch_err = inf_math::psin64(self.pitch_cmd_deg.to_radians()) - fwd.y;
        let roll_err = inf_math::psin64(bank_cmd.to_radians()) - (-right.y);
        let pitch_rate = chassis.angvel.dot(right);
        let roll_rate = chassis.angvel.dot(fwd);
        let yaw_rate = chassis.angvel.dot(up);
        // The gains are an angular ACCELERATION, so the torque is that times the
        // airframe's own inertia — see `ROTOR_GYRADIUS_FRACTION` for where the
        // inertia comes from and why it is derived from the disc.
        let gain = chassis.mass_kg * (disc_r * ROTOR_GYRADIUS_FRACTION).powi(2);
        // THE AXES, spelled out, because getting one of them backwards is a
        // positive feedback loop rather than a wrong number and this model was
        // written with two of them backwards:
        //
        // * `right x fwd == -up`, so a torque along **+right pitches the nose
        //   DOWN**; the nose-UP axis is therefore `-right` and `pitch_rate`
        //   (`angvel . right`) is a nose-DOWN rate.
        // * `fwd x right == +up`, so a torque along **+fwd raises the RIGHT
        //   wing**; the wing-down (positive bank) axis is `-fwd` and
        //   `roll_rate` is a wing-UP rate.
        // * yaw is a rate controller about `+up` and needs no such care: it
        //   drives `angvel . up` at the demand, whatever the sign of either.
        //
        // Which is why both damping terms ADD to their proportional term: each
        // rate is measured about the axis opposite the error's.
        let tau = -right * ((HELI_ATTITUDE_KP * pitch_err + HELI_ATTITUDE_KD * pitch_rate) * gain)
            - fwd * ((HELI_ATTITUDE_KP * roll_err + HELI_ATTITUDE_KD * roll_rate) * gain)
            + up * ((yaw_cmd - yaw_rate) * HELI_ATTITUDE_KD * gain);
        if let Some(pair) = torque_pair(chassis.position, tau, disc_r) {
            out.extend_from_slice(&pair);
        }

        // ── the fuselage. Two coefficients, along and across, both quadratic —
        //    and this is what gives the machine a top speed: nose down tilts the
        //    thrust forward until the drag catches it.
        self.fuselage_drag(chassis, fwd, out);

        // ── the blade, drawn. Visual only; nothing reads it back.
        let turn = ROTOR_SPIN_DEG_PER_S * (self.thrust_n / peak.max(1e-6)).clamp(0.0, 1.0) * dt;
        self.azimuth_deg = (self.azimuth_deg + turn).rem_euclid(360.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One fixed step, seconds — the rate both hosts run at.
    const DT: f64 = 1.0 / 60.0;

    // ── the boat (wave VEH2c) ───────────────────────────────────────────────

    /// A 5 m hull with one screw at the stern, floating with its deck clear.
    fn hull_rig() -> VehicleRig {
        VehicleRig {
            chassis: Uuid::from_u128(0xB0A7),
            // A 1.6 m half-height hull, so the seat (= the top face) is 0.8.
            seat_local: Vec3d::new(0.0, 0.8, 0.0),
            wheels: Vec::new(),
            parts: vec![PartMount {
                guid: Uuid::from_u128(0x5C7E),
                kind: PartKind::Thruster,
                mount_local: Vec3d::new(0.0, -0.5, -2.2),
                size: Vec3d::new(0.25, 0.2, 0.25),
            }],
        }
    }

    /// The chassis of a hull afloat with `water_y` at zero: the origin sits on
    /// the waterline, so the hull is exactly half in.
    fn afloat(linvel: DVec3, angvel: DVec3) -> ChassisState {
        ChassisState {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linvel,
            angvel,
            mass_kg: 2_000.0,
            water_y: Some(0.0),
        }
    }

    /// Sum the forces a solve produced, and the yaw torque about the origin.
    fn resultant(out: &[WheelForce], about: DVec3) -> (DVec3, f64) {
        let mut f = DVec3::ZERO;
        let mut tau = DVec3::ZERO;
        for w in out {
            f += w.force;
            tau += (w.point - about).cross(w.force);
        }
        (f, tau.y)
    }

    /// **A boat pushes only while its screw is in the water** — the one claim
    /// that makes it a boat and not a car with a rudder (wave VEH2c).
    ///
    /// Three states of one hull: afloat, lifted clear by a swell, and on a level
    /// with no water at all. The third is what a boat trailered up a slipway
    /// meets, and it must be inert rather than driveable.
    #[test]
    fn a_screw_out_of_the_water_pushes_nothing() {
        let mut v = HullVehicle::new(hull_rig());
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();

        // Afloat: the screw at y = -0.5 is half a metre under, its band is
        // 0.4 m, so it is fully wetted and pushing.
        v.solve(afloat(DVec3::ZERO, DVec3::ZERO), DT, &mut out);
        let (afloat_f, _) = resultant(&out, DVec3::ZERO);
        assert!(
            afloat_f.z > 1_000.0,
            "a boat at full ahead made {} N",
            afloat_f.z
        );
        assert_eq!(v.immersion(), 0.5, "the origin is on the waterline");

        // Lifted clear: the same hull a metre up. Nothing wet, nothing pushed.
        out.clear();
        let mut high = afloat(DVec3::ZERO, DVec3::ZERO);
        high.position.y = 1.0;
        v.solve(high, DT, &mut out);
        let (clear_f, clear_tau) = resultant(&out, high.position);
        assert_eq!(
            clear_f,
            DVec3::ZERO,
            "a screw in the air pushed {clear_f:?}"
        );
        assert_eq!(clear_tau, 0.0);
        assert_eq!(v.immersion(), 0.0);

        // No water at all: the same answer by a different road, which is the
        // one a slipway takes.
        out.clear();
        let mut dry = afloat(DVec3::ZERO, DVec3::ZERO);
        dry.water_y = None;
        v.solve(dry, DT, &mut out);
        assert_eq!(resultant(&out, DVec3::ZERO).0, DVec3::ZERO);

        // …and the band is a fade, not a cliff: half out is half the push. The
        // screw sits 0.5 m below the origin and its band is 0.4 m, so lifting
        // the hull 0.3 m puts exactly half of it in the air.
        out.clear();
        let mut awash = afloat(DVec3::ZERO, DVec3::ZERO);
        awash.position.y = 0.3;
        v.solve(awash, DT, &mut out);
        let mid = resultant(&out, awash.position).0.z;
        assert!(
            (mid / afloat_f.z - 0.5).abs() < 0.02,
            "half a screw out of the water gave {mid} N of {}",
            afloat_f.z
        );
    }

    /// **The rudder turns the boat toward the side it is put over**, and it does
    /// it by pushing the STERN the other way — which is what makes the moment.
    ///
    /// Asserted as a torque about the hull's own centre rather than as a sign on
    /// a field: a rudder force applied at the centre of mass would sum to the
    /// same lateral force and turn nothing at all.
    #[test]
    fn the_rudder_swings_the_stern_and_turns_the_bow() {
        let mut v = HullVehicle::new(hull_rig());
        v.control(VehicleControls {
            throttle: 1.0,
            steer: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        // Long enough for the rack to reach its stop.
        for _ in 0..120 {
            out.clear();
            v.solve(afloat(DVec3::new(0.0, 0.0, 4.0), DVec3::ZERO), DT, &mut out);
        }
        assert!(
            v.rudder_deg() > 1.0,
            "the rudder never came over: {}",
            v.rudder_deg()
        );
        let (f, tau) = resultant(&out, DVec3::ZERO);
        // Positive yaw about +Y turns +Z toward +X, which is a turn to
        // starboard — the side the wheel was put over.
        assert!(tau > 0.0, "starboard rudder made a yaw torque of {tau}");
        // …and the stern really is being pushed to port: the net side force is
        // negative even though the boat is turning to starboard.
        assert!(f.x < 0.0, "the rudder pushed the stern to {}", f.x);

        // Port rudder is the mirror image, to the digit.
        let mut v2 = HullVehicle::new(hull_rig());
        v2.control(VehicleControls {
            throttle: 1.0,
            steer: -1.0,
            ..Default::default()
        });
        let mut out2 = Vec::new();
        for _ in 0..120 {
            out2.clear();
            v2.solve(
                afloat(DVec3::new(0.0, 0.0, 4.0), DVec3::ZERO),
                DT,
                &mut out2,
            );
        }
        let (f2, tau2) = resultant(&out2, DVec3::ZERO);
        assert!((tau2 + tau).abs() < 1e-9, "{tau} against {tau2}");
        assert!((f2.x + f.x).abs() < 1e-9);
    }

    /// **A boat that has closed the throttle still steers.**
    ///
    /// The half of `RUDDER_FLOW_GAIN` that is not the propeller race, and the
    /// reason it exists: without it a drifting boat has no control at all, which
    /// is wrong about boats and reads as a broken vehicle.
    #[test]
    fn a_boat_carrying_its_way_still_answers_the_helm() {
        let mut v = HullVehicle::new(hull_rig());
        v.control(VehicleControls {
            throttle: 0.0,
            steer: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        for _ in 0..120 {
            out.clear();
            v.solve(afloat(DVec3::new(0.0, 0.0, 8.0), DVec3::ZERO), DT, &mut out);
        }
        let (_, tau) = resultant(&out, DVec3::ZERO);
        assert!(
            tau > 0.0,
            "a boat with no throttle and 8 m/s of way made no turning moment"
        );
        // …and it is genuinely weaker than the same helm under power, which is
        // the thing that makes a burst of throttle a manoeuvre.
        let mut u = HullVehicle::new(hull_rig());
        u.control(VehicleControls {
            throttle: 1.0,
            steer: 1.0,
            ..Default::default()
        });
        let mut out2 = Vec::new();
        for _ in 0..120 {
            out2.clear();
            u.solve(
                afloat(DVec3::new(0.0, 0.0, 8.0), DVec3::ZERO),
                DT,
                &mut out2,
            );
        }
        let (_, powered) = resultant(&out2, DVec3::ZERO);
        assert!(
            powered > tau * 1.5,
            "the race added nothing: {tau} coasting against {powered} under power"
        );
    }

    /// **The keel damps yaw**, because it is applied at two points and not one.
    ///
    /// Mutation-verified in the ledger: moving both lateral-drag forces to the
    /// centre of mass leaves the same sway resistance and **zero** yaw damping,
    /// which is a boat that spins for ever once it is turning.
    #[test]
    fn the_keel_resists_a_spin_and_not_only_a_slide() {
        let v = |angvel: DVec3, linvel: DVec3| {
            let mut h = HullVehicle::new(hull_rig());
            let mut out = Vec::new();
            h.solve(afloat(linvel, angvel), DT, &mut out);
            resultant(&out, DVec3::ZERO)
        };
        // Spinning in place: no net force, and a torque that opposes the spin.
        let (f, tau) = v(DVec3::new(0.0, 0.5, 0.0), DVec3::ZERO);
        assert!(f.length() < 1e-9, "a spin made a net force of {f:?}");
        assert!(tau < 0.0, "the keel did not resist a spin: {tau}");
        let (_, faster) = v(DVec3::new(0.0, 1.0, 0.0), DVec3::ZERO);
        assert!(faster < tau, "the resistance does not grow with the rate");
        // Sliding sideways: a force that opposes the slide, and no torque.
        let (f, tau) = v(DVec3::ZERO, DVec3::new(2.0, 0.0, 0.0));
        assert!(f.x < 0.0, "the keel did not resist a slide: {f:?}");
        assert!(tau.abs() < 1e-9, "a pure slide made a torque of {tau}");
    }

    /// The hull's tunables are the sixty-two by their own names — and the whole
    /// catalogue row is accepted even though most of it means nothing to a boat.
    #[test]
    fn the_hull_reads_the_names_it_claims_and_accepts_the_rest() {
        let mut v = HullVehicle::new(hull_rig());
        // Every one of the sixty-two is taken, so an authored row is never half
        // read — the reason `install` counts what it took.
        for name in VehicleTuning::names() {
            assert!(v.tune(name, 1.0), "the hull refused `{name}`");
        }
        assert!(!v.tune("hoist_speed", 1.0), "the hull invented a name");
        // …and the eleven it READS reach the model. Thrust is the clearest: with
        // `max_engine_force_n` at zero a boat at full ahead makes no push.
        let mut v = HullVehicle::new(hull_rig());
        v.tune("max_engine_force_n", 0.0);
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        v.solve(afloat(DVec3::ZERO, DVec3::ZERO), DT, &mut out);
        assert_eq!(resultant(&out, DVec3::ZERO).0, DVec3::ZERO);
    }

    /// A boat's telegraph, not a car's gearbox: ahead, astern, stopped — and
    /// astern is weaker than ahead, which is why a boat takes so long to stop.
    #[test]
    fn the_telegraph_says_ahead_astern_or_stopped() {
        let push = |c: VehicleControls| {
            let mut v = HullVehicle::new(hull_rig());
            v.control(c);
            let mut out = Vec::new();
            v.solve(afloat(DVec3::ZERO, DVec3::ZERO), DT, &mut out);
            (v.gear(), resultant(&out, DVec3::ZERO).0.z)
        };
        let (g, ahead) = push(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        assert_eq!(g, 1);
        assert!(ahead > 0.0);
        let (g, astern) = push(VehicleControls {
            throttle: -1.0,
            ..Default::default()
        });
        assert_eq!(g, -1);
        assert!(astern < 0.0);
        assert!(
            (astern.abs() / ahead - HULL_ASTERN_FRACTION).abs() < 1e-9,
            "astern made {astern} N against {ahead} N ahead"
        );
        // A pull-back at speed arrives as a BRAKE, and on a boat that is astern
        // power — the mapping `from_intent` already made, with no special case.
        let (g, stopping) = push(VehicleControls {
            brake: 1.0,
            ..Default::default()
        });
        assert_eq!(g, -1);
        assert!(stopping < 0.0, "the brake did not go astern: {stopping}");
        assert_eq!(push(VehicleControls::default()).0, 0);
    }

    /// **`class_for_parts` picks the boat for a thruster** — the one place a
    /// scene's parts become a model (wave VEH2c).
    #[test]
    fn a_thruster_rig_gets_the_hull_and_a_bare_one_gets_nothing() {
        let v = class_for_parts(hull_rig());
        assert_eq!(v.rig().parts.len(), 1);
        // The identity is asserted through BEHAVIOUR, not a downcast: a hull
        // has no suspension and reports its telegraph, an inert `RaycastVehicle`
        // reports the gear its box is in.
        assert_eq!(v.suspension_rest_m(), 0.0);
        assert_eq!(v.gear(), 0, "a stopped boat is not in first");
        // A rig with parts that are neither answers the inert class, which
        // applies no force at all rather than panicking.
        let mut bare = hull_rig();
        bare.parts.clear();
        let mut v = class_for_parts(bare);
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        v.solve(afloat(DVec3::ZERO, DVec3::ZERO), DT, &mut out);
        assert!(out.is_empty(), "the inert class pushed {out:?}");
    }

    // ── the helicopter (wave VEH2c) ─────────────────────────────────────────

    /// A light helicopter: one 5 m disc on a mast above the cabin.
    fn rotor_rig() -> VehicleRig {
        VehicleRig {
            chassis: Uuid::from_u128(0x4E11),
            seat_local: Vec3d::new(0.0, 0.9, 0.0),
            wheels: Vec::new(),
            parts: vec![PartMount {
                guid: Uuid::from_u128(0x0D15),
                kind: PartKind::Rotor,
                mount_local: Vec3d::new(0.0, 1.3, 0.0),
                size: Vec3d::new(5.0, 0.05, 5.0),
            }],
        }
    }

    /// A tuned rotorcraft — the catalogue row's numbers, through the same
    /// `tune` door an authored `VehicleClass` uses. The default tuning is a
    /// CAR's, whose 8 kN could not lift 1 500 kg, and a fixture that flew on it
    /// would be measuring the default rather than the machine.
    fn rotorcraft(rig: VehicleRig) -> RotorVehicle {
        let mut v = RotorVehicle::new(rig);
        for (name, value) in [
            ("max_engine_force_n", 26_000.0),
            ("max_speed_mps", 70.0),
            ("max_steer_deg", 26.0),
            ("min_steer_deg", 14.0),
            ("steer_rate_deg_per_s", 60.0),
            ("steer_return_deg_per_s", 90.0),
            ("drag_n_per_mps2", 3.0),
            ("drag_lateral_n_per_mps2", 40.0),
        ] {
            assert!(v.tune(name, value), "the rotorcraft refused `{name}`");
        }
        v
    }

    /// A hand on the stick, with everything at neutral (wave VEH2c). An
    /// unmanned rotorcraft has its rotor STOPPED, which is a claim of its own
    /// below, so every arm about how one FLIES has to say a pilot is aboard.
    const PILOT: VehicleControls = VehicleControls {
        throttle: 0.0,
        steer: 0.0,
        brake: 0.0,
        handbrake: false,
        vertical: 0.0,
        occupied: true,
    };

    fn airborne(rotation: DQuat, linvel: DVec3, angvel: DVec3) -> ChassisState {
        ChassisState {
            position: DVec3::new(0.0, 40.0, 0.0),
            rotation,
            linvel,
            angvel,
            mass_kg: 1_500.0,
            water_y: None,
        }
    }

    /// **A couple is two forces**, and the pair really carries the torque asked
    /// for while translating nothing (wave VEH2c).
    ///
    /// The whole reason a helicopter needs no new force channel: `WheelForce` is
    /// a force at a point, and every attitude command in `RotorVehicle` is one
    /// of these.
    #[test]
    fn a_torque_pair_carries_its_torque_and_moves_nothing() {
        let centre = DVec3::new(3.0, -7.0, 11.0);
        for tau in [
            DVec3::new(0.0, 1_000.0, 0.0),
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, -250.0),
            DVec3::new(120.0, -340.0, 55.0),
        ] {
            let pair = torque_pair(centre, tau, 2.5).expect("a real torque");
            let sum: DVec3 = pair.iter().map(|w| w.force).sum();
            assert!(sum.length() < 1e-9, "the pair pushes: {sum:?}");
            let moment: DVec3 = pair.iter().map(|w| (w.point - centre).cross(w.force)).sum();
            assert!(
                (moment - tau).length() < 1e-9,
                "asked for {tau:?}, got {moment:?}"
            );
            // …and the moment is the same about ANY point, which is what makes
            // it a pure couple rather than a force that happens to be balanced.
            let elsewhere: DVec3 = pair
                .iter()
                .map(|w| (w.point - DVec3::new(-40.0, 9.0, 2.0)).cross(w.force))
                .sum();
            assert!((elsewhere - tau).length() < 1e-9);
        }
        // Refusals are values.
        assert!(torque_pair(centre, DVec3::ZERO, 2.5).is_none());
        assert!(torque_pair(centre, DVec3::Y, 0.0).is_none());
        assert!(torque_pair(centre, DVec3::new(f64::NAN, 0.0, 0.0), 2.5).is_none());
    }

    /// **Neutral collective hovers**: the rotor exactly carries the weight, so a
    /// level machine with the stick centred neither climbs nor falls.
    #[test]
    fn neutral_collective_is_a_hover_and_the_stick_moves_it_either_way() {
        let mut v = rotorcraft(rotor_rig());
        let mut out = Vec::new();
        let state = airborne(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO);
        let weight = state.mass_kg * 9.81;

        v.control(PILOT);
        v.solve(state, DT, &mut out);
        assert!(
            (v.thrust_n() - weight).abs() < 1e-6,
            "the hover made {} N against {weight} N of weight",
            v.thrust_n()
        );
        // Summed over every force, including the attitude hold's couple, a level
        // hovering machine is in equilibrium.
        let (f, tau) = resultant(&out, state.position);
        assert!((f.y - weight).abs() < 1e-6, "the hover lifted {} N", f.y);
        assert!(
            f.x.abs() < 1e-9 && f.z.abs() < 1e-9,
            "it is being pushed {f:?}"
        );
        assert!(tau.abs() < 1e-9, "a level hover is being yawed: {tau}");

        for (collective, want) in [
            (1.0, 1.0 + HELI_CLIMB_AUTHORITY),
            (-1.0, 1.0 - HELI_CLIMB_AUTHORITY),
        ] {
            out.clear();
            v.control(VehicleControls {
                vertical: collective,
                ..PILOT
            });
            v.solve(state, DT, &mut out);
            assert!(
                (v.thrust_n() / weight - want).abs() < 1e-9,
                "collective {collective} gave {}x hover, wanted {want}x",
                v.thrust_n() / weight
            );
        }
    }

    /// **The governor holds the VERTICAL, and gives up at its own edge.**
    ///
    /// A coordinated turn tilts the thrust vector, so holding the thrust
    /// constant would make every turn a descent. The governor divides by the
    /// bank's cosine — until `HELI_MIN_LIFT_COS`, past which it stops and the
    /// aircraft falls, which is the one honest cliff in the model.
    #[test]
    fn the_collective_governor_holds_height_in_a_bank_until_it_cannot() {
        let weight = 1_500.0 * 9.81;
        let lift_at = |bank_deg: f64| {
            let mut v = rotorcraft(rotor_rig());
            let mut out = Vec::new();
            let rot = DQuat::from_rotation_z(-bank_deg.to_radians());
            let state = airborne(rot, DVec3::ZERO, DVec3::ZERO);
            v.control(PILOT);
            v.solve(state, DT, &mut out);
            // The vertical component of the rotor force alone — the couple sums
            // to zero and the drag is zero at rest.
            (v.thrust_n(), (state.rotation * DVec3::Y).y * v.thrust_n())
        };
        for bank in [0.0, 15.0, 30.0, 45.0] {
            let (thrust, lift) = lift_at(bank);
            assert!(
                (lift - weight).abs() < 1e-6,
                "at {bank} deg of bank the rotor lifted {lift} N of {weight} N"
            );
            assert!(thrust >= weight - 1e-9, "{bank} deg: {thrust} N");
        }
        // THE SECOND EDGE is the governor's own: past `HELI_MIN_LIFT_COS` it
        // stops dividing at all, so a machine rolled onto its side falls rather
        // than asking for an infinite rotor.
        // THE FIRST EDGE is the rotor's own ceiling, and it arrives before
        // `HELI_MIN_LIFT_COS` does: the fixture's 26 kN holds a 1 500 kg
        // machine up to about 56 degrees of bank and no further, so a
        // 60-degree turn descends because the aircraft is out of thrust,
        // which is the right reason for it to.
        let (thrust, lift) = lift_at(60.0);
        assert!(
            (thrust - 26_000.0).abs() < 1e-6,
            "the rotor ceiling is not binding at 60 degrees: {thrust} N"
        );
        assert!(lift < weight, "a 60-degree bank held its height on 26 kN");
        let (_, lift) = lift_at(80.0);
        assert!(
            lift < weight * 0.6,
            "an 80-degree bank still held {lift} N of {weight} N — the governor \
             has no edge"
        );
    }

    /// **The rotorcraft reads the eleven it claims and ACCEPTS the other
    /// fifty-one** — the hull's arm, for the class that did not have one.
    ///
    /// Written by wave VEH2c's audit. The mini-scout's whole ruling is that a
    /// catalogue row is one table and a class that refused half of it would
    /// leave an author's file half-read; that was armed for `HullVehicle` and
    /// asserted only in prose for this one.
    #[test]
    fn the_rotorcraft_reads_the_names_it_claims_and_accepts_the_rest() {
        let mut v = rotorcraft(rotor_rig());
        for name in VehicleTuning::names() {
            assert!(v.tune(name, 1.0), "the rotorcraft refused `{name}`");
        }
        assert!(
            !v.tune("collective_pitch", 1.0),
            "the rotorcraft invented a name"
        );
        // …and the one that matters most REACHES the model: with the rotor's
        // ceiling at zero a machine at full collective makes no thrust at all,
        // which is a tunable being read rather than merely accepted.
        let mut v = rotorcraft(rotor_rig());
        v.tune("max_engine_force_n", 0.0);
        v.control(VehicleControls {
            vertical: 1.0,
            ..PILOT
        });
        let mut out = Vec::new();
        v.solve(
            airborne(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO),
            DT,
            &mut out,
        );
        assert_eq!(v.thrust_n(), 0.0, "the rotor ceiling is not being read");
        assert_eq!(resultant(&out, DVec3::ZERO).0, DVec3::ZERO);
    }

    /// **The pedals ask for a yaw rate and the bank follows the turn** — the
    /// coordinated-turn relation, and the refusal it encodes.
    #[test]
    fn the_pedals_yaw_and_the_bank_is_derived_from_the_turn() {
        let commanded = |steer: f64, speed: f64| {
            let mut v = rotorcraft(rotor_rig());
            let mut out = Vec::new();
            let state = airborne(DQuat::IDENTITY, DVec3::new(0.0, 0.0, speed), DVec3::ZERO);
            v.control(VehicleControls { steer, ..PILOT });
            v.solve(state, DT, &mut out);
            let (_, tau_y) = resultant(&out, state.position);
            // The roll moment about the machine's own forward axis.
            let tau: DVec3 = out
                .iter()
                .map(|w| (w.point - state.position).cross(w.force))
                .sum();
            (tau_y, tau.z)
        };
        // On the spot: a pedal turn, and NO bank — there is no speed to bank
        // against, which is exactly what the relation says.
        //
        // A RIGHT pedal is a NEGATIVE moment about the machine's own up axis.
        // That is not this class's invention: `VehicleControls::steer`'s doc has
        // said "a right turn is a negative yaw rate" since P29.7, and the world
        // arm `the_pedals_turn_the_nose_and_the_bank_only_arrives_with_speed`
        // measures the same stick as **+107 degrees** of the engine's own euler
        // yaw, which is the sense a car's steering already turns in.
        let (yaw, roll) = commanded(1.0, 0.0);
        assert!(yaw < 0.0, "a right pedal made a yaw moment of {yaw}");
        assert!(roll.abs() < 1e-6, "a hovering pedal turn banked: {roll}");
        // At speed: the same pedal, and now it banks INTO the turn. A torque
        // along +Z raises the right wing here, so dropping it is negative.
        let (yaw_fast, roll_fast) = commanded(1.0, 40.0);
        assert!(
            (yaw_fast - yaw).abs() < 1e-6,
            "the yaw command chased the speed"
        );
        assert!(
            roll_fast < 0.0,
            "a turn at 40 m/s did not bank into it: {roll_fast}"
        );
        // …and the other pedal is the mirror image.
        let (yaw_l, roll_l) = commanded(-1.0, 40.0);
        assert!((yaw_l + yaw_fast).abs() < 1e-6);
        assert!((roll_l + roll_fast).abs() < 1e-6);
    }

    /// **The stick commands an attitude**, rate-limited and speed-tapered by the
    /// same law a steering rack uses — and forward is NOSE DOWN.
    #[test]
    fn the_stick_commands_a_pitch_attitude_that_takes_time_to_arrive() {
        let mut v = rotorcraft(rotor_rig());
        v.tune("max_steer_deg", 30.0);
        v.tune("min_steer_deg", 12.0);
        v.tune("max_speed_mps", 60.0);
        v.control(VehicleControls {
            throttle: 1.0,
            ..PILOT
        });
        let mut out = Vec::new();
        let state = airborne(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO);
        v.solve(state, DT, &mut out);
        // One step of a 60 deg/s rack is one degree, and forward is nose DOWN.
        assert!(
            (v.pitch_cmd_deg() + 1.0).abs() < 1e-9,
            "one step gave {} degrees",
            v.pitch_cmd_deg()
        );
        for _ in 0..300 {
            out.clear();
            v.solve(state, DT, &mut out);
        }
        assert!(
            (v.pitch_cmd_deg() + 30.0).abs() < 1e-9,
            "the stick settled at {} of a 30 degree authority",
            v.pitch_cmd_deg()
        );
        // …and at speed the authority tapers, and binds the RESULT: a command
        // already at the hover stop is pulled back as the machine accelerates.
        let fast = airborne(DQuat::IDENTITY, DVec3::new(0.0, 0.0, 60.0), DVec3::ZERO);
        out.clear();
        v.solve(fast, DT, &mut out);
        assert!(
            (v.pitch_cmd_deg() + 12.0).abs() < 1e-9,
            "at 60 m/s the stick still reached {}",
            v.pitch_cmd_deg()
        );
    }

    /// The blade turns while the rotor is turning and the angle stays wrapped —
    /// a visual write, and the only thing in this class that is one.
    #[test]
    fn the_drawn_blade_turns_and_its_angle_stays_bounded() {
        let mut v = rotorcraft(rotor_rig());
        v.control(PILOT);
        let mut out = Vec::new();
        let state = airborne(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..600 {
            out.clear();
            v.solve(state, DT, &mut out);
            let pose = v.part_pose(0).expect("a rotor has a pose");
            assert_eq!(pose.x, 0.0);
            assert_eq!(pose.z, 0.0);
            assert!((0.0..360.0).contains(&pose.y), "azimuth {}", pose.y);
            seen.insert(pose.y.to_bits());
        }
        assert!(
            seen.len() > 100,
            "the blade barely moved: {} angles",
            seen.len()
        );
    }

    /// **`class_for_parts` picks the rotorcraft for a rotor, and rotors outrank
    /// thrusters** — the precedence rule, asserted rather than described.
    #[test]
    fn a_rotor_rig_gets_the_rotorcraft_and_outranks_a_thruster() {
        // A rotor alone.
        let v = class_for_parts(rotor_rig());
        assert_eq!(v.gear(), 0);
        assert_eq!(v.suspension_rest_m(), 0.0);
        // Behaviour, not a downcast: only the rotorcraft lifts against gravity.
        let lifts = |mut v: Box<dyn Vehicle>| {
            v.tune("max_engine_force_n", 26_000.0);
            v.control(PILOT);
            let mut out = Vec::new();
            v.solve(
                airborne(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO),
                DT,
                &mut out,
            );
            out.iter().map(|w| w.force.y).sum::<f64>()
        };
        assert!(lifts(class_for_parts(rotor_rig())) > 14_000.0);
        // A hull does not lift; buoyancy is not its business.
        assert_eq!(lifts(class_for_parts(hull_rig())), 0.0);
        // Both parts: the rotor wins, because what holds an amphibian up is what
        // decides how it behaves.
        let mut both = rotor_rig();
        both.parts.extend(hull_rig().parts);
        both.parts.sort_unstable_by_key(|p| p.guid);
        assert!(lifts(class_for_parts(both)) > 14_000.0);
    }

    /// **Each craft reads its own instruments** (wave VEH2c) — and the
    /// dispatch is the rig, not a kind field somebody has to keep in step.
    #[test]
    fn a_boat_reads_knots_and_an_aircraft_reads_a_height() {
        let car = VehicleRig {
            chassis: Uuid::from_u128(1),
            seat_local: Vec3d::new(0.0, 0.5, 0.0),
            wheels: vec![WheelMount {
                guid: Uuid::from_u128(2),
                mount_local: Vec3d::new(0.9, -0.5, 1.4),
                radius_m: 0.35,
            }],
            parts: Vec::new(),
        };
        // A car is unchanged, to the character: `craft_readout` falls through to
        // `drive_readout` and every VEH2b claim about it still holds.
        assert_eq!(
            craft_readout(&car, 27.777_777_8, 3, None),
            drive_readout(27.777_777_8, 3)
        );
        assert_eq!(
            craft_readout(&car, 27.777_777_8, 3, Some(400.0)),
            "100 km/h    3"
        );

        // A boat reads KNOTS and a telegraph. 8.93 m/s is the launch's own
        // measured top speed, and 17.4 knots is what a helmsman would say.
        let boat = hull_rig();
        assert_eq!(craft_readout(&boat, 8.93, 1, None), "17.4 kn    AHEAD");
        assert_eq!(craft_readout(&boat, 2.0, -1, None), "3.9 kn    ASTERN");
        assert_eq!(craft_readout(&boat, 0.0, 0, None), "0.0 kn    STOP");
        // Backwards is a POSITIVE number beside ASTERN, exactly as a car's
        // speedometer does not go below zero.
        assert_eq!(craft_readout(&boat, -2.0, -1, None), "3.9 kn    ASTERN");

        // An aircraft reads a speed and a HEIGHT, and a gear it does not have
        // never appears.
        let heli = rotor_rig();
        assert_eq!(
            craft_readout(&heli, 38.7, 0, Some(42.4)),
            "139 km/h    ALT 42 m"
        );
        // Below the ground is zero rather than a negative altitude: a machine on
        // a slope reads the honest thing.
        assert_eq!(
            craft_readout(&heli, 0.0, 0, Some(-3.0)),
            "0 km/h    ALT 0 m"
        );
        // …and a host that cannot measure a height prints the speed alone
        // rather than a lie.
        assert_eq!(craft_readout(&heli, 38.7, 0, None), "139 km/h");
        assert_eq!(craft_readout(&heli, 38.7, 0, Some(f64::NAN)), "139 km/h");
        // Nonsense in, a readable line out — `drive_readout`'s own rule.
        assert_eq!(
            craft_readout(&heli, f64::NAN, 0, Some(10.0)),
            "0 km/h    ALT 10 m"
        );
        assert_eq!(
            craft_readout(&boat, f64::INFINITY, 1, None),
            "0.0 kn    AHEAD"
        );
    }

    /// The readout a driver reads, and the two edges of it.
    #[test]
    fn the_readout_says_a_speed_and_a_gear() {
        assert_eq!(drive_readout(0.0, 0), "0 km/h    N");
        assert_eq!(drive_readout(27.7777778, 3), "100 km/h    3");
        // Backwards is a POSITIVE number with an R beside it -- a speedometer
        // does not go below zero.
        assert_eq!(drive_readout(-2.0, -1), "7 km/h    R");
        // Nonsense in, a readable line out.
        assert_eq!(drive_readout(f64::NAN, 0), "0 km/h    N");
        assert_eq!(drive_readout(f64::INFINITY, 1), "0 km/h    1");
        // Every gear the box can be in has a symbol, and one it cannot does not
        // invent one.
        assert_eq!(gear_label(-9), "R");
        assert_eq!(gear_label(MAX_GEARS as i32), "8");
        assert_eq!(gear_label(99), "8");
    }

    // ── the recipe (wave VEH2b) ─────────────────────────────────────────────

    fn spawn_of(name: &str) -> RigSpawn {
        RigSpawn {
            name: name.to_string(),
            at: DVec3::new(10.0, 2.0, -30.0),
            yaw_deg: 45.0,
            paint: crate::math::Color::new(0.2, 0.4, 0.9, 1.0),
            clip: None,
            engine_voice: true,
            livery: None,
        }
    }

    /// The recipe is one list and the RUNTIME door builds every entity in it —
    /// which is what makes "two callers, one car" a structural claim rather
    /// than a comment.
    #[test]
    fn the_runtime_door_builds_every_node_of_the_recipe() {
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0x1234);
        let nodes = rig_nodes(chassis, &def, &spawn_of("Test Car"));
        // A chassis, its body panels, and four wheels each wearing a tyre.
        assert_eq!(nodes.len(), 1 + def.body.parts().len() + 8);
        assert_eq!(nodes[0].guid, chassis);
        assert!(nodes[0].parent.is_none());
        assert!(nodes[0].class.is_some() && nodes[0].audio.is_some());
        for n in &nodes[1..] {
            assert!(n.parent.is_some(), "{} has no parent", n.name);
            assert!(n.class.is_none() && n.audio.is_none(), "{}", n.name);
        }
        // Every parent is introduced before it is named — the property a walk
        // in creation order rests on.
        let mut seen = std::collections::BTreeSet::new();
        for n in &nodes {
            if let Some(p) = n.parent {
                assert!(seen.contains(&p), "{} names an unseen parent", n.name);
            }
            seen.insert(n.guid);
        }

        let mut w = EcsWorld::new();
        assert_eq!(
            spawn_rig(&mut w, chassis, &def, &spawn_of("Test Car")),
            chassis
        );
        for n in &nodes {
            let e = w.entity_of(n.guid).unwrap_or_else(|| panic!("{}", n.name));
            assert_eq!(w.world().get::<Transform>(e).copied(), Some(n.transform));
        }
        // …and the rig the vehicle door reads is derivable from what was built.
        let r = rig_of(&w, chassis).expect("a rig");
        assert_eq!(r.wheels.len(), 4);
    }

    /// **A livery repaints the parts it names, adds the ones it declares, and
    /// leaves an unliveried car BYTE-IDENTICAL** (wave EMS1).
    ///
    /// Three claims, and the third is the one that costs something. The
    /// override is a mutation of the material this loop already built, so "a
    /// part with no override is what it was" is a statement about bytes rather
    /// than a comparison of two literals somebody kept in step — and it is
    /// asserted against a recipe built with `livery: None`, which is every
    /// civilian vehicle and all of traffic.
    #[test]
    fn a_livery_repaints_named_parts_and_leaves_the_rest_exactly_as_they_were() {
        const RED: crate::math::Color = crate::math::Color::new(0.6, 0.05, 0.05, 1.0);
        const BAR: BodyPart = BodyPart {
            name: "light_bar",
            centre: Vec3d::new(0.0, 1.06, 0.0),
            half: Vec3d::new(0.5, 0.06, 0.2),
            primitive: crate::components::Primitive::Cube,
        };
        static LIVERY: Livery = Livery {
            name: "test",
            parts: &[("cabin", PartPaint::flat(RED))],
            extra: &[(
                BAR,
                PartPaint {
                    base_color: crate::math::Color::new(0.1, 0.1, 0.1, 1.0),
                    emissive: crate::math::Color::new(0.2, 0.4, 1.0, 1.0),
                    emissive_intensity: 3.0,
                },
            )],
            service: None,
        };
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0xE_5A1);
        let plain = rig_nodes(chassis, &def, &spawn_of("Plain"));
        let mut liveried_spawn = spawn_of("Plain");
        liveried_spawn.livery = Some(&LIVERY);
        let worn = rig_nodes(chassis, &def, &liveried_spawn);

        // The livery adds exactly its own parts, and they sit after the body's.
        assert_eq!(worn.len(), plain.len() + 1);
        let bar = worn
            .iter()
            .find(|n| n.name == "light_bar")
            .expect("the livery's own part is built");
        assert_eq!(bar.parent, Some(chassis), "a light bar is on the car");
        let m = bar.material.expect("a light bar is drawn");
        assert!(
            m.emissive_linear()[2] > 1.0,
            "a light bar at {:?} x {} does not bloom — the HDR path thresholds \
             at a linear luminance of 1.0",
            m.emissive,
            m.emissive_intensity
        );

        // The named part is repainted…
        let cabin = worn.iter().find(|n| n.name == "cabin").expect("a cabin");
        assert_eq!(cabin.material.expect("drawn").base_color, RED);
        // …and every OTHER node is byte-identical to the unliveried recipe,
        // which is the arm that says a civilian car did not move.
        for p in &plain {
            if p.name == "cabin" {
                continue;
            }
            let w = worn
                .iter()
                .find(|n| n.guid == p.guid)
                .unwrap_or_else(|| panic!("{} vanished under a livery", p.name));
            assert_eq!(w, p, "{} moved under a livery", p.name);
        }
        // …and the plain recipe really did have something to repaint, or the
        // sweep above is a statement about a car with no cabin.
        assert!(plain.iter().any(|n| n.name == "cabin"));
        assert!(!plain.iter().any(|n| n.name == "light_bar"));
    }

    /// Spawning twice is idempotent — a traffic car that materializes on a
    /// guid the level already authored leaves the author's entity alone.
    #[test]
    fn the_runtime_door_never_overwrites_an_entity_the_world_holds() {
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0x99);
        let mut w = EcsWorld::new();
        spawn_rig(&mut w, chassis, &def, &spawn_of("First"));
        let first = w.entity_of(chassis).expect("the chassis");
        spawn_rig(&mut w, chassis, &def, &spawn_of("Second"));
        assert_eq!(w.entity_of(chassis), Some(first));
        assert_eq!(
            w.world().get::<Transform>(first).map(|t| t.translation.x),
            Some(10.0)
        );
    }

    #[test]
    fn a_rig_goes_back_out_of_the_world_it_came_into() {
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0x55);
        let mut w = EcsWorld::new();
        spawn_rig(&mut w, chassis, &def, &spawn_of("Gone"));
        let n = 1 + def.body.parts().len() + 8;
        assert_eq!(despawn_rig(&mut w, chassis, &def), n);
        for node in rig_nodes(chassis, &def, &spawn_of("Gone")) {
            assert!(w.entity_of(node.guid).is_none(), "{}", node.name);
        }
        // …and again is a no-op rather than a panic.
        assert_eq!(despawn_rig(&mut w, chassis, &def), 0);
    }

    /// The placement rule, once: wheel drop plus wheel radius, with the sign
    /// the right way round.
    #[test]
    fn a_rig_rests_with_its_wheels_on_the_ground() {
        let def = VehicleDef::default();
        let y = resting_origin_y(&def, 12.0);
        assert_eq!(y, 12.0 - def.wheel_drop_m + def.wheel_radius_m);
        // The lowest point of a wheel at full extension is the ground.
        assert!((y + def.wheel_drop_m - def.wheel_radius_m - 12.0).abs() < 1e-12);
    }

    fn rig(n: usize) -> VehicleRig {
        let wheels = (0..n)
            .map(|i| WheelMount {
                guid: Uuid::from_u128(0x1000 + i as u128),
                mount_local: Vec3d::new(
                    if i % 2 == 0 { -0.9 } else { 0.9 },
                    -0.5,
                    if i < 2 { 1.4 } else { -1.4 },
                ),
                radius_m: 0.35,
            })
            .collect();
        VehicleRig {
            chassis: Uuid::from_u128(1),
            seat_local: Vec3d::new(0.0, 0.5, 0.0),
            wheels,
            parts: Vec::new(),
        }
    }

    fn resting(mass: f64) -> ChassisState {
        ChassisState {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linvel: DVec3::ZERO,
            angvel: DVec3::ZERO,
            mass_kg: mass,
            water_y: None,
        }
    }

    /// A wheel is a sphere sensor with no body — and each half of that matters.
    #[test]
    fn the_wheel_recogniser_needs_every_clause() {
        let wheel = Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.35,
            sensor: true,
            ..Default::default()
        };
        assert_eq!(wheel_of(Some(&wheel), None), Some(0.35));
        // A body of its own makes it an ordinary child, not a wheel.
        assert_eq!(
            wheel_of(Some(&wheel), Some(&RigidBody3D::default())),
            None,
            "a wheel has no body: the chassis is the body"
        );
        let solid = Collider3D {
            sensor: false,
            ..wheel
        };
        assert_eq!(
            wheel_of(Some(&solid), None),
            None,
            "a solid sphere collides"
        );
        let boxy = Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            ..wheel
        };
        assert_eq!(wheel_of(Some(&boxy), None), None);
        assert_eq!(wheel_of(None, None), None);
    }

    /// **The two recognisers PARTITION the collider vocabulary** (wave VEH2c).
    ///
    /// Not "each answers for its own shape" — that is what a pair of functions
    /// written independently would look like while both claiming a sphere. The
    /// claim asserted here is exclusivity in both directions over the whole
    /// three-shape vocabulary, which is what makes [`PartKind`]'s table a
    /// partition rather than two overlapping opinions.
    #[test]
    fn the_part_recogniser_takes_the_two_shapes_the_wheel_does_not() {
        let sensor = Collider3D {
            sensor: true,
            radius: 0.35,
            half_extents: Vec3d::new(0.4, 0.1, 0.6),
            ..Default::default()
        };
        let of = |k| {
            let c = Collider3D {
                shape_kind: k,
                ..sensor
            };
            (wheel_of(Some(&c), None).is_some(), part_of(Some(&c), None))
        };
        // Sphere: a wheel, and NOT a part.
        assert_eq!(of(ColliderShape3DKind::Sphere), (true, None));
        // Box: a thruster, and NOT a wheel. Its half-extents come through.
        assert_eq!(
            of(ColliderShape3DKind::Box),
            (false, Some((PartKind::Thruster, Vec3d::new(0.4, 0.1, 0.6))))
        );
        // Capsule: a rotor, whose size is its own radius and half-height.
        assert_eq!(
            of(ColliderShape3DKind::Capsule),
            (false, Some((PartKind::Rotor, Vec3d::new(0.35, 0.1, 0.35))))
        );
        // …and every clause of `wheel_of`'s own rule binds here too.
        let boxy = Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            ..sensor
        };
        assert_eq!(
            part_of(Some(&boxy), Some(&RigidBody3D::default())),
            None,
            "a part has no body: the chassis is the body"
        );
        assert_eq!(
            part_of(
                Some(&Collider3D {
                    sensor: false,
                    ..boxy
                }),
                None
            ),
            None,
            "a solid box collides"
        );
        assert_eq!(
            part_of(
                Some(&Collider3D {
                    half_extents: Vec3d::new(0.0, 0.1, 0.6),
                    ..boxy
                }),
                None
            ),
            None,
            "a part with no width can push nothing"
        );
        assert_eq!(part_of(None, None), None);
    }

    /// **WHEELS WIN** — the whole of what this wave's seam costs every level
    /// that already exists (wave VEH2c).
    ///
    /// A car with a trigger volume bolted to it keeps four wheels and NO parts,
    /// so the box sensor is mirrored into rapier exactly as it was before this
    /// wave; the same box on a wheel-less hull is a thruster. One scene shape,
    /// two answers, and the discriminator is the wheels.
    #[test]
    fn wheels_win_so_an_existing_cars_trigger_child_stays_a_trigger() {
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0xB0A7);
        let trigger = Uuid::from_u128(0x7616);
        let boxy = Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.3, 0.2, 0.4),
            sensor: true,
            ..Default::default()
        };

        // (a) With wheels: four wheels, no parts.
        let mut w = EcsWorld::new();
        spawn_rig(&mut w, chassis, &def, &spawn_of("Car"));
        let parent = w.entity_of(chassis).expect("a chassis");
        let e = w.spawn_with_guid(trigger, "Trigger", Some(parent));
        w.world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(0.0, 0.0, -2.0)))
            .insert(boxy);
        let rig = rig_of(&w, chassis).expect("a car is a rig");
        assert_eq!((rig.wheels.len(), rig.parts.len()), (4, 0));

        // (b) The same trigger on a hull with no wheels IS a thruster — so the
        //     discrimination above is the wheels and not the box.
        let mut w = EcsWorld::new();
        spawn_rig_at(&mut w, chassis, &def, &spawn_of("Hull"), false);
        // …and a wheel-less hull was invisible to the recogniser until now.
        assert!(
            rig_of(&w, chassis).is_none(),
            "a body with no wheels and no parts is not a vehicle"
        );
        let parent = w.entity_of(chassis).expect("a chassis");
        let e = w.spawn_with_guid(trigger, "Screw", Some(parent));
        w.world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(0.0, -0.2, -2.0)))
            .insert(boxy);
        // A hull is spawned Kinematic; the chassis rule wants a dynamic body,
        // which is the one thing a wheel-less craft has to say for itself.
        w.world_mut().entity_mut(parent).insert(RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        });
        let rig = rig_of(&w, chassis).expect("a hull with a thruster is a rig");
        assert_eq!((rig.wheels.len(), rig.parts.len()), (0, 1));
        assert_eq!(rig.parts[0].kind, PartKind::Thruster);
        assert_eq!(rig.parts[0].guid, trigger);
        assert_eq!(rig.parts[0].mount_local, Vec3d::new(0.0, -0.2, -2.0));
        assert_eq!(rig.parts_of(PartKind::Thruster).count(), 1);
        assert_eq!(rig.parts_of(PartKind::Rotor).count(), 0);
        // The seat is still derived from the chassis collider, unchanged.
        assert_eq!(rig.seat_local, Vec3d::new(0.0, def.half_extents.y, 0.0));
    }

    /// The parts come back in `Guid` order, like the wheels — so a rig's part
    /// indices are a function of the level's contents and not of an archetype
    /// walk (wave VEH2c).
    #[test]
    fn the_parts_are_sorted_by_guid_whatever_order_they_were_spawned_in() {
        let def = VehicleDef::default();
        let chassis = Uuid::from_u128(0xCAFE);
        let ids = [
            Uuid::from_u128(0x30),
            Uuid::from_u128(0x10),
            Uuid::from_u128(0x20),
        ];
        let mut w = EcsWorld::new();
        spawn_rig_at(&mut w, chassis, &def, &spawn_of("Heli"), false);
        let parent = w.entity_of(chassis).expect("a chassis");
        w.world_mut().entity_mut(parent).insert(RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        });
        for (i, g) in ids.into_iter().enumerate() {
            let e = w.spawn_with_guid(g, "Rotor", Some(parent));
            w.world_mut()
                .entity_mut(e)
                .insert(Transform::from_translation(DVec3::new(0.0, i as f64, 0.0)))
                .insert(Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    radius: 4.0,
                    half_extents: Vec3d::new(4.0, 0.05, 4.0),
                    sensor: true,
                    ..Default::default()
                });
        }
        let rig = rig_of(&w, chassis).expect("a rotorcraft is a rig");
        let got: Vec<Uuid> = rig.parts.iter().map(|p| p.guid).collect();
        let mut want = ids.to_vec();
        want.sort();
        assert_eq!(got, want, "the parts are not in Guid order");
        assert!(rig.parts.iter().all(|p| p.kind == PartKind::Rotor));
        assert_eq!(
            rig.parts[0].size.x, 4.0,
            "a rotor's size is its disc radius"
        );
    }

    /// The vertical axis reaches the controls, is clamped there, and every other
    /// control is untouched by it (wave VEH2c).
    #[test]
    fn the_vertical_axis_reaches_the_controls_and_is_clamped() {
        let none = VehicleControls::from_intent(crate::math::Vec2d::ZERO, 0.0, false, 0.0);
        assert_eq!(none.vertical, 0.0);
        // …and an intent's controls are COMMANDED, which the default is not: a
        // stick at rest is not the same thing as no hand on it
        // (`VehicleControls::occupied`).
        assert!(none.occupied);
        assert!(!VehicleControls::default().occupied);
        assert_eq!(
            none,
            VehicleControls {
                occupied: true,
                ..Default::default()
            }
        );
        let up = VehicleControls::from_intent(crate::math::Vec2d::ZERO, 0.0, false, 4.0);
        assert_eq!(up.vertical, 1.0, "a wild axis is clamped, not refused");
        let down = VehicleControls::from_intent(crate::math::Vec2d::ZERO, 0.0, false, -4.0);
        assert_eq!(down.vertical, -1.0);
        // …and it changes nothing else: the same stick with and without it.
        let stick = crate::math::Vec2d::new(0.5, 1.0);
        let a = VehicleControls::from_intent(stick, 3.0, true, 0.0);
        let b = VehicleControls::from_intent(stick, 3.0, true, 1.0);
        assert_eq!(
            (a.throttle, a.steer, a.brake, a.handbrake),
            (b.throttle, b.steer, b.brake, b.handbrake)
        );
    }

    /// The seat is the chassis collider's top face, so a character's feet land
    /// on it — over the whole authored shape vocabulary.
    #[test]
    fn the_seat_is_the_top_of_whatever_the_chassis_is() {
        let dynamic = RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        };
        let boxy = Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, 0.5, 1.0),
            ..Default::default()
        };
        assert_eq!(
            chassis_of(Some(&boxy), Some(&dynamic)),
            Some(Vec3d::new(0.0, 0.5, 0.0))
        );
        // …and a static body is scenery, not a vehicle.
        assert_eq!(chassis_of(Some(&boxy), Some(&RigidBody3D::default())), None);
        let capsule = Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.8, 0.3),
            radius: 0.3,
            ..Default::default()
        };
        assert_eq!(
            chassis_of(Some(&capsule), Some(&dynamic)).map(|s| s.y),
            Some(1.1)
        );
    }

    /// A rig at rest settles into its travel rather than bottoming out or
    /// floating — the one number every other number here was sized against.
    #[test]
    fn the_default_spring_holds_the_default_rig_inside_its_travel() {
        let t = VehicleTuning::default();
        let load = 1_200.0 * 9.81 / 4.0;
        let x = load / t.stiffness_n_per_m;
        assert!(
            x > 0.05 && x < t.travel_m * 0.8,
            "a 1200 kg rig settles at {x} m of a {} m travel",
            t.travel_m
        );
    }

    /// **The torque curve passes through its three knots**, peaks where it says
    /// it peaks, and the one shape knob moves both halves the way its doc claims.
    #[test]
    fn the_torque_curve_hits_its_three_knots_and_the_bias_moves_both_halves() {
        let t = VehicleTuning::default();
        let at = |rpm| engine_torque_nm(&t, rpm);
        assert!((at(t.idle_rpm) - t.peak_torque_nm * t.idle_torque_frac).abs() < 1e-9);
        assert!((at(t.peak_torque_rpm) - t.peak_torque_nm).abs() < 1e-9);
        assert!((at(t.redline_rpm) - t.peak_torque_nm * t.redline_torque_frac).abs() < 1e-9);
        // The peak really is the peak, swept.
        let mut best = (0.0f64, 0.0f64);
        for i in 0..=200 {
            let rpm = t.idle_rpm + (t.redline_rpm - t.idle_rpm) * i as f64 / 200.0;
            if at(rpm) > best.1 {
                best = (rpm, at(rpm));
            }
        }
        assert!(
            (best.0 - t.peak_torque_rpm).abs() < 60.0,
            "the curve peaks at {} rpm, not at the authored {}",
            best.0,
            t.peak_torque_rpm
        );
        // Below idle and above the redline the argument is clamped — that IS the
        // limiter, and a curve that kept rising past it would make the redline a
        // suggestion.
        assert_eq!(at(0.0), at(t.idle_rpm));
        assert_eq!(at(t.redline_rpm * 3.0), at(t.redline_rpm));
        assert!(at(-1.0).is_finite() && at(f64::NAN).is_finite());

        // A TRUCK engine (bias high) makes more torque low down and less at the
        // top; a SPORTS engine (bias low) does the reverse. Both against the same
        // three knots, so the claim is about the shape and not about the numbers.
        let low = t.idle_rpm + (t.peak_torque_rpm - t.idle_rpm) * 0.35;
        let high = t.peak_torque_rpm + (t.redline_rpm - t.peak_torque_rpm) * 0.6;
        let truck = VehicleTuning {
            torque_curve_bias: 0.8,
            ..t
        };
        let sports = VehicleTuning {
            torque_curve_bias: 0.25,
            ..t
        };
        assert!(engine_torque_nm(&truck, low) > at(low));
        assert!(engine_torque_nm(&truck, high) < at(high));
        assert!(engine_torque_nm(&sports, low) < at(low));
        assert!(engine_torque_nm(&sports, high) > at(high));
    }

    /// **Revs come from the wheels through the gear**, floored at idle and capped
    /// at the redline — and a taller gear means fewer revs for the same speed,
    /// which is the whole reason a gearbox exists.
    #[test]
    fn the_revs_are_the_wheels_through_the_gear() {
        let t = VehicleTuning::default();
        assert_eq!(engine_rpm(&t, 0.0, 1), t.idle_rpm, "a stopped car idles");
        assert_eq!(
            engine_rpm(&t, 40.0, 0),
            t.idle_rpm,
            "neutral turns the engine at idle whatever the wheels do"
        );
        // 30 rad/s on a 0.35 m wheel is 10.5 m/s.
        let first = engine_rpm(&t, 30.0, 1);
        let top = engine_rpm(&t, 30.0, 6);
        assert!(
            top < first,
            "sixth ({top} rpm) is not taller than first ({first} rpm)"
        );
        assert!(first > t.idle_rpm && first <= t.redline_rpm);
        assert_eq!(
            engine_rpm(&t, 500.0, 1),
            t.redline_rpm,
            "the limiter is a clamp, not a suggestion"
        );
        // Reverse turns the engine forwards: a car reversing at 3 m/s is not at
        // minus two thousand rpm.
        assert!(engine_rpm(&t, -9.0, -1) > t.idle_rpm);
    }

    /// **The box shifts up at the top, down at the bottom, and does not hunt.**
    #[test]
    fn the_gearbox_shifts_up_and_down_and_holds_between() {
        let t = VehicleTuning::default();
        let at = |gear, rpm| shift_target(&t, gear, rpm, 1.0, 10.0);
        assert_eq!(at(1, t.shift_up_rpm + 1.0), 2);
        assert_eq!(at(2, t.shift_down_rpm - 1.0), 1);
        assert_eq!(at(3, (t.shift_up_rpm + t.shift_down_rpm) / 2.0), 3);
        assert_eq!(
            at(6, t.shift_up_rpm + 1.0),
            6,
            "a six-speed box does not shift into a seventh"
        );
        assert_eq!(at(1, t.shift_down_rpm - 1.0), 1, "…nor below first");
        // Reverse: asked for backwards while not going forwards.
        assert_eq!(shift_target(&t, 1, 2_000.0, -1.0, 0.0), -1);
        assert_eq!(
            shift_target(&t, 1, 2_000.0, -1.0, 20.0),
            1,
            "asking for reverse at 20 m/s is a BRAKE, and the box stays in gear"
        );
        assert_eq!(shift_target(&t, -1, 2_000.0, 1.0, 0.0), 1);
        // …and the shift band is wide enough that one gear's up point is not the
        // next gear's down point, or the box hunts for ever.
        let step = t.gear_1_ratio / t.gear_2_ratio;
        assert!(
            t.shift_up_rpm / step > t.shift_down_rpm,
            "shifting up at {} lands at {} rpm, at or below the {} downshift point",
            t.shift_up_rpm,
            t.shift_up_rpm / step,
            t.shift_down_rpm
        );
    }

    /// **THE TYRE CURVE**: it rises to exactly 1 at its own peak slip, falls
    /// past it, and settles on the sliding plateau — and the rise stiffness is a
    /// knob that does not move the peak.
    #[test]
    fn the_tyre_peaks_at_one_and_falls_to_its_sliding_plateau() {
        let slide = 0.72;
        for bias in [0.3, 0.5, 0.74, 0.9] {
            assert_eq!(tyre_curve(0.0, bias, slide), 0.0);
            assert!((tyre_curve(1.0, bias, slide) - 1.0).abs() < 1e-12, "{bias}");
            // The peak is at 1 for EVERY stiffness — that is what makes the two
            // knobs independent.
            let mut best = (0.0f64, 0.0f64);
            for i in 0..=400 {
                let s = i as f64 * 0.02;
                let v = tyre_curve(s, bias, slide);
                if v > best.1 {
                    best = (s, v);
                }
            }
            assert!(
                (best.0 - 1.0).abs() < 1e-9,
                "bias {bias} peaks at {} rather than at its own peak slip",
                best.0
            );
            // Past the peak it FALLS, and reaches the plateau exactly at the
            // engine-wide multiple.
            assert!(tyre_curve(2.0, bias, slide) < 1.0);
            assert!(
                (tyre_curve(TYRE_SLIDE_SLIP_MULT, bias, slide) - slide).abs() < 1e-12,
                "{bias}"
            );
            assert!((tyre_curve(50.0, bias, slide) - slide).abs() < 1e-12);
            // Symmetric in the sign of the slip: a tyre does not grip better
            // going backwards.
            assert_eq!(tyre_curve(-0.4, bias, slide), tyre_curve(0.4, bias, slide));
        }
        // A stiffer tyre is above a softer one everywhere below the peak, and
        // they meet AT it — which is the claim `tyre_long_rise_bias`'s doc makes.
        for i in 1..20 {
            let s = i as f64 / 20.0;
            assert!(tyre_curve(s, 0.85, slide) > tyre_curve(s, 0.4, slide));
        }
    }

    /// **µ falls as load rises**, bracketed at both ends — the function that
    /// makes weight transfer cost something.
    #[test]
    fn grip_falls_under_load_and_is_bracketed_at_both_ends() {
        let (mu, stat) = (1.2, 3_000.0);
        assert_eq!(load_sensitive_mu(mu, stat, stat, 0.22), mu);
        assert!(load_sensitive_mu(mu, stat * 2.0, stat, 0.22) < mu);
        assert!(load_sensitive_mu(mu, stat * 0.5, stat, 0.22) > mu);
        assert_eq!(
            load_sensitive_mu(mu, stat * 4.0, stat, 0.0),
            mu,
            "sensitivity 0 is the schoolbook tyre, exactly"
        );
        // The bracket, both ways.
        assert!((load_sensitive_mu(mu, stat * 40.0, stat, 0.5) - mu * MU_MIN_FRAC).abs() < 1e-12);
        assert!((load_sensitive_mu(mu, 0.0, stat, 4.0) - mu * MU_MAX_FRAC).abs() < 1e-12);
        // A refusal is a value: no reference load means no correction.
        assert_eq!(load_sensitive_mu(mu, stat, 0.0, 0.22), mu);
        assert_eq!(load_sensitive_mu(mu, f64::NAN, stat, 0.22), mu);

        // **The consequence, stated in newtons**: transferring load across an
        // axle LOSES that axle grip, which is what an anti-roll bar trades and
        // what a low centre of gravity buys. Zero sensitivity keeps it exactly.
        let total = |sens: f64, transfer: f64| {
            load_sensitive_mu(mu, stat + transfer, stat, sens) * (stat + transfer)
                + load_sensitive_mu(mu, stat - transfer, stat, sens) * (stat - transfer)
        };
        assert!((total(0.0, 1_500.0) - total(0.0, 0.0)).abs() < 1e-9);
        assert!(
            total(0.22, 1_500.0) < total(0.22, 0.0) * 0.95,
            "a 1 500 N transfer cost the axle only {:.1} N of {:.1}",
            total(0.22, 0.0) - total(0.22, 1_500.0),
            total(0.22, 0.0)
        );
    }

    /// **THE FRICTION CIRCLE, and the two boxes that died for it.**
    ///
    /// The headline arm of the wave. P29.7 clamped each axis independently, so a
    /// tyre could hold `µ·N` sideways *and* `µ·N` forwards at once — 1.41 × its
    /// own grip. Here the combined magnitude never exceeds the peak, pure slip on
    /// either axis reproduces that axis's own curve exactly, and asking for more
    /// of one costs the other.
    #[test]
    fn the_two_axis_clamps_are_dead_and_this_is_a_circle() {
        let t = VehicleTuning {
            // One µ and one shape on both axes, so the ellipse is a circle and
            // the claim is about the COUPLING rather than about two numbers.
            lateral_grip: 1.2,
            longitudinal_grip: 1.2,
            tyre_lat_peak_slip: 0.12,
            tyre_long_peak_slip: 0.12,
            tyre_lat_rise_bias: 0.74,
            tyre_long_rise_bias: 0.74,
            tyre_load_sensitivity: 0.0,
            ..VehicleTuning::default()
        };
        let (load, stat) = (3_000.0, 3_000.0);
        let peak = 1.2 * load;
        let mag = |sr, sl| {
            let (x, y) = tyre_force_n(&t, load, stat, sr, sl);
            (x * x + y * y).sqrt()
        };

        // Pure longitudinal at the peak slip is the peak, and pure lateral is the
        // same peak — the two axes agree because they were given one µ.
        assert!((mag(0.12, 0.0) - peak).abs() < 1e-9);
        assert!((mag(0.0, 0.12) - peak).abs() < 1e-9);
        // The two boxes would have delivered √2 × peak here. The circle does not.
        let both = mag(0.12, 0.12);
        assert!(
            both <= peak + 1e-9,
            "combined slip produced {both} N against a peak of {peak} N — that is \
             {:.3} times the grip this tyre has, and it is the two-box defect",
            both / peak
        );
        // …and it is not achieved by simply going to zero: the force is still
        // most of the peak, pointing between the axes.
        assert!(both > peak * 0.6, "{both} N is not a tyre, it is a hole");
        let (fx, fy) = tyre_force_n(&t, load, stat, 0.12, 0.12);
        assert!(
            (fx.abs() - fy.abs()).abs() < 1e-9,
            "equal slip on both axes did not split the force equally"
        );

        // **Asking for more of one costs the other.** At a fixed lateral slip,
        // adding drive slip must REDUCE the sideways force — the property a
        // per-axis clamp cannot have and the reason a car can be steered on the
        // throttle.
        let side_alone = tyre_force_n(&t, load, stat, 0.0, 0.08).1.abs();
        let side_under_power = tyre_force_n(&t, load, stat, 0.30, 0.08).1.abs();
        assert!(
            side_under_power < side_alone * 0.8,
            "wheelspin cost the tyre only {side_alone} → {side_under_power} N of \
             grip; under the old two-box clamp it would have cost none"
        );

        // The signs: a wheel outrunning the road pushes the car forward, and a
        // patch sliding right is resisted to the left.
        assert!(tyre_force_n(&t, load, stat, 0.1, 0.0).0 > 0.0);
        assert!(tyre_force_n(&t, load, stat, -0.1, 0.0).0 < 0.0);
        assert!(tyre_force_n(&t, load, stat, 0.0, 0.1).1 < 0.0);
        // No slip, no force; no load, no force; and no NaN reaches a solver.
        assert_eq!(tyre_force_n(&t, load, stat, 0.0, 0.0), (0.0, 0.0));
        assert_eq!(tyre_force_n(&t, 0.0, stat, 0.5, 0.5), (0.0, 0.0));
        let (x, y) = tyre_force_n(&t, load, stat, f64::NAN, 0.1);
        assert!(x.is_finite() && y.is_finite());
    }

    /// Steering tightens with speed, monotonically, between the two authored
    /// angles.
    #[test]
    fn the_steer_limit_tightens_with_speed() {
        let t = VehicleTuning::default();
        assert_eq!(steer_limit_deg(&t, 0.0), t.max_steer_deg);
        assert_eq!(steer_limit_deg(&t, t.max_speed_mps), t.min_steer_deg);
        assert_eq!(
            steer_limit_deg(&t, t.max_speed_mps * 4.0),
            t.min_steer_deg,
            "and clamps rather than inverting past the top speed"
        );
        let mid = steer_limit_deg(&t, t.max_speed_mps * 0.5);
        assert!(mid < t.max_steer_deg && mid > t.min_steer_deg);
        // The sign of the speed cannot matter: reversing does not straighten the
        // wheels.
        assert_eq!(steer_limit_deg(&t, -8.0), steer_limit_deg(&t, 8.0));
    }

    /// The suspension pushes and never pulls.
    #[test]
    fn the_suspension_never_pulls_the_chassis_down() {
        let t = VehicleTuning::default();
        assert_eq!(suspension_force_n(&t, 0.0, 0.0), 0.0);
        assert!(suspension_force_n(&t, 0.1, 0.0) > 0.0);
        // Extending fast enough that the damper would go negative.
        assert_eq!(
            suspension_force_n(&t, 0.01, -10.0),
            0.0,
            "a strut that pulled down would suck the car onto the road"
        );
        // And the travel is a clamp, not a suggestion.
        assert_eq!(
            suspension_force_n(&t, 10.0, 0.0),
            t.stiffness_n_per_m * t.travel_m
        );
    }

    /// The steer rotation is a plane rotation with portable trigonometry, and it
    /// keeps the basis orthonormal.
    #[test]
    fn steering_rotates_in_the_plane_and_stays_unit() {
        let f = steer_direction(DVec3::Z, DVec3::X, 90.0);
        assert!((f - DVec3::X).length() < 1e-12, "{f}");
        let f = steer_direction(DVec3::Z, DVec3::X, 0.0);
        assert!((f - DVec3::Z).length() < 1e-12);
        let f = steer_direction(DVec3::Z, DVec3::X, 30.0);
        assert!((f.length() - 1.0).abs() < 1e-12);
        assert!(f.y.abs() < 1e-12, "a steer never tilts the wheel");
    }

    /// **The load-bearing arm**: a vehicle at rest under gravity produces four
    /// upward forces that sum to its weight, so it neither sinks nor launches.
    ///
    /// The contacts are placed at the settling length, which is what the door
    /// would have cast; the claim is about the model, and it is a claim about
    /// newtons rather than about a report.
    #[test]
    fn a_resting_rig_holds_its_own_weight() {
        let mass = 1_200.0;
        let mut v = RaycastVehicle::new(rig(4));
        let t = *v.tuning();
        let settle = mass * 9.81 / 4.0 / t.stiffness_n_per_m;
        for (i, w) in v.wheels_mut().iter_mut().enumerate() {
            w.contact = Some(WheelContact {
                point: DVec3::new(i as f64, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - settle + 0.35,
            });
        }
        let mut out = Vec::new();
        v.solve(resting(mass), 1.0 / 60.0, &mut out);
        let up: f64 = out.iter().map(|f| f.force.y).sum();
        assert!(
            (up - mass * 9.81).abs() < 1.0,
            "the rig holds {up} N against a weight of {} N",
            mass * 9.81
        );
    }

    /// A wheel in the air contributes nothing, and the drive budget is shared
    /// over the wheels that are down — the rule that stops a raycast vehicle
    /// climbing walls on one wheel.
    #[test]
    fn a_wheel_in_the_air_carries_no_load_and_no_drive() {
        let mut v = RaycastVehicle::new(rig(4));
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
        assert!(
            out.iter().all(|f| f.force.length() < 1e-9),
            "an airborne rig produces no wheel force at all: {out:?}"
        );
        assert!(v.wheels().iter().all(|w| w.load_n == 0.0));
    }

    /// The tuning door answers by name, refuses what it does not know, and its
    /// name list is the door's own — extracted rather than restated.
    #[test]
    fn the_tuning_door_is_by_name_and_refuses_as_a_value() {
        let mut t = VehicleTuning::default();
        assert!(t.set("stiffness_n_per_m", 42.0));
        assert_eq!(t.stiffness_n_per_m, 42.0);
        assert!(
            !t.set("stiffness", 1.0),
            "a near-miss is refused, not guessed"
        );
        assert!(!t.set("stiffness_n_per_m", f64::NAN));
        assert_eq!(
            t.stiffness_n_per_m, 42.0,
            "…and the refusal changed nothing"
        );
        for name in VehicleTuning::names() {
            assert!(
                VehicleTuning::default().set(name, 1.0),
                "the advertised name {name} must be settable"
            );
        }
        assert_eq!(
            VehicleTuning::names().len(),
            62,
            "a name added to the door and not to the list is invisible to a UI"
        );
        // …and the list has no duplicate, which a sorted check alone allows.
        let mut unique = VehicleTuning::names().to_vec();
        unique.dedup();
        assert_eq!(unique.len(), VehicleTuning::names().len());
        let mut sorted = VehicleTuning::names().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, VehicleTuning::names());
    }

    /// **The projection onto [`VehicleClass`] is TOTAL, and it is checked by
    /// moving every one of the sixty-two** (island wave VEH2a).
    ///
    /// `VehicleClass::from_tuning` is sixty-two hand-written lines and
    /// `settings()` is sixty-two more. Two restatements of one list is the P29.6
    /// A14 shape, and the answer is not to write them once (the component must be
    /// `Reflect` + `Serialize` and the tunable must not) but to make the
    /// restatement **falsifiable**: a field that `from_tuning` forgot, or that
    /// `settings()` does not name, is a field whose distinct value does not
    /// survive the round trip.
    ///
    /// Distinct values, so a copy-paste that reads the *neighbouring* field —
    /// the way a sixty-two-line projection actually goes wrong — is caught too.
    ///
    /// [`VehicleClass`]: crate::components::VehicleClass
    #[test]
    fn the_class_and_the_tuning_are_the_same_sixty_two_numbers() {
        let names = VehicleTuning::names();
        let mut t = VehicleTuning::default();
        for (i, name) in names.iter().enumerate() {
            // 0.5 + i/1000 keeps every value finite, positive, distinct and
            // inside the ranges the shape functions clamp to.
            assert!(
                t.set(name, 0.5 + i as f64 / 1000.0),
                "{name} is not settable"
            );
        }
        let class = crate::components::VehicleClass::from_tuning(&t);
        assert_eq!(
            class.to_tuning(),
            t,
            "a tunable did not survive the round trip through the serialized class"
        );
        let seen: Vec<&str> = class.settings().iter().map(|(n, _)| *n).collect();
        assert_eq!(seen, names, "`settings()` is not `names()`, in order");
        for (i, (name, value)) in class.settings().into_iter().enumerate() {
            assert_eq!(
                value,
                0.5 + i as f64 / 1000.0,
                "`{name}` reports the value of a different field"
            );
        }
        // …and the class's own by-name door is the tuning's, which is what lets a
        // catalogue row say `torque_curve_bias = 0.3` beside `wheel_radius_m`.
        let mut c = crate::components::VehicleClass::default();
        assert!(c.set("peak_torque_nm", 410.0));
        assert_eq!(c.peak_torque_nm, 410.0);
        assert!(!c.set("peak_torque", 1.0));
        assert!(!c.set("peak_torque_nm", f64::NAN));
        assert_eq!(c.peak_torque_nm, 410.0);
    }

    /// **The shape function is monotone, lands on both ends, and `k = 0.5` is the
    /// identity** — the one curve the torque and the tyre both bend with.
    #[test]
    fn the_bias_curve_is_monotone_and_its_slope_is_the_knob() {
        for k in [0.05, 0.2, 0.5, 0.74, 0.95] {
            assert_eq!(curve_bias(0.0, k), 0.0, "k = {k}");
            assert!((curve_bias(1.0, k) - 1.0).abs() < 1e-12, "k = {k}");
            let mut last = -1.0;
            for i in 0..=100 {
                let v = curve_bias(i as f64 / 100.0, k);
                assert!(v >= last - 1e-12, "k = {k} is not monotone at {i}");
                assert!((0.0..=1.0).contains(&v));
                last = v;
            }
        }
        // k = 0.5 is the identity, exactly.
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            assert!((curve_bias(t, 0.5) - t).abs() < 1e-12);
        }
        // The slope at the origin is k/(1-k): 0.8 arrives four times as fast,
        // 0.2 a quarter as fast. Measured over a small step, which is what an
        // author is actually promised.
        let h = 1e-6;
        assert!((curve_bias(h, 0.8) / h - 4.0).abs() < 1e-4);
        assert!((curve_bias(h, 0.2) / h - 0.25).abs() < 1e-4);
        // A high bias is ABOVE the line everywhere in between, a low one below —
        // which is the whole claim ("torque arrives early" / "arrives late").
        assert!(curve_bias(0.3, 0.8) > 0.3 && curve_bias(0.3, 0.2) < 0.3);
        // Refusals are values.
        assert_eq!(curve_bias(f64::NAN, 0.5), 0.0);
        assert_eq!(curve_bias(0.4, f64::NAN), 0.4);
        assert_eq!(curve_bias(4.0, 0.5), 1.0);
        assert_eq!(curve_bias(-4.0, 0.5), 0.0);
    }

    /// **The gearbox answers one ratio per gear, refuses the ones it does not
    /// have, and reverse is one of them.**
    #[test]
    fn the_gearbox_is_a_count_and_a_ratio_per_gear() {
        let t = VehicleTuning::default();
        assert_eq!(t.gears(), 6);
        assert_eq!(t.drive_ratio(1), t.gear_1_ratio * t.final_drive);
        assert_eq!(t.drive_ratio(6), t.gear_6_ratio * t.final_drive);
        assert_eq!(t.drive_ratio(-1), t.reverse_ratio * t.final_drive);
        assert_eq!(t.drive_ratio(0), 0.0, "neutral drives nothing");
        assert_eq!(
            t.drive_ratio(7),
            0.0,
            "a six-speed box does not have a seventh gear even though the field \
             exists"
        );
        // The ratios shorten monotonically, or the box is not a box.
        for g in 1..t.gears() as i32 {
            assert!(
                t.drive_ratio(g) > t.drive_ratio(g + 1),
                "gear {g} is not taller than gear {}",
                g + 1
            );
        }
        // Reverse is between first and second — a real car's is.
        assert!(t.drive_ratio(-1) < t.drive_ratio(1) && t.drive_ratio(-1) > t.drive_ratio(2));
        // A count an author typed wrong is clamped, not a panic and not a
        // division by zero.
        let mut wrong = t;
        assert!(wrong.set("gear_count", 0.0));
        assert_eq!(wrong.gears(), 1);
        assert!(wrong.set("gear_count", 99.0));
        assert_eq!(wrong.gears(), MAX_GEARS);
        assert!(wrong.set("gear_count", 3.4));
        assert_eq!(wrong.gears(), 3);
    }

    /// **The seat warp is two authored numbers again**, and they are the window
    /// v25 could not carry.
    #[test]
    fn the_enter_window_is_authored_and_not_taken_from_the_default() {
        let mut t = VehicleTuning::default();
        assert_eq!(t.enter_window(), inf_anim::WarpWindow::new(0.1, 0.45));
        assert!(t.set("enter_warp_start", 0.2));
        assert!(t.set("enter_warp_end", 0.8));
        assert_eq!(t.enter_window(), inf_anim::WarpWindow::new(0.2, 0.8));
        // …and it survives the serialized class, which is the carried item this
        // closes (VEH1a ledger, "enter_window is still absent from VehicleClass").
        let class = crate::components::VehicleClass::from_tuning(&t);
        assert_eq!(class.to_tuning().enter_window(), t.enter_window());
        assert!(VehicleTuning::names().contains(&"enter_warp_start"));
        assert!(VehicleTuning::names().contains(&"enter_warp_end"));
    }

    /// Back at speed is a **brake**; back at a standstill is reverse.
    #[test]
    fn pressing_back_at_speed_brakes_and_at_rest_reverses() {
        let back = crate::math::Vec2d::new(0.0, -1.0);
        let c = VehicleControls::from_intent(back, 12.0, false, 0.0);
        assert_eq!((c.throttle, c.brake), (0.0, 1.0));
        let c = VehicleControls::from_intent(back, 0.0, false, 0.0);
        assert_eq!((c.throttle, c.brake), (-1.0, 0.0));
        // …and symmetrically, forward while reversing is a brake.
        let fwd = crate::math::Vec2d::new(0.0, 1.0);
        let c = VehicleControls::from_intent(fwd, -12.0, false, 0.0);
        assert_eq!((c.throttle, c.brake), (0.0, 1.0));
        let c = VehicleControls::from_intent(crate::math::Vec2d::new(1.0, 0.0), 0.0, true, 0.0);
        assert_eq!(c.steer, 1.0);
        assert!(c.handbrake);
    }

    /// **Every family's parts stay inside the chassis they are drawn on**, fill
    /// it, and have real air in them.
    ///
    /// The I8b box-builder arms, one content kind over — and the third clause is
    /// the one that says "a car" rather than "a box": a silhouette whose parts
    /// tile the whole hull is a rectangular prism with extra draw calls.
    #[test]
    fn every_body_family_is_a_silhouette_inside_its_own_hull() {
        for family in VehicleBody::ALL {
            let parts = family.parts();
            assert!(
                parts.len() >= 4,
                "{}: {} parts is not a silhouette",
                family.name(),
                parts.len()
            );
            let mut lo = Vec3d::splat(f64::MAX);
            let mut hi = Vec3d::splat(f64::MIN);
            let mut volume = 0.0f64;
            let mut names: Vec<&str> = Vec::new();
            for p in parts {
                for (c, h) in [
                    (p.centre.x, p.half.x),
                    (p.centre.y, p.half.y),
                    (p.centre.z, p.half.z),
                ] {
                    assert!(
                        h > 0.0,
                        "{}/{}: a part with no thickness",
                        family.name(),
                        p.name
                    );
                    assert!(
                        c - h >= -1.0001 && c + h <= 1.0001,
                        "{}/{}: reaches {} to {}, outside its own hull",
                        family.name(),
                        p.name,
                        c - h,
                        c + h
                    );
                }
                lo = Vec3d::new(
                    lo.x.min(p.centre.x - p.half.x),
                    lo.y.min(p.centre.y - p.half.y),
                    lo.z.min(p.centre.z - p.half.z),
                );
                hi = Vec3d::new(
                    hi.x.max(p.centre.x + p.half.x),
                    hi.y.max(p.centre.y + p.half.y),
                    hi.z.max(p.centre.z + p.half.z),
                );
                volume += 8.0 * p.half.x * p.half.y * p.half.z;
                names.push(p.name);
            }
            // It FILLS the hull on every axis, so the drawn car is the size of
            // its own collider rather than a shrunken proxy inside it (the I8b
            // `every_family_is_real_geometry_inside_the_unit_box` clause).
            for (axis, l, h) in [("x", lo.x, hi.x), ("y", lo.y, hi.y), ("z", lo.z, hi.z)] {
                assert!(
                    l <= -0.98 && h >= 0.98,
                    "{}: axis {axis} spans [{l}, {h}] — the body does not fill \
                     its hull",
                    family.name()
                );
            }
            // …and it is not a solid block: the hull is 8 units of volume in
            // fraction space, and a silhouette leaves a good part of it empty.
            assert!(
                volume < 7.0,
                "{}: the parts sum to {volume} of the hull's 8 — that is a \
                 rectangular prism drawn in {} pieces",
                family.name(),
                parts.len()
            );
            // The strongest anti-block clause is the one about the ROOF: the
            // topmost part must be narrower and much shorter than the hull, or
            // the car has no greenhouse and the "silhouette" is a brick.
            let roof = parts
                .iter()
                .max_by(|a, b| (a.centre.y + a.half.y).total_cmp(&(b.centre.y + b.half.y)))
                .expect("a family has parts");
            assert!(
                roof.half.x < 0.96 && roof.half.z < 0.6,
                "{}: its topmost part `{}` is {} x {} of the hull — a roof the \
                 size of the car is a box",
                family.name(),
                roof.name,
                roof.half.x,
                roof.half.z
            );
            names.sort_unstable();
            let n = names.len();
            names.dedup();
            assert_eq!(
                names.len(),
                n,
                "{}: two parts share a name, so they would share a Guid",
                family.name()
            );
            // Every part of ONE family mints a distinct id. (Across families it
            // need not: two silhouettes are never drawn on one chassis, and
            // `lower` deliberately means the same thing in both.)
            let mut ids: Vec<Uuid> = parts
                .iter()
                .map(|p| body_part_guid(Uuid::from_u128(7), p.name))
                .collect();
            ids.sort();
            let n = ids.len();
            ids.dedup();
            assert_eq!(ids.len(), n, "{}: two parts mint one id", family.name());
        }
    }

    /// A part's `Guid` follows the chassis AND the part name, and two parts
    /// never agree.
    #[test]
    fn a_body_parts_guid_is_derived_from_what_it_is() {
        let a = Uuid::from_u128(0xA1);
        let b = Uuid::from_u128(0xB2);
        assert_eq!(body_part_guid(a, "cabin"), body_part_guid(a, "cabin"));
        assert_ne!(body_part_guid(a, "cabin"), body_part_guid(a, "bonnet"));
        assert_ne!(
            body_part_guid(a, "cabin"),
            body_part_guid(b, "cabin"),
            "two cars would share one entity"
        );
        assert!(VehicleBody::ALL
            .into_iter()
            .flat_map(|f| f.parts())
            .all(|p| !body_part_guid(a, p.name).is_nil()));
    }

    /// **A catalogue row is TOML and costs no schema**, and an unknown key is a
    /// refusal by name rather than a silent default.
    #[test]
    fn a_vehicle_catalogue_row_reads_geometry_and_tuning_through_one_door() {
        let mut defs = VehicleDefs::default();
        let n = defs
            .merge_toml(
                "[sedan]\nlabel = \"Saloon\"\n[sedan.vehicle]\nbody = \"sedan\"\n\
                 half_length_m = 2.2\nwheel_radius_m = 0.34\nmax_speed_mps = 32.0\n\
                 [truck.vehicle]\nbody = \"truck\"\nhalf_width_m = 1.1\n\
                 max_engine_force_n = 14000.0\n[bandage]\nstack_max = 4\n",
            )
            .expect("the catalogue parses");
        assert_eq!(
            n, 2,
            "a row with no `[<id>.vehicle]` is skipped, not refused"
        );
        let sedan = defs.get("sedan").expect("the sedan row");
        assert_eq!(sedan.body, VehicleBody::Sedan);
        assert_eq!(sedan.half_extents.z, 2.2);
        assert_eq!(sedan.wheel_radius_m, 0.34);
        assert_eq!(
            sedan.class.max_speed_mps, 32.0,
            "a tuning name reaches `VehicleClass` through the same `set`"
        );
        let truck = defs.get("truck").expect("the truck row");
        assert_eq!(truck.body, VehicleBody::Truck);
        assert_eq!(truck.half_extents.x, 1.1);
        assert_eq!(truck.class.max_engine_force_n, 14_000.0);
        assert_eq!(
            truck.wheel_radius_m,
            VehicleDef::default().wheel_radius_m,
            "an unmentioned key keeps the default"
        );

        // Refusals, by name.
        for (text, why) in [
            ("[x.vehicle]\nbody = \"hovercraft\"\n", "an unknown body"),
            ("[x.vehicle]\nwheelbase = 3.0\n", "an unknown key"),
            ("[x.vehicle]\nwheel_radius_m = \"big\"\n", "a non-number"),
            ("[x]\nvehicle = 3\n", "a `vehicle` that is not a table"),
        ] {
            assert!(
                VehicleDefs::default().merge_toml(text).is_err(),
                "{why} was accepted: {text}"
            );
        }
        // …and a non-finite number is refused rather than stored.
        let mut def = VehicleDef::default();
        assert!(!def.set("wheel_radius_m", f64::NAN));
        assert!(!def.set("max_speed_mps", f64::INFINITY));
        assert_eq!(def, VehicleDef::default());
        for name in VehicleDef::geometry_names() {
            assert!(
                VehicleDef::default().set(name, 1.0),
                "{name} is not settable"
            );
        }
        for name in VehicleTuning::names() {
            assert!(
                VehicleDef::default().set(name, 1.0),
                "the tuning name {name} does not reach a catalogue row"
            );
        }
        let mut sorted = VehicleDef::geometry_names().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, VehicleDef::geometry_names());
    }

    /// **The default def is a car-shaped car** — longer than it is wide, with
    /// its wheels inside its own body.
    ///
    /// The arm the P29.7 fixture did not have. `PHASE29_CAR_HALF` is
    /// `(2.0, 0.5, 1.0)` — four metres across the track and two along the
    /// wheelbase — and it survived a whole phase because nothing drew the body
    /// at the size of its collider.
    #[test]
    fn the_default_vehicle_is_longer_than_it_is_wide_and_covers_its_wheels() {
        let d = VehicleDef::default();
        assert!(
            d.half_extents.z > d.half_extents.x,
            "a car {} m wide and {} m long is a car built sideways",
            d.half_extents.x * 2.0,
            d.half_extents.z * 2.0
        );
        let mounts = d.wheel_mounts();
        assert!(
            mounts[0].z > 0.0 && mounts[1].z > 0.0,
            "the front pair steers"
        );
        assert!(!mounts[2].z.is_sign_positive() && !mounts[3].z.is_sign_positive());
        for m in mounts {
            assert!(
                m.x.abs() <= d.half_extents.x + 0.35,
                "a wheel at x = {} pokes {} m out of a body {} m wide",
                m.x,
                m.x.abs() - d.half_extents.x,
                d.half_extents.x * 2.0
            );
            assert!(
                m.z.abs() < d.half_extents.z,
                "a wheel at z = {} is outside a body {} m long",
                m.z,
                d.half_extents.z * 2.0
            );
        }
        // The mass is a car's mass, which is what the springs were sized for.
        let volume = 8.0 * d.half_extents.x * d.half_extents.y * d.half_extents.z;
        let kg = volume * d.density_kg_m3;
        assert!(
            (900.0..1600.0).contains(&kg),
            "the default car weighs {kg} kg over {volume} m3"
        );
    }

    /// **The engine cue rises with the revs and swells with the throttle**, and
    /// an idling engine is quieter rather than silent.
    ///
    /// Written against the mutation that matters: a cue that read the *load*
    /// into the pitch and the *revs* into the volume would pass a "both move"
    /// check perfectly and make a car that revs when it brakes.
    #[test]
    fn the_engine_cue_reads_revs_as_pitch_and_throttle_as_volume() {
        let idle = engine_cue(0.0, 0.0, 1.0, 1.0);
        let coasting = engine_cue(1.0, 0.0, 1.0, 1.0);
        let stalled_flat_out = engine_cue(0.0, 1.0, 1.0, 1.0);
        assert_eq!(idle.pitch, ENGINE_IDLE_PITCH);
        assert_eq!(idle.volume, ENGINE_IDLE_GAIN);
        assert_eq!(coasting.pitch, ENGINE_TOP_PITCH);
        assert_eq!(
            coasting.volume, ENGINE_IDLE_GAIN,
            "a car coasting at its top speed with the throttle shut is quiet, \
             and it is the revs that make it high"
        );
        assert_eq!(
            stalled_flat_out.pitch, ENGINE_IDLE_PITCH,
            "a car standing still with the throttle flat is LOUD and LOW; a cue \
             that crossed its two inputs would have it screaming"
        );
        assert_eq!(stalled_flat_out.volume, 1.0);
        // Monotone in each input, and the mid-point is a mid-point rather than a
        // switch (the `night_glow_step` lesson: a ramp is asserted through its
        // middle).
        let half = engine_cue(0.5, 0.5, 1.0, 1.0);
        assert!((half.pitch - (ENGINE_IDLE_PITCH + ENGINE_TOP_PITCH) / 2.0).abs() < 1e-12);
        assert!((half.volume - (ENGINE_IDLE_GAIN + 1.0) / 2.0).abs() < 1e-12);
        // The emitter's own authored numbers scale it.
        let truck = engine_cue(0.5, 0.5, 0.4, 0.8);
        assert!((truck.pitch - half.pitch * 0.4).abs() < 1e-12);
        assert!((truck.volume - half.volume * 0.8).abs() < 1e-12);
    }

    /// A refusal is a value, and no NaN reaches a device mixer.
    #[test]
    fn the_engine_cue_clamps_and_refuses_rather_than_propagating() {
        let over = engine_cue(4.0, 9.0, 1.0, 1.0);
        assert_eq!(over, engine_cue(1.0, 1.0, 1.0, 1.0));
        let under = engine_cue(-3.0, -1.0, 1.0, 1.0);
        assert_eq!(under, engine_cue(0.0, 0.0, 1.0, 1.0));
        for cue in [
            engine_cue(f64::NAN, 0.5, 1.0, 1.0),
            engine_cue(0.5, f64::NAN, 1.0, 1.0),
            engine_cue(0.5, 0.5, f64::NAN, 1.0),
            engine_cue(0.5, 0.5, 1.0, f64::INFINITY),
        ] {
            assert!(
                cue.pitch.is_finite() && cue.volume.is_finite(),
                "a non-finite input produced {cue:?}, which reaches a mixer"
            );
            assert!(cue.pitch > 0.0 && cue.volume >= 0.0);
        }
    }

    /// **The class decides what revs mean**, and the default is silence.
    ///
    /// Since VEH2a the road car's revs are its own **rpm** between idle and the
    /// redline, so this drives one rather than asserting an algebraic identity on
    /// the road speed — and it asserts the thing that identity could never say:
    /// that the revs **fall when the box shifts up** while the car keeps
    /// accelerating.
    #[test]
    fn the_engine_state_is_the_classs_own_and_defaults_to_silent() {
        let mut v = RaycastVehicle::new(rig(4));
        // A car at rest with no throttle is at idle, which is revs of zero.
        assert_eq!(v.engine_state(0.0), (0.0, 0.0));
        v.control(VehicleControls {
            throttle: -1.0,
            ..Default::default()
        });
        assert_eq!(
            v.engine_state(0.0).1,
            1.0,
            "reversing is not quieter than driving"
        );

        // Now drive it, on the ground, and watch the revs saw.
        let mut v = RaycastVehicle::new(rig(4));
        let t = *v.tuning();
        for w in v.wheels_mut() {
            w.contact = Some(WheelContact {
                point: DVec3::new(0.0, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.12 + 0.35,
            });
        }
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut chassis = resting(1_200.0);
        let mut out = Vec::new();
        let (mut revs, mut gears, mut drops) = (Vec::new(), Vec::new(), 0usize);
        for _ in 0..600 {
            out.clear();
            v.solve(chassis, 1.0 / 60.0, &mut out);
            // Integrate the chassis crudely: this arm is about the ENGINE, and a
            // rigid body would only add rapier to a claim about revs.
            let fz: f64 = out.iter().map(|f| f.force.z).sum();
            chassis.linvel.z += fz / 1_200.0 / 60.0;
            let r = v.engine_state(chassis.linvel.z).0;
            if revs.last().is_some_and(|p: &f64| r < *p - 0.05) {
                drops += 1;
            }
            revs.push(r);
            gears.push(v.gear());
        }
        assert!(
            chassis.linvel.z > 8.0,
            "the rig only reached {} m/s, so there was nothing to shift",
            chassis.linvel.z
        );
        assert!(
            *gears.last().unwrap() > 1,
            "the box never left first: {:?}",
            &gears[..8]
        );
        assert!(
            drops >= 1,
            "the revs never fell across {} gears — a rev needle that only ever \
             rises is road speed wearing a tachometer, which is exactly what \
             VEH1a shipped",
            gears.last().unwrap()
        );
        assert!(revs.iter().all(|r| (0.0..=1.0).contains(r)));

        // The trait's default is silence: a class that has not thought about
        // sound makes none, rather than inheriting a road car's curve.
        struct Mute(VehicleRig);
        impl Vehicle for Mute {
            fn gear(&self) -> i32 {
                0
            }

            fn rig(&self) -> &VehicleRig {
                &self.0
            }
            fn set_rig(&mut self, rig: VehicleRig) {
                self.0 = rig;
            }
            fn wheels(&self) -> &[WheelState] {
                &[]
            }
            fn wheels_mut(&mut self) -> &mut [WheelState] {
                &mut []
            }
            fn control(&mut self, _: VehicleControls) {}
            fn tune(&mut self, _: &str, _: f64) -> bool {
                false
            }
            fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
                (0.1, inf_anim::WarpWindow::new(0.0, 0.1))
            }
            fn suspension_rest_m(&self) -> f64 {
                0.0
            }
            fn solve(&mut self, _: ChassisState, _: f64, _: &mut Vec<WheelForce>) {}
        }
        assert_eq!(Mute(rig(0)).engine_state(30.0), (0.0, 0.0));
    }

    /// **The handbrake is the rear wheels and only those** — which is what makes
    /// a handbrake turn a turn rather than a stop.
    ///
    /// Asserted at the model, where a wheel's identity is visible: `solve` pushes
    /// **two** forces per grounded wheel in rig order — the suspension and the
    /// tyre — so wheel `i`'s ground force is `out[2i + 1]`, and the fixture's
    /// first two wheels are the front pair (`+Z`, therefore steered). Applied to
    /// all four, this arm would find the front pair braking too.
    ///
    /// Three became two at VEH2a: the lateral and longitudinal forces used to be
    /// pushed separately because they were computed separately, which is the
    /// two-box defect stated in the output vector. One contact patch makes one
    /// force.
    ///
    /// And the handbrake is now a **lock**, not a force: the rear wheels' `ω` is
    /// what it stops, so this arm reads the wheel state as well as the newtons.
    #[test]
    fn the_handbrake_is_the_rear_wheels_and_only_those() {
        let mut v = RaycastVehicle::new(rig(4));
        let t = *v.tuning();
        for (i, w) in v.wheels_mut().iter_mut().enumerate() {
            w.contact = Some(WheelContact {
                point: DVec3::new(i as f64 * 0.1, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.1 + 0.35,
            });
            // Rolling with the road, so the handbrake has something to stop.
            w.omega_rad_s = 8.0 / 0.35;
        }
        v.control(VehicleControls {
            handbrake: true,
            ..Default::default()
        });
        let mut chassis = resting(1_200.0);
        // Rolling forward, so there is something for a brake to resist.
        chassis.linvel = DVec3::new(0.0, 0.0, 8.0);
        let mut out = Vec::new();
        // Half a second, because a handbrake LOCKS a wheel rather than pushing
        // the car: on the first step the wheel is still rolling with the road and
        // the honest ground force is zero. A one-step arm would have measured the
        // model's transient and called it the handbrake.
        for _ in 0..30 {
            out.clear();
            v.solve(chassis, 1.0 / 60.0, &mut out);
        }

        let ground = |i: usize| out[i * 2 + 1].force.z;
        let (front, rear) = ((ground(0) + ground(1)) * 0.5, (ground(2) + ground(3)) * 0.5);
        assert!(
            rear < -100.0,
            "a locked rear wheel must resist the motion; it pushed {rear} N"
        );
        assert!(
            front > rear * 0.2,
            "the front wheels braked {front} N against the rear's {rear} N — the \
             handbrake reached a steered wheel"
        );
        // …and the front pair is only rolling resistance, which barely slows a
        // free-rolling wheel at all.
        let load = v.wheels()[0].load_n;
        assert!(
            front.abs() < load * t.rolling_resistance * 4.0,
            "the front wheels carried {front} N, which is far more than rolling \
             resistance on a {load} N load"
        );
        // THE LOCK, read off the wheels: the rears are turning far slower than
        // the road and the fronts are still rolling with it.
        let w = v.wheels();
        assert!(
            w[2].omega_rad_s < w[0].omega_rad_s * 0.6,
            "the rear wheels are turning at {} rad/s against the front's {} — the \
             handbrake did not lock anything",
            w[2].omega_rad_s,
            w[0].omega_rad_s
        );
        assert!(
            w[2].slip_ratio < -0.05 && w[0].slip_ratio > -0.05,
            "the rears are not slipping ({}) or the fronts are ({})",
            w[2].slip_ratio,
            w[0].slip_ratio
        );
    }

    /// Park a four-wheel rig on flat ground with every wheel in contact, **with
    /// the three driver aids switched off**.
    ///
    /// Off by default because an aid's whole job is to hide the mechanism under
    /// it: a lockup arm with ABS on measures ABS, and a differential arm with
    /// traction control on measures traction control. The aids get their own
    /// fixtures, which turn them back on one at a time.
    fn grounded(tuning: &[(&str, f64)]) -> RaycastVehicle {
        let mut v = RaycastVehicle::new(rig(4));
        for (name, value) in [
            ("abs_slip", 0.0),
            ("traction_control_slip", 0.0),
            ("stability_control", 0.0),
        ] {
            assert!(v.tune(name, value));
        }
        for (name, value) in tuning {
            assert!(v.tune(name, *value), "the fixture set an unknown `{name}`");
        }
        let rest = v.tuning().rest_length_m;
        for w in v.wheels_mut() {
            w.contact = Some(WheelContact {
                point: DVec3::new(0.0, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: rest - 0.1 + 0.35,
            });
        }
        v
    }

    /// **The drivetrain is the split, and the split is which axle pushes.**
    ///
    /// Read off the wheels' own speeds on a LOW-GRIP surface, where the driven
    /// axle spins and the undriven one cannot: a check on the chassis's motion
    /// alone would pass for any split at all, because all three drivetrains move
    /// a car forwards.
    #[test]
    fn the_drivetrain_split_decides_which_axle_pushes() {
        let low = [("longitudinal_grip", 0.25), ("lateral_grip", 0.25)];
        let spun = |split: f64| -> (f64, f64) {
            let mut v = grounded(&[low[0], low[1], ("front_torque_split", split)]);
            v.control(VehicleControls {
                throttle: 1.0,
                ..Default::default()
            });
            let mut out = Vec::new();
            for _ in 0..20 {
                out.clear();
                v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
            }
            let w = v.wheels();
            (
                (w[0].omega_rad_s + w[1].omega_rad_s) / 2.0,
                (w[2].omega_rad_s + w[3].omega_rad_s) / 2.0,
            )
        };
        // The fixture's first two wheels are `+Z` and therefore the front pair.
        let (ff, fr) = spun(1.0);
        assert!(
            ff > 1.0 && fr.abs() < 1e-9,
            "front-wheel drive turned the rears at {fr} rad/s"
        );
        let (rf, rr) = spun(0.0);
        assert!(
            rr > 1.0 && rf.abs() < 1e-9,
            "rear-wheel drive turned the fronts at {rf} rad/s"
        );
        let (af, ar) = spun(AWD_FRONT_SPLIT);
        assert!(af > 0.0 && ar > 0.0, "all-wheel drive turned one axle only");
        assert!(
            ar > af,
            "a {AWD_FRONT_SPLIT} front split must put MORE torque on the rear \
             axle, but the front spun to {af} against the rear's {ar}"
        );
        // …and the whole engine is spent either way: an axle that gets nothing
        // does not shrink the car's total drive.
        assert!(
            (ff - rr).abs() < ff * 0.05,
            "front drive spun to {ff} and rear drive to {rr} — the split is \
             losing torque instead of moving it"
        );
    }

    /// **A locked differential drags a spinning wheel back; an open one does
    /// not** — and it moves speed rather than making it.
    ///
    /// One rear wheel lifted, which is the case a diff exists for: on an open
    /// axle the airborne wheel takes all the speed and the grounded one is left
    /// with nothing.
    #[test]
    fn a_locked_diff_pulls_a_lifted_wheel_back_to_its_partner() {
        // (grounded ω, lifted ω, metres the car covered in half a second)
        let run = |lock: f64| -> (f64, f64, f64) {
            let mut v = grounded(&[
                ("front_torque_split", 0.0),
                ("diff_lock_rear", lock),
                // Enough grip that the grounded wheel really can hold, or both
                // wheels spin and there is no asymmetry to measure.
                ("longitudinal_grip", 2.0),
            ]);
            // Wheel 3 is a rear one, and it is in the air.
            v.wheels_mut()[3].contact = None;
            v.control(VehicleControls {
                throttle: 1.0,
                ..Default::default()
            });
            let mut chassis = resting(1_200.0);
            let mut out = Vec::new();
            let mut travelled = 0.0;
            for _ in 0..30 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
                let fz: f64 = out.iter().map(|f| f.force.z).sum();
                chassis.linvel.z += fz / 1_200.0 / 60.0;
                travelled += chassis.linvel.z / 60.0;
            }
            (
                v.wheels()[2].omega_rad_s,
                v.wheels()[3].omega_rad_s,
                travelled,
            )
        };
        let (open_down, open_up, open_far) = run(0.0);
        let (lock_down, lock_up, lock_far) = run(1.0);
        let (half_down, half_up, half_far) = run(0.5);
        assert!(
            open_up > open_down * 3.0,
            "an OPEN diff left the lifted wheel at {open_up} rad/s against the \
             grounded one's {open_down} — it is behaving like a locked one"
        );
        // A lock **starves** the runaway wheel: it does not force the two speeds
        // together (that would be a rigid axle, and a rigid axle is not what a
        // torque-splitting diff is), it stops feeding the one that is already
        // fastest — so the lifted wheel ends up SLOWER than an open axle left it.
        //
        // Measured: open 4.3 / 52.6 rad/s, locked 1.4 / 46.7.
        assert!(
            lock_up < open_up * 0.95,
            "a locked axle left the lifted wheel at {lock_up:.1} rad/s against an \
             open one's {open_up:.1} — it is not starving the runaway"
        );
        // **What the GROUNDED wheel does is deliberately NOT asserted here**, and
        // the reason is worth writing down. A grounded wheel that sticks turns at
        // the road's speed, so its speed is a statement about the CAR — and the
        // car is confounded: on an open axle the runaway wheel drags the engine
        // into its power band and then into its limiter, so the open car is
        // briefly the faster one for a reason that has nothing to do with
        // differentials (1.4 rad/s locked against 4.3 open, with the half-lock
        // outside both). What the lock is worth has to be read at matched revs,
        // and that is the arm below.
        let _ = (lock_down, open_down, half_down, half_up);

        // **What the lock is WORTH, isolated from the engine.**
        //
        // A distance comparison over the runs above is NOT the arm, and the
        // reason is worth recording: at full throttle the open diff's runaway
        // wheel drags the engine to 3 200 rpm and most of its peak torque while
        // the locked axle is still at 1 400 in first, so the open car goes
        // further (0.39 m against 0.32) for a reason that has nothing to do with
        // differentials. The lock's own contribution is read at MATCHED revs and
        // at a throttle the single grounded wheel can still hold.
        let force = |lock: f64| -> f64 {
            let mut v = grounded(&[
                ("front_torque_split", 0.0),
                ("longitudinal_grip", 2.0),
                ("diff_lock_rear", lock),
            ]);
            v.wheels_mut()[3].contact = None;
            v.wheels_mut()[2].omega_rad_s = 0.0;
            v.wheels_mut()[3].omega_rad_s = 40.0;
            v.control(VehicleControls {
                throttle: 0.3,
                ..Default::default()
            });
            let mut out = Vec::new();
            v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
            out.iter().map(|f| f.force.z).sum()
        };
        let (open_step, lock_step) = (force(0.0), force(1.0));
        assert!(
            lock_step > open_step * 1.8,
            "a locked axle put {lock_step:.0} N down against an open one's \
             {open_step:.0} N from the same state at the same revs — the whole \
             axle's torque did not reach the wheel that has grip"
        );
        let _ = (open_far, lock_far, half_far);
    }

    /// **Brake bias sends the budget to the axle it names.**
    #[test]
    fn the_brake_bias_is_the_share_the_front_axle_takes() {
        let axles = |bias: f64| -> (f64, f64) {
            let mut v = grounded(&[("brake_bias", bias), ("abs_slip", 0.0)]);
            for w in v.wheels_mut() {
                w.omega_rad_s = 12.0 / 0.35;
            }
            v.control(VehicleControls {
                brake: 1.0,
                ..Default::default()
            });
            let mut chassis = resting(1_200.0);
            chassis.linvel = DVec3::new(0.0, 0.0, 12.0);
            let mut out = Vec::new();
            for _ in 0..4 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
            }
            let g = |i: usize| out[i * 2 + 1].force.z;
            ((g(0) + g(1)) * 0.5, (g(2) + g(3)) * 0.5)
        };
        let (f, r) = axles(0.9);
        assert!(
            f < r * 1.5,
            "a 0.9 front bias braked {f} N at the front against {r} N at the rear"
        );
        let (f, r) = axles(0.1);
        assert!(
            r < f * 1.5,
            "a 0.1 front bias braked {r} N at the rear against {f} N at the front"
        );
        // A bias of 0.5 is even, which is the control that says the arm above is
        // reading the bias and not the fixture's geometry.
        let (f, r) = axles(0.5);
        assert!(
            (f - r).abs() < f.abs() * 0.05,
            "an even bias braked {f} N at the front and {r} N at the rear"
        );
    }

    /// **The steering rack has a rate, and it returns to centre faster than it
    /// leaves it.**
    #[test]
    fn the_rack_takes_time_to_turn_and_less_time_to_come_back() {
        let mut v = grounded(&[]);
        let t = *v.tuning();
        let limit = steer_limit_deg(&t, 0.0);
        v.control(VehicleControls {
            steer: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        assert_eq!(v.steer_deg(), 0.0);
        v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
        let after_one = v.steer_deg();
        assert!(
            after_one > 0.0 && after_one < limit,
            "the rack reached {after_one}° of {limit}° in ONE step — that is the \
             instant steering VEH2a exists to retire"
        );
        assert!(
            (after_one - t.steer_rate_deg_per_s / 60.0).abs() < 1e-9,
            "one step of a {}°/s rack moved {after_one}°",
            t.steer_rate_deg_per_s
        );
        // Held, it arrives at full lock and stops there.
        for _ in 0..120 {
            out.clear();
            v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
        }
        assert!((v.steer_deg() - limit).abs() < 1e-9);
        // Released, it comes back at the OTHER rate — faster, because a real
        // rack's castor is what centres it.
        v.control(VehicleControls::default());
        let before = v.steer_deg();
        out.clear();
        v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
        let back = before - v.steer_deg();
        assert!(
            (back - t.steer_return_deg_per_s / 60.0).abs() < 1e-9,
            "the rack returned {back}° in a step against a {}°/s return rate",
            t.steer_return_deg_per_s
        );
        assert!(back > t.steer_rate_deg_per_s / 60.0);
        // …and the speed-sensitive limit binds the RESULT, not just the demand:
        // a car that sped up with the wheel held must give the angle back.
        let mut fast = resting(1_200.0);
        fast.linvel = DVec3::new(0.0, 0.0, t.max_speed_mps);
        v.control(VehicleControls {
            steer: 1.0,
            ..Default::default()
        });
        for _ in 0..240 {
            out.clear();
            v.solve(fast, 1.0 / 60.0, &mut out);
        }
        assert!(
            (v.steer_deg() - t.min_steer_deg).abs() < 1e-9,
            "at the top speed the rack sits at {}° against a {}° limit",
            v.steer_deg(),
            t.min_steer_deg
        );
    }

    /// **Ackermann turns the inside wheel more**, blends to parallel at zero, and
    /// never inverts.
    #[test]
    fn the_inside_front_wheel_turns_more_than_the_outside_one() {
        let (w, l) = (0.84, 2.84);
        for rack in [5.0, 20.0, 35.0, -35.0] {
            let inner = ackermann_deg(rack, true, w, l, 1.0);
            let outer = ackermann_deg(rack, false, w, l, 1.0);
            assert!(
                inner.abs() > rack.abs() && outer.abs() < rack.abs(),
                "at {rack}° the inside wheel took {inner}° and the outside {outer}°"
            );
            assert_eq!(
                inner.signum(),
                rack.signum(),
                "a wheel steered the wrong way"
            );
            assert_eq!(outer.signum(), rack.signum());
            // Zero Ackermann is parallel steering, exactly.
            assert_eq!(ackermann_deg(rack, true, w, l, 0.0), rack);
            assert_eq!(ackermann_deg(rack, false, w, l, 0.0), rack);
            // Half is half way there.
            let half = ackermann_deg(rack, true, w, l, 0.5);
            assert!((half - (rack + inner) / 2.0).abs() < 1e-9);
        }
        // It grows with lock — at a standstill's full lock the difference is
        // worth degrees, and on a motorway it is worth nothing.
        let small = ackermann_deg(2.0, true, w, l, 1.0) - 2.0;
        let large = ackermann_deg(35.0, true, w, l, 1.0) - 35.0;
        assert!(large > small * 10.0, "{small} vs {large}");
        // Degenerate geometry is a refusal, not a division.
        assert_eq!(ackermann_deg(10.0, true, w, 0.0, 1.0), 10.0);
        assert!(ackermann_deg(179.0, true, 4.0, 1.0, 1.0).is_finite());
        // …and the rig reads it: the two front wheels take different angles.
        let mut v = grounded(&[]);
        v.control(VehicleControls {
            steer: 1.0,
            ..Default::default()
        });
        let mut out = Vec::new();
        for _ in 0..60 {
            out.clear();
            v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
        }
        let s = v.wheels();
        assert!(
            (s[0].steer_deg - s[1].steer_deg).abs() > 0.5,
            "the two front wheels took {}° and {}° — the rig is steering parallel",
            s[0].steer_deg,
            s[1].steer_deg
        );
        assert_eq!(s[2].steer_deg, 0.0, "a rear wheel steered");
    }

    /// **The anti-roll bar moves load across an axle and adds none** — and with a
    /// load-sensitive tyre, moving it costs that axle grip.
    #[test]
    fn an_anti_roll_bar_transfers_load_without_adding_any() {
        let loads = |rate: f64| -> (f64, f64, f64) {
            let mut v = grounded(&[("anti_roll_front_n_per_m", rate)]);
            // Roll the rig: the left front is compressed further than the right.
            let t = *v.tuning();
            v.wheels_mut()[0].contact = Some(WheelContact {
                point: DVec3::new(-0.9, -1.0, 1.4),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.18 + 0.35,
            });
            v.wheels_mut()[1].contact = Some(WheelContact {
                point: DVec3::new(0.9, -1.0, 1.4),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.04 + 0.35,
            });
            let mut out = Vec::new();
            v.solve(resting(1_200.0), 1.0 / 60.0, &mut out);
            let w = v.wheels();
            (w[0].load_n, w[1].load_n, v.axle_load_n(true))
        };
        let (bare_l, bare_r, bare_axle) = loads(0.0);
        let (bar_l, bar_r, bar_axle) = loads(8_000.0);
        assert!(
            bar_l > bare_l && bar_r < bare_r + 1e-9 && bar_r < bare_r,
            "the bar did not transfer: {bare_l}/{bare_r} became {bar_l}/{bar_r}"
        );
        assert!(
            (bar_axle - bare_axle).abs() < 1e-9,
            "the bar ADDED {:.1} N to the axle — a bar transfers load, it does \
             not make any",
            bar_axle - bare_axle
        );
        // …and the transfer costs grip, which is the only reason the knob is
        // worth turning. Read through the µ the two loads actually get.
        let stat = 1_200.0 * 9.81 / 4.0;
        let grip = |a: f64, b: f64| {
            load_sensitive_mu(1.2, a, stat, 0.22) * a + load_sensitive_mu(1.2, b, stat, 0.22) * b
        };
        assert!(
            grip(bar_l, bar_r) < grip(bare_l, bare_r),
            "stiffening the bar cost the axle no grip at all"
        );
        // …and a bar stiff enough to lift the inside wheel does NOT pull it
        // down. That IS load the axle gains, and it is gained by the suspension's
        // own rule — a strut that pulled the chassis toward the road would suck
        // the car onto it — rather than by the bar being wrong.
        let (heavy_l, heavy_r, _) = loads(40_000.0);
        assert_eq!(heavy_r, 0.0, "the unloaded wheel took {heavy_r} N");
        assert!(heavy_l > bar_l);
    }

    /// **A high centre of gravity rolls the car harder** — the same tyre force,
    /// a longer moment arm, and nothing else changed.
    #[test]
    fn the_centre_of_gravity_is_where_the_tyre_force_pulls_from() {
        let moment = |cog: f64| -> f64 {
            let mut v = grounded(&[("cog_height_m", cog)]);
            let mut chassis = resting(1_200.0);
            // Sliding sideways, so there is a lateral force to take a moment of.
            chassis.linvel = DVec3::new(4.0, 0.0, 10.0);
            let mut out = Vec::new();
            v.solve(chassis, 1.0 / 60.0, &mut out);
            // Roll moment about the body's own centre of mass, from the forces
            // the model asked for.
            out.iter()
                .map(|f| {
                    let r = f.point - chassis.position;
                    r.cross(f.force).z
                })
                .sum()
        };
        let (low, high) = (moment(-0.45), moment(0.35));
        assert!(
            high.abs() > low.abs() * 1.2,
            "a centre of gravity 0.8 m higher produced a roll moment of {high:.0} \
             N·m against {low:.0} — the height reaches no force"
        );
        assert_eq!(
            high.signum(),
            low.signum(),
            "raising the centre of gravity reversed the roll"
        );
    }

    /// **Downforce presses, drag is anisotropic, and a slide meets more air than
    /// a drive.**
    #[test]
    fn the_air_pushes_down_and_resists_sideways_harder_than_forwards() {
        let aero = |vel: DVec3, df: f64| -> (DVec3, f64) {
            let mut v = grounded(&[
                ("downforce_n_per_mps2", df),
                // No engine, no brake: the only forces are the springs, the
                // tyres and the air, and the springs are vertical.
                ("peak_torque_nm", 0.0),
            ]);
            let mut chassis = resting(1_200.0);
            chassis.linvel = vel;
            let mut out = Vec::new();
            v.solve(chassis, 1.0 / 60.0, &mut out);
            // The aero forces are the ones NOT at a contact point.
            let mut f = DVec3::ZERO;
            let mut down = 0.0;
            for w in out.iter().filter(|w| w.point.y > -0.5) {
                f += w.force;
                down += -w.force.y;
            }
            (f, down)
        };
        let fast = 30.0;
        let (fwd_drag, _) = aero(DVec3::new(0.0, 0.0, fast), 0.0);
        let (side_drag, _) = aero(DVec3::new(fast, 0.0, 0.0), 0.0);
        assert!(
            side_drag.length() > fwd_drag.length() * 2.0,
            "sideways at {fast} m/s met {:.0} N of air and forwards met {:.0} — \
             one isotropic coefficient is exactly the model VEH2a replaced",
            side_drag.length(),
            fwd_drag.length()
        );
        assert!(
            fwd_drag.z < 0.0 && side_drag.x < 0.0,
            "the air pushed ALONG"
        );
        // Downforce is quadratic and it presses DOWN.
        let (_, none) = aero(DVec3::new(0.0, 0.0, fast), 0.0);
        let (_, some) = aero(DVec3::new(0.0, 0.0, fast), 0.5);
        let (_, half) = aero(DVec3::new(0.0, 0.0, fast / 2.0), 0.5);
        assert!(
            none.abs() < 1e-9,
            "there is downforce with the wing switched off"
        );
        assert!((some - 0.5 * fast * fast).abs() < 1e-6, "{some}");
        assert!(
            (some / half - 4.0).abs() < 1e-6,
            "downforce is not quadratic"
        );
    }

    /// **`drivetrain = "awd"` is a spelling of one number**, and the number wins.
    #[test]
    fn a_catalogue_row_may_spell_its_drivetrain() {
        let mut defs = VehicleDefs::default();
        defs.merge_toml(
            "[a.vehicle]\ndrivetrain = \"fwd\"\n\
             [b.vehicle]\ndrivetrain = \"rwd\"\n\
             [c.vehicle]\ndrivetrain = \"awd\"\n\
             [d.vehicle]\ndrivetrain = \"awd\"\nfront_torque_split = 0.15\n",
        )
        .expect("the rows parse");
        assert_eq!(defs.get("a").unwrap().class.front_torque_split, 1.0);
        assert_eq!(defs.get("b").unwrap().class.front_torque_split, 0.0);
        assert_eq!(
            defs.get("c").unwrap().class.front_torque_split,
            AWD_FRONT_SPLIT
        );
        assert_eq!(
            defs.get("d").unwrap().class.front_torque_split,
            0.15,
            "an explicit split must win over the word, whatever order the TOML \
             map iterated in"
        );
        // …and an unknown word is a refusal BY NAME, not a silent rear-drive.
        let err = VehicleDefs::default()
            .merge_toml("[x.vehicle]\ndrivetrain = \"tracks\"\n")
            .expect_err("an unknown drivetrain is refused");
        assert!(err.contains("tracks"), "{err}");
        assert!(VehicleDefs::default()
            .merge_toml("[x.vehicle]\ndrivetrain = 3\n")
            .is_err());
    }

    /// **ABS keeps a braked wheel turning, and it stops the car sooner for it.**
    ///
    /// Both halves matter. An ABS that merely stopped the wheel locking while
    /// lengthening the stop would be a worse brake with a better graph, and that
    /// is exactly what a naive one does.
    #[test]
    fn abs_keeps_the_wheel_out_of_lockup_and_stops_shorter_for_it() {
        let stop = |abs: f64| -> (f64, f64, f64) {
            let mut v = grounded(&[("abs_slip", abs)]);
            let mut chassis = resting(1_200.0);
            chassis.linvel = DVec3::new(0.0, 0.0, 25.0);
            for w in v.wheels_mut() {
                w.omega_rad_s = 25.0 / 0.35;
            }
            v.control(VehicleControls {
                brake: 1.0,
                ..Default::default()
            });
            let mut out = Vec::new();
            let (mut travelled, mut locked) = (0.0, 0usize);
            // Sampled MID-STOP, not at the end: by step 180 both cars are
            // stationary and both wheels read zero, so a comparison there says
            // nothing at all.
            let mut mid = 0.0;
            for step in 0..180 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
                let fz: f64 = out.iter().map(|f| f.force.z).sum();
                chassis.linvel.z = (chassis.linvel.z + fz / 1_200.0 / 60.0).max(0.0);
                travelled += chassis.linvel.z / 60.0;
                if step == 30 {
                    mid = v.wheels()[0].omega_rad_s;
                }
                // "Locked" as a COUNT of steps rather than a worst case: a real
                // anti-lock system PULSES, so its worst single step is a locked
                // wheel too, and only the share of the stop it spends there says
                // anything at all.
                if v.wheels()[0].slip_ratio < -0.8 {
                    locked += 1;
                }
            }
            (travelled, locked as f64 / 180.0, mid)
        };
        let (locked_far, locked_share, locked_omega) = stop(0.0);
        let (abs_far, abs_share, abs_omega) = stop(0.15);
        assert!(
            locked_share > 0.5 && locked_omega < 1.0,
            "with ABS off the wheel spent {locked_share:.2} of the stop locked \
             and ended at {locked_omega} rad/s"
        );
        assert!(
            abs_share < locked_share * 0.5,
            "ABS spent {abs_share:.2} of the stop locked against {locked_share:.2} \
             unaided"
        );
        assert!(
            abs_omega > locked_omega + 5.0,
            "ABS left the wheel at {abs_omega} rad/s against a locked {locked_omega}"
        );
        assert!(
            abs_far < locked_far * 0.95,
            "ABS stopped in {abs_far:.2} m against a locked {locked_far:.2} — an \
             anti-lock that lengthens the stop is a worse brake with a better graph"
        );
    }

    /// **Traction control takes torque off a spinning wheel**, and a slippery
    /// standing start goes further with it than without.
    #[test]
    fn traction_control_cuts_the_torque_a_spinning_wheel_is_given() {
        let launch = |tc: f64| -> (f64, f64) {
            let mut v = grounded(&[
                ("traction_control_slip", tc),
                ("longitudinal_grip", 0.35),
                ("lateral_grip", 0.35),
            ]);
            v.control(VehicleControls {
                throttle: 1.0,
                ..Default::default()
            });
            let mut chassis = resting(1_200.0);
            let mut out = Vec::new();
            let (mut travelled, mut worst) = (0.0, 0.0f64);
            for _ in 0..120 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
                let fz: f64 = out.iter().map(|f| f.force.z).sum();
                chassis.linvel.z += fz / 1_200.0 / 60.0;
                travelled += chassis.linvel.z / 60.0;
                worst = worst.max(v.wheels()[0].slip_ratio);
            }
            (travelled, worst)
        };
        let (free_far, free_slip) = launch(0.0);
        let (tc_far, tc_slip) = launch(0.12);
        assert!(
            free_slip > 0.5,
            "the fixture never span its wheels: {free_slip}"
        );
        assert!(
            tc_slip < free_slip * 0.5,
            "traction control let the slip reach {tc_slip} against {free_slip} \
             unaided"
        );
        assert!(
            tc_far > free_far,
            "traction control carried the car {tc_far:.2} m against {free_far:.2} \
             — an aid that costs distance on a slippery launch is not one"
        );
        // …and it costs nothing at all when the tyres are gripping: an aid that
        // is always taking torque away is a smaller engine.
        let dry = |tc: f64| -> f64 {
            let mut v = grounded(&[("traction_control_slip", tc)]);
            v.control(VehicleControls {
                throttle: 1.0,
                ..Default::default()
            });
            let mut chassis = resting(1_200.0);
            let mut out = Vec::new();
            let mut travelled = 0.0;
            for _ in 0..120 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
                let fz: f64 = out.iter().map(|f| f.force.z).sum();
                chassis.linvel.z += fz / 1_200.0 / 60.0;
                travelled += chassis.linvel.z / 60.0;
            }
            travelled
        };
        assert!(
            (dry(0.12) - dry(0.0)).abs() < dry(0.0) * 0.02,
            "traction control cost a GRIPPING car {:.3} m of {:.3}",
            dry(0.0) - dry(0.12),
            dry(0.0)
        );
    }

    /// **THE AID DOES NOT CHATTER** (`audit:` VEH2a) — the arm the
    /// feed-forward ruling did not have.
    ///
    /// [`aid_torque_cap_nm`]'s own doc records why the first two cuts were
    /// refused: a wheel under 4 kN·m changes speed by 66 rad/s in a single 60 Hz
    /// step, so a controller reading last step's slip is always a step behind an
    /// event that is over inside one step, and `tc / slip` applied directly
    /// oscillated the crank between 416 N·m and 109. That is a measurement in a
    /// comment; **nothing in the tree asserted it**. The arms above measure what
    /// the aid is WORTH (less slip, more distance, nothing lost when dry), and
    /// an oscillating controller can satisfy all three — the refused one did,
    /// and paid for it in 8.7 s to 100 km/h, which only the feel table would
    /// have caught, by 0.3 s.
    ///
    /// So the claim is the shape of the cut itself. `WheelState::tc_cut` is the
    /// share of the requested torque the wheel was actually handed, and under a
    /// feed-forward cap it is a function of the LOAD, which moves over tens of
    /// milliseconds. Under a controller fed by the slip it alternates between
    /// "no cut at all" (the wheel stuck, so the error is zero) and "almost
    /// everything" (the wheel spinning) on consecutive steps. Both states in one
    /// pinned-throttle launch is the signature, and it is what this refuses.
    #[test]
    fn the_traction_control_cut_settles_instead_of_alternating() {
        let mut v = grounded(&[
            ("traction_control_slip", 0.12),
            ("longitudinal_grip", 0.35),
            ("lateral_grip", 0.35),
        ]);
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let mut chassis = resting(1_200.0);
        let mut out = Vec::new();
        let mut cuts: Vec<f64> = Vec::new();
        for _ in 0..120 {
            out.clear();
            v.solve(chassis, 1.0 / 60.0, &mut out);
            let fz: f64 = out.iter().map(|f| f.force.z).sum();
            chassis.linvel.z += fz / 1_200.0 / 60.0;
            cuts.push(v.wheels()[0].tc_cut);
        }
        // Past the first ten steps the suspension has settled and the throttle
        // has been pinned throughout, so every remaining step is the same
        // question asked again.
        let settled = &cuts[10..];
        let uncut = settled.iter().filter(|c| **c > 0.95).count();
        let cutting = settled.iter().filter(|c| **c < 0.9).count();
        let (lo, hi) = settled
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), c| (l.min(*c), h.max(*c)));
        let worst_jump = settled
            .windows(2)
            .fold(0.0f64, |m, w| m.max((w[1] - w[0]).abs()));
        println!(
            "THE AID'S CUT: {cutting} of {} settled steps cut, {uncut} uncut \
             (>0.95), range {lo:.4}..{hi:.4}, worst step-to-step change \
             {worst_jump:.4}",
            settled.len()
        );
        // It really is cutting, or the arm is about a controller that never ran.
        assert!(
            cutting > settled.len() / 2,
            "traction control cut on only {cutting} of {} steps — the fixture is \
             not spinning its wheels and this arm measures nothing",
            settled.len()
        );
        // …and it never lets go: a step at full torque in the middle of a run
        // that is otherwise cutting is the alternation a feed-forward cap does
        // not have.
        assert!(
            uncut < 3,
            "the cut was released entirely on {uncut} of {} settled steps while \
             cutting on {cutting} — that is a controller alternating between a \
             stuck wheel and a spinning one, which is exactly the one-step race \
             `aid_torque_cap_nm` refuses to enter",
            settled.len()
        );
        // **The shape claim.** A cap computed from the LOAD moves at the speed a
        // load moves; a cap computed from last step's slip moves at the speed a
        // wheel spins up, which is 66 rad/s in one step. Measured at **0.0020**
        // per step here, so a fiftyfold margin is still a claim.
        assert!(
            worst_jump < 0.1,
            "the cut moved by {worst_jump:.3} in one step; a cap on the LOAD \
             cannot move that fast, so this is a cap on last step's slip"
        );
    }

    /// **Stability control brakes ONE wheel, and it is the right one.**
    ///
    /// Oversteer takes the outside front; understeer takes the inside rear. Read
    /// as the difference in ground force between the two candidates, which is
    /// what a claim about "which wheel" has to be.
    #[test]
    fn stability_control_brakes_the_outside_front_when_the_car_is_loose() {
        // Steering RIGHT (positive), so the outside is the left pair (x < 0) and
        // the inside is the right pair (x > 0). The fixture's wheels are
        // (-x, +z), (+x, +z), (-x, -z), (+x, -z).
        let run = |yaw: f64, strength: f64| -> [f64; 4] {
            let mut v = grounded(&[
                ("stability_control", strength),
                ("steer_rate_deg_per_s", 1e9),
            ]);
            let mut chassis = resting(1_200.0);
            chassis.linvel = DVec3::new(0.0, 0.0, 20.0);
            chassis.angvel = DVec3::new(0.0, yaw, 0.0);
            for w in v.wheels_mut() {
                w.omega_rad_s = 20.0 / 0.35;
            }
            v.control(VehicleControls {
                steer: 1.0,
                ..Default::default()
            });
            let mut out = Vec::new();
            for _ in 0..3 {
                out.clear();
                v.solve(chassis, 1.0 / 60.0, &mut out);
            }
            let mut omega = [0.0; 4];
            for (i, w) in v.wheels().iter().enumerate() {
                omega[i] = w.omega_rad_s;
            }
            omega
        };
        // A right turn's reference yaw rate is NEGATIVE. Spinning faster than
        // that (more negative) is oversteer.
        // The reference yaw rate this fixture's own steering asks for, so the two
        // cases below are genuinely oversteer and understeer rather than two
        // numbers that happened to work.
        let reference =
            -20.0 * steer_limit_deg(&VehicleTuning::default(), 20.0).to_radians() / (2.0 * 1.4);
        let loose = run(reference * 1.8, 1.0);
        let off = run(reference * 1.8, 0.0);
        assert!(
            loose[0] < off[0] - 1.0,
            "the outside FRONT wheel was not braked: {} against {}",
            loose[0],
            off[0]
        );
        for i in [1, 2, 3] {
            assert!(
                (loose[i] - off[i]).abs() < 1e-9,
                "stability control also braked wheel {i} ({} vs {})",
                loose[i],
                off[i]
            );
        }
        // Understeer — barely rotating into a turn the steering asked for — takes
        // the INSIDE REAR instead.
        let push = run(0.0, 1.0);
        let push_off = run(0.0, 0.0);
        assert!(
            push[3] < push_off[3] - 1.0,
            "the inside REAR wheel was not braked: {} against {}",
            push[3],
            push_off[3]
        );
        assert!((push[0] - push_off[0]).abs() < 1e-9);
        // …and inside the tolerance it does nothing at all.
        let calm = run(reference, 1.0);
        let calm_off = run(reference, 0.0);
        assert_eq!(
            calm, calm_off,
            "stability control fired inside its dead band"
        );
    }

    /// **A wheel is a wheel now**: `ω` is a state, the slip ratio is defined
    /// against it, and the drawn spin is DERIVED from it rather than from how
    /// fast the car is going.
    ///
    /// The clause-1 arm, written against the mutation that matters: a `spin_deg`
    /// still integrated from `along_v / r` would pass any check that only asks
    /// "does the wheel turn", and would show a locked wheel rolling.
    #[test]
    fn the_wheel_has_its_own_speed_and_the_drawn_spin_follows_it() {
        let mut v = grounded(&[]);
        let t = *v.tuning();
        for w in v.wheels_mut() {
            w.contact = Some(WheelContact {
                point: DVec3::new(0.0, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.1 + 0.35,
            });
        }
        let mut chassis = resting(1_200.0);
        chassis.linvel = DVec3::new(0.0, 0.0, 10.0);
        let mut out = Vec::new();

        // Free rolling: ω converges on v/r and the slip goes to nothing.
        for _ in 0..120 {
            out.clear();
            v.solve(chassis, 1.0 / 60.0, &mut out);
        }
        let free = 10.0 / 0.35;
        // EXACTLY rolling, not nearly: a sticking wheel's resist budget is paid
        // inside the stick force the ground supplies, so the wheel really is held
        // at the road's speed rather than dragged a brake-step below it.
        assert!(
            (v.wheels()[0].omega_rad_s - free).abs() < 1e-9,
            "a free-rolling wheel settled at {} rad/s against {free}",
            v.wheels()[0].omega_rad_s
        );
        assert_eq!(
            v.wheels()[0].slip_ratio,
            0.0,
            "a wheel the ground is holding at rolling speed is not slipping"
        );
        // …and the force it takes is exactly the engine braking and the rolling
        // resistance divided by the radius — the equilibrium, not a transient.
        let drag = out
            .iter()
            .map(|f| f.force.z)
            .filter(|z| z.abs() > 1e-9)
            .sum::<f64>();
        assert!(
            drag < 0.0 && drag > -1_200.0,
            "a coasting car's total drag is {drag} N, which is a brake"
        );
        let before = v.wheels()[0].spin_deg;
        out.clear();
        v.solve(chassis, 1.0 / 60.0, &mut out);
        assert_ne!(v.wheels()[0].spin_deg, before, "the drawn wheel is frozen");

        // **LOCKUP**: full brake, and the wheel stops turning while the car does
        // not — which is the state P29.7 could not represent at all. ABS off,
        // because an ABS that let a wheel lock would not be an ABS.
        v.tune("abs_slip", 0.0);
        v.control(VehicleControls {
            brake: 1.0,
            ..Default::default()
        });
        for _ in 0..30 {
            out.clear();
            v.solve(chassis, 1.0 / 60.0, &mut out);
        }
        let w = v.wheels()[0];
        assert!(
            w.omega_rad_s < free * 0.5,
            "the braked wheel is still turning at {} of a free {free}",
            w.omega_rad_s
        );
        assert!(
            w.slip_ratio < -0.1,
            "a locked wheel under a moving car has slip {}",
            w.slip_ratio
        );
        let locked = v.wheels()[0].spin_deg;
        out.clear();
        v.solve(chassis, 1.0 / 60.0, &mut out);
        assert!(
            (v.wheels()[0].spin_deg - locked).abs() < 3.0,
            "a locked wheel turned {:.2}° in one step — the drawn spin is still \
             coming from the road speed",
            v.wheels()[0].spin_deg - locked
        );

        // **WHEELSPIN**: full throttle from a standstill on a LOW-GRIP surface,
        // and ω outruns the road.
        //
        // Low grip on purpose, and the reason is itself a claim worth recording:
        // the default rig on dry tarmac does **not** spin its wheels off the
        // line — 1 852 N·m of wheel torque is 5.3 kN of thrust against 10.3 kN of
        // grip, so the tyres simply hold and the car goes. A model in which every
        // standing start is a burnout is a model whose grip is decorative.
        let mut v = RaycastVehicle::new(rig(4));
        v.tune("longitudinal_grip", 0.3);
        v.tune("lateral_grip", 0.3);
        v.tune("traction_control_slip", 0.0);
        for w in v.wheels_mut() {
            w.contact = Some(WheelContact {
                point: DVec3::new(0.0, -1.0, 0.0),
                normal: DVec3::Y,
                distance_m: t.rest_length_m - 0.1 + 0.35,
            });
        }
        v.control(VehicleControls {
            throttle: 1.0,
            ..Default::default()
        });
        let still = resting(1_200.0);
        for _ in 0..5 {
            out.clear();
            v.solve(still, 1.0 / 60.0, &mut out);
        }
        assert!(
            v.wheels()[0].slip_ratio > 0.2,
            "a standing start produced slip of only {}",
            v.wheels()[0].slip_ratio
        );
        assert!(v.wheels()[0].omega_rad_s > 0.0);
    }
}
