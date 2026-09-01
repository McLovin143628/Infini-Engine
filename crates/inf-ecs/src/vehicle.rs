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

/// A vehicle's geometry: its chassis and its wheels, in `Guid` order.
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
pub fn rig_of(world: &EcsWorld, chassis: Uuid) -> Option<VehicleRig> {
    let entity = world.entity_of(chassis)?;
    let w = world.world();
    let seat_local = chassis_of(w.get::<Collider3D>(entity), w.get::<RigidBody3D>(entity))?;
    let mut wheels: Vec<WheelMount> = Vec::new();
    for child in world.children_of(entity) {
        let Some(guid) = world.guid_of(child) else {
            continue;
        };
        let Some(radius_m) = wheel_of(w.get::<Collider3D>(child), w.get::<RigidBody3D>(child))
        else {
            continue;
        };
        let mount_local = w
            .get::<Transform>(child)
            .map(|t| t.translation)
            .unwrap_or(Vec3d::ZERO);
        wheels.push(WheelMount {
            guid,
            mount_local,
            radius_m,
        });
    }
    if wheels.is_empty() {
        return None;
    }
    wheels.sort_unstable_by_key(|wm| wm.guid);
    Some(VehicleRig {
        chassis,
        seat_local,
        wheels,
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
            // The class's REFERENCE top speed, m/s — what the speed-sensitive
            // steering limit and the drive camera scale against. Since VEH2a the
            // *achieved* top speed is an emergent balance of the torque curve
            // against the drag, so this number is a claim about the car that
            // `the_default_rig_reaches_the_speed_its_own_class_claims` checks.
            max_speed_mps: 55.0,
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

impl VehicleBody {
    /// Every family, in the canonical order.
    pub const ALL: [VehicleBody; 2] = [VehicleBody::Sedan, VehicleBody::Truck];

    /// The stable name a catalogue row names this family by.
    pub fn name(self) -> &'static str {
        match self {
            VehicleBody::Sedan => "sedan",
            VehicleBody::Truck => "truck",
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
        for (k, v) in table {
            if k == "body" {
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
    /// Rolling angle, degrees, integrated from the forward speed and the radius
    /// so a wheel visibly turns at the right rate. Wrapped to `[0, 360)`.
    pub spin_deg: f64,
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

// ── the implementation this phase ships ─────────────────────────────────────

/// The raycast vehicle: a spring, a damper and a friction circle per wheel.
#[derive(Clone, Debug)]
pub struct RaycastVehicle {
    rig: VehicleRig,
    tuning: VehicleTuning,
    controls: VehicleControls,
    wheels: Vec<WheelState>,
}

impl RaycastVehicle {
    /// Build one over a derived rig, with the default tuning.
    pub fn new(rig: VehicleRig) -> Self {
        let wheels = vec![
            WheelState {
                length_m: VehicleTuning::default().rest_length_m,
                ..Default::default()
            };
            rig.wheels.len()
        ];
        Self {
            rig,
            tuning: VehicleTuning::default(),
            controls: VehicleControls::default(),
            wheels,
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
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let k = if k.is_finite() { k.clamp(0.05, 0.95) } else { 0.5 };
    let a = 1.0 / k - 2.0;
    let denom = a * (1.0 - t) + 1.0;
    if denom.abs() < 1e-12 {
        return t;
    }
    (t / denom).clamp(0.0, 1.0)
}

/// **The engine curve**: drive force as a function of throttle and forward speed.
///
/// Linear falloff to zero at [`VehicleTuning::max_speed_mps`], which is what
/// makes that field the top speed on the flat rather than a number that has to
/// be balanced against the drag. Reverse gets a third of the force, which is
/// every road car.
pub fn engine_force_n(tuning: &VehicleTuning, throttle: f64, forward_mps: f64) -> f64 {
    let throttle = throttle.clamp(-1.0, 1.0);
    if throttle == 0.0 || tuning.max_speed_mps <= 0.0 {
        return 0.0;
    }
    let reverse_scale = if throttle < 0.0 { 1.0 / 3.0 } else { 1.0 };
    // The falloff is against speed IN THE DIRECTION OF THE REQUEST, so full
    // force is available when reversing at 20 m/s forward — which is a brake,
    // and is exactly the case `VehicleControls::from_intent` routes elsewhere.
    let along = forward_mps * throttle.signum();
    let headroom = (1.0 - along / tuning.max_speed_mps).clamp(0.0, 1.0);
    tuning.max_engine_force_n * throttle * headroom * reverse_scale
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
    fn rig(&self) -> &VehicleRig {
        &self.rig
    }

    fn set_rig(&mut self, rig: VehicleRig) {
        // The wheel STATE survives a re-derive whose wheel count did not change:
        // a suspension that reset to full extension every time the scene changed
        // would make an editor edit a bounce.
        if rig.wheels.len() != self.wheels.len() {
            self.wheels = vec![
                WheelState {
                    length_m: self.tuning.rest_length_m,
                    ..Default::default()
                };
                rig.wheels.len()
            ];
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

    /// Revs are speed against this class's own top speed; load is the throttle,
    /// unsigned, because reversing is not quieter.
    ///
    /// The brake is deliberately not in it: a car braking hard from its top
    /// speed still has an engine turning over, and folding the brake in would
    /// make the loudest moment of a drive the one where the pedal comes off.
    fn engine_state(&self, forward_mps: f64) -> (f64, f64) {
        let revs = if self.tuning.max_speed_mps > 0.0 {
            (forward_mps.abs() / self.tuning.max_speed_mps).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (revs, self.controls.throttle.abs().clamp(0.0, 1.0))
    }

    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let (fwd, right, up) = chassis.basis();
        let forward_mps = chassis.linvel.dot(fwd);
        let speed = chassis.linvel.length();
        let limit = steer_limit_deg(&self.tuning, forward_mps);
        let steer_deg = self.controls.steer.clamp(-1.0, 1.0) * limit;
        let grounded = self.wheels.iter().filter(|w| w.contact.is_some()).count();
        // The drive and brake budgets are shared over the wheels that can use
        // them: a car with one wheel on the ground does not get four wheels of
        // push, which is how a raycast vehicle climbs walls.
        let drive_total = engine_force_n(&self.tuning, self.controls.throttle, forward_mps);
        let brake_total = self.tuning.brake_force_n * self.controls.brake.clamp(0.0, 1.0);
        let per_wheel = if grounded == 0 {
            0.0
        } else {
            1.0 / grounded as f64
        };
        // The quarter-mass the lateral cancel is sized against. Wheel count, not
        // grounded count: a wheel in the air carries no mass to cancel.
        let share = if self.rig.wheels.is_empty() {
            0.0
        } else {
            chassis.mass_kg / self.rig.wheels.len() as f64
        };

        for (i, mount) in self.rig.wheels.iter().enumerate() {
            let Some(state) = self.wheels.get_mut(i) else {
                break;
            };
            let steer = if mount.steered() { steer_deg } else { 0.0 };
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

            let Some(contact) = state.contact else {
                // In the air the suspension extends and the wheel free-wheels.
                state.length_m = self.tuning.rest_length_m;
                state.load_n = 0.0;
                continue;
            };

            let previous = state.length_m;
            let length = (contact.distance_m - mount.radius_m).clamp(
                self.tuning.rest_length_m - self.tuning.travel_m,
                self.tuning.rest_length_m,
            );
            state.length_m = length;
            let compression = self.tuning.rest_length_m - length;
            // Closing speed from the CONTACT POINT's velocity rather than from
            // the length difference: a finite difference over one step is a step
            // behind and rings at exactly the frequency the damper exists to
            // kill. `previous` is kept for the visual only.
            let _ = previous;
            let point_vel = chassis.point_velocity(contact.point);
            let closing = -point_vel.dot(up);
            let load = suspension_force_n(&self.tuning, compression, closing);
            state.load_n = load;
            // The spring pushes along the SUSPENSION axis (the chassis up), not
            // along the contact normal: a raycast wheel on a slope is still held
            // up by its own strut, and projecting onto the normal is how a car
            // slides sideways off a ramp it should drive up.
            out.push(WheelForce {
                point: contact.point,
                force: up * load,
            });

            // ── the friction circle ──────────────────────────────────────────
            //
            // On the TYRE's velocity, not the suspension's — see
            // `ChassisState::contact_velocity` for the pitch pump this closes.
            let tyre_vel = chassis.contact_velocity(contact.point, up);
            let mu_lat = load * self.tuning.lateral_grip;
            let mu_long = load * self.tuning.longitudinal_grip;
            let side_v = tyre_vel.dot(wheel_right);
            // The force that would cancel the slip exactly in one step, clamped
            // by what the tyre can hold. The clamp is what makes this stable at
            // any timestep: an uncancellable slip becomes a slide, not a spike.
            let lateral = (-share * side_v / dt).clamp(-mu_lat, mu_lat);
            out.push(WheelForce {
                point: contact.point,
                force: wheel_right * lateral,
            });

            let along_v = tyre_vel.dot(wheel_fwd);
            let mut longitudinal = drive_total * per_wheel;
            // **Everything that resists motion is one budget, and it may not
            // reverse the motion.** The most any of it can do in one step is
            // bring this wheel to a stop; a resistive force that overshot would
            // push the car backwards, and then forwards, for ever. Measured
            // before the clamp covered the rolling term too: a braked car crept
            // **5.8 cm backwards per second**, which is what a constant
            // opposing force does at a velocity smaller than one step of it.
            let hand = if self.controls.handbrake && !mount.steered() {
                self.tuning.handbrake_force_n
            } else {
                0.0
            };
            if along_v.abs() > 1e-9 {
                let resist = brake_total * per_wheel + hand + load * self.tuning.rolling_resistance;
                if resist > 0.0 {
                    let stop = (share * along_v / dt).abs();
                    longitudinal -= along_v.signum() * resist.min(stop);
                }
            }
            let longitudinal = longitudinal.clamp(-mu_long, mu_long);
            out.push(WheelForce {
                point: contact.point,
                force: wheel_fwd * longitudinal,
            });

            // The visual roll: the wheel turns at the speed it is travelling.
            if mount.radius_m > 0.0 {
                let deg = along_v / mount.radius_m * dt * 180.0 / std::f64::consts::PI;
                state.spin_deg = (state.spin_deg + deg).rem_euclid(360.0);
            }
        }

        // Aerodynamic drag, once, at the centre of mass — not per wheel, because
        // a car in the air has no wheels on the ground and still has air on it.
        if speed > 1e-6 && self.tuning.drag_n_per_mps2 > 0.0 {
            let drag =
                -chassis.linvel.normalize_or_zero() * self.tuning.drag_n_per_mps2 * speed * speed;
            out.push(WheelForce {
                point: chassis.position,
                force: drag,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    fn resting(mass: f64) -> ChassisState {
        ChassisState {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linvel: DVec3::ZERO,
            angvel: DVec3::ZERO,
            mass_kg: mass,
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

    /// The engine curve reaches zero at the top speed and never pushes backwards
    /// past it — a curve that went negative would make the top speed a wall the
    /// car bounces off.
    #[test]
    fn the_engine_curve_falls_to_zero_at_the_top_speed() {
        let t = VehicleTuning::default();
        assert_eq!(engine_force_n(&t, 1.0, 0.0), t.max_engine_force_n);
        assert!(engine_force_n(&t, 1.0, t.max_speed_mps * 0.5) > 0.0);
        assert_eq!(engine_force_n(&t, 1.0, t.max_speed_mps), 0.0);
        assert_eq!(
            engine_force_n(&t, 1.0, t.max_speed_mps * 2.0),
            0.0,
            "past the top speed the engine stops pushing; it does not pull"
        );
        assert_eq!(engine_force_n(&t, 0.0, 0.0), 0.0);
        // Reverse is a third of the force and full at a standstill.
        assert!((engine_force_n(&t, -1.0, 0.0) + t.max_engine_force_n / 3.0).abs() < 1e-9);
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
            assert!(t.set(name, 0.5 + i as f64 / 1000.0), "{name} is not settable");
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
        let c = VehicleControls::from_intent(back, 12.0, false);
        assert_eq!((c.throttle, c.brake), (0.0, 1.0));
        let c = VehicleControls::from_intent(back, 0.0, false);
        assert_eq!((c.throttle, c.brake), (-1.0, 0.0));
        // …and symmetrically, forward while reversing is a brake.
        let fwd = crate::math::Vec2d::new(0.0, 1.0);
        let c = VehicleControls::from_intent(fwd, -12.0, false);
        assert_eq!((c.throttle, c.brake), (0.0, 1.0));
        let c = VehicleControls::from_intent(crate::math::Vec2d::new(1.0, 0.0), 0.0, true);
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
    #[test]
    fn the_engine_state_is_the_classs_own_and_defaults_to_silent() {
        let mut v = RaycastVehicle::new(rig(4));
        let top = v.tuning().max_speed_mps;
        assert_eq!(v.engine_state(0.0), (0.0, 0.0));
        v.control(VehicleControls {
            throttle: -1.0,
            ..Default::default()
        });
        let (revs, load) = v.engine_state(-top / 2.0);
        assert!(
            (revs - 0.5).abs() < 1e-12,
            "reversing at half speed is {revs}"
        );
        assert_eq!(load, 1.0, "reversing is not quieter");
        // Past the top speed it saturates rather than screaming.
        assert_eq!(v.engine_state(top * 3.0).0, 1.0);

        // The trait's default is silence: a class that has not thought about
        // sound makes none, rather than inheriting a road car's curve.
        struct Mute(VehicleRig);
        impl Vehicle for Mute {
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
    /// Asserted at the model, where a wheel's identity is visible: `solve`
    /// pushes three forces per grounded wheel in rig order (suspension, lateral,
    /// longitudinal), so wheel `i`'s longitudinal is `out[3i + 2]`, and the
    /// fixture's first two wheels are the front pair (`+Z`, therefore steered).
    /// Applied to all four, this arm would find the front pair braking too.
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
        }
        v.control(VehicleControls {
            handbrake: true,
            ..Default::default()
        });
        let mut chassis = resting(1_200.0);
        // Rolling forward, so there is something for a brake to resist.
        chassis.linvel = DVec3::new(0.0, 0.0, 8.0);
        let mut out = Vec::new();
        v.solve(chassis, 1.0 / 60.0, &mut out);

        let long = |i: usize| out[i * 3 + 2].force.z;
        let (front, rear) = ((long(0) + long(1)) * 0.5, (long(2) + long(3)) * 0.5);
        assert!(
            rear < -100.0,
            "a locked rear wheel must resist the motion; it pushed {rear} N"
        );
        assert!(
            front > rear * 0.2,
            "the front wheels braked {front} N against the rear's {rear} N — the \
             handbrake reached a steered wheel"
        );
        // …and the front pair is only rolling resistance, which is a small
        // fraction of the load rather than a brake.
        let load = v.wheels()[0].load_n;
        assert!(
            front.abs() < load * t.rolling_resistance * 1.5,
            "the front wheels carried {front} N, which is more than rolling \
             resistance on a {load} N load"
        );
    }
}
