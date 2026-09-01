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
    pub const ALL: [VehicleBody; 5] = [
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
        if self.rig.wheels.is_empty() || !(mass_kg > 0.0) {
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
/// the redline the argument is clamped, which is the limiter.
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
    if !(top > 0.0) || !forward_mps.is_finite() {
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
    if !(load_n.is_finite() && sensitivity.is_finite()) || !(static_load_n > 0.0) {
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
    if a <= 0.0 || !(wheelbase_m > 0.0) || !rack_deg.is_finite() || !(half_track_m > 0.0) {
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
        // one is limited by the fuel cut above and by the contact patch; clamping
        // its speed clamps the force it can transmit, which is the defect the
        // limiter's own note records.
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
