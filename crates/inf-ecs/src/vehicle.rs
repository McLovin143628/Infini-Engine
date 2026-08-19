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
    /// **The window of that choreography the root motion is warped over**
    /// (`inf_anim::WarpWindow`): before it the character is still walking, after
    /// it the character is seated and the last of the clip plays out.
    ///
    /// A window rather than the whole clip is the point — see
    /// `inf_physics::d3::vehicle`'s enter/exit section, and the P29.4/P29.5/P29.6
    /// ledgers, where this type is named three times as the one warp shape with
    /// no consumer.
    pub enter_window: inf_anim::WarpWindow,
}

impl Default for VehicleTuning {
    fn default() -> Self {
        Self {
            rest_length_m: 0.5,
            travel_m: 0.25,
            stiffness_n_per_m: 20_000.0,
            damping_ns_per_m: 3_000.0,
            max_engine_force_n: 8_000.0,
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
            enter_window: inf_anim::WarpWindow::new(0.1, 0.45),
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
            "rest_length_m" => &mut self.rest_length_m,
            "travel_m" => &mut self.travel_m,
            "stiffness_n_per_m" => &mut self.stiffness_n_per_m,
            "damping_ns_per_m" => &mut self.damping_ns_per_m,
            "max_engine_force_n" => &mut self.max_engine_force_n,
            "max_speed_mps" => &mut self.max_speed_mps,
            "brake_force_n" => &mut self.brake_force_n,
            "handbrake_force_n" => &mut self.handbrake_force_n,
            "max_steer_deg" => &mut self.max_steer_deg,
            "min_steer_deg" => &mut self.min_steer_deg,
            "lateral_grip" => &mut self.lateral_grip,
            "longitudinal_grip" => &mut self.longitudinal_grip,
            "rolling_resistance" => &mut self.rolling_resistance,
            "drag_n_per_mps2" => &mut self.drag_n_per_mps2,
            "enter_time_s" => &mut self.enter_time_s,
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
            "brake_force_n",
            "damping_ns_per_m",
            "drag_n_per_mps2",
            "enter_time_s",
            "handbrake_force_n",
            "lateral_grip",
            "longitudinal_grip",
            "max_engine_force_n",
            "max_speed_mps",
            "max_steer_deg",
            "min_steer_deg",
            "rest_length_m",
            "rolling_resistance",
            "stiffness_n_per_m",
            "travel_m",
        ]
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
        (self.tuning.enter_time_s, self.tuning.enter_window)
    }

    fn suspension_rest_m(&self) -> f64 {
        self.tuning.rest_length_m
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
            15,
            "a name added to the door and not to the list is invisible to a UI"
        );
        let mut sorted = VehicleTuning::names().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, VehicleTuning::names());
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
