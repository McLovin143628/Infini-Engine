//! **BOARDING** (wave VEH3d) — the eight sockets a car offers a body, and the
//! state machine that walks a body into one of them.
//!
//! # Two halves, and this is the one with no world in it
//!
//! Getting into a car is a choreography with a physics world on one end (a door
//! on a revolute joint, a hand that has to arrive within two centimetres of a
//! handle, a capsule that must not be left inside the bodywork) and pure
//! arithmetic on the other (where a seat IS, how long a phase lasts, what a
//! cubic Hermite through two tangents looks like). This module is the second
//! half: it holds the socket derivation, the phase table and the spline, and it
//! can be tested without a `PhysicsBridge3D`. `inf_physics::d3::boarding` is
//! the half that drives a door motor and publishes an IK goal.
//!
//! # THE SOCKETS ARE DERIVED, AND THAT IS A RULING (VEH3a's price, VEH3d's law)
//!
//! Wave VEH3a priced eight socket offsets as persisted `f64` on `VehicleClass`
//! and **refused them by name**: they are a function of the chassis's own
//! half-extents and of the family parts table that put a door on it, so
//! persisting them would be storing a derivation. Nothing in this module is on
//! the wire; [`sockets_of`] is a pure function of `(half_extents, offset,
//! parts)` and every consumer calls it rather than reading a field.
//!
//! The eight, in the chassis frame, metres:
//!
//! | socket | what it is |
//! |---|---|
//! | `seat_l` / `seat_r` | the two front cushions — the **H-point**, where a seated pelvis sits |
//! | `seat_rear` | the rear bench's centre cushion |
//! | `door_handle_l` / `door_handle_r` | the outer handles of the two FRONT doors |
//! | `pedal_throttle` / `pedal_brake` | the pedal faces, on the floor ahead of the driver |
//! | `wheel_hub` | the steering wheel's centre, ahead of the driver at its own rake |
//!
//! # WHICH SEAT IS THE DRIVER'S, and why it is `seat_r`
//!
//! `+X` has been the driver's side of a car in this engine since P29.7: the
//! exit places a driver at `rot * +X` ([`crate::vehicle`]'s
//! `EXIT_CLEARANCE_M` arithmetic) and wave VEH2b's carjack refuses an attempt
//! from anywhere but the `+X` half-space. So [`SeatIndex::Driver`] resolves to
//! `seat_r`, and a family that wanted left-hand drive would move the exit and
//! the carjack with it rather than this table.
//!
//! # THE SEAT IS INSIDE THE CABIN — carried 160, closed here
//!
//! `VehicleRig::seat_local` was the chassis collider's **top face** from P29.7
//! until this wave, so the point a driver's feet were placed on was the car's
//! ROOF. The CHAR1c audit measured three of the island's four drivers standing
//! at `-0.000 m` and `-0.001 m` of their own roof and filed the arm
//! `an_island_driver_is_seated_inside_its_car_and_not_on_top_of_it` as
//! `#[ignore]`d rather than write the defect down as a rule. [`SEAT_FLOOR_FRAC_Y`]
//! is where that number went: the seat a rig publishes is now the **foot well
//! floor** of the driver's side, which is where a seated body's heels are.

use std::collections::BTreeMap;

use uuid::Uuid;

use crate::math::Vec3d;
use crate::world::EcsWorld;

// ── the fractions, stated once ──────────────────────────────────────────────

/// **The foot well floor**, as a fraction of the chassis half-height.
///
/// The point [`VehicleRig::seat_local`](crate::vehicle::VehicleRig::seat_local)
/// publishes, because that field has always been where a seated character's
/// FEET are placed (`step_driving` lifts the capsule from it by
/// `stand_half_height_m + radius`). A road car's floor pan sits a little above
/// the sills, which on a body whose collider spans `±half.y` is a touch above
/// the bottom face — hence `-0.80` and not `-1.0`.
///
/// **It was `-0.55` in the recovered draft and that was a head through the
/// roof.** On the saloon (`half_height_m = 0.62`) a `-0.55` floor and a
/// `-0.10` cushion put the H-point 0.062 m below the chassis centre, and a
/// seated 1.8 m body's crown is about 0.83 m above its H-point — 0.77 m, which
/// is 0.15 m above a 0.62 m roof. A real saloon's floor pan is about 0.3 m off
/// the road and its roof 1.45 m, so the floor sits near the bottom of the
/// body box and the head clears the roof by a hand. `-0.80` and a `-0.45`
/// cushion give the saloon a 0.217 m cushion over its floor and a crown
/// 0.07 m under its roof; the coupe (`0.58`), the lowest body the catalogue
/// has, clears by 0.01 m.
pub const SEAT_FLOOR_FRAC_Y: f64 = -0.80;

/// **The cushion — the H-point**, as a fraction of the chassis half-height.
///
/// 0.35 of the half-height above the floor, which on the island's saloon
/// (`half_height_m = 0.62`) is 0.217 m: a real car's hip point is 0.20–0.35 m
/// above the heel, and this is proportional rather than fixed for
/// [`BodyPart`](crate::vehicle::BodyPart)'s own reason — a fixed 0.28 m cushion
/// would be a bar stool in a truck and a floor mat in a coupe. See
/// [`SEAT_FLOOR_FRAC_Y`] for why the pair moved from the recovered draft's
/// `-0.55` / `-0.10`.
pub const SEAT_CUSHION_FRAC_Y: f64 = -0.45;

/// How far off the centreline a front seat is, as a fraction of the chassis
/// half-width. 0.42 of 0.92 m is 0.386 m on the saloon, which is a real car's
/// seat centre.
pub const SEAT_LATERAL_FRAC_X: f64 = 0.42;

/// How far FORWARD of the chassis centre a front seat is, as a fraction of the
/// half-length.
pub const SEAT_FRONT_FRAC_Z: f64 = 0.10;

/// The rear bench, same units. Behind the front seats by 0.48 of the
/// half-length, which is a saloon's 1.06 m of legroom-plus-seat.
pub const SEAT_REAR_FRAC_Z: f64 = -0.38;

/// The pedal face's height, as a fraction of the chassis half-height — just
/// clear of the floor ([`SEAT_FLOOR_FRAC_Y`] is `-0.80`, so the face is 0.062 m
/// above the pan on the saloon).
pub const PEDAL_FRAC_Y: f64 = -0.70;

/// How far ahead of the chassis centre the pedals are, as a fraction of the
/// half-length. 0.40 of 2.2 m is 0.88 m, and the cushion is at 0.22 m — so the
/// pedal is 0.66 m ahead of the H-point, which is a driving position.
pub const PEDAL_FRAC_Z: f64 = 0.40;

/// The throttle's lateral place, as a fraction of the DRIVER's own lateral
/// fraction — **to the driver's right of the seat's centreline**, because that
/// is where a throttle is in every car, whichever side its driver sits.
///
/// The table is written for a body whose RIGHT is the chassis `+X` — which is
/// `inf_anim::build_template`'s own convention (*"left is −X and right is
/// +X"*). A rig whose right is `-X` (an import that kept glTF's handedness) is
/// seated at the same place and has the two pedals **mirrored about the seat's
/// centreline** by [`mirror_about_seat`], so the throttle is always under the
/// right foot. The pedals are a statement about the driver's body, not about
/// the car's.
pub const PEDAL_THROTTLE_FRAC_X: f64 = 1.25;

/// The brake's, same units: a little to the driver's LEFT of the throttle,
/// on the column's own line. Distinguishable from the throttle in a trace and
/// in a frame by 0.154 m on the saloon.
pub const PEDAL_BRAKE_FRAC_X: f64 = 0.85;

/// How far a pedal's face travels when its input goes to 1, metres. A road
/// car's pedal is 60–80 mm of travel; the foot follows it.
pub const PEDAL_TRAVEL_M: f64 = 0.07;

/// The steering wheel hub's height above the chassis centre, as a fraction of
/// the half-height. With [`SEAT_CUSHION_FRAC_Y`] at `-0.45` the hub is 0.57 of
/// the half-height above the cushion — 0.353 m on the saloon, which is where a
/// wheel centre is relative to an H-point (0.33–0.40 m in a road car).
pub const WHEEL_HUB_FRAC_Y: f64 = 0.12;

/// How far ahead of the chassis centre the hub is, as a fraction of the
/// half-length: 0.30 of 2.2 m is 0.66 m, and the cushion is at 0.22 m, so the
/// wheel is 0.44 m ahead of the hip.
pub const WHEEL_HUB_FRAC_Z: f64 = 0.30;

/// **The column rake**, degrees back from vertical. A road car's steering
/// column sits 20–25° off upright; the rim's plane is normal to it, which is
/// what puts a driver's hands slightly above and behind the hub rather than
/// straight out in front of it.
pub const WHEEL_RAKE_DEG: f64 = 22.0;

/// The steering wheel's rim radius, as a fraction of the chassis half-width.
/// 0.19 of 0.92 m is 0.175 m — a 350 mm wheel.
pub const WHEEL_RIM_FRAC: f64 = 0.19;

/// **How far the rack turns at full lock**, degrees of WHEEL rotation.
///
/// A road car is 2.5 turns lock to lock, so one full lock is 450°. The rim
/// angle is `steer / max_steer * WHEEL_LOCK_DEG`, which is what makes a hand on
/// the rim travel three quarters of a circle rather than the 32° the road wheel
/// turns.
pub const WHEEL_LOCK_DEG: f64 = 450.0;

/// A door handle's height, as a fraction of the door part's OWN half-height,
/// measured from the part's centre. `0.70` puts it just under the waist line —
/// the door's top edge here, because the greenhouse above it is a separate part
/// — which is where a handle is.
///
/// It was `1/3` (VEH3c's sentence, *"two thirds of its height"*) in the
/// recovered draft, and on the saloon that is a handle 0.68 m off the road: a
/// standing 1.8 m body's shoulder is 1.48 m up and its arm 0.54 m long, so it
/// would have to fold nearly half a metre at the hips to touch it. A real
/// saloon's handle is 0.85–0.95 m up. `0.70` is 0.76 m.
pub const HANDLE_HEIGHT_FRAC: f64 = 0.70;

/// How far AFT of the door's centre the handle sits, as a fraction of the
/// door's own half-length. The hinge is at the door's forward edge
/// ([`crate::vehicle::door_hinge`]), so the handle goes toward the trailing one.
pub const HANDLE_AFT_FRAC: f64 = 0.5;

/// The handle socket a car with **no door part** gets, as fractions of the
/// chassis half-extents: `(±1, HANDLE_FALLBACK_Y, HANDLE_FALLBACK_Z)`.
///
/// A refusal is a value, and an absent socket would make every consumer branch.
/// A car with no doors (the VEH2a cube body, a boat, a rotorcraft) still has a
/// flank, and the flank at door height is where a hand would go.
pub const HANDLE_FALLBACK_Y: f64 = -0.05;
/// The other half of [`HANDLE_FALLBACK_Y`].
pub const HANDLE_FALLBACK_Z: f64 = 0.28;

/// **How far out from the flank a boarding body stands to take the handle**,
/// metres — the approach spline's end, measured to the body's centre.
///
/// Not a fraction: it is a property of the BODY, not of the car — a person
/// stands the same distance from a hatchback's door as from a van's. Close,
/// because a hand has to reach the handle from it and an arm is 0.54 m.
pub const STANCE_OUT_M: f64 = 0.30;

/// **How far AFT of the handle that stance is**, metres. The body stands
/// behind the handle, facing the car, so the hand nearest the nose takes it —
/// which is the reference's own picture (`frames/steal-car/0016`).
pub const STANCE_AFT_M: f64 = 0.24;

/// **Where the body steps BACK to while the door swings**, metres behind the
/// door's trailing edge.
///
/// A door hinged at its forward edge sweeps a quarter-disc whose radius is its
/// own length, and a body standing where it took the handle is inside it — the
/// leaf would pass through the body at about fourteen degrees on the saloon.
/// So `OpeningDoor` walks the root from the take to a point this far behind the
/// trailing edge ([`STEP_BACK_OUT_M`] out), which is outside the swept disc by
/// more than a body's radius on every family the catalogue has.
pub const STEP_BACK_AFT_M: f64 = 0.38;

/// …and how far out from the flank that point is, metres.
pub const STEP_BACK_OUT_M: f64 = 0.42;

/// **How far the pelvis dips to put a handle in reach**, metres, at most.
///
/// A door handle on a saloon is about 0.70 m off the road and a standing
/// shoulder is about 1.45 m, so a straight-armed upright body cannot touch one
/// — which is true of a person, and what a person does about it is bend. The
/// dip is derived per character in [`reach_dip_m`] and clamped here.
pub const MAX_REACH_DIP_M: f64 = 0.42;

/// Where a standing body's shoulder is, as a fraction of its own standing
/// height (capsule half-height plus radius, doubled).
pub const SHOULDER_FRAC: f64 = 0.82;

/// How far a shoulder can reach, as a fraction of the same standing height.
///
/// 0.28 of 1.8 m is 0.504 m. `inf_anim::BodyParams::arm_length_ratio` is 0.30
/// (0.54 m shoulder-to-wrist, measured off both of Epic's mannequins), and a
/// two-bone solve asked for its full length has one configuration — the arm
/// locked straight — so the reach a request is PLANNED against keeps six per
/// cent in hand. It was `0.36` (0.65 m) in the recovered draft, which is an
/// arm twelve centimetres longer than any body this engine builds.
pub const ARM_REACH_FRAC: f64 = 0.28;

/// **Where a standing body's hip joint is**, as a fraction of its standing
/// height — `inf_anim::BodyParams::hip_height_ratio`'s own default, 0.53.
///
/// This engine has no seated locomotion clip: `inf_anim::als::LocoMode`
/// reserves 12 for `Driving` and says in its own doc that the state *"is not
/// animated by this graph"*. What it does have is a pelvis offset that the pose
/// step applies to the pelvis joint's local Y before the limbs solve
/// (`inf_ecs::pose`'s `pelvis_drop`), foot IK to a point, and hand IK to a
/// point. A pelvis dropped onto the cushion with the feet on the pedals and the
/// hands on the rim is a seated body made out of the three primitives this
/// engine has — and [`seated_pelvis_drop_m`] is the arithmetic, per body, so a
/// 1.2 m character sits on the same cushion as a 1.9 m one.
pub const PELVIS_FRAC: f64 = 0.53;

/// **How far a seated body's pelvis drops**, metres (negative is down): from
/// where a standing pose puts it to the cushion.
///
/// * `standing_m` — the body's full standing height (capsule, doubled).
/// * `cushion_above_feet_m` — how far the cushion is above the point the body's
///   feet are placed on, which for the driver is the foot-well floor
///   [`crate::vehicle::VehicleRig::seat_local`] publishes.
///
/// Zero for a non-finite or non-positive height, and never positive: a cushion
/// higher than a standing hip is a bar stool, and a body does not stand up to
/// sit on one.
pub fn seated_pelvis_drop_m(standing_m: f64, cushion_above_feet_m: f64) -> f64 {
    if !(standing_m.is_finite() && standing_m > 0.0 && cushion_above_feet_m.is_finite()) {
        return 0.0;
    }
    (cushion_above_feet_m - PELVIS_FRAC * standing_m).min(0.0)
}

// ── the socket table ────────────────────────────────────────────────────────

/// **Which seat**, indexed at last — wave VEH2b carried *"one seat, no index"*
/// as item 4 and this closes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SeatIndex {
    /// The driver's — `seat_r`, the `+X` side. See this module's own note.
    #[default]
    Driver,
    /// The front passenger's — `seat_l`.
    Passenger,
    /// The rear bench.
    Rear,
}

impl SeatIndex {
    /// The frozen wire number this seat folds as — **append only**, exactly as
    /// [`crate::vehicle::BodyPartKind::as_u8`] is, because it reaches the
    /// determinism trace.
    pub fn as_u8(self) -> u8 {
        match self {
            SeatIndex::Driver => 0,
            SeatIndex::Passenger => 1,
            SeatIndex::Rear => 2,
        }
    }

    /// The inverse. An unknown number is the driver's seat, which is the only
    /// seat every vehicle in this engine has.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => SeatIndex::Passenger,
            2 => SeatIndex::Rear,
            _ => SeatIndex::Driver,
        }
    }

    /// A stable short name, for a gate trace, a HUD row and `hero.csv`.
    pub fn name(self) -> &'static str {
        match self {
            SeatIndex::Driver => "driver",
            SeatIndex::Passenger => "passenger",
            SeatIndex::Rear => "rear",
        }
    }

    /// Whether this seat's occupant drives. Exactly one does.
    pub fn drives(self) -> bool {
        self == SeatIndex::Driver
    }

    /// Every seat, in wire order.
    pub const ALL: [SeatIndex; 3] = [SeatIndex::Driver, SeatIndex::Passenger, SeatIndex::Rear];
}

/// **One part's geometry**, as [`sockets_of`] needs it — the three fields of a
/// [`PartState`](crate::bodywork::PartState) a socket is derived from.
///
/// A borrowed slice of these rather than the world, because the derivation has
/// to be callable from a gate with no `EcsWorld` and from the bodywork table
/// with one. `kind` is [`crate::vehicle::BodyPartKind::as_u8`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartGeom {
    /// `BodyPartKind::as_u8`.
    pub kind: u8,
    /// The part's centre, in fractions of the chassis half-extents.
    pub centre_frac: Vec3d,
    /// Its half-extents, same units.
    pub half_frac: Vec3d,
}

/// **The eight sockets**, in the chassis frame, metres — DERIVED, never
/// persisted. See the module docs for the ruling.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VehicleSockets {
    /// The near-side (`-X`) front cushion.
    pub seat_l: Vec3d,
    /// The off-side (`+X`) front cushion — **the driver's**.
    pub seat_r: Vec3d,
    /// The rear bench's centre cushion.
    pub seat_rear: Vec3d,
    /// The near-side front door's outer handle.
    pub door_handle_l: Vec3d,
    /// The off-side front door's outer handle — the one a boarding driver takes.
    pub door_handle_r: Vec3d,
    /// The throttle pedal's face.
    pub pedal_throttle: Vec3d,
    /// The brake pedal's face.
    pub pedal_brake: Vec3d,
    /// The steering wheel's hub.
    pub wheel_hub: Vec3d,
    /// The rim radius, metres — not a socket, but the number a hand on the rim
    /// is placed with, and it is derived from the same half-extents.
    pub wheel_rim_m: f64,
}

impl VehicleSockets {
    /// One seat's **cushion** (the H-point), by index.
    pub fn seat(&self, seat: SeatIndex) -> Vec3d {
        match seat {
            SeatIndex::Driver => self.seat_r,
            SeatIndex::Passenger => self.seat_l,
            SeatIndex::Rear => self.seat_rear,
        }
    }

    /// One seat's **floor** — where that seat's occupant's feet go, which is
    /// the point [`crate::vehicle::VehicleRig::seat_local`] publishes for the
    /// driver.
    ///
    /// The cushion with its Y replaced by the floor's: the two differ only in
    /// height, because a body sits where its feet are.
    pub fn seat_floor(&self, seat: SeatIndex, floor_y: f64) -> Vec3d {
        let c = self.seat(seat);
        Vec3d::new(c.x, floor_y, c.z)
    }

    /// The outer handle of the door that serves `seat`.
    ///
    /// The rear bench is reached through the rear door on the DRIVER's side,
    /// which is the same flank — so it answers the off-side handle. A family
    /// with a rear door of its own is VEH3f's, and when one lands this is the
    /// one function that changes.
    pub fn handle(&self, seat: SeatIndex) -> Vec3d {
        match seat {
            SeatIndex::Passenger => self.door_handle_l,
            SeatIndex::Driver | SeatIndex::Rear => self.door_handle_r,
        }
    }

    /// Which flank `seat` is boarded from: `+1` for the off side, `-1` for the
    /// near one.
    pub fn side_sign(seat: SeatIndex) -> f64 {
        match seat {
            SeatIndex::Passenger => -1.0,
            SeatIndex::Driver | SeatIndex::Rear => 1.0,
        }
    }

    /// **Override one socket by name** — the door a mesh's own socket table
    /// comes through.
    ///
    /// Nothing committed calls it with a mesh yet, and the bound is worth
    /// stating rather than discovering: wave WPN2d's socket door is a
    /// **skeleton** door (`inf_anim::sockets::Socket` carries a joint index),
    /// and a car in this engine has no skeleton — every family's
    /// `RigNode::mesh` is `None` and the body is drawn as `Primitive::Cube`
    /// boxes. So the override that exists today is the PART CENSUS, which is
    /// what [`sockets_of`] takes; this is the seam a real `.inf_mesh` car body
    /// hangs its own table on when VEH3f lands one, and it is exercised by the
    /// gate rather than left as a promise.
    pub fn set(&mut self, name: &str, at: Vec3d) -> bool {
        let slot = match name {
            "seat_l" => &mut self.seat_l,
            "seat_r" => &mut self.seat_r,
            "seat_rear" => &mut self.seat_rear,
            "door_handle_l" => &mut self.door_handle_l,
            "door_handle_r" => &mut self.door_handle_r,
            "pedal_throttle" => &mut self.pedal_throttle,
            "pedal_brake" => &mut self.pedal_brake,
            "wheel_hub" => &mut self.wheel_hub,
            _ => return false,
        };
        if !at.x.is_finite() || !at.y.is_finite() || !at.z.is_finite() {
            return false;
        }
        *slot = at;
        true
    }

    /// Every socket's name, in a stable order — the vocabulary [`set`](Self::set)
    /// accepts and a gate walks.
    pub fn names() -> &'static [&'static str] {
        &[
            "door_handle_l",
            "door_handle_r",
            "pedal_brake",
            "pedal_throttle",
            "seat_l",
            "seat_r",
            "seat_rear",
            "wheel_hub",
        ]
    }

    /// One socket by name — [`set`](Self::set)'s twin, so a name the door
    /// writes is a name something can read back.
    pub fn get(&self, name: &str) -> Option<Vec3d> {
        Some(match name {
            "seat_l" => self.seat_l,
            "seat_r" => self.seat_r,
            "seat_rear" => self.seat_rear,
            "door_handle_l" => self.door_handle_l,
            "door_handle_r" => self.door_handle_r,
            "pedal_throttle" => self.pedal_throttle,
            "pedal_brake" => self.pedal_brake,
            "wheel_hub" => self.wheel_hub,
            _ => return None,
        })
    }
}

/// **DERIVE the eight sockets** from a chassis collider and the parts bolted to
/// it. The one door; every consumer calls this rather than holding a field.
///
/// * `half` — the chassis collider's half-extents, metres. For every catalogue
///   row this IS `VehicleDef::half_extents`: the generator writes one onto the
///   other (`inf_ecs::vehicle::rig_nodes`).
/// * `offset` — the collider's own offset from the body origin, so a chassis
///   whose box is not centred still seats its driver in the cabin.
/// * `parts` — the family's parts, as the world holds them. **Empty is legal**
///   and is what a bare box body gets: the seats, the pedals and the hub are
///   functions of the half-extents alone, and only the two handles look at a
///   door. A car with no doors gets the flank fallback.
pub fn sockets_of(half: Vec3d, offset: Vec3d, parts: &[PartGeom]) -> VehicleSockets {
    let (hx, hy, hz) = (half.x.abs(), half.y.abs(), half.z.abs());
    let at = |fx: f64, fy: f64, fz: f64| {
        Vec3d::new(offset.x + fx * hx, offset.y + fy * hy, offset.z + fz * hz)
    };
    let seat_r = at(SEAT_LATERAL_FRAC_X, SEAT_CUSHION_FRAC_Y, SEAT_FRONT_FRAC_Z);
    let seat_l = at(-SEAT_LATERAL_FRAC_X, SEAT_CUSHION_FRAC_Y, SEAT_FRONT_FRAC_Z);
    let seat_rear = at(0.0, SEAT_CUSHION_FRAC_Y, SEAT_REAR_FRAC_Z);
    // The pedals and the hub belong to the DRIVER's side, whichever that is.
    let d = SEAT_LATERAL_FRAC_X;
    let pedal_throttle = at(d * PEDAL_THROTTLE_FRAC_X, PEDAL_FRAC_Y, PEDAL_FRAC_Z);
    let pedal_brake = at(d * PEDAL_BRAKE_FRAC_X, PEDAL_FRAC_Y, PEDAL_FRAC_Z);
    let wheel_hub = at(d, WHEEL_HUB_FRAC_Y, WHEEL_HUB_FRAC_Z);
    let mut out = VehicleSockets {
        seat_l,
        seat_r,
        seat_rear,
        door_handle_l: at(-1.0, HANDLE_FALLBACK_Y, HANDLE_FALLBACK_Z),
        door_handle_r: at(1.0, HANDLE_FALLBACK_Y, HANDLE_FALLBACK_Z),
        pedal_throttle,
        pedal_brake,
        wheel_hub,
        wheel_rim_m: WHEEL_RIM_FRAC * hx,
    };
    // ── the handles, from the DOORS (VEH3c's own derivation) ────────────────
    //
    // The door's outer face just under its top edge, pushed out along the
    // part's own FACING axis by its half-extent and aft by half of its length
    // (`outer_handle_in_door`, the one arithmetic the live door is read with).
    // `PartState::facing` is derived from the part's offset from the chassis
    // centre, and a door's is X — so this is "the outside of the door skin",
    // proportionally, for any family that moves its doors.
    for side in [-1.0f64, 1.0] {
        let Some(door) = front_door(parts, side) else {
            continue;
        };
        let (c, h) = door_metres(half, offset, door);
        let o = outer_handle_in_door(h, side);
        let socket = Vec3d::new(c.x + o.x, c.y + o.y, c.z + o.z);
        if side > 0.0 {
            out.door_handle_r = socket;
        } else {
            out.door_handle_l = socket;
        }
    }
    out
}

/// **The parts of one car, as [`sockets_of`] wants them** — the bodywork
/// table's own rows, in `Guid` order.
///
/// `O(parts)`, and an empty vec for a car the bodywork has never walked, which
/// is the fallback case [`sockets_of`] is written to take.
pub fn part_geoms(world: &EcsWorld, chassis: Uuid) -> Vec<PartGeom> {
    let Some(row) = crate::bodywork::damage_row(world, chassis) else {
        return Vec::new();
    };
    row.parts
        .values()
        .filter(|s| s.latch.attached())
        .map(|s| PartGeom {
            kind: s.kind,
            centre_frac: s.centre_frac,
            half_frac: s.half_frac,
        })
        .collect()
}

/// **The door part that serves a seat**, and whether it is still on the car.
///
/// Answers the part's `Guid` so [`crate::bodywork`]'s motor can be aimed at it,
/// and `None` for a car with no such door — a boat, a bare box body, or **a car
/// whose door was torn off**, which is VEH3c's `PartLatch::Shed`. The third case
/// is the one the arc brief names: a car with no door is boarded *through the
/// opening*, so the pipeline skips the `OpeningDoor` phase rather than waiting
/// for a hinge that is lying in the road.
pub fn door_for_seat(world: &EcsWorld, chassis: Uuid, seat: SeatIndex) -> Option<Uuid> {
    let side = VehicleSockets::side_sign(seat);
    let row = crate::bodywork::damage_row(world, chassis)?;
    let mut best: Option<(Uuid, f64)> = None;
    let mut attached = false;
    for (guid, st) in row.parts.iter() {
        if st.kind != crate::vehicle::KIND_DOOR {
            continue;
        }
        if st.centre_frac.x * side <= 0.0 {
            continue;
        }
        // The same strict `>` as `front_door`, over the same `Guid`-ordered
        // map, so the handle socket and the door the motor drives are the
        // same door.
        let z = st.centre_frac.z;
        if best.is_none_or(|(_, bz)| z > bz) {
            best = Some((*guid, z));
            attached = st.latch.attached();
        }
    }
    // The seat's door is the FRONT door whatever became of it: a torn-off one
    // answers `None` (board through the opening) rather than the next door
    // back — which the first version answered, measured by the gate's torn-off
    // arm as a driver reaching for the REAR door's handle.
    best.filter(|_| attached).map(|(g, _)| g)
}

// ── the geometry a boarding walks through ───────────────────────────────────

/// **The front door on one flank** of a parts list — `side` is `+1` for the
/// `+X` (driver's) flank and `-1` for the other.
///
/// Of the doors whose centre is on that side, the one furthest forward; a
/// strict `>` so the first of two at the same `z` wins and the answer does not
/// depend on slice order. `None` for a car with no door on that flank, which is
/// a boat, a bare box body or a car whose door VEH3c tore off.
pub fn front_door(parts: &[PartGeom], side: f64) -> Option<&PartGeom> {
    let mut best: Option<&PartGeom> = None;
    for p in parts {
        if p.kind != crate::vehicle::KIND_DOOR {
            continue;
        }
        if p.centre_frac.x * side <= 0.0 {
            continue;
        }
        if best.is_none_or(|b| p.centre_frac.z > b.centre_frac.z) {
            best = Some(p);
        }
    }
    best
}

/// A door's box in METRES, chassis frame: `(centre, half)`.
pub fn door_metres(half: Vec3d, offset: Vec3d, door: &PartGeom) -> (Vec3d, Vec3d) {
    let (hx, hy, hz) = (half.x.abs(), half.y.abs(), half.z.abs());
    (
        Vec3d::new(
            offset.x + door.centre_frac.x * hx,
            offset.y + door.centre_frac.y * hy,
            offset.z + door.centre_frac.z * hz,
        ),
        Vec3d::new(
            door.half_frac.x.abs() * hx,
            door.half_frac.y.abs() * hy,
            door.half_frac.z.abs() * hz,
        ),
    )
}

/// **Where the outer handle is on its door**, metres, from the door's own
/// centre, in the door's frame at zero degrees (which is the chassis's axes).
///
/// The door's outer face ([`HANDLE_HEIGHT_FRAC`] of the way up it, and aft of
/// its centre by [`HANDLE_AFT_FRAC`] of its half-length) — so a door that has
/// swung carries its handle round with it: the physics side rotates this offset
/// by the door body's own live rotation, never re-derives it from a table.
pub fn outer_handle_in_door(door_half_m: Vec3d, side: f64) -> Vec3d {
    Vec3d::new(
        side.signum() * door_half_m.x.abs(),
        HANDLE_HEIGHT_FRAC * door_half_m.y.abs(),
        -HANDLE_AFT_FRAC * door_half_m.z.abs(),
    )
}

/// **The inner pull** — the door's INSIDE face, at the handle's height and a
/// little further forward (the arm-rest pull a driver shuts a door with).
pub fn inner_handle_in_door(door_half_m: Vec3d, side: f64) -> Vec3d {
    Vec3d::new(
        -side.signum() * door_half_m.x.abs(),
        HANDLE_HEIGHT_FRAC * door_half_m.y.abs(),
        INNER_PULL_FWD_FRAC * door_half_m.z.abs(),
    )
}

/// How far FORWARD of the door's centre the inner pull is, as a fraction of its
/// half-length — the arm rest, which is nearer the hinge than the outer handle.
pub const INNER_PULL_FWD_FRAC: f64 = 0.10;

/// **The two ground points a boarding walks between**, chassis frame, metres:
/// `(take, back)` — where the body stands to TAKE the handle (the approach's
/// end), and where it steps BACK to while the door swings clear of it.
///
/// `y` is left at `0`: the ground is a WORLD fact the physics side finds with a
/// ray, and a socket table has no business guessing it.
///
/// * The take is [`STANCE_OUT_M`] out from the flank and [`STANCE_AFT_M`]
///   behind the handle.
/// * The step-back is [`STEP_BACK_OUT_M`] out and [`STEP_BACK_AFT_M`] behind
///   the door's TRAILING edge — outside the quarter-disc the leaf sweeps.
///
/// A car with no door on that flank has no leaf to clear, so both points are
/// the take, and the pipeline boards through the opening.
pub fn stance_points(
    sockets: &VehicleSockets,
    seat: SeatIndex,
    half: Vec3d,
    offset: Vec3d,
    door: Option<&PartGeom>,
) -> (Vec3d, Vec3d) {
    let side = VehicleSockets::side_sign(seat);
    let flank = offset.x + side * half.x.abs();
    let handle = sockets.handle(seat);
    let take = Vec3d::new(flank + side * STANCE_OUT_M, 0.0, handle.z - STANCE_AFT_M);
    let back = match door {
        Some(d) => {
            let (c, h) = door_metres(half, offset, d);
            let trailing = c.z - h.z;
            Vec3d::new(
                flank + side * STEP_BACK_OUT_M,
                0.0,
                (trailing - STEP_BACK_AFT_M).min(take.z),
            )
        }
        None => take,
    };
    (take, back)
}

/// **Where a pulled-out driver is put down**, chassis frame, metres, `y = 0`
/// (the ground is the physics side's).
///
/// [`PULL_OUT_M`] out from the flank and [`PULL_OUT_AFT_M`] behind the seat:
/// through the door opening, clear of the open leaf's trailing edge and clear
/// of the hero, who is standing at the step-back point behind it. The first of
/// the candidates [`exit_candidates`] walks.
///
/// It was level with the seat and 0.85 m out in the first cut, and the gate's
/// own collide-check refused it on the saloon: with the door at the 45 degrees
/// the pull starts at, the leaf's trailing edge is 0.22 m from that point —
/// inside a body's radius — so every pull fell through to the SECOND candidate,
/// 0.7 m further out, and dragged the driver across the edge of the door on the
/// way.
pub fn pull_out_point(
    sockets: &VehicleSockets,
    seat: SeatIndex,
    half: Vec3d,
    offset: Vec3d,
) -> Vec3d {
    let side = VehicleSockets::side_sign(seat);
    let s = sockets.seat(seat);
    Vec3d::new(
        offset.x + side * (half.x.abs() + PULL_OUT_M),
        0.0,
        s.z - PULL_OUT_AFT_M,
    )
}

/// How far from the flank a pulled-out driver lands, metres.
pub const PULL_OUT_M: f64 = 1.0;

/// How far behind the seat it lands, metres.
pub const PULL_OUT_AFT_M: f64 = 0.25;

/// **Every place a body may leave a car to**, chassis frame, metres, `y = 0`,
/// in the order they are tried — the exit's (and the pull-out's) collide check
/// walks this list and takes the first point whose capsule is clear of every
/// collider in the world.
///
/// 1. `preferred` — the step-back point for an exit, the pull-out point for a
///    victim.
/// 2. The same flank, further out.
/// 3. The OTHER flank, mirrored — a car parked against a wall is left by the
///    passenger door.
/// 4. Behind the car, then ahead of it.
///
/// Six points and a fixed order, so the answer is a pure function of the
/// world and both hosts pick the same one.
pub fn exit_candidates(preferred: Vec3d, half: Vec3d, offset: Vec3d) -> [Vec3d; 6] {
    let (hx, hz) = (half.x.abs(), half.z.abs());
    let side = if preferred.x - offset.x >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let further = Vec3d::new(preferred.x + side * EXIT_FURTHER_M, 0.0, preferred.z);
    let mirrored = Vec3d::new(2.0 * offset.x - preferred.x, 0.0, preferred.z);
    let mirrored_further = Vec3d::new(mirrored.x - side * EXIT_FURTHER_M, 0.0, mirrored.z);
    let behind = Vec3d::new(offset.x + side * 0.5 * hx, 0.0, offset.z - hz - EXIT_END_M);
    let ahead = Vec3d::new(offset.x + side * 0.5 * hx, 0.0, offset.z + hz + EXIT_END_M);
    [
        preferred,
        further,
        mirrored,
        mirrored_further,
        behind,
        ahead,
    ]
}

/// How much further out the second candidate is, metres.
pub const EXIT_FURTHER_M: f64 = 0.7;

/// How far past the bumper the end candidates are, metres.
pub const EXIT_END_M: f64 = 0.8;

/// **How far the steering wheel has turned**, degrees of RIM rotation, for a
/// road wheel at `steer_deg` on a rack whose full lock is `max_steer_deg`.
///
/// `steer_deg / max_steer_deg * WHEEL_LOCK_DEG`, clamped to one lock either
/// way — so a hand on the rim travels a quarter turn and more while the road
/// wheel moves thirty degrees, which is the ratio that makes a driver READ as
/// steering. Zero for a rack with no lock.
pub fn rim_angle_deg(steer_deg: f64, max_steer_deg: f64) -> f64 {
    if !(steer_deg.is_finite() && max_steer_deg.is_finite()) || max_steer_deg.abs() < 1e-6 {
        return 0.0;
    }
    (steer_deg / max_steer_deg.abs()).clamp(-1.0, 1.0) * WHEEL_LOCK_DEG
}

/// Where each hand holds the rim at rest, degrees round it from the `+X`
/// spoke, anticlockwise seen from the driver: the quarter-to-three grip.
pub const RIM_GRIP_DEG: f64 = 20.0;

/// How far round the rim the hands RIDE with it, degrees, before the rim
/// slides through them — the push-pull technique driving schools teach.
///
/// A full lock is [`WHEEL_LOCK_DEG`] = 450° of rim, and hands that rode all of
/// it would cross and wind past each other: measured, the gate's full-lock arm
/// found a hand 111.7 mm short of its grip where a grip had wrapped to the far
/// side of the hub. So each hand turns with the rim to this angle and holds
/// there while the rim keeps turning under it; the grip is on the rim at every
/// angle, the hands never cross, and the function stays pure (no re-grip state
/// to fold).
pub const RIM_RIDE_DEG: f64 = 50.0;

/// **The two grips on the rim**, chassis frame, metres: `[the one on the +X
/// side, the one on the -X side]` at rest, both turned by `rim_deg` up to
/// [`RIM_RIDE_DEG`] either way (the rim slides through the hands past it).
///
/// The rim's plane is raked back [`WHEEL_RAKE_DEG`] from vertical, so its "up"
/// axis tilts toward the driver; a positive `rim_deg` turns the wheel the way a
/// right turn does. Portable trig ([`inf_math::psin64`] / `pcos64`): these
/// points become IK goals, the goals become a pose, and the pose is folded.
pub fn wheel_grips(sockets: &VehicleSockets, rim_deg: f64) -> [Vec3d; 2] {
    let rake = WHEEL_RAKE_DEG.to_radians();
    // In-plane axes: `u` is lateral, `v` is the rim's "up", tilted toward the
    // driver (`-Z`) by the rake.
    let (sr, cr) = (inf_math::psin64(rake), inf_math::pcos64(rake));
    let v = Vec3d::new(0.0, cr, -sr);
    let r = sockets.wheel_rim_m;
    let hub = sockets.wheel_hub;
    let at = |deg: f64| {
        let a = deg.to_radians();
        let (s, c) = (inf_math::psin64(a), inf_math::pcos64(a));
        Vec3d::new(hub.x + r * c, hub.y + r * s * v.y, hub.z + r * s * v.z)
    };
    // A right turn is CLOCKWISE seen from the seat, which is a negative angle
    // in this plane's own sense.
    let ride = if rim_deg.is_finite() {
        rim_deg.clamp(-RIM_RIDE_DEG, RIM_RIDE_DEG)
    } else {
        0.0
    };
    [at(RIM_GRIP_DEG - ride), at(180.0 - RIM_GRIP_DEG - ride)]
}

/// **The two pedal faces with the driver's feet on them**, chassis frame,
/// metres: `(throttle, brake)`, each pushed forward and down by
/// [`PEDAL_TRAVEL_M`] times its own input.
///
/// A pedal hinges at its top, so a press moves the face forward mostly and down
/// a little: `(0, -0.35, 1)` normalised is the travel direction, which is a
/// real pedal box's arc at the face.
pub fn pedal_faces(sockets: &VehicleSockets, throttle: f64, brake: f64) -> (Vec3d, Vec3d) {
    let t = throttle.clamp(0.0, 1.0);
    let b = brake.clamp(0.0, 1.0);
    let n = (1.0f64 + 0.35 * 0.35).sqrt();
    let dir = Vec3d::new(0.0, -0.35 / n, 1.0 / n);
    let press = |p: Vec3d, x: f64| {
        Vec3d::new(
            p.x,
            p.y + dir.y * PEDAL_TRAVEL_M * x,
            p.z + dir.z * PEDAL_TRAVEL_M * x,
        )
    };
    (
        press(sockets.pedal_throttle, t),
        press(sockets.pedal_brake, b),
    )
}

/// **Mirror a chassis-frame point about a seat's centreline** — how a rig whose
/// RIGHT is `-X` gets its throttle under its right foot. See
/// [`PEDAL_THROTTLE_FRAC_X`].
pub fn mirror_about_seat(p: Vec3d, seat_x: f64) -> Vec3d {
    Vec3d::new(2.0 * seat_x - p.x, p.y, p.z)
}

/// **Where a passenger's feet go**, chassis frame, metres: the floor ahead of
/// the cushion, a hip's width apart. `floor_y` is the pan's height.
pub fn floor_feet(sockets: &VehicleSockets, seat: SeatIndex, floor_y: f64) -> [Vec3d; 2] {
    let s = sockets.seat(seat);
    let z = s.z + FLOOR_FOOT_FWD_M;
    let y = floor_y + FLOOR_FOOT_UP_M;
    [
        Vec3d::new(s.x - FLOOR_FOOT_SPAN_M, y, z),
        Vec3d::new(s.x + FLOOR_FOOT_SPAN_M, y, z),
    ]
}

/// How far ahead of the cushion a passenger's feet rest, metres.
pub const FLOOR_FOOT_FWD_M: f64 = 0.46;
/// How far above the pan the ankle rests, metres.
pub const FLOOR_FOOT_UP_M: f64 = 0.08;
/// Half the distance between a passenger's feet, metres.
pub const FLOOR_FOOT_SPAN_M: f64 = 0.12;

/// **Where a passenger's hands go**, chassis frame, metres: the grab bar on the
/// dashboard for the front seat, the back of the front seat for the rear bench
/// (the doc's *"Dashboard Grab Bar (Passenger)"*).
pub fn passenger_grips(sockets: &VehicleSockets, seat: SeatIndex) -> [Vec3d; 2] {
    let s = sockets.seat(seat);
    let (y, z) = match seat {
        SeatIndex::Rear => (s.y + 0.42, sockets.seat_r.z - 0.18),
        // The dash face, 10 cm nearer the seat than the wheel hub's plane: at
        // the hub's own depth (the first version) a seated passenger's hands
        // were measured 66 mm short of the bar, because the rim is raked back
        // toward the driver and the dash is not.
        _ => (sockets.wheel_hub.y + 0.04, sockets.wheel_hub.z - 0.08),
    };
    [
        Vec3d::new(s.x - GRIP_SPAN_M, y, z),
        Vec3d::new(s.x + GRIP_SPAN_M, y, z),
    ]
}

/// Half the distance between a passenger's hands on the grab bar, metres.
pub const GRIP_SPAN_M: f64 = 0.17;

// ── the state machine ───────────────────────────────────────────────────────

/// **The boarding machine's phases** (wave VEH3d, the doc's §2).
///
/// `Locked → Unlocking → OpeningDoor → EnteringIK → Seated → Driving`, and the
/// reverse `Driving → Exiting → ClosingDoor → Idle` on the way out.
///
/// It is SIM state on the character ([`crate::components::MovementRuntime`],
/// which is `#[serde(skip)]` and never reflected — so the machine costs no
/// schema at all, which is the law this wave runs under) and it is folded into
/// the determinism trace's nineteenth section by [`boarding_state_bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoardPhase {
    /// Not boarding. The phase every character on a quiet level is in, and the
    /// one [`BoardingState::is_quiet`] folds nothing for.
    #[default]
    Idle,
    /// The press landed and the car is being checked: is there a seat, is it
    /// taken, is there a door. A refusal here is a refusal of the whole press.
    Locked,
    /// **The approach** — root motion along the cubic Hermite from where the
    /// body stood to the door stance node, at walk speed, facing locked to the
    /// vehicle's side normal.
    Unlocking,
    /// The hand reaches the outer handle over [`HAND_REACH_S`], the door opens
    /// on VEH3c's own motor, and the body steps back clear of the leaf. A
    /// CARJACK waits here for the seat to be empty.
    OpeningDoor,
    /// The seat warp — P29.7's quintic window, unchanged in length, now the
    /// path from the step-back point into the seat.
    EnteringIK,
    /// In the seat: the door is pulled shut on the inner handle and the hands
    /// find the wheel.
    Seated,
    /// Driving. The hands are on the rim and the feet are on the pedals, both
    /// following their inputs, for as long as the body is in the car. QUIET for
    /// the trace — see [`BoardingState::is_quiet`].
    Driving,
    /// **The reverse**: the door opens and the body leaves the seat — or, above
    /// [`EXIT_ROLL_MPS`], is thrown clear of a moving one.
    Exiting,
    /// Out, with a hand on the outer handle, pushing it shut.
    ClosingDoor,
    /// **Somebody is pulling this body out** — the victim's half of a carjack.
    /// The car is braked, and the phase ends in a FORCED [`Exiting`](Self::Exiting).
    Jacked,
}

impl BoardPhase {
    /// The frozen wire number this phase folds as — **append only**, because it
    /// reaches the determinism trace.
    pub fn as_u8(self) -> u8 {
        match self {
            BoardPhase::Idle => 0,
            BoardPhase::Locked => 1,
            BoardPhase::Unlocking => 2,
            BoardPhase::OpeningDoor => 3,
            BoardPhase::EnteringIK => 4,
            BoardPhase::Seated => 5,
            BoardPhase::Driving => 6,
            BoardPhase::Exiting => 7,
            BoardPhase::ClosingDoor => 8,
            BoardPhase::Jacked => 9,
        }
    }

    /// A stable short name — what `hero.csv` carries and what a gate trace
    /// prints.
    pub fn name(self) -> &'static str {
        match self {
            BoardPhase::Idle => "-",
            BoardPhase::Locked => "locked",
            BoardPhase::Unlocking => "unlocking",
            BoardPhase::OpeningDoor => "opening",
            BoardPhase::EnteringIK => "entering",
            BoardPhase::Seated => "seated",
            BoardPhase::Driving => "driving",
            BoardPhase::Exiting => "exiting",
            BoardPhase::ClosingDoor => "closing",
            BoardPhase::Jacked => "jacked",
        }
    }

    /// Whether this phase is part of getting IN (as opposed to out, or neither).
    pub fn is_entering(self) -> bool {
        matches!(
            self,
            BoardPhase::Locked
                | BoardPhase::Unlocking
                | BoardPhase::OpeningDoor
                | BoardPhase::EnteringIK
                | BoardPhase::Seated
        )
    }

    /// **Whether the body is on its own feet OUTSIDE the car in this phase** —
    /// the phases the movement step runs as a ground choreography rather than
    /// as a seat.
    pub fn is_on_the_ground(self) -> bool {
        matches!(
            self,
            BoardPhase::Locked
                | BoardPhase::Unlocking
                | BoardPhase::OpeningDoor
                | BoardPhase::ClosingDoor
        )
    }

    /// Whether the boarding machine owns this body's HANDS in this phase, so
    /// the gameplay hand pass must leave them alone.
    pub fn owns_the_hands(self) -> bool {
        !matches!(
            self,
            BoardPhase::Idle | BoardPhase::Locked | BoardPhase::Unlocking
        )
    }

    /// Every phase of the ENTER sequence, in order — what a gate walks and what
    /// [`next_enter_phase`] is the successor function of.
    pub const ENTER_ORDER: [BoardPhase; 6] = [
        BoardPhase::Locked,
        BoardPhase::Unlocking,
        BoardPhase::OpeningDoor,
        BoardPhase::EnteringIK,
        BoardPhase::Seated,
        BoardPhase::Driving,
    ];
}

/// How long the CHECK takes, seconds — the phase that decides whether there is
/// anything to board at all.
pub const LOCKED_S: f64 = 0.08;

/// The shortest the approach may be, seconds. A body already standing at the
/// door still turns to face the car.
pub const UNLOCK_MIN_S: f64 = 0.20;

/// The longest it may be. The reach is 3.00 m on the ground
/// ([`crate::vehicle`]'s `ENTER_REACH_M` on the physics side) and a walk is
/// 1.7 m/s, so the honest bound is under two seconds; this is the clamp that
/// keeps a pathological spline from becoming a cutscene.
pub const UNLOCK_MAX_S: f64 = 2.20;

/// How fast the approach walks, m/s — the ALS walk gait.
pub const APPROACH_MPS: f64 = 1.70;

/// **How long a hand takes to arrive on a handle**, seconds — the weight ramp
/// `0 → 1`, which is the number the arc brief names.
pub const HAND_REACH_S: f64 = 0.20;

/// How long the hand HOLDS the handle once it is on it before the motor is
/// told to open — the moment the latch is pulled, and the window the gate's
/// two-centimetre law is read over.
pub const HANDLE_HOLD_S: f64 = 0.10;

/// How long the step back takes, seconds, from the take to the point clear of
/// the leaf.
pub const STEP_BACK_S: f64 = 0.30;

/// **How far the door has to be open before the body goes through it**,
/// degrees of hinge — read off VEH3c's joint, not assumed from a clock.
pub const DOOR_BOARD_DEG: f64 = 45.0;

/// **How closed a door has to be to count as shut**, degrees.
pub const DOOR_SHUT_DEG: f64 = 3.0;

/// The longest the door phase may wait for the hinge, seconds — a door that is
/// jammed (a hinge the motor cannot move, a car on its side) is boarded through
/// whatever opening it has rather than waited on for ever.
pub const OPENING_MAX_S: f64 = 1.60;

/// The longest a CARJACK waits at the door for the seat to be empty, seconds —
/// past it the pull failed, the door is shut again and the press is refused.
pub const PULL_WAIT_MAX_S: f64 = 1.40;

/// The shortest the settle may be, seconds: the body lands and the hands find
/// the rim.
pub const SEATED_MIN_S: f64 = 0.25;

/// The longest the settle may wait for the door to shut, seconds.
pub const SEATED_MAX_S: f64 = 1.40;

/// How long the exit warp takes, seconds — the reverse of the enter's own
/// window, and deliberately shorter: getting out is a push, getting in is a
/// climb.
pub const EXIT_WARP_S: f64 = 0.45;

/// The longest the exit may wait for its door, seconds, before it leaves
/// through whatever opening there is.
pub const EXIT_DOOR_MAX_S: f64 = 1.00;

/// The longest the door may take to be pushed shut behind a body that has got
/// out, seconds.
pub const CLOSING_MAX_S: f64 = 1.20;

/// How long a victim is held braking before the pull, seconds, at most — the
/// hero's own door phase decides when the pull happens; this is the bound on a
/// hero that never arrives.
pub const JACKED_MAX_S: f64 = 4.0;

/// **Above this the exit is a fall, not a step**, m/s.
///
/// P29.7's own number, hoisted out of `step_driving`'s `linvel.length() > 2.0`
/// so that the exit pipeline, the carjack and the gate all read one constant.
/// Above it the body leaves in [`crate::components::MovementMode::FallControlled`]
/// — which is the mode CHAR1b.2's **roll** clip is reached through, and the
/// reason the arc brief calls a moving exit *"the roll"*: the landing
/// classifier turns a controlled fall that arrives fast into
/// [`crate::components::LandingKind::Rolling`] and the machine plays
/// `land_roll`.
pub const EXIT_ROLL_MPS: f64 = 2.0;

/// **The successor** of an enter phase, or `None` at the end of the sequence.
pub fn next_enter_phase(p: BoardPhase) -> Option<BoardPhase> {
    let at = BoardPhase::ENTER_ORDER.iter().position(|q| *q == p)?;
    BoardPhase::ENTER_ORDER.get(at + 1).copied()
}

/// How long an approach of `len_m` metres takes at [`APPROACH_MPS`], clamped
/// to `[UNLOCK_MIN_S, UNLOCK_MAX_S]`.
pub fn approach_s(len_m: f64) -> f64 {
    if !len_m.is_finite() {
        return UNLOCK_MAX_S;
    }
    (len_m / APPROACH_MPS).clamp(UNLOCK_MIN_S, UNLOCK_MAX_S)
}

/// **The hand weight a phase asks for**, `[0, 1]`, before reach is priced in.
///
/// * `OpeningDoor` — `0 → 1` over [`HAND_REACH_S`] (the doc's *"Blend IK Weight
///   (0.0 -> 1.0) over 0.2s"*), held while the latch is pulled.
/// * `Seated`, `Driving`, `Exiting`, `ClosingDoor` — `0 → 1` over the same
///   ramp from the phase's start.
/// * everything else — `0`.
///
/// Linear, and a pure function of the phase clock, so the gate can compute the
/// weight a step was owed without asking the solver.
pub fn phase_hand_weight(phase: BoardPhase, time_s: f64) -> f64 {
    let ramp = (time_s / HAND_REACH_S).clamp(0.0, 1.0);
    match phase {
        BoardPhase::OpeningDoor
        | BoardPhase::Seated
        | BoardPhase::Driving
        | BoardPhase::Exiting
        | BoardPhase::ClosingDoor => ramp,
        _ => 0.0,
    }
}

/// **How much of a hand request survives the reach**, `[0, 1]`: one inside the
/// arm, falling to zero [`REACH_FADE_M`] past it.
///
/// A hand told to go somewhere its shoulder cannot reach has exactly one pose
/// — the arm locked straight at it — and that is the pose a viewer reads as
/// wrong. So a target that has left the arm's envelope (a door that has swung
/// away, a handle a body has stepped back from) is let go of, smoothly, rather
/// than chased.
pub fn reach_weight(shoulder_to_target_m: f64, reach_m: f64) -> f64 {
    if !(shoulder_to_target_m.is_finite() && reach_m.is_finite()) || reach_m <= 0.0 {
        return 0.0;
    }
    let over = shoulder_to_target_m - reach_m;
    if over <= 0.0 {
        1.0
    } else {
        (1.0 - over / REACH_FADE_M).clamp(0.0, 1.0)
    }
}

/// How far past the arm's reach a hand request fades out over, metres.
pub const REACH_FADE_M: f64 = 0.12;

/// **One character's boarding**, live — never serialized, never reflected.
///
/// It rides [`crate::components::MovementRuntime`] beside
/// [`crate::components::SeatState`] and `CoverState`, which is `#[serde(skip)]`
/// on `CharacterMovement`: the machine is a per-step derivation and an
/// author's document has no business holding one. That is also what makes this
/// wave a **zero-schema** wave, which is the law it runs under — VEH3a's window
/// is spent.
///
/// The two ground points are held in the CHASSIS frame and turned into world
/// points every step from the live chassis pose, so a car nudged while a body
/// walks up to it is still boarded at its door rather than at where its door
/// was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoardingState {
    /// Where in the machine this body is.
    pub phase: BoardPhase,
    /// The chassis it is boarding or leaving, or `Uuid::nil`.
    pub vehicle: Uuid,
    /// Which seat — [`SeatIndex::as_u8`].
    pub seat: u8,
    /// Seconds into the CURRENT phase.
    pub time_s: f64,
    /// How long the current phase lasts, seconds, when that is known up front.
    /// `0` is open-ended: the phase ends on a WORLD fact (a hinge angle, an
    /// empty seat) inside its own `*_MAX_S` bound.
    pub phase_len_s: f64,
    /// The door part this boarding is using, or `Uuid::nil` — a car with no
    /// door, or one whose door is in the road.
    pub door: Uuid,
    /// The hand-IK weight asked for last step, `[0, 1]` — the phase ramp times
    /// the reach.
    pub hand_weight: f64,
    /// Which hand is on the handle: `0` left, `1` right.
    pub hand_side: u8,
    /// The TAKE — where the approach ends and the hand takes the handle.
    /// Chassis frame, metres; `y` unused.
    pub take_local: Vec3d,
    /// The STEP-BACK — clear of the door's swept disc. Chassis frame, metres.
    pub back_local: Vec3d,
    /// The ground under the take, world `y`, metres — found once by a ray.
    pub ground_y: f64,
    /// The facing the approach arrives at, degrees: the flank's inward normal.
    pub stance_yaw_deg: f64,
    /// Where the approach (or the exit warp) began, world metres — the CAPSULE
    /// centre, which is the character's transform.
    pub start: Vec3d,
    /// Its facing then, degrees.
    pub start_yaw_deg: f64,
    /// The Hermite's start tangent, world metres (already scaled).
    pub tangent_start: Vec3d,
    /// Its end tangent, same units — the chassis-frame inward normal, scaled,
    /// re-derived each step from the live chassis rotation.
    pub tangent_end_len: f64,
    /// **Whether this boarding is a CARJACK** — the same pipeline, with the
    /// victim pulled out through the door it opens.
    pub carjack: bool,
    /// The victim a carjack is pulling out, or the hero a `Jacked` body is
    /// being pulled out by. `Uuid::nil` otherwise.
    pub other: Uuid,
    /// When a sub-step inside the current phase began, seconds into it; `-1`
    /// until it has. The door phase's pull, the exit's warp.
    pub mark_s: f64,
    /// The pelvis dip this body needs to put a hand on the handle, metres.
    pub dip_m: f64,
    /// The hinge angle the door stood at last step, degrees — VEH3c's joint,
    /// read back. The instrument `hero.csv` carries.
    pub door_deg: f64,
    /// **How far the hand joint ended from its target last step**, metres.
    /// Measured off the solved pose by the hand pass's own report, never
    /// predicted: it is the number the gate's two-centimetre law reads and the
    /// number `hero.csv` carries.
    pub hand_err_m: f64,
    /// The same for the worst foot against its pedal (or floor) goal, metres.
    pub foot_err_m: f64,
    /// **The throttle this body's seat step sent the car**, `[0, 1]` — what the
    /// right foot presses. Recorded from the same `VehicleControls` the car was
    /// given, so the pedal and the engine cannot disagree. Not folded: it is a
    /// function of the intent, which is.
    pub throttle_in: f64,
    /// The brake, `[0, 1]` — what the left foot presses.
    pub brake_in: f64,
    /// **This body bailed out of a moving car** and has not landed yet — the
    /// next landing is a ROLL whatever its vertical speed (wave VEH3d).
    ///
    /// The landing classifier keys on the VERTICAL impact (`land_hard_mps`,
    /// 7 m/s), and a body thrown out of a car door lands at the car's speed
    /// sideways and a metre's fall downward — a soft landing by that rule, so
    /// CHAR1b.2's `land_roll` would never play for the one case the arc brief
    /// names it for. The flag is the exit's own record: set by the moving
    /// exit, consumed by the first landing, and quiet in the trace (it rides an
    /// `Idle` phase; the mode it produces is folded everywhere a mode is).
    pub bail: bool,
}

impl Default for BoardingState {
    fn default() -> Self {
        Self {
            phase: BoardPhase::Idle,
            vehicle: Uuid::nil(),
            seat: 0,
            time_s: 0.0,
            phase_len_s: 0.0,
            door: Uuid::nil(),
            hand_weight: 0.0,
            hand_side: 1,
            take_local: Vec3d::ZERO,
            back_local: Vec3d::ZERO,
            ground_y: 0.0,
            stance_yaw_deg: 0.0,
            start: Vec3d::ZERO,
            start_yaw_deg: 0.0,
            tangent_start: Vec3d::ZERO,
            tangent_end_len: 0.0,
            carjack: false,
            other: Uuid::nil(),
            mark_s: -1.0,
            dip_m: 0.0,
            door_deg: 0.0,
            hand_err_m: 0.0,
            foot_err_m: 0.0,
            throttle_in: 0.0,
            brake_in: 0.0,
            bail: false,
        }
    }
}

impl BoardingState {
    /// **Nothing is happening that the trace needs to see.** `Idle`, and
    /// `Driving` — a body at the wheel is a SEAT fact (`SeatState` and the
    /// transform section already carry it) and a drive of ten minutes must not
    /// fold ten minutes of a phase clock that nothing reads. The fold is the
    /// PHASE alone on purpose: a stale stance node on an idle body must not keep
    /// a level permanently non-empty ([`crate::vehicle::DrivetrainState::is_quiet`]'s
    /// own ruling).
    pub fn is_quiet(&self) -> bool {
        matches!(self.phase, BoardPhase::Idle | BoardPhase::Driving)
    }

    /// The seat, decoded.
    pub fn seat_index(&self) -> SeatIndex {
        SeatIndex::from_u8(self.seat)
    }

    /// How far through the current phase, `[0, 1]`. An open-ended phase is 0.
    pub fn alpha(&self) -> f64 {
        if self.phase_len_s <= 0.0 {
            return 0.0;
        }
        (self.time_s / self.phase_len_s).clamp(0.0, 1.0)
    }

    /// Whether the current phase has run its length.
    pub fn expired(&self) -> bool {
        self.phase_len_s > 0.0 && self.time_s >= self.phase_len_s
    }

    /// Move to `phase` with `len_s` on its clock, resetting the phase time and
    /// the sub-step mark.
    pub fn enter(&mut self, phase: BoardPhase, len_s: f64) {
        self.phase = phase;
        self.time_s = 0.0;
        self.mark_s = -1.0;
        self.phase_len_s = if len_s.is_finite() && len_s > 0.0 {
            len_s
        } else {
            0.0
        };
    }
}

/// **A cubic Hermite** through two points and two tangents (the doc's approach
/// spline).
///
/// `h00 p0 + h10 m0 + h01 p1 + h11 m1` with the standard basis. Portable: four
/// polynomials in `t` and nothing transcendental, so the curve a body walks is
/// bit-identical on every platform this engine ships to — P14's law, which is
/// why the approach is a Hermite and not a slerp.
pub fn hermite(p0: Vec3d, m0: Vec3d, p1: Vec3d, m1: Vec3d, t: f64) -> Vec3d {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    Vec3d::new(
        h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x,
        h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y,
        h00 * p0.z + h10 * m0.z + h01 * p1.z + h11 * m1.z,
    )
}

/// How many segments [`hermite_len`] measures a curve over.
///
/// Sixteen, because the curve is at most three metres long and a chord error
/// under a millimetre is what "walk speed" has to mean for a duration to be
/// honest. Fixed rather than adaptive so the answer is a pure function of the
/// endpoints on every platform.
pub const HERMITE_SEGMENTS: usize = 16;

/// **How long a Hermite is**, metres — the polyline through
/// [`HERMITE_SEGMENTS`] chords.
pub fn hermite_len(p0: Vec3d, m0: Vec3d, p1: Vec3d, m1: Vec3d) -> f64 {
    let mut prev = p0;
    let mut len = 0.0;
    for i in 1..=HERMITE_SEGMENTS {
        let t = i as f64 / HERMITE_SEGMENTS as f64;
        let at = hermite(p0, m0, p1, m1, t);
        let (dx, dy, dz) = (at.x - prev.x, at.y - prev.y, at.z - prev.z);
        len += (dx * dx + dy * dy + dz * dz).sqrt();
        prev = at;
    }
    len
}

/// **How far a pelvis has to dip for a handle to be in reach**, metres.
///
/// A door handle is about 0.70 m off the road and a standing shoulder is about
/// 1.45 m, so an upright body with a straight arm cannot touch one. Derived
/// from the character's own standing height rather than tabled, so a 1.2 m
/// character bends as much as it needs to and no more, and clamped at
/// [`MAX_REACH_DIP_M`] — past which the answer is "this body cannot reach that
/// handle", which is a refusal and not a limbo.
///
/// * `standing_m` — the body's full standing height (capsule, doubled).
/// * `feet_y`, `handle_y`, `plan_m` — where it is standing, how high the handle
///   is, and how far away on the ground.
pub fn reach_dip_m(standing_m: f64, feet_y: f64, handle_y: f64, plan_m: f64) -> f64 {
    if !(standing_m.is_finite() && standing_m > 0.0) {
        return 0.0;
    }
    let shoulder_y = feet_y + SHOULDER_FRAC * standing_m;
    let reach = ARM_REACH_FRAC * standing_m;
    let dy = shoulder_y - handle_y;
    let d = (plan_m * plan_m + dy * dy).sqrt();
    if d <= reach {
        return 0.0;
    }
    // How far the shoulder has to come down for the same plan distance to be
    // inside the arm: the vertical leg of a right triangle whose hypotenuse is
    // the reach.
    let want_dy = (reach * reach - plan_m * plan_m).max(0.0).sqrt();
    (dy - want_dy).clamp(0.0, MAX_REACH_DIP_M)
}

// ── the trace (the 19th section) ────────────────────────────────────────────

/// How many bytes one BOARDING folds: 16 of guid, a phase, a seat, a hand side,
/// 16 of the vehicle, a carjack flag, 16 of the other party and six `f64`.
pub const BOARDING_TRACE_BYTES: usize = 16 + 1 + 1 + 1 + 16 + 1 + 16 + 6 * 8;

/// **The boarding machine's trace bytes** (wave VEH3d) — the determinism
/// trace's **nineteenth** section, at the tail, after VEH3c's bodywork.
///
/// **EMPTY on a level with nobody boarding**, which is every level committed
/// before this wave and every step of a level whose hero is walking or
/// driving: the fold walks the characters and skips every
/// [`BoardingState::is_quiet`] one, so a world where nobody has pressed E
/// produces a zero-length vec and every committed trace hash stays
/// byte-identical.
///
/// It is folded rather than trusted to the transform section for the reason
/// wave WPN2b gives for folding a spring: two hosts that disagreed about a
/// PHASE agree about every position in the world for as long as the phase's
/// clock is still running, and then one of them is in a car and the other is
/// standing in the road.
pub fn boarding_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let mut rows: BTreeMap<Uuid, BoardingState> = BTreeMap::new();
    for guid in crate::movement::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(cm) = world.world().get::<crate::components::CharacterMovement>(e) else {
            continue;
        };
        if cm.runtime.boarding.is_quiet() {
            continue;
        }
        rows.insert(guid, cm.runtime.boarding);
    }
    if rows.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(rows.len() * BOARDING_TRACE_BYTES);
    for (guid, b) in rows {
        out.extend_from_slice(guid.as_bytes());
        out.push(b.phase.as_u8());
        out.push(b.seat);
        out.push(b.hand_side);
        out.extend_from_slice(b.vehicle.as_bytes());
        out.push(u8::from(b.carjack));
        out.extend_from_slice(b.other.as_bytes());
        for v in [
            b.time_s,
            b.phase_len_s,
            b.hand_weight,
            b.stance_yaw_deg,
            b.mark_s,
            b.dip_m,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
}

/// **What a boarding HUD row says** — the sibling of
/// `inf_ecs::vehicle::drive_readout` and `inf_ecs::bodywork::damage_readout`.
///
/// `BOARDING opening  SEAT driver  HAND 0.014 m  DOOR 32 deg`, and nothing at
/// all while a body is neither boarding nor driving.
pub fn boarding_readout(b: &BoardingState) -> Option<String> {
    if b.phase == BoardPhase::Idle {
        return None;
    }
    let mut row = format!(
        "BOARDING {}  SEAT {}",
        b.phase.name(),
        b.seat_index().name()
    );
    if b.hand_weight > 0.0 {
        row.push_str(&format!("  HAND {:.3} m", b.hand_err_m));
    }
    if b.phase == BoardPhase::Driving {
        row.push_str(&format!("  PEDAL {:.3} m", b.foot_err_m));
    }
    if !b.door.is_nil() {
        row.push_str(&format!("  DOOR {:.0} deg", b.door_deg));
    }
    if b.carjack {
        row.push_str("  CARJACK");
    }
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEDAN_HALF: Vec3d = Vec3d::new(0.92, 0.62, 2.2);

    fn sedan_doors() -> Vec<PartGeom> {
        vec![
            PartGeom {
                kind: crate::vehicle::KIND_DOOR,
                centre_frac: Vec3d::new(-0.955, -0.4, 0.3),
                half_frac: Vec3d::new(0.045, 0.36, 0.26),
            },
            PartGeom {
                kind: crate::vehicle::KIND_DOOR,
                centre_frac: Vec3d::new(0.955, -0.4, 0.3),
                half_frac: Vec3d::new(0.045, 0.36, 0.26),
            },
            PartGeom {
                kind: crate::vehicle::KIND_DOOR,
                centre_frac: Vec3d::new(0.955, -0.4, -0.22),
                half_frac: Vec3d::new(0.045, 0.34, 0.22),
            },
        ]
    }

    #[test]
    fn the_seat_is_inside_the_cabin_and_not_on_the_roof() {
        let s = sockets_of(SEDAN_HALF, Vec3d::ZERO, &[]);
        // The roof is `+half.y`; the cushion and the floor are both below it.
        assert!(
            s.seat_r.y < SEDAN_HALF.y,
            "the cushion is at {:.3} and the roof at {:.3}",
            s.seat_r.y,
            SEDAN_HALF.y
        );
        let floor = SEAT_FLOOR_FRAC_Y * SEDAN_HALF.y;
        assert!(floor < s.seat_r.y, "the floor is below the cushion");
        assert!(
            SEDAN_HALF.y - floor > 0.5,
            "a driver's feet are half a metre below its own roof, not on it"
        );
        // …and the driver is on the `+X` side, which is the exit's side.
        assert!(s.seat_r.x > 0.0 && s.seat_l.x < 0.0);
        assert!(s.seat_rear.z < s.seat_r.z, "the bench is behind the seats");
    }

    #[test]
    fn the_handles_come_off_the_doors_when_there_are_doors() {
        let bare = sockets_of(SEDAN_HALF, Vec3d::ZERO, &[]);
        let with = sockets_of(SEDAN_HALF, Vec3d::ZERO, &sedan_doors());
        assert_ne!(
            bare.door_handle_r, with.door_handle_r,
            "a door moves the handle off the flank fallback"
        );
        // The FRONT door, not the rear: the two are at z = +0.3 and -0.22.
        assert!(with.door_handle_r.z > 0.0);
        // Outboard of the door's own centre, and two thirds up it.
        assert!(with.door_handle_r.x > 0.955 * SEDAN_HALF.x);
        assert!(with.door_handle_r.y > -0.4 * SEDAN_HALF.y);
        assert!(with.door_handle_l.x < 0.0);
    }

    #[test]
    fn every_socket_name_round_trips() {
        let mut s = sockets_of(SEDAN_HALF, Vec3d::ZERO, &sedan_doors());
        for (i, name) in VehicleSockets::names().iter().enumerate() {
            let at = Vec3d::new(i as f64, 2.0 * i as f64, 3.0 * i as f64);
            assert!(s.set(name, at), "`{name}` is not a socket the door writes");
            assert_eq!(s.get(name), Some(at), "`{name}` does not read back");
        }
        assert!(!s.set("not_a_socket", Vec3d::ZERO));
        assert!(s.get("not_a_socket").is_none());
        assert!(!s.set("seat_r", Vec3d::new(f64::NAN, 0.0, 0.0)));
    }

    #[test]
    fn the_hermite_hits_its_endpoints_and_has_a_length() {
        let p0 = Vec3d::new(0.0, 0.0, 0.0);
        let p1 = Vec3d::new(3.0, 0.0, 1.0);
        let m0 = Vec3d::new(0.0, 0.0, 2.0);
        let m1 = Vec3d::new(2.0, 0.0, 0.0);
        assert_eq!(hermite(p0, m0, p1, m1, 0.0), p0);
        assert_eq!(hermite(p0, m0, p1, m1, 1.0), p1);
        let straight = (3.0f64 * 3.0 + 1.0).sqrt();
        let curved = hermite_len(p0, m0, p1, m1);
        assert!(
            curved > straight,
            "a curve through two tangents is longer than the chord: {curved:.4} vs {straight:.4}"
        );
    }

    #[test]
    fn the_dip_is_zero_when_the_handle_is_already_in_reach() {
        // A 1.8 m body standing 0.38 m from a handle 1.30 m up: the shoulder is
        // at 1.476, the drop is 0.176 and the reach is 0.648 — no dip.
        assert_eq!(reach_dip_m(1.8, 0.0, 1.30, 0.38), 0.0);
        // …and the same body at a car's own handle height has to bend.
        let dip = reach_dip_m(1.8, 0.0, 0.70, 0.38);
        assert!(dip > 0.0 && dip <= MAX_REACH_DIP_M, "dip {dip:.3}");
    }

    #[test]
    fn the_enter_sequence_is_six_phases_in_order() {
        let mut p = BoardPhase::Locked;
        let mut seen = vec![p];
        while let Some(n) = next_enter_phase(p) {
            seen.push(n);
            p = n;
        }
        assert_eq!(seen, BoardPhase::ENTER_ORDER.to_vec());
        assert_eq!(p, BoardPhase::Driving);
        // The wire numbers are frozen and unique.
        let mut codes: Vec<u8> = [
            BoardPhase::Idle,
            BoardPhase::Locked,
            BoardPhase::Unlocking,
            BoardPhase::OpeningDoor,
            BoardPhase::EnteringIK,
            BoardPhase::Seated,
            BoardPhase::Driving,
            BoardPhase::Exiting,
            BoardPhase::ClosingDoor,
        ]
        .iter()
        .map(|p| p.as_u8())
        .collect();
        codes.sort_unstable();
        assert_eq!(codes, (0u8..=8).collect::<Vec<_>>());
    }
}
