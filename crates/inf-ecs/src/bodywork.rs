//! **A CAR'S BODYWORK AND WHAT HAPPENS TO IT** (wave VEH3c): the hinges its
//! doors swing on, the impulse that tears them off, the panes that shatter, the
//! dents, the hull's own joules, a dead engine, a flat tyre and a fire.
//!
//! # Where the state lives, and why it is not on the wire
//!
//! In a bevy **resource** ([`VehicleDamageRes`]), keyed by chassis `Guid`, on
//! the `DrivetrainRes` / `DeformFieldRes` precedent and for the same three
//! reasons. VEH3a's schema window is spent — scene v28, `ScenePayload` 13 — and
//! this wave does not reopen it; a component would move every committed level
//! that has a car in it; and damage is a *session's* fact, not an author's, so
//! a level that opened with its doors hanging off would be a level nobody
//! authored.
//!
//! What IS on the wire is the 18th section of the determinism trace
//! ([`damage_state_bytes`]), at the tail, after VEH3b's drivetrain — and
//! **empty until something breaks**, so every quiet level's bytes are what they
//! were.
//!
//! # The three regimes, and the one rule
//!
//! **KINEMATIC UNTIL OPENED OR HIT**, and then the solver's.
//!
//! * **Latched** — a drawn child of the chassis with no `RigidBody3D`, no
//!   `Collider3D`, no rapier body and no joint. Its hinge is integrated here, in
//!   three lines of arithmetic ([`hinge_step`]), and the answer is written onto
//!   its own local `Transform`. A thousand parked cars are a thousand cars in
//!   this state and they cost the solver **nothing** — which is the whole reason
//!   the body is built lazily rather than at spawn.
//! * **Live** — a rapier body of its own on a REAL revolute back to the chassis,
//!   with the hinge's limits, a position motor and its contacts against that
//!   chassis turned off. It is a ROOT entity from that moment (the bridge
//!   mirrors a body at its entity's local transform), and what holds it on is
//!   the joint. `set_part_open` drives the motor; the joint's own impulse is
//!   watched every step by `inf_physics::d3::PhysicsWorld3D::joint_impulse`, and
//!   over [`hinge_tear_ns`] `remove_joint` lets it go.
//! * **Shed** — the same body with the joint gone: free, thrown clear along the
//!   axis it faces, reaped at [`PART_DEBRIS_LIFETIME_S`] under
//!   [`MAX_SHED_PARTS`]. A car can run it over, because it is a thing that is
//!   there.
//!
//! The rule that moves a part between them is one function of one number: how
//! much of a blow reached its mounts. Under `LATCH_POP_FRAC` of the threshold
//! nothing happens; over it a hinged part POPS (it goes live, and swings); over
//! the threshold itself it SHEDS. A bumper has no hinge, so for a bumper there
//! is no middle state and the two branches are the same branch — and a bumper is
//! therefore never on a joint, only ever latched or free.
//!
//! **The two break paths read two different impulses**, which is a distinction
//! the audit had to make: the CRASH path compares a share of the whole car's
//! blow against `part_break_impulse_ns`, and the JOINT path compares what the
//! part's own hinge carried against [`hinge_tear_ns`]. A 22 kg door can deliver
//! 183 N.s at 30 km/h and a 60 km/h shunt puts nine thousand through the same
//! door's load path; one number could not have served both.
//!
//! # What is deliberately NOT here
//!
//! The LOOK. `AdvancedRealisticGlass`, a shatter pattern, sparks, smoke off a
//! dead engine, a dent normal map — every one of those is the PAR arc's, and
//! this module owns the geometry and the state they will be drawn from.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;
use uuid::Uuid;

use crate::components::VehicleClass;
use crate::math::Vec3d;
use crate::vehicle::{BodyPartKind, VehicleTuning};
use crate::world::EcsWorld;

// ── the numbers ─────────────────────────────────────────────────────────────

/// **How many panels' worth of joules a whole car's hull is worth.**
///
/// Four. `panel_health_j` is what ONE body panel absorbs before it is written
/// off, and a car is not written off by the first panel that is: the default
/// 9 000 J a panel makes a 36 000 J car, which is about seventy pistol rounds
/// or twenty rifle ones. That is a car you have to *mean* to destroy, which is
/// the reference footage's own answer.
pub const HULL_PANELS: f64 = 4.0;

/// **How fast a part has to be STOPPED for its hinge to let go**, m/s — at the
/// catalogue's own default mount (wave VEH3c's audit).
///
/// # Why a hinge needs its own number, and it is not `part_break_impulse_ns`
///
/// The two break paths read impulses of two different KINDS and the audit found
/// them being compared to one number:
///
/// * the CRASH path asks what share of the whole car's blow reached a part's
///   mounts — `J · face · impact_share`, where `J` is the CHASSIS' impulse. A
///   60 km/h shunt is 18 879 N.s, and half of that through a door's load path is
///   nine thousand. Against a 4 500 N.s mount it tears, which is right.
/// * the JOINT path asks what the part's OWN hinge carried. A 22 kg door hitting
///   a lamp post at 30 km/h can deliver `m·v` = **183 N.s** and no more — it is
///   a door, not a car. Measured at 68 km/h it reaches **515**, against **2** in
///   open air. Compared to the same 4 500 it never breaks, and an open door
///   would be indestructible by anything smaller than the car it is on.
///
/// So a hinge tears at a SPEED: four metres a second of the part's own momentum,
/// which is a door that hits something at walking-to-jogging pace and loses.
/// Cornering at half a g puts 1.8 N.s through a 22 kg door in a step, so the
/// margin against normal driving is about fifty to one.
///
/// **The tuning still governs it.** [`hinge_tear_ns`] scales this by the row's
/// own `part_break_impulse_ns` against the default, so a car tuned UNBREAKABLE
/// is unbreakable on both paths and a car tuned fragile loses its doors on both.
pub const HINGE_TEAR_MPS: f64 = 4.0;

/// **What one part's hinge lets go at**, newton-seconds — see
/// [`HINGE_TEAR_MPS`].
///
/// `mass · HINGE_TEAR_MPS`, scaled by how far the row's own mount is from the
/// catalogue default. A non-positive or non-finite mount is UNBREAKABLE, which
/// is the same refusal-as-a-value `BreakWatch3D` makes of its own threshold.
pub fn hinge_tear_ns(part_mass_kg: f64, part_break_impulse_ns: f64) -> f64 {
    if !part_break_impulse_ns.is_finite() || part_break_impulse_ns <= 0.0 {
        return f64::INFINITY;
    }
    let default = crate::vehicle::VehicleTuning::default().part_break_impulse_ns;
    if default <= 0.0 {
        return f64::INFINITY;
    }
    (part_mass_kg.max(0.1) * HINGE_TEAR_MPS * (part_break_impulse_ns / default)).max(1.0)
}

/// **The fraction of `part_break_impulse_ns` at which a latch POPS** rather than
/// tearing off.
///
/// Under half. A bonnet that takes a real blow and does not come off is a
/// bonnet standing open, which is what a crashed car looks like and what makes
/// "the bumper came off and the bonnet popped" one rule instead of two.
pub const LATCH_POP_FRAC: f64 = 0.45;

/// **How deep a dent may ever be**, metres — a cap, so a panel can never be
/// pushed through the far side of its own car.
pub const MAX_DENT_M: f64 = 0.22;

/// **How far a panel is pushed in per kilonewton-second of blow**, metres.
///
/// Sized against the crash it has to read on: a 60 km/h wall crash puts about
/// 19 kN.s through the nose, which at this rate is 0.19 m of dent — just inside
/// [`MAX_DENT_M`], so the cap is a bound on the absurd rather than the thing
/// that sets the depth.
pub const DENT_M_PER_KNS: f64 = 0.010;

/// **What a flat tyre's rolling radius is**, as a fraction of the inflated one.
///
/// Three quarters — the research doc's own −25 %. It is what makes the car sit
/// down on that corner, and it is read by the wheel ray, so a flat is visible
/// before it is felt.
pub const FLAT_RADIUS_FRAC: f64 = 0.75;

/// **What a flat tyre's grip is**, as a fraction of the inflated one — −40 %.
pub const FLAT_MU_FRAC: f64 = 0.60;

/// **How close a round has to land to a wheel to flatten it**, as a multiple of
/// that wheel's own radius.
pub const FLAT_HIT_RADII: f64 = 1.25;

/// **How close a round has to land to a pane, in metres, to be a hit on it.**
///
/// A pane is drawn as a thin box and the round stops on the CHASSIS collider,
/// which is the car's outer box — so the impact point is on the skin and the
/// pane is a few centimetres inside it. Quarter of a metre is the band that
/// catches a screen without catching the roof above it.
pub const PANE_HIT_M: f64 = 0.25;

/// **The share of the front of the car that is engine**, as a fraction of the
/// chassis half-length measured from the nose.
///
/// A third. A round through the grille or the bonnet is in the engine bay; one
/// through a door is not.
pub const ENGINE_BAY_FRAC: f64 = 0.34;

/// **The hinge's position-motor stiffness**, per second squared, in DEGREE
/// space.
///
/// With [`HINGE_DAMPING`] it is a damping ratio of 0.77 and a natural period of
/// 0.81 s, so a door commanded open reaches its limit in a little over half a
/// second and does not bounce off it. Degrees rather than radians because the
/// ODE is linear and the unit cancels — and because every other angle on a
/// `Transform` in this engine is in degrees.
pub const HINGE_STIFFNESS: f64 = 60.0;
/// The hinge's velocity damping, per second — see [`HINGE_STIFFNESS`].
pub const HINGE_DAMPING: f64 = 12.0;
/// The most a hinge's motor may accelerate a part, degrees per second squared.
///
/// This is `JointMotor3D::max_force` in the analytic regime: it is what stops a
/// door commanded from shut to fully open from leaving at a speed no hinge
/// could deliver.
pub const HINGE_ACCEL_MAX_DEG_S2: f64 = 2_400.0;

/// **How long a shed part lies where it fell**, seconds.
///
/// `inf_physics::d3::fracture::DEFAULT_DEBRIS_LIFETIME_S`, restated here rather
/// than imported because Ring 0's `inf-ecs` does not depend on `inf-physics` —
/// and pinned to it by
/// `a_shed_part_lives_as_long_as_a_piece_of_p22_debris`.
pub const PART_DEBRIS_LIFETIME_S: f64 = 20.0;

/// **The most shed parts a level may have lying about at once.**
///
/// A quarter of `DEFAULT_DEBRIS_MAX_LIVE` (256), because a shed part is a whole
/// drawn panel with its own body rather than a fracture chunk, and because a
/// pile-up on the island is a dozen cars and not a hundred. Oldest first.
pub const MAX_SHED_PARTS: usize = 64;

/// **How many shards a shattered pane throws.**
pub const GLASS_SHARDS: usize = 6;
/// **How long a glass shard is drawn for**, seconds.
pub const GLASS_SHARD_LIFETIME_S: f64 = 1.6;
/// **The most glass shards a level draws at once** — the dispatcher's `MAX_PUFFS`
/// with a car's worth of panes behind it.
pub const MAX_GLASS_SHARDS: usize = 96;

// ── the state ───────────────────────────────────────────────────────────────

/// **Where one part of a body is in its own life** (wave VEH3c).
///
/// Frozen wire numbers, append only: this reaches the determinism trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PartLatch {
    /// Bolted on and drawn as a child of the chassis. No body, no collider, no
    /// joint — the state a thousand parked cars are in.
    #[default]
    Latched,
    /// **On its own hinge**: a root entity with a rapier body and a real
    /// revolute back to the chassis, watched for its own break.
    ///
    /// `Live` means *open*, and it is the state a part enters lazily — on the
    /// first `set_part_open` or the first blow past `LATCH_POP_FRAC` of its
    /// mount. See this module's own "three regimes" note.
    Live,
    /// Off the car — the same body with its joint gone, free, reaped at
    /// [`PART_DEBRIS_LIFETIME_S`] under [`MAX_SHED_PARTS`]. A car can run it
    /// over; see [`Debris`] for the audit that made that true.
    Shed,
    /// Gone. A pane that shattered leaves nothing to fall.
    Gone,
}

impl PartLatch {
    /// The frozen wire number.
    pub fn as_u8(self) -> u8 {
        match self {
            PartLatch::Latched => 0,
            PartLatch::Live => 1,
            PartLatch::Shed => 2,
            PartLatch::Gone => 3,
        }
    }

    /// A stable short name, for a gate trace and a diagnostic.
    pub fn name(self) -> &'static str {
        match self {
            PartLatch::Latched => "latched",
            PartLatch::Live => "live",
            PartLatch::Shed => "shed",
            PartLatch::Gone => "gone",
        }
    }

    /// Whether the part is still attached to the car at all.
    pub fn attached(self) -> bool {
        matches!(self, PartLatch::Latched | PartLatch::Live)
    }
}

/// **One part's live state** (wave VEH3c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartState {
    /// [`BodyPartKind::as_u8`] — what this part is.
    pub kind: u8,
    /// Where it is in its life.
    pub latch: PartLatch,
    /// The hinge angle, degrees. `0` is shut.
    pub angle_deg: f64,
    /// Its rate, degrees per second.
    pub vel_deg_s: f64,
    /// What the motor is driving it to, degrees.
    pub target_deg: f64,
    /// The joules this part has absorbed.
    pub damage_j: f64,
    /// How far the panel is pushed in, metres — the dent.
    pub dent_m: f64,
    /// The sim step it left the car, or `0`.
    pub shed_step: u64,
    /// **The part's AUTHORED centre**, in fractions of the chassis half-extents.
    ///
    /// **NOT FOLDED and not part of [`is_quiet`](Self::is_quiet)**, for
    /// [`VehicleDamage::last_vel`]'s reason: it is what the level says, not what
    /// happened to it. It is here rather than looked up in a parts table because
    /// the drawn pose is rebuilt from the baseline every step — a dent that
    /// pushed the transform it read next step would deepen for ever — and
    /// because a shipped player has no parts table to look it up in.
    pub centre_frac: Vec3d,
    /// The part's AUTHORED half-extents, same units. Not folded, for the same
    /// reason.
    pub half_frac: Vec3d,
}

impl Default for PartState {
    fn default() -> Self {
        Self {
            kind: crate::vehicle::KIND_PANEL,
            latch: PartLatch::Latched,
            angle_deg: 0.0,
            vel_deg_s: 0.0,
            target_deg: 0.0,
            damage_j: 0.0,
            dent_m: 0.0,
            shed_step: 0,
            centre_frac: Vec3d::ZERO,
            half_frac: Vec3d::ZERO,
        }
    }
}

impl PartState {
    /// A part with this kind, shut and undamaged.
    pub fn new(kind: BodyPartKind) -> Self {
        Self {
            kind: kind.as_u8(),
            ..Default::default()
        }
    }

    /// The same, over a part's authored geometry.
    pub fn authored(kind: BodyPartKind, centre_frac: Vec3d, half_frac: Vec3d) -> Self {
        Self {
            kind: kind.as_u8(),
            centre_frac,
            half_frac,
            ..Default::default()
        }
    }

    /// **Which way this part faces**, as an axis index and a sign — the axis of
    /// its own offset from the chassis centre with the biggest share of it.
    ///
    /// A door's is `X`, a bumper's is `Z`, a roof panel's is `Y`. It is what a
    /// dent is pushed along and what a crash's own direction is compared
    /// against, and deriving it beats authoring it: a family that moves a part
    /// moves which way it faces with it.
    pub fn facing(&self) -> (usize, f64) {
        let c = [self.centre_frac.x, self.centre_frac.y, self.centre_frac.z];
        let mut best = 0usize;
        for (i, v) in c.iter().enumerate() {
            if v.abs() > c[best].abs() {
                best = i;
            }
        }
        let sign = if c[best] < 0.0 { -1.0 } else { 1.0 };
        (best, sign)
    }

    /// **Nothing has happened to this part.**
    ///
    /// Exactly the set [`damage_state_bytes`] folds, which is not a coincidence:
    /// a fold carrying a field this test cannot see would go empty while that
    /// field was still moving (`DrivetrainState::is_quiet`'s own ruling).
    pub fn is_quiet(&self) -> bool {
        self.latch == PartLatch::Latched
            && self.angle_deg == 0.0
            && self.vel_deg_s == 0.0
            && self.target_deg == 0.0
            && self.damage_j == 0.0
            && self.dent_m == 0.0
            && self.shed_step == 0
    }
}

/// **One vehicle's damage** (wave VEH3c).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VehicleDamage {
    /// Its parts, keyed by the part entity's own `Guid` — a `BTreeMap`, so the
    /// fold's order is the level's contents and not a bevy archetype walk.
    pub parts: BTreeMap<Uuid, PartState>,
    /// The joules the hull has absorbed.
    pub hull_j: f64,
    /// The engine's damage, `[0, 1]`. `1` is a dead engine, and
    /// `1 - engine_damage` is the runtime scale on `max_engine_force_n`.
    pub engine_damage: f64,
    /// Which wheels are flat, by rig index — bit `i` is wheel `i`.
    pub flats: u8,
    /// The sim step this car caught fire, or `0`.
    pub fire_step: u64,
    /// **The chassis' velocity at the end of the last step**, m/s — what a
    /// crash's impulse is measured against.
    ///
    /// **NOT FOLDED, and not part of [`is_quiet`](Self::is_quiet).** It is a
    /// derivative of the solver's own state, which the trace's first section
    /// already carries entity by entity, and a car simply driving along would
    /// otherwise keep this section permanently non-empty — which is exactly the
    /// trap `DrivetrainState::clutch_slip_rad_s` is kept out of the VEH3b fold
    /// to avoid.
    pub last_vel: Vec3d,
    /// **Where the chassis was at the end of the last step**, world metres —
    /// what a TELEPORT is measured against.
    ///
    /// NOT FOLDED and not part of [`is_quiet`](Self::is_quiet), for
    /// [`last_vel`](Self::last_vel)'s reason exactly.
    pub last_pos: Vec3d,
    /// Whether [`last_vel`](Self::last_vel) has ever been written. The first
    /// step of a car's life has no previous velocity, and treating a standing
    /// start as a 0 m/s crash would be an impulse of exactly zero — harmless —
    /// while treating a car SPAWNED at speed as one would not.
    pub seen: bool,
    /// **How many of [`parts`](Self::parts) are NOT quiet** — a maintained
    /// count, so [`is_quiet`](Self::is_quiet) is O(1) rather than a walk over
    /// the whole map.
    ///
    /// # Why a count and not a walk
    ///
    /// `is_quiet` is the hot question of the bodywork step: it decides whether a
    /// car pays for a child walk, a hinge pass and a shed gather. Asked of a
    /// `BTreeMap` of fourteen parts, a thousand parked saloons walked
    /// twenty-eight thousand B-tree nodes a step to be told nothing had
    /// happened — and the parked-car arm measured **x1.097 against its x1.05
    /// ceiling in RELEASE**, which is the only configuration that ceiling is
    /// asserted in. (The wave reported x1.083 in debug and never ran the
    /// release number the ceiling governs.) Hoisting the second of the two
    /// questions took it to x1.041 and it still crossed the ceiling on two runs
    /// in eleven; this count is what takes it under for good.
    ///
    /// # It cannot go stale unnoticed
    ///
    /// Refreshed by [`refresh_parts`](Self::refresh_parts) after every burst of
    /// part writes, and `is_quiet` carries a `debug_assert_eq!` against the full
    /// walk — so every gate in this repository, all of which run in debug,
    /// falsifies a count somebody forgot to refresh. In release the assert is
    /// gone and the read is a `u32` compare.
    ///
    /// **NOT FOLDED and NOT PERSISTED**: it is a derivative of `parts`, which is
    /// folded part by part, and `VehicleDamage` lives on a bevy resource.
    pub loud_parts: u32,
}

impl VehicleDamage {
    /// **Nothing has happened to this car.**
    ///
    /// O(1): the parts' half is [`loud_parts`](Self::loud_parts), checked
    /// against the full walk by a `debug_assert` so a stale count reds every
    /// gate in the repository. See that field for the measurement.
    pub fn is_quiet(&self) -> bool {
        debug_assert_eq!(
            self.loud_parts as usize,
            self.parts.values().filter(|p| !p.is_quiet()).count(),
            "VehicleDamage::loud_parts is stale — a part was written without a refresh_parts() after it"
        );
        self.hull_j == 0.0
            && self.engine_damage == 0.0
            && self.flats == 0
            && self.fire_step == 0
            && self.loud_parts == 0
    }

    /// **Re-count the parts that are not quiet.** Call after any burst of writes
    /// to [`parts`](Self::parts) — see [`loud_parts`](Self::loud_parts).
    ///
    /// O(parts), and it runs on the cold path only: a part is written by a
    /// crash, a round, a shatter, a hinge that is moving or an author opening a
    /// door, and every one of those is a car that has already stopped being
    /// quiet.
    pub fn refresh_parts(&mut self) {
        self.loud_parts = self.parts.values().filter(|p| !p.is_quiet()).count() as u32;
    }

    /// The runtime scale on `max_engine_force_n`, `[0, 1]`.
    pub fn engine_scale(&self) -> f64 {
        (1.0 - self.engine_damage).clamp(0.0, 1.0)
    }

    /// Whether wheel `i` is flat.
    pub fn is_flat(&self, i: usize) -> bool {
        i < 8 && (self.flats >> i) & 1 == 1
    }

    /// Flatten wheel `i`. Answers whether it was not already flat.
    pub fn flatten(&mut self, i: usize) -> bool {
        if i >= 8 || self.is_flat(i) {
            return false;
        }
        self.flats |= 1 << i;
        true
    }

    /// How many wheels are flat.
    pub fn flat_count(&self) -> u32 {
        self.flats.count_ones()
    }

    /// How many panes have gone.
    pub fn panes_broken(&self) -> usize {
        self.parts
            .values()
            .filter(|p| p.kind == crate::vehicle::KIND_GLASS && p.latch == PartLatch::Gone)
            .count()
    }

    /// How many parts have left the car.
    pub fn parts_shed(&self) -> usize {
        self.parts
            .values()
            .filter(|p| p.latch == PartLatch::Shed)
            .count()
    }

    /// How many parts are standing open on a live hinge.
    pub fn parts_open(&self) -> usize {
        self.parts
            .values()
            .filter(|p| p.latch == PartLatch::Live)
            .count()
    }

    /// Whether this car is burning.
    pub fn burning(&self) -> bool {
        self.fire_step > 0
    }

    /// The hull's remaining health, `[0, 1]`, against a capacity.
    pub fn hull_frac(&self, capacity_j: f64) -> f64 {
        if capacity_j <= 0.0 {
            return 0.0;
        }
        (1.0 - self.hull_j / capacity_j).clamp(0.0, 1.0)
    }
}

/// **One piece of a car lying in the road** (wave VEH3c), and its clock.
///
/// # It IS a rapier body, and that took an audit
///
/// The wave shipped a shed panel as DRAWN debris the bodywork integrated itself
/// — `at += v·dt`, stop at the ground it was over — and refused to give it a
/// body on three measurements. Two of them were about the solver and are
/// answered (`JointDesc3D::without_contacts` for the overlap, a push clear along
/// the part's own facing axis for the spawn); the third, *"a door on a real
/// hinge, held to a chassis the dispatcher teleports, launched one to 1 705
/// metres"*, **did not reproduce** —
/// `joints3d::a_jointed_rig_survives_being_teleported_as_a_unit` measures the
/// same drag six ways and finds the chassis 1.8 to 6.3 mm off its own schedule
/// in every one. And the first of the two that stood was never a defect at all:
/// a bumper going under a responding ambulance's wheel rays is the POINT, and
/// `a_shed_bumper_in_the_road_is_run_over` measures the car driving over it.
///
/// So this record is no longer a pose. It is a guid and a CLOCK: the body is
/// the solver's, and what the bodywork still owes it is a lifetime
/// ([`PART_DEBRIS_LIFETIME_S`]) and a cap ([`MAX_SHED_PARTS`]).
///
/// It is on the RESOURCE and not in the trace's per-part rows for the reason it
/// always was: a piece of debris is a pose, and the pose reaches the trace
/// through the entity's own `Transform` in `sim_snapshot`'s first section,
/// exactly as every other drawn thing's does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Debris {
    /// The part entity — a rapier body of its own, off the hierarchy, lying
    /// where the solver put it.
    pub guid: Uuid,
    /// The step it left the car. What [`PART_DEBRIS_LIFETIME_S`] is measured
    /// from, and what the [`MAX_SHED_PARTS`] cap evicts by.
    pub born: u64,
}

/// **Every vehicle's damage, keyed by chassis** — the resource.
#[derive(Resource, Clone, Debug, Default)]
pub struct VehicleDamageRes {
    /// The rows, in `Guid` order.
    pub rows: BTreeMap<Uuid, VehicleDamage>,
    /// **The bodywork's own step counter**, advanced once per fixed step by
    /// `inf_physics::d3::bodywork::step_bodywork`.
    ///
    /// Its own rather than `traffic::steps`, which is advanced by the traffic
    /// pass and stands still on every level with no traffic in it — measured: a
    /// fixture's glass shards outlived their own second and a half for ever,
    /// because nothing ever aged. Not folded: it is a clock, and a clock in a
    /// determinism trace is a clock in a determinism trace.
    pub steps: u64,
    /// The shed parts still lying about, oldest first — the debris cap's own
    /// list, and the pose of every piece.
    pub shed: Vec<Debris>,
    /// The glass shards still drawn: `(shard guid, the step it was thrown)`.
    pub shards: Vec<(Uuid, u64)>,
}

/// The damage table, if this level has one.
pub fn damage_of(world: &EcsWorld) -> Option<&VehicleDamageRes> {
    world.world().get_resource::<VehicleDamageRes>()
}

/// The damage table, **making one** if this level has none — the door every
/// writer goes through.
pub fn damage_mut(world: &mut EcsWorld) -> &mut VehicleDamageRes {
    if world.world().get_resource::<VehicleDamageRes>().is_none() {
        world
            .world_mut()
            .insert_resource(VehicleDamageRes::default());
    }
    world
        .world_mut()
        .get_resource_mut::<VehicleDamageRes>()
        .expect("just inserted")
        .into_inner()
}

/// One car's damage, or `None`.
pub fn damage_row(world: &EcsWorld, chassis: Uuid) -> Option<&VehicleDamage> {
    damage_of(world).and_then(|r| r.rows.get(&chassis))
}

/// **Forget every car's damage** — the Simulate-session door, beside
/// `clear_drivetrains` and for its reason exactly: a door the author tore off in
/// run 1 is run 1's, and a second run that began with it on the floor would fold
/// different trace bytes from the shipped player, which starts every car whole.
pub fn clear_damage(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<VehicleDamageRes>();
}

// ── the limits a class sets ─────────────────────────────────────────────────

/// **What one car's own tuning says it can take** (wave VEH3c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageLimits {
    /// The joules a pane absorbs before it shatters.
    pub glass_health_j: f64,
    /// The joules one body panel absorbs before it is written off.
    pub panel_health_j: f64,
    /// The joint impulse over which a hinged part tears off, N.s.
    pub part_break_impulse_ns: f64,
}

impl Default for DamageLimits {
    fn default() -> Self {
        Self::of_tuning(&VehicleTuning::default())
    }
}

impl DamageLimits {
    /// The limits a tuning describes.
    pub fn of_tuning(t: &VehicleTuning) -> Self {
        Self {
            glass_health_j: t.glass_health_j.max(1.0),
            panel_health_j: t.panel_health_j.max(1.0),
            part_break_impulse_ns: t.part_break_impulse_ns.max(0.0),
        }
    }

    /// **The limits a CHASSIS carries**, read off its own `VehicleClass`.
    ///
    /// The world and not the trait: `Vehicle` has a `tune(name, value)` and no
    /// getter, and `VehicleClass` is the component the live tuner writes and the
    /// one a cooked level carries — so this is the same door an author, a
    /// catalogue row and `INF_PIE_TUNE_VEHICLE` all reach through. A chassis
    /// with no class answers the Ring-0 defaults rather than zero, because a car
    /// whose panels absorbed nothing would be written off by its own kerb.
    pub fn of(world: &EcsWorld, chassis: Uuid) -> Self {
        // The three fields, read DIRECTLY off the component rather than through
        // `VehicleClass::to_tuning`. That door builds a hundred-field
        // `VehicleTuning` and walks a hundred `set` calls to fill it, and this
        // is asked once per damaged car per step -- measured at a thousand
        // parked cars, it was most of the 0.92 microseconds a car the whole
        // bodywork pass cost.
        let Some(c) = world
            .entity_of(chassis)
            .and_then(|e| world.world().get::<VehicleClass>(e))
        else {
            return Self::default();
        };
        Self {
            glass_health_j: c.glass_health_j.max(1.0),
            panel_health_j: c.panel_health_j.max(1.0),
            part_break_impulse_ns: c.part_break_impulse_ns.max(0.0),
        }
    }

    /// What a whole hull is worth, joules.
    pub fn hull_capacity_j(&self) -> f64 {
        self.panel_health_j * HULL_PANELS
    }

    /// What an engine is worth, joules — one panel's.
    pub fn engine_capacity_j(&self) -> f64 {
        self.panel_health_j
    }
}

// ── the hinge ───────────────────────────────────────────────────────────────

/// **Advance one hinge a step** — a position motor against a pair of limits.
///
/// The same shape `JointMotor3D` describes and `MotorModel::AccelerationBased`
/// solves: an acceleration proportional to the error, damped by the rate,
/// capped, integrated semi-implicitly, then clamped to the limits with the rate
/// killed at a stop. A part whose LIVE twin is a real rapier revolute is driven
/// by the same three numbers, so a door opens at the same speed either side of
/// the regime change.
///
/// `open_deg` is the far limit and may be either sign; the near limit is always
/// `0`. Returns the new `(angle_deg, vel_deg_s)`.
pub fn hinge_step(
    angle_deg: f64,
    vel_deg_s: f64,
    target_deg: f64,
    open_deg: f64,
    dt: f64,
) -> (f64, f64) {
    if !dt.is_finite() || dt <= 0.0 {
        return (angle_deg, vel_deg_s);
    }
    let (lo, hi) = if open_deg < 0.0 {
        (open_deg, 0.0)
    } else {
        (0.0, open_deg)
    };
    let target = target_deg.clamp(lo, hi);
    let accel = (HINGE_STIFFNESS * (target - angle_deg) - HINGE_DAMPING * vel_deg_s)
        .clamp(-HINGE_ACCEL_MAX_DEG_S2, HINGE_ACCEL_MAX_DEG_S2);
    let mut v = vel_deg_s + accel * dt;
    let mut a = angle_deg + v * dt;
    if a < lo {
        a = lo;
        v = 0.0;
    } else if a > hi {
        a = hi;
        v = 0.0;
    }
    (a, v)
}

// ── where a part is DRAWN ───────────────────────────────────────────────────

/// **The local pose one part is drawn at** (wave VEH3c) — its authored box, bent
/// by its dent and swung on its hinge.
///
/// Rebuilt from the AUTHORED baseline every step rather than accumulated onto
/// the transform it read: a dent applied to a dented transform deepens for ever,
/// and a hinge integrated onto its own output drifts. `chassis_half` is the
/// chassis collider's half-extents in metres; everything else is in fractions of
/// them.
///
/// Returns `(translation, rotation_deg, scale)` in the chassis's own frame — the
/// three fields of a `Transform`.
///
/// # Portable throughout
///
/// The rotation reaches a `Transform`, which `sim_snapshot` folds into the
/// determinism trace's first section — so the sine and the cosine are
/// [`inf_math::psin64`] and [`inf_math::pcos64`], never `f64::sin`. The P14 law's
/// first class, met at a car door.
pub fn part_pose(
    part: &PartState,
    kind: BodyPartKind,
    chassis_half: Vec3d,
    angle_deg: f64,
) -> (Vec3d, Vec3d, Vec3d) {
    let h = chassis_half;
    let mut centre = Vec3d::new(
        part.centre_frac.x * h.x,
        part.centre_frac.y * h.y,
        part.centre_frac.z * h.z,
    );
    let mut scale = Vec3d::new(
        2.0 * part.half_frac.x * h.x,
        2.0 * part.half_frac.y * h.y,
        2.0 * part.half_frac.z * h.z,
    );

    // ── THE DENT ────────────────────────────────────────────────────────────
    //
    // A local push-in along the axis the part faces: the panel's outer face
    // moves toward the middle of the car and its inner face stays where it was,
    // which is what a dent IS. Bounded by `MAX_DENT_M` at the source and by
    // four fifths of the panel's own thickness here, so a panel can never be
    // pushed through the far side of its own car.
    if part.dent_m > 0.0 {
        let (axis, sign) = part.facing();
        let base = [scale.x, scale.y, scale.z][axis];
        let d = part.dent_m.min(MAX_DENT_M).min(0.8 * base);
        let (c, sc) = match axis {
            0 => (&mut centre.x, &mut scale.x),
            1 => (&mut centre.y, &mut scale.y),
            _ => (&mut centre.z, &mut scale.z),
        };
        *c -= sign * d * 0.5;
        *sc = (*sc - d).max(1e-4);
    }

    // ── THE HINGE ───────────────────────────────────────────────────────────
    let Some(hinge) = kind.hinge() else {
        return (centre, Vec3d::ZERO, scale);
    };
    if angle_deg == 0.0 {
        return (centre, Vec3d::ZERO, scale);
    }
    let pivot = Vec3d::new(hinge.at.x * h.x, hinge.at.y * h.y, hinge.at.z * h.z);
    let rad = angle_deg.to_radians();
    let (sn, cs) = (inf_math::psin64(rad), inf_math::pcos64(rad));
    let d = Vec3d::new(centre.x - pivot.x, centre.y - pivot.y, centre.z - pivot.z);
    // The two axes the tables author, and no others: a door swings about `+Y`
    // and a bonnet or a boot lid about `+X`. A hinge on any other axis is a
    // REFUSAL rather than a wrong answer — it draws shut, and
    // `every_authored_part_is_recognised_as_the_kind_it_declares` is what stops
    // one ever being authored.
    if hinge.axis.y.abs() > hinge.axis.x.abs() {
        // R_y(theta): x' = x cos + z sin, z' = -x sin + z cos.
        (
            Vec3d::new(
                pivot.x + d.x * cs + d.z * sn,
                centre.y,
                pivot.z - d.x * sn + d.z * cs,
            ),
            Vec3d::new(0.0, angle_deg, 0.0),
            scale,
        )
    } else if hinge.axis.x != 0.0 {
        // R_x(theta): y' = y cos - z sin, z' = y sin + z cos.
        (
            Vec3d::new(
                centre.x,
                pivot.y + d.y * cs - d.z * sn,
                pivot.z + d.y * sn + d.z * cs,
            ),
            Vec3d::new(angle_deg, 0.0, 0.0),
            scale,
        )
    } else {
        (centre, Vec3d::ZERO, scale)
    }
}

// ── the trace (the 18th section) ────────────────────────────────────────────

/// How many bytes one PART folds: 16 of guid, a kind, a latch, five `f64` and
/// the step it was shed.
pub const PART_TRACE_BYTES: usize = 16 + 1 + 1 + 5 * 8 + 8;

/// How many bytes one CAR folds before its parts: 16 of guid, two `f64`, the
/// flats byte, the fire step and a part count.
pub const VEHICLE_DAMAGE_TRACE_BYTES: usize = 16 + 2 * 8 + 1 + 8 + 2;

/// **The bodywork's trace bytes** (wave VEH3c) — the determinism trace's
/// **18th** section, at the tail, after VEH3b's drivetrain.
///
/// **Empty until something breaks.** A car standing on its wheels with its
/// doors shut folds nothing at all, so every level that was quiet before this
/// wave folds the bytes it always folded — and a level where one pane went
/// folds one car, which is what makes the section a measurement rather than a
/// background hum.
pub fn damage_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(res) = damage_of(world) else {
        return Vec::new();
    };
    let live: Vec<(&Uuid, &VehicleDamage)> =
        res.rows.iter().filter(|(_, d)| !d.is_quiet()).collect();
    if live.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(live.len() * (VEHICLE_DAMAGE_TRACE_BYTES + PART_TRACE_BYTES));
    for (guid, d) in live {
        out.extend_from_slice(guid.as_bytes());
        for v in [d.hull_j, d.engine_damage] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.push(d.flats);
        out.extend_from_slice(&d.fire_step.to_le_bytes());
        // Only the parts that have something to say. A crashed car has fourteen
        // parts and two of them are news.
        let parts: Vec<(&Uuid, &PartState)> =
            d.parts.iter().filter(|(_, p)| !p.is_quiet()).collect();
        out.extend_from_slice(&(parts.len() as u16).to_le_bytes());
        for (pg, p) in parts {
            out.extend_from_slice(pg.as_bytes());
            out.push(p.kind);
            out.push(p.latch.as_u8());
            for v in [p.angle_deg, p.vel_deg_s, p.target_deg, p.damage_j, p.dent_m] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            out.extend_from_slice(&p.shed_step.to_le_bytes());
        }
    }
    out
}

// ── the readout ─────────────────────────────────────────────────────────────

/// **The hero's car damage, as one HUD row** (wave VEH3c) —
/// `inf_ecs::vehicle::drive_readout`'s sibling.
///
/// `HULL 100%  ENG 100%  FLATS 0  GLASS 0/4`, and a burning car says so.
pub fn damage_readout(damage: &VehicleDamage, limits: DamageLimits) -> String {
    let panes = damage
        .parts
        .values()
        .filter(|p| p.kind == crate::vehicle::KIND_GLASS)
        .count();
    let mut row = format!(
        "HULL {:.0}%  ENG {:.0}%  FLATS {}  GLASS {}/{}",
        100.0 * damage.hull_frac(limits.hull_capacity_j()),
        100.0 * damage.engine_scale(),
        damage.flat_count(),
        damage.panes_broken(),
        panes
    );
    let shed = damage.parts_shed();
    if shed > 0 {
        row.push_str(&format!("  SHED {shed}"));
    }
    if damage.burning() {
        row.push_str("  ON FIRE");
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 60.0;

    #[test]
    fn a_hinge_opens_to_its_limit_and_stops_there() {
        let (mut a, mut v) = (0.0, 0.0);
        let mut steps_to_half = None;
        for i in 0..240 {
            let (na, nv) = hinge_step(a, v, 66.0, 66.0, DT);
            a = na;
            v = nv;
            if steps_to_half.is_none() && a >= 33.0 {
                steps_to_half = Some(i);
            }
            assert!(
                (0.0..=66.0 + 1e-9).contains(&a),
                "the hinge left its limits at {a}"
            );
        }
        eprintln!(
            "the door reached half open in {} steps and settled at {a:.3} deg",
            steps_to_half.unwrap_or(999)
        );
        assert!((a - 66.0).abs() < 0.5, "the door settled at {a:.3}, not 66");
        assert!(steps_to_half.is_some_and(|s| (5..=60).contains(&s)));
        // …and it shuts again.
        for _ in 0..240 {
            let (na, nv) = hinge_step(a, v, 0.0, 66.0, DT);
            a = na;
            v = nv;
        }
        assert!(a.abs() < 0.5, "the door shut to {a:.3}, not 0");
    }

    #[test]
    fn a_hinge_that_opens_the_other_way_is_the_same_arithmetic() {
        let (mut a, mut v) = (0.0, 0.0);
        for _ in 0..240 {
            let (na, nv) = hinge_step(a, v, -58.0, -58.0, DT);
            a = na;
            v = nv;
            assert!((-58.0 - 1e-9..=0.0).contains(&a));
        }
        assert!((a + 58.0).abs() < 0.5, "the boot settled at {a:.3}");
    }

    #[test]
    fn a_whole_car_folds_nothing_until_something_happens_to_it() {
        let mut world = EcsWorld::default();
        assert!(damage_state_bytes(&world).is_empty());
        let chassis = Uuid::from_u128(0xC1);
        let part = Uuid::from_u128(0xC2);
        {
            let res = damage_mut(&mut world);
            let row = res.rows.entry(chassis).or_default();
            row.parts.insert(part, PartState::new(BodyPartKind::Bumper));
        }
        assert!(
            damage_state_bytes(&world).is_empty(),
            "an undamaged car folded bytes"
        );
        {
            let res = damage_mut(&mut world);
            let row = res.rows.get_mut(&chassis).unwrap();
            row.parts.get_mut(&part).unwrap().latch = PartLatch::Shed;
            row.parts.get_mut(&part).unwrap().shed_step = 12;
            row.refresh_parts();
        }
        let bytes = damage_state_bytes(&world);
        assert_eq!(
            bytes.len(),
            VEHICLE_DAMAGE_TRACE_BYTES + PART_TRACE_BYTES,
            "one car with one shed part folds one of each"
        );
        assert_eq!(&bytes[..16], chassis.as_bytes());
        clear_damage(&mut world);
        assert!(damage_state_bytes(&world).is_empty());
    }

    #[test]
    fn the_flats_are_a_bitmask_that_refuses_a_wheel_it_has_no_bit_for() {
        let mut d = VehicleDamage::default();
        assert!(d.flatten(0));
        assert!(!d.flatten(0));
        assert!(d.flatten(3));
        assert_eq!(d.flat_count(), 2);
        assert!(d.is_flat(0) && d.is_flat(3) && !d.is_flat(1));
        assert!(!d.flatten(8), "a ninth wheel has no bit");
        assert_eq!(d.flat_count(), 2);
    }

    #[test]
    fn the_readout_says_what_the_bodywork_knows() {
        let limits = DamageLimits::default();
        let mut d = VehicleDamage::default();
        d.parts
            .insert(Uuid::from_u128(1), PartState::new(BodyPartKind::Glass));
        d.parts
            .insert(Uuid::from_u128(2), PartState::new(BodyPartKind::Glass));
        assert_eq!(
            damage_readout(&d, limits),
            "HULL 100%  ENG 100%  FLATS 0  GLASS 0/2"
        );
        d.hull_j = limits.hull_capacity_j() * 0.5;
        d.engine_damage = 0.25;
        d.flatten(1);
        d.parts.get_mut(&Uuid::from_u128(1)).unwrap().latch = PartLatch::Gone;
        d.fire_step = 9;
        assert_eq!(
            damage_readout(&d, limits),
            "HULL 50%  ENG 75%  FLATS 1  GLASS 1/2  ON FIRE"
        );
    }
}
