//! **Doors** (island wave I6): a leaf on a hinge, a lock that is one P22 bond,
//! and the three ways through one.
//!
//! # What a door is
//!
//! A [`DoorSpec`] is a rectangular leaf hinged on one vertical edge. Everything
//! about where it is comes from the *placement* ([`DoorPlacement`]) so that an
//! authored door (an entity with a [`Door`] component and a transform) and a
//! doorway the building grammar emitted (a record on a `PcgVolume`, with no
//! entity at all) can be ranked, swung and broken by the same arithmetic. That
//! is I5's own shape — `inf_ecs::interact::InteractCandidate` flattens a vehicle
//! seat and an authored item onto one rule — applied a second time.
//!
//! # Where the state lives, and why it is not on the component
//!
//! [`DoorField`] is a bevy **resource**, sparse, keyed by the door's `Guid`, and
//! **absent means closed**. Three reasons, in the order they bind:
//!
//! * A grammar doorway has no entity, so it has nowhere to put a component. A
//!   design that gave authored doors their state on the component and derived
//!   doors their state in a map would be two authorities on "is this door open",
//!   which is the defect this repository has paid for at five seams.
//! * It is exactly P22's split — [`crate::components::Destructible`] is the
//!   authored config and `FractureState` is the mutable half a host owns — and
//!   it means **no schema moves**: the scene serializer walks entities and
//!   components, never resources.
//! * The city fixture's hundred blocks plan on the order of twenty thousand
//!   doorways. A `DoorState` per doorway would be state nobody has touched;
//!   sparse, the map holds only doors a player has actually used.
//!
//! The field folds into the sim trace through [`door_state_bytes`], appended to
//! `state_bytes` exactly as P22.1's field and P24.1's poses are, so every gate
//! the engine already has covers doors at once. A level with no doors produces
//! an empty vec and every pre-I6 trace is byte-identical.
//!
//! # The lock is ONE BOND
//!
//! A lock is priced by [`crate::components::Destructible::bond_energy_j`] —
//! `strength × area × CRACK_OPENING_M` — which is the same function the fracture
//! solve prices a chunk face with. That is what "one door for the kick and the
//! crash" means concretely: [`kick_energy_j`] and [`breach_energy_j`] produce
//! joules, [`DoorSpec::lock_energy_j`] says what a break costs, and the
//! comparison is the same one P22 makes. Nothing here invents a damage number.
//!
//! # Portability
//!
//! The swing writes a world transform that reaches a collider and therefore the
//! replay trace, so the trigonometry is `inf_math::psin64`/`pcos64` — the P14
//! law (`f32` std trig is not bit-portable, and `f64`'s is not guaranteed
//! either), not `f64::sin`.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Component, Resource, With};
use glam::DVec3;
use uuid::Uuid;

use crate::components::{Destructible, GlobalTransform, Guid};
use crate::movement::wrap_deg;
use crate::world::EcsWorld;

// ── the numbers, each with its derivation ───────────────────────────────────

/// A domestic door leaf's width, metres.
pub const DEFAULT_DOOR_WIDTH_M: f64 = 0.9;
/// Its height, metres.
pub const DEFAULT_DOOR_HEIGHT_M: f64 = 2.1;
/// Its thickness, metres — 6 cm, a solid-core leaf.
pub const DEFAULT_DOOR_THICKNESS_M: f64 = 0.06;
/// How far a door swings before the frame stops it, degrees.
///
/// Past ninety, so a leaf pushed fully open is unambiguously out of the opening
/// rather than exactly across it.
pub const DEFAULT_OPEN_LIMIT_DEG: f64 = 95.0;

/// How fast a **powered** swing runs, degrees per second — the E key's open and
/// close. 220 deg/s puts a 95-degree leaf through in 0.43 s, which is a door
/// being opened rather than a door being teleported.
pub const DOOR_DRIVE_DPS: f64 = 220.0;
/// Exponential drag on a **free** swing, per second — a kicked or breached leaf
/// coasting to its limit.
pub const DOOR_FREE_DRAG_PER_S: f64 = 1.1;
/// Below this angular speed a free swing has stopped, degrees per second.
///
/// A door that never quite settles is a `DoorField` entry that never stops
/// changing, and therefore a trace that never repeats.
pub const DOOR_SETTLE_DPS: f64 = 8.0;
/// How fast a leaf may swing at all, degrees per second — the bound on a breach
/// that arrives with kilojoules behind it.
///
/// A **bound, not a cure**, in the shape the I5 ragdoll ceiling has: it is what
/// keeps a leaf given absurd energy inside its own frame rather than through it
/// in one step.
pub const DOOR_MAX_DPS: f64 = 900.0;

/// A lock bolt's shear strength, pascals — 300 MPa, structural steel's own
/// class (`docs/memos/p22-strength.md`'s table one row up from concrete).
pub const DEFAULT_LOCK_STRENGTH_PA: f64 = 3.0e8;
/// A lock bolt's shear area, m² — a 20 mm × 20 mm throw.
///
/// With the strength above and P22's millimetre of crack opening that is
/// `3.0e8 × 4.0e-4 × 1.0e-3` = **120 J** exactly, which is the number every
/// door claim in this wave is measured against. It is derived, not chosen: the
/// two inputs are a material and a bolt, and the joules fall out of the P22
/// rule.
pub const DEFAULT_LOCK_AREA_M2: f64 = 4.0e-4;

/// The effective striking mass of a kick, kg — a leg and a hip, not a body.
pub const KICK_MASS_KG: f64 = 15.0;
/// The speed that mass arrives at, m/s.
///
/// `0.5 × 15 × 4.5²` = **151.875 J**, against a default lock's 120 J: one kick
/// opens a domestic door with 31.875 J to spare, and a lock a designer doubles
/// takes two — which is a design knob rather than a hidden constant.
pub const KICK_SPEED_MPS: f64 = 4.5;
/// How close the character's feet must be to kick a door, metres.
///
/// Wider than [`crate::interact::DEFAULT_REACH_M`] would be for a prompt, and
/// deliberately: the leg is the reach.
pub const KICK_REACH_M: f64 = 1.8;
/// The total cone a kick must be aimed within, degrees.
pub const KICK_CONE_DEG: f64 = 100.0;

/// How fast a body must be moving to breach a closed door by running into it,
/// m/s.
///
/// **Above the run speed and below the sprint speed** (`CharacterMovement`'s
/// defaults are 3.75 and 6.5), so jogging into a door is a collision and
/// sprinting into one is a breach. The energy gate below decides whether the
/// *lock* gives; this decides whether the engine treats the contact as an
/// attempt at all, which is the owner's "sprinting/sliding/diving" in a number.
pub const BREACH_SPEED_MPS: f64 = 5.0;
/// The mass a breaching character is priced at, kg.
///
/// A constant rather than a lookup, and the reason is measured elsewhere in this
/// repository: `Collider3D::density` is rapier's 1.0 placeholder and not a
/// material density (the P20.2 finding, met again in P22), so deriving a
/// character's mass from its collider would price an 80 kg adult at a few
/// hundred grams.
pub const DEFAULT_BODY_MASS_KG: f64 = 80.0;
/// How much of its speed a character keeps through a successful breach, after
/// the energy the lock absorbed has been taken off it.
///
/// Two terms, not one: the lock's own joules come off exactly (that is what
/// absorbing energy means) and this is the rest of the impact. At a sprint,
/// 6.5 m/s against a default lock leaves **5.325 m/s** — 81.9 % — and the mode
/// continues, which is the owner's "momentum mostly kept".
pub const BREACH_RESTITUTION: f64 = 0.85;
/// How far past a closed leaf a breach carries the body before the door is
/// considered passed, metres. Used only to seed the leaf's own swing.
pub const BREACH_SWING_DPS: f64 = 520.0;

/// How wide the prompt's cone is on a door, degrees — a door is a flat thing and
/// you have to be facing it.
pub const DOOR_VIEW_CONE_DEG: f64 = 120.0;

/// How close you have to be to use a door, metres.
///
/// **The reach budget includes the leaf's own height**, which is the same thing
/// `ENTER_REACH_M`'s doc says of a vehicle seat, and it is arithmetic rather
/// than generosity: [`crate::interact::resolve`] measures from the character's
/// **feet**, and a door's interaction point is at the leaf's mid-height. A
/// default leaf is 2.1 m tall, so its middle is 1.05 m up, and a character
/// standing `d` metres from the threshold is `sqrt(d² + 1.05²)` from the point.
/// At 2.4 that is **2.16 m of floor**, which is arm's length plus a step; at the
/// 2.0 this was first written as it was 1.70 m, and a player standing a
/// comfortable pace from a door got no prompt at all.
pub const DOOR_REACH_M: f64 = 2.4;

// ── the shapes ──────────────────────────────────────────────────────────────

/// Which face of a door a verb is offered on.
///
/// A lock is lockable **from the inside** — the owner's mandate — so the verb
/// has a side, and the side is a fact about where the character is standing
/// rather than about the door.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum DoorSide {
    /// The face the leaf's inside normal points out of.
    #[default]
    Inside,
    /// The other one.
    Outside,
}

impl DoorSide {
    /// The other side.
    pub fn flip(self) -> Self {
        match self {
            DoorSide::Inside => DoorSide::Outside,
            DoorSide::Outside => DoorSide::Inside,
        }
    }
}

/// **A door's geometry and its lock** — the immutable half.
///
/// `Copy` and scalar throughout, because a `PcgVolume` carries one of these per
/// doorway and a city plans twenty thousand of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorSpec {
    /// The compass yaw, degrees, from the hinge toward the free edge when the
    /// door is shut. `+Z` is zero and `+X` is `+90` — the engine's one
    /// convention (`crate::movement::planar_yaw_deg`).
    pub closed_yaw_deg: f64,
    /// Leaf width, metres — hinge to free edge.
    pub width_m: f64,
    /// Leaf height, metres.
    pub height_m: f64,
    /// Leaf thickness, metres.
    pub thickness_m: f64,
    /// How far it opens, degrees, **signed**: positive swings toward `+yaw`.
    ///
    /// One number rather than a direction and a limit, because the two are never
    /// independent — a door that opens the other way opens the other way *to its
    /// own stop*, and two fields would let an author write a door that swings
    /// left and stops on the right.
    pub open_limit_deg: f64,
    /// The compass yaw of the **inside** face's outward normal, degrees. What
    /// [`DoorSpec::side_of`] compares a character's position against.
    pub inside_yaw_deg: f64,
    /// The lock bolt's shear strength, pascals.
    pub lock_strength_pa: f64,
    /// The lock bolt's shear area, m².
    pub lock_area_m2: f64,
    /// Which side the lock/unlock verb is offered on.
    pub lock_side: DoorSide,
    /// Whether the door starts locked.
    pub locked_at_spawn: bool,
}

impl Default for DoorSpec {
    fn default() -> Self {
        Self {
            closed_yaw_deg: 0.0,
            width_m: DEFAULT_DOOR_WIDTH_M,
            height_m: DEFAULT_DOOR_HEIGHT_M,
            thickness_m: DEFAULT_DOOR_THICKNESS_M,
            open_limit_deg: DEFAULT_OPEN_LIMIT_DEG,
            inside_yaw_deg: 90.0,
            lock_strength_pa: DEFAULT_LOCK_STRENGTH_PA,
            lock_area_m2: DEFAULT_LOCK_AREA_M2,
            lock_side: DoorSide::Inside,
            locked_at_spawn: false,
        }
    }
}

impl DoorSpec {
    /// Whether every number is usable. A door that is not is skipped everywhere
    /// rather than silently drawn at the origin.
    pub fn is_usable(&self) -> bool {
        self.closed_yaw_deg.is_finite()
            && self.inside_yaw_deg.is_finite()
            && self.open_limit_deg.is_finite()
            && self.open_limit_deg != 0.0
            && self.width_m.is_finite()
            && self.width_m > 0.0
            && self.height_m.is_finite()
            && self.height_m > 0.0
            && self.thickness_m.is_finite()
            && self.thickness_m > 0.0
    }

    /// **What breaking this lock costs, joules** — the P22 rule, through the P22
    /// function.
    ///
    /// A lock with no area or no strength costs `0.0`, which is
    /// `bond_energy_j`'s own reading of "this bond cannot hold" and is the right
    /// answer for a door somebody authored with a lock made of nothing.
    pub fn lock_energy_j(&self) -> f64 {
        let d = Destructible {
            strength: self.lock_strength_pa,
            ..Destructible::default()
        };
        d.bond_energy_j(self.lock_area_m2)
    }

    /// Which side of the leaf's plane `point` is on.
    ///
    /// Measured against the **closed** plane, always: a character reaching for a
    /// door that is already half open is still on the side of the wall they were
    /// standing on, and asking the swung leaf would flip the answer half way
    /// through an opening.
    pub fn side_of(&self, hinge: DVec3, point: DVec3) -> DoorSide {
        let n = yaw_dir(self.inside_yaw_deg);
        let d = (point - hinge).dot(n);
        if d >= 0.0 {
            DoorSide::Inside
        } else {
            DoorSide::Outside
        }
    }

    /// The angle the leaf rests at when shut. Always `0.0`; named so call sites
    /// read as intent rather than as a magic zero.
    pub fn closed_deg(&self) -> f64 {
        0.0
    }

    /// Clamp an angle into `[0, open_limit_deg]` respecting the limit's sign.
    pub fn clamp_open(&self, deg: f64) -> f64 {
        if !deg.is_finite() {
            return 0.0;
        }
        if self.open_limit_deg >= 0.0 {
            deg.clamp(0.0, self.open_limit_deg)
        } else {
            deg.clamp(self.open_limit_deg, 0.0)
        }
    }
}

/// **An entity offers a door.**
///
/// A runtime component with no scene slot — [`crate::interact::Interactable`]'s
/// shape and for its reason: `RuntimeEntityGen` decodes positionally, so a slot
/// is a schema bump, and a door's *state* does not belong on an entity anyway
/// (see the module header). The entity's `GlobalTransform` **is the hinge**, at
/// the leaf's mid-height.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Door {
    /// The geometry and the lock.
    pub spec: DoorSpec,
    /// What the prompt calls it.
    pub label: String,
}

impl Default for Door {
    fn default() -> Self {
        Self {
            spec: DoorSpec::default(),
            label: "door".to_string(),
        }
    }
}

/// **One door the rules may act on**, flattened out of whatever produced it —
/// an entity, or a doorway record on a `PcgVolume`.
///
/// The seam that makes an authored door and a grammar doorway one subject, in
/// the same sense `InteractCandidate` makes a seat and an item one.
#[derive(Clone, Debug, PartialEq)]
pub struct DoorPlacement {
    /// The door's stable id. For a derived doorway this is a synthetic guid
    /// folded from its volume and index, exactly as a structure collider's is.
    pub guid: Uuid,
    /// The hinge, world metres, at the leaf's mid-height.
    pub hinge: DVec3,
    /// Its geometry and lock.
    pub spec: DoorSpec,
    /// What to call it.
    pub label: String,
}

/// **A door's mutable half** — where the leaf is, and what the lock is doing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorState {
    /// How far open, degrees, signed to match [`DoorSpec::open_limit_deg`].
    pub open_deg: f64,
    /// Angular velocity, degrees per second.
    pub vel_dps: f64,
    /// Where a **powered** swing is heading, degrees. Meaningless while
    /// [`powered`](Self::powered) is false.
    pub target_deg: f64,
    /// Whether the door is under power (the E key) rather than coasting (a kick
    /// or a breach).
    pub powered: bool,
    /// Whether the lock is engaged.
    pub locked: bool,
    /// Whether the lock has been broken. A broken lock cannot be re-engaged —
    /// that is what breaking it buys, and a door that could be re-locked after
    /// being kicked in would make the kick a delay rather than a way in.
    pub lock_broken: bool,
}

impl DoorState {
    /// The state a door nobody has touched is in.
    pub fn fresh(spec: &DoorSpec) -> Self {
        Self {
            open_deg: 0.0,
            vel_dps: 0.0,
            target_deg: 0.0,
            powered: false,
            locked: spec.locked_at_spawn,
            lock_broken: false,
        }
    }

    /// Whether the leaf is out of the opening far enough to walk through.
    ///
    /// Half the limit rather than "not exactly zero": a door one degree ajar is
    /// not a door you can walk through, and a gate that asked `open_deg != 0.0`
    /// would certify one.
    pub fn is_open(&self, spec: &DoorSpec) -> bool {
        self.open_deg.abs() >= spec.open_limit_deg.abs() * 0.5
    }

    /// Whether the leaf is at rest — settled, with nothing driving it.
    pub fn is_at_rest(&self) -> bool {
        !self.powered && self.vel_dps == 0.0
    }
}

/// **Every door's state, sparsely** — the resource. See the module header for
/// why it is a resource and why absent means closed.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct DoorField(pub BTreeMap<Uuid, DoorState>);

impl DoorField {
    /// This door's state, or the fresh one its spec implies.
    ///
    /// Reading never inserts: a level whose doors nobody has touched keeps an
    /// empty map, which is what makes the trace bytes empty and every pre-I6
    /// trace byte-identical.
    pub fn get(&self, guid: Uuid, spec: &DoorSpec) -> DoorState {
        self.0
            .get(&guid)
            .copied()
            .unwrap_or_else(|| DoorState::fresh(spec))
    }

    /// This door's state for writing, materialised if it was not there.
    pub fn entry(&mut self, guid: Uuid, spec: &DoorSpec) -> &mut DoorState {
        self.0.entry(guid).or_insert_with(|| DoorState::fresh(spec))
    }

    /// How many doors have been touched.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether nothing has been touched.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// **The trace bytes**, in `Guid` order.
    ///
    /// Every field of every touched door, as raw bit patterns — never a rounded
    /// decimal, because a comparison that passes within a tolerance accepts
    /// precisely the drift a PIE-versus-shipping gate exists to catch.
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.0.len() * 48);
        for (guid, s) in &self.0 {
            out.extend_from_slice(guid.as_bytes());
            out.extend_from_slice(&s.open_deg.to_bits().to_le_bytes());
            out.extend_from_slice(&s.vel_dps.to_bits().to_le_bytes());
            out.extend_from_slice(&s.target_deg.to_bits().to_le_bytes());
            out.push(u8::from(s.powered));
            out.push(u8::from(s.locked));
            out.push(u8::from(s.lock_broken));
        }
        out
    }
}

// ── the rules ───────────────────────────────────────────────────────────────

/// The unit world vector at a compass yaw — `+Z` at zero, `+X` at `+90`.
///
/// `psin64`/`pcos64`, not `f64::sin`/`cos`: this reaches a collider pose and
/// therefore the replay trace, and the P14 law is that std trig is not
/// bit-portable.
fn yaw_dir(yaw_deg: f64) -> DVec3 {
    let r = yaw_deg.to_radians();
    DVec3::new(inf_math::psin64(r), 0.0, inf_math::pcos64(r))
}

/// **Where the leaf is**: the centre of its box, its yaw, and its half extents.
///
/// The hinge is on one end of the leaf's length, so the centre is half a width
/// along the leaf's own direction from it. Returned rather than written so the
/// same function serves the collider sync, the interaction position and every
/// test.
///
/// # The half extents are `(thickness, height, width)`, and the order matters
///
/// The yaw is applied as `DQuat::from_rotation_y`, and `yaw_dir` puts yaw
/// **zero at `+Z`** (the engine's one compass convention). So the leaf's *length*
/// runs along its own local `+Z` and its thickness along local `+X` — and a box
/// written `(width, height, thickness)` would be a leaf lying across its own
/// doorway at every angle.
///
/// It cost a real defect: the blocking probe swept a 0.9 m box along the
/// **thin** axis and a 0.06 m box along the long one, so a solid standing
/// squarely in a door's arc was never hit and the leaf swung straight through
/// it. Measured before the fix at 0 blocked steps of 90 against a box the leaf
/// passes through; after it, the leaf stops.
pub fn leaf_pose(placement: &DoorPlacement, open_deg: f64) -> (DVec3, f64, DVec3) {
    let spec = &placement.spec;
    let yaw = wrap_deg(spec.closed_yaw_deg + open_deg);
    let dir = yaw_dir(yaw);
    let centre = placement.hinge + dir * (spec.width_m * 0.5);
    let half = DVec3::new(
        spec.thickness_m * 0.5,
        spec.height_m * 0.5,
        spec.width_m * 0.5,
    );
    (centre, yaw, half)
}

/// Where a **prompt** for this door belongs, world metres.
///
/// The middle of the *closed* opening, not the middle of the leaf: a prompt that
/// rode the leaf would swing away from the player as the door opened, and the
/// thing being talked about is the doorway.
pub fn prompt_position(placement: &DoorPlacement) -> DVec3 {
    let dir = yaw_dir(placement.spec.closed_yaw_deg);
    placement.hinge + dir * (placement.spec.width_m * 0.5)
}

/// What a verb did to a door. **A refusal is a value.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorVerdict {
    /// The door was shut and is now opening.
    Opening,
    /// The door was open and is now closing.
    Closing,
    /// The lock is engaged and the door did not move.
    Locked,
    /// The lock was engaged and is now not.
    Unlocked,
    /// The lock was not engaged and now is.
    LockedNow,
    /// The verb is not offered from where the character is standing.
    WrongSide,
    /// The door's own numbers are unusable.
    Unusable,
}

impl DoorVerdict {
    /// Whether the leaf moved (or started to).
    pub fn moved(self) -> bool {
        matches!(self, DoorVerdict::Opening | DoorVerdict::Closing)
    }
}

/// **The E key on a door.** Toggle it, unless the lock says otherwise.
///
/// A locked door is not skipped — it answers [`DoorVerdict::Locked`], which is
/// what the prompt reads. A resolver that hid a locked door would let the prompt
/// fall through to whatever is behind it, which is `Interactable::enabled`'s own
/// lesson one wave earlier.
pub fn toggle(spec: &DoorSpec, state: &mut DoorState) -> DoorVerdict {
    if !spec.is_usable() {
        return DoorVerdict::Unusable;
    }
    if state.locked && !state.lock_broken {
        return DoorVerdict::Locked;
    }
    let opening = !state.is_open(spec);
    state.powered = true;
    state.vel_dps = 0.0;
    state.target_deg = if opening {
        spec.open_limit_deg
    } else {
        spec.closed_deg()
    };
    if opening {
        DoorVerdict::Opening
    } else {
        DoorVerdict::Closing
    }
}

/// **The lock verb**, offered on one face only.
///
/// `from` is the side the character is standing on. A lock that could be worked
/// from either face would make "locked from the inside" mean nothing.
pub fn set_locked(
    spec: &DoorSpec,
    state: &mut DoorState,
    from: DoorSide,
    locked: bool,
) -> DoorVerdict {
    if !spec.is_usable() {
        return DoorVerdict::Unusable;
    }
    if from != spec.lock_side {
        return DoorVerdict::WrongSide;
    }
    if locked && state.lock_broken {
        // A broken lock does not re-engage. See `DoorState::lock_broken`.
        return DoorVerdict::WrongSide;
    }
    state.locked = locked;
    if locked {
        DoorVerdict::LockedNow
    } else {
        DoorVerdict::Unlocked
    }
}

/// **The kinetic energy a kick delivers, joules** — `½ m v²` over the striking
/// mass and speed above. A constant expressed as its derivation rather than as a
/// number, so the two inputs are what a designer moves.
pub fn kick_energy_j() -> f64 {
    0.5 * KICK_MASS_KG * KICK_SPEED_MPS * KICK_SPEED_MPS
}

/// **The kinetic energy a moving body carries, joules** — `½ m v²`.
///
/// The one expression the crash-through and the kick share, which is what makes
/// "the lock strength decides" a true sentence rather than two tuning tables.
/// Non-finite or negative inputs answer `0.0`: an energy that cannot be compared
/// must not be treated as infinite.
pub fn breach_energy_j(mass_kg: f64, speed_mps: f64) -> f64 {
    if !mass_kg.is_finite() || !speed_mps.is_finite() || mass_kg <= 0.0 {
        return 0.0;
    }
    let e = 0.5 * mass_kg * speed_mps * speed_mps;
    if e.is_finite() {
        e.max(0.0)
    } else {
        0.0
    }
}

/// What an attempt to break a door open decided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakVerdict {
    /// Whether the lock gave.
    pub broke: bool,
    /// Joules the lock absorbed — `0.0` when it did not give, because **damage
    /// is not banked**: that is P22's own rule (`runtime_destruct` spends what
    /// it can afford and dissipates the rest) and a door that accumulated
    /// half-kicks would be a different rule at the same door.
    pub absorbed_j: f64,
    /// Joules left over, which become the leaf's swing.
    pub remainder_j: f64,
    /// What the lock cost.
    pub required_j: f64,
}

/// **Can this energy break this lock**, and what is left if it does.
///
/// The single comparison behind both the kick and the crash. An unlocked (or
/// already-broken) door "breaks" for free and keeps all of the energy, which is
/// the right answer: nothing was holding it.
pub fn try_break(spec: &DoorSpec, state: &DoorState, energy_j: f64) -> BreakVerdict {
    let required = if state.locked && !state.lock_broken {
        spec.lock_energy_j()
    } else {
        0.0
    };
    let e = if energy_j.is_finite() && energy_j > 0.0 {
        energy_j
    } else {
        0.0
    };
    if e < required {
        return BreakVerdict {
            broke: false,
            absorbed_j: 0.0,
            remainder_j: 0.0,
            required_j: required,
        };
    }
    BreakVerdict {
        broke: true,
        absorbed_j: required,
        remainder_j: e - required,
        required_j: required,
    }
}

/// **Apply a break to a door**: burst the lock and send the leaf flying.
///
/// `remainder_j` becomes angular velocity through [`BREACH_SWING_DPS`] scaled by
/// how much energy is left over a lock's own worth, and is bounded by
/// [`DOOR_MAX_DPS`] — a bound, not a cure, in the shape the ragdoll's limb
/// ceiling has.
pub fn apply_break(spec: &DoorSpec, state: &mut DoorState, verdict: &BreakVerdict) {
    if !verdict.broke {
        return;
    }
    // **A LOCK ONLY BREAKS IF IT WAS HOLDING** (I6 audit).
    //
    // [`try_break`] answers `broke` for a door that was merely *shut* as well as
    // for one that was locked — nothing was holding it, so it opens for free and
    // that is the right verdict. Marking it `lock_broken` anyway was not: a
    // broken lock cannot be re-engaged (that is what breaking it buys), so one
    // sprint through a house's own unlocked front door retired that door's lock
    // for the session, and the lock verb stopped being offered on either face.
    // Every door the building grammar emits starts unlocked, so on the shipped
    // city that was every door in it — against a mandate whose first clause is
    // that doors have locks.
    //
    // Read from the STATE rather than from `verdict.required_j`, because a lock
    // an author gave no area is engaged and costs zero: it should still break.
    if state.locked {
        state.locked = false;
        state.lock_broken = true;
    }
    state.powered = false;
    state.target_deg = spec.open_limit_deg;
    // A leaf pushed with exactly a lock's worth of spare energy swings at
    // `BREACH_SWING_DPS`; more is faster, sub-linearly, and the cap is the
    // bound. A default lock is 120 J, so the scale is a real quantity rather
    // than a normalisation of nothing.
    let unit = spec.lock_energy_j().max(1.0);
    let scale = (verdict.remainder_j / unit).max(0.0).sqrt().min(4.0);
    let dps = (BREACH_SWING_DPS * scale.max(0.35)).min(DOOR_MAX_DPS);
    state.vel_dps = if spec.open_limit_deg >= 0.0 {
        dps
    } else {
        -dps
    };
}

/// **The speed a character keeps through a breach, m/s.**
///
/// Two terms and they are different in kind: the lock's joules come off the
/// body's kinetic energy *exactly* — that is what absorbing energy means — and
/// [`BREACH_RESTITUTION`] is the rest of the impact, which is a tunable because
/// nothing about a shoulder against a door is exact.
///
/// A body that would be left with negative energy keeps none, which cannot
/// happen through [`try_break`] (it refuses below the threshold) and is the
/// right answer for any caller that reaches here another way.
pub fn breach_exit_speed_mps(mass_kg: f64, speed_mps: f64, absorbed_j: f64) -> f64 {
    if !mass_kg.is_finite() || mass_kg <= 0.0 || !speed_mps.is_finite() {
        return 0.0;
    }
    let a = if absorbed_j.is_finite() {
        absorbed_j.max(0.0)
    } else {
        0.0
    };
    let v2 = speed_mps * speed_mps - 2.0 * a / mass_kg;
    if !v2.is_finite() || v2 <= 0.0 {
        return 0.0;
    }
    BREACH_RESTITUTION * v2.sqrt()
}

/// **One fixed step of a leaf's motion.**
///
/// Powered swings run at [`DOOR_DRIVE_DPS`] toward their target and stop dead on
/// arrival; free swings coast with drag and settle below [`DOOR_SETTLE_DPS`].
/// `blocked` is the world's answer to "is there something in the way", and a
/// blocked leaf **stops** rather than pushing — which is what makes a door a
/// real obstacle instead of a piston.
///
/// Returns whether the leaf moved, so a caller can skip a pose write for a door
/// that is standing still (`set_body_pose_if_moved`'s own discipline, one layer
/// up).
pub fn advance(spec: &DoorSpec, state: &mut DoorState, dt: f64, blocked: bool) -> bool {
    if !spec.is_usable() || !dt.is_finite() || dt <= 0.0 {
        return false;
    }
    if blocked {
        state.vel_dps = 0.0;
        state.powered = false;
        return false;
    }
    let before = state.open_deg;
    if state.powered {
        let delta = state.target_deg - state.open_deg;
        let step = DOOR_DRIVE_DPS * dt;
        if delta.abs() <= step {
            state.open_deg = state.target_deg;
            state.powered = false;
            state.vel_dps = 0.0;
        } else {
            state.open_deg += step * delta.signum();
            state.vel_dps = DOOR_DRIVE_DPS * delta.signum();
        }
    } else if state.vel_dps != 0.0 {
        state.open_deg += state.vel_dps * dt;
        // Exponential drag, integrated the way every other damping term in this
        // engine is: `v *= 1 / (1 + k dt)`, which is unconditionally stable and
        // has no step size at which it changes sign.
        state.vel_dps /= 1.0 + DOOR_FREE_DRAG_PER_S * dt;
        if state.vel_dps.abs() < DOOR_SETTLE_DPS {
            state.vel_dps = 0.0;
        }
    }
    let clamped = spec.clamp_open(state.open_deg);
    if clamped != state.open_deg {
        // The frame (or the stop) caught it: the leaf is there and it is not
        // moving any more.
        state.open_deg = clamped;
        state.vel_dps = 0.0;
        state.powered = false;
    }
    state.open_deg != before
}

// ── the ECS half ────────────────────────────────────────────────────────────

/// **Every authored door in the world**, in `Guid` order.
///
/// `O(doors)`, and `O(1)` on a level with none — `try_query_filtered` answers
/// `None` when the component was never inserted, which is
/// `interact::candidates_in_world`'s own bound and the same law.
pub fn doors_in_world(world: &EcsWorld) -> Vec<DoorPlacement> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, &Door, &GlobalTransform), With<Door>>() else {
        return Vec::new();
    };
    let mut out: Vec<DoorPlacement> = q
        .iter(w)
        .filter(|(_, d, _)| d.spec.is_usable())
        .map(|(g, d, t)| DoorPlacement {
            guid: g.0,
            hinge: t.translation(),
            spec: d.spec,
            label: d.label.clone(),
        })
        .collect();
    // A bevy archetype walk's order is an implementation detail and must never
    // reach a result.
    out.sort_by_key(|p| p.guid);
    out
}

/// The salt that carves **authored** doors' GUID space out of the scene's own.
const AUTHORED_DOOR_SALT: u128 = 0x6006_0500_444f_4f52_5350_4157_4e21_2121;

/// **The identity of a door a Blueprint hung**, folded from its name and its
/// hinge — a pure function of what the author asked for, never of a counter.
pub fn authored_door_guid(name: &str, hinge: DVec3) -> Uuid {
    let mut x = AUTHORED_DOOR_SALT;
    for b in name.trim().as_bytes() {
        x ^= u128::from(*b);
        x = x
            .rotate_left(11)
            .wrapping_mul(0x0100_0000_01b3_0100_0000_01b3_0100_0001);
    }
    for v in [hinge.x, hinge.y, hinge.z] {
        x ^= u128::from(v.to_bits());
        x = x.rotate_left(29) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    }
    Uuid::from_u128(x)
}

/// **Hang doors from a name-keyed TOML document** — the Blueprint kit's
/// `door.spawn`.
///
/// ```text
/// [front]
/// hinge = [-0.45, 1.05, 4.0]
/// closed_yaw_deg = 90.0
/// inside_yaw_deg = 0.0
/// locked = true
/// ```
///
/// The editor's own doctrine, restated where content reaches it: absent is the
/// default, a malformed document is an **error with nothing applied**, and every
/// numeric is guarded — a non-finite angle takes the default rather than
/// becoming a door nothing can place.
///
/// Returns how many were hung.
pub fn spawn_doors_from_toml(world: &mut EcsWorld, text: &str) -> Result<usize, String> {
    let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    let table = doc
        .as_table()
        .ok_or_else(|| "a door list is a table of named doors".to_string())?;
    // Parsed WHOLE before anything is spawned: a document whose third door is
    // malformed must leave the first two unhung, or a retry hangs them twice.
    let mut plan: Vec<(Uuid, DoorSpec, String, DVec3)> = Vec::new();
    for (name, value) in table {
        let t = value
            .as_table()
            .ok_or_else(|| format!("door {name} is not a table"))?;
        let num = |k: &str, d: f64| -> f64 {
            t.get(k)
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
                .filter(|x| x.is_finite())
                .unwrap_or(d)
        };
        let hinge = match t.get("hinge").and_then(|v| v.as_array()) {
            Some(a) if a.len() == 3 => {
                let c = |i: usize| -> f64 {
                    a[i].as_float()
                        .or_else(|| a[i].as_integer().map(|n| n as f64))
                        .filter(|x| x.is_finite())
                        .unwrap_or(0.0)
                };
                DVec3::new(c(0), c(1), c(2))
            }
            _ => return Err(format!("door {name} needs `hinge = [x, y, z]`")),
        };
        let spec = DoorSpec {
            closed_yaw_deg: num("closed_yaw_deg", 0.0),
            width_m: num("width_m", DEFAULT_DOOR_WIDTH_M),
            height_m: num("height_m", DEFAULT_DOOR_HEIGHT_M),
            thickness_m: num("thickness_m", DEFAULT_DOOR_THICKNESS_M),
            open_limit_deg: num("open_limit_deg", -DEFAULT_OPEN_LIMIT_DEG),
            inside_yaw_deg: num("inside_yaw_deg", 90.0),
            lock_strength_pa: num("lock_strength_pa", DEFAULT_LOCK_STRENGTH_PA),
            lock_area_m2: num("lock_area_m2", DEFAULT_LOCK_AREA_M2),
            lock_side: DoorSide::Inside,
            locked_at_spawn: t.get("locked").and_then(|v| v.as_bool()).unwrap_or(false),
        };
        if !spec.is_usable() {
            return Err(format!("door {name} has unusable geometry"));
        }
        let label = t
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or(name.as_str())
            .to_string();
        plan.push((authored_door_guid(name, hinge), spec, label, hinge));
    }
    let hung = plan.len();
    for (guid, spec, label, hinge) in plan {
        let e = world.spawn_with_guid(guid, &label, None);
        let mut t = crate::components::Transform::IDENTITY;
        t.translation = crate::math::Vec3d::from_dvec3(hinge);
        world
            .world_mut()
            .entity_mut(e)
            .insert((t, Door { spec, label }));
    }
    if hung > 0 {
        world.mark_dirty();
    }
    Ok(hung)
}

/// **Visit every doorway the building grammar planned**, in `(volume, index)`
/// order.
///
/// Here rather than in the physics bridge because this crate is the only one
/// that may name `bevy_ecs` (the facade rule). The bridge turns each into a
/// [`DoorPlacement`] under its own synthetic guid — the identity is the
/// bridge's, because that is where every other synthetic guid in this engine is
/// minted.
///
/// # A VISITOR, and the reason is twenty thousand doorways
///
/// This returned a `Vec` for one afternoon, and on the shipped city that is
/// **19 790 records — about 2 MB — copied out on every call**, three or four
/// times a fixed step, so that a band could then throw 98.8 % of them away. A
/// visitor lets the caller band-check *before* it allocates anything, which is
/// what turns the per-step cost from `O(doorways)` bytes into `O(doorways)`
/// pointer bumps and `O(near)` allocations.
///
/// **Determinism costs nothing here.** The volumes are collected and sorted —
/// there are a hundred of them — and each one's doorways are visited in index
/// order, so the sequence is already `(guid, index)` sorted without a sort over
/// the doorways themselves. A bevy archetype walk's order is an implementation
/// detail and must never reach a result; this is how that is met at this scale.
///
/// `O(doorways)`, and `O(1)` on a level with no `PcgVolume`.
pub fn for_each_volume_doorway(
    world: &EcsWorld,
    mut f: impl FnMut(Uuid, usize, &crate::components::DoorwaySlot),
) {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<
        (&Guid, bevy_ecs::prelude::Entity),
        With<crate::components::PcgVolume>,
    >() else {
        return;
    };
    let mut vols: Vec<(Uuid, bevy_ecs::prelude::Entity)> =
        q.iter(w).map(|(g, e)| (g.0, e)).collect();
    vols.sort_by_key(|(g, _)| *g);
    for (guid, e) in vols {
        let Some(v) = w.get::<crate::components::PcgVolume>(e) else {
            continue;
        };
        for (i, d) in v.doorways.iter().enumerate() {
            f(guid, i, d);
        }
    }
}

/// [`for_each_volume_doorway`] collected — the unbanded list, for a caller that
/// genuinely wants all of them (a test, a debug view). Per-step callers take the
/// visitor.
pub fn volume_doorways(world: &EcsWorld) -> Vec<(Uuid, usize, crate::components::DoorwaySlot)> {
    let mut out = Vec::new();
    for_each_volume_doorway(world, |g, i, d| out.push((g, i, *d)));
    out
}

/// **A kick in flight** — a runtime component on the character, armed by the
/// attack button against a locked door and disarmed by the animation's own
/// notify (or by its fuse).
///
/// It lives in `inf-ecs` rather than beside the gameplay step because this crate
/// is the only one that may name `bevy_ecs` — the facade rule — and a kick has
/// to be a component so that it survives the steps between the press and the
/// impact without a host holding a list.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PendingKick {
    /// The door it is aimed at.
    pub door: Uuid,
    /// Seconds left before it lands anyway.
    pub fuse_s: f64,
}

/// Every kick in flight, in `Guid` order. `O(kicks)`, and `O(1)` when nobody is
/// kicking anything.
pub fn pending_kicks(world: &EcsWorld) -> Vec<(Uuid, PendingKick)> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, &PendingKick), With<PendingKick>>() else {
        return Vec::new();
    };
    let mut out: Vec<(Uuid, PendingKick)> = q.iter(w).map(|(g, k)| (g.0, *k)).collect();
    out.sort_by_key(|(g, _)| *g);
    out
}

/// This character's kick in flight, if it has one.
pub fn pending_kick(world: &EcsWorld, character: Uuid) -> Option<PendingKick> {
    let entity = world.entity_of(character)?;
    world.world().get::<PendingKick>(entity).copied()
}

/// Arm (or re-arm) a kick.
pub fn set_pending_kick(world: &mut EcsWorld, character: Uuid, kick: PendingKick) -> bool {
    let Some(entity) = world.entity_of(character) else {
        return false;
    };
    world.world_mut().entity_mut(entity).insert(kick);
    true
}

/// Disarm a kick.
pub fn clear_pending_kick(world: &mut EcsWorld, character: Uuid) {
    if let Some(entity) = world.entity_of(character) {
        world.world_mut().entity_mut(entity).remove::<PendingKick>();
    }
}

/// The door field, if this world has one.
pub fn door_field(world: &EcsWorld) -> Option<&DoorField> {
    world.world().get_resource::<DoorField>()
}

/// The door field for writing, created on first use.
pub fn door_field_mut(world: &mut EcsWorld) -> &mut DoorField {
    let w = world.world_mut();
    if !w.contains_resource::<DoorField>() {
        w.insert_resource(DoorField::default());
    }
    w.get_resource_mut::<DoorField>()
        .expect("just inserted")
        .into_inner()
}

/// **The trace bytes for doors**, appended to `state_bytes` by both hosts.
///
/// Empty on a level whose doors nobody has touched, which is what keeps every
/// pre-I6 trace byte-identical.
pub fn door_state_bytes(world: &EcsWorld) -> Vec<u8> {
    door_field(world)
        .map(|f| f.state_bytes())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;
    use crate::math::Vec3d;

    const DT: f64 = 1.0 / 60.0;

    fn placement(guid: u128) -> DoorPlacement {
        DoorPlacement {
            guid: Uuid::from_u128(guid),
            hinge: DVec3::ZERO,
            spec: DoorSpec::default(),
            label: "door".into(),
        }
    }

    /// **The lock's price is the P22 rule, arrived at from its two inputs.**
    ///
    /// The number is asserted exactly because it is derived exactly: a steel
    /// bolt (300 MPa) of 4 cm² over P22's millimetre of crack opening is 120 J,
    /// and every other claim in this wave is measured against it.
    #[test]
    fn a_lock_costs_what_the_p22_rule_says_one_bond_of_steel_costs() {
        let spec = DoorSpec::default();
        let want = DEFAULT_LOCK_STRENGTH_PA * DEFAULT_LOCK_AREA_M2 * crate::CRACK_OPENING_M;
        println!(
            "a default lock is {} Pa over {} m2 at a {} m crack: {} J",
            DEFAULT_LOCK_STRENGTH_PA,
            DEFAULT_LOCK_AREA_M2,
            crate::CRACK_OPENING_M,
            spec.lock_energy_j()
        );
        assert!((spec.lock_energy_j() - 120.0).abs() < 1e-9, "not 120 J");
        assert!((spec.lock_energy_j() - want).abs() < 1e-12);
        // …and it is the SAME function the fracture solve prices a chunk face
        // with, which is the whole of "one door for the kick and the crash".
        let d = Destructible {
            strength: DEFAULT_LOCK_STRENGTH_PA,
            ..Destructible::default()
        };
        assert_eq!(d.bond_energy_j(DEFAULT_LOCK_AREA_M2), spec.lock_energy_j());
        // A lock made of nothing cannot hold, and says so rather than dividing.
        let mut hollow = spec;
        hollow.lock_area_m2 = 0.0;
        assert_eq!(hollow.lock_energy_j(), 0.0);
        hollow.lock_area_m2 = f64::NAN;
        assert_eq!(hollow.lock_energy_j(), 0.0);
    }

    /// **One kick opens a domestic door and the margin is a number.**
    #[test]
    fn a_kick_carries_more_than_a_default_lock_and_less_than_two() {
        let spec = DoorSpec::default();
        let kick = kick_energy_j();
        println!(
            "a kick is {kick} J against a {} J lock: {} J to spare",
            spec.lock_energy_j(),
            kick - spec.lock_energy_j()
        );
        assert!((kick - 151.875).abs() < 1e-9);
        assert!(kick > spec.lock_energy_j(), "a kick must open a house door");
        // A lock a designer doubles takes more than one kick — the knob works.
        let mut stout = spec;
        stout.lock_strength_pa *= 2.0;
        assert!(
            kick < stout.lock_energy_j(),
            "the strength knob does nothing"
        );
    }

    /// **The speed gate sits between the run and the sprint**, and the energies
    /// either side of it are what decide the lock.
    #[test]
    fn the_breach_threshold_is_between_a_run_and_a_sprint() {
        let spec = DoorSpec::default();
        let run = breach_energy_j(DEFAULT_BODY_MASS_KG, 3.75);
        let gate = breach_energy_j(DEFAULT_BODY_MASS_KG, BREACH_SPEED_MPS);
        let sprint = breach_energy_j(DEFAULT_BODY_MASS_KG, 6.5);
        println!(
            "run 3.750 m/s = {run} J - gate {BREACH_SPEED_MPS} m/s = {gate} J - sprint 6.500 m/s = {sprint} J, lock {} J",
            spec.lock_energy_j()
        );
        // A claim about the SOURCE — the gate has to sit between the run and
        // the sprint — so the build is where it belongs, which is the same
        // reasoning `interact::VIEW_CONE_EPSILON_DEG`'s arm gives for its own.
        const { assert!(BREACH_SPEED_MPS > 3.75 && BREACH_SPEED_MPS < 6.5) };
        // …and the same claim **derived** rather than restated (I6 audit): the
        // const above pins two literals, so a wave that retuned the character's
        // own speeds would leave it passing while the sentence it encodes went
        // false. `CharacterMovement::default()` is not a const fn, so this half
        // runs.
        let cm = crate::components::CharacterMovement::default();
        println!(
            "the character's own defaults are run {} m/s and sprint {} m/s",
            cm.run_speed_mps, cm.sprint_speed_mps
        );
        assert!(
            BREACH_SPEED_MPS > cm.run_speed_mps && BREACH_SPEED_MPS < cm.sprint_speed_mps,
            "the breach gate is no longer between the run ({}) and the sprint ({})",
            cm.run_speed_mps,
            cm.sprint_speed_mps
        );
        assert!((cm.run_speed_mps - 3.75).abs() < 1e-9);
        assert!((cm.sprint_speed_mps - 6.5).abs() < 1e-9);
        assert!((gate - 1000.0).abs() < 1e-9);
        assert!((sprint - 1690.0).abs() < 1e-9);
        // The energy gate is not the interesting one at these speeds — the SPEED
        // gate is — and a stouter lock is what makes the energy gate bite.
        let mut vault = spec;
        vault.lock_strength_pa = 6.0e11;
        assert!(
            vault.lock_energy_j() > sprint,
            "a vault must refuse a sprint"
        );
        assert!(
            !try_break(
                &vault,
                &DoorState {
                    locked: true,
                    ..DoorState::fresh(&vault)
                },
                sprint
            )
            .broke
        );
    }

    /// **The momentum kept through a breach**, with the two terms separated.
    #[test]
    fn a_breach_keeps_most_of_its_speed_and_the_lock_takes_the_rest() {
        let spec = DoorSpec::default();
        let m = DEFAULT_BODY_MASS_KG;
        let v = 6.5;
        let verdict = try_break(
            &spec,
            &DoorState {
                locked: true,
                ..DoorState::fresh(&spec)
            },
            breach_energy_j(m, v),
        );
        assert!(verdict.broke);
        assert!((verdict.absorbed_j - 120.0).abs() < 1e-9);
        let out = breach_exit_speed_mps(m, v, verdict.absorbed_j);
        println!(
            "a sprint at {v} m/s through a {} J lock leaves {out} m/s, {:.1} % kept, {} J left for the leaf",
            verdict.absorbed_j,
            100.0 * out / v,
            verdict.remainder_j
        );
        assert!(
            out > 0.8 * v,
            "the mode must continue with most of its speed"
        );
        assert!(out < v, "the lock must cost something");
        // The exact arithmetic, so a change to either term is visible.
        let want = BREACH_RESTITUTION * (v * v - 2.0 * 120.0 / m).sqrt();
        assert!((out - want).abs() < 1e-12);
        // Hostile inputs are refusals, not infinities.
        assert_eq!(breach_exit_speed_mps(0.0, v, 0.0), 0.0);
        assert_eq!(breach_exit_speed_mps(m, f64::NAN, 0.0), 0.0);
        assert_eq!(breach_exit_speed_mps(m, 1.0, 1.0e9), 0.0);
    }

    /// **Damage is not banked** — P22's rule, met at a door.
    #[test]
    fn two_half_kicks_do_not_open_what_one_kick_would_not() {
        let spec = DoorSpec {
            locked_at_spawn: true,
            ..Default::default()
        };
        let state = DoorState::fresh(&spec);
        let half = spec.lock_energy_j() * 0.6;
        let a = try_break(&spec, &state, half);
        assert!(!a.broke);
        assert_eq!(a.absorbed_j, 0.0, "a refused blow must absorb nothing");
        let b = try_break(&spec, &state, half);
        assert!(!b.broke, "the second half must not finish the first");
        assert!(
            try_break(&spec, &state, spec.lock_energy_j()).broke,
            "exactly enough must do it"
        );
    }

    /// **The leaf swings, stops at its frame, and settles.**
    #[test]
    fn a_powered_door_reaches_its_stop_and_stays_there() {
        let p = placement(1);
        let mut s = DoorState::fresh(&p.spec);
        assert_eq!(toggle(&p.spec, &mut s), DoorVerdict::Opening);
        let mut steps = 0;
        while !s.is_at_rest() && steps < 600 {
            advance(&p.spec, &mut s, DT, false);
            steps += 1;
        }
        println!(
            "a powered swing took {steps} steps to {} degrees",
            s.open_deg
        );
        assert!((s.open_deg - DEFAULT_OPEN_LIMIT_DEG).abs() < 1e-9);
        assert!(s.is_open(&p.spec));
        assert!(
            steps > 1 && steps < 60,
            "a door is not teleported and not glacial"
        );
        // A settled door does not move again, which is what keeps a trace from
        // changing for ever.
        assert!(!advance(&p.spec, &mut s, DT, false));
        // …and it shuts again.
        assert_eq!(toggle(&p.spec, &mut s), DoorVerdict::Closing);
        for _ in 0..600 {
            advance(&p.spec, &mut s, DT, false);
        }
        assert_eq!(s.open_deg, 0.0);
        assert!(!s.is_open(&p.spec));
    }

    /// **A blocked leaf stops**, which is what makes it an obstacle.
    #[test]
    fn a_blocked_door_stops_instead_of_pushing() {
        let p = placement(1);
        let mut s = DoorState::fresh(&p.spec);
        toggle(&p.spec, &mut s);
        for _ in 0..5 {
            advance(&p.spec, &mut s, DT, false);
        }
        let caught = s.open_deg;
        assert!(caught > 0.0);
        assert!(!advance(&p.spec, &mut s, DT, true), "a blocked leaf moved");
        assert_eq!(s.open_deg, caught);
        assert!(s.is_at_rest(), "a blocked leaf must not stay under power");
    }

    /// **A broken lock does not come back**, and the free swing settles.
    #[test]
    fn a_kicked_door_flies_open_and_cannot_be_relocked() {
        let spec = DoorSpec {
            locked_at_spawn: true,
            ..Default::default()
        };
        let p = DoorPlacement {
            spec,
            ..placement(1)
        };
        let mut s = DoorState::fresh(&spec);
        assert!(s.locked);
        assert_eq!(
            toggle(&spec, &mut s),
            DoorVerdict::Locked,
            "a locked door opened"
        );
        assert_eq!(s.open_deg, 0.0);
        let v = try_break(&spec, &s, kick_energy_j());
        assert!(v.broke);
        apply_break(&spec, &mut s, &v);
        assert!(s.lock_broken && !s.locked);
        assert!(s.vel_dps > 0.0, "the leaf got no swing out of the kick");
        let mut steps = 0;
        while !s.is_at_rest() && steps < 600 {
            advance(&spec, &mut s, DT, false);
            steps += 1;
        }
        println!(
            "a kick with {} J to spare swung the leaf to {} degrees in {steps} steps",
            v.remainder_j, s.open_deg
        );
        assert!(s.is_open(&spec), "a kicked door must be open");
        // Re-locking is refused as a value.
        assert_eq!(
            set_locked(&spec, &mut s, DoorSide::Inside, true),
            DoorVerdict::WrongSide
        );
        assert!(!s.locked);
        // And the door now toggles like any other.
        assert_eq!(toggle(&spec, &mut s), DoorVerdict::Closing);
        // …and the leaf really left the doorway: its centre moved a quarter of
        // its own width, measured against where it started.
        let shut = leaf_pose(&p, 0.0).0;
        let open = leaf_pose(&p, s.open_deg).0;
        println!(
            "the kicked leaf's centre moved {} m",
            (open - shut).length()
        );
        assert!((open - shut).length() > spec.width_m * 0.25);
    }

    /// **THE WEAKEST BREACH STILL REACHES THE FRAME** (I6 audit).
    ///
    /// `apply_break` gives the leaf `sqrt(remainder / lock) × BREACH_SWING_DPS`
    /// with a floor of `0.35`, and nothing pushes it afterwards — so whether a
    /// door that gave with **exactly** nothing to spare ends up open or ends up
    /// ajar is a question about the drag, and it had no arm. `d3::door` used to
    /// carry a `target_deg` re-write that claimed to finish the swing "under
    /// power" and did not (it left `powered` false and wrote the value
    /// `apply_break` had already written); it is gone, and this is what makes
    /// its absence safe rather than lucky.
    ///
    /// Both signs, because a leaf that opens the other way is a leaf whose
    /// clamp has the other bound.
    #[test]
    fn a_breach_with_nothing_to_spare_still_swings_the_leaf_to_its_stop() {
        for limit in [DEFAULT_OPEN_LIMIT_DEG, -DEFAULT_OPEN_LIMIT_DEG] {
            let spec = DoorSpec {
                locked_at_spawn: true,
                open_limit_deg: limit,
                ..Default::default()
            };
            let mut s = DoorState::fresh(&spec);
            // Exactly the lock's own worth: it breaks with a remainder of zero,
            // which is the floor case.
            let v = try_break(&spec, &s, spec.lock_energy_j());
            assert!(v.broke);
            assert_eq!(v.remainder_j, 0.0, "this arm is about the floor");
            apply_break(&spec, &mut s, &v);
            let launch = s.vel_dps;
            let mut steps = 0;
            while !s.is_at_rest() && steps < 3_000 {
                advance(&spec, &mut s, DT, false);
                steps += 1;
            }
            println!(
                "a breach with nothing to spare launched the leaf at {launch} deg/s and it \
                 settled at {} of {limit} degrees after {steps} steps",
                s.open_deg
            );
            assert_eq!(
                s.open_deg, limit,
                "the leaf coasted to a stop short of its own frame"
            );
            assert!(s.is_open(&spec));
        }
    }

    /// **The lock verb has a side.**
    #[test]
    fn a_lock_works_from_its_own_face_and_not_the_other() {
        let spec = DoorSpec::default();
        let hinge = DVec3::ZERO;
        // `inside_yaw_deg` is 90 by default, i.e. the inside normal is `+X`.
        assert_eq!(
            spec.side_of(hinge, DVec3::new(1.0, 0.0, 0.0)),
            DoorSide::Inside
        );
        assert_eq!(
            spec.side_of(hinge, DVec3::new(-1.0, 0.0, 0.0)),
            DoorSide::Outside
        );
        let mut s = DoorState::fresh(&spec);
        assert_eq!(
            set_locked(&spec, &mut s, DoorSide::Outside, true),
            DoorVerdict::WrongSide,
            "the lock worked from the wrong face"
        );
        assert!(!s.locked);
        assert_eq!(
            set_locked(&spec, &mut s, DoorSide::Inside, true),
            DoorVerdict::LockedNow
        );
        assert!(s.locked);
        assert_eq!(
            set_locked(&spec, &mut s, DoorSide::Inside, false),
            DoorVerdict::Unlocked
        );
        assert!(!s.locked);
        assert_eq!(DoorSide::Inside.flip(), DoorSide::Outside);
    }

    /// **The leaf's box is where the geometry says it is**, at both ends of the
    /// swing, and the prompt does not ride it.
    ///
    /// The tolerance is **the portable trigonometry's own error and nothing
    /// else** — `psin64` documents an endpoint error near `5.7e-8` and this
    /// measures it on a 0.45 m lever — so the arm prints what it measured rather
    /// than trusting a round number, exactly as
    /// `interact::VIEW_CONE_EPSILON_DEG`'s own arm does. A wider tolerance would
    /// hide a placement mistake; `f64::sin` would be tighter and not portable,
    /// which is the trade the P14 law already made.
    #[test]
    fn the_leaf_is_half_a_width_from_its_hinge_at_every_angle() {
        /// Metres. Two orders above the measured error and five below a door.
        const EPS_M: f64 = 1.0e-6;
        let mut p = placement(1);
        p.hinge = DVec3::new(10.0, 2.0, -4.0);
        p.spec.closed_yaw_deg = 0.0;
        let (c0, y0, half) = leaf_pose(&p, 0.0);
        let shut_err = (c0 - DVec3::new(10.0, 2.0, -4.0 + 0.45)).length();
        assert!((y0 - 0.0).abs() < 1e-12);
        // **The long axis is local +Z**, because yaw zero is `+Z`. See
        // `leaf_pose` for the defect that costs.
        assert!((half.x - 0.03).abs() < 1e-12, "the thickness is not on x");
        assert!((half.y - 1.05).abs() < 1e-12);
        assert!((half.z - 0.45).abs() < 1e-12, "the width is not on z");
        // …and the box really does cover the leaf: rotating its local far corner
        // by the leaf's own yaw lands on the free edge, which is what a box with
        // its axes the other way round would not do.
        let r = y0.to_radians();
        let local_end = DVec3::new(0.0, 0.0, half.z);
        let end = c0
            + DVec3::new(
                local_end.x * inf_math::pcos64(r) + local_end.z * inf_math::psin64(r),
                0.0,
                -local_end.x * inf_math::psin64(r) + local_end.z * inf_math::pcos64(r),
            );
        let free_edge = p.hinge + DVec3::new(0.0, 0.0, p.spec.width_m);
        assert!(
            (end - free_edge).length() < EPS_M,
            "the box's far face is at {end:?}, the free edge at {free_edge:?}"
        );
        // Ninety degrees round: the leaf points along +X.
        let (c1, y1, _) = leaf_pose(&p, 90.0);
        let open_err = (c1 - DVec3::new(10.45, 2.0, -4.0)).length();
        println!(
            "a leaf hinged at {:?} sits at {c0:?} shut and {c1:?} open; the portable trig places them {shut_err:e} and {open_err:e} m off the exact answer",
            p.hinge
        );
        assert!(shut_err < EPS_M, "{c0:?}");
        assert!(open_err < EPS_M, "{c1:?}");
        assert!((y1 - 90.0).abs() < 1e-12);
        // The distance from the hinge never changes — it is a hinge.
        let mut worst = 0.0_f64;
        for deg in [0.0, 17.5, 45.0, 95.0] {
            let (c, _, _) = leaf_pose(&p, deg);
            worst = worst.max(((c - p.hinge).length() - 0.45).abs());
        }
        println!("the lever arm varies by at most {worst:e} m across the swing");
        assert!(worst < EPS_M, "the hinge is not a hinge");
        // The prompt stays in the doorway — the same point, bit for bit.
        assert_eq!(prompt_position(&p), c0);
    }

    /// Hostile geometry is refused everywhere rather than silently placed.
    #[test]
    fn a_door_with_unusable_numbers_does_nothing() {
        for bad in [
            DoorSpec {
                width_m: 0.0,
                ..Default::default()
            },
            DoorSpec {
                height_m: f64::NAN,
                ..Default::default()
            },
            DoorSpec {
                open_limit_deg: 0.0,
                ..Default::default()
            },
            DoorSpec {
                closed_yaw_deg: f64::INFINITY,
                ..Default::default()
            },
            DoorSpec {
                thickness_m: -0.1,
                ..Default::default()
            },
        ] {
            assert!(!bad.is_usable(), "{bad:?} passed as usable");
            let mut s = DoorState::fresh(&bad);
            assert_eq!(toggle(&bad, &mut s), DoorVerdict::Unusable);
            assert_eq!(
                set_locked(&bad, &mut s, DoorSide::Inside, true),
                DoorVerdict::Unusable
            );
            assert!(!advance(&bad, &mut s, DT, false));
        }
        // A usable door still refuses a hostile step.
        let good = DoorSpec::default();
        let mut s = DoorState::fresh(&good);
        toggle(&good, &mut s);
        assert!(!advance(&good, &mut s, f64::NAN, false));
        assert!(!advance(&good, &mut s, 0.0, false));
        assert_eq!(s.open_deg, 0.0);
    }

    /// **Absent is closed**, and the trace bytes of an untouched level are
    /// empty — which is what makes every pre-I6 trace byte-identical.
    #[test]
    fn an_untouched_door_field_costs_the_trace_nothing() {
        let mut w = EcsWorld::new();
        assert!(door_state_bytes(&w).is_empty());
        let spec = DoorSpec {
            locked_at_spawn: true,
            ..Default::default()
        };
        let guid = Uuid::from_u128(7);
        {
            let f = door_field_mut(&mut w);
            assert!(f.is_empty());
            // Reading does not insert.
            let s = f.get(guid, &spec);
            assert!(s.locked, "a door that spawns locked must read locked");
            assert_eq!(s.open_deg, 0.0);
            assert!(f.is_empty(), "a read materialised an entry");
        }
        assert!(door_state_bytes(&w).is_empty());
        {
            let f = door_field_mut(&mut w);
            toggle(&spec, f.entry(guid, &spec));
            assert_eq!(f.len(), 1);
        }
        let bytes = door_state_bytes(&w);
        assert_eq!(bytes.len(), 16 + 24 + 3, "the record's width moved");
        assert_eq!(&bytes[..16], guid.as_bytes());
    }

    /// **The ECS walk is `Guid`-ordered and costs nothing on a level with no
    /// doors.**
    #[test]
    fn the_door_walk_is_sorted_and_empty_without_doors() {
        let mut w = EcsWorld::new();
        assert!(doors_in_world(&w).is_empty());
        for guid in [9u128, 3, 5] {
            let e = w.spawn_with_guid(Uuid::from_u128(guid), "Door", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(guid as f64, 0.0, 0.0);
            w.world_mut().entity_mut(e).insert((
                Door {
                    spec: DoorSpec::default(),
                    label: "door".into(),
                },
                t,
            ));
        }
        // …and one whose numbers are unusable, which must not be offered.
        let e = w.spawn_with_guid(Uuid::from_u128(1), "Broken", None);
        w.world_mut().entity_mut(e).insert((
            Door {
                spec: DoorSpec {
                    width_m: 0.0,
                    ..Default::default()
                },
                label: "broken".into(),
            },
            Transform::IDENTITY,
        ));
        w.mark_dirty();
        w.propagate();
        let doors = doors_in_world(&w);
        let ids: Vec<u128> = doors.iter().map(|d| d.guid.as_u128()).collect();
        assert_eq!(ids, vec![3, 5, 9], "the walk is not Guid-ordered");
        assert!((doors[0].hinge.x - 3.0).abs() < 1e-12);
    }

    /// **The doorway VISITOR is `(volume, index)` ordered without sorting the
    /// doorways**, which is what lets a city's twenty thousand be walked without
    /// being copied.
    ///
    /// The order is the claim: the volumes are sorted (there are a hundred) and
    /// each one's slots are visited in index order, so the sequence is already
    /// sorted. A bevy archetype walk's own order must never reach a result, and
    /// this is how that is met at a scale where sorting the result would be the
    /// cost the visitor exists to avoid.
    #[test]
    fn the_doorway_visitor_walks_volumes_in_guid_order_and_slots_in_index_order() {
        use crate::components::{DoorwaySlot, PcgVolume};
        let mut w = EcsWorld::new();
        assert!(
            volume_doorways(&w).is_empty(),
            "an empty world walked something"
        );
        let slot = |x: f64| DoorwaySlot {
            hinge: DVec3::new(x, 1.05, 0.0),
            closed_yaw_deg: 0.0,
            width_m: DEFAULT_DOOR_WIDTH_M,
            height_m: DEFAULT_DOOR_HEIGHT_M,
            thickness_m: DEFAULT_DOOR_THICKNESS_M,
            inside_yaw_deg: 90.0,
            exterior: false,
            floor: 0,
        };
        // Spawned HIGH guid first, so a walk that followed insertion order would
        // answer the other way round.
        for (guid, n) in [(9u128, 3usize), (2, 2)] {
            let e = w.spawn_with_guid(Uuid::from_u128(guid), "Vol", None);
            let mut vol = PcgVolume::default();
            let slots: Vec<DoorwaySlot> =
                (0..n).map(|i| slot(guid as f64 + i as f64 * 0.1)).collect();
            vol.set_population(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                slots,
                Vec::new(),
                Default::default(),
                Vec::new(),
                Vec::new(),
            );
            w.world_mut().entity_mut(e).insert(vol);
        }
        w.mark_dirty();
        w.propagate();
        let seen: Vec<(u128, usize)> = volume_doorways(&w)
            .into_iter()
            .map(|(g, i, _)| (g.as_u128(), i))
            .collect();
        println!("the visitor walked {seen:?}");
        assert_eq!(seen, vec![(2, 0), (2, 1), (9, 0), (9, 1), (9, 2)]);
        // …and the slots really arrive, not just their indices.
        let hinges: Vec<f64> = {
            let mut v = Vec::new();
            for_each_volume_doorway(&w, |_, _, d| v.push(d.hinge.x));
            v
        };
        assert_eq!(hinges.len(), 5);
        assert!((hinges[0] - 2.0).abs() < 1e-12, "{hinges:?}");
        assert!((hinges[2] - 9.0).abs() < 1e-12, "{hinges:?}");
    }
}
