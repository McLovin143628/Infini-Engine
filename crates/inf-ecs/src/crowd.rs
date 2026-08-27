//! **THE SIM-LOD TIER SYSTEM** (wave NPC1a) — what a crowd NPC costs, decided
//! once, from sim state, in a door both hosts call.
//!
//! # The measurement this exists for
//!
//! Before this module the engine had **zero animation LOD** (grep-verified): the
//! whole pose pipeline — drive → pelvis → foot IK → hand IK → goals → ragdoll —
//! ran for every [`AnimStateMachine`] entity on every fixed step regardless of
//! where it stood, every character got a rapier controller and a capsule, and
//! every posed character contributed `36 + 40 · joints` bytes to the sim trace:
//! **6 476 B per NPC per step** at the starter character's 161 bones
//! ([`crate::pose::pose_state_bytes`]). A thousand of those is 5.8 GB of retained
//! trace in one gate process and a fixed step three orders past its 6.0 ms
//! ratchet. None of that improves by making any one pass faster.
//!
//! So the crowd pays by **tier**, and the tier is one decision:
//!
//! | tier | rapier | pose | hand IK | position |
//! |---|---|---|---|---|
//! | [`Full`](CrowdTier::Full) | capsule | full | yes | route |
//! | [`Near`](CrowdTier::Near) | capsule | full | **no** | route |
//! | [`Far`](CrowdTier::Far) | **none** | **none** | no | route |
//! | [`Dormant`](CrowdTier::Dormant) | — | — | — | the record remembers |
//!
//! **The position law is the same at every tier**, and that is deliberate rather
//! than unfinished: an agent's place is `route(clock)`, a pure function of its
//! record and the step count, at Full exactly as at Far. NPC1c replaces that
//! function with a path over the road graph and steers the near tiers through
//! `move_and_slide`; until it does, a tier transition is **invisible in the
//! transform**, which is what makes the PIE-==-shipping-across-transitions arm a
//! test of the tier machinery rather than of the route.
//!
//! # THE VISIBILITY LAW: the tier never reads a camera
//!
//! The standing law is *visibility filters what is DRAWN, never what is
//! SIMULATED*. A tier that read the camera would break it outright — two players
//! looking in different directions would simulate different crowds, and PIE would
//! stop equalling shipping the moment the editor's free camera moved.
//!
//! So [`CrowdBand`] is [`crate::band::SimBand`]'s shape, one radius wider:
//! anchors are the entities carrying [`StreamingSource`] (the set P16's cell
//! activation already reads to decide which parts of the world exist at all),
//! snapped to the same [`BAND_LATTICE_M`] lattice, ordered, deduplicated. There
//! is **no camera argument** to [`step_crowd`], so a caller cannot pass one by
//! accident.
//!
//! # Hysteresis is REFUSED, and it is refused for the same reason
//!
//! A tier is a *function of sim state*, not of the history of sim states. A
//! hysteretic tier would agree with itself inside each host and diverge between
//! them the first time one of them started mid-trace — which is the whole
//! property the island gate exists to protect. The cost is the one
//! `SimBand`'s own module states and measures: an agent parked on a lattice line
//! re-tiers every step, alternating between exactly **two** tiers, never
//! wandering. See `an_agent_parked_on_a_tier_boundary_alternates_between_two`.
//!
//! # It fails toward FULL
//!
//! A world with no streaming source at all is [`CrowdBand::unbounded`]: every
//! agent is [`Full`](CrowdTier::Full), which is the pre-NPC1a behaviour of a
//! character in every fixture this tree already has. Same for a world whose only
//! sources carry non-finite positions, and same for an agent whose own position
//! is not finite. Dropping a tier is the dangerous direction (an NPC stops
//! animating, stops colliding, or stops existing); keeping it is merely slow.
//!
//! # No schema moves
//!
//! [`CrowdPopulationRes`] is a bevy **resource** and [`CrowdAgent`] is a
//! component the scene serializer does not know about, exactly as
//! [`crate::deform::DeformFieldRes`] is: the `.inf_lvl` walk writes
//! `RuntimeEntity` fields and never a resource, so nothing here can be saved and
//! **scene v26 does not move**. That is correct for NPC1a — a test population is
//! transient — and it is the shape NPC1d inherits: a population is *data in the
//! recipe*, and bodies materialize by tier.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::{Component, Entity, Resource};
use glam::DVec3;
use uuid::Uuid;

use crate::band::{streaming_sources, BAND_LATTICE_M};
use crate::components::{
    AnimStateMachine, BodyKind3D, CharacterController3D, Collider3D, ColliderShape3DKind,
    RigidBody3D, SkeletalMesh, Transform,
};
use crate::math::Vec3d;
use crate::world::EcsWorld;

// ── the tier ────────────────────────────────────────────────────────────────

/// What one crowd NPC costs this step.
///
/// Ordered cheapest-last: `Full < Near < Far < Dormant`, so `max`/`min` over a
/// set of tiers mean "the dearest" / "the cheapest" without a lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum CrowdTier {
    /// Today's whole pipeline: a capsule in rapier, the full pose evaluation and
    /// the hand IK pass. What every character in this tree got unconditionally
    /// before NPC1a, and what a hero-class actor still gets — see
    /// [`step_crowd`]'s "the hero is untouched" note.
    #[default]
    Full,
    /// A capsule and a full pose, without the hand pass. The reach and the
    /// finger closure are the passes a viewer cannot resolve at distance and the
    /// ones that need a `HandIk` request to do anything at all, so they are the
    /// first thing off the ladder.
    Near,
    /// **Kinematic.** No rapier body, no pose evaluation, no machine advance:
    /// the transform is `route(clock)` and the trace carries a cached digest of
    /// the last pose the agent published instead of 161 joints.
    Far,
    /// **Data only.** The entity is despawned; [`CrowdRecord`] remembers where it
    /// stood, what it was doing and what it looked like, and re-materializes it
    /// the step its tier comes back.
    Dormant,
}

impl CrowdTier {
    /// Whether the pose pipeline runs for this tier (`Full`/`Near`).
    ///
    /// Read by [`crate::pose::step_pose_evaluation`], which is the one place a
    /// pose is evaluated in this engine, so a tier cannot mean one thing in the
    /// editor and another in the player.
    #[inline]
    pub fn poses(self) -> bool {
        matches!(self, CrowdTier::Full | CrowdTier::Near)
    }

    /// Whether the SK1b hand pass runs (`Full` only).
    #[inline]
    pub fn hand_ik(self) -> bool {
        matches!(self, CrowdTier::Full)
    }

    /// Whether this tier is solid — i.e. whether the 3D bridge gives it a rapier
    /// body and collider (`Full`/`Near`).
    #[inline]
    pub fn has_body(self) -> bool {
        matches!(self, CrowdTier::Full | CrowdTier::Near)
    }

    /// Whether an entity exists for this tier at all (everything but `Dormant`).
    #[inline]
    pub fn materialized(self) -> bool {
        !matches!(self, CrowdTier::Dormant)
    }

    /// The byte the trace folds. Frozen: these discriminants are compared
    /// between two hosts and, through the replay path, two machines.
    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            CrowdTier::Full => 0,
            CrowdTier::Near => 1,
            CrowdTier::Far => 2,
            CrowdTier::Dormant => 3,
        }
    }

    /// The label the instruments print.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            CrowdTier::Full => "full",
            CrowdTier::Near => "near",
            CrowdTier::Far => "far",
            CrowdTier::Dormant => "dormant",
        }
    }
}

// ── the radii ───────────────────────────────────────────────────────────────

/// Metres inside which an agent is [`Full`](CrowdTier::Full).
///
/// **Chosen against the collider band, not invented.**
/// [`crate::band::DEFAULT_COLLIDER_NEAR_M`] is 64 m — the radius inside which a
/// building is solid — and an NPC you can walk into has to be inside a world you
/// can walk into. 32 m is half of it, which keeps the dearest tier to the
/// quarter-area a viewer can actually read a finger pose at, and is still nine
/// times a fixed step's travel at a sprint.
pub const DEFAULT_CROWD_FULL_M: f64 = 32.0;

/// Metres inside which an agent is at worst [`Near`](CrowdTier::Near).
///
/// Wider than [`crate::band::DEFAULT_COLLIDER_NEAR_M`] on purpose: an NPC is one
/// capsule where a grammar building is dozens of solids, so the tier that keeps
/// a body can afford to reach past the tier that keeps a *building's* body.
pub const DEFAULT_CROWD_NEAR_M: f64 = 96.0;

/// Metres past which an agent is [`Dormant`](CrowdTier::Dormant).
///
/// [`crate::band::DEFAULT_COLLIDER_FAR_M`] is 1 024 m, and this is half of it,
/// because a dormant agent is *gone* rather than cheap: the radius has to be
/// comfortably inside the cell-activation neighbourhood so an agent does not
/// dematerialize inside the world the player can see. NPC1b's impostors are what
/// moves it out.
pub const DEFAULT_CROWD_FAR_M: f64 = 512.0;

/// The three radii, in metres, ascending — the shape [`CrowdBand`] is built from.
pub const DEFAULT_CROWD_RADII: (f64, f64, f64) = (
    DEFAULT_CROWD_FULL_M,
    DEFAULT_CROWD_NEAR_M,
    DEFAULT_CROWD_FAR_M,
);

// ── the band ────────────────────────────────────────────────────────────────

/// **The one door**: which tier an agent takes this step.
///
/// [`crate::band::SimBand`] with three radii instead of two, and every one of
/// that type's rules restated in code rather than in prose: anchors are
/// [`StreamingSource`] entities, snapped to [`BAND_LATTICE_M`], sorted,
/// deduplicated, non-finite ones dropped; an empty anchor set is
/// [`unbounded`](Self::unbounded); the [`stamp`](Self::stamp) is a membership
/// hash and the only legal operation on it is `==`.
///
/// [`StreamingSource`]: crate::components::StreamingSource
#[derive(Debug, Clone, PartialEq)]
pub struct CrowdBand {
    anchors: Vec<DVec3>,
    full_m: f64,
    near_m: f64,
    far_m: f64,
    unbounded: bool,
    stamp: u64,
}

impl Default for CrowdBand {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl CrowdBand {
    /// Everything is [`Full`](CrowdTier::Full) — the answer for a world with no
    /// streaming source, and the pre-NPC1a behaviour of every fixture in this
    /// tree.
    pub fn unbounded() -> Self {
        Self {
            anchors: Vec::new(),
            full_m: f64::INFINITY,
            near_m: f64::INFINITY,
            far_m: f64::INFINITY,
            unbounded: true,
            stamp: 0,
        }
    }

    /// The band a world's own streaming sources define.
    pub fn from_world(world: &EcsWorld, radii: (f64, f64, f64)) -> Self {
        Self::from_anchors(streaming_sources(world).into_iter().map(|(p, _)| p), radii)
    }

    /// The band a set of anchor positions defines.
    ///
    /// Non-finite anchors are dropped; if that leaves none — or if any radius is
    /// not finite, or they are not ascending — the band is
    /// [`unbounded`](Self::unbounded), failing toward `Full` per the module docs.
    pub fn from_anchors(anchors: impl IntoIterator<Item = DVec3>, radii: (f64, f64, f64)) -> Self {
        let (full_m, near_m, far_m) = radii;
        let mut snapped: Vec<[i64; 2]> = anchors
            .into_iter()
            .filter(|p| p.x.is_finite() && p.z.is_finite())
            .map(|p| [lattice(p.x), lattice(p.z)])
            .collect();
        let ordered = full_m <= near_m && near_m <= far_m;
        let finite = full_m.is_finite() && near_m.is_finite() && far_m.is_finite();
        if snapped.is_empty() || !finite || !ordered {
            return Self::unbounded();
        }
        snapped.sort_unstable();
        snapped.dedup();

        // FNV-1a over the lattice coordinates and all three radii — `SimBand`'s
        // mixer, spelled the same way, because two stamps that meant the same
        // thing and hashed differently would be worse than no stamp at all.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut fold = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for c in &snapped {
            fold(c[0] as u64);
            fold(c[1] as u64);
        }
        fold(full_m.to_bits());
        fold(near_m.to_bits());
        fold(far_m.to_bits());

        Self {
            anchors: snapped
                .iter()
                .map(|c| DVec3::new(unlattice(c[0]), 0.0, unlattice(c[1])))
                .collect(),
            full_m,
            near_m,
            far_m,
            unbounded: false,
            stamp: h,
        }
    }

    /// `true` when nothing is banded — every agent is `Full`.
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.unbounded
    }

    /// The band's membership stamp. `0` for an unbounded band.
    #[inline]
    pub fn stamp(&self) -> u64 {
        self.stamp
    }

    /// The lattice-snapped anchors the band is measured about.
    #[inline]
    pub fn anchors(&self) -> &[DVec3] {
        &self.anchors
    }

    /// The three radii, in metres.
    #[inline]
    pub fn radii(&self) -> (f64, f64, f64) {
        (self.full_m, self.near_m, self.far_m)
    }

    /// **The tier a point takes in this band** — the decision, in one place.
    ///
    /// Measured in the XZ plane from the nearest anchor, exactly as the collider
    /// band measures a building: an NPC on a roof and an NPC in the street below
    /// are the same distance away for the purpose of what they cost, and folding
    /// height in would make a tier depend on terrain the agent is standing on.
    ///
    /// A non-finite point is `Full` in an unbounded band (refusing it would
    /// silently change a fixture's pre-NPC1a behaviour) and `Dormant` in a
    /// banded one, because a NaN distance compares false against every radius
    /// and the fall-through is the cheapest tier.
    #[inline]
    pub fn tier(&self, p: DVec3) -> CrowdTier {
        if self.unbounded {
            return CrowdTier::Full;
        }
        let mut best = f64::INFINITY;
        for a in &self.anchors {
            let (dx, dz) = (p.x - a.x, p.z - a.z);
            let d2 = dx * dx + dz * dz;
            if d2 < best {
                best = d2;
            }
        }
        // `sqrt` and not a squared comparison: IEEE-754 specifies `sqrt`
        // exactly, so the metre-space compare is portable, and the radii are
        // metres everywhere else in this file. NaN falls through every branch.
        let d = best.sqrt();
        if d <= self.full_m {
            CrowdTier::Full
        } else if d <= self.near_m {
            CrowdTier::Near
        } else if d <= self.far_m {
            CrowdTier::Far
        } else {
            CrowdTier::Dormant
        }
    }
}

#[inline]
fn lattice(v: f64) -> i64 {
    let q = (v / BAND_LATTICE_M).floor();
    if q <= -(i64::MAX as f64) {
        i64::MIN + 1
    } else if q >= i64::MAX as f64 {
        i64::MAX
    } else {
        q as i64
    }
}

#[inline]
fn unlattice(c: i64) -> f64 {
    (c as f64 + 0.5) * BAND_LATTICE_M
}

// ── per-agent randomness ────────────────────────────────────────────────────

/// The salt an agent's route speed multiplier is drawn with.
pub const SALT_SPEED: u64 = 0x5350_4545_4400_0001;

/// The salt an agent's route phase offset is drawn with.
pub const SALT_PHASE: u64 = 0x5048_4153_4500_0002;

/// **`mix64(guid ^ tick ^ salt)`** — the house RNG doctrine, as a function.
///
/// There is no engine RNG (no `rand` dependency anywhere in this tree) and there
/// must not be one on a sim path: a stateful generator is state, and state that
/// is not folded into `state_bytes` breaks parity the first time one host starts
/// mid-trace. So every per-agent draw is a **pure function of sim state** — the
/// agent's stable `Guid`, the fixed step it is drawn on, and a compile-time salt
/// naming what it is for.
///
/// The mixer is the SplitMix64 finalizer, the same *specification*
/// `inf_pcg::hash`, `inf_mesh::fracture` and `inf_photo::hash` each spell out;
/// `the_mixer_is_the_splitmix64_finalizer` pins it against the constants rather
/// than against one of those copies.
#[inline]
pub fn agent_rand(guid: Uuid, tick: u64, salt: u64) -> u64 {
    const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
    let bits = guid.as_u128();
    let lo = bits as u64;
    let hi = (bits >> 64) as u64;
    let mut x = lo ^ hi.wrapping_mul(GOLDEN) ^ tick.wrapping_mul(GOLDEN) ^ salt;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// [`agent_rand`] as a uniform in `[0, 1)`.
///
/// The top 53 bits over `2^53` — exact in IEEE-754 double, and therefore the
/// same on every target, which `powf`-based scalings are not.
#[inline]
pub fn agent_unit(guid: Uuid, tick: u64, salt: u64) -> f64 {
    (agent_rand(guid, tick, salt) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

// ── the route ───────────────────────────────────────────────────────────────

/// **Where an agent is at a given sim time** — a pure function of the record and
/// the clock, which is the substrate NPC1c's path-follower replaces.
///
/// A straight there-and-back between two points at a fixed speed. Deliberately
/// the simplest thing that moves: NPC1a's job is the *tier system*, and a route
/// with a graph search in it would make every tier measurement a measurement of
/// the search. `from == to` is a **stand**, which is legal and is what most of a
/// town's population does at any instant.
///
/// Every operation is IEEE-exact (`+ - * / sqrt %`), so two machines derive the
/// same metre — the P14 portable-math law, which binds here because this output
/// reaches a `Transform` and therefore the sim trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrowdRoute {
    /// One end, world metres.
    pub from: DVec3,
    /// The other end, world metres.
    pub to: DVec3,
    /// Metres per second along it. Non-positive is a stand.
    pub speed_mps: f64,
}

impl CrowdRoute {
    /// A route that stands still at `p`.
    pub fn standing(p: DVec3) -> Self {
        Self {
            from: p,
            to: p,
            speed_mps: 0.0,
        }
    }

    /// The agent's position at sim time `t_s`, ping-ponging between the ends.
    ///
    /// `phase_m` shifts the agent along the path so a population drawn from one
    /// route does not march in lockstep; it is the agent's own
    /// [`agent_unit`] draw scaled by the path length, which is why it is a
    /// distance and not an angle.
    pub fn position_at(&self, t_s: f64, phase_m: f64) -> DVec3 {
        let d = self.to - self.from;
        let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        if !(len > 0.0) || !(self.speed_mps > 0.0) || !t_s.is_finite() {
            return self.from;
        }
        let period = 2.0 * len;
        let travelled = self.speed_mps * t_s + phase_m;
        let u = travelled.rem_euclid(period);
        let along = if u <= len { u } else { period - u };
        self.from + d * (along / len)
    }
}

// ── the population ──────────────────────────────────────────────────────────

/// What a crowd NPC is made of, so a [`Dormant`](CrowdTier::Dormant) record can
/// build one back.
///
/// Assets are GUIDs and nothing else: N NPCs on one mannequin share every
/// buffer, every clip, every `.inf_sm` and every `.inf_skel` (the renderer
/// Arc-dedupes by `(mesh, skeleton)`), so a thousand records naming one
/// archetype cost one of each. Body variation without rig variation is NPC1b's
/// packed-channel work and is deliberately not smuggled in here.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrowdArchetype {
    /// `.inf_mesh` with per-vertex skin.
    pub mesh: Option<Uuid>,
    /// `.inf_skel` binding its joint indices.
    pub skeleton: Option<Uuid>,
    /// `.inf_sm` the agent's machine plays.
    pub sm: Option<Uuid>,
    /// Capsule half-height, metres (the segment half-length, `Collider3D`'s
    /// convention).
    pub half_height_m: f64,
    /// Capsule radius, metres.
    pub radius_m: f64,
}

impl CrowdArchetype {
    /// The starter character's proportions — a 1.8 m adult: 0.9 m half-height,
    /// 0.3 m radius, which is the capsule `inf_anim::template` fits a rig to.
    pub fn humanoid(mesh: Option<Uuid>, skeleton: Option<Uuid>, sm: Option<Uuid>) -> Self {
        Self {
            mesh,
            skeleton,
            sm,
            half_height_m: 0.9,
            radius_m: 0.3,
        }
    }
}

/// One member of the population.
///
/// `last` and `pose_digest` are the two fields that make [`Dormant`] and
/// [`Far`] honest rather than lossy: the first is where the agent stood when it
/// stopped having an entity, the second is a fold of the last pose it published
/// before it stopped evaluating one. Both are sim state, both are folded by
/// [`crowd_state_bytes`], and neither is ever written to a file.
///
/// [`Dormant`]: CrowdTier::Dormant
/// [`Far`]: CrowdTier::Far
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrowdRecord {
    /// What it is made of.
    pub archetype: CrowdArchetype,
    /// Where it walks.
    pub route: CrowdRoute,
    /// The tier it took on the last [`step_crowd`].
    pub tier: CrowdTier,
    /// Where it stood then.
    pub last: DVec3,
    /// A fold of the last pose it published, carried while it is not evaluating
    /// one. `0` for an agent that has never posed.
    pub pose_digest: u64,
}

impl CrowdRecord {
    /// A record standing at `p`.
    pub fn standing(archetype: CrowdArchetype, p: DVec3) -> Self {
        Self {
            archetype,
            route: CrowdRoute::standing(p),
            tier: CrowdTier::Dormant,
            last: p,
            pose_digest: 0,
        }
    }

    /// A record walking `route`, starting at its `from` end.
    pub fn walking(archetype: CrowdArchetype, route: CrowdRoute) -> Self {
        Self {
            archetype,
            route,
            tier: CrowdTier::Dormant,
            last: route.from,
            pose_digest: 0,
        }
    }

    /// **Where this agent is at sim time `t_s`** — the record's authored route
    /// plus the two per-agent draws, in the one place that decides them.
    ///
    /// The variation is DERIVED and never stored, which is what keeps it honest:
    /// a jitter written into the record at spawn time would be a second copy of
    /// a pure function, and the copy is the thing that drifts. Both draws are
    /// taken at `tick = 0`, so they are constants of the agent rather than of
    /// the step — a per-step draw uses the same door with the live tick.
    ///
    /// * **speed** — `0.85 … 1.15` of the authored speed, so a population sharing
    ///   one route does not march in lockstep;
    /// * **phase** — up to 8 m of head start along the path, so they do not all
    ///   turn round together either.
    pub fn position_at(&self, guid: Uuid, t_s: f64) -> DVec3 {
        let route = CrowdRoute {
            speed_mps: self.route.speed_mps * (0.85 + 0.3 * agent_unit(guid, 0, SALT_SPEED)),
            ..self.route
        };
        route.position_at(t_s, agent_unit(guid, 0, SALT_PHASE) * 8.0)
    }
}

/// **The population** — every crowd NPC a level has, whether or not it currently
/// has an entity.
///
/// A resource, so no schema moves (see the module docs). Absent until something
/// installs one, so a level with no crowd pays exactly one `get_resource` per
/// fixed step and allocates nothing — the "absent costs nothing" discipline
/// [`crate::deform`] and [`crate::cloth`] already follow.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct CrowdPopulationRes {
    /// The records, in `Guid` order — so every walk over the population is a
    /// function of the level's contents and not of a hash seed.
    pub records: BTreeMap<Uuid, CrowdRecord>,
    /// Fixed steps since the population was installed. The route clock is
    /// `steps · dt` rather than an accumulated `+= dt`, so a long run cannot
    /// drift and two hosts that started at the same step agree exactly.
    pub steps: u64,
}

/// The tier an entity's agent took this step — the component
/// [`crate::pose::step_pose_evaluation`] and the 3D bridge read.
///
/// Written only by [`step_crowd`]. Not reflected and not serialized: it is a
/// *published verdict*, like `AnimStateMachine::runtime`, and an authored one
/// would be a second opinion about a thing that has one door.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrowdAgent {
    /// This step's tier.
    pub tier: CrowdTier,
    /// The agent's own draw seed — `agent_rand(guid, tick, salt)`'s first
    /// argument, cached on the entity so a consumer does not have to look the
    /// `Guid` up again.
    pub guid: Uuid,
}

/// What one [`step_crowd`] did — the instrument's read, and the gate's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrowdStats {
    /// Records at each tier, indexed by [`CrowdTier::as_u8`].
    pub per_tier: [usize; 4],
    /// Entities materialized this step (`Dormant` → anything else).
    pub spawned: u64,
    /// Entities dematerialized this step (anything else → `Dormant`).
    pub despawned: u64,
    /// Records whose tier changed this step — the number the transition arm
    /// asserts is non-zero over a drive.
    pub retiered: u64,
    /// The band's membership stamp (`0` = unbounded).
    pub band_stamp: u64,
}

impl CrowdStats {
    /// Records at `tier`.
    #[inline]
    pub fn at(&self, tier: CrowdTier) -> usize {
        self.per_tier[tier.as_u8() as usize]
    }

    /// Records in total.
    #[inline]
    pub fn total(&self) -> usize {
        self.per_tier.iter().sum()
    }

    /// A one-line summary for the diagnostics log and the instruments.
    pub fn summary(&self) -> String {
        format!(
            "crowd: {} agent(s) — {} full / {} near / {} far / {} dormant, \
             {} spawned / {} despawned / {} re-tiered, band {:#018x}",
            self.total(),
            self.at(CrowdTier::Full),
            self.at(CrowdTier::Near),
            self.at(CrowdTier::Far),
            self.at(CrowdTier::Dormant),
            self.spawned,
            self.despawned,
            self.retiered,
            self.band_stamp,
        )
    }
}

// ── the step ────────────────────────────────────────────────────────────────

/// **Install a population** on `world`, replacing any it already had.
///
/// Records arrive tier-less (`Dormant`, no entity); the first [`step_crowd`]
/// materializes the ones the band wants. That ordering is the point: a spawner
/// that decided tiers itself would be a second copy of the decision.
pub fn set_population(world: &mut EcsWorld, records: BTreeMap<Uuid, CrowdRecord>) {
    world
        .world_mut()
        .insert_resource(CrowdPopulationRes { records, steps: 0 });
}

/// **Forget the crowd**: despawn every materialized agent and remove the
/// resource, so the world is byte-for-byte one that never had a population.
///
/// The editor calls this at both ends of a Simulate session for the reason
/// [`crate::pose::clear_poses`] documents — a `SceneDoc` snapshot carries
/// entities and components and `EcsWorld::clear` despawns entities, and neither
/// touches a resource, so without this a stopped session's crowd would keep
/// standing in the author's document.
pub fn clear_crowd(world: &mut EcsWorld) {
    let materialized: Vec<Uuid> = world
        .world()
        .get_resource::<CrowdPopulationRes>()
        .map(|p| {
            p.records
                .iter()
                .filter(|(_, r)| r.tier.materialized())
                .map(|(g, _)| *g)
                .collect()
        })
        .unwrap_or_default();
    for guid in materialized {
        if let Some(e) = world.entity_of(guid) {
            world.despawn(e);
        }
    }
    world.world_mut().remove_resource::<CrowdPopulationRes>();
}

/// **THE FIXED-STEP CROWD SLOT**: decide every agent's tier, materialize or
/// dematerialize it, and put it where its route says it is.
///
/// ONE function, called from both hosts' fixed steps — the strongest form the
/// MIRROR rule takes, and the same shape [`crate::deform::step_deformation`],
/// [`crate::sky::advance_weather`] and [`crate::pose::step_pose_evaluation`]
/// use. It takes a world and a `dt` and **no camera, no host state and no
/// registries**: everything it reads is in the world.
///
/// The sequence, all of it a pure function of sim state:
///
/// 1. read the band off the world's [`StreamingSource`] entities;
/// 2. walk the records in `Guid` order; for each, the tier of where it *is*
///    (its live transform if it has an entity, its remembered `last` if not);
/// 3. a record leaving [`Full`]/[`Near`] folds its published pose into
///    `pose_digest` — **before** the pose store is rebuilt this step, which is
///    why this phase runs early;
/// 4. `Dormant` → despawn; anything else → spawn if absent;
/// 5. write the route position onto the transform, and the tier onto the
///    [`CrowdAgent`].
///
/// # Where it runs, and why there
///
/// After cell + terrain streaming and the sky, **before** the physics sync, the
/// character step and the animation: the bridge has to see this step's bodies,
/// and [`crate::pose::step_pose_evaluation`] has to see this step's tiers. Its
/// own phase ([`STEP_PHASES`] 24 → 25) rather than a corner of an existing one,
/// because a step that cannot say where its milliseconds went is the defect wave
/// I4b existed to remove.
///
/// # The hero is untouched
///
/// A character carrying no [`CrowdAgent`] — every hero, every authored NPC,
/// every fixture in this tree — is not in the population, is never walked here,
/// and gets exactly the pipeline it got before NPC1a. The tier system is
/// **opt-in by record**, which is what makes "zero cost when absent" a structural
/// claim rather than a benchmark.
///
/// [`Full`]: CrowdTier::Full
/// [`Near`]: CrowdTier::Near
/// [`StreamingSource`]: crate::components::StreamingSource
/// [`STEP_PHASES`]: https://docs.rs/inf-player
pub fn step_crowd(world: &mut EcsWorld, dt: f64) -> CrowdStats {
    step_crowd_banded(world, dt, DEFAULT_CROWD_RADII)
}

/// [`step_crowd`] with explicit radii — the seam the sweep instrument drives to
/// price a tier ladder, and the one a level's own crowd settings will use.
pub fn step_crowd_banded(world: &mut EcsWorld, dt: f64, radii: (f64, f64, f64)) -> CrowdStats {
    // Absent costs nothing: one `contains_resource` on every level with no crowd.
    if !world.world().contains_resource::<CrowdPopulationRes>() {
        return CrowdStats::default();
    }
    let band = CrowdBand::from_world(world, radii);
    let mut stats = CrowdStats {
        band_stamp: band.stamp(),
        ..CrowdStats::default()
    };
    // Lifted out of the world, because the materialization below needs
    // `&mut EcsWorld` and a borrow of the resource would outlive it — the shape
    // `step_pose_evaluation` lifts its goals and blenders with.
    let mut pop = world
        .world_mut()
        .remove_resource::<CrowdPopulationRes>()
        .unwrap_or_default();
    let t_s = pop.steps as f64 * dt;
    // The digests of the poses published on the PREVIOUS step, read once. A
    // demotion folds its own entry out of this map, so an agent that goes Far
    // carries the pose it was last seen in rather than a zero.
    let digests = published_pose_digests(world);

    for (guid, rec) in pop.records.iter_mut() {
        let guid = *guid;
        let entity = world.entity_of(guid);
        let here = match entity.and_then(|e| world.world().get::<Transform>(e)) {
            Some(t) => t.translation.to_dvec3(),
            None => rec.last,
        };
        let tier = band.tier(here);
        let was = rec.tier;
        if tier != was {
            stats.retiered += 1;
        }
        // 3. The cached digest, taken on the way DOWN out of a posing tier.
        if was.poses() && !tier.poses() {
            if let Some(d) = digests.get(&guid) {
                rec.pose_digest = *d;
            }
        }
        rec.tier = tier;
        stats.per_tier[tier.as_u8() as usize] += 1;

        if !tier.materialized() {
            rec.last = here;
            if let Some(e) = entity {
                world.despawn(e);
                stats.despawned += 1;
            }
            continue;
        }
        let entity = match entity {
            Some(e) => e,
            None => {
                stats.spawned += 1;
                materialize(world, guid, rec)
            }
        };
        // 5. Where the route says it is, for every tier — see the module docs
        //    for why the position law does not vary with the tier in NPC1a.
        let p = rec.position_at(guid, t_s);
        rec.last = p;
        let w = world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(entity) {
            t.translation = Vec3d::new(p.x, p.y, p.z);
        }
        if let Some(mut a) = w.get_mut::<CrowdAgent>(entity) {
            a.tier = tier;
        }
    }

    pop.steps += 1;
    world.world_mut().insert_resource(pop);
    stats
}

/// Build the entity a record describes.
///
/// The component set is fixed and small: a skeletal mesh, a machine, a kinematic
/// body with a capsule, a controller and the [`CrowdAgent`] verdict. **No
/// `CharacterMovement`**, deliberately — a crowd agent's position is its route
/// (module docs), and giving it the player's controller as well would be two
/// authorities writing one transform. NPC1c is where the near tiers start
/// steering, and it is where that component arrives.
fn materialize(world: &mut EcsWorld, guid: Uuid, rec: &CrowdRecord) -> Entity {
    let e = world.spawn_with_guid(guid, "Crowd NPC", None);
    let a = rec.archetype;
    world.world_mut().entity_mut(e).insert((
        SkeletalMesh {
            mesh: a.mesh,
            skeleton: a.skeleton,
        },
        AnimStateMachine {
            sm: a.sm,
            ..AnimStateMachine::default()
        },
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            fixed_rotation: true,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(a.radius_m, a.half_height_m, a.radius_m),
            radius: a.radius_m,
            ..Collider3D::default()
        },
        CharacterController3D::default(),
        CrowdAgent {
            tier: rec.tier,
            guid,
        },
    ));
    e
}

/// A fold of every published pose, by entity `Guid` — the source of a record's
/// [`CrowdRecord::pose_digest`].
///
/// FNV-1a over the same bytes [`crate::pose::pose_state_bytes`] emits for that
/// entity, so "the digest of the pose the trace would have carried" is exactly
/// what it says. Computed only for entities the population knows about would be
/// cheaper and is not done, because this runs once per step over a map that is
/// empty on every level with no character at all.
fn published_pose_digests(world: &EcsWorld) -> BTreeMap<Uuid, u64> {
    let Some(store) = world.world().get_resource::<crate::pose::PoseStoreRes>() else {
        return BTreeMap::new();
    };
    store
        .0
        .iter()
        .map(|(g, ep)| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut fold = |bytes: &[u8]| {
                for b in bytes {
                    h ^= u64::from(*b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            };
            fold(ep.skeleton.as_bytes());
            for l in &ep.pose.locals {
                for v in l
                    .translation
                    .iter()
                    .chain(l.rotation.iter())
                    .chain(l.scale.iter())
                {
                    fold(&v.to_le_bytes());
                }
            }
            (*g, h)
        })
        .collect()
}

// ── what the rest of the engine reads ───────────────────────────────────────

/// The tier `entity` took this step, or `None` for anything that is not a crowd
/// agent — which is every hero and every authored character.
#[inline]
pub fn agent_tier(world: &EcsWorld, entity: Entity) -> Option<CrowdTier> {
    world.world().get::<CrowdAgent>(entity).map(|a| a.tier)
}

/// **Every agent the 3D bridge must give no body to this step** — the
/// `Guid`s whose tier is not [`CrowdTier::has_body`].
///
/// Returned as a set rather than read per entity inside the bridge's walk
/// because that walk is `O(entities)` over a furnished town and this is
/// `O(agents)`; an empty set costs one branch per body, which is what a level
/// with no crowd pays. Empty — and allocation-free — when there is no
/// population at all.
pub fn bodiless_agents(world: &EcsWorld) -> BTreeSet<Uuid> {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return BTreeSet::new();
    };
    pop.records
        .iter()
        .filter(|(_, r)| r.tier.materialized() && !r.tier.has_body())
        .map(|(g, _)| *g)
        .collect()
}

/// This step's crowd counters, or the zero stats on a world with no population.
pub fn crowd_stats(world: &EcsWorld) -> CrowdStats {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return CrowdStats::default();
    };
    let mut s = CrowdStats::default();
    for r in pop.records.values() {
        s.per_tier[r.tier.as_u8() as usize] += 1;
    }
    s
}

/// **The crowd's canonical bytes** — the shape a replay / PIE trace folds,
/// exactly like [`crate::deform::deform_state_bytes`] and
/// [`crate::pose::pose_state_bytes`].
///
/// # The trace re-shape, stated as arithmetic
///
/// A posed character contributes `36 + 40 · joints` bytes to
/// [`crate::pose::pose_state_bytes`] — **6 476 B** at the starter character's
/// 161 bones. A [`Far`](CrowdTier::Far) agent evaluates no pose, so it
/// contributes **nothing** there; a [`Dormant`](CrowdTier::Dormant) one has no
/// entity, so it contributes nothing to the sim snapshot either. What would then
/// be invisible is the thing that decided it, so this section carries **49 bytes
/// an agent at every tier**: the `Guid` (16), the tier (1), where it stands (24)
/// and the cached digest of the pose it last published (8).
///
/// That is the whole re-shape: at N = 100 agents all Far the crowd costs
/// `100 × 49 = 4 900` B a step against `100 × 6 476 = 647 600` B — **132×** —
/// and the trace still distinguishes two agents frozen mid-stride in different
/// poses, which a section that emitted only a tier could not.
///
/// The position is folded even for materialized agents, where the sim snapshot
/// already carries it. That is 24 duplicated bytes an agent, spent on purpose: a
/// **Dormant** agent has no snapshot entry at all, and a section whose meaning
/// changed with the tier would be a section a reader has to case-split.
///
/// Appended to the sim's `state_bytes`, which is **hashed and never decoded**,
/// so this needs no version and no reader. A level with no population produces
/// an empty vec and every pre-NPC1a trace is byte-identical.
pub fn crowd_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(pop) = world.world().get_resource::<CrowdPopulationRes>() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(pop.records.len() * AGENT_TRACE_BYTES);
    // BTreeMap: `Guid` order, so the bytes are a property of the level and not
    // of bevy's archetype layout.
    for (guid, rec) in &pop.records {
        out.extend_from_slice(guid.as_bytes());
        out.push(rec.tier.as_u8());
        out.extend_from_slice(&rec.last.x.to_le_bytes());
        out.extend_from_slice(&rec.last.y.to_le_bytes());
        out.extend_from_slice(&rec.last.z.to_le_bytes());
        out.extend_from_slice(&rec.pose_digest.to_le_bytes());
    }
    out
}

/// Bytes one agent contributes to [`crowd_state_bytes`]: 16 `Guid` + 1 tier +
/// 24 position + 8 digest.
///
/// Pinned as a constant because the gates quote it: the re-shape's whole claim
/// is a ratio between this number and a posed character's 6 476, and a ratio
/// with a drifting denominator is not a claim.
pub const AGENT_TRACE_BYTES: usize = 16 + 1 + 24 + 8;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::StreamingSource;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn band_at(x: f64) -> CrowdBand {
        CrowdBand::from_anchors([DVec3::new(x, 0.0, 0.0)], DEFAULT_CROWD_RADII)
    }

    /// **It fails toward FULL.** No sources means no banding, which is what keeps
    /// every pre-NPC1a fixture and every committed level behaving exactly as
    /// before.
    #[test]
    fn a_world_with_no_streaming_source_tiers_everything_full() {
        let b = CrowdBand::from_anchors(Vec::<DVec3>::new(), DEFAULT_CROWD_RADII);
        assert!(b.is_unbounded());
        assert_eq!(b.stamp(), 0);
        assert_eq!(b.tier(DVec3::new(1e9, 0.0, -1e9)), CrowdTier::Full);
        // …and a NaN point in an unbounded band is Full too: refusing it would
        // silently change a fixture that has one.
        assert_eq!(b.tier(DVec3::new(f64::NAN, 0.0, 0.0)), CrowdTier::Full);

        // A source at a non-finite position leaves no anchors, so the band fails
        // open the same way.
        let nan = CrowdBand::from_anchors([DVec3::new(f64::NAN, 0.0, 0.0)], DEFAULT_CROWD_RADII);
        assert!(nan.is_unbounded());
        // A non-finite radius, and radii out of order, are both refused.
        assert!(CrowdBand::from_anchors([DVec3::ZERO], (32.0, f64::NAN, 512.0)).is_unbounded());
        assert!(CrowdBand::from_anchors([DVec3::ZERO], (96.0, 32.0, 512.0)).is_unbounded());
    }

    /// The four tiers over a real anchor, at their own boundaries.
    #[test]
    fn the_band_tiers_a_point_by_its_nearest_anchor() {
        let b = band_at(0.0);
        assert!(!b.is_unbounded());
        // Snapped to the lattice cell centre, exactly as `SimBand` does.
        assert_eq!(b.anchors(), [DVec3::new(8.0, 0.0, 8.0)]);
        let at = |x: f64| b.tier(DVec3::new(x, 0.0, 8.0));
        assert_eq!(at(8.0), CrowdTier::Full);
        assert_eq!(at(39.0), CrowdTier::Full); //  31 m
        assert_eq!(at(41.0), CrowdTier::Near); //  33 m
        assert_eq!(at(103.0), CrowdTier::Near); //  95 m
        assert_eq!(at(105.0), CrowdTier::Far); //  97 m
        assert_eq!(at(519.0), CrowdTier::Far); // 511 m
        assert_eq!(at(521.0), CrowdTier::Dormant); // 513 m

        // A NaN point in a BANDED world falls through to the cheapest tier.
        assert_eq!(b.tier(DVec3::new(f64::NAN, 0.0, 0.0)), CrowdTier::Dormant);

        // The NEAREST anchor wins: a second source next to the far point
        // promotes it all the way back to Full.
        let two = CrowdBand::from_anchors(
            [DVec3::ZERO, DVec3::new(520.0, 0.0, 8.0)],
            DEFAULT_CROWD_RADII,
        );
        assert_eq!(two.tier(DVec3::new(521.0, 0.0, 8.0)), CrowdTier::Full);
        assert_ne!(two.stamp(), b.stamp(), "a new anchor is a new membership");
    }

    /// **The tier is a function of the SET of source positions**, not of the
    /// order a world walk produced them in, nor of duplicates, nor of height.
    #[test]
    fn the_band_is_a_function_of_the_source_set() {
        let a = CrowdBand::from_anchors(
            [DVec3::new(0.0, 0.0, 0.0), DVec3::new(300.0, 0.0, 40.0)],
            DEFAULT_CROWD_RADII,
        );
        let b = CrowdBand::from_anchors(
            [
                DVec3::new(300.0, 0.0, 40.0),
                DVec3::new(1.0, 5.0, 1.0),
                DVec3::new(0.0, 0.0, 0.0),
            ],
            DEFAULT_CROWD_RADII,
        );
        assert_eq!(a, b, "order, height and duplicates must not move the band");
    }

    /// **Hysteresis is refused, and this is what that costs** — the
    /// `SimBand::a_source_parked_on_a_lattice_line_rebands_every_step` arm, one
    /// system over.
    ///
    /// An agent parked on a tier boundary re-tiers on every step, and the bound
    /// this holds is that the thrash alternates between exactly **two** tiers —
    /// the two it is between — rather than wandering. A stateful tier would fix
    /// it and would stop being a pure function of sim state, which is the whole
    /// reason PIE equals shipping here.
    #[test]
    fn an_agent_parked_on_a_tier_boundary_alternates_between_two() {
        let b = band_at(0.0);
        // The Full/Near boundary is 32 m from the snapped anchor at (8, 8).
        let mut tiers = Vec::new();
        for step in 0..60 {
            let x = 8.0 + DEFAULT_CROWD_FULL_M + if step % 2 == 0 { -0.001 } else { 0.001 };
            tiers.push(b.tier(DVec3::new(x, 0.0, 8.0)));
        }
        let distinct: BTreeSet<CrowdTier> = tiers.iter().copied().collect();
        let changes = tiers.windows(2).filter(|w| w[0] != w[1]).count();
        println!(
            "NPC1a tier edge: an agent jittering +/-1 mm across the Full/Near \
             boundary re-tiers {changes} times in {} steps, over {} distinct tiers",
            tiers.len(),
            distinct.len()
        );
        assert_eq!(
            distinct.len(),
            2,
            "a parked agent produced {} tiers — the thrash is a wander, not the \
             two it is between",
            distinct.len()
        );
        assert_eq!(
            changes,
            tiers.len() - 1,
            "the arm is not measuring the edge"
        );

        // THE CONTROL: the same jitter a metre inside the boundary never moves.
        let inside: Vec<CrowdTier> = (0..60)
            .map(|step| {
                let x =
                    8.0 + DEFAULT_CROWD_FULL_M - 1.0 + if step % 2 == 0 { -0.001 } else { 0.001 };
                b.tier(DVec3::new(x, 0.0, 8.0))
            })
            .collect();
        assert!(
            inside.windows(2).all(|w| w[0] == w[1]),
            "the same jitter away from the boundary re-tiered — the band is \
             buying nothing"
        );
    }

    /// The cost ladder is monotone: every tier is cheaper than the one above it,
    /// in every dimension, and nothing costs more as it gets further away.
    #[test]
    fn the_tier_ladder_is_monotone() {
        let ladder = [
            CrowdTier::Full,
            CrowdTier::Near,
            CrowdTier::Far,
            CrowdTier::Dormant,
        ];
        for w in ladder.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(a < b, "{a:?} must order before {b:?}");
            assert!(a.hand_ik() >= b.hand_ik(), "{a:?} vs {b:?}: hand IK grew");
            assert!(a.poses() >= b.poses(), "{a:?} vs {b:?}: pose grew");
            assert!(a.has_body() >= b.has_body(), "{a:?} vs {b:?}: body grew");
            assert!(
                a.materialized() >= b.materialized(),
                "{a:?} vs {b:?}: an entity appeared"
            );
        }
        // …and each rung really is a rung: the four tiers are four distinct
        // cost vectors, so no two of them are the same tier under two names.
        let vectors: BTreeSet<(bool, bool, bool, bool)> = ladder
            .iter()
            .map(|t| (t.hand_ik(), t.poses(), t.has_body(), t.materialized()))
            .collect();
        assert_eq!(vectors.len(), 4, "two tiers cost the same thing");
    }

    /// The trace byte count is what the doc claims, and the re-shape's ratio
    /// with it.
    #[test]
    fn the_agent_trace_section_is_forty_nine_bytes() {
        let mut records = BTreeMap::new();
        for i in 0..7u128 {
            records.insert(
                guid(0x900 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
            );
        }
        let mut world = EcsWorld::new();
        set_population(&mut world, records);
        let bytes = crowd_state_bytes(&world);
        assert_eq!(bytes.len(), 7 * AGENT_TRACE_BYTES);
        assert_eq!(AGENT_TRACE_BYTES, 49);
        // The claim the ledger quotes: a 161-bone posed character is 6 476 B.
        const POSED: usize = 36 + 161 * 40;
        assert_eq!(POSED / AGENT_TRACE_BYTES, 132);
        println!(
            "NPC1a trace: {AGENT_TRACE_BYTES} B an agent against {POSED} B a posed \
             character — {}x",
            POSED / AGENT_TRACE_BYTES
        );

        // A world with no population folds nothing at all, so every pre-NPC1a
        // trace is byte-identical.
        assert!(crowd_state_bytes(&EcsWorld::new()).is_empty());
    }

    /// The mixer is the SplitMix64 finalizer, pinned against the spec rather
    /// than against one of the tree's four copies of it.
    #[test]
    fn the_mixer_is_the_splitmix64_finalizer() {
        fn reference(mut x: u64) -> u64 {
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        }
        const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
        let g = guid(0x1234_5678_9abc_def0);
        for tick in [0u64, 1, 7, u64::MAX] {
            let bits = g.as_u128();
            let want = reference(
                (bits as u64)
                    ^ ((bits >> 64) as u64).wrapping_mul(GOLDEN)
                    ^ tick.wrapping_mul(GOLDEN)
                    ^ SALT_SPEED,
            );
            assert_eq!(agent_rand(g, tick, SALT_SPEED), want);
        }
        // The three arguments each separate: two agents, two ticks and two
        // salts all give different streams.
        assert_ne!(
            agent_rand(g, 0, SALT_SPEED),
            agent_rand(guid(2), 0, SALT_SPEED)
        );
        assert_ne!(agent_rand(g, 0, SALT_SPEED), agent_rand(g, 1, SALT_SPEED));
        assert_ne!(agent_rand(g, 0, SALT_SPEED), agent_rand(g, 0, SALT_PHASE));
        // …and the uniform really is in [0, 1).
        for i in 0..1000u64 {
            let u = agent_unit(guid(i as u128), i, SALT_PHASE);
            assert!((0.0..1.0).contains(&u), "draw {i} was {u}");
        }
    }

    /// A route is a pure function of the clock, ping-pongs, and stands still
    /// when it has nowhere to go.
    #[test]
    fn a_route_is_a_pure_function_of_route_and_clock() {
        let r = CrowdRoute {
            from: DVec3::new(0.0, 0.0, 0.0),
            to: DVec3::new(10.0, 0.0, 0.0),
            speed_mps: 1.0,
        };
        assert_eq!(r.position_at(0.0, 0.0).x, 0.0);
        assert_eq!(r.position_at(5.0, 0.0).x, 5.0);
        assert_eq!(r.position_at(10.0, 0.0).x, 10.0);
        assert_eq!(r.position_at(15.0, 0.0).x, 5.0, "it did not turn round");
        assert_eq!(r.position_at(20.0, 0.0).x, 0.0);
        assert_eq!(r.position_at(21.0, 0.0).x, 1.0, "the period is wrong");
        // Same input, same output — twice, because "pure function" is the claim.
        assert_eq!(r.position_at(7.5, 1.25), r.position_at(7.5, 1.25));
        // A stand stands, whatever the clock says.
        let s = CrowdRoute::standing(DVec3::new(3.0, 4.0, 5.0));
        assert_eq!(s.position_at(1e6, 3.0), DVec3::new(3.0, 4.0, 5.0));
        // …and so does a route with no speed, or a non-finite clock.
        assert_eq!(
            CrowdRoute {
                speed_mps: 0.0,
                ..r
            }
            .position_at(9.0, 0.0),
            r.from
        );
        assert_eq!(r.position_at(f64::NAN, 0.0), r.from);
    }

    /// A world holding a source and a population, with the source at `x`.
    fn crowd_world(records: BTreeMap<Uuid, CrowdRecord>) -> EcsWorld {
        let mut world = EcsWorld::new();
        let src = world.spawn_with_guid(guid(0xF1), "Player", None);
        world
            .world_mut()
            .entity_mut(src)
            .insert(StreamingSource { radius_m: 0.0 });
        world.propagate();
        set_population(&mut world, records);
        world
    }

    fn move_source(world: &mut EcsWorld, x: f64) {
        let e = world.entity_of(guid(0xF1)).expect("the source");
        crate::sim::set_translation(world, e, Vec3d::new(x, 0.0, 0.0));
        world.propagate();
    }

    /// **The step materializes what the band wants and dematerializes what it
    /// does not** — and the walk back proves it is a function of *where the
    /// source is* rather than of what happened first.
    #[test]
    fn the_step_materializes_by_tier_and_dematerializes_by_tier() {
        let mut records = BTreeMap::new();
        // Four agents on a line: 10 m, 50 m, 200 m and 2 km from the origin.
        for (i, x) in [10.0f64, 50.0, 200.0, 2000.0].iter().enumerate() {
            records.insert(
                guid(0xA00 + i as u128),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(*x, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(records);

        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            (
                s.at(CrowdTier::Full),
                s.at(CrowdTier::Near),
                s.at(CrowdTier::Far),
                s.at(CrowdTier::Dormant)
            ),
            (1, 1, 1, 1),
            "the fixture is not posing the problem: {}",
            s.summary()
        );
        assert_eq!(s.spawned, 3, "three tiers materialize, one does not");
        assert_eq!(s.despawned, 0);
        assert!(world.entity_of(guid(0xA00)).is_some());
        assert!(
            world.entity_of(guid(0xA03)).is_none(),
            "a dormant agent has an entity"
        );

        // Walk the source out to the far agent: the near ones dematerialize and
        // the far one comes to life.
        move_source(&mut world, 2000.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Full), 1, "{}", s.summary());
        assert_eq!(s.at(CrowdTier::Dormant), 3, "{}", s.summary());
        assert_eq!(s.spawned, 1);
        assert_eq!(s.despawned, 3);
        assert!(world.entity_of(guid(0xA03)).is_some());
        assert!(world.entity_of(guid(0xA00)).is_none());

        // …and back. A record that dematerialized comes back where it stood,
        // which is what `last` is for.
        move_source(&mut world, 0.0);
        step_crowd(&mut world, 1.0 / 60.0);
        let e = world.entity_of(guid(0xA00)).expect("it came back");
        let t = world.world().get::<Transform>(e).expect("with a transform");
        assert_eq!(t.translation.to_dvec3(), DVec3::new(10.0, 0.0, 0.0));
    }

    /// **A Far agent gets no body and no pose, and the trace says which.**
    ///
    /// The anti-vacuity half matters more than the assertion: an arm that only
    /// checked "the Far agent has no rapier body" would pass on a world where
    /// nothing has one.
    #[test]
    fn the_far_tier_drops_the_body_and_the_pose_and_the_trace_records_it() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xB00),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(10.0, 0.0, 0.0)),
        );
        records.insert(
            guid(0xB01),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(200.0, 0.0, 0.0)),
        );
        let mut world = crowd_world(records);
        step_crowd(&mut world, 1.0 / 60.0);

        let bodiless = bodiless_agents(&world);
        assert_eq!(
            bodiless.len(),
            1,
            "exactly the Far agent loses its body — {bodiless:?}"
        );
        assert!(bodiless.contains(&guid(0xB01)));
        assert!(
            !bodiless.contains(&guid(0xB00)),
            "the near agent lost its body too, so the tier is doing nothing"
        );

        // The published verdict is on the entity, which is what the pose door
        // and the physics bridge read.
        let near = world.entity_of(guid(0xB00)).expect("near");
        let far = world.entity_of(guid(0xB01)).expect("far");
        assert_eq!(agent_tier(&world, near), Some(CrowdTier::Full));
        assert_eq!(agent_tier(&world, far), Some(CrowdTier::Far));
        assert!(
            agent_tier(&world, world.entity_of(guid(0xF1)).unwrap()).is_none(),
            "the streaming source is not a crowd agent and must have no tier"
        );

        // The trace carries the tier byte at the agent's own offset.
        let bytes = crowd_state_bytes(&world);
        assert_eq!(bytes.len(), 2 * AGENT_TRACE_BYTES);
        assert_eq!(bytes[16], CrowdTier::Full.as_u8());
        assert_eq!(bytes[AGENT_TRACE_BYTES + 16], CrowdTier::Far.as_u8());
    }

    /// **The whole step is a pure function of sim state**: two worlds built the
    /// same way and stepped the same way produce the same bytes, and a world
    /// whose source moved produces different ones.
    #[test]
    fn two_identical_worlds_produce_identical_crowd_traces() {
        let build = || {
            let mut records = BTreeMap::new();
            for i in 0..12u128 {
                records.insert(
                    guid(0xC00 + i),
                    CrowdRecord::walking(
                        CrowdArchetype::default(),
                        CrowdRoute {
                            from: DVec3::new(i as f64 * 20.0, 0.0, 0.0),
                            to: DVec3::new(i as f64 * 20.0 + 40.0, 0.0, 0.0),
                            speed_mps: 1.4,
                        },
                    ),
                );
            }
            crowd_world(records)
        };
        let (mut a, mut b) = (build(), build());
        let mut trace_a = Vec::new();
        let mut trace_b = Vec::new();
        for step in 0..90u64 {
            move_source(&mut a, step as f64 * 2.0);
            move_source(&mut b, step as f64 * 2.0);
            step_crowd(&mut a, 1.0 / 60.0);
            step_crowd(&mut b, 1.0 / 60.0);
            trace_a.push(crowd_state_bytes(&a));
            trace_b.push(crowd_state_bytes(&b));
        }
        assert_eq!(trace_a, trace_b, "two identical runs diverged");
        // …and the trace is not a constant, or the comparison is between two
        // recordings of nothing happening.
        let distinct: BTreeSet<&Vec<u8>> = trace_a.iter().collect();
        assert!(
            distinct.len() > 45,
            "only {} of 90 crowd states differ — the agents are not moving",
            distinct.len()
        );
        println!(
            "NPC1a determinism: 90 steps, {} distinct crowd states, {} B a state",
            distinct.len(),
            trace_a[0].len()
        );
    }

    /// **Clearing the crowd leaves a world that never had one.**
    #[test]
    fn clearing_the_crowd_despawns_every_agent_and_removes_the_resource() {
        let mut records = BTreeMap::new();
        for i in 0..5u128 {
            records.insert(
                guid(0xD00 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(i as f64, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(records);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(crowd_stats(&world).at(CrowdTier::Full), 5);

        clear_crowd(&mut world);
        for i in 0..5u128 {
            assert!(
                world.entity_of(guid(0xD00 + i)).is_none(),
                "agent {i} survived"
            );
        }
        assert!(crowd_state_bytes(&world).is_empty());
        assert_eq!(crowd_stats(&world), CrowdStats::default());
        // Idempotent, and a no-op on a world that never had a population.
        clear_crowd(&mut world);
        clear_crowd(&mut EcsWorld::new());
    }

    /// **A world with no population pays nothing** — the anti-vacuity control
    /// for every arm above, and the structural half of "zero cost when absent".
    #[test]
    fn a_world_with_no_population_steps_to_the_zero_stats() {
        let mut world = EcsWorld::new();
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s, CrowdStats::default());
        assert!(bodiless_agents(&world).is_empty());
        assert!(crowd_state_bytes(&world).is_empty());
        assert!(
            !world.world().contains_resource::<CrowdPopulationRes>(),
            "a step over a world with no crowd installed one"
        );
    }
}
