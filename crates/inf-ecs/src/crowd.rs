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
//! | [`Dormant`](CrowdTier::Dormant) | — | — | — | route (no entity to write it to) |
//!
//! **The position law is the same at every tier**, and that is deliberate rather
//! than unfinished: an agent's place is `route(clock)`, a pure function of its
//! record and the step count, at Full exactly as at Far — and at
//! [`Dormant`](CrowdTier::Dormant) too, where there is simply no entity to write
//! it onto. (It was NOT so in the wave's first cut, and the audit arm
//! `a_dormant_agent_keeps_walking_its_route_and_can_come_back` is why: a record
//! that froze where it dematerialized was then *tiered* from that frozen point,
//! so a walking agent whose route carried it home could never come back.) NPC1c
//! replaces that
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
//! **The lattice's slop is inherited too, and it is worth stating in metres.**
//! `SimBand`'s own module measures it: snapping an anchor to a 16 m cell centre
//! moves it by at most `BAND_LATTICE_M · √2 / 2` ≈ **11.3 m**, so every radius
//! below is really "that radius ± 11.3 m" — a third of the 32 m `Full` ring and
//! a fiftieth of the 512 m one. The radii are chosen with that in mind (`Full`
//! at half of `DEFAULT_COLLIDER_NEAR_M` still sits inside the solid world at
//! its worst case), and the arm that bounds it is `SimBand`'s
//! `the_lattice_slop_is_bounded_by_half_a_cell_diagonal` — one lattice, one
//! bound, not two copies of it.
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
//!
//! [`StreamingSource`]: crate::components::StreamingSource

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
    /// **Data only.** The entity is despawned; [`CrowdRecord`] remembers what it
    /// was doing and what it looked like, keeps walking its route as a pure
    /// function of the clock (there is just nothing to write the transform
    /// onto), and re-materializes the step its tier comes back.
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

    /// **Whether this tier casts a shadow of its own shape** (`Full` only), or
    /// through the crowd's shared proxy (wave NPC1b).
    ///
    /// A skinned caster is one *geometry group* in the virtual shadow map, and
    /// `inf_render::VSM_MAX_GROUPS` is 1 024 — so a thousand NPCs each casting
    /// their own silhouette is a thousand groups, past the ceiling, and the
    /// overflow is refused. `Full` is 32 m, which is where a viewer can read that
    /// an arm moved; past it a box the agent's own size is the same handful of
    /// page texels and costs ONE group for the whole crowd.
    ///
    /// The predicate lives here, beside [`poses`](Self::poses) and
    /// [`has_body`](Self::has_body), so the tier means one thing in the editor
    /// and the player — both projectors read it through the same door.
    #[inline]
    pub fn skinned_caster(self) -> bool {
        matches!(self, CrowdTier::Full)
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

// ── what an agent looks like (wave NPC1b) ───────────────────────────────────

/// The salt an agent's palette-swap look is drawn with.
pub const SALT_LOOK: u64 = 0x4c4f_4f4b_0000_0003;

/// The salt an agent's build (its drawn height and girth) is drawn with.
pub const SALT_BUILD: u64 = 0x4255_494c_4400_0004;

/// **The crowd's palette swaps** — linear-space multipliers over whatever base
/// colour the archetype's material resolves to.
///
/// Eight, because a crowd wants to stop reading as clones and does not want to
/// read as a paint chart: at eight looks a group of six is very unlikely to be
/// uniform and no look is rare enough to feel like a special character. They are
/// *multipliers*, not colours, so a character with an authored material keeps its
/// own material and takes the variation on top — which is what makes this work
/// for content that does not exist yet.
///
/// **Body variation without rig variation**, which is the shape
/// [`CrowdArchetype`]'s own doc names: N NPCs share one mesh, one skeleton, one
/// `.inf_sm` and every clip, and what makes them different is per-instance data
/// the renderer already carries. NPC1a deliberately did not smuggle it in here;
/// this is where it lands.
pub const CROWD_LOOKS: [[f32; 3]; 8] = [
    [1.00, 0.98, 0.94], // bone
    [0.52, 0.60, 0.82], // denim
    [0.58, 0.62, 0.42], // olive
    [0.86, 0.52, 0.36], // rust
    [0.38, 0.40, 0.44], // charcoal
    [0.92, 0.82, 0.62], // sand
    [0.40, 0.70, 0.70], // teal
    [0.66, 0.34, 0.40], // maroon
];

/// The narrowest and widest an agent is drawn, as a multiplier on its archetype's
/// proportions.
///
/// **±8 %, and the bound is a physics bound rather than a taste one.** A crowd
/// agent's *collider* is its archetype's capsule and this multiplier does not
/// reach it (see [`agent_look`]), so every centimetre of it is a centimetre of
/// disagreement between what a player sees and what they can walk into. Eight
/// per cent of a 1.8 m adult is 14 cm of height and 2.4 cm of radius — inside the
/// slack a capsule already has around a humanoid mesh, and visible enough that a
/// row of six does not read as one model repeated.
pub const CROWD_BUILD_RANGE: (f32, f32) = (0.92, 1.08);

/// **What one crowd agent looks like** — derived from its `Guid` and nothing
/// else, and therefore never stored.
///
/// A stored look would be a second copy of a pure function, and the copy is what
/// drifts (NPC1a's ruling about the speed multiplier and the route phase, met
/// again). It is also why this is not a component: nothing here is sim state, so
/// nothing here reaches `state_bytes`, and **PIE equals shipping for the same
/// reason it did before the crowd had a face** — both projectors call this one
/// door with the same `Guid` and get the same answer.
///
/// Drawn at `tick = 0`, so an agent's look does not change as it walks.
#[inline]
pub fn agent_look(guid: Uuid) -> CrowdLook {
    let swap = (agent_rand(guid, 0, SALT_LOOK) % CROWD_LOOKS.len() as u64) as usize;
    let u = agent_unit(guid, 0, SALT_BUILD) as f32;
    let (lo, hi) = CROWD_BUILD_RANGE;
    CrowdLook {
        tint: CROWD_LOOKS[swap],
        build: lo + (hi - lo) * u,
    }
}

/// One agent's drawn variation — see [`agent_look`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrowdLook {
    /// A linear-space multiplier over the base colour, from [`CROWD_LOOKS`].
    pub tint: [f32; 3],
    /// A uniform multiplier on the agent's **drawn** scale, inside
    /// [`CROWD_BUILD_RANGE`].
    ///
    /// Drawn and not simulated: it multiplies the render instance's scale and
    /// leaves the rapier capsule at the archetype's own proportions. That is the
    /// honest v1 — a per-agent collider is a per-agent `Collider3D`, which is sim
    /// state, which is a schema question and a trace question and belongs with the
    /// wave that gives a near agent a controller.
    pub build: f32,
}

impl CrowdLook {
    /// This look applied to a base colour, alpha untouched.
    #[inline]
    pub fn over(self, base: [f32; 4]) -> [f32; 4] {
        [
            base[0] * self.tint[0],
            base[1] * self.tint[1],
            base[2] * self.tint[2],
            base[3],
        ]
    }
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
        // Spelled as one conjunction rather than as three negated comparisons:
        // `!(len > 0.0)` reads NaN correctly and reads as a clippy lint, and a
        // route with a NaN end has to answer `from` rather than wander.
        let walkable = len.is_finite()
            && len > 0.0
            && self.speed_mps.is_finite()
            && self.speed_mps > 0.0
            && t_s.is_finite();
        if !walkable {
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
    /// **Poses folded into a cached digest this step** (NPC1a audit) — one per
    /// agent that left a posing tier and had an entry in the store.
    ///
    /// A counter and not a clock, because the property it exists to hold is a
    /// *shape*: the crowd phase's work must be a function of the CROWD. The
    /// first cut folded every entry in the pose store on every step to serve
    /// this, so the phase cost grew with the level's hero count and the wave's
    /// budget was minted against a number that was mostly other systems' poses
    /// (0.282 ms banded against 0.759 all-`Full`, at one thousand agents doing
    /// identical work). This reads 0 on a settled step whatever the level's
    /// character count, which is the machine-independent way to say so.
    pub digests_folded: u64,
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
             {} spawned / {} despawned / {} re-tiered, {} pose digest(s) folded, \
             band {:#018x}",
            self.total(),
            self.at(CrowdTier::Full),
            self.at(CrowdTier::Near),
            self.at(CrowdTier::Far),
            self.at(CrowdTier::Dormant),
            self.spawned,
            self.despawned,
            self.retiered,
            self.digests_folded,
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
///
/// # The `Guid`s are the CALLER's to keep distinct
///
/// A record's key becomes its entity's `Guid` when it materializes, and
/// `EcsWorld::spawn_with_guid` does not refuse a key the world already uses — it
/// would overwrite the index entry, and the level's own entity would become
/// unreachable by `Guid` while still existing. Every caller in this tree draws
/// from a fixed namespace of its own for exactly that reason. A checked door
/// belongs on `spawn_with_guid` rather than here (it is the one place that could
/// answer for *every* spawner), and it is not this wave's to build; NPC1d, which
/// derives a population from a level's own buildings, is the wave that needs it.
///
/// # "Replacing" includes the BODIES (NPC1a audit)
///
/// A population is not only its records: a materialized agent is a real entity
/// carrying a skeletal mesh, a machine, a capsule and a [`CrowdAgent`]. Dropping
/// the resource alone left those standing with a tier frozen at whatever the
/// last step decided and **no record behind them** — so [`bodiless_agents`],
/// which reads records, and the [`CrowdAgent`] component, which the pose door
/// and [`crate::deform::ground_contacts`] read, would answer differently about
/// one entity. That is the same two-opinions shape as this wave's own deform
/// finding, so the old crowd goes through [`clear_crowd`] first.
pub fn set_population(world: &mut EcsWorld, records: BTreeMap<Uuid, CrowdRecord>) {
    clear_crowd(world);
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
/// own phase (`inf_player::step_profile::STEP_PHASES` 24 to 25) rather than a
/// corner of an existing one, because a step that cannot say where its
/// milliseconds went is the defect wave I4b existed to remove.
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
pub fn step_crowd(world: &mut EcsWorld, dt: f64) -> CrowdStats {
    step_crowd_banded(world, dt, DEFAULT_CROWD_RADII)
}

/// **What one agent's tier and place would be** — the step's PURE half, split
/// out so it can be measured and, if it ever pays, parallelized.
///
/// # Why this is a separate function
///
/// The step below is three things in a trench coat: a world read (where is this
/// agent), a decision (`band.tier` + `route(clock)`), and a world write
/// (spawn/despawn/transform). Only the middle one is a pure function, and only a
/// pure function can go through `inf_core::job`'s deterministic in-order map —
/// the ECS mutation cannot, it needs `&mut World`. (Named in prose rather than
/// linked: `inf-ecs` does not depend on `inf-core`, and this wave's measurement
/// says it should not start.)
///
/// So the decision lives here, the step calls it, and the sweep instrument times
/// **this exact function** serially and in parallel at N ∈ {1, 10, 100, 1 000}.
/// A benchmark of a private copy would be a benchmark of something the engine
/// does not run, which is this repository's own "a gate must aim at the thing it
/// names".
///
/// `here` is where the agent is *now* — its live transform if it has an entity,
/// its remembered [`CrowdRecord::last`] if it does not.
#[inline]
pub fn plan_agent(
    band: &CrowdBand,
    guid: Uuid,
    rec: &CrowdRecord,
    here: DVec3,
    t_s: f64,
) -> AgentPlan {
    AgentPlan {
        tier: band.tier(here),
        // **The position law does not vary with the tier, `Dormant` included**
        // (NPC1a audit). It used to: a dematerialized record froze at `last`,
        // the tier was then decided from that frozen point, and a walking agent
        // whose route carried it home could never be re-admitted — it was judged
        // for ever at the metre where it went out of range, while the next
        // materialization would have placed it at `route(now)` somewhere else
        // entirely. Frozen for the decision and live for the placement is two
        // authorities over one thing, which is what this module exists to avoid.
        //
        // A route is a pure function of the clock and costs a handful of flops,
        // so keeping it live while an agent has no entity costs nothing and is
        // the only reading under which `Dormant` is a *cost* tier rather than a
        // one-way door. The day NPC1d gives an off-screen agent a schedule is
        // the day this line reads the schedule instead — for every tier at once.
        at: rec.position_at(guid, t_s),
    }
}

/// One agent's decided tier and place — [`plan_agent`]'s answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentPlan {
    /// What it costs this step.
    pub tier: CrowdTier,
    /// Where it is this step.
    pub at: DVec3,
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

    for (guid, rec) in pop.records.iter_mut() {
        let guid = *guid;
        let entity = world.entity_of(guid);
        let here = match entity.and_then(|e| world.world().get::<Transform>(e)) {
            Some(t) => t.translation.to_dvec3(),
            None => rec.last,
        };
        let plan = plan_agent(&band, guid, rec, here, t_s);
        let tier = plan.tier;
        let was = rec.tier;
        if tier != was {
            stats.retiered += 1;
        }
        // 3. The cached digest, taken on the way DOWN out of a posing tier —
        //    from the store the PREVIOUS step published, and **only for the
        //    agent that is demoting** (NPC1a audit).
        //
        //    The first cut folded every entry in the pose store into a map up
        //    front, which made this phase `O(posed characters × joints)` a step
        //    to serve a demotion that happens to one agent every few hundred.
        //    The wave's own sweep table is what says so: at N = 1 000 the crowd
        //    phase read **0.282 ms banded and 0.759 ms all-`Full`** — the same
        //    thousand agents doing the same work, differing only in how many
        //    characters were in the store — so most of a number minted as "what
        //    a thousand agents cost" was a fold over other systems' poses, and
        //    it grew with the level's hero count rather than with the crowd.
        if was.poses() && !tier.poses() {
            if let Some(d) = published_pose_digest(world, guid) {
                rec.pose_digest = d;
                stats.digests_folded += 1;
            }
        }
        rec.tier = tier;
        stats.per_tier[tier.as_u8() as usize] += 1;

        if !tier.materialized() {
            rec.last = plan.at;
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
        let p = plan.at;
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

/// A fold of **one** entity's published pose — the source of a record's
/// [`CrowdRecord::pose_digest`], and `None` for an entity the store does not
/// hold (which is every entity, on a level with no character at all).
///
/// FNV-1a over the same bytes [`crate::pose::pose_state_bytes`] emits for that
/// entity, so "the digest of the pose the trace would have carried" is exactly
/// what it says.
///
/// # Per agent, on demand, and not a map (NPC1a audit)
///
/// The first cut built a `BTreeMap` of **every** entry in the store at the top
/// of [`step_crowd_banded`], which is `O(posed characters × joints)` a step to
/// serve a demotion that happens to one agent every few hundred steps — and it
/// scaled with the *level's* posed characters rather than with the crowd, so a
/// hero-heavy level paid the crowd system for poses no agent owns. Called here
/// it is `O(demotions × joints)`, and a step on which nobody leaves a posing
/// tier — almost all of them — does not touch the store at all.
fn published_pose_digest(world: &EcsWorld, guid: Uuid) -> Option<u64> {
    let store = world.world().get_resource::<crate::pose::PoseStoreRes>()?;
    let ep = store.0.get(&guid)?;
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
    Some(h)
}

// ── what the rest of the engine reads ───────────────────────────────────────

/// The tier `entity` took this step, or `None` for anything that is not a crowd
/// agent — which is every hero and every authored character.
///
/// **No production caller** (the `set_debris_budget` seam, stated the way this
/// tree states them): the two in-engine readers of the verdict both had a
/// cheaper door already open — the pose evaluation queries `CrowdAgent`
/// alongside its other components, and `deform::ground_contacts` reads it off
/// the `EntityRef` its walk already holds. This is the *ad-hoc* reader, used by
/// the tests and by anything that has an [`Entity`] and one question.
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
/// `100 × 49 = 4 900` B a step against `100 × 6 476 = 647 600` B — **132×**.
///
/// # What the DIGEST buys, precisely
///
/// A demoted agent's pose is not merely small in the trace; it is **gone** —
/// [`crate::pose::step_pose_evaluation`] rebuilds its store from scratch each
/// step, so an agent that stopped posing has no entry and no current pose to
/// describe. The digest is therefore not a summary of live state; it is a fold of
/// the last pose the agent published, carried as **history**.
///
/// That is what makes it worth eight bytes: it puts the *step at which an agent
/// left the pose path* into the trace. Without it, a host that demoted one agent
/// a single step early would produce identical bytes until the two hosts happened
/// to run a pose that differed — which on a stationary crowd could be never. With
/// it, the two diverge on the step they disagreed, which is the step a reader
/// needs.
///
/// A section that emitted only the tier would nearly do the same job and would
/// lose one case: two runs that demoted the same agent on the same step from
/// *different* poses. That is the case a mid-trace start produces.
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

    /// **WHAT REFUSING HYSTERESIS ACTUALLY COSTS** (NPC1a audit) — the
    /// boundary-thrash arm one rung further down, in the currency the thrash is
    /// paid in.
    ///
    /// `an_agent_parked_on_a_tier_boundary_alternates_between_two` measures the
    /// *bound* on the thrash: two tiers, never a wander. That is the right thing
    /// to hold and it is not a cost, and the wave's ledger read it as one ("the
    /// cost of refusing it is measured rather than argued"). On the `Full`/`Near`
    /// line the cost really is nothing — the two tiers differ by a hand pass. On
    /// the **`Far`/`Dormant`** line it is an entity **spawned and despawned every
    /// step**: six components built and a subtree torn down, sixty times a second
    /// per parked agent, plus a `Guid` index insert and remove. A crowd standing
    /// near the edge of the world is the ordinary outcome rather than the exotic
    /// one, so the number belongs in the record NPC1b and NPC1d read.
    ///
    /// The arm asserts the shape (every step materializes and dematerializes)
    /// rather than a millisecond, and PRINTS the count. Fixing it is not this
    /// audit's to do and is not hysteresis: the fixes that keep the tier a pure
    /// function of sim state are a pooled entity or a quantized agent position,
    /// and both belong with the wave that has a renderer in it.
    /// # Where the thrash actually lives
    ///
    /// Not under a jittering source: the lattice exists precisely so that
    /// sub-cell movement cannot move an anchor, and a source wandering inside
    /// one 16 m cell re-tiers nothing. It lives on the **lattice line**, where a
    /// millimetre of solver residue moves the snapped anchor a whole 16 m — the
    /// `SimBand` arm's own mechanism, one system over, and here it is worth an
    /// entity a step rather than a re-stamp.
    #[test]
    fn a_parked_agent_on_the_dormant_edge_spawns_and_despawns_every_step() {
        let mut records = BTreeMap::new();
        records.insert(
            guid(0xE30),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
        );
        let mut world = crowd_world(records);

        // The line at 32 cells snaps to a centre 504 m from the agent on one
        // side (`Far` — it exists) and 520 m on the other (`Dormant` — it does
        // not), across the 512 m `DEFAULT_CROWD_FAR_M` boundary.
        let line = BAND_LATTICE_M * 32.0;
        let (mut spawned, mut despawned) = (0u64, 0u64);
        for step in 0..60 {
            let d = if step % 2 == 0 { -0.001 } else { 0.001 };
            move_source(&mut world, line + d);
            let s = step_crowd(&mut world, 1.0 / 60.0);
            spawned += s.spawned;
            despawned += s.despawned;
        }
        println!(
            "NPC1a hysteresis cost: one agent whose anchor sits on a lattice \
             line across the Far/Dormant boundary cost {spawned} entity \
             spawn(s) and {despawned} despawn(s) in 60 steps — one second of \
             wall clock, for ONE NPC"
        );
        assert!(
            spawned >= 29 && despawned >= 29,
            "the fixture is not straddling the Dormant edge: {spawned} spawn(s) \
             / {despawned} despawn(s) in 60 steps — the arm is measuring a \
             steady state and the cost it names is unrecorded"
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

    /// **THE CROWD PHASE'S WORK IS A FUNCTION OF THE CROWD** (NPC1a audit).
    ///
    /// The cached pose digest is the one thing this phase reads outside its own
    /// population, and the first cut read **all** of it: a `BTreeMap` of every
    /// entry in [`crate::pose::PoseStoreRes`], folded joint by joint, on every
    /// step, to serve a demotion that happens to one agent every few hundred.
    /// So the phase's cost scaled with the *level's* posed characters rather
    /// than with the population, and the budget minted from it was mostly a
    /// measurement of other systems' poses.
    ///
    /// The wave's own sweep table said so and nobody read it that way: at
    /// N = 1 000 the phase charged **0.282 ms banded and 0.759 ms all-`Full`** —
    /// the same thousand agents doing identical work, differing only in how many
    /// characters were in the store. After the fix the two agree (0.103 and
    /// 0.109), which is what a per-agent phase has to look like.
    ///
    /// This arm is a **counter, not a clock**, so it holds on any machine — and
    /// it is worth being exact about which half of the defect it holds. It pins
    /// the **semantics**: a digest is taken on the TRANSITION out of a posing
    /// tier and not once per step per Far agent, and a step on which nobody
    /// leaves a posing tier reads the store zero times however full it is.
    /// It cannot see an implementation that computed a hundred digests and threw
    /// ninety-nine away, because that is a cost and not a behaviour; the arm for
    /// *that* is `crowd_sweep.rs`'s banded-vs-all-`Full` crowd-phase comparison,
    /// which is a wall clock and is asserted where this tree asserts wall clocks
    /// (release, off CI) and reported everywhere else.
    #[test]
    fn a_settled_crowd_folds_no_pose_digests_however_many_characters_pose() {
        use crate::pose::{EvaluatedPose, PoseStoreRes};
        use inf_anim::{JointTransform, Pose};

        let agent = guid(0xF00);
        let mut records = BTreeMap::new();
        records.insert(
            agent,
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(10.0, 0.0, 0.0)),
        );
        let mut world = crowd_world(records);

        // A store with a hundred posed characters in it, one of which is the
        // agent. Fabricated rather than evaluated: this arm is about how many
        // of them the crowd phase touches, not about what a pose contains.
        let pose = Pose {
            locals: vec![JointTransform::IDENTITY; 32],
        };
        let mut store = BTreeMap::new();
        for i in 0..100u128 {
            store.insert(
                guid(0x9000 + i),
                EvaluatedPose {
                    skeleton: guid(0x77),
                    pose: pose.clone(),
                    sockets: Vec::new(),
                },
            );
        }
        store.insert(
            agent,
            EvaluatedPose {
                skeleton: guid(0x77),
                pose,
                sockets: Vec::new(),
            },
        );
        world.world_mut().insert_resource(PoseStoreRes(store));

        // Step 1: the agent classifies Full. Nothing leaves a posing tier.
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Full), 1, "{}", s.summary());
        assert_eq!(
            s.digests_folded, 0,
            "a crowd that promoted nobody folded {} digest(s) out of a \
             101-entry store — the phase is walking the level's poses rather \
             than its own demotions",
            s.digests_folded
        );

        // Step 2: the source walks away and the agent demotes. Exactly one.
        move_source(&mut world, 400.0);
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(s.at(CrowdTier::Far), 1, "{}", s.summary());
        assert_eq!(s.digests_folded, 1, "{}", s.summary());

        // Step 3: it is still Far, so there is nothing to fold — the store is
        // as full as it was and the phase does not look at it.
        let s = step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(
            s.digests_folded, 0,
            "an agent that was already Far folded another digest — the capture \
             is on the tier rather than on the TRANSITION"
        );

        // …and the digest it took is the pose that was published, not a zero.
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        assert_ne!(
            pop.records[&agent].pose_digest, 0,
            "the demotion folded a digest and stored nothing"
        );
    }

    /// **A DORMANT AGENT IS STILL ON ITS ROUTE** (NPC1a audit).
    ///
    /// The module's headline is that *the position law does not vary with the
    /// tier* — an agent's place is `route(clock)` at `Full` exactly as at `Far`.
    /// `Dormant` was the exception, and it was the exception in the one
    /// direction that costs something: a dematerialized record froze at `last`,
    /// **the tier was then decided from that frozen point**, and a walking agent
    /// whose route carried it home could never be re-admitted — it was judged
    /// for ever at the metre where it went out of range, while the *next*
    /// materialization would have placed it at `route(now)` somewhere else
    /// entirely. Frozen for the decision and live for the placement is the two
    /// authorities this module exists to avoid.
    ///
    /// A route is a pure function of the clock and costs a handful of flops, so
    /// keeping it live while an agent has no entity costs nothing and is the
    /// only reading under which `Dormant` is a *cost* tier rather than a
    /// one-way door.
    #[test]
    fn a_dormant_agent_keeps_walking_its_route_and_can_come_back() {
        let mut records = BTreeMap::new();
        // Out to 2 km and back, fast enough that the round trip fits in the
        // step budget of a unit test.
        records.insert(
            guid(0xE00),
            CrowdRecord::walking(
                CrowdArchetype::default(),
                CrowdRoute {
                    from: DVec3::new(0.0, 0.0, 0.0),
                    to: DVec3::new(2000.0, 0.0, 0.0),
                    speed_mps: 1000.0,
                },
            ),
        );
        // The anchor never moves: everything below is the AGENT walking.
        let mut world = crowd_world(records);

        let mut went_dormant = None;
        let mut came_back = None;
        for step in 0..400u64 {
            let s = step_crowd(&mut world, 1.0 / 60.0);
            let dormant = s.at(CrowdTier::Dormant) == 1;
            if dormant && went_dormant.is_none() {
                went_dormant = Some(step);
            }
            if went_dormant.is_some() && !dormant && came_back.is_none() {
                came_back = Some(step);
            }
        }
        let out = went_dormant.expect(
            "the agent never left the band, so this arm is not posing the problem — \
             the route must reach past DEFAULT_CROWD_FAR_M",
        );
        let back = came_back.unwrap_or_else(|| {
            panic!(
                "the agent went Dormant at step {out} and never came back over 400 \
                 steps of a route that returns to its anchor — a dematerialized \
                 agent is being tiered from where it froze rather than from where \
                 its route says it is"
            )
        });
        println!(
            "NPC1a dormancy: an agent walking a 2 km route went Dormant at step \
             {out} and re-materialized at step {back}"
        );

        // …and the record's remembered place tracked the route the whole way,
        // which is what makes the tier above a decision about the present.
        let pop = world
            .world()
            .get_resource::<CrowdPopulationRes>()
            .expect("the population");
        let rec = pop.records[&guid(0xE00)];
        let want = rec.position_at(guid(0xE00), (pop.steps - 1) as f64 * (1.0 / 60.0));
        assert_eq!(
            rec.last, want,
            "a record's `last` diverged from its own route — the trace's 24 \
             position bytes and the tier decision are reading different places"
        );
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

    /// **INSTALLING A POPULATION TAKES THE OLD ONE'S BODIES WITH IT** (NPC1a
    /// audit).
    ///
    /// [`set_population`] says it replaces the population a world already had,
    /// and a population is not only its records: a materialized agent is a real
    /// entity carrying a skeletal mesh, a machine, a capsule and a
    /// [`CrowdAgent`]. Dropping the resource alone left those standing, with a
    /// tier frozen at whatever the last step decided and **no record behind
    /// them** — so [`bodiless_agents`] (which reads records) and the
    /// [`CrowdAgent`] component (which the pose door and the deform pass read)
    /// would answer differently about the same entity, which is the two-opinions
    /// defect this wave's own deform finding is about.
    #[test]
    fn installing_a_second_population_does_not_leave_the_first_standing() {
        let mut first = BTreeMap::new();
        for i in 0..4u128 {
            first.insert(
                guid(0xE10 + i),
                CrowdRecord::standing(CrowdArchetype::default(), DVec3::new(i as f64, 0.0, 0.0)),
            );
        }
        let mut world = crowd_world(first);
        step_crowd(&mut world, 1.0 / 60.0);
        assert_eq!(crowd_stats(&world).at(CrowdTier::Full), 4);

        let mut second = BTreeMap::new();
        second.insert(
            guid(0xE20),
            CrowdRecord::standing(CrowdArchetype::default(), DVec3::ZERO),
        );
        set_population(&mut world, second);
        step_crowd(&mut world, 1.0 / 60.0);

        for i in 0..4u128 {
            assert!(
                world.entity_of(guid(0xE10 + i)).is_none(),
                "agent {i} of the replaced population is still standing in the \
                 world with no record behind it — the crowd door and the tier \
                 component now disagree about whether it exists"
            );
        }
        assert_eq!(crowd_state_bytes(&world).len(), AGENT_TRACE_BYTES);
        assert_eq!(crowd_stats(&world).total(), 1);
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
