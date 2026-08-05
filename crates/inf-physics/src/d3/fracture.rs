//! **Runtime destruction (P22.3): the intact→chunks swap, the structural solve,
//! and the debris budget.**
//!
//! P22.2 built the *content*: a `.inf_fracture` derived at cook for every mesh a
//! [`Destructible`] names, holding watertight convex chunks with volumes, centres
//! of mass and a clean chunk-indexed adjacency graph. This module is what happens
//! when one of those breaks at 60 Hz.
//!
//! # The shape of the thing
//!
//! * [`FractureState`] is one destructible actor's live state — which chunks have
//!   come off, where they are, how old they are. It is **sim state**, owned by the
//!   host (`SimSession::fractures` / `RuntimeSim::fractures`) exactly the way the
//!   voxel volume map is, and for the same reason: it must never be reachable from
//!   anything a camera drives.
//! * [`PhysicsBridge3D::runtime_destruct`] is the **one Ring-0 rule** both hosts'
//!   `destruct.apply_damage` arms call. Refusals are values, never failures.
//! * [`PhysicsBridge3D::step_fractures`] is the fixed-step advance: age the
//!   debris, run the structural solve, apply the budget, latch `Destroyed`.
//! * `PhysicsBridge3D::gather_fracture` (in the bridge) turns the state into
//!   rapier bodies — the `gather_voxels` pattern, one more time.
//!
//! # There are no new entities, and there was never going to be
//!
//! A shattered wall does **not** spawn twelve actors. `engine.spawn` does not
//! exist in this engine, `.inf_lvl` would grow twelve derived rows per wall, undo
//! would mean something new, and two hosts would each need a
//! despawn-before-re-evaluate pass. Chunks live in `FractureState` and reach the
//! solver through **synthetic, content-derived guids**
//! ([`fracture_chunk_guid`]) — the P19.5 scattered-solid decision and the P21.4
//! voxel-chunk decision, taken a third time because it keeps being right.
//!
//! # Anchoring: what holds a chunk up
//!
//! `docs/memos/p22-strength.md` §4.3 settled this at P22.2 so that scene schema
//! **v20 could be frozen**, and this module implements exactly what it says. A
//! chunk is **supported** when it
//!
//! 1. is in contact with **static geometry** — the ground, a terrain tile,
//!    another static body — at detach-evaluation time; or
//! 2. belongs to an entity marked [`AlwaysLoaded`], the explicit override; or
//! 3. reaches a supported chunk through **undetached** neighbours.
//!
//! Everything else falls. Removing a chunk therefore drives *progressive*
//! collapse, one fixed step at a time, deterministically.
//!
//! # Nothing here is scheduled, and that is why there is no pool probe
//!
//! P20.2 needed a **subprocess** gate because its water index sat behind a
//! process-global `OnceLock` pool, so "is this deterministic" could only be asked
//! by starting a fresh process with a different pool size. This module has no
//! such seam to test: it names no job pool (`inf_core::parallel_map` and friends
//! appear nowhere in `inf-physics`), and rapier's own `parallel` feature is off by
//! rule (see the crate docs), so the fixed step is single-threaded from the
//! damage plan through the solve to the budget sweep. A probe binary here would
//! vary nothing, and a gate that varies nothing is the vacuous kind. Every order
//! that could matter is instead a `BTreeMap`/`BTreeSet` or an explicit sort, and
//! `the_same_damage_trace_is_byte_identical` compares the raw bits.
//!
//! The override is **per entity**, because that is the granularity the component
//! set has — and the memo chose that granularity deliberately rather than pay a
//! `StructuralAnchor` slot and a v21 in both codec mirrors. An author who needs a
//! pinned base course splits the structure into two actors and marks the base
//! `AlwaysLoaded`; the memo's own worked example ("the base course of a tower, a
//! bridge abutment") is exactly that shape.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use glam::{DAffine3, DQuat, DVec3};
use inf_ecs::components::{AlwaysLoaded, Destructible, GlobalTransform, MeshRef, Transform};
use inf_ecs::EcsWorld;
use inf_mesh::{FractureAsset, FractureChunk};
use uuid::Uuid;

use super::ecs::PhysicsBridge3D;
use super::world::{BodyKind3D, ColliderDesc3D, ColliderShape3D};
use super::{BodyId3D, ColliderId3D};
use crate::filtering::CollisionLayers;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — every one of them argued
// ─────────────────────────────────────────────────────────────────────────────

/// The characteristic **crack-opening displacement** at failure, in metres: the
/// distance a bond's face separates while the bond is still resisting.
///
/// This is the "documented thickness class" `docs/memos/p22-strength.md` §4
/// leaves to P22.3, and it is the *one* number that turns the memo's force
/// contract into an energy one. The derivation:
///
/// > A bond holds up to `strength × area` **newtons** (the memo's §4.2, and
/// > [`Destructible::bond_force_n`] is the contract). Work is force times
/// > distance. So the **energy** to break that bond is
/// > `strength × area × CRACK_OPENING_M` **joules** — and `Pa · m² · m = N · m =
/// > J` exactly, with no invented conversion constant anywhere.
///
/// **Why 1 mm.** Brittle materials — the whole `strength` table's subject — fail
/// at strains around `1e-4` to `1e-3`. Over a chunk of order one metre that is a
/// displacement of 0.1 to 1 mm, so a millimetre is the top of the honest range
/// and errs toward "things are a little harder to break", which is the direction
/// a designer can fix by turning `strength` down.
///
/// The sanity check that matters: a masonry wall (2 MPa) whose chunks share
/// 1 m² faces costs `2e6 × 1 × 1e-3 = 2 kJ` per bond. A grenade delivers a few
/// kilojoules to a nearby wall and a rocket some tens — so one grenade takes a
/// chunk or two out and a rocket opens a hole, which is the behaviour the numbers
/// exist to produce. A footstep (a few joules) does nothing, and that is the other
/// end of the check.
///
/// It is a **global material constant, not a per-asset field**, for the reason
/// §5 of the memo gives about everything else that looked like a missing field:
/// it describes fracture mechanics, not this actor's material. `strength` is the
/// per-actor knob and it multiplies straight through.
pub const CRACK_OPENING_M: f64 = 1.0e-3;

/// Fraction of the source mesh's extent within which two chunks' hull corners
/// count as **the same corner** — the tolerance the shared-face-area measurement
/// matches on.
///
/// (It used to describe a triangle-on-a-plane test; that algorithm is gone — see
/// [`shared_face_area_m2`] for why it was wrong. The tolerance survives with a
/// new job: deciding when `a`'s corner and `b`'s corner are the same point.)
///
/// A shared Voronoi face's corners are the intersection of the *same* three
/// half-space planes in both cells, so the two hulls carry them at coordinates
/// that agree to within the arithmetic that produced them. `1e-4 × 10 m = 1 mm`
/// clears that comfortably while staying far below any real geometric feature —
/// two genuinely distinct corners of a metre-scale chunk are never a millimetre
/// apart.
const FACE_PLANE_EPS_FRACTION: f64 = 1.0e-4;

/// The vertical skin, in metres, within which a chunk counts as **in contact
/// with** static geometry for the support test.
///
/// Two centimetres: above rapier's own contact prediction distance and below any
/// gap a level designer means to be a gap. A chunk resting on the ground is
/// separated from it by the solver's contact slop, so a strictly-zero test would
/// report a settled building as unsupported and collapse it on the first step.
const SUPPORT_SKIN_M: f64 = 0.02;

/// How many of a chunk's lowest hull vertices the support test probes from.
///
/// Four, not one and not all of them: one probe misses a chunk resting on an
/// edge, and all of them costs a ray per vertex on geometry that is convex
/// anyway. The four lowest points of a convex solid are its contact patch to
/// within the skin above.
const SUPPORT_PROBES: usize = 4;

/// Default level-wide cap on simultaneously-live detached chunks.
///
/// A *default*, not a ratchet. The roadmap puts **per-tier** debris budgets in
/// P22.4, and the tier vocabulary (`inf_render::RenderTier`) lives in the render
/// crate — which physics must never depend on, because that is the wrong
/// direction and would make a fixed step's contents a function of the graphics
/// settings by construction. So the budget arrives here as **data**
/// ([`DebrisBudget`]), the host sets it, and P22.4 fills in the mapping. What
/// this constant buys today is that an unconfigured host still has a bound.
pub const DEFAULT_DEBRIS_MAX_LIVE: u32 = 256;

/// Default seconds a detached chunk survives before the budget may reclaim it.
///
/// Twenty: long enough that a collapse finishes settling and a player who turns
/// around still sees the rubble, short enough that a firefight through a street
/// does not accumulate a thousand resting hulls. Like [`DEFAULT_DEBRIS_MAX_LIVE`]
/// it is a default the host may override, not a ceiling anything asserts.
pub const DEFAULT_DEBRIS_LIFETIME_S: f64 = 20.0;

// ─────────────────────────────────────────────────────────────────────────────
// Identity
// ─────────────────────────────────────────────────────────────────────────────

/// Salt for [`fracture_chunk_guid`]. A fourth distinct constant, so a scattered
/// solid, a voxel chunk, a terrain tile and a fracture chunk can never collide in
/// the bridge's one entity map.
const FRACTURE_CHUNK_SALT: u128 = 0x2203_0200_4652_4143_4348_554e_4b21_0033;

/// The synthetic identity of chunk `index` of the destructible on entity
/// `entity`.
///
/// The [`super::voxel_chunk_guid`] rule with one index folded in instead of three
/// coordinates, and a 128-bit mix rather than a XOR for the same reason: two
/// walls whose guids differ in the low bits must not alias each other's chunks.
/// Stated as one function so a debug view, a save hook or the render projector
/// names one of these the same way the bridge does.
pub fn fracture_chunk_guid(entity: Uuid, index: u32) -> Uuid {
    let mut x = entity.as_u128() ^ FRACTURE_CHUNK_SALT;
    x ^= (index as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// One chunk's runtime state. Parallel to `FractureAsset::chunks`, by index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkState {
    /// Detached from the structure — a dynamic body and a piece of debris.
    pub detached: bool,
    /// Reclaimed by the debris budget or its lifetime: no body, no render, and
    /// it never comes back.
    pub gone: bool,
    /// The sequence number this chunk detached at (`0` while attached). The
    /// budget reclaims the **lowest** first, which is "oldest debris first".
    pub detach_order: u64,
    /// Seconds since detaching (`0.0` while attached).
    pub age_s: f64,
    /// World translation. The rest pose while attached; solver-owned once
    /// detached (written back each step by
    /// [`PhysicsBridge3D::write_back_fractures`]).
    pub translation: DVec3,
    /// World orientation. Identity at rest.
    pub rotation: DQuat,
}

impl ChunkState {
    /// This chunk's state as raw bits — the exact comparator a determinism gate
    /// wants.
    ///
    /// `PartialEq` on the pose is IEEE equality, which is *almost* the same thing
    /// and differs in exactly the case a gate cares about (a NaN that crept into
    /// one host's trace would compare unequal to itself and be reported as
    /// "different", which is right, but `-0.0 == 0.0` would be reported as "the
    /// same", which is not). Bits answer the question that was asked.
    pub fn bits(&self) -> [u64; 8] {
        [
            self.detached as u64,
            self.gone as u64,
            self.detach_order,
            self.age_s.to_bits(),
            self.translation.x.to_bits(),
            self.translation.y.to_bits(),
            self.translation.z.to_bits(),
            // The quaternion folded to one word: four `to_bits` would make the
            // trace twice as long for a value that is one orientation.
            self.rotation.x.to_bits()
                ^ self.rotation.y.to_bits().rotate_left(16)
                ^ self.rotation.z.to_bits().rotate_left(32)
                ^ self.rotation.w.to_bits().rotate_left(48),
        ]
    }
}

/// One destructible actor's live fracture state.
///
/// Seeded **after** sim construction (the [`resolve_fracture_states`] door), for
/// the same reason the voxel map is: resolving a `.inf_fracture` needs the built
/// world to walk, and `BeginPlay` runs inside that construction. A `destruct.*`
/// node on a `BeginPlay` handler therefore sees no state and refuses — stated in
/// the node-kit docs rather than worked around, because the workaround (deferring
/// `BeginPlay`) changes when every handler in the engine runs.
#[derive(Clone, Debug, PartialEq)]
pub struct FractureState {
    /// The cook's chunk set. `Arc` because two hosts, the bridge and the render
    /// projector all read the same bytes and none of them writes them.
    asset: Arc<FractureAsset>,
    /// Runtime state, one entry per `asset.chunks`.
    chunks: Vec<ChunkState>,
    /// Precomputed bond energies, keyed by the canonical `(a, b)` pair with
    /// `a < b`, in **joules**. A pure function of the asset and the component, so
    /// it is computed once at seed time rather than per damage call.
    bonds: BTreeMap<(u32, u32), f64>,
    /// Bonds whose shared face the measurement could **not** find, and which are
    /// therefore priced from the `min(volume)^(2/3)` estimate instead.
    ///
    /// Recorded rather than inferred: "was this area measured or guessed" is the
    /// difference between the shared-face algorithm working and the shared-face
    /// algorithm silently doing nothing, and the P22.3 audit found the previous
    /// algorithm doing exactly the latter for **100%** of a real cook's bonds with
    /// nothing to notice it. A test can now name the pairs.
    estimated: BTreeSet<(u32, u32)>,
    /// Per-chunk **ground-bond** energy, joules: what it costs to break a chunk
    /// away from the static geometry it is standing on, when it is standing on
    /// any.
    ///
    /// # Why this term exists at all
    ///
    /// Without it a tower's base chunk and its top chunk are equally cheap — both
    /// have exactly one neighbour — so a low-index tie-break knocks the base out
    /// for one bond's energy and the whole tower comes down from a tap. (That is
    /// not hypothetical: it is what the first cut of this module did, and
    /// `energy_bookkeeping_is_exact` is what said so.)
    ///
    /// It is also the honest physics. A wall is not merely stacked on its
    /// foundation, it is *bonded* to it — mortar, rebar, a footing — and the memo's
    /// whole model is that a bond is a face of some area at some failure stress.
    /// The ground bond is that same rule with the chunk's own characteristic
    /// cross-section, `volume^(2/3)`, standing in for a contact patch nothing
    /// records. For a unit cube that is exactly `1 m²`, i.e. one ordinary bond,
    /// which is the right order: a base course is about twice as hard to remove as
    /// the brick above it, not ten times and not the same.
    ground_bonds: Vec<f64>,
    /// The actor's material contract, copied at seed time so the solve never has
    /// to reach back into the ECS mid-step.
    destructible: Destructible,
    /// The entity carries [`AlwaysLoaded`] — the memo's explicit structural
    /// anchor override, which makes **every** chunk of this actor an anchor.
    anchored: bool,
    /// The actor's world placement: what maps the asset's source-mesh-local
    /// geometry into the world.
    ///
    /// **Re-derived from the actor's `GlobalTransform` every sync while it is
    /// still intact**, and frozen the instant the first chunk comes off. Both
    /// halves matter:
    ///
    /// * Before the break the actor is a normal entity that a Blueprint, a
    ///   sequencer or a gizmo drag can move, and a placement fixed at seed time
    ///   would shatter a wall at the location it was *loaded* at — the P22.3
    ///   audit's M5, reproduced with a wall at `x = 50` whose chunks, colliders
    ///   and rubble all appeared at `x = 0`.
    /// * After the break there is nothing left to follow. The chunks are
    ///   solver-owned bodies with their own poses, and the entity's transform
    ///   has stopped describing them — re-deriving then would teleport settled
    ///   rubble whenever anything touched the (now invisible) actor.
    placement: DAffine3,
    /// `|det|` of the placement's linear part — the volume scale a non-unit
    /// entity scale applies to every chunk's authored `volume_m3`.
    volume_scale: f64,
    /// Monotone stamp, bumped whenever a chunk's *identity* changes (detached or
    /// reclaimed). The bridge's and the renderer's change stamp; **not** bumped
    /// by pose, which moves every step for every piece of live debris.
    generation: u64,
    /// Mints [`ChunkState::detach_order`].
    next_order: u64,
    /// `Destroyed` has already fired for this actor.
    destroyed_fired: bool,
}

impl FractureState {
    /// Seed a state from a resolved asset, an actor's component, its world
    /// placement and its anchor flag.
    ///
    /// Every chunk starts attached at its rest pose, so a freshly-seeded state is
    /// **indistinguishable from an intact actor** — [`is_intact`](Self::is_intact)
    /// is `true`, the bridge emits no chunk bodies, and the actor keeps its
    /// authored collider and renders its own mesh.
    pub fn new(
        asset: Arc<FractureAsset>,
        destructible: Destructible,
        placement: DAffine3,
        anchored: bool,
    ) -> Self {
        let volume_scale = placement.matrix3.determinant().abs();
        let chunks = asset
            .chunks
            .iter()
            .map(|c| ChunkState {
                detached: false,
                gone: false,
                detach_order: 0,
                age_s: 0.0,
                translation: placement.transform_point3(DVec3::from_array(c.center_of_mass)),
                rotation: DQuat::IDENTITY,
            })
            .collect();
        // The adjacency the solve walks is assumed SYMMETRIC: `bond_energies`
        // canonicalises its pairs, but the support BFS walks each chunk's own
        // `neighbors` list, so a one-sided edge would price correctly and then
        // propagate support in one direction only. P22.2's cook does not enforce
        // symmetry, so this is where a violation is caught — in debug, at seed
        // time, naming the pair — rather than as a building that collapses from
        // one side and not the other.
        debug_assert!(
            asset.chunks.iter().enumerate().all(|(i, c)| {
                c.neighbors.iter().all(|&n| {
                    asset
                        .chunks
                        .get(n as usize)
                        .is_some_and(|o| o.neighbors.contains(&(i as u32)))
                })
            }),
            "the fracture asset has a one-sided adjacency edge; the support solve              would propagate through it in one direction only"
        );
        let (bonds, estimated) = bond_energies(&asset, &destructible, volume_scale);
        let ground_bonds = ground_bond_energies(&asset, &destructible, volume_scale);
        Self {
            asset,
            chunks,
            bonds,
            estimated,
            ground_bonds,
            destructible,
            anchored,
            placement,
            volume_scale,
            generation: 1,
            next_order: 1,
            destroyed_fired: false,
        }
    }

    /// The chunk set this state was seeded from.
    pub fn asset(&self) -> &Arc<FractureAsset> {
        &self.asset
    }

    /// Per-chunk runtime state, by chunk index.
    pub fn chunks(&self) -> &[ChunkState] {
        &self.chunks
    }

    /// The actor's material contract.
    pub fn destructible(&self) -> &Destructible {
        &self.destructible
    }

    /// The actor's world placement (see the field).
    pub fn placement(&self) -> DAffine3 {
        self.placement
    }

    /// Follow the actor's transform while it is still whole (P22.3).
    ///
    /// A no-op once anything has detached, and a no-op when the transform has not
    /// moved — so the steady state is one `DAffine3` comparison per destructible
    /// per sync, and a level whose walls nobody is dragging pays nothing.
    ///
    /// Re-derives every quantity the placement feeds: the chunks' rest poses, the
    /// volume scale (mass), and the bond areas (a bigger wall is harder to break
    /// through). The generation is **not** bumped — moving an intact actor
    /// changes where its chunks would be, not which of them exist, and bumping
    /// would re-describe colliders that do not exist yet.
    pub fn follow(&mut self, placement: DAffine3) {
        if !self.is_intact() || placement == self.placement {
            return;
        }
        self.placement = placement;
        self.volume_scale = placement.matrix3.determinant().abs();
        let (bonds, estimated) = bond_energies(&self.asset, &self.destructible, self.volume_scale);
        self.bonds = bonds;
        self.estimated = estimated;
        self.ground_bonds =
            ground_bond_energies(&self.asset, &self.destructible, self.volume_scale);
        for (i, c) in self.chunks.iter_mut().enumerate() {
            if let Some(src) = self.asset.chunks.get(i) {
                c.translation = placement.transform_point3(DVec3::from_array(src.center_of_mass));
                c.rotation = DQuat::IDENTITY;
            }
        }
    }

    /// The change stamp. Moves when a chunk detaches or is reclaimed; never when
    /// one merely moves.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Nothing has come off yet — the actor still renders its own mesh and keeps
    /// its authored collider.
    pub fn is_intact(&self) -> bool {
        !self.chunks.iter().any(|c| c.detached || c.gone)
    }

    /// How many chunks the cook produced. `0` is impossible for a resident state
    /// (the cook refuses a fracture below [`inf_mesh::MIN_CHUNK_COUNT`]).
    pub fn chunk_count(&self) -> u32 {
        self.chunks.len() as u32
    }

    /// How many chunks are currently detached and not yet reclaimed — live
    /// debris from this actor.
    pub fn live_debris(&self) -> u32 {
        self.chunks.iter().filter(|c| c.detached && !c.gone).count() as u32
    }

    /// The mass, kg, of chunk `index` under this actor's density and scale.
    pub fn chunk_mass_kg(&self, index: usize) -> f64 {
        self.asset
            .chunks
            .get(index)
            .map(|c| {
                self.destructible
                    .chunk_mass_kg(c.volume_m3 * self.volume_scale)
            })
            .unwrap_or(0.0)
    }

    /// The whole state as raw bits — the comparator a cross-host determinism gate
    /// uses. Includes the generation and the destroyed latch, so a run that
    /// merely *reached* a different collapse ordering is caught even if the poses
    /// happen to coincide.
    pub fn bits(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.chunks.len() * 8 + 3);
        out.push(self.generation);
        out.push(self.next_order);
        out.push(self.destroyed_fired as u64);
        for c in &self.chunks {
            out.extend_from_slice(&c.bits());
        }
        out
    }

    /// The bonds priced from the `min(volume)^(2/3)` estimate because their
    /// shared face could not be measured — canonical `(a, b)` pairs with `a < b`.
    ///
    /// **Empty is the healthy answer for geometry that has real faces**, and a
    /// hand-built fixture asserts exactly that. Real cook output is allowed a
    /// small residue: the cook's adjacency comes from the Voronoi *sites*, and a
    /// cell clipped against the source mesh's hull can lose its shared face
    /// entirely — leaving two chunks that are neighbours in the graph and touch
    /// along an edge or not at all. There is nothing to measure there, which is
    /// what the estimate is for.
    pub fn estimated_bonds(&self) -> &BTreeSet<(u32, u32)> {
        &self.estimated
    }

    /// The energy, joules, needed to break chunk `index` away from the static
    /// geometry it rests on — `0.0` for a chunk that rests on nothing (the caller
    /// decides which it is; this is only the price).
    pub fn ground_bond_j(&self, index: usize) -> f64 {
        self.ground_bonds.get(index).copied().unwrap_or(0.0)
    }

    /// The energy, joules, needed to break the bond between chunks `a` and `b`,
    /// or `0.0` when they are not neighbours. Precomputed at seed time.
    pub fn bond_energy_j(&self, a: u32, b: u32) -> f64 {
        let key = if a < b { (a, b) } else { (b, a) };
        self.bonds.get(&key).copied().unwrap_or(0.0)
    }

    /// Detach chunk `index`, minting its order. Idempotent, and a no-op for a
    /// reclaimed chunk.
    fn detach(&mut self, index: usize) -> bool {
        let order = self.next_order;
        let Some(c) = self.chunks.get_mut(index) else {
            return false;
        };
        if c.detached || c.gone {
            return false;
        }
        c.detached = true;
        c.detach_order = order;
        c.age_s = 0.0;
        self.next_order += 1;
        self.generation += 1;
        true
    }

    /// Reclaim chunk `index` — the budget's despawn. A reclaimed chunk leaves the
    /// physics world and the render set together, by omission from both.
    fn reclaim(&mut self, index: usize) -> bool {
        let Some(c) = self.chunks.get_mut(index) else {
            return false;
        };
        if c.gone {
            return false;
        }
        c.gone = true;
        self.generation += 1;
        true
    }
}

/// Per-level debris limits. Owned by the host and handed to
/// [`PhysicsBridge3D::step_fractures`] each step.
///
/// See [`DEFAULT_DEBRIS_MAX_LIVE`] for why this is data rather than a constant
/// physics reads for itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebrisBudget {
    /// Level-wide cap on simultaneously-live detached chunks.
    pub max_live: u32,
    /// Seconds a detached chunk survives before it may be reclaimed.
    pub lifetime_s: f64,
}

impl Default for DebrisBudget {
    fn default() -> Self {
        Self {
            max_live: DEFAULT_DEBRIS_MAX_LIVE,
            lifetime_s: DEFAULT_DEBRIS_LIFETIME_S,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Refusals — values, never failures
// ─────────────────────────────────────────────────────────────────────────────

/// Why a `destruct.*` action did or did not run.
///
/// The [`inf_voxel::RuntimeCarveOutcome`] shape, deliberately: a gameplay refusal
/// must be a **value**, because an erroring node takes its whole handler down and
/// the author fixes it by typing a different number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DestructOutcome {
    /// The damage was applied (possibly absorbing zero energy — see
    /// [`DamageReport::energy_absorbed_j`]).
    Applied,
    /// The entity has no [`Destructible`] component.
    NoDestructible,
    /// The component's `runtime_destruct` gate is off.
    NotPermitted,
    /// No `.inf_fracture` is resident for this entity — it was never cooked, the
    /// mesh reference is missing, or the state has not been seeded yet (which is
    /// what a `BeginPlay` handler sees).
    NoFracture,
    /// Degenerate parameters — a non-finite or non-positive energy.
    Invalid,
}

impl DestructOutcome {
    /// Whether the action ran.
    pub fn applied(&self) -> bool {
        matches!(self, DestructOutcome::Applied)
    }

    /// The log line a host emits for a refusal, or `None` when the action ran.
    ///
    /// One text, shared by both hosts, so a refusal reads identically in the
    /// editor's Output Log and in the shipped player's tracing stream — the same
    /// reason the structural solve is one function.
    pub fn refusal(&self, op_name: &str) -> Option<String> {
        match self {
            DestructOutcome::Applied => None,
            DestructOutcome::NoDestructible => Some(format!(
                "{op_name}: no Destructible on that entity — nothing to break"
            )),
            DestructOutcome::NotPermitted => Some(format!(
                "{op_name}: refused — this actor's `runtime_destruct` is off"
            )),
            DestructOutcome::NoFracture => Some(format!(
                "{op_name}: refused — no fracture data resident for that entity \
                 (was its mesh cooked with a Destructible, and is this a Tick \
                 rather than a BeginPlay?)"
            )),
            DestructOutcome::Invalid => Some(format!(
                "{op_name}: refused — the energy (joules) and the actor's strength \n                 (pascals) must both be positive, finite numbers"
            )),
        }
    }
}

/// What one [`PhysicsBridge3D::runtime_destruct`] call did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageReport {
    /// Applied, or which refusal.
    pub outcome: DestructOutcome,
    /// **Joules actually consumed breaking bonds.** `0.0` for every refusal, and
    /// also for an impact too weak to break the cheapest bond — which is the same
    /// fact from gameplay's side (nothing came off) and differs only in the log.
    ///
    /// Energy rather than a chunk count because a joule is a joule: it composes
    /// (a caller can subtract what was absorbed from what it delivered), it is
    /// comparable against the impulses and forces the solver already produces,
    /// and it is the quantity `strength` is defined against. It is also the
    /// `voxel.carve_*` precedent — those report the **cubic metres** moved, not
    /// the number of samples touched, for the reason
    /// `docs/memos/units-doctrine.md` gives.
    pub energy_absorbed_j: f64,
    /// Chunks this call detached, including any the structural solve then dropped
    /// for lack of support.
    pub detached: u32,
}

impl DamageReport {
    fn refused(outcome: DestructOutcome) -> Self {
        Self {
            outcome,
            energy_absorbed_j: 0.0,
            detached: 0,
        }
    }
}

/// What one [`PhysicsBridge3D::step_fractures`] pass did — the audit instrument,
/// on the `ScatterAudit` precedent rather than a new budget ratchet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FractureAudit {
    /// Destructible actors with a live state this step.
    pub tracked: u32,
    /// Chunks detached by the **structural solve** (not by damage) this step —
    /// progressive collapse, made countable.
    pub collapsed: u32,
    /// Chunks reclaimed because they outlived [`DebrisBudget::lifetime_s`].
    pub expired: u32,
    /// Chunks reclaimed because the level was over [`DebrisBudget::max_live`].
    pub over_budget: u32,
    /// Live detached chunks across the level after the sweep.
    pub live_debris: u32,
}

/// An actor that finished breaking this step — the `Destroyed` event's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestroyedEvent {
    /// The destructible actor.
    pub entity: Uuid,
    /// Chunks detached at the moment it fired.
    pub detached: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Seeding
// ─────────────────────────────────────────────────────────────────────────────

/// Walk the world for [`Destructible`] actors and build their
/// [`FractureState`]s, resolving each one's `.inf_fracture` through `load`.
///
/// **One resolver in Ring 0**, called by both hosts and by the PIE seam, rather
/// than the same walk written out three times — the `inf_voxel::ground_height_at`
/// lesson. `load` is handed the actor's **mesh** GUID and owes back the derived
/// fracture asset; where those bytes come from is the caller's business (a cooked
/// pack entry at [`inf_mesh::derived_fracture_id`], the PIE payload's tail, or a
/// live derivation from the editor's asset DB), and that is the only thing that
/// differs between the three.
///
/// Deterministic: `BTreeMap` keyed by entity `Guid`, and the walk's own order is
/// irrelevant because nothing here depends on it.
///
/// An actor whose mesh has no fracture asset is simply **absent** from the map,
/// which is what makes `destruct.apply_damage` on it a
/// [`DestructOutcome::NoFracture`] value rather than a panic.
pub fn resolve_fracture_states(
    world: &EcsWorld,
    mut load: impl FnMut(Uuid) -> Option<Arc<FractureAsset>>,
) -> BTreeMap<Uuid, FractureState> {
    let mut out = BTreeMap::new();
    // Collect first (the walk holds immutable EntityRefs), then resolve.
    let mut wanted: Vec<(Uuid, Destructible, Uuid, DAffine3, bool)> = Vec::new();
    for e in world.world().iter_entities() {
        let Some(guid) = e.get::<inf_ecs::Guid>().map(|g| g.0) else {
            continue;
        };
        let Some(d) = e.get::<Destructible>() else {
            continue;
        };
        // The component names no asset on purpose (memo §5): the fracture is
        // derived from the entity's own `MeshRef.asset`, so a second authority —
        // and a second dangling reference to advise about — never exists.
        let Some(mesh) = e.get::<MeshRef>().and_then(|m| m.asset) else {
            continue;
        };
        let placement = e.get::<GlobalTransform>().map(|g| g.0).unwrap_or_else(|| {
            e.get::<Transform>()
                .map(|t| DAffine3::from_translation(t.translation.to_dvec3()))
                .unwrap_or(DAffine3::IDENTITY)
        });
        let anchored = e.get::<AlwaysLoaded>().is_some();
        wanted.push((guid, *d, mesh, placement, anchored));
    }
    wanted.sort_by_key(|(g, _, _, _, _)| *g);
    // **`load` is called once per MESH, not once per actor.** A street of 500
    // destructible walls sharing 10 meshes is 500 calls without this, and the
    // editor's `load` runs the whole Voronoi chunking — a level that would take a
    // second to enter Simulate rather than a moment. The `Arc` is what makes the
    // sharing free: 500 states hold 10 chunk sets between them.
    //
    // A cached `None` counts: a mesh that cannot be resolved must not be retried
    // 499 times either.
    let mut cache: BTreeMap<Uuid, Option<Arc<FractureAsset>>> = BTreeMap::new();
    for (guid, d, mesh, placement, anchored) in wanted {
        let asset = cache.entry(mesh).or_insert_with(|| load(mesh)).clone();
        if let Some(asset) = asset {
            out.insert(guid, FractureState::new(asset, d, placement, anchored));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond geometry
// ─────────────────────────────────────────────────────────────────────────────

/// The area, m², of the face chunks `a` and `b` share.
///
/// # How, and why not the obvious way
///
/// The obvious measurement is "the chunks are Voronoi cells, so their shared face
/// lies on the perpendicular bisector of their two centres — sum `a`'s triangles
/// on that plane". **It is wrong, and it fails silently.** A Voronoi face is the
/// bisector of the two *sites*, and a cell's `center_of_mass` is its volume
/// centroid, which is not its site: clip a cell against the source mesh's hull and
/// the centroid slides away from the seed. Measured that way, every bond of a real
/// `fracture_mesh` asset found nothing on the plane and fell through to the
/// fallback below — a whole building's bond energies quietly replaced by an
/// estimate, with nothing to notice it.
///
/// So the face is found from the **geometry the two chunks agree on**: their
/// hull point sets are `f64` and a shared Voronoi face's corners are the
/// intersection of the *same* three half-space planes in both cells, so the two
/// hulls carry those corners at the same coordinates. The common points are the
/// face's polygon; fit a plane through them, order them around their centroid, and
/// sum the fan.
///
/// Working in `f64` hull points rather than `f32` render vertices is the other
/// half of that: the vertices a chunk draws with are quantised, and quantisation
/// noise is exactly the size of the tolerance this measurement needs.
///
/// Measured **once per bond at seed time** ([`bond_energies`]) rather than per
/// damage call, because it is a pure function of the asset.
///
/// Returns `0.0` when the two hulls share fewer than three corners — the caller
/// substitutes a characteristic cross-section rather than letting a bond be free.
fn shared_face_area_m2(asset: &FractureAsset, a: &FractureChunk, b: &FractureChunk) -> f64 {
    let extent = {
        let (lo, hi) = (asset.bounds.min, asset.bounds.max);
        let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        (d[0].max(d[1]).max(d[2]) as f64).max(1.0e-3)
    };
    let eps = extent * FACE_PLANE_EPS_FRACTION;
    let eps2 = eps * eps;
    // The corners both hulls carry — the shared face's polygon, unordered.
    let common: Vec<DVec3> = a
        .hull_points
        .iter()
        .map(|p| DVec3::from_array(*p))
        .filter(|p| {
            b.hull_points
                .iter()
                .any(|q| (DVec3::from_array(*q) - *p).length_squared() <= eps2)
        })
        .collect();
    polygon_area_m2(&common)
}

/// The area of the convex polygon through `points`, which are assumed coplanar
/// and in convex position but in **no particular order** — a shared face's corner
/// set as the two hulls happen to list it.
///
/// Fewer than three points bound no area. The plane's normal is taken from the
/// widest available triangle rather than from the first one, because a face's
/// corner list can start with three nearly-collinear points and a normal fitted to
/// those is noise.
fn polygon_area_m2(points: &[DVec3]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let centre = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let u = {
        // The farthest corner from the centre: a stable in-plane axis.
        let mut best = (0.0_f64, DVec3::X);
        for p in points {
            let d = *p - centre;
            let l = d.length();
            if l > best.0 {
                best = (l, d);
            }
        }
        if best.0 <= 0.0 {
            return 0.0;
        }
        best.1 / best.0
    };
    let normal = {
        let mut best = (0.0_f64, DVec3::Y);
        for p in points {
            let c = u.cross(*p - centre);
            let l = c.length();
            if l > best.0 {
                best = (l, c);
            }
        }
        if best.0 <= 0.0 {
            return 0.0; // collinear: no area
        }
        best.1 / best.0
    };
    let v = normal.cross(u);
    // Angle-order around the centre, then sum the shoelace. Ties break by the
    // input index, so the order is a function of the asset alone.
    let mut planar: Vec<(u64, usize, f64, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let d = *p - centre;
            let (x, y) = (d.dot(u), d.dot(v));
            (order_key(y.atan2(x)), i, x, y)
        })
        .collect();
    planar.sort_by_key(|(k, i, _, _)| (*k, *i));
    let mut twice = 0.0;
    for i in 0..planar.len() {
        let (_, _, x0, y0) = planar[i];
        let (_, _, x1, y1) = planar[(i + 1) % planar.len()];
        twice += x0 * y1 - x1 * y0;
    }
    (twice * 0.5).abs()
}

/// Every chunk's **ground-bond** energy in joules — what it costs to break it
/// away from the static geometry it stands on, over its own characteristic
/// cross-section `volume^(2/3)`. See [`FractureState::ground_bonds`] for why the
/// term exists at all.
fn ground_bond_energies(asset: &FractureAsset, d: &Destructible, volume_scale: f64) -> Vec<f64> {
    asset
        .chunks
        .iter()
        .map(|c| {
            let area = (c.volume_m3 * volume_scale).max(0.0).powf(2.0 / 3.0);
            let e = d.bond_force_n(area) * CRACK_OPENING_M;
            if e.is_finite() {
                e.max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// What [`bond_energies`] answers: every bond's energy, plus the pairs whose
/// area it could not measure and had to estimate.
type BondPricing = (BTreeMap<(u32, u32), f64>, BTreeSet<(u32, u32)>);

/// Every bond's breaking **energy** in joules, keyed by the canonical `(a, b)`
/// pair with `a < b`.
///
/// `strength × area × CRACK_OPENING_M`, with the area scaled by the actor's own
/// scale (`volume_scale^(2/3)` — an area scales as the two-thirds power of a
/// volume) so a wall placed at twice the size is four times as hard to break
/// through, which is what a bigger wall is.
///
/// Where the measurement finds no shared face — a bond the adjacency lists that
/// the triangulation does not express — the area falls back to
/// `min(volume)^(2/3)`, a characteristic cross-section of the smaller chunk. That
/// is deliberately generous rather than zero: a bond with zero energy is a bond
/// that breaks for free, which would make one malformed chunk shatter a building.
fn bond_energies(asset: &FractureAsset, d: &Destructible, volume_scale: f64) -> BondPricing {
    let area_scale = volume_scale.max(0.0).powf(2.0 / 3.0);
    let mut out = BTreeMap::new();
    let mut estimated = BTreeSet::new();
    for (a, b) in asset.adjacency_pairs() {
        let (Some(ca), Some(cb)) = (asset.chunks.get(a as usize), asset.chunks.get(b as usize))
        else {
            continue;
        };
        let mut area = shared_face_area_m2(asset, ca, cb);
        if !area.is_finite() || area <= 0.0 {
            area = ca.volume_m3.min(cb.volume_m3).max(0.0).powf(2.0 / 3.0);
            estimated.insert((a, b));
        }
        let energy = d.bond_force_n(area * area_scale) * CRACK_OPENING_M;
        out.insert(
            (a, b),
            if energy.is_finite() {
                energy.max(0.0)
            } else {
                0.0
            },
        );
    }
    (out, estimated)
}

// ─────────────────────────────────────────────────────────────────────────────
// The bridge half
// ─────────────────────────────────────────────────────────────────────────────

impl PhysicsBridge3D {
    /// **THE ONE RING-0 RULE** behind both hosts' `destruct.apply_damage` arms.
    ///
    /// Spends `energy_j` joules breaking bonds on `entity`'s fracture, then runs
    /// the structural solve so whatever that leaves unsupported collapses in the
    /// same step. Returns what it consumed.
    ///
    /// # Whole-step atomicity (the P21.3 law)
    ///
    /// The set of chunks to detach is **decided in full before a single one is
    /// mutated**. The plan is built against a scratch copy of the attached set, so
    /// a chunk's cost is evaluated against the structure as it stood when the
    /// damage arrived, never against a half-collapsed intermediate that depends on
    /// iteration order.
    ///
    /// # Which chunks, and in what order
    ///
    /// **Cheapest-to-liberate first, ties by ascending chunk index.** A chunk's
    /// cost is the sum of the bond energies holding it to its still-attached
    /// neighbours, so a corner chunk with two bonds goes before a core chunk with
    /// six — which is exactly "fracture creates new surface, and the cheapest new
    /// surface goes first". After each pick the remaining costs are recomputed,
    /// because a chunk whose neighbour just left is genuinely cheaper now; that is
    /// what makes a hole *widen* rather than pepper the wall.
    ///
    /// The roadmap phrases this as "nearest-to-impact first". There is no impact
    /// point to be near: `destruct.apply_damage(entity, energy_j)` takes an actor
    /// and an energy, scene schema v20 carries no impact slot, and inventing one
    /// (the actor's origin, say) would order chunks by distance from the middle of
    /// the wall — which detaches the *core* first and looks like nothing at all.
    /// Cheapest-first is the same physics without the invented input, and it is
    /// stated here rather than left to be discovered.
    ///
    /// Leftover energy **dissipates**. It is not banked: a stored remainder would
    /// be per-actor state that survives across calls, so ten taps that each bounce
    /// off would eventually break a wall that a single blow of ten times the
    /// energy also breaks — and nothing in the memo asks for fatigue.
    pub fn runtime_destruct(
        &mut self,
        fractures: &mut BTreeMap<Uuid, FractureState>,
        entity: Uuid,
        permitted: bool,
        energy_j: f64,
    ) -> DamageReport {
        if !energy_j.is_finite() || energy_j <= 0.0 {
            return DamageReport::refused(DestructOutcome::Invalid);
        }
        // **A MATERIAL can be degenerate too, not just an energy.** `strength` is
        // an authored `f64` on a Details-editable component, so `0`, a negative,
        // a NaN and an infinity are all one keystroke away — and a zero strength
        // makes every bond free, so the very smallest representable joule takes a
        // whole building down. (Measured with `f64::MIN_POSITIVE`: the actor
        // vaporised.) Checked here, beside the energy, because both are "the
        // numbers this blow is computed from" and a refusal must not depend on
        // which of them was wrong.
        if fractures.get(&entity).is_some_and(|s| {
            !s.destructible().strength.is_finite() || s.destructible().strength <= 0.0
        }) {
            return DamageReport::refused(DestructOutcome::Invalid);
        }
        // Permission is checked BEFORE the state lookup so a locked actor reads
        // the same whether or not this sim happened to seed it — a refusal must
        // not depend on load order (the `runtime_carve` rule, verbatim).
        if !permitted {
            return DamageReport::refused(DestructOutcome::NotPermitted);
        }
        if !fractures.contains_key(&entity) {
            return DamageReport::refused(DestructOutcome::NoFracture);
        }

        // ── plan (no mutation yet) ──────────────────────────────────────────
        //
        // The ground set is sampled FIRST, against the world as it stands, and
        // then held fixed for the whole plan — the whole-step atomicity law. A
        // chunk standing on static geometry pays its ground bond as well as its
        // neighbour bonds, which is what stops a tap on a tower from taking the
        // base out and dropping the lot (see `FractureState::ground_bonds`).
        let grounded = self.grounded_chunks(fractures, entity);
        let state = &fractures[&entity];
        let n = state.chunks.len();
        // `attached[i]` tracks the SCRATCH structure the plan is judged against.
        let mut attached: Vec<bool> = state
            .chunks
            .iter()
            .map(|c| !c.detached && !c.gone)
            .collect();
        let mut plan: Vec<usize> = Vec::new();
        let mut remaining = energy_j;
        let mut spent = 0.0_f64;
        loop {
            let mut best: Option<(f64, usize)> = None;
            for i in 0..n {
                if !attached[i] {
                    continue;
                }
                let cost = detach_cost(state, &attached, &grounded, i);
                if cost > remaining {
                    continue;
                }
                // Strictly-less keeps the tie-break at "lowest chunk index",
                // which is the deterministic order the whole batch is built on.
                if best.is_none_or(|(bc, _)| cost < bc) {
                    best = Some((cost, i));
                }
            }
            let Some((cost, i)) = best else { break };
            remaining -= cost;
            // Accumulated separately from `energy_j - remaining`: subtracting a
            // running remainder from the input loses the last bit or two, so a
            // blow that broke exactly one 5 kJ bond would report 4999.999999999999
            // absorbed. The sum of what was actually broken is both exact and the
            // honest definition.
            spent += cost;
            attached[i] = false;
            plan.push(i);
        }

        // ── commit ──────────────────────────────────────────────────────────
        let state = fractures.get_mut(&entity).expect("checked above");
        let mut detached = 0_u32;
        for i in plan {
            if state.detach(i) {
                detached += 1;
            }
        }
        // …then let the structure decide what that left hanging, in the same
        // step. Progressive collapse continues over later steps through
        // `step_fractures`; this call is only the first beat of it.
        detached += self.solve_support_for(fractures, entity);

        DamageReport {
            outcome: DestructOutcome::Applied,
            energy_absorbed_j: spent,
            detached,
        }
    }

    /// **A radial impulse.** Push every dynamic body within `radius_m` of
    /// `center` away from it, and report how many were pushed.
    ///
    /// # Impulses, not forces, and why
    ///
    /// An explosion is an *impulse*: a momentum transfer that happens inside one
    /// step, not a load that persists. [`super::PhysicsWorld3D::apply_force`]'s
    /// docs carry the two rapier laws this obeys — a force **persists** until
    /// `reset_forces` (so a "blast force" would keep pushing for ever) and a force
    /// is **not** an impulse of `F · dt` for the position (rapier substeps). An
    /// impulse has neither problem.
    ///
    /// # Units and falloff, stated because they are a contract
    ///
    /// `impulse_ns` is **newton-seconds delivered at one metre**. Beyond that the
    /// magnitude falls as the **inverse square** of the distance — blast momentum
    /// per unit area spreads over a sphere — and inside one metre it is clamped
    /// to the one-metre value, so a body sitting exactly on the charge gets a
    /// large push rather than an infinite one. Bodies past `radius_m` get nothing
    /// at all; the radius is a hard cutoff, not a fade, so the cost is bounded by
    /// what is nearby rather than by the level.
    ///
    /// A body's *velocity* change is `J / m`, which rapier does for free — so a
    /// concrete chunk and a paper crate at the same distance move differently
    /// without a single tuning constant.
    ///
    /// Deterministic: the tracked-entity map is a `BTreeMap` keyed by `Guid`, and
    /// fracture chunks are in it under their synthetic guids, so debris takes the
    /// blast in the same order every run.
    pub fn radial_impulse(&mut self, center: DVec3, impulse_ns: f64, radius_m: f64) -> u32 {
        if !center.is_finite()
            || !impulse_ns.is_finite()
            || !radius_m.is_finite()
            || impulse_ns == 0.0
            || radius_m <= 0.0
        {
            return 0;
        }
        // Plan under an immutable borrow, in Guid order, then apply — the bridge
        // discipline the water pass states.
        let mut plans: Vec<(BodyId3D, DVec3)> = Vec::new();
        for (_, body, kind) in self.tracked_bodies() {
            if kind != BodyKind3D::Dynamic {
                continue;
            }
            let Some(at) = self.world().body_translation(body) else {
                continue;
            };
            let delta = at - center;
            let d = delta.length();
            if d > radius_m {
                continue;
            }
            let dir = if d > 1.0e-9 {
                delta / d
            } else {
                // A body exactly on the charge has no direction to be pushed in.
                // Up is the one answer that is never wrong-looking, and it is
                // deterministic, which is the property that matters.
                DVec3::Y
            };
            let scale = 1.0 / d.max(1.0).powi(2);
            plans.push((body, dir * impulse_ns * scale));
        }
        let hit = plans.len() as u32;
        for (body, impulse) in plans {
            self.world_mut().apply_impulse(body, impulse);
        }
        hit
    }

    /// Whether `entity`'s destructible is still whole. `false` for an entity with
    /// no fracture state at all — nothing broken, but nothing that *can* break
    /// either, and reporting "intact" for a lamp post would be a lie a Blueprint
    /// cannot distinguish from the truth.
    ///
    /// (The distinguishing query is [`fracture_chunk_count`](Self::fracture_chunk_count),
    /// which is `0` exactly when there is no fracture data.)
    pub fn is_intact(fractures: &BTreeMap<Uuid, FractureState>, entity: Uuid) -> bool {
        fractures.get(&entity).is_some_and(|s| s.is_intact())
    }

    /// How many chunks `entity`'s fracture has, or `0` when it has none.
    pub fn fracture_chunk_count(fractures: &BTreeMap<Uuid, FractureState>, entity: Uuid) -> u32 {
        fractures.get(&entity).map(|s| s.chunk_count()).unwrap_or(0)
    }

    /// **The fixed-step fracture advance**: age the debris, run the structural
    /// solve on every tracked actor, apply the budget, and latch `Destroyed`.
    ///
    /// Runs **after** the solver and the pose write-back, so the support test sees
    /// where bodies actually ended the step, and so a collapse decided here
    /// becomes bodies on the *next* sync — one beat of latency that is what makes
    /// progressive collapse progressive rather than instantaneous.
    ///
    /// Returns the audit plus the actors that finished breaking this step, in
    /// `Guid` order; the host fires `Destroyed` on them.
    pub fn step_fractures(
        &mut self,
        fractures: &mut BTreeMap<Uuid, FractureState>,
        dt: f64,
        budget: DebrisBudget,
    ) -> (FractureAudit, Vec<DestroyedEvent>) {
        let mut audit = FractureAudit {
            tracked: fractures.len() as u32,
            ..Default::default()
        };
        if fractures.is_empty() {
            return (audit, Vec::new());
        }
        // 1. Age. Only live debris ages; an attached chunk has no age.
        for state in fractures.values_mut() {
            for c in &mut state.chunks {
                if c.detached && !c.gone {
                    c.age_s += dt;
                }
            }
        }
        // 2. The structural solve, per actor in Guid order.
        let entities: Vec<Uuid> = fractures.keys().copied().collect();
        for entity in &entities {
            audit.collapsed += self.solve_support_for(fractures, *entity);
        }
        // 3. Lifetime, then the level-wide cap. Both are despawn-by-omission:
        //    a reclaimed chunk stops being emitted by `gather_fracture` and by
        //    the render projector on the same generation, so it leaves the world
        //    and the screen together.
        if budget.lifetime_s.is_finite() && budget.lifetime_s > 0.0 {
            for state in fractures.values_mut() {
                let expiring: Vec<usize> = state
                    .chunks
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.detached && !c.gone && c.age_s >= budget.lifetime_s)
                    .map(|(i, _)| i)
                    .collect();
                for i in expiring {
                    if state.reclaim(i) {
                        audit.expired += 1;
                    }
                }
            }
        }
        let mut live: Vec<(u64, Uuid, usize)> = Vec::new();
        for (entity, state) in fractures.iter() {
            for (i, c) in state.chunks.iter().enumerate() {
                if c.detached && !c.gone {
                    live.push((c.detach_order, *entity, i));
                }
            }
        }
        // Oldest detached first; the entity/index tail makes the order total, so
        // two runs reclaim the same chunks even when two actors broke on the same
        // step and minted the same order.
        live.sort();
        let over = live.len().saturating_sub(budget.max_live as usize);
        for (_, entity, i) in live.iter().take(over) {
            if let Some(state) = fractures.get_mut(entity) {
                if state.reclaim(*i) {
                    audit.over_budget += 1;
                }
            }
        }
        audit.live_debris = fractures.values().map(|s| s.live_debris()).sum();

        // 4. The `Destroyed` latch: an actor fires once, the first step on which
        //    every one of its chunks has come off. "Half a wall is gone" is a
        //    state a handler can poll with `destruct.is_intact`; "this actor is
        //    finished" is an edge, and an edge is what an event is for.
        let mut destroyed = Vec::new();
        for (entity, state) in fractures.iter_mut() {
            if state.destroyed_fired {
                continue;
            }
            if state.chunks.iter().all(|c| c.detached || c.gone) {
                state.destroyed_fired = true;
                destroyed.push(DestroyedEvent {
                    entity: *entity,
                    detached: state.chunks.iter().filter(|c| c.detached).count() as u32,
                });
            }
        }
        (audit, destroyed)
    }

    /// Copy the simulated chunk poses back into the sim's own state (P22.3).
    ///
    /// **Into `FractureState`, not into ECS `Transform`s** — there are no entities
    /// to write to. This is the fracture mirror of
    /// [`write_back_into`](PhysicsBridge3D::write_back_into), and the reason it
    /// exists separately is that a chunk is not an actor: it has no name, no
    /// components, and nothing in the scene document refers to it.
    ///
    /// `Guid` order, and only detached (dynamic) chunks — an attached chunk is a
    /// static body that never moved.
    pub fn write_back_fractures(&mut self, fractures: &mut BTreeMap<Uuid, FractureState>) {
        for (entity, state) in fractures.iter_mut() {
            for (i, c) in state.chunks.iter_mut().enumerate() {
                if !c.detached || c.gone {
                    continue;
                }
                let guid = fracture_chunk_guid(*entity, i as u32);
                let Some(body) = self.body_of(guid) else {
                    continue;
                };
                if let (Some(t), Some(r)) = (
                    self.world().body_translation(body),
                    self.world().body_rotation(body),
                ) {
                    c.translation = t;
                    c.rotation = r;
                }
            }
        }
    }

    /// Which of `entity`'s **attached** chunks are directly supported by static
    /// geometry right now.
    ///
    /// The single door both the damage plan and the structural solve go through,
    /// so "what is holding this chunk up" has exactly one answer per step. A
    /// detached or reclaimed chunk is never grounded: debris is what falls, and a
    /// pile of it holding a wall up would make the collapse depend on where the
    /// last chunk happened to bounce.
    ///
    /// # An [`AlwaysLoaded`] actor short-circuits the whole probe
    ///
    /// Every one of its attached chunks is an anchor by the memo's §4.3 override,
    /// so there is nothing to cast — which also means a pinned structure costs no
    /// rays at all.
    ///
    /// **It also makes the actor harder to BREAK, and that is worth stating
    /// because it is a side effect rather than the intent.** The damage plan
    /// charges a grounded chunk its ground bond as well as its neighbour bonds
    /// (see [`FractureState::ground_bonds`]), and "grounded" is what this
    /// function answers — so on an `AlwaysLoaded` actor *every* chunk pays it, and
    /// a wall an author pinned for structural reasons costs roughly twice as much
    /// energy to chip. The alternative (a support set for the solve and a
    /// different one for the plan) would mean two answers to "what is holding this
    /// chunk", which is exactly the split the P22.3 audit found between the plan
    /// and the probe. One answer, stated consequence.
    ///
    /// # The exclusion set is what makes the probe mean anything
    ///
    /// The rays start just under a chunk's own lowest hull vertices, so without
    /// excluding this actor's own colliders — its authored intact one and every
    /// chunk's — the first thing each ray hits is the chunk it started from, and
    /// every chunk reports "supported". That certifies a building standing on
    /// nothing, silently, in exactly the shape the P21.4 "two empty maps agree"
    /// lesson warns about.
    fn grounded_chunks(
        &mut self,
        fractures: &BTreeMap<Uuid, FractureState>,
        entity: Uuid,
    ) -> Vec<bool> {
        let Some(state) = fractures.get(&entity) else {
            return Vec::new();
        };
        let n = state.chunks.len();
        if state.anchored {
            return state
                .chunks
                .iter()
                .map(|c| !c.detached && !c.gone)
                .collect();
        }
        let mut exclude: BTreeSet<ColliderId3D> = BTreeSet::new();
        if let Some(c) = self.collider_of(entity) {
            exclude.insert(c);
        }
        for i in 0..n {
            if let Some(c) = self.collider_of(fracture_chunk_guid(entity, i as u32)) {
                exclude.insert(c);
            }
        }
        let up = {
            let g = self.world().gravity();
            if g.length_squared() > 0.0 {
                -g.normalize()
            } else {
                DVec3::Y
            }
        };
        // Gather the probe geometry under an immutable borrow first: the raycast
        // needs the world mutably (it rebuilds the query BVH lazily).
        let probes: Vec<(usize, Vec<DVec3>)> = (0..n)
            .filter(|&i| {
                let c = &state.chunks[i];
                !c.detached && !c.gone
            })
            .map(|i| (i, support_probe_points(state, i, up)))
            .collect();
        let mut out = vec![false; n];
        for (i, points) in probes {
            out[i] = points.iter().any(|p| {
                self.static_hit_below(
                    *p + up * SUPPORT_SKIN_M,
                    -up,
                    SUPPORT_SKIN_M * 2.0,
                    &exclude,
                )
            });
        }
        out
    }

    /// Run the support solve for one actor and detach whatever it leaves
    /// hanging. Returns how many chunks fell.
    ///
    /// Union by BFS from the supported seeds, over **undetached** neighbours
    /// only, in ascending chunk index — so the traversal order, and therefore the
    /// `detach_order` the budget later sorts on, is a function of the content
    /// alone.
    fn solve_support_for(
        &mut self,
        fractures: &mut BTreeMap<Uuid, FractureState>,
        entity: Uuid,
    ) -> u32 {
        let Some(state) = fractures.get(&entity) else {
            return 0;
        };
        // An untouched actor is not solved at all: nothing has come off, so
        // nothing can be unsupported, and the probe rays would be a per-step cost
        // on every wall in the level for an answer that cannot change.
        if state.is_intact() {
            return 0;
        }
        let n = state.chunks.len();
        let supported = self.grounded_chunks(fractures, entity);
        // Propagate through undetached neighbours.
        let state = &fractures[&entity];
        let mut reached = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if supported[i] {
                reached[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for &nb in &state.asset.chunks[i].neighbors {
                let nb = nb as usize;
                if nb >= n || reached[nb] {
                    continue;
                }
                let c = &state.chunks[nb];
                if c.detached || c.gone {
                    continue;
                }
                reached[nb] = true;
                queue.push_back(nb);
            }
        }
        let falling: Vec<usize> = (0..n)
            .filter(|&i| {
                let c = &state.chunks[i];
                !c.detached && !c.gone && !reached[i]
            })
            .collect();

        let state = fractures.get_mut(&entity).expect("checked above");
        let mut fell = 0;
        for i in falling {
            if state.detach(i) {
                fell += 1;
            }
        }
        fell
    }

    /// Does a ray from `origin` along `dir` hit a **non-dynamic** collider within
    /// `max_toi`, ignoring `exclude`?
    fn static_hit_below(
        &mut self,
        origin: DVec3,
        dir: DVec3,
        max_toi: f64,
        exclude: &BTreeSet<ColliderId3D>,
    ) -> bool {
        // **Dynamic bodies are excluded in the FILTER, not judged afterwards.**
        // Debris resting on debris is not support — a pile of rubble holding a
        // wall up would make the collapse depend on where the last chunk bounced
        // — but expressing that as "cast, then reject a dynamic hit" is a
        // different rule, and a wrong one: the nearest hit under a chunk is
        // whatever happens to be lying there, so one crate inside the 2 cm skin
        // returned "unsupported" for ground that was plainly underneath it. The
        // P22.3 audit reproduced it with a slab at y = +0.01 collapsing an entire
        // four-cube tower. Asking for the nearest NON-dynamic hit is the question
        // that was meant.
        self.world_mut()
            .cast_ray_excluding(origin, dir, max_toi, exclude, true)
            .is_some()
    }
}

/// The energy needed to liberate chunk `i` from the still-`attached` structure:
/// every bond to a still-attached neighbour, plus its ground bond when it is
/// standing on static geometry.
fn detach_cost(state: &FractureState, attached: &[bool], grounded: &[bool], i: usize) -> f64 {
    let mut cost = 0.0;
    for &nb in &state.asset.chunks[i].neighbors {
        let nb = nb as usize;
        if nb >= attached.len() || !attached[nb] {
            continue;
        }
        cost += state.bond_energy_j(i as u32, nb as u32);
    }
    if grounded.get(i).copied().unwrap_or(false) {
        cost += state.ground_bond_j(i);
    }
    cost
}

/// The world points the support test probes from: the [`SUPPORT_PROBES`] lowest
/// hull vertices of chunk `i` along `-up`, ties by vertex index.
///
/// The hull's own vertices rather than its AABB corners: a Voronoi chunk is a
/// wedge as often as a box, and an AABB corner can hang a decimetre off the
/// actual solid — which would report a chunk as supported by ground its geometry
/// never reaches.
fn support_probe_points(state: &FractureState, i: usize, up: DVec3) -> Vec<DVec3> {
    let Some(chunk) = state.asset.chunks.get(i) else {
        return Vec::new();
    };
    let c = &state.chunks[i];
    // The chunk's world transform: the actor's placement while attached (an
    // attached chunk has never moved), so the hull points map straight through.
    let mut pts: Vec<(u64, usize, DVec3)> = chunk
        .hull_points
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let w = state.placement.transform_point3(DVec3::from_array(*p));
            let w = c.rotation * (w - state.chunk_rest_center(i)) + c.translation;
            // Sort key: height along `up`, ordered as bits so the sort is total
            // and stable without a float comparator.
            (order_key(w.dot(up)), k, w)
        })
        .collect();
    pts.sort_by_key(|(h, k, _)| (*h, *k));
    pts.into_iter()
        .take(SUPPORT_PROBES)
        .map(|(_, _, p)| p)
        .collect()
}

/// Map an `f64` onto a `u64` whose unsigned ordering matches the float's total
/// ordering — so a sort key is exact and needs no float comparator.
fn order_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

impl FractureState {
    /// Chunk `i`'s rest-pose world centre — where its body's origin sits before
    /// anything moves it. The hull points are stored relative to this, so a
    /// detached chunk's collider rotates about its own centre of mass.
    fn chunk_rest_center(&self, i: usize) -> DVec3 {
        self.asset
            .chunks
            .get(i)
            .map(|c| {
                self.placement
                    .transform_point3(DVec3::from_array(c.center_of_mass))
            })
            .unwrap_or(DVec3::ZERO)
    }

    /// The collider description for chunk `i` — a [`ColliderShape3D::ConvexHull`]
    /// of the chunk's own hull point set, expressed **relative to its centre of
    /// mass** and already rotated into the actor's world orientation.
    ///
    /// Centre-relative so the body's origin is the chunk's centre of mass, which
    /// is where a tumbling solid wants to rotate about. Pre-rotated because the
    /// body starts at identity: an actor's rotation and scale are baked into the
    /// points once, at seed time, rather than carried as a body rotation that
    /// would then compose with the solver's.
    ///
    /// `None` when the point set cannot bound a volume — the P22.2 producer gate
    /// (`convex_hull_is_buildable`) already refused those at cook, so this is the
    /// belt to that braces.
    pub(crate) fn chunk_collider(
        &self,
        i: usize,
        layers: CollisionLayers,
    ) -> Option<ColliderDesc3D> {
        let chunk = self.asset.chunks.get(i)?;
        let center = self.chunk_rest_center(i);
        let points: Vec<DVec3> = chunk
            .hull_points
            .iter()
            .map(|p| self.placement.transform_point3(DVec3::from_array(*p)) - center)
            .collect();
        if !super::world::convex_hull_is_buildable(&points) {
            return None;
        }
        // Mass comes from the honest density field and the chunk's own volume,
        // scaled by the actor's scale — `Collider3D::density` is deliberately not
        // consulted (it defaults to rapier's 1.0 placeholder, which is not a
        // material density; the P20.2 buoyancy finding).
        Some(
            ColliderDesc3D::new(ColliderShape3D::ConvexHull { points })
                .density(self.destructible.density_kg_m3.max(0.0))
                .layers(layers),
        )
    }

    /// Chunk `i`'s world pose right now.
    pub(crate) fn chunk_pose(&self, i: usize) -> (DVec3, DQuat) {
        let c = &self.chunks[i];
        (c.translation, c.rotation)
    }
}
