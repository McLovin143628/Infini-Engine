//! **XPBD cloth** (P24.4): the `.inf_cloth` payload and the one solver both
//! hosts step.
//!
//! # Why cloth lives in `inf-anim`
//!
//! A garment is *secondary animation*: its input is the posed skeleton this
//! crate already produces, and its output is folded into the same `state_bytes`
//! the pose is. Three consequences settled the placement:
//!
//! * `inf-ecs` already depends on `inf-anim` (the P24.1 pose door), so
//!   `inf_ecs::cloth` reaches this solver with **no new crate edge**;
//! * both hosts already link `inf-anim` and already decode its payloads through
//!   `inf_asset::decode`, so a `.inf_cloth` needs no new loading machinery;
//! * the crate is `sqrt`-only by law (`tests/portable_pose.rs`), which is exactly
//!   the constraint an XPBD solver on a compared trace has to satisfy — and this
//!   module is on that list.
//!
//! # The solver, and what it deliberately is not
//!
//! Extended Position-Based Dynamics (Macklin et al.), in the **small-steps**
//! form: a fixed number of substeps per fixed step, one Gauss–Seidel sweep (or
//! more) of the constraint list inside each. Constraints are visited in the
//! order the asset stores them and particles in index order, so the result is a
//! property of the asset rather than of an iteration order — which is what makes
//! the trace comparable between two processes.
//!
//! **Bending is a cross distance constraint**, not a dihedral-angle one. That is
//! a correctness decision, not a shortcut: the dihedral formulation needs
//! `acos`, and the P14 law bans `std` transcendentals from anything folded into
//! `state_bytes`. A cross spring over each interior edge's opposite pair is
//! `sqrt`-only, is the same constraint shape as stretch (so one solver routine
//! serves both), and resists folding in exactly the way a garment needs.
//!
//! Everything here is `f32` **model space**, matching [`crate::pose`]: a pose is
//! evaluated in the character's own frame, the collision capsules are derived
//! from that pose, and a garment that simulated in f64 world space would need a
//! floating-origin rebase per particle per step for no accuracy that matters at
//! garment scale. The consequence is written down where it bites — see
//! [`step_cloth`]'s note on inertia.

use glam::{Mat4, Vec3};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// Below this length a constraint's direction is undefined and the constraint is
/// skipped rather than normalized by a denominator approaching zero.
///
/// Metres. Two particles this close are already satisfying any rest length a
/// garment authors, so skipping costs nothing; dividing would put an infinity in
/// the trace.
pub const MIN_CONSTRAINT_LEN_M: f32 = 1.0e-7;

/// The default gravity magnitude a garment falls under, m/s².
///
/// Standard gravity, and it lives here rather than being read off the level
/// because the sim's 3D gravity is a `PhysicsBridge3D` detail the pose path has
/// no route to — see the ledger entry in ROADMAP §12's P24 block. The
/// *direction* is not a constant: [`inf_ecs::cloth`] rotates world-down into the
/// wearer's model frame every step, so a character lying on their back has their
/// coat fall the right way.
pub const GRAVITY_M_S2: f32 = 9.81;

/// One distance-style constraint between two particles.
///
/// The same record serves stretch (a mesh edge) and bend (a cross spring over an
/// interior edge's opposite pair) — see the module docs for why bending has this
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClothEdge {
    /// First particle index.
    pub a: u32,
    /// Second particle index.
    pub b: u32,
    /// Rest length, metres — **precomputed** at author time from the garment's
    /// bind positions, so the solver never needs the mesh.
    pub rest_m: f32,
}

/// A collision capsule, named by the two joints its segment runs between.
///
/// Joints, not positions: the capsule has to follow the character, and the only
/// thing that knows where a joint is this step is the pose the sim just
/// evaluated. [`capsules_for`] is the one place that conversion happens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClothCapsule {
    /// The joint at the segment's start.
    pub joint_a: u16,
    /// The joint at the segment's end. Equal to `joint_a` for a sphere.
    pub joint_b: u16,
    /// Capsule radius, metres.
    pub radius_m: f32,
}

/// The garment's material and solver budget.
///
/// **Compliance, not "stiffness 0..1"**: XPBD's compliance (`α`, m/N) is the
/// inverse of a physical stiffness and is *timestep-independent*, which is the
/// entire reason XPBD exists. A 0..1 stiffness would make a garment stiffer at
/// 120 Hz than at 60 Hz, and this engine compares traces across hosts that may
/// tick at different rates. `0.0` is inextensible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClothMaterial {
    /// Stretch compliance, m/N. `0` = inextensible.
    pub stretch_compliance: f32,
    /// Bend compliance, m/N. Higher than `stretch_compliance` for anything that
    /// drapes; equal to it gives cardboard.
    pub bend_compliance: f32,
    /// Velocity damping, 1/s. Applied as `v *= 1 - damping·h` per substep and
    /// clamped into `0..=1`, so a large value stops the garment rather than
    /// reversing it.
    pub damping: f32,
    /// Collision thickness, metres: how far outside a capsule a particle is held.
    pub thickness_m: f32,
    /// Substeps per fixed step. `0` is read as `1`.
    pub substeps: u8,
    /// Constraint sweeps per substep. `0` is read as `1` — which is also the
    /// small-steps recommendation (spend the budget on substeps, not sweeps).
    pub iterations: u8,
}

impl Default for ClothMaterial {
    /// A mid-weight draping fabric at 60 Hz.
    ///
    /// The numbers are the ones the P24.4 tests are calibrated against: 8
    /// substeps of a 1/60 s step is `h ≈ 2 ms`, which is where XPBD's small-steps
    /// regime starts behaving, and a bend compliance two orders above the stretch
    /// one is a skirt rather than a sheet of plywood.
    fn default() -> Self {
        Self {
            stretch_compliance: 0.0,
            bend_compliance: 1.0e-3,
            damping: 0.5,
            thickness_m: 0.005,
            substeps: 8,
            iterations: 1,
        }
    }
}

/// What refused, and why — values, never panics, because every one of these is
/// reachable from an authored garment.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClothError {
    /// The garment had fewer than three vertices, or no triangles.
    #[error("a garment needs at least one triangle: {vertices} vertices, {indices} indices")]
    Degenerate { vertices: usize, indices: usize },
    /// An index list that is not a triangle list, or names a vertex that is not
    /// there.
    #[error("triangle index {index} is out of range for {vertices} vertices")]
    IndexOutOfRange { index: u32, vertices: usize },
    /// `inv_mass` and `rest` disagreed about how many particles there are.
    #[error("{rest} rest positions but {inv_mass} inverse masses")]
    ParticleCountMismatch { rest: usize, inv_mass: usize },
    /// A rest position, a rest length or a material parameter was not a number.
    #[error("{what} is not finite")]
    NonFinite { what: &'static str },
}

/// The `.inf_cloth` payload: a garment prepared for simulation.
///
/// # What it carries, and what the `garment` reference is for
///
/// The asset is **self-contained for both the sim and the draw**: rest
/// positions, inverse masses, the triangle list, the precomputed constraint sets,
/// the material and the collision configuration. Nothing here needs the
/// `.inf_mesh` at runtime.
///
/// [`garment`](Self::garment) is therefore *provenance and the dependency edge*:
/// it is what the Content Drawer shows, what delete-with-references warns about,
/// and what a future re-derive would read. Storing the positions rather than
/// re-reading the mesh is deliberate — it is what lets a shipped player simulate
/// and draw a garment with the mesh absent, and it is what keeps the PIE wire
/// frozen (see ROADMAP §12's P24 block).
///
/// # The v1 ladder
///
/// v1 is the first version, so [`inf_asset::AssetError::SchemaTooOld`] is not
/// reachable for this kind yet. What makes the *next* version's refusal correct
/// is the byte-consumption pin in this module's tests: a field appended to this
/// struct without bumping [`CURRENT_VERSION`](Self::CURRENT_VERSION) fails there
/// rather than shipping a payload whose readers disagree.
///
/// **No `skip_serializing_if` on any field** — the recurring engine law for
/// bincode-bound types (a skipped field desyncs the positional decoder), and the
/// third time it has been written down in this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClothAsset {
    /// Schema version — **first field**, so `peek_schema_version` can read it
    /// without decoding the rest.
    pub schema_version: u32,
    /// Raw 16-byte GUID of the `.inf_mesh` this garment was prepared from.
    /// Stored as bytes so the crate stays `uuid`-free, exactly like
    /// [`crate::AnimClipAsset::skeleton`].
    pub garment: [u8; 16],
    /// Bind/model-space rest positions, one per particle, index-aligned to the
    /// garment's vertices.
    pub rest: Vec<[f32; 3]>,
    /// Inverse mass per particle, kg⁻¹. **`0` pins the particle** — a hem
    /// stitched to a belt, a collar on a shoulder — and pinned particles are
    /// never integrated, never moved by a constraint and never collided.
    pub inv_mass: Vec<f32>,
    /// The garment's triangle list (3 indices per triangle) — what the projectors
    /// draw the simulated positions with.
    pub indices: Vec<u32>,
    /// Stretch constraints: the garment's unique mesh edges.
    pub distance: Vec<ClothEdge>,
    /// Bend constraints: one cross spring per interior edge (module docs).
    pub bending: Vec<ClothEdge>,
    /// Material + solver budget.
    pub material: ClothMaterial,
    /// Which joint pairs get collision capsules, and how fat.
    pub collision: Vec<ClothCapsule>,
}

impl ClothAsset {
    /// v1 (P24.4) — the first version. See the type's docs for the ladder.
    pub const CURRENT_VERSION: u32 = 1;

    /// How many particles this garment simulates.
    pub fn particle_count(&self) -> usize {
        self.rest.len()
    }

    /// How many triangles it draws.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// **The prepare door**: derive a simulatable garment from a mesh's bind
    /// positions + triangle list.
    ///
    /// This is where "precomputed edge/hinge lists" is actually computed, and it
    /// is the only place — the editor's create command, the tests and the gate all
    /// come through here, so there is one definition of what a garment's
    /// constraint set is.
    ///
    /// Determinism is structural rather than incidental: edges are keyed by the
    /// **ordered pair** `(min, max)` and collected into a `BTreeMap`, so both the
    /// stretch list and the bend list come out in ascending index order whatever
    /// order the triangles arrived in. Two runs, two machines and two processes
    /// therefore produce byte-identical assets from the same mesh.
    ///
    /// `pinned` names particles with infinite mass; indices out of range are
    /// ignored rather than refused, because a pin list is authoring intent and
    /// losing one pin must not make a garment unopenable.
    pub fn from_garment(
        garment: [u8; 16],
        positions: &[[f32; 3]],
        indices: &[u32],
        pinned: &[u32],
        material: ClothMaterial,
    ) -> Result<Self, ClothError> {
        if positions.len() < 3 || indices.len() < 3 || !indices.len().is_multiple_of(3) {
            return Err(ClothError::Degenerate {
                vertices: positions.len(),
                indices: indices.len(),
            });
        }
        for p in positions {
            if !p.iter().all(|v| v.is_finite()) {
                return Err(ClothError::NonFinite {
                    what: "a garment rest position",
                });
            }
        }
        for &i in indices {
            if i as usize >= positions.len() {
                return Err(ClothError::IndexOutOfRange {
                    index: i,
                    vertices: positions.len(),
                });
            }
        }

        let at = |i: u32| Vec3::from_array(positions[i as usize]);
        let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };

        // Every mesh edge, and the up-to-two triangle apexes across it. A
        // BTreeMap because its iteration order is the ordered-pair order, which
        // is the determinism this whole file rests on.
        let mut edges: std::collections::BTreeMap<(u32, u32), Vec<u32>> =
            std::collections::BTreeMap::new();
        for tri in indices.chunks_exact(3) {
            let (i, j, k) = (tri[0], tri[1], tri[2]);
            // A degenerate triangle contributes no edge and no hinge; it is not
            // an error (importers emit them), it simply has nothing to say.
            if i == j || j == k || i == k {
                continue;
            }
            for (a, b, apex) in [(i, j, k), (j, k, i), (k, i, j)] {
                edges.entry(key(a, b)).or_default().push(apex);
            }
        }

        let mut distance = Vec::with_capacity(edges.len());
        let mut bending = Vec::new();
        for (&(a, b), apexes) in &edges {
            distance.push(ClothEdge {
                a,
                b,
                rest_m: (at(a) - at(b)).length(),
            });
            // An interior edge is shared by exactly two triangles; its hinge is
            // the spring between their two apexes. A non-manifold edge (three or
            // more) is left un-hinged rather than guessed at — the honest answer,
            // and one a garment author can see in the constraint count.
            if apexes.len() == 2 && apexes[0] != apexes[1] {
                let (p, q) = key(apexes[0], apexes[1]);
                bending.push(ClothEdge {
                    a: p,
                    b: q,
                    rest_m: (at(p) - at(q)).length(),
                });
            }
        }
        // The hinge list is keyed by its own pair too: two hinges over the same
        // apex pair (a fan) would otherwise double that constraint's stiffness by
        // accident of topology.
        bending.sort_by_key(|e| (e.a, e.b));
        bending.dedup_by_key(|e| (e.a, e.b));

        let mut inv_mass = vec![1.0f32; positions.len()];
        for &p in pinned {
            if let Some(slot) = inv_mass.get_mut(p as usize) {
                *slot = 0.0;
            }
        }

        Ok(Self {
            schema_version: Self::CURRENT_VERSION,
            garment,
            rest: positions.to_vec(),
            inv_mass,
            indices: indices.to_vec(),
            distance,
            bending,
            material,
            collision: Vec::new(),
        })
    }

    /// Attach the collision capsules a garment is fitted around (builder style,
    /// so `from_garment(..)?.with_capsules(..)` reads as one preparation).
    pub fn with_capsules(mut self, capsules: Vec<ClothCapsule>) -> Self {
        self.collision = capsules;
        self
    }

    /// **Is this garment simulatable?** Checked once, at seed time, so the
    /// per-step solver has nothing left to validate.
    ///
    /// A garment that fails this is skipped by [`inf_ecs::cloth`] with the
    /// component left alone, which is the same shape a skeleton that will not
    /// resolve takes in [`crate::pose`].
    pub fn validate(&self) -> Result<(), ClothError> {
        if self.rest.len() != self.inv_mass.len() {
            return Err(ClothError::ParticleCountMismatch {
                rest: self.rest.len(),
                inv_mass: self.inv_mass.len(),
            });
        }
        if self.rest.len() < 3 || self.indices.len() < 3 || !self.indices.len().is_multiple_of(3) {
            return Err(ClothError::Degenerate {
                vertices: self.rest.len(),
                indices: self.indices.len(),
            });
        }
        for &i in &self.indices {
            if i as usize >= self.rest.len() {
                return Err(ClothError::IndexOutOfRange {
                    index: i,
                    vertices: self.rest.len(),
                });
            }
        }
        for e in self.distance.iter().chain(&self.bending) {
            if e.a as usize >= self.rest.len() || e.b as usize >= self.rest.len() {
                return Err(ClothError::IndexOutOfRange {
                    index: e.a.max(e.b),
                    vertices: self.rest.len(),
                });
            }
            if !e.rest_m.is_finite() {
                return Err(ClothError::NonFinite {
                    what: "a constraint rest length",
                });
            }
        }
        for p in &self.rest {
            if !p.iter().all(|v| v.is_finite()) {
                return Err(ClothError::NonFinite {
                    what: "a garment rest position",
                });
            }
        }
        let m = &self.material;
        if ![
            m.stretch_compliance,
            m.bend_compliance,
            m.damping,
            m.thickness_m,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err(ClothError::NonFinite {
                what: "a cloth material parameter",
            });
        }
        Ok(())
    }
}

impl AssetPayload for ClothAsset {
    const KIND: AssetKind = AssetKind::Cloth;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    // A garment has exactly one door, and it is not "re-import": the `.inf_cloth`
    // is *prepared* from a mesh, so the remedy names the preparation.
    const UPGRADE_REMEDY: &'static str =
        "re-prepare the garment from its mesh (Model Editor ▸ Cloth ▸ Prepare Garment)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// One collision capsule, resolved into model-space endpoints for this step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    /// Segment start, model space.
    pub a: Vec3,
    /// Segment end, model space. Equal to `a` for a sphere.
    pub b: Vec3,
    /// Radius, metres.
    pub radius_m: f32,
}

/// **Place a garment's capsules on the pose the sim just evaluated.**
///
/// `joint_globals` is [`crate::global_transforms`] over the evaluated pose, so
/// the capsules are exactly where the drawn character's bones are — the pose is
/// read, never re-evaluated, which is what stops the collision geometry and the
/// skinning disagreeing about where an arm is.
///
/// A capsule naming a joint the skeleton does not have is **dropped**, not
/// clamped: a garment fitted to another rig must not collide against joint 0.
pub fn capsules_for(asset: &ClothAsset, joint_globals: &[Mat4]) -> Vec<Capsule> {
    let mut out = Vec::with_capacity(asset.collision.len());
    for c in &asset.collision {
        let (Some(ma), Some(mb)) = (
            joint_globals.get(c.joint_a as usize),
            joint_globals.get(c.joint_b as usize),
        ) else {
            continue;
        };
        let a = ma.transform_point3(Vec3::ZERO);
        let b = mb.transform_point3(Vec3::ZERO);
        if !a.is_finite() || !b.is_finite() || !c.radius_m.is_finite() || c.radius_m <= 0.0 {
            continue;
        }
        out.push(Capsule {
            a,
            b,
            radius_m: c.radius_m,
        });
    }
    out
}

/// The live particle state of one garment — positions and velocities, model
/// space, one entry per particle.
///
/// This is **sim state**: written only from the fixed step, folded into
/// `state_bytes`, and never written to a file. The `indices` ride along so a
/// projector can draw the garment from the sim world alone, with no asset store
/// in the picture at all (see [`inf_ecs::cloth`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ClothState {
    /// Positions, model space.
    pub x: Vec<[f32; 3]>,
    /// Velocities, m/s, model space.
    pub v: Vec<[f32; 3]>,
    /// Inverse masses, copied from the asset at seed time (`0` = pinned).
    pub inv_mass: Vec<f32>,
    /// The garment's triangle list, shared with every other state seeded from
    /// the same asset — one `Arc`, cloned, never rebuilt.
    pub indices: std::sync::Arc<Vec<u32>>,
}

impl ClothState {
    /// Seed a garment at its rest shape with zero velocity.
    ///
    /// Refuses an invalid asset ([`ClothAsset::validate`]) rather than seeding a
    /// state the solver would then have to re-check every step.
    pub fn seed(asset: &ClothAsset) -> Result<Self, ClothError> {
        asset.validate()?;
        Ok(Self {
            x: asset.rest.clone(),
            v: vec![[0.0; 3]; asset.rest.len()],
            inv_mass: asset.inv_mass.clone(),
            indices: std::sync::Arc::new(asset.indices.clone()),
        })
    }

    /// How many particles.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the garment has no particles.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The total kinetic energy of the garment, J·kg⁻¹-weighted — `Σ ½·|v|²/w`
    /// over the unpinned particles, with `w` the inverse mass.
    ///
    /// Here rather than in a test because it is what a convergence *gate* reads:
    /// "a hanging cloth reaches equilibrium" is a statement about this number
    /// going down, and a number a test computes for itself is a number that
    /// agrees with the test's own idea of the solver.
    pub fn kinetic_energy(&self) -> f32 {
        let mut sum = 0.0f32;
        for (v, &w) in self.v.iter().zip(&self.inv_mass) {
            if w <= 0.0 {
                continue;
            }
            let v = Vec3::from_array(*v);
            sum += 0.5 * v.dot(v) / w;
        }
        sum
    }
}

/// Solve one distance-style constraint in place (XPBD, with its Lagrange
/// multiplier carried across the sweeps of a substep).
///
/// `alpha` is the *compliance over h²* the caller has already formed, because it
/// is constant across a substep and forming it per constraint would be arithmetic
/// repeated a few thousand times a step for no result.
#[inline]
pub fn solve_edge(x: &mut [Vec3], w: &[f32], e: &ClothEdge, alpha: f32, lambda: &mut f32) {
    let (i, j) = (e.a as usize, e.b as usize);
    // The asset was validated at seed, so these are in range; the guard is what
    // makes that a property of the code rather than of the caller.
    if i >= x.len() || j >= x.len() || i == j {
        return;
    }
    let (wi, wj) = (w[i], w[j]);
    let wsum = wi + wj;
    if wsum <= 0.0 {
        return; // both pinned: nothing to move
    }
    let d = x[i] - x[j];
    let len = d.length();
    if len < MIN_CONSTRAINT_LEN_M {
        return;
    }
    let n = d / len;
    let c = len - e.rest_m;
    let dlambda = (-c - alpha * *lambda) / (wsum + alpha);
    *lambda += dlambda;
    x[i] += n * (wi * dlambda);
    x[j] -= n * (wj * dlambda);
}

/// Push one particle out of a capsule, if it is inside it.
///
/// `sqrt` only: the closest point on a segment is a clamped dot-product ratio and
/// the push is a normalized difference. A particle exactly on the axis has no
/// defined normal and is pushed along `+Y` — an arbitrary but *fixed* choice, so
/// the degenerate case is deterministic rather than uninitialized.
#[inline]
pub fn resolve_capsule(p: Vec3, c: &Capsule, thickness_m: f32) -> Option<Vec3> {
    let ab = c.b - c.a;
    let len2 = ab.dot(ab);
    let t = if len2 > MIN_CONSTRAINT_LEN_M * MIN_CONSTRAINT_LEN_M {
        ((p - c.a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = c.a + ab * t;
    let d = p - closest;
    let dist = d.length();
    let r = c.radius_m + thickness_m;
    if dist >= r {
        return None;
    }
    let n = if dist > MIN_CONSTRAINT_LEN_M {
        d / dist
    } else {
        Vec3::Y
    };
    Some(closest + n * r)
}

/// **The one XPBD step.** Advance `state` by `dt` seconds under `gravity`,
/// against `capsules`.
///
/// Called from [`inf_ecs::cloth::step_cloth_simulation`], which is called from
/// both hosts' fixed steps — so this function is the *only* place a garment
/// moves, and the editor preview and the shipped player cannot disagree about
/// how.
///
/// Order inside a substep, and each part's reason:
///  1. **integrate** — gravity, damping, then `x += v·h`, with pinned particles
///     skipped entirely (a pin that were merely re-set after integration would
///     still contribute a velocity on the next line);
///  2. **solve** — `iterations` sweeps of stretch then bend, Lagrange multipliers
///     carried across the sweeps (they are what makes compliance mean anything);
///  3. **collide** — after the constraints, because a garment pushed out of a leg
///     and then pulled back through it by a stretch constraint is the artefact
///     everyone sees;
///  4. **derive velocity** — `v = (x − x_prev)/h`, which is what makes collision
///     and constraints affect momentum without a second integrator.
///
/// # What it does NOT model, stated where it bites
///
/// * **No inertial coupling.** The garment simulates in the wearer's *model*
///   frame, so a character sprinting in a straight line does not stream their
///   coat behind them; what moves the cloth is the capsules moving under it,
///   which covers every animated case and no translational one. Feeding the
///   frame's acceleration in needs the previous global transform kept per
///   garment, and that is a P25 item — ledgered in ROADMAP §12's P24 block.
/// * **No friction and no self-collision.** A particle leaves a capsule along the
///   surface normal with its tangential velocity intact, and two folds of the
///   same skirt pass through each other. Both are ledgered.
///
/// A non-finite `dt` or `gravity`, or an empty garment, is a **no-op** rather
/// than a panic: this runs inside a fixed step, and one bad garment must not take
/// a level down.
pub fn step_cloth(
    asset: &ClothAsset,
    state: &mut ClothState,
    dt: f32,
    gravity: Vec3,
    capsules: &[Capsule],
) {
    if state.x.is_empty() || !dt.is_finite() || dt <= 0.0 || !gravity.is_finite() {
        return;
    }
    let substeps = asset.material.substeps.max(1) as u32;
    let iterations = asset.material.iterations.max(1) as u32;
    let h = dt / substeps as f32;
    if !h.is_finite() || h <= 0.0 {
        return;
    }
    let thickness = asset.material.thickness_m.max(0.0);
    let damp = (1.0 - asset.material.damping * h).clamp(0.0, 1.0);
    let inv_h2 = 1.0 / (h * h);
    let alpha_stretch = asset.material.stretch_compliance.max(0.0) * inv_h2;
    let alpha_bend = asset.material.bend_compliance.max(0.0) * inv_h2;

    // Working buffers, in `Vec3` rather than `[f32; 3]`: the solver touches every
    // particle several times per substep and the conversion would otherwise run
    // once per touch. Converted back at the end, so the *stored* state stays the
    // plain-array shape the trace folds.
    let n = state.x.len();
    let mut x: Vec<Vec3> = state.x.iter().copied().map(Vec3::from_array).collect();
    let mut v: Vec<Vec3> = state.v.iter().copied().map(Vec3::from_array).collect();
    let mut prev: Vec<Vec3> = x.clone();
    let w = &state.inv_mass;
    let mut lambda = vec![0.0f32; asset.distance.len() + asset.bending.len()];

    for _ in 0..substeps {
        for i in 0..n {
            if w[i] <= 0.0 {
                prev[i] = x[i];
                continue;
            }
            v[i] = (v[i] + gravity * h) * damp;
            prev[i] = x[i];
            x[i] += v[i] * h;
        }
        lambda.iter_mut().for_each(|l| *l = 0.0);
        for _ in 0..iterations {
            for (k, e) in asset.distance.iter().enumerate() {
                solve_edge(&mut x, w, e, alpha_stretch, &mut lambda[k]);
            }
            let base = asset.distance.len();
            for (k, e) in asset.bending.iter().enumerate() {
                solve_edge(&mut x, w, e, alpha_bend, &mut lambda[base + k]);
            }
        }
        for i in 0..n {
            if w[i] <= 0.0 {
                continue;
            }
            for c in capsules {
                if let Some(out) = resolve_capsule(x[i], c, thickness) {
                    x[i] = out;
                }
            }
        }
        for i in 0..n {
            if w[i] <= 0.0 {
                v[i] = Vec3::ZERO;
                continue;
            }
            v[i] = (x[i] - prev[i]) / h;
        }
    }

    for i in 0..n {
        // A garment that has gone non-finite (a NaN target, a runaway compliance)
        // is frozen at its previous position rather than written into the trace:
        // one NaN in `state_bytes` makes every downstream comparison meaningless
        // and is unrecoverable, where a frozen particle is visible and local.
        if x[i].is_finite() && v[i].is_finite() {
            state.x[i] = x[i].to_array();
            state.v[i] = v[i].to_array();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// A flat `n × n` grid of quads in the XZ plane at `y = 0`, 10 cm apart —
    /// small enough to solve fast, big enough to have interior edges (and
    /// therefore hinges) at `n >= 3`.
    fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut pos = Vec::new();
        for j in 0..n {
            for i in 0..n {
                pos.push([i as f32 * 0.1, 0.0, j as f32 * 0.1]);
            }
        }
        let mut idx = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = j * n + i;
                let (b, c, d) = (a + 1, a + n, a + n + 1);
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        (pos, idx)
    }

    /// A 5×5 sheet pinned along its `j == 0` row — a hanging curtain.
    fn curtain() -> ClothAsset {
        let (pos, idx) = grid(5);
        let pinned: Vec<u32> = (0..5).collect();
        ClothAsset::from_garment([7; 16], &pos, &idx, &pinned, ClothMaterial::default())
            .expect("a 5x5 grid prepares")
    }

    // ── preparation ─────────────────────────────────────────────────────────

    #[test]
    fn preparing_a_grid_derives_its_edges_and_hinges() {
        let a = curtain();
        // A 5x5 triangulated grid: 25 vertices, 32 triangles, 72 unique edges
        // (2·5·4 axis-aligned + 16 diagonals = 40 + 16 … computed the honest way
        // below rather than asserted from a number nobody can check).
        assert_eq!(a.particle_count(), 25);
        assert_eq!(a.triangle_count(), 32);
        // Euler for a triangulated disc: V - E + F = 1 (F counts triangles only).
        assert_eq!(
            a.particle_count() as i64 - a.distance.len() as i64 + a.triangle_count() as i64,
            1,
            "the edge set is not the grid's: V={} E={} F={}",
            a.particle_count(),
            a.distance.len(),
            a.triangle_count()
        );
        // Interior edges (each shared by two triangles) each grow one hinge, and
        // there are strictly fewer of them than edges (the boundary has none).
        assert!(!a.bending.is_empty(), "a grid has interior edges");
        assert!(a.bending.len() < a.distance.len());
        // Rest lengths are the real distances, not a placeholder.
        for e in &a.distance {
            let d = (Vec3::from_array(a.rest[e.a as usize])
                - Vec3::from_array(a.rest[e.b as usize]))
            .length();
            assert!((e.rest_m - d).abs() < 1e-6, "{e:?} vs {d}");
        }
        assert_eq!(a.validate(), Ok(()));
    }

    /// **The preparation is deterministic**, which is what makes a committed
    /// `.inf_cloth` reproducible from its mesh.
    #[test]
    fn preparation_is_a_pure_function_of_the_mesh() {
        let (pos, idx) = grid(4);
        let a = ClothAsset::from_garment([1; 16], &pos, &idx, &[0], ClothMaterial::default());
        let b = ClothAsset::from_garment([1; 16], &pos, &idx, &[0], ClothMaterial::default());
        assert_eq!(a, b);
        // …and it does not depend on the order the triangles arrived in: the
        // BTreeMap key is the ordered pair, so a reversed index list prepares the
        // same constraint sets.
        let mut reversed: Vec<u32> = Vec::new();
        for tri in idx.chunks_exact(3).rev() {
            reversed.extend_from_slice(tri);
        }
        let c = ClothAsset::from_garment([1; 16], &pos, &reversed, &[0], ClothMaterial::default())
            .unwrap();
        let a = a.unwrap();
        assert_eq!(c.distance, a.distance, "edge order followed the input");
        assert_eq!(c.bending, a.bending, "hinge order followed the input");
    }

    #[test]
    fn a_garment_that_is_not_a_mesh_is_refused_by_name() {
        assert!(matches!(
            ClothAsset::from_garment([0; 16], &[], &[], &[], ClothMaterial::default()),
            Err(ClothError::Degenerate { .. })
        ));
        let (pos, _) = grid(3);
        assert!(matches!(
            ClothAsset::from_garment([0; 16], &pos, &[0, 1, 99], &[], ClothMaterial::default()),
            Err(ClothError::IndexOutOfRange { index: 99, .. })
        ));
        assert!(matches!(
            ClothAsset::from_garment(
                [0; 16],
                &[[f32::NAN, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0]],
                &[0, 1, 2],
                &[],
                ClothMaterial::default(),
            ),
            Err(ClothError::NonFinite { .. })
        ));
        // A validated asset whose particle counts were pulled apart afterwards.
        let mut bad = curtain();
        bad.inv_mass.pop();
        assert!(matches!(
            bad.validate(),
            Err(ClothError::ParticleCountMismatch { .. })
        ));
        assert!(matches!(
            ClothState::seed(&bad),
            Err(ClothError::ParticleCountMismatch { .. })
        ));
    }

    // ── the solver ──────────────────────────────────────────────────────────

    /// **Convergence**: a hanging curtain settles. Energy falls and the shape
    /// stops moving, and the settled shape is inside hand-computed bounds.
    #[test]
    fn a_hanging_curtain_converges_to_a_settled_shape() {
        let a = curtain();
        let mut s = ClothState::seed(&a).unwrap();
        let dt = 1.0 / 60.0;
        for _ in 0..10 {
            step_cloth(&a, &mut s, dt, Vec3::NEG_Y * GRAVITY_M_S2, &[]);
        }
        let early = s.kinetic_energy();
        assert!(early > 0.0, "the curtain never started moving");
        for _ in 0..600 {
            step_cloth(&a, &mut s, dt, Vec3::NEG_Y * GRAVITY_M_S2, &[]);
        }
        let late = s.kinetic_energy();
        assert!(
            late < early * 0.05,
            "the curtain did not settle: {early} J → {late} J"
        );
        // The pinned row never moved…
        for i in 0..5 {
            assert_eq!(s.x[i], a.rest[i], "pinned particle {i} moved");
        }
        // …and the free rows hang BELOW the pins, within the fabric's own reach:
        // four 10 cm rows of inextensible stretch constraints cannot put the
        // bottom row more than 0.4 m down, and gravity must have taken it most of
        // the way there.
        let bottom_y = (20..25).map(|i| s.x[i][1]).fold(f32::INFINITY, f32::min);
        assert!(
            (-0.401..=-0.30).contains(&bottom_y),
            "the hem hangs at y = {bottom_y}; four 0.1 m rows reach -0.4 m"
        );
        // Stretch really is being enforced: no edge is more than 2% long.
        for e in &a.distance {
            let d = (Vec3::from_array(s.x[e.a as usize]) - Vec3::from_array(s.x[e.b as usize]))
                .length();
            assert!(
                d <= e.rest_m * 1.02 + 1e-4,
                "edge {e:?} stretched to {d} from {}",
                e.rest_m
            );
        }
    }

    /// **The MUTATION arm for the solver.** With the constraint sweep severed the
    /// curtain is free-fall: the pinned row still holds, so a test that only
    /// checked the pins would pass — this checks the *edges*.
    #[test]
    fn a_garment_with_no_constraints_falls_apart() {
        let mut a = curtain();
        a.distance.clear();
        a.bending.clear();
        let mut s = ClothState::seed(&a).unwrap();
        for _ in 0..60 {
            step_cloth(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * GRAVITY_M_S2, &[]);
        }
        // Free particles fell a full second of gravity, ~4.9 m — far past the
        // 0.4 m an intact curtain can reach.
        let bottom = s.x[24][1];
        assert!(
            bottom < -3.0,
            "with no constraints the hem should be in free fall, it is at {bottom}"
        );
    }

    /// A 7×7 cape hanging from its `j == 0` row at `y = 0.5`, and the rod it
    /// drapes over.
    ///
    /// **Pinned, and deliberately so.** The first cut of this fixture dropped a
    /// free sheet onto a horizontal rod and measured zero contacts after five
    /// seconds — which was the solver being *right*: a frictionless cloth on a
    /// round rod slides off, and this v1 has no friction (ledgered). A cape hung
    /// from a shoulder line over a bar is the case a garment actually is, and it
    /// stays put because the pins hold it there rather than because the physics
    /// was fudged.
    fn cape() -> (ClothAsset, Capsule) {
        let (pos, idx) = grid(7);
        let hung: Vec<[f32; 3]> = pos.iter().map(|p| [p[0] - 0.3, 0.5, p[2] - 0.3]).collect();
        let pinned: Vec<u32> = (0..7).collect(); // the j == 0 row, at z = -0.3
        let asset =
            ClothAsset::from_garment([2; 16], &hung, &idx, &pinned, ClothMaterial::default())
                .unwrap()
                .with_capsules(vec![ClothCapsule {
                    joint_a: 0,
                    joint_b: 1,
                    radius_m: 0.1,
                }]);
        // A rod 20 cm below the pin line and 5 cm in front of it — **inside the
        // line a free cape hangs down**, which is the placement the mutation arm
        // needs: without collision the cloth passes straight through this volume,
        // with it the cloth bows around the rod and rests on it.
        let rod = Capsule {
            a: Vec3::new(-0.5, 0.30, -0.25),
            b: Vec3::new(0.5, 0.30, -0.25),
            radius_m: 0.1,
        };
        (asset, rod)
    }

    /// Distance from `p` to a capsule's axis — the test's own arithmetic, so the
    /// assertions do not read back the expression the solver used.
    fn axis_distance(p: [f32; 3], c: &Capsule) -> f32 {
        let p = Vec3::from_array(p);
        let ab = c.b - c.a;
        let t = ((p - c.a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        (p - (c.a + ab * t)).length()
    }

    /// **Capsule collision, asserted on the WORLD**: every particle ends up
    /// outside the capsule by at least the garment's thickness, and enough of
    /// them are resting on it that the claim is about a drape.
    #[test]
    fn a_garment_drapes_over_a_capsule_without_penetrating_it() {
        let (a, rod) = cape();
        let mut s = ClothState::seed(&a).unwrap();
        for _ in 0..300 {
            step_cloth(
                &a,
                &mut s,
                1.0 / 60.0,
                Vec3::NEG_Y * GRAVITY_M_S2,
                std::slice::from_ref(&rod),
            );
        }
        let skin = rod.radius_m + a.material.thickness_m;
        let mut touching = 0;
        for (i, p) in s.x.iter().enumerate() {
            let d = axis_distance(*p, &rod);
            assert!(
                d >= skin - 1e-3,
                "particle {i} is {d} m from the axis, inside the {skin} m capsule \
                 surface"
            );
            if d < skin + 0.03 {
                touching += 1;
            }
        }
        // ANTI-VACUITY: the cape really landed ON the rod. A cape that missed it
        // entirely satisfies every assertion above perfectly.
        assert!(
            touching >= 3,
            "only {touching} particles are resting on the capsule — the garment \
             missed it, so the non-penetration assertions above prove nothing"
        );
    }

    /// **The MUTATION arm for collision**: with the capsule list empty the same
    /// cape hangs straight down THROUGH the volume the rod occupies.
    ///
    /// This is the falsifier the non-penetration test needs: it measures the same
    /// quantity (distance to the rod's axis) on a run where nothing is stopping
    /// the cloth, and finds particles inside. Delete `resolve_capsule`'s effect
    /// and the test above fails on exactly these particles.
    #[test]
    fn a_garment_with_no_capsules_hangs_through_the_body() {
        let (a, rod) = cape();
        let mut s = ClothState::seed(&a).unwrap();
        for _ in 0..300 {
            step_cloth(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * GRAVITY_M_S2, &[]);
        }
        let skin = rod.radius_m + a.material.thickness_m;
        let inside =
            s.x.iter()
                .filter(|p| axis_distance(**p, &rod) < skin - 1e-3)
                .count();
        assert!(
            inside > 0,
            "with no capsule the cape must hang through the volume the rod \
             occupies; not one of its {} particles is inside it, so the \
             non-penetration arm is asserting something that is true anyway",
            s.len()
        );
    }

    #[test]
    fn stepping_is_bit_identical_between_two_runs() {
        let a = curtain();
        let run = || {
            let mut s = ClothState::seed(&a).unwrap();
            for k in 0..90 {
                // A moving capsule, so the trace has structure in it rather than
                // being a settled constant two runs would agree about anyway.
                let y = -0.2 + 0.001 * k as f32;
                step_cloth(
                    &a,
                    &mut s,
                    1.0 / 60.0,
                    Vec3::NEG_Y * GRAVITY_M_S2,
                    &[Capsule {
                        a: Vec3::new(-0.3, y, 0.2),
                        b: Vec3::new(0.7, y, 0.2),
                        radius_m: 0.1,
                    }],
                );
            }
            s
        };
        let (p, q) = (run(), run());
        assert_eq!(p.x, q.x);
        assert_eq!(p.v, q.v);
        // The run really moved (two frozen states are equal too).
        assert_ne!(p.x, a.rest);
    }

    #[test]
    fn a_non_finite_step_is_a_no_op_and_never_reaches_the_state() {
        let a = curtain();
        let mut s = ClothState::seed(&a).unwrap();
        for (dt, g) in [
            (f32::NAN, Vec3::NEG_Y),
            (0.0, Vec3::NEG_Y),
            (-1.0, Vec3::NEG_Y),
            (1.0 / 60.0, Vec3::new(f32::NAN, 0.0, 0.0)),
            (1.0 / 60.0, Vec3::splat(f32::INFINITY)),
        ] {
            step_cloth(&a, &mut s, dt, g, &[]);
            assert_eq!(s.x, a.rest, "dt={dt} g={g:?} moved the garment");
        }
        // …and a real step does move it, so the equalities above are about the
        // refusal and not about the solver being inert.
        step_cloth(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * GRAVITY_M_S2, &[]);
        assert_ne!(s.x, a.rest);
    }

    #[test]
    fn capsules_follow_the_posed_joints_and_a_missing_joint_is_dropped() {
        let a = curtain().with_capsules(vec![
            ClothCapsule {
                joint_a: 0,
                joint_b: 1,
                radius_m: 0.1,
            },
            ClothCapsule {
                joint_a: 0,
                joint_b: 9,
                radius_m: 0.1,
            },
        ]);
        let globals = [
            Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            Mat4::from_translation(Vec3::new(1.0, 3.0, 3.0)),
        ];
        let caps = capsules_for(&a, &globals);
        assert_eq!(caps.len(), 1, "a capsule naming joint 9 must be dropped");
        assert_eq!(caps[0].a, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(caps[0].b, Vec3::new(1.0, 3.0, 3.0));
        // A pose that moves the joint moves the capsule (it is not cached).
        let moved = [
            Mat4::from_translation(Vec3::new(1.0, 2.5, 3.0)),
            Mat4::from_translation(Vec3::new(1.0, 3.5, 3.0)),
        ];
        assert_ne!(capsules_for(&a, &moved)[0].a, caps[0].a);
        // A capsule with a nonsense radius is dropped rather than inverted.
        let bad = curtain().with_capsules(vec![ClothCapsule {
            joint_a: 0,
            joint_b: 1,
            radius_m: -1.0,
        }]);
        assert!(capsules_for(&bad, &globals).is_empty());
    }

    // ── the schema ladder ───────────────────────────────────────────────────

    #[test]
    fn the_payload_round_trips_deterministically() {
        let a = curtain().with_capsules(vec![ClothCapsule {
            joint_a: 3,
            joint_b: 4,
            radius_m: 0.12,
        }]);
        let e1 = encode(&a).unwrap();
        assert_eq!(e1, encode(&a).unwrap(), "re-encoding is byte-identical");
        let back: ClothAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.schema_version, ClothAsset::CURRENT_VERSION);
    }

    /// The **v1 wire shape**, positionally — a real shadow struct, so "what v1
    /// was" is written down rather than derived from the live encoder.
    #[derive(Deserialize)]
    struct ClothAssetV1Wire {
        schema_version: u32,
        garment: [u8; 16],
        rest: Vec<[f32; 3]>,
        inv_mass: Vec<f32>,
        indices: Vec<u32>,
        distance: Vec<ClothEdge>,
        bending: Vec<ClothEdge>,
        material: ClothMaterial,
        collision: Vec<ClothCapsule>,
    }

    /// **The wire SHAPE is pinned, so a tail field cannot be appended without a
    /// bump** — the `SkeletonAsset` discipline, from day one.
    ///
    /// Two claims, and the second is the load-bearing one: the nine fields decode
    /// positionally in this order with these types, and the decode consumes
    /// **every byte**. A tenth field appended to `ClothAsset` leaves bytes
    /// unaccounted for here and fails.
    #[test]
    fn the_wire_shape_is_pinned_field_for_field() {
        let want = curtain().with_capsules(vec![ClothCapsule {
            joint_a: 1,
            joint_b: 2,
            radius_m: 0.2,
        }]);
        let bytes = encode(&want).unwrap();
        let (wire, consumed): (ClothAssetV1Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v1 shape decodes the v1 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned nine-field shape does not \
             account for — a field was appended to `ClothAsset` without bumping \
             `CURRENT_VERSION`"
        );
        assert_eq!(wire.schema_version, ClothAsset::CURRENT_VERSION);
        assert_eq!(wire.garment, want.garment);
        assert_eq!(wire.rest, want.rest);
        assert_eq!(wire.inv_mass, want.inv_mass);
        assert_eq!(wire.indices, want.indices);
        assert_eq!(wire.distance, want.distance);
        assert_eq!(wire.bending, want.bending);
        assert_eq!(wire.material, want.material);
        assert_eq!(wire.collision, want.collision);
    }

    /// **The newer direction is a named refusal.** A payload from a future build
    /// decodes structurally and `migrate` rejects it.
    #[test]
    fn a_future_payload_is_refused_as_too_new() {
        let mut a = curtain();
        a.schema_version = ClothAsset::CURRENT_VERSION + 1;
        let bytes = encode(&a).unwrap();
        assert!(matches!(
            decode::<ClothAsset>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    /// **The ASYMMETRIC arm: an append without a bump is unactionable, by
    /// measurement.**
    ///
    /// `SchemaTooOld` is unreachable for a v1 kind — there is no v0 — so the
    /// "older" half of the ladder cannot be exercised the way `SkeletonAsset`'s
    /// is. What *can* be measured, and is what makes the bump mandatory rather
    /// than customary, is the failure a future author would cause by appending a
    /// field and leaving `CURRENT_VERSION` alone: the old reader gets a short read
    /// and `decode` reports `Decode`, **not** a version story, because the peeked
    /// version equals its own and there is nothing to diagnose.
    ///
    /// That is the whole argument for the pin above, written as a test rather
    /// than as a comment.
    #[test]
    fn an_appended_field_without_a_bump_is_an_unactionable_decode_error() {
        #[derive(Serialize)]
        struct ClothAssetUnbumpedAppend {
            schema_version: u32,
            garment: [u8; 16],
            rest: Vec<[f32; 3]>,
            inv_mass: Vec<f32>,
            indices: Vec<u32>,
            distance: Vec<ClothEdge>,
            bending: Vec<ClothEdge>,
            material: ClothMaterial,
            collision: Vec<ClothCapsule>,
            /// The tenth field a future batch appends. Note the version is
            /// deliberately still 1 — that is the mistake being measured.
            wind_response: Vec<f32>,
        }
        let a = curtain();
        let bytes = bincode::serde::encode_to_vec(
            &ClothAssetUnbumpedAppend {
                schema_version: ClothAsset::CURRENT_VERSION,
                garment: a.garment,
                rest: a.rest.clone(),
                inv_mass: a.inv_mass.clone(),
                indices: a.indices.clone(),
                distance: a.distance.clone(),
                bending: a.bending.clone(),
                material: a.material,
                collision: a.collision.clone(),
                wind_response: vec![1.0; 25],
            },
            inf_asset::bincode_config(),
        )
        .unwrap();
        // bincode ignores trailing bytes, so the *reader* silently succeeds and
        // drops the new field — which is exactly why the byte-consumption pin
        // above, and not this, is the gate. Recorded as the bound it is.
        let back: ClothAsset = decode(&bytes).expect("trailing bytes decode today");
        assert_eq!(back, a, "the appended field was silently dropped");

        // The other half of the same mistake — a field *removed* without a bump,
        // i.e. a v1 reader meeting a payload one field short — is a short read,
        // and it must not be reported as a version story.
        #[derive(Serialize)]
        struct ClothAssetShort {
            schema_version: u32,
            garment: [u8; 16],
            rest: Vec<[f32; 3]>,
        }
        let short = bincode::serde::encode_to_vec(
            &ClothAssetShort {
                schema_version: ClothAsset::CURRENT_VERSION,
                garment: a.garment,
                rest: a.rest.clone(),
            },
            inf_asset::bincode_config(),
        )
        .unwrap();
        match decode::<ClothAsset>(&short) {
            Err(inf_asset::AssetError::Decode(_)) => {}
            other => panic!("expected a plain Decode error, got {other:?}"),
        }
    }

    /// The remedy is an **instruction**, and it is readable — the B1 law (a run of
    /// spaces inside a user-facing literal is a scripted edit that ate a line
    /// continuation, which has now cost this repo ten mangled messages).
    #[test]
    fn the_upgrade_remedy_says_what_to_do_and_is_readable() {
        let r = ClothAsset::UPGRADE_REMEDY;
        assert!(r.contains("Model Editor"), "{r}");
        assert!(r.contains("Garment"), "{r}");
        assert!(
            !r.contains("  "),
            "the remedy carries a run of spaces: {r:?}"
        );
    }
}
