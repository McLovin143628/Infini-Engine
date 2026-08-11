//! **Strand hair, v1** (P24.4): the `.inf_hair` payload, the per-strand XPBD
//! chains, and the ribbons they draw as.
//!
//! # It is the cloth module's little brother, deliberately
//!
//! A guide strand is a chain of particles under gravity, colliding against the
//! same posed capsules a garment does. So it uses the *same* solver primitives —
//! [`crate::cloth::solve_edge`] and [`crate::cloth::resolve_capsule`] — rather
//! than a second copy of them, and inherits the same properties for free:
//! `sqrt`-only ([`crate::cloth`]'s module docs say why that is not optional),
//! deterministic constraint order, and a state that folds into `state_bytes`.
//!
//! What is genuinely different is the **anchor**. A garment is pinned by its own
//! inverse masses; a strand is pinned by a *joint* — its root rides the scalp,
//! which rides the head, which the pose moves every step. [`step_hair`] therefore
//! takes this step's root positions and writes them onto particle 0 of each
//! strand before it integrates anything.
//!
//! # v1 honesty
//!
//! Three things this is not, each ledgered in ROADMAP §12's P24 block rather than
//! quietly missing:
//!
//! * **No strand–strand interaction.** Guides pass through each other. Real hair
//!   is dominated by it, and the structures that model it (spatial hashing,
//!   position-based friction) are a phase of their own.
//! * **No grooming brush.** Guides are *generated* — [`HairAsset::grow`] from a
//!   set of scalp roots, shaped by a [`HairGroom`] — and then simulated. There is
//!   no interactive comb, so "authored in the Model Editor" is true of the
//!   generator and its parameters, not of a brush.
//! * **The groom RECIPE is not persisted.** `HairGroom` is a generator input and
//!   the grown guides are what the `.inf_hair` carries, so re-grooming means
//!   re-entering the numbers rather than re-opening them. That is the P23
//!   `Op::Unwrap` doctrine (journal the *result*, so a later build replays a fact
//!   rather than its own opinion of the recipe) and it is what keeps the wire
//!   frozen at v1.
//!
//! # Detail is a RENDER budget
//!
//! [`render_mesh`] takes a [`HairDetail`] — a guide stride and an interpolation
//! count — so the lower tiers draw **cards** (one wide strip per three guides)
//! and the top tier draws interpolated strands around each guide. It is the only
//! place the machine's capability tier reaches hair, and it may reach here
//! precisely because ribbons are *not* folded into `state_bytes`; the substep
//! budget beside it is content and never the tier (see [`HairDetail`] and the
//! P22.4 finding it cites).
//!
//! # The ribbons are STRAND-FRAMED, not camera-facing
//!
//! The phase plan said camera-facing. It is not, and the reason is structural:
//! `project_scene_full` takes no camera, and the two projectors have *different*
//! cameras — so camera-facing ribbon geometry could not be compared between the
//! editor and the shipped player at all, only asserted to exist. A strand-framed
//! ribbon (width axis = tangent × a stable reference) is a pure function of sim
//! state, which means the mirror gate can compare the actual bytes. A view-aligned
//! ribbon wants a hair *pass* that orients in the vertex shader, which is where it
//! belongs and is ledgered as such.

use glam::{Mat4, Vec3};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::cloth::{Capsule, ClothCapsule, ClothEdge, ClothError, MIN_CONSTRAINT_LEN_M};

/// The reference axis a strand's ribbon width is built against when the strand
/// runs across it.
///
/// `+Y` (up) is the right choice for hair specifically: a strand hangs *down*, so
/// its tangent is rarely parallel to up, and when it is (a strand pointing
/// straight up out of a scalp) the fallback below takes over deterministically.
const RIBBON_REFERENCE: Vec3 = Vec3::Y;

/// The fallback width axis for a strand running exactly along [`RIBBON_REFERENCE`].
/// Arbitrary but **fixed**, so the degenerate case is deterministic rather than
/// uninitialized — the same discipline `resolve_capsule` applies to a particle on
/// a capsule's axis.
const RIBBON_FALLBACK: Vec3 = Vec3::X;

/// A right-handed orthonormal pair spanning the plane perpendicular to `dir`
/// (which must already be unit length).
///
/// Deterministic by construction: the seed axis is chosen by comparing `dir`'s
/// components, so the same direction always produces the same pair on every
/// machine, and `sqrt` is the only non-arithmetic operation involved. Used by
/// both the curl generator and the interpolated render strands, so a curl and the
/// strands drawn around it share one idea of "sideways".
fn perpendicular_basis(dir: Vec3) -> (Vec3, Vec3) {
    let seed = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = dir.cross(seed);
    let len = u.length();
    let u = if len > MIN_CONSTRAINT_LEN_M {
        u / len
    } else {
        RIBBON_FALLBACK
    };
    (u, dir.cross(u))
}

/// One guide strand: where it is rooted and the points it runs through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HairStrand {
    /// The skeleton joint this strand's root rides. The pose moves the joint, the
    /// joint moves the root, the root drags the strand — which is the whole of
    /// "hair follows the head".
    pub root_joint: u16,
    /// The root's offset from that joint, in the joint's own space, metres. Kept
    /// separately from `points` so a strand can be re-anchored without re-growing
    /// it.
    pub root_offset: [f32; 3],
    /// The strand's rest points in **model space**, root first. Two or more.
    pub points: Vec<[f32; 3]>,
    /// Rest length of each segment, metres — `points.len() - 1` of them,
    /// precomputed at grow time so the solver never recomputes them.
    pub rest_m: Vec<f32>,
}

impl HairStrand {
    /// How many particles this strand has.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the strand has no particles.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// One root a strand is grown from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairRoot {
    /// The joint the root rides.
    pub joint: u16,
    /// Where it sits in that joint's space, metres.
    pub offset: [f32; 3],
    /// The direction the strand grows in, model space. Normalized by
    /// [`HairAsset::grow`]; a zero or non-finite direction refuses the root.
    pub direction: [f32; 3],
    /// **Which clump this root belongs to** (P24.4 grooming). Roots sharing a
    /// number are drawn toward their common axis by
    /// [`HairGroom::clump_strength`]; a groom with no clumping ignores it
    /// entirely, so `0` on every root is the honest "one clump, or none".
    ///
    /// An authored grouping rather than one the generator derives, for the
    /// [`crate::cloth::ClothAsset::from_garment`] reason: a derived grouping is
    /// "whatever this build's clustering rule says", and the day the rule changes
    /// every committed hairstyle changes with it. The sampler that places the
    /// roots decides, and the grown strands are the fact.
    pub clump: u16,
}

/// **How a hairstyle is groomed** — the shape parameters
/// [`HairAsset::grow`] applies to the straight strands it walks out of the scalp.
///
/// These are *generator inputs*, not payload fields, and that is the P23
/// `Op::Unwrap` doctrine applied to hair: what the `.inf_hair` carries is the
/// grown guides themselves, so re-opening a hairstyle on a later build replays a
/// fact rather than re-running whatever this build's curl model happens to be.
/// The consequence is stated rather than glossed: **the recipe is not persisted**,
/// so re-grooming means re-entering the numbers (ledgered).
///
/// [`Default`] is a **straight** groom — every parameter zero — so a hairstyle
/// grown without one is byte-identical to the pre-groom generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairGroom {
    /// How hard a strand is pulled onto its clump's axis at the tip, `0..=1`
    /// (dimensionless). Clamped, and applied as `strength · t²` along the strand
    /// so the root never leaves the scalp — a root that moved would no longer be
    /// on the head the pose carries it with.
    pub clump_strength: f32,
    /// Curl radius, metres: how far the strand spirals off its own growth axis.
    /// `0` is straight hair.
    pub curl_radius_m: f32,
    /// Curl turns over the strand's whole length (dimensionless revolutions).
    /// `2.0` is a loose wave; `6.0` is a tight coil.
    pub curl_turns: f32,
}

impl Default for HairGroom {
    /// Straight, unclumped hair — the identity groom.
    fn default() -> Self {
        Self {
            clump_strength: 0.0,
            curl_radius_m: 0.0,
            curl_turns: 0.0,
        }
    }
}

/// **How densely a hairstyle draws** — a *render* budget, and the one place the
/// machine's capability tier is allowed to reach hair (P24.4).
///
/// # It is a render budget, and that is load-bearing
///
/// [`crate::cloth::ClothMaterial::substeps`] and `ClothSim::quality` are
/// deliberately **content**, never the tier, because a sim budget folded into
/// `state_bytes` diverges the editor's Simulate from the shipped player on any
/// Medium machine (the P22.4 finding). This is the other half of that ruling: the
/// ribbon geometry is a pure function of the strand positions and is *not* folded
/// into [`inf_ecs::hair::hair_state_bytes`], so it may follow the tier — and the
/// gate that keeps the two halves apart asserts exactly that (changing the detail
/// changes what is drawn and leaves the trace byte-identical).
///
/// # The three levels are two numbers
///
/// An enum would name the tiers; two numbers name the *geometry*, which is what a
/// projector needs and what a gate can measure:
///
/// * `guide_stride` — draw every `n`-th guide, widening its ribbon by `n` so the
///   head keeps its silhouette. `3` is a **card**: one wide quad strip standing
///   for three guides, which is the lower-tier path the phase plan names.
/// * `strands_per_guide` — how many *interpolated* render strands surround each
///   drawn guide, spaced by the hairstyle's own measured guide spacing
///   ([`HairState::spacing_m`]). `1` draws the guide alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HairDetail {
    /// Draw every `n`-th guide (`0` is read as `1`).
    pub guide_stride: u8,
    /// Render strands per drawn guide (`0` is read as `1`).
    pub strands_per_guide: u8,
}

impl HairDetail {
    /// **Cards**: one wide strip per three guides, no interpolation. The lower
    /// tiers' hair.
    pub const CARDS: Self = Self {
        guide_stride: 3,
        strands_per_guide: 1,
    };
    /// **Guides**: one ribbon per guide — what P24.4 v1 drew, and the default.
    pub const GUIDES: Self = Self {
        guide_stride: 1,
        strands_per_guide: 1,
    };
    /// **Interpolated**: three render strands around every guide.
    pub const INTERPOLATED: Self = Self {
        guide_stride: 1,
        strands_per_guide: 3,
    };

    /// The stride, read (`0` means `1`).
    pub fn stride(self) -> usize {
        self.guide_stride.max(1) as usize
    }

    /// The interpolation count, read (`0` means `1`).
    pub fn copies(self) -> usize {
        self.strands_per_guide.max(1) as usize
    }
}

impl Default for HairDetail {
    fn default() -> Self {
        Self::GUIDES
    }
}

/// The hairstyle's simulation and ribbon parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HairMaterial {
    /// Segment compliance, m/N. `0` = inextensible, which is what hair mostly is.
    pub segment_compliance: f32,
    /// Velocity damping, 1/s — higher than a garment's by default, because a
    /// strand with a garment's damping whips.
    pub damping: f32,
    /// Collision thickness, metres: how far outside a capsule a strand is held.
    pub thickness_m: f32,
    /// Substeps per fixed step. `0` is read as `1`.
    pub substeps: u8,
    /// Constraint sweeps per substep. `0` is read as `1`.
    pub iterations: u8,
    /// Ribbon width at the root, metres. Tapers to zero at the tip — see
    /// [`ribbon_mesh`].
    pub ribbon_width_m: f32,
}

impl Default for HairMaterial {
    /// A shoulder-length head of hair at 60 Hz. The 4 mm ribbon is a *clump* of
    /// hair, not one fibre: a guide stands for the strands interpolated around it
    /// ([`HairDetail::INTERPOLATED`] draws them; [`HairDetail::GUIDES`] draws the
    /// guide alone).
    fn default() -> Self {
        Self {
            segment_compliance: 0.0,
            damping: 2.0,
            thickness_m: 0.004,
            substeps: 8,
            iterations: 1,
            ribbon_width_m: 0.004,
        }
    }
}

/// The `.inf_hair` payload: a hairstyle prepared for simulation and drawing.
///
/// Self-contained for both, exactly like [`crate::cloth::ClothAsset`] and for the
/// same reasons: the strand points, the precomputed segment lengths, the material
/// and the collision configuration all live here, and [`scalp`](Self::scalp) is
/// provenance plus the dependency edge rather than something the runtime reads.
///
/// # The v1 ladder
///
/// v1 is the first version, so [`inf_asset::AssetError::SchemaTooOld`] is not
/// reachable for this kind yet; what makes the *next* version's refusal correct is
/// the byte-consumption pin in this module's tests. **No `skip_serializing_if`**
/// on any field — the recurring bincode law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HairAsset {
    /// Schema version — **first field**, so `peek_schema_version` reads it without
    /// decoding the rest.
    pub schema_version: u32,
    /// Raw 16-byte GUID of the scalp `.inf_mesh` the roots were sampled from.
    /// Bytes, so the crate stays `uuid`-free.
    pub scalp: [u8; 16],
    /// The guide strands, in the order [`grow`](Self::grow) produced them.
    pub strands: Vec<HairStrand>,
    /// Simulation + ribbon parameters.
    pub material: HairMaterial,
    /// Which joint pairs get collision capsules, and how fat — the head and
    /// shoulders a strand must not pass through. The same record a garment uses,
    /// so a character's collision proxy is described once.
    pub collision: Vec<ClothCapsule>,
}

impl HairAsset {
    /// v1 (P24.4) — the first version.
    pub const CURRENT_VERSION: u32 = 1;

    /// How many strands.
    pub fn strand_count(&self) -> usize {
        self.strands.len()
    }

    /// How many particles across every strand.
    pub fn particle_count(&self) -> usize {
        self.strands.iter().map(HairStrand::len).sum()
    }

    /// **The generator door**: grow `segments`-segment strands of `length_m` from
    /// a set of scalp roots.
    ///
    /// This is what "guides are authored via a generator, not a brush" means in
    /// code, and it is the only way a `.inf_hair` is made — the editor command,
    /// the tests and the gate all come through here, so there is one definition of
    /// what a guide is.
    ///
    /// Deterministic by construction: strands come out in the order the roots came
    /// in, each strand's points are a fixed walk along its own direction, and
    /// nothing is sorted, hashed or randomized. Two runs and two machines produce
    /// byte-identical assets.
    ///
    /// A root whose direction is zero or non-finite is **refused** by name rather
    /// than grown along an arbitrary axis; a `segments` of zero, or a
    /// non-positive length, is the same refusal.
    ///
    /// `groom` shapes what would otherwise be a straight walk — see [`HairGroom`].
    /// [`HairGroom::default`] is the identity, and the clump/curl arithmetic is
    /// spelled in [`inf_math::psin64`] / [`inf_math::pcos64`] rather than `std`
    /// because the grown points are **committed content**: they land in a
    /// `.inf_hair`, are seeded into the solver and ride `state_bytes` from there.
    /// (`hair.rs` is on the `portable_pose` ban list precisely so this cannot
    /// regress.)
    pub fn grow(
        scalp: [u8; 16],
        roots: &[HairRoot],
        length_m: f32,
        segments: u16,
        material: HairMaterial,
        groom: HairGroom,
    ) -> Result<Self, ClothError> {
        if roots.is_empty() || segments == 0 || !length_m.is_finite() || length_m <= 0.0 {
            return Err(ClothError::Degenerate {
                vertices: roots.len(),
                indices: segments as usize,
            });
        }
        if ![groom.clump_strength, groom.curl_radius_m, groom.curl_turns]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(ClothError::NonFinite {
                what: "a hair groom parameter",
            });
        }
        let seg_len = length_m / segments as f32;
        // Each clump's axis: the mean of its roots' origins and of their
        // directions, in a `BTreeMap` so the walk order is the clump-number order
        // and not a hash order — the determinism `from_garment` rests on, applied
        // to the same problem one module over.
        let mut sums: std::collections::BTreeMap<u16, (Vec3, Vec3, f32)> =
            std::collections::BTreeMap::new();
        for r in roots {
            let e = sums.entry(r.clump).or_insert((Vec3::ZERO, Vec3::ZERO, 0.0));
            e.0 += Vec3::from_array(r.offset);
            e.1 += Vec3::from_array(r.direction);
            e.2 += 1.0;
        }
        let clumps: std::collections::BTreeMap<u16, (Vec3, Vec3)> = sums
            .into_iter()
            .map(|(k, (o, d, n))| {
                let dir = d / n;
                let len = dir.length();
                // A clump whose roots point in opposing directions has no axis;
                // it keeps its own strands' directions rather than collapsing
                // them onto an arbitrary one.
                let dir = if len > MIN_CONSTRAINT_LEN_M {
                    dir / len
                } else {
                    Vec3::ZERO
                };
                (k, (o / n, dir))
            })
            .collect();
        let strength = groom.clump_strength.clamp(0.0, 1.0);
        let mut strands = Vec::with_capacity(roots.len());
        for r in roots {
            let dir = Vec3::from_array(r.direction);
            let len = dir.length();
            if !dir.is_finite() || len < MIN_CONSTRAINT_LEN_M {
                return Err(ClothError::NonFinite {
                    what: "a hair root direction",
                });
            }
            let dir = dir / len;
            let origin = Vec3::from_array(r.offset);
            if !origin.is_finite() {
                return Err(ClothError::NonFinite {
                    what: "a hair root offset",
                });
            }
            let (u, v) = perpendicular_basis(dir);
            let (clump_origin, clump_dir) = clumps
                .get(&r.clump)
                .copied()
                .unwrap_or((origin, Vec3::ZERO));
            let points: Vec<[f32; 3]> = (0..=segments)
                .map(|k| {
                    let t = k as f32 / segments as f32;
                    let along = seg_len * k as f32;
                    let mut p = origin + dir * along;
                    // Clumping: toward the clump's own axis, quadratically along
                    // the strand so the ROOT never moves (t = 0 → no pull).
                    if strength > 0.0 && clump_dir != Vec3::ZERO {
                        let target = clump_origin + clump_dir * along;
                        p += (target - p) * (strength * t * t);
                    }
                    // Curl: a helix about the growth axis, phase-shifted so the
                    // root sits ON the axis (cos(0) - 1 = 0) rather than a radius
                    // away from the scalp it grew out of.
                    // `k > 0` rather than a phase that happens to vanish: the
                    // root must be on the axis EXACTLY, and a minimax polynomial
                    // at θ = 0 is off by an ulp of the radius (measured: 1.1 nm,
                    // which is small and is not zero).
                    if k > 0 && groom.curl_radius_m != 0.0 && groom.curl_turns != 0.0 {
                        // The **f64** portable pair, cast per point: `psin`/`pcos`
                        // are demo-grade (~5e-3) by their own module docs, and
                        // these points are committed content, not a displacement
                        // nobody stores. Same rule the primitive generators
                        // follow.
                        let theta = std::f64::consts::TAU * groom.curl_turns as f64 * t as f64;
                        let (cos, sin) = (inf_math::pcos64(theta), inf_math::psin64(theta));
                        p += u * (((cos - 1.0) as f32) * groom.curl_radius_m)
                            + v * ((sin as f32) * groom.curl_radius_m);
                    }
                    p.to_array()
                })
                .collect();
            // Measured from the shaped points rather than assumed to be
            // `seg_len`: a curled strand's chord is SHORTER than its arc, and a
            // solver told otherwise would stretch every segment on the first
            // step. (A straight groom measures back to `seg_len`, which is what
            // keeps the identity groom byte-identical.)
            let rest_m: Vec<f32> = (0..segments as usize)
                .map(|k| (Vec3::from_array(points[k + 1]) - Vec3::from_array(points[k])).length())
                .collect();
            strands.push(HairStrand {
                root_joint: r.joint,
                root_offset: r.offset,
                rest_m,
                points,
            });
        }
        Ok(Self {
            schema_version: Self::CURRENT_VERSION,
            scalp,
            strands,
            material,
            collision: Vec::new(),
        })
    }

    /// Attach the collision capsules the hair is combed around (builder style).
    pub fn with_capsules(mut self, capsules: Vec<ClothCapsule>) -> Self {
        self.collision = capsules;
        self
    }

    /// **Is this hairstyle simulatable?** Checked once, at seed time, so the
    /// per-step solver has nothing left to validate.
    pub fn validate(&self) -> Result<(), ClothError> {
        if self.strands.is_empty() {
            return Err(ClothError::Degenerate {
                vertices: 0,
                indices: 0,
            });
        }
        for s in &self.strands {
            if s.points.len() < 2 || s.rest_m.len() + 1 != s.points.len() {
                return Err(ClothError::Degenerate {
                    vertices: s.points.len(),
                    indices: s.rest_m.len(),
                });
            }
            for p in &s.points {
                if !p.iter().all(|v| v.is_finite()) {
                    return Err(ClothError::NonFinite {
                        what: "a hair strand point",
                    });
                }
            }
            if !s.rest_m.iter().all(|v| v.is_finite() && *v >= 0.0) {
                return Err(ClothError::NonFinite {
                    what: "a hair segment rest length",
                });
            }
        }
        let m = &self.material;
        if ![
            m.segment_compliance,
            m.damping,
            m.thickness_m,
            m.ribbon_width_m,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err(ClothError::NonFinite {
                what: "a hair material parameter",
            });
        }
        Ok(())
    }
}

impl AssetPayload for HairAsset {
    const KIND: AssetKind = AssetKind::Hair;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    const UPGRADE_REMEDY: &'static str =
        "re-grow the guides from their scalp (Model Editor ▸ Hair ▸ Grow Guides)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// The live particle state of one hairstyle — every strand's particles,
/// flattened, with the strand boundaries alongside.
///
/// Flattened rather than `Vec<Vec<_>>` because the solver touches every particle
/// several times a substep and the trace folds them in one pass; `starts` is what
/// keeps "which strand is this" answerable without a second allocation per strand.
#[derive(Debug, Clone, PartialEq)]
pub struct HairState {
    /// Positions, model space, strand by strand.
    pub x: Vec<[f32; 3]>,
    /// Velocities, m/s.
    pub v: Vec<[f32; 3]>,
    /// `starts[i]` is where strand `i` begins in `x`; `starts` has
    /// `strand_count + 1` entries, so `starts[i+1]` is its end.
    pub starts: Vec<u32>,
    /// **How far apart this hairstyle's guides are**, metres — the mean distance
    /// from a guide's rest root to its nearest neighbour's.
    ///
    /// Measured once, here, because it is a property of the *hairstyle* and the
    /// thing that needs it ([`render_mesh`]'s interpolated strands) runs every
    /// frame: an O(n²) sweep per seed is cheap, and per projection is not.
    ///
    /// Derived rather than authored, so it costs no wire field and cannot drift
    /// from the guides it describes — a hairstyle whose guides were re-grown
    /// closer together interpolates closer together with nothing to re-author.
    /// A single-guide hairstyle has no neighbour and reads `0`, which
    /// [`render_mesh`] takes as "draw the guide alone".
    pub spacing_m: f32,
}

impl HairState {
    /// Seed a hairstyle at its rest shape with zero velocity. Refuses an invalid
    /// asset rather than seeding a state the solver would re-check every step.
    pub fn seed(asset: &HairAsset) -> Result<Self, ClothError> {
        asset.validate()?;
        let mut x = Vec::with_capacity(asset.particle_count());
        let mut starts = Vec::with_capacity(asset.strand_count() + 1);
        for s in &asset.strands {
            starts.push(x.len() as u32);
            x.extend_from_slice(&s.points);
        }
        starts.push(x.len() as u32);
        Ok(Self {
            v: vec![[0.0; 3]; x.len()],
            spacing_m: guide_spacing(asset),
            x,
            starts,
        })
    }

    /// How many particles across every strand.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether there are no particles at all.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// How many strands.
    pub fn strand_count(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }
}

/// The mean nearest-neighbour distance between guide roots, metres — see
/// [`HairState::spacing_m`], which is the only caller and the reason this exists.
///
/// `0` for a hairstyle with fewer than two guides, and for one whose guides are
/// all coincident: both mean "there is no space between the guides to fill".
fn guide_spacing(asset: &HairAsset) -> f32 {
    let roots: Vec<Vec3> = asset
        .strands
        .iter()
        .filter_map(|s| s.points.first().map(|p| Vec3::from_array(*p)))
        .collect();
    if roots.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for (i, a) in roots.iter().enumerate() {
        let mut nearest = f32::INFINITY;
        for (j, b) in roots.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = (*a - *b).length();
            if d < nearest {
                nearest = d;
            }
        }
        if nearest.is_finite() {
            total += nearest;
        }
    }
    let mean = total / roots.len() as f32;
    if mean.is_finite() {
        mean
    } else {
        0.0
    }
}

/// **Where each strand's root is this step**, in model space.
///
/// One entry per strand, in the asset's own order. A strand whose `root_joint` is
/// not in the skeleton keeps its **rest** root — the honest answer, because a
/// hairstyle fitted to another rig must not have its whole scalp collapse onto
/// joint 0.
pub fn roots_for(asset: &HairAsset, joint_globals: &[Mat4]) -> Vec<Vec3> {
    asset
        .strands
        .iter()
        .map(|s| {
            let offset = Vec3::from_array(s.root_offset);
            match joint_globals.get(s.root_joint as usize) {
                Some(m) => {
                    let p = m.transform_point3(offset);
                    if p.is_finite() {
                        p
                    } else {
                        Vec3::from_array(s.points[0])
                    }
                }
                None => Vec3::from_array(s.points[0]),
            }
        })
        .collect()
}

/// **Place a hairstyle's capsules on the pose the sim just evaluated** — the
/// head and shoulders a strand must not pass through.
///
/// A thin forward to [`crate::cloth::capsules_for`]'s rule over this asset's own
/// capsule list, so hair and cloth cannot end up with two different ideas of
/// where a character's collision proxy is. A capsule naming a joint the skeleton
/// does not have is dropped, not clamped, for the same reason it is there.
pub fn capsules_for_hair(asset: &HairAsset, joint_globals: &[Mat4]) -> Vec<Capsule> {
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

/// **The one XPBD hair step.** Advance `state` by `dt` under `gravity`, with each
/// strand's root pinned to `roots[i]`, against `capsules`.
///
/// Called from [`inf_ecs::hair::step_hair_simulation`], which both hosts' fixed
/// steps call — so this is the only place hair moves.
///
/// Order inside a substep mirrors [`crate::cloth::step_cloth`] exactly, with the
/// root pin taking the place of the inverse-mass pin: **anchor**, integrate,
/// solve the chain's distance constraints, collide, derive velocity. The anchor is
/// written *before* integration for the same reason a pinned garment particle is
/// skipped there — a root that were merely re-set afterwards would still have
/// contributed a velocity on the next line, and the strand would snap.
///
/// A non-finite `dt`, `gravity` or root, a `roots` list of the wrong length, or an
/// empty hairstyle is a **no-op** rather than a panic: this runs inside a fixed
/// step, and one bad hairstyle must not take a level down.
pub fn step_hair(
    asset: &HairAsset,
    state: &mut HairState,
    dt: f32,
    gravity: Vec3,
    roots: &[Vec3],
    capsules: &[Capsule],
) {
    if state.x.is_empty()
        || roots.len() != state.strand_count()
        || !dt.is_finite()
        || dt <= 0.0
        || !gravity.is_finite()
        || roots.iter().any(|r| !r.is_finite())
    {
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
    let alpha = asset.material.segment_compliance.max(0.0) / (h * h);

    let n = state.x.len();
    let mut x: Vec<Vec3> = state.x.iter().copied().map(Vec3::from_array).collect();
    let mut v: Vec<Vec3> = state.v.iter().copied().map(Vec3::from_array).collect();
    let mut prev: Vec<Vec3> = x.clone();
    // A root is pinned (`w == 0`) and every other particle is free. Built once
    // rather than branched on per particle, so `solve_edge` — shared with the
    // cloth solver — needs no hair-specific arm.
    let mut w = vec![1.0f32; n];
    for s in 0..state.strand_count() {
        w[state.starts[s] as usize] = 0.0;
    }
    let mut lambda: Vec<f32> = vec![0.0; n];

    for _ in 0..substeps {
        for (s, root) in roots.iter().enumerate() {
            let i = state.starts[s] as usize;
            x[i] = *root;
            v[i] = Vec3::ZERO;
        }
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
            for (s, strand) in asset.strands.iter().enumerate() {
                let base = state.starts[s] as usize;
                for (k, rest_m) in strand.rest_m.iter().enumerate() {
                    let e = ClothEdge {
                        a: (base + k) as u32,
                        b: (base + k + 1) as u32,
                        rest_m: *rest_m,
                    };
                    solve_segment(&mut x, &w, &e, alpha, &mut lambda[base + k]);
                }
            }
        }
        for i in 0..n {
            if w[i] <= 0.0 {
                continue;
            }
            for c in capsules {
                if let Some(out) = crate::cloth::resolve_capsule(x[i], c, thickness) {
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
        // A strand that has gone non-finite is frozen rather than written into the
        // trace — the `step_cloth` rule, for the same reason.
        if x[i].is_finite() && v[i].is_finite() {
            state.x[i] = x[i].to_array();
            state.v[i] = v[i].to_array();
        }
    }
}

/// One segment constraint — a thin forward to the cloth solver's routine, so hair
/// and cloth cannot disagree about what a distance constraint is.
#[inline]
fn solve_segment(x: &mut [Vec3], w: &[f32], e: &ClothEdge, alpha: f32, lambda: &mut f32) {
    crate::cloth::solve_edge(x, w, e, alpha, lambda)
}

/// **Ribbon geometry for a simulated hairstyle** — positions and a triangle list,
/// model space, ready for `inf_render::deformed_skinned_mesh`.
///
/// One quad strip per strand: each particle becomes two vertices offset by
/// ±half-width along that particle's own **width axis**, which is
/// `tangent × RIBBON_REFERENCE` normalized (falling back to
/// [`RIBBON_FALLBACK`] when the strand runs along the reference).
///
/// The width **tapers linearly to zero at the tip**, which is what makes a
/// four-millimetre ribbon read as hair rather than as tape.
///
/// Strand-framed rather than camera-facing — see the module docs for why that is
/// a decision and not an omission. Being a pure function of the sim state is what
/// lets a gate compare the actual bytes between two hosts.
///
/// A strand shorter than two particles contributes nothing; a degenerate segment
/// inherits the previous particle's width axis, so a strand that folds onto itself
/// produces a thin ribbon rather than a NaN.
///
/// This is [`render_mesh`] at [`HairDetail::GUIDES`] and nothing else — kept as a
/// name because "draw the guides" is what a hairstyle means at full detail.
pub fn ribbon_mesh(asset: &HairAsset, state: &HairState) -> (Vec<[f32; 3]>, Vec<u32>) {
    render_mesh(asset, state, HairDetail::GUIDES)
}

/// **Ribbon geometry at a chosen detail** (P24.4) — [`ribbon_mesh`]'s rule with
/// the guide stride and the interpolation count of a [`HairDetail`].
///
/// Three levels out of two numbers, each a pure function of the same sim state:
///
/// * `guide_stride > 1` — **cards**. Every `n`-th guide is drawn, widened by `n`,
///   so a third of the geometry covers the same head. The widening is what makes
///   this a card rather than a bald patch, and it is why the lower tiers get a
///   silhouette instead of a haircut.
/// * `strands_per_guide > 1` — **interpolated render strands**. Each drawn guide
///   is accompanied by `n - 1` copies of itself, displaced around it in the plane
///   perpendicular to the strand at each particle, at the hairstyle's own measured
///   [`HairState::spacing_m`]. They follow the guide exactly, which is what "the
///   guide stands for the strands around it" means geometrically.
/// * both `1` — the guides themselves, byte-identical to what P24.4 v1 drew.
///
/// The displacement angles are [`inf_math::psin64`] / [`inf_math::pcos64`]: the
/// two hosts' projectors compare this geometry byte for byte, so `std` trig here
/// would make the mirror gate a statement about the target triple.
pub fn render_mesh(
    asset: &HairAsset,
    state: &HairState,
    detail: HairDetail,
) -> (Vec<[f32; 3]>, Vec<u32>) {
    let stride = detail.stride();
    let copies = detail.copies();
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(state.len() * 2 * copies);
    let mut idx: Vec<u32> = Vec::new();
    // A card stands for the guides it replaced, so it carries their width.
    let half = (asset.material.ribbon_width_m.max(0.0)) * 0.5 * stride as f32;
    // Interpolated strands fill the space BETWEEN the guides; one strand has no
    // space to fill and is drawn where it is.
    let spread = if copies > 1 {
        state.spacing_m.max(0.0) * 0.5
    } else {
        0.0
    };
    for s in (0..state.strand_count()).step_by(stride) {
        let (from, to) = (state.starts[s] as usize, state.starts[s + 1] as usize);
        let count = to - from;
        if count < 2 {
            continue;
        }
        for c in 0..copies {
            let theta = std::f64::consts::TAU * (c as f64) / (copies as f64);
            let (cos, sin) = (
                inf_math::pcos64(theta) as f32,
                inf_math::psin64(theta) as f32,
            );
            let base = pos.len() as u32;
            let mut axis = RIBBON_FALLBACK;
            let mut binormal = Vec3::Y;
            for k in 0..count {
                let p = Vec3::from_array(state.x[from + k]);
                // The tangent at this particle: forward for the root, backward for the
                // tip, so every particle has one without a special case in the middle.
                let next = Vec3::from_array(state.x[from + (k + 1).min(count - 1)]);
                let prev = Vec3::from_array(state.x[from + k.saturating_sub(1)]);
                let tangent = next - prev;
                let candidate = tangent.cross(RIBBON_REFERENCE);
                let len = candidate.length();
                if len > MIN_CONSTRAINT_LEN_M {
                    axis = candidate / len;
                    binormal = tangent.cross(axis).normalize_or_zero();
                } else if tangent.length() > MIN_CONSTRAINT_LEN_M {
                    // The strand runs along the reference: a fixed fallback, so the
                    // degenerate case is deterministic across runs and hosts.
                    axis = RIBBON_FALLBACK;
                    binormal = tangent.cross(axis).normalize_or_zero();
                }
                let taper = 1.0 - (k as f32 / (count - 1) as f32);
                let w = axis * (half * taper);
                // Copy 0 sits ON the guide (cos 0 = 1, sin 0 = 0 scaled by a
                // spread of zero when there is one copy), so a single-copy detail
                // is the guide and not a strand beside it.
                let off = (axis * cos + binormal * sin) * spread;
                pos.push((p + off - w).to_array());
                pos.push((p + off + w).to_array());
            }
            for k in 0..count as u32 - 1 {
                let (a, b) = (base + k * 2, base + k * 2 + 1);
                let (c, d) = (a + 2, b + 2);
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
    }
    (pos, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// The fixture's four roots: 2 cm apart along x, two per clump, so a clumped
    /// groom has two axes to pull toward rather than one that every strand is
    /// already on (a single-clump fixture would satisfy a clump assertion by
    /// having nowhere to go).
    fn fixture_roots() -> Vec<HairRoot> {
        (0..4)
            .map(|i| HairRoot {
                joint: 1,
                offset: [i as f32 * 0.02 - 0.03, 0.0, 0.0],
                direction: [0.0, -1.0, 0.0],
                clump: i / 2,
            })
            .collect()
    }

    /// Four strands hanging straight down from joint 1, 20 cm long, 4 segments.
    fn hairstyle() -> HairAsset {
        HairAsset::grow(
            [11; 16],
            &fixture_roots(),
            0.2,
            4,
            HairMaterial {
                // A stiff, heavily damped default would settle before the drape
                // arm could see anything move; these are the shipped defaults.
                ..HairMaterial::default()
            },
            HairGroom::default(),
        )
        .expect("the fixture hairstyle grows")
        .with_capsules(vec![ClothCapsule {
            joint_a: 0,
            joint_b: 1,
            radius_m: 0.05,
        }])
    }

    /// A joint-global table putting joint 1 at `y`.
    fn globals(y: f32) -> [Mat4; 2] {
        [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(0.0, y, 0.0)),
        ]
    }

    #[test]
    fn growing_produces_strands_of_the_right_shape() {
        let a = hairstyle();
        assert_eq!(a.strand_count(), 4);
        assert_eq!(a.particle_count(), 4 * 5, "4 segments is 5 particles");
        for s in &a.strands {
            assert_eq!(s.rest_m.len() + 1, s.points.len());
            for (k, rest) in s.rest_m.iter().enumerate() {
                let d =
                    (Vec3::from_array(s.points[k]) - Vec3::from_array(s.points[k + 1])).length();
                assert!((d - rest).abs() < 1e-6, "segment {k}: {d} vs {rest}");
                assert!((rest - 0.05).abs() < 1e-6, "0.2 m over 4 segments");
            }
        }
        assert_eq!(a.validate(), Ok(()));
        // Deterministic: the same roots grow the same guides.
        assert_eq!(a.strands, hairstyle().strands);
    }

    #[test]
    fn a_bad_root_is_refused_by_name() {
        let bad = |direction: [f32; 3]| {
            HairAsset::grow(
                [0; 16],
                &[HairRoot {
                    joint: 0,
                    offset: [0.0; 3],
                    direction,
                    clump: 0,
                }],
                0.1,
                2,
                HairMaterial::default(),
                HairGroom::default(),
            )
        };
        assert!(matches!(
            bad([0.0; 3]),
            Err(ClothError::NonFinite {
                what: "a hair root direction"
            })
        ));
        assert!(matches!(
            bad([f32::NAN, 0.0, 0.0]),
            Err(ClothError::NonFinite { .. })
        ));
        assert!(matches!(
            HairAsset::grow(
                [0; 16],
                &[],
                0.1,
                2,
                HairMaterial::default(),
                HairGroom::default()
            ),
            Err(ClothError::Degenerate { .. })
        ));
        assert!(matches!(
            bad([0.0, -1.0, 0.0]).map(|a| HairAsset::grow(
                [0; 16],
                &[],
                0.0,
                0,
                a.material,
                HairGroom::default()
            )),
            Ok(Err(ClothError::Degenerate { .. }))
        ));
        // A non-finite groom parameter is refused by name too, rather than
        // producing a hairstyle of NaNs the solver would freeze on for ever.
        assert!(matches!(
            HairAsset::grow(
                [0; 16],
                &fixture_roots(),
                0.1,
                2,
                HairMaterial::default(),
                HairGroom {
                    curl_radius_m: f32::NAN,
                    ..HairGroom::default()
                }
            ),
            Err(ClothError::NonFinite {
                what: "a hair groom parameter"
            })
        ));
    }

    /// **The root follows the joint.** A hairstyle whose head has moved has its
    /// strands moved with it — which is the whole of "hair rides the character".
    #[test]
    fn the_roots_follow_the_posed_joint() {
        let a = hairstyle();
        let low = roots_for(&a, &globals(1.0));
        let high = roots_for(&a, &globals(1.5));
        assert_eq!(low.len(), 4);
        for (l, h) in low.iter().zip(&high) {
            assert!((h.y - l.y - 0.5).abs() < 1e-5, "{l:?} → {h:?}");
        }
        // A strand naming a joint the skeleton does not have keeps its REST root
        // rather than collapsing onto joint 0.
        let mut orphan = hairstyle();
        orphan.strands[0].root_joint = 9;
        let r = roots_for(&orphan, &globals(1.0));
        assert_eq!(r[0], Vec3::from_array(orphan.strands[0].points[0]));
        assert_ne!(r[1], Vec3::from_array(orphan.strands[1].points[0]));
    }

    /// **The chain holds and the tip swings.** Segment lengths stay at rest under
    /// gravity while the free end actually moves.
    #[test]
    fn a_strand_holds_its_length_and_its_tip_moves() {
        let a = hairstyle();
        let mut s = HairState::seed(&a).unwrap();
        let roots = roots_for(&a, &globals(1.0));
        let tip0 = s.x[4];
        for _ in 0..120 {
            // Gravity along +X, so a strand grown straight down has somewhere to
            // go: under -Y it is already at equilibrium and nothing would move.
            step_hair(&a, &mut s, 1.0 / 60.0, Vec3::X * 9.81, &roots, &[]);
        }
        assert_ne!(s.x[4], tip0, "the tip never moved");
        for (st, root) in roots.iter().enumerate() {
            let base = s.starts[st] as usize;
            assert_eq!(
                Vec3::from_array(s.x[base]),
                *root,
                "strand {st}'s root left its anchor"
            );
            for k in 0..a.strands[st].rest_m.len() {
                let d = (Vec3::from_array(s.x[base + k]) - Vec3::from_array(s.x[base + k + 1]))
                    .length();
                let rest = a.strands[st].rest_m[k];
                assert!(
                    (d - rest).abs() < rest * 0.02,
                    "strand {st} segment {k} is {d} m, rest {rest}"
                );
            }
        }
    }

    /// **MUTATION arm for the chain solve**: with the segment constraints gone the
    /// strand's particles free-fall away from each other.
    #[test]
    fn a_strand_with_no_segments_falls_apart() {
        let mut a = hairstyle();
        for s in &mut a.strands {
            s.rest_m.clear();
            s.points.truncate(1);
        }
        // A one-particle strand no longer validates, which is itself the refusal.
        assert!(matches!(a.validate(), Err(ClothError::Degenerate { .. })));

        // The reachable half: rest lengths of zero collapse the strand onto its
        // root, which a working solver does and a severed one does not.
        let mut collapsed = hairstyle();
        for s in &mut collapsed.strands {
            s.rest_m.iter_mut().for_each(|r| *r = 0.0);
        }
        let mut st = HairState::seed(&collapsed).unwrap();
        let roots = roots_for(&collapsed, &globals(1.0));
        for _ in 0..120 {
            step_hair(
                &collapsed,
                &mut st,
                1.0 / 60.0,
                Vec3::NEG_Y * 9.81,
                &roots,
                &[],
            );
        }
        let tip = Vec3::from_array(st.x[4]);
        assert!(
            (tip - roots[0]).length() < 0.01,
            "zero-length segments must pull the strand onto its root; the tip is \
             {:.3} m away",
            (tip - roots[0]).length()
        );
    }

    /// **Capsule collision, asserted on the WORLD**, with its falsifier: a strand
    /// blown against a sphere ends up outside it, and the same strand with no
    /// capsule ends up inside.
    #[test]
    fn strands_do_not_pass_through_the_head() {
        let a = hairstyle().with_capsules(vec![ClothCapsule {
            joint_a: 1,
            joint_b: 1,
            radius_m: 0.08,
        }]);
        let roots = roots_for(&a, &globals(1.0));
        // A sphere centred at the root, so a strand hanging from it must bend
        // around rather than through.
        let head = Capsule {
            a: Vec3::new(0.0, 1.0, 0.0),
            b: Vec3::new(0.0, 1.0, 0.0),
            radius_m: 0.08,
        };
        let run = |caps: &[Capsule]| {
            let mut s = HairState::seed(&a).unwrap();
            for _ in 0..180 {
                step_hair(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * 9.81, &roots, caps);
            }
            s
        };
        let skin = head.radius_m + a.material.thickness_m;
        let with = run(std::slice::from_ref(&head));
        let mut inside_with = 0;
        for (i, p) in with.x.iter().enumerate() {
            // The ROOTS are pinned on the sphere's centre by construction, so they
            // are excluded: a pinned particle is not something collision may move.
            if with.starts.contains(&(i as u32)) {
                continue;
            }
            if (Vec3::from_array(*p) - head.a).length() < skin - 1e-3 {
                inside_with += 1;
            }
        }
        assert_eq!(
            inside_with, 0,
            "{inside_with} free particles are inside the head"
        );
        // The falsifier: without the capsule, particles are inside it.
        let without = run(&[]);
        let inside_without = without
            .x
            .iter()
            .enumerate()
            .filter(|(i, _)| !without.starts.contains(&(*i as u32)))
            .filter(|(_, p)| (Vec3::from_array(**p) - head.a).length() < skin - 1e-3)
            .count();
        assert!(
            inside_without > 0,
            "with no capsule the strands must hang through the volume the head \
             occupies; none of them does, so the arm above asserts nothing"
        );
    }

    #[test]
    fn stepping_is_bit_identical_between_two_runs() {
        let a = hairstyle();
        let run = || {
            let mut s = HairState::seed(&a).unwrap();
            for k in 0..90 {
                let roots = roots_for(&a, &globals(1.0 + 0.002 * k as f32));
                step_hair(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * 9.81, &roots, &[]);
            }
            s
        };
        let (p, q) = (run(), run());
        assert_eq!(p.x, q.x);
        assert_eq!(p.v, q.v);
        assert_ne!(p.x, HairState::seed(&a).unwrap().x, "nothing moved");
    }

    #[test]
    fn a_non_finite_step_is_a_no_op() {
        let a = hairstyle();
        let seed = HairState::seed(&a).unwrap();
        let roots = roots_for(&a, &globals(1.0));
        for (dt, g) in [
            (f32::NAN, Vec3::NEG_Y),
            (0.0, Vec3::NEG_Y),
            (1.0 / 60.0, Vec3::splat(f32::NAN)),
        ] {
            let mut s = seed.clone();
            step_hair(&a, &mut s, dt, g, &roots, &[]);
            assert_eq!(s.x, seed.x, "dt={dt} g={g:?} moved the hair");
        }
        // A roots list of the wrong length is refused rather than indexed past.
        let mut s = seed.clone();
        step_hair(&a, &mut s, 1.0 / 60.0, Vec3::NEG_Y * 9.81, &roots[..2], &[]);
        assert_eq!(s.x, seed.x);
        // …and a real step does move it.
        let mut s = seed.clone();
        step_hair(&a, &mut s, 1.0 / 60.0, Vec3::X * 9.81, &roots, &[]);
        assert_ne!(s.x, seed.x);
    }

    /// The ribbons are real geometry: two vertices per particle, six indices per
    /// segment, tapering to a point, and every position finite.
    #[test]
    fn the_ribbons_are_real_geometry_and_taper() {
        let a = hairstyle();
        let s = HairState::seed(&a).unwrap();
        let (pos, idx) = ribbon_mesh(&a, &s);
        assert_eq!(pos.len(), a.particle_count() * 2);
        assert_eq!(idx.len(), a.strand_count() * 4 * 6, "6 indices per segment");
        assert!(idx.iter().all(|i| (*i as usize) < pos.len()));
        assert!(pos.iter().all(|p| p.iter().all(|c| c.is_finite())));
        // Root pair is a full width apart; tip pair is coincident (taper → 0).
        let root_w = (Vec3::from_array(pos[1]) - Vec3::from_array(pos[0])).length();
        assert!(
            (root_w - a.material.ribbon_width_m).abs() < 1e-6,
            "the root ribbon is {root_w} m wide"
        );
        let tip = (a.strands[0].len() - 1) * 2;
        assert!(
            (Vec3::from_array(pos[tip + 1]) - Vec3::from_array(pos[tip])).length() < 1e-6,
            "the ribbon must taper to a point"
        );
        // A one-particle strand contributes nothing rather than a degenerate quad.
        let mut short = a.clone();
        short.strands[0].points.truncate(1);
        short.strands[0].rest_m.clear();
        let mut st = s.clone();
        st.starts = vec![0, 1, 6, 11, 16];
        let (p2, i2) = ribbon_mesh(&short, &st);
        assert!(p2.len() < pos.len() && !i2.is_empty());
    }

    // ── grooming: clump and curl ────────────────────────────────────────────

    /// Grow the fixture's roots under an arbitrary groom.
    fn groomed(groom: HairGroom) -> HairAsset {
        HairAsset::grow(
            [11; 16],
            &fixture_roots(),
            0.2,
            8,
            HairMaterial::default(),
            groom,
        )
        .expect("the groomed hairstyle grows")
    }

    /// **The identity groom changes nothing.** The control for every arm below:
    /// a groom whose parameters are all zero must produce the guides the
    /// pre-groom generator produced, or "clumping moved the tips" would be a
    /// statement about the generator rather than about clumping.
    #[test]
    fn the_default_groom_is_the_straight_generator() {
        let straight = groomed(HairGroom::default());
        for s in &straight.strands {
            let root = Vec3::from_array(s.points[0]);
            let tip = Vec3::from_array(s.points[s.len() - 1]);
            // Straight down, 20 cm, and along nothing else.
            assert!((tip.x - root.x).abs() < 1e-6, "x drifted: {tip:?}");
            assert!((tip.z - root.z).abs() < 1e-6, "z drifted: {tip:?}");
            assert!((root.y - tip.y - 0.2).abs() < 1e-5, "length: {tip:?}");
            for rest in &s.rest_m {
                assert!((rest - 0.025).abs() < 1e-6, "0.2 m over 8 segments");
            }
        }
    }

    /// **Clumping pulls the TIPS together and leaves the ROOTS alone.** Measured
    /// against the straight control on the same roots: within a clump the tips
    /// converge, and every root is where it was — a strand whose root moved would
    /// no longer be on the scalp the pose carries it with.
    #[test]
    fn clumping_converges_the_tips_and_never_moves_a_root() {
        let straight = groomed(HairGroom::default());
        let clumped = groomed(HairGroom {
            clump_strength: 1.0,
            ..HairGroom::default()
        });
        let tip_gap = |a: &HairAsset, i: usize, j: usize| {
            let ti = Vec3::from_array(a.strands[i].points[a.strands[i].len() - 1]);
            let tj = Vec3::from_array(a.strands[j].points[a.strands[j].len() - 1]);
            (ti - tj).length()
        };
        // Roots 0 and 1 share clump 0; 2 and 3 share clump 1.
        assert!(
            tip_gap(&clumped, 0, 1) < tip_gap(&straight, 0, 1) * 0.25,
            "clump 0's tips did not converge: {} vs {}",
            tip_gap(&clumped, 0, 1),
            tip_gap(&straight, 0, 1)
        );
        assert!(
            tip_gap(&clumped, 2, 3) < tip_gap(&straight, 2, 3) * 0.25,
            "clump 1's tips did not converge"
        );
        // …and the two clumps stay APART. A clump strength that merged the whole
        // head into one strand would pass the arm above and be wrong.
        assert!(
            tip_gap(&clumped, 1, 2) > tip_gap(&straight, 1, 2) * 0.5,
            "the two clumps merged into one: {}",
            tip_gap(&clumped, 1, 2)
        );
        for (s, c) in straight.strands.iter().zip(&clumped.strands) {
            assert_eq!(s.points[0], c.points[0], "a root moved");
        }
    }

    /// **Curl leaves the axis and comes back to it**, and the segment rest
    /// lengths are measured from the curled points rather than assumed.
    ///
    /// The second half is the one that bites: a helix is LONGER than the straight
    /// walk it wraps, so a solver handed the straight `length / segments` would
    /// compress every segment on the first step and the curl would shake out.
    #[test]
    fn curl_leaves_the_axis_and_its_rest_lengths_are_measured() {
        let straight = groomed(HairGroom::default());
        let curly = groomed(HairGroom {
            curl_radius_m: 0.02,
            curl_turns: 2.0,
            ..HairGroom::default()
        });
        let s = &curly.strands[0];
        assert_eq!(s.points[0], straight.strands[0].points[0], "the root moved");
        let axis_distance = |p: [f32; 3]| {
            let root = Vec3::from_array(s.points[0]);
            let v = Vec3::from_array(p) - root;
            // The growth axis is -Y, so the off-axis distance is the XZ radius.
            Vec3::new(v.x, 0.0, v.z).length()
        };
        let max_off = s
            .points
            .iter()
            .map(|p| axis_distance(*p))
            .fold(0.0f32, f32::max);
        assert!(
            (max_off - 0.04).abs() < 5e-3,
            "a 2 cm curl radius swings 4 cm across the axis, measured {max_off}"
        );
        // Two full turns: the strand returns to the axis at the halfway point and
        // at the tip.
        assert!(axis_distance(s.points[4]) < 1e-3, "half-way off axis");
        assert!(axis_distance(s.points[8]) < 1e-3, "the tip is off axis");
        let curled_chord: f32 = s.rest_m.iter().sum();
        let straight_chord: f32 = straight.strands[0].rest_m.iter().sum();
        assert!(
            curled_chord > straight_chord,
            "a curl is LONGER end to end than the straight walk it wraps: \
             {curled_chord} vs {straight_chord}"
        );
        for (k, rest) in s.rest_m.iter().enumerate() {
            let d = (Vec3::from_array(s.points[k + 1]) - Vec3::from_array(s.points[k])).length();
            assert!((d - rest).abs() < 1e-6, "segment {k} rest is not measured");
        }
    }

    /// The groom is deterministic — the same roots and the same numbers grow the
    /// same guides, twice.
    #[test]
    fn a_groom_is_deterministic() {
        let g = HairGroom {
            clump_strength: 0.6,
            curl_radius_m: 0.01,
            curl_turns: 3.0,
        };
        assert_eq!(groomed(g).strands, groomed(g).strands);
    }

    // ── detail: cards, guides, interpolated strands ─────────────────────────

    /// **The measured guide spacing is the fixture's own 2 cm.** Derived, not
    /// authored, so it cannot drift from the guides it describes.
    #[test]
    fn the_guide_spacing_is_measured_from_the_guides() {
        let a = hairstyle();
        let s = HairState::seed(&a).unwrap();
        assert!(
            (s.spacing_m - 0.02).abs() < 1e-6,
            "the fixture's roots are 2 cm apart, measured {}",
            s.spacing_m
        );
        // One guide has no neighbour, and reads zero rather than infinity.
        let mut lone = a.clone();
        lone.strands.truncate(1);
        assert_eq!(HairState::seed(&lone).unwrap().spacing_m, 0.0);
    }

    /// **Cards draw a third of the geometry and stay as wide as what they
    /// replaced.** The lower-tier path, measured rather than asserted to exist.
    #[test]
    fn cards_decimate_the_guides_and_widen_what_is_left() {
        let a = hairstyle();
        let s = HairState::seed(&a).unwrap();
        let (guides, gi) = render_mesh(&a, &s, HairDetail::GUIDES);
        let (cards, ci) = render_mesh(&a, &s, HairDetail::CARDS);
        // 4 guides at stride 3 draw strands 0 and 3.
        assert_eq!(cards.len(), guides.len() / 2, "4 guides → 2 cards");
        assert_eq!(ci.len(), gi.len() / 2);
        assert!(ci.iter().all(|i| (*i as usize) < cards.len()));
        let width = |p: &[[f32; 3]]| (Vec3::from_array(p[1]) - Vec3::from_array(p[0])).length();
        assert!(
            (width(&cards) - width(&guides) * 3.0).abs() < 1e-6,
            "a card standing for three guides is three guides wide: {} vs {}",
            width(&cards),
            width(&guides)
        );
    }

    /// **Interpolated strands surround the guide** — three times the geometry,
    /// each copy displaced by the hairstyle's own measured spacing and none of
    /// them sitting on top of another.
    #[test]
    fn interpolated_strands_surround_each_guide() {
        let a = hairstyle();
        let s = HairState::seed(&a).unwrap();
        let (guides, _) = render_mesh(&a, &s, HairDetail::GUIDES);
        let (interp, ii) = render_mesh(&a, &s, HairDetail::INTERPOLATED);
        assert_eq!(interp.len(), guides.len() * 3);
        assert!(ii.iter().all(|i| (*i as usize) < interp.len()));
        assert!(interp.iter().all(|p| p.iter().all(|c| c.is_finite())));
        // The three copies of strand 0's root are distinct, and each is within
        // half the guide spacing of where the guide itself is.
        let per_copy = a.strands[0].len() * 2;
        let root_of = |c: usize| Vec3::from_array(interp[c * per_copy]);
        let guide_root = Vec3::from_array(guides[0]);
        for c in 0..3 {
            let d = (root_of(c) - guide_root).length();
            assert!(
                d > 1e-4 && d < s.spacing_m,
                "copy {c} sits {d} m from its guide"
            );
        }
        assert!(
            (root_of(0) - root_of(1)).length() > 1e-4,
            "copies 0 and 1 coincide"
        );
        assert!(
            (root_of(1) - root_of(2)).length() > 1e-4,
            "copies 1 and 2 coincide"
        );
    }

    /// A zeroed [`HairDetail`] is read as the guides rather than as "draw
    /// nothing" — the `0 is 1` rule the material's substep budget already has.
    #[test]
    fn a_zeroed_detail_reads_as_the_guides() {
        let a = hairstyle();
        let s = HairState::seed(&a).unwrap();
        let zero = HairDetail {
            guide_stride: 0,
            strands_per_guide: 0,
        };
        assert_eq!(
            render_mesh(&a, &s, zero),
            render_mesh(&a, &s, HairDetail::GUIDES)
        );
        assert_eq!(HairDetail::default(), HairDetail::GUIDES);
        assert_eq!(ribbon_mesh(&a, &s), render_mesh(&a, &s, HairDetail::GUIDES));
    }

    // ── the schema ladder ───────────────────────────────────────────────────

    #[test]
    fn the_payload_round_trips_deterministically() {
        let a = hairstyle();
        let e1 = encode(&a).unwrap();
        assert_eq!(e1, encode(&a).unwrap(), "re-encoding is byte-identical");
        assert_eq!(decode::<HairAsset>(&e1).unwrap(), a);
    }

    /// The **v1 wire shape**, positionally — a real shadow struct.
    #[derive(Deserialize)]
    struct HairAssetV1Wire {
        schema_version: u32,
        scalp: [u8; 16],
        strands: Vec<HairStrand>,
        material: HairMaterial,
        collision: Vec<ClothCapsule>,
    }

    /// The wire SHAPE is pinned, and the decode consumes **every byte**: a sixth
    /// field appended without bumping `CURRENT_VERSION` fails here.
    #[test]
    fn the_wire_shape_is_pinned_field_for_field() {
        let want = hairstyle();
        let bytes = encode(&want).unwrap();
        let (wire, consumed): (HairAssetV1Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v1 shape decodes the v1 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned five-field shape does not \
             account for — a field was appended to `HairAsset` without bumping \
             `CURRENT_VERSION`"
        );
        assert_eq!(wire.schema_version, HairAsset::CURRENT_VERSION);
        assert_eq!(wire.scalp, want.scalp);
        assert_eq!(wire.strands, want.strands);
        assert_eq!(wire.material, want.material);
        assert_eq!(wire.collision, want.collision);
    }

    #[test]
    fn a_future_payload_is_refused_as_too_new() {
        let mut a = hairstyle();
        a.schema_version = HairAsset::CURRENT_VERSION + 1;
        let bytes = encode(&a).unwrap();
        assert!(matches!(
            decode::<HairAsset>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    /// The asymmetric arm: a payload one field SHORT is a plain `Decode` error and
    /// never a version story (`SchemaTooOld` is unreachable for a v1 kind — there
    /// is no v0 — and inventing one would send a user to fix the wrong problem).
    #[test]
    fn a_short_payload_is_an_unactionable_decode_error_not_a_version_story() {
        #[derive(Serialize)]
        struct HairAssetShort {
            schema_version: u32,
            scalp: [u8; 16],
        }
        let bytes = bincode::serde::encode_to_vec(
            &HairAssetShort {
                schema_version: HairAsset::CURRENT_VERSION,
                scalp: [11; 16],
            },
            inf_asset::bincode_config(),
        )
        .unwrap();
        match decode::<HairAsset>(&bytes) {
            Err(inf_asset::AssetError::Decode(_)) => {}
            other => panic!("expected a plain Decode error, got {other:?}"),
        }
    }

    #[test]
    fn the_upgrade_remedy_says_what_to_do_and_is_readable() {
        let r = HairAsset::UPGRADE_REMEDY;
        assert!(r.contains("Model Editor"), "{r}");
        assert!(r.contains("Guides"), "{r}");
        assert!(
            !r.contains("  "),
            "the remedy carries a run of spaces: {r:?}"
        );
    }
}
