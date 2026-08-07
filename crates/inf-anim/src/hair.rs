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
//!   set of scalp roots — and then simulated. There is no interactive comb, so
//!   "authored in the Model Editor" is true of the generator and not of a brush.
//! * **No interpolated render strands and no cards.** What draws is the guides
//!   themselves, as ribbons. A hairstyle is therefore as dense as its guide count,
//!   and the lower-tier card path the phase plan names is not built.
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
    /// hair, not one fibre: a guide stands for the strands interpolated around it,
    /// and v1 draws the guide.
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
    pub fn grow(
        scalp: [u8; 16],
        roots: &[HairRoot],
        length_m: f32,
        segments: u16,
        material: HairMaterial,
    ) -> Result<Self, ClothError> {
        if roots.is_empty() || segments == 0 || !length_m.is_finite() || length_m <= 0.0 {
            return Err(ClothError::Degenerate {
                vertices: roots.len(),
                indices: segments as usize,
            });
        }
        let seg_len = length_m / segments as f32;
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
            let points: Vec<[f32; 3]> = (0..=segments)
                .map(|k| (origin + dir * (seg_len * k as f32)).to_array())
                .collect();
            strands.push(HairStrand {
                root_joint: r.joint,
                root_offset: r.offset,
                rest_m: vec![seg_len; segments as usize],
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
pub fn ribbon_mesh(asset: &HairAsset, state: &HairState) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(state.len() * 2);
    let mut idx: Vec<u32> = Vec::new();
    let half = (asset.material.ribbon_width_m.max(0.0)) * 0.5;
    for s in 0..state.strand_count() {
        let (from, to) = (state.starts[s] as usize, state.starts[s + 1] as usize);
        let count = to - from;
        if count < 2 {
            continue;
        }
        let base = pos.len() as u32;
        let mut axis = RIBBON_FALLBACK;
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
            } else if tangent.length() > MIN_CONSTRAINT_LEN_M {
                // The strand runs along the reference: a fixed fallback, so the
                // degenerate case is deterministic across runs and hosts.
                axis = RIBBON_FALLBACK;
            }
            let taper = 1.0 - (k as f32 / (count - 1) as f32);
            let w = axis * (half * taper);
            pos.push((p - w).to_array());
            pos.push((p + w).to_array());
        }
        for k in 0..count as u32 - 1 {
            let (a, b) = (base + k * 2, base + k * 2 + 1);
            let (c, d) = (a + 2, b + 2);
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (pos, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// Four strands hanging straight down from joint 1, 20 cm long, 4 segments.
    fn hairstyle() -> HairAsset {
        let roots: Vec<HairRoot> = (0..4)
            .map(|i| HairRoot {
                joint: 1,
                offset: [i as f32 * 0.02 - 0.03, 0.0, 0.0],
                direction: [0.0, -1.0, 0.0],
            })
            .collect();
        HairAsset::grow(
            [11; 16],
            &roots,
            0.2,
            4,
            HairMaterial {
                // A stiff, heavily damped default would settle before the drape
                // arm could see anything move; these are the shipped defaults.
                ..HairMaterial::default()
            },
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
                }],
                0.1,
                2,
                HairMaterial::default(),
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
            HairAsset::grow([0; 16], &[], 0.1, 2, HairMaterial::default()),
            Err(ClothError::Degenerate { .. })
        ));
        assert!(matches!(
            bad([0.0, -1.0, 0.0]).map(|a| HairAsset::grow([0; 16], &[], 0.0, 0, a.material)),
            Ok(Err(ClothError::Degenerate { .. }))
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
