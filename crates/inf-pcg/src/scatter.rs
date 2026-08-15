//! The scatter kernel — deterministic, massively parallel instance placement.
//!
//! `scatter_region` turns a [`DensityField`] + a [`HeightProvider`] into a list
//! of [`PcgInstance`]s within a world region. It is designed for 1M+ instances
//! and **regional/lazy** generation: the result for a region depends only on the
//! cells that region covers, so adjacent regions tile seamlessly and a moving
//! camera can regenerate just the cells it needs.
//!
//! ## Determinism (the P7.0 guard, extended)
//!
//! Placement is a pure function of `(seed, cell, slot)` via
//! [`Hash64`](crate::hash::Hash64) — never a stateful RNG. Cells are mapped in a
//! fixed order through [`inf_core::parallel_map`] (a deterministic in-order pure
//! map) and their per-cell instance lists concatenated in cell order. The output
//! is therefore **byte-identical for any worker-pool size** (unit-tested against
//! `JobPool::new(1/2/8)`).
//!
//! ## Candidate scheme (exact)
//!
//! For each cell `(cx, cz)` covering the region:
//! 1. The per-cell budget is `target = base_density · cell_size²`. If `target ≤ 0`
//!    the cell is empty. Otherwise the cell holds a `g × g` jittered grid with
//!    `g = round(√target).max(1)` (so a full-density cell yields ≈ `target`
//!    instances). Sub-cell size is `sub = cell_size / g`.
//! 2. Each slot `(i, j)` has a centre at `cell_origin + (i+0.5, j+0.5)·sub`,
//!    displaced by a hashed jitter of up to `±0.5·jitter·sub` per axis.
//! 3. The candidate is clipped to the region with a **half-open** `[min, max)`
//!    test (so tiled regions never double-place a border instance).
//! 4. **Density rejection:** a hashed uniform `u ∈ [0, 1)` accepts the slot iff
//!    `u < density(x, z)`. Because `u` does not depend on the density value,
//!    raising density only ever *adds* instances (pointwise-monotone).
//! 5. The terrain height is looked up (`None` → skip); scale, yaw and normal
//!    alignment are all hashed off the same slot, decorrelated by salts.

use glam::{DQuat, DVec3};

use inf_core::JobPool;

use crate::grammar::span::axis_quat;
use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::sampler::DensityField;

// Per-draw salts so scale/jitter/accept/yaw are decorrelated within one slot.
const SALT_JITTER_X: u64 = 0xA1;
const SALT_JITTER_Z: u64 = 0xB2;
const SALT_ACCEPT: u64 = 0xC3;
const SALT_SCALE: u64 = 0xD4;
const SALT_YAW: u64 = 0xE5;

/// Above this `|dot|` with `+Y`, [`tilt_onto`] takes its degenerate branch.
///
/// glam's own `from_rotation_arc` threshold, kept **character for character**
/// (`1 - 2ε`) rather than rounded to something readable, because this function
/// is that one's portable twin and the two must agree about which inputs are
/// degenerate. A looser threshold here would answer `IDENTITY` for near-flat
/// ground that glam gives a real (if tiny) tilt to, and that is committed
/// content, not a rounding.
const ARC_ONE_MINUS_EPS: f64 = 1.0 - 2.0 * f64::EPSILON;

/// The shortest-arc rotation taking `+Y` onto the unit vector `n` — a terrain
/// normal — built without any transcendental.
///
/// # Why not `DQuat::from_rotation_arc`
///
/// Its ordinary branch is already `sqrt`-only, and its **antiparallel** branch
/// is `from_axis_angle`, which is `sin_cos` inside glam where no grep of this
/// crate can see it. The P14 law says libm is not bit-identical across targets,
/// and a scattered instance's rotation is committed content that *both* hosts
/// re-derive independently and that collider placement follows — so a rotation
/// this crate produces may not have a libm call anywhere inside it, including
/// down a branch. This is `inf_anim::ik::rotation_between`'s shape, one crate
/// over, for the same reason.
///
/// The half-turn is taken about `+X`. Any axis perpendicular to `Y` maps `Y` to
/// `-Y`, so the choice is arbitrary — but it must be *fixed*, and `+X` is
/// perpendicular to `Y` for all time, which is why no search is needed. The
/// branch is unreachable from a heightfield in any case: it is a ground normal
/// pointing straight down.
///
/// # A degenerate normal answers the identity rather than a NaN
///
/// The call site used to be `n.normalize()`, which on a zero-length normal is a
/// vector of NaNs, and `from_rotation_arc` carries those straight through into a
/// NaN rotation and from there into an instance transform. `normalize_or_zero`
/// answers `ZERO`, whose dot with `+Y` is `0` and whose cross with it is `ZERO`,
/// so the general branch below builds the exact identity. Stated rather than
/// left to fall out, because "it happens to work" is the kind of claim that
/// stops being true without saying so.
fn tilt_onto(n: DVec3) -> DQuat {
    let n = n.normalize_or_zero();
    let dot = DVec3::Y.dot(n);
    // The comparisons are glam's, strict side for strict side, so that every
    // input takes the same branch here that it took there.
    if dot > ARC_ONE_MINUS_EPS {
        // Already `+Y`: the arc is empty.
        return DQuat::IDENTITY;
    }
    if dot < -ARC_ONE_MINUS_EPS {
        // Antiparallel: a half turn, about `+X` by fixed choice.
        return DQuat::from_xyzw(1.0, 0.0, 0.0, 0.0);
    }
    let c = DVec3::Y.cross(n);
    DQuat::from_xyzw(c.x, c.y, c.z, 1.0 + dot).normalize()
}

/// How a scattered instance's yaw is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RotationMode {
    /// A hashed random yaw about the (possibly tilted) up axis.
    RandomYaw,
    /// No yaw randomization — orientation comes solely from normal alignment.
    AlignNormal,
}

/// A world-space axis-aligned region to scatter within. Only the XZ extent is
/// used (the terrain supplies Y); `min`/`max` are full [`DVec3`]s for ergonomics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub min: DVec3,
    pub max: DVec3,
}

impl Region {
    /// A region from explicit min/max corners.
    pub fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    /// A region from an XZ rectangle (Y bounds are irrelevant to scattering).
    pub fn from_xz(min_x: f64, min_z: f64, max_x: f64, max_z: f64) -> Self {
        Self {
            min: DVec3::new(min_x, f64::NEG_INFINITY, min_z),
            max: DVec3::new(max_x, f64::INFINITY, max_z),
        }
    }
}

/// Knobs for one scatter pass.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScatterParams {
    /// The master seed; every draw hashes off it.
    pub seed: u64,
    /// Cell edge length in world units (the parallel granularity).
    pub cell_size: f64,
    /// Instances per m² at density 1.0 (sets the per-cell candidate budget).
    pub base_density: f64,
    /// Jitter as a fraction `[0, 1]` of a sub-cell's half-extent.
    pub jitter: f64,
    /// Tilt instances so their up axis follows the terrain normal.
    pub align_to_normal: bool,
    /// Inclusive `[min, max]` uniform scale range.
    pub scale_range: (f64, f64),
    /// How yaw is chosen.
    pub rotation: RotationMode,
    /// Vertical offset added to the sampled ground height.
    pub altitude_offset: f64,
}

impl Default for ScatterParams {
    fn default() -> Self {
        Self {
            seed: 0,
            cell_size: 32.0,
            base_density: 0.1,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (1.0, 1.0),
            rotation: RotationMode::RandomYaw,
            altitude_offset: 0.0,
        }
    }
}

/// One placed instance. `pos` is a world-space [`DVec3`] (architecture rule 3);
/// `kind_index` is resolved by the rule layer (0 from the bare kernel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgInstance {
    pub pos: DVec3,
    pub rotation: DQuat,
    pub scale: f64,
    pub kind_index: u32,
}

/// One placed **collision box** — the solid half of a placement, beside
/// [`PcgInstance`]'s visible half (P19.5).
///
/// A scatter pass never produces one: scattered foliage is decoration, and
/// making a million grass blades solid is not a feature anybody asked for. They
/// come from grammar modules that declare `collider hx,hy,hz` and from a
/// building's own structure (slabs, stairs, lintels), which is what turns
/// "a wall was drawn here" into "you cannot walk through here".
///
/// The rotation is a **yaw-only quaternion**, the same shape
/// [`Frame::rotation`](crate::grammar::span::Frame::rotation) produces, and it is
/// stored as a quaternion rather than as an angle **precisely so no `atan2` is
/// ever needed**: recovering an angle from the frame would put `std` trig on the
/// committed-placement path, which the P14 law forbids. Every query this type
/// answers about its own orientation is a rational function of the quaternion's
/// components (see [`xz_half_extents`](Self::xz_half_extents)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgCollider {
    /// World-space centre of the box.
    pub center: DVec3,
    /// Half-extents in metres, in the box's own (rotated) frame.
    pub half_extents: DVec3,
    /// Yaw-only orientation.
    pub rotation: DQuat,
}

impl PcgCollider {
    /// The XZ half-extents of this box's **axis-aligned bounds**.
    ///
    /// No trigonometry: for a yaw-only quaternion `(0, s, 0, c)` the rotation
    /// angle satisfies `cos θ = 1 − 2s²` and `sin θ = 2sc`, both exact rational
    /// expressions in the stored components — and both *exactly* `1` and `0` for
    /// the identity, which is the overwhelmingly common case (a v1 building's
    /// walls all run along a world axis). An `atan2` here would be `std` trig on
    /// committed data and would also lose that exactness.
    pub fn xz_half_extents(&self) -> (f64, f64) {
        let (s, c) = (self.rotation.y, self.rotation.w);
        let (cos_t, sin_t) = ((1.0 - 2.0 * s * s).abs(), (2.0 * s * c).abs());
        (
            self.half_extents.x * cos_t + self.half_extents.z * sin_t,
            self.half_extents.x * sin_t + self.half_extents.z * cos_t,
        )
    }

    /// The `[min, max]` world-Y band this box occupies.
    #[inline]
    pub fn y_band(&self) -> (f64, f64) {
        (
            self.center.y - self.half_extents.y,
            self.center.y + self.half_extents.y,
        )
    }
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Scatter within `region` on the process-wide job pool.
pub fn scatter_region(
    params: &ScatterParams,
    density: &dyn DensityField,
    height: &dyn HeightProvider,
    region: Region,
) -> Vec<PcgInstance> {
    scatter_region_in(inf_core::global(), params, density, height, region)
}

/// Scatter within `region` on a caller-supplied pool. The result is independent
/// of the pool's thread count — this is the seam the determinism guard uses.
pub fn scatter_region_in(
    pool: &JobPool,
    params: &ScatterParams,
    density: &dyn DensityField,
    height: &dyn HeightProvider,
    region: Region,
) -> Vec<PcgInstance> {
    let cs = params.cell_size;
    if cs <= 0.0 || region.max.x <= region.min.x || region.max.z <= region.min.z {
        return Vec::new();
    }

    // Cells covering the region, in a fixed (cx-major, cz-minor) order.
    let cx0 = (region.min.x / cs).floor() as i64;
    let cx1 = (region.max.x / cs).ceil() as i64;
    let cz0 = (region.min.z / cs).floor() as i64;
    let cz1 = (region.max.z / cs).ceil() as i64;
    let mut cells = Vec::with_capacity(((cx1 - cx0).max(0) * (cz1 - cz0).max(0)) as usize);
    for cx in cx0..cx1 {
        for cz in cz0..cz1 {
            cells.push((cx, cz));
        }
    }

    // Deterministic in-order parallel map, then concatenate in cell order.
    let per_cell: Vec<Vec<PcgInstance>> = pool.parallel_map(cells, |(cx, cz)| {
        scatter_cell(params, density, height, region, cx, cz)
    });
    per_cell.into_iter().flatten().collect()
}

/// Scatter one cell. Pure in `(params, cx, cz)` plus the (pure) field/terrain.
fn scatter_cell(
    params: &ScatterParams,
    density: &dyn DensityField,
    height: &dyn HeightProvider,
    region: Region,
    cx: i64,
    cz: i64,
) -> Vec<PcgInstance> {
    let cs = params.cell_size;
    let target = params.base_density * cs * cs;
    if target <= 0.0 {
        return Vec::new();
    }
    let g = (target.sqrt().round() as i64).max(1);
    let sub = cs / g as f64;
    let cell_x = cx as f64 * cs;
    let cell_z = cz as f64 * cs;

    let mut out = Vec::new();
    for i in 0..g {
        for j in 0..g {
            let slot = Hash64::new(params.seed)
                .mix_i64(cx)
                .mix_i64(cz)
                .mix_i64(i)
                .mix_i64(j);

            let jx = (slot.mix_u64(SALT_JITTER_X).unit() - 0.5) * params.jitter * sub;
            let jz = (slot.mix_u64(SALT_JITTER_Z).unit() - 0.5) * params.jitter * sub;
            let x = cell_x + (i as f64 + 0.5) * sub + jx;
            let z = cell_z + (j as f64 + 0.5) * sub + jz;

            // Half-open region clip → seamless tiling, no double placement.
            if x < region.min.x || x >= region.max.x || z < region.min.z || z >= region.max.z {
                continue;
            }

            // Density rejection: u independent of density → monotone in density.
            let u = slot.mix_u64(SALT_ACCEPT).unit();
            if u >= density.density(x, z) {
                continue;
            }

            // Terrain lookup — no ground here means no instance.
            let h = match height.height(x, z) {
                Some(h) => h,
                None => continue,
            };
            let normal = height.normal(x, z);
            let pos = DVec3::new(x, h + params.altitude_offset, z);

            let scale = lerp(
                params.scale_range.0,
                params.scale_range.1,
                slot.mix_u64(SALT_SCALE).unit(),
            );

            let tilt = if params.align_to_normal {
                normal.map(tilt_onto).unwrap_or(DQuat::IDENTITY)
            } else {
                DQuat::IDENTITY
            };
            let rotation = match params.rotation {
                RotationMode::RandomYaw => {
                    let yaw = slot.mix_u64(SALT_YAW).unit() * std::f64::consts::TAU;
                    tilt * axis_quat(DVec3::Y, yaw)
                }
                RotationMode::AlignNormal => tilt,
            };

            out.push(PcgInstance {
                pos,
                rotation,
                scale,
                kind_index: 0,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::height::FnHeight;
    use crate::noise::ValueNoise;
    use crate::sampler::Constant;
    use crate::sampler::Noise;

    fn flat() -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(|_, _| Some(0.0))
    }

    fn params(base_density: f64) -> ScatterParams {
        ScatterParams {
            seed: 1234,
            cell_size: 16.0,
            base_density,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.8, 1.2),
            rotation: RotationMode::RandomYaw,
            altitude_offset: 0.0,
        }
    }

    #[test]
    fn deterministic_across_pool_sizes() {
        let p = params(0.5);
        let d = Constant(0.7);
        let h = flat();
        let region = Region::from_xz(0.0, 0.0, 256.0, 256.0);

        let a = scatter_region_in(&JobPool::new(1), &p, &d, &h, region);
        let b = scatter_region_in(&JobPool::new(2), &p, &d, &h, region);
        let c = scatter_region_in(&JobPool::new(8), &p, &d, &h, region);

        assert!(!a.is_empty());
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn density_zero_is_empty() {
        let p = params(1.0);
        let d = Constant(0.0);
        let h = flat();
        let out = scatter_region(&p, &d, &h, Region::from_xz(0.0, 0.0, 128.0, 128.0));
        assert!(out.is_empty());
    }

    #[test]
    fn density_scaling_is_monotone() {
        // Same seed/positions; higher density accepts a superset → count grows.
        let h = flat();
        let region = Region::from_xz(0.0, 0.0, 256.0, 256.0);
        let lo = scatter_region(&params(0.5), &Constant(0.25), &h, region).len();
        let hi = scatter_region(&params(0.5), &Constant(0.75), &h, region).len();
        assert!(hi > lo, "expected {hi} > {lo}");
    }

    #[test]
    fn base_density_scales_count() {
        let h = flat();
        let region = Region::from_xz(0.0, 0.0, 256.0, 256.0);
        let sparse = scatter_region(&params(0.05), &Constant(1.0), &h, region).len();
        let dense = scatter_region(&params(1.0), &Constant(1.0), &h, region).len();
        assert!(dense > sparse * 4, "sparse={sparse} dense={dense}");
    }

    #[test]
    fn instances_lie_within_region_and_on_ground() {
        let p = params(0.5);
        let region = Region::from_xz(-40.0, 10.0, 60.0, 90.0);
        let out = scatter_region(&p, &Constant(1.0), &flat(), region);
        assert!(!out.is_empty());
        for inst in &out {
            assert!(
                inst.pos.x >= region.min.x && inst.pos.x < region.max.x,
                "{:?}",
                inst.pos
            );
            assert!(
                inst.pos.z >= region.min.z && inst.pos.z < region.max.z,
                "{:?}",
                inst.pos
            );
            assert_eq!(inst.pos.y, 0.0);
            assert!(inst.scale >= 0.8 && inst.scale <= 1.2);
        }
    }

    #[test]
    fn no_jitter_places_on_subcell_centres() {
        let mut p = params(1.0);
        p.jitter = 0.0;
        let out = scatter_region(
            &p,
            &Constant(1.0),
            &flat(),
            Region::from_xz(0.0, 0.0, 16.0, 16.0),
        );
        // With jitter 0 and cell_size 16, target = 256 → g = 16 → sub = 1.0;
        // centres sit at x = i + 0.5. Every coordinate must have fract 0.5.
        for inst in &out {
            assert!(
                (inst.pos.x.fract().abs() - 0.5).abs() < 1e-9,
                "{}",
                inst.pos.x
            );
            assert!(
                (inst.pos.z.fract().abs() - 0.5).abs() < 1e-9,
                "{}",
                inst.pos.z
            );
        }
    }

    #[test]
    fn jitter_stays_within_subcell() {
        let p = params(1.0); // jitter 1.0, g = 16, sub = 1.0
        let out = scatter_region(
            &p,
            &Constant(1.0),
            &flat(),
            Region::from_xz(0.0, 0.0, 16.0, 16.0),
        );
        let sub = 1.0;
        for inst in &out {
            // Nearest sub-cell centre (…+0.5). Distance from it ≤ 0.5·jitter·sub.
            let nx = (inst.pos.x - 0.5).round() + 0.5;
            let nz = (inst.pos.z - 0.5).round() + 0.5;
            assert!(
                (inst.pos.x - nx).abs() <= 0.5 * sub + 1e-9,
                "x off: {}",
                inst.pos.x
            );
            assert!(
                (inst.pos.z - nz).abs() <= 0.5 * sub + 1e-9,
                "z off: {}",
                inst.pos.z
            );
        }
    }

    #[test]
    fn align_to_normal_tilts_on_slope() {
        // 45° ramp; with alignment the up axis should follow the normal.
        let mut p = params(1.0);
        p.align_to_normal = true;
        p.rotation = RotationMode::AlignNormal;
        let ramp = FnHeight::new(|x, _| Some(x));
        let out = scatter_region(
            &p,
            &Constant(1.0),
            &ramp,
            Region::from_xz(0.0, 0.0, 16.0, 16.0),
        );
        assert!(!out.is_empty());
        let up = out[0].rotation * DVec3::Y;
        // The instance up must lean off world-up on the ramp.
        assert!(up.dot(DVec3::Y) < 0.99, "up={up:?}");
    }

    #[test]
    fn scatter_100k_smoke() {
        // A sanity throughput check (no hard perf assert): ~100k instances.
        let p = ScatterParams {
            seed: 99,
            cell_size: 32.0,
            base_density: 1.0,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (1.0, 1.0),
            rotation: RotationMode::RandomYaw,
            altitude_offset: 0.0,
        };
        let region = Region::from_xz(0.0, 0.0, 320.0, 320.0); // ~102k m²
        let out = scatter_region(&p, &Noise(ValueNoise::default()), &flat(), region);
        // Noise density averages ~0.5, so expect a large-but-not-full population.
        assert!(out.len() > 20_000, "only {} instances", out.len());
    }

    // ── portable placement (Hardening Wave C, L6.F4) ─────────────────────────

    /// The portable yaw agrees with `DQuat::from_rotation_y` everywhere.
    ///
    /// A tolerance and not a bit compare, because it is a *different* function:
    /// `psin64`/`pcos64` are polynomials accurate to ~1e-7 and `sin_cos` is
    /// libm. The point is not that they agree bit for bit — that is impossible
    /// and would defeat the purpose — but that the portable one is the same
    /// rotation to well inside anything a placement can express, while being a
    /// function of its argument on every target rather than of the C library.
    #[test]
    fn the_scatter_yaw_matches_glams_rotation_y() {
        for step in 0..720u32 {
            let yaw = std::f64::consts::TAU * step as f64 / 720.0;
            let got = axis_quat(DVec3::Y, yaw);
            let want = DQuat::from_rotation_y(yaw);
            // Quaternions double-cover: compare the rotated basis, not the
            // components.
            for basis in [DVec3::X, DVec3::Z] {
                let (a, b) = (got * basis, want * basis);
                assert!((a - b).length() < 1e-7, "yaw={yaw}: {a:?} vs {b:?}");
            }
            assert!((got.length() - 1.0).abs() < 1e-12, "not unit at yaw={yaw}");
        }
        // An unrotated instance is EXACTLY unrotated — the short-circuit the
        // `axis_quat` doc explains, worth 5.63e-8 of residual tilt otherwise.
        assert_eq!(axis_quat(DVec3::Y, 0.0), DQuat::IDENTITY);
    }

    /// The portable tilt is **bit-identical** to `DQuat::from_rotation_arc` on
    /// every input a heightfield can produce.
    ///
    /// It can be a bit compare, unlike the yaw above, because glam's ordinary
    /// branch is already `sqrt`-only and this reproduces it operation for
    /// operation; only the antiparallel branch differed, and only that branch
    /// called libm. So the arm says two things at once: no committed placement
    /// moves, and the branch that used to reach `sin_cos` is the only one that
    /// behaves differently.
    ///
    /// The reference is fed `raw.normalize()` because that is literally what the
    /// call site used to write — `from_rotation_arc(Y, n.normalize())` — and the
    /// comparison has to be of two expressions over one input, not of two
    /// functions over two slightly different unit vectors.
    #[test]
    fn the_scatter_tilt_matches_glams_rotation_arc_bit_for_bit() {
        let mut checked = 0u32;
        for i in 0..40u32 {
            for j in 0..40u32 {
                // Slopes up to ~57°, far past anything walkable, and never
                // exactly flat (integer `i` cannot land on 19.5).
                let (sx, sz) = (i as f64 * 0.08 - 1.56, j as f64 * 0.08 - 1.56);
                let raw = DVec3::new(-sx, 1.0, -sz);
                let got = tilt_onto(raw);
                let want = DQuat::from_rotation_arc(DVec3::Y, raw.normalize());
                assert_eq!(
                    got.to_array().map(f64::to_bits),
                    want.to_array().map(f64::to_bits),
                    "normal {raw:?}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 1600, "the sweep shrank");

        // The two ends of the arc, exactly.
        assert_eq!(tilt_onto(DVec3::Y), DQuat::IDENTITY);
        assert_eq!(
            tilt_onto(DVec3::NEG_Y) * DVec3::Y,
            DVec3::NEG_Y,
            "the antiparallel branch must still turn +Y onto -Y"
        );
        // A degenerate normal answers the identity rather than the NaN rotation
        // `n.normalize()` used to hand `from_rotation_arc`.
        assert_eq!(tilt_onto(DVec3::ZERO), DQuat::IDENTITY);
        assert_eq!(tilt_onto(DVec3::splat(f64::NAN)), DQuat::IDENTITY);
    }
}
