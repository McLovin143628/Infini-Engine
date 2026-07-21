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

use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::sampler::DensityField;

// Per-draw salts so scale/jitter/accept/yaw are decorrelated within one slot.
const SALT_JITTER_X: u64 = 0xA1;
const SALT_JITTER_Z: u64 = 0xB2;
const SALT_ACCEPT: u64 = 0xC3;
const SALT_SCALE: u64 = 0xD4;
const SALT_YAW: u64 = 0xE5;

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
                normal
                    .map(|n| DQuat::from_rotation_arc(DVec3::Y, n.normalize()))
                    .unwrap_or(DQuat::IDENTITY)
            } else {
                DQuat::IDENTITY
            };
            let rotation = match params.rotation {
                RotationMode::RandomYaw => {
                    let yaw = slot.mix_u64(SALT_YAW).unit() * std::f64::consts::TAU;
                    tilt * DQuat::from_rotation_y(yaw)
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
}
