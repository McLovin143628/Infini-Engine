//! Dynamic global-illumination math (P13.3b, **rebuilt in P18.4**): the pure,
//! GPU-free half of the real-time GI.
//!
//! The layout of the camera-centred voxel volume and the probe grid, the
//! deterministic golden-spiral ray directions, the L1 spherical-harmonic basis and
//! its radiance/dominant-direction reconstruction, the macro-cell binning that
//! lifted the instance cap, the temporal probe schedule, and the terrain column
//! sampler all live here as pure functions so they unit-test without a device and
//! stay bit-identical to the compute shaders ([`crate::passes::gi`],
//! `shaders/gi_*.wgsl`, `shaders/env_lighting.wgsl`) that mirror them.
//!
//! ## The scheme (what the shaders implement)
//!
//! 1. **Voxelize** a [`GiQuality::voxel_dim`]³ volume centred on the camera,
//!    covering [`crate::GiSettings::extent`] metres. Each voxel stores an albedo +
//!    binary occupancy **and** an injected emissive radiance (two packed `u32`s in
//!    a storage buffer — portable, no 3D storage-texture feature).
//!
//!    Since P18.4 the voxelizer sees the **whole scene**, not just rigid boxes:
//!    * rigid [`MeshInstance`](crate::MeshInstance)s as oriented boxes,
//!    * **skinned** instances as per-joint boxes (the bind-space AABB of each
//!      joint's dominant vertices, carried by the live skinning palette),
//!    * **vgeom** instances as the per-meshlet spheres of the always-resident root
//!      page (the coarsest cut — deterministic whatever the streamer has paged in),
//!    * **terrain** as a per-column height + splat-blended albedo sampled from the
//!      resident tiles ([`sample_terrain_column`]).
//!
//!    The old `MAX_GI_INSTANCES = 256` silent truncation is gone: primitives are
//!    ordered nearest-volume-centre-first ([`priority_order`]), clipped to a
//!    per-frame budget whose overflow is **reported** ([`GiAudit`]), and binned into
//!    [`MACRO_DIM`]³ macro cells ([`bin_macro_cells`]) so a voxel only tests the
//!    primitives that can reach it. That is what makes thousands of primitives
//!    affordable in a *gather* (and therefore deterministic) voxelizer.
//! 2. **March** the [`GiQuality::probe_dims`] probe grid through it: each probe
//!    casts `rays` fixed golden-spiral directions; a ray that hits occupancy gathers
//!    `albedo × sun_visibility(hit) + emissive`, a ray that misses gathers **the
//!    P17.2 sky-view LUT** in that direction (falling back to the authored gradient
//!    for a scene with no atmosphere). The result is projected to **L1 SH** (4
//!    coeffs × RGB) per probe. Probe updates may be **amortized** across frames on a
//!    deterministic round-robin ([`ProbeSchedule`]).
//! 3. **Sample** in the lit passes: the ambient term becomes the trilinearly
//!    probe-interpolated `SH-evaluate(normal)` (× intensity × SSAO), and — new in
//!    P18.4 — a **specular** term reconstructs radiance along the reflection vector
//!    ([`sh_radiance`]), optionally re-anchored at a screen-space ray hit (SSR v1).

use glam::Vec3;

use crate::caps::RenderTier;
use crate::scene::{RenderTerrain, RenderTerrainLayer};

/// Voxel grid resolution per axis at [`GiQuality::High`] (64³ volume). Kept as a
/// free constant because it is the dimension every pre-P18.4 caller and golden
/// rendered with.
pub const GI_DIM: u32 = 64;
/// Probe grid dimensions `[x, y, z]` at [`GiQuality::High`] (16×8×16 = 2048
/// probes). Fewer probes vertically since scenes are wider than tall.
pub const PROBE_DIMS: [u32; 3] = [16, 8, 16];

/// Macro-cell edge in **voxels**: the voxel grid is partitioned into
/// `(dim/MACRO_DIM)³` cells, and each cell carries the list of primitives whose
/// bounds touch it.
///
/// This is the whole reason the 256-instance cap could be lifted. The voxelizer is
/// a *gather* (one thread per voxel, first hit wins) because a scatter would race
/// on the voxel word and race means nondeterminism — and a gather over an unbounded
/// instance list is `O(voxels × instances)`, which is why v1 had a cap at all.
/// Binning makes it `O(voxels × instances-near-that-voxel)`.
///
/// 8 divides every tier's voxel dimension (64 / 48 / 32), which is checked by
/// `macro_dim_divides_every_tier`.
pub const MACRO_DIM: u32 = 8;

/// Largest emissive radiance the voxel volume can carry per channel. Emissive is
/// stored as an RGBA8 word — `rgb` = the colour normalized by its own maximum
/// component, `a` = that maximum divided by this ceiling — so the quantization
/// error is relative rather than absolute and a dim emissive keeps its hue.
/// Mirrors `GI_EMISSIVE_MAX` in `gi_voxelize.wgsl` / `gi_probes.wgsl`.
pub const EMISSIVE_MAX: f32 = 16.0;

/// GI cost tier (P18.4) — the voxel/probe resolution and the per-frame primitive
/// budget. Mirrors [`crate::AtmosphereQuality`]'s shape: authored on
/// [`crate::GiSettings`], clamped **down** by
/// [`RenderTier::apply`](crate::RenderTier::apply), never up.
///
/// [`High`](GiQuality::High) is exactly the pre-P18.4 geometry (64³ / 16×8×16), so
/// a default-settings GI render is unchanged by the tiering itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum GiQuality {
    /// 32³ voxels, 8×4×8 = 256 probes, 512 primitives/frame.
    Low,
    /// 48³ voxels, 12×6×12 = 864 probes, 2048 primitives/frame.
    Medium,
    /// 64³ voxels, 16×8×16 = 2048 probes, 4096 primitives/frame — the pre-P18.4
    /// geometry.
    #[default]
    High,
}

impl GiQuality {
    /// Voxels per axis.
    pub fn voxel_dim(self) -> u32 {
        match self {
            GiQuality::Low => 32,
            GiQuality::Medium => 48,
            GiQuality::High => GI_DIM,
        }
    }

    /// Probe grid dimensions `[x, y, z]`.
    pub fn probe_dims(self) -> [u32; 3] {
        match self {
            GiQuality::Low => [8, 4, 8],
            GiQuality::Medium => [12, 6, 12],
            GiQuality::High => PROBE_DIMS,
        }
    }

    /// Macro cells per axis (`voxel_dim / MACRO_DIM`).
    pub fn macro_dim(self) -> u32 {
        self.voxel_dim() / MACRO_DIM
    }

    /// Ceiling on primitives voxelized per frame. A scene with more spills into
    /// [`GiAudit::dropped`] rather than vanishing silently.
    pub fn instance_budget(self) -> usize {
        match self {
            GiQuality::Low => 512,
            GiQuality::Medium => 2048,
            GiQuality::High => 4096,
        }
    }

    /// Clamp **down** to what a render tier can afford. Never raises quality, so
    /// composing it with a caller's setting can only ever cost less.
    pub fn clamp_to(self, tier: RenderTier) -> Self {
        let ceiling = match tier {
            RenderTier::High => GiQuality::High,
            RenderTier::Medium => GiQuality::Medium,
            // GI is switched off entirely on Low ([`RenderTier::apply`]); the
            // clamp is still defined so the mapping is total.
            RenderTier::Low => GiQuality::Low,
        };
        self.min(ceiling)
    }
}

/// Total probe count at [`GiQuality::High`] (the pre-P18.4 constant).
pub const fn probe_count() -> u32 {
    PROBE_DIMS[0] * PROBE_DIMS[1] * PROBE_DIMS[2]
}

/// Total probe count for an arbitrary probe grid.
pub fn probe_count_of(dims: [u32; 3]) -> u32 {
    dims[0] * dims[1] * dims[2]
}

/// Flat probe index for grid coordinate `(x, y, z)` (`x` fastest) in `dims`.
/// Mirrors the compute shader's `probe_index`.
pub fn probe_index_in(x: u32, y: u32, z: u32, dims: [u32; 3]) -> u32 {
    (z * dims[1] + y) * dims[0] + x
}

/// [`probe_index_in`] at [`PROBE_DIMS`].
pub fn probe_index(x: u32, y: u32, z: u32) -> u32 {
    probe_index_in(x, y, z, PROBE_DIMS)
}

/// Flat voxel index for `(x, y, z)` in a `dim³` grid (`x` fastest). Mirrors the
/// compute shader's `voxel_index`.
pub fn voxel_index_in(x: u32, y: u32, z: u32, dim: u32) -> u32 {
    (z * dim + y) * dim + x
}

/// [`voxel_index_in`] at [`GI_DIM`].
pub fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    voxel_index_in(x, y, z, GI_DIM)
}

/// Flat macro-cell index for cell `(x, y, z)` in a `macro_dim³` grid (`x`
/// fastest). Mirrors `macro_index` in `gi_voxelize.wgsl`.
pub fn macro_index(x: u32, y: u32, z: u32, macro_dim: u32) -> u32 {
    (z * macro_dim + y) * macro_dim + x
}

/// The `i`-th of `n` golden-spiral (Fibonacci-sphere) unit directions — an even,
/// deterministic spread over the sphere with no temporal jitter (v1 determinism).
/// Mirrors `spiral_dir` in `shaders/gi_probes.wgsl`.
pub fn golden_spiral_dir(i: u32, n: u32) -> Vec3 {
    let n = n.max(1) as f32;
    let i = i as f32;
    // Golden angle.
    let phi = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let y = 1.0 - 2.0 * (i + 0.5) / n; // (-1, 1)
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = phi * i;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

/// Real L1 spherical-harmonic basis evaluated in direction `d`
/// `[Y₀₀, Y₁₋₁, Y₁₀, Y₁₁]` = `[0.282095, 0.488603·y, 0.488603·z, 0.488603·x]`.
/// Mirrors `sh_basis` in the shaders.
pub fn sh_l1_basis(d: Vec3) -> [f32; 4] {
    [0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x]
}

/// Reconstruct **radiance** along direction `d` from an L1 SH triple, sharpened by
/// `lobe` (`1` = the full directional reconstruction, `0` = the direction-less DC
/// term alone — what a fully rough surface sees). Mirrors `gi_sh_radiance` in
/// `shaders/env_lighting.wgsl`.
///
/// The projection in `gi_probes.wgsl` normalizes by `4π / rays`, which makes this
/// an *identity* for a constant field: a probe that saw uniform radiance `L`
/// reconstructs exactly `L` in every direction, at every `lobe`. That property is
/// what lets the specular term reduce to the old constant ambient specular
/// (`pre_p18_4_ambient_specular_is_the_uniform_field_limit`) instead of being a
/// free-floating new light source.
pub fn sh_radiance(coeffs: &[[f32; 3]; 4], d: Vec3, lobe: f32) -> [f32; 3] {
    let b = sh_l1_basis(d);
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let dc = coeffs[0][c] * b[0];
        let dir = coeffs[1][c] * b[1] + coeffs[2][c] * b[2] + coeffs[3][c] * b[3];
        *o = (dc + lobe * dir).max(0.0);
    }
    out
}

/// The **dominant light direction** of an L1 SH triple: the direction the linear
/// band points, using per-channel luminance so a coloured bounce still resolves to
/// one direction. Mirrors `gi_sh_dominant_dir` in `shaders/env_lighting.wgsl`.
///
/// Returns `Vec3::ZERO` for a field with no linear band at all (a perfectly
/// uniform environment has no dominant direction, and pretending otherwise would
/// invent a highlight out of a constant).
pub fn sh_dominant_direction(coeffs: &[[f32; 3]; 4]) -> Vec3 {
    // Rec. 709 luma of each linear-band coefficient.
    let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    // Basis order is [Y00, y, z, x] — so the vector is (c3, c1, c2).
    let v = Vec3::new(luma(coeffs[3]), luma(coeffs[1]), luma(coeffs[2]));
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        Vec3::ZERO
    }
}

/// Karis' analytic split-sum environment BRDF: returns `(a, b)` such that the
/// specular response of a surface with reflectance `f0` is `f0·a + b`. Mirrors
/// `gi_env_brdf_ab` in `shaders/env_lighting.wgsl`.
pub fn env_brdf_ab(rough: f32, n_dot_v: f32) -> (f32, f32) {
    let rough = rough.clamp(0.0, 1.0);
    let nov = n_dot_v.clamp(0.0, 1.0);
    let (c0x, c0y, c0z, c0w) = (-1.0f32, -0.0275, -0.572, 0.022);
    let (c1x, c1y, c1z, c1w) = (1.0f32, 0.0425, 1.04, -0.04);
    let (rx, ry, rz, rw) = (
        rough * c0x + c1x,
        rough * c0y + c1y,
        rough * c0z + c1z,
        rough * c0w + c1w,
    );
    let a004 = (rx * rx).min((-9.28 * nov).exp2()) * rx + ry;
    (-1.04 * a004 + rz, 1.04 * a004 + rw)
}

/// The render-local minimum corner of the camera-centred volume: `eye − extent/2`
/// on every axis (the voxel grid and the probe grid share it).
pub fn volume_min(eye_local: Vec3, extent: f32) -> Vec3 {
    eye_local - Vec3::splat(extent * 0.5)
}

/// World size of one voxel at [`GI_DIM`] (`extent / 64`).
pub fn voxel_size(extent: f32) -> f32 {
    extent / GI_DIM as f32
}

/// World size of one voxel for an arbitrary grid dimension.
pub fn voxel_size_of(extent: f32, dim: u32) -> f32 {
    extent / dim.max(1) as f32
}

/// Render-local position of probe `(x, y, z)` in `dims`: probes sit at the **cell
/// corners** spanning the whole extent (so the outermost probes are on the volume
/// faces).
pub fn probe_position_in(
    x: u32,
    y: u32,
    z: u32,
    dims: [u32; 3],
    vol_min: Vec3,
    extent: f32,
) -> Vec3 {
    let frac = |i: u32, n: u32| {
        if n <= 1 {
            0.5
        } else {
            i as f32 / (n - 1) as f32
        }
    };
    vol_min + Vec3::new(frac(x, dims[0]), frac(y, dims[1]), frac(z, dims[2])) * extent
}

/// [`probe_position_in`] at [`PROBE_DIMS`].
pub fn probe_position(x: u32, y: u32, z: u32, vol_min: Vec3, extent: f32) -> Vec3 {
    probe_position_in(x, y, z, PROBE_DIMS, vol_min, extent)
}

// ── temporal probe amortization (P18.4) ─────────────────────────────────────

/// The deterministic round-robin probe-update schedule.
///
/// **Frame-index-free by construction.** The cursor is renderer state that advances
/// by exactly the number of probes actually written, and it is reset (to `0`) when
/// the GI configuration or the probe geometry changes — **not** when the scene's
/// content does (see `GiSweepKey` in `passes::gi` for why that reset had to go).
/// So:
///
/// * two *cold* renders of the same content produce the same slice sequence (both
///   start from a fresh cursor at `0`), and
/// * once a sweep has completed, a static scene is in a **converged steady state**
///   whose probe buffer is a function of the content alone — every probe has been
///   written from the same voxel volume — so it is byte-identical across runs *and*
///   byte-identical to a full-update render.
///
/// Deriving the slice from the *frame index* instead would break the first property
/// the moment a host rendered a warm-up frame; that is why the cursor is state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProbeSchedule {
    /// Index of the next probe to update.
    cursor: u32,
}

impl ProbeSchedule {
    /// A fresh (cold) schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// The current cursor (for tests + the audit report).
    pub fn cursor(self) -> u32 {
        self.cursor
    }

    /// Restart the sweep — called when the GI settings, the probe geometry, the
    /// volume generation or the (bucketed) sun changes, i.e. when the probes'
    /// previous integration cannot be aged into the new one at all.
    ///
    /// **Not** called on a scene-content change: the cursor wraps, so every probe
    /// is revisited within one sweep and staleness is already bounded without a
    /// reset. Resetting there made amortization a no-op in the shipped player,
    /// whose `scene.version` moves every frame — see `GiSweepKey`.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Take the next slice: `(start, count)` over `total` probes, updating at most
    /// `budget` of them. `budget == 0` (or `budget >= total`) means **full update**
    /// — every probe, cursor parked at `0` — which is what the goldens and the
    /// determinism gates render with.
    ///
    /// The slice wraps: `start + count` may exceed `total`, and the shader takes
    /// `(start + i) % total`. Wrapping rather than clamping keeps every probe on the
    /// same period, so no probe is ever starved by a `total` that is not a multiple
    /// of `budget`.
    pub fn next(&mut self, total: u32, budget: u32) -> (u32, u32) {
        if total == 0 {
            self.cursor = 0;
            return (0, 0);
        }
        if budget == 0 || budget >= total {
            self.cursor = 0;
            return (0, total);
        }
        let start = self.cursor % total;
        self.cursor = (start + budget) % total;
        (start, budget)
    }
}

/// Lattice divisions per unit for [`sun_bucket`]'s direction quantization: two
/// unit directions land in the same bucket only if every component agrees to
/// within `1/200`, which bounds the in-bucket angle at `√3 / 200 rad ≈ 0.50°`.
///
/// Chosen against the clock rather than against the shader: the sun sweeps 15°
/// per hour, so at `rate = 1` a bucket lasts **≈ 2 sim-minutes**, and at the 60×
/// rate a level might preview with, ≈ 2 seconds. Long enough that an amortized
/// sweep (8 frames at the documented 256-probe budget) always completes inside a
/// bucket; short enough that the bounded staleness below is invisible.
pub const SUN_BUCKET_LATTICE: f32 = 200.0;

/// Divisions per unit for the sun **radiance** half of [`sun_bucket`] — coarser
/// than the direction, because colour drifts far more slowly than angle over a day
/// and a 1/64 step is well under a quantization step of the 8-bit output.
pub const SUN_RADIANCE_LATTICE: f32 = 64.0;

/// The **quantized** sun, as it enters the amortization sweep key.
///
/// A raw `f32::to_bits()` sun is a correctness bug in the amortization, not a
/// conservatism: under a running `TimeOfDay` clock the projected direction changes
/// in the low bits *every frame*, so the sweep would reset every frame and the
/// cursor would never leave its first slice — amortization would cost exactly what
/// a full update costs while delivering only `probe_budget` probes' worth of
/// freshness. Precisely where it was meant to pay off, it would not.
///
/// Quantizing follows the P17.2 precedent (the sky-view LUT's camera radius is
/// bucketed for the same reason, so a walking camera does not re-bake the sky
/// every frame).
///
/// **Bounded-staleness consequence, stated plainly:** within one bucket the probes
/// are integrated against a sun up to [`SUN_BUCKET_LATTICE`]'s ≈ 0.50° stale, and a
/// probe that has not been revisited yet lags by at most one sweep
/// (`ceil(probe_count / probe_budget)` frames) beyond that. A bucket crossing
/// restarts the sweep, so the lag never accumulates across buckets. At the default
/// `probe_budget = 0` (full update) none of this is reachable at all.
pub fn sun_bucket(dir: Vec3, radiance: [f32; 3]) -> [i32; 6] {
    let q = |v: f32, lattice: f32| {
        if v.is_finite() {
            (v * lattice).round() as i32
        } else {
            i32::MIN
        }
    };
    [
        q(dir.x, SUN_BUCKET_LATTICE),
        q(dir.y, SUN_BUCKET_LATTICE),
        q(dir.z, SUN_BUCKET_LATTICE),
        q(radiance[0], SUN_RADIANCE_LATTICE),
        q(radiance[1], SUN_RADIANCE_LATTICE),
        q(radiance[2], SUN_RADIANCE_LATTICE),
    ]
}

// ── primitive prioritization + macro-cell binning (P18.4) ───────────────────

/// A voxelization primitive's world (render-local) bounding sphere — the only
/// thing prioritization and binning need to know about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiBounds {
    pub center: Vec3,
    pub radius: f32,
}

/// What the voxelizer actually consumed this frame — the instrument that replaced
/// the silent `MAX_GI_INSTANCES` truncation.
///
/// A caller reads it from [`EngineRenderer::gi_audit`](crate::EngineRenderer::gi_audit).
/// `dropped > 0` means the scene has more GI-relevant geometry inside the volume
/// than the tier's budget allows: the *nearest* primitives were kept (so the error
/// is a distant one), but it is a real, reportable loss rather than a comment in a
/// shader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GiAudit {
    /// Primitives whose bounds intersected the volume (the candidate set).
    pub candidates: u32,
    /// Primitives actually uploaded and voxelized.
    pub voxelized: u32,
    /// `candidates - voxelized` — the budget overflow.
    pub dropped: u32,
    /// Macro-cell list entries written (a primitive touching `k` cells costs `k`).
    pub cell_entries: u32,
    /// Terrain columns sampled (`0` for a terrain-free scene).
    pub terrain_columns: u32,
    /// Probes updated this frame (`probe_count` under full update).
    pub probes_updated: u32,
    /// The amortization cursor **after** this frame's slice
    /// ([`ProbeSchedule::cursor`]). `0` under full update. Exposed because the
    /// sweep is otherwise unobservable from outside: a key that resets it every
    /// frame (the raw-`f32`-sun bug [`sun_bucket`] exists to prevent) pins this at
    /// `probe_budget` forever, and nothing in a rendered frame would say so.
    pub probe_cursor: u32,
}

/// Whether a bounding sphere intersects the axis-aligned volume `[min, min+extent]`.
pub fn intersects_volume(b: &GiBounds, vol_min: Vec3, extent: f32) -> bool {
    let vol_max = vol_min + Vec3::splat(extent);
    let closest = b.center.clamp(vol_min, vol_max);
    (closest - b.center).length_squared() <= b.radius * b.radius
}

/// Deterministic nearest-first ordering of `bounds` around `center`.
///
/// Sorted by **surface** distance (`|c − center| − radius`, so a large enclosing
/// primitive like a ground slab sorts first rather than by the accident of where
/// its centre is), with the source index as the tie-break. `f32::total_cmp` rather
/// than `partial_cmp` so a `NaN` from a degenerate transform yields *an* order
/// rather than an unspecified one — determinism does not get to depend on the
/// content being well-formed.
pub fn priority_order(bounds: &[GiBounds], center: Vec3) -> Vec<u32> {
    let mut keyed: Vec<(f32, u32)> = bounds
        .iter()
        .enumerate()
        .map(|(i, b)| ((b.center - center).length() - b.radius, i as u32))
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// Bin `order`-ordered primitives into `macro_dim³` macro cells.
///
/// Returns `(offsets, items)` in CSR form: cell `c` owns
/// `items[offsets[c] .. offsets[c + 1]]`, and every cell's slice is **ascending in
/// the priority order** — which is what makes the shader's "first hit wins" a
/// deterministic, priority-respecting choice rather than a race.
///
/// `offsets` has `macro_dim³ + 1` entries.
pub fn bin_macro_cells(
    bounds: &[GiBounds],
    order: &[u32],
    vol_min: Vec3,
    extent: f32,
    macro_dim: u32,
) -> (Vec<u32>, Vec<u32>) {
    let cells = (macro_dim * macro_dim * macro_dim) as usize;
    let cell_size = extent / macro_dim.max(1) as f32;
    let mut counts = vec![0u32; cells];

    // Cell span of one primitive, clamped to the grid. `None` when it misses.
    let span = |b: &GiBounds| -> Option<([u32; 3], [u32; 3])> {
        let lo = b.center - Vec3::splat(b.radius) - vol_min;
        let hi = b.center + Vec3::splat(b.radius) - vol_min;
        let last = macro_dim.saturating_sub(1) as f32;
        let q = |v: f32| (v / cell_size).floor().clamp(0.0, last) as u32;
        if hi.x < 0.0 || hi.y < 0.0 || hi.z < 0.0 {
            return None;
        }
        if lo.x > extent || lo.y > extent || lo.z > extent {
            return None;
        }
        Some(([q(lo.x), q(lo.y), q(lo.z)], [q(hi.x), q(hi.y), q(hi.z)]))
    };

    for &i in order {
        let Some(b) = bounds.get(i as usize) else {
            continue;
        };
        let Some((lo, hi)) = span(b) else { continue };
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    counts[macro_index(x, y, z, macro_dim) as usize] += 1;
                }
            }
        }
    }

    let mut offsets = vec![0u32; cells + 1];
    for c in 0..cells {
        offsets[c + 1] = offsets[c] + counts[c];
    }
    let total = offsets[cells] as usize;
    let mut items = vec![0u32; total];
    let mut fill = offsets.clone();
    // Walk `order` again: appending in priority order makes each cell's slice
    // ascending in priority, which is the property the shader relies on.
    for (rank, &i) in order.iter().enumerate() {
        let Some(b) = bounds.get(i as usize) else {
            continue;
        };
        let Some((lo, hi)) = span(b) else { continue };
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let c = macro_index(x, y, z, macro_dim) as usize;
                    items[fill[c] as usize] = rank as u32;
                    fill[c] += 1;
                }
            }
        }
    }
    (offsets, items)
}

// ── terrain column sampling (P18.4) ─────────────────────────────────────────

/// One voxel column's terrain occupancy: the world height the column is solid up
/// to, and the splat-blended linear albedo of that surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainColumn {
    /// World-space Y (metres) the terrain surface sits at.
    pub height: f64,
    /// Linear RGB albedo at that point (the four splat layers blended by weight).
    pub albedo: [f32; 3],
}

/// Sample the terrain at world `(x, z)`, over **every** projected tile — the
/// finest one containing the sample wins.
///
/// This is the general sampler, used by tests and any caller that wants "what is
/// drawn there". **The GI voxelizer does not use it**: it goes through
/// [`voxelization_tiles`] + [`sample_terrain_column_in`], restricted to the
/// coarsest asset level, because `terrain.tiles` is the *camera-driven* working
/// set and GI must not be a function of camera history. See
/// [`voxelization_tiles`] for the whole argument.
///
/// Heights are bilinearly interpolated (matching what the terrain shader draws);
/// splat weights take the nearest sample, because a blend of blends is not a
/// different colour to within a voxel and the extra three fetches are not worth it.
/// Returns `None` when no resident tile covers the point (a hole — voxelized as
/// empty, exactly as an unauthored tile has always drawn as nothing).
pub fn sample_terrain_column(terrain: &RenderTerrain, x: f64, z: f64) -> Option<TerrainColumn> {
    let all: Vec<u32> = (0..terrain.tiles.len() as u32).collect();
    sample_terrain_column_in(terrain, &all, x, z)
}

/// The tiles the GI voxelizer is allowed to sample: those at the projection's
/// **coarsest** asset level ([`RenderTerrain::max_lod`]) whose world footprint
/// overlaps the axis-aligned XZ rectangle `[min, max]`, in the projection's own
/// (key-ascending) order.
///
/// ## Why the coarsest level, and not the finest resident one
///
/// `RenderTerrain::tiles` is the **camera-driven** working set: the streamer
/// refines pages near the camera and lets distant ones stand at coarse detail
/// (P16.3b1). Sampling "whichever resident tile is finest" would therefore make GI
/// occupancy and albedo a function of *where the camera has been* — the same leak
/// the vgeom side avoids by voxelizing the always-resident **root page** rather
/// than the live meshlet cut, and it would be invisible to CI, because every
/// golden renders a fully-resident terrain.
///
/// The pyramid's coarsest level is the terrain's analogue of that root page: it is
/// small by construction (`build_pyramid` stops at `PyramidOptions::min_tiles`),
/// it covers the whole terrain, and `TerrainStreamer` seeds and reseeds its
/// published cut from it — so it is there whatever the camera has done. Restricting
/// to it also settles the *albedo* half of the same leak for free: coarse pyramid
/// pages are heights-only, so they project the uniform default weight, and GI can
/// no longer see one splat blend near the camera and another far from it.
///
/// **Fidelity tradeoff, stated plainly.** GI voxels are `extent / dim` — 0.63 m at
/// the default 40 m volume — while a level-`n` terrain sample is `mps · 2ⁿ`. On a
/// streamed terrain with a deep pyramid the coarse lattice is the coarser of the
/// two, so near-field terrain occupancy is blockier than the drawn surface. That
/// is the price of a bounce that does not depend on camera history, and it is the
/// right way round: a *slightly* wrong occluder everywhere beats a *differently*
/// wrong one depending on where the player walked. An inline (non-streamed)
/// terrain has `max_lod() == 0`, so it voxelizes at full authored detail with its
/// painted weights — no tradeoff at all, and no camera dependence either, because
/// nothing streams.
///
/// Filtering once per frame also keeps [`sample_terrain_column_in`]'s inner loop
/// short: the voxelizer samples one column per voxel `(x, z)` — 4096 of them at
/// High — and a streamed terrain can have hundreds of resident tiles.
pub fn voxelization_tiles(terrain: &RenderTerrain, min: (f64, f64), max: (f64, f64)) -> Vec<u32> {
    let res = terrain.tile_resolution;
    if res < 2 || terrain.tiles.is_empty() {
        return Vec::new();
    }
    let lod = terrain.max_lod();
    terrain
        .tiles
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            if t.key.lod != lod {
                return false;
            }
            let span =
                (res - 1) as f64 * terrain.meters_per_sample * (1u64 << t.key.lod.min(62)) as f64;
            let (x0, z0) = (t.origin.x, t.origin.z);
            x0 <= max.0 && x0 + span >= min.0 && z0 <= max.1 && z0 + span >= min.1
        })
        .map(|(i, _)| i as u32)
        .collect()
}

/// [`sample_terrain_column`] restricted to a candidate tile list (see
/// [`voxelization_tiles`], which the GI path always passes). `tiles` must be in
/// the projection's own order for the "finest of the candidates wins" rule to
/// hold — which for the GI path is a formality, since every candidate is at the
/// same (coarsest) level and they do not overlap.
pub fn sample_terrain_column_in(
    terrain: &RenderTerrain,
    tiles: &[u32],
    x: f64,
    z: f64,
) -> Option<TerrainColumn> {
    let res = terrain.tile_resolution;
    if res < 2 || terrain.tiles.is_empty() {
        return None;
    }
    // Finest resident tile wins: `tiles` is ascending by (lod, coord), so the
    // first hit in list order is already the finest.
    for tile in tiles.iter().filter_map(|i| terrain.tiles.get(*i as usize)) {
        let mps = terrain.meters_per_sample * (1u64 << tile.key.lod.min(62)) as f64;
        let fx = (x - tile.origin.x) / mps;
        let fz = (z - tile.origin.z) / mps;
        let last = (res - 1) as f64;
        if fx < 0.0 || fz < 0.0 || fx > last || fz > last {
            continue;
        }
        let x0 = fx.floor().min(last - 1.0).max(0.0);
        let z0 = fz.floor().min(last - 1.0).max(0.0);
        let (tx, tz) = ((fx - x0) as f32, (fz - z0) as f32);
        let (ix, iz) = (x0 as usize, z0 as usize);
        let idx = |cx: usize, cz: usize| cz * res as usize + cx;
        let h = |cx: usize, cz: usize| tile.heights.get(idx(cx, cz)).copied().unwrap_or(0.0);
        let h00 = h(ix, iz);
        let h10 = h(ix + 1, iz);
        let h01 = h(ix, iz + 1);
        let h11 = h(ix + 1, iz + 1);
        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;
        let height = tile.origin.y + (hx0 + (hx1 - hx0) * tz) as f64;

        // Nearest splat weights → blended layer albedo.
        let nx = (fx.round() as usize).min(res as usize - 1);
        let nz = (fz.round() as usize).min(res as usize - 1);
        let w = tile
            .weights
            .get(idx(nx, nz))
            .copied()
            .unwrap_or([255, 0, 0, 0]);
        return Some(TerrainColumn {
            height,
            albedo: blend_layer_albedo(&terrain.layers, w),
        });
    }
    // No resident tile covers this column — a hole, voxelized as empty, exactly
    // as an unauthored tile has always drawn as nothing.
    None
}

/// Blend the four terrain splat layers by an RGBA8 weight sample. Mirrors the
/// terrain shader's `mix` chain closely enough for an occupancy albedo.
pub fn blend_layer_albedo(layers: &[RenderTerrainLayer; 4], w: [u8; 4]) -> [f32; 3] {
    let wf = [
        w[0] as f32 / 255.0,
        w[1] as f32 / 255.0,
        w[2] as f32 / 255.0,
        w[3] as f32 / 255.0,
    ];
    let sum = wf.iter().sum::<f32>();
    let inv = if sum > 1e-4 { 1.0 / sum } else { 0.0 };
    let mut out = [0.0f32; 3];
    for (l, weight) in layers.iter().zip(wf) {
        for (c, o) in out.iter_mut().enumerate() {
            *o += l.albedo[c] * weight * inv;
        }
    }
    if inv == 0.0 {
        layers[0].albedo[..3].try_into().unwrap()
    } else {
        out
    }
}

/// Quantize an emissive radiance to the packed RGBA8 word the voxel volume
/// carries. Mirrors `gi_pack_emissive` in `gi_voxelize.wgsl`.
pub fn pack_emissive(e: [f32; 3]) -> u32 {
    let maxc = e[0].max(e[1]).max(e[2]).clamp(0.0, EMISSIVE_MAX);
    if maxc <= 0.0 {
        return 0;
    }
    let q = |v: f32| ((v / maxc).clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let scale = ((maxc / EMISSIVE_MAX).clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    q(e[0]) | (q(e[1]) << 8) | (q(e[2]) << 16) | (scale << 24)
}

/// Inverse of [`pack_emissive`]. Mirrors `gi_unpack_emissive` in `gi_probes.wgsl`.
pub fn unpack_emissive(v: u32) -> [f32; 3] {
    let f = |s: u32| ((v >> s) & 0xff) as f32 / 255.0;
    let scale = f(24) * EMISSIVE_MAX;
    [f(0) * scale, f(8) * scale, f(16) * scale]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{RenderTerrainTile, TerrainTileKey};
    use glam::DVec3;

    #[test]
    fn spiral_dirs_are_unit_and_spread() {
        let n = 48;
        let mut sum = Vec3::ZERO;
        for i in 0..n {
            let d = golden_spiral_dir(i, n);
            assert!((d.length() - 1.0).abs() < 1e-4, "dir {i} not unit");
            sum += d;
        }
        // An even sphere cover roughly cancels out.
        assert!(sum.length() / (n as f32) < 0.15, "not well spread: {sum:?}");
    }

    #[test]
    fn spiral_is_deterministic() {
        for i in 0..48 {
            assert_eq!(golden_spiral_dir(i, 48), golden_spiral_dir(i, 48));
        }
    }

    #[test]
    fn sh_basis_constant_term() {
        // The l=0 term is direction-independent.
        assert!((sh_l1_basis(Vec3::X)[0] - 0.282095).abs() < 1e-6);
        assert!((sh_l1_basis(Vec3::Y)[0] - 0.282095).abs() < 1e-6);
        // l=1 terms pick up the matching axis.
        assert!((sh_l1_basis(Vec3::Y)[1] - 0.488603).abs() < 1e-6);
        assert!((sh_l1_basis(Vec3::X)[3] - 0.488603).abs() < 1e-6);
    }

    #[test]
    fn probe_and_voxel_indices_are_bounded_and_unique() {
        // Probe corners span the whole grid, endpoints on the faces.
        let vmin = Vec3::ZERO;
        let p0 = probe_position(0, 0, 0, vmin, 40.0);
        let p1 = probe_position(
            PROBE_DIMS[0] - 1,
            PROBE_DIMS[1] - 1,
            PROBE_DIMS[2] - 1,
            vmin,
            40.0,
        );
        assert!(p0.abs_diff_eq(Vec3::ZERO, 1e-4));
        assert!(p1.abs_diff_eq(Vec3::splat(40.0), 1e-4));

        assert_eq!(probe_index(0, 0, 0), 0);
        assert_eq!(probe_count(), 16 * 8 * 16);
        // x is the fastest axis.
        assert_eq!(probe_index(1, 0, 0), 1);
        assert_eq!(probe_index(0, 1, 0), PROBE_DIMS[0]);
        assert_eq!(voxel_index(1, 0, 0), 1);
        assert_eq!(voxel_index(0, 0, 1), GI_DIM * GI_DIM);
    }

    #[test]
    fn volume_is_camera_centred() {
        let eye = Vec3::new(5.0, 2.0, -3.0);
        let vmin = volume_min(eye, 40.0);
        // The eye sits at the volume centre.
        assert!((vmin + Vec3::splat(20.0)).abs_diff_eq(eye, 1e-4));
        assert!((voxel_size(40.0) - 40.0 / 64.0).abs() < 1e-6);
    }

    // ── P18.4 ──

    /// [`GiQuality::High`] must be **exactly** the pre-P18.4 geometry, or every GI
    /// golden moves for a reason that has nothing to do with the feature.
    #[test]
    fn high_quality_is_the_pre_p18_4_geometry() {
        assert_eq!(GiQuality::High.voxel_dim(), GI_DIM);
        assert_eq!(GiQuality::High.probe_dims(), PROBE_DIMS);
        assert_eq!(GiQuality::default(), GiQuality::High);
    }

    #[test]
    fn macro_dim_divides_every_tier() {
        for q in [GiQuality::Low, GiQuality::Medium, GiQuality::High] {
            assert_eq!(
                q.voxel_dim() % MACRO_DIM,
                0,
                "{q:?} voxel dim is not a whole number of macro cells"
            );
            assert_eq!(q.macro_dim(), q.voxel_dim() / MACRO_DIM);
            assert!(q.instance_budget() > 0);
        }
        // Strictly ordered cost.
        assert!(GiQuality::Low.voxel_dim() < GiQuality::Medium.voxel_dim());
        assert!(GiQuality::Medium.voxel_dim() < GiQuality::High.voxel_dim());
        assert!(
            probe_count_of(GiQuality::Low.probe_dims())
                < probe_count_of(GiQuality::Medium.probe_dims())
        );
    }

    #[test]
    fn quality_clamp_only_lowers() {
        for q in [GiQuality::Low, GiQuality::Medium, GiQuality::High] {
            assert!(q.clamp_to(RenderTier::High) <= q);
            assert!(q.clamp_to(RenderTier::Medium) <= q);
            assert!(q.clamp_to(RenderTier::Low) <= q);
        }
        // High tier is a no-op.
        assert_eq!(GiQuality::High.clamp_to(RenderTier::High), GiQuality::High);
        assert_eq!(
            GiQuality::High.clamp_to(RenderTier::Medium),
            GiQuality::Medium
        );
        assert_eq!(GiQuality::Low.clamp_to(RenderTier::High), GiQuality::Low);
        // Idempotent.
        let c = GiQuality::High.clamp_to(RenderTier::Low);
        assert_eq!(c.clamp_to(RenderTier::Low), c);
    }

    /// A constant radiance field reconstructs **exactly** through the SH round
    /// trip, at every lobe sharpness. That identity is why the P18.4 specular term
    /// reduces to the constant it replaced instead of being a new light source.
    #[test]
    fn sh_radiance_is_identity_on_a_uniform_field() {
        // Project a uniform field L over `n` golden-spiral rays, exactly as
        // `gi_probes.wgsl` does.
        let n = 64u32;
        let l = [0.4f32, 0.7, 1.3];
        let mut c = [[0.0f32; 3]; 4];
        for r in 0..n {
            let b = sh_l1_basis(golden_spiral_dir(r, n));
            for (k, cb) in c.iter_mut().enumerate() {
                for (ch, v) in cb.iter_mut().enumerate() {
                    *v += l[ch] * b[k];
                }
            }
        }
        let norm = 4.0 * std::f32::consts::PI / n as f32;
        for cb in c.iter_mut() {
            for v in cb.iter_mut() {
                *v *= norm;
            }
        }
        for lobe in [0.0f32, 0.5, 1.0] {
            for d in [Vec3::X, Vec3::Y, Vec3::NEG_Z, Vec3::ONE.normalize()] {
                let got = sh_radiance(&c, d, lobe);
                for ch in 0..3 {
                    assert!(
                        (got[ch] - l[ch]).abs() < 0.02,
                        "uniform field did not reconstruct: {got:?} vs {l:?} (lobe {lobe})"
                    );
                }
            }
        }
        // ...and a uniform field has (almost) no linear band for a dominant
        // direction to come from: the golden spiral is an even but finite cover, so
        // the residual is a fraction of a percent of the DC term rather than
        // exactly zero. What matters is that it cannot produce a visible highlight.
        let dc = c[0][1].abs();
        for band in &c[1..] {
            for v in band {
                assert!(
                    v.abs() < 0.02 * dc,
                    "uniform field left a linear band of {v} against a DC of {dc}"
                );
            }
        }
    }

    /// **The reduction property** the P18.4 specular design rests on: on a
    /// *uniform* radiance field, a *fully rough* surface's new specular term is
    /// (within the split-sum approximation's own error) the flat
    /// `ambient × f0 × 0.5` it replaced. Turning GI specular on therefore adds
    /// **directionality**, not energy — which is why it can default to on without
    /// re-exposing every GI scene.
    #[test]
    fn specular_reduces_to_the_retired_ambient_constant() {
        let l = 0.8f32; // uniform radiance
        let f0 = 0.06f32;
        // What the lit shaders used to add: `gi_irradiance(n) * f0 * 0.5`, and on a
        // uniform field `gi_irradiance` is `π · L` (its A0 = π cosine convolution
        // with a zero linear band).
        let retired = std::f32::consts::PI * l * f0 * 0.5;
        // What `gi_specular` adds: `sh_radiance(…, lobe = 1 − rough) · π ·
        // (f0·a + b)`, and on a uniform field `sh_radiance` is exactly `L`.
        let (a, b) = env_brdf_ab(1.0, 1.0);
        let now = l * std::f32::consts::PI * (f0 * a + b);
        // Within a quarter: the split-sum BRDF's rough/face-on response is ≈ 0.45,
        // where the retired constant simply said 0.5 — a systematic ~10–20 %
        // (f0-dependent, since the additive `b` matters more for a dielectric).
        // The point is the SCALE, which is what a term with a bug in its π
        // bookkeeping would get wrong by a factor, not by a fifth.
        assert!(
            (now - retired).abs() < 0.25 * retired,
            "the rough/uniform limit drifted from the retired constant: \
             {now} vs {retired}"
        );
        // ...and it is dimmer, never brighter: the new term must not smuggle in
        // energy the old one did not have.
        assert!(now <= retired, "{now} > {retired}");

        // A metal (high f0) tracks it more closely still, because the additive
        // grazing term is a smaller share of the response.
        let f0m = 0.9f32;
        let retired_m = std::f32::consts::PI * l * f0m * 0.5;
        let now_m = l * std::f32::consts::PI * (f0m * a + b);
        assert!((now_m - retired_m).abs() < 0.15 * retired_m);
    }

    #[test]
    fn sh_dominant_direction_points_at_the_bright_side() {
        // A single bright ray along +X projected to L1.
        let d = Vec3::X;
        let b = sh_l1_basis(d);
        let mut c = [[0.0f32; 3]; 4];
        for (k, cb) in c.iter_mut().enumerate() {
            for v in cb.iter_mut() {
                *v = b[k];
            }
        }
        let dom = sh_dominant_direction(&c);
        assert!(dom.dot(Vec3::X) > 0.99, "dominant dir {dom:?} is not +X");
        // Radiance is higher toward the bright side than away from it.
        let toward = sh_radiance(&c, Vec3::X, 1.0)[0];
        let away = sh_radiance(&c, Vec3::NEG_X, 1.0)[0];
        assert!(toward > away, "{toward} !> {away}");
        // Fully-rough (lobe 0) collapses to the same value in every direction.
        assert_eq!(
            sh_radiance(&c, Vec3::X, 0.0),
            sh_radiance(&c, Vec3::NEG_X, 0.0)
        );
    }

    #[test]
    fn env_brdf_is_bounded_and_matches_the_retired_half() {
        for &rough in &[0.04f32, 0.3, 0.7, 1.0] {
            for &nov in &[0.05f32, 0.5, 1.0] {
                let (a, b) = env_brdf_ab(rough, nov);
                assert!(a.is_finite() && b.is_finite());
                assert!((0.0..=1.6).contains(&a), "a {a} out of range");
                // `b` is the grazing-angle Fresnel boost and legitimately
                // approaches 1 for a smooth surface seen edge-on.
                assert!((-0.05..=1.0).contains(&b), "b {b} out of range");
            }
        }
        // The term this replaced was a flat `f0 * 0.5`; a fully-rough,
        // face-on surface must land close to it, or every GI golden's specular
        // energy jumps for no physical reason.
        let (a, b) = env_brdf_ab(1.0, 1.0);
        let f0 = 0.04f32;
        assert!(
            (f0 * a + b - f0 * 0.5).abs() < 0.05,
            "rough face-on env BRDF drifted from the retired 0.5 constant"
        );
    }

    // ── amortization schedule ──

    #[test]
    fn full_update_is_the_zero_budget_default() {
        let mut s = ProbeSchedule::new();
        assert_eq!(s.next(2048, 0), (0, 2048));
        assert_eq!(s.cursor(), 0);
        // A budget at or above the total is also a full update.
        assert_eq!(s.next(2048, 2048), (0, 2048));
        assert_eq!(s.next(2048, 9999), (0, 2048));
        assert_eq!(s.cursor(), 0, "full update must not advance the cursor");
        // Degenerate totals are inert.
        assert_eq!(s.next(0, 128), (0, 0));
    }

    #[test]
    fn amortized_sweep_covers_every_probe_exactly_once() {
        let total = 100u32;
        let budget = 30u32;
        let mut s = ProbeSchedule::new();
        let mut seen = vec![0u32; total as usize];
        // ceil(100/30) = 4 frames covers the sweep (with wrap-around overlap).
        for _ in 0..4 {
            let (start, count) = s.next(total, budget);
            for i in 0..count {
                seen[((start + i) % total) as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c >= 1),
            "a probe was starved by the schedule: {seen:?}"
        );
        // Exactly one full period (100 updates) covers each probe once.
        let mut s2 = ProbeSchedule::new();
        let mut seen2 = vec![0u32; total as usize];
        for _ in 0..5 {
            let (start, count) = s2.next(total, 20);
            for i in 0..count {
                seen2[((start + i) % total) as usize] += 1;
            }
        }
        assert!(seen2.iter().all(|&c| c == 1), "{seen2:?}");
        assert_eq!(s2.cursor(), 0, "a whole number of sweeps returns to 0");
    }

    #[test]
    fn cold_schedules_are_reproducible_and_reset_restarts() {
        let mut a = ProbeSchedule::new();
        let mut b = ProbeSchedule::new();
        let seq_a: Vec<_> = (0..7).map(|_| a.next(2048, 300)).collect();
        let seq_b: Vec<_> = (0..7).map(|_| b.next(2048, 300)).collect();
        assert_eq!(seq_a, seq_b, "two cold schedules diverged");
        assert_ne!(seq_a[0], seq_a[1], "the schedule must actually advance");

        // A reset returns a warm schedule to the cold sequence.
        a.reset();
        let after: Vec<_> = (0..7).map(|_| a.next(2048, 300)).collect();
        assert_eq!(after, seq_a);
    }

    // ── prioritization + binning ──

    fn b(c: Vec3, r: f32) -> GiBounds {
        GiBounds {
            center: c,
            radius: r,
        }
    }

    #[test]
    fn priority_is_nearest_surface_first_and_deterministic() {
        let center = Vec3::ZERO;
        let bounds = [
            b(Vec3::new(10.0, 0.0, 0.0), 0.5),  // surface at 9.5
            b(Vec3::new(3.0, 0.0, 0.0), 0.5),   // surface at 2.5
            b(Vec3::new(50.0, 0.0, 0.0), 49.0), // an enclosing slab: surface at 1.0
        ];
        assert_eq!(priority_order(&bounds, center), vec![2, 1, 0]);
        // Deterministic, and ties break on the source index.
        let tied = [b(Vec3::X, 0.0), b(Vec3::NEG_X, 0.0), b(Vec3::Y, 0.0)];
        assert_eq!(priority_order(&tied, center), vec![0, 1, 2]);
        assert_eq!(priority_order(&tied, center), priority_order(&tied, center));
        // A degenerate (NaN) transform still yields *an* order, not a panic.
        let nan = [b(Vec3::splat(f32::NAN), 1.0), b(Vec3::X, 1.0)];
        assert_eq!(priority_order(&nan, center).len(), 2);
    }

    #[test]
    fn volume_intersection_rejects_the_far_field() {
        let vmin = Vec3::splat(-20.0);
        assert!(intersects_volume(&b(Vec3::ZERO, 0.1), vmin, 40.0));
        // Just outside a face, but its radius reaches in.
        assert!(intersects_volume(
            &b(Vec3::new(21.0, 0.0, 0.0), 1.5),
            vmin,
            40.0
        ));
        assert!(!intersects_volume(
            &b(Vec3::new(25.0, 0.0, 0.0), 1.0),
            vmin,
            40.0
        ));
    }

    #[test]
    fn binning_is_csr_complete_and_priority_ordered() {
        let vmin = Vec3::ZERO;
        let extent = 32.0;
        let macro_dim = 4; // 4³ = 64 cells, 8 m each
        let bounds = [
            b(Vec3::splat(4.0), 1.0),   // cell (0,0,0)
            b(Vec3::splat(28.0), 1.0),  // cell (3,3,3)
            b(Vec3::splat(16.0), 20.0), // spans everything
        ];
        let order = priority_order(&bounds, Vec3::splat(16.0));
        let (offsets, items) = bin_macro_cells(&bounds, &order, vmin, extent, macro_dim);
        assert_eq!(offsets.len(), 64 + 1);
        assert_eq!(*offsets.last().unwrap() as usize, items.len());
        // Offsets are non-decreasing (a valid CSR).
        assert!(offsets.windows(2).all(|w| w[1] >= w[0]));
        // The enclosing primitive reaches every cell, so no cell is empty.
        for c in 0..64 {
            assert!(
                offsets[c + 1] > offsets[c],
                "cell {c} empty despite an all-covering primitive"
            );
            // Each cell's slice is ascending in RANK (priority order).
            let slice = &items[offsets[c] as usize..offsets[c + 1] as usize];
            assert!(slice.windows(2).all(|w| w[0] < w[1]), "cell {c}: {slice:?}");
        }
        // Corner cells carry their own primitive as well as the big one.
        let c000 = macro_index(0, 0, 0, macro_dim) as usize;
        assert_eq!(offsets[c000 + 1] - offsets[c000], 2);
        // Deterministic.
        assert_eq!(
            bin_macro_cells(&bounds, &order, vmin, extent, macro_dim),
            (offsets, items)
        );
    }

    #[test]
    fn binning_drops_primitives_outside_the_volume() {
        let vmin = Vec3::ZERO;
        let bounds = [b(Vec3::splat(-100.0), 1.0), b(Vec3::splat(4.0), 1.0)];
        let order = vec![0u32, 1];
        let (offsets, items) = bin_macro_cells(&bounds, &order, vmin, 32.0, 4);
        assert_eq!(items.len(), 1, "the far primitive was binned anyway");
        assert_eq!(
            items[0], 1,
            "the surviving entry must be the near one's rank"
        );
        assert_eq!(*offsets.last().unwrap(), 1);
    }

    // ── emissive packing ──

    #[test]
    fn emissive_round_trips_within_quantization() {
        for e in [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.4, 0.8],
            [12.0, 6.0, 1.0],
            [40.0, 40.0, 40.0], // above the ceiling → clamps
        ] {
            let got = unpack_emissive(pack_emissive(e));
            for c in 0..3 {
                let want = e[c].min(EMISSIVE_MAX);
                assert!(
                    (got[c] - want).abs() <= 0.08 * EMISSIVE_MAX.min(want.max(1.0)),
                    "emissive {e:?} round-tripped to {got:?}"
                );
            }
        }
        assert_eq!(pack_emissive([0.0; 3]), 0);
        assert_eq!(unpack_emissive(0), [0.0; 3]);
    }

    // ── terrain sampling ──

    fn flat_terrain(height: f32) -> RenderTerrain {
        let res = 5u32;
        RenderTerrain {
            id: 1,
            tile_resolution: res,
            meters_per_sample: 1.0,
            tiles: vec![RenderTerrainTile {
                key: TerrainTileKey::lod0((0, 0)),
                origin: DVec3::new(0.0, 0.0, 0.0),
                heights: vec![height; (res * res) as usize],
                weights: vec![[255, 0, 0, 0]; (res * res) as usize],
                // Unpainted: GI reads heights + splat albedo, never biome ids.
                biomes: Vec::new(),
                height_bounds: (height, height),
                version: 1,
            }],
            layers: [
                RenderTerrainLayer {
                    albedo: [1.0, 0.0, 0.0, 1.0],
                    ..Default::default()
                },
                RenderTerrainLayer {
                    albedo: [0.0, 1.0, 0.0, 1.0],
                    ..Default::default()
                },
                RenderTerrainLayer::default(),
                RenderTerrainLayer::default(),
            ],
            macro_variation: 0.0,
            biome_palette: Vec::new(),
        }
    }

    #[test]
    fn terrain_column_samples_inside_and_misses_outside() {
        let t = flat_terrain(3.5);
        let c = sample_terrain_column(&t, 2.0, 2.0).expect("inside the tile");
        assert!((c.height - 3.5).abs() < 1e-5);
        assert_eq!(c.albedo, [1.0, 0.0, 0.0], "layer 0 at full weight");
        // Outside the resident tile → a hole, not a guess.
        assert!(sample_terrain_column(&t, 99.0, 0.0).is_none());
        assert!(sample_terrain_column(&t, -1.0, 0.0).is_none());
        // An empty projection is inert.
        let empty = RenderTerrain::default();
        assert!(sample_terrain_column(&empty, 0.0, 0.0).is_none());
    }

    #[test]
    fn terrain_column_interpolates_a_ramp() {
        let res = 3u32;
        let mut t = flat_terrain(0.0);
        t.tile_resolution = res;
        t.tiles[0].heights = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        t.tiles[0].weights = vec![[255, 0, 0, 0]; (res * res) as usize];
        // A ramp along +X: the midpoint between samples 0 and 1 is 0.5.
        let c = sample_terrain_column(&t, 0.5, 0.0).unwrap();
        assert!((c.height - 0.5).abs() < 1e-5, "got {}", c.height);
        let c2 = sample_terrain_column(&t, 1.75, 1.0).unwrap();
        assert!((c2.height - 1.75).abs() < 1e-5, "got {}", c2.height);
    }

    /// The residency rule: with a level-0 tile and a coarse tile both covering a
    /// point, the **finest resident** one wins — and with only the coarse tile
    /// resident, the coarse detail is what GI sees (never a hole, only softer).
    #[test]
    fn finest_resident_tile_wins() {
        let mut t = flat_terrain(1.0);
        // A coarse (lod 1) tile covering the same ground at a different height.
        let res = t.tile_resolution;
        t.tiles.push(RenderTerrainTile {
            key: TerrainTileKey::new(1, (0, 0)),
            origin: DVec3::ZERO,
            heights: vec![9.0; (res * res) as usize],
            weights: vec![[255, 0, 0, 0]; (res * res) as usize],
            biomes: Vec::new(),
            height_bounds: (9.0, 9.0),
            version: 1,
        });
        // `tiles` is ascending by (lod, coord) → level 0 first → level 0 wins.
        assert!((sample_terrain_column(&t, 1.0, 1.0).unwrap().height - 1.0).abs() < 1e-5);
        // Evict the fine tile: the coarse one still answers, at its own detail.
        t.tiles.remove(0);
        assert!((sample_terrain_column(&t, 1.0, 1.0).unwrap().height - 9.0).abs() < 1e-5);
    }

    /// The camera-independence rule, at the level it is enforced: the voxelizer's
    /// candidate set is the projection's **coarsest** level and nothing else.
    ///
    /// The regression this guards is quiet — every golden renders a fully-resident
    /// terrain, so a voxelizer that followed the finest resident tile would look
    /// perfect in CI and drift with the camera in a real level.
    #[test]
    fn voxelization_reads_only_the_coarsest_level() {
        let mut t = flat_terrain(1.0);
        let res = t.tile_resolution;
        let span = (res - 1) as f64 * t.meters_per_sample;
        // A level-1 page covering the same ground at a different height — the
        // shape a streamed terrain has when the camera is far away.
        t.tiles.push(RenderTerrainTile {
            key: TerrainTileKey::new(1, (0, 0)),
            origin: DVec3::ZERO,
            heights: vec![9.0; (res * res) as usize],
            weights: vec![[255, 0, 0, 0]; (res * res) as usize],
            biomes: Vec::new(),
            height_bounds: (9.0, 9.0),
            version: 1,
        });
        let rect = ((0.0, 0.0), (span, span));
        let cands = voxelization_tiles(&t, rect.0, rect.1);
        assert_eq!(cands.len(), 1, "expected exactly the coarsest page");
        assert_eq!(t.tiles[cands[0] as usize].key.lod, 1);
        // ...and the sampled column is the coarse one, NOT the finer level-0 page
        // sitting right beside it.
        let col = sample_terrain_column_in(&t, &cands, 1.0, 1.0).unwrap();
        assert!((col.height - 9.0).abs() < 1e-5, "got {}", col.height);
        // The general sampler still answers "what is drawn there" — the two are
        // deliberately different functions.
        assert!((sample_terrain_column(&t, 1.0, 1.0).unwrap().height - 1.0).abs() < 1e-5);

        // **The property itself**: refining the level-0 set (what a camera moving
        // closer does) cannot change the candidate set or the sampled column.
        let before = sample_terrain_column_in(&t, &cands, 1.0, 1.0);
        t.tiles.remove(0); // evict the fine page — the camera walked away
        let after_cands = voxelization_tiles(&t, rect.0, rect.1);
        assert_eq!(
            sample_terrain_column_in(&t, &after_cands, 1.0, 1.0),
            before,
            "the voxelized column moved when level-0 residency changed"
        );

        // An inline terrain (no pyramid) voxelizes at level 0 with its painted
        // weights — max_lod() == 0, so there is no tradeoff and no dependence.
        let inline = flat_terrain(2.0);
        let ic = voxelization_tiles(&inline, rect.0, rect.1);
        assert_eq!(ic.len(), 1);
        assert_eq!(inline.tiles[ic[0] as usize].key.lod, 0);
        assert!(
            (sample_terrain_column_in(&inline, &ic, 1.0, 1.0)
                .unwrap()
                .height
                - 2.0)
                .abs()
                < 1e-5
        );

        // A rectangle outside every page yields no candidates (a hole, not a guess).
        assert!(voxelization_tiles(&t, (1e6, 1e6), (1e6 + 1.0, 1e6 + 1.0)).is_empty());
        assert!(voxelization_tiles(&RenderTerrain::default(), rect.0, rect.1).is_empty());
    }

    // ── sun bucketing (P18.4 amortization) ──

    /// The bug this exists to prevent: a raw-bits sun resets the sweep every frame
    /// under a running clock, so amortization pays full price for one slice of
    /// freshness. The bucket has to be coarse enough that a real-time clock stays
    /// inside it for many frames — and fine enough that the staleness is invisible.
    #[test]
    fn sun_buckets_absorb_a_running_clock_and_break_on_real_motion() {
        let radiance = [3.0f32, 2.9, 2.8];
        // The sun sweeps 15°/hour = 1/240 °/s. One frame at 60 fps is 1/14400 °.
        let at = |deg: f32| {
            let r = deg.to_radians();
            Vec3::new(r.cos(), r.sin(), 0.0)
        };
        let base = sun_bucket(at(45.0), radiance);
        // A single frame's motion at rate = 1 must NOT move the bucket...
        assert_eq!(sun_bucket(at(45.0 + 1.0 / 14400.0), radiance), base);
        // ...nor a whole second of it.
        assert_eq!(sun_bucket(at(45.0 + 1.0 / 240.0), radiance), base);
        // A real move does. 0.5° is the documented bucket width, so a full degree
        // is unambiguous whatever the phase.
        assert_ne!(sun_bucket(at(46.0), radiance), base);
        // Deterministic + pure.
        assert_eq!(sun_bucket(at(45.0), radiance), base);
        // A degenerate direction yields *a* bucket rather than a panic or a NaN
        // key that never compares equal to itself.
        let nan = sun_bucket(Vec3::splat(f32::NAN), radiance);
        assert_eq!(nan, sun_bucket(Vec3::splat(f32::NAN), radiance));

        // Radiance is bucketed too, and coarsely: a colour drift under 1/64 is
        // below what the 8-bit output could show.
        assert_eq!(sun_bucket(at(45.0), [3.0, 2.9, 2.8005]), base);
        assert_ne!(sun_bucket(at(45.0), [3.1, 2.9, 2.8]), base);
    }

    /// The bucket's *width*, as an angle — the number the bounded-staleness claim
    /// in [`sun_bucket`]'s docs is made of.
    #[test]
    fn sun_bucket_width_is_about_half_a_degree() {
        // Same bucket ⇒ every component within 1/lattice ⇒ the chord (and hence
        // the angle, for unit vectors) is under √3 / lattice.
        let bound = (3.0f32).sqrt() / SUN_BUCKET_LATTICE;
        assert!(
            (0.4..0.6).contains(&bound.to_degrees()),
            "bucket width drifted to {}°",
            bound.to_degrees()
        );
        // Sweep a full circle: every pair of directions sharing a bucket really is
        // within that bound, and the buckets actually change along the way.
        let radiance = [1.0f32; 3];
        let mut changes = 0;
        let mut prev_key = sun_bucket(Vec3::X, radiance);
        let mut prev_dir = Vec3::X;
        for i in 1..=3600 {
            let r = (i as f32 * 0.1).to_radians();
            let d = Vec3::new(r.cos(), r.sin(), 0.0);
            let k = sun_bucket(d, radiance);
            if k == prev_key {
                assert!(
                    prev_dir.dot(d).clamp(-1.0, 1.0).acos() <= bound + 1e-4,
                    "two directions shared a bucket while {}° apart",
                    prev_dir.dot(d).acos().to_degrees()
                );
            } else {
                changes += 1;
                prev_key = k;
                prev_dir = d;
            }
        }
        // A full turn crosses ~360/0.5 buckets; the exact count depends on the
        // lattice phase, so this just pins the order of magnitude.
        assert!(
            (300..1600).contains(&changes),
            "a full turn crossed {changes} buckets"
        );
    }

    #[test]
    fn splat_weights_blend_layer_albedo() {
        let layers = [
            RenderTerrainLayer {
                albedo: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
            RenderTerrainLayer {
                albedo: [0.0, 1.0, 0.0, 1.0],
                ..Default::default()
            },
            RenderTerrainLayer::default(),
            RenderTerrainLayer::default(),
        ];
        let half = blend_layer_albedo(&layers, [128, 127, 0, 0]);
        assert!((half[0] - 0.5).abs() < 0.02 && (half[1] - 0.5).abs() < 0.02);
        // An all-zero weight sample falls back to layer 0 rather than to black.
        assert_eq!(blend_layer_albedo(&layers, [0; 4]), [1.0, 0.0, 0.0]);
    }
}
