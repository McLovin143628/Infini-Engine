//! Volumetric clouds (P17.3): the parameter block, the hand-rolled deterministic
//! noise that shapes the field, and the **CPU reference** of the density function
//! the GPU raymarch evaluates.
//!
//! The GPU side lives in [`crate::passes::cloud_bake`] (three compute passes: the
//! two 3D noise textures and the cloud-shadow map), [`crate::passes::cloud`] (the
//! raymarch that composites over the scene) and `shaders/cloud*.wgsl`. As with
//! [`crate::atmosphere`], **everything in this module is pure and GPU-free**, so
//! the field is unit-tested on every CI leg — including the ones with no adapter,
//! where the golden harness skips.
//!
//! # Units (architecture rule 6)
//!
//! Clouds are authored against the **level's** geometry, not against the planet,
//! so — like [`crate::atmosphere::HeightFog`] and unlike the rest of
//! [`crate::atmosphere`] — this module is **SI metres** throughout:
//!
//! | quantity | unit |
//! |---|---|
//! | layer bottom / top altitude, march distance, shadow extent | m |
//! | extinction ([`CloudParams::density`]) | m⁻¹ |
//! | wind | m/s |
//! | time offset | s |
//!
//! The conversion to the atmosphere's kilometres happens exactly once, where a
//! cloud radiance meets the aerial-perspective term in `cloud.wgsl`.
//!
//! # The noise, and why it is hand-rolled
//!
//! The house **deterministic-noise law** (P10, restated by the P17.2 starfield)
//! is that a committed-content field must be a pure function of `(seed,
//! position)` with no wall clock, no `rand`, and no `f32` `std`/WGSL
//! transcendental in the part that is compared across platforms. So:
//!
//! * the lattice values come from [`cloud_hash`], a **pure integer** avalanche
//!   hash — bit-identical on every adapter, exactly like `atmos_hash3`;
//! * the interpolation is `fade` + `lerp` in f32, written in the *same order* in
//!   Rust and WGSL (see [`fade`] / `cloud_fade` in `cloud_noise.wgsl`);
//! * **no trigonometry anywhere** in the field. The one place a periodic function
//!   would be natural — the wind drift — is a linear ramp wrapped with
//!   [`wind_offset`], which is `rem_euclid` on an f64, not a `sin`.
//!
//! What that buys is *near*-bit parity, not bit parity: WGSL permits an
//! implementation to contract `a*b + c` into an FMA, so the two mantissas can
//! differ in the last place and the difference compounds through the octave sum.
//! The parity gate therefore asserts a **documented envelope** rather than
//! equality — see [`CPU_GPU_TEXEL_TOLERANCE`] and the `cloud_noise_bake_matches_the_cpu_reference`
//! golden. That is the same call the P10 erosion port made, for the same reason.
//!
//! # Off path
//!
//! [`CloudParams::default`] is **disabled**, every consumer branches on one
//! uniform flag, and the cloud pass dispatches nothing — so a scene that never
//! opted into clouds renders the exact instruction stream P17.2 left behind.

use crate::atmosphere::AtmosphereQuality;

/// Base extinction the authored [`CloudParams::density`] is expressed in:
/// **m⁻¹**. A real cumulus is 0.01–0.1 m⁻¹; over a 2 km column that is an optical
/// depth of 20–200, i.e. utterly opaque, which is why the default sits at the
/// bottom of the range.
pub const DEFAULT_CLOUD_DENSITY: f32 = 0.04;

/// World size, **metres**, that one full wrap of the **shape** texture covers.
///
/// The 3D textures are tileable (their lattices wrap at the texture edge), so the
/// field repeats every `SHAPE_TILE_M` metres. 8 km is large enough that a player
/// never sees the repeat inside one view and small enough that f32 world
/// coordinates keep their precision at the far end of a 20 km cloud march.
pub const SHAPE_TILE_M: f32 = 8192.0;

/// World size, **metres**, of one wrap of the **detail** (erosion) texture.
pub const DETAIL_TILE_M: f32 = 256.0;

/// World size, **metres**, of one wrap of the 2D coverage/type weather field.
///
/// Deliberately much larger than [`SHAPE_TILE_M`]: coverage is the *weather*, and
/// weather has features tens of kilometres across. A small period here is what
/// makes procedural clouds read as wallpaper.
pub const WEATHER_TILE_M: f32 = 40960.0;

/// Per-channel tolerance of the CPU/GPU noise-bake parity gate, in **RGBA8
/// LSBs**.
///
/// Not a fudge factor, a statement about the platform: WGSL explicitly permits an
/// implementation to contract `a * b + c` into a fused multiply-add, and the
/// octave sum in [`shape_texel`] is a chain of exactly that shape. One
/// contraction shifts the result by ~1 ULP of f32, which after the `* 255.0 +
/// 0.5` quantization is at most one LSB. So the gate asserts **≤ 1 LSB
/// everywhere** and additionally that the overwhelming majority of texels are
/// *exactly* equal (see [`CPU_GPU_EXACT_FRACTION`]) — a real port error moves far
/// more than one texel by far more than one LSB, and would fail both halves.
pub const CPU_GPU_TEXEL_TOLERANCE: u8 = 1;

/// Fraction of noise-bake texels that must match the CPU reference **exactly**.
/// The remainder are allowed [`CPU_GPU_TEXEL_TOLERANCE`].
///
/// The load-bearing assertion is the LSB bound above; this one exists to catch
/// the case where a port is "close enough everywhere" for a *structural* reason
/// (an off-by-half-texel sampling convention, say) that happens to stay inside one
/// LSB in a smooth region. Measured on Windows/Vulkan (RTX 4070 Ti): **88.0 %** of
/// shape texels and **91.1 %** of detail texels are bit-exact, with a worst case
/// of exactly 1 LSB — the shape volume contracts slightly more because its Perlin octave
/// sum is a longer multiply-add chain than the detail volume's single Worley
/// lookup. The floor carries headroom below the measured figure for adapters that
/// contract more eagerly.
pub const CPU_GPU_EXACT_FRACTION: f64 = 0.75;

/// Relative tolerance of the cloud-**shadow** parity gate (dimensionless).
///
/// Much looser than the texel gate, and honestly so: a shadow texel is a
/// 16-step Beer–Lambert march whose *input* is a trilinear filter of the 3D
/// textures. Hardware trilinear filtering is specified to only 8 bits of
/// sub-texel precision (WebGPU inherits this from D3D/Vulkan), so the CPU
/// reference — which filters in full f32 — cannot be expected to agree beyond
/// that. Measured on Windows/Vulkan the agreement is far better than the
/// envelope — **mean |Δ| 4e-5, worst 1.1e-3** over 2704 taps — so 2 % is the
/// cross-adapter allowance rather than the achieved precision, and it is still
/// orders of magnitude tighter than the wholesale disagreement a genuinely wrong
/// march (a flipped height gradient, a dropped erosion channel, a coverage
/// multiply where a dissolve belongs) would produce.
pub const CPU_GPU_SHADOW_TOLERANCE: f32 = 0.02;

/// The scene's clouds. Projected onto [`crate::atmosphere::AtmosphereParams`] by
/// both scene builders from the level's `SkyAtmosphere` component; **disabled** by
/// default so every pre-P17.3 level is untouched.
///
/// All altitudes and distances are **SI metres** (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudParams {
    /// Master switch. Off ⇒ the bake and raymarch passes dispatch nothing and the
    /// lit passes take their byte-identical no-cloud path.
    pub enabled: bool,
    /// Fractional sky coverage, `[0, 1]`. `0` is cloudless, `~0.35` is scattered
    /// cumulus, `1` is solid overcast. This is a *bias* on the procedural weather
    /// field, not a literal area fraction — the field's own variation means the
    /// realised coverage tracks it monotonically without matching it exactly.
    pub coverage: f32,
    /// Cloud **type**, `[0, 1]`: `0` = stratus (a flat sheet hugging the bottom
    /// of the layer), `0.5` = stratocumulus, `1` = cumulus (towering, rounded,
    /// occupying the full slab). Drives the vertical density profile — see
    /// [`height_gradient`].
    pub cloud_type: f32,
    /// Bottom of the cloud slab, **metres** of world altitude.
    pub bottom: f32,
    /// Top of the cloud slab, **metres**. Clamped above [`bottom`](Self::bottom).
    pub top: f32,
    /// Extinction at full density, **m⁻¹**. See [`DEFAULT_CLOUD_DENSITY`].
    pub density: f32,
    /// Strength of the high-frequency **erosion** detail, `[0, 1]`. `0` leaves
    /// smooth blobs; `1` carves the wispy edges that make a cloud read as vapour.
    pub detail: f32,
    /// Field seed. Only the low 24 bits are used, so the value survives the trip
    /// through the f32 uniform exactly (see [`CloudGpuSeed`]).
    pub seed: u32,
    /// Wind velocity in world **X**, m/s. Drifts the whole field.
    pub wind_x: f32,
    /// Wind velocity in world **Z**, m/s.
    pub wind_z: f32,
    /// Time in **seconds** the wind has been blowing — the level's time-of-day
    /// clock, *not* a wall clock. This is what makes the drift a deterministic
    /// function of the document: two runs at the same TOD see the same sky.
    ///
    /// **Not a schema field**: it is projected from the `TimeOfDay` authority
    /// that already lives beside `SkyAtmosphere`.
    pub time_s: f64,
    /// Forward-scattering asymmetry of the dominant Henyey–Greenstein lobe,
    /// `[0, 0.95]`. The back lobe is derived from it (see [`BACK_LOBE_G`]), so a
    /// level authors one number and still gets the two-lobe phase that produces
    /// silver linings.
    pub phase_g: f32,
    /// How much the cloud layer darkens the **sun on the ground**, `[0, 1]`.
    /// `0` disables the cloud-shadow map entirely (and the lit passes take the
    /// byte-identical no-cloud-shadow path); `1` is the physical amount.
    pub shadow_strength: f32,
    /// Multiplier on the ambient (sky-view) term inside the cloud, `[0, 4]`.
    /// `1` is the physical amount; raising it lifts the shadowed undersides of an
    /// overcast layer, which is the usual artistic complaint about correct
    /// clouds.
    pub ambient: f32,
    /// Linear albedo tint of the cloud droplets. White is physical (water is
    /// grey); the tint is for stylised skies.
    pub color: [f32; 3],
}

impl Default for CloudParams {
    /// **Disabled**, with a tasteful scattered-cumulus layer already dialled in so
    /// that enabling clouds is a one-flag change rather than a parameter hunt —
    /// exactly the shape [`crate::atmosphere::AtmosphereParams::default`] takes.
    fn default() -> Self {
        Self {
            enabled: false,
            // 0.35 is a broken-cumulus sky: ~60 % of the dome carries cloud with
            // real gaps between. Measured, not guessed — the coverage knob's
            // response is documented on [`weather`], and `clouds_scattered` is the
            // golden that pins what this number looks like.
            coverage: 0.35,
            cloud_type: 0.7,
            bottom: 1500.0,
            top: 4000.0,
            density: DEFAULT_CLOUD_DENSITY,
            detail: 0.6,
            seed: 0,
            wind_x: 6.0,
            wind_z: 2.0,
            time_s: 0.0,
            phase_g: 0.8,
            shadow_strength: 1.0,
            ambient: 1.0,
            color: [1.0, 1.0, 1.0],
        }
    }
}

/// Asymmetry of the **back** Henyey–Greenstein lobe, as a fraction of the forward
/// lobe's `g`. Negative = back-scattering.
///
/// A single forward lobe makes a cloud look like a lump of plastic: real water
/// droplets scatter strongly forward *and* appreciably back, which is why a cloud
/// between you and the sun has a silver lining and one behind you is still
/// bright. Mie theory gives a whole ripple of lobes; two HG lobes is the standard
/// cheap fit.
pub const BACK_LOBE_G: f32 = -0.45;

/// Weight of the forward lobe in the two-lobe phase mix, `[0, 1]`.
pub const FORWARD_LOBE_WEIGHT: f32 = 0.6;

impl CloudParams {
    /// Whether the raymarch and bake passes should run at all.
    #[inline]
    pub fn active(&self) -> bool {
        self.enabled && self.coverage > 0.0 && self.density > 0.0 && self.thickness() > 0.0
    }

    /// Whether anything here should touch a **lit** pass — i.e. whether the
    /// cloud-shadow map darkens world geometry. `false` keeps the lit shaders
    /// byte-identical.
    #[inline]
    pub fn shadows_world(&self) -> bool {
        self.active() && self.shadow_strength > 0.0
    }

    /// Slab thickness, metres (always ≥ 0).
    #[inline]
    pub fn thickness(&self) -> f32 {
        (self.top - self.bottom).max(0.0)
    }

    /// The seed actually used, masked to the 24 bits that survive the f32
    /// uniform.
    #[inline]
    pub fn masked_seed(&self) -> u32 {
        self.seed & CloudGpuSeed::MASK
    }

    /// Wind displacement of the field at [`time_s`](Self::time_s), **metres**,
    /// wrapped into one shape tile. See [`wind_offset`].
    #[inline]
    pub fn wind_offset(&self) -> [f32; 2] {
        wind_offset(self.wind_x, self.wind_z, self.time_s)
    }
}

/// How the cloud seed crosses the uniform boundary.
///
/// The shader needs the seed as a `u32` for the integer hash, but
/// [`crate::passes::sky_lut::AtmosphereGpu`] is an all-`f32` POD whose
/// `PartialEq` gates the per-frame re-upload — and `f32::from_bits` on an
/// arbitrary `u32` can produce a `NaN`, which would compare unequal to itself and
/// re-upload the uniform every single frame forever. So the seed travels as an
/// **f32 holding a small integer**, exactly representable below 2²⁴, and the
/// shader converts with `u32(...)` rather than `bitcast`.
pub struct CloudGpuSeed;

impl CloudGpuSeed {
    /// Largest seed that survives the round trip exactly (2²⁴ − 1).
    pub const MASK: u32 = 0x00ff_ffff;

    /// Encode a seed for the uniform.
    #[inline]
    pub fn encode(seed: u32) -> f32 {
        (seed & Self::MASK) as f32
    }
}

/// Wind displacement of the cloud field after `time_s` seconds at `(wind_x,
/// wind_z)` m/s, **wrapped into one [`SHAPE_TILE_M`] tile**.
///
/// The wrap is not cosmetic. An hour of 10 m/s wind is 36 km; a day is 864 km, at
/// which point an f32 world coordinate has ~0.06 m of resolution and the
/// high-frequency erosion detail visibly quantizes into stair-steps. Because the
/// field is *tileable*, subtracting whole tiles is exactly a no-op on the sampled
/// value, so the wrap is free of visual consequence and buys back full precision.
/// The arithmetic is done in `f64` so the wrap itself does not lose the bits it
/// exists to protect.
/// Rate, **Hz**, at which the raymarch's blue-noise jitter advances to the next
/// element of its sequence.
///
/// High enough that any plausible frame rate against any plausible time-of-day
/// scale lands on a new element every frame, which is what lets the temporal pass
/// average them; and it is a *rate against the level's clock*, so a paused clock
/// is a frozen jitter and the same frame renders identically twice.
pub const JITTER_PHASE_HZ: f64 = 240.0;

/// Length of the jitter sequence before it repeats.
///
/// The wrap is not cosmetic, for the same reason [`wind_offset`]'s is not: the
/// phase crosses the uniform as an f32, and an unwrapped `time_s × 240` is past
/// 2²⁴ after nineteen hours of level time — at which point the "next element"
/// would silently be the same element, and the temporal pass would average one
/// sample position with itself for ever.
pub const JITTER_PHASE_PERIOD: f64 = 64.0;

/// Which element of the jitter sequence the raymarch offsets by at `time_s`.
///
/// **Never a frame index.** The whole determinism argument for jittering a
/// committed render rests on this: the offset is a function of the level's own
/// clock and the pixel's position, so two runs that reach the same time-of-day
/// with the same camera produce the same pixels — which a frame counter, which
/// depends on how long the process has been up and how many frames it dropped,
/// could never do.
pub fn jitter_phase(time_s: f64) -> f32 {
    let p = time_s * JITTER_PHASE_HZ;
    if !p.is_finite() {
        return 0.0;
    }
    p.floor().rem_euclid(JITTER_PHASE_PERIOD) as f32
}

pub fn wind_offset(wind_x: f32, wind_z: f32, time_s: f64) -> [f32; 2] {
    let tile = SHAPE_TILE_M as f64;
    let wrap = |v: f32| -> f32 {
        let d = v as f64 * time_s;
        if !d.is_finite() {
            return 0.0;
        }
        d.rem_euclid(tile) as f32
    };
    [wrap(wind_x), wrap(wind_z)]
}

// ── the hash ─────────────────────────────────────────────────────────────────

/// The field's one source of randomness: a pure-integer avalanche over four
/// `u32`s.
///
/// Bit-identical on every adapter by construction (no float, no trig), which is
/// what makes the whole field portable. Mirrors `cloud_hash` in
/// `shaders/cloud_noise.wgsl` operation for operation — the multiply constants
/// are odd (hence invertible mod 2³²) and the shift/xor pairs are a standard
/// xorshift-multiply finalizer.
#[inline]
pub fn cloud_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut h = x
        .wrapping_mul(0x8da6_b343)
        .wrapping_add(y.wrapping_mul(0xd816_3841))
        .wrapping_add(z.wrapping_mul(0xcb1a_b31f))
        .wrapping_add(seed.wrapping_mul(0x1652_1623));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    h
}

/// A hash value as a float in `[0, 1)`. 24 bits, so every result is exactly
/// representable in f32 and the CPU and GPU agree on it bit for bit.
#[inline]
pub fn hash_unit(h: u32) -> f32 {
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Quintic fade `6t⁵ − 15t⁴ + 10t³` — C² continuous, so summed octaves have no
/// visible lattice creases. Written in exactly the Horner order `cloud_fade` uses
/// in WGSL.
#[inline]
pub fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Perlin's improved-noise gradient: dot the offset with one of 12 cube-edge
/// directions selected by the low bits of `h`.
///
/// Every component is ±1 or 0, so the dot product is a pair of adds — no
/// multiplication, no table lookup, and nothing that can round differently on a
/// different adapter.
#[inline]
fn grad3(h: u32, x: f32, y: f32, z: f32) -> f32 {
    let hh = h & 15;
    let u = if hh < 8 { x } else { y };
    let v = if hh < 4 {
        y
    } else if hh == 12 || hh == 14 {
        x
    } else {
        z
    };
    let a = if hh & 1 == 0 { u } else { -u };
    let b = if hh & 2 == 0 { v } else { -v };
    a + b
}

/// **Tileable** 3D Perlin gradient noise, returned in roughly `[-1, 1]`.
///
/// `period` is the lattice period: the field repeats exactly every `period` units
/// on each axis, which is what lets the baked 3D texture be sampled with a
/// `Repeat` sampler and tile seamlessly across a 20 km cloud march.
pub fn perlin3_tiled(p: [f32; 3], period: i32, seed: u32) -> f32 {
    let period = period.max(1);
    let fl = [p[0].floor(), p[1].floor(), p[2].floor()];
    let t = [p[0] - fl[0], p[1] - fl[1], p[2] - fl[2]];
    let i = [fl[0] as i32, fl[1] as i32, fl[2] as i32];
    let u = [fade(t[0]), fade(t[1]), fade(t[2])];

    let corner = |dx: i32, dy: i32, dz: i32| -> f32 {
        let wx = (i[0] + dx).rem_euclid(period) as u32;
        let wy = (i[1] + dy).rem_euclid(period) as u32;
        let wz = (i[2] + dz).rem_euclid(period) as u32;
        let h = cloud_hash(wx, wy, wz, seed);
        grad3(h, t[0] - dx as f32, t[1] - dy as f32, t[2] - dz as f32)
    };

    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u[0]);
    let y0 = lerp(x00, x10, u[1]);
    let y1 = lerp(x01, x11, u[1]);
    lerp(y0, y1, u[2])
}

/// **Tileable** 3D Worley (cellular) noise, returned in `[0, 1]` as
/// `1 − distance-to-nearest-feature-point`.
///
/// Inverted so that 1 is the *centre* of a cell: summed, that gives the billowy,
/// cauliflower silhouette a cumulus has, which plain Perlin cannot produce.
/// `cells` is both the grid resolution and the wrap period.
pub fn worley3_tiled(p: [f32; 3], cells: i32, seed: u32) -> f32 {
    let cells = cells.max(1);
    let g = [
        p[0] * cells as f32,
        p[1] * cells as f32,
        p[2] * cells as f32,
    ];
    let base = [g[0].floor(), g[1].floor(), g[2].floor()];
    let f = [g[0] - base[0], g[1] - base[1], g[2] - base[2]];
    let bi = [base[0] as i32, base[1] as i32, base[2] as i32];

    let mut best = 1.0f32;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let wx = (bi[0] + dx).rem_euclid(cells) as u32;
                let wy = (bi[1] + dy).rem_euclid(cells) as u32;
                let wz = (bi[2] + dz).rem_euclid(cells) as u32;
                let h = cloud_hash(wx, wy, wz, seed ^ 0x9e37_79b9);
                let fx = dx as f32 + hash_unit(h);
                let fy = dy as f32 + hash_unit(cloud_hash(h, 1, 0, 0));
                let fz = dz as f32 + hash_unit(cloud_hash(h, 2, 0, 0));
                let d = [fx - f[0], fy - f[1], fz - f[2]];
                let sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if sq < best {
                    best = sq;
                }
            }
        }
    }
    1.0 - best.sqrt().min(1.0)
}

/// Three octaves of [`worley3_tiled`] at `cells`, `2·cells`, `4·cells`, weighted
/// `0.625 / 0.25 / 0.125` (the standard Guerrilla weights).
pub fn worley_fbm(p: [f32; 3], cells: i32, seed: u32) -> f32 {
    worley3_tiled(p, cells, seed) * 0.625
        + worley3_tiled(p, cells * 2, seed.wrapping_add(1)) * 0.25
        + worley3_tiled(p, cells * 4, seed.wrapping_add(2)) * 0.125
}

/// `[a, b] → [0, 1]`, clamped. The remap that makes coverage a *dissolve* rather
/// than a fade: raising the low end erodes the field from its edges inward, which
/// is how a real cloud thins out.
#[inline]
pub fn remap(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0)
}

// ── the baked textures ───────────────────────────────────────────────────────

/// Lattice period of the base Perlin octave inside one shape tile.
pub const SHAPE_PERLIN_PERIOD: i32 = 4;
/// Worley cell count of the base shape octave inside one shape tile.
pub const SHAPE_WORLEY_CELLS: i32 = 4;
/// Worley cell count of the first detail (erosion) octave inside one detail tile.
pub const DETAIL_WORLEY_CELLS: i32 = 2;

/// Highest Worley cell count anywhere in the **shape** volume — the frequency the
/// texture resolution has to resolve.
///
/// This is a real constraint, not bookkeeping: a cell narrower than two texels is
/// not stored, it is *aliased*, and aliased Worley looks like static rather than
/// like clouds. [`CloudQuality`]'s sizes are chosen against it and the tier test
/// asserts the Nyquist relation rather than trusting the table.
pub const SHAPE_MAX_CELLS: i32 = SHAPE_WORLEY_CELLS * 8;
/// Highest Worley cell count in the **detail** volume.
pub const DETAIL_MAX_CELLS: i32 = DETAIL_WORLEY_CELLS * 4;

/// The RGBA8 texel of the **shape** 3D texture at integer coordinate
/// `(x, y, z)` of an `res³` volume.
///
/// * `R` — the Perlin–Worley base: Perlin noise remapped against inverted Worley,
///   which keeps Perlin's connected, wispy topology while giving it Worley's
///   rounded billows. This is the channel that decides where a cloud *is*.
/// * `G`, `B`, `A` — three Worley octaves at increasing frequency, summed by the
///   density function into the erosion that decides what its edge *looks like*.
///
/// **This is the CPU mirror of `cs_cloud_shape` in `shaders/cloud_bake.wgsl`**,
/// texel for texel. The parity gate reads the baked volume back and compares.
pub fn shape_texel(seed: u32, x: u32, y: u32, z: u32, res: u32) -> [u8; 4] {
    shape_value(seed, texel_centre(x, y, z, res)).map(quant)
}

/// The **continuous** shape field at normalized position `p` — what
/// [`shape_texel`] quantizes. Exposed so tileability can be asserted on the field
/// itself (where a seam is a real discontinuity) rather than on two quantized
/// texels a whole texel apart (where a difference proves nothing).
pub fn shape_value(seed: u32, p: [f32; 3]) -> [f32; 4] {
    // Perlin fBm, 3 octaves, mapped to [0, 1].
    let mut perlin = 0.0f32;
    let mut amp = 1.0f32;
    let mut per = SHAPE_PERLIN_PERIOD;
    let mut norm = 0.0f32;
    for o in 0..3u32 {
        let q = [p[0] * per as f32, p[1] * per as f32, p[2] * per as f32];
        perlin += amp * perlin3_tiled(q, per, seed.wrapping_add(o.wrapping_mul(101)));
        norm += amp;
        amp *= 0.5;
        per *= 2;
    }
    let perlin = (perlin / norm * 0.5 + 0.5).clamp(0.0, 1.0);

    let w0 = worley_fbm(p, SHAPE_WORLEY_CELLS, seed.wrapping_add(11));
    // The Perlin–Worley remap (Schneider/Guerrilla): dissolve the Perlin field by
    // the inverted Worley fBm. `w0 - 1` as the low end is what keeps the result in
    // range while letting Worley bite arbitrarily deep.
    let pw = remap(perlin, w0 - 1.0, 1.0);

    // Single octaves, not fBm: three fBms would each reach 4x their base
    // frequency, putting the alpha channel at 128 cells — far past what even a
    // 128^3 volume can store, so what got baked would be aliasing rather than
    // detail. See [`SHAPE_MAX_CELLS`].
    [
        pw,
        worley3_tiled(p, SHAPE_WORLEY_CELLS * 2, seed.wrapping_add(23)),
        worley3_tiled(p, SHAPE_WORLEY_CELLS * 4, seed.wrapping_add(37)),
        worley3_tiled(p, SHAPE_WORLEY_CELLS * 8, seed.wrapping_add(53)),
    ]
}

/// Normalized centre of texel `(x, y, z)` in an `res³` volume. Centres, not
/// corners: a corner convention would make the volume non-tileable at the seam.
#[inline]
fn texel_centre(x: u32, y: u32, z: u32, res: u32) -> [f32; 3] {
    let inv = 1.0 / res.max(1) as f32;
    [
        (x as f32 + 0.5) * inv,
        (y as f32 + 0.5) * inv,
        (z as f32 + 0.5) * inv,
    ]
}

/// The RGBA8 texel of the **detail** (erosion) 3D texture. Three Worley octaves;
/// the alpha channel is unused and pinned to 255 so a debugger sees an opaque
/// volume rather than an invisible one.
///
/// CPU mirror of `cs_cloud_detail` in `shaders/cloud_bake.wgsl`.
pub fn detail_texel(seed: u32, x: u32, y: u32, z: u32, res: u32) -> [u8; 4] {
    detail_value(seed, texel_centre(x, y, z, res)).map(quant)
}

/// The **continuous** detail field at normalized `p`. See [`shape_value`].
pub fn detail_value(seed: u32, p: [f32; 3]) -> [f32; 4] {
    [
        worley3_tiled(p, DETAIL_WORLEY_CELLS, seed.wrapping_add(71)),
        worley3_tiled(p, DETAIL_WORLEY_CELLS * 2, seed.wrapping_add(83)),
        worley3_tiled(p, DETAIL_WORLEY_CELLS * 4, seed.wrapping_add(97)),
        1.0,
    ]
}

/// `[0, 1]` float → RGBA8 LSB, with the same round-half-up the shader's
/// `textureStore` of a normalized format performs.
#[inline]
fn quant(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// ── the weather (2D coverage / type) field ───────────────────────────────────

/// Tileable 2D value noise in `[0, 1]`, on the same integer hash as the 3D field.
fn value2_tiled(x: f32, z: f32, period: i32, seed: u32) -> f32 {
    let period = period.max(1);
    let (fx, fz) = (x.floor(), z.floor());
    let (tx, tz) = (x - fx, z - fz);
    let (ix, iz) = (fx as i32, fz as i32);
    let at = |dx: i32, dz: i32| -> f32 {
        let wx = (ix + dx).rem_euclid(period) as u32;
        let wz = (iz + dz).rem_euclid(period) as u32;
        hash_unit(cloud_hash(wx, 0, wz, seed))
    };
    let (u, w) = (fade(tx), fade(tz));
    lerp(lerp(at(0, 0), at(1, 0), u), lerp(at(0, 1), at(1, 1), u), w)
}

/// Lattice period of the weather field inside one [`WEATHER_TILE_M`] tile.
pub const WEATHER_PERIOD: i32 = 8;

/// How far the raw weather noise is stretched about its midpoint before the
/// authored coverage biases it. See [`weather`] for why this is not cosmetic.
pub const WEATHER_CONTRAST: f32 = 3.0;

/// Slope of the coverage bias. See [`weather`].
pub const COVERAGE_BIAS_SLOPE: f32 = 2.4;
/// Offset of the coverage bias, chosen so `coverage = 0` is exactly cloudless.
pub const COVERAGE_BIAS_OFFSET: f32 = 1.4;

/// The **weather** at world `(x, z)` metres: `[coverage, type]`, both `[0, 1]`.
///
/// This is the 2D map the deliverable calls for, computed analytically rather
/// than baked. That is a deliberate trade: a texture would be one more resizable
/// resource to key a cache on, and this costs four hashes. It is a pure function
/// of `(seed, x, z)` and the authored coverage/type, so it is exactly as
/// deterministic as a baked table would be.
///
/// The authored [`CloudParams::coverage`] biases the field rather than replacing
/// it: at `coverage = 0` nothing survives anywhere, at `1` everything does, and in
/// between the *pattern* of which regions are cloudy is the level's seed.
///
/// CPU mirror of `cloud_weather` in `shaders/cloud_noise.wgsl`.
pub fn weather(seed: u32, x_m: f32, z_m: f32, coverage: f32, cloud_type: f32) -> [f32; 2] {
    let s = WEATHER_PERIOD as f32 / WEATHER_TILE_M;
    let (u, v) = (x_m * s, z_m * s);
    // Two octaves: the synoptic pattern plus a mesoscale break-up.
    let c = value2_tiled(u, v, WEATHER_PERIOD, seed.wrapping_add(211)) * 0.65
        + value2_tiled(u * 3.0, v * 3.0, WEATHER_PERIOD * 3, seed.wrapping_add(223)) * 0.35;
    let t = value2_tiled(u * 2.0, v * 2.0, WEATHER_PERIOD * 2, seed.wrapping_add(233));

    // Widen the raw field before biasing it. Two octaves of interpolated hash
    // pile up around 0.5 — the sum of smooth things is smoother — and a narrow
    // field means the authored slider slides it across the threshold in one go:
    // measured, 0.30 gave 13 % sky cover and 0.45 gave 97 %, so nine tenths of the
    // slider did nothing and a tenth of it did everything. Stretching the field to
    // fill [0, 1] spreads that transition across the range an author actually
    // drags. The stretch is a pure function of the same noise, so nothing about
    // determinism changes.
    let c = ((c - 0.5) * WEATHER_CONTRAST + 0.5).clamp(0.0, 1.0);

    let coverage = coverage.clamp(0.0, 1.0);
    // Bias, not multiply: `coverage` slides the whole field up or down, so 0 is
    // genuinely empty and 1 genuinely solid with a smooth ramp between. The slope
    // and offset are calibrated against the *realised* sky cover rather than
    // against this number: the density function's own threshold means a local
    // `cov` around 0.4 is already visible cloud, so a plain `2c - 1` put the
    // component default at 99 % sky cover. [`COVERAGE_BIAS_SLOPE`] /
    // [`COVERAGE_BIAS_OFFSET`] recentre it so the default reads as broken cumulus,
    // which is what `coverage_is_monotone_and_reaches_both_ends` and the
    // `clouds_scattered` golden between them pin down.
    let cov = (c + (coverage * COVERAGE_BIAS_SLOPE - COVERAGE_BIAS_OFFSET)).clamp(0.0, 1.0);
    // The authored type is the mean; the noise wobbles it by ±0.25 so a sky is
    // not uniformly one cloud species.
    let ty = (cloud_type.clamp(0.0, 1.0) + (t - 0.5) * 0.5).clamp(0.0, 1.0);
    [cov, ty]
}

/// The vertical density profile at relative height `h ∈ [0, 1]` within the slab,
/// for a cloud of type `t ∈ [0, 1]` (0 = stratus, 1 = cumulus).
///
/// Both species taper to zero at the slab's floor and ceiling — a cloud with a
/// hard-edged top is the single most obvious tell of a naive volumetric — but
/// they occupy it differently: a stratus is a thin sheet pinned to the bottom
/// third, a cumulus fills the column and bulges in its upper half.
///
/// CPU mirror of `cloud_height_gradient` in `shaders/cloud_noise.wgsl`.
pub fn height_gradient(h: f32, t: f32) -> f32 {
    let h = h.clamp(0.0, 1.0);
    let t = t.clamp(0.0, 1.0);
    // Stratus: a sheet in [0, 0.35], smooth at both ends.
    let stratus = smoothstep(0.0, 0.08, h) * (1.0 - smoothstep(0.20, 0.40, h));
    // Cumulus: base at 0.05, shoulders through the middle, anvil off by 0.95.
    let cumulus = smoothstep(0.02, 0.22, h) * (1.0 - smoothstep(0.60, 1.0, h));
    stratus + (cumulus - stratus) * t
}

/// Hermite smoothstep, written exactly as WGSL's builtin computes it.
#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Two-lobe Henyey–Greenstein phase function at `cos_theta` with forward
/// asymmetry `g`. CPU mirror of `cloud_phase` in `shaders/cloud_noise.wgsl`.
pub fn phase(cos_theta: f32, g: f32) -> f32 {
    let hg = |g: f32| -> f32 {
        let g2 = g * g;
        (1.0 - g2)
            / (4.0 * std::f32::consts::PI * (1.0 + g2 - 2.0 * g * cos_theta).max(1e-4).powf(1.5))
    };
    let g = g.clamp(0.0, 0.95);
    hg(g) * FORWARD_LOBE_WEIGHT + hg(BACK_LOBE_G * (g / 0.8).min(1.0)) * (1.0 - FORWARD_LOBE_WEIGHT)
}

/// Octaves of the Hillaire multiple-scattering approximation the march sums.
pub const MS_OCTAVES: u32 = 3;

/// Exponent of the **powder** term (SKY2). `1 − T^K == 1 − exp(−K·τ)`, i.e. the
/// Guerrilla `1 − exp(−2d)` written against a transmittance rather than an
/// optical depth, so it costs one multiply instead of a second march.
pub const POWDER_K: f32 = 2.0;

/// The sun's contribution at one march sample, per unit extinction: Hillaire's
/// multiple-scattering octaves, with the **powder** correction on the first.
///
/// CPU mirror of `cloud_sun_energy` in `shaders/cloud.wgsl`.
///
/// # What powder is for
///
/// Beer's law says a point with nothing between it and the sun is fully lit, and
/// for a *volume* that is the wrong answer: the light gets there, but there is
/// almost no material at that point to scatter it toward the eye, so it
/// contributes almost nothing. Leaving the correction out is what makes a
/// volumetric read as airbrushed — the sunward face and the shaded face differ
/// only by a shadow, and the medium never darkens where it thins.
///
/// Three things gate it, and each is a defect the other two do not catch:
///
/// * **`facing`** — the deficit is only *visible* from the side turned away from
///   the sun. Applied looking into the sun it would erase the silver lining,
///   which is the one effect the two-lobe phase function exists to produce.
/// * **`sun_y`** — below the horizon `sun_t` is the documented early-out's `1.0`,
///   which means "no march was run" and not "no material". Read as an optical
///   depth of zero it takes the entire single-scattering term off a night sky:
///   measured at a **45 % drop** in `clouds_night`'s starless cloud brightness
///   before the gate existed.
/// * **the first octave only** — octaves 2 and 3 stand in for light that has
///   already bounced several times inside the layer and arrives from every
///   direction rather than along the sun ray. Darkening those takes back exactly
///   the term that keeps a thick deck's interior out of soot.
pub fn sun_energy(sun_t: f32, cos_theta: f32, g: f32, sun_y: f32) -> f32 {
    let st = sun_t.clamp(0.0, 1.0);
    let powder = 1.0 - st.powf(POWDER_K);
    let facing = (-cos_theta * 0.5 + 0.5).clamp(0.0, 1.0);
    let single = if sun_y > 1e-3 {
        1.0 + (powder - 1.0) * facing
    } else {
        1.0
    };

    let mut e = 0.0f32;
    let mut att = 1.0f32;
    let mut sca = 1.0f32;
    let mut ecc = 1.0f32;
    for n in 0..MS_OCTAVES {
        let p = if n == 0 { single } else { 1.0 };
        e += sca * sun_t.max(0.0).powf(att) * phase(cos_theta, g * ecc) * p;
        att *= 0.5;
        sca *= 0.5;
        ecc *= 0.5;
    }
    e
}

// ── the CPU reference of the density function ────────────────────────────────

/// A read-back copy of the two baked 3D volumes, so the CPU reference evaluates
/// **the same field the GPU does** rather than a re-derivation of it.
///
/// This is the whole point of the parity gate's design: comparing a CPU density
/// against a GPU density is only meaningful if any disagreement is attributable
/// to the *density function*. Re-baking the noise on the CPU would fold the bake's
/// own (already separately gated) rounding into the answer.
pub struct CloudVolumes {
    /// `res³` RGBA8 texels, x-major then y then z (the compute shader's order).
    pub shape: Vec<u8>,
    pub shape_res: u32,
    pub detail: Vec<u8>,
    pub detail_res: u32,
}

impl CloudVolumes {
    /// Trilinearly filtered sample of a wrapping RGBA8 volume, mirroring a
    /// `Repeat` + `Linear` sampler. Returns the four channels in `[0, 1]`.
    ///
    /// **Precision note:** real hardware trilinear filtering carries only ~8 bits
    /// of sub-texel precision, while this computes in full f32 — which is exactly
    /// why the shadow parity gate uses a relative envelope
    /// ([`CPU_GPU_SHADOW_TOLERANCE`]) rather than an equality.
    fn sample(data: &[u8], res: u32, uvw: [f32; 3]) -> [f32; 4] {
        let res_f = res as f32;
        let mut out = [0.0f32; 4];
        // Texel-centre convention: uvw * res - 0.5.
        let c = [
            uvw[0] * res_f - 0.5,
            uvw[1] * res_f - 0.5,
            uvw[2] * res_f - 0.5,
        ];
        let base = [c[0].floor(), c[1].floor(), c[2].floor()];
        let f = [c[0] - base[0], c[1] - base[1], c[2] - base[2]];
        for k in 0i32..8 {
            let o = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
            let w = (if o[0] == 0 { 1.0 - f[0] } else { f[0] })
                * (if o[1] == 0 { 1.0 - f[1] } else { f[1] })
                * (if o[2] == 0 { 1.0 - f[2] } else { f[2] });
            let ix = (base[0] as i32 + o[0]).rem_euclid(res as i32) as usize;
            let iy = (base[1] as i32 + o[1]).rem_euclid(res as i32) as usize;
            let iz = (base[2] as i32 + o[2]).rem_euclid(res as i32) as usize;
            let idx = ((iz * res as usize + iy) * res as usize + ix) * 4;
            for ch in 0..4 {
                out[ch] += w * (data[idx + ch] as f32 / 255.0);
            }
        }
        out
    }

    /// The cloud **density** at world position `p` metres — the CPU mirror of
    /// `cloud_density` in `shaders/cloud_noise.wgsl`, evaluated against the same
    /// baked volumes.
    ///
    /// Returns extinction in **m⁻¹** (0 outside the slab).
    pub fn density(&self, params: &CloudParams, p: [f32; 3]) -> f32 {
        let thickness = params.thickness();
        if thickness <= 0.0 {
            return 0.0;
        }
        let h = (p[1] - params.bottom) / thickness;
        if !(0.0..=1.0).contains(&h) {
            return 0.0;
        }
        let off = params.wind_offset();
        let seed = params.masked_seed();

        // Weather is sampled at the *undrifted* position: the coverage pattern is
        // the geography of the weather system, and a weather system that slid with
        // its own clouds would never let a gap pass overhead.
        let [cov, ty] = weather(seed, p[0], p[2], params.coverage, params.cloud_type);
        if cov <= 0.0 {
            return 0.0;
        }
        let grad = height_gradient(h, ty);
        if grad <= 0.0 {
            return 0.0;
        }

        // The shape volume drifts with the wind; the vertical axis maps the slab
        // onto one full wrap so a tall cloud is not a stretched copy of a short one.
        let sp = [
            (p[0] + off[0]) / SHAPE_TILE_M,
            p[1] / thickness.max(1.0),
            (p[2] + off[1]) / SHAPE_TILE_M,
        ];
        let s = Self::sample(&self.shape, self.shape_res, sp);
        let low = s[1] * 0.625 + s[2] * 0.25 + s[3] * 0.125;
        let shape = remap(s[0], low - 1.0, 1.0) * grad;

        // Coverage as a dissolve: raising the floor eats the field from its edges.
        let mut d = remap(shape, 1.0 - cov, 1.0) * cov;
        if d <= 0.0 {
            return 0.0;
        }

        // Erosion. Wispy (inverted) at the base, billowy at the top — the standard
        // trick that makes a cumulus fray downward and stay solid on its shoulders.
        let detail_strength = params.detail.clamp(0.0, 1.0);
        if detail_strength > 0.0 {
            // Detail drifts faster than the shape: relative motion inside a cloud
            // is what makes it look alive rather than like a rigid sculpture.
            let dp = [
                (p[0] + off[0] * 2.0) / DETAIL_TILE_M,
                p[1] / DETAIL_TILE_M,
                (p[2] + off[1] * 2.0) / DETAIL_TILE_M,
            ];
            let e = Self::sample(&self.detail, self.detail_res, dp);
            let fbm = e[0] * 0.625 + e[1] * 0.25 + e[2] * 0.125;
            let wispy = fbm + (1.0 - fbm - fbm) * smoothstep(0.0, 0.35, h);
            d = remap(d, wispy * 0.6 * detail_strength, 1.0);
        }
        d.max(0.0) * params.density.max(0.0)
    }

    /// Beer–Lambert transmittance of the cloud layer from `p` toward `sun_dir`
    /// (unit, pointing **at** the sun), marched in `steps` steps.
    ///
    /// CPU mirror of the inner march of `cs_cloud_shadow` in
    /// `shaders/cloud_bake.wgsl`; the shadow parity gate compares this against the
    /// baked shadow map.
    pub fn sun_transmittance(
        &self,
        params: &CloudParams,
        p: [f32; 3],
        sun_dir: [f32; 3],
        steps: u32,
    ) -> f32 {
        let steps = steps.max(1);
        // The slab crossing length along the sun ray, capped so a sun on the
        // horizon does not march to the other side of the world.
        let sy = sun_dir[1];
        if sy <= 1e-3 {
            return 1.0;
        }
        let span = (params.thickness() / sy).min(MAX_SHADOW_MARCH_M);
        let dt = span / steps as f32;
        let mut depth = 0.0f32;
        for i in 0..steps {
            let t = (i as f32 + 0.5) * dt;
            let q = [
                p[0] + sun_dir[0] * t,
                p[1] + sun_dir[1] * t,
                p[2] + sun_dir[2] * t,
            ];
            depth += self.density(params, q) * dt;
        }
        (-depth).exp()
    }
}

/// Longest sun ray the shadow bake will march, **metres**. A sun 2° above the
/// horizon would otherwise want a 70 km path through the slab; capping it keeps
/// the bake's cost bounded and its error confined to the moments when the sun is
/// already reddened into insignificance.
pub const MAX_SHADOW_MARCH_M: f32 = 20_000.0;

// ── quality tiers ────────────────────────────────────────────────────────────

/// Cloud noise-texture sizes, march budgets and shadow-map resolution, derived
/// from [`AtmosphereQuality`] rather than authored separately.
///
/// One knob, not two: a machine that can afford a 256×64 transmittance LUT can
/// afford a 128³ cloud volume, and letting them disagree would only ever produce
/// combinations nobody tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CloudQuality {
    /// 64³ shape / 16³ detail / 256² shadow, 32-step primary march.
    Low,
    /// 96³ shape / 24³ detail / 384² shadow, 64-step primary march.
    Medium,
    /// 128³ shape / 32³ detail / 512² shadow, 96-step primary march.
    High,
}

impl CloudQuality {
    /// The cloud tier that goes with an atmosphere tier.
    pub fn from_atmosphere(q: AtmosphereQuality) -> Self {
        match q {
            AtmosphereQuality::High => CloudQuality::High,
            AtmosphereQuality::Medium => CloudQuality::Medium,
            AtmosphereQuality::Low => CloudQuality::Low,
        }
    }

    /// Edge of the cubic **shape** volume.
    pub fn shape_res(self) -> u32 {
        match self {
            CloudQuality::High => 128,
            CloudQuality::Medium => 96,
            CloudQuality::Low => 64,
        }
    }

    /// Edge of the cubic **detail** (erosion) volume.
    pub fn detail_res(self) -> u32 {
        match self {
            CloudQuality::High => 32,
            CloudQuality::Medium => 24,
            CloudQuality::Low => 16,
        }
    }

    /// Edge of the square cloud-**shadow** map.
    pub fn shadow_res(self) -> u32 {
        match self {
            CloudQuality::High => 512,
            CloudQuality::Medium => 384,
            CloudQuality::Low => 256,
        }
    }

    /// Steps of the primary (view-ray) march. The march is adaptive — it takes
    /// long strides through empty air and refines on contact — so this is a
    /// **ceiling**, not a per-pixel cost.
    pub fn march_steps(self) -> u32 {
        match self {
            CloudQuality::High => 96,
            CloudQuality::Medium => 64,
            CloudQuality::Low => 32,
        }
    }

    /// Steps of the secondary (toward-the-sun) transmittance march at each
    /// primary sample. Kept tiny: this is the term that multiplies the primary
    /// cost, and a cloud's self-shadowing is low-frequency.
    pub fn light_steps(self) -> u32 {
        match self {
            CloudQuality::High => 6,
            CloudQuality::Medium => 5,
            CloudQuality::Low => 4,
        }
    }

    /// Steps of the cloud-shadow-map bake's march.
    pub fn shadow_steps(self) -> u32 {
        match self {
            CloudQuality::High => 16,
            CloudQuality::Medium => 12,
            CloudQuality::Low => 8,
        }
    }

    /// Whether the primary march reads the **detail** volume at all.
    ///
    /// This is the documented Low-tier cheat: a Low-tier march skips the erosion
    /// texture entirely and shades the smooth base shape. The silhouette loses its
    /// wisps, which is visible; it stays a *pure function of the same inputs*,
    /// which is what determinism requires. (A billboard fallback would not — it
    /// would need a screen-space fade, and screen-space is where determinism goes
    /// to die.)
    pub fn uses_detail(self) -> bool {
        !matches!(self, CloudQuality::Low)
    }

    /// Approximate GPU cost of the once-per-seed bake, in texels written. Reported
    /// by the tier documentation rather than measured, because it happens once.
    pub fn bake_texels(self) -> u64 {
        let s = self.shape_res() as u64;
        let d = self.detail_res() as u64;
        s * s * s + d * d * d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> CloudParams {
        CloudParams {
            enabled: true,
            ..CloudParams::default()
        }
    }

    /// The default block is inert: nothing that has not opted in can be touched.
    #[test]
    fn default_is_disabled_and_inert() {
        let c = CloudParams::default();
        assert!(!c.enabled);
        assert!(!c.active());
        assert!(!c.shadows_world());
        // ...but the authored defaults are already a usable sky, so enabling is
        // one flag rather than a parameter hunt.
        assert!(on().active());
        assert!(on().shadows_world());
        // Zeroing either of the two knobs opts back out of the lit passes.
        let quiet = CloudParams {
            shadow_strength: 0.0,
            ..on()
        };
        assert!(quiet.active() && !quiet.shadows_world());
        let empty = CloudParams {
            coverage: 0.0,
            ..on()
        };
        assert!(!empty.active());
    }

    /// The whole determinism law, as a property: the field is a pure function of
    /// `(seed, position)` and nothing else.
    #[test]
    fn noise_is_seed_stable_and_pure() {
        for &(x, y, z) in &[(0u32, 0u32, 0u32), (7, 3, 19), (63, 63, 63), (17, 40, 5)] {
            let a = shape_texel(1234, x, y, z, 64);
            let b = shape_texel(1234, x, y, z, 64);
            assert_eq!(a, b, "shape texel not pure at ({x},{y},{z})");
            let d = detail_texel(1234, x, y, z, 32);
            assert_eq!(d, detail_texel(1234, x, y, z, 32));
        }
        // A different seed is a different field (somewhere, not everywhere).
        let differs = (0..32).any(|i| shape_texel(1, i, i, i, 64) != shape_texel(2, i, i, i, 64));
        assert!(differs, "the seed does not reach the field");
    }

    /// The **committed** values: if any of these ever change, a level's sky
    /// changed under it and a golden has to be re-blessed on purpose. This is the
    /// bit-stability half of the determinism law, and it runs with no GPU.
    #[test]
    fn committed_noise_values_are_bit_stable() {
        // Seed 0 (the component default), a 64³ volume, four scattered texels.
        assert_eq!(shape_texel(0, 0, 0, 0, 64), [178, 119, 83, 184]);
        assert_eq!(shape_texel(0, 31, 17, 5, 64), [153, 99, 96, 92]);
        assert_eq!(shape_texel(0, 63, 63, 63, 64), [174, 76, 119, 108]);
        assert_eq!(detail_texel(0, 9, 21, 30, 32), [113, 150, 144, 255]);
        // The hash itself, at the bottom of everything.
        assert_eq!(cloud_hash(0, 0, 0, 0), 0);
        assert_eq!(cloud_hash(1, 2, 3, 4), 2_928_021_154);
        assert_eq!(cloud_hash(0xffff_ffff, 0, 0, 7), 2_111_138_898);
    }

    /// The noise must not depend on how the bake is *scheduled* — the "pool
    /// sizes" half of the determinism law. Each texel is computed from its own
    /// coordinate alone, so evaluating them in any order (here: reversed, and
    /// chunked) gives the identical volume.
    #[test]
    fn bake_order_does_not_change_the_volume() {
        let res = 16;
        let forward: Vec<[u8; 4]> = (0..res * res * res)
            .map(|i| {
                let (x, y, z) = (i % res, (i / res) % res, i / (res * res));
                shape_texel(99, x, y, z, res)
            })
            .collect();
        let mut reversed: Vec<[u8; 4]> = (0..res * res * res)
            .rev()
            .map(|i| {
                let (x, y, z) = (i % res, (i / res) % res, i / (res * res));
                shape_texel(99, x, y, z, res)
            })
            .collect();
        reversed.reverse();
        assert_eq!(forward, reversed);
    }

    /// Tileability is what lets the march wrap a 20 km ray through an 8 km
    /// texture without a seam. Asserted on the underlying generators, where a
    /// failure is legible, rather than on the quantized texels.
    #[test]
    fn noise_tiles_seamlessly() {
        for i in 0..24 {
            let t = i as f32 / 24.0;
            // Perlin: p and p + period are the same lattice point.
            let a = perlin3_tiled([t, t * 2.0, t * 0.5], 4, 7);
            let b = perlin3_tiled([t + 4.0, t * 2.0, t * 0.5], 4, 7);
            assert!((a - b).abs() < 1e-5, "perlin seam at {t}: {a} vs {b}");
            // Worley wraps in the *unit* domain (cells span [0,1)).
            let c = worley3_tiled([t, 0.3, 0.7], 4, 7);
            let d = worley3_tiled([t + 1.0, 0.3, 0.7], 4, 7);
            assert!((c - d).abs() < 1e-5, "worley seam at {t}: {c} vs {d}");
        }
        // ...and so does the composed field, on every channel and every axis.
        // Asserted on the CONTINUOUS value rather than on two quantized texels a
        // whole texel apart: the latter would differ for perfectly good reasons
        // (they are different points), which makes it a test of nothing.
        for i in 0..16 {
            let t = i as f32 / 16.0;
            let p = [t, t * 0.37 + 0.1, t * 0.71 + 0.2];
            for axis in 0..3 {
                let mut q = p;
                q[axis] += 1.0;
                let a = shape_value(3, p);
                let b = shape_value(3, q);
                for c in 0..4 {
                    assert!(
                        (a[c] - b[c]).abs() < 1e-4,
                        "shape channel {c} seams on axis {axis} at {t}: {} vs {}",
                        a[c],
                        b[c]
                    );
                }
                let a = detail_value(3, p);
                let b = detail_value(3, q);
                for c in 0..3 {
                    assert!(
                        (a[c] - b[c]).abs() < 1e-4,
                        "detail channel {c} seams on axis {axis} at {t}"
                    );
                }
            }
        }
    }

    /// Both generators must stay in the range the density function assumes, or
    /// the remaps silently clip and the sky turns into a flat sheet.
    #[test]
    fn generators_stay_in_range() {
        for i in 0..500 {
            let p = [i as f32 * 0.017, i as f32 * -0.031 + 0.5, i as f32 * 0.0073];
            let w = worley3_tiled(p, 4, 5);
            assert!((0.0..=1.0).contains(&w), "worley out of range: {w}");
            let f = worley_fbm(p, 4, 5);
            assert!((0.0..=1.0001).contains(&f), "worley fbm out of range: {f}");
            let pn = perlin3_tiled(p, 4, 5);
            assert!(pn.abs() <= 1.5, "perlin out of range: {pn}");
        }
    }

    /// Coverage must be monotone or the World Settings slider lies. Asserted on
    /// the mean of the weather field over a wide area — the *realised* coverage,
    /// not the authored number.
    #[test]
    fn coverage_is_monotone_and_reaches_both_ends() {
        let mean = |c: f32| -> f32 {
            let mut sum = 0.0;
            let n = 40;
            for i in 0..n {
                for j in 0..n {
                    let x = i as f32 * 1500.0;
                    let z = j as f32 * 1500.0;
                    sum += weather(0, x, z, c, 0.7)[0];
                }
            }
            sum / (n * n) as f32
        };
        let mut prev = -1.0;
        for i in 0..=10 {
            let c = i as f32 / 10.0;
            let m = mean(c);
            assert!(
                m + 1e-4 >= prev,
                "coverage not monotone at {c}: {m} < {prev}"
            );
            prev = m;
        }
        assert_eq!(mean(0.0), 0.0, "coverage 0 must be a cloudless sky");
        assert!(mean(1.0) > 0.99, "coverage 1 must be solid overcast");
    }

    /// The vertical profile: zero at both boundaries for every species, and the
    /// two species genuinely differ in where they put their mass.
    #[test]
    fn height_gradient_closes_at_both_ends() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            assert_eq!(height_gradient(0.0, t), 0.0, "open floor at type {t}");
            assert_eq!(height_gradient(1.0, t), 0.0, "open ceiling at type {t}");
            // Non-degenerate somewhere in between.
            let peak = (0..=20)
                .map(|k| height_gradient(k as f32 / 20.0, t))
                .fold(0.0f32, f32::max);
            assert!(peak > 0.5, "type {t} has no cloud at all: {peak}");
        }
        // Stratus lives low, cumulus lives high — the defining difference.
        assert!(height_gradient(0.15, 0.0) > height_gradient(0.15, 1.0));
        assert!(height_gradient(0.55, 1.0) > height_gradient(0.55, 0.0));
    }

    /// The two-lobe phase must be forward-dominant, strictly positive, and
    /// carry a visible back lobe — the term the silver lining comes from.
    #[test]
    fn phase_is_two_lobed_and_positive() {
        let g = 0.8;
        let fwd = phase(1.0, g);
        let side = phase(0.0, g);
        let back = phase(-1.0, g);
        assert!(fwd > side && side > 0.0, "{fwd} {side}");
        assert!(back > side, "no back lobe: {back} vs {side}");
        assert!(fwd > back * 2.0, "back lobe swamps the forward one");
        for i in 0..=64 {
            let c = -1.0 + i as f32 / 32.0;
            let p = phase(c, g);
            assert!(p.is_finite() && p > 0.0, "phase {p} at cos {c}");
        }
        // g = 0 is (nearly) isotropic in the forward lobe.
        let iso = phase(1.0, 0.0);
        assert!(
            (iso - phase(-1.0, 0.0)).abs() < 0.02,
            "g=0 not near-isotropic"
        );
    }

    /// **The powder term does the three things it exists to do, and nothing
    /// else.** Each assertion here is a defect the wave actually had to fix or
    /// deliberately avoid, written as the property that catches it.
    #[test]
    fn powder_darkens_the_thin_sunward_side_only() {
        let g = 0.8;
        let up = 0.5; // a sun well above the horizon

        // (a) A point with nothing between it and the sun — the thin edge of a
        // cloud — is DARKER than Beer alone says, seen with the sun behind the
        // eye. That is the dark rim, and it is the whole term.
        let lit_edge = sun_energy(1.0, -1.0, g, up);
        let beer_edge = sun_energy(1.0, -1.0, g, -1.0); // same call, powder gated off
        assert!(
            lit_edge < beer_edge * 0.75,
            "powder did not darken a thin sunward edge: {lit_edge} vs {beer_edge}"
        );

        // (b) ...and it leaves the SILVER LINING alone. Looking into the sun the
        // eye sees forward-scattered light from the near surface; darkening that
        // would erase the effect the two-lobe phase exists for.
        let into_sun = sun_energy(1.0, 1.0, g, up);
        let into_sun_beer = sun_energy(1.0, 1.0, g, -1.0);
        assert_eq!(
            into_sun, into_sun_beer,
            "powder reached the forward lobe: {into_sun} vs {into_sun_beer}"
        );

        // (c) A sun below the horizon is the early-out, not an optical depth of
        // zero. Every geometry must be untouched at night.
        for &c in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            assert_eq!(
                sun_energy(1.0, c, g, -0.3),
                sun_energy(1.0, c, g, 0.0),
                "the night path moved at cos {c}"
            );
        }

        // (d) DEEP inside a cloud, where the sun ray is already extinguished,
        // powder is ~1 and takes nothing away — an overcast deck must not go to
        // soot. Measured against the same sample with the term gated off.
        let deep = sun_energy(0.02, -1.0, g, up);
        let deep_beer = sun_energy(0.02, -1.0, g, -1.0);
        assert!(
            deep > deep_beer * 0.995,
            "powder darkened a fully-shadowed interior: {deep} vs {deep_beer}"
        );

        // (e) The multiple-scattering octaves survive. Powder on ALL of them
        // would drive the edge to zero; on the first only it cannot, because the
        // remaining octaves carry 0.5 + 0.25 of the weight.
        assert!(
            lit_edge > 0.0,
            "the powdered edge went to zero — powder reached the MS octaves"
        );
        // Monotone and finite over the whole domain, so no configuration of a
        // level can produce a NaN in the march's accumulator.
        for i in 0..=32 {
            let t = i as f32 / 32.0;
            let e = sun_energy(t, -0.3, g, up);
            assert!(e.is_finite() && e >= 0.0, "sun_energy({t}) = {e}");
        }
    }

    /// The wind is a deterministic function of the level's clock, wraps into one
    /// tile, and survives hostile input.
    #[test]
    fn wind_drifts_deterministically_and_wraps() {
        let a = wind_offset(6.0, 2.0, 3600.0);
        assert_eq!(a, wind_offset(6.0, 2.0, 3600.0));
        // 6 m/s for an hour is 21.6 km = 2 tiles + 5216 m.
        assert!(
            (a[0] - (21_600.0 - 2.0 * SHAPE_TILE_M)).abs() < 1e-3,
            "{a:?}"
        );
        // Always inside one tile, however long the level has been running.
        for t in [0.0, 1.0, 86_400.0, 1e9, -3600.0] {
            let o = wind_offset(11.0, -7.0, t);
            assert!(o[0] >= 0.0 && o[0] < SHAPE_TILE_M, "t={t}: {o:?}");
            assert!(o[1] >= 0.0 && o[1] < SHAPE_TILE_M, "t={t}: {o:?}");
        }
        // Non-finite input cannot poison the uniform.
        let bad = wind_offset(f32::NAN, f32::INFINITY, 1.0);
        assert_eq!(bad, [0.0, 0.0]);
        assert_eq!(wind_offset(1.0, 1.0, f64::NAN), [0.0, 0.0]);
        // Zero wind is a static sky.
        assert_eq!(wind_offset(0.0, 0.0, 99_999.0), [0.0, 0.0]);
    }

    /// **The jitter's determinism story, as a property.** The march's blue-noise
    /// offset advances with the LEVEL CLOCK and with nothing else: the same clock
    /// is the same offset however many frames the process has drawn, a paused
    /// clock is a frozen pattern, and the phase stays an exactly-representable
    /// small integer however long the level has been running.
    #[test]
    fn the_jitter_phase_follows_the_level_clock_and_nothing_else() {
        // Pure: the same clock gives the same phase, always.
        for t in [0.0, 1.0 / 60.0, 43_200.0, 86_400.0, 1e9] {
            assert_eq!(jitter_phase(t), jitter_phase(t));
            let p = jitter_phase(t);
            assert!(p.is_finite() && (0.0..JITTER_PHASE_PERIOD as f32).contains(&p));
            // Integer-valued, so it survives the f32 uniform exactly and the
            // shader's multiply by the golden ratio is reproducible.
            assert_eq!(p, p.floor(), "phase {p} is not an integer at t={t}");
        }
        // A clock that advances by one frame at 60 Hz advances the sequence —
        // which is what the temporal pass averages over.
        assert_ne!(jitter_phase(10.0), jitter_phase(10.0 + 1.0 / 60.0));
        // A clock that does NOT advance does not: a paused editor renders the
        // same sky twice, which is what the golden harness asserts.
        assert_eq!(jitter_phase(10.0), jitter_phase(10.0 + 1e-9));
        // The wrap: 19 hours of level time is past 2^24 unwrapped, and the phase
        // is still a small exact integer.
        let far = jitter_phase(1e9);
        assert!(far < JITTER_PHASE_PERIOD as f32, "{far}");
        // Hostile input cannot poison the uniform.
        assert_eq!(jitter_phase(f64::NAN), 0.0);
        assert_eq!(jitter_phase(f64::INFINITY), 0.0);
        // Negative clocks (a level rewound past midnight) stay in range.
        assert!((0.0..JITTER_PHASE_PERIOD as f32).contains(&jitter_phase(-7.5)));
    }

    /// The seed's round trip through the f32 uniform must be exact, and must
    /// never produce the `NaN` that would re-upload the uniform forever.
    #[test]
    fn seed_survives_the_uniform_exactly() {
        for s in [0u32, 1, 12345, CloudGpuSeed::MASK] {
            let e = CloudGpuSeed::encode(s);
            assert!(e.is_finite());
            assert_eq!(e as u32, s, "seed {s} did not round-trip");
        }
        // Above the mask it truncates rather than lying about precision.
        let big = CloudGpuSeed::encode(u32::MAX);
        assert_eq!(big as u32, CloudGpuSeed::MASK);
        assert!(big.is_finite());
    }

    /// The CPU density reference behaves: zero outside the slab, zero where the
    /// weather says clear, and monotone in the authored density.
    #[test]
    fn density_reference_respects_the_slab() {
        let res = 16;
        let dres = 8;
        let mut shape = Vec::with_capacity((res * res * res * 4) as usize);
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    shape.extend_from_slice(&shape_texel(0, x, y, z, res));
                }
            }
        }
        let mut detail = Vec::with_capacity((dres * dres * dres * 4) as usize);
        for z in 0..dres {
            for y in 0..dres {
                for x in 0..dres {
                    detail.extend_from_slice(&detail_texel(0, x, y, z, dres));
                }
            }
        }
        let vols = CloudVolumes {
            shape,
            shape_res: res,
            detail,
            detail_res: dres,
        };
        let p = CloudParams {
            coverage: 1.0,
            ..on()
        };
        // Outside the slab, in both directions.
        assert_eq!(vols.density(&p, [0.0, 0.0, 0.0]), 0.0);
        assert_eq!(vols.density(&p, [0.0, 9999.0, 0.0]), 0.0);
        // Inside it, solid overcast has to produce *something* somewhere.
        let any = (0..64).any(|i| {
            let x = i as f32 * 137.0;
            vols.density(&p, [x, 2400.0, x * 0.7]) > 0.0
        });
        assert!(any, "an overcast slab produced no cloud anywhere");
        // A cloudless sky produces nothing anywhere.
        let clear = CloudParams { coverage: 0.0, ..p };
        for i in 0..64 {
            let x = i as f32 * 137.0;
            assert_eq!(vols.density(&clear, [x, 2400.0, x * 0.7]), 0.0);
        }
        // Density scales the extinction linearly (it is a multiplier, by design).
        let dense = CloudParams {
            density: p.density * 2.0,
            ..p
        };
        for i in 0..32 {
            let q = [i as f32 * 211.0, 2600.0, i as f32 * -97.0];
            let a = vols.density(&p, q);
            let b = vols.density(&dense, q);
            assert!((b - 2.0 * a).abs() < 1e-6, "{a} {b}");
        }
        // The sun transmittance is bounded and deterministic.
        let sun = [0.3, 0.9, 0.31];
        let t = vols.sun_transmittance(&p, [0.0, 1600.0, 0.0], sun, 8);
        assert!((0.0..=1.0).contains(&t), "{t}");
        assert_eq!(t, vols.sun_transmittance(&p, [0.0, 1600.0, 0.0], sun, 8));
        // A sun on the horizon is the documented early-out, not a 70 km march.
        assert_eq!(
            vols.sun_transmittance(&p, [0.0, 1600.0, 0.0], [1.0, 0.0, 0.0], 8),
            1.0
        );
    }

    // ── tiers ───────────────────────────────────────────────────────────────

    #[test]
    fn quality_shrinks_with_the_tier() {
        use CloudQuality::*;
        for (lo, hi) in [(Low, Medium), (Medium, High)] {
            assert!(lo.shape_res() < hi.shape_res());
            assert!(lo.detail_res() < hi.detail_res());
            assert!(lo.shadow_res() < hi.shadow_res());
            assert!(lo.march_steps() < hi.march_steps());
            assert!(lo.light_steps() < hi.light_steps());
            assert!(lo.shadow_steps() < hi.shadow_steps());
            assert!(lo.bake_texels() < hi.bake_texels());
        }
        // The documented Low-tier cheat, pinned so it cannot silently spread.
        assert!(!Low.uses_detail());
        assert!(Medium.uses_detail() && High.uses_detail());
        // NYQUIST — the one sizing constraint that is not free. A Worley cell
        // narrower than two texels is not *stored*, it is aliased, and aliased
        // Worley reads as television static rather than as cloud. (Tileability
        // needs no divisibility: the field is periodic with period 1 in uvw by
        // construction, whatever the resolution — see `noise_tiles_seamlessly`.)
        for q in [Low, Medium, High] {
            assert!(
                q.shape_res() >= 2 * SHAPE_MAX_CELLS as u32,
                "{q:?}: {}^3 cannot resolve {SHAPE_MAX_CELLS} Worley cells",
                q.shape_res()
            );
            assert!(
                q.detail_res() >= 2 * DETAIL_MAX_CELLS as u32,
                "{q:?}: {}^3 cannot resolve {DETAIL_MAX_CELLS} Worley cells",
                q.detail_res()
            );
        }
    }

    #[test]
    fn quality_follows_the_atmosphere_tier() {
        assert_eq!(
            CloudQuality::from_atmosphere(AtmosphereQuality::High),
            CloudQuality::High
        );
        assert_eq!(
            CloudQuality::from_atmosphere(AtmosphereQuality::Medium),
            CloudQuality::Medium
        );
        assert_eq!(
            CloudQuality::from_atmosphere(AtmosphereQuality::Low),
            CloudQuality::Low
        );
    }
}
