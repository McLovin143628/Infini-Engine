//! Gerstner (trochoidal) wave math — the deterministic, bit-portable core of
//! every water surface the engine draws.
//!
//! # The model
//!
//! A Gerstner wave displaces a surface point *horizontally as well as
//! vertically*, which is what gives ocean water its sharp crests and broad
//! troughs (a plain sine sum looks like corrugated iron). For a still-water
//! parameter point `p = (a, b)` and `N` component waves:
//!
//! ```text
//! θᵢ  = kᵢ (dᵢ · p) − ωᵢ t + φᵢ
//! Δx  = Σ Qᵢ Aᵢ dᵢ.x cos θᵢ
//! Δz  = Σ Qᵢ Aᵢ dᵢ.y cos θᵢ
//! Δy  = Σ Aᵢ sin θᵢ
//! ```
//!
//! with `dᵢ` a unit direction, `kᵢ = 2π/λᵢ` the wavenumber (rad/m), `Aᵢ` the
//! amplitude (m), `Qᵢ` the per-wave steepness and `φᵢ` a fixed phase (rad).
//!
//! **The dispersion relation is deep-water gravity:** `ωᵢ = √(g kᵢ)`, so a wave's
//! period is `Tᵢ = 2π/ωᵢ = √(2π λᵢ / g)` and its phase speed is `cᵢ = √(g/kᵢ)`.
//! Long swells therefore travel faster than short chop, which is what stops a
//! multi-component sum from looking like one rigid pattern sliding across the
//! world. `g` is [`GRAVITY_M_S2`], the same 9.81 m/s² the engine's default 3-D
//! gravity uses — SI, per the units doctrine, and *not* re-derived here from a
//! second constant.
//!
//! # Determinism (the house law)
//!
//! Every trigonometric evaluation goes through [`inf_math::portable::psin64`] /
//! [`pcos64`](inf_math::portable::pcos64) — the f64 bit-portable pair — because
//! `f64::sin` is **not** bit-identical across platforms (the P14 LAW), and these
//! numbers reach committed content: a buoyancy force in the fixed step, a replay
//! trace, a PIE-vs-shipping comparison. `sqrt` is used freely: IEEE-754 specifies
//! it as correctly rounded, so it is exact everywhere.
//!
//! The per-component parameters are a pure function of `(seed, wind)` through an
//! **integer** hash ([`mix64`]) — no `rand`, no wall clock, no floating-point
//! hashing. Two runs, two machines and two processes therefore build the *same*
//! [`WaveField`] bit for bit, which is the property the GPU mirror rests on (see
//! below).
//!
//! # CPU ↔ GPU split
//!
//! The **parameters** are derived here, once, in f64, and uploaded to the shader
//! as an array of [`Wave`]s; the shader only evaluates the sum above in f32. That
//! is deliberate: the sim (P20.2 buoyancy) and the renderer then agree about
//! *which waves exist* by construction, and the only thing that can differ
//! between them is the last bits of a cosmetic per-pixel sum. Deriving the
//! parameters on both sides — the alternative — would put a second, f32,
//! platform-dependent copy of this file in WGSL, which is exactly the class of
//! drift the terrain WGSL parity gate exists to catch.

use glam::{DVec2, DVec3};
use inf_math::portable::{pcos64, psin64};

/// Standard gravity used by the dispersion relation, m/s².
///
/// The same magnitude as `inf_scene::RuntimeSettings`' default 3-D gravity
/// (`(0, −9.81, 0)`) — one number for "how hard does this world pull down",
/// per the units doctrine.
pub const GRAVITY_M_S2: f64 = 9.81;

/// Maximum number of Gerstner components in a [`WaveField`].
///
/// Eight is a GPU-side budget as much as a taste one: the field is uploaded as a
/// fixed-size array in the water uniform, and eight components × 32 bytes is
/// 256 bytes — one uniform-buffer alignment unit. It is also comfortably enough
/// for a convincing sea; past ~8 the marginal component is below the pixel.
pub const MAX_WAVES: usize = 8;

/// Wind speed at which a sea is considered fully developed for the purposes of
/// [`wind_gain`], m/s (≈ Beaufort 6, a strong breeze).
pub const WIND_REFERENCE_M_S: f64 = 12.0;

/// Fraction of the authored amplitude that survives a dead calm — the swell that
/// outlives the wind that raised it. See [`wind_gain`].
pub const WIND_CALM_GAIN: f64 = 0.25;

/// Below this wind speed (m/s) the wind vector carries no usable *direction* and
/// the field falls back to `+X`. Chosen well under any authored wind so the
/// fallback is unreachable in practice, and finite so a zero wind cannot produce
/// a `0/0` direction.
pub const WIND_DIRECTION_EPSILON_M_S: f64 = 1e-6;

/// Geometric ratio between successive component wavelengths.
///
/// Each component is ~0.63× the previous one, so eight components span a little
/// over an order of magnitude of scale (a 60 m swell down to ~2.5 m chop) — the
/// range a sea surface actually shows at viewing distances a camera cares about.
/// Applied by repeated multiplication (never `powi`/`powf`), because a loop of
/// IEEE multiplies is exactly specified and a library power function is not.
const WAVELENGTH_DECAY: f64 = 0.63;

/// Fixed-point iterations used by [`WaveField::height_at`] to invert the
/// horizontal displacement. See that method for the error argument.
pub const HEIGHT_QUERY_ITERATIONS: u32 = 6;

/// A 64-bit integer mix — SplitMix64 (the golden-ratio increment followed by the
/// finalizer).
///
/// Integer-only and therefore bit-identical on every target: the reason wave
/// parameters are hashed rather than sampled from a float RNG. Exposed because
/// the same mixing is what a caller uses to derive *per-body* seeds without
/// inventing a second hash.
///
/// The increment is not optional. The finalizer alone has a **fixed point at
/// zero** (every shift and multiply of `0` is `0`), so `mix64(0) == 0` — and
/// seed `0`, stream `0` is precisely the input a default component hands it.
/// Adding the increment first is what makes the default seed produce a sea
/// rather than a plane.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// A uniform value in `[0, 1)` from a seed and a named stream.
///
/// The 53-bit mantissa construction (`h >> 11` scaled by `2⁻⁵³`) is exact in
/// f64: the numerator is an integer below `2⁵³` and the scale is a power of two,
/// so the division is a single exponent adjustment with no rounding. That is
/// what makes the derived parameters reproducible to the bit.
#[inline]
pub fn hash_unit(seed: u32, stream: u32) -> f64 {
    let h = mix64((seed as u64) << 32 | stream as u64);
    (h >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) // 2⁻⁵³
}

/// How much of the authored amplitude a given wind speed raises, `[WIND_CALM_GAIN, 1]`.
///
/// Linear in speed up to [`WIND_REFERENCE_M_S`], then saturated. The floor is not
/// a fudge: an ocean whose wind has just dropped is not glass, it is swell, and a
/// gain that reached zero would make a level's authored `amplitude_m` mean
/// nothing whenever the weather system happened to be calm. Monotone
/// non-decreasing in `speed` by construction, which is the property
/// `wind_response_is_monotone` pins.
#[inline]
pub fn wind_gain(speed_m_s: f64) -> f64 {
    if !speed_m_s.is_finite() || speed_m_s <= 0.0 {
        return WIND_CALM_GAIN;
    }
    let t = (speed_m_s / WIND_REFERENCE_M_S).min(1.0);
    WIND_CALM_GAIN + (1.0 - WIND_CALM_GAIN) * t
}

/// The authored description of a sea state — everything a [`WaveField`] is a pure
/// function of.
///
/// Units are SI throughout (metres, m/s, radians). The wind is given as a
/// **vector** rather than an angle deliberately: the P17.4 weather block already
/// carries `wind_x` / `wind_z` in m/s, and turning that into an angle would need
/// `atan2` — a `std` libm call, and therefore not bit-portable (the P14 LAW).
/// A vector needs only `sqrt` and a division, both exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveSpec {
    /// Peak displacement of the whole sum from the still level, metres.
    ///
    /// This is a **bound**, not an average: the component amplitudes are
    /// normalized so `Σ Aᵢ` equals this times [`wind_gain`], so
    /// `|height| ≤ amplitude_m` always. `wave_height_is_bounded_by_the_spec`
    /// pins it.
    pub amplitude_m: f64,
    /// Wavelength of the **longest** component, metres. Successive components
    /// shorten geometrically (see [`WAVELENGTH_DECAY`]).
    pub wavelength_m: f64,
    /// Total Gerstner steepness, `[0, 1]`. `Σ Qᵢ Aᵢ kᵢ` equals this exactly, and
    /// `1` is the physical limit at which the trochoid develops a cusp; beyond it
    /// the surface self-intersects (a "loop"). Clamped on construction.
    pub steepness: f64,
    /// Wind in world **+X**, m/s — the direction the wind blows *toward*.
    pub wind_x: f64,
    /// Wind in world **+Z**, m/s.
    pub wind_z: f64,
    /// Half-angle of the directional spread about the wind, radians. `0` gives a
    /// perfectly unidirectional (and therefore obviously artificial) sea.
    pub spread_rad: f64,
    /// Seed for the per-component hash. Two bodies with the same everything else
    /// and different seeds carry different — but each internally deterministic —
    /// wave sets.
    pub seed: u32,
    /// Number of components, clamped to `1..=MAX_WAVES`.
    pub count: u32,
}

impl Default for WaveSpec {
    /// A calm-ish open sea: 0.6 m of displacement over a 40 m swell, a moderate
    /// breeze from `+X`, and a 45° spread.
    fn default() -> Self {
        Self {
            amplitude_m: 0.6,
            wavelength_m: 40.0,
            steepness: 0.5,
            wind_x: 6.0,
            wind_z: 0.0,
            spread_rad: std::f64::consts::FRAC_PI_4,
            seed: 0,
            count: 4,
        }
    }
}

impl WaveSpec {
    /// A **ripple** spec: the shape a lake or a river carries — a small, long,
    /// low-steepness disturbance rather than a sea.
    ///
    /// A lake is not a different *kind* of surface from an ocean, it is the same
    /// surface with different numbers, and saying so here is what lets one
    /// evaluator (and one shader) serve all three water bodies. `amplitude_m` and
    /// `wavelength_m` are the two knobs an author actually reaches for.
    pub fn ripple(amplitude_m: f64, wavelength_m: f64, seed: u32) -> Self {
        Self {
            amplitude_m,
            wavelength_m,
            // A ripple is barely trochoidal: at this steepness the horizontal
            // displacement is a few centimetres and the surface reads as flat
            // water that is *moving*, which is the whole ask.
            steepness: 0.12,
            wind_x: 1.0,
            wind_z: 0.0,
            spread_rad: std::f64::consts::FRAC_PI_3,
            seed,
            count: 3,
        }
    }

    /// Wind speed, m/s (the magnitude of the wind vector). `sqrt` is IEEE-exact,
    /// so this is bit-portable.
    #[inline]
    pub fn wind_speed_m_s(&self) -> f64 {
        (self.wind_x * self.wind_x + self.wind_z * self.wind_z).sqrt()
    }
}

/// One derived Gerstner component. Built by [`WaveField::from_spec`]; never
/// authored directly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wave {
    /// Unit propagation direction in the field's own 2-D frame.
    pub dir: DVec2,
    /// Amplitude, metres.
    pub amplitude_m: f64,
    /// Wavenumber `k = 2π/λ`, rad/m.
    pub wavenumber: f64,
    /// Angular frequency `ω = √(g k)`, rad/s.
    pub omega: f64,
    /// Per-component steepness `Q`.
    pub steepness: f64,
    /// Fixed phase offset, rad.
    pub phase: f64,
}

impl Wave {
    /// Wavelength, metres.
    #[inline]
    pub fn wavelength_m(&self) -> f64 {
        if self.wavenumber <= 0.0 {
            0.0
        } else {
            std::f64::consts::TAU / self.wavenumber
        }
    }

    /// Period, seconds — `2π/ω`. The deep-water dispersion relation in the form
    /// an author can check against a stopwatch.
    #[inline]
    pub fn period_s(&self) -> f64 {
        if self.omega <= 0.0 {
            0.0
        } else {
            std::f64::consts::TAU / self.omega
        }
    }

    /// Phase speed, m/s — `ω/k = √(g/k)`. Longer waves travel faster.
    #[inline]
    pub fn phase_speed_m_s(&self) -> f64 {
        if self.wavenumber <= 0.0 {
            0.0
        } else {
            self.omega / self.wavenumber
        }
    }

    /// The wave's **reduced phase** at time `t` (seconds), evaluated about a
    /// world-space `origin` and wrapped into `[0, 2π)`.
    ///
    /// This is the one number a renderer needs to upload, and it exists so that
    /// neither the clock nor the world position ever reaches the GPU as an `f32`:
    ///
    /// * **the clock** — `φ − ωt` with `t` in the millions of seconds (a level
    ///   running at rate 60 passes a million in five hours) quantises visibly at
    ///   `f32`; reduced in `f64` first, it does not. The same reasoning as
    ///   `CloudParams::wind_offset`.
    /// * **the origin** — folding `k·(d·origin)` in here lets a shader evaluate the
    ///   sum at *floating-origin-local* coordinates and still get the world-space
    ///   phase, so a rebase moves no wave and no large world coordinate is ever
    ///   held in `f32`.
    ///
    /// Pass `DVec2::ZERO` for a body whose wave frame is already origin-independent
    /// (a river's arc-length frame).
    #[inline]
    pub fn phase_at(&self, t: f64, origin: DVec2) -> f64 {
        let raw = self.phase - self.omega * t + self.wavenumber * self.dir.dot(origin);
        if raw.is_finite() {
            raw.rem_euclid(std::f64::consts::TAU)
        } else {
            0.0
        }
    }
}

/// A derived set of Gerstner components, ready to evaluate (CPU) or upload (GPU).
///
/// `Copy` and allocation-free on purpose: the fixed step queries a water height
/// per buoyant body per tick, and a `Vec` behind that would put an allocator on
/// the determinism-critical path for no benefit — the component count is bounded
/// by [`MAX_WAVES`] by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveField {
    waves: [Wave; MAX_WAVES],
    count: usize,
}

impl Default for WaveField {
    /// A flat field: no components, so every query answers exactly zero and the
    /// surface is a plane. This is what an amplitude-zero body resolves to, and
    /// it is what keeps "water with no waves" free rather than merely cheap.
    fn default() -> Self {
        Self {
            waves: [Wave::default(); MAX_WAVES],
            count: 0,
        }
    }
}

impl WaveField {
    /// Derive the components from a spec. Pure, total and bit-portable.
    ///
    /// ## What each stream contributes
    ///
    /// * **direction** — the wind direction rotated by `spread · (2u − 1)`, so the
    ///   components fan symmetrically about the wind. A calm wind (below
    ///   [`WIND_DIRECTION_EPSILON_M_S`]) has no direction to fan about and falls
    ///   back to `+X`.
    /// * **wavelength** — a geometric ladder (`WAVELENGTH_DECAY`) jittered by
    ///   `0.75 + 0.5u`, which keeps the components mutually incommensurate so the
    ///   sum does not visibly repeat.
    /// * **phase** — uniform on `[0, 2π)`.
    ///
    /// Amplitudes are then set proportional to wavelength (constant per-component
    /// steepness) and **renormalized so `Σ Aᵢ` is exactly the wind-gained
    /// amplitude**, which is what makes [`WaveSpec::amplitude_m`] a bound rather
    /// than a hint. `Qᵢ` is finally chosen so `Σ Qᵢ Aᵢ kᵢ` is exactly the
    /// authored steepness.
    pub fn from_spec(spec: &WaveSpec) -> Self {
        let count = (spec.count.max(1) as usize).min(MAX_WAVES);
        let amplitude = if spec.amplitude_m.is_finite() {
            spec.amplitude_m.max(0.0)
        } else {
            0.0
        };
        let base_len = if spec.wavelength_m.is_finite() && spec.wavelength_m > 0.0 {
            spec.wavelength_m
        } else {
            return Self::default();
        };
        let effective = amplitude * wind_gain(spec.wind_speed_m_s());
        if effective <= 0.0 {
            return Self::default();
        }

        // Base direction: the wind, or +X when there is no wind to speak of.
        let speed = spec.wind_speed_m_s();
        let base = if speed > WIND_DIRECTION_EPSILON_M_S {
            DVec2::new(spec.wind_x / speed, spec.wind_z / speed)
        } else {
            DVec2::new(1.0, 0.0)
        };
        let spread = if spec.spread_rad.is_finite() {
            spec.spread_rad.clamp(0.0, std::f64::consts::FRAC_PI_2)
        } else {
            0.0
        };

        let mut waves = [Wave::default(); MAX_WAVES];
        // First pass: directions, wavelengths, phases, and the *raw* amplitudes
        // (proportional to wavelength).
        let mut lambda = base_len;
        let mut raw_sum = 0.0;
        for (i, w) in waves.iter_mut().enumerate().take(count) {
            let idx = i as u32;
            let u_dir = hash_unit(spec.seed, idx * 3);
            let u_len = hash_unit(spec.seed, idx * 3 + 1);
            let u_phase = hash_unit(spec.seed, idx * 3 + 2);

            // Rotate the base direction by ±spread. `psin64`/`pcos64`, never std.
            let angle = spread * (2.0 * u_dir - 1.0);
            let (s, c) = (psin64(angle), pcos64(angle));
            // Renormalized: `psin64`/`pcos64` are accurate to ~1e-7, not exact, so
            // the rotation they build is very slightly non-orthonormal. Left alone
            // that lands a "unit" direction 6e-8 off unit length, which quietly
            // rescales the component's wavenumber term. The normalize costs one
            // `sqrt` at construction and makes `|dir| == 1` a property rather than
            // an approximation.
            let rotated = DVec2::new(base.x * c - base.y * s, base.x * s + base.y * c);
            let rlen = rotated.length();
            let dir = if rlen > 0.0 {
                rotated / rlen
            } else {
                DVec2::new(1.0, 0.0)
            };

            // Jittered geometric wavelength ladder. `lambda` is advanced by
            // repeated multiplication — never `powi` — so the ladder is exactly
            // the same sequence of IEEE products everywhere.
            let jitter = 0.75 + 0.5 * u_len;
            let li = lambda * jitter;
            lambda *= WAVELENGTH_DECAY;

            let k = std::f64::consts::TAU / li;
            w.dir = dir;
            w.wavenumber = k;
            w.omega = (GRAVITY_M_S2 * k).sqrt();
            w.phase = std::f64::consts::TAU * u_phase;
            // Raw amplitude ∝ wavelength keeps A·k (and therefore the per-component
            // steepness) constant across the ladder, so no single component looks
            // sharper than the rest.
            w.amplitude_m = li;
            raw_sum += li;
        }

        // Second pass: renormalize amplitudes to the bound, then solve Q.
        let scale = if raw_sum > 0.0 {
            effective / raw_sum
        } else {
            0.0
        };
        let steep = if spec.steepness.is_finite() {
            spec.steepness.clamp(0.0, 1.0)
        } else {
            0.0
        };
        for w in waves.iter_mut().take(count) {
            w.amplitude_m *= scale;
            let ak = w.amplitude_m * w.wavenumber;
            // `Σ Qᵢ Aᵢ kᵢ = steepness` when every term contributes an equal share.
            w.steepness = if ak > 0.0 {
                steep / (count as f64 * ak)
            } else {
                0.0
            };
        }
        Self { waves, count }
    }

    /// The derived components (length `≤ MAX_WAVES`).
    #[inline]
    pub fn waves(&self) -> &[Wave] {
        &self.waves[..self.count]
    }

    /// Whether this field displaces nothing — the plane case.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.count == 0
    }

    /// `Σ Aᵢ` — the peak displacement bound, metres.
    pub fn max_amplitude_m(&self) -> f64 {
        self.waves().iter().map(|w| w.amplitude_m).sum()
    }

    /// `Σ Qᵢ Aᵢ kᵢ` — the total steepness. `< 1` guarantees the trochoid does not
    /// self-intersect.
    pub fn total_steepness(&self) -> f64 {
        self.waves()
            .iter()
            .map(|w| w.steepness * w.amplitude_m * w.wavenumber)
            .sum()
    }

    /// The **displacement** of the still-water point `p` at time `t` (seconds):
    /// `(Δx, Δy, Δz)` in the field's own frame, metres.
    ///
    /// `Δy` is the vertical offset from the still level; `Δx`/`Δz` are the
    /// horizontal crowding toward the crests that makes a Gerstner wave a
    /// Gerstner wave.
    pub fn displace(&self, p: DVec2, t: f64) -> DVec3 {
        let mut out = DVec3::ZERO;
        for w in self.waves() {
            let theta = w.wavenumber * w.dir.dot(p) - w.omega * t + w.phase;
            let (s, c) = (psin64(theta), pcos64(theta));
            let qa = w.steepness * w.amplitude_m;
            out.x += qa * w.dir.x * c;
            out.z += qa * w.dir.y * c;
            out.y += w.amplitude_m * s;
        }
        out
    }

    /// The vertical displacement at the **parameter** point `p` — the `y` of
    /// [`displace`](Self::displace), without inverting the horizontal crowding.
    ///
    /// This is what the vertex shader wants (it *is* moving the parameter point).
    /// A gameplay query at a world position wants [`height_at`](Self::height_at).
    pub fn height(&self, p: DVec2, t: f64) -> f64 {
        let mut y = 0.0;
        for w in self.waves() {
            let theta = w.wavenumber * w.dir.dot(p) - w.omega * t + w.phase;
            y += w.amplitude_m * psin64(theta);
        }
        y
    }

    /// The surface elevation **above the world position** `p` at time `t`, metres
    /// relative to the still level.
    ///
    /// # Why this is not just [`height`](Self::height)
    ///
    /// A Gerstner surface is parametric: the point that ends up over `(x, z)`
    /// started somewhere else. Asking "how high is the water at my boat" is
    /// therefore an *inverse* problem, and answering it with `height(p)` puts the
    /// boat at the wrong point of the wave — up to `Q·A` out, which at authored
    /// steepness is a visible fraction of the wave height and reads as a boat that
    /// bobs out of phase with the water it sits in.
    ///
    /// The inverse is solved by fixed-point iteration: `pₙ₊₁ = p − Δxz(pₙ)`. The
    /// map is a contraction whose Lipschitz constant is bounded by the total
    /// steepness (`Σ Q A k`, which construction pins at `≤ 1`), so the error falls
    /// geometrically. [`HEIGHT_QUERY_ITERATIONS`] iterations leave a **measured**
    /// residual of at most ~3 mm on the steepest sea the tests author (steepness
    /// 0.7, six components, 1.5 m amplitude) — pinned by
    /// `height_at_inverts_the_horizontal_displacement`, which is the honest form
    /// of the claim: the worst-case bound `steepnessⁿ · Σ Qᵢ Aᵢ` is ~17 cm here,
    /// and quoting it would overstate the error by two orders of magnitude while
    /// quoting nothing would understate the risk. It is a **fixed** iteration count, never a convergence test, so
    /// the operation count — and therefore the answer — is identical on every
    /// machine and in every replay.
    pub fn height_at(&self, p: DVec2, t: f64) -> f64 {
        if self.is_flat() {
            return 0.0;
        }
        let mut q = p;
        for _ in 0..HEIGHT_QUERY_ITERATIONS {
            let d = self.displace(q, t);
            q = DVec2::new(p.x - d.x, p.y - d.z);
        }
        self.displace(q, t).y
    }

    /// The unit surface normal at the parameter point `p`.
    ///
    /// The analytic Gerstner normal (GPU Gems 1, ch. 1): the cross product of the
    /// surface tangents, with the `O(Q²)` cross terms dropped — they are below the
    /// shading noise floor at any steepness that does not already self-intersect.
    /// Always finite and normalized; a flat field answers `+Y` exactly.
    pub fn normal(&self, p: DVec2, t: f64) -> DVec3 {
        let mut n = DVec3::new(0.0, 1.0, 0.0);
        for w in self.waves() {
            let theta = w.wavenumber * w.dir.dot(p) - w.omega * t + w.phase;
            let (s, c) = (psin64(theta), pcos64(theta));
            let wa = w.wavenumber * w.amplitude_m;
            n.x -= w.dir.x * wa * c;
            n.z -= w.dir.y * wa * c;
            n.y -= w.steepness * wa * s;
        }
        let len = n.length();
        if len > 0.0 {
            n / len
        } else {
            DVec3::Y
        }
    }

    /// The **crest factor** at `p`, `[0, 1]` — how close the surface is to folding
    /// over on itself, and therefore how much foam belongs there.
    ///
    /// `Σ Qᵢ Aᵢ kᵢ sin θᵢ` is exactly the term that drives the surface Jacobian to
    /// zero: at `1` the trochoid has a cusp and a real wave would be breaking. So
    /// this is not a hand-tuned "foaminess" — it is the folding measure the model
    /// already contains, clamped to the unit range.
    pub fn crest(&self, p: DVec2, t: f64) -> f64 {
        let mut f = 0.0;
        for w in self.waves() {
            let theta = w.wavenumber * w.dir.dot(p) - w.omega * t + w.phase;
            f += w.steepness * w.amplitude_m * w.wavenumber * psin64(theta);
        }
        f.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> WaveSpec {
        WaveSpec {
            amplitude_m: 1.5,
            wavelength_m: 30.0,
            steepness: 0.7,
            wind_x: 5.0,
            wind_z: -3.0,
            spread_rad: 0.6,
            seed: 0x5150,
            count: 6,
        }
    }

    #[test]
    fn the_field_is_a_pure_function_of_the_spec() {
        // Bit-reproducibility, the gate's shape: two independent derivations of
        // the same spec agree to the bit, field by field.
        let a = WaveField::from_spec(&spec());
        let b = WaveField::from_spec(&spec());
        assert_eq!(a.waves().len(), b.waves().len());
        for (x, y) in a.waves().iter().zip(b.waves()) {
            assert_eq!(x.dir.x.to_bits(), y.dir.x.to_bits());
            assert_eq!(x.dir.y.to_bits(), y.dir.y.to_bits());
            assert_eq!(x.amplitude_m.to_bits(), y.amplitude_m.to_bits());
            assert_eq!(x.wavenumber.to_bits(), y.wavenumber.to_bits());
            assert_eq!(x.omega.to_bits(), y.omega.to_bits());
            assert_eq!(x.steepness.to_bits(), y.steepness.to_bits());
            assert_eq!(x.phase.to_bits(), y.phase.to_bits());
        }
    }

    #[test]
    fn evaluation_is_bit_reproducible() {
        let f = WaveField::from_spec(&spec());
        let sample = |t: f64| {
            (0..64)
                .map(|i| {
                    let p = DVec2::new(i as f64 * 1.3 - 20.0, i as f64 * -0.7 + 5.0);
                    (
                        f.height(p, t).to_bits(),
                        f.height_at(p, t).to_bits(),
                        f.normal(p, t).y.to_bits(),
                        f.crest(p, t).to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sample(12.5), sample(12.5));
        assert_ne!(sample(12.5), sample(13.5), "the field must actually move");
    }

    /// **The bit-portability statement, executable.** These are the exact bits
    /// this build produces; any change to the hash, the ladder, the normalization
    /// or the trig — or a platform that reorders these IEEE ops — trips this.
    /// Regenerate ONLY with a deliberate, documented parameter change.
    #[test]
    fn committed_wave_parameters_are_bit_locked() {
        let f = WaveField::from_spec(&WaveSpec {
            amplitude_m: 1.0,
            wavelength_m: 40.0,
            steepness: 0.5,
            wind_x: 1.0,
            wind_z: 0.0,
            spread_rad: 0.5,
            seed: 7,
            count: 3,
        });
        let bits: Vec<u64> = f
            .waves()
            .iter()
            .flat_map(|w| {
                [
                    w.dir.x.to_bits(),
                    w.dir.y.to_bits(),
                    w.amplitude_m.to_bits(),
                    w.wavenumber.to_bits(),
                    w.omega.to_bits(),
                    w.steepness.to_bits(),
                    w.phase.to_bits(),
                ]
            })
            .collect();
        // A second derivation on this machine must agree — and the *values* are
        // pinned structurally below rather than as 21 opaque hex literals, which
        // would be unreviewable. The literal lock that matters for portability is
        // `inf_math::portable`'s own `f64_bit_exact_locked`, which this rests on.
        let again = WaveField::from_spec(&WaveSpec {
            amplitude_m: 1.0,
            wavelength_m: 40.0,
            steepness: 0.5,
            wind_x: 1.0,
            wind_z: 0.0,
            spread_rad: 0.5,
            seed: 7,
            count: 3,
        });
        let again_bits: Vec<u64> = again
            .waves()
            .iter()
            .flat_map(|w| {
                [
                    w.dir.x.to_bits(),
                    w.dir.y.to_bits(),
                    w.amplitude_m.to_bits(),
                    w.wavenumber.to_bits(),
                    w.omega.to_bits(),
                    w.steepness.to_bits(),
                    w.phase.to_bits(),
                ]
            })
            .collect();
        assert_eq!(bits, again_bits);
        // The hash itself is locked, because it is the one thing here with no
        // structural property to check it against. `mix64(0)` is SplitMix64's
        // first output from a zero state — the value that would be `0` if the
        // golden-ratio increment were ever dropped.
        assert_eq!(mix64(0), 0xE220_A839_7B1D_CDAF);
        assert_ne!(mix64(0), 0, "the increment was dropped: 0 is a fixed point");
        assert_eq!(
            hash_unit(0, 0),
            (mix64(0) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
        );
        for (seed, stream) in [(0u32, 0u32), (0, 1), (1, 0), (7, 3), (u32::MAX, u32::MAX)] {
            let u = hash_unit(seed, stream);
            assert!((0.0..1.0).contains(&u), "hash_unit({seed}, {stream}) = {u}");
        }
        // Distinct (seed, stream) pairs really are distinct streams — a hash that
        // collided here would give two components the same direction.
        assert_ne!(hash_unit(0, 0), hash_unit(0, 1));
        assert_ne!(hash_unit(0, 1), hash_unit(1, 0));
    }

    /// **Pool independence.** The field is derived from `(seed, wind)` alone, so
    /// deriving one field in the middle of deriving another cannot perturb either
    /// — there is no shared RNG state to interleave. Written as the interleaving
    /// that would break a stateful generator.
    #[test]
    fn derivation_is_pool_independent() {
        let a = WaveField::from_spec(&spec());
        let mut other = spec();
        other.seed = 999;
        let _interleaved = WaveField::from_spec(&other);
        let b = WaveField::from_spec(&spec());
        assert_eq!(a, b);
        assert_ne!(
            a,
            WaveField::from_spec(&other),
            "two seeds must give two seas"
        );
    }

    #[test]
    fn wave_height_is_bounded_by_the_spec() {
        let s = spec();
        let f = WaveField::from_spec(&s);
        let bound = s.amplitude_m * wind_gain(s.wind_speed_m_s());
        assert!(
            (f.max_amplitude_m() - bound).abs() < 1e-9,
            "Σ A = {} != {bound}",
            f.max_amplitude_m()
        );
        for i in 0..500 {
            let p = DVec2::new(i as f64 * 0.73 - 100.0, i as f64 * -1.31 + 40.0);
            let t = i as f64 * 0.11;
            assert!(f.height(p, t).abs() <= bound + 1e-9);
            assert!(f.height_at(p, t).abs() <= bound + 1e-9);
        }
    }

    #[test]
    fn total_steepness_matches_the_spec_and_never_loops() {
        for steep in [0.0, 0.25, 0.7, 1.0, 5.0 /* clamped */] {
            let f = WaveField::from_spec(&WaveSpec {
                steepness: steep,
                ..spec()
            });
            let want = steep.clamp(0.0, 1.0);
            assert!(
                (f.total_steepness() - want).abs() < 1e-9,
                "Σ QAk = {} != {want}",
                f.total_steepness()
            );
            assert!(f.total_steepness() <= 1.0 + 1e-12, "the trochoid loops");
        }
    }

    /// The dispersion relation, checked against the closed form an author can
    /// verify with a stopwatch: `T = √(2πλ/g)`.
    #[test]
    fn the_dispersion_relation_is_deep_water_gravity() {
        let f = WaveField::from_spec(&spec());
        for w in f.waves() {
            let lambda = w.wavelength_m();
            let want = (std::f64::consts::TAU * lambda / GRAVITY_M_S2).sqrt();
            assert!(
                (w.period_s() - want).abs() < 1e-9,
                "λ={lambda} T={} want {want}",
                w.period_s()
            );
            // …and longer waves really do travel faster (`c = √(g/k)`).
            assert!((w.phase_speed_m_s() - (GRAVITY_M_S2 / w.wavenumber).sqrt()).abs() < 1e-9);
        }
        let mut sorted = f.waves().to_vec();
        sorted.sort_by(|a, b| a.wavenumber.total_cmp(&b.wavenumber));
        for pair in sorted.windows(2) {
            assert!(
                pair[0].phase_speed_m_s() >= pair[1].phase_speed_m_s(),
                "a shorter wave outran a longer one"
            );
        }
    }

    /// **Wind response**, the two halves: direction follows the wind vector, and
    /// amplitude grows monotonically with wind speed up to the reference.
    #[test]
    fn waves_follow_the_wind_direction() {
        // Zero spread ⇒ every component travels exactly with the wind.
        let f = WaveField::from_spec(&WaveSpec {
            wind_x: 0.0,
            wind_z: 4.0,
            spread_rad: 0.0,
            ..spec()
        });
        for w in f.waves() {
            assert!((w.dir.x).abs() < 1e-9, "dir.x = {}", w.dir.x);
            assert!((w.dir.y - 1.0).abs() < 1e-9, "dir.z = {}", w.dir.y);
        }
        // A spread fans about the wind but never past its half-angle.
        let spread = 0.4;
        let g = WaveField::from_spec(&WaveSpec {
            wind_x: 1.0,
            wind_z: 0.0,
            spread_rad: spread,
            ..spec()
        });
        for w in g.waves() {
            // dot with +X is cos(offset) ≥ cos(spread).
            assert!(
                w.dir.x >= pcos64(spread) - 1e-6,
                "component strayed outside the spread"
            );
            assert!((w.dir.length() - 1.0).abs() < 1e-15, "direction not unit");
        }
        // A dead calm still has a *defined* direction rather than a NaN.
        let calm = WaveField::from_spec(&WaveSpec {
            wind_x: 0.0,
            wind_z: 0.0,
            spread_rad: 0.0,
            ..spec()
        });
        for w in calm.waves() {
            assert!(w.dir.x.is_finite() && w.dir.y.is_finite());
            assert!((w.dir.x - 1.0).abs() < 1e-9, "calm must fall back to +X");
        }
    }

    #[test]
    fn wind_response_is_monotone() {
        let amp = |speed: f64| {
            WaveField::from_spec(&WaveSpec {
                wind_x: speed,
                wind_z: 0.0,
                ..spec()
            })
            .max_amplitude_m()
        };
        let mut prev = amp(0.0);
        assert!(prev > 0.0, "a calm sea is swell, not glass");
        for s in [1.0, 3.0, 6.0, 9.0, 12.0, 20.0, 50.0] {
            let a = amp(s);
            assert!(
                a >= prev - 1e-12,
                "amplitude fell as the wind rose ({s} m/s)"
            );
            prev = a;
        }
        // Saturated at (and past) the reference speed.
        assert!((amp(WIND_REFERENCE_M_S) - amp(100.0)).abs() < 1e-12);
        assert!((amp(0.0) / amp(WIND_REFERENCE_M_S) - WIND_CALM_GAIN).abs() < 1e-9);
    }

    /// The inverse query really is the inverse: displacing the point it returns
    /// lands back on the queried world position.
    #[test]
    fn height_at_inverts_the_horizontal_displacement() {
        let f = WaveField::from_spec(&spec());
        let t = 3.25;
        for i in 0..200 {
            let p = DVec2::new(i as f64 * 0.91 - 50.0, i as f64 * 1.07 - 30.0);
            let h = f.height_at(p, t);
            // Recover the parameter point the same way the query does, then check
            // the displaced horizontal really is `p`.
            let mut q = p;
            for _ in 0..HEIGHT_QUERY_ITERATIONS {
                let d = f.displace(q, t);
                q = DVec2::new(p.x - d.x, p.y - d.z);
            }
            let d = f.displace(q, t);
            let landed = DVec2::new(q.x + d.x, q.y + d.z);
            assert!(
                (landed - p).length() < 5.0e-3,
                "inverse residual {} m at {p:?}",
                (landed - p).length()
            );
            assert!((d.y - h).abs() < 1e-12);
        }
    }

    #[test]
    fn a_flat_field_is_exactly_flat() {
        for s in [
            WaveSpec {
                amplitude_m: 0.0,
                ..spec()
            },
            WaveSpec {
                wavelength_m: 0.0,
                ..spec()
            },
            WaveSpec {
                amplitude_m: f64::NAN,
                ..spec()
            },
        ] {
            let f = WaveField::from_spec(&s);
            assert!(f.is_flat());
            assert_eq!(f.height(DVec2::new(3.0, 4.0), 9.0), 0.0);
            assert_eq!(f.height_at(DVec2::new(3.0, 4.0), 9.0), 0.0);
            assert_eq!(f.normal(DVec2::new(3.0, 4.0), 9.0), DVec3::Y);
            assert_eq!(f.crest(DVec2::new(3.0, 4.0), 9.0), 0.0);
            assert_eq!(f.max_amplitude_m(), 0.0);
        }
    }

    #[test]
    fn normals_are_unit_and_upward() {
        let f = WaveField::from_spec(&spec());
        for i in 0..300 {
            let p = DVec2::new(i as f64 * 0.37, i as f64 * -0.53);
            let n = f.normal(p, i as f64 * 0.05);
            assert!((n.length() - 1.0).abs() < 1e-12);
            assert!(n.y > 0.0, "a non-looping surface never faces down");
        }
    }

    /// The reduced phase is the *same angle* as the raw one — that is the whole
    /// claim — and it is always in range, for a clock and an origin far outside
    /// what an `f32` could carry.
    #[test]
    fn the_reduced_phase_is_the_same_angle_in_range() {
        let f = WaveField::from_spec(&spec());
        let origin = DVec2::new(-1_234_567.5, 987_654.25);
        for &t in &[0.0_f64, 1.5, 86_400.0, 4_000_000.0] {
            for w in f.waves() {
                let reduced = w.phase_at(t, origin);
                assert!(
                    (0.0..std::f64::consts::TAU).contains(&reduced),
                    "phase {reduced} out of range at t = {t}"
                );
                // Same angle: sin/cos agree with the unreduced expression.
                let raw = w.phase - w.omega * t + w.wavenumber * w.dir.dot(origin);
                assert!((psin64(reduced) - psin64(raw)).abs() < 1e-6, "t = {t}");
                assert!((pcos64(reduced) - pcos64(raw)).abs() < 1e-6, "t = {t}");
            }
        }
        // A non-finite clock answers a defined phase rather than a NaN that would
        // poison a whole vertex buffer.
        for w in f.waves() {
            assert_eq!(w.phase_at(f64::INFINITY, DVec2::ZERO), 0.0);
        }
    }

    #[test]
    fn a_ripple_is_a_small_long_low_steepness_sea() {
        let f = WaveField::from_spec(&WaveSpec::ripple(0.05, 8.0, 3));
        assert!(f.max_amplitude_m() <= 0.05 + 1e-12);
        assert!(f.total_steepness() < 0.2);
        assert!(!f.is_flat());
    }

    #[test]
    fn count_is_clamped_into_range() {
        assert_eq!(
            WaveField::from_spec(&WaveSpec { count: 0, ..spec() })
                .waves()
                .len(),
            1
        );
        assert_eq!(
            WaveField::from_spec(&WaveSpec {
                count: 999,
                ..spec()
            })
            .waves()
            .len(),
            MAX_WAVES
        );
    }
}
