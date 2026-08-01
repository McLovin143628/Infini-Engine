//! Precipitation v1 (P17.4) — a world-anchored rain/snow particle layer.
//!
//! Deliberately **not** a general VFX system. This is one small dedicated pass
//! with one job: put rain streaks or snowflakes in the air around the camera,
//! deterministically, driven by the weather block, and get out of the way when
//! the weather is dry. A real particle system (emitters, curves, collision,
//! GPU simulation) is future scope, and calling this one would be a lie.
//!
//! # What makes it deterministic
//!
//! Everything about a particle is a pure function of two integers and the
//! level's clock:
//!
//! * its **base position** comes from [`crate::clouds::cloud_hash`] — the same
//!   pure-integer avalanche the cloud field uses (the `inf-pcg` counter-hash
//!   doctrine: a draw is a function of a coordinate tuple, never of a stateful
//!   RNG), with a WGSL mirror that is bit-identical by construction because it
//!   contains no float and no trigonometry;
//! * its **motion** is `wind · t` horizontally and `fall_speed · t` vertically,
//!   where `t` is `ResolvedSky::cloud_time_s` — the *document's* clock, never a
//!   wall clock and never a frame index. Two runs at the same time of day
//!   produce the same rain, in the editor and in a shipped build alike.
//!
//! The three displacements are wrapped into the box **on the CPU in `f64`**
//! (see [`PrecipParams::offsets`]), exactly as [`crate::clouds::wind_offset`]
//! does and for the same reason: a day of 9 m/s fall is 780 km, at which point
//! an `f32` metre has ~0.06 m of resolution and the flakes would visibly
//! quantize into stair-steps.
//!
//! # World-anchored, without a world coordinate
//!
//! Particles must move through the world rather than with the camera, while
//! always filling the view. The shader gets `eye_mod` — the camera's world
//! position reduced modulo the box, computed in `f64` — and places a particle at
//!
//! ```text
//! p_world = eye_world + wrap_signed(base + drift − eye_mod, box)
//! ```
//!
//! Since `eye_mod ≡ eye_world (mod box)`, `p_world ≡ base + drift (mod box)`:
//! the lattice is locked to world space, so the rain does not slide when the
//! camera moves, and it cannot pop when the floating origin rebases (the origin
//! never enters the expression).
//!
//! # Off ⇒ instruction-neutral
//!
//! [`PrecipParams::active`] false ⇒ the pass returns before touching the
//! encoder: no render pass, no pipeline bind, no draw. Every pre-P17.4 golden
//! renders the exact command stream it did before.

/// Terminal fall speed of rain, **m/s**. A 2 mm raindrop falls at about this.
pub const RAIN_FALL_SPEED: f32 = 9.0;
/// Terminal fall speed of a snowflake, **m/s** — an order of magnitude slower,
/// which is most of why snow reads as snow.
pub const SNOW_FALL_SPEED: f32 = 1.0;

/// Horizontal wrap period of the particle volume, **metres**. The rain fills a
/// `PRECIP_BOX_XZ_M` square around the camera and repeats outside it.
pub const PRECIP_BOX_XZ_M: f32 = 40.0;
/// Vertical wrap period, **metres**.
pub const PRECIP_BOX_Y_M: f32 = 28.0;

// ── the supra-physical constants, disclosed together ────────────────────────
//
// ALL THREE of the size constants below are larger than the hydrometeor they
// stand for, and they are grouped here so the disclosure is one list rather than
// a claim attached to whichever one was written first. Against reality:
//
//   * a raindrop is ~2 mm across      -> RAIN_RADIUS_M (1 mm real) is 20x;
//   * a snowflake is ~5-10 mm across  -> SNOW_RADIUS_M (~3 mm real) is ~15x;
//   * a 9 m/s drop smears ~0.15 m in one 60 Hz frame
//                                     -> RAIN_STREAK_M is ~7x that smear.
//
// One reason covers all three: a drop is genuinely sub-pixel at any resolution,
// the visible phenomenon is a *sheet* of them, and 48 000 particles standing in
// for millions can only read as a sheet by drawing each sample bigger than what
// it represents. `PRECIP_MAX_ALPHA_COMPENSATION` in `precip.wgsl` is the same
// argument applied to alpha, and `PRECIP_MIN_PIXELS` beside it is the
// rasterization floor that makes any of it visible at all. These are the batch's
// only departures from the SI-and-physical rule; everything else here (fall
// speeds, box extents, wind, accumulation rate) is real.

/// Length of a rain streak, **metres**. A photographic exposure smears a 9 m/s
/// drop; drawing a point instead is the single thing that makes rain look like
/// static rather than weather. Supra-physical — see the disclosure above.
pub const RAIN_STREAK_M: f32 = 1.1;
/// Radius of a rain streak, **metres** (its half-width). Supra-physical.
pub const RAIN_RADIUS_M: f32 = 0.02;
/// Radius of a snowflake, **metres**. Snow is drawn round rather than streaked,
/// which is what its two orders of magnitude lower speed actually looks like.
/// Supra-physical.
pub const SNOW_RADIUS_M: f32 = 0.05;

/// Per-particle alpha at the centre of a drop, before the shape falloff.
///
/// Low on purpose: precipitation reads as weather when it is a *density* of
/// faint marks, not a scattering of opaque white dashes. The count is the knob
/// intensity moves; this is a constant of the look.
pub const PRECIP_ALPHA: f32 = 0.45;

/// Salt folded into [`crate::clouds::cloud_hash`] so precipitation positions are
/// decorrelated from the cloud field even at the same seed.
pub const PRECIP_HASH_SALT: u32 = 0x9e37_79b9;

/// How many particles a render tier will draw at **full** intensity.
///
/// Derived from [`crate::atmosphere::AtmosphereQuality`] rather than being its
/// own knob, exactly as [`crate::clouds::CloudQuality`] is — one sky quality
/// setting, not three, because a machine that can afford a 128³ cloud volume can
/// afford 48 k billboards and letting them disagree would only produce
/// combinations nobody tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecipQuality {
    Low,
    Medium,
    High,
}

impl PrecipQuality {
    /// The precipitation tier implied by an atmosphere tier.
    pub fn from_atmosphere(q: crate::atmosphere::AtmosphereQuality) -> Self {
        match q {
            crate::atmosphere::AtmosphereQuality::Low => PrecipQuality::Low,
            crate::atmosphere::AtmosphereQuality::Medium => PrecipQuality::Medium,
            crate::atmosphere::AtmosphereQuality::High => PrecipQuality::High,
        }
    }

    /// Particle budget at full intensity. `48 000` in a 40 × 40 × 28 m box is
    /// ~1.1 particles/m³ — orders below real rain (which is invisible per drop
    /// anyway) and comfortably above the density at which a downpour stops
    /// reading as one.
    pub fn max_particles(self) -> u32 {
        match self {
            PrecipQuality::High => 48_000,
            PrecipQuality::Medium => 24_000,
            PrecipQuality::Low => 9_000,
        }
    }
}

/// The precipitation layer's parameters, projected from the weather block.
///
/// Pure and GPU-free, like [`crate::clouds::CloudParams`] and
/// [`crate::atmosphere::AtmosphereParams`], so every derivation below is
/// unit-tested on **every** CI leg — including the ones with no adapter, where
/// the golden harness skips.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecipParams {
    /// Master switch. Off ⇒ the pass dispatches nothing at all.
    pub enabled: bool,
    /// Intensity `[0, 1]`. Scales the particle count linearly — the honest knob,
    /// since a heavier shower is more drops rather than bigger ones.
    pub intensity: f32,
    /// Phase `[0, 1]`: `0` = rain (fast, streaked, small), `1` = snow (slow,
    /// round, larger). Blendable, so a rain→snow transition is continuous.
    pub snowiness: f32,
    /// Wind in world **X**, m/s — slants the fall.
    pub wind_x: f32,
    /// Wind in world **Z**, m/s.
    pub wind_z: f32,
    /// The level's clock in seconds (`ResolvedSky::cloud_time_s`), **not** a wall
    /// clock. Projected, never authored — the same field, and the same reason, as
    /// [`crate::clouds::CloudParams::time_s`].
    pub time_s: f64,
    /// Linear tint of the droplet. White is physical.
    pub color: [f32; 3],
}

impl Default for PrecipParams {
    /// **Disabled**, dry, with a light breeze — enabling precipitation is a
    /// consequence of the weather block, never a parameter hunt.
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.0,
            snowiness: 0.0,
            wind_x: 0.0,
            wind_z: 0.0,
            time_s: 0.0,
            color: [1.0, 1.0, 1.0],
        }
    }
}

impl PrecipParams {
    /// Whether the pass should draw anything at all.
    #[inline]
    pub fn active(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }

    /// Fall speed at this phase, **m/s** — the rain/snow lerp.
    #[inline]
    pub fn fall_speed(&self) -> f32 {
        let s = self.snowiness.clamp(0.0, 1.0);
        RAIN_FALL_SPEED + (SNOW_FALL_SPEED - RAIN_FALL_SPEED) * s
    }

    /// Half-length of the drawn particle along its velocity, **metres**: a
    /// streak for rain, collapsing to the flake's radius for snow.
    #[inline]
    pub fn half_length(&self) -> f32 {
        let s = self.snowiness.clamp(0.0, 1.0);
        let rain = RAIN_STREAK_M * 0.5;
        rain + (SNOW_RADIUS_M - rain) * s
    }

    /// Half-width of the drawn particle, **metres**.
    #[inline]
    pub fn radius(&self) -> f32 {
        let s = self.snowiness.clamp(0.0, 1.0);
        RAIN_RADIUS_M + (SNOW_RADIUS_M - RAIN_RADIUS_M) * s
    }

    /// How many particles to draw at this intensity and tier.
    ///
    /// Zero when inactive, so the caller's "no particles" and "no draw" agree.
    #[inline]
    pub fn count(&self, quality: PrecipQuality) -> u32 {
        if !self.active() {
            return 0;
        }
        let n = quality.max_particles() as f32 * self.intensity.clamp(0.0, 1.0);
        n.round().max(1.0) as u32
    }

    /// The three wrapped displacements the shader needs, **metres**:
    /// `[drift_x, drift_z, fall]`.
    ///
    /// Computed in `f64` and reduced into one box period before it becomes an
    /// `f32`, for the reason [`crate::clouds::wind_offset`] spells out: the raw
    /// displacement grows without bound with the level's clock, and an `f32` at
    /// 780 km has 6 cm of resolution. Because the lattice is periodic in exactly
    /// these periods, subtracting whole boxes is a **no-op on the result**, so
    /// the wrap costs nothing and buys back the whole mantissa.
    ///
    /// `fall` is returned **positive** (metres already fallen); the shader
    /// subtracts it.
    #[inline]
    pub fn offsets(&self) -> [f32; 3] {
        let wrap = |v: f64, period: f32| -> f32 {
            let d = v * self.time_s;
            if !d.is_finite() {
                return 0.0;
            }
            d.rem_euclid(period as f64) as f32
        };
        [
            wrap(self.wind_x as f64, PRECIP_BOX_XZ_M),
            wrap(self.wind_z as f64, PRECIP_BOX_XZ_M),
            wrap(self.fall_speed() as f64, PRECIP_BOX_Y_M),
        ]
    }

    /// The camera's world position reduced modulo the box, **metres** — the
    /// world-anchoring term described in the module docs.
    ///
    /// Taken in `f64` because a world coordinate is `f64` by architecture rule 3
    /// and the modulo of a large `f32` would already have lost the fraction this
    /// exists to preserve.
    #[inline]
    pub fn eye_mod(eye_world_m: glam::DVec3) -> [f32; 3] {
        let m = |v: f64, period: f32| -> f32 {
            if !v.is_finite() {
                return 0.0;
            }
            v.rem_euclid(period as f64) as f32
        };
        [
            m(eye_world_m.x, PRECIP_BOX_XZ_M),
            m(eye_world_m.y, PRECIP_BOX_Y_M),
            m(eye_world_m.z, PRECIP_BOX_XZ_M),
        ]
    }
}

// ── CPU references for `shaders/precip.wgsl` ────────────────────────────────
//
// These three functions have **no production caller**: the placement they
// describe happens in the vertex shader, and this module's job on the CPU is to
// fill the uniform. They exist for the reason `clouds::cloud_density`'s CPU
// reference exists — a GPU-side property that only a golden can observe is a
// property that goes unexamined on every CI leg without an adapter, and the
// world-anchoring invariant here (a floating-origin rebase must not move a
// single particle) is exactly that kind of property. Each mirrors its WGSL
// counterpart operation for operation, named in its doc comment, and
// [`particle_offset`] composes them the way `vs` composes them, so what the
// gates below evaluate is the shader's arithmetic rather than a paraphrase of it.

/// Base position of particle `i` inside the box, **metres** in `[0, box)`.
///
/// CPU reference for `precip_base` in `shaders/precip.wgsl`: three
/// [`crate::clouds::cloud_hash`] draws at distinct salts, each taken through
/// [`crate::clouds::hash_unit`] (24 bits, so every value is exactly
/// representable in `f32` and the CPU and GPU agree bit for bit).
pub fn precip_base(i: u32) -> [f32; 3] {
    let h = |salt: u32| {
        crate::clouds::hash_unit(crate::clouds::cloud_hash(i, salt, 0, PRECIP_HASH_SALT))
    };
    [
        h(0) * PRECIP_BOX_XZ_M,
        h(1) * PRECIP_BOX_Y_M,
        h(2) * PRECIP_BOX_XZ_M,
    ]
}

/// Reduce `v` into `[−period/2, period/2)` — CPU reference for
/// `precip_wrap_signed` in WGSL. (`fract` there is `x − floor(x)`, i.e. the
/// Euclidean remainder, which is what `rem_euclid` is here.)
#[inline]
pub fn wrap_signed(v: f32, period: f32) -> f32 {
    let half = period * 0.5;
    (v + half).rem_euclid(period) - half
}

/// A particle's offset **from the eye**, metres — CPU reference for the
/// world-anchoring block of `vs` in `shaders/precip.wgsl`, composed exactly as
/// the shader composes it:
///
/// ```text
/// offset = wrap_signed(base(i) + drift − eye_mod, box)
/// p_world = eye_world + offset
/// ```
///
/// `drift` is `[wind_x·t, −fallen, wind_z·t]` already wrapped by
/// [`PrecipParams::offsets`]; `eye_mod` is [`PrecipParams::eye_mod`]. Because
/// `eye_mod ≡ eye_world (mod box)`, the resulting world position is congruent to
/// `base + drift` modulo the box **whatever the eye is** — which is the whole
/// world-anchoring claim, and what
/// `a_floating_origin_rebase_moves_no_particle` evaluates.
pub fn particle_offset(i: u32, drift: [f32; 3], eye_mod: [f32; 3]) -> [f32; 3] {
    let base = precip_base(i);
    let periods = [PRECIP_BOX_XZ_M, PRECIP_BOX_Y_M, PRECIP_BOX_XZ_M];
    std::array::from_fn(|k| wrap_signed(base[k] + drift[k] - eye_mod[k], periods[k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_off_path_is_dead() {
        let p = PrecipParams::default();
        assert!(!p.active());
        assert_eq!(p.count(PrecipQuality::High), 0);
        // Enabled but dry is still nothing: intensity is the master, so a weather
        // block that has blended down to zero rain stops drawing rather than
        // drawing invisible particles forever.
        let dry = PrecipParams {
            enabled: true,
            ..PrecipParams::default()
        };
        assert!(!dry.active());
        assert_eq!(dry.count(PrecipQuality::High), 0);
    }

    #[test]
    fn intensity_scales_the_count_linearly_and_tiers_clamp_down() {
        let p = |i: f32| PrecipParams {
            enabled: true,
            intensity: i,
            ..PrecipParams::default()
        };
        assert_eq!(p(1.0).count(PrecipQuality::High), 48_000);
        assert_eq!(p(0.5).count(PrecipQuality::High), 24_000);
        assert_eq!(p(0.25).count(PrecipQuality::High), 12_000);
        // A tier only ever reduces.
        assert!(p(1.0).count(PrecipQuality::Medium) < p(1.0).count(PrecipQuality::High));
        assert!(p(1.0).count(PrecipQuality::Low) < p(1.0).count(PrecipQuality::Medium));
        // Hostile input cannot ask for a billion particles.
        assert_eq!(p(1e9).count(PrecipQuality::High), 48_000);
        // …and a tiny-but-nonzero intensity still draws *something*, so the knob
        // has no dead band at the bottom.
        assert_eq!(p(1e-9).count(PrecipQuality::High), 1);
    }

    #[test]
    fn rain_and_snow_differ_in_the_ways_that_matter() {
        let rain = PrecipParams {
            enabled: true,
            intensity: 1.0,
            snowiness: 0.0,
            ..PrecipParams::default()
        };
        let snow = PrecipParams {
            snowiness: 1.0,
            ..rain
        };
        assert_eq!(rain.fall_speed(), RAIN_FALL_SPEED);
        assert_eq!(snow.fall_speed(), SNOW_FALL_SPEED);
        assert!(
            snow.fall_speed() < rain.fall_speed() * 0.2,
            "snow must drift"
        );
        // Rain is a long thin streak; snow is a round flake.
        assert!(rain.half_length() > rain.radius() * 20.0);
        assert!((snow.half_length() - snow.radius()).abs() < 1e-6);
        // The blend is continuous — a half-frozen shower is between the two.
        let half = PrecipParams {
            snowiness: 0.5,
            ..rain
        };
        assert!(half.fall_speed() < rain.fall_speed() && half.fall_speed() > snow.fall_speed());
        assert!(half.radius() > rain.radius() && half.radius() < snow.radius());
        // Out-of-range phase is clamped rather than extrapolated into a flake
        // that falls upward.
        let hostile = PrecipParams {
            snowiness: 4.0,
            ..rain
        };
        assert_eq!(hostile.fall_speed(), SNOW_FALL_SPEED);
    }

    /// The whole point of wrapping in `f64`: a day of falling must still resolve
    /// millimetres. Unwrapped, `9 m/s × 86 400 s` is 778 km, where an `f32` step
    /// is 6 cm.
    #[test]
    fn offsets_wrap_into_one_box_and_stay_precise_all_day() {
        let p = PrecipParams {
            enabled: true,
            intensity: 1.0,
            wind_x: 6.0,
            wind_z: -2.0,
            time_s: 86_400.0,
            ..PrecipParams::default()
        };
        let [dx, dz, fall] = p.offsets();
        for (v, period) in [
            (dx, PRECIP_BOX_XZ_M),
            (dz, PRECIP_BOX_XZ_M),
            (fall, PRECIP_BOX_Y_M),
        ] {
            assert!((0.0..period).contains(&v), "{v} outside [0, {period})");
        }
        // A whole box of drift is a NO-OP — simultaneously the tileability proof
        // and what keeps an all-day session from quantizing into stair-steps.
        let one_box = PrecipParams {
            time_s: 86_400.0 + PRECIP_BOX_XZ_M as f64 / 6.0,
            wind_z: 0.0,
            ..p
        };
        let base = PrecipParams { wind_z: 0.0, ..p };
        assert!((one_box.offsets()[0] - base.offsets()[0]).abs() < 1e-3);

        // Sub-second resolution survives a full day, which is the claim.
        let later = PrecipParams {
            time_s: 86_400.0 + 0.001,
            ..p
        };
        assert_ne!(later.offsets()[2], p.offsets()[2]);

        // Non-finite input yields zero rather than a NaN that would blank the
        // uniform's `PartialEq` and re-upload every frame forever.
        let bad = PrecipParams {
            time_s: f64::INFINITY,
            ..p
        };
        assert_eq!(bad.offsets(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn eye_mod_reduces_a_world_coordinate_without_losing_the_fraction() {
        // Far out in an f64 world: the modulo must still carry millimetres, which
        // an f32 at 3.2 million metres (0.25 m of resolution) could not.
        let e = PrecipParams::eye_mod(glam::DVec3::new(3_200_000.125, -17.5, -4.25));
        assert!(
            (e[0] - (3_200_000.125f64).rem_euclid(f64::from(PRECIP_BOX_XZ_M)) as f32).abs() < 1e-4
        );
        assert!(e[0] >= 0.0 && e[0] < PRECIP_BOX_XZ_M);
        // Negative altitudes reduce into the positive range (rem_euclid, not %).
        assert!(e[1] >= 0.0 && e[1] < PRECIP_BOX_Y_M);
        assert!(e[2] >= 0.0 && e[2] < PRECIP_BOX_XZ_M);
        assert_eq!(
            PrecipParams::eye_mod(glam::DVec3::new(f64::NAN, 0.0, 0.0))[0],
            0.0
        );
    }

    /// The determinism backbone: base positions are a pure function of the index,
    /// they fill the box, and they are **exactly representable in `f32`** so the
    /// CPU reference and the WGSL mirror cannot disagree by a ULP.
    #[test]
    fn base_positions_are_pure_uniform_and_exactly_representable() {
        for i in [0u32, 1, 2, 47, 1_000, 47_999] {
            let a = precip_base(i);
            assert_eq!(a, precip_base(i), "not a pure function of the index");
            assert!(a[0] >= 0.0 && a[0] < PRECIP_BOX_XZ_M);
            assert!(a[1] >= 0.0 && a[1] < PRECIP_BOX_Y_M);
            assert!(a[2] >= 0.0 && a[2] < PRECIP_BOX_XZ_M);
        }
        assert_ne!(precip_base(0), precip_base(1), "the hash is not diffusing");

        // Roughly uniform over the box: an 8-bucket histogram of 48 000 draws
        // must not have a bucket outside ±20 % of the mean. This is what catches
        // a salt collision, which would otherwise show up as rain falling in
        // sheets.
        let mut hist = [0u32; 8];
        for i in 0..48_000u32 {
            let b = precip_base(i);
            let k = ((b[0] / PRECIP_BOX_XZ_M) * 8.0) as usize;
            hist[k.min(7)] += 1;
        }
        let mean = 48_000.0 / 8.0;
        for (k, n) in hist.iter().enumerate() {
            let r = *n as f32 / mean;
            assert!(r > 0.8 && r < 1.2, "bucket {k} is {r:.2}× the mean");
        }

        // Every draw is 24-bit, so `f32` holds it exactly and the WGSL mirror
        // (which does the same integer arithmetic) lands on the same bits.
        let h = crate::clouds::hash_unit(crate::clouds::cloud_hash(7, 1, 0, PRECIP_HASH_SALT));
        assert_eq!((h * 16_777_216.0).fract(), 0.0);
    }

    /// **The world-anchoring gate**, evaluated through the composed CPU
    /// reference rather than through its pieces — the pieces are individually
    /// harmless and the property lives entirely in how they compose.
    ///
    /// A floating-origin rebase moves `origin` by a multiple of
    /// [`inf_math::ORIGIN_SNAP`] and moves the render-local eye by the same
    /// amount in the opposite direction. `p_world = origin + eye_local + offset`
    /// must be **unchanged**: the origin cancels algebraically, and `offset`
    /// depends on the eye only through `eye_mod`, which is computed from the
    /// *world* eye and so does not move at all. Get that wrong and the whole
    /// sheet of rain jumps every time the camera crosses a rebase boundary.
    ///
    /// Asserted at f64 world scale, where an `f32` shortcut would already have
    /// lost the fraction: 3 200 km out, an `f32` metre has 0.25 m of resolution.
    #[test]
    fn a_floating_origin_rebase_moves_no_particle() {
        use glam::DVec3;
        use inf_math::{FloatingOrigin, ORIGIN_SNAP};

        let p = PrecipParams {
            enabled: true,
            intensity: 1.0,
            wind_x: 22.0,
            wind_z: 9.0,
            time_s: 4_321.5,
            ..PrecipParams::default()
        };
        let [dx, dz, fall] = p.offsets();
        let drift = [dx, -fall, dz];

        // One eye, two origins: the origin *before* a rebase and the one after a
        // camera walk of several snap quanta. The shader sees `eye_local` and
        // `eye_mod`; the world position it produces must be identical.
        let eye_world = DVec3::new(3_200_000.125, 137.75, -412.5);
        let eye_mod = PrecipParams::eye_mod(eye_world);
        let before = FloatingOrigin::new(eye_world - DVec3::splat(4.0 * ORIGIN_SNAP));
        let after = FloatingOrigin::new(eye_world + DVec3::splat(7.0 * ORIGIN_SNAP));
        assert_ne!(before.origin(), after.origin(), "the rebase did nothing");

        let world_of = |o: &FloatingOrigin, i: u32| -> DVec3 {
            // `view.eye.xyz` is the render-local eye; `centre = view.eye + offset`
            // is render-local too, so the world position is `origin + centre`.
            let eye_local = o.to_render(eye_world);
            let off = particle_offset(i, drift, eye_mod);
            o.to_world(glam::Vec3::new(
                eye_local.x + off[0],
                eye_local.y + off[1],
                eye_local.z + off[2],
            ))
        };

        let periods = [
            f64::from(PRECIP_BOX_XZ_M),
            f64::from(PRECIP_BOX_Y_M),
            f64::from(PRECIP_BOX_XZ_M),
        ];
        for i in [0u32, 1, 17, 4_097, 47_999] {
            let a = world_of(&before, i);
            let b = world_of(&after, i);
            assert!(
                (a - b).length() < 1e-3,
                "particle {i} moved {} m across a rebase",
                (a - b).length()
            );
            // …and it sits where the LATTICE says, not merely somewhere stable:
            // congruent to `base + drift` modulo the box, which is what makes it
            // world-anchored rather than camera-anchored.
            let base = precip_base(i);
            for (k, axis) in [a.x, a.y, a.z].into_iter().enumerate() {
                let want = f64::from(base[k] + drift[k]).rem_euclid(periods[k]);
                let got = axis.rem_euclid(periods[k]);
                let d = (got - want).abs().min(periods[k] - (got - want).abs());
                assert!(d < 1e-2, "particle {i} axis {k}: {got} vs lattice {want}");
            }
        }

        // The complement: moving the EYE (not the origin) really does re-wrap
        // particles into a new box around it — otherwise the two assertions above
        // would hold for a layer that simply never moves.
        let far = DVec3::new(3_200_000.125 + 21.5, 137.75, -412.5);
        let far_mod = PrecipParams::eye_mod(far);
        assert_ne!(
            particle_offset(17, drift, eye_mod),
            particle_offset(17, drift, far_mod),
            "a 21.5 m camera move re-wrapped nothing"
        );
    }

    #[test]
    fn wrap_signed_centres_on_zero() {
        assert_eq!(wrap_signed(0.0, 48.0), 0.0);
        assert_eq!(wrap_signed(1.0, 48.0), 1.0);
        assert_eq!(wrap_signed(47.0, 48.0), -1.0);
        assert_eq!(wrap_signed(-1.0, 48.0), -1.0);
        assert_eq!(wrap_signed(49.0, 48.0), 1.0);
        for v in [-500.0f32, -24.0, 0.0, 23.9, 96.5, 1e5] {
            let w = wrap_signed(v, 48.0);
            assert!((-24.0..24.0).contains(&w), "{v} -> {w}");
        }
    }

    /// The tier mapping is total and injective, so a tier switch always changes
    /// the particle budget — the same property `CloudQuality` is held to.
    #[test]
    fn precip_quality_is_a_total_injective_function_of_atmosphere_quality() {
        use crate::atmosphere::AtmosphereQuality;
        let tiers = [
            AtmosphereQuality::Low,
            AtmosphereQuality::Medium,
            AtmosphereQuality::High,
        ];
        let mut seen = Vec::new();
        for q in tiers {
            let n = PrecipQuality::from_atmosphere(q).max_particles();
            assert!(!seen.contains(&n), "{q:?} shares a particle budget");
            seen.push(n);
            assert_eq!(
                PrecipQuality::from_atmosphere(q),
                PrecipQuality::from_atmosphere(q)
            );
        }
    }
}
