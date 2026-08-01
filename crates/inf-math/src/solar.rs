//! Deterministic solar & lunar position from a time of day and a place (P17.1).
//!
//! This is the engine's **only** source of a sun direction. Phase 2 through 16
//! shipped a compile-time `inf_render::camera::SUN_DIR` constant; P17.1 retires
//! it in favour of a `TimeOfDay` + `SkyAtmosphere` component pair whose direction
//! is computed here and projected into the renderer by both scene builders.
//!
//! # Placement
//!
//! `inf-math` — not `inf-render` — because every ring of the engine needs it and
//! none of them should pay for `wgpu` to get it: the editor viewport projector,
//! the shipped player projector, the fixed-step simulation that advances the
//! clock, the Blueprint `sky.*` host namespace, and the sequencer property track
//! that keys [`TimeOfDay::seconds`]. `inf-math` is the leaf every one of those
//! already depends on (directly or through `inf-ecs`), and it is where the
//! bit-portable trigonometry this module is built on ([`crate::portable`]) lives.
//!
//! [`TimeOfDay::seconds`]: https://docs.rs/inf-ecs
//!
//! # Determinism & the psin/pcos law
//!
//! The house law (see [`crate::portable`]) is that `f64::sin` is **not**
//! bit-identical across platforms, so anything feeding *committed bytes* must use
//! the portable polynomials. A sun direction feeds GPU uniforms and a
//! `RenderScene`, not a serialized asset — the committed bytes here are
//! `TimeOfDay`'s own plain `f64` fields, and [`advance`] touches them with nothing
//! but IEEE add/mul/floor. So the law does not strictly *bind* this module.
//!
//! It is followed anyway, and that is load-bearing for two gates that do compare
//! numbers across process boundaries:
//!
//! * **PIE == shipping** compares a sun-direction trace captured in the editor's
//!   in-process simulation against one captured in a shipped player subprocess.
//! * The **replay determinism** harness compares two runs of the same sim.
//!
//! Both would pass with `std` trig on one machine and could rot the moment a
//! trace is recorded on one platform and replayed on another. Using
//! [`psin64`]/[`pcos64`] (plus `sqrt`, `floor`, add and mul — all exactly
//! specified by IEEE-754) makes every function below bit-identical on every
//! target, so the traces are portable artefacts rather than machine-local ones.
//!
//! The two *display-only* helpers [`elevation_deg`] and [`azimuth_deg`] do call
//! `std` trig (`asin`/`atan2`); they exist for World Settings readouts and unit
//! tests and are documented as never feeding render or committed state.
//!
//! # Coordinate frame
//!
//! Engine world axes are Y-up and right-handed. This module maps the standard
//! East/North/Up topocentric frame onto them as:
//!
//! | engine axis | compass |
//! |-------------|---------|
//! | `+X`        | East    |
//! | `+Y`        | Up      |
//! | `+Z`        | South   |
//!
//! (`+X × +Y = +Z` ⇒ East × Up = South, which is right-handed — the same
//! handedness as the rest of the renderer.) North is therefore `-Z`, which is
//! also the engine's conventional "forward".
//!
//! Every direction returned points **toward** the body, matching
//! `inf_render::scene::RenderLight::direction` and the `view.sun_dir` uniform.
//!
//! # Time
//!
//! [`SolarInput::seconds`] is **UTC seconds since midnight**, in `[0, 86400)`.
//! Local solar time is `UTC + longitude_deg / 15` hours, so longitude is a real
//! knob: 12:00 with `longitude_deg = 0` is solar noon on the prime meridian,
//! and 12:00 with `longitude_deg = 15` is one hour *past* local noon.
//!
//! The engine's year is a fixed **365 days** with no leap day and no year field;
//! [`SolarInput::day_of_year`] is `1..=365` (larger values clamp). This is a
//! deliberate simplification — a game clock, not an ephemeris.
//!
//! # Accuracy
//!
//! Declination and the equation of time use Spencer (1971) Fourier fits, which
//! are the standard "good enough for solar engineering" approximations:
//! declination within ≈0.03°, equation of time within ≈0.3 min (≈0.08° of hour
//! angle) of the true value. Combined with the exact spherical-astronomy
//! transform below, sun **elevation** lands within a few hundredths of a degree
//! of a reference ephemeris for the solstice/equinox cases the tests pin.
//! Atmospheric refraction (which lifts a body near the horizon by up to ≈0.57°)
//! is deliberately **not** applied: a renderer wants the geometric direction.
//!
//! The moon is an explicitly documented v1 approximation — see [`moon_direction`].

use glam::DVec3;
use std::f64::consts::{PI, TAU};

use crate::portable::{pcos64, psin64};

/// Seconds in one solar day — the modulus [`SolarInput::seconds`] lives in.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Days in the engine's (fixed, leap-free) year.
pub const DAYS_PER_YEAR: u32 = 365;

/// Mean obliquity of the ecliptic, degrees (Earth's axial tilt). Used by the
/// lunar model; the solar declination comes from Spencer's fit instead.
pub const OBLIQUITY_DEG: f64 = 23.4393;

/// Mean synodic month (new moon to new moon), days — the lunar phase period.
pub const SYNODIC_MONTH_DAYS: f64 = 29.530_588_9;

/// Day of year of the northward (vernal) equinox in the fixed 365-day year —
/// the zero point of the ecliptic longitude used by the lunar model.
const VERNAL_EQUINOX_DAY: f64 = 80.0;

/// Where and when: the inputs a sun/moon direction is a pure function of.
///
/// Mirrors the `TimeOfDay` ECS component field-for-field (minus `rate`, which
/// only drives [`advance`]); the projectors build one of these from the
/// component and hand it to [`bodies`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarInput {
    /// UTC seconds since midnight. Values outside `[0, 86400)` are wrapped, so a
    /// caller may pass an unwrapped accumulator.
    pub seconds: f64,
    /// Day of year, `1..=365` (clamped). No leap day; see the module docs.
    pub day_of_year: u32,
    /// Geodetic latitude in degrees, `+` north. Clamped to `[-90, 90]`.
    pub latitude_deg: f64,
    /// Longitude in degrees, `+` east. Wrapped into `[-180, 180)`.
    pub longitude_deg: f64,
}

impl Default for SolarInput {
    /// The engine's default time and place — see the `TimeOfDay` component,
    /// whose defaults these mirror: 10:00 UTC on day 172 (the June solstice) at
    /// 48.9° N on the prime meridian.
    ///
    /// Chosen so [`sun_direction`] lands within **1.6°** of the legacy
    /// `SUN_DIR` constant (`normalize(0.45, 0.75, 0.3)`) the engine rendered with
    /// from Phase 2 to Phase 16 — a scene that opts into time of day keeps
    /// essentially the look it had.
    fn default() -> Self {
        Self {
            seconds: 36_000.0,
            day_of_year: 172,
            latitude_deg: 48.9,
            longitude_deg: 0.0,
        }
    }
}

impl SolarInput {
    /// `seconds` wrapped into `[0, 86400)`.
    #[inline]
    pub fn wrapped_seconds(&self) -> f64 {
        wrap_seconds(self.seconds)
    }

    /// `day_of_year` clamped into `1..=365`.
    #[inline]
    pub fn clamped_day(&self) -> u32 {
        self.day_of_year.clamp(1, DAYS_PER_YEAR)
    }

    /// The fraction of the day elapsed, `[0, 1)`.
    #[inline]
    pub fn day_fraction(&self) -> f64 {
        self.wrapped_seconds() / SECONDS_PER_DAY
    }
}

/// Unit directions toward the sun and the moon, plus the scalars a renderer
/// wants for free (they fall out of the same trigonometry).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyBodies {
    /// Unit direction **toward** the sun (engine axes; see the module docs).
    pub sun: DVec3,
    /// Unit direction **toward** the moon.
    pub moon: DVec3,
    /// Lunar phase in `[0, 1)`: `0` = new moon (with the sun), `0.5` = full moon
    /// (opposite the sun).
    pub moon_phase: f64,
}

/// Wrap an unbounded seconds accumulator into `[0, 86400)`.
///
/// `rem_euclid` on a finite `f64` is exact; a non-finite input returns `0.0`
/// rather than propagating a `NaN` into a direction.
#[inline]
pub fn wrap_seconds(seconds: f64) -> f64 {
    if !seconds.is_finite() {
        return 0.0;
    }
    let s = seconds.rem_euclid(SECONDS_PER_DAY);
    // rem_euclid can round to the modulus itself for inputs a hair below zero.
    if s >= SECONDS_PER_DAY {
        0.0
    } else {
        s
    }
}

/// Roll a `1..=365` day of year by `delta` whole days, wrapping the year.
#[inline]
pub fn roll_day(day_of_year: u32, delta: i64) -> u32 {
    let zero_based = i64::from(day_of_year.clamp(1, DAYS_PER_YEAR)) - 1;
    let rolled = (zero_based + delta).rem_euclid(i64::from(DAYS_PER_YEAR));
    (rolled + 1) as u32
}

/// Advance a clock by `rate` sim-seconds per real second over `dt` real seconds,
/// wrapping the day.
///
/// Returns the new `(seconds, day_of_year)`. `rate == 0` freezes the clock;
/// a negative rate runs it backwards (and rolls the day back). Non-finite inputs
/// leave the clock untouched — a `NaN` `dt` must never poison a saved level.
///
/// Only IEEE add/mul/floor are used, so the advanced clock is bit-identical on
/// every platform: this is the function the replay-determinism and
/// PIE-vs-shipping gates run thousands of times.
///
/// Rolling the day at the wrap is not cosmetic. The fractional-year angle γ that
/// drives declination is defined as `(day − 1) + (hours − 12)/24`, so
/// `(day, 86400⁻)` and `(day + 1, 0)` evaluate to the *same* γ — the sun's path
/// is continuous across midnight only because the day increments.
#[inline]
pub fn advance(seconds: f64, day_of_year: u32, rate: f64, dt: f64) -> (f64, u32) {
    let day = day_of_year.clamp(1, DAYS_PER_YEAR);
    if !seconds.is_finite() || !rate.is_finite() || !dt.is_finite() {
        return (wrap_seconds(seconds), day);
    }
    let t = seconds + rate * dt;
    if !t.is_finite() {
        return (wrap_seconds(seconds), day);
    }
    let days = (t / SECONDS_PER_DAY).floor();
    let rolled = if days.abs() < 9.0e15 { days as i64 } else { 0 };
    (wrap_seconds(t), roll_day(day, rolled))
}

/// Spencer's fractional-year angle γ, radians.
///
/// `γ = 2π/365 · ((day − 1) + (hours − 12) / 24)` — the argument both Fourier
/// fits below are expressed in.
#[inline]
fn fractional_year(input: &SolarInput) -> f64 {
    let day = f64::from(input.clamped_day());
    let hours = input.wrapped_seconds() / 3600.0;
    TAU / f64::from(DAYS_PER_YEAR) * (day - 1.0 + (hours - 12.0) / 24.0)
}

/// Solar declination in radians — Spencer (1971), accurate to ≈0.03°.
///
/// Positive is north of the celestial equator (northern summer).
pub fn declination_rad(input: &SolarInput) -> f64 {
    let g = fractional_year(input);
    let (s1, c1) = (psin64(g), pcos64(g));
    let (s2, c2) = (psin64(2.0 * g), pcos64(2.0 * g));
    let (s3, c3) = (psin64(3.0 * g), pcos64(3.0 * g));
    0.006_918 - 0.399_912 * c1 + 0.070_257 * s1 - 0.006_758 * c2 + 0.000_907 * s2 - 0.002_697 * c3
        + 0.001_48 * s3
}

/// The equation of time in **minutes** — Spencer (1971), accurate to ≈0.3 min.
///
/// How far ahead (positive) true solar time runs of mean solar time, the
/// combined effect of Earth's orbital eccentricity and axial tilt. It is what
/// makes the analemma a figure eight.
pub fn equation_of_time_minutes(input: &SolarInput) -> f64 {
    let g = fractional_year(input);
    let (s1, c1) = (psin64(g), pcos64(g));
    let (s2, c2) = (psin64(2.0 * g), pcos64(2.0 * g));
    229.18 * (0.000_075 + 0.001_868 * c1 - 0.032_077 * s1 - 0.014_615 * c2 - 0.040_849 * s2)
}

/// The sun's **hour angle** in radians: `0` at local solar noon, increasing
/// westward (afternoon positive), wrapped to `(-π, π]`.
pub fn hour_angle_rad(input: &SolarInput) -> f64 {
    // Earth's rotation since UTC midnight, re-zeroed at 12:00 UTC …
    let rotation = TAU * input.day_fraction() - PI;
    // … plus the longitude offset (east is ahead) and the equation of time
    // (minutes → radians: one minute of time is 2π/1440 of a rotation).
    let longitude = wrap_longitude(input.longitude_deg).to_radians();
    let eot = equation_of_time_minutes(input) * TAU / 1440.0;
    wrap_pi(rotation + longitude + eot)
}

/// Wrap an angle into `(-π, π]`.
#[inline]
fn wrap_pi(a: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let x = (a + PI).rem_euclid(TAU);
    x - PI
}

/// Wrap a longitude into `[-180, 180)`.
#[inline]
fn wrap_longitude(deg: f64) -> f64 {
    if !deg.is_finite() {
        return 0.0;
    }
    (deg + 180.0).rem_euclid(360.0) - 180.0
}

/// The topocentric direction toward a body from its declination and hour angle.
///
/// The textbook spherical-astronomy transform, written **directly as a vector**
/// so no `asin`/`atan2` is ever needed — which is exactly what lets the whole
/// module stay on the bit-portable [`psin64`]/[`pcos64`] pair:
///
/// ```text
/// East  = −cos δ · sin H
/// Up    =  sin δ · sin φ + cos δ · cos φ · cos H
/// North =  sin δ · cos φ − cos δ · sin φ · cos H      (engine Z = −North)
/// ```
fn direction_from(decl_rad: f64, hour_rad: f64, latitude_deg: f64) -> DVec3 {
    let lat = latitude_deg.clamp(-90.0, 90.0).to_radians();
    let (sd, cd) = (psin64(decl_rad), pcos64(decl_rad));
    let (sh, ch) = (psin64(hour_rad), pcos64(hour_rad));
    let (sp, cp) = (psin64(lat), pcos64(lat));

    let east = -cd * sh;
    let up = sd * sp + cd * cp * ch;
    let north = sd * cp - cd * sp * ch;

    let v = DVec3::new(east, up, -north);
    // Length is 1 ± ~1e-7 (the polynomials' error); renormalising costs one sqrt
    // and one reciprocal, both IEEE-exact, so the result stays bit-portable.
    let len_sq = v.length_squared();
    if len_sq > 1e-12 {
        v / len_sq.sqrt()
    } else {
        DVec3::Y
    }
}

/// Unit direction **toward the sun** (engine axes; see the module docs).
pub fn sun_direction(input: &SolarInput) -> DVec3 {
    direction_from(
        declination_rad(input),
        hour_angle_rad(input),
        input.latitude_deg,
    )
}

/// Lunar phase in `[0, 1)`: `0` = new (with the sun), `0.5` = full (opposite).
///
/// Derived from the day of year alone, so it repeats every simulated year — the
/// engine has no year field, and a 365-day cycle over a 29.53-day month leaves a
/// visible seam at the new year. That is accepted, documented v1 behaviour: this
/// is set dressing, not an almanac.
pub fn moon_phase(input: &SolarInput) -> f64 {
    let days = f64::from(input.clamped_day()) - 1.0 + input.day_fraction();
    (days / SYNODIC_MONTH_DAYS).rem_euclid(1.0)
}

/// Unit direction **toward the moon** — an explicitly approximate v1 model.
///
/// The moon is placed on the **ecliptic**, `phase · 360°` of ecliptic longitude
/// east of the sun:
///
/// * its hour angle is the sun's minus `phase · 2π` (a body further east
///   transits later, so its hour angle is smaller);
/// * its declination follows from `sin δ = sin ε · sin λ`, with `λ` the moon's
///   ecliptic longitude and `ε` the obliquity. `cos δ = √(1 − sin²δ)` is
///   unambiguous because `|δ| ≤ ε < 90°`.
///
/// At a full moon (`phase = 0.5`) this puts the moon opposite the sun, up at
/// local midnight, riding high in winter and low in summer — the behaviour a
/// player reads as "correct".
///
/// **Not modelled (v1):** the 5.14° inclination of the lunar orbit to the
/// ecliptic, the orbit's eccentricity (the equation of the centre, ±6°),
/// nodal/apsidal precession, parallax, and libration. Expect errors of several
/// degrees against a real ephemeris. P17.2 (stars and moon at night) may refine
/// this; nothing outside this function depends on the model.
pub fn moon_direction(input: &SolarInput) -> DVec3 {
    let phase = moon_phase(input);
    let hour = wrap_pi(hour_angle_rad(input) - phase * TAU);

    // Sun's ecliptic longitude, measured from the northward equinox.
    let days = f64::from(input.clamped_day()) - 1.0 + input.day_fraction();
    let sun_lambda = TAU * (days - VERNAL_EQUINOX_DAY) / f64::from(DAYS_PER_YEAR);
    let moon_lambda = sun_lambda + phase * TAU;

    let sin_decl = psin64(OBLIQUITY_DEG.to_radians()) * psin64(moon_lambda);
    let sin_decl = sin_decl.clamp(-1.0, 1.0);
    let cos_decl = (1.0 - sin_decl * sin_decl).max(0.0).sqrt();

    let lat = input.latitude_deg.clamp(-90.0, 90.0).to_radians();
    let (sh, ch) = (psin64(hour), pcos64(hour));
    let (sp, cp) = (psin64(lat), pcos64(lat));

    let east = -cos_decl * sh;
    let up = sin_decl * sp + cos_decl * cp * ch;
    let north = sin_decl * cp - cos_decl * sp * ch;

    let v = DVec3::new(east, up, -north);
    let len_sq = v.length_squared();
    if len_sq > 1e-12 {
        v / len_sq.sqrt()
    } else {
        DVec3::NEG_Y
    }
}

/// Both bodies plus the phase, in one pass — what the scene projectors call.
pub fn bodies(input: &SolarInput) -> SkyBodies {
    SkyBodies {
        sun: sun_direction(input),
        moon: moon_direction(input),
        moon_phase: moon_phase(input),
    }
}

/// Altitude above the horizon in degrees, from a direction.
///
/// **Display only** (World Settings readouts, unit tests): this calls `std`
/// `asin`, which is not bit-portable. Never feed its result into render state,
/// a trace, or committed bytes — use the direction vector itself.
pub fn elevation_deg(dir: DVec3) -> f64 {
    dir.normalize_or_zero()
        .y
        .clamp(-1.0, 1.0)
        .asin()
        .to_degrees()
}

/// Compass azimuth in degrees, `0` = north, increasing clockwise through east.
///
/// **Display only** — see [`elevation_deg`].
pub fn azimuth_deg(dir: DVec3) -> f64 {
    let d = dir.normalize_or_zero();
    // North is −Z, East is +X.
    let a = d.x.atan2(-d.z).to_degrees();
    if a < 0.0 {
        a + 360.0
    } else {
        a
    }
}

/// Format `seconds` as `HH:MM:SS` — the World Settings readback, shared so the
/// backend and any CLI print the clock identically.
pub fn format_clock(seconds: f64) -> String {
    let s = wrap_seconds(seconds);
    let total = s as u32;
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total / 60) % 60,
        total % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(seconds: f64, day: u32, lat: f64, lon: f64) -> SolarInput {
        SolarInput {
            seconds,
            day_of_year: day,
            latitude_deg: lat,
            longitude_deg: lon,
        }
    }

    /// Local solar noon in UTC seconds for a longitude, undoing the equation of
    /// time so the hour angle is (numerically) zero. Iterated because the eot
    /// itself depends (weakly) on the time of day.
    fn solar_noon(day: u32, lon: f64) -> f64 {
        let base = 43_200.0 - lon / 15.0 * 3600.0;
        let mut t = base;
        for _ in 0..4 {
            t = base - equation_of_time_minutes(&at(t, day, 0.0, lon)) * 60.0;
        }
        t
    }

    #[test]
    fn noon_hour_angle_is_zero() {
        for &(day, lon) in &[
            (1u32, 0.0f64),
            (81, 0.0),
            (172, 0.0),
            (172, 45.0),
            (300, -120.0),
        ] {
            let t = solar_noon(day, lon);
            let h = hour_angle_rad(&at(t, day, 0.0, lon));
            assert!(h.abs() < 1e-6, "day {day} lon {lon}: hour angle {h}");
        }
    }

    #[test]
    fn equinox_noon_elevation_is_ninety_minus_latitude() {
        // Day 81 is the northward equinox in the fixed 365-day year: the
        // declination is within a third of a degree of zero, so noon elevation is
        // 90° − |lat| to that accuracy.
        let day = 81;
        let t = solar_noon(day, 0.0);
        let decl = declination_rad(&at(t, day, 0.0, 0.0)).to_degrees();
        assert!(decl.abs() < 0.5, "equinox declination {decl}");
        for &lat in &[0.0_f64, 23.4393, 45.0, -33.9, 60.0] {
            let el = elevation_deg(sun_direction(&at(t, day, lat, 0.0)));
            // Exact spherical geometry, with the residual declination folded in.
            let expected = 90.0 - (lat - decl).abs();
            assert!(
                (el - expected).abs() < 0.02,
                "lat {lat}: elevation {el} vs {expected}"
            );
            assert!(
                (el - (90.0 - lat.abs())).abs() < 0.6,
                "lat {lat}: elevation {el} far from 90−|lat|"
            );
        }
    }

    #[test]
    fn june_solstice_noon_elevation() {
        // Day 172 ≈ the June solstice: declination ≈ +23.44°, so noon elevation
        // is 90 − (lat − decl). Reference values from the geometry itself; the
        // tolerance covers Spencer's ≈0.03° declination fit.
        let day = 172;
        let t = solar_noon(day, 0.0);
        let decl = declination_rad(&at(t, day, 0.0, 0.0)).to_degrees();
        assert!(
            (decl - 23.44).abs() < 0.1,
            "june solstice declination {decl}"
        );
        for &lat in &[0.0_f64, 45.0, 66.5, -45.0] {
            let el = elevation_deg(sun_direction(&at(t, day, lat, 0.0)));
            let expected = 90.0 - (lat - decl).abs();
            assert!(
                (el - expected).abs() < 0.05,
                "lat {lat}: elevation {el} vs {expected}"
            );
        }
    }

    #[test]
    fn december_solstice_declination_is_southern() {
        // Day 355 ≈ the December solstice: declination ≈ −23.44°.
        let decl = declination_rad(&at(43_200.0, 355, 0.0, 0.0)).to_degrees();
        assert!((decl + 23.44).abs() < 0.15, "december declination {decl}");
    }

    #[test]
    fn polar_night_and_midnight_sun() {
        // Above the Arctic circle the sun never sets at the June solstice and
        // never rises at the December one.
        let lat = 78.0;
        for step in 0..24 {
            let t = f64::from(step) * 3600.0;
            let summer = elevation_deg(sun_direction(&at(t, 172, lat, 0.0)));
            let winter = elevation_deg(sun_direction(&at(t, 355, lat, 0.0)));
            assert!(summer > 0.0, "midnight sun failed at {t}s: {summer}");
            assert!(winter < 0.0, "polar night failed at {t}s: {winter}");
        }
    }

    #[test]
    fn sun_rises_in_the_east_and_sets_in_the_west() {
        // Equinox on the equator: 06:00 local ⇒ due east, 18:00 ⇒ due west.
        let day = 81;
        let noon = solar_noon(day, 0.0);
        let dawn = sun_direction(&at(noon - 6.0 * 3600.0, day, 0.0, 0.0));
        let dusk = sun_direction(&at(noon + 6.0 * 3600.0, day, 0.0, 0.0));
        assert!(dawn.x > 0.99, "dawn not east: {dawn}");
        assert!(dusk.x < -0.99, "dusk not west: {dusk}");
        assert!(dawn.y.abs() < 0.02 && dusk.y.abs() < 0.02, "not at horizon");
    }

    #[test]
    fn northern_noon_sun_is_south() {
        // Mid-northern latitude at noon: the sun is due south (engine +Z).
        let day = 81;
        let t = solar_noon(day, 0.0);
        let d = sun_direction(&at(t, day, 45.0, 0.0));
        assert!(d.z > 0.7, "noon sun not south: {d}");
        assert!(d.x.abs() < 0.01, "noon sun not on the meridian: {d}");
        let az = azimuth_deg(d);
        assert!((az - 180.0).abs() < 1.0, "azimuth {az}");
    }

    #[test]
    fn southern_hemisphere_noon_sun_is_north() {
        let day = 81;
        let t = solar_noon(day, 151.2);
        let d = sun_direction(&at(t, day, -33.9, 151.2));
        assert!(d.z < -0.5, "southern noon sun not north: {d}");
        assert!(d.y > 0.8, "southern noon sun not high: {d}");
    }

    #[test]
    fn longitude_shifts_local_noon() {
        // 15° of longitude is exactly one hour of local time.
        let day = 200;
        let a = solar_noon(day, 0.0);
        let b = solar_noon(day, 15.0);
        assert!((a - b - 3600.0).abs() < 1.0, "{a} vs {b}");
    }

    #[test]
    fn direction_is_continuous_across_midnight() {
        // Stepping over the day boundary (with the day rolling, as `advance`
        // does) must not jump the sun.
        let lat = 51.5;
        let before = at(SECONDS_PER_DAY - 0.5, 200, lat, 0.0);
        let (s, d) = advance(before.seconds, before.day_of_year, 1.0, 1.0);
        let after = at(s, d, lat, 0.0);
        assert_eq!((s, d), (0.5, 201));
        let delta = sun_direction(&before).distance(sun_direction(&after));
        // One second of Earth rotation is 15 arc-seconds ≈ 7.3e-5 rad.
        assert!(delta < 1e-3, "midnight discontinuity {delta}");
        // …and the whole-day seam is continuous in declination too.
        let g_before = fractional_year(&before);
        let g_after = fractional_year(&after);
        assert!((g_before - g_after).abs() < 1e-4, "gamma seam");
    }

    #[test]
    fn direction_is_continuous_across_the_year() {
        let lat = 51.5;
        let before = at(SECONDS_PER_DAY - 0.5, 365, lat, 0.0);
        let (s, d) = advance(before.seconds, before.day_of_year, 1.0, 1.0);
        assert_eq!((s, d), (0.5, 1));
        let delta = sun_direction(&before).distance(sun_direction(&at(s, d, lat, 0.0)));
        assert!(delta < 1e-3, "new-year discontinuity {delta}");
    }

    #[test]
    fn advance_is_deterministic_and_bit_exact() {
        // The gate: two independent accumulations of the same schedule agree to
        // the bit, which is what makes the replay and PIE traces comparable.
        let run = || {
            let (mut s, mut d) = (0.0_f64, 1_u32);
            for _ in 0..10_000 {
                let step = advance(s, d, 60.0, 1.0 / 60.0);
                s = step.0;
                d = step.1;
            }
            (s, d)
        };
        let a = run();
        let b = run();
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1, b.1);
        // 10 000 steps × 1 sim-second = 10 000 s.
        assert!((a.0 - 10_000.0).abs() < 1e-6, "{}", a.0);
        assert_eq!(a.1, 1);
    }

    #[test]
    fn advance_wraps_days_and_freezes_at_rate_zero() {
        assert_eq!(advance(0.0, 1, 0.0, 1000.0), (0.0, 1));
        assert_eq!(advance(43_200.0, 5, 0.0, 1e9), (43_200.0, 5));
        // Two and a half days forward.
        let (s, d) = advance(0.0, 10, SECONDS_PER_DAY, 2.5);
        assert!((s - 43_200.0).abs() < 1e-6, "{s}");
        assert_eq!(d, 12);
        // Backwards over the year boundary.
        let (s, d) = advance(3600.0, 1, -SECONDS_PER_DAY, 1.0);
        assert!((s - 3600.0).abs() < 1e-6, "{s}");
        assert_eq!(d, 365);
    }

    #[test]
    fn advance_rejects_non_finite() {
        assert_eq!(advance(1234.0, 3, f64::NAN, 1.0), (1234.0, 3));
        assert_eq!(advance(1234.0, 3, 1.0, f64::INFINITY), (1234.0, 3));
        assert_eq!(advance(f64::NAN, 3, 1.0, 1.0), (0.0, 3));
    }

    #[test]
    fn wrap_and_roll_edges() {
        assert_eq!(wrap_seconds(-1.0), SECONDS_PER_DAY - 1.0);
        assert_eq!(wrap_seconds(SECONDS_PER_DAY), 0.0);
        assert_eq!(wrap_seconds(f64::NAN), 0.0);
        assert_eq!(roll_day(1, -1), 365);
        assert_eq!(roll_day(365, 1), 1);
        assert_eq!(roll_day(0, 0), 1);
        assert_eq!(roll_day(9_999, 0), 365);
    }

    #[test]
    fn directions_are_unit_length_everywhere() {
        for day in (1..=365).step_by(29) {
            for step in 0..24 {
                for &lat in &[-90.0_f64, -45.0, 0.0, 37.0, 90.0] {
                    let i = at(f64::from(step) * 3600.0, day, lat, -71.0);
                    let b = bodies(&i);
                    assert!((b.sun.length() - 1.0).abs() < 1e-9, "sun {b:?}");
                    assert!((b.moon.length() - 1.0).abs() < 1e-9, "moon {b:?}");
                    assert!((0.0..1.0).contains(&b.moon_phase));
                }
            }
        }
    }

    #[test]
    fn full_moon_is_opposite_the_sun() {
        // Find the day whose phase is closest to 0.5 and check the moon sits
        // opposite the sun (the defining property of the v1 model).
        let mut best = (1_u32, 1.0_f64);
        for day in 1..=365 {
            let p = moon_phase(&at(0.0, day, 0.0, 0.0));
            let e = (p - 0.5).abs();
            if e < best.1 {
                best = (day, e);
            }
        }
        let i = at(0.0, best.0, 20.0, 0.0);
        let dot = sun_direction(&i).dot(moon_direction(&i));
        assert!(dot < -0.99, "full moon dot {dot} on day {}", best.0);
    }

    #[test]
    fn new_moon_rides_with_the_sun() {
        let i = at(0.0, 1, 20.0, 0.0);
        assert!(moon_phase(&i) < 0.01);
        let dot = sun_direction(&i).dot(moon_direction(&i));
        assert!(dot > 0.98, "new moon dot {dot}");
    }

    #[test]
    fn default_reproduces_the_legacy_sun_constant() {
        // The compile-time sun the engine shipped from Phase 2 to Phase 16.
        let legacy = DVec3::new(0.45, 0.75, 0.3).normalize();
        let d = sun_direction(&SolarInput::default());
        let angle = d.dot(legacy).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(angle < 1.6, "default sun is {angle}° off legacy ({d})");
    }

    #[test]
    fn clock_formats_hh_mm_ss() {
        assert_eq!(format_clock(0.0), "00:00:00");
        assert_eq!(format_clock(36_000.0), "10:00:00");
        assert_eq!(format_clock(86_399.0), "23:59:59");
        assert_eq!(format_clock(-1.0), "23:59:59");
    }

    #[test]
    fn elevation_and_azimuth_round_trip() {
        for &(el, az) in &[
            (0.0_f64, 0.0_f64),
            (45.0, 90.0),
            (-30.0, 200.0),
            (80.0, 350.0),
        ] {
            let (e, a) = (el.to_radians(), az.to_radians());
            let d = DVec3::new(e.cos() * a.sin(), e.sin(), -e.cos() * a.cos());
            assert!((elevation_deg(d) - el).abs() < 1e-9, "el {el}");
            assert!((azimuth_deg(d) - az).abs() < 1e-9, "az {az}");
        }
    }
}
