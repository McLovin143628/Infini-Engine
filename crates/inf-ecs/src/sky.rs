//! The sky authority: resolving one [`TimeOfDay`] + [`SkyAtmosphere`] pair out of
//! a world, and advancing the clock in the fixed step (P17.1).
//!
//! # Why this lives in `inf-ecs`
//!
//! Two projectors build a `RenderScene` from an ECS world — the editor viewport
//! (`inf_viewport::host::rebuild_scene`) and the shipped player
//! (`inf_player::render::project_scene`) — and they walk the world in *different*
//! orders (document order vs `Guid` order). For a per-entity thing like a light
//! that is fine; for a **singleton** it is a divergence waiting to happen: "the
//! first `TimeOfDay` I meet" is a different entity on each side.
//!
//! So the *resolution rule* lives here, once, in Ring 0, where both sides reach
//! it: [`sky_authority`] picks the **lowest `Guid`**, which is a property of the
//! data rather than of the traversal. `inf-render` cannot host this (it does not
//! depend on `inf-ecs`), and `inf-ecs` cannot return renderer types (it does not
//! depend on `inf-render`), so this module returns plain data and each projector
//! does the ~15-line mapping into `RenderScene` — the same MIRROR arrangement
//! `project_light` already uses, but with the part that *can* silently diverge
//! moved out of the mirrors.
//!
//! # Determinism
//!
//! Everything here is a pure function of world contents (never of iteration
//! order), and [`advance_time_of_day`] uses only IEEE add/mul/floor via
//! [`inf_math::solar::advance`]. Two runs of the same simulation therefore
//! produce bit-identical clocks and bit-identical sun directions — which is what
//! the replay-determinism and PIE-vs-shipping gates compare.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bevy_ecs::prelude::{Entity, With};
use glam::DVec3;
use uuid::Uuid;

use crate::components::{Guid, SkyAtmosphere, TimeOfDay, WeatherParams, WeatherPreset};
use crate::world::EcsWorld;

/// Longest weather transition the engine will run, **seconds** (one hour).
///
/// Not a taste limit — an arithmetic one. `advance_weather` counts down in `f32`,
/// and once `ulp(remaining)/2 ≥ dt` the subtraction `remaining − dt` **rounds
/// back to `remaining`**: the countdown freezes, `k = dt / remaining` underflows
/// toward zero, and the blender is armed forever — writing the component every
/// step while never settling, which voids the "settled ⇒ the blender never writes
/// these fields again" contract the whole design rests on. At 60 Hz that begins
/// around 2¹⁹ s (≈ 6 days), with a band below it where the countdown still moves
/// but only by one ULP a step — tens of millions of steps to finish, i.e. frozen
/// in practice. A 240 Hz sim brings both down by a factor of four. An hour is
/// three orders of magnitude inside either, and is already longer than any
/// transition a level has a use for.
///
/// This is the **one** definition of the bound. `set_weather` clamps to it, and
/// `inf_editor_core::ipc::WeatherDto::to_component` clamps to it by *reference*
/// rather than by repeating `3600.0`, so the two doors into the field cannot
/// drift apart. A hand-edited sidecar bypasses both doors, which is what the
/// no-progress backstop in [`advance_weather`] is for.
pub const MAX_WEATHER_BLEND_S: f32 = 3_600.0;

/// Snow depth accumulated per second at **full** intensity and full snowiness,
/// metres per second — the P22 hook's calibration constant.
///
/// `1.4e-5 m/s` is ≈ 5 cm/hour, which is what a meteorologist calls heavy
/// snowfall. Stated here, once, so P22.1's deformation code does not have to
/// invent a rate and the number is reviewable against reality rather than
/// against taste.
pub const SNOW_ACCUMULATION_MAX_M_PER_S: f32 = 1.4e-5;

/// The weather actually in force, after the enable flag has chosen between the
/// weather block and the authored cloud/fog fields (P17.4).
///
/// This lives in Ring 0 for the same reason [`sky_authority`] and
/// [`ResolvedSky::cloud_time_s`] do: it is the kind of small derivation that two
/// byte-identical MIRROR bodies in the editor and the player would eventually
/// stop agreeing about. Both projectors call [`ResolvedSky::weather`] once and
/// copy fields out of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedWeather {
    /// Cloud coverage `[0, 1]` the renderer should use.
    pub cloud_coverage: f32,
    /// Cloud type `[0, 1]`.
    pub cloud_type: f32,
    /// Cloud/precipitation wind in world X, m/s.
    pub wind_x: f32,
    /// Cloud/precipitation wind in world Z, m/s.
    pub wind_z: f32,
    /// Height-fog extinction, m⁻¹.
    pub fog_density: f32,
    /// Precipitation intensity `[0, 1]`; `0` when weather is off.
    pub precipitation: f32,
    /// Precipitation phase `[0, 1]` — 0 rain, 1 snow.
    pub snowiness: f32,
}

/// The resolved sky: the authority's components plus the astronomy they imply.
///
/// Returned by [`resolve_sky`]; consumed by both scene projectors and by any
/// gate that wants to assert a sun direction without a GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedSky {
    /// The authority entity's stable id.
    pub guid: Uuid,
    /// The authority's clock.
    pub time_of_day: TimeOfDay,
    /// The authority's atmosphere parameters. A [`TimeOfDay`] without a
    /// [`SkyAtmosphere`] beside it resolves the component default (a sun that
    /// lights the scene with the engine's historic intensity).
    pub atmosphere: SkyAtmosphere,
    /// Unit direction **toward** the sun (engine axes: `+X` east, `+Y` up, `+Z`
    /// south).
    pub sun: DVec3,
    /// Unit direction **toward** the moon.
    pub moon: DVec3,
    /// Lunar phase, `[0, 1)` — `0` new, `0.5` full.
    pub moon_phase: f64,
}

impl ResolvedSky {
    /// Whether the sun is above the geometric horizon.
    #[inline]
    pub fn is_day(&self) -> bool {
        self.sun.y > 0.0
    }

    /// The body that currently lights the scene: `(direction, linear colour,
    /// intensity)` — the sun while it is up, the moon once it has set.
    ///
    /// Returns `None` when [`SkyAtmosphere::enabled`] is `false`: the clock still
    /// runs and the sky still tints, but the level authors its own lights.
    pub fn key_light(&self) -> Option<(DVec3, [f32; 3], f32)> {
        if !self.atmosphere.enabled {
            return None;
        }
        let a = &self.atmosphere;
        if self.is_day() {
            let c = a.sun_color;
            Some((self.sun, [c.r, c.g, c.b], a.sun_intensity))
        } else {
            let c = a.moon_color;
            Some((self.moon, [c.r, c.g, c.b], a.moon_intensity))
        }
    }

    /// How much the authored sky gradient is scaled at this time of day, `[0, 1]`.
    ///
    /// A smoothstep on the sun's elevation: **exactly `1.0` whenever the sun is
    /// more than ~9° above the horizon** (so any daytime scene draws the authored
    /// colours untouched, byte-for-byte), falling to
    /// `1 − night_darkening` once it is more than ~9° below.
    pub fn sky_dim(&self) -> f32 {
        const BAND: f64 = 0.15; // sin(≈8.6°)
        let t = ((self.sun.y + BAND) / (2.0 * BAND)).clamp(0.0, 1.0);
        let s = (t * t * (3.0 - 2.0 * t)) as f32;
        let floor = 1.0 - self.atmosphere.night_darkening.clamp(0.0, 1.0);
        floor + (1.0 - floor) * s
    }

    /// The clock reading the **cloud wind** drifts by, in seconds (P17.3).
    ///
    /// `(day_of_year − 1) · 86400 + seconds`, so the drift is continuous across
    /// midnight instead of snapping back — a level whose clock runs at rate 60
    /// would otherwise show the whole sky jump every 24 in-game minutes.
    ///
    /// This lives here, in Ring 0, rather than in either projector, for the same
    /// reason [`sky_authority`] does: it is the kind of one-line derivation that
    /// two byte-identical MIRROR bodies would eventually stop agreeing about.
    /// The wind is a function of the **document's** clock and of nothing else —
    /// no wall clock, no frame counter — which is what makes two runs at the same
    /// time of day render the same sky.
    #[inline]
    pub fn cloud_time_s(&self) -> f64 {
        let day = self.time_of_day.day_of_year.max(1) as f64 - 1.0;
        day * 86_400.0 + self.time_of_day.seconds
    }

    /// The clock and wind a **water body** responds to: `(level clock in seconds,
    /// (wind_x, wind_z) in m/s)`.
    ///
    /// One derivation, in Ring 0, for the same reason [`cloud_time_s`](Self::cloud_time_s)
    /// is here: both scene projectors need it, and two byte-identical MIRROR
    /// bodies deciding for themselves what a wave's clock is would eventually
    /// stop agreeing. A body that has opted out of the weather
    /// (`WaterBody::wind_from_weather == false`) still gets the clock from here —
    /// only the wind is its own.
    #[inline]
    pub fn water_environment(&self) -> (f64, (f64, f64)) {
        let w = self.weather();
        (self.cloud_time_s(), (w.wind_x as f64, w.wind_z as f64))
    }

    /// The weather in force (P17.4): the `weather_*` block when
    /// [`SkyAtmosphere::weather_enabled`], else the authored cloud/fog fields
    /// with no precipitation.
    ///
    /// The `false` branch reproduces the v13 projection **field for field**,
    /// which is what keeps every pre-P17.4 golden and every existing level
    /// byte-identical; the `true` branch is what makes the authored cloud
    /// coverage/type/wind and fog density *driven* rather than merely
    /// overwritten — they stay authorable and stay meaningful the moment weather
    /// is switched back off.
    pub fn weather(&self) -> ResolvedWeather {
        let a = &self.atmosphere;
        if a.weather_enabled {
            ResolvedWeather {
                cloud_coverage: a.weather_coverage,
                cloud_type: a.weather_cloud_type,
                wind_x: a.weather_wind_x,
                wind_z: a.weather_wind_z,
                fog_density: a.weather_fog_density,
                precipitation: a.weather_precipitation.clamp(0.0, 1.0),
                snowiness: a.weather_snowiness.clamp(0.0, 1.0),
            }
        } else {
            ResolvedWeather {
                cloud_coverage: a.cloud_coverage,
                cloud_type: a.cloud_type,
                wind_x: a.cloud_wind_x,
                wind_z: a.cloud_wind_z,
                fog_density: a.fog_density,
                precipitation: 0.0,
                snowiness: 0.0,
            }
        }
    }

    /// **The P22 hook.** Snow depth accumulating on upward-facing surfaces right
    /// now, in **metres per second** (SI).
    ///
    /// `intensity × snowiness × `[`SNOW_ACCUMULATION_MAX_M_PER_S`], so rain
    /// accumulates nothing, a Snow preset at full blast accumulates ≈ 5 cm/hour,
    /// and a rain→snow transition ramps continuously through the blend.
    ///
    /// **P22.1 (deformation & destruction) consumes this**; nothing renders
    /// accumulation yet. It is defined here, now, because the *rate* is a
    /// property of the resolved sky and P22 must not re-derive it from the
    /// component (which would put two definitions of "how snowy is it" in the
    /// engine). Zero whenever weather is disabled, so the hook is inert for
    /// every level that has not opted in.
    #[inline]
    pub fn snow_accumulation_rate(&self) -> f32 {
        let w = self.weather();
        w.precipitation * w.snowiness * SNOW_ACCUMULATION_MAX_M_PER_S
    }

    /// The sky gradient colours to hand the renderer: the authored zenith /
    /// horizon / ground scaled by [`sky_dim`](Self::sky_dim).
    pub fn sky_gradient(&self) -> [[f32; 3]; 3] {
        let d = self.sky_dim();
        let a = &self.atmosphere;
        [
            [a.zenith.r * d, a.zenith.g * d, a.zenith.b * d],
            [a.horizon.r * d, a.horizon.g * d, a.horizon.b * d],
            [a.ground.r * d, a.ground.g * d, a.ground.b * d],
        ]
    }
}

/// The entity that owns the level's clock: the **lowest `(Guid, Entity)`**
/// carrying a [`TimeOfDay`].
///
/// Lowest-`Guid` (rather than "first found") is what makes the two projectors
/// agree: `Guid` order is a property of the data, so the editor viewport and the
/// shipped player resolve the same entity even though one walks document order
/// and the other walks `Guid` order. A level with two clocks is an authoring
/// mistake, not a crash — it deterministically picks one.
///
/// `Entity` breaks a tie. Two entities *should* never share a `Guid`, but a
/// merge-mangled `.inf_lvl` or a bad paste can produce it, and a strict `<` on
/// the `Guid` alone would then keep whichever the traversal happened to reach
/// first — i.e. exactly the iteration-order dependence this function exists to
/// remove. `Entity` is a property of the world rather than of the walk, so each
/// side stays internally deterministic and stable frame to frame. (The two sides
/// may still disagree about *which* duplicate wins; nothing can fix that, because
/// the level no longer names one — the tie-break's job is to stop a single side
/// from flickering.)
///
/// ## Cost
///
/// `O(clocks)`, not `O(entities)`: `try_query_filtered` restricts the walk to
/// archetypes that actually contain a [`TimeOfDay`] — normally one entity, often
/// zero. That matters because this is on the hot path of the four `sky.*`
/// Blueprint seams, which a script may call several times per tick in a world
/// with tens of thousands of entities. `None` from `try_query_filtered` means the
/// component has never been inserted in this world at all, which is the `O(1)`
/// answer for every pre-P17.1 level.
pub fn sky_authority(world: &EcsWorld) -> Option<Entity> {
    let w = world.world();
    let mut q = w.try_query_filtered::<(Entity, &Guid), With<TimeOfDay>>()?;
    let mut best: Option<(Uuid, Entity)> = None;
    for (entity, guid) in q.iter(w) {
        let key = (guid.0, entity);
        if best.is_none_or(|b| key < b) {
            best = Some(key);
        }
    }
    best.map(|(_, e)| e)
}

/// One-shot latch for the orphaned-[`SkyAtmosphere`] diagnostic below.
static ORPHAN_ATMOSPHERE_WARNED: AtomicBool = AtomicBool::new(false);
/// How many times that diagnostic has actually fired (test observability).
static ORPHAN_ATMOSPHERE_WARNINGS: AtomicUsize = AtomicUsize::new(0);

/// How many orphaned-[`SkyAtmosphere`] warnings have been emitted this process.
/// Test/diagnostic hook; the warning is latched, so this saturates at 1.
#[doc(hidden)]
pub fn orphan_atmosphere_warnings() -> usize {
    ORPHAN_ATMOSPHERE_WARNINGS.load(Ordering::Relaxed)
}

/// Clear the one-shot latch. Test-only: production code must never re-arm a
/// diagnostic that exists precisely so it cannot spam a per-frame log.
#[doc(hidden)]
pub fn reset_orphan_atmosphere_warning() {
    ORPHAN_ATMOSPHERE_WARNED.store(false, Ordering::Relaxed);
    ORPHAN_ATMOSPHERE_WARNINGS.store(0, Ordering::Relaxed);
}

/// Warn (once per process) about a [`SkyAtmosphere`] that will never be read.
///
/// The atmosphere is resolved **from the authority entity**, so a `SkyAtmosphere`
/// with no [`TimeOfDay`] beside it is silently inert — the level looks like it has
/// been configured and renders as if it had not. That is a nasty failure mode
/// precisely because nothing is broken: no panic, no missing asset, just a sun
/// that ignores everything you typed. Both shapes are covered:
///
/// * **no clock anywhere** — the atmosphere does nothing at all;
/// * **a clock exists, but the atmosphere sits on a different entity** — the
///   authority's own (possibly defaulted) atmosphere wins and this one is ignored.
///
/// Latched, because `resolve_sky` runs every frame in two projectors. Cheap after
/// the first hit (one relaxed atomic load), and cheap before it too: the query is
/// archetype-scoped exactly like [`sky_authority`].
fn warn_orphan_atmosphere(world: &EcsWorld, authority: Option<Entity>) {
    if ORPHAN_ATMOSPHERE_WARNED.load(Ordering::Relaxed) {
        return;
    }
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<Entity, With<SkyAtmosphere>>() else {
        return;
    };
    let Some(orphan) = q.iter(w).find(|e| Some(*e) != authority) else {
        return;
    };
    if ORPHAN_ATMOSPHERE_WARNED.swap(true, Ordering::Relaxed) {
        return; // another thread won the race; it logged.
    }
    ORPHAN_ATMOSPHERE_WARNINGS.fetch_add(1, Ordering::Relaxed);
    let guid = w.get::<Guid>(orphan).map(|g| g.0);
    let name = w
        .get::<crate::components::Name>(orphan)
        .map(|n| n.as_str().to_string())
        .unwrap_or_default();
    if authority.is_some() {
        tracing::warn!(
            entity = ?guid,
            name = %name,
            "SkyAtmosphere is ignored: it sits on an entity that is not the level's \
             sky authority (the lowest-Guid entity carrying a TimeOfDay). Move it onto \
             that entity, or remove the level's other TimeOfDay components."
        );
    } else {
        tracing::warn!(
            entity = ?guid,
            name = %name,
            "SkyAtmosphere has no effect: this level has no TimeOfDay, so there is no \
             sun for it to describe and the renderer falls back to its fixed default \
             sun. Add a Time of Day component to the same entity (World Settings → \
             Time of Day does it for you)."
        );
    }
}

/// Resolve the level's sky, or `None` when no entity carries a [`TimeOfDay`].
///
/// `None` is the pre-P17.1 world: the projectors leave the renderer's default sun
/// (the retired `SUN_DIR` value) in place, so every scene that has not opted in
/// renders exactly the pixels it always did.
pub fn resolve_sky(world: &EcsWorld) -> Option<ResolvedSky> {
    let entity = sky_authority(world);
    // Diagnose an atmosphere that nothing will ever read — including in the
    // no-authority case, which is why this runs before the `?`.
    warn_orphan_atmosphere(world, entity);
    let entity = entity?;
    let w = world.world();
    let time_of_day = *w.get::<TimeOfDay>(entity)?;
    let atmosphere = w.get::<SkyAtmosphere>(entity).copied().unwrap_or_default();
    let guid = w.get::<Guid>(entity).map(|g| g.0)?;
    let bodies = inf_math::solar::bodies(&time_of_day.solar_input());
    Some(ResolvedSky {
        guid,
        time_of_day,
        atmosphere,
        sun: bodies.sun,
        moon: bodies.moon,
        moon_phase: bodies.moon_phase,
    })
}

/// Advance the level's clock by `dt` simulated seconds — the fixed-step system.
///
/// Only the **authority** advances (a stray second `TimeOfDay` is left alone, so
/// which entity moves never depends on iteration order), and only at a non-zero
/// `rate`. Returns the new time when it changed, so a caller can trace it.
///
/// Called from both fixed steps (`SimSession::fixed_step` and
/// `RuntimeSim::fixed_step`) in the same slot; never from the editor's authoring
/// path, so an idle editor never moves the sun or dirties the document.
pub fn advance_time_of_day(world: &mut EcsWorld, dt: f64) -> Option<TimeOfDay> {
    let entity = sky_authority(world)?;
    let mut tod = *world.world().get::<TimeOfDay>(entity)?;
    if tod.rate == 0.0 || !tod.rate.is_finite() {
        return None;
    }
    tod.advance(dt);
    *world.world_mut().get_mut::<TimeOfDay>(entity)? = tod;
    Some(tod)
}

/// Advance the level's **weather blend** by `dt` simulated seconds — the second
/// fixed-step system (P17.4), run in the same slot as
/// [`advance_time_of_day`].
///
/// Only the **authority**'s atmosphere blends, and only while a transition is in
/// flight (`weather_blend_remaining > 0`) on an enabled weather block. Returns
/// the new live parameters when they moved, so a caller can trace them.
///
/// # The blend, and why it needs no `from` snapshot
///
/// Each step closes `k = dt / remaining` of the *remaining* gap to the target:
///
/// ```text
/// p ← p + (target − p) · k        remaining ← max(remaining − dt, 0)
/// ```
///
/// After `n` steps of a fixed `dt` out of a total `T` the surviving gap is
/// `∏(1 − dt/(T − i·dt)) = (T − n·dt)/T`, so in **exact arithmetic** `p` is the
/// linear interpolation `p₀ + (target − p₀)·(n·dt/T)`. In `f32` it is that to
/// within a measured **17 ULP over a 240-step (4 s at 60 Hz) blend** — the
/// accumulation of a `1/60` that is not representable — which is why the gate
/// asserts linearity against an envelope rather than an equality. The
/// **endpoint** is exact regardless, because the last step assigns (below).
///
/// That is what lets the component store only the live values and the time left,
/// with no `from` copy to keep consistent, and it is pure IEEE add/mul/div, so
/// two runs (and PIE vs shipping) are bit-identical.
///
/// A non-finite or non-positive `dt` is ignored rather than poisoning the state.
pub fn advance_weather(world: &mut EcsWorld, dt: f64) -> Option<WeatherParams> {
    if !dt.is_finite() || dt <= 0.0 {
        return None;
    }
    let entity = sky_authority(world)?;
    let mut a = *world.world().get::<SkyAtmosphere>(entity)?;
    // `is_sign_positive` would accept a NaN remaining; `> 0.0` is deliberately
    // the *positive* test, so a NaN that reached the field (it cannot through the
    // DTO clamp, but a hand-edited sidecar could) settles the blend rather than
    // spinning it forever.
    let in_flight = a.weather_blend_remaining > 0.0;
    if !a.weather_enabled || !in_flight {
        return None;
    }
    let dt = dt as f32;
    let target = a.weather_target.params();
    // **The termination backstop.** `set_weather` and the DTO both clamp to
    // [`MAX_WEATHER_BLEND_S`], but a hand-edited `.inf_lvl.toml` (or a `.inf_lvl`
    // written by a future tool) can put any `f32` in this field, and the failure
    // it produces is the one this whole design cannot tolerate: a countdown that
    // never reaches zero leaves this function writing the component every step
    // forever, which voids "settled ⇒ the blender never writes these fields
    // again" — the contract the sequencer and the Details grid rest on.
    //
    // Two conditions, because there are two shapes of it. `no_progress` is the
    // exact arithmetic test — above ~2¹⁹ s at 60 Hz, `remaining − dt` rounds back
    // to `remaining` — and it is written as arithmetic rather than as a magnitude
    // so it holds for any `dt`, including a subnormal one. `over_ceiling` catches
    // the band just below that, where the subtraction still moves but only by one
    // ULP a step, so the blend would need tens of millions of steps: progress in
    // principle, frozen in practice, and out of contract either way.
    let remaining = a.weather_blend_remaining;
    let over_ceiling = remaining > MAX_WEATHER_BLEND_S;
    let no_progress = remaining - dt >= remaining;
    let settle = over_ceiling || remaining <= dt || no_progress;
    let live = if settle {
        // The last step **assigns** the target rather than closing the final
        // `k = 1` of the gap. Arithmetically those are the same thing; in f32
        // they are not, because `remaining` has by then accumulated ~240 lossy
        // subtractions of a `1/60` that is not representable, so the computed
        // `k` lands short and the state settles a measured ~17 ULP away from the
        // preset it is supposed to BE. Assigning makes "a settled blend equals
        // its target" a bit-identity instead of a tolerance.
        a.weather_blend_remaining = 0.0;
        target
    } else {
        let mut live = a.weather_params();
        let k = dt / a.weather_blend_remaining;
        let step = |p: &mut f32, t: f32| *p += (t - *p) * k;
        step(&mut live.coverage, target.coverage);
        step(&mut live.cloud_type, target.cloud_type);
        step(&mut live.wind_x, target.wind_x);
        step(&mut live.wind_z, target.wind_z);
        step(&mut live.fog_density, target.fog_density);
        step(&mut live.precipitation, target.precipitation);
        step(&mut live.snowiness, target.snowiness);
        a.weather_blend_remaining -= dt;
        live
    };
    a.set_weather_params(live);
    *world.world_mut().get_mut::<SkyAtmosphere>(entity)? = a;
    Some(live)
}

// ── the `sky.*` Blueprint host surface (P17.1) ──────────────────────────────
//
// Four one-line seams shared verbatim by the editor's `SimHost` and the shipped
// player's `RuntimeHost`, so preview == shipped by construction rather than by
// two hand-written match arms agreeing. Units per architecture rule 6: seconds
// for the clock, a dimensionless multiplier for the rate.

/// The clock and wind every water body in `world` responds to (P20.1), resolved
/// **once per projection**.
///
/// A world with no sky authority answers `(0, (0, 0))` — a defined value rather
/// than an error, matching how `terrain.height_at` reads a flat plane out of a
/// terrain-less world. That is also what makes a water body render identically
/// in a level that has never heard of a clock.
pub fn water_environment(world: &EcsWorld) -> (f64, (f64, f64)) {
    resolve_sky(world)
        .map(|s| s.water_environment())
        .unwrap_or((0.0, (0.0, 0.0)))
}

/// The level clock's current time, UTC seconds since midnight. `0` when the
/// level has no clock — a defined answer rather than an error, matching how
/// `terrain.height_at` reads a flat plane out of a terrain-less world.
pub fn time_of_day_seconds(world: &EcsWorld) -> f64 {
    sky_authority(world)
        .and_then(|e| world.world().get::<TimeOfDay>(e))
        .map(|t| t.seconds)
        .unwrap_or(0.0)
}

/// The level clock's rate (simulated seconds per simulated second); `0` when the
/// level has no clock, which is also what "frozen" means.
pub fn time_of_day_rate(world: &EcsWorld) -> f64 {
    sky_authority(world)
        .and_then(|e| world.world().get::<TimeOfDay>(e))
        .map(|t| t.rate)
        .unwrap_or(0.0)
}

/// Set the level clock, wrapping into `[0, 86400)`. Non-finite input is ignored.
/// Returns whether a clock was there to set.
pub fn set_time_of_day_seconds(world: &mut EcsWorld, seconds: f64) -> bool {
    if !seconds.is_finite() {
        return false;
    }
    let Some(entity) = sky_authority(world) else {
        return false;
    };
    match world.world_mut().get_mut::<TimeOfDay>(entity) {
        Some(mut t) => {
            t.seconds = inf_math::solar::wrap_seconds(seconds);
            true
        }
        None => false,
    }
}

/// Set the level clock's rate. Non-finite input is ignored. Returns whether a
/// clock was there to set.
pub fn set_time_of_day_rate(world: &mut EcsWorld, rate: f64) -> bool {
    if !rate.is_finite() {
        return false;
    }
    let Some(entity) = sky_authority(world) else {
        return false;
    };
    match world.world_mut().get_mut::<TimeOfDay>(entity) {
        Some(mut t) => {
            t.rate = rate;
            true
        }
        None => false,
    }
}

// ── the weather half of the `sky.*` host surface (P17.4) ───────────────────
//
// Same doctrine as the four clock seams above: one-line functions over the
// authority, shared verbatim by the editor's `SimHost` and the shipped player's
// `RuntimeHost`, so preview == shipped by construction. A clock-less (or
// atmosphere-less) world answers a defined value rather than an error, matching
// `terrain.height_at`'s precedent.

/// The atmosphere of the sky authority, or `None` when there is no authority.
fn authority_atmosphere(world: &EcsWorld) -> Option<SkyAtmosphere> {
    sky_authority(world).and_then(|e| world.world().get::<SkyAtmosphere>(e).copied())
}

/// Begin a transition to `preset`, taking `blend_seconds` to get there.
///
/// `blend_seconds` is read the literal way, because that is the only reading a
/// Blueprint author can predict:
///
/// * **`0`** — snap. The target's values are written immediately and nothing is
///   left in flight. An unwired Float pin lowers to `0.0`, so a
///   `sky.set_weather` node with only its preset wired changes the weather *now*,
///   which is exactly what it looks like it will do. It is also what the
///   editor's preset buttons need, since an idle editor runs no fixed step.
/// * **`> 0`** — blend over that many simulated seconds, **clamped to
///   [`MAX_WEATHER_BLEND_S`]**. That clamp is arithmetic rather than taste: past
///   it the `f32` countdown stops making progress and the blend never settles
///   (see the constant's docs).
/// * **negative or non-finite** — fall back to the level's authored
///   [`SkyAtmosphere::weather_blend_seconds`]. The escape hatch, not the default,
///   because "I passed a number and got a different one" is worse than
///   "I passed nothing and got nothing".
///
/// Setting the preset **also enables the weather block** — a script naming a
/// preset plainly means for it to take effect, and doing nothing because a
/// checkbox was clear is the worst of the available behaviours.
///
/// Returns whether a sky authority was there to set.
pub fn set_weather(world: &mut EcsWorld, preset: WeatherPreset, blend_seconds: f64) -> bool {
    let Some(entity) = sky_authority(world) else {
        return false;
    };
    let Some(mut a) = world.world_mut().get_mut::<SkyAtmosphere>(entity) else {
        return false;
    };
    let requested = if blend_seconds.is_finite() && blend_seconds >= 0.0 {
        blend_seconds as f32
    } else {
        a.weather_blend_seconds.max(0.0)
    }
    // Both branches clamp: the authored default reaches this field through the
    // DTO (which clamps to the same constant) but also through Details and a
    // sidecar, which do not.
    .clamp(0.0, MAX_WEATHER_BLEND_S);
    a.weather_enabled = true;
    a.weather_target = preset;
    if requested > 0.0 {
        a.weather_blend_remaining = requested;
    } else {
        a.set_weather_params(preset.params());
        a.weather_blend_remaining = 0.0;
    }
    true
}

/// The **target** preset's identifier (`"clear"`, `"storm"`, …). `"clear"` when
/// the level has no sky authority — the same defined-answer convention the clock
/// getters use, and the value a default component carries anyway.
pub fn weather_preset_name(world: &EcsWorld) -> &'static str {
    authority_atmosphere(world)
        .map(|a| a.weather_target.as_str())
        .unwrap_or(WeatherPreset::Clear.as_str())
}

/// Live precipitation intensity `[0, 1]`; `0` with no authority or with weather
/// disabled. Reads the **resolved** weather, so a Blueprint sees exactly what
/// the renderer draws.
pub fn weather_precipitation(world: &EcsWorld) -> f64 {
    resolve_sky(world)
        .map(|s| s.weather().precipitation as f64)
        .unwrap_or(0.0)
}

/// Live wind **speed** in m/s — the magnitude of the resolved wind vector, which
/// is what a gameplay script (sway, sailing, particle drag) actually wants. The
/// direction is available through the components; a second pure node for it
/// would be the multi-output fan-out the single-purpose rule exists to avoid.
pub fn weather_wind_speed(world: &EcsWorld) -> f64 {
    resolve_sky(world)
        .map(|s| {
            let w = s.weather();
            ((w.wind_x as f64) * (w.wind_x as f64) + (w.wind_z as f64) * (w.wind_z as f64)).sqrt()
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with(clocks: &[(Uuid, TimeOfDay)]) -> EcsWorld {
        let mut w = EcsWorld::new();
        for (guid, tod) in clocks {
            let e = w.spawn_with_guid(*guid, "Sky", None);
            w.world_mut().entity_mut(e).insert(*tod);
        }
        w
    }

    fn uuid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    #[test]
    fn empty_world_has_no_sky() {
        let w = EcsWorld::new();
        assert!(sky_authority(&w).is_none());
        assert!(resolve_sky(&w).is_none());
    }

    #[test]
    fn lowest_guid_wins_regardless_of_spawn_order() {
        let a = uuid(0x11);
        let b = uuid(0x22);
        let low = TimeOfDay {
            seconds: 1.0,
            ..TimeOfDay::default()
        };
        let high = TimeOfDay {
            seconds: 2.0,
            ..TimeOfDay::default()
        };
        // Spawned low-then-high and high-then-low must resolve the SAME entity —
        // this is the property that keeps the two projectors in agreement.
        let w1 = world_with(&[(a, low), (b, high)]);
        let w2 = world_with(&[(b, high), (a, low)]);
        assert_eq!(resolve_sky(&w1).unwrap().guid, a);
        assert_eq!(resolve_sky(&w2).unwrap().guid, a);
        assert_eq!(resolve_sky(&w1).unwrap().time_of_day.seconds, 1.0);
    }

    #[test]
    fn missing_atmosphere_resolves_the_default() {
        let w = world_with(&[(uuid(1), TimeOfDay::default())]);
        let sky = resolve_sky(&w).unwrap();
        assert_eq!(sky.atmosphere, SkyAtmosphere::default());
        assert!(sky.is_day(), "the default clock is mid-morning");
        let (dir, color, intensity) = sky.key_light().unwrap();
        assert_eq!(dir, sky.sun);
        assert_eq!(intensity, 3.0);
        assert_eq!(color, [1.0, 0.98, 0.95]);
    }

    #[test]
    fn night_switches_the_key_light_to_the_moon() {
        let midnight = TimeOfDay {
            seconds: 0.0,
            ..TimeOfDay::default()
        };
        let w = world_with(&[(uuid(1), midnight)]);
        let sky = resolve_sky(&w).unwrap();
        assert!(!sky.is_day(), "00:00 UTC at longitude 0 is night");
        let (dir, _, intensity) = sky.key_light().unwrap();
        assert_eq!(dir, sky.moon);
        assert_eq!(intensity, 0.15);
    }

    #[test]
    fn disabled_atmosphere_projects_no_key_light() {
        let mut w = world_with(&[(uuid(1), TimeOfDay::default())]);
        let e = sky_authority(&w).unwrap();
        w.world_mut().entity_mut(e).insert(SkyAtmosphere {
            enabled: false,
            ..SkyAtmosphere::default()
        });
        assert!(resolve_sky(&w).unwrap().key_light().is_none());
    }

    #[test]
    fn daytime_sky_gradient_is_the_authored_colours_untouched() {
        // The byte-stability claim: with the sun up, `sky_dim` is exactly 1.0 and
        // the gradient is the authored value bit-for-bit.
        let w = world_with(&[(uuid(1), TimeOfDay::default())]);
        let sky = resolve_sky(&w).unwrap();
        assert_eq!(sky.sky_dim(), 1.0);
        assert_eq!(
            sky.sky_gradient(),
            [
                [0.012, 0.021, 0.038],
                [0.055, 0.081, 0.120],
                [0.009, 0.011, 0.015]
            ]
        );
    }

    #[test]
    fn night_sky_gradient_darkens_to_the_floor() {
        let midnight = TimeOfDay {
            seconds: 0.0,
            ..TimeOfDay::default()
        };
        let w = world_with(&[(uuid(1), midnight)]);
        let sky = resolve_sky(&w).unwrap();
        // Deep night: the smoothstep has bottomed out at 1 − night_darkening.
        assert!((sky.sky_dim() - 0.15).abs() < 1e-6, "{}", sky.sky_dim());
        assert!(sky.sky_gradient()[0][2] < 0.038);
    }

    #[test]
    fn advance_moves_only_the_authority_and_only_at_a_rate() {
        let running = TimeOfDay {
            seconds: 100.0,
            rate: 60.0,
            ..TimeOfDay::default()
        };
        let frozen = TimeOfDay {
            seconds: 500.0,
            ..TimeOfDay::default()
        };
        let mut w = world_with(&[(uuid(1), running), (uuid(2), frozen)]);
        let out = advance_time_of_day(&mut w, 1.0).unwrap();
        assert_eq!(out.seconds, 160.0);
        assert_eq!(resolve_sky(&w).unwrap().time_of_day.seconds, 160.0);
        // The non-authority clock is untouched.
        let other = w.entity_of(uuid(2)).unwrap();
        assert_eq!(w.world().get::<TimeOfDay>(other).unwrap().seconds, 500.0);
        // A frozen authority reports no change at all.
        let mut w2 = world_with(&[(uuid(1), frozen)]);
        assert!(advance_time_of_day(&mut w2, 10.0).is_none());
    }

    #[test]
    fn advance_is_bit_identical_across_runs() {
        // The gate this exists for: two independent 600-step runs must agree to
        // the bit, in both the clock and the resulting sun direction.
        let run = || {
            let mut w = world_with(&[(
                uuid(7),
                TimeOfDay {
                    seconds: 0.0,
                    rate: 600.0,
                    ..TimeOfDay::default()
                },
            )]);
            for _ in 0..600 {
                advance_time_of_day(&mut w, 1.0 / 60.0);
            }
            let sky = resolve_sky(&w).unwrap();
            (sky.time_of_day.seconds.to_bits(), sky.sun)
        };
        let (a_bits, a_sun) = run();
        let (b_bits, b_sun) = run();
        assert_eq!(a_bits, b_bits);
        assert_eq!(a_sun.x.to_bits(), b_sun.x.to_bits());
        assert_eq!(a_sun.y.to_bits(), b_sun.y.to_bits());
        assert_eq!(a_sun.z.to_bits(), b_sun.z.to_bits());
    }

    /// The P20.1 water seam: one Ring-0 derivation of the clock and wind a wave
    /// responds to, so two MIRROR projectors cannot disagree about it.
    #[test]
    fn water_environment_is_the_clock_and_the_weather_wind() {
        // No authority at all: a defined answer, never an error.
        assert_eq!(water_environment(&EcsWorld::new()), (0.0, (0.0, 0.0)));

        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        });
        assert!(set_weather(&mut w, WeatherPreset::Storm, 0.0));
        let (t, (wx, wz)) = water_environment(&w);
        let storm = WeatherPreset::Storm.params();
        assert_eq!(wx, storm.wind_x as f64);
        assert_eq!(wz, storm.wind_z as f64);
        // The clock is the DOCUMENT's, continuous across midnight — the same one
        // the clouds drift on, not a second definition of "now".
        let sky = resolve_sky(&w).unwrap();
        assert_eq!(t, sky.cloud_time_s());

        // Weather OFF falls back to the authored cloud wind, which is what keeps
        // a pre-P17.4 level's sea responding to something rather than nothing.
        let calm = weather_world(SkyAtmosphere {
            weather_enabled: false,
            cloud_wind_x: 3.5,
            cloud_wind_z: -1.25,
            ..SkyAtmosphere::default()
        });
        assert_eq!(water_environment(&calm).1, (3.5, -1.25));
    }

    #[test]
    fn blueprint_host_seams_read_and_write_the_authority() {
        let mut w = world_with(&[(uuid(1), TimeOfDay::default())]);
        assert_eq!(time_of_day_seconds(&w), 36_000.0);
        assert_eq!(time_of_day_rate(&w), 0.0);

        assert!(set_time_of_day_seconds(&mut w, 90_000.0));
        assert_eq!(time_of_day_seconds(&w), 3_600.0, "the setter wraps the day");
        assert!(set_time_of_day_seconds(&mut w, -1.0));
        assert_eq!(time_of_day_seconds(&w), 86_399.0);
        assert!(set_time_of_day_rate(&mut w, 120.0));
        assert_eq!(time_of_day_rate(&w), 120.0);

        // Non-finite input is ignored rather than poisoning a saved level.
        assert!(!set_time_of_day_seconds(&mut w, f64::NAN));
        assert_eq!(time_of_day_seconds(&w), 86_399.0);
        assert!(!set_time_of_day_rate(&mut w, f64::INFINITY));
        assert_eq!(time_of_day_rate(&w), 120.0);

        // A clock-less world answers 0 and refuses the writes — a defined
        // answer, never an error (the `terrain.height_at` precedent).
        let mut empty = EcsWorld::new();
        assert_eq!(time_of_day_seconds(&empty), 0.0);
        assert_eq!(time_of_day_rate(&empty), 0.0);
        assert!(!set_time_of_day_seconds(&mut empty, 1.0));
        assert!(!set_time_of_day_rate(&mut empty, 1.0));
    }

    /// Duplicate `Guid`s are a corrupt level, not a crash — but the answer must
    /// still be a property of the *world*, not of the traversal, or one side could
    /// flicker frame to frame. Spawning the pair in either order must resolve the
    /// same entity.
    #[test]
    fn equal_guids_break_the_tie_deterministically() {
        let dup = uuid(0x5150);
        let a = TimeOfDay {
            seconds: 1.0,
            ..TimeOfDay::default()
        };
        let b = TimeOfDay {
            seconds: 2.0,
            ..TimeOfDay::default()
        };
        let pick = |first: TimeOfDay, second: TimeOfDay| {
            let mut w = EcsWorld::new();
            // `spawn_with_guid` indexes by Guid, so the second insert overwrites the
            // index entry — exactly the corrupt shape a bad merge produces.
            let e1 = w.spawn_with_guid(dup, "Sky A", None);
            w.world_mut().entity_mut(e1).insert(first);
            let e2 = w.spawn_with_guid(dup, "Sky B", None);
            w.world_mut().entity_mut(e2).insert(second);
            let chosen = sky_authority(&w).expect("an authority is still resolved");
            // The documented rule, recomputed here independently of the traversal:
            // the minimum of `(Guid, Entity)`.
            let expected = std::cmp::min((dup, e1), (dup, e2)).1;
            assert_eq!(chosen, expected, "the tie must break on (Guid, Entity)");
            // Repeated calls agree — no flicker frame to frame.
            assert_eq!(sky_authority(&w), Some(chosen));
            (
                chosen == e1,
                w.world().get::<TimeOfDay>(chosen).unwrap().seconds,
            )
        };
        // The SAME position wins for either spawn order, so the answer is a
        // property of the world rather than of the insert sequence …
        let (a_first_won, secs_a) = pick(a, b);
        let (b_first_won, secs_b) = pick(b, a);
        assert_eq!(
            a_first_won, b_first_won,
            "the tie-break must not depend on insert order"
        );
        // … and because it is positional, the payload follows the position.
        assert_eq!(secs_a, if a_first_won { 1.0 } else { 2.0 });
        assert_eq!(secs_b, if b_first_won { 2.0 } else { 1.0 });
    }

    /// GATE for the `sky.*` host seams: resolution is archetype-scoped, so a world
    /// with tens of thousands of entities costs the same as a tiny one.
    ///
    /// Measured **relatively** (min-of-N on both sizes) rather than against an
    /// absolute millisecond budget: a shared CI runner's absolute timings are
    /// noise, but an `O(entities)` scan would make the big world ~80× the small
    /// one here, which no amount of noise reaches from ~1×.
    #[test]
    fn authority_lookup_does_not_scale_with_world_size() {
        fn world_of(n: usize) -> EcsWorld {
            let mut w = EcsWorld::new();
            let clock = w.spawn_with_guid(uuid(1), "Sky", None);
            w.world_mut().entity_mut(clock).insert(TimeOfDay::default());
            for i in 0..n {
                w.spawn_with_guid(uuid(1000 + i as u128), "Prop", None);
            }
            w
        }
        fn min_lookup_nanos(w: &EcsWorld, reps: usize) -> u128 {
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        std::hint::black_box(sky_authority(std::hint::black_box(w)));
                    }
                    t.elapsed().as_nanos().max(1)
                })
                .min()
                .unwrap()
        }
        const SMALL: usize = 100;
        const BIG: usize = 8_000;
        const REPS: usize = 200;

        let small = world_of(SMALL);
        let big = world_of(BIG);
        // Correctness first — a fast wrong answer is not the point.
        assert_eq!(sky_authority(&big), sky_authority(&big));
        assert!(resolve_sky(&big).is_some());

        let ratio = min_lookup_nanos(&big, REPS) as f64 / min_lookup_nanos(&small, REPS) as f64;
        assert!(
            ratio < 20.0,
            "authority lookup scales with world size (×{ratio:.1} for {BIG} vs {SMALL} \
             entities) — the archetype-scoped query regressed to a full scan"
        );
    }

    /// A clock-less world answers `None` without touching entities at all: the
    /// component was never inserted, so the query state cannot even be built.
    /// This is the `O(1)` path every pre-P17.1 level takes, every frame.
    #[test]
    fn a_world_that_never_saw_a_clock_short_circuits() {
        let mut w = EcsWorld::new();
        for i in 0..500 {
            w.spawn_with_guid(uuid(i), "Prop", None);
        }
        assert!(sky_authority(&w).is_none());
        assert!(resolve_sky(&w).is_none());
    }

    /// An atmosphere nothing will ever read must say so — once — rather than
    /// leaving an author staring at a configured-looking level that renders under
    /// the fixed default sun.
    #[test]
    fn an_orphaned_atmosphere_warns_exactly_once() {
        reset_orphan_atmosphere_warning();

        // Shape 1: an atmosphere with no clock anywhere in the level.
        let mut w = EcsWorld::new();
        let e = w.spawn_with_guid(uuid(1), "Sky", None);
        w.world_mut().entity_mut(e).insert(SkyAtmosphere::default());
        assert!(resolve_sky(&w).is_none());
        assert_eq!(orphan_atmosphere_warnings(), 1);
        // Latched: the projectors call this every frame.
        for _ in 0..10 {
            resolve_sky(&w);
        }
        assert_eq!(
            orphan_atmosphere_warnings(),
            1,
            "the warning must be one-shot"
        );

        // Shape 2: a clock exists, but the atmosphere sits on another entity.
        reset_orphan_atmosphere_warning();
        let mut w = EcsWorld::new();
        let clock = w.spawn_with_guid(uuid(1), "Sky", None);
        w.world_mut().entity_mut(clock).insert(TimeOfDay::default());
        let stray = w.spawn_with_guid(uuid(2), "Stray Atmosphere", None);
        w.world_mut()
            .entity_mut(stray)
            .insert(SkyAtmosphere::default());
        let sky = resolve_sky(&w).expect("the clock still resolves");
        assert_eq!(
            sky.atmosphere,
            SkyAtmosphere::default(),
            "the authority's own (defaulted) atmosphere is what was used"
        );
        assert_eq!(orphan_atmosphere_warnings(), 1);

        // The healthy shape — atmosphere ON the authority — must stay silent.
        reset_orphan_atmosphere_warning();
        let mut w = EcsWorld::new();
        let e = w.spawn_with_guid(uuid(1), "Sky", None);
        w.world_mut().entity_mut(e).insert(TimeOfDay::default());
        w.world_mut().entity_mut(e).insert(SkyAtmosphere::default());
        for _ in 0..5 {
            resolve_sky(&w);
        }
        assert_eq!(
            orphan_atmosphere_warnings(),
            0,
            "a correct level must not warn"
        );
        reset_orphan_atmosphere_warning();
    }

    #[test]
    fn authority_ignores_entities_without_a_clock() {
        let mut w = EcsWorld::new();
        // A lower-Guid entity without a clock must not win the resolution.
        w.spawn_with_guid(uuid(1), "Plain", None);
        let clock = w.spawn_with_guid(uuid(9), "Sky", None);
        w.world_mut().entity_mut(clock).insert(TimeOfDay::default());
        assert_eq!(sky_authority(&w), Some(clock));
    }

    // ── weather (P17.4) ─────────────────────────────────────────────────

    fn weather_world(atmos: SkyAtmosphere) -> EcsWorld {
        let mut w = world_with(&[(uuid(1), TimeOfDay::default())]);
        let e = sky_authority(&w).unwrap();
        w.world_mut().entity_mut(e).insert(atmos);
        w
    }

    /// The **byte-stability** claim for every pre-P17.4 level: with weather off,
    /// the resolved weather is the authored cloud/fog fields field for field and
    /// there is no precipitation at all.
    #[test]
    fn weather_off_resolves_the_authored_cloud_and_fog_fields() {
        let a = SkyAtmosphere {
            cloud_coverage: 0.42,
            cloud_type: 0.31,
            cloud_wind_x: 7.5,
            cloud_wind_z: -2.25,
            fog_density: 3.0e-4,
            // Deliberately hostile: the weather block is fully populated with
            // values that must NOT be read while the flag is clear.
            weather_enabled: false,
            weather_coverage: 1.0,
            weather_cloud_type: 0.0,
            weather_wind_x: 99.0,
            weather_wind_z: 99.0,
            weather_fog_density: 1.0,
            weather_precipitation: 1.0,
            weather_snowiness: 1.0,
            ..SkyAtmosphere::default()
        };
        let sky = resolve_sky(&weather_world(a)).unwrap();
        let w = sky.weather();
        assert_eq!(w.cloud_coverage, 0.42);
        assert_eq!(w.cloud_type, 0.31);
        assert_eq!(w.wind_x, 7.5);
        assert_eq!(w.wind_z, -2.25);
        assert_eq!(w.fog_density, 3.0e-4);
        assert_eq!(
            w.precipitation, 0.0,
            "disabled weather must not precipitate"
        );
        assert_eq!(w.snowiness, 0.0);
        assert_eq!(sky.snow_accumulation_rate(), 0.0);
    }

    /// …and with weather **on** the same four fields come from the weather block
    /// instead. Both directions matter: this is what "driven, not duplicated"
    /// means.
    #[test]
    fn weather_on_drives_clouds_and_fog() {
        let a = SkyAtmosphere {
            cloud_coverage: 0.42,
            cloud_wind_x: 7.5,
            fog_density: 3.0e-4,
            weather_enabled: true,
            ..SkyAtmosphere::default()
        };
        let mut w = weather_world(a);
        assert!(set_weather(&mut w, WeatherPreset::Storm, 0.0), "snap");
        let sky = resolve_sky(&w).unwrap();
        let r = sky.weather();
        let storm = WeatherPreset::Storm.params();
        assert_eq!(r.cloud_coverage, storm.coverage);
        assert_eq!(r.wind_x, storm.wind_x);
        assert_eq!(r.fog_density, storm.fog_density);
        assert_eq!(r.precipitation, 1.0);
        // Storm is rain, so the P22 hook stays at zero however hard it pours.
        assert_eq!(sky.snow_accumulation_rate(), 0.0);
    }

    /// The P22 hook: snow — and only snow — accumulates, proportionally to both
    /// intensity and phase.
    #[test]
    fn snow_accumulation_rate_is_the_p22_hook() {
        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        });
        assert!(set_weather(&mut w, WeatherPreset::Snow, 0.0));
        let rate = resolve_sky(&w).unwrap().snow_accumulation_rate();
        let snow = WeatherPreset::Snow.params();
        assert_eq!(rate, snow.precipitation * SNOW_ACCUMULATION_MAX_M_PER_S);
        // ~5 cm/hour at full blast, ~3.5 cm/hour at the Snow preset's 0.7.
        assert!(
            (rate * 3600.0 - 0.035).abs() < 1e-3,
            "{} m/h",
            rate * 3600.0
        );
    }

    /// The blend is **exactly linear** and lands on the target on time — the
    /// property that lets the component store no `from` snapshot. Checked at
    /// three fractions of the way, not just at the end, because an exponential
    /// approach would also arrive (asymptotically) and would pass an
    /// endpoints-only test.
    #[test]
    fn the_weather_blend_is_exactly_linear_and_lands_on_time() {
        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        });
        let from = WeatherPreset::Clear.params();
        let to = WeatherPreset::Storm.params();
        // 4 s at 60 Hz ≈ 240 steps. Run until the blender goes inert rather than
        // for a fixed count: `1/60` is not representable, so 240 subtractions
        // leave a few microseconds behind and the blend legitimately takes one
        // more step. Asserting the count *afterwards* keeps the duration claim
        // without pretending f32 arithmetic is exact.
        assert!(set_weather(&mut w, WeatherPreset::Storm, 4.0));
        let dt = 1.0 / 60.0;
        let mut steps = 0u32;
        while let Some(live) = advance_weather(&mut w, dt) {
            steps += 1;
            assert!(steps <= 300, "the blend never settled");
            let t = (steps as f32 * (dt as f32) / 4.0).min(1.0);
            let want = from.coverage + (to.coverage - from.coverage) * t;
            // 1e-3 of a 0.92 range: loose enough for 240 f32 steps, tight enough
            // that a smoothstep (off by 0.09 at t = 0.25) or an exponential ease
            // fails immediately. Linearity is the claim, not the endpoints.
            assert!(
                (live.coverage - want).abs() < 1e-3,
                "step {steps}: coverage {} != linear {want}",
                live.coverage
            );
        }
        assert!(
            (240..=242).contains(&steps),
            "a 4 s blend at 60 Hz took {steps} steps"
        );
        // Landed exactly (an identity, not a tolerance — see `advance_weather`),
        // and settled. A settled blend is INERT: the blender must never write
        // these fields again, which is what lets the sequencer own them, and the
        // `while` above already proved it by terminating.
        let a = authority_atmosphere(&w).unwrap();
        assert_eq!(
            a.weather_params(),
            to,
            "the blend did not land on the target"
        );
        assert_eq!(a.weather_blend_remaining, 0.0);
    }

    /// A settled state is a fixed point in the other direction too: once the
    /// blender is inert, a hand-written (sequencer / Details / Blueprint) value
    /// survives the fixed step untouched.
    #[test]
    fn a_settled_blend_leaves_hand_authored_values_alone() {
        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: true,
            weather_target: WeatherPreset::Storm,
            weather_precipitation: 0.25, // deliberately NOT the Storm value
            ..SkyAtmosphere::default()
        });
        for _ in 0..120 {
            assert!(advance_weather(&mut w, 1.0 / 60.0).is_none());
        }
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_precipitation,
            0.25
        );
    }

    /// Determinism, the gate's shape: two independent runs of the same blend
    /// agree **to the bit**, including at the intermediate steps.
    #[test]
    fn the_weather_blend_is_bit_identical_across_runs() {
        let run = || {
            let mut w = weather_world(SkyAtmosphere {
                weather_enabled: true,
                ..SkyAtmosphere::default()
            });
            set_weather(&mut w, WeatherPreset::Snow, 3.0);
            let mut trace = Vec::new();
            for _ in 0..300 {
                advance_weather(&mut w, 1.0 / 60.0);
                let a = authority_atmosphere(&w).unwrap();
                trace.push((
                    a.weather_coverage.to_bits(),
                    a.weather_wind_x.to_bits(),
                    a.weather_precipitation.to_bits(),
                    a.weather_snowiness.to_bits(),
                    a.weather_blend_remaining.to_bits(),
                ));
            }
            trace
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "the weather blend is not deterministic");
        assert_ne!(a.first(), a.last(), "the blend never moved — vacuous gate");
    }

    /// A disabled weather block does not blend at all, so a level that has not
    /// opted in cannot have its authored clouds silently walked somewhere else.
    #[test]
    fn a_disabled_weather_block_never_blends() {
        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: false,
            weather_target: WeatherPreset::Storm,
            weather_blend_remaining: 10.0,
            ..SkyAtmosphere::default()
        });
        for _ in 0..60 {
            assert!(advance_weather(&mut w, 1.0 / 60.0).is_none());
        }
        let a = authority_atmosphere(&w).unwrap();
        assert_eq!(
            a.weather_params(),
            SkyAtmosphere::default().weather_params()
        );
    }

    /// **The blender must always terminate.** A `weather_blend_remaining` large
    /// enough that `remaining − dt` rounds back to `remaining` would freeze the
    /// countdown, underflow `k` to zero, and leave `advance_weather` writing the
    /// component every step forever without ever settling — which voids the
    /// "settled ⇒ the blender never writes these fields again" contract the
    /// sequencer and the Details grid depend on.
    ///
    /// Every one of these is hand-set **on the component**, bypassing both doors
    /// (`set_weather` and the DTO clamp), because that is exactly what a
    /// hand-edited `.inf_lvl.toml` can do — the same reasoning the `> 0.0`
    /// in-flight test above is written for.
    #[test]
    fn a_hostile_blend_remaining_still_settles() {
        for remaining in [
            f32::MAX,
            1e30,
            1e12,
            // Past the 60 Hz freeze point (`ulp(x)/2 >= 1/60` from x ≈ 2¹⁹) …
            (1u32 << 20) as f32,
            // … and exactly ON the binade boundary, where the subtraction still
            // moves (the binade *below* has half the ULP) but by one ULP a step,
            // which is 16.8 million steps to settle. The `over_ceiling` arm is
            // what catches this one; the arithmetic test alone would not.
            (1u32 << 19) as f32,
            // Just over the ceiling: progress is normal here, so ONLY the ceiling
            // test settles it — the pair of conditions is not redundant.
            MAX_WEATHER_BLEND_S + 1.0,
            f32::INFINITY,
            f32::NAN,
            -1.0,
            f32::NEG_INFINITY,
        ] {
            let mut w = weather_world(SkyAtmosphere {
                weather_enabled: true,
                weather_target: WeatherPreset::Storm,
                weather_blend_remaining: remaining,
                ..SkyAtmosphere::default()
            });
            // At most one step of work: either the blend was never in flight
            // (negative / NaN), or the no-progress backstop settles it at once.
            advance_weather(&mut w, 1.0 / 60.0);
            assert!(
                advance_weather(&mut w, 1.0 / 60.0).is_none(),
                "a blend of {remaining} s is still armed after one step — the \
                 blender never terminates"
            );
            // A *finite positive* remainder is consumed and zeroed. A negative
            // or NaN one never entered the body at all (the `> 0.0` in-flight
            // test rejected it), so the field is left exactly as found — which is
            // the same contract stated the other way round: the blender does not
            // write. Asserting "it becomes 0" for those would be asserting a
            // write we specifically do not want.
            let a = authority_atmosphere(&w).unwrap();
            if remaining > 0.0 {
                assert_eq!(
                    a.weather_blend_remaining, 0.0,
                    "a blend of {remaining} s did not settle"
                );
            } else {
                assert_eq!(
                    a.weather_params(),
                    SkyAtmosphere::default().weather_params(),
                    "a blend of {remaining} s moved the parameters"
                );
            }
        }
    }

    /// …and the door that a *script* comes through clamps before it gets there,
    /// so the backstop above is a second line rather than the only one.
    #[test]
    fn set_weather_clamps_the_blend_to_the_arithmetic_ceiling() {
        let mut w = weather_world(SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        });
        assert!(set_weather(&mut w, WeatherPreset::Storm, 5e6));
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_blend_remaining,
            MAX_WEATHER_BLEND_S,
            "`sky.set_weather` must clamp to the ceiling, not store 5e6"
        );
        // An absurd *authored* default reaches the same field through the
        // negative-argument fallback, so it is clamped on that path too.
        let mut w2 = weather_world(SkyAtmosphere {
            weather_enabled: true,
            weather_blend_seconds: 1e9,
            ..SkyAtmosphere::default()
        });
        assert!(set_weather(&mut w2, WeatherPreset::Fog, -1.0));
        assert_eq!(
            authority_atmosphere(&w2).unwrap().weather_blend_remaining,
            MAX_WEATHER_BLEND_S
        );

        // …and a clamped blend genuinely completes rather than merely being
        // smaller. Stepped at a coarse dt so the whole hour runs in the test:
        // the claim is termination, and `dt` is the fixed step's only role here.
        let mut steps = 0u32;
        while advance_weather(&mut w, 1.0).is_some() {
            steps += 1;
            assert!(steps <= 3_700, "a clamped blend still did not finish");
        }
        assert_eq!(steps, MAX_WEATHER_BLEND_S as u32);
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_params(),
            WeatherPreset::Storm.params()
        );
    }

    #[test]
    fn weather_host_seams_read_and_write_the_authority() {
        let mut w = weather_world(SkyAtmosphere::default());
        assert_eq!(weather_preset_name(&w), "clear");
        assert_eq!(weather_precipitation(&w), 0.0);
        // With weather off the wind getter still answers: the *cloud* wind is
        // the only wind there is, and reporting 0 would be a lie a sway/sailing
        // script would act on. (6, 2) m/s is the cloud block's default.
        let d = SkyAtmosphere::default();
        let cloud_speed =
            ((d.cloud_wind_x as f64).powi(2) + (d.cloud_wind_z as f64).powi(2)).sqrt();
        assert!((weather_wind_speed(&w) - cloud_speed).abs() < 1e-9);

        // set_weather ENABLES the block — a script that names a preset means it.
        assert!(set_weather(&mut w, WeatherPreset::Storm, 0.0));
        assert!(authority_atmosphere(&w).unwrap().weather_enabled);
        assert_eq!(weather_preset_name(&w), "storm");
        assert_eq!(weather_precipitation(&w), 1.0);
        let s = WeatherPreset::Storm.params();
        let want = ((s.wind_x as f64).powi(2) + (s.wind_z as f64).powi(2)).sqrt();
        assert!((weather_wind_speed(&w) - want).abs() < 1e-9);

        // A NEGATIVE / non-finite blend falls back to the authored default (8 s);
        // an explicit 0 (the unwired-pin value) snaps, which the Storm write above
        // already proved by landing on the preset with nothing in flight.
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_blend_remaining,
            0.0
        );
        assert!(set_weather(&mut w, WeatherPreset::Clear, -1.0));
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_blend_remaining,
            8.0
        );
        assert!(set_weather(&mut w, WeatherPreset::Fog, f64::NAN));
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_blend_remaining,
            8.0
        );
        // …and an explicit duration is taken literally.
        assert!(set_weather(&mut w, WeatherPreset::Snow, 12.5));
        assert_eq!(
            authority_atmosphere(&w).unwrap().weather_blend_remaining,
            12.5
        );

        // A clock-less world answers the defined values and refuses the write.
        let mut empty = EcsWorld::new();
        assert_eq!(weather_preset_name(&empty), "clear");
        assert_eq!(weather_precipitation(&empty), 0.0);
        assert_eq!(weather_wind_speed(&empty), 0.0);
        assert!(!set_weather(&mut empty, WeatherPreset::Storm, 1.0));
    }

    /// The preset name round-trip a Blueprint depends on, plus the rule that an
    /// unknown name is a **no-op**: `parse` returning `None` is what stops a
    /// typo'd `sky.set_weather("stormy")` from quietly picking a different sky.
    #[test]
    fn preset_names_round_trip_and_reject_typos() {
        for p in WeatherPreset::ALL {
            assert_eq!(WeatherPreset::parse(p.as_str()), Some(p));
            assert_eq!(WeatherPreset::parse(&p.as_str().to_uppercase()), Some(p));
            assert_eq!(
                WeatherPreset::parse(&format!("  {}  ", p.as_str())),
                Some(p)
            );
        }
        assert_eq!(WeatherPreset::parse("stormy"), None);
        assert_eq!(WeatherPreset::parse(""), None);
        // Every preset is distinct in what it actually does — a copy-pasted
        // table entry would otherwise be invisible.
        for (i, a) in WeatherPreset::ALL.iter().enumerate() {
            for b in &WeatherPreset::ALL[i + 1..] {
                assert_ne!(a.params(), b.params(), "{a:?} and {b:?} are the same sky");
            }
        }
        // The component's defaults ARE a settled Clear state, which is what makes
        // enabling weather a single boolean rather than a parameter hunt.
        let d = SkyAtmosphere::default();
        assert_eq!(d.weather_params(), WeatherPreset::Clear.params());
        assert_eq!(d.weather_target, WeatherPreset::Clear);
        assert!(!d.weather_enabled);
        assert_eq!(d.weather_blend_remaining, 0.0);
    }
}
