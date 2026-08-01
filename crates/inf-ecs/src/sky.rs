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

use crate::components::{Guid, SkyAtmosphere, TimeOfDay};
use crate::world::EcsWorld;

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

// ── the `sky.*` Blueprint host surface (P17.1) ──────────────────────────────
//
// Four one-line seams shared verbatim by the editor's `SimHost` and the shipped
// player's `RuntimeHost`, so preview == shipped by construction rather than by
// two hand-written match arms agreeing. Units per architecture rule 6: seconds
// for the clock, a dimensionless multiplier for the rate.

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
}
