//! **Sound through a door** (island wave VEN1b) — the occlusion model, given
//! the doorways it has been walking past since I6.
//!
//! # What this replaces
//!
//! Since P12.3 the whole of audio occlusion has been one unfiltered raycast and
//! a flat cut: hit anything → −12 dB, hit nothing → unity, and the test taken
//! **once, at `Play` time**, so a listener who walked from the street into a
//! club heard the same muffled loop all the way in. It is a good first model
//! and it has one wrong answer that matters: *a doorway is not a wall*.
//!
//! Every element the fix needs already existed and audio read none of it.
//! `d3::door::placements` flattens authored `Door` components **and** every
//! grammar `PcgDoorway` into one list; `door::DoorField` holds each leaf's
//! `open_deg`. A venue's doorway is therefore a queryable hole in a queryable
//! wall, and this module is the rule that reads it.
//!
//! # The rule, in four cases
//!
//! ```text
//!   ray clear                 -> Clear    unity, no filter   "you are inside"
//!   blocked, no doorway near  -> Wall     -12 dB             (P12.3, unchanged)
//!   blocked, OPEN doorway     -> Doorway  falls off with the LISTENER's own
//!                                         distance from the opening, no filter
//!   blocked, SHUT doorway     -> Shut     -24 dB + a low-pass
//! ```
//!
//! The third case is the one the mandate names — *attenuated, not occluded,
//! with a swell as you cross*. A portal's attenuation is a function of the
//! **detour**: how much further the sound has to travel to get through the hole
//! than it would through the wall. Standing in the doorway the detour is zero
//! and the music is at full; a pace to the side and it starts to fall; past
//! [`PORTAL_LISTENER_REACH_M`] the doorway is not the way this sound arrives and
//! the wall's own answer takes over. Nothing here is a state machine, so walking
//! in and out is continuous by construction.
//!
//! # The second case is UNCHANGED, deliberately
//!
//! `AudioSource::occlusion` is `false` in `Default` and no committed content
//! sets it, so the only source on this path today is a venue's own music — but
//! the rule is written so that the day one does, its verdict is the number
//! P12.3 gave it. `the_wall_answer_is_the_one_p12_gave` is the arm.
//!
//! # Portable math
//!
//! Distances are `sqrt` of sums of products and the interpolations are `lerp`s.
//! No trigonometry — this gain reaches the audio command stream, which is what
//! `physics_demo`'s gate (c) compares between the editor's PIE and the shipped
//! player.

use glam::DVec3;
use inf_ecs::door::DoorState;
use inf_ecs::EcsWorld;

use super::world::PhysicsWorld3D;

/// The obstruction gain (linear) of a source behind a plain wall — a **−12 dB**
/// cut.
///
/// P12.3's own number, kept to the digit. It is here rather than in each host
/// because two hosts that muffled by different amounts would fork the audio
/// command stream, which is the one thing `physics_demo`'s gate (c) compares.
pub const WALL_CUT_LINEAR: f64 = 0.251_188_643_150_958;

/// The obstruction gain (linear) of a source behind a **shut door** — a −24 dB
/// cut.
///
/// Twice the wall's cut in dB, and that is the honest ordering rather than an
/// arbitrary depth: a door leaf is thinner than a wall and would transmit
/// *more*, but a shut door is what stands between a listener and a hole that
/// would otherwise be open, and the number a player reads is the difference
/// between "the club is over there" and "the club is through this door". The
/// low-pass below is what carries most of the impression; this carries the
/// rest.
pub const DOOR_SHUT_CUT_LINEAR: f64 = 0.063_095_734_448_019;

/// The cutoff of the one-pole low-pass a shut door puts on a source, hertz.
///
/// Five hundred: above a kick drum's fundamental and below almost everything
/// else, which is what a club sounds like from the pavement.
pub const DOOR_SHUT_LOWPASS_HZ: f64 = 500.0;

/// How far a sound may **detour** through a doorway before the doorway stops
/// being on its way at all, metres.
///
/// `|listener → door| + |door → emitter| − |listener → emitter|`, which is zero
/// when the opening is exactly on the straight line and grows as it moves off
/// it. Four metres rejects a door behind the listener and a door on the far
/// side of the building.
///
/// **It is a filter and not the attenuation**, and finding that out is the
/// measurement this pair of constants exists because of. The first cut used the
/// detour for both, and a detour is nearly *scale-free*: measured on this
/// module's own fixture, a listener 4 m straight out from the opening detoured
/// **0.22 m** and one 5 m round the corner detoured **1.99 m** — barely apart on
/// an eight-metre scale — while a listener **sixty metres** away detoured
/// **5.2 m** and was accepted as hearing the club through its front door. What
/// separates the three is how far the LISTENER is from the opening, so that is
/// what the gain is a function of.
pub const PORTAL_DETOUR_MAX_M: f64 = 4.0;

/// How near the opening a listener has to be to hear a source through it
/// unattenuated, metres.
///
/// Two: standing in the doorway. This is the top of the swell.
pub const PORTAL_LISTENER_FULL_M: f64 = 2.0;

/// How far from the opening a listener may be before the doorway stops being
/// the way the sound arrives, metres.
///
/// Twelve, which is the width of a street plus a pavement: a body on the far
/// kerb still hears the club through its door, and one at the end of the block
/// hears a wall. Between [`PORTAL_LISTENER_FULL_M`] and this the gain falls
/// from unity to [`WALL_CUT_LINEAR`], which is the swell the mandate asks for
/// written as an interpolation rather than as a fade.
pub const PORTAL_LISTENER_REACH_M: f64 = 12.0;

/// **How the sound got here.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalVerdict {
    /// Nothing between the listener and the emitter. Inside the room.
    Clear,
    /// Something solid, and no doorway near enough to be the way through.
    Wall,
    /// Through an open doorway — attenuated by the detour, and **not**
    /// filtered.
    Doorway,
    /// Through a shut door — cut hard and low-passed.
    Shut,
}

impl PortalVerdict {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            PortalVerdict::Clear => "clear",
            PortalVerdict::Wall => "wall",
            PortalVerdict::Doorway => "doorway",
            PortalVerdict::Shut => "shut",
        }
    }
}

/// What one occlusion query answered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortalGain {
    /// The obstruction gain in `[0, 1]` — what `inf_audio`'s
    /// `PlayCommand::occlusion_gain` carries. (Named and not linked: this crate
    /// does not depend on `inf-audio`, and the seam is deliberately one way —
    /// the physics side decides a number and the host puts it on a command.)
    pub gain: f64,
    /// The one-pole low-pass this path puts on the source, hertz, or `None` for
    /// a path that only attenuates.
    ///
    /// **Carried in the command and not yet applied by the device.** The mixer
    /// has held `Effect::Lowpass` as data since P12 and `backend.rs` has no
    /// filter code at all — its own module doc calls wiring per-bus kira
    /// sub-tracks "the documented device-side follow-up". So this is a decision
    /// the sim makes, observable in the command stream both hosts are compared
    /// on and readable through `AudioEngine::effective_lowpass_hz`; what it is
    /// not yet is a filter you can hear. Priced, taken to the seam it can
    /// honestly reach, and carried past it. See the wave ledger.
    pub lowpass_hz: Option<f64>,
    /// How the sound got here.
    pub verdict: PortalVerdict,
    /// The detour through the doorway that produced it, metres — `0.0` for
    /// [`Clear`](PortalVerdict::Clear) and [`Wall`](PortalVerdict::Wall).
    pub detour_m: f64,
}

impl PortalGain {
    /// Unobstructed.
    pub const CLEAR: Self = Self {
        gain: 1.0,
        lowpass_hz: None,
        verdict: PortalVerdict::Clear,
        detour_m: 0.0,
    };

    /// Behind a plain wall — P12.3's answer, unchanged.
    pub const WALL: Self = Self {
        gain: WALL_CUT_LINEAR,
        lowpass_hz: None,
        verdict: PortalVerdict::Wall,
        detour_m: 0.0,
    };

    /// The gain in decibels, for a report. `-inf` at silence.
    pub fn db(&self) -> f64 {
        if self.gain <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * self.gain.log10()
    }
}

/// **The occlusion a listener hears an emitter through** — the one door both
/// hosts call.
///
/// `phys` is taken mutably because `PhysicsWorld3D::cast_ray` is: rapier's
/// query pipeline is updated in place.
///
/// The ray is the P12.3 one, unchanged and unfiltered — it can be blocked by a
/// dynamic body or by the emitter's own collider, which is the same bound that
/// model has always had and is stated here rather than silently inherited.
pub fn portal_gain(
    world: &EcsWorld,
    phys: &mut PhysicsWorld3D,
    listener: DVec3,
    emitter: DVec3,
) -> PortalGain {
    let delta = emitter - listener;
    let dist = delta.length();
    if !(dist.is_finite() && dist > 1e-6) {
        return PortalGain::CLEAR;
    }
    let dir = delta / dist;
    let blocked = match phys.cast_ray(listener, dir, dist) {
        Some(hit) => (hit.point - listener).length() + 1e-3 < dist,
        None => false,
    };
    if !blocked {
        return PortalGain::CLEAR;
    }
    portal_of(world, listener, emitter, dist).unwrap_or(PortalGain::WALL)
}

/// **The doorway this sound comes through, if any** — the half of
/// [`portal_gain`] that needs no physics and is therefore testable without one.
///
/// The portal is the door with the smallest **detour**; ties break on the
/// door's own `Guid`, which `d3::door::placements` has already sorted by, so
/// two hosts pick the same hole out of the same world.
pub fn portal_of(
    world: &EcsWorld,
    listener: DVec3,
    emitter: DVec3,
    direct_m: f64,
) -> Option<PortalGain> {
    let field = inf_ecs::door::door_field(world);
    // The nearest opening the sound could plausibly come through: near the
    // LISTENER, and on the way. Ties break on the `Guid` order
    // `d3::door::placements` has already sorted by, so two hosts pick the same
    // hole out of the same world.
    let mut best: Option<(f64, f64, bool)> = None;
    for p in super::door::placements(world) {
        if !p.hinge.is_finite() {
            continue;
        }
        // The hinge, and not the opening's centre: a leaf hangs off one edge of
        // the hole, so this is up to half a door width (0.45 m) off the middle
        // of it — an error two orders below the twelve metres it is compared
        // against, and one that costs no trigonometry on a path that reaches
        // the audio command stream.
        let mouth = p.hinge;
        let near = (mouth - listener).length();
        let detour = near + (emitter - mouth).length() - direct_m;
        if !(near.is_finite() && detour.is_finite()) {
            continue;
        }
        if near > PORTAL_LISTENER_REACH_M || detour > PORTAL_DETOUR_MAX_M {
            continue;
        }
        let open = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec))
            .is_open(&p.spec);
        if best.is_none_or(|(bn, _, _)| near < bn) {
            best = Some((near, detour, open));
        }
    }
    let (near, detour, open) = best?;
    if !open {
        return Some(PortalGain {
            gain: DOOR_SHUT_CUT_LINEAR,
            lowpass_hz: Some(DOOR_SHUT_LOWPASS_HZ),
            verdict: PortalVerdict::Shut,
            detour_m: detour,
        });
    }
    // **Attenuated, not occluded.** Unity in the opening, falling to the wall's
    // own cut by the time the doorway has stopped being the way through — so
    // crossing the threshold is a swell rather than a switch, and the
    // continuity is a property of the arithmetic rather than of a fade.
    let span = (PORTAL_LISTENER_REACH_M - PORTAL_LISTENER_FULL_M).max(1e-6);
    let u = ((near - PORTAL_LISTENER_FULL_M) / span).clamp(0.0, 1.0);
    Some(PortalGain {
        gain: 1.0 + (WALL_CUT_LINEAR - 1.0) * u,
        lowpass_hz: None,
        verdict: PortalVerdict::Doorway,
        detour_m: detour,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::{DoorwaySlot, GlobalTransform, PcgVolume, Transform};
    use uuid::Uuid;

    /// A world holding one grammar doorway at `hinge`.
    fn world_with_door(hinge: DVec3) -> EcsWorld {
        let mut w = EcsWorld::new();
        let v = Uuid::from_u128(0xD0_04);
        w.spawn_with_guid(v, "block", None);
        let e = w.entity_of(v).expect("the block");
        let mut vol = PcgVolume::default();
        vol.set_population(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![DoorwaySlot {
                hinge,
                closed_yaw_deg: 0.0,
                width_m: 0.9,
                height_m: 2.1,
                thickness_m: 0.2,
                inside_yaw_deg: 180.0,
                exterior: true,
                floor: 0,
            }],
            Vec::new(),
            Default::default(),
            Vec::new(),
            Vec::new(),
        );
        w.world_mut().entity_mut(e).insert((
            Transform::IDENTITY,
            GlobalTransform(glam::DAffine3::IDENTITY),
            vol,
        ));
        w.mark_dirty();
        w.propagate();
        w
    }

    fn door_guid(w: &EcsWorld) -> Uuid {
        super::super::door::placements(w)
            .first()
            .expect("the fixture door")
            .guid
    }

    /// **THE THREE POINTS**, which is the clause's own measurement: standing at
    /// the doorway is louder than standing outside it, and standing outside a
    /// SHUT one is quieter still and filtered.
    #[test]
    fn a_doorway_attenuates_where_a_wall_occludes() {
        let w = world_with_door(DVec3::new(0.0, 1.05, 0.0));
        // The emitter is 5 m inside; the listener walks away from the opening.
        let emitter = DVec3::new(0.0, 2.5, 5.0);
        let at = |z: f64, x: f64| -> PortalGain {
            let l = DVec3::new(x, 1.7, z);
            portal_of(&w, l, emitter, (emitter - l).length()).expect("a portal")
        };
        let doorway = at(-0.5, 0.0);
        let outside = at(-4.0, 0.0);
        let aside = at(-2.0, 5.0);
        println!(
            "VEN1b portal: doorway {:.3} ({:.1} dB, detour {:.2}); outside \
             {:.3} ({:.1} dB, detour {:.2}); round the corner {:.3} ({:.1} dB, \
             detour {:.2})",
            doorway.gain,
            doorway.db(),
            doorway.detour_m,
            outside.gain,
            outside.db(),
            outside.detour_m,
            aside.gain,
            aside.db(),
            aside.detour_m
        );
        // A shut door is the fixture's default (`DoorState::fresh`), so all
        // three are `Shut` until it is opened — assert that first, because a
        // gate that measured the open case on a shut door would be measuring
        // nothing.
        assert_eq!(doorway.verdict, PortalVerdict::Shut);
        assert_eq!(doorway.gain, DOOR_SHUT_CUT_LINEAR);
        assert_eq!(doorway.lowpass_hz, Some(DOOR_SHUT_LOWPASS_HZ));

        // Now open it, and the three points separate.
        {
            let mut w2 = world_with_door(DVec3::new(0.0, 1.05, 0.0));
            let g = door_guid(&w2);
            let spec = super::super::door::placement_of(&w2, g)
                .expect("a door")
                .spec;
            let f = inf_ecs::door::door_field_mut(&mut w2);
            f.entry(g, &spec).open_deg = spec.open_limit_deg;
            let at = |z: f64, x: f64| -> PortalGain {
                let l = DVec3::new(x, 1.7, z);
                portal_of(&w2, l, emitter, (emitter - l).length()).expect("a portal")
            };
            let (d, o, a) = (at(-0.5, 0.0), at(-4.0, 0.0), at(-2.0, 5.0));
            println!(
                "VEN1b portal (open): doorway {:.1} dB, outside {:.1} dB, round \
                 the corner {:.1} dB",
                d.db(),
                o.db(),
                a.db()
            );
            assert_eq!(d.verdict, PortalVerdict::Doorway);
            assert_eq!(d.lowpass_hz, None, "an open door filters the sound");
            // In the opening: essentially unattenuated.
            assert!(d.gain > 0.95, "at the open door the gain is {:.3}", d.gain);
            // Straight out from it: quieter, but nowhere near a wall.
            assert!(
                o.gain < d.gain && o.gain > WALL_CUT_LINEAR,
                "outside {:.3} against the doorway's {:.3} and a wall's {:.3}",
                o.gain,
                d.gain,
                WALL_CUT_LINEAR
            );
            // Round the corner: further from the opening, and quieter for it.
            assert!(
                a.gain < o.gain,
                "standing beside the wall is {:.3} against {:.3} straight out \
                 — the gain is not falling with the listener's distance from \
                 the opening",
                a.gain,
                o.gain
            );
            assert!(
                a.detour_m > o.detour_m,
                "standing beside the wall detoured {:.2} m against {:.2} m \
                 straight out",
                a.detour_m,
                o.detour_m
            );
            // …and the SHUT answer is quieter than any of them.
            assert!(DOOR_SHUT_CUT_LINEAR < a.gain);
        }
    }

    /// **A doorway too far away is not the way the sound arrives** — and the
    /// answer is then exactly the one P12.3 gave, to the digit.
    #[test]
    fn the_wall_answer_is_the_one_p12_gave() {
        let w = world_with_door(DVec3::new(0.0, 1.05, 0.0));
        // A listener a long way round the block: the detour through the door is
        // enormous.
        let emitter = DVec3::new(0.0, 2.5, 5.0);
        let listener = DVec3::new(60.0, 1.7, 5.0);
        assert!(
            portal_of(&w, listener, emitter, (emitter - listener).length()).is_none(),
            "a door sixty metres off the line was taken as the way through"
        );
        assert_eq!(PortalGain::WALL.gain, WALL_CUT_LINEAR);
        assert_eq!(PortalGain::WALL.lowpass_hz, None);
        // The P12.3 constant, spelled out here so a drift is a red test rather
        // than a quieter game: 10^(-12/20).
        assert!(
            (WALL_CUT_LINEAR - 10f64.powf(-12.0 / 20.0)).abs() < 1e-12,
            "the wall cut is no longer -12 dB"
        );
        assert!(
            (DOOR_SHUT_CUT_LINEAR - 10f64.powf(-24.0 / 20.0)).abs() < 1e-12,
            "the shut-door cut is no longer -24 dB"
        );
    }

    /// A level with no doors at all answers `None`, so a world that never had a
    /// doorway is the wall model exactly.
    #[test]
    fn a_level_with_no_doors_is_the_wall_model() {
        let w = EcsWorld::new();
        assert!(portal_of(&w, DVec3::ZERO, DVec3::new(0.0, 0.0, 5.0), 5.0).is_none());
    }
}
