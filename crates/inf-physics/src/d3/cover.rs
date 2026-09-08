//! **THE COVER PROBE** (wave COV1) — what a character could put its back
//! against, how tall it is, and where it ends.
//!
//! # A second reader of the mantle's own sweeps
//!
//! [`super::traversal::probe_ledge`] and this module ask two different questions
//! of the same geometry. A mantle asks *what is the **top** of the thing in
//! front of me and can I get onto it*; a cover press asks *what is the **face**
//! of it, how high does it come, and where does it stop*. Both begin with a
//! forward capsule sweep across a band of heights and both use a downward sphere
//! sweep to find a surface top, so those two are one door each —
//! [`super::traversal::sweep_forward_face`] and
//! [`super::traversal::sweep_surface_top`] — and this module is the second
//! reader of both. That is the wave brief's "one door, two readers", made out of
//! functions rather than out of a comment.
//!
//! The band differs, and that is the whole of the difference. A ledge probe
//! sweeps 0.50–2.50 m because that is what a body can climb. A cover probe
//! sweeps [`CoverSettings::band_low_m`]–[`CoverSettings::band_high_m`] because
//! that is what a body can hide behind: it starts BELOW the ledge probe's floor
//! (a 0.6 m car door is cover and is not a ledge) and it stops well above head
//! height (a wall is cover no matter how tall it is).
//!
//! # The classification is ALS's split, one threshold lower
//!
//! [`inf_anim::MANTLE_HIGH_SPLIT_M`] is the donor's hard 125 cm line between the
//! one-metre and two-metre mantles, and it is the same line GTA's cover system
//! draws between crouching behind a car and standing against a wall. So
//! [`CoverClass`] reuses it rather than inventing a second number, and the only
//! new threshold is the FLOOR — [`CoverSettings::min_height_m`], below which a
//! surface is a kerb rather than cover.
//!
//! # Everything refuses by value, and the refusal says what it hit
//!
//! [`probe_cover`] always answers a [`CoverProbe`], never an `Option`, and a
//! probe that found nothing carries the [`CoverRefusal`] that says why and — for
//! the two refusals that had a surface to look at — the
//! [`ColliderLabel`](super::label::ColliderLabel) of the thing it looked at.
//! Carried 162 is what makes that possible: a façade on this island has no
//! entity at all, so without the label door a refusal could only ever have said
//! "a guid".
//!
//! # The cost, and it is zero until somebody asks
//!
//! Nothing here runs on an ordinary step. [`super::movement`] calls it on the
//! cover **press** and on the steps a character is **in** cover, and
//! [`CoverProbe::sweeps`] counts the shape casts it spent so a gate can assert
//! the zero rather than believe it. The extents — the lateral search that finds
//! the corners a slide must stop at — are the expensive half
//! ([`CoverSettings::extent_samples`] casts a side) and they are measured only
//! once a surface has been classified as cover at all.

use glam::DVec3;
use uuid::Uuid;

use super::label::{ColliderFamily, ColliderLabel};
use super::traversal::{is_walkable, sweep_forward_face, sweep_surface_top};
use super::{ColliderId3D, PhysicsBridge3D};

/// The class, the side and the rule that decides them all live in
/// [`inf_ecs::cover`] — this crate is the half that needs a world, and a second
/// spelling of "what counts as cover" is exactly the thing the ring split
/// exists to prevent. Re-exported so a caller with a `PhysicsBridge3D` in hand
/// does not have to name two crates to read one probe.
pub use inf_ecs::cover::{classify, CoverClass, CoverSide, MIN_COVER_HEIGHT_M};

/// **Why a cover probe found nothing** — a value, with the thing it looked at
/// named where there was one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoverRefusal {
    /// It found a surface and that surface is cover.
    #[default]
    None,
    /// The character is not facing anywhere (a zero forward, a non-finite
    /// position).
    NoFacing,
    /// The forward sweep hit nothing within reach, or hit only floors.
    NoSurface,
    /// A surface, but its face turns away from the approach by more than
    /// [`CoverSettings::approach_deg`] — a wall you are running ALONG is not a
    /// wall you are taking cover behind.
    FaceTurnedAway,
    /// A surface, and it is one of the families that is never cover: the ground,
    /// a kerb slab, a door leaf. See [`ColliderFamily::is_coverable`].
    NotCoverable,
    /// A surface whose top is below [`CoverSettings::min_height_m`]. **The
    /// kerb's refusal**, when the kerb is a box the grammar drew rather than a
    /// labelled slab.
    TooLow,
    /// A surface narrower than the character's own capsule. A lamp post is not
    /// cover, however tall it is.
    TooNarrow,
    /// A surface with no room for the character to stand at its standoff — it is
    /// in a gap it cannot fit into.
    NoRoom,
}

impl CoverRefusal {
    /// The sentence a failure message uses.
    pub fn why(self) -> &'static str {
        match self {
            CoverRefusal::None => "it is cover",
            CoverRefusal::NoFacing => "the character is not facing anywhere",
            CoverRefusal::NoSurface => "nothing in front within reach",
            CoverRefusal::FaceTurnedAway => "the face turns away from the approach",
            CoverRefusal::NotCoverable => "that family is never cover",
            CoverRefusal::TooLow => "its top is below the cover floor",
            CoverRefusal::TooNarrow => "narrower than the character's own capsule",
            CoverRefusal::NoRoom => "no room to stand at the standoff",
        }
    }
}

/// How the cover probe reaches, in metres and degrees.
///
/// Every number is here rather than in the probe's body, for
/// [`super::traversal::LedgeSettings`]' reason: a project that wants a taller
/// cover floor or a longer reach retunes one struct, and a gate that wants to
/// falsify a threshold mutates one line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverSettings {
    /// The bottom of the band the forward sweep covers, metres above the feet.
    ///
    /// **Below the ledge probe's own floor on purpose.** `LedgeSettings`'
    /// `min_height_m` is 0.50 m because anything lower is a step the autostep
    /// owns; a cover surface at 0.55 m is a thing to crouch behind and the
    /// sweep has to be able to *see* it before the classifier can refuse it.
    pub band_low_m: f64,
    /// The top of that band, metres above the feet. Head height and a little
    /// more: a sweep that reached three metres would find the underside of a
    /// balcony and call it a wall.
    pub band_high_m: f64,
    /// How far forward the character reaches for a surface, metres.
    pub reach_m: f64,
    /// Radius of the forward sweeping capsule, metres. Shared with the ledge
    /// probe's own, because a cover surface a mantle cannot see is a surface the
    /// vault-from-cover could never take.
    pub forward_radius_m: f64,
    /// Radius of the downward sweeping sphere, metres.
    pub down_radius_m: f64,
    /// The shortest surface that is cover, metres above the feet. **A kerb is
    /// 0.15 m and is not cover**; a car door is 0.8 m and is.
    pub min_height_m: f64,
    /// How high the top sweep looks before it gives up and calls the surface
    /// tall, metres.
    pub max_top_m: f64,
    /// How far off head-on the surface's face may be and still be cover,
    /// degrees.
    pub approach_deg: f64,
    /// How far the character's capsule SURFACE sits off the cover surface,
    /// metres — the standoff. The capsule centre is this plus its radius.
    pub standoff_m: f64,
    /// How far along the surface the extent search looks each way, metres.
    pub extent_max_m: f64,
    /// How many coarse samples that search takes each way. The cost knob: the
    /// search is `2 * (extent_samples + EXTENT_REFINEMENTS)` forward sweeps and
    /// it runs only once a surface has been classified as cover.
    pub extent_samples: u32,
}

impl Default for CoverSettings {
    fn default() -> Self {
        Self {
            band_low_m: 0.35,
            band_high_m: 1.90,
            reach_m: 0.90,
            forward_radius_m: 0.30,
            down_radius_m: 0.30,
            min_height_m: MIN_COVER_HEIGHT_M,
            max_top_m: 2.50,
            approach_deg: 45.0,
            standoff_m: 0.06,
            extent_max_m: 4.0,
            extent_samples: 8,
        }
    }
}

/// How many bisection steps the extent search spends refining each edge.
///
/// Three, which takes a 0.5 m coarse step down to 6.25 cm — inside the 10 cm the
/// corner-stop arm asserts against, and cheap: three casts a side.
pub const EXTENT_REFINEMENTS: u32 = 3;

/// **What the probe found.** Always a value; `class` is
/// [`CoverClass::None`] when it found nothing and `refusal` says why.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverProbe {
    /// What it found.
    pub class: CoverClass,
    /// Why not, when it found nothing.
    pub refusal: CoverRefusal,
    /// The point on the cover's face the probe reached, world metres.
    pub point: DVec3,
    /// That face's outward normal — it points back AT the character.
    pub normal: DVec3,
    /// Where the character's FEET go: the face point, out along the normal by
    /// the capsule radius plus the standoff, at the feet's own height.
    pub anchor: DVec3,
    /// How high the surface's top is above the character's feet, metres.
    /// [`f64::INFINITY`] when there is no top within
    /// [`CoverSettings::max_top_m`], which is what makes a wall a wall.
    pub top_m: f64,
    /// The yaw the character faces while in cover: **into** the surface, exactly
    /// as a mantle faces into its ledge.
    pub yaw_deg: f64,
    /// How far the surface runs to the character's LEFT before it ends, metres,
    /// measured from [`anchor`](Self::anchor) along the face tangent, with the
    /// sweeper's own radius taken back off so it is the CORNER's distance and
    /// not the reach of the thing that found it. Capped at
    /// [`CoverSettings::extent_max_m`], which means "further than we looked".
    pub left_m: f64,
    /// The same to the right.
    pub right_m: f64,
    /// What it is — the family, the block and the ordinal (carried 162).
    pub label: ColliderLabel,
    /// The collider the face belongs to, so a caller can exclude it or compare
    /// it against a weapon ray's own hit.
    pub collider: Option<ColliderId3D>,
    /// **How many shape casts this probe spent.** The budget arm's number: zero
    /// on a step that never asked.
    pub sweeps: u32,
}

impl Default for CoverProbe {
    fn default() -> Self {
        Self {
            class: CoverClass::None,
            refusal: CoverRefusal::NoSurface,
            point: DVec3::ZERO,
            normal: DVec3::Z,
            anchor: DVec3::ZERO,
            top_m: 0.0,
            yaw_deg: 0.0,
            left_m: 0.0,
            right_m: 0.0,
            label: ColliderLabel::entity(Uuid::nil()),
            collider: None,
            sweeps: 0,
        }
    }
}

impl CoverProbe {
    /// A refusal with a reason and nothing else.
    fn refused(refusal: CoverRefusal, sweeps: u32) -> Self {
        Self {
            refusal,
            sweeps,
            ..Self::default()
        }
    }

    /// **Say what happened, in words** — the failure message every cover arm
    /// prints, with the thing it looked at NAMED.
    pub fn explain(&self, world: &inf_ecs::world::EcsWorld) -> String {
        match self.class {
            CoverClass::None => format!(
                "no cover: {} (looked at {}), {} sweeps",
                self.refusal.why(),
                self.label.describe(world),
                self.sweeps
            ),
            c => format!(
                "{:?} cover on {} -- top {:.4} m, extents L {:.3} m / R {:.3} m, {} sweeps",
                c,
                self.label.describe(world),
                self.top_m,
                self.left_m,
                self.right_m,
                self.sweeps
            ),
        }
    }
}

/// **THE COVER PROBE.**
///
/// `feet` is the character's ground point, `forward` the direction it is
/// reaching in (its facing, or the stick direction when one is held), `radius` /
/// `half_height` its capsule, and `exclude` the colliders it must not see — its
/// own, and every other character's, which is the ledge probe's `IgnoreOnlyPawn`
/// exactly.
///
/// The five steps, and each one can refuse:
///
/// 1. **The face.** [`super::traversal::sweep_forward_face`] across the cover
///    band. A walkable face is the floor and is refused there.
/// 2. **The approach.** The face's normal must oppose `forward` within
///    [`CoverSettings::approach_deg`].
/// 3. **The family.** [`ColliderFamily::is_coverable`] — the ground, a kerb and
///    a door leaf are refused by NAME, which is what carried 162's label door
///    bought.
/// 4. **The top.** [`super::traversal::sweep_surface_top`], and
///    [`classify`] over what it answered.
/// 5. **The extents.** Lateral forward sweeps out to
///    [`CoverSettings::extent_max_m`] each way, bisected
///    [`EXTENT_REFINEMENTS`] times, so the corners a slide must stop at are
///    measured rather than guessed. Runs only when steps 1–4 said cover.
#[allow(clippy::too_many_arguments)]
pub fn probe_cover(
    bridge: &mut PhysicsBridge3D,
    feet: DVec3,
    forward: DVec3,
    radius: f64,
    half_height: f64,
    slope_limit_deg: f64,
    settings: &CoverSettings,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
) -> CoverProbe {
    let fwd = DVec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
    if fwd == DVec3::ZERO || !feet.is_finite() {
        return CoverProbe::refused(CoverRefusal::NoFacing, 0);
    }
    let centre = (settings.band_high_m + settings.band_low_m) * 0.5;
    let span = (settings.band_high_m - settings.band_low_m) * 0.5;
    let mut sweeps = 0u32;

    // ── 1. the face, through the mantle's own door.
    sweeps += 1;
    let Some(face) = sweep_forward_face(
        bridge.world_mut(),
        feet,
        fwd,
        centre,
        span,
        settings.reach_m,
        settings.forward_radius_m,
        slope_limit_deg,
        exclude,
    ) else {
        return CoverProbe::refused(CoverRefusal::NoSurface, sweeps);
    };
    let label = bridge.label_of(face.collider).unwrap_or_default();

    // ── 2. the approach. `-fwd` is the direction the character came FROM, and a
    //    cover face has to point that way: a wall the character is running
    //    ALONG is not a wall it is taking cover behind.
    let cos_limit = inf_math::pcos64(settings.approach_deg.to_radians());
    let normal = face.normal.normalize_or_zero();
    let facing = DVec3::new(normal.x, 0.0, normal.z).normalize_or_zero();
    if facing == DVec3::ZERO || (-fwd).dot(facing) < cos_limit {
        return CoverProbe {
            label,
            collider: Some(face.collider),
            ..CoverProbe::refused(CoverRefusal::FaceTurnedAway, sweeps)
        };
    }

    // ── 3. the family, by NAME (carried 162).
    if !label.family.is_coverable() {
        return CoverProbe {
            label,
            collider: Some(face.collider),
            ..CoverProbe::refused(CoverRefusal::NotCoverable, sweeps)
        };
    }

    // ── 4. the top, and the class.
    sweeps += 1;
    let top = sweep_surface_top(
        bridge.world_mut(),
        feet.y,
        face.point,
        normal,
        settings.max_top_m,
        settings.down_radius_m,
        slope_limit_deg,
        exclude,
    );
    let top_m = match top {
        Some(p) => p.y - feet.y,
        // No walkable top within the look: it is taller than we care about,
        // which is exactly what a wall is.
        None => f64::INFINITY,
    };
    let class = classify(top_m, settings.min_height_m);
    if !class.is_cover() {
        return CoverProbe {
            label,
            collider: Some(face.collider),
            top_m,
            ..CoverProbe::refused(CoverRefusal::TooLow, sweeps)
        };
    }

    // **Where the character's feet end up**: exactly where they are, moved along
    // the face's own normal until the capsule's surface is `standoff_m` off it.
    //
    // The correction is along the NORMAL only, and that is the whole of it. The
    // first cut anchored at the face's witness point instead — which reads as
    // "against the wall, where the probe touched it" and is wrong for a reason
    // that took a step-by-step trace to see: a capsule swept against a box
    // reports its witness at whatever feature of the manifold the narrow phase
    // picked, which for a flat wall is a corner metres away. The anchor then
    // walked sideways a few centimetres per step under a character that was
    // standing still, and a slide along a 1.5 m wall drifted 1.75 m past its
    // end before the extents stopped moving with it. Measured; the fix is this
    // line.
    let depth = (feet - face.point).dot(facing);
    let anchor = feet + facing * (radius + settings.standoff_m - depth);
    let yaw_deg = inf_math::patan2_64(-facing.x, -facing.z).to_degrees();

    // ── 4b. room. The character's own capsule must fit where it is going —
    //    `probe_ledge`'s step 3, one place along.
    sweeps += 1;
    let centre_at = anchor + DVec3::Y * (half_height + radius);
    if bridge
        .world_mut()
        .cast_shape(
            &super::ColliderShape3D::Capsule {
                half_height: half_height.max(1e-3),
                radius: radius.max(1e-3),
            },
            centre_at,
            glam::DQuat::IDENTITY,
            DVec3::Y,
            1e-3,
            exclude,
        )
        .is_some_and(|h| h.started_penetrating)
    {
        return CoverProbe {
            label,
            collider: Some(face.collider),
            top_m,
            ..CoverProbe::refused(CoverRefusal::NoRoom, sweeps)
        };
    }

    // ── 5. the extents. The tangent is the face's own, in the ground plane: the
    //    character's LEFT while it faces INTO the wall.
    //
    //    Spelled once, in `inf_ecs::cover::tangent_left`, because the movement
    //    step derives the same vector every step to slide along and a second
    //    spelling is a left that is a right in one of the two readers.
    let tangent = inf_ecs::cover::tangent_left(inf_ecs::math::Vec3d::from_dvec3(facing)).to_dvec3();
    let (left_m, left_sweeps) = extent_along(
        bridge,
        anchor,
        fwd,
        tangent,
        centre,
        span,
        settings,
        slope_limit_deg,
        exclude,
    );
    let (right_m, right_sweeps) = extent_along(
        bridge,
        anchor,
        fwd,
        -tangent,
        centre,
        span,
        settings,
        slope_limit_deg,
        exclude,
    );
    sweeps += left_sweeps + right_sweeps;

    // A surface narrower than the capsule it is meant to hide is a lamp post.
    if left_m + right_m < radius * 2.0 {
        return CoverProbe {
            label,
            collider: Some(face.collider),
            top_m,
            left_m,
            right_m,
            ..CoverProbe::refused(CoverRefusal::TooNarrow, sweeps)
        };
    }

    CoverProbe {
        class,
        refusal: CoverRefusal::None,
        point: face.point,
        normal: facing,
        anchor,
        top_m,
        yaw_deg,
        left_m,
        right_m,
        label,
        collider: Some(face.collider),
        sweeps,
    }
}

/// **How far the surface runs in one direction** before the forward sweep stops
/// finding it, metres — and how many sweeps that cost.
///
/// A coarse march out to [`CoverSettings::extent_max_m`] followed by
/// [`EXTENT_REFINEMENTS`] bisections of the last good step, so the answer is
/// within `extent_max / samples / 2^refinements` of the true edge — 6.25 cm on
/// the shipped numbers, which is inside the 10 cm the corner arm asserts.
///
/// The whole extent is reported when nothing ended within the look; a caller
/// reads that as "further than we measured", which is the right answer for a
/// city block's wall.
#[allow(clippy::too_many_arguments)]
fn extent_along(
    bridge: &mut PhysicsBridge3D,
    anchor: DVec3,
    fwd: DVec3,
    tangent: DVec3,
    centre: f64,
    span: f64,
    settings: &CoverSettings,
    slope_limit_deg: f64,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
) -> (f64, u32) {
    let samples = settings.extent_samples.max(1);
    let step = settings.extent_max_m / f64::from(samples);
    let mut sweeps = 0u32;
    let still_there = |bridge: &mut PhysicsBridge3D, d: f64, sweeps: &mut u32| -> bool {
        *sweeps += 1;
        sweep_forward_face(
            bridge.world_mut(),
            anchor + tangent * d,
            fwd,
            centre,
            span,
            settings.reach_m,
            settings.forward_radius_m,
            slope_limit_deg,
            exclude,
        )
        .is_some()
    };
    let mut good = 0.0f64;
    let mut bad = f64::NAN;
    for i in 1..=samples {
        let d = step * f64::from(i);
        if still_there(bridge, d, &mut sweeps) {
            good = d;
        } else {
            bad = d;
            break;
        }
    }
    if !bad.is_finite() {
        // Never ended: the wall runs further than we looked.
        return (settings.extent_max_m, sweeps);
    }
    for _ in 0..EXTENT_REFINEMENTS {
        let mid = (good + bad) * 0.5;
        if still_there(bridge, mid, &mut sweeps) {
            good = mid;
        } else {
            bad = mid;
        }
    }
    // **The sweeper's own radius comes back off.** The bisection finds the
    // offset at which a capsule of `forward_radius_m` stops touching the
    // surface, which is the corner PLUS that radius — so reporting `good` would
    // over-measure every extent by 30 cm and a slide clamped against it would
    // stop with the character's centre exactly on the corner and its whole
    // outboard half hanging over the drop. Measured before this line: a 1.5 m
    // wall reported 1.75 m of extent.
    ((good - settings.forward_radius_m).max(0.0), sweeps)
}

/// **Is this face still the same cover** — the question a slide asks every step.
///
/// A convenience over [`probe_cover`] for the caller that already has a probe:
/// the class may CHANGE along a surface (a car's flank into its roof line is
/// `Low` into `High`, which is the re-classification clause 3 names) and the
/// caller wants to know that without re-deciding what cover means.
pub fn reclassify(previous: CoverClass, now: &CoverProbe) -> CoverClass {
    if now.class.is_cover() {
        now.class
    } else {
        // The probe lost the surface for a step (a doorway, a sample between two
        // solids): hold what we had rather than dropping the character out of
        // cover on one frame of geometry. Leaving cover is a decision the
        // movement step makes on the stick and the button, not a thing a missed
        // sweep does.
        previous
    }
}

/// Whether `normal` is a floor, re-exported so a caller measuring a cover
/// surface by hand uses the same test the probe did.
pub fn face_is_floor(normal: DVec3, slope_limit_deg: f64) -> bool {
    is_walkable(normal, slope_limit_deg)
}

/// The family a probe found, for a caller that wants to print it without
/// resolving a name.
pub fn family_of(probe: &CoverProbe) -> ColliderFamily {
    probe.label.family
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal keeps the class it had rather than dropping cover on one
    /// missed sweep; a probe that found something replaces it.
    #[test]
    fn a_missed_sweep_does_not_leave_cover() {
        let held = CoverClass::Low;
        let lost = CoverProbe::refused(CoverRefusal::NoSurface, 1);
        assert_eq!(reclassify(held, &lost), CoverClass::Low);
        let found = CoverProbe {
            class: CoverClass::High,
            ..CoverProbe::default()
        };
        assert_eq!(reclassify(held, &found), CoverClass::High);
    }

    /// Every refusal has a sentence, and no two of them share one — a message
    /// that could mean two things is a message nobody can act on.
    #[test]
    fn every_refusal_says_something_of_its_own() {
        let all = [
            CoverRefusal::None,
            CoverRefusal::NoFacing,
            CoverRefusal::NoSurface,
            CoverRefusal::FaceTurnedAway,
            CoverRefusal::NotCoverable,
            CoverRefusal::TooLow,
            CoverRefusal::TooNarrow,
            CoverRefusal::NoRoom,
        ];
        let mut said: Vec<&str> = all.iter().map(|r| r.why()).collect();
        said.sort_unstable();
        let n = said.len();
        said.dedup();
        assert_eq!(said.len(), n, "two refusals say the same thing");
    }
}
