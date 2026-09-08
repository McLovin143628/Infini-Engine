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
//! [`super::label::ColliderLabel`] of the thing it looked at.
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
    /// A surface that is **moving**. A parked car is cover; one pulling away is
    /// not, and a character glued to its flank would be dragged down the street.
    ///
    /// This is where ALS's "do not mantle a moving platform" rule lives for the
    /// cover probe. The ledge probe expresses it as a broad-phase filter
    /// (`CastTargets::Fixed`) and a cover probe cannot: the island's parked
    /// cars are DYNAMIC bodies, and filtering them out made every one of them
    /// invisible — 247 cover surfaces in the wave's census and not a car among
    /// them. So the question is asked directly, of the body's own velocity.
    Moving,
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
            CoverRefusal::Moving => "the surface is moving",
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
    /// **How fast a surface may be moving and still be cover**, m/s.
    ///
    /// A parked car is cover and one pulling away is not. Ten centimetres a
    /// second: a rigid body settled on its suspension jitters below this and a
    /// vehicle under power is past it inside one step.
    pub max_surface_speed_mps: f64,
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
            max_surface_speed_mps: 0.10,
        }
    }
}

/// **How many times the forward sweep is retaken past a MOVING surface.**
///
/// Two. A car driving between a character and the wall it is pressing against is
/// one skip; two moving things in a row is a street the press should simply
/// refuse. The bound is what keeps the P22.3 M4 remedy from being unbounded —
/// the whole reason that audit preferred a broad-phase filter is that a
/// post-hoc one can retry for ever.
pub const MOVING_RETRIES: u32 = 2;

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
    /// **The face normal as an `inf_ecs` vector**, so a caller can hand it to
    /// [`inf_ecs::cover::tangent_left`] without spelling the conversion.
    pub fn normal_v(&self) -> inf_ecs::math::Vec3d {
        inf_ecs::math::Vec3d::from_dvec3(self.normal)
    }

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

    // ── 1. the face, through the mantle's own door — and 2b's retry, which is
    //    why this is a loop rather than one sweep.
    //
    //    **A MOVING surface is skipped, not refused** (the P22.3 audit's M4).
    //    `cast_shape_where`'s own doc states the rule this obeys: filtering
    //    *after* a cast hides whatever was behind the rejected hit, so a car
    //    driving past a wall would make the WALL un-cover-able rather than the
    //    car. The ledge probe avoids it by asking the broad phase for `Fixed`
    //    only — which a cover probe cannot do, because a parked car IS a
    //    dynamic body and filtering the class out made every car on the island
    //    invisible (the wave's census: 247 cover surfaces, not one of them a
    //    car). So the moving hit is added to the exclusion set and the sweep is
    //    taken again, at most [`MOVING_RETRIES`] times, and the thing behind it
    //    gets its turn.
    let cos_limit = inf_math::pcos64(settings.approach_deg.to_radians());
    let mut skip: std::collections::BTreeSet<ColliderId3D> = exclude.clone();
    let mut moving_label: Option<ColliderLabel> = None;
    let mut found: Option<(super::ShapeHit3D, ColliderLabel, DVec3)> = None;
    for _ in 0..=MOVING_RETRIES {
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
            &skip,
            super::CastTargets::All,
        ) else {
            break;
        };
        let label = bridge.label_of(face.collider).unwrap_or_default();
        let normal = face.normal.normalize_or_zero();
        let facing = DVec3::new(normal.x, 0.0, normal.z).normalize_or_zero();
        // The surface is not driving away. A collider with no body at all —
        // every static structure — answers zero and passes.
        let speed = bridge
            .guid_of_collider(face.collider)
            .and_then(|g| bridge.body_of(g))
            .and_then(|b| bridge.world_mut().body_linvel(b))
            .map(|v| v.length())
            .unwrap_or(0.0);
        if speed > settings.max_surface_speed_mps {
            moving_label = Some(label);
            skip.insert(face.collider);
            continue;
        }
        found = Some((face, label, facing));
        break;
    }
    let Some((face, label, facing)) = found else {
        // Nothing at all, or nothing that was not moving. The two are different
        // answers and the refusal says which.
        return match moving_label {
            Some(l) => CoverProbe {
                label: l,
                ..CoverProbe::refused(CoverRefusal::Moving, sweeps)
            },
            None => CoverProbe::refused(CoverRefusal::NoSurface, sweeps),
        };
    };

    // ── 2. the approach. `-fwd` is the direction the character came FROM, and a
    //    cover face has to point that way: a wall the character is running
    //    ALONG is not a wall it is taking cover behind.
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
        // The face's own PLANAR normal: the top sweep steps into the surface
        // along it, and stepping along a normal with a vertical component would
        // walk the sample up or down the face rather than into it.
        facing,
        settings.max_top_m,
        settings.down_radius_m,
        slope_limit_deg,
        exclude,
        super::CastTargets::All,
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
            super::CastTargets::All,
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
    ///
    /// **The list is exhaustive by construction** (the COV1 audit): it was a
    /// hand-written array of eight and the enum has nine, so `Moving` — the
    /// variant the census's own two refusals are — was never compared against
    /// anything and a tenth variant would have been just as invisible. The
    /// `match` below has no wildcard, so adding a refusal is a compile error
    /// here rather than a silent gap.
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
            CoverRefusal::Moving,
        ];
        for r in all {
            // The exhaustiveness itself: a new variant that is not added to the
            // array above cannot reach this `match` without naming itself.
            let named = match r {
                CoverRefusal::None => "None",
                CoverRefusal::NoFacing => "NoFacing",
                CoverRefusal::NoSurface => "NoSurface",
                CoverRefusal::FaceTurnedAway => "FaceTurnedAway",
                CoverRefusal::NotCoverable => "NotCoverable",
                CoverRefusal::TooLow => "TooLow",
                CoverRefusal::TooNarrow => "TooNarrow",
                CoverRefusal::NoRoom => "NoRoom",
                CoverRefusal::Moving => "Moving",
            };
            assert!(
                all.iter().any(|x| format!("{x:?}") == named),
                "{named} is a refusal the sameness list has never seen"
            );
            assert!(!r.why().is_empty(), "{named} says nothing at all");
        }
        let mut said: Vec<&str> = all.iter().map(|r| r.why()).collect();
        said.sort_unstable();
        let n = said.len();
        said.dedup();
        assert_eq!(said.len(), n, "two refusals say the same thing");
    }
}

// ── NPCs IN COVER (wave COV1, clause 5) ─────────────────────────────────────

use inf_ecs::components::{CharacterMovement, MovementMode, Transform};
use inf_ecs::cover::NpcCoverRes;
use inf_ecs::math::Vec3d;
use inf_ecs::world::EcsWorld;

/// **What one step of the NPC cover pass did.**
///
/// Every field is an *engagement* counter rather than a "the pass ran" flag —
/// the EMS2 law: a gate that cannot tell "the officers held" from "no officer
/// was ever in the radius" certifies a no-op.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NpcCoverReport {
    /// Responders the pass looked at — the cost.
    pub considered: usize,
    /// Responders inside a shot's radius this step.
    pub under_fire: usize,
    /// Units that ran a cover SEARCH this step (the expensive half).
    pub searched: usize,
    /// Units walking to a cover anchor they found.
    pub moving: usize,
    /// Units in [`MovementMode::Cover`] right now.
    pub in_cover: usize,
    /// Units leaned out of it right now.
    pub peeking: usize,
    /// **Shape casts this pass spent.** ZERO on a step with no gunfire, which is
    /// the budget arm's own number and the reason it is counted.
    pub probes: u32,
    /// Units that took HIGH cover because the town is at its top rung.
    pub swat_high: usize,
}

/// **THE NPC COVER PASS.**
///
/// One pass, called by both hosts from the gameplay step, over the responders
/// [`inf_ecs::dispatch::RespondersRes`] names.
///
/// * a responder inside a shot's radius is **under fire**
///   ([`inf_ecs::cover::under_fire`] — EMS2's own exemption predicate, read the
///   other way round);
/// * one that is not already in cover **searches** for it, on the compass, at
///   most once every [`inf_ecs::cover::NPC_SEARCH_PERIOD`] steps;
/// * what it finds it **walks to**, along an `inf_nav::NavPath`, and presses
///   the same `press_cover` edge a player's key raises — one door, two callers;
/// * once in cover it **peeks** on a duty cycle and points its **body's aim**
///   at the threat — `MovementRuntime::aim_yaw_deg` and `want_aim`, the same
///   two fields a player's right mouse button writes.
///
/// # The seam, said exactly (the COV1 audit)
///
/// This pass does **not** call [`super::gameplay::npc_aim_at`], and an earlier
/// spelling of this list said it did. It cannot: `npc_aim_at` takes the
/// **guid** of a target and this pass is handed the *places* a step's gunfire
/// came from ([`super::gameplay::panic_sources_for`]'s own coalesced list), so
/// there is no shooter entity here to name. What the pass hands WPN2e is a unit
/// standing behind cover, turned toward the threat, leaning out on a duty cycle
/// — and WPN2e resolves the shooter, calls `npc_aim_at` and owns the trigger.
///
/// # The path is two points, and that is stated rather than hidden
///
/// `NavPath::new([here, anchor])` — a straight leg, arc-length parameterized,
/// which is what `CrowdRoute::between` is and what every tier measurement in
/// this tree is written against. A searched route over the street graph is what
/// `inf_ecs::crowd`'s own routes carry and what a unit *driving* to an incident
/// already gets; an officer crossing twenty metres of pavement to a wall is a
/// straight leg in every reference frame this campaign has looked at, and
/// building a search here would be a wave rather than a clause.
///
/// # The cost, and the zero
///
/// `O(responders)` on a step with gunfire and **one resource lookup** on every
/// step without any — the sources are empty, the loop is not entered, and
/// [`NpcCoverReport::probes`] is zero. A level with no responders pays the same
/// nothing.
pub fn step_npc_cover(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    sources: &[DVec3],
    radius_m: f64,
    step: u64,
    dt: f64,
) -> NpcCoverReport {
    let mut report = NpcCoverReport::default();
    let responders = inf_ecs::dispatch::responders(world);
    if responders.is_empty() {
        // A level with no responders keeps no state either — the crowd's own
        // prune rule, so a session that despawned its units does not carry them.
        if world.world().get_resource::<NpcCoverRes>().is_some() {
            world.world_mut().insert_resource(NpcCoverRes::default());
        }
        return report;
    }
    report.considered = responders.len();
    let prefer_high = inf_ecs::cover::prefers_high(hottest_response(world));
    let mut res = world
        .world_mut()
        .remove_resource::<NpcCoverRes>()
        .unwrap_or_default();
    // A unit that has gone away forgets what it was doing, so a guid that comes
    // back does not inherit a stale anchor.
    let live: std::collections::BTreeSet<uuid::Uuid> = responders.iter().copied().collect();
    res.units.retain(|g, _| live.contains(g));

    let settings = CoverSettings::default();
    for unit in responders {
        let Some(here) = body_at(world, unit) else {
            res.units.remove(&unit);
            continue;
        };
        let threat = inf_ecs::cover::under_fire(
            Vec3d::from_dvec3(here),
            &sources
                .iter()
                .map(|s| Vec3d::from_dvec3(*s))
                .collect::<Vec<_>>(),
            radius_m,
        );
        let in_cover = mode_of(world, unit) == Some(MovementMode::Cover);
        let Some(threat) = threat else {
            // **Not under fire: nothing at all happens.** No probe, no path, no
            // state — which is the budget arm's whole claim, and the reason the
            // early return is here rather than inside the search.
            if !in_cover {
                res.units.remove(&unit);
            }
            continue;
        };
        let threat: Vec3d = threat;
        report.under_fire += 1;
        let slot = res.units.entry(unit).or_default();
        slot.threat = threat;

        if in_cover {
            report.in_cover += 1;
            slot.drop_leg();
            slot.peek_s += dt;
            let (out, _) = inf_ecs::cover::peek_cycle(slot.peek_s);
            slot.peeking = out;
            if out {
                report.peeking += 1;
            }
            // The body's own aim goes to the threat, and the trigger stays OPEN.
            // **WPN2e wires the firing** — this wave points and leans.
            aim_and_lean(world, unit, threat.to_dvec3(), out);
            continue;
        }

        // ── the SEARCH, at most once every `NPC_SEARCH_PERIOD` steps.
        let stale = step.saturating_sub(slot.searched_step) >= inf_ecs::cover::NPC_SEARCH_PERIOD;
        let has_anchor = slot.path.is_some();
        if !has_anchor && stale {
            slot.searched_step = step;
            report.searched += 1;
            let (found, spent) = search_for_cover(
                world,
                bridge,
                unit,
                here,
                threat.to_dvec3(),
                prefer_high,
                &settings,
            );
            report.probes += spent;
            if let Some((anchor, face_yaw, high)) = found {
                if high && prefer_high {
                    report.swat_high += 1;
                }
                slot.lay_leg(Vec3d::from_dvec3(here), Vec3d::from_dvec3(anchor), face_yaw);
            }
        }

        // ── walk the leg, and press the same key a player would.
        if slot.walking() {
            report.moving += 1;
            let anchor = slot.anchor.to_dvec3();
            let gap = ((anchor.x - here.x).powi(2) + (anchor.z - here.z).powi(2)).sqrt();
            // **The arrival distance is the PROBE's reach, not the snap's.**
            //
            // A unit that pressed anywhere inside `MAX_SNAP_M` (1.5 m) would
            // press from 1.2 m back — where the surface is 1.6 m away and
            // `CoverSettings::reach_m` is 0.9, so the press probes, finds
            // nothing, and the unit searches, walks nowhere and presses again.
            // Measured on the fixture: an officer parked at 1.23 m from its own
            // anchor for the whole fifteen seconds, spending 1 961 shape casts.
            // Half the probe's reach is inside what the press can see with room
            // for the last step's overshoot.
            if gap <= settings.reach_m * 0.5 {
                // Close enough for the press to take it the rest of the way.
                press_cover(world, unit, slot.face_yaw_deg);
                slot.drop_leg();
                slot.searched_step = step;
            } else if let Some(dir) = slot.heading_from(Vec3d::from_dvec3(here)) {
                walk_toward(world, unit, dir.to_dvec3());
            }
        }
    }
    world.world_mut().insert_resource(res);
    report
}

/// The hottest response rung any open profile is at — `Cold` when nobody is
/// wanted, which is every level before somebody does something.
fn hottest_response(world: &EcsWorld) -> inf_ecs::crime::Response {
    inf_ecs::crime::wanted(world)
        .into_iter()
        .filter_map(|s| inf_ecs::crime::profile_of(world, s).map(|p| p.response()))
        .max()
        .unwrap_or(inf_ecs::crime::Response::Cold)
}

/// Where a body is, world metres.
fn body_at(world: &EcsWorld, guid: uuid::Uuid) -> Option<DVec3> {
    let e = world.entity_of(guid)?;
    let p = world.world().get::<Transform>(e)?.translation.to_dvec3();
    p.is_finite().then_some(p)
}

fn mode_of(world: &EcsWorld, guid: uuid::Uuid) -> Option<MovementMode> {
    let e = world.entity_of(guid)?;
    Some(world.world().get::<CharacterMovement>(e)?.mode)
}

/// **Look for cover on the compass**, and answer where its anchor is and
/// whether it is HIGH — plus what the look cost in shape casts.
///
/// The bearings are the eight the CHAR1b.2 census used. Each one probes from the
/// unit's own feet, and the candidates are scored by **how much they put between
/// the unit and the threat**: a surface whose face points back at the shooter is
/// cover, and one on the far side of the unit is a wall to be shot against.
///
/// `prefer_high` is the SWAT behaviour: with it, a `High` candidate wins over
/// any `Low` one regardless of distance, and without it the nearest wins.
#[allow(clippy::too_many_arguments)]
fn search_for_cover(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    unit: uuid::Uuid,
    here: DVec3,
    threat: DVec3,
    prefer_high: bool,
    settings: &CoverSettings,
) -> (Option<(DVec3, f64, bool)>, u32) {
    let Some(e) = world.entity_of(unit) else {
        return (None, 0);
    };
    let (radius, half) = {
        let w = world.world();
        let cm = match w.get::<CharacterMovement>(e) {
            Some(cm) => cm,
            None => return (None, 0),
        };
        let r = w
            .get::<inf_ecs::components::Collider3D>(e)
            .map(|c| c.radius)
            .unwrap_or(0.3);
        (r, cm.half_height_for(MovementMode::Grounded))
    };
    let slope = world
        .world()
        .get::<CharacterMovement>(e)
        .map(|c| c.slope_limit_deg)
        .unwrap_or(50.0);
    let feet = here - DVec3::Y * (half + radius);
    let mut exclude = std::collections::BTreeSet::new();
    if let Some(c) = bridge.collider_of(unit) {
        exclude.insert(c);
    }
    let to_threat = DVec3::new(threat.x - here.x, 0.0, threat.z - here.z).normalize_or_zero();
    let mut spent = 0u32;
    let mut best: Option<(f64, DVec3, f64, bool)> = None;
    let samples =
        (inf_ecs::cover::NPC_COVER_SEARCH_M / inf_ecs::cover::NPC_SEARCH_STEP_M).ceil() as i32;
    for i in 0..inf_ecs::cover::NPC_SEARCH_BEARINGS {
        let a = std::f64::consts::TAU * i as f64 / inf_ecs::cover::NPC_SEARCH_BEARINGS as f64;
        let dir = DVec3::new(inf_math::psin64(a), 0.0, inf_math::pcos64(a));
        // **Between the unit and the threat**: the surface has to be on the
        // side the shooting is coming from, or the unit is hiding behind
        // something with its back to the gun. Roughly half the compass is
        // dropped here, which is half the cost.
        if to_threat != DVec3::ZERO && dir.dot(to_threat) < 0.0 {
            continue;
        }
        // **March along the bearing.** The probe answers about the place it is
        // standing and its reach is under a metre, so a search that asked once
        // from the unit's own feet could only ever find cover the unit was
        // already touching -- which is what the first cut did, and what the
        // fixture measured as 116 shape casts and no cover at all.
        for k in 0..=samples {
            let d = f64::from(k) * inf_ecs::cover::NPC_SEARCH_STEP_M;
            let at = feet + dir * d;
            let p = probe_cover(bridge, at, dir, radius, half, slope, settings, &exclude);
            spent += p.sweeps;
            if !p.class.is_cover() {
                continue;
            }
            let reach = ((p.anchor.x - feet.x).powi(2) + (p.anchor.z - feet.z).powi(2)).sqrt();
            if reach > inf_ecs::cover::NPC_COVER_SEARCH_M {
                break;
            }
            let high = p.class == CoverClass::High;
            // The score: distance, with a HIGH candidate given a large discount
            // when the town is at its top rung. A number rather than a branch so
            // the preference is a strength and not a veto -- a SWAT unit with
            // only a car beside it still takes the car.
            let score = if prefer_high && high {
                reach - 1000.0
            } else {
                reach
            };
            if best.is_none_or(|(b, _, _, _)| score < b) {
                best = Some((score, p.anchor, p.yaw_deg, high));
            }
            // The nearest surface along this bearing is the one; anything
            // further along it is behind that one.
            break;
        }
    }
    (best.map(|(_, a, y, h)| (a, y, h)), spent)
}

/// Point the unit at where it is going and hold the movement stick that way.
fn walk_toward(world: &mut EcsWorld, unit: uuid::Uuid, dir: DVec3) {
    let Some(e) = world.entity_of(unit) else {
        return;
    };
    let planar = DVec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
    if planar == DVec3::ZERO {
        return;
    }
    let yaw = inf_math::patan2_64(planar.x, planar.z).to_degrees();
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        // The intent is in the AIM frame, and the aim is being pointed the same
        // way — so "forward" is the whole of it.
        cm.runtime.aim_yaw_deg = yaw;
        cm.runtime.intent_move = inf_ecs::math::Vec2d::new(0.0, 1.0);
        cm.runtime.want_sprint = true;
    }
}

/// **Press the cover key**, exactly as a player's keyboard does.
///
/// The one door: the AI raises the same `press_cover` edge `apply_intent` raises
/// and the movement step answers it identically. An NPC cover system that
/// entered `MovementMode::Cover` by writing the mode would be a second
/// implementation of the thing this wave built, and the two would disagree the
/// first day one of them was tuned.
fn press_cover(world: &mut EcsWorld, unit: uuid::Uuid, face_yaw_deg: f64) {
    let Some(e) = world.entity_of(unit) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        if face_yaw_deg.is_finite() {
            cm.runtime.body_yaw_deg = face_yaw_deg;
            cm.runtime.aim_yaw_deg = face_yaw_deg;
            cm.runtime.target_yaw_deg = face_yaw_deg;
        }
        cm.runtime.intent_move = inf_ecs::math::Vec2d::ZERO;
        cm.runtime.want_sprint = false;
        cm.runtime.press_cover = true;
    }
}

/// Aim at the threat and hold (or release) the peek.
///
/// The **body's** aim, not the weapon's: `aim_yaw_deg` and `want_aim` are the
/// two fields a player's own right mouse button writes, and pointing them is
/// what leans a unit out of cover. `gameplay::npc_aim_at` — the weapon door —
/// is **not** called here and needs a target guid this pass has not got; WPN2e
/// owns that call and the trigger with it. See `step_npc_cover`'s own seam
/// paragraph.
fn aim_and_lean(world: &mut EcsWorld, unit: uuid::Uuid, threat: DVec3, out: bool) {
    let Some(e) = world.entity_of(unit) else {
        return;
    };
    let Some(here) = body_at(world, unit) else {
        return;
    };
    let to = DVec3::new(threat.x - here.x, 0.0, threat.z - here.z);
    if to.length_squared() > 1.0e-12 {
        let yaw = inf_math::patan2_64(to.x, to.z).to_degrees();
        if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.aim_yaw_deg = yaw;
        }
    }
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        // `want_aim` is what the movement step's cover block reads to lean the
        // body out, and it is the same field a player's right mouse button
        // sets — one door again.
        cm.runtime.want_aim = out;
        cm.runtime.intent_move = inf_ecs::math::Vec2d::ZERO;
    }
}
