//! **Traversal detectors** (P29.4, clause 4): the ledge probe a mantle is
//! decided by, and the velocity-aligned sweep that lets the landing classifier
//! run *before* the character touches anything.
//!
//! # The ledge probe is ALS's five steps, and where it differs it says so
//!
//! Port map §3.2. The shape is the donor's: a forward capsule sweep across the
//! band of heights a ledge could be in, a downward sphere sweep to find the top
//! surface, a room check for the character's own capsule at the target, then the
//! height classification. Four things are ours:
//!
//! * the numbers are **metres** (the donor's are centimetres; converted once, at
//!   [`LedgeSettings::default`] and at [`inf_anim::MANTLE_HIGH_SPLIT_M`]);
//! * "do not mantle a moving platform" is a **broad-phase filter**
//!   ([`CastTargets::Fixed`]) rather than a velocity check on the hit, because the
//!   P22.3 audit's M4 is what a downstream filter does — it hides whatever was
//!   behind the thing it rejected;
//! * `IgnoreOnlyPawn` is the caller's **exclusion set**, because a character in
//!   this engine is a set of colliders somebody knows about rather than a body
//!   class rapier can name;
//! * the target is a plain [`Ledge`] value and the *placement* is motion warping
//!   ([`inf_anim::warp`]), so none of ALS's animated-start-offset bookkeeping
//!   exists here.
//!
//! # Everything refuses by value
//!
//! Every function returns `Option`, and every rejection is a `None` with a reason
//! in the code beside it. A traversal probe that could fail a handler would take
//! down a `Tick` body for standing near a wall.

use glam::{DQuat, DVec3};

use super::{CastTargets, ColliderId3D, ColliderShape3D, PhysicsWorld3D};

/// How far the probes reach, in metres.
///
/// ALS's `FALSMantleTraceSettings`, converted once. The port map records that the
/// shipped values live in a Blueprint CDO and were **not** extractable from the
/// binary asset, so these are the documented stock V4 defaults for the *grounded*
/// settings (250 / 50 / 75 / 30 / 30 cm) and they are treated as a starting point
/// rather than as a measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LedgeSettings {
    /// The tallest ledge that may be mantled, metres above the feet.
    pub max_height_m: f64,
    /// The shortest — anything lower is a step, and the mover's autostep owns it.
    pub min_height_m: f64,
    /// How far forward the character reaches for a wall, metres.
    pub reach_m: f64,
    /// Radius of the forward sweeping capsule, metres.
    pub forward_radius_m: f64,
    /// Radius of the downward sweeping sphere, metres.
    pub down_radius_m: f64,
}

impl Default for LedgeSettings {
    fn default() -> Self {
        Self {
            max_height_m: 2.50,
            min_height_m: 0.50,
            reach_m: 0.75,
            forward_radius_m: 0.30,
            down_radius_m: 0.30,
        }
    }
}

impl LedgeSettings {
    /// The **falling** settings: a shorter reach and a lower ceiling, because a
    /// character in the air catching a ledge is a different move from one
    /// climbing off the ground (ALS `FallingTraceSettings`, 150 / 50 / 70 cm).
    pub fn falling() -> Self {
        Self {
            max_height_m: 1.50,
            reach_m: 0.70,
            ..Self::default()
        }
    }
}

/// A ledge the character could mantle onto.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ledge {
    /// Where the character's **feet** end up, world space.
    pub feet: DVec3,
    /// The wall's outward normal at the point that was reached for.
    pub normal: DVec3,
    /// How far above the character's current feet the ledge is, metres.
    pub height_m: f64,
    /// `true` past [`inf_anim::MANTLE_HIGH_SPLIT_M`] — ALS's hard 125 cm split
    /// between the one-metre and two-metre mantle animations.
    pub high: bool,
    /// The yaw the character should face to climb it: **into** the wall.
    pub yaw_deg: f64,
}

/// Whether a surface with this normal is a floor rather than a wall.
pub fn is_walkable(normal: DVec3, slope_limit_deg: f64) -> bool {
    let n = normal.normalize_or_zero();
    if n == DVec3::ZERO {
        return false;
    }
    // `pacos64`, not `f64::acos`: this decides whether a mantle happens, and a
    // mantle writes a Transform.
    inf_math::pacos64(n.y.clamp(-1.0, 1.0)).to_degrees() <= slope_limit_deg
}

/// The compass yaw of a planar direction, degrees, `+Z` zero and `+X` +90 — the
/// convention `inf_ecs::movement::planar_yaw_deg` uses, spelled here because this
/// crate does not depend on that module for one function.
fn planar_yaw_deg(x: f64, z: f64) -> f64 {
    if x.abs() < 1e-12 && z.abs() < 1e-12 {
        return 0.0;
    }
    inf_math::patan2_64(x, z).to_degrees()
}

/// How far BEHIND the character the forward sweep starts, metres.
///
/// ALS's own trick, and the reason it is a named constant rather than a literal
/// in two probes: a character already touching a wall must still *sweep into*
/// it, and a sweep that begins in contact reports `started_penetrating` with no
/// usable normal. Wave COV1's cover probe is the second reader.
pub const SWEEP_BACKOFF_M: f64 = 0.30;

/// How far INTO a surface the downward top-finding sweep steps, metres.
///
/// The sweep lands on the surface rather than on its lip. Shared for
/// [`SWEEP_BACKOFF_M`]'s reason.
pub const TOP_SWEEP_INSET_M: f64 = 0.15;

/// **The forward face sweep** -- step 1 of the ledge probe, and step 1 of wave
/// COV1's cover probe.
///
/// One door, two readers, and the split is where the two questions diverge: a
/// mantle asks *what is the TOP of this thing* and a cover press asks *what is
/// the FACE of it*, but both begin by finding a non-walkable surface in front
/// of the character across a band of heights. The band is the caller's, because
/// it is the only part that differs: a ledge probe sweeps the heights it could
/// climb (0.5-2.5 m) and a cover probe sweeps the heights it could hide behind.
///
/// `centre_m` and `span_m` are the band's centre and half-height above `feet`;
/// `reach_m` is how far forward, measured from the capsule (the
/// [`SWEEP_BACKOFF_M`] behind it is added here, once).
///
/// `None` for a degenerate facing, a sweep that started inside something (no
/// face to read a normal off), or a face that is **walkable** -- a floor is not
/// a wall, and climbing onto the ground you are standing on is not a mantle any
/// more than pressing your back against it is cover (ALS `.cpp:196`).
///
/// # `targets` is the two readers' one disagreement, and it was measured
///
/// A mantle asks [`CastTargets::Fixed`], because ALS's "do not mantle a moving
/// platform" is a broad-phase filter here (see this module's own header) and a
/// character that climbed onto a moving car would arrive somewhere else.
///
/// A cover press asks [`CastTargets::All`], and the reason is a **measurement**:
/// wave COV1's census over the island found 247 cover surfaces in 2 256 probes
/// and **every one of them was a building façade**. Not one parked car answered,
/// because VEH2b's fleet is a set of DYNAMIC chassis boxes and `Fixed` cannot
/// see one — and a car is the canonical low cover in the game this wave is
/// named after. The moving-platform rule is kept where it belongs instead:
/// `probe_cover` refuses a body that is actually MOVING, by its velocity,
/// which is the question the filter was standing in for.
#[allow(clippy::too_many_arguments)]
pub fn sweep_forward_face(
    world: &mut PhysicsWorld3D,
    feet: DVec3,
    fwd: DVec3,
    centre_m: f64,
    span_m: f64,
    reach_m: f64,
    radius_m: f64,
    slope_limit_deg: f64,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
    targets: CastTargets,
) -> Option<super::ShapeHit3D> {
    if fwd == DVec3::ZERO
        || !feet.is_finite()
        || !span_m.is_finite()
        || span_m <= 0.0
        || !reach_m.is_finite()
        || reach_m <= 0.0
    {
        return None;
    }
    let start = feet - fwd * SWEEP_BACKOFF_M + DVec3::Y * centre_m;
    let sweeper = ColliderShape3D::Capsule {
        half_height: 0.01 + span_m,
        radius: radius_m,
    };
    let wall = world.cast_shape_where(
        &sweeper,
        start,
        DQuat::IDENTITY,
        fwd,
        reach_m + SWEEP_BACKOFF_M,
        exclude,
        targets,
    )?;
    if wall.started_penetrating || is_walkable(wall.normal, slope_limit_deg) {
        return None;
    }
    Some(wall)
}

/// **The downward top sweep** -- step 2 of the ledge probe, and the height
/// classifier of wave COV1's cover probe.
///
/// A sphere dropped from `rise_m` above the character's feet onto the surface
/// behind `face`, stepped [`TOP_SWEEP_INSET_M`] into it so the sweep lands on
/// the surface rather than on its lip. Answers the **world point** of the top it
/// found, or `None` when there is no walkable top within the rise -- which a
/// mantle reads as "not a ledge" and a cover press reads as "tall enough to
/// stand behind".
#[allow(clippy::too_many_arguments)]
pub fn sweep_surface_top(
    world: &mut PhysicsWorld3D,
    feet_y: f64,
    face_point: DVec3,
    face_normal: DVec3,
    rise_m: f64,
    radius_m: f64,
    slope_limit_deg: f64,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
    targets: CastTargets,
) -> Option<DVec3> {
    let down_end = DVec3::new(face_point.x, feet_y, face_point.z) - face_normal * TOP_SWEEP_INSET_M;
    let rise = rise_m + radius_m + 0.01;
    let down_start = down_end + DVec3::Y * rise;
    let top = world.cast_shape_where(
        &ColliderShape3D::Sphere { radius: radius_m },
        down_start,
        DQuat::IDENTITY,
        -DVec3::Y,
        rise,
        exclude,
        targets,
    )?;
    if top.started_penetrating || !is_walkable(top.normal, slope_limit_deg) {
        return None;
    }
    // The sphere's centre at impact, with the surface's own height: ALS's
    // `(hit.Location.XY, hit.ImpactPoint.Z)`.
    let sphere_centre = down_start - DVec3::Y * top.toi;
    Some(DVec3::new(sphere_centre.x, top.point.y, sphere_centre.z))
}

/// **The five-step ledge probe.**
///
/// `feet` is the character's ground point, `forward` its facing (planar; a
/// non-planar vector is flattened), `radius`/`half_height` its capsule, and
/// `exclude` the colliders that must not be seen — the character's own, plus
/// every other character's, which is `IgnoreOnlyPawn`.
///
/// `None` for each of ALS's six rejections, and one of ours (a degenerate
/// facing).
#[allow(clippy::too_many_arguments)]
pub fn probe_ledge(
    world: &mut PhysicsWorld3D,
    feet: DVec3,
    forward: DVec3,
    radius: f64,
    half_height: f64,
    slope_limit_deg: f64,
    settings: &LedgeSettings,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
) -> Option<Ledge> {
    let fwd = DVec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
    if fwd == DVec3::ZERO || !feet.is_finite() {
        return None;
    }
    let band = (settings.max_height_m + settings.min_height_m) * 0.5;
    let span = (settings.max_height_m - settings.min_height_m) * 0.5;
    if !span.is_finite() || span <= 0.0 || !settings.reach_m.is_finite() || settings.reach_m <= 0.0
    {
        return None;
    }

    // ── 1. Forward capsule sweep across the band of heights a ledge can be in.
    //    **Wave COV1 hoisted this into `sweep_forward_face`** — one door, two
    //    readers: a mantle asks what the TOP of the thing in front is and a
    //    cover press asks what its FACE is, and both start here.
    let wall = sweep_forward_face(
        world,
        feet,
        fwd,
        band,
        span,
        settings.reach_m,
        settings.forward_radius_m,
        slope_limit_deg,
        exclude,
        CastTargets::Fixed,
    )?;

    // ── 2. Downward sphere sweep to find the ledge's top surface, stepping 15 cm
    //    INTO the ledge so the sweep lands on the surface rather than on its lip.
    //    Hoisted with step 1, and for the same reason.
    let landing = sweep_surface_top(
        world,
        feet.y,
        wall.point,
        wall.normal,
        settings.max_height_m,
        settings.down_radius_m,
        slope_limit_deg,
        exclude,
        CastTargets::Fixed,
    )?;

    // ── 3. Room check: the character's OWN capsule must fit where it is going.
    //    Lifted by a 2 cm skin so the check is about the space above the ledge
    //    and not about the ledge itself.
    let target_centre = landing + DVec3::Y * (half_height + radius + 0.02);
    let fits = world.cast_shape(
        &ColliderShape3D::Capsule {
            half_height: half_height.max(1e-3),
            radius: radius.max(1e-3),
        },
        target_centre,
        DQuat::IDENTITY,
        DVec3::Y,
        1e-3,
        exclude,
    );
    if fits.is_some_and(|h| h.started_penetrating) {
        return None;
    }

    // ── 4. Height, and the classification ALS hard-codes at 125 cm.
    let height_m = landing.y - feet.y;
    if !height_m.is_finite() || height_m < settings.min_height_m || height_m > settings.max_height_m
    {
        return None;
    }
    Some(Ledge {
        feet: landing,
        normal: wall.normal,
        height_m,
        high: height_m > inf_anim::MANTLE_HIGH_SPLIT_M,
        // Face INTO the wall: the opposite of its outward normal.
        yaw_deg: planar_yaw_deg(-wall.normal.x, -wall.normal.z),
    })
}

/// The fall speed below which land prediction says nothing, m/s (ALS −200 cm/s):
/// a character stepping off a kerb is not landing, it is walking.
pub const LAND_PREDICT_MIN_MPS: f64 = -2.0;
/// The fall speed at which the prediction sweep is at its longest, m/s (ALS
/// −4000 cm/s).
pub const LAND_PREDICT_FULL_MPS: f64 = -40.0;
/// The shortest prediction sweep, metres (ALS 50 cm).
pub const LAND_PREDICT_NEAR_M: f64 = 0.5;
/// The longest, metres (ALS 2000 cm).
pub const LAND_PREDICT_FAR_M: f64 = 20.0;

/// What the velocity-aligned sweep saw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandPrediction {
    /// How close the landing is, `[0, 1]` — `1` is about to happen, `0` is at the
    /// far end of the sweep. This is the value an animation blends on.
    pub alpha: f64,
    /// Seconds until impact at the current speed.
    pub time_s: f64,
    /// Where the sweep hit.
    pub point: DVec3,
    /// The surface normal there.
    pub normal: DVec3,
    /// **The speed the character will hit at**, m/s, accounting for the gravity
    /// it will pick up on the way down — which is what makes this usable by the
    /// landing classifier *before* the touch rather than after it.
    pub impact_mps: f64,
}

/// **Land prediction**: a capsule swept along the character's own velocity,
/// whose length scales with fall speed.
///
/// Along the velocity and not straight down, which is the point: a character
/// thrown off a ledge lands somewhere in front of itself, and a downward ray
/// would predict the void it is currently over.
///
/// `None` when the character is not falling fast enough to be landing, when the
/// sweep hits nothing, or when what it hits is a wall rather than a floor.
#[allow(clippy::too_many_arguments)]
pub fn predict_landing(
    world: &mut PhysicsWorld3D,
    centre: DVec3,
    velocity: DVec3,
    radius: f64,
    half_height: f64,
    slope_limit_deg: f64,
    gravity_mps2: f64,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
) -> Option<LandPrediction> {
    if !velocity.is_finite() || !centre.is_finite() || velocity.y > LAND_PREDICT_MIN_MPS {
        return None;
    }
    let vy = velocity
        .y
        .clamp(LAND_PREDICT_FULL_MPS, LAND_PREDICT_MIN_MPS);
    let dir = DVec3::new(velocity.x, vy, velocity.z).normalize_or_zero();
    if dir == DVec3::ZERO {
        return None;
    }
    // The sweep's length scales 0.5 m -> 20 m with fall speed.
    let t = ((velocity.y.max(LAND_PREDICT_FULL_MPS) - LAND_PREDICT_MIN_MPS)
        / (LAND_PREDICT_FULL_MPS - LAND_PREDICT_MIN_MPS))
        .clamp(0.0, 1.0);
    let length = LAND_PREDICT_NEAR_M + (LAND_PREDICT_FAR_M - LAND_PREDICT_NEAR_M) * t;
    let hit = world.cast_shape(
        &ColliderShape3D::Capsule {
            half_height: half_height.max(1e-3),
            radius: radius.max(1e-3),
        },
        centre,
        DQuat::IDENTITY,
        dir,
        length,
        exclude,
    )?;
    if hit.started_penetrating || !is_walkable(hit.normal, slope_limit_deg) {
        return None;
    }
    let speed = velocity.length();
    let time_s = if speed > 1e-9 { hit.toi / speed } else { 0.0 };
    // The drop between here and there, and therefore the speed it arrives at.
    // `v² = v0² + 2 g h` — a sqrt, which IEEE-754 specifies exactly.
    let drop = (-dir.y * hit.toi).max(0.0);
    let impact_mps = (velocity.y * velocity.y + 2.0 * gravity_mps2.max(0.0) * drop).sqrt();
    Some(LandPrediction {
        alpha: (1.0 - hit.toi / length).clamp(0.0, 1.0),
        time_s,
        point: hit.point,
        normal: hit.normal,
        impact_mps,
    })
}
