//! **TAKING COVER** (wave COV1) — the deciding half.
//!
//! The user's brief, verbatim: *"taking cover … just like how the characters can
//! take cover (by either crouching or standing depending on the object) behind
//! objects in GTA 6. It will function literally just like how it does in GTA 6
//! and the new Gears of War game."* Two verbs come out of that sentence — the
//! **stance is the object's**, and the **camera and the gun come out around an
//! edge** — and everything here serves one or the other.
//!
//! # The split, which is this engine's own
//!
//! Everything in this module is a pure function of numbers, exactly as
//! [`crate::movement`] is to `inf_physics::d3::movement` and
//! [`crate::dispatch`] is to `inf_physics::d3::dispatch`. The half that needs a
//! world — the sweeps that find a surface, measure its top and walk out to its
//! corners — is `inf_physics::d3::cover`. So the *rules* (what counts as cover,
//! which stance it implies, how far a peek leans, when a character leaves) can
//! be measured on their own, and a gate that wants to falsify a threshold
//! mutates one line in one crate.
//!
//! # It is SIM state, and it moves no schema
//!
//! [`CoverState`] lives on [`crate::components::MovementRuntime`], which is
//! `#[serde(skip)]` — the [`crate::components::MantleState`] and
//! `SeatState` precedent, and for their reason: the mode is serialized (it
//! always was, and `Cover` claimed reserved slot **14**), the state around it is
//! derived, and a level saved with a character in cover loads with that
//! character standing where the cover was rather than with a dangling reference
//! to a wall. Scene **v27**, `ScenePayload` **13** and `EXPECTED_LEVELS` **24**
//! were all untouched by this wave. (The scene is **v28** since wave VEH3a
//! spent the VEH3 arc's window on the vehicle class; nothing in COV1 moved
//! with it, and `ScenePayload` is still 13.)
//!
//! Derived does not mean unsimulated: the mode, the snapped transform and the
//! peek are all fixed-step state, so PIE and shipping integrate them identically
//! and the replay gate reproduces them.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;
use uuid::Uuid;

use crate::math::Vec3d;

/// **The shortest surface a character will take cover behind**, metres above its
/// feet.
///
/// 0.55 m, chosen against the two things on this island that bracket it: a
/// street kerb is [`crate::traffic::KERB_HEIGHT_M`] (0.15 m) and a parked car's
/// flank is about 0.8 m. Below this a body cannot get enough of itself behind
/// the thing even crouched, which is what "not cover" means.
pub const MIN_COVER_HEIGHT_M: f64 = 0.55;

/// **How long the snap into cover takes**, seconds.
///
/// A quarter of a second — GTA's own feel, and short enough that the press feels
/// like a press. It is a BLEND and not a teleport: the capsule travels the
/// distance over this window, and
/// [`max_snap_speed_mps`] is what a gate bounds it with.
pub const SNAP_S: f64 = 0.25;

/// **The longest snap the press will take at all**, metres.
///
/// A character further than this from the surface walks to it rather than being
/// pulled: a snap that crossed a room would be a teleport with a ramp on it.
/// The cover probe's own reach is shorter than this, so this is the belt to its
/// braces — it bounds a snap whose surface moved between the probe and the
/// blend.
pub const MAX_SNAP_M: f64 = 1.50;

/// **A sprint-to-cover snap takes longer**, seconds — the slide-in.
///
/// A body arriving at 6.5 m/s does not stop in a quarter of a second, and
/// CHAR1b.2's authored `INF_Slide` is 1.10 s. This is the window the slide-in
/// plays over; the snap itself is the same blend, given more of it.
pub const SLIDE_IN_S: f64 = 0.55;

/// **How long the stick must be held AWAY from the surface to leave cover**,
/// seconds.
///
/// GTA's rule, and the reason it is a *duration*: the stick points away from the
/// wall for a frame every time a player corrects a slide, and a character that
/// popped out of cover on that frame would be unusable.
pub const AWAY_LEAVE_S: f64 = 0.30;

/// How far the stick must be pushed away from the surface for that clock to run,
/// `[0, 1]` of full deflection.
pub const AWAY_DEADZONE: f64 = 0.5;

/// **How far a corner peek moves the character sideways**, metres.
///
/// Both halves of the design rest on this being a real displacement of the
/// CAPSULE rather than a lean of the pose alone: the head has to actually clear
/// the corner for a shot to reach it, and it has to actually go back for the
/// wall to stop the next one. 0.45 m puts a 0.30 m-radius capsule's outer edge
/// 0.15 m past the corner.
pub const PEEK_LATERAL_M: f64 = 0.45;

/// How fast a peek leans out and back, `[0, 1]` per second.
pub const PEEK_RATE_PER_S: f64 = 4.0;

/// **How far the character's back sits off the surface**, metres — the standoff,
/// measured from the capsule's SURFACE.
pub const STANDOFF_M: f64 = 0.06;

/// The tolerance a gate allows on that standoff, metres. Five centimetres, which
/// is the brief's own number.
pub const STANDOFF_TOLERANCE_M: f64 = 0.05;

/// **What a cover surface is, by height** — the whole of "crouching or standing
/// depending on the object".
///
/// **Not a wire enum**: it never reaches a file. It lives on the runtime and
/// rides into the animation graph as a float
/// (`inf_anim::als::COVER_VAR`), which is what the CHAR1b.2 law means by *the
/// door is a MODE plus a PARAMETER*.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoverClass {
    /// Nothing to hide behind.
    #[default]
    None,
    /// A top between [`MIN_COVER_HEIGHT_M`] and [`inf_anim::MANTLE_HIGH_SPLIT_M`]
    /// — a car's flank, a low wall, a bar counter. The character **crouches**,
    /// and can rise over the top to shoot.
    Low,
    /// A top above that split, or no top at all within the probe's look — a
    /// façade, a shipping container. The character **stands**, and can only
    /// shoot around an edge.
    High,
}

impl CoverClass {
    /// Whether this is cover at all.
    pub fn is_cover(self) -> bool {
        !matches!(self, CoverClass::None)
    }

    /// **Whether the character crouches behind it.** The user's sentence, as a
    /// function.
    pub fn crouches(self) -> bool {
        matches!(self, CoverClass::Low)
    }

    /// The value the animation parameter carries: `0` none, `1` low, `2` high.
    pub fn param(self) -> f64 {
        match self {
            CoverClass::None => 0.0,
            CoverClass::Low => 1.0,
            CoverClass::High => 2.0,
        }
    }

    /// The class a parameter value names — the inverse, so a gate reads back
    /// what a machine was told rather than what a table says.
    pub fn from_param(v: f64) -> Self {
        if v >= 1.5 {
            CoverClass::High
        } else if v >= 0.5 {
            CoverClass::Low
        } else {
            CoverClass::None
        }
    }
}

/// **Classify a measured surface top.** The one place the two thresholds are
/// compared against, so a mutation has one line to change.
///
/// `top_m` is the surface's height above the character's feet, and
/// [`f64::INFINITY`] means the probe found no top within its look — which is
/// what a wall is, and why the third case exists at all.
pub fn classify(top_m: f64, min_height_m: f64) -> CoverClass {
    if !top_m.is_finite() {
        return CoverClass::High;
    }
    if top_m < min_height_m {
        CoverClass::None
    } else if top_m <= inf_anim::MANTLE_HIGH_SPLIT_M {
        CoverClass::Low
    } else {
        CoverClass::High
    }
}

/// **Which way the character is leaning out of cover.**
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoverSide {
    /// Tucked in — nothing of the character is past the cover.
    #[default]
    Behind,
    /// Around the LEFT edge (the character's own left while facing the surface).
    Left,
    /// Around the right edge.
    Right,
    /// **Over the top** — a `Low` cover only. The character rises out of its
    /// crouch instead of stepping sideways, which is the whole difference
    /// between shooting over a car and shooting round a wall.
    Over,
}

impl CoverSide {
    /// The value the animation parameter carries: `0` behind, `-1` left, `+1`
    /// right, `2` over. A signed number for the two lateral cases so an additive
    /// lean can read it directly.
    pub fn param(self) -> f64 {
        match self {
            CoverSide::Behind => 0.0,
            CoverSide::Left => -1.0,
            CoverSide::Right => 1.0,
            CoverSide::Over => 2.0,
        }
    }

    /// **The sign of the lateral displacement, in the TANGENT's own frame.**
    ///
    /// The tangent is [`tangent_left`], so `+1` is toward the character's left
    /// and `-1` toward its right; the two that do not move sideways are `0`.
    ///
    /// It is deliberately NOT [`param`](Self::param)'s sign. That one is an
    /// animation number, where `-1`/`+1` is the additive lean's own left/right
    /// convention; this one multiplies a world vector, and the two disagreeing
    /// is what made the first corner peek step the capsule the wrong way round
    /// the corner — measured: a lean at the left-hand end of a wall moved the
    /// character 0.45 m to its RIGHT, away from the corner it was leaning
    /// around.
    pub fn lateral_sign(self) -> f64 {
        match self {
            CoverSide::Left => 1.0,
            CoverSide::Right => -1.0,
            _ => 0.0,
        }
    }

    /// Whether this peek clears the cover by STANDING rather than by stepping.
    pub fn is_over(self) -> bool {
        matches!(self, CoverSide::Over)
    }

    /// Whether the character is exposed at all.
    pub fn is_out(self) -> bool {
        !matches!(self, CoverSide::Behind)
    }
}

/// **A character in cover** — everything the fixed step owns about it.
///
/// Plain `Copy` scalars, because [`crate::components::MovementRuntime`] is
/// inlined into an ECS component and must not allocate — the
/// [`crate::components::MantleState`] rule.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoverState {
    /// Whether the character is in cover.
    pub active: bool,
    /// What the surface is.
    pub class: CoverClass,
    /// **Whether the stance is crouched**, derived from `class` on entry and on
    /// every re-classification.
    ///
    /// Stored rather than recomputed because
    /// [`CharacterMovement::half_height_for`](crate::components::CharacterMovement::half_height_for)
    /// reads it, and that function takes a mode and nothing else — a capsule
    /// that resized by asking the class through three layers would be a capsule
    /// two authorities could disagree about.
    pub crouched: bool,
    /// Where the character's FEET sit against the surface, world metres — the
    /// probe's anchor, slid along the surface by [`along_m`](Self::along_m).
    pub anchor: Vec3d,
    /// The surface's outward normal: it points back AT the character.
    pub normal: Vec3d,
    /// The facing while in cover: **into** the surface, degrees.
    pub yaw_deg: f64,
    /// The surface's top above the anchor's feet, metres. [`f64::INFINITY`] for
    /// a wall the probe found no top on.
    pub top_m: f64,
    /// How far the surface runs to the character's left before it ends, metres.
    pub left_m: f64,
    /// The same to the right.
    pub right_m: f64,
    /// **How far along the surface the character has slid** from where it
    /// entered, metres, positive toward its own left.
    pub along_m: f64,
    /// **Whether the slide is against a corner right now** — the fact a peek and
    /// a caption both read, and the engagement counter behind "it stopped at the
    /// end" as opposed to "it never got there".
    pub at_corner: bool,
    /// Which way it is leaning out.
    pub side: CoverSide,
    /// How far out, `[0, 1]`.
    pub peek: f64,
    /// Seconds into the snap. The snap is over at [`snap_s`](Self::snap_s).
    pub blend_s: f64,
    /// How long the whole snap takes, seconds — [`SNAP_S`], or [`SLIDE_IN_S`]
    /// when the character arrived sprinting.
    pub snap_s: f64,
    /// Whether the entry was a sprint-to-cover slide.
    pub slide_in: bool,
    /// Where the capsule was when the snap began, world metres.
    pub start: Vec3d,
    /// Its facing then, degrees.
    pub start_yaw_deg: f64,
    /// **How long the stick has been pushed away from the surface**, seconds —
    /// the leave clock.
    pub away_s: f64,
    /// What the last probe cost, in shape casts. Zero on a step that did not
    /// probe, which is the budget arm's own number.
    pub sweeps: u32,
}

impl CoverState {
    /// How far through the snap, `[0, 1]`. `1` once it is over, which is also
    /// what a state with no snap answers.
    pub fn alpha(&self) -> f64 {
        if self.snap_s <= 0.0 {
            return 1.0;
        }
        (self.blend_s / self.snap_s).clamp(0.0, 1.0)
    }

    /// Whether the snap is still running.
    pub fn snapping(&self) -> bool {
        self.active && self.alpha() < 1.0
    }

    /// The value [`crate::anim_bridge`] publishes for the class.
    pub fn param(&self) -> f64 {
        if self.active {
            self.class.param()
        } else {
            0.0
        }
    }
}

/// **The fastest the capsule may travel during a snap**, m/s — the bound the
/// gate's "it is not a teleport" arm reads.
///
/// The longest snap the press accepts, over the shortest window it takes. Any
/// real snap is slower than this, and a teleport is infinitely faster.
pub fn max_snap_speed_mps() -> f64 {
    MAX_SNAP_M / SNAP_S
}

/// **Where the character should be, this step of the snap.**
///
/// A smoothstep between where it was and where the surface wants it, so the
/// press reads as a move rather than as a cut. Pure, so the "bounded speed" arm
/// can integrate it without a world.
pub fn snap_position(start: Vec3d, target: Vec3d, alpha: f64) -> Vec3d {
    let a = smoothstep(alpha);
    Vec3d::new(
        start.x + (target.x - start.x) * a,
        start.y + (target.y - start.y) * a,
        start.z + (target.z - start.z) * a,
    )
}

/// The same for the facing, over the shortest arc.
pub fn snap_yaw_deg(start_yaw_deg: f64, target_yaw_deg: f64, alpha: f64) -> f64 {
    let d = crate::movement::angle_delta_deg(target_yaw_deg, start_yaw_deg);
    crate::movement::wrap_deg(start_yaw_deg + d * smoothstep(alpha))
}

/// The ease both halves of the snap use. `inf_anim::warp_ease`'s shape, spelled
/// here because `inf-ecs` states its own rules.
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// **The character's LEFT while it faces into the surface**, in the ground
/// plane — the direction a cover slide runs along.
///
/// One spelling, because three readers need it: the probe measures its extents
/// along it, the movement step slides along it, and a peek steps along it. Two
/// spellings would be a left that is a right in one of the three.
///
/// The convention is the engine's own compass — yaw zero is `+Z` and `+X` is
/// `+90` — so a forward `f` has its left at `(-f.z, f.x)`. The character faces
/// **into** the surface, which is `-normal`, so its left is `(normal.z,
/// -normal.x)`.
pub fn tangent_left(normal: Vec3d) -> Vec3d {
    let n = Vec3d::new(normal.x, 0.0, normal.z);
    let len = (n.x * n.x + n.z * n.z).sqrt();
    if len < 1.0e-9 {
        return Vec3d::new(0.0, 0.0, 0.0);
    }
    Vec3d::new(n.z / len, 0.0, -n.x / len)
}

/// **Where along the surface the character may stand** — the corner stop.
///
/// `along_m` is where it wants to be (positive toward its own left) and the two
/// extents are what the probe measured. The character's own half-width is kept
/// off each end, so the capsule stops with its EDGE at the corner rather than
/// its centre hanging over the drop — which is what "never slides off the end"
/// means when the thing sliding is 0.6 m wide.
///
/// Answers the clamped position and whether it was clamped, because a slide that
/// has reached a corner is a fact the caller acts on (it is the corner a peek
/// leans around).
pub fn clamp_along(along_m: f64, left_m: f64, right_m: f64, half_width_m: f64) -> (f64, bool) {
    let lo = -(right_m - half_width_m).max(0.0);
    let hi = (left_m - half_width_m).max(0.0);
    if along_m > hi {
        (hi, true)
    } else if along_m < lo {
        (lo, true)
    } else {
        (along_m, false)
    }
}

/// **Which corner is nearer**, given where along the surface the character is.
///
/// The side a corner peek leans toward: whichever edge it is closer to. A
/// character in the middle of a long wall has no near corner within
/// `reach_m` and gets [`CoverSide::Behind`] — it cannot lean around something
/// that is four metres away.
pub fn nearer_corner(
    along_m: f64,
    left_m: f64,
    right_m: f64,
    half_width_m: f64,
    reach_m: f64,
) -> CoverSide {
    let to_left = (left_m - half_width_m - along_m).max(0.0);
    let to_right = (right_m - half_width_m + along_m).max(0.0);
    if to_left.min(to_right) > reach_m {
        return CoverSide::Behind;
    }
    if to_left <= to_right {
        CoverSide::Left
    } else {
        CoverSide::Right
    }
}

/// **Which way the character leans when it aims from cover.**
///
/// Low cover rises over the top; high cover goes round the nearer corner, and
/// answers [`CoverSide::Behind`] when there is no corner in reach — which is a
/// character pinned in the middle of a wall, and is exactly the situation where
/// blind fire is the only shot there is.
pub fn peek_side(
    class: CoverClass,
    along_m: f64,
    left_m: f64,
    right_m: f64,
    half_width_m: f64,
    reach_m: f64,
) -> CoverSide {
    match class {
        CoverClass::None => CoverSide::Behind,
        CoverClass::Low => CoverSide::Over,
        CoverClass::High => nearer_corner(along_m, left_m, right_m, half_width_m, reach_m),
    }
}

/// **Whether the stick is pushing away from the surface**, given the planar
/// intent in the AIM frame and the surface normal expressed in the same frame.
///
/// `intent` is `(right, forward)` and `normal_local` is the surface's outward
/// normal rotated into that same frame, so the test is one dot product: a stick
/// pointing along the normal is a stick pointing away from the wall.
pub fn pushing_away(intent_x: f64, intent_y: f64, normal_x: f64, normal_y: f64) -> bool {
    let m = (intent_x * intent_x + intent_y * intent_y).sqrt();
    if m < 1.0e-6 {
        return false;
    }
    let dot = (intent_x * normal_x + intent_y * normal_y) / m;
    dot >= AWAY_DEADZONE
}

/// **How far out of cover a peek has leaned the capsule**, metres, signed
/// positive toward the character's own left.
///
/// Zero for a peek that rises rather than steps — an `Over` peek moves the head
/// by STANDING, and its displacement is a capsule half-height rather than a
/// sideways offset.
pub fn peek_lateral_m(side: CoverSide, peek: f64) -> f64 {
    side.lateral_sign() * PEEK_LATERAL_M * peek.clamp(0.0, 1.0)
}

// ── BLIND FIRE (wave WPN2e) ─────────────────────────────────────────────────

/// **How far past the surface a blind-fired round leaves from**, metres.
///
/// **Thirty-five centimetres** — an outstretched forearm past the face of the
/// cover, and it is not decoration: a round that left from where the *hand*
/// actually is would start behind the wall and stop in it on segment zero. The
/// whole point of blind fire is that the weapon clears the cover and the head
/// does not.
///
/// It is measured from [`CoverState::anchor`], which is where the feet sit
/// **against** the surface, so the origin is 0.35 m beyond the surface plane
/// whatever the character's own posture is doing.
pub const BLIND_REACH_M: f64 = 0.35;

/// **How far above a low cover's top a blind-fired round leaves from**, metres.
///
/// **Fifteen centimetres** — the weapon held over the parapet, clear of it, with
/// the shooter still crouched behind. Smaller and the round clips its own cover
/// on a surface the probe measured a centimetre low; larger and the weapon is
/// waving in the air.
pub const BLIND_CLEAR_M: f64 = 0.15;

/// **How far around a high cover's edge a blind-fired round leaves from**,
/// metres.
///
/// **`PEEK_LATERAL_M`** — the same 0.45 m a real peek steps, because the arm
/// goes exactly as far round the corner as the body would have and the point is
/// that the head stays behind it.
pub const BLIND_LATERAL_M: f64 = PEEK_LATERAL_M;

/// **How much wider a blind shot's cone is**, degrees.
///
/// **Nine.** Firing at something you cannot see is the least accurate thing a
/// person with a gun does, and this is added to whatever cone the weapon, the
/// stance and the bloom already resolved — so a blind burst from a bench-rested
/// sniper rifle is still tighter than a blind burst from a submachine gun, and
/// both are wild.
///
/// Nine degrees is 1.6 m of scatter at 10 m and 5.5 m at 35 m, which is the
/// engagement range: at the far end of the policy's own reach a blind shot is a
/// suppression tool and not an aimed one, which is what it should be.
pub const BLIND_FIRE_CONE_DEG: f64 = 9.0;

/// **Where a BLIND round leaves from, and which way it goes** (wave WPN2e) —
/// the whole rule, as one pure function.
///
/// Blind fire is *"the weapon clears the cover and the head does not"*, and this
/// is the two facts that statement implies:
///
/// * **the direction** is the cover's own outward normal.
///   [`CoverState::normal`] points back **at** the character, so out is
///   `-normal`, and the answer is a yaw because a blind shot is level: nobody
///   aims a weapon they are not looking down;
/// * **the origin** is past the surface, and *how* it gets past depends on the
///   class, which is the same distinction [`peek_side`] draws:
///   - **`Low`** — over the top. [`BLIND_CLEAR_M`] above the surface's own
///     measured top, [`BLIND_REACH_M`] beyond its face.
///   - **`High`** — round the nearer end, whichever of `left_m` / `right_m` is
///     shorter, [`BLIND_LATERAL_M`] along the tangent and [`BLIND_REACH_M`]
///     beyond the face, at whatever height the caller's muzzle already is.
///     Unlike [`nearer_corner`] there is no reach gate: a character cannot
///     *lean* round a corner four metres away and can perfectly well point a
///     weapon at it, and a wall with no end at all (`left_m` and `right_m` both
///     infinite) takes the left arm of the tie, which is a direction rather than
///     a refusal.
///
/// `muzzle_y` is the height the shooter's own muzzle is at, used for the high
/// case; the low case ignores it, because a weapon held over a parapet is at the
/// parapet's height and not at the crouched shooter's chest.
///
/// Answers `None` for a state that is not in cover and for a degenerate normal —
/// refusals as values, and the caller fires normally.
///
/// Portable: [`inf_math::patan2_64`], because this yaw reaches a ray cast, whose
/// hit reaches the damage door, which reaches the trace.
pub fn blind_fire_shot(cover: &CoverState, muzzle_y: f64) -> Option<(Vec3d, f64)> {
    if !cover.active {
        return None;
    }
    let n = Vec3d::new(cover.normal.x, 0.0, cover.normal.z);
    let len = (n.x * n.x + n.z * n.z).sqrt();
    if !len.is_finite() || len < 1.0e-9 {
        return None;
    }
    // Out of cover is AWAY from the character, and the normal points at it.
    let out = Vec3d::new(-n.x / len, 0.0, -n.z / len);
    let yaw = inf_math::patan2_64(out.x, out.z).to_degrees();
    let anchor = cover.anchor;
    let origin = if cover.class.crouches() {
        // Over the top. `top_m` is measured above the anchor's feet, and an
        // unmeasured top (an infinite wall the probe found no lid on) is not a
        // low cover by `classify`'s own rule, so it cannot reach here.
        let top = if cover.top_m.is_finite() {
            cover.top_m
        } else {
            return None;
        };
        Vec3d::new(
            anchor.x + out.x * BLIND_REACH_M,
            anchor.y + top + BLIND_CLEAR_M,
            anchor.z + out.z * BLIND_REACH_M,
        )
    } else {
        // Round the nearer end. The tangent is the same one the probe measured
        // its extents along, so `left_m` and `right_m` are in its frame.
        let t = tangent_left(cover.normal);
        let sign = if cover.left_m <= cover.right_m {
            1.0
        } else {
            -1.0
        };
        Vec3d::new(
            anchor.x + out.x * BLIND_REACH_M + t.x * sign * BLIND_LATERAL_M,
            muzzle_y,
            anchor.z + out.z * BLIND_REACH_M + t.z * sign * BLIND_LATERAL_M,
        )
    };
    (origin.x.is_finite() && origin.y.is_finite() && origin.z.is_finite() && yaw.is_finite())
        .then_some((origin, yaw))
}

/// **Whether a peek should stand the character up.**
///
/// Only the `Over` peek does. Half-way through the lean it is still crouched, so
/// the threshold is where the head clears the top rather than where the lean
/// begins — the same 0.5 the pose's own blends use.
pub fn peek_stands(side: CoverSide, peek: f64) -> bool {
    side.is_over() && peek >= 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The classification table, on numbers.** Mutate either threshold and this
    /// reds — which is the point, because both of them decide what a character
    /// does with its whole body.
    #[test]
    fn the_class_is_decided_by_the_measured_top() {
        let f = MIN_COVER_HEIGHT_M;
        assert_eq!(classify(0.15, f), CoverClass::None, "a kerb is not cover");
        assert_eq!(classify(f - 0.001, f), CoverClass::None);
        assert_eq!(classify(f, f), CoverClass::Low);
        assert_eq!(classify(0.82, f), CoverClass::Low, "a car's flank");
        assert_eq!(classify(inf_anim::MANTLE_HIGH_SPLIT_M, f), CoverClass::Low);
        assert_eq!(
            classify(inf_anim::MANTLE_HIGH_SPLIT_M + 0.001, f),
            CoverClass::High
        );
        assert_eq!(classify(2.4, f), CoverClass::High, "a container");
        assert_eq!(classify(f64::INFINITY, f), CoverClass::High, "a wall");
        // The user's own sentence, as an assertion.
        assert!(classify(0.82, f).crouches(), "behind a car you crouch");
        assert!(!classify(2.4, f).crouches(), "against a wall you stand");
    }

    /// The parameter is a round trip, because a machine compares against it.
    #[test]
    fn the_class_and_the_side_survive_their_parameters() {
        for c in [CoverClass::None, CoverClass::Low, CoverClass::High] {
            assert_eq!(CoverClass::from_param(c.param()), c, "{c:?}");
        }
        // The sides are four distinct numbers, so an edge cannot mean two of
        // them.
        let mut p: Vec<i64> = [
            CoverSide::Behind,
            CoverSide::Left,
            CoverSide::Right,
            CoverSide::Over,
        ]
        .iter()
        .map(|s| (s.param() * 1000.0) as i64)
        .collect();
        p.sort_unstable();
        let n = p.len();
        p.dedup();
        assert_eq!(p.len(), n);
    }

    /// **The snap is a blend and it is bounded.** The distance covered per step
    /// never exceeds the bound the gate reads, and it ARRIVES.
    #[test]
    fn the_snap_arrives_and_never_teleports() {
        let start = Vec3d::new(0.0, 0.0, 0.0);
        let target = Vec3d::new(MAX_SNAP_M, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let mut t = 0.0;
        let mut prev = start;
        let mut worst = 0.0f64;
        while t < SNAP_S {
            t += dt;
            let now = snap_position(start, target, t / SNAP_S);
            let d = ((now.x - prev.x).powi(2) + (now.z - prev.z).powi(2)).sqrt();
            worst = worst.max(d / dt);
            prev = now;
        }
        assert!(
            (prev.x - MAX_SNAP_M).abs() < 1.0e-6,
            "the snap arrives: {}",
            prev.x
        );
        // A smoothstep's peak rate is 1.5x the average, and the bound is the
        // whole distance over the whole window -- so the peak is 1.5x the bound
        // and the arm asserts against that rather than against a number that
        // would pass a teleport.
        assert!(
            worst <= max_snap_speed_mps() * 1.5 + 1.0e-9,
            "peak {worst} m/s against {} m/s",
            max_snap_speed_mps() * 1.5
        );
        // A teleport is not this: one step covering the whole distance is
        // 90 m/s, twelve times the bound.
        assert!(MAX_SNAP_M / dt > max_snap_speed_mps() * 1.5 * 2.0);
    }

    /// **The corner stop, with the capsule's own width kept off the end.**
    #[test]
    fn a_slide_stops_before_the_corner_by_its_own_half_width() {
        let (a, clamped) = clamp_along(3.0, 2.0, 2.0, 0.3);
        assert!(clamped);
        assert!((a - 1.7).abs() < 1.0e-9, "{a}");
        let (b, clamped) = clamp_along(-9.0, 2.0, 1.0, 0.3);
        assert!(clamped);
        assert!((b + 0.7).abs() < 1.0e-9, "{b}");
        // Inside, nothing moves.
        let (c, clamped) = clamp_along(0.5, 2.0, 2.0, 0.3);
        assert!(!clamped);
        assert!((c - 0.5).abs() < 1.0e-9);
        // A surface narrower than the character pins it at zero rather than
        // inverting the range.
        let (d, _) = clamp_along(1.0, 0.1, 0.1, 0.3);
        assert_eq!(d, 0.0);
    }

    /// **Low cover rises, high cover goes round, and a wall with no corner in
    /// reach offers neither.**
    #[test]
    fn the_peek_side_is_the_class_and_the_nearer_corner() {
        assert_eq!(
            peek_side(CoverClass::Low, 0.0, 2.0, 2.0, 0.3, 1.2),
            CoverSide::Over
        );
        // Nearer the left edge.
        assert_eq!(
            peek_side(CoverClass::High, 1.2, 2.0, 2.0, 0.3, 1.2),
            CoverSide::Left
        );
        // Nearer the right.
        assert_eq!(
            peek_side(CoverClass::High, -1.2, 2.0, 2.0, 0.3, 1.2),
            CoverSide::Right
        );
        // The middle of a long wall: no corner within reach, so nothing to lean
        // around -- which is where blind fire is the only shot.
        assert_eq!(
            peek_side(CoverClass::High, 0.0, 6.0, 6.0, 0.3, 1.2),
            CoverSide::Behind
        );
        assert_eq!(
            peek_side(CoverClass::None, 0.0, 2.0, 2.0, 0.3, 1.2),
            CoverSide::Behind
        );
    }

    /// The peek displaces the capsule sideways for a corner and stands it up for
    /// a top, and the two are never both.
    #[test]
    fn a_corner_peek_steps_and_a_top_peek_stands() {
        // Positive is along `tangent_left`, which is the character's own left —
        // so a LEFT peek is positive and a RIGHT one negative. The animation
        // parameter's sign is the other convention and is asserted apart.
        assert!((peek_lateral_m(CoverSide::Left, 1.0) - PEEK_LATERAL_M).abs() < 1.0e-12);
        assert!((peek_lateral_m(CoverSide::Right, 1.0) + PEEK_LATERAL_M).abs() < 1.0e-12);
        assert_eq!(
            CoverSide::Left.param(),
            -1.0,
            "the ANIMATION lean is signed the other way, and the two must not be confused"
        );
        assert_eq!(CoverSide::Right.param(), 1.0);
        assert_eq!(peek_lateral_m(CoverSide::Over, 1.0), 0.0);
        assert_eq!(peek_lateral_m(CoverSide::Behind, 1.0), 0.0);
        assert!(!peek_stands(CoverSide::Over, 0.4));
        assert!(peek_stands(CoverSide::Over, 0.5));
        assert!(!peek_stands(CoverSide::Left, 1.0));
    }

    /// **Leaving is a duration, not a frame.** A stick that flicks away for one
    /// step does not leave cover; one held away past the window does.
    #[test]
    fn the_stick_has_to_be_held_away() {
        // Straight away from the wall.
        assert!(pushing_away(0.0, 1.0, 0.0, 1.0));
        // Along it.
        assert!(!pushing_away(1.0, 0.0, 0.0, 1.0));
        // Into it.
        assert!(!pushing_away(0.0, -1.0, 0.0, 1.0));
        // Diagonally away, past the deadzone.
        assert!(pushing_away(0.7, 0.7, 0.0, 1.0));
        // Centred.
        assert!(!pushing_away(0.0, 0.0, 0.0, 1.0));
        // The clock: at 60 Hz it takes 18 steps.
        let steps = (AWAY_LEAVE_S * 60.0).ceil() as i32;
        assert_eq!(steps, 18);
    }

    /// **The tangent is the character's LEFT, and it is a left in every reader.**
    ///
    /// A wall whose outward normal points at `+X` faces a character standing to
    /// its east; that character looks west (`-X`) and its left hand points
    /// south, which in this engine's compass is `-Z`.
    #[test]
    fn the_cover_tangent_is_the_characters_own_left() {
        let t = tangent_left(Vec3d::new(1.0, 0.0, 0.0));
        assert!((t.x).abs() < 1.0e-12, "{t:?}");
        assert!((t.z + 1.0).abs() < 1.0e-12, "{t:?}");
        // A normal pointing at `+Z` (the character stands to the north, looking
        // south): its left is east, `+X`.
        let t = tangent_left(Vec3d::new(0.0, 0.0, 1.0));
        assert!((t.x - 1.0).abs() < 1.0e-12, "{t:?}");
        assert!((t.z).abs() < 1.0e-12, "{t:?}");
        // It is a unit vector, and the vertical component of the normal is
        // dropped rather than tilting it.
        let t = tangent_left(Vec3d::new(3.0, 9.0, 4.0));
        assert!(((t.x * t.x + t.z * t.z).sqrt() - 1.0).abs() < 1.0e-12);
        assert_eq!(t.y, 0.0);
        // A vertical normal has no tangent at all rather than a NaN one.
        let t = tangent_left(Vec3d::new(0.0, 1.0, 0.0));
        assert_eq!((t.x, t.y, t.z), (0.0, 0.0, 0.0));
    }

    /// A fresh state is not in cover, makes no claim and costs nothing — the
    /// thing every character in every committed level carries.
    #[test]
    fn a_default_cover_state_is_not_in_cover() {
        let c = CoverState::default();
        assert!(!c.active);
        assert!(!c.snapping());
        assert_eq!(c.param(), 0.0);
        assert_eq!(c.alpha(), 1.0, "no snap is a finished snap");
        assert_eq!(c.sweeps, 0);
        assert!(!c.crouched);
        assert!(!c.at_corner);
        assert_eq!(c.side, CoverSide::Behind);
    }
}

// ── NPCs in cover (wave COV1, clause 5) ─────────────────────────────────────

/// **How far a unit will walk to reach cover**, metres.
///
/// Twenty. Far enough that a cruiser's crew standing in the open has somewhere
/// to go, short enough that an officer does not abandon the incident it was
/// sent to in order to get behind a wall two streets away.
pub const NPC_COVER_SEARCH_M: f64 = 20.0;

/// **How often a unit that has not found cover looks again**, steps.
///
/// Thirty — half a second at 60 Hz. The search is the expensive half (a probe
/// per bearing) and a unit under fire that failed to find anything last frame
/// will fail again this frame; re-asking every step would make the cost a
/// function of how many officers are pinned rather than of how many are moving.
pub const NPC_SEARCH_PERIOD: u64 = 30;

/// **How far apart the samples along one bearing are**, metres.
///
/// A metre and a half, and it is the number that makes the search WORK at all.
/// The first cut probed once, from the unit's own feet — and `CoverSettings::
/// reach_m` is 0.90 m, so an officer five metres from a wall found nothing and
/// stood in the open through a whole firefight. Measured: 116 shape casts spent
/// and zero cover found. A cover probe answers about the place it is standing,
/// so a search has to MOVE the place.
///
/// One and a half metres is under the probe's own reach plus its backoff, so no
/// surface can fall between two samples.
pub const NPC_SEARCH_STEP_M: f64 = 1.50;

/// **How many bearings a unit's cover search tries.**
///
/// Eight — the compass, which is the same census the CHAR1b.2 audit's ledge
/// sweep used and for the same reason: it is the coarsest set that cannot miss
/// a wall a body could get behind, and the cost bound is what makes the search
/// affordable at all.
pub const NPC_SEARCH_BEARINGS: usize = 8;

/// **How long a unit stays leaned out before ducking back**, seconds.
pub const NPC_PEEK_OUT_S: f64 = 1.20;
/// **…and how long it stays tucked in between peeks.**
///
/// Longer than the lean, so a unit under fire is behind its cover more often
/// than it is out of it — which is what taking cover is for.
pub const NPC_PEEK_IN_S: f64 = 1.80;

/// **What one unit is doing about cover** — derived, never saved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnitCover {
    /// Where it is going, world metres. The zero vector when it has nowhere.
    pub anchor: Vec3d,
    /// The leg it is walking, or `None` when it is not walking one.
    pub path: Option<inf_nav::NavPath>,
    /// The step its last search ran on.
    pub searched_step: u64,
    /// Seconds into the peek cycle.
    pub peek_s: f64,
    /// Whether it is leaned out right now.
    pub peeking: bool,
    /// Who it is taking cover from, world metres.
    pub threat: Vec3d,
    /// **The facing the cover it found wants**, degrees — into the surface.
    ///
    /// Remembered rather than re-derived on arrival: at the anchor the
    /// direction to the anchor is zero, so a unit that turned toward where it
    /// was going would be facing nowhere on the step it presses.
    pub face_yaw_deg: f64,
}

impl UnitCover {
    /// **Lay a leg to `to`** — the `inf_nav::NavPath` a unit walks to its cover.
    ///
    /// A two-point path, arc-length parameterized, which is exactly what
    /// [`crate::crowd::CrowdRoute::between`] is and what every tier measurement
    /// in this tree is written against. A searched route over the street graph
    /// is what a unit *driving* to an incident already gets; an officer
    /// crossing twenty metres of pavement to a wall is a straight leg in every
    /// reference frame this campaign has looked at, and a search here would be
    /// a wave rather than a clause. Stated, not hidden.
    ///
    /// The constructor lives here rather than at the caller because `inf-nav`
    /// is `inf-ecs`' dependency and not `inf-physics`': the applying half asks
    /// for a leg and gets one, and no new dependency edge is added to reach a
    /// type through a crate that already holds it.
    pub fn lay_leg(&mut self, from: Vec3d, to: Vec3d, face_yaw_deg: f64) {
        self.anchor = to;
        self.face_yaw_deg = face_yaw_deg;
        self.path = Some(inf_nav::NavPath::new([from.to_dvec3(), to.to_dvec3()]));
    }

    /// **Which way to walk from `here`** along the leg, or `None` when there is
    /// no leg. A unit already at the end gets the leg's own last direction,
    /// which is what `NavPath::direction_at` answers past its length.
    pub fn heading_from(&self, here: Vec3d) -> Option<Vec3d> {
        let p = self.path.as_ref()?;
        let s = p.project(here.to_dvec3()).s_m;
        Some(Vec3d::from_dvec3(p.direction_at(s)))
    }

    /// Forget the leg — the unit has arrived, or has stopped needing one.
    pub fn drop_leg(&mut self) {
        self.path = None;
    }

    /// Whether it is walking one.
    pub fn walking(&self) -> bool {
        self.path.is_some()
    }
}

/// **Every unit's cover state** (wave COV1) — [`crate::dispatch::RespondersRes`]'
/// shape, and for its reason: a responder may be a `Dormant` crowd agent with no
/// entity at all, so a marker component would be silently absent on exactly the
/// agents a shot at the edge of a radius reaches.
///
/// Derived, never serialized, no schema moves.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct NpcCoverRes {
    /// Keyed by the unit's guid, in `Guid` order because a `BTreeMap` is.
    pub units: BTreeMap<Uuid, UnitCover>,
}

/// **Is this unit under fire?** — the predicate clause 5 is written in terms of.
///
/// A responder ([`crate::dispatch::is_responder`]) inside `radius_m` of a place
/// this step's gunfire came from, that it did not fire itself. It is EMS2's
/// panic exemption read the other way round: the officers the flee door refuses
/// to rout are exactly the officers who are under fire, and this is the same
/// two facts — *is a responder* and *is inside the radius* — asked as a
/// question about the world rather than as a filter inside a pass.
///
/// Pure, so it can be measured without a crowd: the sources are the caller's,
/// already coalesced and capped by `inf_physics::d3::gameplay`'s own bound.
pub fn under_fire(at: Vec3d, sources: &[Vec3d], radius_m: f64) -> Option<Vec3d> {
    let mut best: Option<(f64, Vec3d)> = None;
    for s in sources {
        let d = ((s.x - at.x).powi(2) + (s.y - at.y).powi(2) + (s.z - at.z).powi(2)).sqrt();
        if d <= radius_m && best.is_none_or(|(b, _)| d < b) {
            best = Some((d, *s));
        }
    }
    best.map(|(_, s)| s)
}

/// **Whether a unit is leaned out right now**, and how far into the cycle it is.
///
/// A duty cycle rather than a decision: an officer in cover comes out, shoots,
/// and goes back in, and the cadence is what `npc_aim_at` is called on. The
/// firing itself is **WPN2e's seam** — this wave points the weapon and leans the
/// body, and holds the trigger open.
pub fn peek_cycle(peek_s: f64) -> (bool, f64) {
    let period = NPC_PEEK_OUT_S + NPC_PEEK_IN_S;
    let t = if period > 0.0 {
        peek_s.rem_euclid(period)
    } else {
        0.0
    };
    (t < NPC_PEEK_OUT_S, t)
}

/// **Which class of cover a unit prefers**, given the response rung it is on.
///
/// The EMS3 carried item — *"Swat is a COUNT not a crew"* — becoming a
/// behaviour. At [`crate::crime::Response::Swat`] the town has sent everything
/// it has and the units that arrive are the ones that stack on corners: they
/// take **HIGH** cover, which is the only class you can shoot around an edge
/// from, and they pass over a car's flank to get it. Every other rung takes
/// whatever is nearest, because an officer answering a shoplifting is not
/// clearing a building.
pub fn prefers_high(response: crate::crime::Response) -> bool {
    matches!(response, crate::crime::Response::Swat)
}
