//! **Derive at import** — pillar S2 (P29.5).
//!
//! Epic ships **seven** AnimModifiers with the stock Game Animation Sample and a
//! human has to remember to run each of them over each of 891 clips (§13's
//! measured baseline). This module is the answer §13 promised: the same facts,
//! computed from the clip by code, every time it is imported, with nothing to
//! remember and nothing to forget.
//!
//! # What it produces, and where it puts it
//!
//! Everything lands in the `.inf_anim` **v2** fields P29.2 reserved for exactly
//! this ([`crate::channels`]) — no schema moves, because that was the point of
//! reserving them:
//!
//! * [`RootMotionTrack`] — the root joint's translation (**Y included**) and yaw
//!   over the clip, **with the residual removed from the joints**, so the clip
//!   plays in place and the track carries the displacement. That is what makes a
//!   clip usable by a character controller *and* by [`crate::warp_offset`] at the
//!   same time.
//! * [`DistanceTrack`] — cumulative ground metres, the input to distance
//!   matching.
//! * **Foot-plant sync markers** and **footstep event markers**, derived from the
//!   feet's own height over the clip rather than placed by hand.
//! * **Curve channels**: the clip's ground speed, each foot's speed, each foot's
//!   lock window, and the gait tier — the `W_Gait`/`FootLock_*` half of the
//!   reference vocabulary P29.4's consumers already read.
//!
//! # Determinism
//!
//! A derivation writes bytes into a committed `.inf_anim`, so it is on the P14
//! law's side of the line: no `std` transcendental, no `glam` constructor that
//! reaches one. The single rotation this file builds — the inverse yaw it removes
//! from the root — is written out as `(0, sin(−θ/2), 0, cos(−θ/2))` through
//! [`inf_math::psin64`]/[`inf_math::pcos64`], the same way [`crate::locomotion`]
//! writes its clips. `tests/portable_pose.rs` carries this file for that reason.
//!
//! Everything else is adds, multiplies, comparisons and `sqrt` (which IEEE-754
//! specifies exactly). Iteration is over `Vec`s in index order and over the
//! skeleton in joint order; nothing is keyed by a hash.
//!
//! # Idempotence, and why it needs a rule
//!
//! Derivation **removes** the root residual, so a naive re-run over an
//! already-derived clip would extract a track of zero and overwrite the real one.
//! [`derive_clip`] is therefore a pure function of the clip *as imported*: it
//! **un-bakes first** ([`unbake_root_motion`], the exact inverse of the removal)
//! and then derives from what it recovered. `derive(derive(c)) == derive(c)`,
//! byte for byte, is a test rather than a hope.
//!
//! # What it does NOT do
//!
//! It does not invent a curve nothing measured. `Enable_FootIK_L`,
//! `Layering_*`, `Mask_AimOffset` and the rest of the vocabulary are *authoring*
//! decisions — a clip does not know whether its arms should be overridden — so
//! they are left to the author and read with a fallback ([`AnimClip::curve_value`]).
//! Deriving them as constants would be manufacturing consent.

use glam::{Quat, Vec3};

use crate::channels::{als, AnimMarker, CurveChannel, DistanceTrack, RootMotionTrack};
use crate::clip::{AnimClip, Interpolation, QuatTrack, Vec3Track};
use crate::pose::{global_transforms, sample_clip};
use crate::root_motion::root_joint_index;
use crate::skeleton::Skeleton;

/// The clip's own **ground speed**, metres per second, over its timeline.
///
/// The stock sample bakes this by hand with `AM_MoveData_Speed`
/// (a `MotionExtractor` in `TranslationSpeed` mode over the root); the name is
/// byte-exact so an imported clip that already carries Epic's curve and one this
/// engine derived are the same channel.
pub const MOVE_DATA_SPEED: &str = "MoveData_Speed";

/// Prefix of the per-foot speed channels — `FootSpeed_L`, `FootSpeed_R`
/// (`AM_FootSpeed_L`/`_R` in the stock sample, run by hand over `ball_l`/`ball_r`).
pub const FOOT_SPEED_PREFIX: &str = "FootSpeed_";

/// Prefix of the per-foot lock channels, [`als::FOOT_LOCK_L`]'s family. Derived
/// from the plant windows, which is what P29.4's foot lock consumes.
pub const FOOT_LOCK_PREFIX: &str = "FootLock_";

/// The prefix an **event** marker must carry to be heard as a footstep.
///
/// The one spelling in the engine: `inf_ecs::anim_bridge::FOOTSTEP_PREFIX` is
/// this constant re-exported, so the producer and the consumer cannot drift.
pub const FOOTSTEP_PREFIX: &str = "footstep";

/// The prefix a derived **sync** marker carries.
pub const PLANT_PREFIX: &str = "plant_";

/// **The one spelling of a derived footstep event's name** (P29.6 audit, A17).
///
/// The rule is `footstep_<leg>` and it used to be written twice: once by
/// `derived_markers`, which is what actually goes on the clip, and once by
/// [`DerivedNames::of`], which is what every consumer asks for the name. Two
/// `format!`s five hundred lines apart, agreeing by inspection, for a wire
/// string — and a producer and a consumer that each spell one string in their
/// own place is exactly how the P29.6 course found the wizard counting nothing.
/// That defect was closed at the *wizard's* boundary; this is the same defect
/// one layer down, in the crate that owns both halves.
pub fn footstep_marker(leg: &str) -> String {
    format!("{FOOTSTEP_PREFIX}_{leg}")
}

/// …and the sync marker beside it, for the same reason.
pub fn plant_marker(leg: &str) -> String {
    format!("{PLANT_PREFIX}{leg}")
}

/// How the vertical half of the root's motion is split between the track and the
/// pose.
///
/// The distinction is not a preference, it is a property of the rig. A clip's
/// root joint is whichever joint has no parent, and on a rig authored the way
/// [`crate::template`] authors one that joint is **`hips`** — which *bobs*. A
/// walk cycle's vertical is then oscillation, not travel, and moving all of it
/// onto the track would bob the capsule instead of the character.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerticalPolicy {
    /// The track carries only the clip's **net** vertical travel — the straight
    /// ramp from the first frame's height to the last — and every oscillation
    /// about it stays in the pose. The default, because a looping gait ends where
    /// it began vertically and therefore has a net of zero: the bob stays where
    /// it belongs and the track is honest about carrying nothing.
    #[default]
    NetDrift,
    /// The track carries the whole vertical. What a **traversal** clip wants: a
    /// mantle's rise *is* the movement, and a controller re-deriving it from
    /// gravity puts the hands somewhere other than the ledge.
    Full,
}

/// The knobs [`derive_clip`] turns, each with the reason its default is what it
/// is.
#[derive(Clone, Debug, PartialEq)]
pub struct DeriveOptions {
    /// Sampling rate for the derived tracks and curves, Hz. The keys of the
    /// clip's own root tracks are always included on top of this grid, so the
    /// residual removal is exact at every authored key whatever this is.
    pub fps: f32,
    /// How the vertical is split — see [`VerticalPolicy`].
    pub vertical: VerticalPolicy,
    /// Whether the clip is meant to loop. Contact windows wrap across the clip
    /// end when it is, which is what stops a cycle that begins mid-stance from
    /// reporting a footstep at time zero on every loop.
    pub looping: bool,
    /// How far above its own lowest point a foot may be and still count as
    /// touching the ground, metres.
    ///
    /// A *height* test rather than a speed test, because contact is a height
    /// fact: a foot on the floor is on the floor whether the body above it is
    /// walking or sprinting. The speed is derived too — as
    /// [`FOOT_SPEED_PREFIX`] channels — and is what the lock consumes, not what
    /// decides the window.
    pub contact_lift_m: f32,
    /// The shortest contact window that counts as a plant, seconds. Below it a
    /// foot brushing the floor at the bottom of a swing is noise, not a step.
    pub min_contact_s: f32,
    /// The speeds that separate the gait tiers, m/s — walk, run, sprint.
    /// ALS's `WalkSpeed`/`RunSpeed`/`SprintSpeed`, converted once
    /// ([`crate::channels::als::W_GAIT`] is 1 / 2 / 3 at these three).
    pub gait_speeds_mps: [f32; 3],
}

impl Default for DeriveOptions {
    fn default() -> Self {
        Self {
            fps: 30.0,
            vertical: VerticalPolicy::NetDrift,
            looping: true,
            contact_lift_m: 0.02,
            min_contact_s: 0.05,
            gait_speeds_mps: [1.65, 3.75, 6.5],
        }
    }
}

impl DeriveOptions {
    /// The options a **traversal** clip is derived with: the whole vertical on
    /// the track (a mantle's rise is travel) and no loop wrap (a vault is played
    /// once).
    pub fn traversal() -> Self {
        Self {
            vertical: VerticalPolicy::Full,
            looping: false,
            ..Self::default()
        }
    }
}

/// One contact window of one foot.
#[derive(Clone, Debug, PartialEq)]
pub struct FootPlant {
    /// The joint that touched down.
    pub joint: u16,
    /// The sync marker's name — the leg's own name where the rig has one (so a
    /// derived clip and a [`crate::locomotion`]-generated one name the same
    /// marker and blend against each other), else the foot joint's.
    pub name: String,
    /// Clip time the foot touched down, seconds.
    pub start_s: f32,
    /// Clip time it left, seconds. Wraps past `duration` on a looping clip whose
    /// contact spans the seam.
    pub end_s: f32,
}

impl FootPlant {
    /// How long the foot was down, seconds.
    pub fn duration_s(&self) -> f32 {
        self.end_s - self.start_s
    }
}

/// What a derivation found — the numbers a panel shows and a proposal clusters
/// on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeriveReport {
    /// Total root translation over the clip, metres, clip space (Y included).
    pub translation: [f32; 3],
    /// Total root turn over the clip, radians.
    pub yaw_rad: f32,
    /// Total ground distance travelled, metres.
    pub distance_m: f32,
    /// **The speed this clip depicts**, m/s — the number a state graph clusters
    /// on. The greater of [`travel_speed_mps`](Self::travel_speed_mps) and
    /// [`stride_speed_mps`](Self::stride_speed_mps); see the latter for why
    /// exactly one of them is meaningful on any given clip.
    pub avg_speed_mps: f32,
    /// `distance_m / duration` — what the **root** travels.
    ///
    /// Zero for an in-place cycle, which is most authored locomotion and every
    /// clip [`crate::locomotion`] generates.
    pub travel_speed_mps: f32,
    /// **What the FEET say**, m/s: `stride × cadence`, where the stride is
    /// [`stride_m`](Self::stride_m) and the cadence is how often the clip plants
    /// that foot.
    ///
    /// The other half of the same question, and the reason a proposal works on
    /// in-place content at all — which is most authored locomotion, and every
    /// clip this engine generates. A cycle that translates nothing still says
    /// how fast it is: it says it in the length of its own stride and the rate it
    /// takes them at, which is the definition of gait speed and does not care
    /// whether anything moved.
    ///
    /// It is measured on the clip **after** the root residual is removed, so a
    /// root-motion clip and an in-place one are the same measurement — which is
    /// what makes the two comparable rather than merely both present. Their
    /// disagreement is itself a reading: a root-motion clip whose stride speed
    /// is a long way from its travel speed is a clip whose feet slide, and no
    /// runtime lock can put back what an animator did not author.
    pub stride_speed_mps: f32,
    /// **The stride**, metres: how far a foot travels along the ground over one
    /// cycle, averaged over the feet that plant at all.
    ///
    /// Measured as the foot's own planar excursion in the residual-removed clip,
    /// which is the one definition that reads the same on a root-motion clip and
    /// an in-place one: in both, a foot goes back by one stride under the body
    /// and forward by one stride over it. `0` for a clip with no feet or no
    /// plants.
    pub stride_m: f32,
    /// The gait tier this clip depicts on ALS's 0–3 scale (0 stopped, 1 walk,
    /// 2 run, 3 sprint) — the value written into [`als::W_GAIT`].
    pub gait: f32,
    /// Every contact window found, foot by foot in joint order.
    pub plants: Vec<FootPlant>,
    /// The curve channels this derivation wrote, in the order it wrote them.
    pub curves: Vec<String>,
    /// The markers it wrote (sync and event together).
    pub markers: usize,
}

/// Why a clip could not be derived. Each is a **value**: an import that meets one
/// keeps the clip it already had and says what it could not do, because a clip
/// with no root joint is still a perfectly good clip.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DeriveError {
    #[error("the skeleton has no root joint (every joint claims a parent)")]
    NoRootJoint,
    #[error("clip {name:?} has no positive duration")]
    NoDuration { name: String },
    #[error("derivation was asked for {fps} fps, which is not a positive rate")]
    BadRate { fps: f32 },
    /// The clip poses to something that is not a number, so what would be baked
    /// is not either. See [`derive_clip`]'s step 6a for why this is a refusal
    /// and not a sanitised value.
    #[error(
        "clip {name:?} derives a non-finite {what}, so nothing was baked — the pose it was \
         sampled from is not a number"
    )]
    NonFinite { name: String, what: String },
}

// ── the entry point ─────────────────────────────────────────────────────────

/// **Derive every field this module owns** onto a copy of `clip`.
///
/// Pure: `clip` is not touched, and the answer depends only on `(clip, skeleton,
/// opts)`. An already-derived clip is un-baked first, so this is idempotent up to
/// bytes — see the module docs.
pub fn derive_clip(
    clip: &AnimClip,
    skeleton: &Skeleton,
    opts: &DeriveOptions,
) -> Result<(AnimClip, DeriveReport), DeriveError> {
    if !crate::positive(opts.fps) {
        return Err(DeriveError::BadRate { fps: opts.fps });
    }
    let root = root_joint_index(skeleton).ok_or(DeriveError::NoRootJoint)? as u16;
    if !crate::positive(clip.duration) {
        return Err(DeriveError::NoDuration {
            name: clip.name.clone(),
        });
    }

    // 1. Recover the clip as it was imported.
    let mut out = clip.clone();
    unbake_root_motion(&mut out, root);
    let source = out.clone();

    // 2. The sample grid: the root's own keys plus a uniform rate, so the
    //    removal below is exact at every authored key and dense enough between
    //    them for the distance track to be a polyline and not a chord.
    let times = sample_times(&source, root, opts.fps);

    // 3. The root's trajectory, and the distance it implies.
    let (root_track, distance) = bake_tracks(&source, skeleton, root, &times, opts.vertical);

    // 4. Remove exactly what was extracted, so the clip plays in place.
    remove_root_residual(&mut out, root, &root_track);

    // 5. The feet: contact windows, and the markers and lock curves they imply.
    //    One grid and one set of traces, shared by the plants and the channels,
    //    so a lock window cannot land between two samples of its own curve.
    //    Measured on `out` -- the residual-removed clip -- and not on the source,
    //    for two reasons that are the same reason: it is the clip every consumer
    //    will actually play, so the channels describe what will be sampled; and a
    //    stride is a foot's excursion *under the body*, which is only a stable
    //    quantity once the body's own travel has been taken out.
    let feet = foot_joints(skeleton);
    let foot_times = uniform_times(clip.duration, opts.fps);
    let traces = foot_traces(&out, skeleton, &feet, &foot_times);
    let plants = foot_plants(skeleton, &feet, &foot_times, &traces, clip.duration, opts);

    // 6. Assemble. Everything this module owns is replaced; anything an author
    //    wrote that this rig's derivation would not itself write survives —
    //    see [`DerivedNames`] for where that boundary is drawn and why.
    let total = *root_track.translation.last().unwrap_or(&[0.0; 3]);
    let first = *root_track.translation.first().unwrap_or(&[0.0; 3]);
    let translation = [
        total[0] - first[0],
        total[1] - first[1],
        total[2] - first[2],
    ];
    let yaw_rad = root_track.yaw_rad.last().copied().unwrap_or(0.0)
        - root_track.yaw_rad.first().copied().unwrap_or(0.0);
    let distance_m = distance.distance_m.last().copied().unwrap_or(0.0);
    let travel_speed_mps = distance_m / clip.duration;
    let (stride_m, stride_speed_mps) = stride_and_speed(&feet, &traces, &plants, clip.duration);
    // The greater, because a clip answers this question from one end or the
    // other: a root-motion clip travels and an in-place one strides. See
    // `DeriveReport::stride_speed_mps` for what their disagreement means when a
    // clip answers from both.
    let avg_speed_mps = travel_speed_mps.max(stride_speed_mps);
    let gait = gait_of(avg_speed_mps, opts.gait_speeds_mps);

    let mut curves = derived_curves(
        skeleton,
        &feet,
        &times,
        &foot_times,
        &traces,
        &distance,
        &plants,
        clip.duration,
        gait,
    );
    let mut markers = derived_markers(&plants);

    let owned_curves: Vec<String> = curves.iter().map(|c| c.name.clone()).collect();

    // 6a. **Nothing non-finite reaches the file.** A clip whose joints hold a
    //     NaN is a legal `.inf_anim` today — `validate_timing` asks about key
    //     *times*, not values — and sampling one poses a NaN, which every step
    //     below propagates: the root-motion track, the distance track and the
    //     foot-speed channels all become NaN. Two of those are worse than
    //     useless. The distance track is `partition_point`ed by
    //     `DistanceTrack::time_at_distance` and its monotonicity **is** checked
    //     at the door, so a derivation that baked one would turn a clip that
    //     loads into a clip that does not: the writer manufacturing a file its
    //     own reader rejects (P23.6's closing ruling). And the foot-speed
    //     channels are read by P29.4's lock with no guard at all.
    //
    //     A refusal and not a sanitised zero, because a zero here is a
    //     measurement nobody took: the honest answer is that this clip cannot be
    //     measured, which the import records as an advisory and the panel shows.
    if let Some(what) = first_non_finite(&root_track, &distance, &curves) {
        return Err(DeriveError::NonFinite {
            name: clip.name.clone(),
            what,
        });
    }

    // 6b. Replace exactly the names this rig implies — see [`DerivedNames`] for
    //     why the boundary is the rig's own and not a prefix.
    let owned = DerivedNames::of(skeleton, &feet);
    out.curves.retain(|c| !owned.curves.contains(&c.name));
    out.curves.append(&mut curves);
    out.markers.retain(|m| !owned.owns_marker(m));
    out.markers.append(&mut markers);

    let report = DeriveReport {
        translation,
        yaw_rad,
        distance_m,
        avg_speed_mps,
        travel_speed_mps,
        stride_speed_mps,
        stride_m,
        gait,
        plants,
        curves: owned_curves,
        markers: out.markers.iter().filter(|m| owned.owns_marker(m)).count(),
    };
    out.root_motion = Some(root_track);
    out.distance = Some(distance);
    Ok((out, report))
}

/// **Every name a derivation over `skeleton` writes** — and therefore the exact
/// set it is entitled to replace.
///
/// The boundary is the **rig's own** and not a prefix, and that is the whole
/// point of the type. The audio drain hears any marker starting with
/// [`FOOTSTEP_PREFIX`], and its own documentation invites a project to add a
/// `footstep_land` for free — so a prefix rule reads that authored notify as
/// something this module wrote and deletes it on the next press of the
/// re-derive button. So does it read a `plant_hand_l` sync marker somebody put
/// in a hands group. Silently destroying authored content is not a boundary, it
/// is the absence of one.
///
/// It is a set of **could-write** names rather than the did-write ones, which
/// matters in the other direction: a re-derive with a stricter
/// [`DeriveOptions::contact_lift_m`] may find no plants at all, and a rule that
/// only replaced what it wrote this time would leave the *previous* run's
/// markers behind. The set is a function of the rig alone, so it is the same
/// whatever the options say.
///
/// The cost of that choice is stated rather than hidden: a rig whose feet were
/// renamed strands the old run's channels under the old names. That is stale
/// derived data an author can see and delete, which is the cheaper of the two
/// mistakes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DerivedNames {
    /// Curve channels: [`MOVE_DATA_SPEED`], [`als::W_GAIT`], and a
    /// [`FOOT_SPEED_PREFIX`]/[`FOOT_LOCK_PREFIX`] pair per foot.
    pub curves: std::collections::BTreeSet<String>,
    /// Sync markers — `plant_<leg>`, one per foot, in
    /// [`crate::locomotion::FOOT_SYNC_GROUP`].
    pub sync_markers: std::collections::BTreeSet<String>,
    /// Event markers — `footstep_<leg>`, one per foot.
    pub event_markers: std::collections::BTreeSet<String>,
}

impl DerivedNames {
    /// The names a derivation over `skeleton` with `feet` ([`foot_joints`])
    /// writes.
    pub fn of(skeleton: &Skeleton, feet: &[u16]) -> Self {
        let mut out = Self {
            curves: [MOVE_DATA_SPEED.to_string(), als::W_GAIT.to_string()]
                .into_iter()
                .collect(),
            ..Self::default()
        };
        for &foot in feet {
            let suffix = foot_suffix(skeleton, foot);
            out.curves.insert(format!("{FOOT_SPEED_PREFIX}{suffix}"));
            out.curves.insert(format!("{FOOT_LOCK_PREFIX}{suffix}"));
            let leg = leg_name_of(skeleton, foot);
            out.sync_markers.insert(plant_marker(&leg));
            out.event_markers.insert(footstep_marker(&leg));
        }
        out
    }

    /// The names a derivation over `skeleton` writes, asking [`foot_joints`] for
    /// the feet — the door the asset panel's `anim_clip_info` reads to badge a
    /// channel as derived, so the badge and the re-derive button agree about
    /// which channels the button will replace.
    pub fn of_skeleton(skeleton: &Skeleton) -> Self {
        Self::of(skeleton, &foot_joints(skeleton))
    }

    /// Whether `marker` is one this module writes. **The group is part of the
    /// question** for a sync marker: a `plant_hand_l` an author put in a hands
    /// group is not a foot plant however it is spelled.
    pub fn owns_marker(&self, marker: &AnimMarker) -> bool {
        if marker.is_sync() {
            marker.group == crate::locomotion::FOOT_SYNC_GROUP
                && self.sync_markers.contains(&marker.name)
        } else {
            self.event_markers.contains(&marker.name)
        }
    }
}

/// Whether a curve name belongs to the **derived vocabulary** — the display
/// question, not the ownership one.
///
/// A panel badging a channel "derived" has a clip and may have no rig, so it
/// asks about the family. [`DerivedNames`] is the authority on what a derivation
/// actually replaces, and the two differ only on a name that borrows the family
/// prefix without naming a foot of the rig (`FootLock_Weapon`): badged here,
/// left alone there — which is the safe direction for both.
pub fn is_derived_curve(name: &str) -> bool {
    name == MOVE_DATA_SPEED
        || name == als::W_GAIT
        || name.starts_with(FOOT_SPEED_PREFIX)
        || name.starts_with(FOOT_LOCK_PREFIX)
}

/// Whether a marker name belongs to the derived vocabulary — the display
/// question. [`DerivedNames::owns_marker`] is the ownership one.
pub fn is_derived_marker(name: &str) -> bool {
    name.starts_with(PLANT_PREFIX) || name.starts_with(FOOTSTEP_PREFIX)
}

/// The first non-finite number in what a derivation is about to bake, named.
///
/// Every field checked here is one a consumer reads without asking: the mantle
/// reads the root track, `time_at_distance` binary-searches the distance track,
/// and P29.4's foot lock reads the channels. `AnimClip::validate_timing` asks
/// about key *times* only, so none of these values has a door anywhere else.
fn first_non_finite(
    root: &RootMotionTrack,
    distance: &DistanceTrack,
    curves: &[CurveChannel],
) -> Option<String> {
    if root
        .translation
        .iter()
        .flatten()
        .chain(root.yaw_rad.iter())
        .any(|v| !v.is_finite())
    {
        return Some("root-motion track".to_string());
    }
    if distance.distance_m.iter().any(|v| !v.is_finite()) {
        return Some("distance track".to_string());
    }
    curves
        .iter()
        .find(|c| c.values.iter().any(|v| !v.is_finite()))
        .map(|c| format!("curve channel `{}`", c.name))
}

/// **Put a baked root-motion track back onto the root joint** — the exact inverse
/// of the residual removal, so a clip can be re-derived from what was imported.
///
/// A clip with no baked track is left alone, which is the case that matters: this
/// is called on every derivation and does nothing on the first one.
pub fn unbake_root_motion(clip: &mut AnimClip, root: u16) {
    let Some(track) = clip.root_motion.take() else {
        return;
    };
    clip.distance = None;
    let Some(jt) = clip.tracks.iter_mut().find(|t| t.joint == root) else {
        return;
    };
    if let Some(ch) = jt.translation.as_mut() {
        for (i, t) in ch.times.iter().enumerate() {
            let Some((p, _)) = track.sample(*t) else {
                continue;
            };
            if let Some(v) = ch.values.get_mut(i) {
                v[0] += p[0];
                v[1] += p[1];
                v[2] += p[2];
            }
        }
    }
    if let Some(ch) = jt.rotation.as_mut() {
        for (i, t) in ch.times.iter().enumerate() {
            let Some((_, yaw)) = track.sample(*t) else {
                continue;
            };
            if let Some(v) = ch.values.get_mut(i) {
                let q = yaw_quat(yaw as f64) * Quat::from_array(*v);
                *v = q.to_array();
            }
        }
    }
}

// ── the pieces ──────────────────────────────────────────────────────────────

/// A yaw rotation about +Y, built by hand.
///
/// `Quat::from_rotation_y` is `f32::sin_cos`, which the P14 law bans on any path
/// that reaches committed bytes — and this one *is* the bytes. Written the way
/// [`crate::locomotion`] writes its own: the two components computed in `f64`
/// through the portable helpers and cast once, at the wire.
fn yaw_quat(yaw_rad: f64) -> Quat {
    let h = yaw_rad * 0.5;
    Quat::from_xyzw(
        0.0,
        inf_math::psin64(h) as f32,
        0.0,
        inf_math::pcos64(h) as f32,
    )
}

/// The union of the root joint's own key times and a uniform `fps` grid over the
/// clip, ascending and deduplicated.
///
/// Both halves earn their place. The **keys** make the residual removal exact:
/// the track passes through every authored value, so subtracting it at that key
/// leaves exactly zero. The **grid** makes the distance track a polyline rather
/// than a chord across a curved path, and gives the derived curves somewhere to
/// put a sample on a clip whose root has two keys and whose feet have forty.
fn sample_times(clip: &AnimClip, root: u16, fps: f32) -> Vec<f32> {
    let dur = clip.duration;
    let steps = ((dur * fps).ceil() as usize).max(1);
    let mut times: Vec<f32> = Vec::with_capacity(steps + 8);
    for k in 0..=steps {
        times.push(dur * (k as f32) / (steps as f32));
    }
    if let Some(jt) = clip.tracks.iter().find(|t| t.joint == root) {
        for ch in [jt.translation.as_ref().map(|c| &c.times)]
            .into_iter()
            .flatten()
        {
            times.extend(ch.iter().copied().filter(|t| t.is_finite()));
        }
        if let Some(c) = jt.rotation.as_ref() {
            times.extend(c.times.iter().copied().filter(|t| t.is_finite()));
        }
    }
    times.retain(|t| t.is_finite() && *t >= 0.0 && *t <= dur);
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    if times.is_empty() {
        times.push(0.0);
    }
    times
}

/// Sample the root joint at every `t` and turn it into the two v2 tracks.
fn bake_tracks(
    clip: &AnimClip,
    skeleton: &Skeleton,
    root: u16,
    times: &[f32],
    vertical: VerticalPolicy,
) -> (RootMotionTrack, DistanceTrack) {
    let n = times.len();
    let mut raw: Vec<(Vec3, f32)> = Vec::with_capacity(n);
    for &t in times {
        let pose = sample_clip(skeleton, clip, t, false);
        let local = pose.locals.get(root as usize).copied().unwrap_or_default();
        raw.push((
            Vec3::from_array(local.translation),
            inf_math::pyaw(Quat::from_array(local.rotation)),
        ));
    }
    let y0 = raw.first().map(|(p, _)| p.y).unwrap_or(0.0);
    let y1 = raw.last().map(|(p, _)| p.y).unwrap_or(0.0);
    let span = times.last().copied().unwrap_or(0.0) - times.first().copied().unwrap_or(0.0);

    let mut translation = Vec::with_capacity(n);
    let mut yaw_rad = Vec::with_capacity(n);
    let mut distance_m = Vec::with_capacity(n);
    let mut travelled = 0.0f32;
    let mut prev: Option<Vec3> = None;
    for (i, (p, yaw)) in raw.iter().enumerate() {
        let y = match vertical {
            VerticalPolicy::Full => p.y,
            VerticalPolicy::NetDrift => {
                if crate::positive(span) {
                    let a = (times[i] - times[0]) / span;
                    y0 + (y1 - y0) * a
                } else {
                    y0
                }
            }
        };
        let v = Vec3::new(p.x, y, p.z);
        if let Some(q) = prev {
            let d = v - q;
            travelled += (d.x * d.x + d.z * d.z).sqrt();
        }
        prev = Some(v);
        translation.push(v.to_array());
        yaw_rad.push(*yaw);
        distance_m.push(travelled);
    }
    (
        RootMotionTrack {
            times: times.to_vec(),
            translation,
            yaw_rad,
        },
        DistanceTrack {
            times: times.to_vec(),
            distance_m,
        },
    )
}

/// Subtract the baked track from the root joint's own keys, so what is left plays
/// in place.
///
/// Exact, not approximate: every key time of the root's channels is in the
/// track's own time list ([`sample_times`]), so `track.sample(key)` is the stored
/// value and the subtraction cancels. Between two keys the track carries only
/// collinear extra samples of the same segment, so the residual is zero there
/// too — for a linear channel exactly, for a stepped one by the same argument,
/// and for the rotation to within the difference between a slerp and a lerp of
/// one angle, which is what removing a yaw from a quaternion track costs.
fn remove_root_residual(clip: &mut AnimClip, root: u16, track: &RootMotionTrack) {
    if !clip.tracks.iter().any(|t| t.joint == root) {
        return;
    }
    let Some(jt) = clip.tracks.iter_mut().find(|t| t.joint == root) else {
        return;
    };
    if let Some(ch) = jt.translation.as_mut() {
        for (i, t) in ch.times.iter().enumerate() {
            let Some((p, _)) = track.sample(*t) else {
                continue;
            };
            if let Some(v) = ch.values.get_mut(i) {
                v[0] -= p[0];
                v[1] -= p[1];
                v[2] -= p[2];
            }
        }
    }
    if let Some(ch) = jt.rotation.as_mut() {
        for (i, t) in ch.times.iter().enumerate() {
            let Some((_, yaw)) = track.sample(*t) else {
                continue;
            };
            if let Some(v) = ch.values.get_mut(i) {
                let q = yaw_quat(-(yaw as f64)) * Quat::from_array(*v);
                *v = q.to_array();
            }
        }
    }
}

/// **The rig's feet**, in joint order.
///
/// By name, because a foot is a role and not a position: `foot_l`, `foot_r`,
/// `foot_l0` … are what [`crate::template`] emits and what the retarget
/// vocabulary uses. An imported rig that spells them `ik_foot_l` or `LeftFoot`
/// is caught by the second pass, which is a `contains` rather than a prefix — and
/// only runs when the first found nothing, so a rig with both does not report
/// four feet.
pub fn foot_joints(skeleton: &Skeleton) -> Vec<u16> {
    let strict: Vec<u16> = skeleton
        .joints()
        .iter()
        .enumerate()
        .filter(|(_, j)| j.name.starts_with("foot_"))
        .map(|(i, _)| i as u16)
        .collect();
    if !strict.is_empty() {
        return strict;
    }
    skeleton
        .joints()
        .iter()
        .enumerate()
        .filter(|(_, j)| j.name.to_ascii_lowercase().contains("foot"))
        .map(|(i, _)| i as u16)
        .collect()
}

/// The name a foot's markers are spelled with.
///
/// Walks **up the rig** for the leg the foot belongs to (`upper_leg_*`), because
/// that is the name [`crate::locomotion::FOOT_SYNC_GROUP`]'s generated markers
/// already carry — so a derived run and a generated walk on the same rig name the
/// same marker and a blend between them aligns by name rather than by falling
/// back to the ordinal. A rig with no such ancestor is named after its foot,
/// which is the honest answer when there is no leg to name.
fn leg_name_of(skeleton: &Skeleton, foot: u16) -> String {
    let joints = skeleton.joints();
    let mut cur = Some(foot);
    let mut hops = 0usize;
    while let Some(i) = cur {
        let Some(j) = joints.get(i as usize) else {
            break;
        };
        if j.name.starts_with("upper_leg_") {
            return j.name.clone();
        }
        // Bounded: a malformed chain must not spin. `Skeleton::new` already
        // refuses cycles, so this is the belt to that brace.
        hops += 1;
        if hops > joints.len() {
            break;
        }
        cur = j.parent;
    }
    joints
        .get(foot as usize)
        .map(|j| j.name.clone())
        .unwrap_or_default()
}

/// The suffix a per-foot channel is named with: `foot_l` → `L`, `foot_r0` → `R0`.
///
/// Uppercased so a biped's channels come out as `FootSpeed_L` / `FootLock_R`,
/// which is both ALS's vocabulary ([`als::FOOT_LOCK_L`]) and the stock sample's
/// (`AM_FootSpeed_L`) — the same names two different donors chose, which is a
/// good sign they are the right ones.
fn foot_suffix(skeleton: &Skeleton, foot: u16) -> String {
    let name = skeleton
        .joints()
        .get(foot as usize)
        .map(|j| j.name.as_str())
        .unwrap_or("");
    let tail = name.rsplit('_').next().unwrap_or(name);
    tail.to_ascii_uppercase()
}

/// One foot at one sample.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FootSample {
    /// Model-space position.
    pos: Vec3,
    /// Full speed, m/s — what the stock sample's `AM_FootSpeed_*` bakes
    /// (`MotionExtractor` in `TranslationSpeed` mode) and what the channel
    /// carries.
    speed: f32,
}

/// Every foot's position and speed, sampled on `times`.
fn foot_traces(
    clip: &AnimClip,
    skeleton: &Skeleton,
    feet: &[u16],
    times: &[f32],
) -> Vec<Vec<FootSample>> {
    let mut out: Vec<Vec<FootSample>> = vec![Vec::with_capacity(times.len()); feet.len()];
    let mut prev: Vec<Option<Vec3>> = vec![None; feet.len()];
    let mut prev_t = times.first().copied().unwrap_or(0.0);
    for &t in times {
        let pose = sample_clip(skeleton, clip, t, false);
        let globals = global_transforms(skeleton, &pose);
        let dt = t - prev_t;
        for (k, &joint) in feet.iter().enumerate() {
            let pos = globals
                .get(joint as usize)
                .map(|m| m.w_axis.truncate())
                .unwrap_or(Vec3::ZERO);
            let speed = match (prev[k], crate::positive(dt)) {
                (Some(q), true) => (pos - q).length() / dt,
                _ => 0.0,
            };
            out[k].push(FootSample { pos, speed });
            prev[k] = Some(pos);
        }
        prev_t = t;
    }
    // The first sample has no predecessor, so it borrows the second's speed
    // rather than reporting a zero that would read as a plant.
    for tr in out.iter_mut() {
        if tr.len() >= 2 {
            tr[0].speed = tr[1].speed;
        }
    }
    out
}

/// **The stride, and the speed it implies** — `(stride_m, stride × cadence)`.
///
/// The stride is a foot's **planar excursion** over the clip: how far it travels
/// along the ground between the extremes of its own swing. That is the one
/// definition that reads the same on a root-motion clip and an in-place one,
/// because it is measured after the root residual is removed — in both, a foot
/// goes back by one stride under the body and forward by one stride over it.
///
/// The cadence is how often the clip **plants** that foot: `plants / duration`.
/// `stride × cadence` is then the definition of gait speed, and it does not care
/// whether anything translated. That is what makes a proposal work over content
/// that plays in place, which is most authored locomotion and every clip
/// [`crate::locomotion`] generates.
///
/// Averaged **per foot** rather than pooled: a creature whose left leg swings
/// further than its right depicts one speed, not a weighted vote, and pooling
/// would let the busier leg decide.
fn stride_and_speed(
    feet: &[u16],
    traces: &[Vec<FootSample>],
    plants: &[FootPlant],
    duration: f32,
) -> (f32, f32) {
    if !crate::positive(duration) {
        return (0.0, 0.0);
    }
    let mut stride_total = 0.0f32;
    let mut speed_total = 0.0f32;
    let mut counted = 0usize;
    for (k, &joint) in feet.iter().enumerate() {
        let steps = plants.iter().filter(|p| p.joint == joint).count();
        if steps == 0 {
            continue;
        }
        let Some(trace) = traces.get(k) else {
            continue;
        };
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for s in trace {
            if s.pos.is_finite() {
                lo = lo.min(s.pos);
                hi = hi.max(s.pos);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            continue;
        }
        let (dx, dz) = (hi.x - lo.x, hi.z - lo.z);
        let stride = (dx * dx + dz * dz).sqrt();
        stride_total += stride;
        speed_total += stride * (steps as f32 / duration);
        counted += 1;
    }
    if counted == 0 {
        (0.0, 0.0)
    } else {
        (stride_total / counted as f32, speed_total / counted as f32)
    }
}

/// Contact windows, foot by foot, off a grid and traces the caller already has.
fn foot_plants(
    skeleton: &Skeleton,
    feet: &[u16],
    times: &[f32],
    traces: &[Vec<FootSample>],
    duration: f32,
    opts: &DeriveOptions,
) -> Vec<FootPlant> {
    if feet.is_empty() || times.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (k, &joint) in feet.iter().enumerate() {
        let tr = &traces[k];
        let lowest = tr
            .iter()
            .map(|s| s.pos.y)
            .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
        if !lowest.is_finite() {
            continue;
        }
        let ceiling = lowest + opts.contact_lift_m;
        let down: Vec<bool> = tr
            .iter()
            .map(|s| crate::at_most(s.pos.y, ceiling))
            .collect();
        // Every frame down is a foot that never leaves the floor — a clip with no
        // step in it (an idle, an aim offset). No plant, which is the answer: a
        // footstep sound on an idle pose is exactly the noise a hand-placed
        // notify makes.
        if down.iter().all(|d| *d) {
            continue;
        }
        let name = leg_name_of(skeleton, joint);
        for (start, end) in contact_runs(&down, opts.looping) {
            let t0 = times[start];
            let t1 = if end >= times.len() {
                duration + times[end - times.len()]
            } else {
                times[end]
            };
            if !crate::greater(t1 - t0, opts.min_contact_s) {
                continue;
            }
            out.push(FootPlant {
                joint,
                name: name.clone(),
                start_s: t0,
                end_s: t1,
            });
        }
    }
    out
}

/// The uniform grid the foot analysis runs on. Separate from [`sample_times`]
/// deliberately: a foot's contact is a question about the *whole* rig, so it is
/// asked at a fixed rate rather than wherever the root happens to have keys.
fn uniform_times(duration: f32, fps: f32) -> Vec<f32> {
    let steps = ((duration * fps).ceil() as usize).max(1);
    (0..steps)
        .map(|k| duration * (k as f32) / (steps as f32))
        .collect()
}

/// Contiguous runs of `true`, as `[start, end)` index pairs.
///
/// When `looping`, a run that reaches the last frame and a run that starts at the
/// first are one run through the seam, reported once with the **later** start —
/// which is what stops a cycle authored to begin mid-stance from firing a
/// footstep at time zero on every loop.
fn contact_runs(down: &[bool], looping: bool) -> Vec<(usize, usize)> {
    let n = down.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if down[i] {
            let start = i;
            while i < n && down[i] {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    if looping && runs.len() >= 2 {
        let last = runs.len() - 1;
        if runs[0].0 == 0 && runs[last].1 == n {
            let head = runs.remove(0);
            let tail = runs.pop().expect("two runs, one removed");
            runs.push((tail.0, n + head.1));
            runs.sort_by_key(|r| r.0);
        }
    }
    runs
}

/// The markers a set of plants implies: one sync marker and one footstep event
/// per contact, at the instant the foot came down.
fn derived_markers(plants: &[FootPlant]) -> Vec<AnimMarker> {
    let mut out = Vec::with_capacity(plants.len() * 2);
    for p in plants {
        out.push(AnimMarker::sync(
            p.start_s,
            plant_marker(&p.name),
            crate::locomotion::FOOT_SYNC_GROUP,
        ));
        out.push(AnimMarker::event(p.start_s, footstep_marker(&p.name)));
    }
    out.sort_by(|a, b| {
        a.time_s
            .total_cmp(&b.time_s)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

/// The curve channels a derivation writes.
#[allow(clippy::too_many_arguments)]
fn derived_curves(
    skeleton: &Skeleton,
    feet: &[u16],
    times: &[f32],
    foot_times: &[f32],
    traces: &[Vec<FootSample>],
    distance: &DistanceTrack,
    plants: &[FootPlant],
    duration: f32,
    gait: f32,
) -> Vec<CurveChannel> {
    let mut out = Vec::new();
    // The clip's ground speed, from the distance track's own slope.
    let mut speed = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        let (i0, i1) = (i.saturating_sub(1), (i + 1).min(times.len() - 1));
        let dt = times[i1] - times[i0];
        let dd = distance.distance_m[i1] - distance.distance_m[i0];
        speed.push(if crate::positive(dt) { dd / dt } else { 0.0 });
    }
    out.push(CurveChannel::new(
        MOVE_DATA_SPEED,
        times.to_vec(),
        speed,
        Interpolation::Linear,
    ));
    out.push(CurveChannel::constant(als::W_GAIT, gait));

    for (k, &joint) in feet.iter().enumerate() {
        let Some(trace) = traces.get(k) else {
            continue;
        };
        let suffix = foot_suffix(skeleton, joint);
        out.push(CurveChannel::new(
            format!("{FOOT_SPEED_PREFIX}{suffix}"),
            foot_times.to_vec(),
            trace.iter().map(|s| s.speed).collect(),
            Interpolation::Linear,
        ));
        out.push(CurveChannel::new(
            format!("{FOOT_LOCK_PREFIX}{suffix}"),
            foot_times.to_vec(),
            foot_times
                .iter()
                .map(|t| {
                    let locked = plants.iter().any(|p| {
                        p.joint == joint
                            && ((*t >= p.start_s && *t <= p.end_s)
                                || (p.end_s > duration && *t <= p.end_s - duration))
                    });
                    if locked {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            Interpolation::Step,
        ));
    }
    out
}

/// **ALS's `W_Gait` scale**, derived: 0 stopped, 1 at walk speed, 2 at run,
/// 3 at sprint, piecewise-linear between and clamped outside.
///
/// The same shape as P29.3's `GetMappedSpeed` model on the runtime side, spelled
/// here because this crate does not depend on that one and because the number
/// this writes is a *clip* property while that one is a *character* property.
pub fn gait_of(speed_mps: f32, tiers: [f32; 3]) -> f32 {
    let [walk, run, sprint] = tiers;
    if !speed_mps.is_finite() || speed_mps <= 0.0 {
        return 0.0;
    }
    let seg = |v: f32, a: f32, b: f32| -> f32 {
        if crate::greater(b, a) {
            ((v - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    if !crate::greater(speed_mps, walk) {
        seg(speed_mps, 0.0, walk)
    } else if !crate::greater(speed_mps, run) {
        1.0 + seg(speed_mps, walk, run)
    } else {
        2.0 + seg(speed_mps, run, sprint)
    }
}

/// **[`gait_of`] read backwards**: the speed a gait value depicts.
///
/// The inverse exists because `W_Gait` is the one derived number a clip carries
/// on the **wire** — the derivation's own `avg_speed_mps` lives in a report that
/// is thrown away after the import — and [`crate::propose`] has to recover a
/// depicted speed from a clip it is reading off disk. An in-place cycle's
/// distance track is flat, so the channel is the only place its speed survives.
///
/// Exact on the round trip at every tier boundary and linear between them, which
/// is what makes `speed_of_gait(gait_of(v)) == v` for any `v` inside the ladder.
pub fn speed_of_gait(gait: f32, tiers: [f32; 3]) -> f32 {
    let [walk, run, sprint] = tiers;
    if !gait.is_finite() || gait <= 0.0 {
        return 0.0;
    }
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t.clamp(0.0, 1.0);
    if gait <= 1.0 {
        lerp(0.0, walk, gait)
    } else if gait <= 2.0 {
        lerp(walk, run, gait - 1.0)
    } else {
        lerp(run, sprint, gait - 2.0)
    }
}

/// Rebuild `clip`'s root joint channels from a `[times, values]` pair — the
/// helper the tests below and [`crate::propose`]'s fixtures share.
pub fn set_root_translation(
    clip: &mut AnimClip,
    root: u16,
    times: Vec<f32>,
    values: Vec<[f32; 3]>,
) {
    let track = Vec3Track::new(times, values, Interpolation::Linear);
    match clip.tracks.iter_mut().find(|t| t.joint == root) {
        Some(jt) => jt.translation = Some(track),
        None => {
            let mut jt = crate::clip::JointTrack::new(root);
            jt.translation = Some(track);
            clip.tracks.push(jt);
        }
    }
}

/// [`set_root_translation`] for the rotation half.
pub fn set_root_rotation(clip: &mut AnimClip, root: u16, times: Vec<f32>, values: Vec<[f32; 4]>) {
    let track = QuatTrack::new(times, values, Interpolation::Linear);
    match clip.tracks.iter_mut().find(|t| t.joint == root) {
        Some(jt) => jt.rotation = Some(track),
        None => {
            let mut jt = crate::clip::JointTrack::new(root);
            jt.rotation = Some(track);
            clip.tracks.push(jt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locomotion::{build_locomotion, GaitParams};
    use crate::skeleton::{Joint, JointTransform};
    use crate::template::{build_template, BodyParams, BodyPlan};
    use glam::Mat4;

    fn one_joint() -> Skeleton {
        Skeleton::new(vec![Joint {
            name: "hips".into(),
            parent: None,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }])
        .unwrap()
    }

    fn biped() -> crate::asset::SkeletonAsset {
        build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    fn bytes(clip: &AnimClip) -> Vec<u8> {
        bincode::serde::encode_to_vec(clip, inf_asset::bincode_config()).unwrap()
    }

    /// **The headline**: a clip whose root travels ends up playing *in place*,
    /// with every metre of that travel on the track — and the two halves still
    /// add up to the clip that was imported.
    #[test]
    fn a_travelling_clip_plays_in_place_and_its_travel_moves_onto_the_track() {
        let sk = one_joint();
        let clip = crate::root_motion::straight_line_clip("walk", Vec3::Z, 2.0, 1.0);
        // Before: the root joint really does move.
        let before = sample_clip(&sk, &clip, 1.0, false).locals[0].translation;
        assert!((before[2] - 2.0).abs() < 1e-5, "{before:?}");

        let (out, report) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();

        // After: it does not.
        let after = sample_clip(&sk, &out, 1.0, false).locals[0].translation;
        assert!(
            after[2].abs() < 1e-5,
            "the residual was not removed: {after:?}"
        );
        assert!(sample_clip(&sk, &out, 0.5, false).locals[0].translation[2].abs() < 1e-5);

        // The travel is on the track, and the distance track agrees with it.
        let track = out.root_motion.as_ref().expect("a baked track");
        let (end, _) = track.sample(1.0).unwrap();
        assert!((end[2] - 2.0).abs() < 1e-5, "{end:?}");
        assert!((report.distance_m - 2.0).abs() < 1e-4, "{report:?}");
        assert!((report.avg_speed_mps - 2.0).abs() < 1e-4, "{report:?}");
        assert!((out.distance.as_ref().unwrap().sample(0.5).unwrap() - 1.0).abs() < 1e-4);

        // …and the consumer that matters reads the track, so a derived clip
        // drives a character exactly as far as the clip it came from.
        let d = crate::root_delta_3d(&out, &sk, 0.0, 1.0, false);
        assert!((d.translation.z - 2.0).abs() < 1e-5, "{d:?}");
        assert!((d.distance_m - 2.0).abs() < 1e-4, "{d:?}");

        // The two halves add back up: un-baking recovers the import.
        let mut restored = out.clone();
        unbake_root_motion(&mut restored, 0);
        let back = sample_clip(&sk, &restored, 0.5, false).locals[0].translation;
        assert!(
            (back[2] - 1.0).abs() < 1e-5,
            "un-bake did not restore: {back:?}"
        );
    }

    /// A yaw is extracted and removed the same way a translation is.
    #[test]
    fn a_turning_clips_yaw_moves_onto_the_track_too() {
        let sk = one_joint();
        let mut clip = AnimClip::new("turn", Vec::new());
        // Half a right angle over one second, written by hand (no `from_rotation_y`
        // on a path that produces committed bytes).
        let q = |a: f64| {
            let h = a * 0.5;
            [
                0.0f32,
                inf_math::psin64(h) as f32,
                0.0,
                inf_math::pcos64(h) as f32,
            ]
        };
        set_root_rotation(
            &mut clip,
            0,
            vec![0.0, 1.0],
            vec![q(0.0), q(std::f64::consts::FRAC_PI_4)],
        );
        clip.duration = 1.0;

        let (out, report) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();
        assert!(
            (report.yaw_rad - std::f32::consts::FRAC_PI_4).abs() < 1e-4,
            "{report:?}"
        );
        let posed = sample_clip(&sk, &out, 1.0, false).locals[0].rotation_quat();
        assert!(
            inf_math::pyaw(posed).abs() < 1e-4,
            "the yaw residual was left in the pose: {}",
            inf_math::pyaw(posed)
        );
        let d = crate::root_delta_3d(&out, &sk, 0.0, 1.0, false);
        assert!((d.yaw - std::f32::consts::FRAC_PI_4).abs() < 1e-4, "{d:?}");
    }

    /// **Conservation, on a clip that does both** (P29.5 audit).
    ///
    /// The in-tree arms measure the residual on a *straight line* and the yaw on
    /// a clip that only turns. A traversal clip does both at once, and the two
    /// removals compose — the translation is subtracted from the root's own
    /// local keys while the yaw is left-multiplied out of its rotation — so
    /// "each half works alone" is not the same claim as "the pair is invertible".
    /// Both halves here: the clip plays in place **and** un-baking recovers the
    /// pose it arrived with, sample for sample, at times that are not keys.
    #[test]
    fn a_clip_that_travels_and_turns_plays_in_place_and_un_bakes_back() {
        let sk = one_joint();
        let q = |a: f64| {
            let h = a * 0.5;
            [
                0.0f32,
                inf_math::psin64(h) as f32,
                0.0,
                inf_math::pcos64(h) as f32,
            ]
        };
        let mut clip = AnimClip::new("vault", Vec::new());
        set_root_translation(
            &mut clip,
            0,
            vec![0.0, 0.5, 1.0],
            vec![[0.0, 0.0, 0.0], [0.3, 0.6, 0.55], [0.4, 0.9, 1.2]],
        );
        set_root_rotation(
            &mut clip,
            0,
            vec![0.0, 0.5, 1.0],
            vec![
                q(0.0),
                q(std::f64::consts::FRAC_PI_6),
                q(std::f64::consts::FRAC_PI_3),
            ],
        );
        clip.duration = 1.0;

        let before: Vec<_> = (0..=20)
            .map(|k| {
                let t = k as f32 / 20.0;
                let p = sample_clip(&sk, &clip, t, false).locals[0];
                (t, p.translation, p.rotation)
            })
            .collect();

        let (out, report) = derive_clip(&clip, &sk, &DeriveOptions::traversal()).unwrap();

        // 1. THE RESIDUAL. Not "a track exists" — the root itself no longer
        //    moves or turns, at every sample and not only at the keys.
        for (t, _, _) in &before {
            let p = sample_clip(&sk, &out, *t, false).locals[0];
            assert!(
                p.translation_vec().length() < 1e-5,
                "the root still travels at t={t}: {:?}",
                p.translation
            );
            assert!(
                inf_math::pyaw(p.rotation_quat()).abs() < 1e-4,
                "the root still turns at t={t}: {}",
                inf_math::pyaw(p.rotation_quat())
            );
        }
        // …and what it stopped doing is on the track.
        assert!((report.translation[1] - 0.9).abs() < 1e-5, "{report:?}");
        assert!(
            (report.yaw_rad - std::f32::consts::FRAC_PI_3).abs() < 1e-4,
            "{report:?}"
        );

        // 2. CONSERVATION. Un-baking gives back the clip that arrived.
        let mut restored = out.clone();
        unbake_root_motion(&mut restored, 0);
        for (t, translation, rotation) in &before {
            let p = sample_clip(&sk, &restored, *t, false).locals[0];
            let d = p.translation_vec() - Vec3::from_array(*translation);
            assert!(
                d.length() < 1e-5,
                "t={t}: {:?} vs {translation:?}",
                p.translation
            );
            let a = p.rotation_quat();
            let b = Quat::from_array(*rotation);
            assert!(
                a.dot(b).abs() > 1.0 - 1e-6,
                "t={t}: {:?} vs {rotation:?}",
                p.rotation
            );
        }
    }

    /// **Idempotence, byte for byte.** The property the whole un-bake-first rule
    /// exists for: a re-derive from the panel must not eat the track it already
    /// wrote.
    #[test]
    fn deriving_twice_is_deriving_once() {
        let sk = one_joint();
        let clip = crate::root_motion::straight_line_clip("walk", Vec3::Z, 2.0, 1.0);
        let (once, r1) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();
        let (twice, r2) = derive_clip(&once, &sk, &DeriveOptions::default()).unwrap();
        assert_eq!(bytes(&once), bytes(&twice), "a re-derive moved bytes");
        assert_eq!(r1, r2);
        let (thrice, _) = derive_clip(&twice, &sk, &DeriveOptions::default()).unwrap();
        assert_eq!(bytes(&once), bytes(&thrice));
    }

    /// **Two processes, one answer.** Derivation is a pure function of its
    /// inputs, so this asserts the only thing a single process can: that two
    /// independent derivations of two independently-built inputs agree byte for
    /// byte. The cross-process half is `anim_derivation_twin`.
    #[test]
    fn two_derivations_of_the_same_input_are_byte_equal() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let a = derive_clip(&set.walk, &sk.skeleton, &DeriveOptions::default()).unwrap();
        let set2 = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let b = derive_clip(&set2.walk, &sk.skeleton, &DeriveOptions::default()).unwrap();
        assert_eq!(bytes(&a.0), bytes(&b.0));
        assert_eq!(a.1, b.1);
    }

    /// The vertical policy decides where a bob lives, and the default keeps it in
    /// the pose — which is the whole reason the policy exists (see
    /// [`VerticalPolicy`]).
    #[test]
    fn the_vertical_policy_decides_where_the_bob_goes() {
        let sk = one_joint();
        let mut clip = AnimClip::new("bob", Vec::new());
        // Up 10 cm and back down: net zero, oscillation 0.1.
        set_root_translation(
            &mut clip,
            0,
            vec![0.0, 0.5, 1.0],
            vec![[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]],
        );
        clip.duration = 1.0;

        let (net, _) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();
        let track = net.root_motion.as_ref().unwrap();
        assert!(
            track.sample(0.5).unwrap().0[1].abs() < 1e-6,
            "a net-zero bob must not reach the track"
        );
        assert!(
            (sample_clip(&sk, &net, 0.5, false).locals[0].translation[1] - 0.1).abs() < 1e-6,
            "the bob must stay in the pose"
        );

        let full = derive_clip(
            &clip,
            &sk,
            &DeriveOptions {
                vertical: VerticalPolicy::Full,
                ..DeriveOptions::default()
            },
        )
        .unwrap()
        .0;
        assert!(
            (full.root_motion.as_ref().unwrap().sample(0.5).unwrap().0[1] - 0.1).abs() < 1e-6,
            "Full must put the whole vertical on the track"
        );
        assert!(
            sample_clip(&sk, &full, 0.5, false).locals[0].translation[1].abs() < 1e-6,
            "…and take it out of the pose"
        );
    }

    /// **A generated walk's feet are found, named the way the generator names
    /// them, and marked once each.**
    ///
    /// The name agreement is the load-bearing half: a derived run and a
    /// [`crate::locomotion`]-generated walk on the same rig must place markers a
    /// sync group can match **by name**, not by falling through to the ordinal.
    #[test]
    fn a_generated_walk_gets_one_plant_per_foot_named_after_its_leg() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let (out, report) =
            derive_clip(&set.walk, &sk.skeleton, &DeriveOptions::default()).unwrap();
        assert_eq!(
            report.plants.len(),
            2,
            "a biped walk cycle plants twice: {:?}",
            report
                .plants
                .iter()
                .map(|p| (p.name.as_str(), p.start_s, p.end_s))
                .collect::<Vec<_>>()
        );
        let derived: std::collections::BTreeSet<&str> = out
            .markers
            .iter()
            .filter(|m| m.is_sync() && m.name.starts_with(PLANT_PREFIX))
            .map(|m| m.name.as_str())
            .collect();
        let generated: std::collections::BTreeSet<&str> = set
            .walk
            .markers
            .iter()
            .filter(|m| m.is_sync())
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(
            derived, generated,
            "a derived plant and a generated one must be the same marker"
        );
        // A footstep rings for each, and it is spelled the way the audio drain
        // recognises (`inf_ecs::anim_bridge::FOOTSTEP_PREFIX`).
        assert_eq!(
            out.markers
                .iter()
                .filter(|m| !m.is_sync() && m.name.starts_with(FOOTSTEP_PREFIX))
                .count(),
            2
        );
        // And the sync group is the one generated content already uses.
        assert!(out
            .sync_groups()
            .contains(crate::locomotion::FOOT_SYNC_GROUP));
    }

    /// A pose that never lifts a foot is not a walk, and gets no footsteps — the
    /// failure mode a hand-placed notify makes most often.
    #[test]
    fn an_idle_clip_gets_no_footsteps() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let (out, report) =
            derive_clip(&set.idle, &sk.skeleton, &DeriveOptions::default()).unwrap();
        assert!(report.plants.is_empty(), "{:?}", report.plants);
        assert!(out
            .markers
            .iter()
            .all(|m| !m.name.starts_with(FOOTSTEP_PREFIX)));
        // It is still derived: the speed and gait channels are there, and the
        // gait is zero because the clip depicts standing still.
        assert!(out.curve(MOVE_DATA_SPEED).is_some());
        assert_eq!(out.curve_value(als::W_GAIT, 0.0, -1.0), 0.0);
    }

    /// The lock channel is high **exactly** during the plant it came from, which
    /// is the contract P29.4's foot lock reads (`LOCK_ENGAGE` is 0.99).
    #[test]
    fn the_lock_channel_is_high_exactly_during_its_own_plant() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let (out, report) =
            derive_clip(&set.walk, &sk.skeleton, &DeriveOptions::default()).unwrap();
        let plant = report.plants.first().expect("a plant");
        let suffix = foot_suffix(&sk.skeleton, plant.joint);
        let name = format!("{FOOT_LOCK_PREFIX}{suffix}");
        let mid = (plant.start_s + plant.end_s.min(set.walk.duration)) * 0.5;
        assert!(
            out.curve_value(&name, mid, 0.0) >= crate::foot::LOCK_ENGAGE,
            "the lock is not engaged mid-plant"
        );
        // Somewhere outside every window of that foot it is not engaged.
        let free = (0..60)
            .map(|k| set.walk.duration * k as f32 / 60.0)
            .find(|t| {
                report
                    .plants
                    .iter()
                    .filter(|p| p.joint == plant.joint)
                    .all(|p| *t < p.start_s || *t > p.end_s)
            })
            .expect("a frame with this foot in the air");
        assert!(out.curve_value(&name, free, 1.0) < crate::foot::LOCK_ENGAGE);
    }

    /// **An in-place cycle says how fast it is.**
    ///
    /// The check that matters is not a tolerance, it is a *second opinion*:
    /// `LocomotionSet::walk_speed_m_s` is the generator's own number, computed
    /// from the leg geometry it authored the clip with (`stride × cadence`, where
    /// `stride = 2 L sin θ`). The derivation never sees any of that — it watches
    /// the feet slide backwards under a body that does not move. Two independent
    /// routes to the same metres per second, and if they agree the mechanism is
    /// measuring the thing it claims to.
    #[test]
    fn an_in_place_cycle_reports_the_speed_its_planted_feet_slide_at() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        for (clip, depicted, what) in [
            (&set.walk, set.walk_speed_m_s as f32, "walk"),
            (&set.run, set.run_speed_m_s as f32, "run"),
        ] {
            let (_, r) = derive_clip(clip, &sk.skeleton, &DeriveOptions::default()).unwrap();
            assert_eq!(
                r.travel_speed_mps, 0.0,
                "a generated cycle is in place, so nothing travels ({what})"
            );
            assert!(
                r.stride_speed_mps > 0.0,
                "{what}: the feet must slide backwards under a body that does not move"
            );
            assert_eq!(r.avg_speed_mps, r.stride_speed_mps);
            let err = (r.stride_speed_mps - depicted).abs() / depicted;
            // Measured: 4.1% on the walk, 3.9% on the run. The residue is the
            // knee — the generator's `mean_leg_m` is the straight hip-to-foot
            // distance and the animated leg flexes, so the foot's real excursion
            // is not exactly `2 L sin θ`. Bounded at 8%, twice the measurement.
            assert!(
                err < 0.08,
                "{what}: the derivation reads {:.3} m/s where the generator's own \
                 stride x cadence says {depicted:.3} m/s ({:.0}% apart)",
                r.stride_speed_mps,
                err * 100.0
            );
            // …and the run really is faster than the walk, which is the property
            // a proposal clusters on.
            assert!(r.stride_m > 0.0, "{what}: an in-place stride is not zero");
        }
        let walk = derive_clip(&set.walk, &sk.skeleton, &DeriveOptions::default())
            .unwrap()
            .1;
        let run = derive_clip(&set.run, &sk.skeleton, &DeriveOptions::default())
            .unwrap()
            .1;
        assert!(
            run.stride_speed_mps > walk.stride_speed_mps * 1.2,
            "a run must read faster than a walk: {} vs {}",
            run.stride_speed_mps,
            walk.stride_speed_mps
        );
    }

    /// The other side of the same coin: a clip whose ROOT travels reports travel
    /// and not stance, because a planted foot in such a clip really is stationary.
    #[test]
    fn a_root_motion_clip_reports_its_travel() {
        let sk = one_joint();
        let clip = crate::root_motion::straight_line_clip("walk", Vec3::Z, 2.0, 1.0);
        let (_, r) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();
        assert!((r.travel_speed_mps - 2.0).abs() < 1e-4, "{r:?}");
        assert_eq!(r.stride_speed_mps, 0.0, "a one-joint rig has no feet");
        assert_eq!(r.avg_speed_mps, r.travel_speed_mps);
    }

    /// The gait scale inverts exactly at every tier boundary — which is what lets
    /// a proposal recover an in-place clip's speed from the one channel that
    /// survives onto the wire.
    #[test]
    fn the_gait_scale_reads_backwards() {
        let t = [1.65f32, 3.75, 6.5];
        for v in [0.0f32, 0.4, 1.65, 2.7, 3.75, 5.0, 6.5] {
            let back = speed_of_gait(gait_of(v, t), t);
            assert!((back - v).abs() < 1e-4, "{v} -> {back}");
        }
        // Past the top the ladder clamps, so the inverse clamps with it.
        assert_eq!(speed_of_gait(gait_of(99.0, t), t), 6.5);
        assert_eq!(speed_of_gait(f32::NAN, t), 0.0);
        assert_eq!(speed_of_gait(-1.0, t), 0.0);
    }

    /// The gait scale is ALS's, and the stride is measured off the distance track
    /// rather than guessed.
    #[test]
    fn the_gait_scale_is_zero_one_two_three() {
        let t = [1.65f32, 3.75, 6.5];
        assert_eq!(gait_of(0.0, t), 0.0);
        assert_eq!(gait_of(1.65, t), 1.0);
        assert_eq!(gait_of(3.75, t), 2.0);
        assert_eq!(gait_of(6.5, t), 3.0);
        assert!((gait_of(0.825, t) - 0.5).abs() < 1e-6);
        assert!((gait_of(2.7, t) - 1.5).abs() < 1e-6);
        assert_eq!(gait_of(99.0, t), 3.0, "clamps rather than extrapolates");
        assert_eq!(gait_of(f32::NAN, t), 0.0);
    }

    /// Every refusal is a **value**: a clip this module cannot derive is still a
    /// clip, and the import that meets one keeps it.
    #[test]
    fn refusals_are_values_and_name_themselves() {
        let sk = one_joint();
        let clip = crate::root_motion::straight_line_clip("walk", Vec3::Z, 1.0, 1.0);
        assert_eq!(
            derive_clip(
                &clip,
                &sk,
                &DeriveOptions {
                    fps: 0.0,
                    ..DeriveOptions::default()
                }
            ),
            Err(DeriveError::BadRate { fps: 0.0 })
        );
        let empty = AnimClip::new("static", Vec::new());
        assert_eq!(
            derive_clip(&empty, &sk, &DeriveOptions::default()),
            Err(DeriveError::NoDuration {
                name: "static".into()
            })
        );
    }

    /// A traversal clip asks for the whole vertical, and gets it — this is the
    /// shape `inf_physics::d3::movement::step_mantle` warps onto a ledge.
    #[test]
    fn a_traversal_clip_keeps_its_whole_rise() {
        let sk = one_joint();
        let mut clip = AnimClip::new("mantle", Vec::new());
        set_root_translation(
            &mut clip,
            0,
            vec![0.0, 0.5, 1.0],
            // An ease-out arc: most of the rise happens early, so a warp that
            // scales this is visibly not a lerp.
            vec![[0.0, 0.0, 0.0], [0.0, 0.75, 0.3], [0.0, 1.0, 0.8]],
        );
        clip.duration = 1.0;
        let (out, report) = derive_clip(&clip, &sk, &DeriveOptions::traversal()).unwrap();
        assert!((report.translation[1] - 1.0).abs() < 1e-6, "{report:?}");
        assert!((report.translation[2] - 0.8).abs() < 1e-6, "{report:?}");
        let track = out.root_motion.as_ref().unwrap();
        assert!((track.sample(0.5).unwrap().0[1] - 0.75).abs() < 1e-6);
        assert!(
            sample_clip(&sk, &out, 0.5, false).locals[0]
                .translation_vec()
                .length()
                < 1e-6,
            "a traversal clip plays in place too"
        );
    }

    /// **A poisoned pose is refused, not baked** (P29.5 audit, A1).
    ///
    /// `validate_timing` asks about key *times*, so a clip holding a NaN
    /// translation **value** is a perfectly legal `.inf_anim` — and sampling one
    /// poses a NaN, which every derived field inherits. Two of them are worse
    /// than useless: `DistanceTrack::distance_m` has a monotonicity check at the
    /// asset door, so baking a NaN one turns a clip that loads into a clip that
    /// does not (the writer manufacturing a file its own reader rejects); and
    /// the foot-speed channels are read by P29.4's lock with no guard at all.
    ///
    /// The control is the same clip with the NaN removed, which must derive —
    /// otherwise this asserts nothing about NaN and everything about the fixture.
    #[test]
    fn a_clip_that_poses_a_nan_is_refused_rather_than_baked() {
        let sk = one_joint();
        let mut clip = AnimClip::new("poisoned", Vec::new());
        set_root_translation(
            &mut clip,
            0,
            vec![0.0, 1.0],
            vec![[0.0, 0.0, 0.0], [f32::NAN, 0.0, 1.0]],
        );
        clip.duration = 1.0;
        // The fixture really is a legal clip today — that is what makes this a
        // door and not a decode error somewhere upstream.
        assert!(clip.validate_timing().is_ok(), "the fixture must be legal");

        let err = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap_err();
        assert_eq!(
            err,
            DeriveError::NonFinite {
                name: "poisoned".into(),
                what: "root-motion track".into(),
            },
            "a NaN pose must be a refusal by name"
        );

        // THE CONTROL: the same clip, finite, derives — and the refused one is
        // still exactly the clip that arrived (a refusal is a value: the caller
        // keeps what it had).
        let mut ok = clip.clone();
        set_root_translation(
            &mut ok,
            0,
            vec![0.0, 1.0],
            vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        );
        let (baked, _) = derive_clip(&ok, &sk, &DeriveOptions::default()).unwrap();
        assert!(baked.validate_timing().is_ok());
        assert!(baked.root_motion.is_some());

        // …and the arm that says why it matters: had the NaN been baked, the
        // asset's own door would have refused the result.
        let mut poisoned = clip.clone();
        poisoned.distance = Some(DistanceTrack {
            times: vec![0.0, 1.0],
            distance_m: vec![0.0, f32::NAN],
        });
        assert!(
            poisoned.validate_timing().is_err(),
            "a NaN distance track must be refused by the asset door, or this \
             refusal is defending nothing"
        );
    }

    /// **The ownership boundary** (P29.5 audit, A2): a re-derive replaces what
    /// this rig's derivation writes and leaves everything else alone.
    ///
    /// `footstep_land` is the name `anim_bridge::FOOTSTEP_PREFIX`'s own docs
    /// invite an author to add ("a project that adds `footstep_land` gets it for
    /// free"), and a prefix rule deletes it on the next press of the re-derive
    /// button. So does it delete a `plant_hand_l` sync marker somebody put in a
    /// hands group. Both are authored content and neither is this module's.
    #[test]
    fn a_re_derive_keeps_authored_markers_it_did_not_write() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let authored = set
            .walk
            .clone()
            .with_markers(vec![
                AnimMarker::event(0.4, "footstep_land"),
                AnimMarker::sync(0.25, "plant_hand_l", "hands"),
            ])
            .with_curves(vec![CurveChannel::constant("FootLock_Weapon", 0.5)]);
        // The generator already put its own `plant_*` markers on: keep them
        // beside the authored ones, or the assertion below is about an empty set.
        let (out, report) =
            derive_clip(&authored, &sk.skeleton, &DeriveOptions::default()).unwrap();

        assert!(
            out.markers.iter().any(|m| m.name == "footstep_land"),
            "an authored footstep notify was deleted: {:?}",
            out.markers.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
        assert!(
            out.markers
                .iter()
                .any(|m| m.name == "plant_hand_l" && m.group == "hands"),
            "an authored sync marker in another group was deleted"
        );
        assert_eq!(
            out.curve_value("FootLock_Weapon", 0.0, -1.0),
            0.5,
            "a channel that borrows a derived prefix without naming a foot is not this module's"
        );

        // …and what it DOES own is still replaced rather than duplicated: the
        // derived plants appear once each, and the report counts only its own.
        let names = DerivedNames::of_skeleton(&sk.skeleton);
        for n in &names.sync_markers {
            assert_eq!(
                out.markers
                    .iter()
                    .filter(|m| m.is_sync() && &m.name == n)
                    .count(),
                1,
                "{n} was duplicated by a re-derive"
            );
        }
        assert_eq!(
            report.markers, 4,
            "two feet, one plant and one footstep each"
        );

        // Re-deriving is still byte-identical with the authored data present.
        let (again, _) = derive_clip(&out, &sk.skeleton, &DeriveOptions::default()).unwrap();
        assert_eq!(bytes(&out), bytes(&again));
    }

    /// A derivation that finds **no** plants still clears the previous run's
    /// markers — the reason [`DerivedNames`] is a could-write set rather than a
    /// did-write one.
    #[test]
    fn a_re_derive_that_finds_no_plants_clears_the_last_ones() {
        let sk = biped();
        let set = build_locomotion(BodyPlan::Biped, &sk, &GaitParams::default()).unwrap();
        let (with, r1) = derive_clip(&set.walk, &sk.skeleton, &DeriveOptions::default()).unwrap();
        assert_eq!(r1.plants.len(), 2);

        // A contact window nothing can meet: every plant disappears.
        let strict = DeriveOptions {
            min_contact_s: 1e6,
            ..DeriveOptions::default()
        };
        let (without, r2) = derive_clip(&with, &sk.skeleton, &strict).unwrap();
        assert!(r2.plants.is_empty(), "{:?}", r2.plants);
        assert_eq!(
            without
                .markers
                .iter()
                .filter(|m| m.name.starts_with(PLANT_PREFIX) || m.name.starts_with(FOOTSTEP_PREFIX))
                .count(),
            0,
            "the previous run's derived markers survived a run that found none: {:?}",
            without.markers.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
    }

    /// A name this module does not own survives a derivation, and one it does is
    /// replaced rather than duplicated.
    #[test]
    fn an_authored_channel_survives_and_a_derived_one_is_replaced() {
        let sk = one_joint();
        let mut clip = crate::root_motion::straight_line_clip("walk", Vec3::Z, 2.0, 1.0);
        clip = clip
            .with_curves(vec![
                CurveChannel::constant(als::MASK_AIM_OFFSET, 0.25),
                CurveChannel::constant(als::W_GAIT, 99.0),
            ])
            .with_markers(vec![AnimMarker::event(0.5, "reload_done")]);
        let (out, _) = derive_clip(&clip, &sk, &DeriveOptions::default()).unwrap();
        assert_eq!(out.curve_value(als::MASK_AIM_OFFSET, 0.0, -1.0), 0.25);
        assert_eq!(
            out.curves.iter().filter(|c| c.name == als::W_GAIT).count(),
            1,
            "a derived channel must be replaced, not duplicated"
        );
        assert_ne!(out.curve_value(als::W_GAIT, 0.0, -1.0), 99.0);
        assert!(out.markers.iter().any(|m| m.name == "reload_done"));
    }
}
