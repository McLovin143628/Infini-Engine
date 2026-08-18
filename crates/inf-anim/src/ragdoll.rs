//! **The ragdoll bridge's animation half** (P29.4, clause 6) — every rule the
//! bridge needs that is a function of numbers, and nothing that needs a
//! simulation.
//!
//! # The doctrine this file exists to satisfy
//!
//! §13's catalogue amendment binds the ragdoll to the **P12 command-queue
//! doctrine**: *blend weights are a pure function of sim state, and physics never
//! mutates the machine directly*. ALS violates both halves — its AnimBP reads the
//! physics body for `FlailRate` and the pose snapshot is pushed imperatively at
//! `RagdollEnd` (port map §4.3). Here the physics side publishes numbers and the
//! animation side reads them, and every decision in between is one of the
//! functions below, so "the same sim state gives the same blend weight" is a
//! property a test can assert rather than a claim about call order.
//!
//! `inf_physics::ragdoll` (P12.1) is the *other* half: the pure builder that
//! turns a humanoid skeleton into body / collider / joint descriptors. It has
//! been in the tree since P12 with no runtime consumer; this wave is its first.
//!
//! # The constants are ALS's, converted once
//!
//! The motor-drive band `0…1000 → 0…25000` (`:666–668`) is centimetres per second
//! on its input: `1000 cm/s` is [`MOTOR_SPEED_BAND_MPS`]`.1` = 10 m/s. The
//! gravity cutoff `LastRagdollVelocity.Z > -4000` (`:672`) is −40 m/s. The flail
//! rate's `0…1000` is the same 10 m/s. Converted here and nowhere else.

/// The ragdoll's phase — what the bridge is doing with this character.
///
/// A three-state phase rather than a bool because the get-up is not "not
/// ragdolling": the bodies are gone, the machine is playing a get-up, and the
/// blend weight is still a function of how long that has been true.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RagdollPhase {
    /// No ragdoll: the machine owns the pose entirely.
    #[default]
    Inactive,
    /// The articulated bodies own the pose.
    Simulating,
    /// The bodies are gone and the machine is blending back in from the pose
    /// they left behind.
    GettingUp,
}

/// The input band of ALS's velocity-scaled motor drive, **metres per second**
/// (`0…1000` cm/s).
pub const MOTOR_SPEED_BAND_MPS: (f64, f64) = (0.0, 10.0);
/// The output band of that drive — angular-drive stiffness, unit-free as rapier
/// takes it (`0…25000`).
pub const MOTOR_STIFFNESS_BAND: (f64, f64) = (0.0, 25_000.0);
/// Below this vertical speed the ragdoll's gravity is cut, m/s (ALS −4000 cm/s):
/// "prevents continual acceleration … also prevents the ragdoll from going
/// through the floor".
pub const GRAVITY_CUTOFF_MPS: f64 = -40.0;
/// The speed at which the flail blend reaches 1, m/s (ALS 1000 cm/s).
pub const FLAIL_FULL_MPS: f64 = 10.0;

/// **The velocity-scaled motor drive** (ALS `RagdollUpdate`): a fast-moving
/// ragdoll is *stiffer* — it holds its animated pose more — and a slow one goes
/// limp.
///
/// A pure map from the body's speed onto an angular-drive stiffness. Clamped at
/// both ends, and a non-finite speed answers the limp end rather than a stiffness
/// nobody can reason about.
pub fn motor_stiffness(speed_mps: f64) -> f64 {
    map_range(speed_mps.abs(), MOTOR_SPEED_BAND_MPS, MOTOR_STIFFNESS_BAND)
}

/// The most angular damping a ragdoll's limbs are given, at
/// [`MOTOR_SPEED_BAND_MPS`]'s top end.
///
/// See [`angular_damping`] for why this is the unit the drive is expressed in.
pub const MAX_ANGULAR_DAMPING: f64 = 8.0;

/// **The velocity-scaled stiffness, applied**: how much angular damping a
/// ragdoll's bodies get, from the same band [`motor_stiffness`] maps.
///
/// # This is a behaviour port, not a mechanism port, and it says so
///
/// ALS drives the physical-animation **angular motors** of its skeletal mesh
/// (`SetAllMotorsAngularDriveParams`), which pull each bone toward the pose the
/// animation would have had. This engine's `JointKind3D::Spherical` has no motor
/// — a shoulder is a plain ball joint — so there is nothing to give a stiffness
/// to. What is portable is the *behaviour* the donor's comment describes: a fast
/// ragdoll holds its shape and a slow one goes limp. Angular damping is the
/// mechanism this facade has for exactly that, and it is driven from the same
/// 0…10 m/s band so the ported constant stays load-bearing rather than becoming
/// a number in a comment.
///
/// The day `JointKind3D::Spherical` grows a motor, this becomes the drive's
/// stiffness and the constant is already in the right units.
pub fn angular_damping(speed_mps: f64) -> f64 {
    let top = MOTOR_STIFFNESS_BAND.1;
    if !(top > 0.0) {
        return 0.0;
    }
    motor_stiffness(speed_mps) / top * MAX_ANGULAR_DAMPING
}

/// Whether the ragdoll should still feel gravity, given its vertical speed.
pub fn gravity_enabled(vertical_mps: f64) -> bool {
    // A NaN answers `true`: a ragdoll that keeps its gravity behaves like every
    // other body, and a ragdoll that silently loses it floats.
    !(vertical_mps <= GRAVITY_CUTOFF_MPS)
}

/// The flail blend `[0, 1]` — how much of a limb-waving overlay a falling
/// ragdoll gets, from the root body's speed (ALS `UpdateRagdollValues`).
pub fn flail_rate(speed_mps: f64) -> f32 {
    map_range(speed_mps.abs(), (0.0, FLAIL_FULL_MPS), (0.0, 1.0)) as f32
}

/// **Face-up, read from the pelvis roll sign** — the exact mechanism §13's
/// catalogue row names (`bRagdollFaceUp = PelvisRotation.Roll < 0`).
///
/// `reversed_pelvis` is ALS's `bReversedPelvis` escape hatch for a rig whose
/// pelvis bone points the other way; it flips the comparison rather than the
/// answer, which is what keeps a roll of exactly zero ("on its side, ambiguous")
/// answering the same way for both conventions.
pub fn face_up_from_pelvis_roll(roll_deg: f64, reversed_pelvis: bool) -> bool {
    if !roll_deg.is_finite() {
        return false;
    }
    if reversed_pelvis {
        roll_deg > 0.0
    } else {
        roll_deg < 0.0
    }
}

/// The body yaw a ragdolled capsule should take from its pelvis — **yaw only**,
/// flipped 180° when the character is face-up (ALS `:702`).
///
/// The flip is not cosmetic: a supine character's pelvis points its "forward"
/// into the ground, so a capsule that took the pelvis yaw unflipped would stand
/// the character up backwards.
pub fn body_yaw_from_pelvis(pelvis_yaw_deg: f64, face_up: bool) -> f64 {
    if !pelvis_yaw_deg.is_finite() {
        return 0.0;
    }
    let y = if face_up {
        pelvis_yaw_deg - 180.0
    } else {
        pelvis_yaw_deg
    };
    let r = y % 360.0;
    if r < 0.0 {
        r + 360.0
    } else {
        r
    }
}

/// Which get-up a character on the ground needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GetUp {
    /// On its back.
    Supine,
    /// On its front.
    Prone,
}

impl GetUp {
    /// The get-up owed to a pelvis roll — the two halves of §13's row in one
    /// call.
    pub fn from_pelvis_roll(roll_deg: f64, reversed_pelvis: bool) -> Self {
        if face_up_from_pelvis_roll(roll_deg, reversed_pelvis) {
            GetUp::Supine
        } else {
            GetUp::Prone
        }
    }

    /// The name this get-up sets on the machine's `get_up_face_up` parameter:
    /// `1.0` supine, `0.0` prone. A number rather than a string because a
    /// condition tree compares numbers, and because a parameter that crosses the
    /// `anim.*` kit is `f64` by that kit's own signature.
    pub fn param_value(self) -> f64 {
        match self {
            GetUp::Supine => 1.0,
            GetUp::Prone => 0.0,
        }
    }
}

/// **The blend weight, as a pure function of sim state.**
///
/// `1.0` means the *physics* pose is what is drawn; `0.0` means the machine's is.
/// A simulating ragdoll is fully physical; a get-up blends the physics pose out
/// over `blend_s`; anything else is fully the machine's.
///
/// The ease is [`crate::warp::warp_ease`]'s smoothstep, so the hand-off has no
/// velocity step at either end — the same reason the warp uses it.
///
/// This function is the doctrine's load-bearing sentence: it takes the phase and
/// a clock and returns a number, so two worlds in the same sim state cannot
/// produce two different blends no matter what order their steps ran in.
pub fn blend_weight(phase: RagdollPhase, time_in_phase_s: f64, blend_s: f64) -> f32 {
    match phase {
        RagdollPhase::Inactive => 0.0,
        RagdollPhase::Simulating => 1.0,
        RagdollPhase::GettingUp => {
            if !(blend_s > 0.0) || !time_in_phase_s.is_finite() {
                return 0.0;
            }
            let a = (time_in_phase_s / blend_s).clamp(0.0, 1.0);
            (1.0 - crate::warp::warp_ease(a)) as f32
        }
    }
}

/// Linear map with a clamp and a degenerate-range guard.
fn map_range(value: f64, from: (f64, f64), to: (f64, f64)) -> f64 {
    let span = from.1 - from.0;
    if !span.is_finite() || span.abs() < f64::EPSILON || !value.is_finite() {
        return to.0;
    }
    let t = ((value - from.0) / span).clamp(0.0, 1.0);
    to.0 + (to.1 - to.0) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_motor_drive_is_als_band_in_metres() {
        assert_eq!(motor_stiffness(0.0), 0.0);
        assert!((motor_stiffness(5.0) - 12_500.0).abs() < 1e-9);
        assert_eq!(motor_stiffness(10.0), 25_000.0);
        assert_eq!(motor_stiffness(99.0), 25_000.0, "clamped, not extrapolated");
        assert_eq!(
            motor_stiffness(-3.0),
            motor_stiffness(3.0),
            "speed, not velocity"
        );
        assert_eq!(motor_stiffness(f64::NAN), 0.0);
        // 1000 cm/s is 10 m/s: the conversion happened once, here.
        assert_eq!(MOTOR_SPEED_BAND_MPS.1, 10.0);
    }

    /// The applied half of the drive: same band, same shape, a unit this facade
    /// actually has.
    #[test]
    fn the_drive_is_applied_as_angular_damping_over_the_same_band() {
        assert_eq!(angular_damping(0.0), 0.0, "a stopped ragdoll is limp");
        assert!((angular_damping(10.0) - MAX_ANGULAR_DAMPING).abs() < 1e-12);
        assert!((angular_damping(5.0) - MAX_ANGULAR_DAMPING / 2.0).abs() < 1e-12);
        assert_eq!(angular_damping(1e9), MAX_ANGULAR_DAMPING, "clamped");
        // It is DERIVED from the ported constant, not a second table: halving the
        // stiffness halves the damping, at every speed.
        for i in 0..=20 {
            let v = i as f64 * 0.7;
            let ratio = angular_damping(v) / MAX_ANGULAR_DAMPING;
            let stiff = motor_stiffness(v) / MOTOR_STIFFNESS_BAND.1;
            assert!(
                (ratio - stiff).abs() < 1e-12,
                "at {v} m/s: {ratio} vs {stiff}"
            );
        }
    }

    #[test]
    fn gravity_is_cut_below_forty_metres_per_second() {
        assert!(gravity_enabled(0.0));
        assert!(gravity_enabled(-39.9));
        assert!(!gravity_enabled(-40.0));
        assert!(!gravity_enabled(-500.0));
        assert!(
            gravity_enabled(f64::NAN),
            "a NaN keeps gravity rather than floating"
        );
    }

    #[test]
    fn the_pelvis_roll_sign_picks_the_get_up() {
        assert!(face_up_from_pelvis_roll(-30.0, false));
        assert!(!face_up_from_pelvis_roll(30.0, false));
        // The reversed-pelvis convention flips the comparison…
        assert!(face_up_from_pelvis_roll(30.0, true));
        // …and zero answers the same way for both, because it is ambiguous.
        assert!(!face_up_from_pelvis_roll(0.0, false));
        assert!(!face_up_from_pelvis_roll(0.0, true));
        assert_eq!(GetUp::from_pelvis_roll(-1.0, false), GetUp::Supine);
        assert_eq!(GetUp::from_pelvis_roll(1.0, false), GetUp::Prone);
        assert_eq!(GetUp::Supine.param_value(), 1.0);
        assert_eq!(GetUp::Prone.param_value(), 0.0);
    }

    #[test]
    fn a_supine_capsule_takes_the_flipped_pelvis_yaw() {
        assert_eq!(body_yaw_from_pelvis(90.0, false), 90.0);
        assert_eq!(body_yaw_from_pelvis(90.0, true), 270.0);
        assert_eq!(body_yaw_from_pelvis(10.0, true), 190.0);
        // Folded into [0, 360) rather than left negative.
        assert_eq!(body_yaw_from_pelvis(-90.0, false), 270.0);
        assert_eq!(body_yaw_from_pelvis(f64::NAN, true), 0.0);
    }

    /// **The doctrine, asserted.** The weight is a function of `(phase, clock)`
    /// and of nothing else — so the same sim state gives the same number, and a
    /// get-up really does end at zero rather than merely approaching it.
    #[test]
    fn the_blend_weight_is_a_pure_function_of_sim_state() {
        assert_eq!(blend_weight(RagdollPhase::Inactive, 99.0, 0.5), 0.0);
        assert_eq!(blend_weight(RagdollPhase::Simulating, 99.0, 0.5), 1.0);
        assert_eq!(blend_weight(RagdollPhase::GettingUp, 0.0, 0.5), 1.0);
        assert_eq!(blend_weight(RagdollPhase::GettingUp, 0.5, 0.5), 0.0);
        assert_eq!(blend_weight(RagdollPhase::GettingUp, 9.9, 0.5), 0.0);
        assert!((blend_weight(RagdollPhase::GettingUp, 0.25, 0.5) - 0.5).abs() < 1e-6);
        // Monotone: a get-up never blends back toward the physics pose.
        let mut last = 1.0f32;
        for i in 0..=50 {
            let w = blend_weight(RagdollPhase::GettingUp, i as f64 * 0.01, 0.5);
            assert!(w <= last + 1e-7, "step {i}: {w} rose above {last}");
            last = w;
        }
        // Determinism: the same inputs, twice, bit for bit.
        for i in 0..37 {
            let t = i as f64 * 0.013;
            assert_eq!(
                blend_weight(RagdollPhase::GettingUp, t, 0.5).to_bits(),
                blend_weight(RagdollPhase::GettingUp, t, 0.5).to_bits()
            );
        }
        // A degenerate blend time hands over immediately rather than dividing.
        assert_eq!(blend_weight(RagdollPhase::GettingUp, 0.0, 0.0), 0.0);
    }

    #[test]
    fn the_flail_rate_saturates_at_ten_metres_per_second() {
        assert_eq!(flail_rate(0.0), 0.0);
        assert!((flail_rate(5.0) - 0.5).abs() < 1e-6);
        assert_eq!(flail_rate(10.0), 1.0);
        assert_eq!(flail_rate(1e9), 1.0);
    }
}
