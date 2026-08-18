//! **Foot IK and foot locking** (P29.4, clause 5) — the pure half.
//!
//! Two mechanisms that look alike and are not:
//!
//! * **Foot IK** puts a foot on the *ground it is standing on*: a downward probe
//!   under the foot, a vertical offset so the sole rests on the surface, and a
//!   pitch/roll so it lies flat on a slope. The probe needs a world; the
//!   arithmetic that turns a hit into an offset does not, and is
//!   [`ground_offset`].
//! * **Foot lock** pins a *planted* foot to the place in the world where it was
//!   planted, and counter-moves it as the body travels, so a foot that the clip
//!   says is on the ground does not skate across it. That is [`FootLock`], and it
//!   is what the wave's gate measures — **foot slide, in metres**.
//!
//! # The constants are ALS's, converted once
//!
//! Every centimetre in the donor becomes a metre here, at this boundary and
//! nowhere else (port map §2.12): the trace envelope is `+50 / −45` cm →
//! [`TRACE_ABOVE_M`] / [`TRACE_BELOW_M`], and the foot's own thickness is
//! `13.5` cm → [`FOOT_HEIGHT_M`]. The interpolation speeds are unit-free rates
//! and carry over unchanged.
//!
//! # The one rule that is not obvious
//!
//! ALS's `SetFootLocking` updates the lock alpha **only if the new value is at
//! least 0.99, or is lower than the current one** — "the foot can only blend out
//! of the locked position or lock to a new position, and never blend in"
//! (`:376–377`). [`FootLock::update`] implements exactly that, and the arm
//! `a_lock_never_blends_in` is what stops a well-meaning smoothing pass from
//! removing it: blending *in* means the foot is dragged toward its lock point
//! while it is still moving, which is a skate with extra steps.
//!
//! What is **not** ported is the donor's `1 / AnimUpdateRateParams->UpdateRate`
//! scaling of the lock curve (`:372`). That is a frame-rate dependency inside a
//! value that decides where a foot is, and this engine's pose rides
//! `state_bytes`.
//!
//! # Portability
//!
//! `sqrt`, adds and multiplies, plus `inf_math::patan2_64` for the two surface
//! angles — the same helper `inf_ecs::movement` uses for the movement quadrant,
//! and for the same reason: this result reaches the pose.

use glam::Vec3;

/// How far **above** the foot the ground probe starts, metres (ALS
/// `IK_TraceDistanceAboveFoot = 50` cm).
pub const TRACE_ABOVE_M: f64 = 0.50;
/// How far **below** the foot the ground probe ends, metres (ALS
/// `IK_TraceDistanceBelowFoot = 45` cm).
pub const TRACE_BELOW_M: f64 = 0.45;
/// The foot's own thickness — the distance from the ankle's floor point to the
/// sole, metres (ALS `FootHeight = 13.5` cm).
pub const FOOT_HEIGHT_M: f64 = 0.135;

/// Interp speed pulling the foot offset **down** onto a surface (ALS 30).
pub const OFFSET_DOWN_INTERP: f64 = 30.0;
/// Interp speed pulling the foot offset **up** off a surface (ALS 15).
pub const OFFSET_UP_INTERP: f64 = 15.0;
/// Interp speed for the foot's pitch/roll onto the surface normal (ALS 30).
pub const ROTATION_INTERP: f64 = 30.0;
/// Interp speed for the pelvis offset going **up** (ALS 10).
pub const PELVIS_UP_INTERP: f64 = 10.0;
/// Interp speed for the pelvis offset going **down** (ALS 15).
pub const PELVIS_DOWN_INTERP: f64 = 15.0;
/// Interp speed everything resets at when IK is disabled (ALS 15).
pub const RESET_INTERP: f64 = 15.0;

/// The curve value at or above which a lock **engages** (ALS 0.99).
pub const LOCK_ENGAGE: f32 = 0.99;

/// One frame's first-order interpolation toward `target` at `speed` — ALS's
/// `FInterpTo`, which is `clamp(speed × dt, 0, 1)` and not an exponential (that
/// would need a transcendental, and this value reaches the pose).
pub fn interp_to(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    if !dt.is_finite() || dt <= 0.0 || !target.is_finite() {
        return current;
    }
    let a = (speed.max(0.0) * dt).clamp(0.0, 1.0);
    current + (target - current) * a
}

/// [`interp_to`] on a vector, componentwise at one speed.
pub fn interp_to_vec(current: Vec3, target: Vec3, speed: f64, dt: f64) -> Vec3 {
    Vec3::new(
        interp_to(current.x as f64, target.x as f64, speed, dt) as f32,
        interp_to(current.y as f64, target.y as f64, speed, dt) as f32,
        interp_to(current.z as f64, target.z as f64, speed, dt) as f32,
    )
}

/// What a ground hit under one foot means for that foot's IK.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroundOffset {
    /// Where the foot must move to, relative to where the pose put it, metres.
    pub offset: Vec3,
    /// Pitch about the character's right axis, degrees, so the sole lies on the
    /// surface.
    pub pitch_deg: f64,
    /// Roll about the character's forward axis, degrees.
    pub roll_deg: f64,
}

/// Turn one downward probe hit into a [`GroundOffset`] — ALS's `SetFootOffsets`
/// arithmetic (`:464–535`), with UE's Z-up swapped for this engine's Y-up.
///
/// * `impact` is the world point the probe hit and `normal` the surface normal
///   there.
/// * `foot_floor` is where the *pose* put the foot's floor point (the ankle
///   socket with its Y replaced by the root joint's Y — the character's own
///   ground plane).
///
/// The offset lifts the foot by its own thickness along the surface normal, so a
/// foot on a slope rests on the slope rather than sinking a sole's depth into it.
/// A non-finite normal answers the zero offset, which is "leave the pose alone".
pub fn ground_offset(impact: Vec3, normal: Vec3, foot_floor: Vec3) -> GroundOffset {
    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO || !impact.is_finite() || !foot_floor.is_finite() {
        return GroundOffset::default();
    }
    let h = FOOT_HEIGHT_M as f32;
    let target = impact + n * h;
    let from = foot_floor + Vec3::new(0.0, h, 0.0);
    // UE: Pitch = -atan2(N.X, N.Z), Roll = atan2(N.Y, N.Z) about a Z-up normal.
    // Here the up axis is +Y, so the two planar components are X (roll) and Z
    // (pitch), each against the vertical.
    let pitch = -inf_math::patan2_64(n.z as f64, n.y.max(1e-6) as f64).to_degrees();
    let roll = inf_math::patan2_64(n.x as f64, n.y.max(1e-6) as f64).to_degrees();
    GroundOffset {
        offset: target - from,
        pitch_deg: pitch,
        roll_deg: roll,
    }
}

/// **A planted foot's world-space lock.**
///
/// `alpha` is how much of the lock is applied, `position` the world point the
/// foot is pinned to and `yaw_deg` the facing it was pinned at. The whole struct
/// is `Copy` scalars: it lives on a per-entity runtime that must not allocate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FootLock {
    /// How much of the lock is applied, `[0, 1]`.
    pub alpha: f32,
    /// The world point the foot is pinned to, metres.
    pub position: Vec3,
    /// The facing the foot was pinned at, degrees.
    pub yaw_deg: f64,
    /// Whether a lock is currently held at all.
    pub locked: bool,
}

impl FootLock {
    /// Advance the lock for one fixed step.
    ///
    /// * `enable` is the clip's `Enable_FootIK_L/R` curve — at or below zero the
    ///   whole mechanism is off and the lock is released.
    /// * `lock_curve` is the clip's `FootLock_L/R` curve: the animator's
    ///   statement that this foot is planted right now.
    /// * `turning` is `true` while the body is turning under the foot (ALS gates
    ///   on `|RotationAmount| <= 0.001`); a turn breaks the lock, because a
    ///   pinned foot with a rotating hip is a foot pointing the wrong way.
    /// * `foot_world` / `foot_yaw_deg` are where the pose put the foot this step,
    ///   which is what a *new* lock snaps to.
    ///
    /// The alpha only ever moves toward the engaged value or downward — see the
    /// module docs.
    pub fn update(
        &mut self,
        enable: f32,
        lock_curve: f32,
        turning: bool,
        foot_world: Vec3,
        foot_yaw_deg: f64,
    ) {
        if !(enable > 0.0) || turning {
            self.alpha = 0.0;
            self.locked = false;
            return;
        }
        let want = if lock_curve.is_finite() {
            lock_curve.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // THE rule (ALS `:376–377`): engage at the threshold, otherwise only ever
        // release. A value between the two that is *higher* than the current
        // alpha is ignored — that is what "never blend in" means.
        if want >= LOCK_ENGAGE {
            self.alpha = want;
        } else if want < self.alpha {
            self.alpha = want;
        }
        if self.alpha >= LOCK_ENGAGE && !self.locked {
            // A fresh lock snaps to where the foot is NOW.
            self.position = foot_world;
            self.yaw_deg = foot_yaw_deg;
            self.locked = true;
        }
        if self.alpha <= 0.0 {
            self.locked = false;
        }
    }

    /// Release the lock outright (leaving the ground, ragdolling, dying).
    pub fn release(&mut self) {
        self.alpha = 0.0;
        self.locked = false;
    }

    /// Where the foot should be drawn this step, given where the pose put it.
    ///
    /// The lock is applied by `alpha`, so a releasing lock slides the foot back
    /// onto the animation rather than popping it.
    pub fn resolve(&self, posed_world: Vec3) -> Vec3 {
        if !self.locked || !(self.alpha > 0.0) {
            return posed_world;
        }
        let a = self.alpha.clamp(0.0, 1.0);
        posed_world + (self.position - posed_world) * a
    }

    /// **The gate's measurement**: how far the foot has slid from where it was
    /// planted, in **metres**, on the ground plane.
    ///
    /// Ground-plane rather than 3D deliberately: a foot rising off a step has not
    /// slid, and a gate that counted it would be measuring the clip's stride
    /// instead of the defect. `0.0` when nothing is locked, which is the honest
    /// answer — an unplanted foot cannot slide.
    pub fn slide_m(&self, posed_world: Vec3) -> f64 {
        if !self.locked || !(self.alpha > 0.0) {
            return 0.0;
        }
        let d = self.resolve(posed_world) - self.position;
        ((d.x as f64) * (d.x as f64) + (d.z as f64) * (d.z as f64)).sqrt()
    }
}

/// The pelvis offset owed to a pair of foot offsets — ALS's `SetPelvisIKOffset`
/// (`:427–449`).
///
/// The target is the **lower** of the two feet: when one foot is on a step below
/// the other, the whole body drops to reach it rather than the low leg
/// straightening past its limit. `alpha` is the mean of the two `Enable_FootIK`
/// curves, so a character with IK on one leg only gets half the drop.
pub fn pelvis_offset(enable_l: f32, enable_r: f32, offset_l: Vec3, offset_r: Vec3) -> Vec3 {
    let a = ((enable_l.clamp(0.0, 1.0) + enable_r.clamp(0.0, 1.0)) * 0.5).clamp(0.0, 1.0);
    if !(a > 0.0) {
        return Vec3::ZERO;
    }
    let lower = if offset_l.y <= offset_r.y {
        offset_l
    } else {
        offset_r
    };
    // Only the vertical: a pelvis that chased a foot sideways would lean the
    // character off its own capsule.
    Vec3::new(0.0, lower.y.min(0.0) * a, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_als_envelope_is_in_metres() {
        // 50 / 45 / 13.5 cm, converted once at this boundary and nowhere else.
        assert_eq!(TRACE_ABOVE_M, 0.50);
        assert_eq!(TRACE_BELOW_M, 0.45);
        assert_eq!(FOOT_HEIGHT_M, 0.135);
    }

    #[test]
    fn a_flat_surface_lifts_the_foot_by_its_own_thickness_and_does_not_tilt_it() {
        let g = ground_offset(Vec3::new(0.0, 0.2, 0.0), Vec3::Y, Vec3::new(0.0, 0.0, 0.0));
        assert!((g.offset.y - 0.2).abs() < 1e-6, "{g:?}");
        assert!(g.pitch_deg.abs() < 1e-9 && g.roll_deg.abs() < 1e-9, "{g:?}");
        // A 30-degree slope about the character's right axis reads as pitch.
        let n = Vec3::new(0.0, 30f32.to_radians().cos(), 30f32.to_radians().sin());
        let g = ground_offset(Vec3::ZERO, n, Vec3::ZERO);
        assert!((g.pitch_deg + 30.0).abs() < 0.5, "{g:?}");
        // A degenerate normal is an answer, not a NaN in the pose.
        assert_eq!(
            ground_offset(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
            GroundOffset::default()
        );
    }

    /// **THE rule.** A lock engages at the threshold and afterwards may only
    /// release. A mid value that is *higher* than the current alpha changes
    /// nothing — which is what stops a foot being dragged toward a lock point it
    /// has not reached.
    #[test]
    fn a_lock_never_blends_in() {
        let mut l = FootLock::default();
        // Below the threshold from cold: nothing engages.
        l.update(1.0, 0.5, false, Vec3::new(1.0, 0.0, 0.0), 0.0);
        assert!(!l.locked, "{l:?}");
        assert_eq!(l.alpha, 0.0, "a rising mid value must not blend in");
        // At the threshold: engaged, snapped to where the foot is now.
        l.update(1.0, 1.0, false, Vec3::new(1.0, 0.0, 2.0), 90.0);
        assert!(l.locked);
        assert_eq!(l.position, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(l.yaw_deg, 90.0);
        // Releasing: a lower value DOES take, and the lock point does not move.
        l.update(1.0, 0.4, false, Vec3::new(5.0, 0.0, 5.0), 0.0);
        assert!((l.alpha - 0.4).abs() < 1e-6);
        assert_eq!(
            l.position,
            Vec3::new(1.0, 0.0, 2.0),
            "a releasing lock keeps its point"
        );
        // …and a value back UP is ignored while the lock is still held.
        l.update(1.0, 0.9, false, Vec3::new(5.0, 0.0, 5.0), 0.0);
        assert!((l.alpha - 0.4).abs() < 1e-6, "blended in: {l:?}");
        // Turning breaks it outright.
        l.update(1.0, 1.0, true, Vec3::ZERO, 0.0);
        assert!(!l.locked && l.alpha == 0.0, "{l:?}");
        // So does the IK gate curve going to zero.
        l.update(1.0, 1.0, false, Vec3::ZERO, 0.0);
        assert!(l.locked);
        l.update(0.0, 1.0, false, Vec3::ZERO, 0.0);
        assert!(!l.locked);
    }

    /// The measurement the wave's gate is built on: a locked foot under a moving
    /// body does not slide, and an unlocked one does.
    #[test]
    fn a_locked_foot_does_not_slide_and_an_unlocked_one_does() {
        let mut l = FootLock::default();
        l.update(1.0, 1.0, false, Vec3::new(0.0, 0.0, 0.0), 0.0);
        // The pose walks the foot 12 cm forward; fully locked, it stays put.
        let posed = Vec3::new(0.0, 0.0, 0.12);
        assert!(l.slide_m(posed) < 1e-9, "{}", l.slide_m(posed));
        assert_eq!(l.resolve(posed), Vec3::ZERO);
        // Half-released, half the slide comes back — which is the blend-out.
        l.update(1.0, 0.5, false, posed, 0.0);
        assert!(
            (l.slide_m(posed) - 0.06).abs() < 1e-6,
            "{}",
            l.slide_m(posed)
        );
        // Released entirely: the lock has no opinion, and reports none.
        l.release();
        assert_eq!(l.slide_m(posed), 0.0);
        assert_eq!(l.resolve(posed), posed);
        // The vertical is not slide: a foot rising off a step reads zero.
        let mut up = FootLock::default();
        up.update(1.0, 1.0, false, Vec3::ZERO, 0.0);
        assert!(up.slide_m(Vec3::new(0.0, 0.3, 0.0)) < 1e-9);
    }

    #[test]
    fn the_pelvis_follows_the_lower_foot_and_only_downward() {
        let p = pelvis_offset(
            1.0,
            1.0,
            Vec3::new(0.0, -0.2, 0.0),
            Vec3::new(0.0, -0.05, 0.0),
        );
        assert!((p.y + 0.2).abs() < 1e-6, "{p:?}");
        assert_eq!(p.x, 0.0, "the pelvis does not chase a foot sideways");
        // Half the IK, half the drop — the alpha is the MEAN of the two gates,
        // so a character with IK on one leg only drops half as far.
        let half = pelvis_offset(1.0, 0.0, Vec3::new(0.0, -0.2, 0.0), Vec3::ZERO);
        assert!((half.y + 0.1).abs() < 1e-6, "{half:?}");
        let half = pelvis_offset(
            1.0,
            0.0,
            Vec3::new(0.0, -0.2, 0.0),
            Vec3::new(0.0, -0.4, 0.0),
        );
        assert!(
            (half.y + 0.2).abs() < 1e-6,
            "the LOWER foot, at half weight: {half:?}"
        );
        // Feet ABOVE the pose do not lift the pelvis (that is a jump, not IK).
        let up = pelvis_offset(1.0, 1.0, Vec3::new(0.0, 0.3, 0.0), Vec3::new(0.0, 0.2, 0.0));
        assert_eq!(up.y, 0.0);
        // No IK at all is no offset at all.
        assert_eq!(
            pelvis_offset(0.0, 0.0, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
            Vec3::ZERO
        );
    }

    #[test]
    fn interp_to_is_the_first_order_chase_and_is_inert_on_a_dead_step() {
        assert!((interp_to(0.0, 1.0, 30.0, 1.0 / 60.0) - 0.5).abs() < 1e-12);
        assert_eq!(interp_to(0.25, 1.0, 30.0, 0.0), 0.25);
        assert_eq!(interp_to(0.25, f64::NAN, 30.0, 0.1), 0.25);
        // A speed×dt past 1 arrives rather than overshooting.
        assert_eq!(interp_to(0.0, 1.0, 1000.0, 0.5), 1.0);
    }
}
