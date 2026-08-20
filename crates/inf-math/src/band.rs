//! **Distance bands** (I3): the three-way tier rule and the oriented-box
//! distance it reads, in the one crate every consumer already depends on.
//!
//! Two very different subsystems band content by distance and they must not
//! disagree about what "far" means:
//!
//! * the **physics bridge** bands a city's static colliders off the simulation's
//!   own streaming sources, so a building nobody can reach costs no solver time;
//! * each **host's projection** bands the same buildings off the camera, so a
//!   building nobody can see costs one instance instead of a thousand.
//!
//! The two take different *inputs* by law — visibility filters what is drawn,
//! never what is simulated — but they share this arithmetic, because the day
//! they stop sharing it is the day a building draws as a solid block you can
//! still walk into, or as parts you fall through.
//!
//! Nothing here is a policy: the radii are the caller's, and the defaults live
//! beside the consumers that measured them.

use glam::{DQuat, DVec3};

/// What a piece of content is, at a given distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Full detail: every part drawn, every solid simulated.
    Near,
    /// A stand-in: one instance, one collider — the content's own shell.
    Far,
    /// Out of range: nothing drawn, nothing simulated.
    Out,
}

impl Tier {
    /// `true` when this tier carries the content's own parts.
    #[inline]
    pub fn is_near(self) -> bool {
        matches!(self, Self::Near)
    }

    /// `true` when this tier contributes anything at all.
    #[inline]
    pub fn is_resident(self) -> bool {
        !matches!(self, Self::Out)
    }
}

/// The tier something at `distance_m` belongs to.
///
/// **Closed on the near boundary** (`distance <= near_m` is [`Tier::Near`]), so
/// content exactly on the line takes the *richer* tier: a boundary case that
/// resolves towards "enterable" cannot strand a character inside a stand-in.
///
/// A non-finite distance is [`Tier::Out`]. Every comparison against `NaN` is
/// false, so a naive `if d < near {Near} else if d < far {Far} else {Out}` would
/// fall through to `Out` by accident and a `<=`-chained variant could fall
/// through to `Near`; stating it explicitly makes the answer a decision rather
/// than a consequence of operator order (the NaN-at-doors law).
#[inline]
pub fn tier_at(distance_m: f64, near_m: f64, far_m: f64) -> Tier {
    if !distance_m.is_finite() {
        return Tier::Out;
    }
    if distance_m <= near_m {
        Tier::Near
    } else if distance_m <= far_m {
        Tier::Far
    } else {
        Tier::Out
    }
}

/// Distance from `p` to an oriented box, measured on the **XZ plane only**.
///
/// Zero inside the box's footprint. Y is deliberately ignored: a character
/// twelve metres below a tower's ground slab is *at* the tower, and a band that
/// measured the full 3-D distance would drop a skyscraper's colliders the moment
/// somebody walked into its basement — or, on the render side, swap a tower for
/// its silhouette when the camera looks up at it from the street.
///
/// Trig-free (the P14 portability law): one quaternion inverse-rotation, two
/// clamped subtractions and a `sqrt`.
#[inline]
pub fn distance_xz_to_box(p: DVec3, center: DVec3, half: DVec3, rotation: DQuat) -> f64 {
    let rel = rotation.inverse() * (p - center);
    let dx = (rel.x.abs() - half.x).max(0.0);
    let dz = (rel.z.abs() - half.z).max(0.0);
    (dx * dx + dz * dz).sqrt()
}

/// The **nearest** of a set of anchors to an oriented box, on XZ.
///
/// `f64::INFINITY` for an empty anchor set — which [`tier_at`] reads as
/// [`Tier::Out`]. Callers that mean "no anchors, therefore no banding" must say
/// so themselves rather than relying on this to mean it; `inf_ecs::band` does.
pub fn nearest_xz_to_box(anchors: &[DVec3], center: DVec3, half: DVec3, rotation: DQuat) -> f64 {
    let mut best = f64::INFINITY;
    for a in anchors {
        let d = distance_xz_to_box(*a, center, half, rotation);
        if d < best {
            best = d;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn the_tier_rule_is_three_way_and_nan_is_out() {
        assert_eq!(tier_at(0.0, 10.0, 100.0), Tier::Near);
        assert_eq!(tier_at(10.0, 10.0, 100.0), Tier::Near);
        assert_eq!(tier_at(10.000_1, 10.0, 100.0), Tier::Far);
        assert_eq!(tier_at(100.0, 10.0, 100.0), Tier::Far);
        assert_eq!(tier_at(100.1, 10.0, 100.0), Tier::Out);
        assert_eq!(tier_at(f64::NAN, 10.0, 100.0), Tier::Out);
        assert_eq!(tier_at(f64::INFINITY, 10.0, 100.0), Tier::Out);
        // A degenerate pair still answers: near >= far means nothing is Far.
        assert_eq!(tier_at(50.0, 100.0, 10.0), Tier::Near);
        assert!(Tier::Near.is_near() && Tier::Near.is_resident());
        assert!(!Tier::Far.is_near() && Tier::Far.is_resident());
        assert!(!Tier::Out.is_resident());
    }

    #[test]
    fn the_distance_is_to_the_box_on_the_xz_plane_and_ignores_height() {
        let (c, h) = (DVec3::ZERO, DVec3::new(10.0, 20.0, 5.0));
        // Inside the footprint, at any height — the basement case.
        assert_eq!(
            distance_xz_to_box(DVec3::new(0.0, 900.0, 0.0), c, h, DQuat::IDENTITY),
            0.0
        );
        assert_eq!(
            distance_xz_to_box(DVec3::new(0.0, -900.0, 0.0), c, h, DQuat::IDENTITY),
            0.0
        );
        // Straight out along X, and diagonally off a corner (3-4-5).
        let d = distance_xz_to_box(DVec3::new(13.0, 0.0, 0.0), c, h, DQuat::IDENTITY);
        assert!((d - 3.0).abs() < 1e-12, "{d}");
        let d = distance_xz_to_box(DVec3::new(13.0, 0.0, 9.0), c, h, DQuat::IDENTITY);
        assert!((d - 5.0).abs() < 1e-12, "{d}");

        // Rotated: the point is measured against the box's OWN axes.
        let yaw = DQuat::from_xyzw(0.0, 0.6, 0.0, 0.8).normalize();
        let turned = yaw * DVec3::new(13.0, 0.0, 9.0);
        let d = distance_xz_to_box(turned, c, h, yaw);
        assert!((d - 5.0).abs() < 1e-9, "{d}");

        // THE CONTROL, and it is the case that discriminates: a point INSIDE
        // the turned box reads zero, and reads 2.52 m outside if the rotation is
        // dropped. (An outside point is a weak control — the same 13,9 corner is
        // only 0.46 m off when the rotation is ignored, which a tolerance would
        // swallow.)
        let inside = yaw * DVec3::new(9.0, 0.0, 4.0);
        assert_eq!(distance_xz_to_box(inside, c, h, yaw), 0.0);
        let ignored = distance_xz_to_box(inside, c, h, DQuat::IDENTITY);
        assert!(
            ignored > 1.0,
            "the control does not discriminate: {ignored}"
        );
    }

    #[test]
    fn the_nearest_anchor_wins_and_an_empty_set_is_infinite() {
        let (c, h) = (DVec3::ZERO, DVec3::splat(1.0));
        assert_eq!(nearest_xz_to_box(&[], c, h, DQuat::IDENTITY), f64::INFINITY);
        assert_eq!(
            tier_at(nearest_xz_to_box(&[], c, h, DQuat::IDENTITY), 10.0, 100.0),
            Tier::Out
        );
        let anchors = [
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::new(90.0, 0.0, 0.0),
        ];
        let d = nearest_xz_to_box(&anchors, c, h, DQuat::IDENTITY);
        assert!((d - 3.0).abs() < 1e-12, "{d}");
        // Order does not decide it.
        let mut rev = anchors;
        rev.reverse();
        assert_eq!(d, nearest_xz_to_box(&rev, c, h, DQuat::IDENTITY));
    }
}
