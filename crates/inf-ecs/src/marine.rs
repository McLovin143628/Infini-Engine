//! **Planing hulls and sails** (wave VEH3g) -- the two things VEH2c's
//! [`HullVehicle`](crate::vehicle::HullVehicle) could not do, on the two v28
//! fields VEH3a landed for them.
//!
//! # Planing vs displacement
//!
//! A displacement hull is held up by what it displaces at every speed, so its
//! draught does not change as it goes faster (it squats a little -- not modelled).
//! A PLANING hull climbs onto its own bow wave: above
//! [`planing_speed_mps`](crate::vehicle::VehicleTuning::planing_speed_mps) the
//! water's DYNAMIC pressure on the bottom carries a share of the weight, the hull
//! rises, the wetted area falls and with it the drag -- which is why a jetski
//! goes from wallowing to skimming across a narrow band of speed.
//!
//! The model is exactly that and no more: a vertical hydrodynamic lift, a
//! smoothstep in forward speed from [`PLANING_ONSET_FRAC`] of the planing speed
//! to the planing speed itself, carrying up to [`PLANING_LIFT_FRAC`] of the
//! weight while the hull touches the water. Buoyancy (P20.2's Archimedes) is
//! untouched and carries the rest -- so the hull RISES by itself until the two
//! add up to the weight, and the draught-vs-speed table is read off the hull's
//! world position, never off this function. `planing_speed_mps = 0` is a
//! displacement hull and this function answers zero for it.
//!
//! # The sail
//!
//! A sail is a wing standing on its end in the APPARENT wind -- the P17 weather's
//! own wind vector minus the hull's velocity -- trimmed by the crew to the best
//! angle for the point of sail. Its lift is perpendicular to the apparent wind
//! and its drag along it; what drives the boat is their sum's component along
//! the keel, and what the keel resists is the rest. The trim is a polar, not a
//! sheet: a crew that cannot point closer than [`SAIL_NO_GO_DEG`] to the wind gets
//! a luffing sail (drag only) there, which is why a sailing boat does NOT sail
//! straight into the wind and the gate asserts it.
//!
//! The force acts at the sail's centre of effort, [`SAIL_CE_HEIGHT_FRAC`] of the
//! mast above the deck, so its sideways half is a heeling moment the hull's
//! buoyancy has to right -- the yacht HEELS, and the heel is measured.

use glam::DVec3;

use crate::vehicle::AIR_DENSITY_KG_M3;

/// The share of its planing speed at which a hull begins to climb.
pub const PLANING_ONSET_FRAC: f64 = 0.6;

/// The share of the weight the water's dynamic lift carries at full plane.
///
/// Sixty per cent: a planing hull at speed is still wetted aft, so buoyancy
/// carries the rest -- which, on a box hull, is the draught falling to about
/// two-fifths of its rest value.
pub const PLANING_LIFT_FRAC: f64 = 0.6;

/// The immersion below which the planing lift fades out, `[0, 1]`: a hull
/// skimming on its last twentieth still has its bottom in the water; one launched
/// clear of a wave has none.
pub const PLANING_WET_FLOOR: f64 = 0.05;

/// The hull's heave (and pitch) damping ratio against its own heave frequency
/// `sqrt(g / draught)` -- the water's dashpot beside P20.2's Archimedes spring.
pub const HULL_HEAVE_DAMPING_RATIO: f64 = 0.7;

/// The hull's roll damping ratio -- lighter than heave, because a yacht is
/// SUPPOSED to heel under sail and settle back slowly.
pub const HULL_ROLL_DAMPING_RATIO: f64 = 0.3;

/// How close to the apparent wind a sail can be trimmed to draw, degrees.
pub const SAIL_NO_GO_DEG: f64 = 30.0;

/// The degrees past [`SAIL_NO_GO_DEG`] over which the sail fills.
pub const SAIL_FILL_DEG: f64 = 15.0;

/// A trimmed sail's lift coefficient.
pub const SAIL_CL: f64 = 1.3;

/// A trimmed sail's drag coefficient close-hauled.
pub const SAIL_CD_MIN: f64 = 0.15;

/// A sail's drag coefficient dead downwind -- a flat plate square to the air.
pub const SAIL_CD_RUN: f64 = 1.2;

/// A luffing sail's drag coefficient -- flogging, head to wind.
pub const SAIL_CD_LUFF: f64 = 0.2;

/// Where the sail's force acts: this fraction of the mast (√area, a sail's
/// height to a first approximation) above the chassis origin.
pub const SAIL_CE_HEIGHT_FRAC: f64 = 0.6;

/// The smoothstep `3t² − 2t³`, clamped -- portable, no transcendental.
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// **The planing lift**, newtons, upward: `0` below [`PLANING_ONSET_FRAC`] of the
/// planing speed and for a displacement hull (`planing_speed_mps <= 0`), rising
/// to [`PLANING_LIFT_FRAC`] of the weight at it.
///
/// `wet` is the hull's immersion `[0, 1]`: a hull clear of the water has no
/// bottom pressure, so a boat launched off a wave does not keep flying.
pub fn planing_lift_n(planing_speed_mps: f64, forward_mps: f64, weight_n: f64, wet: f64) -> f64 {
    if !(planing_speed_mps.is_finite() && planing_speed_mps > 0.0) || wet <= 0.0 {
        return 0.0;
    }
    let onset = planing_speed_mps * PLANING_ONSET_FRAC;
    let t = (forward_mps - onset) / (planing_speed_mps - onset).max(1e-9);
    PLANING_LIFT_FRAC
        * weight_n.max(0.0)
        * smoothstep(t)
        * (wet / PLANING_WET_FLOOR).clamp(0.0, 1.0)
}

/// The sail polar: lift and drag coefficients at an apparent-wind angle off the
/// bow, degrees `[0, 180]` (0 = head to wind).
pub fn sail_coefficients(theta_deg: f64) -> (f64, f64) {
    let th = theta_deg.abs().min(180.0);
    let fill = smoothstep((th - SAIL_NO_GO_DEG) / SAIL_FILL_DEG);
    // Past a broad reach the sail is eased until it stalls and pushes as a
    // plate: the lift fades out by a dead run and the drag grows to a plate's.
    let run = ((th - 100.0) / 80.0).clamp(0.0, 1.0);
    let cl = SAIL_CL * fill * (1.0 - run);
    let cd_trim = SAIL_CD_MIN + (SAIL_CD_RUN - SAIL_CD_MIN) * run;
    let cd = SAIL_CD_LUFF + (cd_trim - SAIL_CD_LUFF) * fill;
    (cl, cd)
}

/// **The sail's force**, world newtons, and the apparent-wind angle it met,
/// degrees -- given the hull's basis, velocity, the WIND vector and the sail
/// area. Horizontal only: the sail is trimmed in the horizontal plane and the
/// heel reduces its projected area by `up.y`.
pub fn sail_force(fwd: DVec3, up: DVec3, linvel: DVec3, wind: DVec3, area_m2: f64) -> (DVec3, f64) {
    if !(area_m2.is_finite() && area_m2 > 0.0) {
        return (DVec3::ZERO, 0.0);
    }
    let flat = |v: DVec3| DVec3::new(v.x, 0.0, v.z);
    // The air's velocity over the deck.
    let apparent = flat(wind - linvel);
    let va = apparent.length();
    let bow = flat(fwd).normalize_or_zero();
    if va < 1e-6 || bow == DVec3::ZERO {
        return (DVec3::ZERO, 0.0);
    }
    let toward = apparent / va;
    // The angle the wind comes FROM, off the bow.
    let from = -toward;
    let side = DVec3::Y.cross(bow); // world-right-of-bow in the plane
    let theta = inf_math::portable::patan2_64(from.dot(-side).abs(), from.dot(bow)).to_degrees();
    let (cl, cd) = sail_coefficients(theta);
    let q = 0.5 * AIR_DENSITY_KG_M3 * va * va * area_m2 * up.y.clamp(0.0, 1.0);
    // Lift perpendicular to the apparent wind, on the side that drives forward.
    let mut lift_dir = DVec3::Y.cross(toward);
    if lift_dir.dot(bow) < 0.0 {
        lift_dir = -lift_dir;
    }
    (lift_dir * (q * cl) + toward * (q * cd), theta)
}

/// Where the sail's force acts, metres above the chassis origin along its up.
pub fn sail_ce_height_m(area_m2: f64) -> f64 {
    SAIL_CE_HEIGHT_FRAC * area_m2.max(0.0).sqrt()
}

/// **What the hull did at its last solve** (wave VEH3g) -- the marine
/// instruments' and traces' one source.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarineState {
    /// Forward speed through the water, m/s.
    pub forward_mps: f64,
    /// The hull's immersion, `[0, 1]`, from the chassis half-height.
    pub immersion: f64,
    /// The draught, metres: the water surface minus the hull bottom, off the
    /// chassis position the solve was handed.
    pub draught_m: f64,
    /// The planing lift, newtons.
    pub planing_lift_n: f64,
    /// The sail's force, world newtons (zero for a hull with no sail).
    pub sail_force: DVec3,
    /// The apparent wind's angle off the bow, degrees.
    pub apparent_wind_deg: f64,
    /// The heel, degrees, positive to starboard.
    pub heel_deg: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_displacement_hull_never_planes_and_a_planing_hull_does_above_its_speed() {
        let w = 10_000.0;
        assert_eq!(planing_lift_n(0.0, 50.0, w, 1.0), 0.0);
        assert_eq!(planing_lift_n(16.0, 16.0 * PLANING_ONSET_FRAC, w, 1.0), 0.0);
        let full = planing_lift_n(16.0, 16.0, w, 1.0);
        assert!((full - PLANING_LIFT_FRAC * w).abs() < 1e-9);
        assert_eq!(
            planing_lift_n(16.0, 30.0, w, 0.0),
            0.0,
            "clear of the water"
        );
    }

    #[test]
    fn a_sail_draws_on_a_reach_luffs_head_to_wind_and_pushes_on_a_run() {
        // Wind from the north (blowing toward +Z), a boat pointing north (-Z).
        let wind = DVec3::new(0.0, 0.0, 8.0);
        let head = sail_force(-DVec3::Z, DVec3::Y, DVec3::ZERO, wind, 100.0);
        assert!(head.1 < 1.0, "head to wind: {}", head.1);
        assert!(
            head.0.dot(-DVec3::Z) < 0.0,
            "head to wind the sail pushes the boat BACK"
        );
        // Beam reach: pointing east.
        let beam = sail_force(DVec3::X, DVec3::Y, DVec3::ZERO, wind, 100.0);
        assert!((beam.1 - 90.0).abs() < 1e-9);
        assert!(beam.0.dot(DVec3::X) > 0.0);
        // Run: pointing south, with the wind.
        let run = sail_force(DVec3::Z, DVec3::Y, DVec3::ZERO, wind, 100.0);
        assert!(run.0.dot(DVec3::Z) > 0.0);
        assert!(
            beam.0.dot(DVec3::X) > run.0.dot(DVec3::Z) * 0.9,
            "a reach is the fast point"
        );
    }
}
