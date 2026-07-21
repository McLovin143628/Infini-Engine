//! World-space **ray → heightfield** intersection (P10.2b).
//!
//! Sculpting needs the world point under the cursor: the editor unprojects a
//! screen pixel to a world ray and asks the terrain where that ray first meets
//! the surface. There is no closed form for an arbitrary heightfield, so this is
//! a **fixed-step ray march** that brackets the first sign change of the signed
//! height residual `f(t) = ray_y(t) − terrain_height(x, z)` and then **bisects**
//! that bracket to a tight world-space crossing.
//!
//! Design points the march must survive:
//! * **Above or below starts.** A camera above the terrain starts with `f > 0`
//!   and crosses to `f < 0` (entering the ground); a ray beginning under the
//!   surface starts with `f < 0`. Either way the *first* sign change is the
//!   surface, so we bracket on any flip, not just `+→−`.
//! * **Unauthored gaps.** [`TerrainData::height_at`] returns `None` off the
//!   authored tiles. A sample with no height can't be compared, so it resets the
//!   bracket — the march simply skips the hole and resumes when the ray re-enters
//!   authored space (a ray that only ever grazes holes misses).
//! * **Near-horizontal rays.** The step is taken *along the ray* at ~half a
//!   sample spacing, so even a shallow ray advances in fine height increments and
//!   never steps over a thin ridge.

use glam::{DVec2, DVec3};

use crate::data::TerrainData;

/// Where a ray meets the terrain surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainHit {
    /// World-space intersection point (on the surface).
    pub point: DVec3,
    /// Ray parameter of the hit: `point == origin + dir · t` (world units when
    /// `dir` is unit length).
    pub t: f64,
}

/// Bisection iterations refining a bracketed crossing — each roughly halves the
/// interval, so ~40 pins it to far below floating-point-meaningful precision.
const REFINE_ITERS: u32 = 40;
/// Stop refining once the bracket is this tight (world units along the ray).
const REFINE_EPSILON: f64 = 1e-4;

/// Cast the ray `origin + dir · t` (`t ∈ [0, max_t]`) against `data` and return
/// the first surface hit, or `None` if the ray never crosses authored terrain
/// inside the range.
///
/// `dir` need not be normalized, but `TerrainHit::t` and `max_t` are measured in
/// `dir` lengths; pass a unit `dir` for `t` to read as world metres. The march
/// steps at ~half [`TerrainData::meters_per_sample`] along the ray.
pub fn raycast_terrain(
    data: &TerrainData,
    origin: DVec3,
    dir: DVec3,
    max_t: f64,
) -> Option<TerrainHit> {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO || max_t <= 0.0 {
        return None;
    }
    // Empty terrain is one big hole — nothing to hit.
    if data.is_empty() {
        return None;
    }

    // Step along the ray at half a sample spacing so a thin feature is never
    // stepped over. The horizontal component of a step is `step · dir_xz`, still
    // sub-sample for any non-degenerate ray.
    let step = (0.5 * data.meters_per_sample()).max(1e-3);

    // Previous *authored* sample: `(t, residual)`. Reset to `None` across holes.
    let mut prev: Option<(f64, f64)> = None;
    let mut t = 0.0;
    // One extra iteration so the final `t == max_t` endpoint is evaluated.
    while t <= max_t + step {
        let tc = t.min(max_t);
        let p = origin + dir * tc;
        match residual(data, p) {
            Some(fc) => {
                if fc == 0.0 {
                    // Exactly on the surface — a hit with no bracket to refine.
                    return Some(TerrainHit { point: p, t: tc });
                }
                if let Some((tp, fp)) = prev {
                    if (fp > 0.0) != (fc > 0.0) {
                        // Sign flip between two authored samples brackets the
                        // surface; bisect for the precise crossing.
                        let th = refine(data, origin, dir, tp, fp, tc, fc);
                        return Some(TerrainHit {
                            point: origin + dir * th,
                            t: th,
                        });
                    }
                }
                prev = Some((tc, fc));
            }
            // A hole breaks the bracket: we can't compare across unauthored space.
            None => prev = None,
        }
        if tc >= max_t {
            break;
        }
        t += step;
    }
    None
}

/// Signed height residual `ray_y − terrain_height` at world point `p`, or `None`
/// where the terrain is unauthored (a hole). Positive above the surface.
#[inline]
fn residual(data: &TerrainData, p: DVec3) -> Option<f64> {
    data.height_at(DVec2::new(p.x, p.z)).map(|h| p.y - h)
}

/// Bisect a bracket `[ta, tb]` whose residuals `fa`, `fb` have opposite signs to
/// the surface crossing. Returns the crossing's ray parameter. If a refinement
/// sample lands on a hole (vanishingly rare inside a one-step bracket), falls
/// back to the secant estimate of the current bracket.
fn refine(
    data: &TerrainData,
    origin: DVec3,
    dir: DVec3,
    mut ta: f64,
    mut fa: f64,
    mut tb: f64,
    mut fb: f64,
) -> f64 {
    for _ in 0..REFINE_ITERS {
        if (tb - ta).abs() < REFINE_EPSILON {
            break;
        }
        let tm = 0.5 * (ta + tb);
        let Some(fm) = residual(data, origin + dir * tm) else {
            // Gap inside the bracket: linear (secant) estimate of the crossing.
            return ta + (tb - ta) * fa / (fa - fb);
        };
        if fm == 0.0 {
            return tm;
        }
        if (fm > 0.0) == (fa > 0.0) {
            ta = tm;
            fa = fm;
        } else {
            tb = tm;
            fb = fm;
        }
    }
    0.5 * (ta + tb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-tile terrain shaped like a smooth hill peaking at the origin:
    /// `h(x, z) = peak · exp(-(x² + z²)/σ²)`, centered on the tile.
    fn hill(peak: f64, sigma: f64) -> (TerrainData, DVec2) {
        let res = 65;
        let mps = 1.0;
        let mut data = TerrainData::new(res, mps);
        let span = data.tile_span();
        let center = DVec2::new(span * 0.5, span * 0.5);
        data.author_tile((0, 0), |x, z| {
            let dx = x - center.x;
            let dz = z - center.y;
            peak * (-(dx * dx + dz * dz) / (sigma * sigma)).exp()
        });
        (data, center)
    }

    #[test]
    fn straight_down_hits_the_peak() {
        let (data, center) = hill(20.0, 12.0);
        let origin = DVec3::new(center.x, 100.0, center.y);
        let hit = raycast_terrain(&data, origin, DVec3::NEG_Y, 500.0).expect("ray must hit");
        // The peak is ~20 m; the hit sits on the surface at the tile centre.
        assert!((hit.point.x - center.x).abs() < 1e-3);
        assert!((hit.point.z - center.y).abs() < 1e-3);
        let surf = data
            .height_at(DVec2::new(hit.point.x, hit.point.z))
            .unwrap();
        assert!(
            (hit.point.y - surf).abs() < 1e-2,
            "hit off surface: {hit:?}"
        );
        assert!((hit.point.y - 20.0).abs() < 0.2, "peak height {hit:?}");
    }

    #[test]
    fn angled_ray_lands_on_the_surface() {
        let (data, center) = hill(15.0, 16.0);
        // Come in from the +X/+Y side at 45° toward the hill.
        let origin = DVec3::new(center.x + 60.0, 60.0, center.y);
        let dir = DVec3::new(-1.0, -1.0, 0.0);
        let hit = raycast_terrain(&data, origin, dir, 500.0).expect("angled ray must hit");
        // Whatever it hits, the point lies on the terrain surface.
        let surf = data
            .height_at(DVec2::new(hit.point.x, hit.point.z))
            .unwrap();
        assert!(
            (hit.point.y - surf).abs() < 1e-2,
            "angled hit off surface: {hit:?} surf {surf}"
        );
        // And it is the FIRST crossing: the ray is above terrain before it.
        let before = origin + dir.normalize() * (hit.t - 1.0);
        let surf_before = data
            .height_at(DVec2::new(before.x, before.z))
            .unwrap_or(f64::NEG_INFINITY);
        assert!(before.y > surf_before, "not the first crossing: {hit:?}");
    }

    #[test]
    fn misses_off_terrain() {
        let (data, center) = hill(10.0, 10.0);
        // A vertical ray far outside the single authored tile hits nothing.
        let origin = DVec3::new(center.x + 10_000.0, 100.0, center.y);
        assert!(raycast_terrain(&data, origin, DVec3::NEG_Y, 500.0).is_none());
        // A ray pointing UP and away from an above start never re-enters terrain.
        let up = DVec3::new(center.x, 50.0, center.y);
        assert!(raycast_terrain(&data, up, DVec3::Y, 500.0).is_none());
    }

    #[test]
    fn empty_terrain_never_hits() {
        let data = TerrainData::new(16, 1.0);
        assert!(raycast_terrain(&data, DVec3::new(0.0, 10.0, 0.0), DVec3::NEG_Y, 100.0).is_none());
    }

    #[test]
    fn refine_matches_analytic_plane() {
        // A tilted plane h(x, z) = 0.3x − 0.15z + 2 has an exact ray crossing;
        // the marched+bisected hit must match it to well under a sample.
        let (a, b, c) = (0.3, -0.15, 2.0);
        let res = 33;
        let mut data = TerrainData::new(res, 1.0);
        data.author_tile((0, 0), |x, z| a * x + b * z + c);
        let span = data.tile_span();

        let origin = DVec3::new(span * 0.3, 40.0, span * 0.6);
        let dir = DVec3::new(0.2, -1.0, 0.1).normalize();
        let hit = raycast_terrain(&data, origin, dir, 200.0).expect("plane ray must hit");

        // Solve o.y + t·d.y = a(o.x + t·d.x) + b(o.z + t·d.z) + c for t.
        let num = a * origin.x + b * origin.z + c - origin.y;
        let den = dir.y - a * dir.x - b * dir.z;
        let t_analytic = num / den;
        assert!(
            (hit.t - t_analytic).abs() < 1e-2,
            "t {} vs analytic {t_analytic}",
            hit.t
        );
        let expect = origin + dir * t_analytic;
        assert!((hit.point - expect).length() < 1e-2, "point {hit:?}");
    }

    #[test]
    fn near_horizontal_ray_finds_a_ridge() {
        // A shallow ray skimming toward the hill must still catch its slope
        // rather than stepping over it.
        let (data, center) = hill(25.0, 10.0);
        let origin = DVec3::new(center.x - 40.0, 6.0, center.y);
        let dir = DVec3::new(1.0, -0.05, 0.0); // almost flat, drifting down
        let hit = raycast_terrain(&data, origin, dir, 400.0).expect("shallow ray must hit");
        let surf = data
            .height_at(DVec2::new(hit.point.x, hit.point.z))
            .unwrap();
        assert!((hit.point.y - surf).abs() < 1e-2, "shallow hit off surface");
    }
}
