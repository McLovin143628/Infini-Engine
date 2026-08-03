//! **Authoring-time hydrology** (P20.4): the questions a placement tool asks a
//! terrain that the cook cannot.
//!
//! # Why these live here and not in the cook
//!
//! P20.1's downhill advisory reads the river's *spline* and queries no terrain at
//! all, deliberately: the cook validates a `.inf_terrain` structurally and never
//! pages a tile in, so anything that needs a ground height is unanswerable there.
//! The editor has the opposite problem and the opposite luxury — the terrain is
//! resident, in `Terrain::data`, while the author is standing on it.
//!
//! So the split is by *what is knowable*, not by taste:
//!
//! * the **surface** climbs in the direction it flows — spline only, so the cook
//!   checks it ([`crate::river::uphill_spans`] over
//!   [`RiverPath::surface_profile`]);
//! * the **authored bed** climbs — spline + profile, still no terrain, so the
//!   cook checks that too ([`RiverPath::bed_profile_from_depth`], P20.4);
//! * the river is **buried in** or **perched over** the ground — needs the
//!   heightfield, so it lives here and the tools report it
//!   ([`bed_conflicts`]);
//! * where a lake's waterline lands — needs the heightfield, so it lives here
//!   ([`fill_preview`]).
//!
//! # Determinism
//!
//! Everything is a pure function of `(the path or the rectangle, the height
//! function, the resolution)`. Fixed iteration counts, row-major order, no RNG,
//! no wall clock, no trigonometry. Two calls with the same arguments give
//! bit-identical answers — which matters because the same functions back an
//! editor preview and a report an author acts on, and a preview that moved
//! between two identical frames would be worse than none.
//!
//! Nothing here reaches the fixed step. These are *authoring* queries; the sim's
//! water is [`crate::WaterSurface`] and only that.

use glam::{DVec2, DVec3};

use crate::river::RiverPath;

/// The smallest grid a [`fill_preview`] is computed on, per side. Below this a
/// preview is a caricature: four samples cannot show where a waterline runs.
pub const MIN_FILL_RESOLUTION: u32 = 4;

/// The largest, per side. `256² = 65 536` height queries is a comfortable
/// interactive budget (a drag is re-previewed per frame) and well past the point
/// where more samples change the picture.
pub const MAX_FILL_RESOLUTION: u32 = 256;

/// Default grid for a lake fill preview, per side.
pub const DEFAULT_FILL_RESOLUTION: u32 = 64;

/// Where a still-water level of `level_m` lands on the ground inside a rectangle.
///
/// Produced by [`fill_preview`]; consumed by the lake tool's viewport preview and
/// by its "this lake covers 38 % of the box, up to 4.2 m deep" readout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillPreview {
    /// The level this preview was computed for, metres of world Y.
    pub level_m: f64,
    /// Fraction of the *known* samples that are at or below the level, `[0, 1]`.
    /// `0` for a rectangle entirely off the terrain.
    pub covered_fraction: f64,
    /// Deepest submersion found, metres (`0` when nothing is covered).
    pub max_depth_m: f64,
    /// Mean submersion over the **covered** samples, metres (`0` when nothing is
    /// covered). The mean over the whole rectangle would be a different, and much
    /// less useful, number — it drops as the box grows.
    pub mean_depth_m: f64,
    /// Total grid samples taken (`(resolution + 1)²`).
    pub samples: u32,
    /// How many of them the terrain answered for. `samples - known` is the part
    /// of the rectangle that is over a hole or off the authored extent, and a
    /// preview whose `known` is `0` is saying "there is no ground here", not
    /// "the lake is empty".
    pub known: u32,
    /// The **waterline**: world-XZ segments where the ground crosses the level,
    /// in row-major cell order. Draw them and you have drawn the shore.
    pub waterline: Vec<[DVec2; 2]>,
}

impl FillPreview {
    /// Whether the terrain answered for any sample at all.
    #[inline]
    pub fn has_ground(&self) -> bool {
        self.known > 0
    }
}

/// Sample the ground under an axis-aligned world-XZ rectangle and report where a
/// still-water level of `level_m` would sit on it.
///
/// `center` / `half_extent` are the lake's own convention (`WaterBody::extent` is
/// a half-extent centred on the entity). `resolution` is cells per side and is
/// clamped to `[MIN_FILL_RESOLUTION, MAX_FILL_RESOLUTION]`; the grid sampled is
/// `(resolution + 1)²` corners.
///
/// `height_at` returns `None` over a hole in the heightfield or outside the
/// authored extent, and those samples are **excluded** rather than defaulted:
/// a rectangle half off the terrain must not report itself half dry, and a
/// waterline must not be drawn along the edge of the data.
///
/// # The waterline is marching squares, and the ambiguous case is resolved by
/// rule, not by luck
///
/// Each cell's four corners are classified against the level and the standard
/// 16-case table is walked with linear interpolation along the crossed edges. The
/// two **saddle** cases (opposite corners on opposite sides) are genuinely
/// ambiguous — the same four heights describe both "two channels" and "one
/// isthmus" — and are always resolved the *same* way (as two separate segments
/// joining adjacent edges) so the output is a function of the heights alone. A
/// cell with any unknown corner emits nothing.
pub fn fill_preview(
    height_at: impl Fn(DVec2) -> Option<f64>,
    center: DVec2,
    half_extent: DVec2,
    level_m: f64,
    resolution: u32,
) -> FillPreview {
    let n = resolution.clamp(MIN_FILL_RESOLUTION, MAX_FILL_RESOLUTION);
    let hx = half_extent.x.abs();
    let hz = half_extent.y.abs();
    let min = DVec2::new(center.x - hx, center.y - hz);
    let span = DVec2::new(2.0 * hx, 2.0 * hz);
    let side = (n + 1) as usize;

    // Row-major corner grid. `f64::NAN` is never a terrain height, so `Option`
    // is kept rather than encoded — the marching pass has to know.
    let mut heights: Vec<Option<f64>> = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let p = DVec2::new(
                min.x + span.x * i as f64 / n as f64,
                min.y + span.y * j as f64 / n as f64,
            );
            heights.push(height_at(p).filter(|h| h.is_finite()));
        }
    }

    let mut known = 0u32;
    let mut covered = 0u32;
    let mut max_depth = 0.0f64;
    let mut depth_sum = 0.0f64;
    for h in heights.iter().flatten() {
        known += 1;
        let d = level_m - *h;
        if d >= 0.0 {
            covered += 1;
            depth_sum += d;
            if d > max_depth {
                max_depth = d;
            }
        }
    }

    let corner = |i: usize, j: usize| -> DVec2 {
        DVec2::new(
            min.x + span.x * i as f64 / n as f64,
            min.y + span.y * j as f64 / n as f64,
        )
    };
    let mut waterline: Vec<[DVec2; 2]> = Vec::new();
    for j in 0..n as usize {
        for i in 0..n as usize {
            let idx = |di: usize, dj: usize| heights[(j + dj) * side + (i + di)];
            let (Some(h00), Some(h10), Some(h01), Some(h11)) =
                (idx(0, 0), idx(1, 0), idx(0, 1), idx(1, 1))
            else {
                continue;
            };
            march_cell(
                [h00, h10, h11, h01],
                [
                    corner(i, j),
                    corner(i + 1, j),
                    corner(i + 1, j + 1),
                    corner(i, j + 1),
                ],
                level_m,
                &mut waterline,
            );
        }
    }

    FillPreview {
        level_m,
        covered_fraction: if known > 0 {
            covered as f64 / known as f64
        } else {
            0.0
        },
        max_depth_m: max_depth,
        mean_depth_m: if covered > 0 {
            depth_sum / covered as f64
        } else {
            0.0
        },
        samples: (side * side) as u32,
        known,
        waterline,
    }
}

/// One marching-squares cell. Corners are given in **winding** order
/// (`00, 10, 11, 01`) so edge `e` runs from corner `e` to corner `(e + 1) % 4`.
fn march_cell(h: [f64; 4], p: [DVec2; 4], level: f64, out: &mut Vec<[DVec2; 2]>) {
    // Bit `k` set ⇒ corner `k` is submerged (at or below the level). "At" counts
    // as submerged so a perfectly flat cell exactly at the level is inside rather
    // than a ring of degenerate segments.
    let mut mask = 0u8;
    for (k, hk) in h.iter().enumerate() {
        if level >= *hk {
            mask |= 1 << k;
        }
    }
    if mask == 0 || mask == 0b1111 {
        return;
    }
    // Interpolated crossing on edge `e`.
    let cross = |e: usize| -> DVec2 {
        let (a, b) = (e, (e + 1) % 4);
        let (ha, hb) = (h[a], h[b]);
        let d = hb - ha;
        let t = if d.abs() > 0.0 {
            ((level - ha) / d).clamp(0.0, 1.0)
        } else {
            0.5
        };
        p[a] + (p[b] - p[a]) * t
    };
    // Which edges each configuration cuts. The two saddles (0b0101, 0b1010) are
    // always split the same way — see the fn docs.
    let pairs: &[(usize, usize)] = match mask {
        0b0001 | 0b1110 => &[(3, 0)],
        0b0010 | 0b1101 => &[(0, 1)],
        0b0100 | 0b1011 => &[(1, 2)],
        0b1000 | 0b0111 => &[(2, 3)],
        0b0011 | 0b1100 => &[(3, 1)],
        0b0110 | 0b1001 => &[(0, 2)],
        0b0101 => &[(3, 0), (1, 2)],
        0b1010 => &[(0, 1), (2, 3)],
        _ => &[],
    };
    for &(a, b) in pairs {
        out.push([cross(a), cross(b)]);
    }
}

/// What is wrong where a river meets the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedIssue {
    /// The **terrain is above the water surface**: the river runs *inside* the
    /// hill. Nothing renders (the surface is below the ground that occludes it)
    /// and nothing complains — the most silent failure hydrology has.
    Buried,
    /// The **terrain is below the authored bed**: the ribbon hangs in the air
    /// over a gorge the depth profile does not reach. It draws, and it draws
    /// visibly wrong.
    Perched,
}

impl BedIssue {
    /// A short, stable id — the string an advisory or a DTO carries.
    pub const fn id(self) -> &'static str {
        match self {
            BedIssue::Buried => "buried",
            BedIssue::Perched => "perched",
        }
    }
}

/// A contiguous stretch of river with the same [`BedIssue`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BedConflict {
    /// What is wrong.
    pub issue: BedIssue,
    /// Arc length where the stretch starts, metres.
    pub from_s: f64,
    /// …and ends, metres.
    pub to_s: f64,
    /// The worst discrepancy over the stretch, metres — how far the terrain rises
    /// above the surface ([`BedIssue::Buried`]) or falls below the authored bed
    /// ([`BedIssue::Perched`]).
    pub worst_m: f64,
    /// World XZ of the worst frame, so a tool can fly the camera to it.
    pub worst_xz: DVec2,
}

impl BedConflict {
    /// Length of the stretch, metres.
    #[inline]
    pub fn length_m(&self) -> f64 {
        self.to_s - self.from_s
    }
}

/// Compare a river against the ground it crosses.
///
/// `height_at` is the terrain (the editor's resident `Terrain::data`, offset by
/// the terrain entity's own origin); frames it answers `None` for are **skipped**,
/// exactly as [`RiverPath::bed_profile`] skips them, so a river crossing an
/// unloaded region is not reported as flying.
///
/// `tolerance_m` absorbs the same two wobbles the cook's tolerance does — spline
/// overshoot between knots and arc-length resampling — plus bilinear heightfield
/// noise where the centreline crosses a tile diagonal, which is why the tools
/// pass a slightly larger value than the cook's 0.5 m. Below it, nothing is
/// reported: an author who sculpted a bank to within a few centimetres of the
/// water has done the job.
///
/// Adjacent offending frames are **merged into spans**, and a span carries its
/// worst frame, because "your river is buried for 40 m, worst by 3.1 m" is
/// actionable and forty separate lines are not. Deterministic: one pass in frame
/// order, no sorting, no map.
pub fn bed_conflicts(
    path: &RiverPath,
    height_at: impl Fn(DVec2) -> Option<f64>,
    tolerance_m: f64,
) -> Vec<BedConflict> {
    let tol = if tolerance_m.is_finite() {
        tolerance_m.max(0.0)
    } else {
        0.0
    };
    let mut out: Vec<BedConflict> = Vec::new();
    let mut open: Option<BedConflict> = None;
    for f in &path.frames {
        let xz = DVec2::new(f.center.x, f.center.z);
        let Some(ground) = height_at(xz).filter(|h| h.is_finite()) else {
            // A hole closes any open span: the stretch we can speak about ended
            // where the data did.
            if let Some(c) = open.take() {
                out.push(c);
            }
            continue;
        };
        let surface = f.center.y;
        let bed = surface - f.depth_m.max(0.0);
        let (issue, amount) = if ground - surface > tol {
            (Some(BedIssue::Buried), ground - surface)
        } else if bed - ground > tol {
            (Some(BedIssue::Perched), bed - ground)
        } else {
            (None, 0.0)
        };
        match (issue, open.as_mut()) {
            (Some(kind), Some(c)) if c.issue == kind => {
                c.to_s = f.s;
                if amount > c.worst_m {
                    c.worst_m = amount;
                    c.worst_xz = xz;
                }
            }
            (Some(kind), _) => {
                if let Some(c) = open.take() {
                    out.push(c);
                }
                open = Some(BedConflict {
                    issue: kind,
                    from_s: f.s,
                    to_s: f.s,
                    worst_m: amount,
                    worst_xz: xz,
                });
            }
            (None, _) => {
                if let Some(c) = open.take() {
                    out.push(c);
                }
            }
        }
    }
    if let Some(c) = open {
        out.push(c);
    }
    out
}

/// The world-XZ footprint of a river, dilated by half its widest cross-section —
/// the box a tool frames the camera on, and the bound a spatial reject would use.
///
/// `None` for an empty path.
pub fn river_bounds(path: &RiverPath) -> Option<(DVec2, DVec2)> {
    if path.frames.is_empty() {
        return None;
    }
    let pad = 0.5 * path.max_width_m();
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    for f in &path.frames {
        min = min.min(DVec2::new(f.center.x, f.center.z));
        max = max.max(DVec2::new(f.center.x, f.center.z));
    }
    Some((min - DVec2::splat(pad), max + DVec2::splat(pad)))
}

/// The still-water level a river's own surface implies at its **source** — what a
/// lake feeding it should sit at.
///
/// `None` for an empty path. Used by the lake tool when it is placed at the head
/// of an existing river: the two must agree, or the river starts with a step.
pub fn river_source_level_m(path: &RiverPath) -> Option<f64> {
    path.frames.first().map(|f: &crate::RiverFrame| f.center.y)
}

/// A world point on a river's centreline at arc length `s`, clamped to the path.
/// `None` for an empty path. The tools use it to place a camera or a probe.
pub fn river_point_at(path: &RiverPath, s: f64) -> Option<DVec3> {
    if path.frames.is_empty() {
        return None;
    }
    let s = s.clamp(0.0, path.length_m);
    for w in path.frames.windows(2) {
        if s <= w[1].s {
            let d = w[1].s - w[0].s;
            let u = if d > 0.0 { (s - w[0].s) / d } else { 0.0 };
            return Some(w[0].center.lerp(w[1].center, u));
        }
    }
    path.frames.last().map(|f| f.center)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::river::{RiverProfile, DEFAULT_SAMPLES_PER_SEGMENT};
    use inf_math::spline::SplineInterp;

    /// A bowl: `y = (|x| + |z|) * 0.1 - 5`, so the terrain rises away from the
    /// origin and a level of `-5` touches exactly one point.
    fn bowl(p: DVec2) -> Option<f64> {
        Some((p.x.abs() + p.y.abs()) * 0.1 - 5.0)
    }

    fn straight_river(depth: f64) -> RiverPath {
        RiverPath::build(
            &[
                DVec3::new(0.0, 10.0, 0.0),
                DVec3::new(100.0, 5.0, 0.0),
                DVec3::new(200.0, 0.0, 0.0),
            ],
            false,
            SplineInterp::Linear,
            &RiverProfile {
                width_start_m: 8.0,
                width_end_m: 8.0,
                depth_start_m: depth,
                depth_end_m: depth,
                flow_speed_m_s: 1.0,
            },
            DEFAULT_SAMPLES_PER_SEGMENT,
        )
    }

    #[test]
    fn a_fill_preview_measures_coverage_and_depth() {
        let p = fill_preview(bowl, DVec2::ZERO, DVec2::splat(50.0), 0.0, 64);
        assert_eq!(p.level_m, 0.0);
        assert_eq!(p.known, p.samples, "the bowl answers everywhere");
        assert!(p.has_ground());
        // The level-0 contour of `(|x|+|z|)*0.1 - 5` is the diamond
        // |x| + |z| = 50, whose area is half the 100×100 box.
        assert!(
            (p.covered_fraction - 0.5).abs() < 0.02,
            "covered {}",
            p.covered_fraction
        );
        // Deepest at the centre: 0 - (-5) = 5 m.
        assert!((p.max_depth_m - 5.0).abs() < 0.05, "{}", p.max_depth_m);
        assert!(p.mean_depth_m > 0.0 && p.mean_depth_m < p.max_depth_m);
        assert!(!p.waterline.is_empty(), "the shore must be drawable");
        // Every waterline vertex sits on the contour, to within a cell.
        let cell = 100.0 / 64.0;
        for [a, b] in &p.waterline {
            for v in [a, b] {
                let h = bowl(*v).unwrap();
                assert!(h.abs() < cell * 0.15, "vertex off the contour: {h}");
            }
        }
    }

    #[test]
    fn a_fill_preview_is_deterministic_and_resolution_is_clamped() {
        let a = fill_preview(bowl, DVec2::new(3.0, -7.0), DVec2::splat(30.0), -1.5, 32);
        let b = fill_preview(bowl, DVec2::new(3.0, -7.0), DVec2::splat(30.0), -1.5, 32);
        assert_eq!(a, b);
        // Absurd resolutions are clamped, not honoured or rejected.
        let lo = fill_preview(bowl, DVec2::ZERO, DVec2::splat(10.0), 0.0, 0);
        assert_eq!(lo.samples, (MIN_FILL_RESOLUTION + 1).pow(2));
        let hi = fill_preview(bowl, DVec2::ZERO, DVec2::splat(10.0), 0.0, 100_000);
        assert_eq!(hi.samples, (MAX_FILL_RESOLUTION + 1).pow(2));
    }

    #[test]
    fn a_hole_is_excluded_rather_than_defaulted() {
        // Ground at −2 everywhere, with a hole over the right half.
        let holed = |p: DVec2| if p.x > 0.0 { None } else { Some(-2.0) };
        let p = fill_preview(holed, DVec2::ZERO, DVec2::splat(10.0), 0.0, 16);
        assert!(p.known < p.samples, "the hole was not detected");
        assert!(p.known > 0);
        // Every KNOWN sample is covered, so the coverage is 1 — not 0.5, which is
        // what defaulting the hole to y = 0 would have produced.
        assert!(
            (p.covered_fraction - 1.0).abs() < 1e-12,
            "{}",
            p.covered_fraction
        );
        // …and no waterline is drawn along the edge of the DATA.
        assert!(p.waterline.is_empty(), "{:?}", p.waterline);

        // Entirely off the terrain: "there is no ground here", not "empty lake".
        let none = fill_preview(|_| None, DVec2::ZERO, DVec2::splat(10.0), 0.0, 8);
        assert_eq!(none.known, 0);
        assert!(!none.has_ground());
        assert_eq!(none.covered_fraction, 0.0);
    }

    #[test]
    fn a_level_below_everything_covers_nothing() {
        let p = fill_preview(bowl, DVec2::ZERO, DVec2::splat(50.0), -99.0, 32);
        assert_eq!(p.covered_fraction, 0.0);
        assert_eq!(p.max_depth_m, 0.0);
        assert_eq!(p.mean_depth_m, 0.0);
        assert!(p.waterline.is_empty());
        // …and one above everything covers all of it, with no shore inside the box.
        let q = fill_preview(bowl, DVec2::ZERO, DVec2::splat(50.0), 99.0, 32);
        assert_eq!(q.covered_fraction, 1.0);
        assert!(q.waterline.is_empty());
    }

    /// Ground that hugs the river's own authored bed — the "correct content"
    /// baseline every conflict case perturbs.
    fn hugging_bed(p: DVec2) -> Option<f64> {
        straight_river(1.5)
            .sample(p)
            .map(|s| s.surface_y - s.depth_m)
    }

    #[test]
    fn a_buried_river_is_reported_with_its_worst_frame() {
        // A ridge across the middle of an otherwise correctly-bedded river.
        let ground = |p: DVec2| {
            if (80.0..120.0).contains(&p.x) {
                Some(20.0) // well above the surface (~5 m there)
            } else {
                hugging_bed(p)
            }
        };
        let path = straight_river(1.5);
        let issues = bed_conflicts(&path, ground, 0.5);
        assert_eq!(issues.len(), 1, "{issues:?}");
        let c = issues[0];
        assert_eq!(c.issue, BedIssue::Buried);
        assert_eq!(c.issue.id(), "buried");
        assert!(c.from_s > 50.0 && c.to_s < 150.0, "{c:?}");
        assert!(c.length_m() > 20.0);
        assert!(c.worst_m > 10.0, "{c:?}");
        assert!((80.0..=120.0).contains(&c.worst_xz.x), "{c:?}");

        // ANTI-VACUITY: the same river over ground that hugs its bed is clean.
        assert!(bed_conflicts(&path, hugging_bed, 0.5).is_empty());
    }

    #[test]
    fn a_perched_river_is_reported_and_the_tolerance_bites() {
        // Ground 40 m below everything: the ribbon hangs in the air.
        let path = straight_river(1.5);
        let issues = bed_conflicts(&path, |_| Some(-40.0), 0.5);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].issue, BedIssue::Perched);
        assert_eq!(issues[0].issue.id(), "perched");
        assert!(issues[0].worst_m > 30.0);

        // A 30 cm gap under the bed is inside a 50 cm tolerance and is silent…
        let near = |p: DVec2| hugging_bed(p).map(|h| h - 0.3);
        assert!(bed_conflicts(&path, near, 0.5).is_empty());
        // …and is reported once the tolerance stops covering it, which is what
        // proves the tolerance is doing the suppressing rather than the geometry.
        assert!(!bed_conflicts(&path, near, 0.1).is_empty());
    }

    #[test]
    fn holes_close_a_span_rather_than_bridging_it() {
        // Buried, hole, buried: two spans, not one straddling the gap.
        let ground = |p: DVec2| {
            if (90.0..110.0).contains(&p.x) {
                None
            } else {
                Some(50.0) // far above the surface everywhere else
            }
        };
        let path = straight_river(1.5);
        let issues = bed_conflicts(&path, ground, 0.5);
        assert_eq!(issues.len(), 2, "{issues:?}");
        assert!(issues.iter().all(|c| c.issue == BedIssue::Buried));
        assert!(issues[0].to_s < issues[1].from_s);
    }

    #[test]
    fn bounds_and_probes_answer_for_a_real_path_and_not_for_an_empty_one() {
        let path = straight_river(1.5);
        let (min, max) = river_bounds(&path).unwrap();
        // The river runs 0 → 200 in x at z = 0, 8 m wide: padded by 4 m each way.
        assert!((min.x + 4.0).abs() < 1e-6, "{min:?}");
        assert!((max.x - 204.0).abs() < 1e-6, "{max:?}");
        assert!((min.y + 4.0).abs() < 1e-6 && (max.y - 4.0).abs() < 1e-6);
        assert!((river_source_level_m(&path).unwrap() - 10.0).abs() < 1e-9);
        let mid = river_point_at(&path, path.length_m * 0.5).unwrap();
        assert!((mid.x - 100.0).abs() < 1.0, "{mid:?}");
        // Clamped, never extrapolated.
        assert_eq!(
            river_point_at(&path, 1e9).unwrap(),
            path.frames.last().unwrap().center
        );
        assert_eq!(river_point_at(&path, -1e9).unwrap(), path.frames[0].center);

        let empty = RiverPath::default();
        assert!(river_bounds(&empty).is_none());
        assert!(river_source_level_m(&empty).is_none());
        assert!(river_point_at(&empty, 0.0).is_none());
    }
}
