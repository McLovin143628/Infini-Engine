//! **The two filters**, each a separately testable pass over depth maps the
//! sweep has already produced: left-right consistency, and speckle removal.
//!
//! # Why filtering is not part of the sweep
//!
//! The sweep's confidence measure asks a *local* question — did this pixel's
//! cost curve have one clear minimum? A pixel on a repeated texture, or one on
//! the occluded side of an edge, can answer that question confidently and be
//! wrong. The two things that catch it are both **non-local**, and neither can
//! run inside the sweep:
//!
//! * **Left-right consistency** needs the neighbour views' finished depth maps,
//!   which do not exist while any one view is being swept.
//! * **Speckle removal** needs connected components over the whole map.
//!
//! Splitting them out is also what makes them testable: each is a pure function
//! from depth maps to a **keep-mask**, and the gate can sever one and watch a
//! specific arm go red.
//!
//! # Masks, not mutations
//!
//! Both passes return a `Vec<bool>` and mutate nothing. That is not tidiness: a
//! consistency check that invalidated pixels as it went would be comparing some
//! pixels against a neighbour's *original* depths and others against its
//! already-filtered ones, and the answer would depend on the order the views
//! were visited. Computing every mask from the unfiltered maps and applying them
//! afterwards makes the pass a gather, which is the same rule TSDF fusion
//! follows and the same rule the rest of the engine follows.
//!
//! # Neither pass runs on the GPU, and why
//!
//! Consistency is a handful of projections per pixel — real work, but nothing
//! next to the sweep, and it needs `f64` world positions the GPU does not have.
//! Speckle removal is a **connected-component labelling**: inherently
//! sequential, and its GPU form is an iterated label-propagation whose
//! termination is data-dependent — which is a loop count that varies with the
//! input, which is exactly the shape of thing the determinism rules keep out.
//! They are CPU passes on purpose, and this paragraph is the ruling rather than
//! an omission.

use inf_photo::sfm::RegisteredCamera;

use crate::sweep::{backproject_world, DepthMap};

/// How the two filters are tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterConfig {
    /// Two views agree about a surface point when their back-projections land
    /// within this fraction of the reference depth of each other.
    ///
    /// **Relative**, not absolute, because a depth map's error grows with depth:
    /// the disparity a plane step corresponds to shrinks as `1/z`, so a fixed
    /// metric tolerance would be lenient near and impossible far.
    pub lr_tolerance: f32,
    /// How many neighbour views must agree for a pixel to survive.
    pub lr_min_agree: usize,
    /// Two adjacent pixels belong to the same region when their depths are
    /// within this fraction of each other. Relative, for the same reason.
    pub speckle_delta: f32,
    /// A connected region smaller than this many pixels is noise.
    pub speckle_min_region: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            lr_tolerance: 0.02,
            lr_min_agree: 1,
            speckle_delta: 0.02,
            speckle_min_region: 24,
        }
    }
}

/// A keep-mask for one reference view, from agreement with its neighbours'
/// depth maps.
///
/// For every valid pixel: back-project it to a world point, project that point
/// into each neighbour, read the neighbour's own depth at the pixel it lands on,
/// back-project *that* to a world point, and count the neighbour as agreeing
/// when the two world points are within `lr_tolerance` of the reference depth of
/// each other. A pixel survives when at least [`FilterConfig::lr_min_agree`]
/// neighbours agree.
///
/// A neighbour that cannot see the point at all — behind it, outside its frame,
/// or with no depth there — is neither agreement nor disagreement. It simply
/// does not vote, which is what makes an occluded region fail the check rather
/// than a partly-visible one.
///
/// `maps` and `cameras` are parallel slices in the same order; `reference` and
/// `neighbours` index into them.
pub fn consistency_mask(
    maps: &[DepthMap],
    cameras: &[RegisteredCamera],
    reference: usize,
    neighbours: &[usize],
    cfg: &FilterConfig,
) -> Vec<bool> {
    let map = &maps[reference];
    let cam = &cameras[reference];
    let w = map.width;
    let h = map.height;
    let mut keep = vec![false; map.pixels()];
    for y in 0..h {
        for x in 0..w {
            let i = y as usize * w as usize + x as usize;
            let d = map.depth[i];
            if !(d > 0.0) {
                continue;
            }
            let world = backproject_world(cam, x, y, d);
            let tolerance = (cfg.lr_tolerance * d) as f64;
            let mut agree = 0usize;
            for &ni in neighbours {
                let nb_cam = &cameras[ni];
                let nb_map = &maps[ni];
                let camera_space = nb_cam.pose.transform(world);
                if camera_space.z <= 0.0 {
                    continue;
                }
                let Some(px) = nb_cam.intrinsics.project(camera_space) else {
                    continue;
                };
                let sx = (px.x + 0.5).floor();
                let sy = (px.y + 0.5).floor();
                if sx < 0.0
                    || sy < 0.0
                    || sx > (nb_map.width - 1) as f64
                    || sy > (nb_map.height - 1) as f64
                {
                    continue;
                }
                let (sx, sy) = (sx as u32, sy as u32);
                let j = sy as usize * nb_map.width as usize + sx as usize;
                let nd = nb_map.depth[j];
                if !(nd > 0.0) {
                    continue;
                }
                let theirs = backproject_world(nb_cam, sx, sy, nd);
                if (theirs - world).length() <= tolerance {
                    agree += 1;
                }
            }
            keep[i] = agree >= cfg.lr_min_agree;
        }
    }
    keep
}

/// What one speckle pass did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeckleReport {
    /// How many pixels the pass would drop.
    pub dropped: usize,
    /// How many separate regions were dropped.
    pub regions: usize,
    /// How many connected regions there were in total.
    pub total_regions: usize,
    /// The largest region's size, in pixels.
    pub largest: usize,
}

/// A keep-mask that removes small connected regions.
///
/// Four-connected flood fill in **raster order** with an explicit stack, so the
/// labelling is a pure function of the map and not of a recursion depth or a
/// container's iteration order. Two neighbouring pixels are linked when both
/// carry a depth and those depths are within [`FilterConfig::speckle_delta`] of
/// the larger — so a depth *discontinuity* is a region boundary, which is what
/// makes a wrong patch floating in front of a wall its own small region rather
/// than part of the wall.
pub fn speckle_mask(map: &DepthMap, cfg: &FilterConfig) -> (Vec<bool>, SpeckleReport) {
    let w = map.width as usize;
    let h = map.height as usize;
    let n = w * h;
    let mut keep = vec![false; n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut region: Vec<usize> = Vec::new();
    let mut report = SpeckleReport {
        dropped: 0,
        regions: 0,
        total_regions: 0,
        largest: 0,
    };
    for start in 0..n {
        if seen[start] || !(map.depth[start] > 0.0) {
            continue;
        }
        stack.clear();
        region.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = (i % w, i / w);
            let visit = |j: usize, stack: &mut Vec<usize>, seen: &mut Vec<bool>| {
                if seen[j] || !(map.depth[j] > 0.0) {
                    return;
                }
                let a = map.depth[i];
                let b = map.depth[j];
                let larger = if a > b { a } else { b };
                if (a - b).abs() <= cfg.speckle_delta * larger {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1, &mut stack, &mut seen);
            }
            if x + 1 < w {
                visit(i + 1, &mut stack, &mut seen);
            }
            if y > 0 {
                visit(i - w, &mut stack, &mut seen);
            }
            if y + 1 < h {
                visit(i + w, &mut stack, &mut seen);
            }
        }
        report.total_regions += 1;
        report.largest = report.largest.max(region.len());
        if region.len() >= cfg.speckle_min_region {
            for &i in &region {
                keep[i] = true;
            }
        } else {
            report.dropped += region.len();
            report.regions += 1;
        }
    }
    (keep, report)
}

/// Apply a keep-mask, invalidating both channels together.
///
/// Returns how many pixels it took.
pub fn apply_mask(map: &mut DepthMap, keep: &[bool]) -> usize {
    let mut dropped = 0usize;
    for (i, kept) in keep.iter().enumerate().take(map.pixels()) {
        if map.depth[i] > 0.0 && !kept {
            map.invalidate(i);
            dropped += 1;
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec2, DVec3};
    use inf_photo::camera::{Intrinsics, Pose};

    fn camera(view: u32, eye: DVec3) -> RegisteredCamera {
        RegisteredCamera {
            view,
            pose: Pose::look_at(eye, DVec3::new(0.0, 0.0, 4.0), DVec3::NEG_Y),
            intrinsics: Intrinsics::centred(32, 24, 30.0),
            observations: 0,
        }
    }

    /// A depth map of a plane at `z = plane` as seen by `cam`.
    fn plane_map(cam: &RegisteredCamera, plane: f64) -> DepthMap {
        let mut map = DepthMap::empty(cam.view, 32, 24);
        for y in 0..24u32 {
            for x in 0..32u32 {
                // The plane z = `plane` in WORLD space; the depth along the
                // camera's own +Z is what a depth map holds.
                let n = cam.intrinsics.to_normalized(DVec2::new(x as f64, y as f64));
                let dir = cam.pose.rotation.inverse() * DVec3::new(n.x, n.y, 1.0);
                let eye = cam.pose.center();
                if dir.z.abs() < 1e-9 {
                    continue;
                }
                let t = (plane - eye.z) / dir.z;
                if t <= 0.0 {
                    continue;
                }
                let i = y as usize * 32 + x as usize;
                map.depth[i] = t as f32;
                map.confidence[i] = 0.5;
            }
        }
        map
    }

    #[test]
    fn agreeing_views_survive_the_consistency_check() {
        let cams = vec![
            camera(0, DVec3::new(0.0, 0.0, 0.0)),
            camera(1, DVec3::new(0.6, 0.0, 0.0)),
            camera(2, DVec3::new(-0.5, 0.1, 0.0)),
        ];
        let maps: Vec<DepthMap> = cams.iter().map(|c| plane_map(c, 4.0)).collect();
        let cfg = FilterConfig::default();
        let keep = consistency_mask(&maps, &cams, 0, &[1, 2], &cfg);
        let valid = maps[0].valid_count();
        let kept = keep.iter().filter(|k| **k).count();
        assert!(valid > 600, "the fixture map is nearly empty: {valid}");
        assert!(
            kept * 10 >= valid * 9,
            "only {kept} of {valid} pixels survived agreement between three views of one plane"
        );
    }

    #[test]
    fn a_disagreeing_neighbour_takes_the_pixel_with_it() {
        // Anti-vacuity from the other side. Same three cameras, but the
        // neighbours' maps describe a plane a whole unit nearer than the
        // reference's. Nothing should agree.
        let cams = vec![
            camera(0, DVec3::new(0.0, 0.0, 0.0)),
            camera(1, DVec3::new(0.6, 0.0, 0.0)),
            camera(2, DVec3::new(-0.5, 0.1, 0.0)),
        ];
        let mut maps: Vec<DepthMap> = cams.iter().map(|c| plane_map(c, 4.0)).collect();
        maps[1] = plane_map(&cams[1], 3.0);
        maps[2] = plane_map(&cams[2], 3.0);
        let keep = consistency_mask(&maps, &cams, 0, &[1, 2], &FilterConfig::default());
        let kept = keep.iter().filter(|k| **k).count();
        assert_eq!(
            kept, 0,
            "{kept} pixels agreed with a neighbour a whole unit out"
        );
    }

    #[test]
    fn speckle_removal_takes_the_island_and_leaves_the_continent() {
        let mut map = DepthMap::empty(0, 32, 24);
        // A big block of consistent depth...
        for y in 4..20u32 {
            for x in 4..28u32 {
                let i = y as usize * 32 + x as usize;
                map.depth[i] = 3.0;
                map.confidence[i] = 1.0;
            }
        }
        // ...and a 2x2 island at a different depth, inside it.
        for y in 10..12u32 {
            for x in 10..12u32 {
                let i = y as usize * 32 + x as usize;
                map.depth[i] = 1.0;
            }
        }
        let (keep, report) = speckle_mask(&map, &FilterConfig::default());
        assert_eq!(report.dropped, 4, "report: {report:?}");
        assert_eq!(report.regions, 1);
        assert_eq!(report.total_regions, 2);
        assert_eq!(report.largest, 16 * 24 - 4);
        for y in 10..12u32 {
            for x in 10..12u32 {
                assert!(!keep[y as usize * 32 + x as usize], "the island survived");
            }
        }
        assert!(keep[4 * 32 + 4], "the continent was taken");
    }

    #[test]
    fn a_depth_step_splits_a_region_even_when_the_pixels_touch() {
        // The property the relative link tolerance exists for: two slabs that
        // are adjacent in the image and a long way apart in depth are two
        // regions, not one.
        let mut map = DepthMap::empty(0, 32, 24);
        for y in 0..24u32 {
            for x in 0..32u32 {
                let i = y as usize * 32 + x as usize;
                map.depth[i] = if x < 16 { 2.0 } else { 5.0 };
                map.confidence[i] = 1.0;
            }
        }
        let (_, report) = speckle_mask(&map, &FilterConfig::default());
        assert_eq!(report.total_regions, 2, "report: {report:?}");
        assert_eq!(report.dropped, 0, "both halves are large enough to keep");
    }

    #[test]
    fn applying_a_mask_clears_both_channels_together() {
        let mut map = DepthMap::empty(0, 4, 1);
        for i in 0..4 {
            map.depth[i] = 1.0 + i as f32;
            map.confidence[i] = 0.5;
        }
        let dropped = apply_mask(&mut map, &[true, false, true, false]);
        assert_eq!(dropped, 2);
        assert_eq!(map.depth, vec![1.0, 0.0, 3.0, 0.0]);
        assert_eq!(map.confidence, vec![0.5, 0.0, 0.5, 0.0]);
    }
}
