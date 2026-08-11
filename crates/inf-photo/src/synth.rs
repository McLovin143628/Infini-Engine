//! The **known-truth dataset**: an analytic scene, a tiny deterministic
//! software renderer, and the exact poses the images were taken from.
//!
//! # Why a renderer lives in a Ring-0 photogrammetry crate
//!
//! A reconstruction can only be *measured* against poses that are given rather
//! than estimated. There are three ways to get those:
//!
//! 1. **Analytic projections** — skip the images, feed the solver the exact
//!    pixel coordinates a known point projects to. Cheap, and it measures the
//!    geometry stages while proving nothing at all about feature detection or
//!    matching, which is where most of a photogrammetry pipeline's risk lives.
//! 2. **Real renders** through `inf-render`. Faithful, and it makes the SfM core
//!    depend on a GPU, on an adapter, and on a golden-image harness — an
//!    inversion of the rings for a *test* fixture.
//! 3. **This.** A few textured planes, a ray per sub-sample, `f64` throughout,
//!    no GPU and no committed image files. The whole pipeline runs — detector,
//!    descriptor, matcher, RANSAC, bundle — on images whose camera poses are
//!    known exactly.
//!
//! The third is the one that keeps `inf-photo` in Ring 0 and still tests the
//! thing end to end, so it is what this module is. A real-render arm belongs in
//! a higher crate once P25.2 gives the dense stage something to compare against.
//!
//! # The scene, and why it is lopsided
//!
//! Three planes: a floor, a back wall, and a **slanted** wall on one side. The
//! slant is not decoration — a fixture with a mirror symmetry cannot detect a
//! sign error in a reconstruction, because the mirrored answer fits it exactly.
//! The camera stations are lopsided for the same reason: unequal spacing,
//! unequal heights, unequal distances, and a different look-at target and roll
//! for each, so no rotation, reflection or index permutation of the truth maps
//! onto itself.
//!
//! # The texture, and why it is dots
//!
//! FAST looks for a pixel that differs from a contiguous arc of the ring around
//! it. A small high-contrast disc is the ideal such thing, and a *field* of them
//! at hash-jittered positions with hash-varied radii and polarities is locally
//! distinctive — which is what a BRIEF descriptor needs and what a repeating
//! checkerboard conspicuously is not. Coverage is a smooth falloff at the disc
//! edge, so a feature has a well-defined sub-pixel position and does not jump a
//! whole pixel when the camera moves slightly.
//!
//! # Determinism
//!
//! `f64` arithmetic, integer loop bounds, no `std` transcendental (rays come
//! from [`crate::camera::Intrinsics::to_normalized`], which is polynomial;
//! camera frames come from cross products). The dot field is a pure function of
//! its cell coordinates through [`crate::hash::Hash64`], so it does not depend
//! on the order pixels are shaded in and there is no texture image anywhere.

use glam::{DVec2, DVec3};

use crate::camera::{Intrinsics, Pose};
use crate::gray::GrayImage;
use crate::hash::Hash64;
use crate::linalg::normalize_or;
use crate::sfm::View;

/// A field of jittered discs on a plane, defined by arithmetic rather than by
/// pixels.
#[derive(Clone, Copy, Debug)]
pub struct DotField {
    /// Seed for the cell hash.
    pub seed: u64,
    /// Grid pitch, in **plane units** (metres, in the scene's own frame).
    pub cell: f64,
    /// Background level.
    pub background: u8,
    /// Dark-dot level.
    pub dark: u8,
    /// Light-dot level.
    pub light: u8,
    /// Disc radius as a fraction of `cell`, minimum.
    pub min_radius: f64,
    /// Disc radius as a fraction of `cell`, maximum. Must stay at or below
    /// [`DotField::MAX_SAFE_RADIUS`] — see the derivation there.
    pub max_radius: f64,
}

impl DotField {
    /// The largest `max_radius` the four-cell scan in [`DotField::sample`] is
    /// exact for.
    ///
    /// A disc in cell `C` has its centre in `[C + 0.2, C + 0.8]` and radius at
    /// most `R`, so it covers `[C + 0.2 - R, C + 0.8 + R]`. A sample in the
    /// lower half of cell `C` scans `{C-1, C}`; the nearest disc it *does not*
    /// scan lives in `C+1` and reaches down to `C + 1.2 - R`, which must stay
    /// above `C + 0.5`. That gives `R <= 0.7`, and the upper-half case gives the
    /// same bound by symmetry.
    pub const MAX_SAFE_RADIUS: f64 = 0.7;

    /// The intensity at a point on the plane.
    ///
    /// Discs are composited in **ascending cell order** rather than
    /// winner-takes-all, which makes the field continuous: at the boundary where
    /// the scan swaps one neighbour for another, both the departing and the
    /// arriving cell contribute exactly zero (that is what
    /// [`DotField::MAX_SAFE_RADIUS`] guarantees), so nothing steps.
    pub fn sample(&self, u: f64, v: f64) -> f64 {
        debug_assert!(self.max_radius <= Self::MAX_SAFE_RADIUS);
        let base = Hash64::new(self.seed);
        let cu = (u / self.cell).floor();
        let cv = (v / self.cell).floor();
        let fu = u / self.cell - cu;
        let fv = v / self.cell - cv;
        let du = if fu < 0.5 { [-1i64, 0] } else { [0, 1] };
        let dv = if fv < 0.5 { [-1i64, 0] } else { [0, 1] };
        let mut value = self.background as f64;
        for &dj in &dv {
            for &di in &du {
                let ci = cu as i64 + di;
                let cj = cv as i64 + dj;
                let h = base.mix_i64(ci).mix_i64(cj).finish();
                let jx = (h & 0xffff) as f64 / 65536.0;
                let jy = ((h >> 16) & 0xffff) as f64 / 65536.0;
                let jr = ((h >> 32) & 0xffff) as f64 / 65536.0;
                let polarity = (h >> 48) & 1;
                let cx = (ci as f64 + 0.2 + 0.6 * jx) * self.cell;
                let cy = (cj as f64 + 0.2 + 0.6 * jy) * self.cell;
                let radius =
                    self.cell * (self.min_radius + (self.max_radius - self.min_radius) * jr);
                let d = ((u - cx) * (u - cx) + (v - cy) * (v - cy)).sqrt();
                let soft = radius * 0.25;
                let t = ((radius - d) / soft).clamp(0.0, 1.0);
                let coverage = t * t * (3.0 - 2.0 * t);
                if coverage <= 0.0 {
                    continue;
                }
                let target = if polarity == 0 {
                    self.dark as f64
                } else {
                    self.light as f64
                };
                value += (target - value) * coverage;
            }
        }
        value
    }
}

/// A finite textured plane.
#[derive(Clone, Copy, Debug)]
pub struct TexturedPlane {
    /// A point on the plane; also the texture's origin.
    pub origin: DVec3,
    /// Unit normal.
    pub normal: DVec3,
    /// Unit in-plane axis for the texture's `u`.
    pub u_axis: DVec3,
    /// Unit in-plane axis for the texture's `v` (`normal x u_axis`).
    pub v_axis: DVec3,
    /// Half-extent along `u_axis`, in metres.
    pub u_extent: f64,
    /// Half-extent along `v_axis`, in metres.
    pub v_extent: f64,
    /// The texture.
    pub texture: DotField,
}

impl TexturedPlane {
    /// Build a plane, orthonormalising `u_hint` against the normal.
    pub fn new(
        origin: DVec3,
        normal: DVec3,
        u_hint: DVec3,
        u_extent: f64,
        v_extent: f64,
        texture: DotField,
    ) -> Self {
        let normal = normalize_or(normal, DVec3::Y);
        let u_axis = normalize_or(u_hint - normal * normal.dot(u_hint), DVec3::X);
        let v_axis = normal.cross(u_axis);
        Self {
            origin,
            normal,
            u_axis,
            v_axis,
            u_extent,
            v_extent,
            texture,
        }
    }

    /// Ray-plane intersection. Returns the ray parameter and the texture
    /// coordinates, or `None` for a miss.
    fn intersect(&self, ray_origin: DVec3, ray_dir: DVec3) -> Option<(f64, f64, f64)> {
        let denom = ray_dir.dot(self.normal);
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = (self.origin - ray_origin).dot(self.normal) / denom;
        if t <= 1e-6 || !t.is_finite() {
            return None;
        }
        let p = ray_origin + ray_dir * t;
        let d = p - self.origin;
        let u = d.dot(self.u_axis);
        let v = d.dot(self.v_axis);
        if u.abs() > self.u_extent || v.abs() > self.v_extent {
            return None;
        }
        Some((t, u, v))
    }
}

/// The scene: some planes and the level a ray that hits nothing reports.
#[derive(Clone, Debug)]
pub struct SynthScene {
    /// The planes.
    pub planes: Vec<TexturedPlane>,
    /// What a missing ray reports.
    pub background: u8,
}

impl SynthScene {
    /// The committed fixture scene: floor, back wall, and a slanted side wall.
    ///
    /// Deliberately **asymmetric** — the slant is on one side only, the three
    /// dot fields have different pitches and different seeds, and the extents
    /// differ, so no reflection of this scene is this scene.
    pub fn alcove() -> Self {
        let floor = TexturedPlane::new(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::Y,
            DVec3::X,
            5.0,
            5.0,
            DotField {
                seed: 0x5f10_0f00,
                cell: 0.170,
                background: 205,
                dark: 30,
                light: 250,
                min_radius: 0.18,
                max_radius: 0.34,
            },
        );
        let back = TexturedPlane::new(
            DVec3::new(0.0, 2.0, 3.0),
            DVec3::NEG_Z,
            DVec3::X,
            5.0,
            2.0,
            DotField {
                seed: 0x0bac_c001,
                cell: 0.230,
                background: 190,
                dark: 25,
                light: 245,
                min_radius: 0.16,
                max_radius: 0.33,
            },
        );
        let slant = TexturedPlane::new(
            DVec3::new(2.8, 1.5, 0.3),
            DVec3::new(-0.82, 0.15, -0.55),
            DVec3::Y,
            1.6,
            2.2,
            DotField {
                seed: 0x5aa1_7ee0,
                cell: 0.150,
                background: 215,
                dark: 35,
                light: 255,
                min_radius: 0.19,
                max_radius: 0.36,
            },
        );
        Self {
            planes: vec![floor, back, slant],
            background: 12,
        }
    }

    /// The nearest plane a world-space ray hits, as `(plane index, distance)`.
    pub fn nearest_hit(&self, origin: DVec3, dir: DVec3) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, plane) in self.planes.iter().enumerate() {
            if let Some((t, _, _)) = plane.intersect(origin, dir) {
                if best.map(|(_, bt)| t < bt).unwrap_or(true) {
                    best = Some((i, t));
                }
            }
        }
        best
    }

    /// The intensity along a world-space ray.
    fn shade(&self, origin: DVec3, dir: DVec3) -> f64 {
        let mut nearest = f64::INFINITY;
        let mut value = self.background as f64;
        for plane in &self.planes {
            if let Some((t, u, v)) = plane.intersect(origin, dir) {
                if t < nearest {
                    nearest = t;
                    value = plane.texture.sample(u, v);
                }
            }
        }
        value
    }
}

/// One committed camera station: eye, look-at target, and the "up" that sets
/// the roll. Given as raw arrays so the table reads as data.
///
/// Every row differs from every other in position, target **and** roll: a ring
/// of cameras at one radius and one height around the origin is the classic
/// fixture that cannot see a sign error, and this is deliberately not one.
pub const STATIONS: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
    (
        [0.00, 1.50, -3.20],
        [-0.20, 1.00, 1.50],
        [-0.05, -1.00, 0.00],
    ),
    (
        [0.85, 1.35, -3.00],
        [-0.10, 1.05, 1.50],
        [-0.02, -1.00, 0.03],
    ),
    ([1.75, 1.70, -2.50], [0.05, 1.10, 1.55], [0.01, -1.00, 0.05]),
    ([2.50, 1.25, -1.70], [0.20, 1.15, 1.60], [0.04, -1.00, 0.02]),
    (
        [-0.80, 1.60, -3.35],
        [-0.30, 1.02, 1.45],
        [-0.06, -1.00, -0.02],
    ),
    (
        [-1.70, 1.15, -2.85],
        [-0.35, 1.08, 1.50],
        [-0.03, -1.00, 0.04],
    ),
];

/// How the dataset is rendered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthConfig {
    /// Image width, pixels.
    pub width: u32,
    /// Image height, pixels.
    pub height: u32,
    /// Focal length, pixels (square pixels).
    pub focal_px: f64,
    /// First radial coefficient. Non-zero on purpose: it exercises the
    /// undistortion in both directions, which nothing else in the crate's tests
    /// does end to end.
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
    /// Samples per pixel axis (so `2` means four samples).
    pub supersample: u32,
    /// How many of [`STATIONS`] to render.
    pub stations: usize,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            width: 320,
            height: 240,
            focal_px: 300.0,
            k1: -0.09,
            k2: 0.02,
            supersample: 3,
            stations: STATIONS.len(),
        }
    }
}

impl SynthConfig {
    /// The intrinsics every station is rendered with.
    pub fn intrinsics(&self) -> Intrinsics {
        Intrinsics::centred(self.width, self.height, self.focal_px).with_radial(self.k1, self.k2)
    }

    /// The truth pose of station `i`.
    pub fn pose(&self, i: usize) -> Pose {
        let (eye, target, up) = STATIONS[i];
        Pose::look_at(
            DVec3::from_array(eye),
            DVec3::from_array(target),
            DVec3::from_array(up),
        )
    }
}

/// Views and the poses they were rendered from.
#[derive(Clone, Debug)]
pub struct SynthDataset {
    /// The rendered views, in station order.
    pub views: Vec<View>,
    /// The exact poses, in station order.
    pub truth: Vec<Pose>,
}

impl SynthDataset {
    /// Render the committed alcove from the first `cfg.stations` stations.
    pub fn alcove(cfg: &SynthConfig) -> Self {
        Self::render(&SynthScene::alcove(), cfg)
    }

    /// Render an arbitrary scene from the committed stations.
    pub fn render(scene: &SynthScene, cfg: &SynthConfig) -> Self {
        let intrinsics = cfg.intrinsics();
        // Camera-space ray directions, computed once: they depend on the
        // intrinsics and the sub-sample grid alone, not on the station. This is
        // also the only place undistortion runs during rendering.
        let rays = camera_rays(&intrinsics, cfg.supersample);
        let n = cfg.stations.min(STATIONS.len());
        let mut views = Vec::with_capacity(n);
        let mut truth = Vec::with_capacity(n);
        for i in 0..n {
            let pose = cfg.pose(i);
            let image = render_view(scene, cfg, &rays, &pose);
            views.push(View { image, intrinsics });
            truth.push(pose);
        }
        Self { views, truth }
    }
}

/// Unit camera-space ray directions for every sub-sample of every pixel, in
/// row-major pixel order with sub-samples in row-major order inside a pixel.
fn camera_rays(intrinsics: &Intrinsics, supersample: u32) -> Vec<DVec3> {
    let s = supersample.max(1);
    let mut out = Vec::with_capacity((intrinsics.width * intrinsics.height * s * s) as usize);
    for y in 0..intrinsics.height {
        for x in 0..intrinsics.width {
            for sy in 0..s {
                for sx in 0..s {
                    let px = DVec2::new(
                        x as f64 + (sx as f64 + 0.5) / s as f64,
                        y as f64 + (sy as f64 + 0.5) / s as f64,
                    );
                    let n = intrinsics.to_normalized(px);
                    out.push(DVec3::new(n.x, n.y, 1.0).normalize());
                }
            }
        }
    }
    out
}

fn render_view(scene: &SynthScene, cfg: &SynthConfig, rays: &[DVec3], pose: &Pose) -> GrayImage {
    let s = cfg.supersample.max(1);
    let per_pixel = (s * s) as usize;
    let eye = pose.center();
    let inverse = pose.rotation.inverse();
    let mut image = GrayImage::new(cfg.width, cfg.height).expect("non-zero dimensions");
    let mut index = 0usize;
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            let mut acc = 0.0f64;
            for k in 0..per_pixel {
                let dir = inverse * rays[index + k];
                acc += scene.shade(eye, dir);
            }
            index += per_pixel;
            let v = (acc / per_pixel as f64).round().clamp(0.0, 255.0) as u8;
            image.set(x, y, v);
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> SynthConfig {
        SynthConfig {
            width: 96,
            height: 72,
            focal_px: 90.0,
            supersample: 2,
            stations: 3,
            ..SynthConfig::default()
        }
    }

    #[test]
    fn the_stations_are_asymmetric() {
        // The law: a symmetric fixture cannot see a sign error. Assert the
        // asymmetry rather than trusting the table to stay lopsided.
        let eyes: Vec<DVec3> = STATIONS.iter().map(|s| DVec3::from_array(s.0)).collect();
        let centroid = eyes.iter().fold(DVec3::ZERO, |a, b| a + *b) / eyes.len() as f64;
        assert!(
            centroid.length() > 0.2,
            "the stations are centred on their own centroid: {centroid}"
        );
        // No station is the mirror of another about x = centroid.x.
        for (i, a) in eyes.iter().enumerate() {
            for (j, b) in eyes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let mirrored = DVec3::new(2.0 * centroid.x - a.x, a.y, a.z);
                assert!(
                    (mirrored - *b).length() > 0.15,
                    "stations {i} and {j} are mirror images about the centroid"
                );
            }
        }
        // Heights and distances vary.
        let heights: Vec<f64> = eyes.iter().map(|e| e.y).collect();
        let hi = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = heights.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(hi - lo > 0.4, "every station is at the same height");
        // And the rolls differ.
        let ups: Vec<DVec3> = STATIONS.iter().map(|s| DVec3::from_array(s.2)).collect();
        assert!(
            ups.windows(2).any(|w| (w[0] - w[1]).length() > 1e-6),
            "every station has the same roll"
        );
    }

    #[test]
    fn the_scene_geometry_is_not_mirror_symmetric() {
        // The law: a fixture with a mirror symmetry cannot see a sign error,
        // because the mirrored reconstruction fits it exactly. Measured on the
        // **geometry**, not on the texture — two hash-jittered dot fields
        // disagree pixel by pixel whatever the shape of the room, so a texture
        // comparison would pass on a perfectly symmetric box.
        let scene = SynthScene::alcove();
        let origin = DVec3::new(0.0, 1.5, -3.0);
        let mut differed = 0usize;
        let mut compared = 0usize;
        for i in 0..40 {
            let t = -0.6 + 0.03 * i as f64;
            let a = scene.nearest_hit(origin, DVec3::new(t, -0.1, 1.0).normalize());
            let b = scene.nearest_hit(origin, DVec3::new(-t, -0.1, 1.0).normalize());
            compared += 1;
            match (a, b) {
                (Some((pa, ta)), Some((pb, tb))) => {
                    if pa != pb || (ta - tb).abs() > 0.02 {
                        differed += 1;
                    }
                }
                (None, None) => {}
                _ => differed += 1,
            }
        }
        assert!(
            differed * 4 > compared,
            "only {differed} of {compared} mirrored rays met different geometry — \
             the slanted wall is not breaking the symmetry"
        );
    }

    #[test]
    fn a_render_is_bit_identical_across_calls() {
        let cfg = small();
        let a = SynthDataset::alcove(&cfg);
        let b = SynthDataset::alcove(&cfg);
        for (i, (va, vb)) in a.views.iter().zip(&b.views).enumerate() {
            assert_eq!(va.image.pixels(), vb.image.pixels(), "view {i} differed");
        }
    }

    #[test]
    fn the_render_has_content_and_is_not_a_flat_field() {
        let cfg = small();
        let ds = SynthDataset::alcove(&cfg);
        assert_eq!(ds.views.len(), 3);
        for (i, v) in ds.views.iter().enumerate() {
            let px = v.image.pixels();
            let lo = *px.iter().min().unwrap();
            let hi = *px.iter().max().unwrap();
            assert!(
                hi as i32 - lo as i32 > 150,
                "view {i} has no contrast: {lo}..{hi}"
            );
            // And it is not mostly background: the planes should fill the frame.
            let sky = px.iter().filter(|&&p| p <= 14).count();
            assert!(
                sky * 4 < px.len(),
                "view {i} is {sky} of {} pixels of empty background",
                px.len()
            );
        }
    }

    #[test]
    fn different_stations_render_different_images() {
        let cfg = small();
        let ds = SynthDataset::alcove(&cfg);
        for i in 1..ds.views.len() {
            assert_ne!(
                ds.views[0].image.pixels(),
                ds.views[i].image.pixels(),
                "station {i} rendered the same image as station 0"
            );
        }
    }

    #[test]
    fn the_truth_poses_actually_look_at_the_scene() {
        let cfg = SynthConfig::default();
        let scene = SynthScene::alcove();
        for i in 0..STATIONS.len() {
            let pose = cfg.pose(i);
            let dir = pose.rotation.inverse() * DVec3::Z;
            let hit = scene
                .planes
                .iter()
                .filter_map(|p| p.intersect(pose.center(), dir))
                .map(|(t, _, _)| t)
                .fold(f64::INFINITY, f64::min);
            assert!(
                hit.is_finite() && hit > 0.5,
                "station {i}'s principal ray hits nothing (t = {hit})"
            );
        }
    }

    #[test]
    fn the_dot_field_produces_both_polarities_and_real_edges() {
        let field = DotField {
            seed: 7,
            cell: 0.1,
            background: 200,
            dark: 20,
            light: 250,
            min_radius: 0.2,
            max_radius: 0.35,
        };
        let mut darkest = 255.0f64;
        let mut lightest = 0.0f64;
        for i in 0..400 {
            for j in 0..400 {
                let v = field.sample(i as f64 * 0.005, j as f64 * 0.005);
                darkest = darkest.min(v);
                lightest = lightest.max(v);
            }
        }
        assert!(darkest < 40.0, "no dark dots: darkest sample is {darkest}");
        assert!(
            lightest > 240.0,
            "no light dots: lightest sample is {lightest}"
        );
    }

    #[test]
    fn the_dot_field_never_misses_a_disc_the_four_cell_scan_should_have_found() {
        // Anti-vacuity for the optimisation: sweep finely across a cell boundary
        // and assert the value is continuous — a missed cell shows up as a step.
        let field = DotField {
            seed: 0x1234,
            cell: 0.1,
            background: 200,
            dark: 20,
            light: 250,
            min_radius: 0.2,
            max_radius: 0.45,
        };
        let mut worst_step = 0.0f64;
        let mut prev = field.sample(0.0, 0.0503);
        for i in 1..20_000 {
            let u = i as f64 * 0.00005;
            let v = field.sample(u, 0.0503);
            worst_step = worst_step.max((v - prev).abs());
            prev = v;
        }
        assert!(
            worst_step < 5.0,
            "the texture stepped by {worst_step} — a disc was clipped by the cell scan"
        );
    }
}
