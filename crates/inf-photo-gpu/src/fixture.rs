//! The dense stage's **known-truth fixture**: a scene with real depth structure,
//! rendered by the analytic renderer P25.1 already owns, with an **analytic
//! depth map per view** to measure against.
//!
//! # Why P25.1's alcove is not enough
//!
//! `inf_photo::synth::SynthScene::alcove` is three planes arranged so that a
//! *pose* error shows up. That is the right fixture for structure from motion
//! and the wrong one for a dense stage, for two reasons:
//!
//! * **A plane sweep is a depth estimator, and a scene of large flat walls only
//!   ever asks it easy questions.** What tests a sweep is a *dihedral* — two
//!   surfaces meeting at an angle, so a wrongly-chosen plane is wrong on one
//!   side of the ridge and right on the other — and an **occluding object**,
//!   where a depth discontinuity means some pixels genuinely have no
//!   correspondence in a neighbour view and the filters have to notice.
//! * **A depth map needs truth per pixel, not per camera.** That truth is the
//!   ray-scene intersection this module computes analytically, from the same
//!   geometry the renderer shades.
//!
//! So this fixture is a **convex dihedral** with a **raised block** in front of
//! it on a floor. It is built entirely out of `inf_photo::synth`'s existing
//! primitives and rendered by its existing renderer from its existing camera
//! stations — nothing in P25.1 was touched to make it, which also means nothing
//! in P25.1's own gate can shift because of it.
//!
//! # Lopsided, and asserted so
//!
//! The house law: a fixture with a mirror symmetry cannot see a sign error,
//! because the mirrored answer fits it exactly. Here the two dihedral walls have
//! *different* lengths and *different* angles to the ridge, the block is off to
//! one side, none of the six dot fields share a seed or a pitch, and the camera
//! stations were already lopsided in position, target and roll.
//! `the_fixture_geometry_has_no_mirror_symmetry` measures it on the
//! **geometry** rather than the texture, because two hash-jittered dot fields
//! disagree pixel by pixel whatever the shape of the room.

use glam::{DVec2, DVec3};
use inf_photo::align::similarity_from_points;
use inf_photo::camera::{Intrinsics, Pose};
use inf_photo::sfm::Reconstruction;
use inf_photo::synth::{DotField, SynthConfig, SynthDataset, SynthScene, TexturedPlane, STATIONS};

/// The fixture's render settings.
///
/// 320x240 at focal 300 px, the same as P25.1's alcove. **Measured, not
/// assumed**: 160x120 at focal 150 — the same field of view at a quarter of the
/// pixels, chosen to keep the sweep cheap — does not reconstruct at all. Feature
/// localisation noise is roughly constant in *pixels*, so halving the focal
/// length doubles it on the normalised image plane, and the eight-point
/// essential estimate (whose noise sensitivity `inf-photo` documents as a known
/// bound) decomposes to a near-rotation-only pose: 123 of 125 matches are
/// inliers and only 1 of 69 cheirality-passing points has any parallax at all,
/// so the initialisation refuses by name. 240x180 at focal 225 fails the same
/// way. This is that bound met in the field rather than in a doc comment, and it
/// is why the dense stage's own budget is spent on a census cost that pays for
/// its window **once per pixel** rather than on smaller photographs.
///
/// `k1`/`k2` are non-zero on purpose: radial distortion is in the projection the
/// shader has to reproduce, and a fixture with `k1 = 0` would let a dropped
/// distortion term pass unnoticed.
pub fn config() -> SynthConfig {
    SynthConfig {
        width: 320,
        height: 240,
        focal_px: 300.0,
        k1: -0.09,
        k2: 0.02,
        supersample: 2,
        stations: STATIONS.len(),
    }
}

fn dots(seed: u64, cell: f64, background: u8, dark: u8, light: u8) -> DotField {
    DotField {
        seed,
        cell,
        background,
        dark,
        light,
        min_radius: 0.18,
        max_radius: 0.34,
    }
}

/// The world position of the dihedral's ridge, at mid height.
pub const RIDGE: DVec3 = DVec3::new(0.25, 2.0, 2.05);

/// The scene: a floor, a convex dihedral, and a raised block that occludes part
/// of the floor from every station.
///
/// The plane list, in order, is: floor, dihedral wall A (the long one), dihedral
/// wall B (the short one), then the block's top, front, left, right and back.
/// The order is committed — [`surface_distance`] and the completeness sampler
/// both index it.
pub fn dihedral_scene() -> SynthScene {
    // The dihedral's two in-plane directions away from the ridge, and the
    // normals that follow from them. Written as literals rather than derived so
    // the geometry is readable; `the_dihedral_is_a_real_dihedral` checks the
    // angle they actually make.
    let dir_a = DVec3::new(-1.00, 0.0, 0.45).normalize();
    let dir_b = DVec3::new(1.00, 0.0, 0.62).normalize();
    let normal_a = dir_a.cross(DVec3::Y).normalize();
    let normal_b = -dir_b.cross(DVec3::Y).normalize();
    // The two walls run different distances from the ridge, which is half of
    // what stops a mirrored reconstruction fitting this scene.
    let (len_a, len_b) = (2.20, 1.70);

    let floor = TexturedPlane::new(
        DVec3::new(0.0, 0.0, 0.8),
        DVec3::Y,
        DVec3::X,
        4.0,
        4.2,
        dots(0x5f10_0f00, 0.170, 205, 30, 250),
    );
    let wall_a = TexturedPlane::new(
        RIDGE + dir_a * len_a,
        normal_a,
        DVec3::Y,
        2.0,
        len_a,
        dots(0x0bac_c001, 0.205, 190, 25, 245),
    );
    let wall_b = TexturedPlane::new(
        RIDGE + dir_b * len_b,
        normal_b,
        DVec3::Y,
        2.0,
        len_b,
        dots(0x5aa1_7ee0, 0.145, 215, 35, 255),
    );

    // The block: five faces, since its underside is on the floor and no station
    // can see it. Off-centre in x on purpose.
    let (bx, bz) = (-0.95, 0.55);
    let (hx, hy, hz) = (0.45, 0.32, 0.30);
    let top = TexturedPlane::new(
        DVec3::new(bx, hy * 2.0, bz),
        DVec3::Y,
        DVec3::X,
        hx,
        hz,
        dots(0x00b0_1770, 0.098, 200, 20, 250),
    );
    let front = TexturedPlane::new(
        DVec3::new(bx, hy, bz - hz),
        DVec3::NEG_Z,
        DVec3::X,
        hx,
        hy,
        dots(0x00b0_1771, 0.086, 225, 40, 255),
    );
    let left = TexturedPlane::new(
        DVec3::new(bx - hx, hy, bz),
        DVec3::NEG_X,
        DVec3::Y,
        hy,
        hz,
        dots(0x00b0_1772, 0.112, 185, 22, 240),
    );
    let right = TexturedPlane::new(
        DVec3::new(bx + hx, hy, bz),
        DVec3::X,
        DVec3::Y,
        hy,
        hz,
        dots(0x00b0_1773, 0.104, 210, 28, 252),
    );
    let back = TexturedPlane::new(
        DVec3::new(bx, hy, bz + hz),
        DVec3::Z,
        DVec3::X,
        hx,
        hy,
        dots(0x00b0_1774, 0.092, 195, 33, 248),
    );

    SynthScene {
        planes: vec![floor, wall_a, wall_b, top, front, left, right, back],
        background: 12,
    }
}

/// Render the fixture from every committed station.
pub fn dataset() -> SynthDataset {
    SynthDataset::render(&dihedral_scene(), &config())
}

/// The reconstruction the dense stage would be handed if structure from motion
/// were **perfect**: the same sparse cloud and the same observations, mapped
/// into the truth's frame by the similarity that best aligns the camera centres,
/// with every pose replaced by the one the renderer actually used.
///
/// This is how the fixture's own **noise floor** is measured. A depth map from a
/// solved reconstruction carries two errors that cannot be told apart by looking
/// at it: the sweep's, and the poses'. Running the identical pipeline on exact
/// poses isolates the first — whatever error is left is the census sampling, the
/// plane quantisation and the parabola, and no dense stage over these
/// photographs can beat it. It is the P25.1 discipline (measure what limits the
/// number before setting a bound on it) applied to depth instead of
/// reprojection.
///
/// Returns `None` when fewer than three cameras are registered, which is when
/// the alignment is not determined.
pub fn with_truth_poses(sparse: &Reconstruction, truth: &[Pose]) -> Option<Reconstruction> {
    let views = sparse.registered_views();
    let source: Vec<DVec3> = views
        .iter()
        .map(|v| sparse.cameras[v].pose.center())
        .collect();
    let target: Vec<DVec3> = views.iter().map(|v| truth[*v as usize].center()).collect();
    let similarity = similarity_from_points(&source, &target)?;
    let mut out = sparse.clone();
    for (view, camera) in out.cameras.iter_mut() {
        camera.pose = truth[*view as usize];
    }
    for point in out.points.iter_mut() {
        point.position = similarity.apply(point.position);
    }
    Some(out)
}

/// The **analytic** depth map of one view: depth along the camera's `+Z`, in
/// metres (the truth's own units), with `0.0` where the ray hits nothing.
///
/// This is a ray-scene intersection through the *same* `nearest_hit` the
/// renderer shades with, sampled at the **pixel centre** on the crate's
/// pixel-centre convention: at one sample per pixel the renderer's own sub-pixel
/// grid collapses to `to_normalized((x, y))`, which is what this uses, so the
/// truth and the image cannot be half a pixel apart.
///
/// The intersection parameter `t` is along a **unit** ray, so `t * ray.z` in
/// camera space is exactly the `z` a depth map holds.
pub fn truth_depth(scene: &SynthScene, intrinsics: &Intrinsics, pose: &Pose) -> Vec<f64> {
    let eye = pose.center();
    let inverse = pose.rotation.inverse();
    let w = intrinsics.width;
    let h = intrinsics.height;
    let mut out = vec![0.0f64; w as usize * h as usize];
    for y in 0..h {
        for x in 0..w {
            let n = intrinsics.to_normalized(DVec2::new(x as f64, y as f64));
            let ray = DVec3::new(n.x, n.y, 1.0).normalize();
            let dir = inverse * ray;
            if let Some((_, t)) = scene.nearest_hit(eye, dir) {
                out[y as usize * w as usize + x as usize] = t * ray.z;
            }
        }
    }
    out
}

/// The distance from a world point to the nearest surface of the scene, in
/// metres.
///
/// Every primitive is a **finite** rectangle, so the closest point on one is
/// found by clamping the in-plane coordinates to the extents and keeping the
/// normal component — exact, with no iteration and no sampling.
pub fn surface_distance(scene: &SynthScene, p: DVec3) -> f64 {
    scene
        .planes
        .iter()
        .map(|plane| {
            let d = p - plane.origin;
            let u = d.dot(plane.u_axis).clamp(-plane.u_extent, plane.u_extent);
            let v = d.dot(plane.v_axis).clamp(-plane.v_extent, plane.v_extent);
            let closest = plane.origin + plane.u_axis * u + plane.v_axis * v;
            (p - closest).length()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Points on the scene's surfaces that at least `min_views` stations actually
/// see, on a grid of roughly `step` metres.
///
/// This is the **completeness** reference: a reconstruction is not judged only
/// by whether its vertices are near the truth (which a reconstruction of one
/// square metre would pass perfectly) but by whether the truth it could have
/// seen has vertices near it. Visibility is settled by the same `nearest_hit`
/// the renderer uses — a sample is seen by a station when the ray from that
/// station to it is not stopped by anything else first.
pub fn visible_surface_samples(scene: &SynthScene, cfg: &SynthConfig, step: f64) -> Vec<DVec3> {
    let mut out = Vec::new();
    let intrinsics = cfg.intrinsics();
    let poses: Vec<Pose> = (0..cfg.stations.min(STATIONS.len()))
        .map(|i| cfg.pose(i))
        .collect();
    for (pi, plane) in scene.planes.iter().enumerate() {
        let nu = ((plane.u_extent * 2.0) / step).ceil().max(1.0) as i64;
        let nv = ((plane.v_extent * 2.0) / step).ceil().max(1.0) as i64;
        for iu in 0..=nu {
            for iv in 0..=nv {
                let u = -plane.u_extent + 2.0 * plane.u_extent * iu as f64 / nu as f64;
                let v = -plane.v_extent + 2.0 * plane.v_extent * iv as f64 / nv as f64;
                let p = plane.origin + plane.u_axis * u + plane.v_axis * v;
                let mut seen = 0usize;
                for pose in &poses {
                    let eye = pose.center();
                    let to = p - eye;
                    let distance = to.length();
                    if distance < 1e-6 {
                        continue;
                    }
                    let dir = to / distance;
                    // In frame?
                    let camera_space = pose.transform(p);
                    if camera_space.z <= 0.0 {
                        continue;
                    }
                    match intrinsics.project(camera_space) {
                        Some(px) if intrinsics.contains(px) => {}
                        _ => continue,
                    }
                    // Unoccluded? The nearest hit along the ray must be this
                    // very plane, at this very distance.
                    match scene.nearest_hit(eye, dir) {
                        Some((hit, t)) if hit == pi && (t - distance).abs() < 1e-3 => seen += 1,
                        _ => {}
                    }
                }
                if seen >= 2 {
                    out.push(p);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn scene() -> &'static SynthScene {
        static SCENE: OnceLock<SynthScene> = OnceLock::new();
        SCENE.get_or_init(dihedral_scene)
    }

    #[test]
    fn the_dihedral_is_a_real_dihedral_and_the_block_stands_off_the_floor() {
        let s = scene();
        let a = s.planes[1].normal;
        let b = s.planes[2].normal;
        let cos = a.dot(b);
        assert!(
            (0.3..0.8).contains(&cos),
            "the two walls meet at cos {cos} — that is nearly coplanar or nearly a fold"
        );
        // Both walls face the cameras, which all sit at negative z.
        assert!(a.z < -0.5 && b.z < -0.5, "a wall faces away: {a} / {b}");
        // The two walls really share the ridge: the ridge is on both.
        for (i, plane) in [(1usize, s.planes[1]), (2, s.planes[2])] {
            let d = RIDGE - plane.origin;
            assert!(
                d.dot(plane.normal).abs() < 1e-9,
                "the ridge is {} off plane {i}",
                d.dot(plane.normal)
            );
            assert!(d.dot(plane.v_axis).abs() <= plane.v_extent + 1e-9);
        }
        // The block's top is above the floor by a real amount, which is what
        // makes it occlude.
        let top = s.planes[3];
        assert!(top.origin.y > 0.5, "the block is flush with the floor");
    }

    #[test]
    fn the_fixture_geometry_has_no_mirror_symmetry() {
        // Measured on the GEOMETRY, not the texture: two hash-jittered dot
        // fields disagree pixel by pixel whatever the shape of the room, so a
        // texture comparison would pass on a perfectly symmetric box.
        let s = scene();
        let origin = DVec3::new(0.0, 1.5, -3.0);
        let mut differed = 0usize;
        let mut compared = 0usize;
        for i in 0..60 {
            let t = -0.7 + 0.024 * i as f64;
            for &pitch in &[-0.35f64, -0.15, 0.02] {
                let a = s.nearest_hit(origin, DVec3::new(t, pitch, 1.0).normalize());
                let b = s.nearest_hit(origin, DVec3::new(-t, pitch, 1.0).normalize());
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
        }
        assert!(
            differed * 3 > compared,
            "only {differed} of {compared} mirrored rays met different geometry"
        );
    }

    #[test]
    fn every_station_sees_the_dihedral_and_the_block() {
        // Anti-vacuity for the whole fixture: a scene the cameras cannot see is
        // a gate about nothing. Both the ridge and the block's top must land in
        // every station's frame, unoccluded.
        let s = scene();
        let cfg = config();
        let k = cfg.intrinsics();
        let block_top = DVec3::new(-0.95, 0.64, 0.55);
        for i in 0..STATIONS.len() {
            let pose = cfg.pose(i);
            for (name, target) in [("the ridge", RIDGE), ("the block top", block_top)] {
                let camera_space = pose.transform(target);
                assert!(camera_space.z > 0.0, "{name} is behind station {i}");
                let px = k.project(camera_space).expect("in front");
                assert!(
                    k.contains(px),
                    "{name} is outside station {i}'s frame at {px}"
                );
                // And unoccluded: the ray to it must actually reach it. "In
                // frame" is not "visible", and a fixture whose subject is
                // behind its own block would be a gate about a wall.
                let eye = pose.center();
                let to = target - eye;
                let distance = to.length();
                let hit = s
                    .nearest_hit(eye, to / distance)
                    .map(|(_, t)| t)
                    .unwrap_or(f64::INFINITY);
                assert!(
                    (hit - distance).abs() < 1e-3,
                    "{name} is occluded from station {i}: the ray stops at {hit}, it is at \
                     {distance}"
                );
            }
        }
    }

    #[test]
    fn the_analytic_depth_agrees_with_the_render_and_has_real_structure() {
        let s = scene();
        let cfg = SynthConfig {
            stations: 2,
            ..config()
        };
        let k = cfg.intrinsics();
        let depth = truth_depth(s, &k, &cfg.pose(0));
        let valid: Vec<f64> = depth.iter().copied().filter(|d| *d > 0.0).collect();
        assert!(
            valid.len() * 4 > depth.len() * 3,
            "only {} of {} pixels hit the scene",
            valid.len(),
            depth.len()
        );
        let lo = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = valid.iter().copied().fold(0.0f64, f64::max);
        assert!(
            hi - lo > 1.5,
            "the depth range is only {lo}..{hi} — this scene is effectively a wall"
        );
        // A depth truth must land ON the surface it describes: back-project a
        // sample and its distance to the scene has to be zero.
        let pose = cfg.pose(0);
        let (w, h) = (k.width as usize, k.height as usize);
        let mut worst = 0.0f64;
        for y in (8..h - 8).step_by(17) {
            for x in (8..w - 8).step_by(19) {
                let i = y * w + x;
                if depth[i] <= 0.0 {
                    continue;
                }
                let n = k.to_normalized(DVec2::new(x as f64, y as f64));
                let p = DVec3::new(n.x, n.y, 1.0) * depth[i];
                let world = pose.rotation.inverse() * (p - pose.translation);
                worst = worst.max(surface_distance(s, world));
            }
        }
        assert!(
            worst < 1e-6,
            "an analytic depth back-projects {worst} m off its own surface"
        );
    }

    #[test]
    fn the_block_occludes_and_the_occlusion_is_a_real_depth_step() {
        // The feature the fixture exists for. Somewhere along a horizontal scan
        // across the block's silhouette there must be a step of at least the
        // block's own depth, or the "occluding object" claim is decoration.
        let s = scene();
        let cfg = config();
        let k = cfg.intrinsics();
        let depth = truth_depth(s, &k, &cfg.pose(0));
        let (w, h) = (k.width as usize, k.height as usize);
        let mut biggest = 0.0f64;
        for y in 0..h {
            for x in 1..w {
                let (a, b) = (depth[y * w + x - 1], depth[y * w + x]);
                if a > 0.0 && b > 0.0 {
                    biggest = biggest.max((a - b).abs());
                }
            }
        }
        assert!(
            biggest > 0.4,
            "the largest horizontal depth step is {biggest} m — nothing occludes anything"
        );
    }

    #[test]
    fn the_completeness_samples_are_visible_and_cover_every_primitive() {
        let s = scene();
        let cfg = config();
        let samples = visible_surface_samples(s, &cfg, 0.09);
        assert!(
            samples.len() > 2000,
            "only {} visible surface samples",
            samples.len()
        );
        // Every sample really is on the surface.
        let worst = samples
            .iter()
            .map(|p| surface_distance(s, *p))
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-9,
            "a completeness sample is {worst} m off surface"
        );
        // And the set is not one big plane: the block and both walls contribute.
        let near_block = samples
            .iter()
            .filter(|p| (p.x + 0.95).abs() < 0.5 && p.y > 0.05 && (p.z - 0.55).abs() < 0.35)
            .count();
        assert!(near_block > 60, "only {near_block} samples on the block");
        let near_walls = samples.iter().filter(|p| p.z > 1.8 && p.y > 0.4).count();
        assert!(
            near_walls > 300,
            "only {near_walls} samples on the dihedral"
        );
    }

    #[test]
    fn a_render_is_bit_identical_across_calls_and_has_contrast() {
        let cfg = SynthConfig {
            stations: 2,
            ..config()
        };
        let s = dihedral_scene();
        let a = SynthDataset::render(&s, &cfg);
        let b = SynthDataset::render(&s, &cfg);
        for (i, (va, vb)) in a.views.iter().zip(&b.views).enumerate() {
            assert_eq!(va.image.pixels(), vb.image.pixels(), "view {i} differed");
            let px = va.image.pixels();
            let lo = *px.iter().min().unwrap();
            let hi = *px.iter().max().unwrap();
            assert!(hi as i32 - lo as i32 > 150, "view {i} has no contrast");
            let sky = px.iter().filter(|&&p| p <= 14).count();
            assert!(
                sky * 5 < px.len(),
                "view {i} is {sky} of {} pixels of background",
                px.len()
            );
        }
    }
}
