//! End-to-end headless smoke: bring up a GPU context (skipping gracefully on
//! CI runners without any adapter), render the standard passes into an
//! offscreen target, and sanity-check the pixels. Golden-image comparisons
//! live in `golden.rs` (P2.5); this test only asserts structure.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderScene, RenderView,
    HEADLESS_FORMAT,
};

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

#[test]
fn renders_sky_grid_and_cube() {
    let Some(gpu) = gpu_or_skip() else { return };

    const W: u32 = 320;
    const H: u32 = 180;
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);

    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::ONE,
        [0.8, 0.1, 0.1, 1.0],
        1,
    ));
    scene.mark_dirty();

    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 1.5, 4.0),
        forward: Vec3::new(0.0, -0.2, -1.0).normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
    };

    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let rgba = target.read_rgba(&gpu).expect("readback");
    assert_eq!(rgba.len(), (W * H * 4) as usize);

    // Top row: sky — not black, blue-ish channel dominates red.
    let sky = pixel(&rgba, W, W / 2, 2);
    assert!(sky[2] > 0, "sky should not be pure black: {sky:?}");
    assert!(sky[2] >= sky[0], "sky should lean blue: {sky:?}");

    // Center: the red cube fills the middle of the frame.
    let cube = pixel(&rgba, W, W / 2, H / 2);
    assert!(
        cube[0] > cube[2] && cube[0] > 40,
        "expected the red cube at center: {cube:?}"
    );

    // The frame must not be uniform (grid/sky/cube variety).
    let corner = pixel(&rgba, W, 4, H - 4);
    assert_ne!(corner, cube);
}

#[test]
fn resize_recreates_targets_without_validation_errors() {
    let Some(gpu) = gpu_or_skip() else { return };

    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (w, h) in [(64, 64), (200, 120), (64, 64)] {
        let target = HeadlessTarget::new(&gpu, w, h);
        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 2.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: w,
            height: h,
        };
        renderer.render(&gpu, &scene, &view, &target.view, (w, h));
        let rgba = target.read_rgba(&gpu).expect("readback");
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }
}
