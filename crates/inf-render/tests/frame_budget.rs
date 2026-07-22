//! Render frame-budget smoke (P15.1 / §8 ratchet).
//!
//! Renders a representative scene (a grid of lit cubes + editor grid + sky, the
//! full pass pipeline) headlessly for N frames and asserts the **mean** frame
//! time is under a hard budget. Like the golden harness this **skips** when no
//! GPU adapter is available, and on a **software** adapter (llvmpipe/WARP on CI)
//! it only *smoke-renders* — the strict budget is enforced on real hardware,
//! where the number is meaningful. Per §8 the budget is a tripwire that only
//! ratchets **down**.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderScene, RenderView,
    HEADLESS_FORMAT,
};

/// Hard mean-per-frame budget, in milliseconds (a 30 FPS floor). Real GPUs render
/// this scene in a small fraction of it; the margin absorbs driver/CI variance.
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Lower it as the
/// measured floor drops; never raise it to hide a regression.
const FRAME_BUDGET_MS: f64 = 33.0;

const W: u32 = 640;
const H: u32 = 360;

fn overlook_view() -> RenderView {
    let eye = DVec3::new(14.0, 10.0, 20.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::ZERO - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// A grid of lit cubes — a representative, non-trivial draw load.
fn cube_field(side: i32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let mut id = 1u32;
    for x in -side..side {
        for z in -side..side {
            scene.instances.push(MeshInstance::lit(
                DVec3::new(x as f64 * 1.5, 0.5, z as f64 * 1.5),
                Quat::from_rotation_y(0.3),
                Vec3::ONE,
                [0.6, 0.6, 0.7, 1.0],
                id,
            ));
            id += 1;
        }
    }
    scene.mark_dirty();
    scene
}

#[test]
fn frame_stays_under_budget() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP frame_budget: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    // Software rasterizers (llvmpipe/WARP) and virtualized GPUs (the CI macOS
    // runner reports "Apple Paravirtual device") have non-representative,
    // run-to-run-noisy timing — smoke-only for both. The strict budget is a
    // REAL-hardware gate and still only ratchets down.
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    // ~22x22 = 484 cubes.
    let scene = cube_field(11);
    let view = overlook_view();
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);

    const WARMUP: u32 = 10;
    const MEASURED: u32 = 60;

    for _ in 0..WARMUP {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let start = std::time::Instant::now();
    for _ in 0..MEASURED {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        // Sync each frame so the measured time reflects real GPU+CPU frame cost
        // (not just submission).
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64;

    eprintln!(
        "frame_budget: mean {mean_ms:.3} ms/frame over {MEASURED} frames on {} ({:?}); budget {FRAME_BUDGET_MS} ms{}",
        info.name,
        info.device_type,
        if software { " [software — smoke only]" } else { "" }
    );

    if software {
        // A CPU rasterizer's timing is not representative; we only prove the frame
        // pipeline runs. The strict budget is a hardware gate.
        return;
    }
    assert!(
        mean_ms < FRAME_BUDGET_MS,
        "frame mean {mean_ms:.3} ms exceeded the {FRAME_BUDGET_MS} ms budget on {} \
         (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)",
        info.name
    );
}
