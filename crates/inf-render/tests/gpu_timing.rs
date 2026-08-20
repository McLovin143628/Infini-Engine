//! **The per-pass GPU clock, proven on a device** (island wave I4).
//!
//! `crates/inf-render/src/timing.rs` is the instrument's renderer half; this file
//! is what says it works. Three claims, and the third is the one the 54 frozen
//! goldens depend on:
//!
//! 1. every graph node the renderer built appears in the report, in run order,
//!    together with the four out-of-graph segments;
//! 2. the segments **tile** the frame — their sum is the frame's own first-to-last
//!    span, so a per-pass breakdown can be read as a budget rather than as a set
//!    of unrelated numbers;
//! 3. attaching the instrument does not change a single pixel.
//!
//! Claim 3 is asserted against the rendered image rather than against the command
//! stream, because "the commands are identical" is a sentence and "the frame is
//! byte-identical" is a comparison.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderScene, RenderView,
    HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;

fn view() -> RenderView {
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

fn cubes() -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let mut id = 1u32;
    for x in -4..4 {
        for z in -4..4 {
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

/// The report names every pass the renderer built, in the order it ran them, and
/// the segments sum to the frame.
///
/// The name list is compared against `EngineRenderer::pass_names` rather than
/// against a literal, because a literal would be a second declaration of the pass
/// order and this tree has paid for those. The four out-of-graph segments ARE
/// literal, because they are hand-bracketed in `render` and a gate that read them
/// from the same place they are written could not see them go missing.
#[test]
fn the_report_names_every_pass_and_the_segments_tile_the_frame() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gpu_timing: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let scene = cubes();
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);

    if !r.set_gpu_timing(&gpu, true) {
        eprintln!(
            "SKIP gpu_timing: {} ({:?}) cannot time an encoder segment",
            info.name, info.device_type
        );
        // The refusal itself is asserted: a device that says no must hand back
        // nothing rather than a report of zeros a caller would print as a
        // measurement.
        r.render(&gpu, &scene, &view(), &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        assert!(
            r.gpu_timings(&gpu).is_none(),
            "a device without timestamp queries must report NO timings, not zeros"
        );
        return;
    }

    // Warm once: the first frame builds pipelines and its segment times are a
    // statement about shader compilation.
    r.render(&gpu, &scene, &view(), &target.view, (W, H));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let _ = r.gpu_timings(&gpu);

    r.render(&gpu, &scene, &view(), &target.view, (W, H));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let t = r.gpu_timings(&gpu).expect("a timed frame reports timings");

    let names: Vec<&str> = t.passes.iter().map(|p| p.name).collect();
    let mut expected: Vec<&str> = vec!["vt-stream", "vsm-raster"];
    expected.extend(r.pass_names());
    expected.extend(["vt-feedback", "vsm-mark"]);
    assert_eq!(
        names, expected,
        "the report must name every segment the frame recorded, in order"
    );

    let sum: f64 = t.passes.iter().map(|p| p.ms).sum();
    // Exactly, not approximately: the segments are consecutive differences of one
    // monotone tick sequence, so their sum IS the first-to-last span. Compared
    // with a tolerance only because the sum is accumulated in a different order
    // than the subtraction that produced `total_ms`.
    assert!(
        (sum - t.total_ms).abs() < 1.0e-9,
        "the segments do not tile the frame: {sum:.6} ms of parts against a \
         {:.6} ms whole",
        t.total_ms
    );
    assert!(
        t.total_ms > 0.0,
        "a frame that drew {} instances took zero GPU time — the query set \
         resolved nothing",
        scene.instances.len()
    );

    let dearest = t.by_cost();
    eprintln!(
        "gpu_timing on {} ({:?}): frame {:.3} ms over {} segments; dearest: {}",
        info.name,
        info.device_type,
        t.total_ms,
        t.passes.len(),
        dearest
            .iter()
            .take(5)
            .map(|(n, ms)| format!("{n} {ms:.3} ms"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// **Attaching the instrument does not move a pixel** — the claim the frozen
/// goldens rest on.
///
/// `set_gpu_timing(false)` is the shipped configuration and the mark call sites
/// are a `None` check there; with it on, five extra `write_timestamp`s, a
/// `resolve_query_set` and a `copy_buffer_to_buffer` join the same encoder. None
/// of them touches a render target, and this is where that stops being an
/// argument and becomes a comparison: the same scene, the same view, rendered by
/// two renderers that differ only in the flag, read back and compared byte for
/// byte.
#[test]
fn timing_changes_no_pixel() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gpu_timing pixels: no GPU adapter");
        return;
    };
    let scene = cubes();
    let v = view();

    let shoot = |timed: bool| -> Vec<u8> {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        let live = r.set_gpu_timing(&gpu, timed);
        assert_eq!(live, timed && gpu.supports_timestamp_query());
        for _ in 0..3 {
            r.render(&gpu, &scene, &v, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = r.gpu_timings(&gpu);
        }
        target.read_rgba(&gpu).expect("read back the frame")
    };

    let plain = shoot(false);
    let timed = shoot(true);
    assert_eq!(
        plain.len(),
        timed.len(),
        "the two frames are not the same size"
    );
    let differing = plain.iter().zip(&timed).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} bytes moved when the GPU stopwatch was attached — a \
         golden would move with them",
        plain.len()
    );
}
