//! **TAA under a MOVING camera** (audit FIX1).
//!
//! `taa_multiframe_stable` — the only other TAA arm in this repository — renders
//! twelve frames from a `view` it never changes. With a static camera the history
//! and the current frame hold the same content, so the reprojection cannot be
//! wrong and the arm converges whatever the reprojection does. **A moving camera
//! is the only condition under which TAA's reprojection is exercised at all**,
//! and nothing here moved one, which is why the frame the author reported —
//! *"washed out and heavily ghosted"* — had no gate anywhere that could have
//! caught it.
//!
//! # The measure
//!
//! Not mean luminance: a wrong history only shows there once it has DILATED,
//! which takes seconds. The measure is how far the TAA-**resolved** frame is from
//! the **un-resolved** frame at the same final camera pose. TAA is an
//! anti-aliaser; a correct resolve differs from its source only by the sub-pixel
//! detail it removed, whether or not the camera moved. So the quantity that
//! matters is the RATIO of that distance moving to that distance static, and the
//! control is the same street built from geometry the depth prepass does cover.
//!
//! # What it caught
//!
//! `taa.wgsl` fell through with `hist_uv = in.uv` — an identity reprojection —
//! for every pixel the depth prepass does not cover, which since VIS-C1b is every
//! meshlet and every scattered instance: on the showcase island, the buildings
//! and the vegetation. Measured here at the head that shipped it:
//!
//! | | static | moving | growth |
//! |---|---|---|---|
//! | control, rigid meshes (write the prepass) | 0.0021 | 0.0016 | **x0.73** |
//! | subject, meshlets (write nothing) | 0.0012 | 0.0051 | **x4.30** |
//!
//! and after the shader refuses a history it cannot locate, **x1.15** — the
//! subject behaves like the control, and the control does not move.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::golden::image_diff;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;

const W: u32 = 320;
const H: u32 = 180;
const ASSET: u128 = 0xF1F1_0000_0000_0001;
/// Long enough for a wrong history to compound, short enough to stay a smoke.
const FRAMES: usize = 180;
/// The hero's own `Run` gait, 3.75 m/s at 60 Hz.
const STEP_M: f64 = 0.0625;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP taa_motion: no GPU adapter ({e})");
            None
        }
    }
}

fn sun(scene: &mut RenderScene) {
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
}

fn floor(scene: &mut RenderScene) {
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 1.0, 400.0),
        [0.15, 0.22, 0.12, 1.0],
        1,
    ));
}

/// Where the blocks stand: a 14 m-wide street the camera drives down, so the
/// frame is a STREET and not a horizon. The first cut of this instrument put them
/// 60 m ahead of the camera and measured two specks against the sky — a vacuous
/// check, and it reported no defect on a head that has one.
fn block_at(k: usize) -> (f64, f64) {
    (70.0 - k as f64 * 14.0, 7.0)
}

/// The blocks either side as rigid `MeshInstance`s — geometry the depth prepass
/// DOES cover. The control.
fn street_prims() -> RenderScene {
    let mut scene = RenderScene::default();
    floor(&mut scene);
    let mut id = 2;
    for k in 0..24 {
        let (z, half) = block_at(k);
        for side in [-half, half] {
            scene.instances.push(MeshInstance::lit(
                DVec3::new(side, 6.0, z),
                Quat::IDENTITY,
                Vec3::new(6.0, 14.0, 10.0),
                [0.80, 0.82, 0.78, 1.0],
                id,
            ));
            id += 1;
        }
    }
    sun(&mut scene);
    scene.mark_dirty();
    scene
}

/// The same street with the blocks drawn as MESHLETS — the path the island's
/// buildings take, and the one that writes no prepass depth (VIS-C1b). The
/// subject.
fn street_vgeom(mesh: &Arc<inf_render::VgeomMesh>) -> RenderScene {
    let mut scene = RenderScene {
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, mesh).expect("index the vmesh")],
        ..Default::default()
    };
    floor(&mut scene);
    let mut id = 2;
    for k in 0..24 {
        let (z, half) = block_at(k);
        for side in [-half, half] {
            scene.vgeom_instances.push(VgeomInstance::lit(
                ASSET,
                DVec3::new(side, 6.0, z),
                Quat::IDENTITY,
                Vec3::splat(3.5),
                [0.80, 0.82, 0.78, 1.0],
                id,
            ));
            id += 1;
        }
    }
    sun(&mut scene);
    scene.mark_dirty();
    scene
}

fn view_at(z: f64) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 1.6, z),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Mean luminance, and the fraction of pixels over 240 — the two numbers the
/// author's own screenshots were scored with, printed so a reader can line this
/// up against them.
fn stats(rgba: &[u8]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut hot = 0usize;
    let n = rgba.len() / 4;
    for p in rgba.chunks(4) {
        let l = 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]);
        sum += l;
        if l > 240.0 {
            hot += 1;
        }
    }
    (sum / n as f64, hot as f64 / n as f64)
}

/// Render [`FRAMES`] frames on ONE renderer (so the history accumulates) and
/// return the last one.
fn run(gpu: &GpuContext, scene: &RenderScene, taa: bool, vgeom: bool, moving: bool) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(RenderSettings {
        taa,
        vgeom: VgeomSettings {
            enabled: vgeom,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    });
    let mut last = Vec::new();
    for f in 0..FRAMES {
        let z = 60.0 - if moving { STEP_M * f as f64 } else { 0.0 };
        renderer.render(gpu, scene, &view_at(z), &target.view, (W, H));
        last = target.read_rgba(gpu).expect("readback");
    }
    last
}

/// How far moving the camera may push TAA's resolve away from the frame it is
/// built from, as a multiple of the same distance with the camera still. `1.0` is
/// "the camera's motion cost the resolve nothing"; the defect this file was
/// written for measured **4.30**.
const MOTION_GROWTH_CEILING: f64 = 2.5;

#[test]
fn taa_under_a_moving_camera_resolves_to_the_frame_it_is_built_from() {
    let Some(gpu) = gpu_or_skip() else { return };
    let prims = street_prims();
    let mesh = Arc::new(dense_grid_mesh(40));
    let vg = street_vgeom(&mesh);

    let row = |label: &str, scene: &RenderScene, vgeom: bool, moving: bool| {
        let off = run(&gpu, scene, false, vgeom, moving);
        let on = run(&gpu, scene, true, vgeom, moving);
        let (mean, max) = image_diff(&off, &on, W, H);
        let (lum_off, hot_off) = stats(&off);
        let (lum_on, hot_on) = stats(&on);
        println!(
            "  {label:32} diff mean {mean:.4} max {max:.4} | lum {lum_off:7.3} -> {lum_on:7.3} \
             | hot {hot_off:.4} -> {hot_on:.4}"
        );
        f64::from(mean)
    };
    println!("FIX1 audit — TAA under motion, {FRAMES} frames, {STEP_M} m a frame");
    let prim_static = row("CONTROL rigid,   camera STATIC", &prims, false, false);
    let prim_moving = row("CONTROL rigid,   camera MOVING", &prims, false, true);
    let vgeom_static = row("SUBJECT meshlet, camera STATIC", &vg, true, false);
    let vgeom_moving = row("SUBJECT meshlet, camera MOVING", &vg, true, true);

    let growth = |still: f64, moving: f64| moving / still.max(1.0e-6);
    let control = growth(prim_static, prim_moving);
    let subject = growth(vgeom_static, vgeom_moving);
    println!("  motion growth: control x{control:.2}  subject x{subject:.2}");

    // The control first, because a subject bound the control also fails is a
    // statement about this fixture and not about the prepass.
    assert!(
        control < MOTION_GROWTH_CEILING,
        "the RIGID-mesh control degrades x{control:.2} under motion \
         ({prim_static:.4} -> {prim_moving:.4}); this fixture is measuring \
         something other than the reprojection"
    );
    assert!(
        subject < MOTION_GROWTH_CEILING,
        "TAA's resolve of the MESHLET street is x{subject:.2} further from the \
         frame it is built from once the camera moves ({vgeom_static:.4} -> \
         {vgeom_moving:.4}), against x{control:.2} for the rigid-mesh control \
         ({prim_static:.4} -> {prim_moving:.4}). Meshlets write no depth prepass \
         (VIS-C1b), so a history the resolve cannot locate is being blended in \
         anyway — see the no-depth branch in `shaders/taa.wgsl`."
    );
}
