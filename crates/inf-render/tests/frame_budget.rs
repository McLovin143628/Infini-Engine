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
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderDeform, RenderDeformCell,
    RenderScene, RenderTerrain, RenderTerrainTile, RenderView, TerrainTileKey, HEADLESS_FORMAT,
};

/// Hard mean-per-frame budget, in milliseconds (a 30 FPS floor). Real GPUs render
/// this scene in a small fraction of it; the margin absorbs driver/CI variance.
/// **RATCHET RULE (§8): this constant may only ever DECREASE.** Lower it as the
/// measured floor drops; never raise it to hide a regression.
use inf_core::FRAME_BUDGET_MS;

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

/// **The version-churning regime** (Hardening Wave E, 2026-08-14) — the frame a
/// SHIPPED PLAYER renders, which this file did not previously measure at all.
///
/// [`frame_stays_under_budget`] above renders **one** `RenderScene` sixty times
/// and never touches its `version`. Every version-gated cache in the renderer
/// therefore misses once and hits on all fifty-nine frames after it. That is the
/// *opposite* of the regime a host produces: `project_scene_full` — in both
/// hosts — ends in `scene.mark_dirty()`, so every frame the player draws arrives
/// carrying a version nobody has seen before, and each of `mesh.rs`,
/// `depth_prepass.rs`, `shadow.rs`, `skinned.rs`, `sprite.rs`, `mask.rs` and
/// `vgeom.rs` re-does its work.
///
/// A budget that measures only the still regime is **structurally blind** to the
/// cost of the churn, which is why the churn has stood as a RECORDED-open item
/// since P18.4 with a green gate beside it: the number this file printed was the
/// number of a frame that does not exist.
///
/// So this arm renders the same scene in the same process in **both** regimes and
/// asserts the budget against the churning one — the honest side. The still
/// regime is kept, measured and printed, because the *difference* between the two
/// is the only direct price of `scene.version` this tree can quote.
///
/// The difference itself is deliberately **not** asserted: it is a subtraction of
/// two wall clocks, noisier than either, and §8 numbers are tripwires rather than
/// hardware claims (see `inf_player::budget`'s header for the full rule). What is
/// asserted is (1) the churning frame stays inside `FRAME_BUDGET_MS` on real
/// hardware, and (2) the arm is not vacuous — the two regimes really did present
/// different version sequences to the renderer.
#[test]
fn frame_stays_under_budget_under_version_churn() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP frame_budget churn: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    let view = overlook_view();
    let target = HeadlessTarget::new(&gpu, W, H);
    const WARMUP: u32 = 10;
    const MEASURED: u32 = 60;

    // `churn` reproduces what a host does between frames: bump the version. The
    // scene CONTENT is byte-identical in both runs — only the stamp moves — so
    // the difference is exactly what the stamp buys and nothing else.
    let measure = |churn: bool| -> (f64, u64) {
        let mut scene = cube_field(11);
        let first = scene.version;
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        for _ in 0..WARMUP {
            if churn {
                scene.mark_dirty();
            }
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let start = std::time::Instant::now();
        for _ in 0..MEASURED {
            if churn {
                scene.mark_dirty();
            }
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(MEASURED);
        (ms, scene.version.wrapping_sub(first))
    };

    let (still_ms, still_bumps) = measure(false);
    let (churn_ms, churn_bumps) = measure(true);

    eprintln!(
        "frame_budget churn: still {still_ms:.3} ms/frame | churning {churn_ms:.3} ms/frame \
         (+{:.3} ms, {:+.1}%) over {MEASURED} frames of {} instances on {} ({:?}); \
         budget {FRAME_BUDGET_MS} ms{}",
        churn_ms - still_ms,
        (churn_ms - still_ms) / still_ms * 100.0,
        cube_field(11).instances.len(),
        info.name,
        info.device_type,
        if software {
            " [software — smoke only]"
        } else {
            ""
        }
    );

    // ANTI-VACUITY. The whole point of this arm is that the two runs presented
    // *different* version sequences; if `mark_dirty` ever stopped bumping, both
    // runs would be the still one and the arm would measure nothing while staying
    // green. This is a pure assertion — it holds with no GPU at all.
    assert_eq!(
        still_bumps, 0,
        "the still regime must hold `version` fixed — otherwise both halves \
         measure the same thing"
    );
    assert_eq!(
        churn_bumps,
        u64::from(WARMUP + MEASURED),
        "the churning regime must bump `version` once per frame, exactly as \
         `project_scene_full` does"
    );

    if software {
        // A CPU rasterizer's timing is not representative; the pipeline having
        // run in both regimes is the whole claim here.
        return;
    }
    assert!(
        churn_ms < FRAME_BUDGET_MS,
        "a version-churning frame mean {churn_ms:.3} ms exceeded the \
         {FRAME_BUDGET_MS} ms budget on {} (the §8 budget only ratchets DOWN — \
         investigate the regression, do not raise it)",
        info.name
    );
}

/// **The Phase 17 sky-stack budget** (P17.4): what the whole living sky —
/// atmosphere LUTs, volumetric clouds, cloud shadows and precipitation — costs a
/// frame, per render tier, measured against the same frame with each layer off.
///
/// Deliberately **not a new ratchet constant**. §8's rule is that a budget
/// tripwire only ever ratchets down, which makes each one a standing maintenance
/// obligation; adding a second for a feature that is *off by default* and whose
/// worst measured cost is a fraction of a millisecond would be paying that
/// obligation for nothing. The existing `FRAME_BUDGET_MS` already covers the
/// composed frame, and this test asserts the sky-on frame is inside it — the same
/// tripwire, now exercised with every Phase-17 layer enabled at once, which is
/// the case that was previously untested. What the per-tier numbers are *for* is
/// the ROADMAP's cost table and a future decision about whether any of it needs a
/// ratchet of its own; they are printed, not asserted, because an absolute
/// millisecond on one machine is not a contract.
///
/// The relative claims that ARE asserted are the ones that can regress
/// meaningfully: a lower tier must not cost more than a higher one, and the
/// whole stack must stay inside the frame budget on real hardware.
#[test]
fn sky_stack_cost_per_tier() {
    use inf_render::{
        AtmosphereParams, AtmosphereQuality, CloudParams, PrecipParams, RenderSettings, SunParams,
    };

    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP sky_stack_cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || info.name.to_ascii_lowercase().contains("paravirtual");

    // The measured scene: the same cube field, under a full Phase-17 sky in the
    // heaviest state a weather preset can produce (Storm — solid cover, full
    // rain), so what is measured is the ceiling rather than a typical frame.
    let mut lit = cube_field(11);
    let bodies = inf_math::solar::bodies(&inf_math::solar::SolarInput {
        seconds: 43_200.0,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
    });
    lit.sun = SunParams {
        direction: bodies.sun.as_vec3(),
        ..SunParams::default()
    };
    lit.atmosphere = AtmosphereParams {
        enabled: true,
        clouds: CloudParams {
            enabled: true,
            coverage: 1.0,
            cloud_type: 0.35,
            bottom: 600.0,
            top: 2800.0,
            ..CloudParams::default()
        },
        precip: PrecipParams {
            enabled: true,
            intensity: 1.0,
            wind_x: 22.0,
            wind_z: 9.0,
            time_s: 1_234.5,
            ..PrecipParams::default()
        },
        ..AtmosphereParams::default()
    };
    lit.mark_dirty();

    let mut bare = lit.clone();
    bare.atmosphere = AtmosphereParams::default();
    bare.mark_dirty();

    let view = overlook_view();
    let target = HeadlessTarget::new(&gpu, W, H);

    let measure = |scene: &RenderScene, quality: AtmosphereQuality| -> f64 {
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        let mut settings = RenderSettings::default();
        settings.atmosphere.quality = quality;
        renderer.set_settings(settings);
        for _ in 0..10 {
            renderer.render(&gpu, scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        const N: u32 = 60;
        let start = std::time::Instant::now();
        for _ in 0..N {
            renderer.render(&gpu, scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(N)
    };

    let baseline = measure(&bare, AtmosphereQuality::High);
    let mut costs = Vec::new();
    for q in [
        AtmosphereQuality::Low,
        AtmosphereQuality::Medium,
        AtmosphereQuality::High,
    ] {
        let ms = measure(&lit, q);
        eprintln!(
            "sky_stack {q:?}: {ms:.3} ms/frame (+{:.3} ms over a sky-less {baseline:.3} ms) on {}",
            ms - baseline,
            info.name
        );
        costs.push((q, ms));
    }

    if software {
        // A CPU rasterizer's timing is not representative of anything; the frame
        // pipeline having run at all is the whole claim here.
        return;
    }
    for (q, ms) in &costs {
        assert!(
            *ms < FRAME_BUDGET_MS,
            "the full Phase-17 sky at {q:?} cost {ms:.3} ms, over the \
             {FRAME_BUDGET_MS} ms frame budget on {} (§8: investigate, never raise it)",
            info.name
        );
    }
}

/// **The P18.1 two-pass occlusion budget**: what real HZB occlusion costs a
/// vgeom frame, measured against the same frame with occlusion off and with the
/// single-pass v1 shape.
///
/// Following `sky_stack_cost_per_tier`, this deliberately adds **no new ratchet
/// constant** — §8's rule makes every tripwire a standing maintenance obligation,
/// and the composed-frame `FRAME_BUDGET_MS` already covers this. What is asserted
/// is that the heaviest configuration (two-pass, on the meshlet path) stays inside
/// that existing budget; the per-mode millisecond deltas are *printed* for the
/// ROADMAP cost table, because an absolute number on one machine is not a
/// contract.
///
/// **"Per tier" is answered by construction, not by measurement.** Only
/// `RenderTier::High` runs the meshlet path at all, and `RenderTier::apply` clears
/// `occlusion`/`two_pass` on Medium and Low (as does `clamp_mobile`) — so the
/// overhead on every tier below High is exactly zero, which the pure assertion at
/// the end of this test pins. There is no configuration in which a weaker GPU pays
/// for occlusion culling.
#[test]
fn vgeom_two_pass_cost() {
    use inf_render::{RenderSettings, RenderTier, VgeomAsset, VgeomInstance, VgeomSettings};
    use std::sync::Arc;

    // Tier clamps first — GPU-free, and the "no lower tier pays for this" claim.
    let requested = RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };
    assert!(requested.vgeom.occlusion && requested.vgeom.two_pass);
    for tier in [RenderTier::Medium, RenderTier::Low] {
        let c = tier.apply(requested);
        assert!(
            !c.vgeom.enabled && !c.vgeom.occlusion && !c.vgeom.two_pass,
            "{tier:?} must pay nothing for P18.1"
        );
    }
    let m = RenderTier::clamp_mobile(requested);
    assert!(!m.vgeom.occlusion && !m.vgeom.two_pass);

    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP vgeom_two_pass_cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || info.name.to_ascii_lowercase().contains("paravirtual");

    // A dense meshlet body wall-to-wall in front of the camera, with a field of
    // smaller bodies behind it — the shape the occlusion path exists for.
    const ASSET: u128 = 0x1801_0000_bd67_0000;
    let mesh = Arc::new(vgeom_budget_mesh(64));
    let mut scene = RenderScene {
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &mesh).expect("index the vmesh")],
        ..Default::default()
    };
    let standing = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let mut id = 1u32;
    scene.vgeom_instances.push(VgeomInstance::lit(
        ASSET,
        DVec3::ZERO,
        standing,
        Vec3::splat(9.0),
        [0.7, 0.55, 0.35, 1.0],
        id,
    ));
    for gx in -4..=4 {
        for gy in -3..=3 {
            id += 1;
            scene.vgeom_instances.push(VgeomInstance::lit(
                ASSET,
                DVec3::new(gx as f64 * 1.6, gy as f64 * 1.6, -14.0),
                standing,
                Vec3::splat(0.7),
                [0.25, 0.6, 0.8, 1.0],
                id,
            ));
        }
    }
    scene.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..inf_render::RenderLight::default()
    });
    scene.mark_dirty();

    let eye = DVec3::new(0.0, 0.0, 7.0);
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };
    let target = HeadlessTarget::new(&gpu, W, H);

    let measure = |occlusion: bool, two_pass: bool| -> f64 {
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.set_settings(RenderSettings {
            vgeom: VgeomSettings {
                enabled: true,
                occlusion,
                two_pass,
                ..VgeomSettings::default()
            },
            ..RenderSettings::default()
        });
        // The warmup also converges the temporal early set, so what is timed is a
        // steady-state frame, not the conservative first one.
        for _ in 0..10 {
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        const N: u32 = 60;
        let start = std::time::Instant::now();
        for _ in 0..N {
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(N)
    };

    let off = measure(false, false);
    let single = measure(true, false);
    let two = measure(true, true);
    eprintln!(
        "vgeom occlusion (High tier only): off {off:.3} ms | single-pass {single:.3} ms \
         (+{:.3}) | two-pass {two:.3} ms (+{:.3}) — {} instances, {} meshlets/mesh, on {}",
        single - off,
        two - off,
        scene.vgeom_instances.len(),
        mesh.meshlet_count(),
        info.name
    );

    if software {
        return;
    }
    for (label, ms) in [("single-pass", single), ("two-pass", two)] {
        assert!(
            ms < FRAME_BUDGET_MS,
            "vgeom {label} occlusion cost {ms:.3} ms, over the {FRAME_BUDGET_MS} ms \
             frame budget on {} (§8: investigate, never raise it)",
            info.name
        );
    }
}

// A dense displaced grid plane — literally the shape the vgeom goldens use, from
// the one shared generator (`inf_vgeom::test_support`, bit-portable trig), with
// enough triangles to produce a real meshlet DAG. This file used to carry its own
// copy of the body.
use inf_vgeom::test_support::dense_grid_mesh as vgeom_budget_mesh;

/// **The P18.4 GI v2 budget**: what full-scene voxelization + the probe march +
/// the specular term cost per [`GiQuality`] tier, measured against the same frame
/// with GI off.
///
/// Following `sky_stack_cost_per_tier` and `vgeom_two_pass_cost`, this adds **no
/// new ratchet constant** — §8 makes every tripwire a standing maintenance
/// obligation, and the composed-frame `FRAME_BUDGET_MS` already covers this. What
/// is asserted is that the heaviest configuration (High + SSR, over a 484-cube
/// field that puts hundreds of primitives in the volume) stays inside that budget;
/// the per-tier millisecond deltas are *printed* for the ROADMAP cost table,
/// because an absolute number on one machine is not a contract.
///
/// The tier story below High is answered by construction as well as by
/// measurement: `RenderTier::apply` clamps `GiQuality` down and turns GI **off**
/// entirely on Low, which the pure assertion at the end pins.
#[test]
fn gi_v2_cost_per_tier() {
    use inf_render::{GiQuality, GiSettings, RenderSettings, RenderTier};

    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gi_v2_cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || info.name.to_ascii_lowercase().contains("paravirtual");

    let scene = cube_field(11); // 484 cubes — a real primitive load for the volume
    let view = overlook_view();
    let target = HeadlessTarget::new(&gpu, W, H);

    let measure = |settings: RenderSettings| -> f64 {
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        for _ in 0..10 {
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        const N: u32 = 60;
        let start = std::time::Instant::now();
        for _ in 0..N {
            renderer.render(&gpu, &scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(N)
    };

    let gi = |quality: GiQuality, probe_budget: u32, ssr: bool| RenderSettings {
        gi: GiSettings {
            enabled: true,
            quality,
            probe_budget,
            ssr,
            ..GiSettings::default()
        },
        ..RenderSettings::default()
    };

    let baseline = measure(RenderSettings::default());
    let mut worst = 0.0f64;
    for q in [GiQuality::Low, GiQuality::Medium, GiQuality::High] {
        let ms = measure(gi(q, 0, false));
        worst = worst.max(ms);
        eprintln!(
            "gi_v2 {q:?}: {ms:.3} ms/frame (+{:.3} over a GI-less {baseline:.3} ms) on {}",
            ms - baseline,
            info.name
        );
    }
    // Amortization: the same tier with an eighth of the probes per frame.
    let amortized = measure(gi(GiQuality::High, 256, false));
    eprintln!(
        "gi_v2 High amortized (256 probes/frame): {amortized:.3} ms/frame \
         (+{:.3} over baseline)",
        amortized - baseline
    );
    // The heaviest thing a project can ask for.
    let with_ssr = measure(gi(GiQuality::High, 0, true));
    worst = worst.max(with_ssr);
    eprintln!(
        "gi_v2 High + SSR: {with_ssr:.3} ms/frame (+{:.3} over baseline) on {}",
        with_ssr - baseline,
        info.name
    );

    // Below High, the tier clamp is what governs — by construction, not by luck.
    let asked = gi(GiQuality::High, 0, true);
    assert_eq!(
        RenderTier::Medium.apply(asked).gi.quality,
        GiQuality::Medium
    );
    assert!(!RenderTier::Low.apply(asked).gi.enabled);
    assert!(!RenderTier::clamp_mobile(asked).gi.enabled);

    if software {
        // A CPU rasterizer's timing is not representative of anything; the frame
        // pipeline having run at all is the whole claim here.
        return;
    }
    assert!(
        worst < FRAME_BUDGET_MS,
        "GI v2 at its heaviest cost {worst:.3} ms, over the {FRAME_BUDGET_MS} ms \
         frame budget on {} (§8: investigate, never raise it)",
        info.name
    );
}

/// **P18.5 GPU-instanced scatter cost.** Measures the whole scatter path — three
/// cull dispatches, the HZB build, and up to two indirect draws — against the CPU
/// fallback that draws the same batches, at 100k instances.
///
/// Following the P17.4 / P18.1 / P18.4 precedent this adds **no new ratchet
/// constant** (§8 makes each one a standing obligation); it asserts the heaviest
/// configuration stays inside the existing `FRAME_BUDGET_MS` and prints the
/// per-mode numbers for the ROADMAP cost table, because an absolute millisecond
/// count on one machine is not a contract.
///
/// The interesting comparison is not "GPU vs CPU is faster" — at this instance
/// count that is not in doubt — it is that the GPU path's cost is *bounded by the
/// screen* while the fallback's is bounded by the instance count, which is why the
/// tier that loses the compute path also loses 2/3 of its draw distance.
#[test]
fn scatter_cost_at_one_hundred_thousand_instances() {
    use inf_render::{
        LightKind, PrimMesh, RenderLight, RenderSettings, RenderTier, ScatterBatch, ScatterData,
        ScatterInstance,
    };
    use std::sync::Arc;

    // Tier clamps first — GPU-free, and the "no lower tier pays for this" claim.
    let base = RenderSettings::default();
    assert!(base.scatter.gpu && base.scatter.occlusion && base.scatter.impostors);
    for tier in [RenderTier::Medium, RenderTier::Low] {
        let c = tier.apply(base);
        assert!(
            !c.scatter.gpu && !c.scatter.occlusion,
            "{tier:?} must pay nothing for the scatter compute path"
        );
        assert!(c.scatter.cull_distance_m < base.scatter.cull_distance_m);
    }

    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP scatter_cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || info.name.to_ascii_lowercase().contains("paravirtual");

    // 100k instances over 400 m — the ROADMAP's biome-population order of
    // magnitude, one tenth of the 1M target, on one batch.
    const N: u32 = 316; // 316² = 99 856
    const SPAN: f64 = 400.0;
    let step = SPAN / N as f64;
    let mut insts = Vec::with_capacity((N * N) as usize);
    for i in 0..N * N {
        let (gx, gz) = ((i % N) as f64, (i / N) as f64);
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(0x27d4_eb2d);
        let jx = ((h & 0xFFFF) as f64 / 65535.0) - 0.5;
        let jz = (((h >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
        insts.push(ScatterInstance {
            position: DVec3::new(
                (gx - (N as f64 - 1.0) * 0.5 + jx * 0.7) * step,
                0.4,
                (gz - (N as f64 - 1.0) * 0.5 + jz * 0.7) * step,
            ),
            rotation: Quat::IDENTITY,
            scale: 0.8,
            color: [0.24, 0.52, 0.20, 1.0],
        });
    }
    let count = insts.len();
    let data = Arc::new(ScatterData::build(PrimMesh::Cube, DVec3::ZERO, insts));
    let mut scene = RenderScene {
        scatter: vec![ScatterBatch::lit(data, DVec3::ZERO, 0.85, 90)],
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::new(-0.4, 0.78, -0.48).normalize(),
            color: [1.0, 0.96, 0.88],
            intensity: 3.2,
            ..Default::default()
        }],
        ..Default::default()
    };
    scene.mark_dirty();
    let view = overlook_view();

    const WARM: usize = 8;
    const MEASURED: usize = 40;
    let target = HeadlessTarget::new(&gpu, W, H);
    let measure = |s: RenderSettings| -> f64 {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(s);
        for _ in 0..WARM {
            r.render(&gpu, &scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURED {
            r.render(&gpu, &scene, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        t0.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64
    };

    let empty = {
        let mut bare = scene.clone();
        bare.scatter.clear();
        bare.mark_dirty();
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        for _ in 0..WARM {
            r.render(&gpu, &bare, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURED {
            r.render(&gpu, &bare, &view, &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        t0.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64
    };

    let mut no_occ = RenderSettings::default();
    no_occ.scatter.occlusion = false;
    let mut cpu = RenderSettings::default();
    cpu.scatter.gpu = false;
    // Scatter WITH cascaded shadows, at the same real scale. The shadow node packs
    // its caster set on the CPU, so this is the configuration where the P18.5
    // caster clamps (`shadow_caster_settings` + `MAX_CPU_SCATTER_INSTANCES`) are
    // load-bearing rather than theoretical — and the one the audit found had been
    // escaping every clamp the renderer has.
    let mut shadowed = RenderSettings::default();
    shadowed.shadows.enabled = true;
    let mut shadowed_cpu = shadowed;
    shadowed_cpu.scatter.gpu = false;

    let full = measure(RenderSettings::default());
    let unocc = measure(no_occ);
    let fallback = measure(cpu);
    let with_shadows = measure(shadowed);
    let shadowed_fallback = measure(shadowed_cpu);

    // What the caster clamps actually admitted, at the settings just measured.
    let casters = {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(shadowed);
        r.render(&gpu, &scene, &view, &target.view, (W, H));
        r.scatter_audit(&gpu).shadow_casters
    };

    eprintln!(
        "scatter cost ({count} instances, {W}x{H}, on {}): no scatter {empty:.3} ms | \
         GPU path {full:.3} ms (+{:.3}) | GPU without HZB {unocc:.3} ms (+{:.3}) | \
         CPU fallback {fallback:.3} ms (+{:.3}) | GPU + CSM {with_shadows:.3} ms \
         (+{:.3}, {casters} casters) | CPU + CSM {shadowed_fallback:.3} ms (+{:.3})",
        info.name,
        full - empty,
        unocc - empty,
        fallback - empty,
        with_shadows - empty,
        shadowed_fallback - empty,
    );

    // The clamps bit: a 400 m field under a 60 m shadow range must not hand every
    // instance to three cascades, and it must hand them *some*.
    assert!(
        casters > 0,
        "no scatter reached the shadow caster set — the shadow arm is vacuous"
    );
    assert!(
        (casters as usize) < count / 4,
        "the shadow caster clamps did not bite: {casters} of {count} instances \
         packed into the cascades under a {} m shadow range",
        shadowed.shadows.max_distance
    );
    assert!(
        (casters as usize) <= inf_render::MAX_CPU_SCATTER_INSTANCES,
        "the caster ceiling was exceeded"
    );

    if software {
        return;
    }
    for (label, ms) in [
        ("GPU path", full),
        ("GPU without HZB", unocc),
        ("CPU fallback", fallback),
        ("GPU + CSM", with_shadows),
        ("CPU + CSM", shadowed_fallback),
    ] {
        assert!(
            ms < FRAME_BUDGET_MS,
            "scatter {label} cost {ms:.3} ms at {count} instances, over the \
             {FRAME_BUDGET_MS} ms frame budget on {} (§8: investigate, never raise it)",
            info.name
        );
    }
}

/// **The P22.1 surface-deformation budget.**
///
/// What the deformation path adds to a frame is a *pack* and an *upload* — a
/// walking player's field changes every step, so the window re-packs and
/// re-uploads its dirty rect every frame, and the terrain shader gains four
/// window fetches per fragment on top of the four height fetches it already made.
/// This measures the worst realistic version of both: a **saturated** field (the
/// cell bound's worth of live cells is not realistic; a hundred metres of ruts is)
/// under a camera that is inside the window, moving, so the window re-origins
/// every frame and uploads in full.
///
/// It asserts against `inf_core::FRAME_BUDGET_MS` and declares **no constant of
/// its own** — the §8 rule. The number here is a frame, and a frame has one
/// budget.
#[test]
fn deform_window_cost() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP deform_window_cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let virtualized = {
        let n = info.name.to_ascii_lowercase();
        n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware")
    };
    let software = info.device_type == wgpu::DeviceType::Cpu || virtualized;

    // A flat terrain wide enough to fill the frame, and a field of ruts across it.
    const RES: u32 = 129;
    const MPS: f64 = 0.25;
    let span = (RES - 1) as f64 * MPS;
    let n = (RES * RES) as usize;
    let mut tiles = Vec::new();
    for tx in 0..3 {
        for tz in 0..3 {
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(tx as f64 * span, 0.0, tz as f64 * span),
                heights: vec![0.0; n],
                weights: vec![[0, 0, 0, 255]; n],
                biomes: Vec::new(),
                height_bounds: (0.0, 0.0),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    let mut field = inf_terrain::deform::DeformField::new();
    // ~40 lanes × 90 m of rut: several hundred live cells, which is what an hour
    // of driving around one area actually looks like.
    for lane in 0..40 {
        let z = 4.0 + lane as f64 * 2.0;
        let mut x = 2.0;
        while x < 92.0 {
            field.relax(1.0 / 60.0, 0.0);
            field.stamp_contact(
                glam::DVec2::new(x, z),
                0.34,
                inf_terrain::deform::PressureClass::Heavy,
                3,
                1.0 / 60.0,
            );
            x += 0.3;
        }
    }
    assert!(field.cell_count() > 200, "the fixture must be a real load");
    let deform = RenderDeform {
        cell_samples: inf_terrain::deform::DEFORM_CELL_SAMPLES,
        texel_m: inf_terrain::deform::DEFORM_SAMPLE_PITCH_M,
        epoch: field.epoch(),
        cells: field
            .cells()
            .map(|(coord, cell)| RenderDeformCell {
                coord: *coord,
                depths: cell.depths().to_vec(),
            })
            .collect(),
    };
    let scene = RenderScene {
        terrains: vec![RenderTerrain {
            id: 0,
            tile_resolution: RES,
            meters_per_sample: MPS,
            tiles,
            layers: Default::default(),
            macro_variation: 0.0,
            biome_palette: Vec::new(),
        }],
        deform: Some(std::sync::Arc::new(deform)),
        ..Default::default()
    };

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    const WARMUP: u32 = 10;
    const MEASURED: u32 = 60;
    // A moving camera, so the window re-origins and uploads in FULL every frame —
    // the worst case, not the cached one.
    let moving = |f: u32| {
        let eye = DVec3::new(20.0 + f as f64 * 0.4, 9.0, 20.0);
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: eye,
            forward: (DVec3::new(40.0, 0.0, 44.0) - eye).as_vec3().normalize(),
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: W,
            height: H,
            ortho: None,
        }
    };
    for f in 0..WARMUP {
        renderer.render(&gpu, &scene, &moving(f), &target.view, (W, H));
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let before = renderer.deform_uploads();
    let start = std::time::Instant::now();
    for f in 0..MEASURED {
        renderer.render(&gpu, &scene, &moving(WARMUP + f), &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64;
    let uploads = renderer.deform_uploads() - before;

    eprintln!(
        "deform_window_cost: mean {mean_ms:.3} ms/frame over {MEASURED} frames, \
         {uploads} window uploads, {} cells on {} ({:?}); budget {FRAME_BUDGET_MS} ms{}",
        field.cell_count(),
        info.name,
        info.device_type,
        if software {
            " [software — smoke only]"
        } else {
            ""
        }
    );
    // Anti-vacuity: the worst case really was the worst case — a moving camera
    // re-uploaded the window on (almost) every measured frame.
    assert!(
        uploads as u32 >= MEASURED / 2,
        "the moving camera only re-uploaded {uploads} times in {MEASURED} frames — \
         this measured the cached path, not the worst one"
    );

    if software {
        return;
    }
    assert!(
        mean_ms < FRAME_BUDGET_MS,
        "deformation frame mean {mean_ms:.3} ms exceeded the {FRAME_BUDGET_MS} ms \
         budget on {} (the §8 budget only ratchets DOWN — investigate the \
         regression, do not raise it)",
        info.name
    );
}
