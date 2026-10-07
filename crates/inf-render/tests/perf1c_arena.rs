//! **THE MESHLET NODE'S ARGS ARENA** (wave PERF1c, clause 2).
//!
//! Every drawn meshlet asset used to own five tiny buffers -- the early and late
//! cull uniforms, the early and late indirect args, a debug-flags uniform -- and
//! the record path wrote all five through `queue.write_buffer` every frame:
//! ~1 600 queue writes a frame on the shipped island, each its own staging
//! allocation. They now live at per-asset offsets in two shared buffers
//! (`ArgsArena`), filled as one byte image per half and written once.
//!
//! What this file holds it to:
//!
//! * **equivalence over a drive** -- the same moving multi-asset scene, with
//!   assets leaving and re-entering the draw set (so slots are freed and
//!   re-seated), drawn through a renderer whose arena keeps each asset's slot
//!   and through one whose arena re-seats EVERY asset in a new slot EVERY frame,
//!   leaves byte-identical frames. A bind group or an indirect draw that read
//!   the wrong offset would draw one asset with another's visible set, and the
//!   shuffled renderer would see a different asset there each frame;
//! * **engagement** -- the frames are not empty, the draw set really churns, and
//!   the node makes at most three arena writes a frame (plus one when the debug
//!   flags change) where the per-asset path made five per drawn asset.
//!
//! Byte-identity with the PER-ASSET path that preceded the arena is held by the
//! strict golden class (`golden.rs`, `INF_GOLDEN_STRICT=1`): every meshlet
//! golden was blessed through the per-asset buffers and none moved.
//!
//! Skips cleanly with no GPU adapter (CI's software adapter runs it).

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;
use inf_vgeom::VgeomMesh;

const W: u32 = 256;
const H: u32 = 144;
const ASSETS: u128 = 5;
const FRAMES: usize = 24;

/// Five assets -- five meshes of different density, so each has its own
/// meshlet count, its own triangle ceiling and therefore its own indirect args
/// -- in a row the camera walks along; asset `k` sits out of every frame where
/// `(frame + k) % 4 == 0`, so the draw set changes on most frames.
fn scene(meshes: &[Arc<VgeomMesh>], frame: usize) -> RenderScene {
    let mut sc = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    for k in 0..ASSETS {
        if (frame as u128 + k) % 4 == 0 {
            continue;
        }
        let id = 0x1c00_0000_0000_0000 + k;
        sc.vgeom_assets
            .push(VgeomAsset::from_mesh(id, &meshes[k as usize]).expect("index the vmesh"));
        sc.vgeom_instances.push(VgeomInstance::lit(
            id,
            DVec3::new(-6.0 + k as f64 * 3.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(2.5),
            [0.4 + 0.12 * k as f32, 0.55, 0.9 - 0.15 * k as f32, 1.0],
            1 + k as u32,
        ));
    }
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
        ..RenderLight::default()
    });
    sc.mark_dirty();
    sc
}

fn view(frame: usize) -> RenderView {
    let eye = DVec3::new(-4.0 + frame as f64 * 0.35, 4.0, 9.0);
    let at = DVec3::new(eye.x * 0.5, 0.0, 0.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// The shipping meshlet path: occlusion and two-pass on, so both arena halves
/// (early and late) are exercised.
fn settings() -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// Drive the scene for [`FRAMES`] frames; `(frames, arena writes, drawn assets)`.
fn drive(gpu: &GpuContext, shuffle: bool) -> (Vec<Vec<u8>>, u64, usize) {
    let meshes: Vec<Arc<VgeomMesh>> = (0..ASSETS)
        .map(|k| Arc::new(dense_grid_mesh(12 + 10 * k as usize)))
        .collect();
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(settings());
    r.set_vgeom_arena_shuffle(shuffle);
    let mut frames = Vec::with_capacity(FRAMES);
    let mut drawn = 0usize;
    let before = r.vgeom_arg_write_count();
    for f in 0..FRAMES {
        let sc = scene(&meshes, f);
        drawn += sc.vgeom_assets.len();
        r.render(gpu, &sc, &view(f), &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        frames.push(target.read_rgba(gpu).expect("readback"));
    }
    (frames, r.vgeom_arg_write_count() - before, drawn)
}

#[test]
fn the_meshlet_frame_is_the_same_frame_through_the_arena() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP perf1c_arena: no GPU adapter");
        return;
    };
    let (kept, writes, drawn) = drive(&gpu, false);
    let (shuffled, _, _) = drive(&gpu, true);
    println!(
        "PERF1c arena: {FRAMES} frames, {drawn} asset draws, {writes} arena writes \
         (the per-asset path: {} queue writes)",
        drawn * 5
    );
    // Anti-vacuity: the frames show geometry, and differ from one another (the
    // camera moves and the draw set churns).
    let background = kept[0][0..4].to_vec();
    let lit = kept[FRAMES / 2]
        .chunks_exact(4)
        .filter(|p| *p != background.as_slice())
        .count();
    assert!(
        lit > (W * H / 20) as usize,
        "only {lit} pixels differ from the background -- the meshlets did not draw"
    );
    assert!(
        kept[3] != kept[4],
        "consecutive frames are identical -- nothing moved"
    );
    for (k, (a, b)) in kept.iter().zip(&shuffled).enumerate() {
        assert!(
            a == b,
            "frame {k}: re-seating every asset in a new arena slot changed the pixels"
        );
    }
    // Engagement: at most three arena writes a frame plus the one debug-flags
    // write, against five queue writes per drawn asset before the arena.
    assert!(
        writes <= 3 * FRAMES as u64 + 1,
        "{writes} arena writes over {FRAMES} frames"
    );
    assert!(
        drawn >= 3 * FRAMES,
        "the fixture drew only {drawn} assets over {FRAMES} frames"
    );
}
