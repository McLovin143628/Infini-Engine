//! **Wave PAR0 — THE MANY-LIGHTS SUBSTRATE, gated.**
//!
//! Until PAR0 a frame held sixteen lights, first sixteen in scene order, in a
//! 1 040 B uniform six shaders each looped over privately; point and spot
//! lights lit through walls; a stage rig burned at 11 a.m.; the content layer
//! pre-deleted 87 of the 99 fixtures a nightlife strip builds so the sixteen
//! would hold. Every arm below reads what the SHADER saw, a measured
//! luminance, or a byte comparison — never the CPU vector that was submitted.
//!
//! | arm | reads | would the 16-light forward path pass? |
//! |---|---|---|
//! | `two_thousand_lights_reach_their_froxels_through_the_ceiling_read_by_name` | the froxel grid read back from the GPU (distinct light indices, a per-froxel probe) | no — 16 |
//! | `a_closed_room_lights_nothing_outside_and_the_control_leaks` | per-pixel luminance of the exterior, shadows on vs the policy's budget at 0 vs no fixture | no — every point light leaked |
//! | `a_culled_light_changes_no_pixel` | the frame's bytes with culling on vs off | n/a (it had no cull: both lights burned a slot) |
//! | `a_shuffled_scene_order_moves_no_shadow` | the frame's bytes with the scene's light order reversed | no — the slot was "the n-th caster in scene order" |
//! | `a_point_light_over_terrain_raises_its_luminance` | terrain pixels with vs without a lamp | no — terrain read no light |
//! | `a_stage_rig_is_dark_at_eleven_and_lit_at_nine` | the lights both projectors push, at 11:00 and 21:00 | no — 26.0 at 11 a.m. |
//! | `pie_and_shipping_build_byte_identical_light_lists_over_a_night_walk` | `LightPlan::bytes` + `state_bytes` per step on both hosts | n/a (new comparison) |
//! | `the_island_strip_at_nine_shows_its_fixtures_in_the_shader` (ignored: `INF_ISLAND_PACK`) | the froxel census on Harbour City's strip at 21:00 | no — 12 of 99 |

use std::path::{Path, PathBuf};

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;
use inf_render::lights::{froxel_of, LightPlan, LightSettings, LIGHTS_PER_FRAME_CEILING};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance, RenderLight, RenderScene,
    RenderSettings, RenderView, HEADLESS_FORMAT,
};

const W: u32 = 640;
const H: u32 = 360;
/// Frames rendered before a measured readback: the virtual-shadow pages are
/// marked from frame F's depth and rastered in F + 2.
const WARM: u32 = 8;
/// One 8-bit step, the unit every leak and every "changes no pixel" is in.
const STEP: u8 = 1;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP par0_lights_gate: no GPU adapter ({e})");
            None
        }
    }
}

fn look(eye: DVec3, target: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Render `frames` frames and read the last back; the renderer is returned so
/// an arm can read the froxel census of the frame it measured.
fn render(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> (Vec<u8>, EngineRenderer) {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(settings);
    for _ in 0..WARM {
        r.render(gpu, scene, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    (target.read_rgba(gpu).expect("readback"), r)
}

/// (largest per-channel delta, mean per-channel delta) between two frames.
/// With `PAR0_DUMP` naming a directory, write a frame there as raw RGBA
/// (`<name>.rgba`, W × H) for the ledger's frames. A no-op otherwise.
fn dump(name: &str, img: &[u8]) {
    if let Some(dir) = std::env::var_os("PAR0_DUMP") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(format!("{name}.rgba")), img);
    }
}

fn delta(a: &[u8], b: &[u8]) -> (u8, f64) {
    let mut max = 0u8;
    let mut sum = 0u64;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.len() as f64)
}

fn floor(scene: &mut RenderScene, half: f32, id: u32) {
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(half * 2.0, 1.0, half * 2.0),
        [0.7, 0.7, 0.72, 1.0],
        id,
    ));
}

/// A dim moon so the scene is never empty of lights (an empty list would draw
/// the fallback editor sun and the "no fixture" control would not be one).
fn moon() -> RenderLight {
    RenderLight {
        kind: LightKind::Directional,
        color: [0.6, 0.7, 1.0],
        intensity: 0.05,
        direction: Vec3::new(0.3, 0.8, 0.2).normalize(),
        cast_shadows: false,
        ..RenderLight::default()
    }
}

fn vsm_on() -> RenderSettings {
    let mut s = RenderSettings::default();
    s.vsm.enabled = true;
    s
}

fn point(at: DVec3, intensity: f32, range: f32, shadow: bool) -> RenderLight {
    RenderLight {
        kind: LightKind::Point,
        color: [1.0, 0.9, 0.75],
        intensity,
        position: at,
        range,
        cast_shadows: shadow,
        ..RenderLight::default()
    }
}

/// **THE WALL IS GONE, MEASURED** (clauses 1 + 8): 2 000 point lights over a
/// floor, every one inside the view, and every one reaches at least one froxel
/// of the GPU-built grid — read back, not submitted. A per-froxel probe then
/// asks the froxel under a sample of lights' own pixels whether it lists that
/// light. The ceiling the list lives under is read BY NAME.
#[test]
fn two_thousand_lights_reach_their_froxels_through_the_ceiling_read_by_name() {
    let Some(gpu) = gpu() else { return };
    const N: usize = 2000;
    assert!(
        LIGHTS_PER_FRAME_CEILING >= N,
        "the frame ceiling ({LIGHTS_PER_FRAME_CEILING}) no longer holds the arm's {N}"
    );
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor(&mut scene, 150.0, 1);
    scene.lights.push(moon());
    let (cols, rows) = (50usize, 40usize);
    for k in 0..N {
        let (i, j) = ((k % cols) as f64, (k / cols) as f64);
        let at = DVec3::new(-98.0 + i * 4.0, 1.2, -60.0 + j * 4.0);
        let mut l = point(at, 6.0, 3.0, false);
        l.color = [
            0.4 + 0.6 * (i / cols as f64) as f32,
            0.6,
            0.4 + 0.6 * (j / rows as f64) as f32,
        ];
        scene.lights.push(l);
    }
    scene.mark_dirty();
    let view = look(DVec3::new(0.0, 150.0, -150.0), DVec3::new(0.0, 0.0, 10.0));
    let (img, r) = render(&gpu, &scene, &view, RenderSettings::default());

    let plan: &LightPlan = r.light_plan();
    let census = r.light_census(&gpu).expect("the froxel grid reads back");
    println!(
        "PAR0 2000-light frame: plan {} local ({} culled frustum, {} energy), shader saw {} distinct local + {} directional over {} non-empty froxels, longest list {}, {} overflowed, {} entries (a tiled grid would list {}, longest tile {})",
        plan.local(),
        plan.culled_frustum,
        plan.culled_energy,
        census.distinct_local,
        census.directional,
        census.nonempty_froxels,
        census.max_per_froxel,
        census.overflowed_froxels,
        census.entries,
        census.tiled_entries,
        census.tiled_max
    );
    assert_eq!(
        plan.local() as usize,
        N,
        "a light inside the view was culled"
    );
    assert_eq!(
        census.distinct_local as usize, N,
        "only {} of {N} lights reached any froxel the GPU built",
        census.distinct_local
    );
    assert!(
        census.lights_in_shader() as usize > 16 * 100,
        "the 16-light wall"
    );
    for (c, &n) in census.counts.iter().enumerate() {
        if n > inf_render::lights::CLUSTER_MAX_LIGHTS {
            let c = c as u32;
            let (x, y, z) = (
                c % inf_render::lights::CLUSTER_X,
                (c / inf_render::lights::CLUSTER_X) % inf_render::lights::CLUSTER_Y,
                c / (inf_render::lights::CLUSTER_X * inf_render::lights::CLUSTER_Y),
            );
            println!("  overflow froxel ({x},{y},{z}) lists {n}");
        }
    }
    dump("2000_lights", &img);
    assert_eq!(census.overflowed_froxels, 0, "a froxel overflowed");

    // THE PROBE: the froxel under a light's own pixel, at the floor point
    // beneath it, lists that light.
    let vp = view.view_proj();
    let mut probed = 0usize;
    for k in (0..N).step_by(97) {
        let li = 1 + k; // scene index (the moon is 0)
        let at = scene.lights[li].position;
        let ground = DVec3::new(at.x, 0.0, at.z);
        let c = vp * (ground - view.origin.origin()).as_vec3().extend(1.0);
        let ndc = c.truncate() / c.w;
        if ndc.x.abs() > 0.98 || ndc.y.abs() > 0.98 {
            continue;
        }
        let px = (
            (ndc.x * 0.5 + 0.5) * W as f32,
            (0.5 - ndc.y * 0.5) * H as f32,
        );
        let fx = froxel_of(&view, px, ground);
        let list = r.light_froxel(&gpu, fx).expect("a froxel reads back");
        let record = plan
            .scene_index
            .iter()
            .position(|&s| s as usize == li)
            .expect("the light is in the plan") as u32;
        assert!(
            list.contains(&record),
            "light {li} (record {record}) is missing from the froxel {fx} under its own pool ({} entries)",
            list.len()
        );
        probed += 1;
    }
    assert!(probed >= 15, "the probe sampled only {probed} lights");
    // …and the pools are on the floor: the frame is not the moonlit floor.
    let lit = img
        .chunks(4)
        .filter(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 120)
        .count();
    println!("PAR0 2000-light frame: {lit} bright pixels, {probed} froxels probed");
    assert!(
        lit > (W * H / 50) as usize,
        "the 2 000 pools did not light the floor ({lit} px)"
    );
}

/// A closed 6 × 3 × 6 m room (0.3 m walls, floor, ceiling) on a wide floor,
/// with `fixture` inside it.
fn closed_room(fixture: Option<RenderLight>) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor(&mut scene, 40.0, 1);
    let wall = [0.75, 0.72, 0.68, 1.0];
    let boxes = [
        (DVec3::new(0.0, 1.5, 3.15), Vec3::new(6.6, 3.0, 0.3)),
        (DVec3::new(0.0, 1.5, -3.15), Vec3::new(6.6, 3.0, 0.3)),
        (DVec3::new(3.15, 1.5, 0.0), Vec3::new(0.3, 3.0, 6.0)),
        (DVec3::new(-3.15, 1.5, 0.0), Vec3::new(0.3, 3.0, 6.0)),
        (DVec3::new(0.0, 3.15, 0.0), Vec3::new(6.6, 0.3, 6.6)),
    ];
    for (k, (c, s)) in boxes.iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            *c,
            Quat::IDENTITY,
            *s,
            wall,
            k as u32 + 2,
        ));
    }
    scene.lights.push(moon());
    if let Some(l) = fixture {
        scene.lights.push(l);
    }
    scene.mark_dirty();
    scene
}

/// **THE WALL-LEAK ARM** (clause 2): a shadowed fixture in a closed room lights
/// nothing outside it — the exterior frame is within one 8-bit step of the
/// same room with no fixture at all — and the CONTROL, the same fixture with
/// the shadow policy's budget at zero, leaks.
#[test]
fn a_closed_room_lights_nothing_outside_and_the_control_leaks() {
    let Some(gpu) = gpu() else { return };
    let lamp = point(DVec3::new(0.0, 2.2, 0.0), 120.0, 16.0, true);
    let view = look(DVec3::new(11.0, 6.0, 9.0), DVec3::new(3.0, 0.5, 2.0));

    let (dark, _) = render(&gpu, &closed_room(None), &view, vsm_on());
    let (shadowed, r) = render(&gpu, &closed_room(Some(lamp)), &view, vsm_on());
    let mut control_settings = vsm_on();
    control_settings.lights.local_shadow_budget = 0;
    let (leaky, _) = render(&gpu, &closed_room(Some(lamp)), &view, control_settings);

    let slots = r.vsm_light_slots().to_vec();
    dump("wall_dark", &dark);
    dump("wall_shadowed", &shadowed);
    dump("wall_leaky", &leaky);
    let over: Vec<(u32, u32, u8)> = shadowed
        .chunks(4)
        .zip(dark.chunks(4))
        .enumerate()
        .filter_map(|(i, (a, b))| {
            let d = a[..3]
                .iter()
                .zip(&b[..3])
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            (d > STEP).then_some((i as u32 % W, i as u32 / W, d))
        })
        .collect();
    println!(
        "PAR0 wall leak: {} pixels over one step, first {:?}",
        over.len(),
        &over[..over.len().min(12)]
    );
    let (leak_max, leak_mean) = delta(&shadowed, &dark);
    let (ctl_max, ctl_mean) = delta(&leaky, &dark);
    println!(
        "PAR0 wall leak: shadowed fixture vs no fixture max {leak_max} mean {leak_mean:.4}; control (no shadow) max {ctl_max} mean {ctl_mean:.4}; slots {slots:?}"
    );
    assert!(
        slots.get(1).copied().unwrap_or(0) > 0,
        "the policy granted the fixture no page tree, so this arm measured nothing"
    );
    assert!(
        ctl_max > 8,
        "the control does not leak (max {ctl_max}), so the arm proves nothing"
    );
    // THE RESIDUE, measured and bounded rather than tolerated by a loose max:
    // the shipped 3 x 3 PCF kernel straddles a thin wall's OUTER CORNER where
    // it meets the ground, and a tap beside the silhouette reads the unoccluded
    // floor beyond — a texel-wide penumbra at the contact corner, ten pixels of
    // 230 400 on this frame. The far side as a whole moves by under a hundredth
    // of a step, and with the kernel at one tap (hard shadows) NOTHING moves.
    assert!(
        leak_mean < 0.01,
        "a shadowed fixture in a closed room moved the exterior by {leak_mean:.4} steps on average"
    );
    assert!(
        over.len() * 10_000 <= (W * H) as usize,
        "{} exterior pixels moved by more than one step — more than the corner penumbra",
        over.len()
    );
    let mut hard = vsm_on();
    hard.vsm.pcf_radius = 0;
    let (hard_dark, _) = render(&gpu, &closed_room(None), &view, hard);
    let (hard_lit, _) = render(&gpu, &closed_room(Some(lamp)), &view, hard);
    let (hard_max, hard_mean) = delta(&hard_lit, &hard_dark);
    println!("PAR0 wall leak, one-tap kernel: max {hard_max} mean {hard_mean:.5}");
    assert!(
        hard_max <= STEP,
        "with a one-tap kernel a shadowed fixture still changed an exterior pixel by {hard_max} steps — a real leak, not a penumbra"
    );
}

/// **A culled light changes no pixel** (clause 3): one light behind the camera
/// (frustum-culled) and one whose energy sphere cannot reach the view though
/// its range would (energy-culled). The frame with culling OFF — both in the
/// buffer — is within one step of the frame with culling ON, and the plan
/// says which cull removed each.
#[test]
fn a_culled_light_changes_no_pixel() {
    let Some(gpu) = gpu() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor(&mut scene, 60.0, 1);
    scene.lights.push(moon());
    scene
        .lights
        .push(point(DVec3::new(0.0, 2.0, 2.0), 30.0, 10.0, false));
    // Behind the camera, range 4: outside the frustum.
    scene
        .lights
        .push(point(DVec3::new(0.0, 6.0, -30.0), 500.0, 4.0, false));
    // Off to the left beyond the view, range 40 reaches the floor in view but
    // its energy radius (sqrt(0.01 / 9.5e-4) ~ 3.2 m) does not.
    scene
        .lights
        .push(point(DVec3::new(-60.0, 1.0, 20.0), 0.01, 40.0, false));
    scene.mark_dirty();
    let view = look(DVec3::new(0.0, 8.0, -12.0), DVec3::new(0.0, 0.0, 8.0));
    let (on, r_on) = render(&gpu, &scene, &view, RenderSettings::default());
    let mut off_settings = RenderSettings::default();
    off_settings.lights.cull = false;
    let (off, r_off) = render(&gpu, &scene, &view, off_settings);
    let (p_on, p_off) = (r_on.light_plan().clone(), r_off.light_plan().clone());
    let (max, mean) = delta(&on, &off);
    println!(
        "PAR0 cull: on {} local (frustum {}, energy {}), off {} local; frame delta max {max} mean {mean:.5}",
        p_on.local(),
        p_on.culled_frustum,
        p_on.culled_energy,
        p_off.local()
    );
    assert_eq!(
        (p_on.culled_frustum, p_on.culled_energy),
        (1, 1),
        "the culls removed the wrong lights"
    );
    assert_eq!(
        p_off.local(),
        3,
        "culling off kept fewer than every local light"
    );
    assert!(
        max <= STEP,
        "removing two culled lights moved a pixel by {max} steps"
    );
}

/// **THE SLOT SEAM** (clause 1): the virtual-shadow slot rides each record by
/// SCENE index, so reversing the scene's light order moves no shadow. Two
/// shadowed point lights and a sun-free moon, in both orders.
#[test]
fn a_shuffled_scene_order_moves_no_shadow() {
    let Some(gpu) = gpu() else { return };
    let build = |reverse: bool| {
        let mut scene = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        floor(&mut scene, 30.0, 1);
        for (k, x) in [-2.5f64, 2.5].iter().enumerate() {
            scene.instances.push(MeshInstance::lit(
                DVec3::new(*x, 0.75, 0.0),
                Quat::from_rotation_y(0.3),
                Vec3::new(1.2, 1.5, 1.2),
                [0.8, 0.45, 0.3, 1.0],
                k as u32 + 2,
            ));
        }
        let mut lights = vec![
            moon(),
            point(DVec3::new(-4.0, 3.5, 1.0), 200.0, 30.0, true),
            point(DVec3::new(4.0, 3.0, -1.0), 160.0, 30.0, true),
        ];
        if reverse {
            lights.reverse();
        }
        scene.lights = lights;
        scene.mark_dirty();
        scene
    };
    let view = look(DVec3::new(0.0, 8.0, 9.0), DVec3::new(0.0, 0.3, 0.0));
    let (a, ra) = render(&gpu, &build(false), &view, vsm_on());
    let (b, rb) = render(&gpu, &build(true), &view, vsm_on());
    let (max, mean) = delta(&a, &b);
    println!(
        "PAR0 slot seam: forward slots {:?}, reversed {:?}; frame delta max {max} mean {mean:.5}",
        ra.vsm_light_slots(),
        rb.vsm_light_slots()
    );
    assert!(
        ra.vsm_light_slots().iter().filter(|&&s| s > 0).count() == 2,
        "both point lights should hold a page tree"
    );
    assert!(
        max <= STEP,
        "reversing the scene's light order moved a pixel by {max} steps"
    );
}

/// **A lamp lights the road it stands on** (clause 7): the terrain pass reads
/// the frame's local lights through the shared library.
#[test]
fn a_point_light_over_terrain_raises_its_luminance() {
    let Some(gpu) = gpu() else { return };
    let res = 33u32;
    let mut heights = vec![0f32; (res * res) as usize];
    for (k, h) in heights.iter_mut().enumerate() {
        *h = 0.05 * ((k % 7) as f32);
    }
    let terrain = inf_render::RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: 1.0,
        tiles: vec![inf_render::RenderTerrainTile {
            key: inf_render::TerrainTileKey::lod0((0, 0)),
            origin: DVec3::ZERO,
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (0.0, 0.3),
            holes: Vec::new(),
            version: 1,
        }],
        layers: [inf_render::RenderTerrainLayer {
            albedo: [0.30, 0.30, 0.30, 1.0],
            roughness: 0.9,
            tex_scale: 4.0,
            vt: Default::default(),
        }; 4],
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    };
    let build = |lamp: bool| {
        let mut scene = RenderScene {
            grid_enabled: false,
            terrains: vec![terrain.clone()],
            ..Default::default()
        };
        scene.lights.push(moon());
        if lamp {
            scene
                .lights
                .push(point(DVec3::new(16.0, 2.5, 16.0), 60.0, 12.0, false));
        }
        scene.mark_dirty();
        scene
    };
    let view = look(DVec3::new(16.0, 14.0, 2.0), DVec3::new(16.0, 0.0, 16.0));
    let (dark, _) = render(&gpu, &build(false), &view, RenderSettings::default());
    let (lit, _) = render(&gpu, &build(true), &view, RenderSettings::default());
    let luma = |img: &[u8], x0: u32, y0: u32, w: u32, h: u32| -> f64 {
        let mut s = 0u64;
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let i = ((y * W + x) * 4) as usize;
                s += u64::from(img[i]) + u64::from(img[i + 1]) + u64::from(img[i + 2]);
            }
        }
        s as f64 / f64::from(w * h * 3)
    };
    let (cx, cy) = (W / 2 - 40, H / 2 - 20);
    let (d, l) = (luma(&dark, cx, cy, 80, 40), luma(&lit, cx, cy, 80, 40));
    println!("PAR0 terrain: the pool's mean 8-bit level {d:.2} without the lamp, {l:.2} with it");
    assert!(
        l > d + 10.0,
        "a point light 2.5 m over terrain raised its pool by {:.2} levels",
        l - d
    );
}

/// A 6 × 3 × 6 m room whose front wall (z = -3.15) holds a 2 × 1.5 m window
/// at x ∈ [-1, 1], y ∈ [1, 2.5]; `pane` puts a scattered glass leaf in the void
/// at that transmission (`None` = an open hole); a red cabinet stands against
/// the back wall; `fixture` hangs inside; `sun` adds a daytime sun.
fn windowed_room(pane: Option<f32>, fixture: bool, sun: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    floor(&mut scene, 40.0, 1);
    let wall = [0.75, 0.72, 0.68, 1.0];
    let boxes = [
        (DVec3::new(0.0, 1.5, 3.15), Vec3::new(6.6, 3.0, 0.3)),
        (DVec3::new(3.15, 1.5, 0.0), Vec3::new(0.3, 3.0, 6.0)),
        (DVec3::new(-3.15, 1.5, 0.0), Vec3::new(0.3, 3.0, 6.0)),
        (DVec3::new(0.0, 3.15, 0.0), Vec3::new(6.6, 0.3, 6.6)),
        // The front wall around the window void.
        (DVec3::new(-2.15, 1.5, -3.15), Vec3::new(2.3, 3.0, 0.3)),
        (DVec3::new(2.15, 1.5, -3.15), Vec3::new(2.3, 3.0, 0.3)),
        (DVec3::new(0.0, 0.5, -3.15), Vec3::new(2.0, 1.0, 0.3)),
        (DVec3::new(0.0, 2.75, -3.15), Vec3::new(2.0, 0.5, 0.3)),
    ];
    for (k, (c, sz)) in boxes.iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            *c,
            Quat::IDENTITY,
            *sz,
            wall,
            k as u32 + 2,
        ));
    }
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.0, 2.6),
        Quat::IDENTITY,
        Vec3::new(2.4, 2.0, 0.6),
        [0.85, 0.12, 0.08, 1.0],
        20,
    ));
    if let Some(t) = pane {
        let anchor = DVec3::new(0.0, 1.75, -3.15);
        let leaf = inf_render::ScatterInstance {
            position: anchor,
            rotation: Quat::IDENTITY,
            scale: Vec3::new(2.0, 1.5, 0.3),
            color: [0.86, 0.93, 0.92, 1.0],
        };
        let mut b = inf_render::ScatterBatch::lit(
            std::sync::Arc::new(inf_render::ScatterData::build(
                inf_render::PrimMesh::Cube,
                anchor,
                vec![leaf],
            )),
            anchor,
            0.05,
            30,
        );
        b.transmission = t;
        b.casts_shadows = t <= 0.0;
        scene.scatter.push(b);
    }
    scene.lights.push(moon());
    if sun {
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 0.96, 0.9],
            intensity: 3.0,
            // From BEHIND the room: no direct sun enters the window, so the
            // interior is lit only by what the probe field carries in.
            direction: Vec3::new(0.2, 0.55, 0.8).normalize(),
            cast_shadows: true,
            ..RenderLight::default()
        });
    }
    if fixture {
        scene
            .lights
            .push(point(DVec3::new(0.0, 2.4, 0.5), 30.0, 16.0, true));
    }
    scene.mark_dirty();
    scene
}

/// Mean 8-bit level (r+g+b)/3 over a pixel rectangle.
fn region(img: &[u8], x0: u32, y0: u32, w: u32, h: u32) -> f64 {
    let mut s = 0u64;
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let i = ((y * W + x) * 4) as usize;
            s += u64::from(img[i]) + u64::from(img[i + 1]) + u64::from(img[i + 2]);
        }
    }
    s as f64 / f64::from(w * h * 3)
}

/// The screen rectangle (x0, y0, w, h) a world box covers, shrunk by `inset`
/// of its own size on every side.
fn screen_rect(view: &RenderView, lo: DVec3, hi: DVec3, inset: f32) -> (u32, u32, u32, u32) {
    let vp = view.view_proj();
    let mut min = glam::Vec2::splat(f32::INFINITY);
    let mut max = glam::Vec2::splat(f32::NEG_INFINITY);
    for k in 0..8 {
        let p = DVec3::new(
            if k & 1 == 0 { lo.x } else { hi.x },
            if k & 2 == 0 { lo.y } else { hi.y },
            if k & 4 == 0 { lo.z } else { hi.z },
        );
        let c = vp * (p - view.origin.origin()).as_vec3().extend(1.0);
        let n = c.truncate() / c.w;
        let px = glam::Vec2::new((n.x * 0.5 + 0.5) * W as f32, (0.5 - n.y * 0.5) * H as f32);
        min = min.min(px);
        max = max.max(px);
    }
    let size = max - min;
    let (a, b) = (min + size * inset, max - size * inset);
    let a = a.clamp(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32 - 1.0, H as f32 - 1.0),
    );
    let b = b.clamp(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32 - 1.0, H as f32 - 1.0),
    );
    (
        a.x as u32,
        a.y as u32,
        ((b.x - a.x) as u32).max(1),
        ((b.y - a.y) as u32).max(1),
    )
}

/// **GLASS TRANSMITS** (clause 4): at night, a fixture in a room seen from the
/// street — the window's pixels are brighter through the pane than the wall's
/// beside it; the pane is not opaque (the same window with an opaque leaf is
/// dark, and the red cabinet behind it is red through the glass); and the
/// fixture's light spills out through the glazed window onto the pavement.
#[test]
fn a_fixture_behind_a_window_is_seen_through_the_pane_and_spills_out() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(0.0, 1.8, -11.0), DVec3::new(0.0, 1.4, -3.0));
    let (glass, rg) = render(
        &gpu,
        &windowed_room(Some(0.85), true, false),
        &view,
        vsm_on(),
    );
    let (opaque, _) = render(
        &gpu,
        &windowed_room(Some(0.0), true, false),
        &view,
        vsm_on(),
    );
    let (dark, _) = render(
        &gpu,
        &windowed_room(Some(0.85), false, false),
        &view,
        vsm_on(),
    );
    dump("window_glass_night", &glass);
    dump("window_opaque_night", &opaque);
    let win = screen_rect(
        &view,
        DVec3::new(-1.0, 1.0, -3.3),
        DVec3::new(1.0, 2.5, -3.3),
        0.2,
    );
    let wall = screen_rect(
        &view,
        DVec3::new(1.4, 1.0, -3.3),
        DVec3::new(2.9, 2.5, -3.3),
        0.2,
    );
    let pavement = screen_rect(
        &view,
        DVec3::new(-1.0, 0.0, -9.0),
        DVec3::new(1.0, 0.0, -6.5),
        0.15,
    );
    let (g_win, g_wall, o_win) = (
        region(&glass, win.0, win.1, win.2, win.3),
        region(&glass, wall.0, wall.1, wall.2, wall.3),
        region(&opaque, win.0, win.1, win.2, win.3),
    );
    let (g_pave, o_pave, d_pave) = (
        region(&glass, pavement.0, pavement.1, pavement.2, pavement.3),
        region(&opaque, pavement.0, pavement.1, pavement.2, pavement.3),
        region(&dark, pavement.0, pavement.1, pavement.2, pavement.3),
    );
    // The colour probe: the window's centre pixel shows the red cabinet.
    let (cx, cy) = (win.0 + win.2 / 2, win.1 + win.3 / 2);
    let i = ((cy * W + cx) * 4) as usize;
    let centre = [glass[i], glass[i + 1], glass[i + 2]];
    println!(
        "PAR0 glass: window {g_win:.1} vs wall beside it {g_wall:.1}; opaque leaf {o_win:.1}; pavement glass {g_pave:.1} / opaque {o_pave:.1} / no fixture {d_pave:.1}; window centre {centre:?}; glass batches drawn {}",
        rg.scatter_glass_draws()
    );
    assert!(rg.scatter_glass_draws() > 0, "the glass pass drew nothing");
    assert!(g_win > g_wall + 10.0, "the lit room is not brighter through the pane ({g_win:.1}) than the wall beside it ({g_wall:.1})");
    assert!(
        g_win > o_win + 10.0,
        "the pane is as opaque as a solid leaf ({g_win:.1} vs {o_win:.1})"
    );
    assert!(
        u16::from(centre[0]) > u16::from(centre[1]) + 20,
        "the red cabinet is not red through the glass ({centre:?})"
    );
    assert!(
        g_pave > d_pave + 3.0,
        "no light spills through the glazed window ({g_pave:.1} vs {d_pave:.1} with no fixture)"
    );
    assert!(
        g_pave > o_pave + 3.0,
        "the glazed window spills no more than a solid leaf ({g_pave:.1} vs {o_pave:.1})"
    );
}

/// **GLASS IS A WINDOW TO THE PROBE MARCH** (clause 4 + the FIX3 audit's
/// CARRIED 68): with GI on, the glazed room's pane is NOT staged into the GI
/// voxel volume (the voxelizer's own reject counter, read back) while the
/// same leaf made opaque is — so the probe march sees an opening, not a wall.
///
/// The pixel half is printed, and it is a finding rather than an assertion:
/// this 6 m room's floor reads the SAME p50 open, glazed and sealed, because
/// the shipped probe grid (5.71 m vertical spacing over a 40 m box) places no
/// probe inside it and the interior's ambient is the sky's own irradiance
/// whatever encloses it — FIX3's CARRIED 62, which PAR0 does not close (see
/// the ledger's carried list). A "within 10 %" pixel arm would be green for
/// the wrong reason here.
#[test]
fn a_pane_is_an_opening_to_the_gi_voxelizer_and_an_opaque_leaf_is_a_wall() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(0.0, 2.2, 1.5), DVec3::new(0.0, 0.0, -1.5));
    let mut gi = vsm_on();
    gi.gi.enabled = true;
    let p50 = |img: &[u8]| -> f64 {
        let (x0, y0, w, h) = (W / 4, H / 2, W / 2, H / 2 - 4);
        let mut v: Vec<u16> = Vec::new();
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let i = ((y * W + x) * 4) as usize;
                v.push(u16::from(img[i]) + u16::from(img[i + 1]) + u16::from(img[i + 2]));
            }
        }
        v.sort_unstable();
        f64::from(v[v.len() / 2]) / 3.0
    };
    let (open, ro) = render(&gpu, &windowed_room(None, false, true), &view, gi);
    let (glazed, rg) = render(&gpu, &windowed_room(Some(0.85), false, true), &view, gi);
    let (sealed, rs) = render(&gpu, &windowed_room(Some(0.0), false, true), &view, gi);
    let (ao, ag, as_) = (ro.gi_audit(), rg.gi_audit(), rs.gi_audit());
    dump("room_gi_open", &open);
    dump("room_gi_glazed", &glazed);
    dump("room_gi_sealed", &sealed);
    let (o, g, s) = (p50(&open), p50(&glazed), p50(&sealed));
    println!(
        "PAR0 glass GI: scatter batches rejected open {} / glazed {} / opaque {}; voxelized open {} / glazed {} / opaque {}; interior floor p50 open {o:.2}, glazed {g:.2}, opaque {s:.2} (CARRIED 62: no probe inside a 6 m room)",
        ao.scatter_rejected,
        ag.scatter_rejected,
        as_.scatter_rejected,
        ao.voxelized,
        ag.voxelized,
        as_.voxelized
    );
    assert_eq!(
        ag.scatter_rejected,
        as_.scatter_rejected + 1,
        "the glass leaf was staged into the GI volume like an opaque one"
    );
    assert_eq!(
        ag.voxelized, ao.voxelized,
        "glazed and open rooms voxelize differently"
    );
    assert_eq!(
        as_.voxelized,
        ao.voxelized + 1,
        "the opaque leaf was not voxelized — the control proves nothing"
    );
}

// ── the island fixture: the night schedule + the both-hosts light list ───────

fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

fn build_project(tmp: &Path) -> PathBuf {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    inf_island::write_content(&build, &proj.join("Content")).expect("the island's content writes");
    proj
}

fn cook(proj: &Path, tmp: &Path) -> PathBuf {
    let out = tmp.join("out");
    inf_packager::cook(proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    out
}

fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_bindings(inf_player::level::load_actor_classes_by_guid_from_dir(
        content,
    ))
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

fn hero_entity(sim: &RuntimeSim) -> Option<inf_ecs::Entity> {
    let world = sim.world().world();
    let mut found = None;
    for e in world.iter_entities() {
        if e.get::<inf_ecs::components::CharacterMovement>()
            .is_some_and(|m| m.player_controlled)
        {
            found = Some(e.id());
        }
    }
    found
}

fn set_hero(sim: &mut RuntimeSim, e: inf_ecs::Entity, p: DVec3) {
    if let Some(mut t) = sim
        .world_mut()
        .world_mut()
        .get_mut::<inf_ecs::components::Transform>(e)
    {
        t.translation = inf_ecs::math::Vec3d::new(p.x, p.y, p.z);
    }
    sim.world_mut().mark_dirty();
}

fn freeze_clock(sim: &mut RuntimeSim, hour: f64) {
    let w = sim.world_mut().world_mut();
    let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
    let mut wound = 0usize;
    for mut tod in q.iter_mut(w) {
        tod.seconds = (hour * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
        tod.rate = 0.0;
        wound += 1;
    }
    assert!(wound > 0, "the island carries no clock to freeze");
    sim.world_mut().mark_dirty();
}

fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn project(sim: &RuntimeSim) -> RenderScene {
    let mut scene = RenderScene::default();
    let voxels = inf_voxel::VoxelVolumes::default();
    let mut meshes = inf_render::ScatterMeshes::new();
    inf_player::scatter_mesh::add_building_modules(&mut meshes);
    inf_player::render::project_scene_full(
        &mut scene,
        sim,
        1.0,
        &inf_player::vmesh::VmeshRegistry::default(),
        &inf_player::skinned::SkinnedRegistry::new(),
        &voxels,
        &mut inf_render::DebrisCache::default(),
        None,
        &meshes,
        &std::collections::HashMap::new(),
    );
    scene
}

/// Every resident volume's fixtures (the derived content, before any clock).
fn resident_fixtures(sim: &RuntimeSim) -> usize {
    let w = sim.world().world();
    w.iter_entities()
        .filter_map(|e| {
            e.get::<inf_ecs::components::PcgVolume>()
                .map(|v| v.lights.len())
        })
        .sum()
}

fn local_lights(scene: &RenderScene) -> Vec<&RenderLight> {
    scene
        .lights
        .iter()
        .filter(|l| l.kind != LightKind::Directional)
        .collect()
}

/// The fixture island's venue, and its slug.
fn venue() -> (DVec3, String) {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let v = plans
        .iter()
        .flat_map(|p| p.blocks.iter())
        .find(|b| b.archetype.is_venue())
        .copied()
        .expect("the fixture places at least one venue");
    (DVec3::new(v.centre.x, 0.0, v.centre.y), slug)
}

/// Stand the hero at `at`, freeze the clock at `hour`, and step until the
/// ground (and the venue's derived fixtures) have streamed in.
fn stand(sim: &mut RuntimeSim, at: DVec3, hour: f64) {
    let hero = hero_entity(sim).expect("a hero");
    set_hero(sim, hero, at + DVec3::new(0.0, 2.0, 0.0));
    freeze_clock(sim, hour);
    for _ in 0..90 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
}

/// **THE NIGHT SCHEDULE, in the projector** (clause 5): the venue's rig is
/// derived content at every hour, and the lights the projector pushes are
/// zero at 11:00 and every fixture at 21:00 — on both hosts.
#[test]
fn a_stage_rig_is_dark_at_eleven_and_lit_at_nine() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, tmp.path());
    let (at, slug) = venue();
    for (label, mut sim) in [
        ("shipping", pack_sim(&pack)),
        ("PIE", loose_sim(&proj.join("Content"), &slug)),
    ] {
        stand(&mut sim, at, 11.0);
        let fixtures = resident_fixtures(&sim);
        let day = project(&sim);
        stand(&mut sim, at, 21.0);
        let night = project(&sim);
        let (d, n) = (local_lights(&day), local_lights(&night));
        let watts = |ls: &[&RenderLight]| ls.iter().map(|l| l.intensity).sum::<f32>();
        println!(
            "PAR0 schedule ({label}): {fixtures} fixtures derived; 11:00 pushes {} local lights ({:.1} intensity), 21:00 pushes {} ({:.1})",
            d.len(),
            watts(&d),
            n.len(),
            watts(&n)
        );
        assert!(
            fixtures > 0,
            "{label}: the venue derived no fixture, so the arm is about nothing"
        );
        assert_eq!(d.len(), 0, "{label}: a night venue's rig burns at 11:00");
        assert_eq!(
            n.len(),
            fixtures,
            "{label}: not every fixture is lit at 21:00"
        );
        assert!(
            n.iter().all(|l| l.cast_shadows),
            "{label}: a fixture stopped asking for a shadow"
        );
    }
}

/// **PIE == SHIPPING ON THE LIGHT LIST** (clause 3): both hosts at 21:00 walk
/// the hero past the venue; every step, each host's projected scene goes
/// through the renderer's own plan for the SAME camera, and the two lists'
/// bytes — and the two worlds' bytes — are identical.
#[test]
fn pie_and_shipping_build_byte_identical_light_lists_over_a_night_walk() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, tmp.path());
    let (at, slug) = venue();
    let mut hosts = [pack_sim(&pack), loose_sim(&proj.join("Content"), &slug)];
    for sim in hosts.iter_mut() {
        stand(sim, at - DVec3::new(30.0, 0.0, 0.0), 21.0);
    }
    // Street level: the venue's fixtures hang just under its ceilings, so the
    // lowest of them less a storey is the pavement the walk is at.
    let street = local_lights(&project(&hosts[0]))
        .iter()
        .map(|l| l.position.y)
        .fold(f64::INFINITY, f64::min)
        - 3.0;
    assert!(
        street.is_finite(),
        "no fixture is resident beside the venue at 21:00"
    );
    let target = DVec3::new(at.x, street + 2.0, at.z);
    let settings = LightSettings::default();
    let mut lit_steps = 0usize;
    let mut most = 0u32;
    for step in 0..60 {
        let p = DVec3::new(at.x - 30.0 + step as f64, street + 1.7, at.z - 8.0);
        let mut plans = Vec::new();
        let mut worlds = Vec::new();
        for sim in hosts.iter_mut() {
            let hero = hero_entity(sim).expect("a hero");
            set_hero(sim, hero, p);
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let scene = project(sim);
            let view = RenderView {
                origin: FloatingOrigin::new(p),
                eye_world: p,
                forward: (target - p).as_vec3().normalize(),
                up: Vec3::Y,
                fov_y: 60f32.to_radians(),
                near: 0.05,
                width: 1920,
                height: 1080,
                ortho: None,
            };
            let plan = inf_render::lights::plan_lights(&scene, &view, &[], &settings, 1.0);
            plans.push(plan);
            worlds.push(digest(&sim.state_bytes()));
        }
        assert_eq!(
            plans[0].bytes(),
            plans[1].bytes(),
            "step {step}: the light lists differ"
        );
        assert_eq!(worlds[0], worlds[1], "step {step}: the worlds differ");
        if plans[0].local() > 0 {
            lit_steps += 1;
        }
        most = most.max(plans[0].local());
    }
    println!("PAR0 light lists PIE == shipping: 60 steps, {lit_steps} with a local light, at most {most}");
    assert!(
        lit_steps >= 30,
        "only {lit_steps} of 60 steps saw a local light — the walk compared empty lists"
    );
}

/// **THE STRIP, ON THE REAL ISLAND** (clauses 6 + 8): Harbour City's
/// nightlife strip at 21:00 — every fixture its venue blocks derive, against
/// the lights the froxel grid LISTS (read back from the GPU), from a camera
/// over the strip at 1080p. Before PAR0 `VOLUME_LIGHT_CAP` = 4 kept 4 a block
/// (12 for the strip's three) and the 16-light uniform held the frame.
///
/// Run with `INF_ISLAND_PACK=<cooked island>`; needs a real GPU.
#[test]
#[ignore = "needs a cooked island pack (INF_ISLAND_PACK) and a real GPU"]
fn the_island_strip_at_nine_shows_its_fixtures_in_the_shader() {
    let Some(pack) = std::env::var_os("INF_ISLAND_PACK").map(PathBuf::from) else {
        println!("SKIP: INF_ISLAND_PACK names no pack");
        return;
    };
    let Some(gpu) = gpu() else { return };
    let recipe_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml");
    let recipe = inf_island::IslandRecipe::load(&recipe_path).expect("the island recipe loads");
    let design = inf_island::read_design(&recipe).expect("the island design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let city = plans
        .iter()
        .find(|p| p.name == "Harbour City")
        .expect("the island plans Harbour City");
    let strip: Vec<DVec3> = city
        .blocks
        .iter()
        .filter(|b| b.archetype.is_venue())
        .map(|b| DVec3::new(b.centre.x, 0.0, b.centre.y))
        .collect();
    assert!(!strip.is_empty(), "{} has no venue block", city.name);
    let centre = strip.iter().fold(DVec3::ZERO, |a, &b| a + b) / strip.len() as f64;
    println!(
        "PAR0 island strip: {} — {} venue blocks around ({:.0}, {:.0})",
        city.name,
        strip.len(),
        centre.x,
        centre.z
    );
    let mut sim = pack_sim(&pack);
    stand(&mut sim, centre, 21.0);
    for _ in 0..240 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    // The strip's fixtures: every resident volume's lights within 120 m.
    let mut strip_fixtures = 0usize;
    let mut ground = f64::INFINITY;
    {
        let w = sim.world().world();
        for e in w.iter_entities() {
            if let Some(v) = e.get::<inf_ecs::components::PcgVolume>() {
                for l in &v.lights {
                    // On the strip: within a block's half-diagonal of a venue
                    // block's centre.
                    let near = strip
                        .iter()
                        .any(|b| DVec3::new(l.at.x - b.x, 0.0, l.at.z - b.z).length() < 75.0);
                    if near {
                        strip_fixtures += 1;
                        ground = ground.min(l.at.y);
                    }
                }
            }
        }
    }
    let scene = project(&sim);
    let pushed = local_lights(&scene).len();
    // High enough over the strip that every venue block is in the frame.
    let reach = strip
        .iter()
        .map(|b| DVec3::new(b.x - centre.x, 0.0, b.z - centre.z).length())
        .fold(0.0, f64::max)
        + 80.0;
    let eye = DVec3::new(centre.x, ground + reach * 1.3, centre.z - reach * 1.1);
    let view = RenderView {
        origin: FloatingOrigin::new(eye),
        eye_world: eye,
        forward: (DVec3::new(centre.x, ground, centre.z) - eye)
            .as_vec3()
            .normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.1,
        width: 1920,
        height: 1080,
        ortho: None,
    };
    let target = HeadlessTarget::new(&gpu, 1920, 1080);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    for _ in 0..WARM {
        r.render(&gpu, &scene, &view, &target.view, (1920, 1080));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let census = r.light_census(&gpu).expect("the froxel grid reads back");
    let plan = r.light_plan().clone();
    println!(
        "PAR0 island strip at 21:00: {strip_fixtures} fixtures derived on the strip, {pushed} local lights pushed frame-wide, plan {} local (frustum {}, energy {}), shader lists {} distinct local over {} froxels (longest {}, {} overflowed); pre-PAR0 the strip kept {}",
        plan.local(),
        plan.culled_frustum,
        plan.culled_energy,
        census.distinct_local,
        census.nonempty_froxels,
        census.max_per_froxel,
        census.overflowed_froxels,
        strip.len() * 4
    );
    if let Some(dir) = std::env::var_os("PAR0_DUMP") {
        let img = target.read_rgba(&gpu).expect("readback");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(PathBuf::from(dir).join("island_strip_2100_1080p.rgba"), img);
    }
    assert!(
        strip_fixtures >= 99,
        "the strip derives {strip_fixtures} fixtures — the cap still deletes some"
    );
    assert!(
        census.distinct_local as usize >= strip_fixtures,
        "the shader lists {} local lights against the strip's {strip_fixtures}",
        census.distinct_local
    );
}
