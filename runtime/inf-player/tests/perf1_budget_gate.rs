//! **WAVE PERF1 — THE ISLAND AT FRAME RATE: the gate.**
//!
//! The wave's arms that are not clocks run everywhere; the clock arms (the
//! shipped island's frame against `SHIPPING_FRAME_CEILING_MS`) need the cooked
//! island and a real adapter and live in `fps_instrument` beside the rest of the
//! frame table, on the house law: they print everywhere, return under
//! `cfg!(debug_assertions)` and under `CI`, and assert only in release off CI
//! over the minimum of five rounds.
//!
//! Here:
//!
//! * **the skinned LOD reader** (clause 3) -- the hosts' bind-space rebuild
//!   carries rungs, the pass selects them by projected error, and the chosen
//!   rung's error is under the stated pop bound at every distance;
//! * **the pop, measured** -- a 1080p frame of a body at the distance its first
//!   rung is chosen, drawn with the rung and with the full buffer, compared
//!   pixel for pixel;
//! * **the two constants are one number** -- the renderer's pop bound and the
//!   import crate's.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, SkinnedInstance, SkinnedLodView, SkinnedMeshData, SkinnedShadow,
    HEADLESS_FORMAT,
};

/// A skinned body stand-in: a capsule-ish closed surface 1.8 m tall, `rings`
/// rings, every vertex bound to joint 0 -- a mesh over the rung floor with a
/// silhouette and a shading gradient a pop can show up in.
fn body(rings: u32) -> MeshAsset {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let segs = rings * 2;
    for r in 0..=rings {
        let t = r as f32 / rings as f32;
        let th = std::f32::consts::PI * t;
        let y = 0.9 - 0.9 * th.cos();
        let rad = 0.28 * th.sin() * (1.0 + 0.15 * (t * 9.0).sin());
        for s in 0..=segs {
            let ph = 2.0 * std::f32::consts::PI * s as f32 / segs as f32;
            let n =
                Vec3::new(th.sin() * ph.cos(), -th.cos(), th.sin() * ph.sin()).normalize_or_zero();
            vertices.push(MeshVertex {
                position: [rad * ph.cos(), y, rad * ph.sin()],
                normal: n.to_array(),
                uv: [s as f32 / segs as f32, t],
                ..Default::default()
            });
        }
    }
    let w = segs + 1;
    for r in 0..rings {
        for s in 0..segs {
            let a = r * w + s;
            indices.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
        }
    }
    let n = vertices.len();
    MeshAsset::new(
        vec![SubMesh {
            name: "body".into(),
            vertices,
            indices,
            material_slot: None,
            skin: vec![VertexSkin::default(); n],
        }],
        vec![],
    )
}

fn instance(at: DVec3) -> SkinnedInstance {
    SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        translation: at,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.8, 0.62, 0.5, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        palette: Arc::new(vec![glam::Mat4::IDENTITY]),
        shadow: SkinnedShadow::None,
        vt: inf_render::VtTextureSet::NONE,
        sections: Vec::new(),
    }
}

fn view_1080p() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 1.0, 0.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: 1920,
        height: 1080,
        ortho: None,
    }
}

/// **THE TWO POP BOUNDS ARE ONE NUMBER.** The renderer restates the import
/// crate's constant (it names no import crate); a drift between them would
/// select rungs under one bound and document another.
#[test]
fn the_renderers_pop_bound_is_the_import_crates() {
    assert_eq!(
        inf_render::SKINNED_LOD_PIXEL_ERROR,
        inf_mesh::optimize::INDEX_LOD_PIXEL_ERROR
    );
}

/// **THE LOD READER READS** (clause 3). The hosts' `skinned_mesh_data` gives a
/// body over the floor its rungs; the planner draws the full buffer near and
/// coarser rungs as the body walks away; and at every distance the chosen
/// rung's error, projected at the body's nearest point, is under the pop bound
/// -- the rung one finer than it is the last one that is not over it.
#[test]
fn a_body_draws_coarser_rungs_with_distance_and_never_past_its_pop_bound() {
    let data = inf_player::skinned::skinned_mesh_data(&body(96)).expect("the body is skinned");
    let tris = data.indices.len() / 3;
    assert!(
        data.lods.len() >= 2,
        "a {tris}-triangle body built {} rung(s)",
        data.lods.len()
    );
    let view = SkinnedLodView::of(&view_1080p()).expect("a perspective view");
    let mesh = Arc::new(data);
    let mut seen = std::collections::BTreeSet::new();
    let mut last = 0u8;
    for step in 0..400 {
        let d = 1.0 + f64::from(step) * 1.5;
        let at = DVec3::new(0.0, 0.0, -d);
        let scene = RenderScene {
            skinned_meshes: vec![mesh.clone()],
            skinned: vec![instance(at)],
            ..Default::default()
        };
        let plan = inf_render::plan_skinned_batches_at(&scene, Some(view));
        let lod = plan.runs[0].lod;
        assert!(
            lod >= last,
            "the rung went finer ({last} -> {lod}) as the body walked away at {d} m"
        );
        last = lod;
        seen.insert(lod);
        let dist = (at - view.eye).length() as f32;
        let near = (dist - mesh.lod_radius_m).max(1.0e-3);
        if lod > 0 {
            let err = mesh.lods[usize::from(lod) - 1].error_m;
            let px = err / near * view.px_per_m;
            assert!(
                px <= view.pixel_error,
                "rung {lod} at {d} m projects {px:.3} px"
            );
        }
        if let Some(next) = mesh.lods.get(usize::from(lod)) {
            let px = next.error_m / near * view.px_per_m;
            assert!(
                px > view.pixel_error,
                "rung {} was allowed at {d} m ({px:.3} px) and not taken",
                lod + 1
            );
        }
    }
    println!(
        "PERF1 skinned LOD: {tris} tris, rungs {:?} (error m {:?}), rungs drawn over 1..600 m: {seen:?}",
        mesh.lods.iter().map(|l| l.indices.len() / 3).collect::<Vec<_>>(),
        mesh.lods.iter().map(|l| l.error_m).collect::<Vec<_>>()
    );
    assert!(
        seen.contains(&0) && seen.len() >= 3,
        "only rungs {seen:?} were drawn"
    );
    // No view, no rung: the plan every pre-PERF1 caller makes.
    let scene = RenderScene {
        skinned_meshes: vec![mesh.clone()],
        skinned: vec![instance(DVec3::new(0.0, 0.0, -500.0))],
        ..Default::default()
    };
    assert_eq!(inf_render::plan_skinned_batches(&scene).runs[0].lod, 0);
}

/// Render one body at `at` through the real renderer, its mesh as given.
fn frame(gpu: &GpuContext, mesh: Arc<SkinnedMeshData>, at: DVec3) -> Vec<u8> {
    let v = view_1080p();
    let target = HeadlessTarget::new(gpu, v.width, v.height);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(RenderSettings::default());
    let mut sc = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        skinned: vec![instance(at)],
        ..Default::default()
    };
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        ..RenderLight::default()
    });
    sc.mark_dirty();
    for _ in 0..3 {
        r.render(gpu, &sc, &v, &target.view, (v.width, v.height));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// The same frame with no body in it -- the reference the body's pixels are
/// counted against.
fn frame_empty(gpu: &GpuContext) -> Vec<u8> {
    let v = view_1080p();
    let target = HeadlessTarget::new(gpu, v.width, v.height);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(RenderSettings::default());
    let mut sc = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        ..RenderLight::default()
    });
    sc.mark_dirty();
    for _ in 0..3 {
        r.render(gpu, &sc, &v, &target.view, (v.width, v.height));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// **THE POP, MEASURED** at 1080p: for each rung, the body placed at the first
/// distance (on a 0.25 m walk) where the planner takes that rung, drawn with
/// the rung and with the rung one finer, compared pixel for pixel -- and
/// against a frame with no body at all, so "the body's pixels" counts the
/// pixels the body actually changes.
///
/// What a half-pixel geometric error can do is flip the coverage of a pixel ON
/// THE SILHOUETTE -- a whole body-to-sky step in that one pixel, which is why
/// the per-pixel maximum is printed and not bounded -- and nudge the INTERIOR
/// shading by the coarser triangles' interpolation. So the frame is split:
///
/// * the **silhouette band** (body pixels with a non-body 4-neighbour, in
///   either frame, dilated by one pixel): the pixels that moved by more than
///   8 / 255 there must number no more than the silhouette's own length -- the
///   outline moved by under a pixel;
/// * the **interior** (every other body pixel): under 0.5 % of it may move by
///   more than 8 / 255.
///
/// (The first cut asserted a share of ALL body pixels and a third rung at
/// 2.5 %, drawn ~80 px tall, failed it at ~10 %: 107 of those on the outline,
/// which is the wrong denominator for a silhouette, and 19 of 918 INSIDE it,
/// which is a real shading pop -- so that rung is no longer built
/// (`inf_mesh::optimize::INDEX_LOD_RATIOS`).)
#[test]
fn a_rung_switch_at_its_distance_moves_almost_nothing_at_1080p() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP perf1 pop: no GPU adapter");
        return;
    };
    let data = inf_player::skinned::skinned_mesh_data(&body(96)).expect("skinned");
    let view = SkinnedLodView::of(&view_1080p()).expect("perspective");
    let full = Arc::new(data);
    let mut measured = 0usize;
    for rung in 1..=full.lods.len() {
        // The first distance this rung is chosen at.
        let mut d = 1.0f64;
        let switch = loop {
            let at = DVec3::new(0.0, 0.0, -d);
            let scene = RenderScene {
                skinned_meshes: vec![full.clone()],
                skinned: vec![instance(at)],
                ..Default::default()
            };
            let lod = inf_render::plan_skinned_batches_at(&scene, Some(view)).runs[0].lod;
            if usize::from(lod) >= rung {
                break d;
            }
            d += 0.25;
            if d > 3000.0 {
                break f64::NAN;
            }
        };
        if !switch.is_finite() {
            continue;
        }
        let at = DVec3::new(0.0, 0.0, -switch);
        // The body with rungs up to `rung` (the planner takes `rung` here) and
        // with rungs up to `rung - 1` (it takes the finer one).
        let mut coarse = (*full).clone();
        coarse.lods.truncate(rung);
        let mut finer = (*full).clone();
        finer.lods.truncate(rung - 1);
        let a = frame(&gpu, Arc::new(coarse), at);
        let b = frame(&gpu, Arc::new(finer), at);
        let empty = frame_empty(&gpu);
        let (w, h) = (1920usize, 1080usize);
        let px = |img: &[u8], i: usize| -> [u8; 3] { [img[i * 4], img[i * 4 + 1], img[i * 4 + 2]] };
        let differs = |x: [u8; 3], y: [u8; 3]| x.iter().zip(&y).any(|(p, q)| p.abs_diff(*q) > 4);
        let body: Vec<bool> = (0..w * h)
            .map(|i| differs(px(&a, i), px(&empty, i)) || differs(px(&b, i), px(&empty, i)))
            .collect();
        // The outline: a body pixel with a non-body 4-neighbour.
        let edge: Vec<bool> = (0..w * h)
            .map(|i| {
                if !body[i] {
                    return false;
                }
                let (x, y) = (i % w, i / w);
                [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)]
                    .iter()
                    .any(|(dx, dy)| {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        nx < 0
                            || ny < 0
                            || nx >= w as i64
                            || ny >= h as i64
                            || !body[ny as usize * w + nx as usize]
                    })
            })
            .collect();
        // The band: the outline dilated by one pixel either way.
        let band: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                (-1i64..=1).any(|dy| {
                    (-1i64..=1).any(|dx| {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        nx >= 0
                            && ny >= 0
                            && nx < w as i64
                            && ny < h as i64
                            && edge[ny as usize * w + nx as usize]
                    })
                })
            })
            .collect();
        let (mut body_px, mut outline, mut interior) = (0usize, 0usize, 0usize);
        let (mut moved_band, mut moved_interior, mut max) = (0usize, 0usize, 0u8);
        for i in 0..w * h {
            let d = px(&a, i)
                .iter()
                .zip(&px(&b, i))
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            max = max.max(d);
            body_px += usize::from(body[i]);
            outline += usize::from(edge[i]);
            if band[i] {
                moved_band += usize::from(d > 8);
            } else if body[i] {
                interior += 1;
                moved_interior += usize::from(d > 8);
            }
        }
        println!(
            "PERF1 pop rung {rung} at {switch:.2} m: body {body_px} px (outline {outline}, interior {interior}); moved > 8/255: {moved_band} on the silhouette band, {moved_interior} inside; max {max}/255"
        );
        assert!(
            body_px > 0,
            "rung {rung}: the body drew nothing at {switch} m"
        );
        assert!(
            moved_band <= outline,
            "rung {rung} at {switch:.2} m: {moved_band} silhouette pixels moved against a {outline}-pixel outline -- more than a pixel's shift"
        );
        assert!(
            moved_interior * 200 <= interior.max(1),
            "rung {rung} at {switch:.2} m: {moved_interior} of {interior} interior pixels moved by more than 8/255"
        );
        measured += 1;
    }
    assert!(measured >= 2, "only {measured} rung switches were measured");
}

/// **THE CATCH-UP SPIRAL IS CAPPED** (the PERF1 audit, priority a').
///
/// A windowed frame runs as many fixed steps as its wall time owes. On the
/// island a step costs ~13-17 ms, so a slow frame owed more steps, the extra
/// steps made it slower still, and the loop sat at `FixedStep`'s default cap
/// of eight: measured in the shipped player's own window at noon, 7.8 steps a
/// frame over the run and an interval of 147 ms (p50). The cap is now
/// [`WINDOWED_MAX_CATCH_UP_STEPS`] -- two -- and the backlog past it is
/// DROPPED, so under overload the simulation runs slower than the wall clock
/// instead of the frame dying (measured: 63.9 ms p50 on the same run).
///
/// The arms: the constant is at most three; a half-second frame runs exactly
/// the cap; the frame after it, of one step's length, runs ONE step (the
/// surplus was dropped, not banked). And the source: the one windowed loop
/// calls `run_frame` once, and the sim's stepper is built from the constant.
#[test]
fn a_slow_windowed_frame_runs_at_most_the_catch_up_cap_and_drops_the_rest() {
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim, WINDOWED_MAX_CATCH_UP_STEPS};
    assert!(
        (1..=3).contains(&WINDOWED_MAX_CATCH_UP_STEPS),
        "the catch-up cap is {WINDOWED_MAX_CATCH_UP_STEPS}: above three the island's step cost spirals the window again"
    );
    let mut sim = RuntimeSim::new(
        inf_ecs::EcsWorld::new(),
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    let before = sim.steps();
    let ran = sim.run_frame(0.5, RuntimeInput::default());
    assert_eq!(
        ran, WINDOWED_MAX_CATCH_UP_STEPS,
        "a half-second frame owes thirty steps and must run the cap"
    );
    assert_eq!(sim.steps() - before, u64::from(ran));
    let next = sim.run_frame(1.0 / 60.0, RuntimeInput::default());
    assert_eq!(
        next, 1,
        "the backlog past the cap was banked, not dropped: the next frame ran {next} steps"
    );
    let window = include_str!("../src/window.rs");
    assert_eq!(
        window.matches(".run_frame(").count(),
        1,
        "the windowed loop must reach the sim through ONE run_frame call"
    );
    let rs = include_str!("../src/runtime_sim.rs");
    assert!(
        rs.contains("FixedStep::with_max_steps(1.0 / hz, WINDOWED_MAX_CATCH_UP_STEPS)"),
        "the sim's stepper is not built from WINDOWED_MAX_CATCH_UP_STEPS"
    );
    println!(
        "PERF1 audit (a'): cap {WINDOWED_MAX_CATCH_UP_STEPS}; a 0.5 s frame ran {ran}, the next 1/60 s frame ran {next}"
    );
}

/// **A WEARER AND WHAT IT WEARS DRAW ONE RUNG** (the PERF1 audit, e').
///
/// A garment is its own `SkinnedInstance` on the wearer's transform and the
/// wearer's palette `Arc` (wave OUTFIT1). Each used to pick its rung from its
/// own mesh's error, so a body could take its coarse rung under a jacket still
/// drawn whole. Two bodies of different density stand in for a body and a
/// garment: at a distance where, planned ALONE, they choose different rungs,
/// planned TOGETHER (one palette, one translation) they choose the same one --
/// the finer. The control: the same two with separate palettes keep their own
/// rungs (crowd agents in one pose share a palette and stand metres apart, so
/// the grouping must not reach past the wearer).
#[test]
fn a_wearer_and_its_wearables_draw_the_same_rung_in_the_same_frame() {
    let a = Arc::new(inf_player::skinned::skinned_mesh_data(&body(96)).expect("skinned"));
    let b = Arc::new(inf_player::skinned::skinned_mesh_data(&body(48)).expect("skinned"));
    assert!(
        !a.lods.is_empty() && !b.lods.is_empty(),
        "both stand-ins need rungs"
    );
    let view = SkinnedLodView::of(&view_1080p()).expect("a perspective view");
    let alone = |mesh: &Arc<SkinnedMeshData>, at: DVec3| {
        let scene = RenderScene {
            skinned_meshes: vec![mesh.clone()],
            skinned: vec![instance(at)],
            ..Default::default()
        };
        inf_render::plan_skinned_batches_at(&scene, Some(view)).runs[0].lod
    };
    let together = |at: DVec3, shared: bool| {
        let body = instance(at);
        let mut worn = instance(at);
        worn.mesh = 1;
        if shared {
            worn.palette = body.palette.clone();
        }
        let scene = RenderScene {
            skinned_meshes: vec![a.clone(), b.clone()],
            skinned: vec![body, worn],
            ..Default::default()
        };
        let plan = inf_render::plan_skinned_batches_at(&scene, Some(view));
        let of = |mesh: usize| {
            plan.runs
                .iter()
                .find(|r| r.mesh == mesh)
                .expect("a run")
                .lod
        };
        (of(0), of(1))
    };
    let mut differing = 0usize;
    for step in 0..1600 {
        let at = DVec3::new(0.0, 0.0, -(1.0 + f64::from(step) * 0.375));
        let (ra, rb) = (alone(&a, at), alone(&b, at));
        let (ta, tb) = together(at, true);
        assert_eq!(
            ta, tb,
            "at {:.1} m the wearer drew rung {ta} and its wearable rung {tb}",
            -at.z
        );
        assert_eq!(
            ta,
            ra.min(rb),
            "the agreed rung is not the finer of the two"
        );
        if ra != rb {
            differing += 1;
            assert_eq!(
                together(at, false),
                (ra, rb),
                "two instances that do NOT share a wearer were made to agree"
            );
        }
    }
    println!("PERF1 audit (e'): {differing} of 1600 distances where the two would have drawn different rungs; all agreed");
    assert!(
        differing > 10,
        "the stand-ins never chose different rungs alone ({differing}) -- the arm is vacuous"
    );
}
