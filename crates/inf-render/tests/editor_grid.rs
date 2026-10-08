//! **THE EDITOR'S GROUND GRID OVER A LEVEL WITH GROUND** (wave PAR1b.2,
//! clause 6) — through the same `EngineRenderer` the studio viewport draws
//! with (`RenderScene::grid_enabled`, the viewport host's default `true`).
//!
//! The PAR1b audit's `frames-editor\02-editor-21h.png`: the release editor on
//! the island at 21:00 showed the grid as a bright white lattice over every
//! street and pavement. Two causes, both in `grid.wgsl`: the plane was at
//! RENDER-LOCAL y = 0 — the floating origin's snapped height of the camera, a
//! few metres over the island's streets, so it was drawn in front of them —
//! and its lines are HDR emission drawn before the eye, so the night eye's
//! x40 - x180 tonemapped them white. The fix: the plane is WORLD y = 0
//! (`mode_axis.y`, render-local `-origin.y`), depth-tested like any surface, so
//! ground above sea level occludes it; and below the horizon it dims six
//! stops. Not hidden by default: an empty level (no terrain) keeps its grid at
//! full strength by day, which is the editor every golden draws.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderScene, RenderSettings,
    RenderView, HEADLESS_FORMAT,
};

const W: u32 = 640;
const H: u32 = 360;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP editor_grid: no GPU adapter ({e})");
            None
        }
    }
}

/// The most a grid line may lift the night frame, 8-bit codes — an eighth of
/// full scale: a line a user can find, not a lattice.
const NIGHT_LINE_MAX: u8 = 32;

/// A street's height on the island's spawn settlement, metres.
const ROAD_Y: f64 = 16.6;

/// The camera a pavement-height editor fly-through puts over that street:
/// 5 m up, looking down the street, the floating origin snapped near it (as
/// the viewport host's origin follows its camera).
fn street_view() -> RenderView {
    let eye = DVec3::new(-1750.0, ROAD_Y + 5.0, 2040.0);
    RenderView {
        origin: FloatingOrigin::new(eye),
        eye_world: eye,
        forward: Vec3::new(0.0, -0.25, 1.0).normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn scene(grid: bool, night: bool, with_road: bool) -> RenderScene {
    let mut s = RenderScene {
        grid_enabled: grid,
        ..RenderScene::default()
    };
    if night {
        s.sun.direction = Vec3::new(0.2, -0.35, 1.0).normalize();
    }
    if with_road {
        // The street and its pavements: a wide slab whose top is the road.
        s.instances.push(MeshInstance::lit(
            DVec3::new(-1750.0, ROAD_Y - 0.5, 2100.0),
            Quat::IDENTITY,
            Vec3::new(400.0, 1.0, 400.0),
            [0.16, 0.16, 0.17, 1.0],
            1,
        ));
    }
    s
}

fn render(gpu: &GpuContext, s: &RenderScene, view: &RenderView, exposure: f32) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    let mut st = RenderSettings::default();
    st.exposure = exposure;
    r.set_settings(st);
    for _ in 0..4 {
        r.render(gpu, s, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// Pixels in the lower half (the road, below the horizon) whose colour the
/// grid moved by more than one 8-bit step.
fn grid_pixels(with: &[u8], without: &[u8]) -> usize {
    let mut n = 0;
    for y in H / 2..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            let d = (0..3)
                .map(|c| with[i + c].abs_diff(without[i + c]))
                .max()
                .unwrap_or(0);
            n += usize::from(d > 1);
        }
    }
    n
}

/// **NO GRID OVER THE STREET AT 21:00, AND THE GRID STILL DRAWS WHERE THERE
/// IS NO GROUND.** The island-street camera at night, at the shipped kerb eye
/// x46.2: the frame with the grid on against the same frame with it off — 0
/// grid pixels over the road. Controls: an empty level by day (no ground, the
/// origin at zero, the editor every golden draws) shows the grid (pixels > 0),
/// so the arm is looking at a grid that renders. Mutation (measured, in the
/// PAR1b.2 report): the plane back at render-local 0 (`t = -ro.y / rd.y`) ->
/// the street camera's grid pixels > 0, red.
#[test]
fn the_grid_is_not_drawn_over_a_street_at_night_and_still_draws_on_an_empty_level() {
    let Some(gpu) = gpu() else { return };
    let view = street_view();
    let on = render(&gpu, &scene(true, true, true), &view, 46.2);
    let off = render(&gpu, &scene(false, true, true), &view, 46.2);
    let over_road = grid_pixels(&on, &off);

    let empty_view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 5.0, 0.0),
        ..street_view()
    };
    let day_on = render(&gpu, &scene(true, false, false), &empty_view, 1.0);
    let day_off = render(&gpu, &scene(false, false, false), &empty_view, 1.0);
    let empty_day = grid_pixels(&day_on, &day_off);
    // …and where the grid IS seen at night (no ground over world zero), it is
    // a faint line at the night eye — under an eighth of full scale over the
    // ground (`NIGHT_LINE_MAX`), not the lattice the night eye made of it.
    // Mutation (measured): the night dim removed -> 59 codes, red.
    let night_on = render(&gpu, &scene(true, true, false), &empty_view, 46.2);
    let night_off = render(&gpu, &scene(false, true, false), &empty_view, 46.2);
    let peak = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    };
    let (day_peak, night_peak) = (peak(&day_on, &day_off), peak(&night_on, &night_off));
    println!(
        "PAR1b.2 EDITOR GRID: over the street at 21:00 (x46.2) {over_road} grid pixel(s); empty level by day {empty_day} grid pixel(s) (control); line peak by day (x1) {day_peak} vs at night (x46.2) {night_peak} codes"
    );
    assert!(
        night_peak < NIGHT_LINE_MAX,
        "the grid at night reads {night_peak} codes over the ground (by day {day_peak}) — the night eye whitens it"
    );
    assert!(
        empty_day > 1000,
        "the grid draws nothing on an empty level by day ({empty_day}) — the arm is not looking at a grid"
    );
    assert_eq!(
        over_road, 0,
        "{over_road} grid pixel(s) over the street at night"
    );
}
