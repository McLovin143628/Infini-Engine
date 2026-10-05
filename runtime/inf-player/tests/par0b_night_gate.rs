//! **Wave PAR0b — THE NIGHT AND THE ROOM, gated.**
//!
//! PAR0 gave the renderer many shadowed lights; its audit measured that the
//! DARK was wrong: an unlit sealed room read as bright as an open one (no GI
//! probe inside it, the sky's ambient flooding in), and the night had no floor
//! (no moonlight, a sun-only sky term, manual exposure 1.0). Every arm below
//! reads a rendered luminance, a GPU readback or a byte comparison.
//!
//! | arm | reads | the pre-PAR0b renderer passes? |
//! |---|---|---|
//! | `a_sealed_unlit_room_is_dark_and_one_lamp_reads_its_pool` | floor p50 of a sealed 6 x 3 x 6 m room by day, unlit and with one lamp | no — the sealed floor read the street's ambient |
//! | `an_open_doorway_admits_a_falloff_of_daylight` | floor luminance at five depths from a doorway | no — a 3 m room held no probe: flat |
//! | `a_glazed_room_reads_within_ten_percent_of_the_unglazed` | floor p50 behind a glazed vs an open window | (passed for the wrong reason: both read the sky) |
//! | `a_porch_that_sees_sky_is_not_zeroed_by_a_buried_probe` | porch floor vs open ground ambient | — |
//! | `the_open_air_does_not_darken_under_probe_visibility` | an open courtyard's walls, visibility on vs off | n/a |

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    AtmosphereParams, EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance,
    RenderLight, RenderScene, RenderSettings, RenderView, SunParams, HEADLESS_FORMAT,
};

const W: u32 = 640;
const H: u32 = 360;
/// Frames before a measured readback: VSM pages are marked in F and rastered
/// in F + 2; the GI probes run a full update every frame.
const WARM: u32 = 8;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP par0b_night_gate: no GPU adapter ({e})");
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
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView, s: RenderSettings) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(s);
    for _ in 0..WARM {
        r.render(gpu, scene, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// With `PAR0B_DUMP` naming a directory, write a frame there as raw RGBA.
fn dump(name: &str, img: &[u8]) {
    if let Some(dir) = std::env::var_os("PAR0B_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(format!("{name}.rgba")), img);
    }
}

/// The shipped lit stack's relevant half: GI on, shadows on, virtual shadow
/// maps on (so a lamp is shadowed), manual exposure 1 (the arms read radiance
/// at a fixed exposure, so the eye cannot hide a difference).
fn gi_settings(visibility: bool) -> RenderSettings {
    let mut s = RenderSettings::default();
    s.gi.enabled = true;
    s.gi.probe_visibility = visibility;
    s.shadows.enabled = true;
    s.shadows.max_distance = 120.0;
    s.vsm.enabled = true;
    s
}

/// The noon sun, 40 deg up from +Z.
fn sun_dir() -> Vec3 {
    let e = 40f32.to_radians();
    Vec3::new(0.0, e.sin(), e.cos())
}

/// A sky (the P17 atmosphere) and, if `sun`, its directional light. Without
/// the light the scene still carries the sky's ambient and a near-zero fill so
/// the light list is never empty (an empty list draws the editor sun).
fn sky_scene(sun: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        sun: SunParams {
            direction: sun_dir(),
            color: [1.0, 0.98, 0.95],
            intensity: 3.0,
            ..SunParams::default()
        },
        atmosphere: AtmosphereParams {
            enabled: true,
            aerial_perspective: 0.0,
            ..AtmosphereParams::default()
        },
        ..Default::default()
    };
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: if sun { 3.0 } else { 1.0e-4 },
        direction: sun_dir(),
        cast_shadows: sun,
        ..RenderLight::default()
    });
    // The ground, far past every room.
    let mut g = MeshInstance::lit(
        DVec3::new(0.0, -0.8, 0.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.35, 0.35, 0.35, 1.0],
        1,
    );
    g.roughness = 1.0;
    scene.instances.push(g);
    scene
}

fn slab(scene: &mut RenderScene, id: u32, lo: DVec3, hi: DVec3) {
    let mut m = MeshInstance::lit(
        (lo + hi) * 0.5,
        Quat::IDENTITY,
        (hi - lo).as_vec3(),
        [0.7, 0.7, 0.7, 1.0],
        id,
    );
    m.roughness = 1.0;
    scene.instances.push(m);
}

/// Which opening the room's -Z wall (the wall AWAY from the sun) has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Opening {
    Sealed,
    /// A 1.6 x 2.2 m doorway (a shop door; a 0.9 m one is narrower than
    /// two GI voxels, which the march cannot resolve — see the report).
    Door,
    /// A 2 x 1.5 m window hole, sill at 0.9 m.
    Window,
    /// The same window with a transmitting pane in it.
    Glazed,
}

/// Interior half-extent and height of the ordinary room, metres; walls and
/// slabs 0.3 m — what the island's buildings use, thinner than one GI voxel.
const HALF: f64 = 3.0;
const TALL: f64 = 3.0;
const T: f64 = 0.3;

/// An ordinary room on the ground: floor top at y = 0, interior 6 x 3 x 6 m,
/// the -Z wall holding `opening`, `lamp` hanging at 2.6 m.
fn room(opening: Opening, sun: bool, lamp: bool) -> RenderScene {
    let mut s = sky_scene(sun);
    let (h, t, tall) = (HALF, T, TALL);
    slab(
        &mut s,
        2,
        DVec3::new(-h - t, -t, -h - t),
        DVec3::new(h + t, 0.0, h + t),
    );
    slab(
        &mut s,
        3,
        DVec3::new(-h - t, tall, -h - t),
        DVec3::new(h + t, tall + t, h + t),
    );
    slab(
        &mut s,
        4,
        DVec3::new(-h - t, 0.0, -h - t),
        DVec3::new(-h, tall, h + t),
    );
    slab(
        &mut s,
        5,
        DVec3::new(h, 0.0, -h - t),
        DVec3::new(h + t, tall, h + t),
    );
    slab(
        &mut s,
        6,
        DVec3::new(-h, 0.0, h),
        DVec3::new(h, tall, h + t),
    );
    let (z0, z1) = (-h - t, -h);
    match opening {
        Opening::Sealed => slab(&mut s, 7, DVec3::new(-h, 0.0, z0), DVec3::new(h, tall, z1)),
        Opening::Door => {
            slab(
                &mut s,
                7,
                DVec3::new(-h, 0.0, z0),
                DVec3::new(-0.8, tall, z1),
            );
            slab(&mut s, 8, DVec3::new(0.8, 0.0, z0), DVec3::new(h, tall, z1));
            slab(
                &mut s,
                9,
                DVec3::new(-0.8, 2.2, z0),
                DVec3::new(0.8, tall, z1),
            );
        }
        Opening::Window | Opening::Glazed => {
            slab(
                &mut s,
                7,
                DVec3::new(-h, 0.0, z0),
                DVec3::new(-1.0, tall, z1),
            );
            slab(&mut s, 8, DVec3::new(1.0, 0.0, z0), DVec3::new(h, tall, z1));
            slab(
                &mut s,
                9,
                DVec3::new(-1.0, 0.0, z0),
                DVec3::new(1.0, 0.9, z1),
            );
            slab(
                &mut s,
                10,
                DVec3::new(-1.0, 2.4, z0),
                DVec3::new(1.0, tall, z1),
            );
            if opening == Opening::Glazed {
                let anchor = DVec3::new(0.0, 1.65, z0 + t * 0.5);
                let leaf = inf_render::ScatterInstance {
                    position: anchor,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(2.0, 1.5, 0.05),
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
                b.transmission = 0.85;
                b.casts_shadows = false;
                s.scatter.push(b);
            }
        }
    }
    if lamp {
        s.lights.push(RenderLight {
            kind: LightKind::Point,
            color: [1.0, 0.86, 0.68],
            intensity: 6.0,
            position: DVec3::new(0.0, 2.6, 0.0),
            range: 8.0,
            cast_shadows: true,
            ..RenderLight::default()
        });
    }
    s.mark_dirty();
    s
}

/// Luma (Rec.709, 0..255) of one pixel.
fn luma(img: &[u8], x: u32, y: u32) -> f64 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f64::from(img[i]) + 0.7152 * f64::from(img[i + 1]) + 0.0722 * f64::from(img[i + 2])
}

/// Median luma over a pixel rectangle.
fn p50(img: &[u8], x0: u32, y0: u32, w: u32, h: u32) -> f64 {
    let mut v: Vec<f64> = Vec::with_capacity((w * h) as usize);
    for y in y0..(y0 + h).min(H) {
        for x in x0..(x0 + w).min(W) {
            v.push(luma(img, x, y));
        }
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The screen position of a world point under `view`.
fn project(view: &RenderView, p: DVec3) -> (u32, u32) {
    let c = view.view_proj() * (p - view.origin.origin()).as_vec3().extend(1.0);
    let n = c.truncate() / c.w;
    (
        ((n.x * 0.5 + 0.5) * W as f32).clamp(0.0, W as f32 - 1.0) as u32,
        ((0.5 - n.y * 0.5) * H as f32).clamp(0.0, H as f32 - 1.0) as u32,
    )
}

/// Median luma of a 12 x 12 px patch around a world point.
fn patch_at(img: &[u8], view: &RenderView, p: DVec3) -> f64 {
    let (x, y) = project(view, p);
    p50(img, x.saturating_sub(6), y.saturating_sub(6), 12, 12)
}

/// The camera every room arm stands at: inside, back to the +Z wall, looking
/// down at the floor toward the -Z wall.
fn inside_view() -> RenderView {
    look(DVec3::new(0.0, 1.7, 2.2), DVec3::new(0.0, 0.0, -1.0))
}

/// Floor sample points from the -Z wall inward, metres from the wall.
const DEPTHS: [f64; 5] = [0.5, 1.5, 2.5, 3.5, 4.5];

fn floor_point(depth: f64) -> DVec3 {
    DVec3::new(0.0, 0.0, -HALF + depth)
}

/// **AN UNLIT SEALED ROOM IS DARK; ONE LAMP READS ITS POOL** (clause 1).
///
/// A 6 x 3 x 6 m room with 0.3 m walls and slabs — the island's own
/// thicknesses, thinner than a 0.625 m GI voxel — sealed on every side, under a
/// noon sun and a clear P17 sky, GI on. No light can enter, so every code its
/// floor shows is light the renderer invented. **The bound: the floor's p50
/// at most 3 / 255** — two codes of headroom over the black-plus-dither floor
/// (`0.5`), and under ACES that is a scene radiance below ~0.0008, which no
/// viewer reads as lit (a moonlit kerb is ten times that). The same room with
/// one 6-intensity lamp at 2.6 m must read its pool: the floor under the lamp
/// at least 40 / 255 and brighter than the floor 2.5 m away.
///
/// Mutation: `GiSettings::probe_visibility = false` (the pre-PAR0b blend) —
/// the sealed floor reads the sky's ambient, RED.
#[test]
fn a_sealed_unlit_room_is_dark_and_one_lamp_reads_its_pool() {
    let Some(gpu) = gpu() else { return };
    let view = inside_view();
    let dark = render(
        &gpu,
        &room(Opening::Sealed, true, false),
        &view,
        gi_settings(true),
    );
    let lit = render(
        &gpu,
        &room(Opening::Sealed, true, true),
        &view,
        gi_settings(true),
    );
    let blind = render(
        &gpu,
        &room(Opening::Sealed, true, false),
        &view,
        gi_settings(false),
    );
    dump("sealed_dark", &dark);
    dump("sealed_lit", &lit);
    dump("sealed_no_visibility", &blind);
    let floor = |img: &[u8]| {
        let (x0, y0) = project(&view, floor_point(1.0));
        let (x1, y1) = project(&view, floor_point(4.0));
        p50(
            img,
            x0.min(x1).max(W / 3),
            y0.min(y1),
            W / 3,
            y0.max(y1) - y0.min(y1),
        )
    };
    let (d, l, b) = (floor(&dark), floor(&lit), floor(&blind));
    let under = patch_at(&lit, &view, floor_point(HALF));
    let away = patch_at(&lit, &view, floor_point(0.5));
    let frame_p95 = {
        let mut v: Vec<f64> = (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .map(|(x, y)| luma(&dark, x, y))
            .collect();
        v.sort_by(f64::total_cmp);
        v[v.len() * 95 / 100]
    };
    println!(
        "PAR0b SEALED ROOM: unlit floor p50 {d:.2} (frame p95 {frame_p95:.2}); one lamp floor p50 {l:.2}, under the lamp {under:.2}, 2.5 m away {away:.2}; visibility OFF (the pre-PAR0b blend) {b:.2}"
    );
    assert!(
        d <= 3.0,
        "an unlit sealed room's floor reads {d:.2} / 255 — light the renderer invented"
    );
    assert!(
        frame_p95 <= 8.0,
        "an unlit sealed room's frame p95 is {frame_p95:.2}"
    );
    assert!(l >= 40.0, "one lamp lights the room's floor to only {l:.2}");
    assert!(
        under > away + 5.0,
        "no pool: under the lamp {under:.2}, 2.5 m away {away:.2}"
    );
}

/// **AN OPEN DOORWAY ADMITS A FALLOFF OF DAYLIGHT** (clause 1): the room's
/// door in the wall AWAY from the sun (no direct beam on the floor), the floor
/// sampled at 0.5 / 1.5 / 2.5 / 3.5 / 4.5 m from the doorway wall, photographed
/// at a manual exposure of 16 (a shaded doorway's skylight is a few percent of
/// the sunlit street, under one code at the street's exposure). Not a step:
/// no sample more than 3x its deeper neighbour. Not a flood: the floor by the
/// door under 0.85 of the open ground outside it, and the falloff real (the
/// nearest at least 1.3x the deepest). And not the black of a sealed room: the
/// deepest sample brighter than the sealed room's floor by 2 codes.
#[test]
fn an_open_doorway_admits_a_falloff_of_daylight() {
    let Some(gpu) = gpu() else { return };
    let view = inside_view();
    let bright = RenderSettings {
        exposure: 16.0,
        ..gi_settings(true)
    };
    let door = render(&gpu, &room(Opening::Door, true, false), &view, bright);
    let door1 = render(
        &gpu,
        &room(Opening::Door, true, false),
        &view,
        gi_settings(true),
    );
    let sealed = render(&gpu, &room(Opening::Sealed, true, false), &view, bright);
    dump("doorway", &door);
    let samples: Vec<f64> = DEPTHS
        .iter()
        .map(|&d| patch_at(&door, &view, floor_point(d)))
        .collect();
    let sealed_floor = patch_at(&sealed, &view, floor_point(2.5));
    // The open ground just outside the door, read from outside.
    let out_view = look(DVec3::new(0.0, 1.7, -8.0), DVec3::new(0.0, 0.0, -4.5));
    let outside_img = render(
        &gpu,
        &room(Opening::Door, true, false),
        &out_view,
        gi_settings(true),
    );
    let outside = patch_at(&outside_img, &out_view, DVec3::new(0.0, 0.0, -4.5));
    println!(
        "PAR0b DOORWAY: floor luma by depth {:?} (m {DEPTHS:?}); open ground outside {outside:.2}; sealed room floor {sealed_floor:.2}",
        samples.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()
    );
    for w in samples.windows(2) {
        assert!(
            w[0] <= w[1] * 3.0 + 1.0,
            "a STEP in the doorway's light: {samples:?}"
        );
    }
    let by_door = patch_at(&door1, &view, floor_point(DEPTHS[0]));
    assert!(by_door < outside * 0.85, "a FLOOD: the floor by the door {by_door:.2} vs the open ground {outside:.2} at the same exposure");
    assert!(samples[0] >= samples[4] * 1.3, "no falloff: {samples:?}");
    assert!(
        samples[4] > sealed_floor + 2.0,
        "the doorway admits nothing to 4.5 m: {samples:?} vs sealed {sealed_floor:.2}"
    );
}

/// **A GLAZED ROOM READS LIKE AN UNGLAZED ONE** (clause 1 + the FIX3 audit's
/// CARRIED 68): the window in the shaded wall as an open hole vs with a 0.85
/// transmission pane — the floor's p50 within 10 %. Meaningful only because
/// the sealed room is now dark (the PAR0 report: 190 / 190 / 190 for open,
/// glazed and sealed — green for the wrong reason); so the arm also asserts the
/// window lets light in (the open room's floor 2 codes over the sealed one).
#[test]
fn a_glazed_room_reads_within_ten_percent_of_the_unglazed() {
    let Some(gpu) = gpu() else { return };
    let view = inside_view();
    let open = render(
        &gpu,
        &room(Opening::Window, true, false),
        &view,
        gi_settings(true),
    );
    let glazed = render(
        &gpu,
        &room(Opening::Glazed, true, false),
        &view,
        gi_settings(true),
    );
    let sealed = render(
        &gpu,
        &room(Opening::Sealed, true, false),
        &view,
        gi_settings(true),
    );
    dump("window_open", &open);
    dump("window_glazed", &glazed);
    let floor = |img: &[u8]| {
        let (x0, y0) = project(&view, floor_point(0.8));
        let (_, y1) = project(&view, floor_point(3.5));
        p50(
            img,
            x0.saturating_sub(60),
            y0.min(y1),
            120,
            y0.max(y1) - y0.min(y1),
        )
    };
    let (o, g, s) = (floor(&open), floor(&glazed), floor(&sealed));
    println!("PAR0b GLAZED ROOM: floor p50 open window {o:.2}, glazed {g:.2}, sealed {s:.2}; glazed/open {:.3}", g / o.max(1e-3));
    assert!(
        o > s + 2.0,
        "the open window lets no light in ({o:.2} vs sealed {s:.2})"
    );
    assert!(
        (g - o).abs() <= 0.10 * o,
        "glazed {g:.2} vs open {o:.2} — more than 10 % apart"
    );
}

/// **A PORCH THAT SEES SKY IS NOT ZEROED** (clause 1 + CARRIED 61): a 6 x 6 m
/// roof slab on four posts, open on all four sides, no direct sun (the sky
/// alone lights both readings). Its floor sees the sky sideways under the
/// roof; at least 0.30 of the open ground's ambient beside it. A buried or
/// roofed probe must not zero it.
#[test]
fn a_porch_that_sees_sky_is_not_zeroed_by_a_buried_probe() {
    let Some(gpu) = gpu() else { return };
    let mut s = sky_scene(false);
    slab(
        &mut s,
        2,
        DVec3::new(-3.3, -0.3, -3.3),
        DVec3::new(3.3, 0.0, 3.3),
    );
    slab(
        &mut s,
        3,
        DVec3::new(-3.3, 3.0, -3.3),
        DVec3::new(3.3, 3.3, 3.3),
    );
    for (k, (x, z)) in [(-3.0, -3.0), (3.0, -3.0), (-3.0, 3.0), (3.0, 3.0)]
        .iter()
        .enumerate()
    {
        slab(
            &mut s,
            4 + k as u32,
            DVec3::new(x - 0.15, 0.0, z - 0.15),
            DVec3::new(x + 0.15, 3.0, z + 0.15),
        );
    }
    s.mark_dirty();
    let view = look(DVec3::new(0.0, 1.7, 7.5), DVec3::new(0.0, 0.0, 1.0));
    let img = render(&gpu, &s, &view, gi_settings(true));
    dump("porch", &img);
    let porch = patch_at(&img, &view, DVec3::new(0.0, 0.0, 1.0));
    let open = patch_at(&img, &view, DVec3::new(0.0, -0.3, 5.5));
    println!(
        "PAR0b PORCH: porch floor {porch:.2}, open ground {open:.2}, ratio {:.3}",
        porch / open.max(1e-3)
    );
    assert!(
        porch >= 0.30 * open,
        "the porch floor reads {porch:.2} against the open ground's {open:.2}"
    );
}

/// **THE OPEN AIR DOES NOT DARKEN** (clause 1): probe visibility refuses a
/// probe a surface it cannot see — it must not refuse what it can. An open
/// courtyard (four 12 m walls, no roof) under the noon sun: the shaded walls'
/// and floor's luminance with visibility on within 8 % (or 2 codes) of off.
#[test]
fn the_open_air_does_not_darken_under_probe_visibility() {
    let Some(gpu) = gpu() else { return };
    let mut s = sky_scene(true);
    let (h, t, tall) = (12.0, 1.0, 12.0);
    slab(
        &mut s,
        2,
        DVec3::new(-h - t, -t, -h - t),
        DVec3::new(h + t, 0.0, h + t),
    );
    slab(
        &mut s,
        4,
        DVec3::new(-h - t, 0.0, -h - t),
        DVec3::new(-h, tall, h + t),
    );
    slab(
        &mut s,
        5,
        DVec3::new(h, 0.0, -h - t),
        DVec3::new(h + t, tall, h + t),
    );
    slab(
        &mut s,
        6,
        DVec3::new(-h, 0.0, -h - t),
        DVec3::new(h, tall, -h),
    );
    s.mark_dirty();
    let view = look(DVec3::new(6.0, 3.0, 6.0), DVec3::new(-12.0, 2.0, -6.0));
    let on = render(&gpu, &s, &view, gi_settings(true));
    let off = render(&gpu, &s, &view, gi_settings(false));
    dump("courtyard_on", &on);
    dump("courtyard_off", &off);
    let pts = [
        DVec3::new(-12.0, 2.0, -6.0),
        DVec3::new(-12.0, 6.0, 0.0),
        DVec3::new(-6.0, 2.0, -12.0),
        DVec3::new(-4.0, 0.0, -4.0),
    ];
    for p in pts {
        let (a, b) = (patch_at(&on, &view, p), patch_at(&off, &view, p));
        println!("PAR0b OPEN AIR at {p:?}: visibility on {a:.2}, off {b:.2}");
        assert!(
            a >= b * 0.92 - 2.0,
            "the open air darkened at {p:?}: {a:.2} with visibility vs {b:.2} without"
        );
    }
}

/// The same sealed room at night (no sun), lit by one downlight: a spot at
/// 2.9 m aimed straight down, 60 deg cone, so the ceiling receives NO direct
/// light and reads its bounce alone.
fn downlit_room(bounce: bool) -> (RenderScene, RenderSettings) {
    let mut s = room(Opening::Sealed, false, false);
    s.lights.push(RenderLight {
        kind: LightKind::Spot,
        color: [1.0, 0.9, 0.75],
        intensity: 12.0,
        position: DVec3::new(0.0, 2.9, 0.0),
        direction: Vec3::Y,
        range: 10.0,
        inner_cos: 50f32.to_radians().cos(),
        outer_cos: 60f32.to_radians().cos(),
        cast_shadows: true,
        ..RenderLight::default()
    });
    s.mark_dirty();
    let mut set = gi_settings(true);
    set.gi.local_lights = bounce;
    (s, set)
}

/// **GI READS LOCAL LIGHTS** (clause 2): a sealed room at night lit by one
/// downlight. The ceiling gets no direct light, so what it shows is bounce:
/// with the local lights in the probe march it must read brighter than
/// without (by 5 codes), and — the energy bound — no brighter than the lit
/// floor times the slabs' albedo (0.7): a bounce cannot return more light than
/// arrived.
///
/// Mutation: `GiSettings::local_lights = false` IS the control here; the arm's
/// own mutation is the probe march dropping `gi_local_bounce` (RED: ceiling
/// on == off).
#[test]
fn a_lit_rooms_ceiling_receives_its_fixtures_bounce_within_the_albedo_bound() {
    let Some(gpu) = gpu() else { return };
    let up = look(DVec3::new(0.0, 1.2, 2.0), DVec3::new(0.0, 3.0, -1.0));
    let down = look(DVec3::new(0.0, 2.0, 2.4), DVec3::new(0.0, 0.0, 0.0));
    let (on_scene, on_set) = downlit_room(true);
    let (off_scene, off_set) = downlit_room(false);
    let ceil_on = render(&gpu, &on_scene, &up, on_set);
    let ceil_off = render(&gpu, &off_scene, &up, off_set);
    let floor_on = render(&gpu, &on_scene, &down, on_set);
    dump("downlit_ceiling_bounce", &ceil_on);
    dump("downlit_ceiling_no_bounce", &ceil_off);
    dump("downlit_floor", &floor_on);
    let c_on = patch_at(&ceil_on, &up, DVec3::new(0.0, TALL, -1.0));
    let c_off = patch_at(&ceil_off, &up, DVec3::new(0.0, TALL, -1.0));
    let f = patch_at(&floor_on, &down, DVec3::new(0.0, 0.0, 0.0));
    // Codes are not radiance; compare in (approximately) linear terms through
    // the sRGB decode — the tonemap's toe is near-linear at these levels.
    let lin = |c: f64| {
        let v = c / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    println!(
        "PAR0b BOUNCE: ceiling with the fixture in the march {c_on:.2}, without {c_off:.2}; lit floor {f:.2}; ceiling/floor (linear) {:.3} against the albedo bound 0.7",
        lin(c_on) / lin(f).max(1e-6)
    );
    assert!(
        c_on > c_off + 5.0,
        "the ceiling shows no bounce from its own fixture ({c_on:.2} vs {c_off:.2})"
    );
    assert!(
        lin(c_on) <= 0.7 * lin(f) + 1e-3,
        "the bounce returns more light than arrived: ceiling {c_on:.2} vs floor {f:.2}"
    );
}

/// **NO LIGHT LEAKS THROUGH A WALL VIA THE PROBES** (clause 2): PAR0's wall-leak
/// arm with GI on and the local lights in the march — the closed room's
/// exterior with its lamp vs without, seen from the street at night: the mean
/// delta under 0.05 of a step and no more than 1 % of pixels moved by more
/// than 2 codes (the PCF corner penumbra is clause 7's, read by max there).
#[test]
fn the_wall_leak_holds_with_gi_reading_the_local_lights() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(9.0, 2.5, -9.0), DVec3::new(0.0, 0.8, 0.0));
    let (lamp, set) = downlit_room(true);
    let dark = room(Opening::Sealed, false, false);
    let a = render(&gpu, &lamp, &view, set);
    let b = render(&gpu, &dark, &view, set);
    dump("gi_wall_leak", &a);
    let mut sum = 0u64;
    let mut moved = 0usize;
    let mut max = 0u8;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = (0..3).map(|k| x[k].abs_diff(y[k])).max().unwrap_or(0);
        sum += u64::from(d);
        max = max.max(d);
        if d > 2 {
            moved += 1;
        }
    }
    let n = (W * H) as f64;
    let mean = sum as f64 / n;
    println!("PAR0b GI WALL LEAK: exterior delta mean {mean:.4} max {max}, {moved} px over 2 codes ({:.3} %)", 100.0 * moved as f64 / n);
    assert!(
        mean < 0.05,
        "light leaks out of a closed room through the probes: mean {mean:.4}"
    );
    assert!(
        (moved as f64) < 0.01 * n,
        "{moved} exterior pixels moved by more than 2 codes"
    );
}

/// A lit white wall 6 m behind one (or two) glass panes, at night, seen
/// head-on: `tints` are the panes' colours, nearest first, each its own batch.
fn panes_scene(tints: &[[f32; 4]], reverse: bool) -> RenderScene {
    let mut s = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    s.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0e-4,
        direction: Vec3::Y,
        ..RenderLight::default()
    });
    let mut wall = MeshInstance::lit(
        DVec3::new(0.0, 1.5, 6.0),
        Quat::IDENTITY,
        Vec3::new(8.0, 3.0, 0.3),
        [0.9, 0.9, 0.9, 1.0],
        1,
    );
    wall.roughness = 1.0;
    s.instances.push(wall);
    s.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [1.0, 1.0, 1.0],
        intensity: 8.0,
        position: DVec3::new(0.0, 1.5, 4.0),
        range: 8.0,
        ..RenderLight::default()
    });
    let mut batches = Vec::new();
    for (k, tint) in tints.iter().enumerate() {
        let anchor = DVec3::new(0.15 * k as f64, 1.5, 1.0 + 1.5 * k as f64);
        let leaf = inf_render::ScatterInstance {
            position: anchor,
            rotation: Quat::IDENTITY,
            scale: Vec3::new(2.0, 2.0, 0.05),
            color: *tint,
        };
        let mut b = inf_render::ScatterBatch::lit(
            std::sync::Arc::new(inf_render::ScatterData::build(
                inf_render::PrimMesh::Cube,
                anchor,
                vec![leaf],
            )),
            anchor,
            0.05,
            40 + k as u32,
        );
        b.transmission = 0.85;
        b.casts_shadows = false;
        batches.push(b);
    }
    if reverse {
        batches.reverse();
    }
    s.scatter = batches;
    s.mark_dirty();
    s
}

/// Mean RGB over the centre 40 x 40 px.
fn centre_rgb(img: &[u8]) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    let mut n = 0.0;
    for y in H / 2 - 20..H / 2 + 20 {
        for x in W / 2 - 20..W / 2 + 20 {
            let i = ((y * W + x) * 4) as usize;
            for (c, a) in acc.iter_mut().enumerate() {
                *a += f64::from(img[i + c]);
            }
            n += 1.0;
        }
    }
    acc.map(|v| v / n)
}

/// **A TINTED PANE TINTS THE ROOM BEHIND IT** (clause 6): a white wall lit by
/// a lamp, seen through a clear pane and through a green one. On an adapter
/// with dual-source blending (the glass pass's per-channel path; the arm
/// prints which ran) the green pane's view is GREEN — its green channel over
/// its red and blue by 20 codes — and the clear pane's is grey (channels
/// within 8). Without the feature the honest single-source fallback cannot
/// tint and the arm asserts only that the fallback ran and transmits.
///
/// Mutation: the dual entry writing a grey pass-through — the green view
/// goes grey, RED.
#[test]
fn a_tinted_pane_tints_the_room_behind_it() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(0.0, 1.5, -3.0), DVec3::new(0.0, 1.5, 6.0));
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    r.set_settings(RenderSettings::default());
    let mut shot = |scene: &RenderScene| {
        for _ in 0..4 {
            r.render(&gpu, scene, &view, &target.view, (W, H));
        }
        target.read_rgba(&gpu).expect("readback")
    };
    let clear = shot(&panes_scene(&[[0.92, 0.95, 0.95, 1.0]], false));
    let green = shot(&panes_scene(&[[0.25, 0.95, 0.3, 1.0]], false));
    dump("pane_clear", &clear);
    dump("pane_green", &green);
    let (c, g) = (centre_rgb(&clear), centre_rgb(&green));
    let dual = r.scatter_glass_dual_source();
    println!(
        "PAR0b GLASS TINT: dual-source {dual}; through a clear pane {c:?}, through a green pane {g:?}; glass draws {}",
        r.scatter_glass_draws()
    );
    assert!(r.scatter_glass_draws() > 0, "the glass pass drew nothing");
    if dual {
        assert!(
            g[1] > g[0] + 20.0 && g[1] > g[2] + 20.0,
            "the green pane does not tint the wall: {g:?}"
        );
        assert!(
            (c[0] - c[1]).abs() < 8.0 && (c[1] - c[2]).abs() < 8.0,
            "the clear pane tints: {c:?}"
        );
    } else {
        assert!(
            g[1] > 20.0,
            "the single-source fallback transmits nothing: {g:?}"
        );
    }
}

/// **TWO OVERLAPPING PANES COMPOSITE IN ONE ORDER** (clause 6): two panes in
/// two batches, one behind the other over the lit wall; the scene's batch
/// order reversed — the glass pass sorts back to front by distance, so the
/// frame must not change (max delta at most 1 code), and the overlap must
/// differ from the near pane alone (both are drawn).
#[test]
fn two_overlapping_panes_composite_the_same_in_either_scene_order() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(0.0, 1.5, -3.0), DVec3::new(0.0, 1.5, 6.0));
    let tints = [[0.9, 0.5, 0.5, 1.0], [0.5, 0.5, 0.9, 1.0]];
    let a = render(
        &gpu,
        &panes_scene(&tints, false),
        &view,
        RenderSettings::default(),
    );
    let b = render(
        &gpu,
        &panes_scene(&tints, true),
        &view,
        RenderSettings::default(),
    );
    let one = render(
        &gpu,
        &panes_scene(&tints[..1], false),
        &view,
        RenderSettings::default(),
    );
    dump("panes_two", &a);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    let (ca, c1) = (centre_rgb(&a), centre_rgb(&one));
    println!(
        "PAR0b PANES: scene order reversed moves max {max} code(s); centre two panes {ca:?}, one pane {c1:?}"
    );
    assert!(
        max <= 1,
        "reversing the scene order moved the overlap by {max} codes"
    );
    assert!(
        (ca[2] - c1[2]).abs() > 3.0 || (ca[0] - c1[0]).abs() > 3.0,
        "the second pane is not drawn: {ca:?} vs {c1:?}"
    );
}

/// **GLASS ON THE CPU FALLBACK IS A WINDOW, NOT A WALL** (clause 6): the
/// fixture-lit room behind a pane with the scatter GPU path off (every tier
/// below High). The pane must not draw as an opaque box: the window reads
/// within 15 % of the same window with no pane, and brighter than the wall.
#[test]
fn the_cpu_fallback_draws_a_pane_as_an_opening() {
    let Some(gpu) = gpu() else { return };
    let view = look(DVec3::new(0.0, 1.8, -11.0), DVec3::new(0.0, 1.4, -3.0));
    let mut s = gi_settings(false);
    s.gi.enabled = false;
    s.scatter.gpu = false;
    let glazed = render(&gpu, &room(Opening::Glazed, false, true), &view, s);
    let open = render(&gpu, &room(Opening::Window, false, true), &view, s);
    dump("fallback_glazed", &glazed);
    let (wx, wy) = project(&view, DVec3::new(0.0, 1.65, -HALF - T));
    let (gx, gy) = project(&view, DVec3::new(2.0, 1.65, -HALF - T));
    let g = p50(&glazed, wx - 20, wy - 15, 40, 30);
    let o = p50(&open, wx - 20, wy - 15, 40, 30);
    let wall = p50(&glazed, gx - 10, gy - 10, 20, 20);
    println!("PAR0b FALLBACK GLASS: window glazed {g:.2}, open {o:.2}, wall {wall:.2}");
    assert!(
        (g - o).abs() <= 0.15 * o.max(1.0),
        "the fallback's pane is not an opening: {g:.2} vs open {o:.2}"
    );
    assert!(
        g > wall + 5.0,
        "the fallback window reads like the wall: {g:.2} vs {wall:.2}"
    );
}

/// **WATER READS THE LIGHTS** (clause 6, the PAR0 carried arm): a calm harbour
/// at night with a lamp 3 m over it vs without — the water under the lamp
/// brighter by 5 codes.
#[test]
fn a_lamp_over_the_harbour_raises_the_waters_luminance() {
    let Some(gpu) = gpu() else { return };
    let harbour = |lamp: bool| {
        let mut s = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        s.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [0.6, 0.7, 1.0],
            intensity: 0.01,
            direction: Vec3::new(0.2, 0.6, 0.3).normalize(),
            ..RenderLight::default()
        });
        let mut bed = MeshInstance::lit(
            DVec3::new(0.0, -4.5, 0.0),
            Quat::IDENTITY,
            Vec3::new(200.0, 1.0, 200.0),
            [0.3, 0.3, 0.3, 1.0],
            1,
        );
        bed.roughness = 1.0;
        s.instances.push(bed);
        s.waters = vec![inf_render::RenderWater {
            id: 1,
            kind: inf_render::WaterKindGpu::Ocean,
            level_m: 0.0,
            time_s: 10.0,
            ..inf_render::RenderWater::default()
        }];
        if lamp {
            s.lights.push(RenderLight {
                kind: LightKind::Point,
                color: [1.0, 0.85, 0.6],
                intensity: 20.0,
                position: DVec3::new(0.0, 3.0, 8.0),
                range: 14.0,
                ..RenderLight::default()
            });
        }
        s.mark_dirty();
        s
    };
    let view = look(DVec3::new(0.0, 4.0, -4.0), DVec3::new(0.0, 0.0, 8.0));
    let lit = render(&gpu, &harbour(true), &view, RenderSettings::default());
    let dark = render(&gpu, &harbour(false), &view, RenderSettings::default());
    dump("harbour_lamp", &lit);
    let at = DVec3::new(0.0, 0.0, 8.0);
    let (l, d) = (patch_at(&lit, &view, at), patch_at(&dark, &view, at));
    println!("PAR0b HARBOUR: water under the lamp {l:.2}, without it {d:.2}");
    assert!(
        l > d + 5.0,
        "a lamp over the harbour raises the water by {:.2} codes",
        l - d
    );
}

/// **THE CORNER PENUMBRA IS GONE, READ BY MAX** (clause 7): the sealed room at
/// night with one shadowed 30-intensity point lamp, seen from outside across
/// two of its outer corners, against the same room with no lamp — the shipped
/// virtual-shadow settings (radius-1 PCF). PAR0's audit measured 10 exterior
/// pixels of up to 121 steps at a closed room's outer corners (the slope bias
/// of a grazing local light reaching past a 0.3 m wall); the arm reads the
/// MAXIMUM per-channel delta, not the mean, and bounds it at 2 codes.
///
/// Mutation: `VSM_LOCAL_MAX_BIAS_M` raised to 1.0 (no cap within a wall) —
/// RED (the corner pixels return).
#[test]
fn a_closed_rooms_outer_corners_stay_dark_by_max() {
    let Some(gpu) = gpu() else { return };
    let lamp = |on: bool| {
        let mut s = room(Opening::Sealed, false, false);
        if on {
            s.lights.push(RenderLight {
                kind: LightKind::Point,
                color: [1.0, 0.9, 0.75],
                intensity: 30.0,
                position: DVec3::new(0.0, 2.0, 0.0),
                range: 16.0,
                cast_shadows: true,
                ..RenderLight::default()
            });
        }
        s.mark_dirty();
        s
    };
    let mut set = RenderSettings::default();
    set.vsm.enabled = true;
    set.shadows.enabled = true;
    let mut worst = 0u8;
    let mut over = 0usize;
    for eye in [
        DVec3::new(9.0, 4.0, 9.0),
        DVec3::new(-9.0, 4.0, -9.0),
        DVec3::new(8.0, 1.2, -8.0),
    ] {
        let view = look(eye, DVec3::new(0.0, 0.5, 0.0));
        let a = render(&gpu, &lamp(true), &view, set);
        let b = render(&gpu, &lamp(false), &view, set);
        dump("corner_leak", &a);
        for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let d = (0..3).map(|k| x[k].abs_diff(y[k])).max().unwrap_or(0);
            worst = worst.max(d);
            if d > 2 {
                over += 1;
            }
        }
    }
    println!(
        "PAR0b CORNER: exterior max delta {worst} code(s), {over} px over 2 codes, three views"
    );
    assert!(
        worst <= 2,
        "a closed room's corner leaks {worst} codes ({over} px)"
    );
}

/// **EIGHT STORAGE BUFFERS** (clause 5): the engine device asks for exactly
/// `wgpu::Limits::default()`'s eight fragment storage buffers and the meshlet
/// tier admits an adapter that grants eight — the capability an 8-limit
/// WebGPU / mobile adapter lost in PAR0. The binding census itself is
/// `passes::visbuffer`'s unit arm (environment 4 + resolve 4).
#[test]
fn the_lit_path_fits_eight_storage_buffers_again() {
    let generous = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 1_000_000,
        ..wgpu::Limits::default()
    };
    let asked = inf_render::gpu::engine_limits(&generous).max_storage_buffers_per_shader_stage;
    println!(
        "PAR0b STORAGE: the device asks {asked}; the meshlet tier needs {}; wgpu default {}",
        inf_render::caps::VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE,
        wgpu::Limits::default().max_storage_buffers_per_shader_stage
    );
    assert_eq!(
        asked,
        wgpu::Limits::default().max_storage_buffers_per_shader_stage
    );
    assert_eq!(
        inf_render::caps::VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE,
        wgpu::Limits::default().max_storage_buffers_per_shader_stage
    );
}
