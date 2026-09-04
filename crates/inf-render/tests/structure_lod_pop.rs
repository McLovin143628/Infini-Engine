//! **The parts↔shell pop, measured at 1080p before anything is built for it**
//! (island wave I4, clause 6).
//!
//! Wave I3 shipped the structure LOD as a **hard cut**: inside
//! [`STRUCTURE_LOD_M`] a building draws its own boxes, outside it one oriented
//! shell, and the swap has no cross-fade. The I3 ledger carried that as an open
//! bound — *"at 192 m the pop is small; closing it needs the fade to become a
//! property of the band pair"* — and the I4 brief's instruction is the right one:
//! **measure the pop first; if it is invisible at 192 m at 1080p, REFUSE the
//! cross-fade with the measurement.**
//!
//! So this file renders the two sides of the swap at a shipping resolution, from
//! the swap distance, and reports the difference in the same units the golden
//! harness judges a re-render in. It asserts nothing about which way the answer
//! falls: it prints it, and the wave's ledger records the disposition. What it
//! *does* assert is that the comparison is not vacuous — the building is really
//! on screen, and the two arms really drew different geometry.
//!
//! # Why a dither would not be free
//!
//! The scatter path's existing `fade_band_m` resolves mesh↔impostor with a
//! complementary dither, which is **per-pipeline** rather than per-band. Making
//! the parts and shell bands fade into each other means both are drawn through
//! the overlap — and the I3 audit's `reach` already makes them overlap by the
//! widest shell's half-diagonal, where the shell's faces and the parts'
//! outermost faces are coplanar. A dither there is a dither between two surfaces
//! at the same depth, which is a different problem from the one the scatter path
//! solved.
//!
//! # What the refusal is a refusal ABOUT (the I4 audit)
//!
//! Two readings are taken, and only one of them is the shipped configuration.
//! With `impostors = false` the swap is between the parts' **geometry** and the
//! shell's, and it is invisible — that is the reading the cross-fade is refused
//! on. With `impostors = true`, which is the default, **both** sides are drawn as
//! billboards and 91.6 % of a silhouette **9.2×** the mesh's moves. The change
//! there is still the band pair's; what the impostor contributes is its *size*,
//! because a billboard is sized from the instance's bounding sphere and a
//! 20 × 30 × 7.4 m box's sphere is far wider than the box.
//!
//! So the refusal is **conditional**: no cross-fade, because as geometry there is
//! nothing to fade — and the first repair at this distance is the billboard's
//! sizing, not a fade.
//!
//! # The sizing HAS been repaired, and the refusal still stands (island wave I4b)
//!
//! I4 measured the ratio at **19.2×** (55 868 px against 2 903) and named the
//! sizing as the repair that comes first. I4b took it: `impostor_radius` in
//! `scatter_mesh.wgsl` now answers the instance's own bounding sphere per
//! primitive kind instead of `unit_radius × max(sx, sy, sz)`, and the ratio
//! halved to **9.2×** (26 792 px) — exactly the `0.866 × 30 = 25.98 m` against
//! `0.5 × |(20, 30, 7.4)| = 18.38 m` the arithmetic predicts, squared.
//!
//! What is left is not a defect but what an impostor *is*: a screen-facing card
//! sized to a bounding sphere is intrinsically about twice a box's silhouette,
//! and 91.6 % of it still moves at the swap. So the conditional refusal is
//! **re-taken and kept**, on the same geometry numbers, and the arms below are
//! re-aimed at the repaired ratio rather than deleted — the next thing that could
//! move them is an oriented card or a real impostor atlas, which is a project.
//!
//! And the honest half: the repair **did not move the frame**. Measured on the
//! fps instrument's city at 1080p, MIN of rounds, the scatter pass is
//! 2.98 ms unlit against 2.70–3.31 before and 7.44 ms lit against 7.49 — inside
//! the run-to-run spread on both. Halving a billboard's area is a fidelity fix
//! here, not a performance one, because this scene's scatter cost is its mesh
//! band rather than impostor overdraw.
//!
//! # THE SWAP MOVED, AND WITH IT WHAT EACH ROW MEANS (island wave I8c)
//!
//! Everything above was measured at **192 m**, because that is what
//! [`STRUCTURE_LOD_M`] was — and the I8b audit's LOW-7 records that this file's
//! own note ("at the 192 m structure swap the scatter path is ALREADY in its
//! impostor band") had become the *record of a defect* rather than a caveat: the
//! swap sat 72 m past `ScatterSettings::mesh_distance_m`, so a shell was a
//! billboard from the moment it existed and a building spent the whole
//! 120–230 m annulus as hundreds of cards. Island wave I8c re-ordered the two —
//! the swap is `inf_pcg`'s own 96 m against a 120 m mesh band — and every row
//! below re-aims off the constant.
//!
//! Neither carried bound was ever a claim about the *swap*: both are claims about
//! what a bounding-sphere **card** looks like beside the geometry it stands in
//! for, and the swap merely happened to sit inside the impostor band. They move
//! to `2 × STRUCTURE_LOD_M`, which is the same 192 m, and re-read to **26 792 px
//! against 2 903 (9.2×), 91.5 % moving** — I4b's own figures to the pixel.
//!
//! What is new is the third arm, and it is the ordering's own consequence in
//! pixels: at the swap the shipped picture is **1.37×** the geometry picture
//! rather than 9.2×. The residue above 1.00 is this fixture rather than the band
//! — the building is 30 m deep, so a shot from 96 m spans 81 m to 111 m and its
//! far wall is already inside the 100–120 m mesh→impostor fade.
//!
//! The geometry refusal survived the move on its own numbers and nothing was
//! re-blessed for it: **398 of 13 766 px move (2.9 %), worst channel 29/255** at
//! 96 m, against the 5 % and 32/255 the arms have held since I4.
//!
//! # The fixture is hand-authored, and that is a bound
//!
//! `parts()` and `shell()` are `ScatterInstance` literals shaped like what
//! `inf_pcg::building::assemble` and `group_shell` produce. Neither production
//! function is called here, and neither is the band test that *chooses* between
//! them (`push_shells` / `push_pcg_scatter` are private to the two hosts). So this
//! file measures the difference between the two sides of the swap; it does not
//! prove that the engine's own two sides are those two.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, PrimMesh, RenderLight, RenderScene,
    RenderView, ScatterBatch, ScatterData, ScatterInstance, HEADLESS_FORMAT, STRUCTURE_LOD_M,
};

const W: u32 = 1920;
const H: u32 = 1080;

/// The city's own building: `CITY_FLOORS = 2` storeys on a 20 × 30 m lot, walls
/// and floor slabs as boxes — the shape `inf_pcg::building::assemble` produces
/// and `push_pcg_scatter` hands the GPU.
const FOOTPRINT: (f64, f64) = (20.0, 30.0);
const FLOORS: i32 = 2;
const STOREY_M: f64 = 3.6;
const WALL_M: f64 = 0.25;

/// The building's parts, in world space.
fn parts() -> Vec<ScatterInstance> {
    let (hx, hz) = (FOOTPRINT.0 * 0.5, FOOTPRINT.1 * 0.5);
    let mut v = Vec::new();
    let colour = [0.62, 0.60, 0.55, 1.0];
    for f in 0..FLOORS {
        let y0 = f as f64 * STOREY_M;
        // Slab.
        v.push(ScatterInstance {
            position: DVec3::new(0.0, y0 + 0.1, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(FOOTPRINT.0 as f32, 0.2, FOOTPRINT.1 as f32),
            color: colour,
        });
        // Four walls.
        for (dx, dz, sx, sz) in [
            (hx, 0.0, WALL_M, FOOTPRINT.1),
            (-hx, 0.0, WALL_M, FOOTPRINT.1),
            (0.0, hz, FOOTPRINT.0, WALL_M),
            (0.0, -hz, FOOTPRINT.0, WALL_M),
        ] {
            v.push(ScatterInstance {
                position: DVec3::new(dx, y0 + STOREY_M * 0.5, dz),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(sx as f32, STOREY_M as f32, sz as f32),
                color: colour,
            });
        }
    }
    // Roof.
    v.push(ScatterInstance {
        position: DVec3::new(0.0, FLOORS as f64 * STOREY_M + 0.1, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(FOOTPRINT.0 as f32, 0.2, FOOTPRINT.1 as f32),
        color: colour,
    });
    v
}

/// The shell: the smallest box containing the parts, which is exactly what
/// `inf_pcg::building::lod::group_shell` derives and `push_shells` draws.
fn shell() -> ScatterInstance {
    let height = FLOORS as f64 * STOREY_M + 0.2;
    ScatterInstance {
        position: DVec3::new(0.0, height * 0.5, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(FOOTPRINT.0 as f32, height as f32, FOOTPRINT.1 as f32),
        // `push_shells` gives the shell the first part's colour, so a district of
        // offices does not turn grey at the LOD distance.
        color: [0.62, 0.60, 0.55, 1.0],
    }
}

fn scene(instances: Vec<ScatterInstance>) -> RenderScene {
    let data = Arc::new(ScatterData::build(PrimMesh::Cube, DVec3::ZERO, instances));
    let mut s = RenderScene {
        scatter: vec![ScatterBatch::lit(data, DVec3::ZERO, 0.85, 1)],
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::new(-0.4, 0.78, -0.48).normalize(),
            color: [1.0, 0.96, 0.88],
            intensity: 3.2,
            ..Default::default()
        }],
        grid_enabled: false,
        ..Default::default()
    };
    s.mark_dirty();
    s
}

/// The eye at `d` metres from the building's centre, at eye height, looking at
/// it — the shot a player gets at the swap.
fn view(d: f64) -> RenderView {
    let centre = DVec3::new(0.0, FLOORS as f64 * STOREY_M * 0.5, 0.0);
    let eye = DVec3::new(0.0, 1.7, d);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (centre - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// The building's **silhouette**: every pixel either frame paints differently
/// from the same shot with nothing in it, plus that region's size in pixels.
///
/// Measured against an EMPTY RENDER rather than against a background colour. The
/// first version of this file took the corner pixel as "the background" and
/// counted every pixel unlike it as building — which reported the building as
/// **99.6 % of the frame**, because the sky is a gradient. A coverage number
/// that includes the sky makes every ratio below meaningless.
fn silhouette(parts: &[u8], shell: &[u8], empty: &[u8]) -> (usize, u32, u32) {
    let mut count = 0usize;
    let (mut x0, mut x1, mut y0, mut y1) = (u32::MAX, 0u32, u32::MAX, 0u32);
    for i in 0..(W * H) as usize {
        let px = i * 4;
        let differs = |a: &[u8]| (0..3).any(|c| a[px + c] != empty[px + c]);
        if differs(parts) || differs(shell) {
            count += 1;
            let (x, y) = (i as u32 % W, i as u32 / W);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    (
        count,
        x1.saturating_sub(x0).saturating_add(1),
        y1.saturating_sub(y0).saturating_add(1),
    )
}

/// Pixels whose RGB differs between two frames, and the worst channel delta.
fn moved(a: &[u8], b: &[u8]) -> (usize, u8) {
    let mut differing = 0usize;
    let mut worst = 0u8;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = (0..3).map(|c| pa[c].abs_diff(pb[c])).max().unwrap_or(0);
        if d > 0 {
            differing += 1;
            worst = worst.max(d);
        }
    }
    (differing, worst)
}

/// **THE MEASUREMENT.** What the parts↔shell swap looks like, at the distance it
/// happens, at the resolution it ships at.
#[test]
fn the_parts_to_shell_swap_measured_at_1080p() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP structure_lod_pop: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let target = HeadlessTarget::new(&gpu, W, H);
    let parts_scene = scene(parts());
    let shell_scene = scene(vec![shell()]);

    let shoot = |s: &RenderScene, d: f64, impostors: bool| -> Vec<u8> {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        let mut settings = inf_render::RenderSettings::default();
        settings.scatter.impostors = impostors;
        r.set_settings(settings);
        for _ in 0..3 {
            r.render(&gpu, s, &view(d), &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        target.read_rgba(&gpu).expect("read back")
    };

    // **THE NOISE FLOOR, ONCE PER CONFIGURATION.** The same scene, rendered
    // twice by two fresh renderers. Whatever differs here is not a LOD pop — it
    // is MSAA resolve, the scatter path's GPU cull, and driver scheduling — and a
    // "pixels moved" number quoted without it is a number about the renderer's
    // repeatability as much as about the swap.
    //
    // **Per configuration**, because the impostor path and the mesh path are
    // different pipelines: the I4 audit found the floor measured with
    // `impostors = true` and then compared against a `differing` taken with
    // `impostors = false`, which is a floor for one renderer held against a delta
    // from another.
    let (noise_imp, noise_imp_worst) = moved(
        &shoot(&parts_scene, STRUCTURE_LOD_M, true),
        &shoot(&parts_scene, STRUCTURE_LOD_M, true),
    );
    let (noise, noise_worst) = moved(
        &shoot(&parts_scene, STRUCTURE_LOD_M, false),
        &shoot(&parts_scene, STRUCTURE_LOD_M, false),
    );

    println!(
        "structure LOD pop on {} ({:?}), {W}x{H}, fov 70°, swap at {STRUCTURE_LOD_M} m:",
        info.name, info.device_type
    );
    println!(
        "  noise floor (the same scene twice): impostors OFF {noise} px differ, worst channel {noise_worst}/255; impostors ON {noise_imp} px, worst {noise_imp_worst}/255"
    );
    // **The determinism claim, ARMED.** "The frame is bit-deterministic across
    // two fresh renderers" was printed and never asserted, so a renderer that
    // became non-deterministic would have raised the floor silently and gutted
    // every ratio below without a red arm anywhere (the I4 audit).
    assert_eq!(
        (noise, noise_imp),
        (0, 0),
        "two fresh renderers drew the same scene differently ({noise} px mesh / {noise_imp} px impostor) — the frame is no longer bit-deterministic, and every 'pixels moved' number below is measuring the renderer's repeatability as well as the swap"
    );
    println!(
        "  NOTE: `ScatterSettings::default().mesh_distance_m` is {} m and the swap is at {STRUCTURE_LOD_M} m, so the shipped configuration is in its MESH band at the swap and in its IMPOSTOR band at 2x it (island wave I8c re-ordered the two). All four readings are given.",
        inf_render::RenderSettings::default().scatter.mesh_distance_m
    );
    let mut at_near = None;
    let mut at_near_mesh = None;
    let mut at_swap = None;
    let mut at_swap_mesh = None;
    let mut at_far = None;
    let mut at_far_mesh = None;
    for impostors in [true, false] {
        let label = if impostors {
            "impostors ON (shipped)"
        } else {
            "impostors OFF     "
        };
        for d in [
            STRUCTURE_LOD_M * 0.5,
            STRUCTURE_LOD_M,
            STRUCTURE_LOD_M * 2.0,
        ] {
            // The empty shot is taken at the SAME distance: the sky is a
            // gradient in view space, so an empty frame from somewhere else
            // makes the whole sky read as building. (The first version of this
            // file took one empty shot at the swap distance and reported the
            // building as 68 % of the frame at 96 m.)
            let empty = shoot(&scene(Vec::new()), d, impostors);
            let p = shoot(&parts_scene, d, impostors);
            let s = shoot(&shell_scene, d, impostors);
            let (mean, max) = inf_render::golden::image_diff(&p, &s, W, H);
            let (differing, worst) = moved(&p, &s);
            let (covered, bw, bh) = silhouette(&p, &s, &empty);
            println!(
                "  {label} {d:>6.1} m: building {bw} x {bh} px ({covered} px, {:.4} % of frame); {differing} px move ({:.1} % of it), worst channel {worst}/255; perceptual mean {mean:.5} / max {max:.5}",
                covered as f64 / f64::from(W * H) * 100.0,
                differing as f64 / covered.max(1) as f64 * 100.0,
            );
            let row = (mean, max, differing, covered, worst, bw, bh);
            let slot = if (d - STRUCTURE_LOD_M).abs() < 1.0e-9 {
                if impostors {
                    &mut at_swap
                } else {
                    &mut at_swap_mesh
                }
            } else if (d - STRUCTURE_LOD_M * 2.0).abs() < 1.0e-9 {
                // **The impostor band, which is now PAST the swap** (island wave
                // I8c). The carried bound about a bounding-sphere card did not
                // stop being true when the bands were re-ordered; it moved to the
                // distances where a card is what draws.
                if impostors {
                    &mut at_far
                } else {
                    &mut at_far_mesh
                }
            } else if impostors {
                &mut at_near
            } else {
                &mut at_near_mesh
            };
            *slot = Some(row);
        }
    }

    let (imp_mean, imp_max, imp_moved, imp_covered, imp_worst, imp_w, imp_h) =
        at_swap.expect("the impostor row ran");
    let (mean, max, differing, covered, worst, bw, bh) = at_swap_mesh.expect("the mesh row ran");
    let (_, _, far_moved, far_covered, _, _, _) = at_far.expect("the far impostor row ran");
    let (_, _, _, far_mesh_covered, _, _, _) = at_far_mesh.expect("the far mesh row ran");

    // **THE TWO CONFIGURATIONS ARE ONE PIPELINE, WELL INSIDE THE MESH BAND.**
    // At half the swap distance nothing is near any band edge, so `impostors`
    // must make no difference at all — and it makes none, to the pixel. Without
    // this the ratios below could be reading two renderers rather than two bands.
    assert_eq!(
        at_near.expect("the near impostor row ran"),
        at_near_mesh.expect("the near mesh row ran"),
        "at {:.1} m — half the swap and far inside the mesh band — the shipped configuration and the impostor-free one drew different pictures, so every comparison below is between two renderers rather than between two bands",
        STRUCTURE_LOD_M * 0.5
    );

    // ANTI-VACUITY (1): the building is really on screen at the swap distance.
    assert!(
        covered > 400,
        "only {covered} pixels are the building at {STRUCTURE_LOD_M} m — the shot is empty and the comparison is between two skies"
    );
    // ANTI-VACUITY (2): the two arms really drew different geometry, by more than
    // the renderer's own repeatability (which measured ZERO — the frame is
    // bit-deterministic). Eleven boxes against one that rasterized identically
    // would not be a LOD at all.
    assert!(
        differing > noise,
        "the swap moved {differing} pixels against a {noise}-pixel noise floor — the two arms are indistinguishable from one scene rendered twice"
    );

    println!(
        "
  THE VERDICT — the parts->shell swap, as GEOMETRY, at {STRUCTURE_LOD_M} m:
             the building is {bw} x {bh} px (the I3 ledger's \"about thirty pixels tall\", measured); {differing} of its {covered} px move ({:.1} %), worst channel {worst}/255, frame-level perceptual mean {mean:.5} against a {:.2} tolerance. The pop is INVISIBLE and the cross-fade is REFUSED.",
        differing as f64 / covered.max(1) as f64 * 100.0,
        inf_render::golden::GOLDEN_MEAN_TOLERANCE,
    );
    println!(
        "  …and the finding beside it, RE-AIMED (island wave I8c): the swap is at {STRUCTURE_LOD_M} m and `mesh_distance_m` is {} m, so the SHIPPED configuration meets the swap inside its MESH band — its silhouette there is {imp_w} x {imp_h} px / {imp_covered} px against the mesh reading's {covered} px ({:.2}x rather than the 9.2x it was), {imp_moved} px move ({:.1} %), worst {imp_worst}/255, mean {imp_mean:.5} / max {imp_max:.5}. The residue above 1.00x is this 30 m-deep FIXTURE, whose far wall sits at 111 m and is therefore already inside the 100-120 m fade. The bounding-sphere card is still what an impostor is; it lives PAST the swap now, and at {:.1} m the same building's shipped silhouette is {far_covered} px against the mesh band's {far_mesh_covered} px ({:.1}x), of which {far_moved} px move.",
        inf_render::RenderSettings::default().scatter.mesh_distance_m,
        imp_covered as f64 / covered.max(1) as f64,
        imp_moved as f64 / imp_covered.max(1) as f64 * 100.0,
        STRUCTURE_LOD_M * 2.0,
        far_covered as f64 / far_mesh_covered.max(1) as f64,
    );

    // **THE REFUSAL, ARMED — AT THE BUILDING'S OWN SCALE.**
    //
    // The wave refuses a band-pair cross-fade on the strength of the GEOMETRY
    // reading, so that is the number the arm holds: a 33-pixel-tall building
    // whose swap moves 63 of its 2 903 pixels by at most 18/255.
    //
    // The first version held it against `GOLDEN_MEAN_TOLERANCE` alone, and the I4
    // audit measured that clause: `image_diff` averages `|Δ|` over a **64 × 36**
    // downscale, so a change confined to a 2 903-pixel object can move the mean
    // by at most `3 × 2903 / 900 / 6912 = 0.0014` — **43× under the 0.06
    // tolerance at its arithmetic maximum**. The clause could not fail for any
    // change to this building, whatever the LOD did. A frame-scale tolerance
    // cannot arm a claim about an object that is 0.14 % of the frame.
    //
    // So the refusal is armed on the object: the fraction of the building's own
    // pixels that move, and the worst channel step. Both are measured (2.2 %,
    // 18/255) and both can fail — a shell that rendered as a different silhouette
    // or a different colour moves them immediately. The frame-scale pair is kept
    // *beside* them, because it is what the repository means by "a viewer would
    // not call these two frames different", with its own ceiling written down.
    let moved_pct = differing as f64 / covered.max(1) as f64 * 100.0;
    assert!(
        moved_pct <= 5.0,
        "the parts->shell swap moves {moved_pct:.1} % of the building's own {covered} pixels (measured at 2.2 % when island wave I4 refused a cross-fade on the strength of it). The refusal no longer has evidence, and the band pair needs the dither the I3 ledger describes."
    );
    assert!(
        worst <= 32,
        "the parts->shell swap's worst channel step is {worst}/255 against the 18/255 island wave I4 refused a cross-fade on. A step this size is a visible edge on a 33-pixel-tall building."
    );
    assert!(
        mean <= inf_render::golden::GOLDEN_MEAN_TOLERANCE
            && max <= inf_render::golden::GOLDEN_MAX_TOLERANCE,
        "the parts->shell swap now moves the frame by mean {mean:.5} / max {max:.5}, past the golden harness's own re-render tolerance — which for an object this small takes a change far larger than the LOD swap itself."
    );

    // **THE TWO CARRIED BOUNDS, RE-AIMED AT THE DISTANCE THEY WERE ALWAYS
    // MEASURED AT** (island wave I8c).
    //
    // I4 and I4b armed two claims about the *shipped* reading — a silhouette
    // `> 4x` the mesh's, and `> 50 %` of it moving — and both were measured at
    // **192 m**, because that is what `STRUCTURE_LOD_M` was. Neither was ever a
    // claim about the swap: they are claims about what a **bounding-sphere card**
    // looks like beside the geometry it stands in for, and the swap merely
    // happened to be inside the impostor band. I8c re-ordered the bands (96 m
    // against a 120 m `mesh_distance_m`), so the swap is now inside the *mesh*
    // band and the card lives past it. The two bounds move to `2 x
    // STRUCTURE_LOD_M`, which is the same 192 m, and re-read to the same numbers:
    // **26 792 px against 2 903 (9.2x), 91.5 % moving** — I4b's own figures, to
    // the pixel.
    //
    // The `> 4x` bound stays at four for I4b's reason: the arm is about "a
    // billboard is much bigger than the silhouette it stands in for", which is
    // what an impostor *is*, and a bound re-tightened onto today's ratio would go
    // red for an improvement.
    assert!(
        far_covered > far_mesh_covered * 4,
        "at {:.1} m the shipped impostor silhouette ({far_covered} px) is no longer far larger than the same building's mesh silhouette ({far_mesh_covered} px) — island wave I4 carried that ratio as a bound on the impostor's SIZING and I8c moved it to this distance; the ledger needs re-reading",
        STRUCTURE_LOD_M * 2.0
    );
    let far_pct = far_moved as f64 / far_covered.max(1) as f64 * 100.0;
    assert!(
        far_pct > 50.0,
        "at {:.1} m the parts->shell swap now moves only {far_pct:.1} % of the building's impostor silhouette, against the 91.6 % island wave I4b measured after re-sizing the card. The band pair has gained a fade, or the impostor has become something other than a bounding-sphere billboard; either way the refusal of a band-pair cross-fade was conditional on this number and has to be re-taken.",
        STRUCTURE_LOD_M * 2.0
    );
    // …and the far row is a real reading rather than an empty frame.
    assert!(
        far_mesh_covered > 100,
        "only {far_mesh_covered} px of building at {:.1} m — the far comparison is between two skies",
        STRUCTURE_LOD_M * 2.0
    );

    // **AND THE ORDERING, IN PIXELS** — the claim island wave I8c earns, which no
    // arm above makes.
    //
    // `the_structure_swap_happens_inside_the_scatter_mesh_band` compares two
    // constants; this is the claim that the renderer *acts* on them, which is the
    // P21.4 law this file exists under. At the swap the shipped picture is close
    // to the geometry picture; at twice it, it is the `> 4x` card above. The
    // bound is `< 2x` and the measurement is **1.37x**, and the gap between 1.0
    // and 1.37 is the FIXTURE rather than the band: this building is 30 m deep,
    // so a shot from 96 m spans 81 m to 111 m and its far wall is already inside
    // the 100–120 m mesh→impostor fade. A shallower building would read 1.00.
    let swap_ratio = imp_covered as f64 / covered.max(1) as f64;
    let far_ratio = far_covered as f64 / far_mesh_covered.max(1) as f64;
    assert!(
        swap_ratio < 2.0 && swap_ratio < far_ratio * 0.5,
        "the shipped silhouette at the {STRUCTURE_LOD_M} m swap is {swap_ratio:.2}x the geometry's against {far_ratio:.1}x at twice the distance — the swap is drawing a card again, so island wave I8b's band-ordering defect has returned"
    );

    // **AND THE SHIPPED POP AT THE SWAP, ARMED** (the island wave I8c audit).
    //
    // I4 armed `imp_pct > 50` on the shipped reading at the swap, and I8c moved
    // that bound out to `2 x STRUCTURE_LOD_M` with the card — which left the
    // number a *player* sees at the swap distance asserted by nothing at all. It
    // is the ordering's own dividend and it is worth a tripwire: **91.5 % of a
    // 26 792 px silhouette moved at the old 192 m swap and 29.6 % of an 18 884 px
    // one moves at 96 m**, worst channel 172/255 against 159/255. The bound is
    // the far row's own `> 50 %` turned around, so the two arms cannot both be
    // satisfied by a renderer that has stopped distinguishing the bands.
    let swap_pct = imp_moved as f64 / imp_covered.max(1) as f64 * 100.0;
    assert!(
        swap_pct < 50.0,
        "the SHIPPED parts->shell swap at {STRUCTURE_LOD_M} m moves {swap_pct:.1} % of the building's {imp_covered} px silhouette, against the 29.6 % island wave I8c's band re-order left it at and the 91.5 % it was at the 192 m swap. The swap has drifted back out past the scatter mesh band, or the mesh band has been narrowed under it"
    );
}

/// **A BUILDING IS NEVER A BALL** (the EDIT1 audit) — the shell batch as
/// `push_shells` actually bands it, at the distance the author sees it.
///
/// # What the author reported, and what it was
///
/// The showcase island's editor and PIE frames both carry a row of smooth white
/// **domes** standing among Harbour City's buildings: evenly spaced, roughly
/// half-buried, at building scale. Wave EDIT1 measured that they are not the P17
/// cloud slab (`clouds_enabled` is `false` on that level, so the cloud pass never
/// runs) and carried them as `PrimMesh::Sphere` "from a scatter or foliage
/// palette, drawn where a real mesh was meant to be".
///
/// **Nothing in the island places a sphere primitive**, and the scatter path's
/// missing-mesh placeholder is a `PrimMesh::Cube`. The domes are the subject of
/// this very file seen from the other side: the per-building **shell** box,
/// banded `[STRUCTURE_LOD_M, draw_distance)`, drawn past `mesh_distance_m` as a
/// scatter IMPOSTOR — a screen-facing card of the box's bounding-sphere radius,
/// shaded with a spherical normal so that it reads as a solid ball
/// (`vs_impostor`: "the card shades as a blob of the right size rather than as a
/// flat sticker"), centred at the building's mid-height so its lower half is
/// under the ground, and tinted from `pcg_kind_color`'s five-entry debug palette.
/// The arms above have printed its size beside the geometry's since island wave
/// I4; what nobody had done was band the fixture the way the engine bands it and
/// look at the result.
///
/// # Why this is a defect where the cross-fade refusal is not
///
/// The refusal above is about the *parts to shell* swap and is re-taken on
/// geometry numbers every wave. This is a different claim: a shell is ALREADY
/// the coarse tier of a complementary LOD pair — one box standing in for the
/// ~1 500 that are no longer drawn — so a card is not a cheaper stand-in for
/// anything, it is an approximation of an approximation. And it is not even a
/// trade: a shell is one twelve-triangle box per building against a card's two,
/// while the card covers 9.2x the fill.
///
/// # THE ARM
///
/// At `2 x STRUCTURE_LOD_M`, well inside the impostor band, the SHIPPED
/// configuration and the impostor-free one must draw the shell **identically** —
/// the same box, to the pixel. Before the rule they differed by the whole disc.
#[test]
fn a_far_lod_shell_is_never_replaced_by_a_card() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP structure_lod_pop: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let target = HeadlessTarget::new(&gpu, W, H);

    let shoot = |s: &RenderScene, d: f64, impostors: bool| -> Vec<u8> {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        let mut settings = inf_render::RenderSettings::default();
        settings.scatter.impostors = impostors;
        r.set_settings(settings);
        for _ in 0..3 {
            r.render(&gpu, s, &view(d), &target.view, (W, H));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        target.read_rgba(&gpu).expect("read back")
    };

    // The shell batch AS `push_shells` BANDS IT. `scene()` leaves
    // `near_distance` at zero, which is an ordinary scatter batch and not the
    // thing the island draws; the inner cut is what makes this the far half of a
    // complementary pair, and `inf-player`'s island gate identifies the shell
    // batch by exactly this field.
    let mut far_lod = scene(vec![shell()]);
    far_lod.scatter[0].near_distance = STRUCTURE_LOD_M;
    far_lod.mark_dirty();

    // Twice the swap distance: inside the impostor band by 72 m, which is where
    // the author's domes stand.
    let d = STRUCTURE_LOD_M * 2.0;
    let empty = shoot(&scene(Vec::new()), d, true);
    let shipped = shoot(&far_lod, d, true);
    let geometry = shoot(&far_lod, d, false);

    let (shipped_px, sw, sh) = silhouette(&shipped, &shipped, &empty);
    let (geom_px, gw, gh) = silhouette(&geometry, &geometry, &empty);
    let (differ, worst) = moved(&shipped, &geometry);
    println!(
        "EDIT1 audit - the far-LOD SHELL on {} ({:?}), {W}x{H}, at {d:.1} m:",
        info.name, info.device_type
    );
    println!(
        "  shipped   {sw:>4} x {sh:>4} px ({shipped_px:>6} px)
  geometry  {gw:>4} x {gh:>4} px ({geom_px:>6} px)   ratio {:.2}x
  {differ} px differ between them, worst channel {worst}/255",
        shipped_px as f64 / geom_px.max(1) as f64,
    );

    // ANTI-VACUITY: the building is really on screen as geometry, so "they
    // match" is not two empty frames matching.
    assert!(
        geom_px > 100,
        "only {geom_px} px of shell at {d:.1} m as geometry - the comparison is between two skies"
    );
    // THE CLAIM. Identical, not merely similar: with the impostor band refused
    // for this batch the two configurations write the same uniforms and run the
    // same pipeline, and this file has already asserted the renderer is
    // bit-deterministic across two fresh renderers.
    assert_eq!(
        (differ, worst),
        (0, 0),
        "at {d:.1} m the shipped configuration drew the shell as something other than its own geometry: {shipped_px} px against {geom_px} px ({:.1}x), {differ} px differing by up to {worst}/255. A shell is the FAR half of a complementary LOD pair and must be rasterized - see `effective_bands`. This is the building-sized white dome the EDIT1 audit found standing in Harbour City",
        shipped_px as f64 / geom_px.max(1) as f64,
    );
}
