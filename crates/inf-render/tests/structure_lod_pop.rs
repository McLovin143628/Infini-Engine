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
        "  NOTE: `ScatterSettings::default().mesh_distance_m` is {} m, so at the {STRUCTURE_LOD_M} m structure swap the scatter path is ALREADY in its impostor band. Both readings are given.",
        inf_render::RenderSettings::default().scatter.mesh_distance_m
    );
    let mut at_swap = None;
    let mut at_swap_mesh = None;
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
            if (d - STRUCTURE_LOD_M).abs() < 1.0e-9 {
                let row = (mean, max, differing, covered, worst, bw, bh);
                if impostors {
                    at_swap = Some(row);
                } else {
                    at_swap_mesh = Some(row);
                }
            }
        }
    }

    let (imp_mean, imp_max, imp_moved, imp_covered, imp_worst, imp_w, imp_h) =
        at_swap.expect("the impostor row ran");
    let (mean, max, differing, covered, worst, bw, bh) = at_swap_mesh.expect("the mesh row ran");

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
        "  …and the finding beside it: with the SHIPPED impostor band on (`mesh_distance_m` {} m < {STRUCTURE_LOD_M} m), the same building's silhouette is {imp_w} x {imp_h} px / {imp_covered} px — {:.1}x the mesh's {covered} px — and {imp_moved} of it ({:.1} %) moves at the swap, worst channel {imp_worst}/255, mean {imp_mean:.5} / max {imp_max:.5}. Both frames there are impostors, so the CHANGE is still the band pair's; what the impostor owns is its SIZE — a billboard sized from the instance's bounding sphere. The repair that comes first is the sizing, not the fade.",
        inf_render::RenderSettings::default().scatter.mesh_distance_m,
        imp_covered as f64 / covered.max(1) as f64,
        imp_moved as f64 / imp_covered.max(1) as f64 * 100.0,
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

    // **THE CARRIED BOUND, ARMED.** The impostor silhouette being many times the
    // mesh's is a measured fact this wave records rather than fixes — a scatter
    // impostor is sized from the instance's bounding sphere, and a 20 x 30 x 7.4 m
    // box's sphere is much wider than the box. The day that changes, this arm
    // says so instead of the ledger quietly going stale.
    //
    // *The attribution, corrected by the I4 audit.* Both frames in the shipped
    // reading are drawn as impostors, so the change between them is the **band
    // pair's** — parts against shell — seen through a billboard. What the
    // impostor owns is the change's SIZE, not its cause: it is what turns a
    // 2 903-pixel difference into a 26 792-pixel one. The first write-up said the
    // discontinuity "belongs to the impostor band rather than to the structure
    // band pair", which sends the next reader to the wrong repair.
    //
    // **The bound is `> 4x` and the measurement is 9.2x** (island wave I4b sized
    // the card to the instance's own bounding sphere and halved it from 19.2x).
    // It stays at 4x because what the arm is about is "a billboard is much bigger
    // than the silhouette it stands in for", which is still true and is what an
    // impostor is; a bound re-tightened to 9x would go red for an *improvement*.
    assert!(
        imp_covered > covered * 4,
        "the impostor silhouette ({imp_covered} px) is no longer far larger than the mesh's ({covered} px) — island wave I4 carried that ratio as the reason the discontinuity at {STRUCTURE_LOD_M} m is so much bigger than the geometry reading, and the ledger needs re-reading"
    );
    // **AND THE SHIPPED POP IS NOT SMALL, ARMED AS THE REFUSAL'S CONDITION.**
    // The refusal above is a statement about geometry; the configuration that
    // ships draws both sides as impostors and moves 91.6 % of a silhouette 9.2x
    // the mesh's.
    //
    // **The condition FIRED and the refusal was re-taken** (island wave I4b).
    // I4 wrote this arm so that repairing the billboard's sizing would turn the
    // file red and force clause 6 to be re-decided. I4b repaired it — the ratio
    // went 19.2x -> 9.2x — and the re-decision is: the refusal STANDS, because
    // the geometry reading it rests on did not move at all (63 of 2 903 px), and
    // what remains is a bounding-sphere card, which is what an impostor is rather
    // than a defect it has. So the arm is kept, aimed at the repaired numbers.
    let imp_pct = imp_moved as f64 / imp_covered.max(1) as f64 * 100.0;
    assert!(
        imp_pct > 50.0,
        "the SHIPPED parts->shell swap now moves only {imp_pct:.1} % of the building's impostor silhouette, against the 91.6 % island wave I4b measured after re-sizing the card. The band pair has gained a fade, or the impostor has become something other than a bounding-sphere billboard; either way the refusal of a band-pair cross-fade was conditional on this number and has to be re-taken."
    );
}
