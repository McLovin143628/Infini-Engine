//! **The P28.1 parity nucleus**: the visibility-buffer path shades meshlets at
//! golden-parity with the forward path, per pixel, on a real device.
//!
//! The phase's "Done when" opens with that sentence, and this file is what makes
//! it a claim rather than a hope. Every arm renders ONE scene twice — once
//! through `vgeom_mesh.wgsl`'s fragment stage and once through
//! `visbuffer.wgsl` + `vis_resolve.wgsl` — and compares the two images byte for
//! byte.
//!
//! # Why the comparison is trustworthy
//!
//! The two paths are **independent constructions**. `vis_resolve.wgsl` is not a
//! refactor of `vgeom_mesh.wgsl`, does not `include` it, and does not share a
//! function with it: it re-derives the vertex pull, solves for barycentrics the
//! forward path gets from the rasterizer's interpolators, and computes gradients
//! the forward path gets from `dpdx`. What the two share is their INPUT — one
//! flat instance table, one set of meshlet pools, one `view.view_proj` — because
//! a parity claim about two programs fed different data is about nothing.
//!
//! That independence is the point. P27.5's closing law is that *a gate measures
//! agreement between subsystems and cannot see an error the subsystems share*;
//! the forward path is retained by the ROADMAP as the fallback AND as this
//! gate's oracle precisely so that there is a second derivation to disagree with.
//!
//! # Why it compares INTERIOR pixels
//!
//! [`crate::renderer::SCENE_SAMPLES`] is a compile-time `4` and the visibility
//! buffer is single-sample by the ROADMAP's own clause 1. At a silhouette the two
//! are *obliged* to differ: the forward path resolves four samples of partial
//! coverage, the resolve writes one colour to all four. Comparing there would be
//! measuring MSAA, not the resolve.
//!
//! So each arm asks the visibility buffer itself which pixels are interior — a
//! pixel whose four neighbours carry its own id is one the forward path's four
//! samples all lie inside, so it is shaded ONCE at the pixel centre and resolves
//! to four identical samples, which is exactly what the resolve computes. On
//! those pixels the two paths must agree **to the byte**, and every arm asserts
//! zero mismatches. The edge population is not swept under the rug: it is
//! measured by `the_edge_population_is_where_the_two_paths_are_allowed_to_differ`
//! and it is the whole content of the P28.1 MSAA ruling
//! (`docs/memos/p28-1-visbuffer.md` §5).
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::passes::visbuffer::VisImage;
use inf_render::visbuffer::{
    parity_ok, parity_verdict, ParityVerdict, VisPacking, VIS_EMPTY,
};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomMesh, VgeomSettings,
    HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;

const W: u32 = 320;
const H: u32 = 180;
const ASSET: u128 = 0x2801_0000_0715_0000;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP visbuffer_parity ({what}): no GPU adapter ({e})");
            None
        }
    }
}

// ── the fixture ──────────────────────────────────────────────────────────────

/// A 16 x 16 displaced grid — 512 triangles, several meshlets, a real DAG.
///
/// **Deliberately coarser than the occlusion suite's 48.** The parity comparison
/// lives on INTERIOR pixels, and a triangle's interior is what is left after its
/// one-pixel rim: a 48 x 48 grid at this viewport puts ~10 pixels under each
/// triangle, of which 81.6 % are rim (measured, first draft of this file), so the
/// population the arms compare over was a tenth of what it should be. Sixteen
/// puts ~40 pixels under a triangle and the interior population is most of the
/// coverage. The geometry under test is identical in kind; only the ratio of
/// silhouette to surface moved, and that ratio is the *test's* subject, not the
/// resolve's.
fn mesh() -> Arc<VgeomMesh> {
    Arc::new(dense_grid_mesh(16))
}

/// A dense grid quad stood up in world XY so it faces the camera and fills the
/// frame — the same fixture shape the vgeom goldens and the VT sampling suite
/// use. `variants` places that many instances side by side, each with its own
/// colour, metallic and roughness, which is the "per-instance-set variation"
/// arm's content: the resolve reads its material from the FLAT instance table
/// through the id it unpacked, so two instances of one asset differing only in
/// material is exactly the case a wrong instance field silently passes.
fn scene(variants: usize, alpha: f32, lights: bool) -> RenderScene {
    scene_vt(variants, alpha, lights, inf_render::VtTextureSet::NONE)
}

fn scene_vt(
    variants: usize,
    alpha: f32,
    lights: bool,
    vt: inf_render::VtTextureSet,
) -> RenderScene {
    let m = mesh();
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &m).expect("index the vmesh")],
        ..Default::default()
    };
    let standing = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    for i in 0..variants {
        let t = i as f32 / variants.max(1) as f32;
        let mut inst = VgeomInstance::lit(
            ASSET,
            // Spread in x AND stepped back in z. The z step is load-bearing:
            // instances at one depth overlap in a band where their displaced
            // surfaces are near-coplanar, and a 4x-MSAA depth buffer and a 1x one
            // resolve a z-fight differently — which showed up as interior pixels
            // differing by 40 of 255 and reads exactly like a resolve defect. It
            // is a fixture defect. 2 m of separation against a +/-0.66 m
            // displacement makes the depth order unambiguous for both.
            DVec3::new(
                (i as f64 - (variants as f64 - 1.0) * 0.5) * 2.4,
                0.0,
                -(i as f64) * 2.0,
            ),
            standing,
            Vec3::splat(2.2),
            [0.2 + 0.7 * t, 0.65 - 0.4 * t, 0.30 + 0.5 * t, alpha],
            i as u32 + 1,
        );
        inst.metallic = 0.1 + 0.8 * t;
        inst.roughness = 0.15 + 0.7 * (1.0 - t);
        inst.vt = vt;
        sc.vgeom_instances.push(inst);
    }
    if lights {
        sc.lights.push(RenderLight {
            kind: LightKind::Directional,
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
            position: DVec3::ZERO,
            range: 0.0,
            ..RenderLight::default()
        });
        sc.lights.push(RenderLight {
            kind: LightKind::Point,
            color: [0.4, 0.7, 1.0],
            intensity: 8.0,
            position: DVec3::new(2.0, 1.5, 3.0),
            range: 20.0,
            ..RenderLight::default()
        });
        sc.lights.push(RenderLight {
            kind: LightKind::Spot,
            color: [1.0, 0.6, 0.3],
            intensity: 10.0,
            position: DVec3::new(-2.0, 2.0, 4.0),
            direction: Vec3::new(0.3, -0.4, -0.8).normalize(),
            range: 25.0,
            inner_cos: 30f32.to_radians().cos(),
            outer_cos: 40f32.to_radians().cos(),
            ..RenderLight::default()
        });
    }
    sc.mark_dirty();
    sc
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 7.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Meshlet settings with occlusion off. Two-pass HZB is a temporal feature and
/// this suite is about one frame's shading; the occlusion-on case is
/// `vgeom_occlusion.rs`'s subject and its contract (purely subtractive) means it
/// cannot change what either path shades.
fn settings(visbuffer: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion: false,
            two_pass: false,
            visbuffer,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

struct Frame {
    rgba: Vec<u8>,
    ids: Option<VisImage>,
    audit: inf_render::passes::visbuffer::VisAudit,
}

fn render(gpu: &GpuContext, sc: &RenderScene, st: RenderSettings, ids: bool) -> Frame {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(st);
    r.set_visbuffer_readback(ids);
    let v = view();
    r.render(gpu, sc, &v, &target.view, (W, H));
    Frame {
        rgba: target.read_rgba(gpu).expect("readback"),
        ids: if ids { r.read_visbuffer(gpu) } else { None },
        audit: r.vis_audit(),
    }
}

/// **The parity bound, and why it is one least-significant step rather than
/// zero.**
///
/// The forward path's barycentrics come from the rasterizer's interpolators; the
/// resolve solves for them. Those are the same numbers in exact arithmetic and
/// not always the same `f32`, and the frame is written to an **8-bit** target, so
/// a pixel whose shaded value sits within an f32 ulp of a `1/255` boundary can
/// round the other way. That is a quantization artefact of the comparison, not a
/// disagreement about shading — and the way to say so honestly is to bound it at
/// exactly one step and forbid two.
///
/// Measured on this fixture: 10 to 20 of ~2 700 interior pixels differ, **always
/// by exactly 1**, never in more than one channel's low bit. A wrong barycentric
/// or a wrong gradient does not look like this — mutation-verified in the P28.1
/// ledger, where transposing the resolve's `d_dx`/`d_dy` puts the worst delta at
/// 60+ and the differing fraction past a third.
use inf_render::visbuffer::PARITY_MAX_STEP;

/// Forward vs resolved on every interior pixel.
///
/// **The criterion's own classifier**, not a second copy of it (P28.5 audit).
/// `inf_render::visbuffer::parity_verdict` walks the same interior set this file
/// used to walk by hand; sharing the five constants left two implementations of
/// *bordering* and *solid centre* free to drift, which is the fork the "one
/// door" claim says does not exist.
///
/// The caller asserts the population was not empty as well — a parity claim over
/// nothing is the vacuous shape the P19 audit named.
fn compare_interior(fwd: &[u8], res: &[u8], ids: &VisImage) -> ParityVerdict {
    parity_verdict(fwd, res, ids.width, &ids.interior())
}

/// The assertion every matrix row makes, in one place so the bound is stated
/// once and cannot drift between rows — and it is
/// [`inf_render::visbuffer::parity_ok`], the same predicate `phase28_gate` holds
/// its rows to.
fn assert_parity(label: &str, v: &ParityVerdict, floor: usize) {
    assert!(
        v.interior > floor,
        "{label}: only {} interior pixels; a parity claim over a handful of \
         pixels is not one",
        v.interior
    );
    parity_ok(v, false).unwrap_or_else(|e| panic!("{label}: {e}"));
    eprintln!("parity {label}: {}", v.summary());
}

/// One matrix row. Renders both paths, checks the visibility path was actually
/// taken, and asserts byte parity over a non-trivial interior population.
fn parity_row(gpu: &GpuContext, label: &str, sc: &RenderScene) -> (usize, usize) {
    let fwd = render(gpu, sc, settings(false), false);
    let res = render(gpu, sc, settings(true), true);
    assert_eq!(
        res.audit.refused(),
        0,
        "{label}: the visibility path was refused ({:?})",
        res.audit
    );
    assert!(
        res.audit.frames > 0,
        "{label}: no frame rasterized a visibility buffer, so this row compared \
         the forward path with itself"
    );
    let ids = res.ids.as_ref().expect("readback was on");
    let covered = ids.covered();
    assert!(
        covered > (W * H / 20) as usize,
        "{label}: the visibility buffer covers only {covered} of {} pixels — the \
         fixture is not on screen and the comparison below is about the sky",
        W * H
    );
    let v = compare_interior(&fwd.rgba, &res.rgba, ids);
    assert_parity(label, &v, (W * H / 100) as usize);
    (v.interior, covered)
}

// ── the matrix ───────────────────────────────────────────────────────────────

/// **Untextured, scalar material, no lights** — the analytic sun path
/// (`lights.count == 0`), which is where `shade_light` runs against
/// `view.sun_dir` and nothing else.
#[test]
fn parity_untextured_scalar() {
    let Some(gpu) = gpu_or_skip("untextured") else {
        return;
    };
    parity_row(&gpu, "untextured/scalar", &scene(1, 1.0, false));
}

/// **Per-instance-set variation** — four instances of ONE asset differing in
/// colour, metallic and roughness. The resolve reads its material from the flat
/// instance table through the instance field it unpacked, so an off-by-one there
/// shades every pixel with a neighbour's material and this is the arm that sees
/// it. A single-instance fixture cannot.
#[test]
fn parity_per_instance_variation() {
    let Some(gpu) = gpu_or_skip("instance variation") else {
        return;
    };
    let sc = scene(4, 1.0, false);
    let (_n, _) = parity_row(&gpu, "instance-set", &sc);
    // Anti-vacuity: the instances really do carry different materials, so
    // "they agree" is a claim about the decode and not about four copies of one
    // number.
    let mats: std::collections::BTreeSet<_> = sc
        .vgeom_instances
        .iter()
        .map(|i| {
            (
                i.metallic.to_bits(),
                i.roughness.to_bits(),
                i.color[0].to_bits(),
            )
        })
        .collect();
    assert_eq!(
        mats.len(),
        sc.vgeom_instances.len(),
        "the instances share a material, so this row would pass with the \
         instance field decoded wrong"
    );
}

/// **Masked** — instance alpha below one, which both paths carry through to the
/// output alpha channel untouched by lighting. The comparison includes the alpha
/// channel (`compare_interior` walks all four), so a resolve that dropped it
/// fails here and nowhere else.
#[test]
fn parity_masked_alpha() {
    let Some(gpu) = gpu_or_skip("masked") else {
        return;
    };
    parity_row(&gpu, "masked", &scene(2, 0.45, false));
}

/// **The full analytic light loop** — one directional, one point, one spot, so
/// the `count != 0` branch runs with all three kinds, `point_attenuation` and the
/// spot cone included.
#[test]
fn parity_three_light_kinds() {
    let Some(gpu) = gpu_or_skip("lights") else {
        return;
    };
    parity_row(&gpu, "dir+point+spot", &scene(3, 1.0, true));
}

/// **GI on** — `gi_irradiance` and `gi_ambient_specular` reached from the resolve
/// through the same environment group the forward path reads them from.
#[test]
fn parity_gi_on() {
    let Some(gpu) = gpu_or_skip("GI") else {
        return;
    };
    let sc = scene(2, 1.0, true);
    let mut st_off = settings(false);
    st_off.gi.enabled = true;
    let mut st_on = settings(true);
    st_on.gi.enabled = true;
    let fwd = render(&gpu, &sc, st_off, false);
    let res = render(&gpu, &sc, st_on, true);
    let ids = res.ids.as_ref().expect("readback on");
    let v = compare_interior(&fwd.rgba, &res.rgba, ids);
    assert_parity("GI on", &v, 100);
}

/// **Shadows on** — the sun's `shadow_factor`, reached through the same one door
/// `vsm_receive.wgsl` declares. With the cascade enabled `sun_shadowing_enabled()`
/// is true, so both paths take the multiply and a resolve that had skipped it
/// would show as a whole-surface brightness difference.
#[test]
fn parity_sun_shadowed() {
    let Some(gpu) = gpu_or_skip("shadows") else {
        return;
    };
    let sc = scene(2, 1.0, true);
    let mut st_off = settings(false);
    st_off.shadows.enabled = true;
    let mut st_on = settings(true);
    st_on.shadows.enabled = true;
    let fwd = render(&gpu, &sc, st_off, false);
    let res = render(&gpu, &sc, st_on, true);
    let ids = res.ids.as_ref().expect("readback on");
    let v = compare_interior(&fwd.rgba, &res.rgba, ids);
    assert_parity("sun shadowed", &v, 100);
}

// ── determinism, on the device ───────────────────────────────────────────────

/// **Two runs, identical buffers.** The ROADMAP's clause 1 says depth-tested
/// last-writer-wins keeps determinism by the same argument today's opaque raster
/// rests on; this is that argument pinned rather than asserted.
///
/// The argument, stated where it can be checked: the id a texel holds is the id
/// of the fragment that won the depth test, and the depth test is `Greater`
/// against a buffer cleared to a constant. Two fragments can tie only if they are
/// coplanar to the last bit of the f32, and a tie is broken by primitive order —
/// which is the cull compute's append order, and *that* is not ordered. So the
/// guarantee is the one the opaque path makes and no more: the same draw stream
/// replays to the same buffer, because the same appends happen in the same order
/// for the same committed input.
#[test]
fn a_visbuffer_is_bit_identical_across_two_runs() {
    let Some(gpu) = gpu_or_skip("determinism") else {
        return;
    };
    let sc = scene(3, 1.0, true);
    let a = render(&gpu, &sc, settings(true), true);
    let b = render(&gpu, &sc, settings(true), true);
    let (ia, ib) = (a.ids.expect("ids a"), b.ids.expect("ids b"));
    assert_eq!(
        ia.covered(),
        ib.covered(),
        "the two runs covered different pixel counts"
    );
    assert!(ia.covered() > 1000, "the fixture drew almost nothing");
    assert_eq!(
        ia.ids, ib.ids,
        "two runs of one committed scene produced different visibility buffers"
    );
    // …and the frames they shade from agree too, which is the half a buffer
    // comparison alone does not cover.
    assert_eq!(a.rgba, b.rgba, "the resolved frames differ");
}

// ── the packing, on the device ───────────────────────────────────────────────

/// **Every id the device wrote decodes to a real instance and a real triangle.**
///
/// The Rust round-trip arms prove `unpack(pack(x)) == x`; this proves the *shader*
/// packs what the Rust unpacks, which is a different claim and the one that
/// catches a shifted field. It is measured against the scene: the instance field
/// must name an instance the scene placed, and the triangle field must be inside
/// the asset's cooked maximum.
#[test]
fn the_devices_ids_decode_to_the_scene_the_host_placed() {
    let Some(gpu) = gpu_or_skip("id decode") else {
        return;
    };
    let sc = scene(4, 1.0, false);
    let n_inst = sc.vgeom_instances.len() as u32;
    let f = render(&gpu, &sc, settings(true), true);
    let ids = f.ids.expect("ids");
    let mut seen_instances = std::collections::BTreeSet::new();
    let mut max_tri = 0u32;
    let mut empties = 0usize;
    for &id in &ids.ids {
        match VisPacking::unpack(id) {
            None => {
                empties += 1;
                assert_eq!(id, VIS_EMPTY, "a non-empty id unpacked to nothing");
            }
            Some(f) => {
                assert!(
                    f.instance < n_inst,
                    "the device wrote instance {} and the scene placed {n_inst}",
                    f.instance
                );
                seen_instances.insert(f.instance);
                max_tri = max_tri.max(f.tri);
            }
        }
    }
    assert!(
        empties > 0,
        "the fixture covers the whole frame, so the empty \
             sentinel is never exercised — move the camera back"
    );
    assert_eq!(
        seen_instances.len() as u32,
        n_inst,
        "only {:?} of the {n_inst} instances reached the buffer; an instance \
         field that decoded constant would look exactly like this",
        seen_instances
    );
    assert!(
        max_tri < 128,
        "the device wrote triangle {max_tri}, past the 7-bit field's range"
    );
    eprintln!(
        "device ids: {} instances, max tri {max_tri}, {empties} empty of {}",
        seen_instances.len(),
        ids.ids.len()
    );
}

// ── the MSAA measurement (clause 3's evidence) ───────────────────────────────

/// **The edge population, measured.** This is the number the P28.1 MSAA ruling
/// is made of, and the arm exists so the memo cites a measurement rather than an
/// estimate.
///
/// Covered pixels split into interior (the two paths must agree) and edge (they
/// are obliged to differ, because one resolves four samples of partial coverage
/// and the other writes one colour to all four). The arm asserts the split is
/// real in BOTH directions — a fixture with no edges would make every parity row
/// above vacuous, and a fixture that is all edge would make them empty — and
/// prints the ratio.
#[test]
fn the_edge_population_is_where_the_two_paths_are_allowed_to_differ() {
    let Some(gpu) = gpu_or_skip("edge cost") else {
        return;
    };
    let sc = scene(3, 1.0, true);
    let fwd = render(&gpu, &sc, settings(false), false);
    let res = render(&gpu, &sc, settings(true), true);
    let ids = res.ids.as_ref().expect("ids");
    let covered = ids.covered();
    let interior = ids.interior().len();
    let edge = covered - interior;
    assert!(
        interior > 0 && edge > 0,
        "the fixture has {interior} interior and {edge} edge pixels; a split \
         that is empty on either side makes the parity rows either vacuous or \
         impossible"
    );
    // How many of the EDGE pixels actually differ, which is the cost the ruling
    // weighs. Counted, not asserted to a threshold: it is a property of the
    // fixture's silhouette density, and freezing it would be freezing the
    // fixture.
    let mut edge_bad = 0usize;
    let interior_set: std::collections::BTreeSet<(u32, u32)> = ids.interior().into_iter().collect();
    for y in 0..ids.height {
        for x in 0..ids.width {
            if ids.at(x, y) == VIS_EMPTY || interior_set.contains(&(x, y)) {
                continue;
            }
            let i = ((y * ids.width + x) * 4) as usize;
            if (0..4).any(|c| fwd.rgba[i + c] != res.rgba[i + c]) {
                edge_bad += 1;
            }
        }
    }
    let frame = (W * H) as f64;
    eprintln!(
        "P28.1 MSAA ruling evidence: covered {covered} ({:.1}% of frame), \
         interior {interior}, edge {edge} ({:.1}% of covered), \
         edge pixels differing {edge_bad} ({:.2}% of frame)",
        100.0 * covered as f64 / frame,
        100.0 * edge as f64 / covered as f64,
        100.0 * edge_bad as f64 / frame,
    );
    // The claim the ruling rests on, asserted rather than only printed: the
    // difference between the two paths lives at the edges and nowhere else.
    let v = compare_interior(&fwd.rgba, &res.rgba, ids);
    let (n, interior_bad) = (v.interior, v.differing);
    assert_parity("edge-arm interior", &v, 100);
    // …and the edges are where the difference actually is, by a wide margin. The
    // ratio is the ruling's content: a single-sample buffer costs silhouettes and
    // costs surfaces nothing.
    let edge_rate = edge_bad as f64 / edge.max(1) as f64;
    let interior_rate = interior_bad as f64 / n.max(1) as f64;
    assert!(
        edge_rate > 10.0 * interior_rate,
        "edge pixels differ at {:.1} % and interior at {:.1} % — the ruling \
         rests on those being different populations",
        100.0 * edge_rate,
        100.0 * interior_rate
    );
}

// ── the refusal, end to end ──────────────────────────────────────────────────

/// **A scene the packing cannot address renders through the forward path**, and
/// the counter says which ceiling. The refusal is the P27.1 grid-overflow law's
/// shape: typed, at registration, and falling back rather than failing.
///
/// Driven by an asset whose cooked meshlets exceed the triangle field, which is
/// the one ceiling a test can reach without placing two thousand instances.
#[test]
fn a_scene_past_a_ceiling_falls_back_to_the_forward_path_and_says_which() {
    let Some(gpu) = gpu_or_skip("refusal") else {
        return;
    };
    // 256 triangles a meshlet — legal for `meshopt` (its cap is 512, divisible
    // by four), past the 7-bit field. Built through the same generator the rest
    // of this file uses, so the ONLY thing that differs is the parameter the
    // ceiling is about.
    let (pos, nrm, uv, idx) = inf_vgeom::test_support::displaced_grid(
        48,
        inf_vgeom::test_support::DEFAULT_AMPLITUDE,
        inf_vgeom::test_support::GridNormals::Analytic,
    );
    let m = Arc::new(inf_vgeom::build_vgeom(
        &pos,
        &nrm,
        &uv,
        &[],
        &idx,
        inf_vgeom::BuildParams {
            // BOTH limits: `max_vertices` binds first at the default 64 (a
            // 64-vertex meshlet holds ~124 triangles whatever the triangle cap
            // says), so raising only `max_triangles` produces meshlets the
            // packing is perfectly happy with — measured, and it made the first
            // draft of this arm assert a refusal that never happened.
            max_vertices: 255,
            max_triangles: 256,
            ..inf_vgeom::BuildParams::default()
        },
    ));
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &m).expect("index")],
        ..Default::default()
    };
    sc.vgeom_instances.push(VgeomInstance::lit(
        ASSET,
        DVec3::ZERO,
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Vec3::splat(2.0),
        [0.6, 0.6, 0.6, 1.0],
        1,
    ));
    sc.mark_dirty();
    let over = sc.vgeom_assets[0].source.max_tri();
    assert!(
        over > 128,
        "the fixture cooked {over} triangles a meshlet, inside the 7-bit field —          meshopt did not honour the parameter and this arm would assert a          refusal that cannot happen"
    );
    let refused = render(&gpu, &sc, settings(true), true);
    assert_eq!(
        refused.audit.refused_triangles, 1,
        "the over-wide asset was not refused by the triangle ceiling ({:?})",
        refused.audit
    );
    assert_eq!(refused.audit.frames, 0, "a refused frame rasterized ids");
    assert_eq!(
        refused.audit.refused_instances + refused.audit.refused_meshlet_slots,
        0,
        "the refusal was counted against the wrong ceiling — the P27.5 ruling is \
         that a single counter would wear both names"
    );
    // …and the frame is the forward path's, byte for byte. This is the half that
    // makes "falls back" a claim: a refusal that dropped the geometry would also
    // satisfy every counter above.
    let forward = render(&gpu, &sc, settings(false), false);
    assert_eq!(
        refused.rgba, forward.rgba,
        "a refused frame is not the forward frame — the fallback dropped or \
         changed the geometry it was supposed to hand back"
    );

    // **THE ANTI-VACUITY THAT MATTERS, and the first draft did not have it**
    // (P28.1 audit). Two frames agreeing is not evidence that either drew
    // anything: both take the SAME `raster_draw` call site, so deleting that
    // call leaves them identically empty and the assertion above green.
    // Mutation-measured — the fallback's only draw was removed and this arm
    // survived, while every parity row died. The old guard (`p[3] > 0`) could
    // not see it either: the sky writes an opaque alpha over the whole frame.
    //
    // So the claim is made against a frame that provably has no meshlet geometry
    // in it: the same scene with its instances removed. The refused frame must
    // differ from THAT over a real population, which is the statement "the
    // fallback handed the geometry back" rather than "two paths lost it
    // together".
    let bare = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &m).expect("index")],
        ..Default::default()
    };
    let empty = render(&gpu, &bare, settings(false), false);
    let painted = refused
        .rgba
        .chunks_exact(4)
        .zip(empty.rgba.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        painted > (W * H / 20) as usize,
        "the refused frame differs from a geometry-free frame on only {painted} \
         of {} pixels — the fallback did not draw the asset it was handed, and \
         'the refused frame equals the forward frame' is two empty frames \
         agreeing",
        W * H
    );
    eprintln!("refusal fallback: {painted} pixels carry the handed-back geometry");
}

// ── the textured row (the analytic gradients' only device witness) ───────────

/// **Textured (virtual textures), the row the gradients live or die by.**
///
/// Every other row in the matrix shades from per-instance constants, and a
/// constant does not care what its screen derivative is: transpose the resolve's
/// `d_dx` and `d_dy` and they all still pass. `vt_surface` is the one consumer —
/// it takes the gradients as explicit arguments and picks a mip from them — so
/// this row is where a wrong gradient becomes a wrong texel and a visible
/// disagreement.
///
/// **And it is the row where the two paths are NOT expected to agree exactly**,
/// which is the honest finding this suite exists to produce rather than to hide.
/// The forward path's `dpdx(in.uv)` is a finite difference over a 2 x 2 quad —
/// one value shared by four pixels, first-order accurate. The resolve's is the
/// exact analytic derivative, per pixel. Those choose the same mip almost
/// everywhere and a different one where the footprint sits on a level boundary,
/// and a different mip is a different texel, which is a large delta rather than a
/// rounding step. So the bound here is **stated as a population**, not as a step:
/// the vast majority of interior pixels agree to within rounding, and the rest
/// are the mip-boundary set.
///
/// # What is claimed here, and what was withdrawn (P28.1 audit)
///
/// **Claimed and measured:** the resolve's gradient is the exact analytic
/// derivative and the forward path's is a first-order quad difference — asserted
/// on the Rust twin, in `crates/inf-render/src/visbuffer.rs`, by
/// `the_analytic_gradient_is_the_limit_of_the_finite_difference`, as a
/// convergence signature with a 400× wrong-formula control. That is a statement
/// about the *gradient*.
///
/// **Withdrawn:** "the resolve's number is the better one", as a statement about
/// the *image*. The audit built the measurement the claim needed — a 16×
/// supersampled forward reference, box-downsampled in linear space — and it does
/// not support it: over the 5 132 interior pixels the forward frame's mean
/// absolute error against the reference is 0.06554 and the resolve's is 0.06558,
/// and on the differing pixels alone the resolve is closer on **47.6 %** of them.
/// A coin flip. Nor could the design have found a small effect: the reference's
/// own noise floor, measured by running the identical comparison with **no
/// texture bound**, is **0.0868** — larger than the whole textured signal and
/// ~200× the difference between the two paths. Repeating it on the isolated
/// texture contribution (textured minus untextured, which cancels the
/// resolution-dependent shading) gives 51.2 %, the same coin flip.
///
/// So the honest bound is the one this arm asserts: the two agree except on a
/// **thin boundary set**, and which of them is nearer the truth there is
/// undecided by measurement.
#[test]
fn parity_textured_virtual_texture() {
    let Some(gpu) = gpu_or_skip("textured") else {
        return;
    };
    // A prime-period diagonal ramp: a transposed lookup, a mirrored row order or
    // an off-by-a-tile address all read a different value (the fixture-asymmetry
    // law `vt_sampling.rs` states).
    const N: u32 = 256;
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            rgba.extend_from_slice(&[
                ((x * 7 + y * 3) % 251) as u8,
                ((x * 11 + y * 29) % 241) as u8,
                ((x * 5 + y * 17) % 239) as u8,
                255,
            ]);
        }
    }
    let bytes = inf_material::build_tiled_texture(
        rgba,
        N,
        N,
        inf_material::TextureImportSettings {
            srgb: false,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
        },
    )
    .expect("the fixture tiles")
    .into_bytes();

    let run = |visbuffer: bool| {
        let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
            format: inf_vt::PageFormat::Rgba8,
            stored_tile_size: inf_vt::STORED_TILE_SIZE,
            budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 256,
            max_texture_dim: 8192,
        });
        lib.register_or_record(1, Arc::new(bytes.clone()))
            .unwrap_or_else(|| panic!("the fixture registers: {:?}", lib.refusals()));
        let pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(settings(visbuffer));
        r.set_visbuffer_readback(visbuffer);
        r.set_vt_level(Some((lib, pools)));
        let v = view();
        // Three frames: the registry warms, the floor pages in, and the third
        // draws through a resident set. Both paths get the same three.
        let mut rgba = Vec::new();
        for _ in 0..3 {
            let set = r
                .vt_textures()
                .map(|l| l.set_for(Some(1), None, None))
                .unwrap_or(inf_render::VtTextureSet::NONE);
            let sc = scene_vt(2, 1.0, true, set);
            r.render(&gpu, &sc, &v, &target.view, (W, H));
            rgba = target.read_rgba(&gpu).expect("readback");
        }
        (
            rgba,
            if visbuffer {
                r.read_visbuffer(&gpu)
            } else {
                None
            },
            r.vt_pop_in().admits,
        )
    };
    let (fwd, _, fwd_admits) = run(false);
    let (res, ids, res_admits) = run(true);
    assert!(
        fwd_admits > 0 && res_admits > 0,
        "nothing was paged in ({fwd_admits} / {res_admits}) — this row would \
         compare two untextured frames and call it parity"
    );
    let ids = ids.expect("readback on");
    let v = compare_interior(&fwd, &res, &ids);
    assert!(
        v.interior > 1000,
        "textured: only {} interior pixels",
        v.interior
    );
    eprintln!("parity textured: {}", v.summary());

    // **THE CRITERION, THROUGH THE ONE DOOR** (P28.5 audit). The bound and the
    // shape are `inf_render::visbuffer::parity_ok(_, true)` — the same predicate
    // `phase28_gate`'s row holds to, over the same `parity_verdict` classifier.
    //
    // Until this audit the two files shared the five CONSTANTS and each owned its
    // own implementation of *differing*, *bordering* and *solid centre*, which is
    // two spellings of the criterion wearing one door's name: a change to what
    // "bordering" means would have moved the gate and left the nucleus still
    // green. The population bound alone cannot tell a mip BOUNDARY — a curve one
    // or two pixels thick where the footprint crosses a power of two — from a
    // broad smear over whole triangles, which is what a wrong gradient produces;
    // the bordering and solid-centre clauses are what do, and they are now stated
    // exactly once in the tree.
    parity_ok(&v, true).unwrap_or_else(|e| panic!("textured: {e}"));

    // …and anti-vacuity in the other direction: the texture actually modulates
    // the surface, so "they agree" is a claim about sampling and not about two
    // untextured frames.
    let flat = {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(settings(true));
        let v = view();
        r.render(&gpu, &scene(2, 1.0, true), &v, &target.view, (W, H));
        target.read_rgba(&gpu).expect("readback")
    };
    assert_ne!(
        flat, res,
        "the textured frame is byte-identical to the untextured one, so nothing \
         was sampled and this row is about nothing"
    );
}

// ── the reconstructed depth (M8's answer) ────────────────────────────────────

/// **The resolve's depth is the rasterizer's**, and a rigid object interleaved
/// with the meshlet geometry is what makes that a claim.
///
/// The resolve does not store the visibility pass's depth; it recovers it from
/// the same barycentric solve, on the identity that `vis_bary` solves
/// `M · mu = (ndc.x, ndc.y, 1)` EXACTLY, so the reconstructed clip point is
/// `(ndc.x, ndc.y, dot(mu, z), 1)`. It then writes it as `@builtin(frag_depth)`
/// into the MSAA depth, which is where translucency, water and the shadow
/// marking pass expect to find meshlet depth.
///
/// **Every other row in this file is blind to it.** With only meshlet geometry in
/// the frame the MSAA depth buffer holds nothing else, so any depth passes the
/// test and any depth writes the same colour — mutation-measured: replacing the
/// whole expression with the constant `0.5` leaves `parity_three_light_kinds`
/// green. This arm puts a rigid cube through the middle of the meshlet grid, so
/// which of the two a pixel shows is decided by the depth comparison and nothing
/// else.
#[test]
fn parity_interleaved_with_rigid_geometry() {
    let Some(gpu) = gpu_or_skip("depth order") else {
        return;
    };
    let mut sc = scene(2, 1.0, true);
    // A slab between the two meshlet grids in depth: nearer than the far
    // instance, further than the near one, and wide enough on screen that a
    // mis-ordered frame is a population and not a rim.
    //
    // **Its z extent is chosen to INTERSECT neither**, and that is the arm's own
    // finding rather than caution. The first draft used a 1.4 m half-depth slab
    // at z = -1, which reaches z = -0.3 — inside the near grid's own
    // displacement, which spans +/-0.66 m about z = 0. Along the intersection
    // curve the two paths disagreed by up to 104 of 255 on **thirteen** pixels,
    // and that is not a resolve defect: a 4x MSAA depth buffer resolves an
    // intersection per SAMPLE and a single-sample reconstruction cannot, so the
    // two are obliged to differ exactly there — the same obligation the
    // silhouette carries, one geometric case over. Recorded in
    // `docs/memos/p28-1-visbuffer.md` §5 beside the edge measurement, because it
    // is the second thing a single-sample visibility buffer gives up.
    //
    // Near grid: z in [-0.66, 0.66]. Far grid: z in [-2.66, -1.34]. Slab:
    // z in [-1.15, -0.85]. Clear of both.
    sc.instances.push(inf_render::MeshInstance::lit(
        DVec3::new(0.0, 0.0, -1.0),
        Quat::IDENTITY,
        Vec3::new(6.0, 1.4, 0.3),
        [0.9, 0.2, 0.15, 1.0],
        900,
    ));
    sc.mark_dirty();
    let fwd = render(&gpu, &sc, settings(false), false);
    let res = render(&gpu, &sc, settings(true), true);
    let ids = res.ids.as_ref().expect("ids");
    // Anti-vacuity: the cube has to be VISIBLE and has to OCCLUDE meshlet
    // geometry, or the depth comparison this arm is about never happens.
    let occluded = (0..ids.height)
        .flat_map(|y| (0..ids.width).map(move |x| (x, y)))
        .filter(|&(x, y)| ids.at(x, y) == VIS_EMPTY)
        .count();
    assert!(
        occluded > 1000,
        "only {occluded} pixels are free of meshlet ids; the rigid cube is not \
         in the frame and this arm compares what every other row already does"
    );
    let v = compare_interior(&fwd.rgba, &res.rgba, ids);
    assert_parity("interleaved rigid", &v, 100);
    // …and the two frames agree about the CUBE's pixels too, which is the half
    // that fails when the resolve writes a wrong depth: a meshlet pixel that
    // should have lost the depth test paints over the cube.
    let mut cube_bad = 0usize;
    let mut cube_seen = 0usize;
    for y in 1..ids.height - 1 {
        for x in 1..ids.width - 1 {
            // Empty AND surrounded by empty — the mirror of `interior()`, and
            // needed for the mirror reason: a pixel whose CENTRE no meshlet
            // covers can still have MSAA samples that one does, so the forward
            // path blends a meshlet into it and the single-sample resolve cannot.
            // Measured: 128 of 45 274 empty pixels differ without this, all of
            // them one step off a silhouette.
            if ids.at(x, y) != VIS_EMPTY
                || ids.at(x - 1, y) != VIS_EMPTY
                || ids.at(x + 1, y) != VIS_EMPTY
                || ids.at(x, y - 1) != VIS_EMPTY
                || ids.at(x, y + 1) != VIS_EMPTY
            {
                continue;
            }
            let i = ((y * ids.width + x) * 4) as usize;
            cube_seen += 1;
            if (0..4).any(|c| fwd.rgba[i + c].abs_diff(res.rgba[i + c]) > PARITY_MAX_STEP) {
                cube_bad += 1;
            }
        }
    }
    assert_eq!(
        cube_bad, 0,
        "{cube_bad} of {cube_seen} pixels the visibility buffer calls EMPTY \
         differ between the two paths by more than a rounding step — the \
         resolve is writing a depth that loses or wins the wrong comparisons"
    );
    assert!(
        cube_seen > 10_000,
        "only {cube_seen} pixels are meshlet-free and surrounded by meshlet-free          pixels; the rigid slab and the background between them are what this          half of the arm is about"
    );
    eprintln!(
        "parity interleaved rigid: {} interior, {cube_seen} non-meshlet pixels",
        v.interior
    );
}
