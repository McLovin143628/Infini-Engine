//! GPU-instanced scatter gates (P18.5).
//!
//! Four properties, each gated rather than described:
//!
//! 1. **Occlusion is subtractive.** The scatter cull reads the same
//!    `hzb_occlusion.wgsl` test P18.1 proved conservative, so an occlusion-on
//!    frame is **byte-identical** to an occlusion-off frame — and the audit
//!    counters are asserted separately to prove the culling actually happened, so
//!    the equality cannot be satisfied by a no-op.
//! 2. **The compaction is deterministic.** Two fresh renderers must agree byte for
//!    byte on the image and on every audit counter. That is the whole reason the
//!    compaction is a prefix sum instead of an atomic append.
//! 3. **The LOD bands are real and ordered.** Pushing the camera back moves
//!    instances from the mesh list to the impostor list to nothing, monotonically.
//! 4. **The GPU path and the CPU fallback draw the same content.** Not the same
//!    pixels — the fallback has no impostors and no per-instance occlusion — but
//!    the same batches, in the same place, so a tier drop never empties a field.
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance, PrimMesh, RenderLight,
    RenderScene, RenderSettings, RenderView, ScatterBatch, ScatterData, ScatterInstance,
    ScatterSettings, HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP scatter: no GPU adapter ({e})");
            None
        }
    }
}

// ── content ──────────────────────────────────────────────────────────────────

/// A deterministic `n×n` field of scattered cubes on the ground plane, spanning
/// `span` metres, centred on `anchor`. Positions come from an integer hash so the
/// field is irregular (a lattice would let a whole row fall in or out of a band
/// together and hide an off-by-one) but reproducible on every platform — no `std`
/// trig anywhere near committed test data (the P14 LAW).
fn scatter_field(anchor: DVec3, n: u32, span: f64, scale: f32) -> ScatterBatch {
    let mut out = Vec::with_capacity((n * n) as usize);
    for i in 0..n * n {
        let (gx, gz) = ((i % n) as f64, (i / n) as f64);
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(0x27d4_eb2d);
        let jx = ((h & 0xFFFF) as f64 / 65535.0) - 0.5;
        let jz = (((h >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
        let step = span / n as f64;
        out.push(ScatterInstance {
            position: anchor
                + DVec3::new(
                    (gx - (n as f64 - 1.0) * 0.5 + jx * 0.6) * step,
                    scale as f64 * 0.5,
                    (gz - (n as f64 - 1.0) * 0.5 + jz * 0.6) * step,
                ),
            rotation: Quat::IDENTITY,
            scale,
            color: [0.25, 0.55, 0.22, 1.0],
        });
    }
    ScatterBatch::lit(
        Arc::new(ScatterData::build(PrimMesh::Cube, anchor, out)),
        anchor,
        0.85,
        7,
    )
}

fn lit_scene(batches: Vec<ScatterBatch>) -> RenderScene {
    let mut scene = RenderScene {
        scatter: batches,
        lights: vec![RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::new(-0.4, -0.8, -0.45).normalize(),
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            ..Default::default()
        }],
        ..Default::default()
    };
    scene.mark_dirty();
    scene
}

/// The camera looks down the +Z corridor from `back` metres away, so distance
/// bands map onto depth in the obvious way.
fn corridor_view(back: f64) -> RenderView {
    let eye = DVec3::new(0.0, 6.0, -back);
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

fn settings(f: impl FnOnce(&mut ScatterSettings)) -> RenderSettings {
    let mut s = RenderSettings::default();
    f(&mut s.scatter);
    s
}

fn shot(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    s: &RenderSettings,
    audit: bool,
) -> (Vec<u8>, inf_render::ScatterAudit) {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(*s);
    r.set_scatter_audit(audit);
    r.render(gpu, scene, view, &target.view, (W, H));
    let img = target.read_rgba(gpu).expect("readback");
    (img, r.scatter_audit(gpu))
}

// ── 1. determinism ───────────────────────────────────────────────────────────

/// Two fresh renderers, same inputs, byte-identical image **and** byte-identical
/// counters.
///
/// This is the gate the compaction design exists for. An `atomicAdd`-appended
/// visible list would very likely still pass a pixel comparison on a static opaque
/// scene — which is exactly why the counters are compared too, and why
/// `passes::scatter::tests::shader_constants_match_the_rust_side` additionally
/// asserts the *source* has not grown an atomic append.
#[test]
fn a_scattered_frame_is_reproducible_across_renderers() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = lit_scene(vec![scatter_field(DVec3::ZERO, 48, 90.0, 1.2)]);
    let view = corridor_view(70.0);
    let s = settings(|_| {});

    let (a, audit_a) = shot(&gpu, &scene, &view, &s, true);
    let (b, audit_b) = shot(&gpu, &scene, &view, &s, true);
    assert_eq!(a, b, "a scattered frame must be reproducible");
    assert_eq!(
        audit_a, audit_b,
        "the instance-cull counters must be reproducible"
    );

    // Anti-vacuity: the fixture must actually exercise the cull.
    assert_eq!(
        audit_a.candidates,
        48 * 48,
        "every instance must be considered"
    );
    assert!(
        audit_a.mesh > 0 && audit_a.impostor > 0,
        "the fixture must straddle the LOD bands (mesh {}, impostor {})",
        audit_a.mesh,
        audit_a.impostor
    );
    assert!(
        a.chunks(4).any(|p| p[1] > p[0] && p[1] > p[2]),
        "the frame must actually contain green scatter"
    );
}

/// The same content, expressed twice, is **one** GPU upload: `ScatterData::key`
/// is a hash of the packed bytes, so two batches that agree byte for byte agree on
/// their key. (The renderer's per-batch cache is keyed on it, so this is what
/// makes a duplicated scatter free.)
#[test]
fn identical_content_is_one_content_key() {
    let a = ScatterData::build(
        PrimMesh::Cube,
        DVec3::new(10.0, 0.0, -3.0),
        (0..64).map(|i| ScatterInstance {
            position: DVec3::new(10.0 + i as f64, 0.0, -3.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            color: [1.0, 0.0, 0.0, 1.0],
        }),
    );
    let b = ScatterData::build(
        PrimMesh::Cube,
        DVec3::new(10.0, 0.0, -3.0),
        (0..64).map(|i| ScatterInstance {
            position: DVec3::new(10.0 + i as f64, 0.0, -3.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            color: [1.0, 0.0, 0.0, 1.0],
        }),
    );
    assert_eq!(a.key(), b.key(), "identical content must share a key");

    // The primitive kind is part of the identity — the same placements drawn as
    // spheres are a different asset, not the same one with a different pipeline.
    let spheres = ScatterData::build(
        PrimMesh::Sphere,
        DVec3::new(10.0, 0.0, -3.0),
        a.instances.iter().map(|i| ScatterInstance {
            position: DVec3::new(10.0, 0.0, -3.0)
                + DVec3::new(i.offset[0] as f64, i.offset[1] as f64, i.offset[2] as f64),
            rotation: Quat::from_array(i.rotation),
            scale: i.scale,
            color: i.color,
        }),
    );
    assert_ne!(a.key(), spheres.key());

    // One moved instance is a different asset — which is what makes a stale
    // render unrepresentable rather than merely unlikely.
    let mut moved: Vec<ScatterInstance> = a
        .instances
        .iter()
        .map(|i| ScatterInstance {
            position: DVec3::new(10.0, 0.0, -3.0)
                + DVec3::new(i.offset[0] as f64, i.offset[1] as f64, i.offset[2] as f64),
            rotation: Quat::from_array(i.rotation),
            scale: i.scale,
            color: i.color,
        })
        .collect();
    moved[7].position.y += 0.001;
    let c = ScatterData::build(PrimMesh::Cube, DVec3::new(10.0, 0.0, -3.0), moved);
    assert_ne!(a.key(), c.key());
}

/// The packed payload is **origin-independent**: it is a function of the anchor,
/// not of the floating origin, so a rebase cannot invalidate a scatter upload.
#[test]
fn the_payload_does_not_depend_on_the_floating_origin() {
    let anchor = DVec3::new(12_345.0, 0.0, -6_789.0);
    let insts: Vec<ScatterInstance> = (0..16)
        .map(|i| ScatterInstance {
            position: anchor + DVec3::new(i as f64 * 0.5, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            color: [1.0; 4],
        })
        .collect();
    let d = ScatterData::build(PrimMesh::Cube, anchor, insts);
    // Offsets are small and exact regardless of where in the world the batch sits.
    assert_eq!(d.instances[0].offset, [0.0, 0.0, 0.0]);
    assert_eq!(d.instances[15].offset, [7.5, 0.0, 0.0]);
    // …and no floating-origin type was needed to produce them.
    assert_eq!(d.max_scale(), 1.0);
}

// ── 2. occlusion is subtractive ──────────────────────────────────────────────

/// A wall of opaque geometry in front of a scatter field: turning HZB occlusion on
/// must remove instances (`occluded > 0`) and must not change a single pixel.
#[test]
fn occlusion_on_is_pixel_identical_to_occlusion_off() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = lit_scene(vec![scatter_field(
        DVec3::new(0.0, 0.0, 30.0),
        40,
        60.0,
        1.0,
    )]);
    // A slab straddling the corridor, close to the camera: everything behind it is
    // provably invisible, which is what gives the cull something real to prove.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 8.0, -20.0),
        Quat::IDENTITY,
        Vec3::new(120.0, 40.0, 1.0),
        [0.5, 0.5, 0.55, 1.0],
        1,
    ));
    scene.mark_dirty();
    let view = corridor_view(60.0);

    let (on, audit_on) = shot(&gpu, &scene, &view, &settings(|s| s.occlusion = true), true);
    let (off, audit_off) = shot(
        &gpu,
        &scene,
        &view,
        &settings(|s| s.occlusion = false),
        true,
    );

    assert_eq!(
        on, off,
        "scatter occlusion must be purely subtractive — it may only remove \
         instances the hierarchical depth PROVES contribute zero fragments"
    );
    assert_eq!(audit_off.occluded, 0, "occlusion off must prove nothing");
    assert!(
        audit_on.occluded > 0,
        "the occluder fixture must actually occlude something (else the pixel \
         equality above is vacuous)"
    );
    // Subtractive means exactly that: the surviving set shrinks, never grows.
    assert!(audit_on.mesh + audit_on.impostor <= audit_off.mesh + audit_off.impostor);
}

// ── 3. LOD bands ─────────────────────────────────────────────────────────────

/// Distance bands select monotonically: as the camera retreats, instances leave
/// the mesh list, pass through the impostor list, and are finally culled.
#[test]
fn the_lod_bands_move_instances_mesh_to_impostor_to_culled() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = lit_scene(vec![scatter_field(DVec3::ZERO, 32, 40.0, 1.0)]);
    let s = settings(|s| {
        s.mesh_distance_m = 60.0;
        s.cull_distance_m = 150.0;
        s.fade_band_m = 10.0;
    });

    let mut last_mesh = u32::MAX;
    let mut saw_impostor = false;
    let mut saw_cull = false;
    for back in [40.0, 80.0, 130.0, 200.0] {
        let (_, a) = shot(&gpu, &scene, &corridor_view(back), &s, true);
        assert!(
            a.mesh <= last_mesh,
            "the full-mesh count must fall monotonically with distance \
             (at {back} m: {} after {last_mesh})",
            a.mesh
        );
        last_mesh = a.mesh;
        saw_impostor |= a.impostor > 0;
        saw_cull |= a.distance_culled > 0;
    }
    assert_eq!(last_mesh, 0, "nothing draws as a full mesh past the band");
    assert!(saw_impostor, "the impostor band must be exercised");
    assert!(saw_cull, "the cull distance must be exercised");
}

/// The authored draw distance (`PcgVolume::draw_distance` — the one *content* LOD
/// knob, and the one both hosts used to disagree about) clamps the tier's band
/// **down** and never up.
#[test]
fn the_authored_draw_distance_only_pulls_the_band_in() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut near = scatter_field(DVec3::ZERO, 32, 40.0, 1.0);
    near.draw_distance = 45.0;
    let mut far = scatter_field(DVec3::ZERO, 32, 40.0, 1.0);
    far.draw_distance = 100_000.0;

    let s = settings(|s| {
        s.mesh_distance_m = 60.0;
        s.cull_distance_m = 150.0;
    });
    let view = corridor_view(70.0);
    let (_, a_near) = shot(&gpu, &lit_scene(vec![near]), &view, &s, true);
    let (_, a_far) = shot(&gpu, &lit_scene(vec![far]), &view, &s, true);

    assert!(
        a_near.distance_culled > a_far.distance_culled,
        "a 45 m draw distance must cull more than the tier's 150 m band \
         ({} vs {})",
        a_near.distance_culled,
        a_far.distance_culled
    );
    let (_, a_unlimited) = shot(
        &gpu,
        &lit_scene(vec![scatter_field(DVec3::ZERO, 32, 40.0, 1.0)]),
        &view,
        &s,
        true,
    );
    assert_eq!(
        a_far.distance_culled, a_unlimited.distance_culled,
        "content must not be able to EXTEND the tier's band"
    );
}

/// Impostors off is not "impostors, invisibly": the impostor list stays empty and
/// the mesh band stretches to the cull distance, so the field still covers the
/// same ground.
#[test]
fn impostors_off_keeps_the_ground_covered() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = lit_scene(vec![scatter_field(DVec3::ZERO, 32, 40.0, 1.0)]);
    let view = corridor_view(90.0);
    let base = settings(|s| {
        s.mesh_distance_m = 40.0;
        s.cull_distance_m = 200.0;
        s.fade_band_m = 5.0;
    });
    let mut off = base;
    off.scatter.impostors = false;

    let (_, on_a) = shot(&gpu, &scene, &view, &base, true);
    let (img_off, off_a) = shot(&gpu, &scene, &view, &off, true);
    assert!(
        on_a.impostor > 0,
        "the reference must actually use impostors"
    );
    assert_eq!(off_a.impostor, 0, "impostors off must draw none");
    assert_eq!(
        off_a.mesh + off_a.distance_culled,
        off_a.candidates,
        "with impostors off every surviving instance is a full mesh"
    );
    assert!(
        img_off.chunks(4).any(|p| p[1] > p[0] && p[1] > p[2]),
        "the field must still be visible with impostors off"
    );
}

// ── 4. the CPU fallback ──────────────────────────────────────────────────────

/// The tier decides the *mechanism*, never whether there is any foliage. With the
/// GPU path off the same batches draw through the rigid mesh pipeline.
#[test]
fn the_cpu_fallback_draws_the_same_field() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = lit_scene(vec![scatter_field(DVec3::ZERO, 24, 30.0, 1.2)]);
    let view = corridor_view(40.0);

    let (gpu_img, _) = shot(&gpu, &scene, &view, &settings(|_| {}), false);
    let (cpu_img, cull) = shot(&gpu, &scene, &view, &settings(|s| s.gpu = false), true);

    let green = |img: &[u8]| img.chunks(4).filter(|p| p[1] > p[0] && p[1] > p[2]).count();
    let (g, c) = (green(&gpu_img), green(&cpu_img));
    assert!(
        g > 200 && c > 200,
        "both paths must draw the field (gpu {g}, cpu {c})"
    );
    // Not byte-identical, and the test says why rather than pretending: the GPU
    // path additionally draws impostors and culls per instance against the HZB.
    // What must hold is that the fallback covers comparably much of the screen.
    let ratio = c as f64 / g as f64;
    assert!(
        (0.5..2.0).contains(&ratio),
        "the fallback covers a wildly different area than the GPU path ({ratio:.2}×)"
    );
    assert_eq!(
        cull,
        inf_render::ScatterAudit::default(),
        "the fallback runs no cull compute, so it publishes no counters"
    );

    // Two fallback frames agree — the bucketed-eye re-pack key is deterministic.
    let (cpu_b, _) = shot(&gpu, &scene, &view, &settings(|s| s.gpu = false), false);
    assert_eq!(cpu_img, cpu_b, "the fallback must be reproducible too");
}

/// A scene with no scatter is untouched by any of this — the node returns before
/// it builds a pyramid, allocates a buffer or records a pass. This is why all 39
/// goldens are byte-identical.
#[test]
fn a_scene_without_scatter_is_byte_identical_with_every_knob_wound() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene::default();
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::splat(2.0),
        [0.8, 0.3, 0.2, 1.0],
        1,
    ));
    scene.mark_dirty();
    let view = corridor_view(8.0);

    let (base, _) = shot(&gpu, &scene, &view, &RenderSettings::default(), false);
    let wound = settings(|s| {
        s.gpu = false;
        s.impostors = false;
        s.occlusion = false;
        s.frustum_cull = false;
        s.mesh_distance_m = 3.0;
        s.cull_distance_m = 5.0;
        s.fade_band_m = 1.0;
    });
    let (other, _) = shot(&gpu, &scene, &view, &wound, true);
    assert_eq!(
        base, other,
        "every scatter knob must be inert on a scene with no scatter batches"
    );
}

// ── 5. scatter casts shadows ─────────────────────────────────────────────────

/// Before P18.5 a scatter's instances *were* `RenderScene::instances`, so they
/// cast cascaded shadows for free. Moving them onto the GPU path would have
/// silently deleted every one of those shadows — invisible to the compiler, to the
/// unit tests and to a pixel comparison with no shadow in it, and obvious the
/// moment anyone looks at the ground. The shadow pass therefore packs scatter
/// casters alongside the rigid ones, clipped to the shadow range.
///
/// The claim is isolated by toggling **shadows**, not the scatter: the ground is
/// flat and the only thing standing on it is scatter, so every ground pixel that
/// darkens when shadows come on is a scatter shadow. The control repeats the
/// experiment with the scatter removed and must find essentially none.
#[test]
fn scatter_casts_cascaded_shadows() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = lit_scene(vec![scatter_field(DVec3::new(0.0, 0.0, 6.0), 8, 9.0, 1.4)]);
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.1, 6.0),
        Quat::IDENTITY,
        Vec3::new(60.0, 0.2, 60.0),
        [0.85, 0.85, 0.85, 1.0],
        1,
    ));
    scene.lights[0].direction = Vec3::new(0.35, 0.80, 0.49).normalize();
    scene.lights[0].cast_shadows = true;
    scene.mark_dirty();
    let view = {
        let eye = DVec3::new(0.0, 16.0, -14.0);
        let target = DVec3::new(0.0, 0.0, 6.0);
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
    };

    let mut shadows_on = RenderSettings::default();
    shadows_on.shadows.enabled = true;
    shadows_on.shadows.max_distance = 80.0;
    let shadows_off = RenderSettings::default();

    let render = |sc: &RenderScene, s: &RenderSettings| {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(*s);
        r.render(&gpu, sc, &view, &target.view, (W, H));
        target.read_rgba(&gpu).expect("readback")
    };
    // Ground pixels (grey receiver, not the green scatter) that darken when the
    // shadow pass turns on.
    let ground_darkened = |lit: &[u8], shadowed: &[u8]| {
        lit.chunks(4)
            .zip(shadowed.chunks(4))
            .filter(|(l, s)| {
                let ground = l[1] <= l[0] + 4 && l[0] > 20;
                ground && (s[0] as i32) < (l[0] as i32) - 12
            })
            .count()
    };

    // CONTROL A — the anti-vacuity guard, and it earned its place: the first draft
    // of this test put the sun 20° above the horizon and found 1 shadowed ground
    // pixel, which reads exactly like "scatter casts no shadows". Eight rigid cubes
    // in the same fixture found 6, i.e. the FIXTURE was blind, not the code. This
    // control fails first and says so.
    let mut rigid = scene.clone();
    rigid.scatter.clear();
    for i in 0..8 {
        rigid.instances.push(MeshInstance::lit(
            DVec3::new(-4.0 + i as f64, 0.7, 6.0),
            Quat::IDENTITY,
            Vec3::splat(1.4),
            [0.25, 0.55, 0.22, 1.0],
            20 + i as u32,
        ));
    }
    rigid.mark_dirty();
    let rigid_cast = ground_darkened(&render(&rigid, &shadows_off), &render(&rigid, &shadows_on));
    assert!(
        rigid_cast > 100,
        "the FIXTURE cannot show a shadow at all ({rigid_cast} ground px from eight \
         rigid cubes) — retune the sun angle or the camera before trusting the \
         scatter measurement below"
    );

    let with_on = render(&scene, &shadows_on);
    let with_off = render(&scene, &shadows_off);
    let cast = ground_darkened(&with_off, &with_on);

    let mut bare = scene.clone();
    bare.scatter.clear();
    bare.mark_dirty();
    let control = ground_darkened(&render(&bare, &shadows_off), &render(&bare, &shadows_on));

    assert!(
        cast > 200,
        "the scatter cast no cascaded shadow onto the receiver ({cast} ground px) — \
         the pre-P18.5 CPU path did, so this would be a silent regression"
    );
    assert!(
        control * 8 < cast,
        "the control (same scene, no scatter) darkened {control} ground px against \
         the scatter's {cast} — the measurement is not isolating scatter shadows"
    );

    // …and the frame is still reproducible with the caster set in play.
    assert_eq!(
        with_on,
        render(&scene, &shadows_on),
        "shadowed scatter must be reproducible"
    );
}

// ── 6. same content, two anchors ─────────────────────────────────────────────

/// **Two batches sharing a content key at different anchors must BOTH draw.**
///
/// The first cut of P18.5 keyed all of a batch's GPU state — its compacted list,
/// its indirect args, *and its uniforms* — by content alone, on the reasoning that
/// content addressing dedupes uploads. It does, and that part is right; but the
/// per-frame state is not content, it is one draw. Two same-content batches
/// therefore wrote the same uniform buffer in sequence and the last anchor won, so
/// one of the two fields silently vanished — with the module's own documentation
/// naming duplicated foliage strokes as the reason content addressing exists.
///
/// Reproduced before the fix at 615 painted pixels against this control's 1223.
/// The control is the load-bearing half: without it "both fields drew" is
/// satisfied by one field drawing twice as densely, or by any regression that
/// happens to paint enough pixels.
#[test]
fn two_batches_of_the_same_content_at_different_anchors_both_draw() {
    let Some(gpu) = gpu_or_skip() else { return };

    // One payload, built once and shared by `Arc` — identical bytes, identical
    // key, deliberately: this is the case that used to collide.
    let field = scatter_field(DVec3::ZERO, 14, 10.0, 0.9);
    let data = field.data.clone();
    let left = ScatterBatch {
        data: data.clone(),
        anchor: DVec3::new(-11.0, 0.0, 6.0),
        id: 11,
        ..field.clone()
    };
    let right = ScatterBatch {
        data: data.clone(),
        anchor: DVec3::new(11.0, 0.0, 6.0),
        id: 12,
        ..field.clone()
    };
    assert_eq!(
        left.data.key(),
        right.data.key(),
        "the fixture must be two batches of IDENTICAL content, or it tests nothing"
    );

    let view = corridor_view(34.0);
    let s = settings(|s| {
        s.mesh_distance_m = 400.0;
        s.cull_distance_m = 800.0;
    });
    let green = |img: &[u8]| {
        img.chunks(4)
            .filter(|p| p[1] > p[0] + 4 && p[1] > p[2] + 4)
            .count()
    };

    let (both, audit) = shot(
        &gpu,
        &lit_scene(vec![left.clone(), right.clone()]),
        &view,
        &s,
        true,
    );
    let (only_left, _) = shot(&gpu, &lit_scene(vec![left.clone()]), &view, &s, false);
    let (only_right, _) = shot(&gpu, &lit_scene(vec![right.clone()]), &view, &s, false);

    // Both halves of the frame are painted, and each matches its solo render —
    // i.e. neither field was drawn at the other's anchor.
    assert!(
        green(&only_left) > 100 && green(&only_right) > 100,
        "each field must paint on its own (left {}, right {})",
        green(&only_left),
        green(&only_right)
    );
    assert_eq!(
        green(&both),
        green(&only_left) + green(&only_right),
        "the two same-content fields do not sum: one of them was dropped or drawn \
         on top of the other (both {}, left {}, right {})",
        green(&both),
        green(&only_left),
        green(&only_right)
    );

    // The cull saw both payloads, not one.
    assert_eq!(
        audit.candidates,
        2 * 14 * 14,
        "the cull ran over one batch, not two: {audit:?}"
    );

    // …and the *upload* really is shared: a third batch of the same content adds
    // candidates without adding a distinct payload, which is the property content
    // addressing was introduced for and which the fix must not have spent.
    let third = ScatterBatch {
        data: data.clone(),
        anchor: DVec3::new(0.0, 0.0, 26.0),
        id: 13,
        ..field
    };
    let (_, audit3) = shot(&gpu, &lit_scene(vec![left, right, third]), &view, &s, true);
    assert_eq!(audit3.candidates, 3 * 14 * 14);
    // …and the dedup is *observed*, not inferred: three batches, ONE payload on the
    // GPU. Without this the candidate count above is equally consistent with three
    // separate uploads — which is exactly the property the collision fix had to
    // preserve while it stopped sharing everything else.
    assert_eq!(
        audit3.uploads, 1,
        "three same-content batches must share one instance buffer"
    );
    assert_eq!(audit.uploads, 1, "two same-content batches likewise");
}
