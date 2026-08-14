//! **The ray-query shadow experiment** (P28.5, the ROADMAP's last clause) — the
//! probe, the clamp, the BLAS/TLAS over meshlet clusters, and the comparison
//! against the shipped virtual shadow map.
//!
//! The clause, in full:
//!
//! > 1. adapter/feature probe for ray queries in the pinned wgpu; where present,
//! >    BLAS over meshlet clusters + per-frame TLAS, sun shadows via ray query
//! >    compared against VSM output on goldens; 2. a memo with the measured
//! >    verdict (quality, cost, coverage); VSM remains the shipped path on every
//! >    tier; the experiment lands behind a default-off setting with a `caps.rs`
//! >    clamp.
//!
//! # The file's two halves, and why the split is the point
//!
//! **The never-load-bearing half runs everywhere**: the setting is off, no tier
//! or preset turns it on, the clamp only ever clears it, and a device without
//! the feature refuses to build an acceleration structure with a message rather
//! than producing an empty one. Those are the arms that keep the experiment
//! honest on the machines that cannot run it, which is most of them.
//!
//! **The measuring half runs behind the probe.**
//! `wgpu::Features::EXPERIMENTAL_RAY_QUERY` is documented **Vulkan-only and
//! native-only** in the pinned wgpu 30, and this tree's instance is
//! `VULKAN | METAL`, so on macOS it is absent by construction and on any
//! software adapter (lavapipe, WARP) it is absent in fact. Every device arm
//! prints the adapter it ran on, because a measurement whose platform is not
//! stated is the P25 one-platform trap.
//!
//! # The oracle
//!
//! A gate cannot see an error the subsystems share, so the correctness arm does
//! not compare the trace against another trace: it compares it against a
//! **CPU ray caster written out in this file** (Möller–Trumbore, f32, no
//! acceleration structure at all), over the same triangles and the same rays.
//! A wrong transform, a wrong index base or a transposed camera basis all fail
//! there and cannot fail "by agreement".
//!
//! The **surface offset is the one input the two sides deliberately share** —
//! it is a parameter of the experiment, not of the intersector, so moving it
//! moves both. It gets its own arm instead
//! (`a_zero_surface_offset_shadows_every_surface_with_itself`), because a
//! shared parameter with no arm of its own is a thing no gate can see.
//!
//! The VSM comparison is a different kind of claim and is treated as one: two
//! shadow techniques disagree at penumbrae and at every bias, so it is
//! **measured and classified**, not asserted equal. What is asserted is that
//! both produce a shadow of the same caster in the same place — falsified by an
//! anti-vacuity pair (each mask must have both classes) and by the requirement
//! that they agree on most of the frame.

use std::sync::Arc;

use glam::{DVec3, Mat3, Quat, Vec3};
use inf_render::caps::AdapterCaps;
use inf_render::raytrace::{
    RtBlasSource, RtInstance, RtScene, RtSunShadow, RtView, RT_LIT, RT_MISS, RT_SHADOWED,
};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RaytraceSettings, RenderLight,
    RenderScene, RenderSettings, RenderTier, RenderView, VgeomAsset, VgeomInstance, VgeomMesh,
    VsmSettings, HEADLESS_FORMAT,
};
// (the fixture calls `inf_vgeom::test_support` by path — see `mesh`)

const W: u32 = 256;
const H: u32 = 144;
const ASSET: u128 = 0x2805_0000_0000_0001;
/// Frames the VSM arm renders before it reads the frame — the marking ring is
/// pinned at `frame − 2` and a page has to be marked, rasterized and cached
/// before a receiver can read it.
const FRAMES: u64 = 8;
/// **A narrow field of view**, on `vsm_receiver.rs`'s own ruling: the shadow
/// level rule is one shadow texel per SCREEN pixel, so a 256 × 144 target at 60°
/// asks for 70 mm texels — coarser than any shipped configuration, and coarse
/// enough that the derived bias swamps a metre-scale caster. Every arm in this
/// tree that measures shadow DETAIL narrows the field of view instead, which is
/// the honest way to raise pixel density on a small render target.
const NARROW_FOV_DEG: f32 = 25.0;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP ray_query ({what}): no GPU adapter ({e})");
            None
        }
    }
}

/// The probe: the experiment's own context, or a stated skip.
///
/// A **separate** context from the shipped one, and that separation is P28.5's
/// own finding — see [`GpuContext::headless_ray_query`].
fn ray_query_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless_ray_query() {
        Ok(gpu) => {
            let info = gpu.adapter.get_info();
            assert!(
                gpu.supports_ray_query(),
                "the experiment's own context does not support ray queries"
            );
            eprintln!(
                "ray_query ({what}): running on {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            );
            Some(gpu)
        }
        Err(e) => {
            eprintln!(
                "SKIP ray_query ({what}): {e}. The feature is Vulkan-only in \
                 wgpu 30. The probe, clamp and refusal arms still ran."
            );
            None
        }
    }
}

// ── the fixture ──────────────────────────────────────────────────────────────

/// The DAG both halves use: a **flat** 16 × 16 quad grid, 512 triangles, a real
/// meshlet DAG — `visbuffer_parity`'s fixture with the displacement amplitude
/// taken to zero.
///
/// **Flat is load-bearing here and nowhere else in the tree.** A displaced
/// surface shadows *itself*, and self-shadowing is where two shadow techniques
/// legitimately disagree most: a rasterized shadow map's depth bias produces
/// acne along every grazing slope, and a ray query has no bias to produce it
/// with. Measured on the displaced fixture, the two masks agreed on **56.7 %**
/// of the covered frame — nearly all of the disagreement in the direction "the
/// shipped path says shadowed, the trace says lit", i.e. exactly acne. That is
/// a real property of shadow maps and it is not what this comparison is for, so
/// the fixture removes it and the memo records the number.
///
/// It removes a second confound with it: a plane simplifies to a plane, so the
/// meshlet cut the raster picks is geometrically the level the BLAS holds, and
/// no disagreement below is about LOD.
fn mesh() -> Arc<VgeomMesh> {
    Arc::new(inf_vgeom::test_support::build_grid(
        16,
        0.0,
        inf_vgeom::test_support::GridNormals::Analytic,
    ))
}

/// **The finest level of the DAG**, which is `levels.len() − 1`: the payload is
/// stored coarse→fine and `lod_level == 0` is the original geometry. Asserted
/// rather than assumed, because a BLAS built over the *coarsest* level would
/// trace a different surface from the one the raster draws and every
/// disagreement below would be about LOD.
fn finest_level(m: &VgeomMesh) -> usize {
    let last = m.levels.len() - 1;
    assert_eq!(
        m.levels[last].lod_level, 0,
        "the DAG's last level is not the finest — the BLAS would trace a \
         different surface from the one the raster draws"
    );
    last
}

/// A row-major 3 × 4 affine, the layout `wgpu::TlasInstance` takes.
fn xform(t: Vec3, r: Quat, s: Vec3) -> [f32; 12] {
    let m = Mat3::from_quat(r) * Mat3::from_diagonal(s);
    [
        m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x, //
        m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y, //
        m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z,
    ]
}

/// The two placements the whole file shares: a large ground quad in XZ, and a
/// smaller slab floating above it that casts across it.
///
/// The fixture's grid already spans `x, z ∈ [-1, 1]`, so both instances take
/// the identity rotation and the second is simply raised — a rotation here
/// would stand a plane on edge and cast nothing, which is what the first draft
/// of this file did.
///
/// One asset, two instances — which is also what makes the TLAS meaningful: a
/// scene with one instance can be traced with a BLAS alone.
fn placements() -> [(Vec3, Quat, Vec3); 2] {
    [
        (
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(14.0, 1.0, 14.0),
        ),
        (
            Vec3::new(0.0, 2.6, 0.4),
            Quat::IDENTITY,
            Vec3::new(1.6, 1.0, 1.6),
        ),
    ]
}

/// Unit direction **toward** the sun — steep and offset, so the wall's shadow
/// lands on the ground inside the frame rather than off the far edge.
fn sun_dir() -> Vec3 {
    Vec3::new(0.25, 0.90, 0.35).normalize()
}

/// The camera: above and behind, looking down at the ground the wall shadows.
fn view() -> RenderView {
    let eye = DVec3::new(0.0, 7.0, 9.0);
    let at = DVec3::new(0.0, -0.6, -1.0);
    RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: NARROW_FOV_DEG.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn rt_view() -> RtView {
    let v = view();
    RtView {
        eye: v.eye_world.as_vec3(),
        forward: v.forward,
        up: v.up,
        fov_y: v.fov_y,
        sun: sun_dir(),
        near: v.near,
        far: 400.0,
        shadow_bias: inf_render::raytrace::RT_SHADOW_BIAS_M,
    }
}

/// **The correctness arm's camera**: lower and wider, aimed so the frame's top
/// rows look past the ground's far edge.
///
/// A separate view because the two arms want opposite things. The VSM
/// comparison wants every pixel covered and a narrow field of view, for the
/// shadow-texel-density reason [`NARROW_FOV_DEG`] gives. The correctness arm
/// wants all **three** verdicts present — including `RT_MISS`, which only
/// exists where a ray escapes — because an oracle comparison that never sees a
/// miss cannot tell a trace that hits everything from a correct one.
fn wide_rt_view() -> RtView {
    let eye = Vec3::new(0.0, 3.0, 10.0);
    let at = Vec3::new(0.0, 0.0, -6.0);
    RtView {
        eye,
        forward: (at - eye).normalize(),
        up: Vec3::Y,
        fov_y: 45_f32.to_radians(),
        sun: sun_dir(),
        near: 0.05,
        far: 400.0,
        shadow_bias: inf_render::raytrace::RT_SHADOW_BIAS_M,
    }
}

/// The meshlet scene the raster draws — the same asset at the same two
/// placements the TLAS gets.
fn scene(cast: bool) -> RenderScene {
    let m = mesh();
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &m).expect("index the vmesh")],
        ..Default::default()
    };
    for (i, (t, r, s)) in placements().into_iter().enumerate() {
        sc.vgeom_instances.push(VgeomInstance::lit(
            ASSET,
            t.as_dvec3(),
            r,
            s,
            [0.82, 0.82, 0.84, 1.0],
            i as u32 + 1,
        ));
    }
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.94],
        intensity: 3.0,
        direction: sun_dir(),
        position: DVec3::ZERO,
        range: 0.0,
        cast_shadows: cast,
        ..Default::default()
    });
    sc.mark_dirty();
    sc
}

/// The TLAS the experiment traces: one BLAS over the finest meshlet level, two
/// instances at the raster's own placements.
fn build_scene(gpu: &GpuContext) -> (RtScene, std::time::Duration) {
    let m = mesh();
    let level = finest_level(&m);
    let src = RtBlasSource::from_meshlet_level(&m, level).expect("the finest level exists");
    let instances: Vec<RtInstance> = placements()
        .into_iter()
        .map(|(t, r, s)| RtInstance {
            blas: 0,
            transform: xform(t, r, s),
        })
        .collect();
    let t0 = std::time::Instant::now();
    let rt = RtScene::build(gpu, std::slice::from_ref(&src), &instances).expect("the build");
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    (rt, t0.elapsed())
}

// ── the CPU oracle ───────────────────────────────────────────────────────────

/// Every triangle of the fixture, **in world space**, derived independently of
/// anything the GPU was handed: the DAG's own vertices through this file's own
/// transform arithmetic.
fn world_triangles() -> Vec<[Vec3; 3]> {
    let m = mesh();
    let level = finest_level(&m);
    let src = RtBlasSource::from_meshlet_level(&m, level).expect("the finest level");
    let mut out = Vec::new();
    for (t, r, s) in placements() {
        let rot = Mat3::from_quat(r) * Mat3::from_diagonal(s);
        for tri in src.indices.chunks_exact(3) {
            out.push([
                rot * Vec3::from(src.positions[tri[0] as usize]) + t,
                rot * Vec3::from(src.positions[tri[1] as usize]) + t,
                rot * Vec3::from(src.positions[tri[2] as usize]) + t,
            ]);
        }
    }
    out
}

/// Möller–Trumbore, written out here so the comparison is against a second
/// derivation and not against the same intersector twice. `None` on a miss.
fn hit_triangle(o: Vec3, d: Vec3, tri: &[Vec3; 3], tmin: f32, tmax: f32) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let tv = o - tri[0];
    let u = tv.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tv.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t >= tmin && t <= tmax).then_some(t)
}

fn closest(o: Vec3, d: Vec3, tris: &[[Vec3; 3]], tmin: f32, tmax: f32) -> Option<f32> {
    tris.iter()
        .filter_map(|t| hit_triangle(o, d, t, tmin, tmax))
        .fold(None, |acc: Option<f32>, t| {
            Some(acc.map_or(t, |a| a.min(t)))
        })
}

/// The same three verdicts the shader produces, from the CPU caster.
fn cpu_verdicts(tris: &[[Vec3; 3]], v: &RtView) -> Vec<u32> {
    let right = v.forward.cross(v.up).normalize();
    let up = right.cross(v.forward).normalize();
    let half = (v.fov_y * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let mut out = vec![RT_MISS; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let px = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / H as f32 * 2.0;
            let dir = (v.forward.normalize() + right * (px * half * aspect) + up * (py * half))
                .normalize();
            let Some(t) = closest(v.eye, dir, tris, v.near, v.far) else {
                continue;
            };
            let surface = v.eye + dir * t;
            let occluded =
                closest(surface, v.sun.normalize(), tris, v.shadow_bias, v.far).is_some();
            out[(y * W + x) as usize] = if occluded { RT_SHADOWED } else { RT_LIT };
        }
    }
    out
}

// ── the never-load-bearing half (no adapter) ─────────────────────────────────

/// **The experiment is off, and nothing in this tree turns it on.**
///
/// The ROADMAP's clause has two halves — "default-off setting" and "VSM remains
/// the shipped path on every tier" — and this is both, asserted structurally
/// rather than read off a literal: every tier and both presets are applied to a
/// settings block with the experiment ON, and the field must come back off or
/// unchanged, never set.
#[test]
fn the_experiment_is_off_by_default_and_no_tier_or_preset_turns_it_on() {
    assert!(
        !RaytraceSettings::default().sun_shadows,
        "the experiment shipped ON"
    );
    assert!(!RenderSettings::default().raytrace.sun_shadows);

    // THE LAW: a tier never turns it on. High is the tier that would be tempted.
    let off = RenderSettings::default();
    for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
        assert!(
            !tier.apply(off).raytrace.sun_shadows,
            "{tier:?} ENABLED the ray-query experiment"
        );
    }
    assert!(!RenderTier::mobile_default().raytrace.sun_shadows);
    assert!(!RenderTier::clamp_mobile(off).raytrace.sun_shadows);

    // …and a tier does not *clear* it either, which is the other half of "no
    // tier has an opinion": the only thing that clears it is the adapter clamp,
    // so a host that deliberately turned the experiment on to measure it does
    // not silently lose it to a tier decision.
    let mut on = RenderSettings::default();
    on.raytrace.sun_shadows = true;
    for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
        assert!(
            tier.apply(on).raytrace.sun_shadows,
            "{tier:?} took an opinion about an experiment"
        );
    }

    // And VSM is still the shipped shadow path on every tier that runs one —
    // the clause's own words, pinned beside the experiment rather than only in
    // `caps.rs`'s own suite.
    let mut vsm_on = RenderSettings::default();
    vsm_on.vsm = VsmSettings {
        enabled: true,
        ..vsm_on.vsm
    };
    assert!(RenderTier::High.apply(vsm_on).vsm.enabled);
    assert!(RenderTier::Medium.apply(vsm_on).vsm.enabled);
    assert!(
        !RenderTier::Low.apply(vsm_on).vsm.enabled,
        "Low keeps CSM — the P27.5 clause"
    );
}

/// **The clamp clears and never sets**, in both directions, on synthetic caps —
/// so it is checked on the adapter class this machine is not.
#[test]
fn the_capability_clamp_only_ever_clears_the_experiment() {
    fn caps(ray_query: bool) -> AdapterCaps {
        AdapterCaps {
            compute_shaders: true,
            indirect_execution: true,
            max_storage_buffers_per_stage: 8,
            max_storage_buffer_binding_size: 128 << 20,
            max_compute_workgroups_per_dim: 65535,
            max_storage_textures_per_stage: 1,
            is_cpu: false,
            texture_compression_bc: true,
            ray_query,
            polygon_mode_line: true,
        }
    }
    let mut on = RenderSettings::default();
    on.raytrace.sun_shadows = true;
    let off = RenderSettings::default();

    // No ray queries: cleared, whatever the caller asked for.
    assert!(!caps(false).clamp_ray_query(on).raytrace.sun_shadows);
    assert!(!caps(false).clamp_ray_query(off).raytrace.sun_shadows);
    // Ray queries: a caller who asked keeps it, a caller who did not is NOT
    // given it. That second one is the law.
    assert!(caps(true).clamp_ray_query(on).raytrace.sun_shadows);
    assert!(
        !caps(true).clamp_ray_query(off).raytrace.sun_shadows,
        "an adapter that CAN trace enabled an experiment nobody asked for"
    );
    // Idempotent, and reached through the composite door a host actually calls.
    for c in [caps(false), caps(true)] {
        let a = c.clamp_occlusion(on);
        assert_eq!(c.clamp_occlusion(a), a, "the clamp is not idempotent");
        assert_eq!(
            a.raytrace.sun_shadows,
            c.clamp_ray_query(on).raytrace.sun_shadows,
            "clamp_occlusion did not compose clamp_ray_query"
        );
    }
    // The capability is orthogonal to the tier, like BC and line raster.
    assert_eq!(
        inf_render::caps::choose_tier(&caps(true)),
        inf_render::caps::choose_tier(&caps(false)),
        "ray queries moved the render tier"
    );
}

/// **A build over nothing is refused**, and so is a build on a device without
/// the feature — with a message rather than an empty structure a trace would
/// then report as "the sun sees everything".
#[test]
fn an_acceleration_structure_over_nothing_or_without_the_feature_is_refused() {
    let Some(gpu) = gpu_or_skip("the refusal") else {
        return;
    };
    let src = RtBlasSource {
        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices: vec![0, 1, 2],
    };
    let one = [RtInstance {
        blas: 0,
        transform: xform(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
    }];
    // **The SHIPPED device refuses**, and that is the arm's first half: the
    // renderer's own context never signs wgpu's experimental token, so it never
    // carries the feature, whatever the adapter can do. On this machine the
    // adapter has ray queries and the shipped device still does not — which is
    // exactly the separation P28.5 had to introduce.
    assert!(
        !gpu.supports_ray_query(),
        "the SHIPPED context acquired an experimental feature — `from_adapter`'s \
         optional mask has grown one and the experiment is no longer separate"
    );
    let e = RtScene::build(&gpu, std::slice::from_ref(&src), &one)
        .expect_err("a device without the feature built one");
    assert!(e.contains("EXPERIMENTAL_RAY_QUERY"), "{e}");

    // The other half needs a device that CAN build one: the degenerate inputs
    // are refused there too, rather than producing an empty structure a trace
    // would report as "the sun sees everything".
    let Some(rq) = ray_query_or_skip("the refusal") else {
        return;
    };
    assert!(RtScene::build(&rq, &[], &one).is_err());
    assert!(RtScene::build(&rq, std::slice::from_ref(&src), &[]).is_err());
    let bad = RtBlasSource {
        positions: src.positions.clone(),
        indices: vec![0, 1, 9],
    };
    let e = RtScene::build(&rq, std::slice::from_ref(&bad), &one)
        .expect_err("an out-of-range index was accepted");
    assert!(e.contains("indexes vertex 9"), "{e}");
}

// ── the measuring half (behind the probe) ────────────────────────────────────

/// **THE CORRECTNESS ARM**: the device's ray query against a CPU ray caster
/// written out in this file.
///
/// Same triangles, same rays, same bias, same three verdicts. This is what
/// makes every number in the comparison arm below mean something: a transposed
/// camera basis, a wrong TLAS transform convention, an index base off by the
/// vertex offset, or a shadow bias that lets a surface shadow itself all fail
/// here, and none of them can fail by agreement — the CPU side owns no
/// acceleration structure at all.
#[test]
fn a_blas_over_meshlet_clusters_traces_what_a_cpu_ray_caster_traces() {
    let Some(gpu) = ray_query_or_skip("the correctness arm") else {
        return;
    };
    let (rt, build_ms) = build_scene(&gpu);
    let stats = rt.stats();
    let pass = RtSunShadow::new(&gpu);
    let v = wide_rt_view();
    let t0 = std::time::Instant::now();
    let gpu_verdicts = pass.trace(&gpu, &rt, &v, W, H);
    let trace_ms = t0.elapsed();

    let tris = world_triangles();
    let cpu = cpu_verdicts(&tris, &v);

    // ANTI-VACUITY, before anything is compared: all three verdicts have to be
    // present on both sides, or "they agree" is a statement about a blank frame.
    for (name, m) in [("gpu", &gpu_verdicts), ("cpu", &cpu)] {
        for (label, want) in [
            ("miss", RT_MISS),
            ("lit", RT_LIT),
            ("shadowed", RT_SHADOWED),
        ] {
            let n = m.iter().filter(|x| **x == want).count();
            assert!(
                n > 64,
                "{name} produced only {n} {label} pixels — this fixture is not \
                 exercising the class"
            );
        }
    }

    let differ: Vec<usize> = (0..(W * H) as usize)
        .filter(|i| gpu_verdicts[*i] != cpu[*i])
        .collect();
    // Disagreement is CLASSIFIED, not merely bounded (the P28.1 criterion's
    // shape): a difference at a silhouette or a shadow edge is a rounding
    // choice between two intersectors, and a difference in the middle of a
    // solid block is a defect.
    let interior = differ
        .iter()
        .filter(|i| {
            let (x, y) = ((**i as u32) % W, (**i as u32) / W);
            x > 0 && y > 0 && x + 1 < W && y + 1 < H
        })
        .filter(|i| {
            let n = [**i - 1, **i + 1, **i - W as usize, **i + W as usize];
            n.iter().all(|k| gpu_verdicts[*k] != cpu[*k])
        })
        .count();

    println!(
        "RAY QUERY vs CPU CASTER — {} of {} pixels differ ({:.4} %), {interior} of them \
         interior to a differing block",
        differ.len(),
        W * H,
        differ.len() as f64 / f64::from(W * H) * 100.0
    );
    println!(
        "  BLAS {} / {} triangles / {} instances / {:.1} KiB input — build {:.2} ms, \
         trace {:.2} ms for {} rays",
        stats.blas,
        stats.triangles,
        stats.instances,
        stats.input_bytes as f64 / 1024.0,
        build_ms.as_secs_f64() * 1e3,
        trace_ms.as_secs_f64() * 1e3,
        (W * H) * 2
    );

    assert!(
        differ.len() * 200 < (W * H) as usize,
        "the device and the CPU caster disagree on {} of {} pixels — over half a \
         percent is not a rounding boundary",
        differ.len(),
        W * H
    );
    assert_eq!(
        interior, 0,
        "{interior} disagreements are interior to a solid differing block, which \
         is a wrong ray and not a rounding boundary"
    );
    // The BLAS really is over the meshlet clusters, and the TLAS really has two
    // instances of it — a single-instance trace would not exercise the TLAS at
    // all and this file's transform convention would be untested.
    assert_eq!(stats.blas, 1);
    assert_eq!(stats.instances, 2);
    assert_eq!(stats.triangles, world_triangles().len() / 2);
}

/// **THE COMPARISON THE CLAUSE ASKS FOR**: ray-queried sun shadows against the
/// shipped virtual shadow map, on the same scene, through the same camera.
///
/// The shipped mask is read the way `vsm_receiver.rs` reads it — the frame with
/// the sun casting, against the same frame with it not casting — because that
/// is the *output* the clause names, and a receiver function called by hand
/// would be a comparison against a component rather than against the path.
///
/// **What is asserted, and what is only measured.** Two shadow techniques
/// disagree at every penumbra and at every bias, so equality is not the claim
/// and asserting it would be asserting a coincidence. What is asserted:
///
/// * both masks are non-trivial (each has lit and shadowed pixels) — the
///   anti-vacuity, in both directions, before anything is read;
/// * they agree on the great majority of the covered frame;
/// * the disagreement is concentrated at the boundary rather than spread —
///   which is what says the two techniques found the *same shadow*, offset,
///   rather than two different ones.
///
/// **The coverage bound is structural and is the honest part**: the TLAS holds
/// the meshlet clusters this fixture put in it and nothing else — no terrain,
/// no voxels, no skinned geometry, no scatter, no translucency — so the
/// comparison is restricted to pixels where the primary ray hit, and the
/// numbers below are about those pixels. That restriction is not a fixture
/// choice; it is what a per-frame TLAS over meshlet clusters *is*.
#[test]
fn the_ray_queried_sun_shadow_against_the_shipped_virtual_shadow_map() {
    let Some(gpu) = ray_query_or_skip("the VSM comparison") else {
        return;
    };
    let (rt, _) = build_scene(&gpu);
    let pass = RtSunShadow::new(&gpu);
    let v = rt_view();
    let traced = pass.trace(&gpu, &rt, &v, W, H);

    // **The shipped path runs on the SHIPPED device**, not on the experiment's
    // — the comparison is against what a player's machine draws, and the
    // experiment's context is a device no host ever builds.
    let Some(shipped) = gpu_or_skip("the VSM comparison's shipped half") else {
        return;
    };
    let gpu = &shipped;
    // Twice: sun casting through the page atlas, and the same frame with the
    // sun not casting at all.
    let render = |cast: bool| -> Vec<u8> {
        let target = HeadlessTarget::new(gpu, W, H);
        let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
        let mut s = *r.settings();
        s.vsm = VsmSettings {
            enabled: cast,
            budget_bytes: 512 * 64 * 1024,
            clipmap_pages_per_side: 32,
            clipmap_levels: 8,
            first_level_extent_m: 12.0,
            ..s.vsm
        };
        s.shadows.enabled = cast;
        r.try_set_settings(s).expect("legal settings");
        let sc = scene(cast);
        let view = view();
        for _ in 0..FRAMES {
            r.render(gpu, &sc, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        target.read_rgba(gpu).expect("readback")
    };
    let shadowed = render(true);
    let lit = render(false);

    let luma = |img: &[u8], i: usize| -> f32 {
        let p = &img[i * 4..i * 4 + 3];
        0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])
    };
    // The shipped mask: a pixel the sun's shadow darkened. The threshold is a
    // whole 8-bit step, so dithering and tone-mapping rounding cannot enter it.
    let vsm_shadowed: Vec<bool> = (0..(W * H) as usize)
        .map(|i| luma(&lit, i) - luma(&shadowed, i) > 1.0)
        .collect();

    let covered: Vec<usize> = (0..(W * H) as usize)
        .filter(|i| traced[*i] != RT_MISS)
        .collect();
    let rt_shadowed: Vec<bool> = traced.iter().map(|v| *v == RT_SHADOWED).collect();

    // ANTI-VACUITY, both masks, before any comparison.
    let rt_dark = covered.iter().filter(|i| rt_shadowed[**i]).count();
    let vsm_dark = covered.iter().filter(|i| vsm_shadowed[**i]).count();
    assert!(
        !covered.is_empty() && rt_dark > 64 && rt_dark < covered.len() - 64,
        "the traced mask is trivial ({rt_dark} of {} covered)",
        covered.len()
    );
    assert!(
        vsm_dark > 64 && vsm_dark < covered.len() - 64,
        "the shipped mask is trivial ({vsm_dark} of {} covered) — the virtual \
         shadow map did not engage and this arm compares against nothing",
        covered.len()
    );

    let disagree: Vec<usize> = covered
        .iter()
        .copied()
        .filter(|i| rt_shadowed[*i] != vsm_shadowed[*i])
        .collect();
    // Boundary = a pixel with a neighbour of the other class in EITHER mask.
    // A disagreement there is the two techniques' edges landing a pixel apart;
    // a disagreement away from any edge is a different shadow.
    let boundary = |i: usize, m: &[bool]| -> bool {
        let (x, y) = (i as u32 % W, i as u32 / W);
        if x == 0 || y == 0 || x + 1 >= W || y + 1 >= H {
            return true;
        }
        [i - 1, i + 1, i - W as usize, i + W as usize]
            .iter()
            .any(|k| m[*k] != m[i])
    };
    let at_edge = disagree
        .iter()
        .filter(|i| boundary(**i, &rt_shadowed) || boundary(**i, &vsm_shadowed))
        .count();

    let agree = covered.len() - disagree.len();
    println!(
        "RAY QUERY vs VSM — covered {} of {} pixels ({:.1} %); agree {agree} \
         ({:.2} %); disagree {} of which {at_edge} ({:.1} %) at a shadow edge",
        covered.len(),
        W * H,
        covered.len() as f64 / f64::from(W * H) * 100.0,
        agree as f64 / covered.len() as f64 * 100.0,
        disagree.len(),
        at_edge as f64 / disagree.len().max(1) as f64 * 100.0,
    );
    println!(
        "  traced shadowed {rt_dark}, shipped shadowed {vsm_dark}, of {} covered",
        covered.len()
    );

    assert!(
        agree * 10 >= covered.len() * 8,
        "the two techniques agree on only {:.1} % of the covered frame — they \
         are not finding the same shadow",
        agree as f64 / covered.len() as f64 * 100.0
    );
    assert!(
        at_edge * 4 >= disagree.len() * 3,
        "only {at_edge} of {} disagreements are at a shadow edge — the rest are \
         a different shadow, not the same one offset",
        disagree.len()
    );
}

/// **The surface offset is load-bearing, and this is the arm that says so.**
///
/// `RtView::shadow_bias` is a parameter rather than a constant precisely so a
/// gate can take it to zero. A shadow ray fired from exactly the hit point
/// re-hits the triangle it came from at `t = 0` on any real intersector, so
/// every lit surface reads occluded and the whole covered frame goes dark —
/// which is a *plausible-looking* shadow mask, and is the failure this bias
/// exists to prevent.
///
/// Both sides are checked, because both take the parameter: the device's trace
/// and this file's own CPU caster must **both** collapse. A bias that only the
/// shader read would show up as the two disagreeing, and that is a different
/// defect from this one.
#[test]
fn a_zero_surface_offset_shadows_every_surface_with_itself() {
    let Some(gpu) = ray_query_or_skip("the surface offset") else {
        return;
    };
    let (rt, _) = build_scene(&gpu);
    let pass = RtSunShadow::new(&gpu);
    let good = wide_rt_view();
    let zero = RtView {
        shadow_bias: 0.0,
        ..good
    };
    let tris = world_triangles();

    let dark = |v: &[u32]| v.iter().filter(|x| **x == RT_SHADOWED).count();
    let covered = |v: &[u32]| v.iter().filter(|x| **x != RT_MISS).count();

    let ok = pass.trace(&gpu, &rt, &good, W, H);
    let bad = pass.trace(&gpu, &rt, &zero, W, H);
    let cpu_ok = cpu_verdicts(&tris, &good);
    let cpu_bad = cpu_verdicts(&tris, &zero);

    println!(
        "surface offset {} m -> {} shadowed of {} covered; 0 m -> {} (cpu {} -> {})",
        good.shadow_bias,
        dark(&ok),
        covered(&ok),
        dark(&bad),
        dark(&cpu_ok),
        dark(&cpu_bad),
    );

    // ANTI-VACUITY: at the shipped offset the frame is mostly LIT, or "it went
    // dark" is not a change.
    assert!(
        dark(&ok) * 2 < covered(&ok),
        "the shipped offset already shadows most of the frame"
    );
    // **Both sides collapse**, and by a lot. Measured on this adapter: 830 of
    // 19 322 covered pixels at 1 mm, **7 697** at zero on the device and
    // **14 123** in the CPU caster.
    assert!(
        dark(&bad) > dark(&ok) * 5 && dark(&cpu_bad) > dark(&cpu_ok) * 5,
        "a zero surface offset barely changed the mask ({} -> {} device, {} -> \
         {} cpu) — this intersector does not self-hit and the bias is measuring \
         nothing",
        dark(&ok),
        dark(&bad),
        dark(&cpu_ok),
        dark(&cpu_bad)
    );

    // **And what it produces is NOISE rather than a shadow**, which is the real
    // finding and the stronger assertion: at the shipped offset the device and
    // the CPU caster agree pixel for pixel, and at zero they do not agree at
    // all. Neither goes uniformly dark — `eye + dir · t` lands microscopically
    // above or below its own triangle depending on the last bits, so each
    // intersector speckles its own way. A bias is not a tuning knob here; it is
    // what makes the answer a function of the geometry instead of of the
    // rounding.
    let differ = |a: &[u32], b: &[u32]| (0..a.len()).filter(|i| a[*i] != b[*i]).count();
    assert_eq!(
        differ(&ok, &cpu_ok),
        0,
        "the two sides already disagree at the shipped offset"
    );
    let noise = differ(&bad, &cpu_bad);
    println!("  at a zero offset the two intersectors differ on {noise} pixels");
    assert!(
        noise * 20 > covered(&bad),
        "at a zero offset the two intersectors still agree on all but {noise} \
         pixels, so the self-hit is deterministic and this arm is not measuring \
         what it says"
    );
}
