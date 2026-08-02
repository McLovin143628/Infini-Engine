//! Two-pass HZB occlusion gates (P18.1).
//!
//! The pipeline under test: early draw of last frame's visible meshlets → HZB
//! from the depth they wrote (min over MSAA subsamples) → late cull + draw of the
//! remainder. Everything here rests on one contract, stated in
//! `inf_render::passes::vgeom`:
//!
//! > **Occlusion is purely subtractive.** The HZB test only ever removes meshlets
//! > it *proves* contribute zero fragments, so for the same frame inputs the
//! > occlusion-on image is **pixel-identical** to the occlusion-off image — under
//! > any temporal state, at any point in the convergence, including the frame
//! > after a teleport.
//!
//! That turns a temporal feature into something the house determinism doctrine
//! can actually gate: every test below that renders occluded geometry asserts
//! byte equality against the unoccluded reference, and the counters
//! ([`VgeomAudit`]) are asserted separately to prove the culling is real and not
//! a no-op that trivially satisfies the equality.
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    is_camera_cut, EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomAudit, VgeomInstance, VgeomMesh, VgeomSettings,
    HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP vgeom_occlusion: no GPU adapter ({e})");
            None
        }
    }
}

// ── Content ──────────────────────────────────────────────────────────────────

// The same dense procedural mesh the vgeom goldens use: an `n×n` grid quad-plane
// in the local XZ plane, displaced by a smooth bi-sinusoid so it has real
// curvature (nontrivial normal cones + several LOD levels). One shared generator
// in `inf_vgeom::test_support`, with bit-portable trig — this file used to carry
// its own copy, displaced with `std` sin/cos, which cooked a different DAG per
// platform.
use inf_vgeom::test_support::dense_grid_mesh;

const ASSET: u128 = 0x1801_0000_0cc1_0000;

/// The occlusion fixture: a **wall** of meshlets filling the view, and a second,
/// smaller meshlet body parked well behind it. Rotating the plane +90° about X
/// stands it up in world XY with its normals along +Z, i.e. facing the camera —
/// so it rasterizes (the pipeline back-face culls) and writes the depth the HZB
/// then has to prove things against.
///
/// Geometry (camera at `z = +6`, 60° fov, 320×180):
/// * wall — scale 8 ⇒ ±8 m in XY at z ∈ [−2.4, 2.4]; the frame's half-width at
///   the nearest wall depth is ~3.7 m, so it covers the viewport completely.
/// * hidden — scale 2 at z = −20 ⇒ ±2 m, 26 m out, entirely inside the wall's
///   screen coverage and entirely behind it.
fn occluder_scene(mesh: Arc<VgeomMesh>) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &mesh).expect("index the vmesh")],
        ..Default::default()
    };
    let standing = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    scene.vgeom_instances.push(VgeomInstance::lit(
        ASSET,
        DVec3::ZERO,
        standing,
        Vec3::splat(8.0),
        [0.72, 0.52, 0.30, 1.0],
        1,
    ));
    scene.vgeom_instances.push(VgeomInstance::lit(
        ASSET,
        DVec3::new(0.0, 0.0, -20.0),
        standing,
        Vec3::splat(2.0),
        [0.20, 0.65, 0.85, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    scene
}

fn look_from(eye: DVec3, at: DVec3, w: u32, h: u32) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: w,
        height: h,
        ortho: None,
    }
}

fn front_view() -> RenderView {
    look_from(DVec3::new(0.0, 0.0, 6.0), DVec3::ZERO, W, H)
}

// ── Settings ─────────────────────────────────────────────────────────────────

fn vgeom(occlusion: bool, two_pass: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion,
            two_pass,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// The reference: the meshlet path with the HZB test off entirely. Every
/// occlusion mode must reproduce this image byte for byte.
fn unoccluded() -> RenderSettings {
    vgeom(false, false)
}

// ── Rig ──────────────────────────────────────────────────────────────────────

/// One renderer + headless target, driven for several frames so the temporal
/// (last-frame-visible) state actually accumulates. A fresh `Rig` is a cold
/// renderer — which is exactly the state every golden renders from.
struct Rig {
    renderer: EngineRenderer,
    target: HeadlessTarget,
    w: u32,
    h: u32,
}

impl Rig {
    fn new(gpu: &GpuContext, w: u32, h: u32, settings: RenderSettings) -> Self {
        let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        Self {
            renderer,
            target: HeadlessTarget::new(gpu, w, h),
            w,
            h,
        }
    }

    fn audited(gpu: &GpuContext, w: u32, h: u32, settings: RenderSettings) -> Self {
        let mut rig = Self::new(gpu, w, h, settings);
        rig.renderer.set_vgeom_audit(true);
        rig
    }

    fn frame(&mut self, gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
        self.renderer
            .render(gpu, scene, view, &self.target.view, (self.w, self.h));
        self.target.read_rgba(gpu).expect("readback")
    }

    fn audit(&self, gpu: &GpuContext) -> VgeomAudit {
        self.renderer.vgeom_audit(gpu)
    }
}

/// Render `n` frames of the same scene+view on one renderer, returning them all.
fn frames(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
    n: usize,
) -> Vec<Vec<u8>> {
    let mut rig = Rig::new(gpu, W, H, settings);
    (0..n).map(|_| rig.frame(gpu, scene, view)).collect()
}

/// One frame from a cold renderer — the state every golden is captured in.
fn cold_frame(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    s: RenderSettings,
) -> Vec<u8> {
    Rig::new(gpu, W, H, s).frame(gpu, scene, view)
}

#[track_caller]
fn assert_identical(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: frame sizes differ");
    let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
    assert_eq!(
        diff,
        0,
        "{what}: {diff} of {} bytes differ — occlusion must be purely subtractive",
        a.len()
    );
}

/// How many pixels the frame actually paints (anything above the cleared sky) —
/// a guard that "pixel-identical" is not trivially satisfied by two empty frames.
fn painted(img: &[u8]) -> usize {
    img.chunks(4).filter(|p| p[0] > 24 || p[1] > 24).count()
}

// ── (a) the core correctness gate ────────────────────────────────────────────

/// **Converged two-pass == single-pass, pixel-identical.**
///
/// Eight frames on one renderer (frame 1 is conservative, 2+ inherit last frame's
/// visible set, and the set converges), then a byte comparison against
///   * a cold occlusion-**off** render — the subtractive contract, and
///   * a cold single-pass (v1) occlusion render — the two shapes agree.
///
/// Every intermediate frame is compared too: the invariant is per-frame, not just
/// at the fixed point, so a mid-convergence hole is a failure.
#[test]
fn converged_two_pass_matches_the_unoccluded_and_single_pass_frames() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));
    let view = front_view();

    let reference = cold_frame(&gpu, &scene, &view, unoccluded());
    assert!(
        painted(&reference) > (W * H / 8) as usize,
        "fixture must actually fill the frame with meshlet geometry"
    );

    let two = frames(&gpu, &scene, &view, vgeom(true, true), 8);
    for (i, f) in two.iter().enumerate() {
        assert_identical(f, &reference, &format!("two-pass frame {i}"));
    }

    let single = cold_frame(&gpu, &scene, &view, vgeom(true, false));
    assert_identical(&single, &reference, "single-pass (v1) occlusion");
}

// ── (b) occlusion actually culls, and only culls the invisible ───────────────

/// The **subtractive-only proof with teeth**: the audit readback must show a
/// nonzero occluded count on the same frames that are pixel-identical to the
/// unoccluded render. Without the counter check, (a) would also pass if the HZB
/// test never fired.
///
/// It also pins the *shape* of the convergence:
/// * frame 1 is conservative — the early pass draws the whole base cut and the
///   late pass adds nothing (`early_drawn == base_cut`, `late_drawn == 0`);
/// * a converged frame draws strictly less — the occluded meshlets have dropped
///   out of the early set (`early_drawn == base_cut - occluded`), which is the
///   entire point of persisting the visible list.
#[test]
fn occlusion_culls_a_nonzero_set_and_the_frame_is_unchanged() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));
    let view = front_view();
    let reference = cold_frame(&gpu, &scene, &view, unoccluded());

    let mut rig = Rig::audited(&gpu, W, H, vgeom(true, true));

    let first = rig.frame(&gpu, &scene, &view);
    let a1 = rig.audit(&gpu);
    assert_identical(&first, &reference, "first (conservative) two-pass frame");
    assert!(a1.base_cut > 0, "the base cut selected nothing: {a1:?}");
    assert!(
        a1.occluded > 0,
        "the wall must occlude some of the hidden body's meshlets: {a1:?}"
    );
    assert_eq!(
        (a1.early_drawn, a1.late_drawn),
        (a1.base_cut, 0),
        "a cold frame must be conservative — early set == the whole base cut: {a1:?}"
    );

    // Converge.
    let mut audit = a1;
    for _ in 0..7 {
        let f = rig.frame(&gpu, &scene, &view);
        audit = rig.audit(&gpu);
        assert_identical(&f, &reference, "converging two-pass frame");
    }
    assert_eq!(
        audit.base_cut, a1.base_cut,
        "the base cut is temporal-state-free: {audit:?} vs {a1:?}"
    );
    assert_eq!(
        audit.occluded, a1.occluded,
        "occlusion converged to the same subtracted set: {audit:?} vs {a1:?}"
    );
    assert_eq!(
        audit.early_drawn + audit.occluded,
        audit.base_cut,
        "converged early set == base cut minus the proven-occluded: {audit:?}"
    );
    assert_eq!(
        audit.late_drawn, 0,
        "nothing is newly disoccluded on a static converged frame: {audit:?}"
    );
    assert!(
        audit.early_drawn < a1.early_drawn,
        "convergence must actually save draws ({} → {})",
        a1.early_drawn,
        audit.early_drawn
    );
    // And the saving is real work, not rounding.
    eprintln!(
        "P18.1 occlusion: base cut {} pairs, {} proven occluded ({:.1}%), \
         drawn {} → {} after convergence",
        audit.base_cut,
        audit.occluded,
        100.0 * audit.occluded as f64 / audit.base_cut as f64,
        a1.early_drawn,
        audit.early_drawn,
    );
}

// ── (c) first-frame-conservative determinism ────────────────────────────────

/// **Double-render from cold == double-render from cold.** Two independent
/// renderers, driven through the same six frames, must agree frame by frame — the
/// temporal state is seeded identically (from nothing) and evolves identically,
/// so a frame is still a pure function of (scene, view, settings, frame ordinal).
///
/// The counters are compared too: they are the only observable that *can* differ
/// between the passes, so if the split ever became order-dependent this catches it
/// where the pixel comparison (which is subtractive-proof) could not.
#[test]
fn cold_start_is_frame_by_frame_deterministic() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));
    let view = front_view();

    let mut a = Rig::audited(&gpu, W, H, vgeom(true, true));
    let mut b = Rig::audited(&gpu, W, H, vgeom(true, true));
    for i in 0..6 {
        let fa = a.frame(&gpu, &scene, &view);
        let aa = a.audit(&gpu);
        let fb = b.frame(&gpu, &scene, &view);
        let ab = b.audit(&gpu);
        assert_identical(&fa, &fb, &format!("cold-start frame {i}"));
        assert_eq!(aa, ab, "cold-start frame {i} counters diverged");
    }
}

// ── (d) a camera cut leaves no holes ────────────────────────────────────────

/// A teleport invalidates last frame's visible set. Two things must hold on the
/// frame that lands:
///   1. it is **conservative** — the whole base cut goes through the early pass
///      (this is the [`is_camera_cut`] trigger doing its job), and
///   2. it has **no holes** — byte-identical to a cold unoccluded render of that
///      same view. Note (2) is guaranteed by the subtractive contract *regardless*
///      of (1); the cut heuristic is a cost optimisation, and this test pins both
///      so a future change to the threshold cannot quietly become load-bearing.
#[test]
fn a_camera_cut_renders_a_conservative_frame_with_no_holes() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));
    let near = front_view();
    // 54 m in one frame — a teleport, not a walk.
    let far = look_from(DVec3::new(0.0, 0.0, 60.0), DVec3::ZERO, W, H);
    assert!(is_camera_cut(&near, &far), "fixture must trip the cut test");

    let mut rig = Rig::audited(&gpu, W, H, vgeom(true, true));
    for _ in 0..6 {
        rig.frame(&gpu, &scene, &near);
    }
    let converged = rig.audit(&gpu);
    assert!(converged.late_drawn == 0 && converged.early_drawn < converged.base_cut);

    let cut_frame = rig.frame(&gpu, &scene, &far);
    let cut_audit = rig.audit(&gpu);
    assert_eq!(
        (cut_audit.early_drawn, cut_audit.late_drawn),
        (cut_audit.base_cut, 0),
        "the frame after a teleport must be conservative: {cut_audit:?}"
    );

    let reference = cold_frame(&gpu, &scene, &far, unoccluded());
    assert!(
        painted(&reference) > 200,
        "the teleported view must show geometry"
    );
    assert_identical(&cut_frame, &reference, "frame after a camera cut");

    // And it re-converges from there without ever diverging from the reference.
    for _ in 0..5 {
        let f = rig.frame(&gpu, &scene, &far);
        assert_identical(&f, &reference, "re-convergence after the cut");
    }
}

/// A small pan is NOT a cut (otherwise two-pass would reset every frame in a
/// moving game and buy nothing), while a teleport, a snap turn and a resolution
/// change all are. Pure — no GPU.
#[test]
fn camera_cut_thresholds() {
    let base = front_view();
    assert!(
        !is_camera_cut(&base, &base),
        "identical views are not a cut"
    );

    let pan = look_from(DVec3::new(0.4, 0.0, 6.0), DVec3::new(0.4, 0.0, 0.0), W, H);
    assert!(!is_camera_cut(&base, &pan), "a 40 cm step is not a cut");

    let teleport = look_from(DVec3::new(0.0, 0.0, 200.0), DVec3::ZERO, W, H);
    assert!(is_camera_cut(&base, &teleport), "a 194 m jump is a cut");

    let snap = look_from(DVec3::new(0.0, 0.0, 6.0), DVec3::new(0.0, 0.0, 12.0), W, H);
    assert!(is_camera_cut(&base, &snap), "a 180° turn is a cut");

    let resized = look_from(DVec3::new(0.0, 0.0, 6.0), DVec3::ZERO, W * 2, H);
    assert!(
        is_camera_cut(&base, &resized),
        "a resolution change is a cut"
    );
}

// ── (g) resizable-resource discipline ───────────────────────────────────────

/// The HZB is viewport-sized and its bind groups embed a `FrameTargets` view, so
/// they are `GenCache`d on `(targets.generation, hzb.generation)`. A stale key
/// there samples the *previous* size's pyramid — no validation error, just wrong
/// occlusion. Resize away and back on ONE renderer (which reallocates the targets
/// and the pyramid twice, and returns to a size whose cache entry has since been
/// evicted) and require every frame to still match a cold render of that size.
#[test]
fn resizing_rebuilds_the_hzb_and_its_bind_groups() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));

    // Odd dimensions on purpose: the mip chain floors, so an odd level would drop
    // a trailing row/column from the min-downsample if `cs_down` did not extend
    // its gather — a coverage hole that over-culls.
    for (w, h) in [(W, H), (211, 97), (W, H)] {
        let view = look_from(DVec3::new(0.0, 0.0, 6.0), DVec3::ZERO, w, h);
        let mut rig = Rig::new(&gpu, w, h, vgeom(true, true));
        let mut two = Vec::new();
        for _ in 0..4 {
            two.push(rig.frame(&gpu, &scene, &view));
        }
        let mut plain = Rig::new(&gpu, w, h, unoccluded());
        let reference = plain.frame(&gpu, &scene, &view);
        for (i, f) in two.iter().enumerate() {
            assert_identical(f, &reference, &format!("{w}x{h} two-pass frame {i}"));
        }
    }
}

// ── Settings surface ────────────────────────────────────────────────────────

/// Occlusion is on by default now, and single-pass remains reachable as a
/// settings-selectable fallback that behaves identically on screen.
#[test]
fn defaults_and_fallback() {
    let d = VgeomSettings::default();
    assert!(d.occlusion, "P18.1: occlusion ships on");
    assert!(d.two_pass, "P18.1: two-pass ships on");
    assert!(
        !d.enabled,
        "the meshlet path itself is still opt-in per host"
    );

    let Some(gpu) = gpu_or_skip() else { return };
    let scene = occluder_scene(Arc::new(dense_grid_mesh(40)));
    let view = front_view();
    let reference = cold_frame(&gpu, &scene, &view, unoccluded());
    // `two_pass = false` with occlusion on ⇒ v1: one cull, one draw, HZB from the
    // depth already in the scene target. Still subtractive, still identical.
    for f in frames(&gpu, &scene, &view, vgeom(true, false), 3) {
        assert_identical(&f, &reference, "single-pass fallback");
    }

    // The audit means the same thing in single-pass: its one dispatch sees every
    // pair, so `base_cut` matches two-pass exactly. It reports no late draws
    // (there is no late pass) and — because the pyramid only holds classic depth
    // when this node starts, and this scene has no classic geometry — it proves
    // nothing occluded, which is precisely why two-pass exists.
    let mut one = Rig::audited(&gpu, W, H, vgeom(true, false));
    for _ in 0..3 {
        one.frame(&gpu, &scene, &view);
    }
    let single = one.audit(&gpu);

    let mut both = Rig::audited(&gpu, W, H, vgeom(true, true));
    for _ in 0..3 {
        both.frame(&gpu, &scene, &view);
    }
    let two = both.audit(&gpu);

    assert_eq!(
        single.base_cut, two.base_cut,
        "the base cut does not depend on the occlusion shape"
    );
    assert_eq!(single.late_drawn, 0, "single-pass has no late pass");
    assert_eq!(
        single.occluded, 0,
        "v1 can only occlude against classic depth, and this scene has none: {single:?}"
    );
    assert!(
        two.occluded > single.occluded,
        "two-pass must cull strictly more than v1 on a vgeom-occludes-vgeom scene \
         ({two:?} vs {single:?})"
    );

    // Occlusion off entirely ⇒ the pre-P18.1 path, unchanged across frames.
    for f in frames(&gpu, &scene, &view, unoccluded(), 3) {
        assert_identical(&f, &reference, "occlusion off");
    }
}
