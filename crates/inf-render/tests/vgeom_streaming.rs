//! Meshlet streaming gates (P18.2).
//!
//! The pipeline under test: one streaming sync point per frame (a pure function of
//! wants, current residency and budget) → the four suballocated meshlet pools →
//! a cull compute whose LOD cut is *clamped* to the finest resident level. The
//! contract is stated in `inf_vgeom::stream` and `inf_render::passes::vgeom`:
//!
//! > **Residency changes WHICH meshlet on a root-to-leaf path draws, never
//! > WHETHER one does.** Pages are ordered coarsest-first with page 0 = every
//! > root and are never evicted out of prefix order, so a budget clamp yields
//! > softer detail, never a hole — and at the streamer's own *wanted* floor the
//! > clamped cut is byte-for-byte the cut the pre-P18.2 whole-upload path
//! > produced, which is why all 36 goldens stayed byte-identical.
//!
//! Every test below therefore asserts one of three things: the clamped GPU cut
//! equals the CPU twin exactly (parity), a scripted flythrough reproduces a
//! byte-identical pixel *and* residency trace (determinism), or a clamped frame
//! still covers the reference frame's geometry (no holes). The audit counter is
//! asserted separately so "identical" can never be satisfied by a clamp that
//! never fired.
//!
//! # The one place "never a hole" stops being true, and why the sweeps avoid it
//!
//! The guarantee has a floor: it holds only while page 0 itself is resident, and
//! page 0 needs to fit. Since the four pools share **one** budget (they used to
//! carry fixed shares, and a share-based split could refuse the roots while the
//! byte ledger said there was room — see `VgeomPools`), that reduces to the one
//! honest condition `budget_bytes >= page0.resident_bytes()`; on this fixture
//! page 0 costs 7 828 B. Below it the asset has **zero** resident pages and the
//! frame draws nothing, and `AssetResidency::floor_lod` then reports `max_lod` —
//! the same value a roots-only residency reports — so only `resident_pages == 0`
//! distinguishes the two. Every budget in the sweeps below is comfortably above
//! that floor, and the sweeps assert `resident_pages >= 1` so a future fixture
//! change that crosses it fails loudly instead of silently testing an empty
//! frame. (`inf_vgeom`'s `the_root_page_survives_every_budget_that_can_hold_it`
//! pins the floor itself, with no GPU.)
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{DVec3, Mat3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::passes::vgeom::{cpu_visible_set, cull_flags, frustum_planes, lod_threshold};
use inf_render::{
    cull_visible_streamed, EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight,
    RenderScene, RenderSettings, RenderView, VgeomAsset, VgeomAudit, VgeomInstance, VgeomMesh,
    VgeomSettings, HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;

/// Grid resolution of the fixture mesh. 48 gives 4 608 triangles → 115 meshlets
/// over six LOD levels, which the container lays out as **six pages** — deep
/// enough that a budget sweep can land on several genuinely different residency
/// depths, small enough that seven cull readbacks and forty renders stay inside a
/// couple of seconds.
const GRID: usize = 48;

const ASSET: u128 = 0x1802_0000_57e1_0000;

fn gpu_or_skip(name: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP {name}: no adapter ({e})");
            None
        }
    }
}

// ── Content ──────────────────────────────────────────────────────────────────

/// The same dense procedural mesh the vgeom goldens and the occlusion gates use:
/// an `n×n` grid quad-plane in the local XZ plane, displaced by a smooth
/// bi-sinusoid so it has real curvature (nontrivial normal cones + several LOD
/// levels, hence several pages).
fn dense_grid_mesh(n: usize) -> VgeomMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;
            let y = 0.3 * (x * 3.0).sin() * (z * 3.0).cos();
            let dydx = 0.3 * 3.0 * (x * 3.0).cos() * (z * 3.0).cos();
            let dydz = -0.3 * 3.0 * (x * 3.0).sin() * (z * 3.0).sin();
            let nrm = Vec3::new(-dydx, 1.0, -dydz).normalize();
            positions.push([x, y, z]);
            normals.push(nrm.to_array());
            uvs.push([u, v]);
        }
    }
    let stride = (n + 1) as u32;
    let mut indices = Vec::new();
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &indices,
        inf_vgeom::BuildParams::default(),
    )
}

/// The streaming fixture: one instance of the dense mesh, scaled up to 4 m so a
/// close camera genuinely wants the finest pages and a distant one does not — the
/// spread the budget then has to arbitrate. `instances = false` renders the same
/// frame with the meshlet body removed, which is the exact background the
/// coverage gate measures against.
fn streaming_scene(mesh: &Arc<VgeomMesh>, instances: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, mesh).expect("index the vmesh")],
        ..Default::default()
    };
    if instances {
        scene.vgeom_instances.push(VgeomInstance::lit(
            ASSET,
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(4.0),
            [0.72, 0.52, 0.30, 1.0],
            1,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    scene
}

/// The single instance the cull-readback gates cut, matching `streaming_scene`.
fn fixture_instance() -> VgeomInstance {
    VgeomInstance::lit(
        ASSET,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(4.0),
        [0.72, 0.52, 0.30, 1.0],
        1,
    )
}

fn look_from(eye: DVec3, at: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// The fixed three-quarter view every non-flythrough test renders. The whole 4 m
/// body sits well inside a 60° frustum here, so no meshlet is near a frustum
/// boundary and float divergence cannot flip a cull.
fn overlook_view() -> RenderView {
    look_from(DVec3::new(0.0, 4.4, 8.4), DVec3::ZERO)
}

// ── Settings ─────────────────────────────────────────────────────────────────

/// The **cut** settings for the parity gates: cone culling off (its per-normal
/// boundary is the one place CPU and GPU can legitimately disagree, and it is
/// covered by the pure `cpu_visible_set` unit tests), occlusion off (the readback
/// path has no depth context anyway), so what is compared is exactly the
/// residency-clamped LOD + frustum base cut.
fn cut_settings(budget_bytes: u64) -> VgeomSettings {
    VgeomSettings {
        enabled: true,
        cone_cull: false,
        frustum_cull: true,
        occlusion: false,
        two_pass: false,
        pixel_error: 1.0,
        debug_meshlets: false,
        stream: inf_vgeom::VgeomStreamBudget {
            budget_bytes,
            ..inf_vgeom::VgeomStreamBudget::default()
        },
    }
}

/// The **render** settings: the shipping meshlet path (occlusion and two-pass on,
/// as they ship) with only the streaming budget varied, so what the pixel gates
/// measure is streaming and nothing else.
fn render_settings(budget_bytes: u64) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            stream: inf_vgeom::VgeomStreamBudget {
                budget_bytes,
                ..inf_vgeom::VgeomStreamBudget::default()
            },
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// The stock budget (256 MiB) — every shipping sample is want-limited under it,
/// which is the state the goldens were captured in.
fn default_settings() -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// A budget no fixture can exhaust — the "streaming is off in all but name" end
/// of the range.
const UNLIMITED_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

/// Budgets to sweep, **largest first**, as fractions of the asset's full VRAM
/// cost. Every divisor here leaves the budget above the page-0 floor described in
/// the module docs (the smallest is `total / 12` ≈ 11.5 KB against a ~10.3 KB
/// floor), and together they land on four distinct residency depths on this
/// fixture: 4, 3, 2 and 1 of six pages.
fn budget_sweep(total_resident_bytes: u64) -> Vec<u64> {
    [1u64, 2, 3, 4, 6, 8, 12]
        .into_iter()
        .map(|d| total_resident_bytes / d)
        .collect()
}

// ── Rig ──────────────────────────────────────────────────────────────────────

/// One renderer + headless target, driven for several frames so both the
/// streamer's residency and the two-pass temporal state actually accumulate. A
/// fresh `Rig` is a cold renderer with empty pools — the state every golden and
/// every shipping first frame starts from.
struct Rig {
    renderer: EngineRenderer,
    target: HeadlessTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, settings: RenderSettings) -> Self {
        let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        Self {
            renderer,
            target: HeadlessTarget::new(gpu, W, H),
        }
    }

    fn audited(gpu: &GpuContext, settings: RenderSettings) -> Self {
        let mut rig = Self::new(gpu, settings);
        rig.renderer.set_vgeom_audit(true);
        rig
    }

    fn frame(&mut self, gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
        self.renderer
            .render(gpu, scene, view, &self.target.view, (W, H));
        self.target.read_rgba(gpu).expect("readback")
    }

    /// Render `n` frames of one static view and return the last.
    fn settle(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        n: usize,
    ) -> Vec<u8> {
        let mut img = Vec::new();
        for _ in 0..n {
            img = self.frame(gpu, scene, view);
        }
        img
    }

    fn residency(&self) -> Residency {
        Residency::of(&self.renderer)
    }

    fn audit(&self, gpu: &GpuContext) -> VgeomAudit {
        self.renderer.vgeom_audit(gpu)
    }
}

/// One frame's residency observation — the streaming half of a render trace.
/// Everything here is a CPU counter the streamer already maintains, so capturing
/// it costs nothing and a trace can be compared byte for byte.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Residency {
    floor_lod: BTreeMap<u128, u32>,
    pages: BTreeMap<u128, (usize, usize)>,
    resident_bytes: u64,
    budget_clamped: bool,
}

impl Residency {
    fn of(renderer: &EngineRenderer) -> Self {
        let r = renderer.vgeom_stream_report();
        Self {
            floor_lod: r.floor_lod,
            pages: r.pages,
            resident_bytes: r.stats.resident_bytes,
            budget_clamped: r.stats.budget_clamped,
        }
    }

    fn floor(&self) -> u32 {
        self.floor_lod.get(&ASSET).copied().unwrap_or(u32::MAX)
    }

    fn resident_pages(&self) -> usize {
        self.pages.get(&ASSET).map_or(0, |p| p.0)
    }
}

#[track_caller]
fn assert_identical(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: frame sizes differ");
    let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
    assert_eq!(diff, 0, "{what}: {diff} of {} bytes differ", a.len());
}

/// Which pixels this frame paints over the background — the identically-rendered
/// frame with the meshlet body removed. Comparing against a real background
/// render rather than a brightness heuristic makes "covered" exact: the sky is
/// bit-identical in both, so every differing pixel is geometry.
fn coverage(img: &[u8], background: &[u8]) -> Vec<bool> {
    img.chunks(4)
        .zip(background.chunks(4))
        .map(|(a, b)| a != b)
        .collect()
}

/// The CPU twin of one instance's GPU cut at residency floor `floor_lod` — the
/// identical LOD + frustum filter, with the threshold derived from the same
/// projection the shader's per-instance scalar came from.
fn cpu_cut(
    mesh: &VgeomMesh,
    inst: &VgeomInstance,
    view: &RenderView,
    settings: &VgeomSettings,
    floor_lod: u8,
) -> Vec<u32> {
    let model = view
        .origin
        .model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = inst.scale.abs().max_element().max(1e-6);
    let normal_mat = Mat3::from_quat(inst.rotation)
        * Mat3::from_diagonal(inst.scale.max(Vec3::splat(1e-6)).recip());
    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(mesh.center));
    let t = lod_threshold(
        eye,
        center_world,
        mesh.radius * max_scale,
        max_scale,
        view,
        settings.pixel_error,
    );
    cpu_visible_set(
        mesh,
        model,
        normal_mat,
        eye,
        t,
        max_scale,
        &frustum_planes(view.view_proj()),
        cull_flags(settings),
        floor_lod,
    )
}

/// One row of a budget sweep: what the GPU cut and the streamer agreed on.
struct Sweep {
    budget: u64,
    floor_lod: u8,
    resident_pages: usize,
    total_pages: usize,
    visible: usize,
}

/// Run the cull readback once per budget, asserting CPU/GPU parity on every row.
/// Shared by the parity gate and the monotonicity gate so both see exactly the
/// same residency ladder.
fn sweep_cuts(gpu: &GpuContext, mesh: &VgeomMesh, budgets: &[u64]) -> Vec<Sweep> {
    let inst = fixture_instance();
    let view = overlook_view();
    budgets
        .iter()
        .map(|&budget| {
            let settings = cut_settings(budget);
            let rb =
                cull_visible_streamed(gpu, mesh, std::slice::from_ref(&inst), &view, &settings);
            assert!(
                rb.resident_pages >= 1,
                "budget {budget} left NOTHING resident — page 0 no longer fits its pool share, \
                 so this row tests an empty frame rather than a clamped one (see the module \
                 docs' page-0 floor). Raise the budget or shrink the fixture."
            );
            let cpu = cpu_cut(mesh, &inst, &view, &settings, rb.floor_lod);
            let gpu_ids: Vec<u32> = rb.pairs.iter().map(|p| p[1]).collect();
            assert_eq!(
                gpu_ids, cpu,
                "budget {budget} (floor_lod {}, {} of {} pages resident): the GPU's clamped cut \
                 must equal cpu_visible_set at the SAME floor",
                rb.floor_lod, rb.resident_pages, rb.total_pages
            );
            Sweep {
                budget,
                floor_lod: rb.floor_lod,
                resident_pages: rb.resident_pages,
                total_pages: rb.total_pages,
                visible: rb.pairs.len(),
            }
        })
        .collect()
}

// ── (1) the parity gate, under punched residency ─────────────────────────────

/// **The clamped cut is the CPU cut, at every residency depth.**
///
/// `golden.rs::vgeom_cpu_gpu_cut_parity` proves parity at the streamer's own
/// *wanted* floor, where the clamp is provably a no-op on the drawn set. That
/// leaves the interesting half untested: what the shader does when the budget
/// really did punch pages out and the clamp changes which meshlets draw. This
/// sweeps budgets down to a twelfth of the asset's full VRAM cost, lands on four
/// different residency depths, and requires the GPU's visible set to equal
/// `cpu_visible_set` at the floor the readback reports — exactly, not
/// approximately.
///
/// That equality is what makes `VgeomMesh::select_with_residency` a usable
/// reference for everything else in the engine (the player's activation check,
/// the cook's coverage assertions): if the branchless `eff_error = (lod_level <=
/// floor_lod) ? 0 : error` in `vgeom_cull.wgsl` ever drifted from the Rust twin,
/// no pixel test would catch it — a coarser-but-complete surface still looks
/// plausible — and this one fails immediately.
///
/// Two non-vacuity guards: at least one budget must be genuinely partially
/// resident, and at least one must sit *below* what an unlimited budget would
/// hold, so the sweep cannot pass by never clamping anything.
#[test]
fn cut_clamp_parity_cpu_equals_gpu_under_punched_residency() {
    let Some(gpu) = gpu_or_skip("cut_clamp_parity") else {
        return;
    };
    let mesh = dense_grid_mesh(GRID);
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let total = source.total_resident_bytes();

    // The want-limited reference: what this camera asks for when nothing is
    // scarce. Every clamped row below must be coarser than or equal to it.
    let unlimited = sweep_cuts(&gpu, &mesh, &[UNLIMITED_BUDGET]).remove(0);
    let rows = sweep_cuts(&gpu, &mesh, &budget_sweep(total));

    eprintln!(
        "P18.2 cut parity: {} meshlets in {} pages, {total} B fully resident; unlimited budget \
         holds {}/{} pages (floor_lod {}, {} visible)",
        mesh.meshlet_count(),
        unlimited.total_pages,
        unlimited.resident_pages,
        unlimited.total_pages,
        unlimited.floor_lod,
        unlimited.visible
    );
    for r in &rows {
        eprintln!(
            "  budget {:>9} B: {}/{} pages, floor_lod {}, {} meshlets visible (== CPU)",
            r.budget, r.resident_pages, r.total_pages, r.floor_lod, r.visible
        );
    }

    assert!(
        rows.iter().any(|r| r.resident_pages < r.total_pages),
        "no budget produced partial residency — the parity sweep would be vacuous"
    );
    assert!(
        rows.iter().any(|r| r.floor_lod > unlimited.floor_lod),
        "no budget clamped BELOW what the camera wanted (every row sat at floor_lod {}), so the \
         sweep only re-tested the want-limited case golden.rs already covers",
        unlimited.floor_lod
    );
}

// ── (2) a clamp coarsens monotonically and never empties the frame ───────────

/// **Shrinking the budget only ever coarsens.** Across the same ladder, the
/// visible meshlet count is non-increasing, the residency floor is
/// non-decreasing, and the cut is never empty.
///
/// This is the degradation curve the whole feature promises, and it is not
/// self-evident from the code: the clamp *zeroes* a meshlet's error, which widens
/// the interval it wins, so one could imagine a floor that adds meshlets rather
/// than removing them. It cannot, because along a root-to-leaf path the clamped
/// intervals still tile `[0, ∞)` and the selected meshlet simply moves to level
/// `max(floor, L)` — a coarsening, and coarser levels have strictly fewer
/// meshlets. Asserting the monotonicity end to end pins that argument against the
/// shader, not just against the prose.
///
/// "Never zero" is the same claim the no-holes pixel gate makes, measured at the
/// cut instead of at the frame: page 0 is every root, it is never evicted, so the
/// cut always has something to select.
#[test]
fn a_budget_clamped_frame_is_coarser_but_never_empty() {
    let Some(gpu) = gpu_or_skip("budget_clamp_monotonicity") else {
        return;
    };
    let mesh = dense_grid_mesh(GRID);
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let rows = sweep_cuts(&gpu, &mesh, &budget_sweep(source.total_resident_bytes()));

    // `budget_sweep` is ordered largest budget first, so each row is the same
    // camera with strictly less VRAM than the row before it.
    for pair in rows.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            b.budget < a.budget,
            "the sweep must shrink monotonically ({} then {})",
            a.budget,
            b.budget
        );
        assert!(
            b.visible <= a.visible,
            "a smaller budget drew MORE meshlets: {} B → {} visible, then {} B → {} visible",
            a.budget,
            a.visible,
            b.budget,
            b.visible
        );
        assert!(
            b.floor_lod >= a.floor_lod,
            "a smaller budget resolved a FINER floor: {} B → floor_lod {}, then {} B → floor_lod {}",
            a.budget,
            a.floor_lod,
            b.budget,
            b.floor_lod
        );
    }
    for r in &rows {
        assert!(
            r.visible > 0,
            "budget {} B produced an EMPTY cut at floor_lod {} with {}/{} pages resident — \
             residency must degrade to the root floor, never to nothing",
            r.budget,
            r.floor_lod,
            r.resident_pages,
            r.total_pages
        );
    }
    let coarsest = rows.last().expect("the sweep is non-empty");
    let finest = &rows[0];
    assert!(
        coarsest.visible < finest.visible && coarsest.floor_lod > finest.floor_lod,
        "the ladder must actually descend ({} visible at floor {} → {} visible at floor {})",
        finest.visible,
        finest.floor_lod,
        coarsest.visible,
        coarsest.floor_lod
    );
    eprintln!(
        "P18.2 degradation: {} → {} visible meshlets as floor_lod goes {} → {} over budgets \
         {} → {} B",
        finest.visible,
        coarsest.visible,
        finest.floor_lod,
        coarsest.floor_lod,
        finest.budget,
        coarsest.budget
    );
}

// ── (3) the determinism gate ─────────────────────────────────────────────────

/// A scripted camera move: dolly in over six frames, back out over five, then a
/// snap to point-blank range. The dolly walks the want up through several pages,
/// eviction hysteresis makes the way out *not* retrace the way in, and the snap
/// is a camera cut that also invalidates the two-pass temporal set — so the trace
/// exercises loads, evictions, a budget clamp and a conservative frame.
fn flythrough() -> Vec<RenderView> {
    let mut views = Vec::new();
    for k in 0..6 {
        let f = f64::from(k) / 5.0;
        views.push(look_from(
            DVec3::new(0.0, 9.0 - 6.0 * f, 17.0 - 11.5 * f),
            DVec3::ZERO,
        ));
    }
    for k in 1..6 {
        let f = f64::from(k) / 5.0;
        views.push(look_from(
            DVec3::new(0.0, 3.0 + 6.0 * f, 5.5 + 11.5 * f),
            DVec3::ZERO,
        ));
    }
    views.push(look_from(DVec3::new(0.0, 1.6, 2.4), DVec3::ZERO));
    views
}

/// **A streamed flythrough is reproducible, frame for frame, in pixels and in
/// residency.**
///
/// This is the gate the whole "wants are CPU-side, not cull feedback" decision
/// exists to make passable. Residency is frame-history-dependent state — what is
/// paged in this frame depends on what was paged in last frame, through the
/// hysteresis band and the bounded load batch — which is precisely the shape of
/// state that normally destroys reproducibility. `VgeomStreamer::plan` is a pure
/// function of `(wants, residency, budget)` with a total grant order, so two cold
/// renderers driven through the same views must produce identical residency, and
/// therefore identical pixels.
///
/// Both halves are asserted. Pixels alone would not be enough: a bug that paged
/// in a different prefix but still drew the same coarse surface would pass, and
/// then diverge under a different camera. Residency alone would not be enough
/// either: the pools' offsets are what the shader reads through, so an allocator
/// that handed out the same page count at different addresses would corrupt the
/// image while the counters agreed.
///
/// The trace is also required to *move* — a budget that pinned residency for
/// twelve frames would make this a very expensive tautology.
#[test]
fn streaming_renders_a_deterministic_pixel_trace() {
    let Some(gpu) = gpu_or_skip("streaming_determinism") else {
        return;
    };
    let mesh = Arc::new(dense_grid_mesh(GRID));
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let scene = streaming_scene(&mesh, true);
    let views = flythrough();

    // Half the asset's full cost: enough for the far end of the dolly to be
    // want-limited, not enough for the near end — so the trace both converges and
    // clamps.
    let budget = source.total_resident_bytes() / 2;

    let run = |gpu: &GpuContext| -> Vec<(Vec<u8>, Residency)> {
        let mut rig = Rig::new(gpu, render_settings(budget));
        views
            .iter()
            .map(|v| {
                let img = rig.frame(gpu, &scene, v);
                (img, rig.residency())
            })
            .collect()
    };

    let a = run(&gpu);
    let b = run(&gpu);
    assert_eq!(a.len(), views.len());
    for (i, ((ia, ra), (ib, rb))) in a.iter().zip(&b).enumerate() {
        assert_eq!(ra, rb, "flythrough frame {i}: residency traces diverged");
        assert_identical(ia, ib, &format!("flythrough frame {i}"));
    }

    let floors: Vec<u32> = a.iter().map(|(_, r)| r.floor()).collect();
    let pages: Vec<usize> = a.iter().map(|(_, r)| r.resident_pages()).collect();
    eprintln!(
        "P18.2 flythrough (budget {budget} B): resident pages {pages:?}, floor_lod {floors:?}"
    );
    assert!(
        pages.iter().any(|p| *p != pages[0]),
        "residency never moved across the flythrough ({pages:?}) — the budget or the camera \
         script no longer exercises paging, so this determinism gate is vacuous"
    );
    assert!(
        a.iter().any(|(_, r)| r.budget_clamped),
        "no frame in the flythrough was budget-clamped — the trace never left the want-limited \
         regime the goldens already cover"
    );
    assert!(
        pages.iter().all(|p| *p >= 1),
        "a flythrough frame lost its root floor entirely ({pages:?})"
    );
}

// ── (4) the equivalence twin of the goldens ──────────────────────────────────

/// **An unlimited budget renders the stock frame, byte for byte.**
///
/// The primary equivalence gate for this claim is the golden suite: all 36
/// committed images were re-rendered with streaming on and stayed byte-identical,
/// which is a far broader proof than any single fixture. This is its focused
/// twin — the one place the claim is asserted *as a claim*, against a budget
/// deliberately set to a number no content can exhaust, so a regression that only
/// showed up as "the goldens need re-blessing" is instead attributed here.
///
/// The residency report is asserted too, because "identical" would also hold if
/// both frames were clamped the same way. Both must show the streamer freely
/// granting everything the camera asked for: nothing budget-clamped, no asset
/// short of its want. Note that "full residency" is deliberately *not* the
/// assertion — the streamer declines to page in pages the cut could never select
/// (`ideal_page_count`), so want-limited is the correct and expected state.
#[test]
fn an_unlimited_budget_is_pixel_identical_to_the_reference_frame() {
    let Some(gpu) = gpu_or_skip("unlimited_budget_equivalence") else {
        return;
    };
    let mesh = Arc::new(dense_grid_mesh(GRID));
    let scene = streaming_scene(&mesh, true);
    let view = overlook_view();

    let mut stock = Rig::new(&gpu, default_settings());
    let mut huge = Rig::new(&gpu, render_settings(UNLIMITED_BUDGET));
    for i in 0..3 {
        let a = stock.frame(&gpu, &scene, &view);
        let b = huge.frame(&gpu, &scene, &view);
        assert_identical(
            &a,
            &b,
            &format!("frame {i}: default budget vs an unexhaustible one"),
        );
    }

    let (ra, rb) = (stock.residency(), huge.residency());
    assert_eq!(
        ra, rb,
        "the two budgets settled on different residency: {ra:?} vs {rb:?}"
    );
    for (label, r) in [("default", &ra), ("unlimited", &rb)] {
        assert!(
            !r.budget_clamped,
            "{label} budget reported a clamp — this gate must compare two UNclamped frames: {r:?}"
        );
        assert!(
            r.resident_pages() >= 1,
            "{label} budget holds no pages at all: {r:?}"
        );
    }
    eprintln!(
        "P18.2 equivalence: default and unlimited budgets both settle at {}/{} pages, floor_lod \
         {}, {} B resident",
        ra.resident_pages(),
        ra.pages.get(&ASSET).map_or(0, |p| p.1),
        ra.floor(),
        ra.resident_bytes
    );
}

// ── (5) no holes ─────────────────────────────────────────────────────────────

/// **A clamped frame still covers the geometry the reference frame drew.**
///
/// The cut-level gates prove the clamped set is non-empty; this proves the
/// *surface* stays closed on screen, which is the property a player would
/// actually notice. A coarser LOD moves silhouettes and re-shades interiors, so
/// byte equality is out of the question — what must hold is coverage: nearly
/// every pixel the reference frame painted must still be painted.
///
/// The threshold is 99 %, chosen from what this fixture produces. Measured on the
/// P18.2 implementation, a budget of a sixth of the asset (residency floor 4 of
/// 6, two pages of six, five meshlets instead of seventeen) retains **99.95 %** of
/// the reference's covered pixels, and even the extreme roots-only clamp retains
/// 98.3 % — the residual in both cases is silhouette drift of a pixel or two at
/// the body's edge, not a hole in its middle. 99 % therefore has ample margin
/// against a real hole, which would cost whole percent-scale regions at once.
///
/// The clamped frame is also required to differ from the reference: if the clamp
/// did nothing, coverage would trivially be 100 % and the test would prove
/// nothing.
#[test]
fn residency_never_leaves_a_hole_in_the_frame() {
    let Some(gpu) = gpu_or_skip("residency_no_holes") else {
        return;
    };
    let mesh = Arc::new(dense_grid_mesh(GRID));
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let scene = streaming_scene(&mesh, true);
    let bare = streaming_scene(&mesh, false);
    let view = overlook_view();

    // The background: the identical frame with the meshlet body removed, so
    // "covered" is exact rather than a brightness guess.
    let background = Rig::new(&gpu, default_settings()).settle(&gpu, &bare, &view, 3);
    let reference = Rig::new(&gpu, default_settings()).settle(&gpu, &scene, &view, 3);

    let budget = source.total_resident_bytes() / 6;
    let mut clamped_rig = Rig::new(&gpu, render_settings(budget));
    let clamped = clamped_rig.settle(&gpu, &scene, &view, 3);
    let residency = clamped_rig.residency();

    let cov_ref = coverage(&reference, &background);
    let cov_clamped = coverage(&clamped, &background);
    let n_ref = cov_ref.iter().filter(|c| **c).count();
    let n_clamped = cov_clamped.iter().filter(|c| **c).count();
    let kept = cov_ref
        .iter()
        .zip(&cov_clamped)
        .filter(|(r, c)| **r && **c)
        .count();
    let ratio = kept as f64 / n_ref as f64;

    eprintln!(
        "P18.2 no-holes: reference covers {n_ref} px, clamped ({}/{} pages, floor_lod {}, {} B) \
         covers {n_clamped} px and keeps {kept} of the reference's ({:.4})",
        residency.resident_pages(),
        residency.pages.get(&ASSET).map_or(0, |p| p.1),
        residency.floor(),
        residency.resident_bytes,
        ratio
    );

    assert!(
        n_ref > (W * H / 16) as usize,
        "the fixture must actually paint the frame ({n_ref} px of {})",
        W * H
    );
    assert!(
        residency.budget_clamped && residency.resident_pages() >= 1,
        "the clamped frame must be genuinely budget-clamped AND still hold its root floor: \
         {residency:?}"
    );
    assert_ne!(
        clamped, reference,
        "the clamped frame is byte-identical to the reference — the budget did not bite, so the \
         coverage assertion below would be vacuous"
    );
    assert!(
        ratio >= 0.99,
        "the clamped frame left holes: it covers only {kept} of the reference's {n_ref} painted \
         pixels ({ratio:.4}); a residency clamp must coarsen the surface, never open it"
    );
}

// ── (6) the cull-feedback instrument ─────────────────────────────────────────

/// **`VgeomAudit::clamped` reports requested-but-missing, and only that.**
///
/// The cull compute is the one place that knows a meshlet only won its interval
/// because the finer page is out. That signal is deliberately *not* fed back into
/// what gets loaded — a GPU readback is one frame latent, so residency would
/// become a function of frame history and the trace gate above would stop being
/// provable. It survives as an audit counter, which means nothing in the shipping
/// frame depends on it and nothing else would notice if it silently broke.
///
/// This pins it from both sides: a budget-clamped frame must report a nonzero
/// count, and a want-limited frame must report exactly zero. The zero is the
/// sharper half — it is the same statement as "streaming costs VRAM, never
/// detail", counted per meshlet by the shader rather than argued from the page
/// bounds, so a drift in `ideal_page_count` would surface here as a nonzero
/// counter on an unclamped frame.
#[test]
fn the_clamped_audit_counter_reports_requested_but_missing() {
    let Some(gpu) = gpu_or_skip("clamped_audit_counter") else {
        return;
    };
    let mesh = Arc::new(dense_grid_mesh(GRID));
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let scene = streaming_scene(&mesh, true);
    let view = overlook_view();

    let mut wanted = Rig::audited(&gpu, default_settings());
    wanted.settle(&gpu, &scene, &view, 3);
    let want_audit = wanted.audit(&gpu);
    let want_res = wanted.residency();

    let budget = source.total_resident_bytes() / 6;
    let mut clamped = Rig::audited(&gpu, render_settings(budget));
    clamped.settle(&gpu, &scene, &view, 3);
    let clamp_audit = clamped.audit(&gpu);
    let clamp_res = clamped.residency();

    eprintln!(
        "P18.2 clamp audit: want-limited ({}/{} pages, floor {}) base_cut {} clamped {}; \
         budget-clamped ({}/{} pages, floor {}) base_cut {} clamped {}",
        want_res.resident_pages(),
        want_res.pages.get(&ASSET).map_or(0, |p| p.1),
        want_res.floor(),
        want_audit.base_cut,
        want_audit.clamped,
        clamp_res.resident_pages(),
        clamp_res.pages.get(&ASSET).map_or(0, |p| p.1),
        clamp_res.floor(),
        clamp_audit.base_cut,
        clamp_audit.clamped,
    );

    assert!(
        want_audit.base_cut > 0 && clamp_audit.base_cut > 0,
        "both frames must have selected something: {want_audit:?} / {clamp_audit:?}"
    );
    assert!(
        !want_res.budget_clamped,
        "the reference frame must not be budget-clamped: {want_res:?}"
    );
    assert_eq!(
        want_audit.clamped, 0,
        "a frame the streamer fully satisfied reported {} meshlets drawn coarser than asked for \
         — streaming must cost VRAM, never detail ({want_audit:?})",
        want_audit.clamped
    );
    assert!(
        clamp_res.budget_clamped,
        "the clamped frame is not actually budget-clamped: {clamp_res:?}"
    );
    assert!(
        clamp_audit.clamped > 0,
        "a budget-clamped frame reported no requested-but-missing meshlets ({clamp_audit:?}) — \
         the cull-feedback instrument is dead"
    );
    assert!(
        clamp_audit.clamped <= clamp_audit.base_cut,
        "the clamped count must be a subset of the base cut: {clamp_audit:?}"
    );
}

// ── (7) what the sync point costs ────────────────────────────────────────────

/// Mirror of the §8 frame-budget tripwire in `tests/frame_budget.rs`, which is a
/// private `const` of that test binary and therefore not importable here.
///
/// **This is deliberately NOT a new ratchet.** §8's rule makes every budget
/// constant a standing maintenance obligation, and P17.4 (`sky_stack_cost_per_
/// tier`) and P18.1 (`vgeom_two_pass_cost`) both set the precedent of measuring a
/// new subsystem against the existing composed-frame budget rather than minting
/// another one. If `frame_budget.rs` ratchets its number down, this copy follows
/// it down; it may never be raised independently, and it may never be raised at
/// all.
const FRAME_BUDGET_MS: f64 = 33.0;

/// **The streaming sync point is not a per-frame cost.**
///
/// `VgeomStreamer::plan` runs once per frame before culling, and on a converged
/// streamer it is supposed to do nothing at all: the wants match the residency,
/// nothing is granted, nothing is evicted, no page is staged and no buffer is
/// written. What this measures is whether that "nothing" is actually free —
/// walking the page directory and re-deriving the wants every frame is cheap, but
/// it is not zero, and it sits on the critical path of every meshlet frame.
///
/// The cold frame is measured separately and printed rather than asserted: it
/// pays the full page-in (staging, pool growth, and the re-upload a pool growth
/// forces), which is a one-off whose absolute cost depends entirely on how much
/// content the first frame sees. It is here because a regression that moved work
/// from the cold frame into the steady state would otherwise look like an
/// improvement.
///
/// Only the steady state is asserted, against the existing frame budget, and only
/// on real hardware — a software rasterizer's timing is not representative of
/// anything, exactly as `frame_budget.rs` treats it.
#[test]
fn streaming_overhead_is_bounded() {
    let Some(gpu) = gpu_or_skip("streaming_overhead") else {
        return;
    };
    let info = gpu.adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || info.name.to_ascii_lowercase().contains("paravirtual");

    let mesh = Arc::new(dense_grid_mesh(GRID));
    let source = inf_vgeom::VgeomSource::from_mesh(&mesh).expect("index the vmesh");
    let scene = streaming_scene(&mesh, true);
    let view = overlook_view();
    // A budget that keeps the streamer honest (it clamps, so the plan re-evaluates
    // a real auction every frame) rather than one that trivially holds everything.
    let budget = source.total_resident_bytes() / 4;
    let mut rig = Rig::new(&gpu, render_settings(budget));

    let cold_start = std::time::Instant::now();
    rig.frame(&gpu, &scene, &view);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

    // Converge both the streamer (idle plans from here) and the two-pass early set.
    for _ in 0..10 {
        rig.frame(&gpu, &scene, &view);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    const MEASURED: u32 = 60;
    let start = std::time::Instant::now();
    for _ in 0..MEASURED {
        rig.frame(&gpu, &scene, &view);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let steady_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(MEASURED);

    let residency = rig.residency();
    eprintln!(
        "P18.2 streaming overhead on {} ({:?}), {W}x{H}, budget {budget} B: cold (full page-in) \
         frame {cold_ms:.3} ms, steady-state mean {steady_ms:.3} ms over {MEASURED} frames, \
         streamer holding {} B resident across {}/{} pages{}",
        info.name,
        info.device_type,
        residency.resident_bytes,
        residency.resident_pages(),
        residency.pages.get(&ASSET).map_or(0, |p| p.1),
        if software {
            " [software — smoke only]"
        } else {
            ""
        }
    );

    assert!(
        residency.resident_pages() >= 1,
        "the measured frames drew nothing: {residency:?}"
    );
    if software {
        return;
    }
    assert!(
        steady_ms < FRAME_BUDGET_MS,
        "a converged streaming frame cost {steady_ms:.3} ms, over the {FRAME_BUDGET_MS} ms \
         frame budget on {} (§8: investigate the regression, never raise the budget)",
        info.name
    );
}
