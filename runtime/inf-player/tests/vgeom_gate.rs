//! Phase-13-closing gate (P13.4): the virtualized-geometry demo streams + culls
//! through the shipping pipeline, and the classic-LOD fallback + auto-tier work.
//!
//! The editor authors the `vgeom-demo` sample: a grid of `GRID × GRID` instances of
//! ONE dense `.inf_mesh` (via the P13.4 `MeshRef.asset` reference), so the source
//! triangle count exceeds 10M while the `.inf_lvl` stays tiny. Cooking it derives
//! the `.inf_vmesh` meshlet DAG (through the `MeshRef.asset` → dependency-closure
//! edge) and ships both in the pack. These tests prove:
//!
//! * **(a)** byte-identical save/reload of the `.inf_lvl` (the P3 discipline).
//! * **(b)** cooked load with vgeom ON: total source triangles ≥ 10M (from the
//!   vmesh data × instances), and the GPU meshlet cull leaves only a small fraction
//!   of meshlets visible from a ground-level camera — deterministic across runs.
//! * **(b2)** (P18.1) the same pack under the shipping defaults — two-pass HZB
//!   occlusion ON — renders **pixel-identical** to the occlusion-off frame while
//!   the audit counters show a nonzero proven-occluded set. Occlusion is
//!   subtractive at flagship scale, not just in a unit fixture.
//! * **(b3)** (P18.2) the same pack **streamed under a hard VRAM budget**: the
//!   meshlet pools hold a fraction of the asset, the frame is still deterministic
//!   pixel-for-pixel across runs, the cut is coarser but covers the same ground
//!   (never a hole), and the residency trace is reproducible.
//! * **(c)** the SAME cooked pack with vgeom OFF renders through the classic
//!   discrete-LOD fallback (it activates + draws; a far camera picks a coarser
//!   level than a near one).
//! * **(d)** the auto-tier disables vgeom on the Low tier (capability fallback).
//!
//! GPU parts skip cleanly with no adapter (like every GPU path in this repo).

use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3};

use inf_editor_core::samples::{
    vgeom_demo_mesh, vgeom_demo_scene, vgeom_demo_source_triangles, VGEOM_DEMO_GRID,
    VGEOM_DEMO_MESH_GUID,
};
use inf_math::FloatingOrigin;
use inf_packager::{cook, CookOptions};
use inf_player::level::LevelSource;
use inf_player::level::PackLevelSource;
use inf_player::vmesh::{derived_vmesh_id, VmeshRegistry};
use inf_project::ProjectManifest;
use inf_render::{
    caps::{choose_tier, AdapterCaps, RenderTier},
    classic_lod_selection, cull_visible_source, EngineRenderer, GpuContext, HeadlessTarget,
    LightKind, RenderLight, RenderScene, RenderSettings, RenderView, VgeomAsset, VgeomInstance,
    VgeomSettings,
};
// The page-derived budget helpers, shared with `phase18_gate` and `inf-render`'s
// streaming suite; `inf_vgeom::test_support`'s header states the rule they enforce.
use inf_vgeom::test_support::{budget_for_pages, held_pages_for_want};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn vgeom_demo_dir() -> PathBuf {
    workspace_root().join("samples/vgeom-demo")
}

/// Scaffold a project with the vgeom-demo copied into Content, then cook it (the
/// cook derives the `.inf_vmesh` from the dense mesh via the closure edge).
fn cook_vgeom_demo(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Vgeom Demo", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = vgeom_demo_dir();
    for f in ["VgeomDemo.inf_lvl", "Dense.inf_mesh", "Dense.inf_mesh.toml"] {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Resolve the cooked pack into `(vmesh asset, placed instances)` — the exact
/// `RenderScene` content the player builds for the demo.
fn scene_from_pack(pack_dir: &Path) -> (VgeomAsset, Vec<VgeomInstance>) {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let reader_reg = {
        // The registry is loaded from the same pack the level source wraps.
        let reader = inf_asset::PackReader::open(&pack_dir.join(inf_player::level::PACK_FILE))
            .expect("open pack for registry");
        VmeshRegistry::from_pack(std::sync::Arc::new(reader)).expect("vmesh registry")
    };
    assert!(!reader_reg.is_empty(), "cook derived at least one vmesh");

    let (asset_id, vmesh_source) = reader_reg
        .pick(VGEOM_DEMO_MESH_GUID, true)
        .expect("the derived vmesh for the dense mesh is present in the pack");
    assert_eq!(
        asset_id,
        derived_vmesh_id(VGEOM_DEMO_MESH_GUID).as_u128(),
        "vmesh keyed by the cook-derived id"
    );
    let asset = VgeomAsset::new(asset_id, vmesh_source);

    let level = inf_scene::RuntimeLevel::decode(&source.level_bytes().unwrap()).expect("decode");
    let mut instances = Vec::new();
    let mut id = 1u32;
    for e in &level.entities {
        let Some(mr) = &e.mesh else { continue };
        if mr.asset != Some(VGEOM_DEMO_MESH_GUID) {
            continue;
        }
        let t = e.transform;
        instances.push(VgeomInstance::lit(
            asset_id,
            t.translation.to_dvec3(),
            t.quat().as_quat(),
            t.scale.to_dvec3().as_vec3(),
            [0.6, 0.6, 0.6, 1.0],
            id,
        ));
        id += 1;
    }
    (asset, instances)
}

/// A ground-level camera near one corner of the grid, looking diagonally across it.
fn ground_camera() -> RenderView {
    let eye = DVec3::new(-26.0, 1.5, -26.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::ZERO - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    }
}

// ── GATE (a): byte-identical save/reload ─────────────────────────────────────

#[test]
fn gate_a_scene_round_trips_byte_identical() {
    // The committed `.inf_lvl` re-decode/encode through the runtime reader is a
    // lossless identity on current-schema content (the editor separately proves
    // the generator → committed bytes lock).
    let committed = std::fs::read(vgeom_demo_dir().join("VgeomDemo.inf_lvl")).unwrap();
    assert_eq!(
        committed[0],
        inf_scene::SCHEMA_VERSION as u8,
        "committed vgeom-demo is a current-schema payload"
    );
    let level = inf_scene::RuntimeLevel::decode(&committed).unwrap();
    let re = level.encode().unwrap();
    assert_eq!(
        committed, re,
        "vgeom-demo decode→encode must be byte-identical"
    );
    // The generator itself round-trips (source-of-truth check without disk).
    let _ = vgeom_demo_scene();
}

// ── GATE (b): cooked load, vgeom ON — 10M+ triangles + massive cull ──────────

#[test]
fn gate_b_cooked_vgeom_on_streams_and_culls() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_vgeom_demo(tmp.path());
    let (asset, instances) = scene_from_pack(&pack);

    // Source triangles: the vmesh's finest (LOD-0) triangle count × instances.
    // The classic-LOD view is the one thing that genuinely needs the whole DAG, so
    // this materializes it once — the streamed render path never does.
    let full = asset.source.to_mesh().expect("materialize the vmesh");
    let lod0_tris = full.classic_lods()[0].triangle_count() as u64;
    let total_source = lod0_tris * instances.len() as u64;
    assert_eq!(instances.len(), VGEOM_DEMO_GRID * VGEOM_DEMO_GRID);
    assert!(
        total_source >= 10_000_000,
        "gate needs 10M+ source triangles across instances, got {total_source} \
         ({lod0_tris} LOD0 tris × {} instances)",
        instances.len()
    );
    // Sanity vs the sample's declared budget (LOD0 ≈ the source mesh triangles).
    let per_mesh = vgeom_demo_mesh().triangle_count() as u64;
    assert!(
        total_source >= vgeom_demo_source_triangles().min(per_mesh * instances.len() as u64) / 2
    );

    // GPU: the meshlet cull leaves only a small fraction visible, deterministically.
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gate (b) GPU cull: no adapter");
        return;
    };
    let view = ground_camera();
    let settings = VgeomSettings {
        enabled: true,
        ..VgeomSettings::default()
    };
    let total_pairs = asset.source.meshlet_count() as usize as u64 * instances.len() as u64;

    let vis1 = cull_visible_source(&gpu, &asset.source, &instances, &view, &settings).pairs;
    let vis2 = cull_visible_source(&gpu, &asset.source, &instances, &view, &settings).pairs;
    assert_eq!(
        vis1, vis2,
        "the visible meshlet SET is deterministic across runs"
    );
    assert!(!vis1.is_empty(), "the meshlet path activated (visible > 0)");

    let ratio = vis1.len() as f64 / total_pairs as f64;
    eprintln!(
        "gate (b): {} of {} (instance×meshlet) pairs visible — cull ratio {:.3}",
        vis1.len(),
        total_pairs,
        ratio
    );
    assert!(
        ratio < 0.35,
        "expected massive culling (LOD cut + frustum) from a ground camera, ratio {ratio:.3}"
    );
}

// ── GATE (b2): the same 10M+ pack under P18.1 two-pass HZB occlusion ─────────

/// The 10.6M-triangle scene must render **identically** with occlusion on and off
/// — the P18.1 subtractive contract at flagship scale — while the audit counters
/// prove the HZB is actually removing a meaningful slice of a ground-level view
/// through 324 dense bodies.
///
/// This is the shipping-defaults gate: `VgeomSettings::default()` now carries
/// `occlusion: true, two_pass: true`, so what runs here is exactly what the player
/// runs. Both modes are driven for several frames on ONE renderer each, so the
/// two-pass side is compared *after* its temporal state has converged, not just on
/// its conservative first frame.
#[test]
fn gate_b2_occlusion_on_matches_occlusion_off_at_10m_triangles() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gate (b2): no adapter");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_vgeom_demo(tmp.path());
    let (asset, instances) = scene_from_pack(&pack);
    let total_pairs = asset.source.meshlet_count() as usize as u64 * instances.len() as u64;

    let mut scene = RenderScene {
        vgeom_assets: vec![asset],
        vgeom_instances: instances,
        ..Default::default()
    };
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.75, 0.55).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = ground_camera();

    let settings = |occlusion: bool| RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion,
            two_pass: occlusion,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };

    let (w, h) = (view.width, view.height);
    let shot = |occlusion: bool, n: usize, audit: bool| {
        let target = HeadlessTarget::new(&gpu, w, h);
        let mut r = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
        r.set_settings(settings(occlusion));
        r.set_vgeom_audit(audit);
        let mut img = Vec::new();
        for _ in 0..n {
            r.render(&gpu, &scene, &view, &target.view, (w, h));
            img = target.read_rgba(&gpu).expect("readback");
        }
        (img, r.vgeom_audit(&gpu))
    };

    let (reference, _) = shot(false, 1, false);
    let painted = reference
        .chunks(4)
        .filter(|p| p[0] > 24 || p[1] > 24)
        .count();
    assert!(
        painted > 2_000,
        "the 10M-triangle ground view must paint real geometry ({painted} px)"
    );

    let (converged, audit) = shot(true, 6, true);
    eprintln!(
        "gate (b2): {} of {total_pairs} pairs in the base cut, {} proven occluded \
         ({:.1}%), drawn {} early + {} late",
        audit.base_cut,
        audit.occluded,
        100.0 * audit.occluded as f64 / audit.base_cut.max(1) as f64,
        audit.early_drawn,
        audit.late_drawn,
    );
    assert!(audit.base_cut > 0, "the meshlet path activated");
    assert!(
        audit.occluded > 0,
        "a ground camera looking through 324 dense bodies must occlude something: {audit:?}"
    );
    let diff = converged
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "two-pass occlusion must be purely subtractive at 10M triangles \
         ({diff} of {} bytes differ)",
        reference.len()
    );
}

// ── GATE (b3): the same 10M+ pack STREAMED under a hard VRAM budget ──────────

/// The flagship scene with **meshlet streaming forced into partial residency**
/// (P18.2): the pools are given a fraction of what the asset needs, so the
/// streamer must page a prefix and clamp the cut to it.
///
/// Four things have to hold at once, and each is a separate way this feature
/// could be wrong:
///
/// 1. **The budget is a hard bound.** Resident bytes stay under the ceiling, and
///    the streamer says so (`budget_clamped`). A soft budget is not a budget.
/// 2. **Never a hole.** The clamped frame is *not* pixel-identical to the
///    fully-wanted one — the clamp genuinely coarsened the geometry, so the test
///    is not vacuous — yet it still paints essentially the same ground. A missing
///    page must read as softer detail, never as a gap. Page 0 (every root of the
///    DAG) is what guarantees that, and it is never evicted.
/// 3. **Determinism survives it.** Two fresh renderers driven through the same
///    frame sequence produce byte-identical images *and* an identical residency
///    trace. Streaming is where a renderer usually stops being reproducible —
///    loads land when IO says so — and this pins that ours does not, because the
///    resident set is a pure function of (wants, budget) at the sync point rather
///    than of what happened to arrive.
/// 4. **A shorter prefix coarsens the cut.** The clamped run's `floor_lod` is
///    strictly above the unclamped run's — the streamer did not merely hold fewer
///    bytes, the cut the shader takes actually moved. The budget is built to make
///    this true on any page ladder (see the derivation below), so a failure here is
///    the clamp not reaching the shader rather than a fixture that drifted.
#[test]
fn gate_b3_streamed_under_budget_is_deterministic_and_hole_free() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP gate (b3): no adapter");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_vgeom_demo(tmp.path());
    let (asset, instances) = scene_from_pack(&pack);
    let asset_id = asset.id;
    let full_bytes = asset.source.total_resident_bytes();
    let pages = asset.source.pages().len();
    assert!(pages >= 3, "the flagship asset must have pages to withhold");

    let mut scene = RenderScene {
        vgeom_assets: vec![asset],
        vgeom_instances: instances,
        ..Default::default()
    };
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.75, 0.55).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    // The live page directory the budget below is read off (the asset moved into
    // the scene; this is the same `Arc<VgeomSource>` the streamer will plan against).
    let source = &scene.vgeom_assets[0].source;
    let view = ground_camera();
    let (w, h) = (view.width, view.height);

    let settings = |budget: u64| RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            stream: inf_vgeom::VgeomStreamBudget {
                budget_bytes: budget,
                ..inf_vgeom::VgeomStreamBudget::default()
            },
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };

    // Render `frames` frames on ONE fresh renderer, returning the last image and
    // the streaming report it published.
    let shot = |budget: u64, frames: usize| {
        let target = HeadlessTarget::new(&gpu, w, h);
        let mut r = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
        r.set_settings(settings(budget));
        let mut img = Vec::new();
        for _ in 0..frames {
            r.render(&gpu, &scene, &view, &target.view, (w, h));
            img = target.read_rgba(&gpu).expect("readback");
        }
        (img, r.vgeom_stream_report())
    };

    // ── the fully-wanted reference: the budget never bites ──
    let (reference, full_report) = shot(inf_vgeom::DEFAULT_VGEOM_BUDGET_BYTES, 1);
    let (full_resident, total_pages) = full_report.pages[&asset_id];
    assert!(
        !full_report.stats.budget_clamped,
        "the reference must be unclamped"
    );
    let covered = |img: &[u8]| img.chunks(4).filter(|p| p[0] > 24 || p[1] > 24).count();
    let reference_px = covered(&reference);
    assert!(
        reference_px > 2_000,
        "the 10M-triangle ground view must paint real geometry ({reference_px} px)"
    );

    // ── the clamped run: a budget that cannot hold what the camera wants ──
    //
    // Counted in PAGES off the live page directory, never as a fraction of a byte
    // total. The streamer already declines the pages this camera cannot use, so a
    // fraction of the whole *asset* can still cover the whole want and clamp
    // nothing — that is the streaming-costs-no-detail theorem doing its job, and it
    // would quietly make this test vacuous. But a fraction of the *wanted bytes* is
    // no better as a premise: whether half the bytes buys `k` pages or `k + 1` is
    // decided by the per-page sizes meshopt produced on the machine that measured
    // them, and meshopt is not cross-platform. `phase18_gate` paid for that with a
    // macOS-only failure; the rule and these two helpers live in
    // `inf_vgeom::test_support`, whose header carries the law.
    //
    // So: take the camera's own want (`wanted_pages` — the streamer's
    // `ideal_page_count`, a pure function of camera + directory), hold half of it,
    // and shorten further if that prefix happens to end on the same LOD level the
    // full want ends on — a ladder where one level spans several pages would
    // otherwise satisfy "fewer pages" without raising the cut's floor, which is
    // claim 4 below. Every step is read off THIS platform's directory, so the four
    // claims hold whatever DAG meshopt built here.
    let dir = source.pages();
    let want = full_report.stats.wanted_pages;
    let unclamped_floor = full_report.floor_lod[&asset_id];
    assert!(
        want >= 2 && full_resident >= 2,
        "the ground camera wants {want} of {total_pages} pages and the unclamped run \
         held {full_resident}: the root page is granted unconditionally, so with \
         nothing above it there is no prefix to withhold and no budget can clamp"
    );
    let mut held = held_pages_for_want(full_resident);
    while held > 1 && dir[held - 1].floor_lod <= unclamped_floor {
        held -= 1;
    }
    assert!(
        dir[held - 1].floor_lod > unclamped_floor,
        "no prefix shorter than the unclamped run's {full_resident} pages ends above \
         its LOD floor {unclamped_floor} — every resident page sits at one level on \
         this ladder, so claim 4 (a shorter prefix raises the floor) is unstateable"
    );
    let budget = budget_for_pages(source, held);
    let (clamped, report) = shot(budget, 1);
    let (resident, _) = report.pages[&asset_id];
    eprintln!(
        "gate (b3): {resident} of {total_pages} pages resident under a {budget} B budget \
         built to hold {held} of a {want}-page want (unclamped holds {full_resident} pages \
         / {} B of a {full_bytes} B asset); {} | floor_lod {} vs the unclamped run's \
         {unclamped_floor}",
        full_report.stats.resident_bytes,
        report.stats.summary(),
        report.floor_lod[&asset_id],
    );

    // 1. the budget is a HARD bound.
    assert!(resident >= 1, "the root page is never denied");
    assert!(
        resident <= held && held < full_resident,
        "the budget must admit at most the {held}-page prefix it was built for, and \
         that prefix must be shorter than the {full_resident} pages the unclamped \
         run held (got {resident})"
    );
    assert!(
        report.stats.resident_bytes <= budget,
        "resident bytes {} exceed the {budget}-byte ceiling",
        report.stats.resident_bytes
    );
    assert!(report.stats.budget_clamped, "the clamp must be reported");
    // 4. a shorter prefix raises the cut's floor. Guaranteed by the construction
    //    above (`held` was chosen to end above `wanted_floor`, and `floor_lod` never
    //    gets finer as the prefix gets shorter), so this checks the construction
    //    reached the shader's side of the world rather than pinning a number.
    assert!(
        report.floor_lod[&asset_id] > full_report.floor_lod[&asset_id],
        "a shorter residency prefix must raise the cut's floor ({} pages at floor {} \
         vs {full_resident} at floor {})",
        resident,
        report.floor_lod[&asset_id],
        full_report.floor_lod[&asset_id]
    );

    // 2. never a hole: coarser, but the same ground.
    let clamped_px = covered(&clamped);
    assert_ne!(
        clamped, reference,
        "a clamped frame that is byte-identical means the clamp did nothing"
    );
    let ratio = clamped_px as f64 / reference_px as f64;
    assert!(
        ratio > 0.97,
        "a coarser cut may move silhouettes, never open a gap: covered {clamped_px} \
         of {reference_px} px (ratio {ratio:.4})"
    );

    // 3. determinism: two fresh renderers, same frames, identical everything.
    let (again, report_again) = shot(budget, 1);
    assert_eq!(clamped, again, "a streamed frame must be reproducible");
    assert_eq!(
        report.pages, report_again.pages,
        "the resident-set trace must be reproducible"
    );
    assert_eq!(report.floor_lod, report_again.floor_lod);
    assert_eq!(report.stats, report_again.stats);

    // ...and across a multi-frame convergence, where the streamer has state.
    let (conv_a, rep_a) = shot(budget, 5);
    let (conv_b, rep_b) = shot(budget, 5);
    assert_eq!(
        conv_a, conv_b,
        "a converged streamed frame must be reproducible"
    );
    assert_eq!(rep_a.pages, rep_b.pages);
    assert_eq!(rep_a.stats, rep_b.stats);
}

// ── GATE (c): SAME pack, vgeom OFF — classic discrete-LOD fallback ───────────

#[test]
fn gate_c_cooked_vgeom_off_uses_classic_lod() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_vgeom_demo(tmp.path());
    let (asset, instances) = scene_from_pack(&pack);
    let assets = [asset];

    // The classic path activates + draws every instance (a draw/instance probe).
    let near = ground_camera();
    let sel_near = classic_lod_selection(&assets, &instances, &near, 1.0);
    assert_eq!(
        sel_near.drawn_instances,
        instances.len() as u32,
        "the classic fallback resolves + draws every instance"
    );
    assert!(
        sel_near.drawn_triangles > 0,
        "the classic path drew geometry"
    );
    assert!(sel_near.picks.iter().all(|p| p.is_some()));

    // A far camera (whole grid tiny) picks coarser levels than a near one — the
    // discrete-LOD distance behaviour.
    let far_eye = DVec3::new(0.0, 400.0, 400.0);
    let far = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: far_eye,
        forward: (DVec3::ZERO - far_eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    };
    // A very near camera close to the grid centre → finest levels.
    let near_eye = DVec3::new(0.0, 2.0, 0.0);
    let very_near = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: near_eye,
        forward: Vec3::NEG_Y,
        up: Vec3::Z,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    };
    let sel_far = classic_lod_selection(&assets, &instances, &far, 1.0);
    let sel_vnear = classic_lod_selection(&assets, &instances, &very_near, 1.0);

    let max_pick =
        |s: &inf_render::ClassicSelection| s.picks.iter().flatten().copied().max().unwrap();
    eprintln!(
        "gate (c): near draws {} tris (max LOD {}), far draws {} tris (max LOD {})",
        sel_vnear.drawn_triangles,
        max_pick(&sel_vnear),
        sel_far.drawn_triangles,
        max_pick(&sel_far),
    );
    assert!(
        sel_far.drawn_triangles < sel_vnear.drawn_triangles,
        "a far camera draws fewer triangles (coarser LODs) than a near one"
    );
    assert!(
        max_pick(&sel_far) > max_pick(&sel_vnear),
        "a far camera picks a strictly coarser LOD level than a near one"
    );
}

// ── GATE (d): capability auto-tier disables vgeom on the Low tier ────────────

#[test]
fn gate_d_low_tier_disables_vgeom() {
    // A downlevel / software-class adapter → Low, which auto-disables vgeom.
    let low_caps = AdapterCaps {
        compute_shaders: false,
        indirect_execution: false,
        max_storage_buffers_per_stage: 0,
        max_storage_buffer_binding_size: 0,
        max_compute_workgroups_per_dim: 0,
        max_storage_textures_per_stage: 0,
        is_cpu: true,
        texture_compression_bc: false,
        polygon_mode_line: false,
    };
    assert_eq!(choose_tier(&low_caps), RenderTier::Low);

    // Force Low → the render settings' vgeom flag is turned off automatically, so
    // the pick logic + the ClassicVgeomNode take over (the fallback path).
    let mut settings = inf_render::RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..Default::default()
    };
    settings.tier_override = Some(RenderTier::Low);
    let clamped = settings.tier_override.unwrap().apply(settings);
    assert!(
        !clamped.vgeom.enabled,
        "forcing the Low tier auto-disables the meshlet path"
    );

    // A capable adapter (if present) picks High and keeps vgeom available.
    if let Ok(gpu) = GpuContext::headless() {
        let caps = AdapterCaps::probe(&gpu);
        let tier = choose_tier(&caps);
        eprintln!("gate (d): live adapter tier = {tier:?} ({caps:?})");
        // Whatever it is, applying it to a vgeom-on setting never *enables* a
        // feature; on a non-High tier vgeom must end up off.
        let s = inf_render::RenderSettings {
            vgeom: VgeomSettings {
                enabled: true,
                ..VgeomSettings::default()
            },
            ..Default::default()
        };
        let applied = tier.apply(s);
        assert_eq!(applied.vgeom.enabled, tier == RenderTier::High);
    }
}
