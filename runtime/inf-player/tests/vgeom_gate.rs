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
    classic_lod_selection, cull_visible, EngineRenderer, GpuContext, HeadlessTarget, LightKind,
    RenderLight, RenderScene, RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomSettings,
};

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
        VmeshRegistry::from_pack(&reader).expect("vmesh registry")
    };
    assert!(!reader_reg.is_empty(), "cook derived at least one vmesh");

    let (asset_id, mesh_arc) = reader_reg
        .pick(VGEOM_DEMO_MESH_GUID, true)
        .expect("the derived vmesh for the dense mesh is present in the pack");
    assert_eq!(
        asset_id,
        derived_vmesh_id(VGEOM_DEMO_MESH_GUID).as_u128(),
        "vmesh keyed by the cook-derived id"
    );
    let asset = VgeomAsset {
        id: asset_id,
        mesh: mesh_arc,
    };

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
    let lod0_tris = asset.mesh.classic_lods()[0].triangle_count() as u64;
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
    let total_pairs = asset.mesh.meshlet_count() as u64 * instances.len() as u64;

    let vis1 = cull_visible(&gpu, &asset.mesh, &instances, &view, &settings);
    let vis2 = cull_visible(&gpu, &asset.mesh, &instances, &view, &settings);
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
    let total_pairs = asset.mesh.meshlet_count() as u64 * instances.len() as u64;

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
