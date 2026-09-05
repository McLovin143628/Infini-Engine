//! **PIE == shipping, over a scene that runs virtual shadow maps** (P27.4).
//!
//! Phase 27's own "Done when" asks for *"the page-allocation trace is
//! deterministic and PIE == shipping"*. P27.1 through P27.3 could not say it
//! about anything a player renders, because virtual shadows were inert until a
//! receiver existed; this is the first batch where they reach pixels, so it is
//! the first batch where the claim has a subject.
//!
//! The shape is `phase26_gate`'s `pie_equals_shipping_on_the_residency_trace`,
//! moved from texture tiles to shadow pages: one project, cooked; two worlds
//! built from it — one off the **cooked pack**, one off the **editor's
//! `ScenePayload`** — and both projected through
//! `inf_player::render::project_scene`, which is the ONE door a shipped frame's
//! scene comes through. What is compared is
//!
//! 1. the **resident page trace**, one line a frame, named by (light handle,
//!    face, level, x, y) rather than by slot, because a slot is an allocation
//!    order and a page is a place; and
//! 2. the **frame bytes**, because a receiver that resolved the same pages and
//!    read them differently would pass (1) and fail here.
//!
//! Anti-vacuity comes first in both directions: the trace has to be non-empty
//! and it has to MOVE along the walk, and the shadowed frame has to differ from
//! the same frame with virtual shadows off.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{Light, LightKind, Transform};
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use inf_render::{GpuContext, RenderScene, RenderSettings, RenderView, VsmSettings};
use uuid::Uuid;

const LEVEL: Uuid = Uuid::from_u128(0x2704_0000_0000_0001);
const W: u32 = 256;
const H: u32 = 144;
/// Frames each side renders. The marking ring is pinned at `frame − 2`, so the
/// residency cannot hold anything before frame 2; eight leaves the trace a
/// stretch of settled frames to differ over.
const FRAMES: u64 = 8;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP {what}: no GPU adapter ({e})");
            None
        }
    }
}

fn put(content: &Path, file: &str, guid: Uuid, bytes: &[u8], kind: AssetKind) {
    let path = content.join(file);
    std::fs::write(&path, bytes).expect("write asset");
    AssetSidecar::new(AssetId(guid), kind, ContentHash::of(bytes))
        .save(&path)
        .expect("write sidecar");
}

fn cook_opts() -> CookOptions {
    CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// A project whose level is a floor slab, three boxes standing on it and one
/// **shadow-casting** directional light — the smallest thing that makes a page
/// atlas do work.
fn shadow_project(tmp: &Path) -> (PathBuf, SceneDoc) {
    let proj = tmp.join("shadowed");
    ProjectManifest::new("Phase 27 Receiver", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mut doc = SceneDoc::new();
    let place =
        |doc: &mut SceneDoc, name: &str, t: inf_ecs::math::Vec3d, s: inf_ecs::math::Vec3d| {
            let id = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, name, None);
            let world = doc.world_mut();
            let e = world.entity_of(id).expect("the cube exists");
            world.world_mut().entity_mut(e).insert(Transform {
                translation: t,
                scale: s,
                ..Default::default()
            });
        };
    place(
        &mut doc,
        "Floor",
        inf_ecs::math::Vec3d::new(0.0, -0.5, 0.0),
        inf_ecs::math::Vec3d::new(60.0, 1.0, 60.0),
    );
    for (i, (x, z)) in [(-2.2f64, 0.6f64), (1.2, -1.4), (2.8, 1.4)]
        .into_iter()
        .enumerate()
    {
        place(
            &mut doc,
            &format!("Box{i}"),
            inf_ecs::math::Vec3d::new(x, 0.75, z),
            inf_ecs::math::Vec3d::new(1.4, 1.5, 1.4),
        );
    }
    // The sun. `cast_shadows` defaults to true (schema v8), and it is written
    // out explicitly here because it is the field this whole file is about.
    let sun = doc.edit_create(
        inf_editor_core::ipc::SpawnKind::DirectionalLight,
        "Sun",
        None,
    );
    {
        let world = doc.world_mut();
        let e = world.entity_of(sun).expect("the light exists");
        world.world_mut().entity_mut(e).insert(Light {
            kind: LightKind::Directional,
            intensity: 3.0,
            cast_shadows: true,
            ..Default::default()
        });
        world.world_mut().entity_mut(e).insert(Transform {
            // The projector derives a directional light's direction from its
            // rotation; a pitch puts the sun above the horizon.
            rotation: inf_ecs::math::Vec3d::new(-55.0, 24.0, 0.0),
            ..Default::default()
        });
    }

    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(
        &content,
        "Shadowed.inf_lvl",
        LEVEL,
        &level,
        AssetKind::Level,
    );
    (proj, doc)
}

/// The PIE payload for `doc`, served from the project on disk — `phase26_gate`'s
/// `payload_for`, kept identical so the two suites build the wire the same way.
fn payload_for(proj: &Path, doc: &SceneDoc) -> inf_runtime::pie::ScenePayload {
    let content = proj.join("Content");
    let mut by_guid: HashMap<Uuid, PathBuf> = HashMap::new();
    for e in std::fs::read_dir(&content).expect("content dir") {
        let p = e.expect("dir entry").path();
        if let Ok(side) = AssetSidecar::load(&p) {
            by_guid.insert(side.guid.uuid(), p);
        }
    }
    let read = move |g: Uuid| by_guid.get(&g).and_then(|p| std::fs::read(p).ok());
    inf_editor_core::pie::build_scene_payload(
        doc,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        read,
        |_| None,
        0,
        false,
    )
    .expect("the payload builds")
}

fn view(step: u64) -> RenderView {
    // A slow approach, so the clipmap's levels shift and the trace is a WALK
    // rather than a still.
    let eye = glam::DVec3::new(0.0, 5.0, 10.0 - 0.55 * step as f64);
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: eye,
        forward: (glam::DVec3::new(0.0, 0.4, 0.0) - eye)
            .as_vec3()
            .normalize(),
        up: glam::Vec3::Y,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn vsm_on() -> VsmSettings {
    VsmSettings {
        enabled: true,
        budget_bytes: 256 * 64 * 1024,
        clipmap_pages_per_side: 32,
        clipmap_levels: 8,
        first_level_extent_m: 12.0,
        ..Default::default()
    }
}

/// What one scripted run produced.
struct Run {
    /// One line a frame: every resident page, by (light, face, level, x, y).
    trace: Vec<String>,
    /// The last frame's pixels.
    frame: Vec<u8>,
    /// Frames the lit passes were handed the atlas.
    receiver_frames: u64,
}

/// Render the scripted walk over `sim`, through the **player's own projector**.
fn scripted(
    gpu: &GpuContext,
    sim: &inf_player::runtime_sim::RuntimeSim,
    vsm: Option<VsmSettings>,
) -> Run {
    let target = inf_render::HeadlessTarget::new(gpu, W, H);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    if let Some(v) = vsm {
        settings.vsm = v;
    }
    renderer
        .try_set_settings(settings)
        .expect("the arm's settings are legal");
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();

    let mut trace = Vec::with_capacity(FRAMES as usize);
    for step in 0..FRAMES {
        let mut scene = RenderScene::default();
        // **The one door.** Both worlds are projected by the function the
        // shipped render host calls, so a difference in the trace below is a
        // difference in the WORLD rather than in two projectors that were
        // written twice.
        inf_player::render::project_scene(&mut scene, sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        let v = view(step);
        renderer.render(gpu, &scene, &v, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

        let mut line = String::new();
        if let Some(sys) = renderer.vsm() {
            let res = sys.residency();
            let geom = res.geometry();
            let mut pages: Vec<(u32, u32, u32, u32, u32)> = (0..geom.slot_count())
                .filter_map(|s| res.slot_occupant(s))
                .map(|(l, p)| (l.0, p.face, p.level, p.x, p.y))
                .collect();
            // Sorted by ADDRESS, never by slot: a slot index is an allocation
            // order and two hosts that seated the same pages in a different
            // order describe the same shadow.
            pages.sort_unstable();
            for (l, f, lv, x, y) in pages {
                line.push_str(&format!("{l}/{f}/{lv}/{x}/{y};"));
            }
        }
        trace.push(line);
    }
    Run {
        trace,
        frame: target.read_rgba(gpu).expect("readback"),
        receiver_frames: renderer.vsm_receiver_frames(),
    }
}

/// **PIE == shipping on the shadow-page trace and on the pixels.**
#[test]
fn pie_equals_shipping_on_a_virtual_shadow_scene() {
    let Some(gpu) = gpu_or_skip("the P27.4 PIE-vs-shipping receiver") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = shadow_project(tmp.path());
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    // The shipping side: the cooked pack.
    let source =
        inf_player::level::PackLevelSource::open(&report.pack_path).expect("open the cooked pack");
    let built = inf_player::build_world_from_pack(&source).expect("build the world");
    let shipped_sim = inf_player::sim_from_built(built);
    // The preview side: the editor's payload.
    let previewed_sim = inf_player::sim_from_payload(&payload_for(&proj, &doc))
        .expect("the payload builds a sim")
        .sim;

    // ASSERT THE WORLD BEFORE COMPARING TWO OF THEM (the P21.4 law): both sides
    // carry a shadow-casting directional light, or the equalities below are
    // about two scenes with no shadows in them.
    for (name, sim) in [("shipped", &shipped_sim), ("previewed", &previewed_sim)] {
        let mut scene = RenderScene::default();
        inf_player::render::project_scene(
            &mut scene,
            sim,
            0.0,
            &inf_player::vmesh::VmeshRegistry::new(),
        );
        assert!(
            scene
                .lights
                .iter()
                .any(|l| l.cast_shadows && l.kind == inf_render::LightKind::Directional),
            "the {name} world projects no shadow-casting directional light"
        );
        assert!(
            scene.instances.len() >= 4,
            "the {name} world projects {} instances, not the floor and its three \
             casters",
            scene.instances.len()
        );
    }

    let shipped = scripted(&gpu, &shipped_sim, Some(vsm_on()));
    let previewed = scripted(&gpu, &previewed_sim, Some(vsm_on()));

    // ANTI-VACUITY FIRST. A trace of empty strings agrees with itself perfectly.
    assert_eq!(shipped.trace.len(), FRAMES as usize);
    assert_eq!(shipped.receiver_frames, FRAMES);
    assert_eq!(previewed.receiver_frames, FRAMES);
    assert!(
        shipped.trace.iter().filter(|l| !l.is_empty()).count() >= (FRAMES - 2) as usize,
        "the shipped run held no resident page on most frames: {:?}",
        shipped.trace
    );
    assert!(
        shipped.trace.windows(2).any(|w| w[0] != w[1]),
        "residency never changed across {FRAMES} frames of a scripted approach, \
         so the equality below is about a still frame"
    );

    assert_eq!(
        shipped.trace, previewed.trace,
        "a cooked pack and a PIE payload made different shadow pages resident \
         for one level"
    );
    assert_eq!(
        shipped.frame, previewed.frame,
        "the two boot paths resolved the same pages and shaded them differently"
    );

    // …and the frame really is shadowed, or "the two agree" is a statement about
    // two identical un-shadowed pictures.
    let unshadowed = scripted(&gpu, &shipped_sim, None);
    assert_eq!(unshadowed.receiver_frames, 0);
    assert_ne!(
        shipped.frame, unshadowed.frame,
        "virtual shadows changed no pixel of the shipped frame"
    );
}

/// **The page trace is deterministic** — twice, on the shipping path, bit for
/// bit. The other half of Phase 27's "Done when" clause, and the one the P27.5
/// gate restates over a longer scripted path.
#[test]
fn the_shadow_page_trace_is_bit_exact_twice() {
    let Some(gpu) = gpu_or_skip("the P27.4 page-trace determinism arm") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, _doc) = shadow_project(tmp.path());
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");

    let once = scripted(
        &gpu,
        &inf_player::sim_from_built(
            inf_player::build_world_from_pack(&source).expect("build the world"),
        ),
        Some(vsm_on()),
    );
    let twice = scripted(
        &gpu,
        &inf_player::sim_from_built(
            inf_player::build_world_from_pack(&source).expect("build the world"),
        ),
        Some(vsm_on()),
    );
    assert!(once.trace.iter().any(|l| !l.is_empty()));
    assert!(once.trace.windows(2).any(|w| w[0] != w[1]));
    assert_eq!(
        once.trace, twice.trace,
        "the page trace is not deterministic"
    );
    assert_eq!(once.frame, twice.frame, "the frame is not deterministic");
}
