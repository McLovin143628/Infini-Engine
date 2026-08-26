//! **PIE == shipping, over a trace whose exposure adapts** (wave VIS1b).
//!
//! Auto exposure is the first thing this renderer draws that depends on the
//! frames *before* the one being drawn: the multiplier a frame is tonemapped
//! with is the previous frame's multiplier moved toward this frame's target, by
//! the level clock's own delta. That is exactly the shape a PIE-vs-shipping claim
//! has to be made about, because two hosts that agree about a still frame can
//! still disagree about a sequence.
//!
//! The shape is `vsm_pie.rs`'s: one project, cooked; two worlds built from it —
//! one off the **cooked pack**, one off the editor's **`ScenePayload`** — stepped
//! in lockstep and projected through `inf_player::render::project_scene`, the one
//! door a shipped frame's scene comes through. What is compared is the exposure
//! trace, sixteen bytes a frame, read off the GPU.
//!
//! **Anti-vacuity comes first**, in three directions and in this order: the
//! level's clock has to actually run, the exposure has to actually MOVE along the
//! trace, and it has to arrive somewhere different from where it started. A
//! trace of one repeated number is equal to itself on both sides and says
//! nothing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{Light, LightKind, Material, SkyAtmosphere, TimeOfDay, Transform};
use inf_ecs::math::{Color, Vec3d};
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use inf_render::{
    EngineRenderer, ExposureMode, ExposureSettings, GpuContext, HeadlessTarget, RenderScene,
    RenderSettings, RenderView, HEADLESS_FORMAT,
};
use uuid::Uuid;

const LEVEL: Uuid = Uuid::from_u128(0x0165_1b00_0000_0001);
const W: u32 = 256;
const H: u32 = 144;
/// Steps each side takes. Enough that the adaptation has crossed most of the way
/// to its target without arriving, so the trace is a ramp rather than a step.
const STEPS: u64 = 24;
/// Simulated seconds of clock per real second. **Very** fast on purpose: twenty
/// four steps at 60 Hz is 0.4 real seconds, and the target the exposure chases is
/// the scene's own brightness — which only moves if the sun does. At this rate
/// the trace covers 16:56 to 20:16 and the sun sets inside it.
const CLOCK_RATE: f64 = 30_000.0;

/// The level's authored adaptation speed, **stops per second of LEVEL clock**.
///
/// It reads absurdly small and is not: `adaptation_speed` is stops per second of
/// the *document's* clock, and this level's clock runs thirty thousand times real
/// time. That the two are the same number is the whole point of stepping the eye
/// by the document, and it is the sentence a designer has to have read before
/// typing into the Adaptation row.
///
/// **And this level is past the discontinuity guard**, which is why the arithmetic
/// below is the guard's rather than the rate's. `passes::exposure::MAX_STEP_S`
/// clamps one frame's clock delta to **10 s**, so this trace adapts at
/// `4e-4 × 10 = 0.004` stops a frame — 0.096 over twenty-four, which is the
/// **0.0920** span the arm measures. A level at `rate == 30 000` is a fixture, not
/// a game; the guard's own doc names the ceiling (`rate == 600` at 60 fps) and
/// what happens above it.
const ADAPTATION_STOPS_PER_CLOCK_S: f32 = 4.0e-4;

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

/// A project whose level is a floor, a few emissive lamps, a sun, and — the part
/// this file is about — a **running** time-of-day clock.
fn dusk_project(tmp: &Path) -> (PathBuf, SceneDoc) {
    let proj = tmp.join("dusk");
    ProjectManifest::new("VIS1b Exposure", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mut doc = SceneDoc::new();
    let place = |doc: &mut SceneDoc, name: &str, t: Vec3d, s: Vec3d, mat: Option<Material>| {
        let id = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, name, None);
        let world = doc.world_mut();
        let e = world.entity_of(id).expect("the cube exists");
        world.world_mut().entity_mut(e).insert(Transform {
            translation: t,
            scale: s,
            ..Default::default()
        });
        if let Some(m) = mat {
            world.world_mut().entity_mut(e).insert(m);
        }
    };
    place(
        &mut doc,
        "Floor",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(80.0, 1.0, 80.0),
        None,
    );
    for (i, x) in [-3.0f64, 0.0, 3.0].into_iter().enumerate() {
        place(
            &mut doc,
            &format!("Lamp{i}"),
            Vec3d::new(x, 1.0, 0.0),
            Vec3d::new(1.0, 2.0, 1.0),
            Some(Material {
                base_color: Color::new(0.04, 0.04, 0.05, 1.0),
                // An 8-bit colour and an authored intensity — the clause-2 path,
                // exercised here for free because a scene with something bright
                // in it is what makes an exposure trace interesting.
                emissive: Color::new(1.0, 0.7, 0.4, 1.0),
                emissive_intensity: 6.0,
                ..Material::default()
            }),
        );
    }
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
            ..Default::default()
        });
        // **The clock, and the atmosphere that gives it something to change.**
        // `rate` is what makes this a trace rather than a still: the sim advances
        // `TimeOfDay::seconds` by `rate * dt` every fixed step, on both sides,
        // through the one Ring-0 door `inf_ecs::sky::advance_time_of_day`.
        world.world_mut().entity_mut(e).insert(TimeOfDay {
            seconds: 61_000.0, // ~16:56 UTC — the sun on its way down
            day_of_year: 172,
            latitude_deg: 48.9,
            longitude_deg: 0.0,
            rate: CLOCK_RATE,
        });
        world.world_mut().entity_mut(e).insert(SkyAtmosphere {
            enabled: true,
            ..SkyAtmosphere::default()
        });
    }

    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(&content, "Dusk.inf_lvl", LEVEL, &level, AssetKind::Level);
    (proj, doc)
}

/// The PIE payload for `doc`, served from the project on disk.
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
        0,
        false,
    )
    .expect("the payload builds")
}

fn view() -> RenderView {
    let eye = glam::DVec3::new(0.0, 2.4, 9.0);
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: eye,
        forward: (glam::DVec3::new(0.0, 1.0, 0.0) - eye)
            .as_vec3()
            .normalize(),
        up: glam::Vec3::Y,
        fov_y: 55f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn auto_exposure() -> RenderSettings {
    RenderSettings {
        exposure_control: ExposureSettings {
            mode: ExposureMode::Auto,
            // Deliberately slow relative to the clock: a fast adaptation arrives
            // on frame two and the rest of the trace is a constant, which is the
            // vacuous shape this file exists to avoid.
            adaptation_speed: ADAPTATION_STOPS_PER_CLOCK_S,
            ..ExposureSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// One scripted run: step the sim, project, render `renders_per_step` times, read
/// the exposure back.
///
/// `renders_per_step` is the arm's own falsifier. A host that drew two frames per
/// simulation step — a windowed PIE at 120 Hz over a 60 Hz sim, which is the
/// ordinary case — must land on the SAME exposure as one that drew one, because
/// the extra render advances no clock and therefore adapts by nothing. An
/// adaptation stepped by a frame counter would fail this and pass everything
/// else in the file.
fn scripted(
    gpu: &GpuContext,
    sim: &mut inf_player::runtime_sim::RuntimeSim,
    renders_per_step: usize,
) -> Vec<(f64, f32, f32)> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(auto_exposure());
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();

    let mut trace = Vec::with_capacity(STEPS as usize);
    for _ in 0..STEPS {
        sim.step_once(Default::default());
        let mut scene = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        inf_player::render::project_scene(&mut scene, sim, 0.0, &vmeshes);
        scene.grid_enabled = false;
        scene.mark_dirty();
        let clock = scene.atmosphere.clouds.time_s;
        for _ in 0..renders_per_step.max(1) {
            renderer.render(gpu, &scene, &view(), &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let s = renderer.read_exposure(gpu).expect("exposure readback");
        trace.push((clock, s.ev, s.avg_luminance));
    }
    trace
}

#[test]
fn pie_equals_shipping_on_a_trace_whose_exposure_adapts() {
    let Some(gpu) = gpu_or_skip("the VIS1b PIE-vs-shipping exposure trace") else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = dusk_project(tmp.path());
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    // The shipping side: the cooked pack.
    let source =
        inf_player::level::PackLevelSource::open(&report.pack_path).expect("open the cooked pack");
    let built = inf_player::build_world_from_pack(&source).expect("build the world");
    let mut shipped = inf_player::sim_from_built(built);
    // The preview side: the editor's payload.
    let mut previewed = inf_player::sim_from_payload(&payload_for(&proj, &doc))
        .expect("the payload builds a sim")
        .sim;

    // ── ASSERT THE WORLD BEFORE COMPARING TWO OF THEM (the P21.4 law) ──
    //
    // Both sides must carry the sky authority, or the "trace" below is two
    // renders of a frozen clock agreeing about nothing.
    for (name, sim) in [
        ("shipped", &shipped as &inf_player::runtime_sim::RuntimeSim),
        ("previewed", &previewed),
    ] {
        let sky = inf_ecs::sky::resolve_sky(sim.world())
            .unwrap_or_else(|| panic!("{name} has no sky authority"));
        assert_eq!(
            sky.time_of_day.rate, CLOCK_RATE,
            "{name} clock is not running"
        );
    }

    let a = scripted(&gpu, &mut shipped, 1);
    let b = scripted(&gpu, &mut previewed, 1);

    // ── anti-vacuity, three ways ──
    assert_eq!(a.len(), STEPS as usize);
    let clock_moved = a.last().unwrap().0 - a[0].0;
    assert!(
        clock_moved > 1.0,
        "the level clock did not run: {clock_moved} s over {STEPS} steps"
    );
    assert!(
        a.iter().any(|s| s.2 > 0.0),
        "the histogram measured nothing on the shipping side"
    );
    let ev_span = a.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max)
        - a.iter().map(|s| s.1).fold(f32::INFINITY, f32::min);
    assert!(
        ev_span > 0.02,
        "the exposure never adapted: EV span {ev_span} over {STEPS} frames"
    );
    assert!(
        a.windows(2).filter(|w| w[0].1 != w[1].1).count() > STEPS as usize / 2,
        "the exposure is a step rather than a ramp"
    );

    eprintln!(
        "exposure trace: clock +{clock_moved:.1} s, EV {:.4} -> {:.4} (span {ev_span:.4}), \
         avg luminance {:.4} -> {:.4}",
        a[0].1,
        a.last().unwrap().1,
        a[0].2,
        a.last().unwrap().2
    );

    // ── and the two are the same trace ──
    assert_eq!(
        a, b,
        "PIE and shipping disagreed about the exposure of a level they were both handed"
    );

    // ── the frame rate is not an input ──
    //
    // The same sim, the same clock, THREE renders per step instead of one. A
    // windowed PIE running at 180 Hz over a 60 Hz sim is exactly this, and it
    // must land on the same exposures — the extra renders advance no clock and
    // therefore adapt by nothing.
    let (proj2, doc2) = dusk_project(tmp.path());
    let report2 = cook(&proj2, &tmp.path().join("out2"), &cook_opts()).expect("the project cooks");
    let source2 = inf_player::level::PackLevelSource::open(&report2.pack_path).expect("open");
    let mut fast = inf_player::sim_from_built(
        inf_player::build_world_from_pack(&source2).expect("build the world"),
    );
    let _ = doc2;
    let c = scripted(&gpu, &mut fast, 3);
    assert_eq!(
        a, c,
        "the exposure moved with the FRAME count rather than with the clock"
    );
}
