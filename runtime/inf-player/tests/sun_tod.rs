//! P17.1 gate: the time-of-day sun is deterministic, and **PIE == shipping** on a
//! sun-direction trace over a scripted clock ramp.
//!
//! The sun stopped being a compile-time constant in P17.1: it is now a pure
//! function of a `TimeOfDay` component that the *simulation* advances. That makes
//! it exactly the kind of state the house gates exist for — if the clock or the
//! solar math drifted between the editor's in-process PIE world and a cooked,
//! shipped pack, the shipped game would light differently from the preview.
//!
//! * **(a) determinism** — two runs of the same scripted ramp produce
//!   bit-identical clock + sun traces (`to_bits`, not an epsilon).
//! * **(b) PIE == shipping** — the cooked-pack world and the PIE-payload world
//!   produce the *same* trace, including the projected `RenderScene.sun` the
//!   renderer would consume.
//! * **(c) the clock is only the sim's** — `rate == 0` freezes the sun no matter
//!   how many steps run, and a level with no clock projects the retired
//!   `SUN_DIR` direction exactly.
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state the render path consumes (`project_scene`'s output),
//! headlessly.

use std::path::{Path, PathBuf};

use inf_ecs::components::{SkyAtmosphere, TimeOfDay};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_packager::{cook, CookOptions};
use inf_player::level::{BuiltWorld, InfSceneWorldBuilder, PackLevelSource};
use inf_player::render::project_scene;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::vmesh::VmeshRegistry;
use inf_project::ProjectManifest;
use inf_render::RenderScene;
use uuid::Uuid;

const STEPS: usize = 240;
const SKY_GUID: Uuid = Uuid::from_u128(0x17_0001);
const PROP_GUID: Uuid = Uuid::from_u128(0x17_0002);

/// One recorded step: the clock and the sun direction, as raw bits.
///
/// Bits rather than floats on purpose — an `f64` comparison that "passes within
/// 1e-12" is exactly the drift this gate is supposed to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sample {
    seconds: u64,
    sun: [u64; 3],
    /// The `f32` sun the renderer would actually receive, via `project_scene`.
    projected: [u32; 3],
    /// Whether the projection published a directional key light this step.
    key_light: bool,
    /// The cloud field's **wind displacement** this step, as `f32` bits (P17.3).
    ///
    /// This is the only *derived* value in the sky projection — everything else
    /// is a component field copied across — so it is the one most able to drift
    /// between the two hosts without anything else noticing. It is a pure
    /// function of the level's clock (`ResolvedSky::cloud_time_s`), which is
    /// exactly the claim this gate exists to hold: two runs at the same time of
    /// day must see the same sky, in the editor and in a shipped build alike.
    cloud_wind: [u32; 2],
}

/// The retired `SUN_DIR`, as the `f32` bits a projection publishes — what a
/// clockless level (and a deleted feature) would produce.
fn retired_sun_bits() -> [u32; 3] {
    let d = inf_render::DEFAULT_SUN_DIR.normalize();
    [d.x.to_bits(), d.y.to_bits(), d.z.to_bits()]
}

fn sample(sim: &RuntimeSim) -> Sample {
    let sun = sim.sun_direction();
    // Start from a deliberately WRONG sun so each sample proves the projection
    // wrote the block this frame, rather than inheriting a default that happens to
    // match. (Same reasoning as the `project_sky` unit tests.)
    let mut scene = RenderScene {
        sun: inf_render::SunParams {
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            intensity: 999.0,
            ..inf_render::SunParams::default()
        },
        ..RenderScene::default()
    };
    project_scene(&mut scene, sim, 0.0, &VmeshRegistry::new());
    let projected = scene.sun.unit_direction();
    Sample {
        seconds: sim.time_of_day_seconds().to_bits(),
        sun: [sun.x.to_bits(), sun.y.to_bits(), sun.z.to_bits()],
        projected: [
            projected.x.to_bits(),
            projected.y.to_bits(),
            projected.z.to_bits(),
        ],
        key_light: scene
            .lights
            .iter()
            .any(|l| l.kind == inf_render::LightKind::Directional),
        cloud_wind: scene.atmosphere.clouds.wind_offset().map(|v| v.to_bits()),
    }
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Sample> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            sample(sim)
        })
        .collect()
}

/// A tiny level: one sky authority running a fast clock, plus a prop so the
/// projected scene is not empty.
fn sky_doc(rate: f64) -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Sun TOD Gate");
    doc.create_with_guid(SKY_GUID, SpawnKind::Empty, "Sky", None);
    doc.create_with_guid(PROP_GUID, SpawnKind::Cube, "Prop", None);
    let e = doc.world().entity_of(SKY_GUID).expect("sky entity");
    let w = doc.world_mut().world_mut();
    w.entity_mut(e).insert(TimeOfDay {
        seconds: 0.0,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
        // 600× — the whole day sweeps past in 144 s of sim, so 240 fixed steps at
        // 60 Hz walk the sun from night through dawn into morning.
        rate,
    });
    // Clouds ON: the trace then covers the wind's derivation from the clock,
    // which is the projection's only computed field (P17.3).
    w.entity_mut(e).insert(SkyAtmosphere {
        clouds_enabled: true,
        ..SkyAtmosphere::default()
    });
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

/// Scaffold a project holding `doc`, cook it, return the pack dir.
fn cook_doc(tmp: &Path, doc: &SceneDoc) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Sun TOD Gate", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    inf_editor_core::scene::serialize::save(doc, &content.join("Sky.inf_lvl"), None)
        .expect("save level");
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let builder = InfSceneWorldBuilder::with_defaults(actors);
    let built: BuiltWorld = inf_player::level::load(&source, &builder).expect("pack level builds");
    inf_player::sim_from_built(built)
}

fn pie_sim(doc: &SceneDoc) -> RuntimeSim {
    let payload = inf_editor_core::pie::build_scene_payload(
        doc,
        |_guid| None, // no blueprint actors
        |_guid| None, // no PCG graphs
        |_guid| None, // no animation assets
        |_guid| None, // no biome sets
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// GATE (a): the same scripted ramp twice is bit-identical.
#[test]
fn tod_ramp_is_deterministic() {
    let doc = sky_doc(600.0);
    let a = run_trace(&mut pie_sim(&doc));
    let b = run_trace(&mut pie_sim(&doc));
    assert_eq!(a, b, "the time-of-day ramp is not deterministic");

    // …and it actually moved: a frozen trace would pass a determinism check
    // trivially, which is the classic way this gate rots.
    assert_ne!(
        a.first().unwrap().sun,
        a.last().unwrap().sun,
        "the sun never moved — the ramp proves nothing"
    );
    assert_ne!(a.first().unwrap().seconds, a.last().unwrap().seconds);
    // 240 steps × 1/60 s × 600 = 2400 sim-seconds.
    let end = f64::from_bits(a.last().unwrap().seconds);
    assert!((end - 2_400.0).abs() < 1e-6, "clock ended at {end}");
    // Every step published the sun (or moon) as a directional key light, and none
    // of them is the retired constant — so a deleted projection cannot pass.
    assert!(a.iter().all(|s| s.key_light));
    assert!(
        a.iter().all(|s| s.projected != retired_sun_bits()),
        "the trace fell back to the retired sun — the projection is not running"
    );
}

/// GATE (b): PIE payload round-trip == shipping, on the sun trace.
#[test]
fn pie_sun_trace_matches_shipping() {
    let doc = sky_doc(600.0);
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_doc(dir.path(), &doc);

    let ship = run_trace(&mut pack_sim(&pack));
    let pie = run_trace(&mut pie_sim(&doc));

    assert_eq!(
        pie, ship,
        "PIE sun trace != shipping (PIE == shipping for time of day)"
    );
    // The projected f32 the renderer consumes agrees too, not just the f64 the
    // sim holds — the two projector MIRRORs are the thing most likely to drift.
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.projected == b.projected));
    // ...and so does the cloud wind, which is derived rather than copied and is
    // therefore the field a MIRROR drift would move first (P17.3).
    assert!(
        pie.iter()
            .zip(&ship)
            .all(|(a, b)| a.cloud_wind == b.cloud_wind),
        "PIE and shipping disagree about the cloud wind"
    );
    // The wind actually moved during the trace, or the assertion above compares
    // 240 copies of zero and proves nothing.
    assert!(
        pie.first().map(|s| s.cloud_wind) != pie.last().map(|s| s.cloud_wind),
        "the cloud wind never moved across the trace — the gate is vacuous"
    );
}

/// The level really did survive the cook with its clock intact — otherwise
/// GATE (b) would compare two identical *default* suns and pass vacuously.
#[test]
fn the_cooked_level_carries_its_clock() {
    let doc = sky_doc(600.0);
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_doc(dir.path(), &doc);
    let sim = pack_sim(&pack);
    assert_eq!(sim.time_of_day_seconds(), 0.0);

    let mut ramped = pack_sim(&pack);
    for _ in 0..60 {
        ramped.step_once(RuntimeInput::default());
    }
    assert!(
        (ramped.time_of_day_seconds() - 600.0).abs() < 1e-6,
        "cooked clock did not advance: {}",
        ramped.time_of_day_seconds()
    );
}

/// GATE (c1): `rate == 0` freezes the sun, however long the sim runs.
#[test]
fn a_frozen_clock_never_moves_the_sun() {
    let doc = sky_doc(0.0);
    let mut sim = pie_sim(&doc);
    let first = sample(&sim);
    // The projection must actually be RUNNING, or "nothing moved" is what a
    // deleted feature also reports. The clock is present, the sun is published as
    // a key light, and it is NOT the retired default.
    assert!(first.key_light, "the sky authority published no key light");
    assert_ne!(
        first.projected,
        retired_sun_bits(),
        "the frozen clock must still project ITS sun, not the retired constant"
    );
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    let last = sample(&sim);
    assert_eq!(first, last, "a frozen clock moved the sun");
    assert_eq!(f64::from_bits(last.seconds), 0.0);
}

/// GATE (c2): a level with **no** clock projects the retired `SUN_DIR` exactly —
/// the byte-stability promise that keeps every pre-P17.1 level and golden intact.
#[test]
fn a_clockless_level_keeps_the_retired_sun() {
    let mut doc = SceneDoc::new();
    doc.set_title("No Clock");
    doc.create_with_guid(PROP_GUID, SpawnKind::Cube, "Prop", None);
    let mut sim = pie_sim(&doc);
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }

    // Deliberately dirty the scene first — mirroring the `project_sky` unit test in
    // `inf_viewport::host`. Starting from `RenderScene::default()` would make every
    // assertion below hold even with `project_sky` deleted; dirtying proves the
    // projection actively RESETS the block to the retired defaults each frame.
    let mut scene = RenderScene {
        sun: inf_render::SunParams {
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            intensity: 999.0,
            moon_intensity: 42.0,
            ..inf_render::SunParams::default()
        },
        sky: inf_render::SkyParams {
            zenith: [1.0, 0.0, 1.0],
            horizon: [1.0, 0.0, 1.0],
            ground: [1.0, 0.0, 1.0],
        },
        ..RenderScene::default()
    };
    project_scene(&mut scene, &sim, 0.0, &VmeshRegistry::new());
    assert_eq!(
        scene.sun,
        inf_render::SunParams::default(),
        "a clockless level must be RESET to the retired constant, not left as found"
    );
    assert_eq!(
        scene.sun.unit_direction(),
        inf_render::DEFAULT_SUN_DIR.normalize()
    );
    assert_eq!(scene.sky, inf_render::SkyParams::default());
    assert!(
        !scene
            .lights
            .iter()
            .any(|l| l.kind == inf_render::LightKind::Directional),
        "a clockless level must not gain a sun light"
    );
    assert_eq!(sim.time_of_day_seconds(), 0.0);
}
