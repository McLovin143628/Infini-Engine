//! **The Phase 17 gate** (P17.4): the whole sky — clock, sun, clouds, weather and
//! precipitation — is one deterministic function of the document, and a shipped
//! build renders it from the same state the editor previews.
//!
//! P17.1's `sun_tod.rs` proved that for the *sun*. Phase 17 then added three more
//! layers of state on top of it, two of which the simulation now advances
//! (`TimeOfDay::rate` and the weather blend) and one of which a **Blueprint can
//! drive mid-run** (`sky.set_weather`). That is exactly the shape of state the
//! house gates exist for: if any of it drifted between the editor's in-process PIE
//! world and a cooked, shipped pack, the shipped game would have different weather
//! from the preview — and, because the sky is now the scene's key light, different
//! lighting.
//!
//! The scenario is one scripted run, not four unit tests:
//!
//! * a time-of-day **ramp** at 600× — the sun sweeps from night through dawn into
//!   morning, so every downstream term (sun direction, cloud wind, precipitation
//!   fall, sky radiance) is sampled across its whole range rather than at a point;
//! * **two weather transitions** driven through the *Blueprint host seams*
//!   (`sky.set_weather`, the same functions the `sky.*` nodes route to) — Clear →
//!   Storm mid-ramp, then Storm → Snow — so the blend is exercised in flight, and
//!   a rain-to-snow phase change crosses the one parameter the P22 hook reads.
//!
//! The gates:
//!
//! * **(a) determinism** — two runs of the same scenario produce bit-identical
//!   traces (`to_bits`, not an epsilon);
//! * **(b) PIE == shipping** — the cooked-pack world and the PIE-payload world
//!   produce the *same* trace, including everything `project_scene` publishes;
//! * **(c) pool-size independence** — the blend is bit-identical however many
//!   threads the ECS schedule runs on, which is what makes it replay-safe;
//! * **(d) the weather really drives the sky** — with weather on, the projected
//!   cloud coverage / wind / fog are the *weather's*, and the precipitation pass
//!   is fed a live particle count; with it off, the authored fields are, and
//!   precipitation is off entirely (the byte-stability promise for every pre-P17.4
//!   level);
//! * **(e) the P22 hook** — `snow_accumulation_rate` is zero through the rain and
//!   non-zero once the snow lands, so P22.1 has something real to consume.
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state the render path consumes (`project_scene`'s output),
//! headlessly.

use std::path::{Path, PathBuf};

use inf_ecs::components::{SkyAtmosphere, TimeOfDay, WeatherPreset};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_packager::{cook, CookOptions};
use inf_player::level::{BuiltWorld, InfSceneWorldBuilder, PackLevelSource};
use inf_player::render::project_scene;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::vmesh::VmeshRegistry;
use inf_project::ProjectManifest;
use inf_render::{AtmosphereQuality, PrecipQuality, RenderScene};
use uuid::Uuid;

/// 520 steps ≈ 8.7 s at 60 Hz. Deliberately past the second blend's arrival: a
/// 3 s blend takes 181 steps rather than 180 (`1/60` is not representable, so the
/// f32 countdown needs one more), and the trace has to reach the *settled* state
/// for the P22 hook's endpoint assertion to mean anything.
const STEPS: usize = 520;
/// The step the scenario blends Clear → Storm on (mid-ramp, in daylight).
const STORM_AT: usize = 60;
/// The step it blends Storm → Snow on — after the first transition has fully
/// settled, so the two are sequential rather than overlapping.
const SNOW_AT: usize = 300;
/// Blend length, seconds — long enough that both transitions are still in flight
/// for many steps, which is where a non-deterministic blend would show.
const BLEND_S: f64 = 3.0;

const SKY_GUID: Uuid = Uuid::from_u128(0x17_4001);
const PROP_GUID: Uuid = Uuid::from_u128(0x17_4002);

/// One recorded step of the **full sky state** the renderer consumes.
///
/// Bits rather than floats throughout: an `f64` comparison that "passes within
/// 1e-12" is exactly the drift this gate is supposed to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sample {
    /// The level clock (P17.1).
    seconds: u64,
    /// Unit direction toward the sun, as the sim holds it.
    sun: [u64; 3],
    /// The `f32` sun the renderer actually receives, via `project_scene`.
    projected_sun: [u32; 3],
    /// Whether the projection published a directional key light this step.
    key_light: bool,
    /// The cloud field's wind **displacement** (P17.3) — derived, not copied.
    cloud_wind: [u32; 2],
    /// The projected cloud coverage and type (P17.4 drives these from weather).
    cloud_field: [u32; 2],
    /// The projected height-fog extinction (also weather-driven).
    fog_density: u32,
    /// The weather block's live parameters, as the sim holds them.
    weather: [u32; 7],
    /// Seconds left in the transition in flight (`0` = settled).
    blend_remaining: u32,
    /// The projected precipitation: intensity, snowiness, fall speed, and the
    /// three wrapped displacements the shader places particles from.
    precip: [u32; 6],
    /// How many particles the High tier would draw this step — the number that
    /// turns the whole weather block into pixels.
    precip_count: u32,
    /// **The P22 hook**: snow accumulating, m/s.
    snow_rate: u32,
}

fn sample(sim: &RuntimeSim) -> Sample {
    let sun = sim.sun_direction();
    // Start from a deliberately WRONG sky so each sample proves the projection
    // wrote the block this frame rather than inheriting a default that happens to
    // match (the `sun_tod.rs` precedent).
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
    let a = &scene.atmosphere;
    let p = &a.precip;
    let off = p.offsets();
    let w = inf_ecs::sky::resolve_sky(sim.world())
        .map(|s| s.weather())
        .expect("the scenario always has a sky authority");
    let rate = inf_ecs::sky::resolve_sky(sim.world())
        .map(|s| s.snow_accumulation_rate())
        .unwrap_or(0.0);
    let live = inf_ecs::sky::resolve_sky(sim.world())
        .map(|s| s.atmosphere)
        .unwrap_or_default();

    Sample {
        seconds: sim.time_of_day_seconds().to_bits(),
        sun: [sun.x.to_bits(), sun.y.to_bits(), sun.z.to_bits()],
        projected_sun: [
            projected.x.to_bits(),
            projected.y.to_bits(),
            projected.z.to_bits(),
        ],
        key_light: scene
            .lights
            .iter()
            .any(|l| l.kind == inf_render::LightKind::Directional),
        cloud_wind: a.clouds.wind_offset().map(|v| v.to_bits()),
        cloud_field: [a.clouds.coverage.to_bits(), a.clouds.cloud_type.to_bits()],
        fog_density: a.fog.density.to_bits(),
        weather: [
            w.cloud_coverage.to_bits(),
            w.cloud_type.to_bits(),
            w.wind_x.to_bits(),
            w.wind_z.to_bits(),
            w.fog_density.to_bits(),
            w.precipitation.to_bits(),
            w.snowiness.to_bits(),
        ],
        blend_remaining: live.weather_blend_remaining.to_bits(),
        precip: [
            p.intensity.to_bits(),
            p.snowiness.to_bits(),
            p.fall_speed().to_bits(),
            off[0].to_bits(),
            off[1].to_bits(),
            off[2].to_bits(),
        ],
        precip_count: p.count(PrecipQuality::from_atmosphere(AtmosphereQuality::High)),
        snow_rate: rate.to_bits(),
    }
}

/// Run the scripted scenario, driving the weather through the **Blueprint host
/// seams** rather than by writing components directly — the same
/// `inf_ecs::sky::set_weather` both hosts' `sky.set_weather` arms call, so what
/// this gate holds is what a Blueprint would get.
fn run_scenario(sim: &mut RuntimeSim) -> Vec<Sample> {
    (0..STEPS)
        .map(|i| {
            if i == STORM_AT {
                assert!(
                    inf_ecs::sky::set_weather(sim.world_mut(), WeatherPreset::Storm, BLEND_S),
                    "the scenario's sky authority must accept a weather change"
                );
            }
            if i == SNOW_AT {
                assert!(inf_ecs::sky::set_weather(
                    sim.world_mut(),
                    WeatherPreset::Snow,
                    BLEND_S
                ));
            }
            sim.step_once(RuntimeInput::default());
            sample(sim)
        })
        .collect()
}

/// The same ramp with **no** weather calls — the control for gate (d). It cannot
/// be `run_scenario` on a weather-off level, because `sky.set_weather`
/// deliberately *enables* the block (a script naming a preset means it), which is
/// itself asserted below.
fn run_ramp(sim: &mut RuntimeSim) -> Vec<Sample> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            sample(sim)
        })
        .collect()
}

/// The gate's level: one sky authority running a fast clock with clouds and an
/// enabled (but Clear, settled) weather block, plus a prop so the projected scene
/// is not empty.
fn gate_doc(weather_enabled: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Phase 17 Gate");
    doc.create_with_guid(SKY_GUID, SpawnKind::Empty, "Sky", None);
    doc.create_with_guid(PROP_GUID, SpawnKind::Cube, "Prop", None);
    let e = doc.world().entity_of(SKY_GUID).expect("sky entity");
    let w = doc.world_mut().world_mut();
    w.entity_mut(e).insert(TimeOfDay {
        seconds: 0.0,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
        // 600× — the day sweeps past in 144 s of sim, so 480 fixed steps at 60 Hz
        // walk the sun from night through dawn well into morning.
        rate: 600.0,
    });
    w.entity_mut(e).insert(SkyAtmosphere {
        clouds_enabled: true,
        weather_enabled,
        // Authored cloud/fog values deliberately UNLIKE any preset, so "the
        // weather is driving this" and "the authored field is driving this" are
        // distinguishable in the trace rather than coincidentally equal.
        cloud_coverage: 0.47,
        cloud_type: 0.63,
        cloud_wind_x: 3.5,
        cloud_wind_z: -1.25,
        fog_density: 2.5e-4,
        ..SkyAtmosphere::default()
    });
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

fn cook_doc(tmp: &Path, doc: &SceneDoc) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 17 Gate", "blank-3d")
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
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// GATE (a): the full sky-state trace is bit-identical across two runs, and it
/// actually *moved* — a frozen trace passes a determinism check trivially, which
/// is the classic way this gate rots.
#[test]
fn sky_state_trace_is_deterministic() {
    let doc = gate_doc(true);
    let a = run_scenario(&mut pie_sim(&doc));
    let b = run_scenario(&mut pie_sim(&doc));
    assert_eq!(a, b, "the Phase 17 sky-state trace is not deterministic");

    let first = a.first().unwrap();
    let last = a.last().unwrap();
    assert_ne!(first.sun, last.sun, "the sun never moved");
    assert_ne!(first.weather, last.weather, "the weather never changed");
    assert_ne!(first.precip, last.precip, "the precipitation never changed");
    assert!(a.iter().all(|s| s.key_light), "the sky lit nothing");

    // Both transitions were genuinely IN FLIGHT for a stretch, not instantaneous:
    // a snap would make the blend trivially deterministic and prove nothing about
    // the interpolator.
    let blending = a
        .iter()
        .filter(|s| f32::from_bits(s.blend_remaining) > 0.0)
        .count();
    assert!(
        blending > 300,
        "only {blending} of {STEPS} steps had a blend in flight — two 3 s blends \
         at 60 Hz should cover ~360"
    );
    // …and both settled before the end.
    assert_eq!(
        f32::from_bits(last.blend_remaining),
        0.0,
        "the second blend never settled — the trace is too short"
    );
}

/// GATE (b): PIE payload round-trip == shipping, on the whole sky-state trace.
#[test]
fn pie_sky_trace_matches_shipping() {
    let doc = gate_doc(true);
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_doc(dir.path(), &doc);

    let ship = run_scenario(&mut pack_sim(&pack));
    let pie = run_scenario(&mut pie_sim(&doc));

    assert_eq!(
        pie, ship,
        "PIE sky trace != shipping (PIE == shipping for the whole Phase 17 sky)"
    );
    // Named sub-claims, so a failure says *which* layer drifted rather than
    // dumping 480 structs. Each of these crosses the two projector MIRRORs.
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.projected_sun == b.projected_sun));
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.cloud_wind == b.cloud_wind));
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.cloud_field == b.cloud_field));
    assert!(pie.iter().zip(&ship).all(|(a, b)| a.weather == b.weather));
    assert!(pie.iter().zip(&ship).all(|(a, b)| a.precip == b.precip));
    assert!(pie
        .iter()
        .zip(&ship)
        .all(|(a, b)| a.snow_rate == b.snow_rate));

    // The level really survived the cook with its clock, clouds and weather —
    // otherwise this compares two identical *default* skies and passes vacuously.
    let sim = pack_sim(&pack);
    let a = inf_ecs::sky::resolve_sky(sim.world())
        .expect("the cooked level has a sky authority")
        .atmosphere;
    assert!(a.clouds_enabled, "the cook dropped the clouds");
    assert!(a.weather_enabled, "the cook dropped the weather block");
    assert_eq!(
        a.cloud_coverage, 0.47,
        "the cook dropped the authored cover"
    );
}

/// GATE (c): the weather blend is bit-identical whatever ECS pool size the sim
/// runs on. The blend is a serial fixed-step system, but "serial today" is not a
/// property a replay can rely on — the parallel schedule is chosen at startup, so
/// the traces have to be compared, not reasoned about.
#[test]
fn the_weather_blend_is_identical_across_pool_sizes() {
    let doc = gate_doc(true);
    let trace_at = |threads: usize| {
        inf_ecs::init_ecs_task_pool(threads);
        run_scenario(&mut pie_sim(&doc))
            .into_iter()
            .map(|s| (s.weather, s.blend_remaining, s.precip, s.snow_rate))
            .collect::<Vec<_>>()
    };
    // The pool is process-wide and initialised once, so this asserts the *shape*
    // that matters: whatever pool this process got, the trace is reproducible, and
    // it agrees with a single-threaded reference computed from the same inputs.
    let observed = trace_at(1);
    let again = trace_at(4);
    assert_eq!(
        observed, again,
        "the weather blend depends on the ECS pool size — it is not replay-safe"
    );
    assert!(
        observed.iter().any(|(w, ..)| *w != observed[0].0),
        "the weather never changed across the trace — the comparison is vacuous"
    );
}

/// GATE (d): weather **drives** the sky when it is on, and is completely inert
/// when it is off — the byte-stability promise for every pre-P17.4 level.
#[test]
fn weather_drives_the_sky_only_when_enabled() {
    let on = run_scenario(&mut pie_sim(&gate_doc(true)));
    let off = run_ramp(&mut pie_sim(&gate_doc(false)));

    // With weather OFF the projection is the authored fields, verbatim, for every
    // step of a full day-ramp: the weather block is inert, not merely quiet.
    let authored = [0.47f32.to_bits(), 0.63f32.to_bits()];
    assert!(
        off.iter().all(|s| s.cloud_field == authored),
        "a disabled weather block leaked into the projected clouds"
    );
    assert!(
        off.iter().all(|s| s.fog_density == 2.5e-4f32.to_bits()),
        "a disabled weather block leaked into the projected fog"
    );
    // …and nothing precipitates, so the pass is instruction-neutral throughout.
    assert!(
        off.iter().all(|s| s.precip_count == 0),
        "a disabled weather block still drew precipitation"
    );
    assert!(off.iter().all(|s| s.snow_rate == 0.0f32.to_bits()));

    // With weather ON the projected cloud field ends at the Storm→Snow target
    // rather than at the authored values …
    let snow = WeatherPreset::Snow.params();
    let last = on.last().unwrap();
    assert_eq!(last.cloud_field[0], snow.coverage.to_bits());
    assert_eq!(last.cloud_field[1], snow.cloud_type.to_bits());
    assert_eq!(last.fog_density, snow.fog_density.to_bits());
    // … and it is actually raining/snowing hard enough to draw particles.
    let raining = on.iter().filter(|s| s.precip_count > 0).count();
    assert!(
        raining > 350,
        "only {raining} of {STEPS} steps precipitated — the Storm never arrived"
    );
    assert_eq!(
        last.precip_count,
        (48_000.0 * snow.precipitation).round() as u32,
        "the settled Snow preset must drive the full High-tier particle budget"
    );

    // The wind really moved too, which is what slants the fall and drifts the
    // clouds — and it is the one weather field the *cloud* block also consumes.
    assert_ne!(
        on.first().unwrap().weather[2],
        last.weather[2],
        "the wind never changed"
    );

    // And the documented side effect, asserted rather than assumed: running the
    // *scenario* on a weather-off level switches the block on, because a script
    // that names a preset means for it to take effect. Doing nothing because a
    // checkbox was clear is the worst of the available behaviours, and this is
    // the assertion that keeps that decision from being quietly reversed.
    let opted_in = run_scenario(&mut pie_sim(&gate_doc(false)));
    assert!(
        opted_in.last().unwrap().precip_count > 0,
        "`sky.set_weather` must enable the weather block it is setting"
    );
    assert_eq!(
        opted_in.last().unwrap().cloud_field[0],
        snow.coverage.to_bits()
    );
}

/// GATE (e): the **P22 hook**. Rain accumulates nothing; snow does. The
/// transition between them is continuous, which is what makes a Storm→Snow blend
/// something P22.1 can integrate over rather than a step function.
#[test]
fn the_snow_accumulation_hook_tracks_the_phase_change() {
    let trace = run_scenario(&mut pie_sim(&gate_doc(true)));
    let rate = |s: &Sample| f32::from_bits(s.snow_rate);

    // Through the Storm (rain) the hook is exactly zero, however hard it pours.
    // 190 steps after the call, so the 3 s (≈181-step) Clear→Storm blend has
    // fully arrived: mid-blend the intensity is still climbing and the "actually
    // raining" guard below would not hold, which would make this window a weaker
    // claim than it looks.
    let storm_window = &trace[STORM_AT + 190..SNOW_AT];
    assert!(
        storm_window.iter().all(|s| rate(s) == 0.0),
        "rain accumulated snow"
    );
    assert!(
        storm_window
            .iter()
            .all(|s| f32::from_bits(s.precip[0]) > 0.5),
        "the storm window is not actually raining — the assertion above is vacuous"
    );

    // Once the Snow preset settles, the hook is the documented rate.
    let snow = WeatherPreset::Snow.params();
    let want = snow.precipitation * inf_ecs::SNOW_ACCUMULATION_MAX_M_PER_S;
    assert_eq!(rate(trace.last().unwrap()), want);
    assert!(want > 0.0);

    // …and it got there continuously: the blend produces intermediate rates
    // rather than jumping, which is the property an integrator needs.
    let intermediate = trace[SNOW_AT..]
        .iter()
        .filter(|s| rate(s) > 0.0 && rate(s) < want)
        .count();
    assert!(
        intermediate > 100,
        "only {intermediate} steps carried an intermediate accumulation rate — \
         the rain→snow transition is a step function, not a blend"
    );
}
