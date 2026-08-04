//! P20.1 gate: a water scene projects **deterministically**, and **PIE ==
//! shipping** on a water projection trace over a scripted clock ramp.
//!
//! Water is derived rather than authored. A `WaterBody` carries an amplitude, a
//! wavelength and a seed; what a player *sees* is a Gerstner sum whose every
//! component's direction, wavenumber, frequency, steepness and phase come out of
//! a hash and the level's clock. That makes it exactly the kind of state the
//! house gates exist for: if the derivation, the clock or the wind rule drifted
//! between the editor's in-process PIE world and a cooked, shipped pack, the
//! shipped game's sea would be a *different sea* from the preview's — and it
//! would still look like a sea, so nothing would notice.
//!
//! * **(a) determinism** — two runs of the same scripted ramp produce
//!   bit-identical water traces (`to_bits`, not an epsilon).
//! * **(b) PIE == shipping** — the cooked-pack world and the PIE-payload world
//!   produce the *same* trace, including the derived wave parameters and the
//!   river ribbon the renderer would consume.
//! * **(c) the trace is not vacuous** — the sea really moves with the clock, the
//!   weather wind really reaches the waves, and a body that opted out of the
//!   weather really is unmoved by it.
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state the render path consumes (`project_scene`'s output),
//! headlessly.

use std::path::{Path, PathBuf};

use inf_ecs::components::{
    SkyAtmosphere, Spline, SplineInterp, TimeOfDay, Transform, WaterBody, WaterKind, WeatherPreset,
};
use inf_ecs::math::{Vec2d, Vec3d};
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

const STEPS: usize = 120;
const SKY_GUID: Uuid = Uuid::from_u128(0x20_0001);
const OCEAN_GUID: Uuid = Uuid::from_u128(0x20_0002);
const LAKE_GUID: Uuid = Uuid::from_u128(0x20_0003);
const RIVER_GUID: Uuid = Uuid::from_u128(0x20_0004);

/// One projected water body, as raw bits.
///
/// Bits rather than floats on purpose: a comparison that "passes within 1e-12" is
/// exactly the drift this gate is supposed to catch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WaterSample {
    kind: u32,
    level: u64,
    time_s: u64,
    flow: u64,
    /// Every derived Gerstner component: `(dir.x, dir.y, amplitude, wavenumber,
    /// omega, steepness, phase)`. This is the part that is *computed* rather than
    /// copied, and therefore the part a drift moves first.
    waves: Vec<[u64; 7]>,
    /// The river ribbon: how many frames, and the first and last frame's centre +
    /// arc length. A ribbon built from a different spline, a different
    /// interpolation or a different sample density fails here.
    frames: usize,
    ends: Vec<[u64; 4]>,
}

fn sample(sim: &RuntimeSim) -> Vec<WaterSample> {
    // Start from a scene carrying a deliberately WRONG water list, so each sample
    // proves the projection rebuilt it this frame rather than inheriting
    // something that happens to match.
    let mut scene = RenderScene {
        waters: vec![inf_render::RenderWater {
            level_m: -9_999.0,
            ..inf_render::RenderWater::default()
        }],
        ..Default::default()
    };
    let vmeshes = VmeshRegistry::default();
    project_scene(&mut scene, sim, 0.0, &vmeshes);
    scene
        .waters
        .iter()
        .map(|w| WaterSample {
            kind: w.kind.code(),
            level: w.level_m.to_bits(),
            time_s: w.time_s.to_bits(),
            flow: w.flow_speed_m_s.to_bits(),
            waves: w
                .waves
                .waves()
                .iter()
                .map(|c| {
                    [
                        c.dir.x.to_bits(),
                        c.dir.y.to_bits(),
                        c.amplitude_m.to_bits(),
                        c.wavenumber.to_bits(),
                        c.omega.to_bits(),
                        c.steepness.to_bits(),
                        c.phase.to_bits(),
                    ]
                })
                .collect(),
            frames: w.frames.len(),
            ends: [w.frames.first(), w.frames.last()]
                .into_iter()
                .flatten()
                .map(|f| {
                    [
                        f.center.x.to_bits(),
                        f.center.y.to_bits(),
                        f.center.z.to_bits(),
                        f.s.to_bits(),
                    ]
                })
                .collect(),
        })
        .collect()
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Vec<WaterSample>> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            sample(sim)
        })
        .collect()
}

/// The coastal scene: a clock running a storm, an ocean that follows the weather
/// wind, a lake that does **not**, and a spline river running downhill.
fn water_doc() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Water Projection Gate");
    doc.create_with_guid(SKY_GUID, SpawnKind::Empty, "Sky", None);
    doc.create_with_guid(OCEAN_GUID, SpawnKind::Empty, "Ocean", None);
    doc.create_with_guid(LAKE_GUID, SpawnKind::Empty, "Lake", None);
    doc.create_with_guid(RIVER_GUID, SpawnKind::Empty, "River", None);

    let sky = doc.world().entity_of(SKY_GUID).expect("sky entity");
    let ocean = doc.world().entity_of(OCEAN_GUID).unwrap();
    let lake = doc.world().entity_of(LAKE_GUID).unwrap();
    let river = doc.world().entity_of(RIVER_GUID).unwrap();
    let w = doc.world_mut().world_mut();

    w.entity_mut(sky).insert(TimeOfDay {
        seconds: 0.0,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
        // 600× — the clock really moves over 120 steps, so a wave phase that
        // ignored it would be obvious.
        rate: 600.0,
    });
    // Weather ON and blending toward a storm, so the wind the ocean responds to
    // is itself changing every step. A projection that read the *authored* wind
    // instead of the resolved one would be frozen and would fail (c).
    w.entity_mut(sky).insert(SkyAtmosphere {
        weather_enabled: true,
        weather_target: WeatherPreset::Storm,
        weather_blend_seconds: 4.0,
        weather_blend_remaining: 4.0,
        ..SkyAtmosphere::default()
    });

    w.entity_mut(ocean).insert(WaterBody {
        kind: WaterKind::Ocean,
        level_m: 2.0,
        wave_seed: 0xC0FFEE,
        ..WaterBody::default()
    });
    w.entity_mut(lake)
        .insert(WaterBody::lake(11.0, Vec2d::new(60.0, 40.0)));

    w.entity_mut(river).insert(WaterBody::river(9.0, 2.0, 2.5));
    w.entity_mut(river).insert(Spline {
        points: vec![
            Vec3d::new(0.0, 20.0, 0.0),
            Vec3d::new(60.0, 16.0, 10.0),
            Vec3d::new(120.0, 12.0, -10.0),
            Vec3d::new(180.0, 8.0, 0.0),
        ],
        closed: false,
        interp: SplineInterp::CatmullRom,
    });
    // A non-identity transform, so the ribbon exercises the world-space mapping
    // rather than an accidental identity.
    w.entity_mut(river).insert(Transform {
        translation: Vec3d::new(25.0, 1.0, -12.0),
        ..Transform::IDENTITY
    });

    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

fn cook_doc(tmp: &Path, doc: &SceneDoc) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Water Projection Gate", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    inf_editor_core::scene::serialize::save(doc, &content.join("Water.inf_lvl"), None)
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
        |_guid| None,
        |_guid| None,
        |_guid| None,
        |_guid| None,
        |_guid| None,
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// GATE (a): the same scripted ramp twice is bit-identical.
#[test]
fn the_water_trace_is_deterministic() {
    let doc = water_doc();
    let a = run_trace(&mut pie_sim(&doc));
    let b = run_trace(&mut pie_sim(&doc));
    assert_eq!(a, b, "the water projection is not deterministic");

    // Not vacuous: three bodies really projected, each with derived components…
    let first = a.first().expect("a trace");
    assert_eq!(first.len(), 3, "expected an ocean, a lake and a river");
    assert!(first.iter().all(|s| !s.waves.is_empty()));
    assert!(
        first.iter().any(|s| s.kind == 2 && s.frames >= 4),
        "the river projected no ribbon: {first:?}"
    );
    // …and the projection really REBUILT the list rather than inheriting the
    // deliberately-wrong seed scene.
    assert!(
        first.iter().all(|s| f64::from_bits(s.level) > -9_000.0),
        "the seed body survived the projection"
    );
}

/// GATE (c1): the sea moves with the **level clock**. Every water body's
/// `time_s` advances, and the trace's first and last steps differ — a frozen
/// projection would satisfy (a) trivially.
#[test]
fn the_sea_advances_with_the_level_clock() {
    let doc = water_doc();
    let trace = run_trace(&mut pie_sim(&doc));
    let first = trace.first().unwrap();
    let last = trace.last().unwrap();
    assert_ne!(first, last, "nothing about the water moved over 120 steps");
    for (a, b) in first.iter().zip(last) {
        assert_ne!(
            f64::from_bits(a.time_s),
            f64::from_bits(b.time_s),
            "a body's clock never advanced"
        );
    }
    // 120 steps x 1/60 s x 600 = 1200 sim-seconds of clock.
    let dt = f64::from_bits(last[0].time_s) - f64::from_bits(first[0].time_s);
    assert!((dt - 1_190.0).abs() < 20.0, "the clock advanced {dt} s");
}

/// GATE (c2): the **weather wind** reaches the waves — and only the bodies that
/// asked for it.
///
/// The ocean follows the level's wind, which is blending toward a storm, so its
/// derived components must change across the ramp. The lake opted out
/// (`WaterBody::lake` sets `wind_from_weather: false`, because a lake has no
/// fetch), so its components must be **bit-identical** from first step to last.
/// That pairing is what makes this a test of the *rule* rather than of "some
/// number changed".
#[test]
fn the_weather_wind_reaches_only_the_bodies_that_asked_for_it() {
    let doc = water_doc();
    let trace = run_trace(&mut pie_sim(&doc));
    let first = trace.first().unwrap();
    let last = trace.last().unwrap();

    let by_kind =
        |t: &Vec<WaterSample>, k: u32| t.iter().find(|s| s.kind == k).cloned().expect("a body");
    let ocean_a = by_kind(first, 0);
    let ocean_b = by_kind(last, 0);
    assert_ne!(
        ocean_a.waves, ocean_b.waves,
        "the storm never reached the ocean's waves"
    );

    let lake_a = by_kind(first, 1);
    let lake_b = by_kind(last, 1);
    assert_eq!(
        lake_a.waves, lake_b.waves,
        "the storm raised a swell on a lake — `wind_from_weather: false` is not \
         being honoured"
    );

    // A river's ripple travels downstream, so it is likewise unmoved by the wind.
    assert_eq!(by_kind(first, 2).waves, by_kind(last, 2).waves);
}

/// GATE (b): PIE payload round-trip == shipping, on the water trace.
///
/// This is the headline: the cooked pack and the PIE payload must project the
/// *same* water, wave for wave and frame for frame, at every step of a moving
/// clock under changing weather.
#[test]
fn pie_water_trace_matches_shipping() {
    let doc = water_doc();
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_doc(dir.path(), &doc);

    let ship = run_trace(&mut pack_sim(&pack));
    let pie = run_trace(&mut pie_sim(&doc));

    assert_eq!(
        pie.len(),
        ship.len(),
        "the two runs took different numbers of steps"
    );
    for (i, (a, b)) in pie.iter().zip(&ship).enumerate() {
        assert_eq!(
            a, b,
            "PIE water trace != shipping at step {i} (PIE == shipping for water)"
        );
    }

    // A guard on the guard: the trace has to have content. Both sides carried
    // three bodies with real derived components and a real ribbon.
    let last = ship.last().unwrap();
    assert_eq!(last.len(), 3);
    assert!(last.iter().all(|s| s.waves.len() >= 3));
    let river = last.iter().find(|s| s.kind == 2).unwrap();
    assert!(river.frames >= 4, "the shipped river lost its ribbon");
    assert_eq!(river.ends.len(), 2);
    assert_ne!(river.ends[0], river.ends[1], "the ribbon is degenerate");
}

/// **THE P19.1 FLOW MAP REACHES A RIVER'S FOAM** (P20.4).
///
/// The wiring is additive-only by construction — `inf_water::flow_foam_gain`
/// returns exactly `1.0` where there is no flow — so this is asserted as a pair:
/// a river over a terrain that was **never eroded** carries the exact identity in
/// every frame, and the *same* river over a terrain carrying a flow map carries
/// more. Without the first half the second would not prove the coupling is
/// additive; without the second the first would be satisfied by wiring nothing at
/// all.
///
/// Pinned here rather than in a golden because it is a claim about a *number the
/// projector computes*, not about pixels: the golden that showed it would differ
/// from `water_river` by a foam intensity, while this differs from a broken
/// implementation by `1.0` vs `1.6`.
#[test]
fn the_flow_map_modulates_a_rivers_foam_and_nothing_else() {
    use inf_ecs::components::Terrain;

    let flow_gains = |flow: f32| -> Vec<u64> {
        let mut doc = water_doc();
        // A terrain under the river, optionally carrying a flow map. 4 m spacing
        // over 129 samples = 512 m per tile; four tiles cover the river's whole
        // run, which straddles z = 0 once the river's transform is applied.
        const RES: u32 = 129;
        const TILES: [(i32, i32); 4] = [(0, -1), (0, 0), (-1, -1), (-1, 0)];
        let terrain_guid = Uuid::from_u128(0x20_0005);
        doc.create_with_guid(terrain_guid, SpawnKind::Empty, "Ground", None);
        {
            let e = doc.world().entity_of(terrain_guid).unwrap();
            let mut t = Terrain {
                meters_per_sample: 4.0,
                tile_resolution: RES,
                data: inf_terrain::TerrainData::new(RES, 4.0),
                ..Terrain::default()
            };
            for key in TILES {
                t.data.author_tile(key, |_, _| 0.0);
                if flow != 0.0 {
                    let tile = t.data.get_tile_mut(key).unwrap();
                    for j in 0..RES {
                        for i in 0..RES {
                            tile.set_map_texel(RES, i, j, [flow, 0.0, 0.0]);
                        }
                    }
                }
            }
            doc.world_mut().world_mut().entity_mut(e).insert(t);
        }
        doc.world_mut().mark_dirty();
        doc.world_mut().propagate();

        let mut sim = pie_sim(&doc);
        sim.step_once(RuntimeInput::default());
        let mut scene = RenderScene::default();
        project_scene(&mut scene, &sim, 0.0, &VmeshRegistry::default());
        let river = scene
            .waters
            .iter()
            .find(|w| w.kind == inf_render::WaterKindGpu::River)
            .expect("the fixture's river projected");
        assert!(river.frames.len() > 8, "the ribbon is degenerate");
        river.frames.iter().map(|f| f.flow_gain.to_bits()).collect()
    };

    // Never eroded ⇒ the EXACT identity, frame for frame. Not "close to 1".
    let dry = flow_gains(0.0);
    assert!(
        dry.iter().all(|&b| b == 1.0f64.to_bits()),
        "an unmapped terrain moved a river's foam"
    );

    // A fully-channelled terrain ⇒ the saturated gain, everywhere the river
    // crosses it.
    let wet = flow_gains(inf_water::FLOW_FOAM_REFERENCE_M3 as f32);
    assert_eq!(wet.len(), dry.len());
    let want = inf_water::flow_foam_gain(inf_water::FLOW_FOAM_REFERENCE_M3);
    assert!(want > 1.0, "the curve itself is a no-op");
    let boosted = wet.iter().filter(|&&b| b == want.to_bits()).count();
    assert!(
        boosted * 2 > wet.len(),
        "only {boosted} of {} frames took the flow boost",
        wet.len()
    );
    // …and the two runs really differ, which is the mutation check on the whole
    // wiring: a projector that ignored the map would produce identical vectors.
    assert_ne!(dry, wet);
}

/// **THE VISIBILITY LAW** (P20.4), pinned from the render side.
///
/// Hiding a water body removes it from `RenderScene::waters` — no surface, no
/// underwater fog, no wetness band — and changes the simulation by nothing. The
/// sim half is `crates/inf-physics/tests/water_visibility_3d.rs`; the decision
/// and its evidence live on `RenderWater::surface()`.
///
/// The two halves are deliberately in different crates because they are two
/// different claims about the same law, and a single file asserting both would
/// let a future change satisfy one by breaking the other.
#[test]
fn a_hidden_water_body_is_not_drawn_but_is_still_simulated() {
    let mut doc = water_doc();
    let mut sim = pie_sim(&doc);
    sim.step_once(RuntimeInput::default());
    let all = sample(&sim);
    assert_eq!(all.len(), 3, "the fixture must carry ocean + lake + river");

    // Hide the LAKE only.
    doc.edit_set_visible(LAKE_GUID, false);
    let mut hidden_sim = pie_sim(&doc);
    hidden_sim.step_once(RuntimeInput::default());
    let drawn = sample(&hidden_sim);
    assert_eq!(
        drawn.len(),
        2,
        "a hidden body still reached the renderer: {drawn:?}"
    );
    // The two that remain are exactly the two that were not hidden — the lake's
    // kind is Lake (code 1), so its absence is checkable rather than assumed.
    assert!(
        !drawn.iter().any(|w| w.kind == 1),
        "the hidden lake is still in the frame: {drawn:?}"
    );
    assert!(drawn.iter().any(|w| w.kind == 0), "the ocean vanished too");
    assert!(drawn.iter().any(|w| w.kind == 2), "the river vanished too");

    // …and the SIM still has it. `water.surface_height` over the lake's own
    // footprint answers the same number either way — the poll the Blueprint node
    // makes, so this is the user-visible half of the law rather than a private
    // one.
    let shown_h = sim
        .bridge3d()
        .water_surface_height(0.0, 0.0)
        .expect("the fixture's lake covers the origin");
    let hidden_h = hidden_sim
        .bridge3d()
        .water_surface_height(0.0, 0.0)
        .expect("a hidden lake is still water to the simulation");
    assert_eq!(
        shown_h.to_bits(),
        hidden_h.to_bits(),
        "hiding the lake moved its surface for the simulation"
    );
    // ANTI-VACUITY: the answer is the LAKE's (level 11), not the unbounded
    // ocean's (level 2) which covers the origin too — so the equality above is a
    // claim about the hidden body rather than about the one beneath it.
    assert!(
        shown_h > 10.0,
        "the probe answered the ocean, not the lake ({shown_h})"
    );
}

/// The cook is **silent** on this level: its river runs downhill, so the P20.1
/// advisory must not fire. An advisory that fires on correct content is one
/// nobody reads.
#[test]
fn the_downhill_river_draws_no_advisory() {
    let doc = water_doc();
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Water Projection Gate", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    inf_editor_core::scene::serialize::save(&doc, &content.join("Water.inf_lvl"), None).unwrap();
    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).unwrap();
    assert!(
        !report.warnings.iter().any(|w| w.contains("climbs")),
        "the downhill river was reported: {:?}",
        report.warnings
    );
}
