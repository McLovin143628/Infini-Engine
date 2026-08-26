//! **Auto exposure on the GPU** (wave VIS1b) — the arms that need a device.
//!
//! The rule's arithmetic is unit-tested without one (`inf_render::settings`);
//! what needs a device is that the histogram *sees the frame*, that the
//! adaptation is a function of the level clock and of nothing else, that manual
//! mode still renders the bytes it always did, and that the bloom threshold now
//! keys on the exposed image rather than on the raw radiance.
//!
//! Every arm here renders **several frames on one renderer**, because an
//! adaptation that has no history is a snap: the state buffer is what carries the
//! previous frame's EV forward, and a fresh `EngineRenderer` has none. That is
//! the shape `taa_multiframe_stable` already has.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    BloomSettings, EngineRenderer, ExposureMode, ExposureSettings, ExposureState, GpuContext,
    HeadlessTarget, MeshInstance, RenderScene, RenderSettings, RenderView, HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP exposure: no GPU adapter ({e})");
            None
        }
    }
}

fn view() -> RenderView {
    let eye = DVec3::new(0.0, 1.5, 7.0);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::new(0.0, 0.5, 0.0) - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// One emissive slab at `radiance`, filling the whole frame — the simplest scene
/// whose average luminance is a number the arm chooses.
///
/// Emissive rather than lit, deliberately: a lit surface's luminance is a
/// function of the sun, the ambient, the BRDF and the tone of the albedo, and an
/// arm that has to reason about all four in order to say what the histogram
/// should have measured is an arm that will be wrong quietly.
///
/// **Filling the frame is load-bearing, not cosmetic.** The first draft of this
/// fixture was a 3×3 wall of small cubes, and a *tenfold* change in their
/// radiance moved the measured average by 1.22× — because the sky behind them
/// held eight ninths of the log-average's weight. The arm was measuring the
/// backdrop.
fn emissive_wall(radiance: f32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    let mut m = MeshInstance::lit(
        DVec3::new(0.0, 1.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(40.0, 30.0, 0.5),
        [0.0, 0.0, 0.0, 1.0],
        1,
    );
    m.emissive = [radiance, radiance, radiance];
    scene.instances.push(m);
    scene.mark_dirty();
    scene
}

/// Set the level clock the exposure node steps by. It is `cloud_time_s` — the
/// document's own clock — which the renderer already carries for the wind, the
/// waves and the rain.
fn set_clock(scene: &mut RenderScene, t: f64) {
    scene.atmosphere.clouds.time_s = t;
}

fn auto(speed: f32) -> RenderSettings {
    RenderSettings {
        exposure_control: ExposureSettings {
            mode: ExposureMode::Auto,
            adaptation_speed: speed,
            ..ExposureSettings::default()
        },
        ..RenderSettings::default()
    }
}

struct Rig {
    target: HeadlessTarget,
    renderer: EngineRenderer,
}

impl Rig {
    fn new(gpu: &GpuContext, settings: RenderSettings) -> Self {
        let target = HeadlessTarget::new(gpu, W, H);
        let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
        renderer.set_settings(settings);
        Self { target, renderer }
    }

    fn frame(&mut self, gpu: &GpuContext, scene: &RenderScene) -> ExposureState {
        self.renderer
            .render(gpu, scene, &view(), &self.target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        self.renderer.read_exposure(gpu).expect("exposure readback")
    }

    fn pixels(&self, gpu: &GpuContext) -> Vec<u8> {
        self.target.read_rgba(gpu).expect("readback")
    }
}

/// Render `steps` frames advancing the level clock by `dt` each, and return the
/// exposure trace.
fn trace(
    gpu: &GpuContext,
    settings: RenderSettings,
    scene: &mut RenderScene,
    dt: f64,
    steps: usize,
) -> Vec<ExposureState> {
    let mut rig = Rig::new(gpu, settings);
    (0..steps)
        .map(|i| {
            set_clock(scene, i as f64 * dt);
            rig.frame(gpu, scene)
        })
        .collect()
}

fn mean_luma(rgba: &[u8]) -> f64 {
    let n = (rgba.len() / 4) as f64;
    rgba.chunks(4)
        .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
        .sum::<f64>()
        / n
}

/// **Manual mode is the off path, and it is byte-identical.**
///
/// The claim the whole wave rests on: a level that never touches the exposure
/// block renders the pixels it rendered at `278fb3c6`. It cannot be proved
/// against the old binary from inside this tree, so it is proved the way the
/// engine proves every opt-in: the manual frame equals a frame rendered with the
/// feature's own inputs at their identities, and the exposure state the two
/// readers see is the authored scalar **bit for bit**.
#[test]
fn manual_exposure_records_nothing_and_multiplies_by_the_authored_scalar() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = emissive_wall(2.0);
    set_clock(&mut scene, 100.0);

    for e in [1.0f32, 0.4, 2.5] {
        let settings = RenderSettings {
            exposure: e,
            ..RenderSettings::default()
        };
        let mut rig = Rig::new(&gpu, settings);
        // Three frames: manual writes on the first and records nothing after it,
        // so a stale buffer would show up as a moving trace.
        let a = rig.frame(&gpu, &scene);
        let b = rig.frame(&gpu, &scene);
        let c = rig.frame(&gpu, &scene);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(
            a.multiplier.to_bits(),
            e.to_bits(),
            "manual exposure {e} reached the shaders as {}",
            a.multiplier
        );
        assert_eq!(a.valid, 0.0, "manual mode must not claim an adapted EV");
        assert_eq!(
            a.avg_luminance, 0.0,
            "manual mode must not have built a histogram"
        );
    }

    // And the frame really is the frame: manual at 1.0 against the same scene
    // through a renderer that never had the exposure block touched.
    let mut plain = Rig::new(&gpu, RenderSettings::default());
    plain.frame(&gpu, &scene);
    let want = plain.pixels(&gpu);
    let mut manual = Rig::new(
        &gpu,
        RenderSettings {
            exposure: 1.0,
            exposure_control: ExposureSettings::default(),
            ..RenderSettings::default()
        },
    );
    manual.frame(&gpu, &scene);
    assert_eq!(want, manual.pixels(&gpu), "manual mode moved a byte");
}

/// **Auto exposure measures the frame it is given.**
///
/// A wall ten times brighter must be measured as roughly ten times brighter and
/// exposed roughly ten times less — which is the one claim that a histogram
/// wired to the wrong texture, or reduced in the wrong space, cannot satisfy by
/// accident.
#[test]
fn the_histogram_measures_the_scene_and_the_exposure_follows_it() {
    let Some(gpu) = gpu_or_skip() else { return };

    let mut dim = emissive_wall(0.3);
    let mut bright = emissive_wall(3.0);
    // One frame each: with no history the resolve SNAPS to the target, which is
    // what makes a single frame a measurement of the scene rather than of the
    // path taken to it.
    let a = trace(&gpu, auto(1.5), &mut dim, 0.0, 1)[0];
    let b = trace(&gpu, auto(1.5), &mut bright, 0.0, 1)[0];

    assert!(a.valid > 0.5 && b.valid > 0.5, "auto mode must adapt");
    assert!(
        a.avg_luminance > 0.0 && b.avg_luminance > 0.0,
        "the histogram measured nothing: {a:?} / {b:?}"
    );
    let ratio = b.avg_luminance / a.avg_luminance;
    assert!(
        (5.0..20.0).contains(&ratio),
        "a ten-times-brighter wall measured {ratio}x brighter ({a:?} vs {b:?})"
    );
    assert!(
        b.multiplier < a.multiplier,
        "the brighter scene must be exposed LESS: {} vs {}",
        b.multiplier,
        a.multiplier
    );
    // And the two frames land near each other on screen, which is the point of
    // an auto exposure at all.
    let mut rig_a = Rig::new(&gpu, auto(1.5));
    rig_a.frame(&gpu, &dim);
    let mut rig_b = Rig::new(&gpu, auto(1.5));
    rig_b.frame(&gpu, &bright);
    let (la, lb) = (
        mean_luma(&rig_a.pixels(&gpu)),
        mean_luma(&rig_b.pixels(&gpu)),
    );
    let auto_spread = (lb / la.max(1e-6)).max(la / lb.max(1e-6));

    let fixed_a = {
        let mut r = Rig::new(&gpu, RenderSettings::default());
        r.frame(&gpu, &dim);
        mean_luma(&r.pixels(&gpu))
    };
    let fixed_b = {
        let mut r = Rig::new(&gpu, RenderSettings::default());
        r.frame(&gpu, &bright);
        mean_luma(&r.pixels(&gpu))
    };
    let fixed_spread = (fixed_b / fixed_a.max(1e-6)).max(fixed_a / fixed_b.max(1e-6));
    assert!(
        auto_spread < fixed_spread,
        "auto exposure did not close the gap: auto {auto_spread:.3}x vs manual {fixed_spread:.3}x"
    );
    eprintln!(
        "exposure: dim avg {:.4} -> x{:.3}; bright avg {:.4} -> x{:.3}; \
         screen spread manual {fixed_spread:.3}x -> auto {auto_spread:.3}x",
        a.avg_luminance, a.multiplier, b.avg_luminance, b.multiplier
    );
}

/// **The adaptation is stepped by the level clock, and by nothing else.**
///
/// Three claims in one arm, because they are one claim seen three ways:
///
/// * a frozen clock freezes the adaptation, however many frames run — which is
///   what "a paused sim is a frozen eye" means at the buffer;
/// * the same clock interval crossed in four steps and in twenty lands on the
///   same exposure, so a level does not look different on a faster machine;
/// * running long enough arrives at the target rather than approaching it, which
///   is the property a linear ramp in stops has and an exponential decay does
///   not.
#[test]
fn the_adaptation_follows_the_level_clock_and_never_the_frame_count() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = emissive_wall(4.0);

    // A frozen clock. Frame 0 snaps (no history); every frame after it must hold.
    let frozen = trace(&gpu, auto(0.5), &mut scene, 0.0, 6);
    for (i, s) in frozen.iter().enumerate().skip(1) {
        assert_eq!(
            s.ev, frozen[1].ev,
            "frame {i} moved the exposure on a frozen clock: {s:?}"
        );
    }

    // The same clock interval, two frame rates. Both start from a snap on frame
    // 0, so both are adapting from the same EV over the same 1.0 s.
    let coarse = trace(&gpu, auto(0.5), &mut scene, 0.25, 5);
    let fine = trace(&gpu, auto(0.5), &mut scene, 0.05, 21);
    let (a, b) = (coarse.last().unwrap().ev, fine.last().unwrap().ev);
    assert!(
        (a - b).abs() < 1e-4,
        "4 x 0.25 s gave EV {a}, 20 x 0.05 s gave EV {b}"
    );

    // And the same trace twice on two renderers is the same trace.
    let again = trace(&gpu, auto(0.5), &mut scene, 0.25, 5);
    let l: Vec<f32> = coarse.iter().map(|s| s.ev).collect();
    let r: Vec<f32> = again.iter().map(|s| s.ev).collect();
    assert_eq!(l, r, "two runs of one clock disagreed");
}

/// **The eye still works on a level whose clock never runs** (VIS1b audit).
///
/// `dt` is a `cloud_time_s` delta, `cloud_time_s` is a pure function of
/// `TimeOfDay`, and **`TimeOfDay::rate` defaults to `0.0`** — so on most levels
/// the delta is zero on every frame after the first. Read as "a frozen clock is a
/// frozen eye" that made auto exposure adapt once, to whatever the first frame
/// happened to look like, and then hold that for ever: a player walking out of a
/// lit courtyard into a cellar saw **no** adaptation at all.
///
/// The rule now has two halves, and this arm is both of them:
///
/// * a clock that has **never** moved has no rate for a ramp to be expressed in,
///   so the eye tracks — every frame snaps to its own target;
/// * a clock that **has** moved is a running one, and a zero-delta frame after
///   that is a paused world or a second render of one simulation step, both of
///   which must hold. That half is what `exposure_pie.rs`'s three-renders-per-step
///   arm rests on, so it is pinned here rather than left to it.
#[test]
fn the_eye_adapts_on_a_level_whose_clock_never_runs() {
    let Some(gpu) = gpu_or_skip() else { return };

    // ONE scene, mutated in place — the instance upload is version-gated, so two
    // separately-built scenes both at version 1 would hand the renderer the first
    // one twice and the arm would certify a frozen eye as a working one. (It did,
    // on the first run: the histogram read the same 0.2975 both times.)
    let mut scene = emissive_wall(0.3);
    let brighten = |s: &mut RenderScene, r: f32| {
        s.instances[0].emissive = [r; 3];
        s.mark_dirty();
    };
    // The clock is never touched: this is a level at the default `rate == 0`.
    let mut rig = Rig::new(&gpu, auto(1.5));

    let a = rig.frame(&gpu, &scene);
    let b = rig.frame(&gpu, &scene);
    assert!(
        a.avg_luminance > 0.0,
        "the histogram measured nothing: {a:?}"
    );
    assert_eq!(
        a.multiplier.to_bits(),
        b.multiplier.to_bits(),
        "the same frame twice must expose the same: {a:?} vs {b:?}"
    );

    // Ten times brighter with the clock still at zero. Before the audit this
    // returned `a.multiplier` unchanged, for ever.
    brighten(&mut scene, 3.0);
    let c = rig.frame(&gpu, &scene);
    eprintln!(
        "static clock: dim x{:.4} (avg {:.4}) -> bright x{:.4} (avg {:.4})",
        a.multiplier, a.avg_luminance, c.multiplier, c.avg_luminance
    );
    assert!(
        c.avg_luminance > a.avg_luminance * 2.0,
        "the fixture did not actually get brighter ({} -> {}), so the assertion \
         below would pass for the wrong reason",
        a.avg_luminance,
        c.avg_luminance
    );
    assert!(
        c.multiplier < a.multiplier * 0.8,
        "the eye is frozen on a level whose clock never runs: {} then {}",
        a.multiplier,
        c.multiplier
    );
    // …and it comes back, which says it is tracking rather than drifting.
    brighten(&mut scene, 0.3);
    let d = rig.frame(&gpu, &scene);
    assert_eq!(
        d.multiplier.to_bits(),
        a.multiplier.to_bits(),
        "the tracked exposure is not a function of the frame alone: {a:?} vs {d:?}"
    );

    // THE OTHER HALF. Once the clock has moved, a zero-delta frame HOLDS — that
    // is a paused world, or a second render of one simulation step.
    let mut scene = emissive_wall(0.3);
    let mut rig = Rig::new(&gpu, auto(0.5));
    set_clock(&mut scene, 0.0);
    rig.frame(&gpu, &scene);
    set_clock(&mut scene, 0.2);
    let moved = rig.frame(&gpu, &scene);
    // The scene changes and the clock does not: the eye must not follow.
    brighten(&mut scene, 3.0);
    let held = rig.frame(&gpu, &scene);
    assert!(
        held.avg_luminance > moved.avg_luminance * 2.0,
        "the second half's fixture did not change either ({} -> {})",
        moved.avg_luminance,
        held.avg_luminance
    );
    assert_eq!(
        held.ev.to_bits(),
        moved.ev.to_bits(),
        "a zero-delta frame on a RUNNING clock must hold, not snap: {moved:?} then {held:?}"
    );
}

/// **The meter reads the frame before the lens writes to it** (VIS1b audit).
///
/// The histogram taps `post_hdr`, and the flare adds light to the frame. If the
/// two were the other way round the eye would meter its own glare — a brighter
/// frame would stop down, which dims the glare, which opens the eye — and the
/// exposure would be a feedback loop rather than a measurement. It is not, and
/// this is the assertion rather than the sentence.
///
/// The same ordering carries the clause-1 decision: the bloom prefilter and the
/// tonemap both read the exposure this node writes, so it has to precede both.
///
/// **And, while a renderer is standing here, the timestamp budget.**
/// `timing::FRAME_MARKS_NEEDED` is the graph's node count written down by hand,
/// and it had drifted three short in both directions across two waves — 29 and
/// then 31 against a graph of 32 and then 34. `mark` drops silently past
/// `MAX_FRAME_MARKS`, so the constant that exists to make that impossible was
/// itself the thing nobody was checking. This is the comparison that cannot go
/// stale: the graph's own length, plus the five out-of-graph segments and the
/// origin mark, against the query set's size.
#[test]
fn the_meter_reads_the_frame_before_the_lens_writes_to_it() {
    let Some(gpu) = gpu_or_skip() else { return };
    let renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let names = renderer.pass_names();
    let at = |n: &str| {
        names
            .iter()
            .position(|p| *p == n)
            .unwrap_or_else(|| panic!("no `{n}` node in the graph: {names:?}"))
    };
    assert!(
        at("exposure") < at("bloom"),
        "the exposure must be measured before the bloom prefilter thresholds against it"
    );
    assert!(
        at("bloom") < at("flare"),
        "the flare gathers the frame's bright part and belongs after the bloom"
    );
    assert!(
        at("flare") < at("tonemap"),
        "the tonemap adds the flare, so the flare must have been drawn"
    );
    assert!(
        at("exposure") < at("flare"),
        "the eye would be metering its own glare"
    );

    // Five out-of-graph segments (`vt-stream`, `vsm-sync`, `vsm-raster`,
    // `vt-feedback`, `vsm-mark`) plus the frame's origin mark.
    const OUT_OF_GRAPH: usize = 5;
    const ORIGIN: usize = 1;
    let needed = names.len() + OUT_OF_GRAPH + ORIGIN;
    eprintln!(
        "frame marks: {} graph nodes + {OUT_OF_GRAPH} + {ORIGIN} = {needed} of \
         {} query slots",
        names.len(),
        inf_render::MAX_FRAME_MARKS
    );
    assert!(
        needed <= inf_render::MAX_FRAME_MARKS as usize,
        "the frame writes {needed} timestamps into {} slots — `FrameTimer::mark` \
         drops the tail silently, so the per-pass report would stop naming the \
         last passes rather than fail",
        inf_render::MAX_FRAME_MARKS
    );
}

/// **The bloom threshold is exposure-relative — the ordering decision, priced.**
///
/// Before this wave the tonemap did `(hdr + bloom) * exposure`, so the prefilter
/// keyed on raw radiance and the threshold meant a different thing at every
/// exposure. With auto exposure that is a real defect rather than a stylistic
/// one: a dim scene the eye has opened four stops has *nothing* over a linear
/// threshold of 1.0, so it cannot bloom at all, while the same scene four stops
/// brighter blooms off everything.
///
/// The measurement is the ratio of "how much energy did bloom add" between a dim
/// scene and a bright one, both under auto exposure. Exposure-relative keeps that
/// ratio near 1; the old order does not, and the arm prices both orders by
/// driving the shipped one with a manual exposure that reproduces them.
#[test]
fn the_bloom_threshold_is_exposure_relative() {
    let Some(gpu) = gpu_or_skip() else { return };
    let bloom = BloomSettings {
        enabled: true,
        threshold: 1.0,
        knee: 0.6,
        intensity: 0.5,
        karis: false,
    };

    // The fixture is a dim backdrop with a small bright patch on it — a scene
    // with a *highlight*, which is the only kind of scene a threshold is a
    // question about. A uniform wall cannot answer it: auto exposure puts the
    // whole thing at middle grey and nothing is above any threshold at all.
    // Scaling `radiance` scales both together, so the scene's *structure* — the
    // patch is twenty times the backdrop — is what stays fixed between the two
    // measurements.
    let scene_at = |radiance: f32| -> RenderScene {
        let mut scene = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        let mut back = MeshInstance::lit(
            DVec3::new(0.0, 1.5, 0.0),
            Quat::IDENTITY,
            Vec3::new(40.0, 30.0, 0.5),
            [0.0, 0.0, 0.0, 1.0],
            1,
        );
        back.emissive = [radiance; 3];
        scene.instances.push(back);
        let mut patch = MeshInstance::lit(
            DVec3::new(0.0, 1.0, 3.0),
            Quat::IDENTITY,
            Vec3::new(0.5, 0.5, 0.2),
            [0.0, 0.0, 0.0, 1.0],
            2,
        );
        patch.emissive = [radiance * 20.0; 3];
        scene.instances.push(patch);
        scene.mark_dirty();
        scene
    };

    // `gain(radiance, settings)` — the energy bloom adds to the frame.
    let gain = |radiance: f32, settings: RenderSettings| -> f64 {
        let mut scene = scene_at(radiance);
        set_clock(&mut scene, 0.0);
        let on = {
            let mut r = Rig::new(&gpu, RenderSettings { bloom, ..settings });
            r.frame(&gpu, &scene);
            mean_luma(&r.pixels(&gpu))
        };
        let off = {
            let mut r = Rig::new(&gpu, settings);
            r.frame(&gpu, &scene);
            mean_luma(&r.pixels(&gpu))
        };
        on - off
    };

    // The shipped order: auto exposure, exposure-relative threshold.
    let dim = gain(0.35, auto(1.5));
    let bright = gain(3.5, auto(1.5));
    assert!(
        dim > 0.5,
        "a dim scene under auto exposure must still bloom, added {dim:.3}"
    );
    let relative = (bright / dim).max(dim / bright);

    // The old order, reproduced: a FIXED exposure of 1.0 is exactly what
    // "threshold keys on raw radiance" means, because then the two orders are
    // the same expression. The dim scene's raw radiance never crosses 1.0.
    let dim_fixed = gain(0.35, RenderSettings::default());
    let bright_fixed = gain(3.5, RenderSettings::default());
    let absolute = (bright_fixed / dim_fixed.max(1e-6)).max(dim_fixed.max(1e-6) / bright_fixed);

    eprintln!(
        "bloom gain: relative dim {dim:.3} / bright {bright:.3} = {relative:.2}x spread; \
         absolute dim {dim_fixed:.3} / bright {bright_fixed:.3} = {absolute:.2}x spread"
    );
    // Measured on an RTX 4070 Ti at 320x180: **1.02x** with the shipped order
    // against **2.02x** with the old one. The thresholds sit between those two
    // numbers rather than beside one of them, so reverting the ordering fails
    // the first assert and a scene that stopped blooming at all fails the
    // `dim > 0.5` above.
    assert!(
        absolute > 1.8,
        "the fixture no longer separates the two orders: absolute spread {absolute:.2}x"
    );
    assert!(
        relative < 1.5,
        "the bloom threshold is not exposure-relative: {relative:.2}x spread \
         against the absolute order's {absolute:.2}x"
    );
}

/// **The Karis first downsample kills a firefly** (clause 4), and is measured
/// against the thing it exists to kill rather than against a screenshot.
///
/// One tiny very bright cube in an otherwise dim frame is what a specular
/// highlight on a wet leaf is at a distance: a sub-pixel sample that survives
/// every box downsample at full weight and crawls. Weighting each tap by
/// `1/(1+luma)` before averaging is Karis' fix; what it costs is genuine
/// highlight energy, which is why it is opt-in and why the arm measures the cost
/// as well as the benefit.
#[test]
fn the_karis_downsample_costs_a_firefly_more_than_it_costs_a_highlight() {
    let Some(gpu) = gpu_or_skip() else { return };
    let bloom = |karis: bool| RenderSettings {
        bloom: BloomSettings {
            enabled: true,
            threshold: 1.0,
            knee: 0.6,
            intensity: 0.5,
            karis,
        },
        ..RenderSettings::default()
    };

    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // The broad highlight: a wall at a modest radiance.
    let mut wall = MeshInstance::lit(
        DVec3::new(-2.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(1.2, 1.2, 0.2),
        [0.0, 0.0, 0.0, 1.0],
        1,
    );
    wall.emissive = [4.0, 4.0, 4.0];
    scene.instances.push(wall);
    scene.mark_dirty();

    let broad = |s: RenderSettings| {
        let mut r = Rig::new(&gpu, s);
        r.frame(&gpu, &scene);
        mean_luma(&r.pixels(&gpu))
    };
    let highlight_plain = broad(bloom(false));
    let highlight_karis = broad(bloom(true));

    // The firefly: a very small, very bright cube beside it.
    let mut fly = MeshInstance::lit(
        DVec3::new(2.2, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::splat(0.03),
        [0.0, 0.0, 0.0, 1.0],
        2,
    );
    fly.emissive = [900.0, 900.0, 900.0];
    scene.instances.push(fly);
    scene.mark_dirty();

    let with_fly = |s: RenderSettings| {
        let mut r = Rig::new(&gpu, s);
        r.frame(&gpu, &scene);
        mean_luma(&r.pixels(&gpu))
    };
    let fly_plain = with_fly(bloom(false)) - highlight_plain;
    let fly_karis = with_fly(bloom(true)) - highlight_karis;

    // The cost, and the benefit, side by side.
    let highlight_cost = 1.0 - highlight_karis / highlight_plain.max(1e-9);
    let firefly_cut = 1.0 - fly_karis / fly_plain.max(1e-9);
    eprintln!(
        "karis: firefly bloom {fly_plain:.4} -> {fly_karis:.4} (cut {:.1}%); \
         broad highlight {highlight_plain:.4} -> {highlight_karis:.4} (cost {:.1}%)",
        firefly_cut * 100.0,
        highlight_cost * 100.0
    );
    assert!(
        fly_plain > 0.0,
        "the fixture's firefly did not bloom at all ({fly_plain})"
    );
    assert!(
        firefly_cut > 2.0 * highlight_cost.max(0.0),
        "Karis took as much from the highlight as from the firefly: \
         cut {firefly_cut:.4} vs cost {highlight_cost:.4}"
    );
}
