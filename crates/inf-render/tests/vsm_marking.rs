//! **The P27.1 shadow-page marking pass**, on a real device — the depth-buffer
//! consumer, the residency it feeds, and the mirror it publishes through.
//!
//! What these arms are built to falsify:
//!
//! * a marking pass that marks **nothing** (an empty mask allocates no page, and
//!   every other assertion in the phase still passes — which is exactly the
//!   shape of a feature that does nothing);
//! * a pass that marks the **wrong level** or the wrong page (the residency
//!   would then hold pages no receiver will ever read, and P27.4 would render
//!   shadows nowhere);
//! * a pass that ignores the depth buffer entirely (a sky-only frame would mark
//!   the whole clipmap, which is the failure that looks like success);
//! * a mirror that agrees with the residency because both were written from the
//!   same variable (**mirrored ≠ measured**, the P26.5 law) — the table is read
//!   back off the GPU and compared against an independent walk.
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.
//!
//! # The fixture, and why its numbers are what they are
//!
//! One cube, head-on, on an empty sky — so every non-sky pixel belongs to one
//! flat face at a known world position, and the CPU twin can say exactly which
//! pages that face covers. The clipmap is deliberately configured so the level
//! the pixel footprint justifies is **2 of 6**: at an end of the tree the arm
//! could not tell a rule from a clamp.

use std::collections::BTreeSet;

use inf_render::{
    mark_page_for, GpuContext, RenderScene, RenderView, VsmLightHandle, VsmPage, VsmSettings,
};

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

const FW: u32 = 320;
const FH: u32 = 180;
/// The cube's side, in metres. Its front face is a `SIDE × SIDE` square at
/// `z = SIDE / 2`.
const SIDE: f32 = 2.0;
/// Where the cube stands, in the ground plane.
///
/// **Off-centre in y and centred in x, and neither is decoration.** Centred in
/// x, the face straddles a page boundary, so the marked set is more than one
/// page. Off-centre in y, the marked set is *asymmetric about the clipmap
/// centre* — which is what makes the page grid's vertical flip observable at
/// all. Measured: with the cube at the origin, deleting the flip from
/// `vsm_mark.wgsl` maps the marked set onto itself and every arm in this file
/// stays green.
const CUBE_XY: (f32, f32) = (0.0, 1.3);

/// The clipmap the arms run against: six levels of an 8×8 page grid over ±6 m.
///
/// 6 m at 8 pages of 128 texels is an 11.7 mm level-0 texel; the fixture's pixel
/// footprint at 5 m is 32.1 mm, so the justified level is `ceil(log2(2.74)) = 2`
/// — neither 0 nor 5, which is what lets the level arm below tell a rule from a
/// clamp.
fn settings() -> VsmSettings {
    VsmSettings {
        enabled: true,
        clipmap_pages_per_side: 8,
        clipmap_levels: 6,
        first_level_extent_m: 6.0,
        ..Default::default()
    }
}

fn view(eye_z: f64) -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 0.0, eye_z),
        forward: glam::Vec3::NEG_Z,
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// One cube at the origin under `lights` directional shadow casters, on an empty
/// sky (no grid: the infinite grid writes depth everywhere and would make every
/// arm here a statement about the grid).
fn scene(lights: &[glam::Vec3]) -> RenderScene {
    let mut scene = RenderScene {
        // No grid: the infinite grid writes depth at every pixel, which would
        // make every arm in this file a statement about the grid.
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(CUBE_XY.0 as f64, CUBE_XY.1 as f64, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(SIDE),
        [1.0, 1.0, 1.0, 1.0],
        1,
    ));
    for d in lights {
        scene.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: *d,
            cast_shadows: true,
            ..Default::default()
        });
    }
    scene.mark_dirty();
    scene
}

/// Render `frames` frames and hand back the renderer, so an arm can assert on
/// the WORLD (the resident page set) rather than on a report.
fn run(
    gpu: &GpuContext,
    scene: &RenderScene,
    v: &RenderView,
    set: &VsmSettings,
    frames: u64,
) -> inf_render::EngineRenderer {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    s.vsm = *set;
    renderer.set_settings(s);
    for _ in 0..frames {
        renderer.render(gpu, scene, v, &target.view, (FW, FH));
        // A host's frame pacing; here, the poll that lets the ring's maps fire.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    renderer
}

/// Every page of `light` that is resident — the WORLD, read out of the live
/// residency rather than out of a counter.
fn resident_pages(renderer: &inf_render::EngineRenderer, light: u32) -> BTreeSet<VsmPage> {
    let sys = renderer.vsm().expect("a live vsm system");
    let handle = VsmLightHandle(light);
    let desc = sys.residency().desc(handle).expect("registered").clone();
    let mut out = BTreeSet::new();
    for face in 0..desc.faces() {
        for level in 0..desc.level_count() {
            let g = desc.levels[level as usize];
            for y in 0..g.pages_y {
                for x in 0..g.pages_x {
                    let at = VsmPage::new(face, level, x, y);
                    if sys.residency().is_resident(handle, at) {
                        out.insert(at);
                    }
                }
            }
        }
    }
    out
}

/// The pages the cube's front face covers, according to the **CPU twin**, over a
/// dense sampling of the face dilated by one pixel's world size.
///
/// Dilated deliberately: the rasterizer covers a pixel whose centre lands inside
/// the projected face, and that centre reconstructs to a point up to half a pixel
/// outside the face's world rectangle. So this is a **superset** of what the GPU
/// can legitimately have marked, which is the direction that makes "every page
/// the GPU marked is one the geometry covers" a real assertion rather than a
/// tautology.
fn twin_pages(
    renderer: &inf_render::EngineRenderer,
    v: &RenderView,
    light: usize,
) -> BTreeSet<VsmPage> {
    let sys = renderer.vsm().expect("a live vsm system");
    let proj = &sys.projections()[light];
    let desc = &sys.trees()[light];
    let eye = v.eye_local();
    let scale = inf_render::projection_scale(v);
    let half = SIDE * 0.5;
    let pad = ((eye.z - half).abs() / scale) * 0.75;
    let mut out = BTreeSet::new();
    const N: i32 = 160;
    for iy in 0..=N {
        for ix in 0..=N {
            let fx = CUBE_XY.0 - half - pad + (2.0 * (half + pad)) * (ix as f32 / N as f32);
            let fy = CUBE_XY.1 - half - pad + (2.0 * (half + pad)) * (iy as f32 / N as f32);
            let p = glam::Vec3::new(fx, fy, half);
            let pixel_world = (p - eye).length() / scale;
            if let Some(page) = mark_page_for(proj, desc, p, pixel_world) {
                out.insert(page);
            }
        }
    }
    out
}

// ── (a) the pass runs, marks real pages, and marks the level the rule names ──

/// **THE MARKING ARM**: a real frame's depth buffer becomes a real set of
/// allocated shadow pages, at the level the CPU rule names and at the pages the
/// geometry covers.
///
/// This is the arm that fails when the pass does nothing (nothing is resident),
/// when it marks the wrong level (the level set is not `{2}`), and when its page
/// arithmetic is wrong (a page outside the twin's superset).
#[test]
fn the_marking_pass_allocates_the_pages_the_geometry_covers() {
    let Some(gpu) = gpu_or_skip("the VSM marking pass") else {
        return;
    };
    let set = settings();
    let v = view(6.0);
    let renderer = run(&gpu, &scene(&[glam::Vec3::Z]), &v, &set, 6);

    let sys = renderer.vsm().expect("a live vsm system");
    let stats = sys.stats();
    assert_eq!(
        renderer.vsm_engaged_frames(),
        6,
        "the renderer did not record the marking pass every frame: {}",
        sys.summary()
    );
    assert!(
        stats.projections >= 6,
        "no projection was dispatched — the pass ran over nothing: {}",
        sys.summary()
    );
    assert!(
        stats.mark_frames > 0,
        "the ring never landed in 6 frames, so the GPU pass is unobserved: {}",
        sys.summary()
    );
    assert_eq!(
        stats.mark_frames + stats.mark_misses,
        stats.frames,
        "a frame neither read a mask nor recorded a miss"
    );
    assert!(
        stats.admits > 0,
        "the loop allocated no page at all: {}",
        sys.summary()
    );
    assert_eq!(
        stats.deferred, 0,
        "the fixture's atlas is not under pressure"
    );

    // The RING landed, read through its own counters rather than through the
    // stats incremented beside them — a `mark_frames` that counted a miss as a
    // hit would agree with itself and disagree with this.
    assert!(
        sys.marker().ring().hits() > 0,
        "the ring reports no hit while the loop reports {} mask frames",
        stats.mark_frames
    );
    assert_eq!(
        sys.marker().ring().hits(),
        stats.mark_frames,
        "the ring and the loop disagree about how many masks arrived"
    );
    assert_eq!(sys.marker().ring().latency(), 2, "the pinned latency moved");
    // …and the line a host logs names the numbers it carries — the shape
    // `VtStats::summary` is pinned in, so the accessor is not shipped untested.
    let line = renderer.vsm_summary().expect("a live system has a line");
    assert!(
        line.contains("vsm residency") && line.contains("vsm marking"),
        "{line}"
    );
    assert!(line.contains("1 lights"), "{line}");
    assert!(
        line.contains(&format!("{} admits", stats.admits)),
        "the marking half does not name its admits: {line}"
    );

    // ASSERT THE WORLD: what is resident, page by page.
    let got = resident_pages(&renderer, 0);
    assert!(!got.is_empty(), "nothing is resident: {}", sys.summary());
    let levels: BTreeSet<u32> = got.iter().map(|p| p.level).collect();
    assert_eq!(
        levels,
        [2u32].into_iter().collect::<BTreeSet<_>>(),
        "the pass marked levels {levels:?} where the CPU rule says 2 — the level \
         rule and its mirror disagree about what this frame needs"
    );
    // ANTI-VACUITY on the level: it is at neither end of the tree, so this arm
    // can tell a rule from a clamp.
    let tree_levels = sys.trees()[0].level_count();
    assert!(
        tree_levels == 6 && !levels.contains(&0) && !levels.contains(&(tree_levels - 1)),
        "the fixture's level sits at an end of a {tree_levels}-level tree"
    );

    // …and every page it marked is one the geometry actually covers.
    let twin = twin_pages(&renderer, &v, 0);
    assert!(
        got.is_subset(&twin),
        "the GPU marked pages the geometry does not cover: {:?} (twin held {:?})",
        got.difference(&twin).collect::<Vec<_>>(),
        twin
    );
    // …and the page under the face's CENTRE was marked, which is the half a
    // subset check cannot make.
    let face_centre = glam::Vec3::new(CUBE_XY.0, CUBE_XY.1, SIDE * 0.5);
    let centre = mark_page_for(
        &sys.projections()[0],
        &sys.trees()[0],
        face_centre,
        (v.eye_local() - face_centre).length() / inf_render::projection_scale(&v),
    )
    .expect("the face centre is inside the clipmap");
    assert!(
        got.contains(&centre),
        "the page under the face's centre ({centre:?}) was never marked; got {got:?}"
    );
    // ANTI-VACUITY on the page set: the face straddles a page boundary, so this
    // is more than one page and "the marked set" is not "the one page there is".
    assert!(
        got.len() > 1,
        "the fixture covers one page ({got:?}), so a shader that marked a constant \
         page would pass"
    );
}

// ── (b) the control: a frame with nothing in it marks nothing ──

/// **The marked set is a function of the DEPTH BUFFER.** A sky-only frame marks
/// zero pages.
///
/// The arm that catches a pass that ignores depth: feed it a constant instead
/// and the clipmap fills with pages nothing will ever read, which looks — from
/// every counter above — exactly like success. Measured: replacing `depth` with
/// `0.5` in the reconstruction fails this arm and two others.
///
/// **What it does *not* pin, honestly**: the shader's explicit `depth <= 0.0`
/// early-out. Measured — weakening it to `depth < 0.0`, so a sky pixel falls
/// through, leaves every arm in this file green, because a reverse-Z depth of 0
/// unprojects to a point at infinity that the clipmap's containing-level test
/// rejects anyway. The early-out is a cost guard (most of a frame is sky, and it
/// skips the whole projection loop for those threads) and it is stated as intent
/// rather than defended by this arm.
#[test]
fn a_frame_with_no_geometry_marks_no_page() {
    let Some(gpu) = gpu_or_skip("the VSM empty-frame control") else {
        return;
    };
    let set = settings();
    let v = view(6.0);
    let mut empty = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    empty.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    empty.mark_dirty();

    let renderer = run(&gpu, &empty, &v, &set, 6);
    let sys = renderer.vsm().expect("a live vsm system");
    // The pass RAN — it just found nothing. Both halves matter: a pass that never
    // ran would also mark nothing.
    assert_eq!(renderer.vsm_engaged_frames(), 6, "{}", sys.summary());
    assert!(sys.stats().projections >= 6, "{}", sys.summary());
    assert!(sys.stats().mark_frames > 0, "{}", sys.summary());
    assert_eq!(
        sys.stats().admits,
        0,
        "a sky-only frame allocated {} shadow pages — the pass is not reading the \
         depth buffer: {}",
        sys.stats().admits,
        sys.summary()
    );
    assert!(resident_pages(&renderer, 0).is_empty());
    // …and every address still resolves, to NOTHING, which is the law.
    assert_eq!(
        sys.residency()
            .resolve(VsmLightHandle(0), VsmPage::flat(0, 0, 0)),
        None
    );
}

// ── (c) refinement under camera approach, on the device ──

/// **A closer camera marks a finer level**, and a farther one a coarser — the
/// property P26's gate could only infer and the one P27's gate has to state over
/// the page-allocation trace (the P26.5 audit's carried item, seeded here).
#[test]
fn the_marked_level_gets_finer_as_the_camera_closes() {
    let Some(gpu) = gpu_or_skip("the VSM level rule on a device") else {
        return;
    };
    let set = settings();
    let s = scene(&[glam::Vec3::Z]);
    let level_at = |eye_z: f64| -> BTreeSet<u32> {
        let v = view(eye_z);
        let r = run(&gpu, &s, &v, &set, 6);
        resident_pages(&r, 0).iter().map(|p| p.level).collect()
    };
    let near = level_at(3.0);
    let far = level_at(24.0);
    assert!(
        !near.is_empty() && !far.is_empty(),
        "one camera marked nothing"
    );
    let (near_max, far_min) = (
        *near.iter().next_back().expect("non-empty"),
        *far.iter().next().expect("non-empty"),
    );
    assert!(
        near_max < far_min,
        "the near camera marked levels {near:?} and the far one {far:?} — the \
         level rule does not track the camera"
    );
}

// ── (d) order independence, on the device ──

/// **The mask does not depend on the order the projections were uploaded in.**
///
/// `inf_vsm::mark`'s own arm reorders `mark` calls on the CPU. What it cannot say
/// is whether the *pass* is order-independent: the projection list is uploaded in
/// scene-light order and the threads that read it run in whatever order the
/// device chooses. The claim the whole replay-doctrine reconciliation rests on is
/// that neither matters.
#[test]
fn the_marked_set_does_not_depend_on_the_light_order() {
    let Some(gpu) = gpu_or_skip("the VSM mask's order independence") else {
        return;
    };
    let set = settings();
    let v = view(6.0);
    // Two suns from different directions, so the two lights genuinely mark
    // different page sets and an agreement is not an agreement about one answer.
    let (a, b) = (glam::Vec3::Z, glam::Vec3::new(0.4, 0.6, 0.7).normalize());
    let forward = run(&gpu, &scene(&[a, b]), &v, &set, 6);
    let reversed = run(&gpu, &scene(&[b, a]), &v, &set, 6);

    let fa = resident_pages(&forward, 0);
    let fb = resident_pages(&forward, 1);
    let ra = resident_pages(&reversed, 0);
    let rb = resident_pages(&reversed, 1);
    assert_eq!(
        fa, rb,
        "the first light's pages changed when it moved second"
    );
    assert_eq!(
        fb, ra,
        "the second light's pages changed when it moved first"
    );
    // ANTI-VACUITY: the two lights really do mark different sets, so the equality
    // above is not two copies of one answer.
    assert!(
        !fa.is_empty() && !fb.is_empty() && fa != fb,
        "the fixture's two suns mark the same pages ({} and {})",
        fa.len(),
        fb.len()
    );
}

// ── (e) the mirror is MEASURED, not mirrored ──

/// **The indirection buffer on the device says what the residency says** — read
/// back as bytes and compared against an independent walk.
///
/// The P26.5 law, applied at the moment the pair is created: every mirror-pinned
/// pair needs one independent-bytes arm, because two things written from the same
/// variable agree however wrong they both are.
#[test]
fn the_table_on_the_device_is_the_table_the_residency_resolved() {
    let Some(gpu) = gpu_or_skip("the VSM table readback") else {
        return;
    };
    let set = settings();
    let v = view(6.0);
    let renderer = run(&gpu, &scene(&[glam::Vec3::Z]), &v, &set, 6);
    let sys = renderer.vsm().expect("a live vsm system");
    let table = sys.pools().table();

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vsm-table-readback"),
        size: table.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(table, 0, &staging, 0, table.size());
    gpu.queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let words: Vec<u32> = {
        let view = staging.slice(..).get_mapped_range().expect("mapped");
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    staging.unmap();

    assert_eq!(
        words,
        sys.residency().table_words(),
        "the bytes on the device are not the words the residency holds"
    );
    assert_eq!(
        words[0],
        inf_vsm::VSM_TABLE_MAGIC,
        "the buffer was never written"
    );

    // …and the entries in those bytes decode to exactly what an independent walk
    // of the residency resolves — the half a `==` against `table_words()` cannot
    // make, since that is the same array the mirror was written from.
    let handle = VsmLightHandle(0);
    let desc = sys.residency().desc(handle).expect("registered").clone();
    let (base, _) = sys.residency().table_block(handle).expect("registered");
    let entries = base
        + inf_vsm::VSM_TABLE_LIGHT_HEADER_WORDS
        + inf_vsm::VSM_TABLE_LEVEL_REC_WORDS * desc.levels.len();
    let mut resolved = 0usize;
    let mut none = 0usize;
    for (&(_, page), want) in inf_vsm::resolved_table(sys.residency()).iter() {
        let idx = entries + desc.entry_index(page).expect("in tree") as usize;
        match (inf_vsm::unpack_entry(words[idx]), want) {
            (None, None) => none += 1,
            (Some(e), Some(w)) => {
                assert_eq!(e.slot, w.slot, "{page:?}");
                assert_eq!(
                    desc.ancestor(page, e.level),
                    Some(w.page),
                    "{page:?} resolves to a different ancestor on the device"
                );
                resolved += 1;
            }
            (got, _) => panic!("{page:?}: the device says {got:?} and the walk says {want:?}"),
        }
    }
    // ANTI-VACUITY, both directions: the comparison saw real entries AND real
    // absences, so it is not a sweep over one answer.
    assert!(
        resolved > 0 && none > 0,
        "{resolved} resolved and {none} empty entries — the readback compared one \
         kind of answer"
    );
}

// ── (f) the off path ──

/// **With the setting off, none of it exists.** No atlas, no pass, no counter —
/// which is what makes "no golden moved" a measurement of the command stream
/// rather than an expectation about pixels.
#[test]
fn a_renderer_with_virtual_shadows_off_records_nothing() {
    let Some(gpu) = gpu_or_skip("the VSM off path") else {
        return;
    };
    // The DEFAULT settings, not a hand-cleared copy: what every golden renders
    // with, and what P27.4 will have to flip deliberately.
    let off = VsmSettings::default();
    assert!(!off.enabled, "virtual shadow maps default ON");
    let renderer = run(&gpu, &scene(&[glam::Vec3::Z]), &view(6.0), &off, 6);
    assert!(renderer.vsm().is_none(), "a system was built anyway");
    assert_eq!(renderer.vsm_engaged_frames(), 0);
    assert!(renderer.vsm_summary().is_none());

    // …and a scene with no shadow-casting light builds nothing either, even with
    // the setting ON — the second half of the off path, and the one a level with
    // VSM enabled and the sun turned off actually takes.
    let mut dark = scene(&[]);
    dark.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: false,
        ..Default::default()
    });
    dark.mark_dirty();
    let renderer = run(&gpu, &dark, &view(6.0), &settings(), 6);
    assert!(
        renderer.vsm().is_none(),
        "a light that does not cast shadows built a page atlas"
    );
    assert_eq!(renderer.vsm_engaged_frames(), 0);
}

// ── (g) the perspective lights, on the device ──

/// **A point light's SIX faces mark through the device's own face arithmetic.**
///
/// Every arm above drives a directional clipmap, and the clipmap is the one tree
/// kind whose bit index never touches `face_stride`: it has one face, so
/// `face * face_stride` is zero and a shader that dropped the term entirely would
/// be green everywhere. A point light is where that arithmetic is load-bearing —
/// six faces sharing one table block and one bit range — and it is exactly the
/// shape the P26.3 law names: never claim the GPU ran something it did not.
///
/// The fixture wraps the cube in a point light, so the face the camera is looking
/// at is marked and the pages carry a **non-zero face index**.
#[test]
fn a_point_lights_faces_mark_through_their_own_stride() {
    let Some(gpu) = gpu_or_skip("the VSM cube-face marking") else {
        return;
    };
    let set = settings();
    let v = view(6.0);
    let mut s = scene(&[]);
    // A point light 10 m in front of the cube's face, along the camera's own
    // axis, so the lit face is squarely inside ONE cube face's frustum (index 5,
    // `-Z`) and the level its footprint justifies lands in the middle of the
    // tree. The distance is not decoration: a point light's level rule is
    // `ceil(log2(pixel_world × 64 × 2^(levels-1) / d))`, so at 3 m the answer is
    // the coarsest level for EVERY level count — measured, when this arm's first
    // fixture put the light there and the anti-vacuity check below caught it.
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Point,
        position: glam::DVec3::new(CUBE_XY.0 as f64, CUBE_XY.1 as f64, 11.0),
        range: 0.0,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();
    let renderer = run(&gpu, &s, &v, &set, 6);

    let sys = renderer.vsm().expect("a live vsm system");
    assert_eq!(
        sys.trees()[0].kind,
        inf_render::VsmTreeKind::Cube,
        "the fixture's light is not a cube"
    );
    assert_eq!(
        sys.projections().len(),
        6,
        "a point light is six projections: {}",
        sys.summary()
    );
    assert_eq!(renderer.vsm_engaged_frames(), 6);

    let got = resident_pages(&renderer, 0);
    assert!(
        !got.is_empty(),
        "a point light marked no page at all: {}",
        sys.summary()
    );
    // THE POINT OF THE ARM: the marked pages are on a face the stride has to
    // reach. Face 4 is `+Z` in `CUBE_FACE_BASES`, which is the face looking back
    // at the camera — a non-zero index, so `face * face_stride` is load-bearing.
    let faces: BTreeSet<u32> = got.iter().map(|p| p.face).collect();
    assert!(
        faces.iter().any(|f| *f > 0),
        "every marked page is on face 0, so the face stride was never exercised \
         ({faces:?})"
    );
    // …and every marked page is one the CPU twin agrees a face can hold: the
    // face index and the page grid come back consistent through the table's own
    // `first_entry` / `face_stride` pair, which is the arithmetic under test.
    let desc = &sys.trees()[0];
    for page in &got {
        assert!(page.face < 6, "{page:?}");
        let g = desc.levels[page.level as usize];
        assert!(page.x < g.pages_x && page.y < g.pages_y, "{page:?}");
    }
    // ANTI-VACUITY on the level, as everywhere else: a point light's level rule
    // is distance-scaled, so a marked level at an end of the tree would mean the
    // clamp answered rather than the rule.
    let levels: BTreeSet<u32> = got.iter().map(|p| p.level).collect();
    assert!(
        levels.iter().all(|l| *l > 0 && *l + 1 < desc.level_count()),
        "the marked levels {levels:?} reach an end of a {}-level tree, so a clamp          answered rather than the rule",
        desc.level_count()
    );
}
