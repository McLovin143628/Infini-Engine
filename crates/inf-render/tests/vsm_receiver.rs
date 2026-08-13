//! **The P27.4 receiver**, on a real device — `shadow_factor` through the page
//! table, the clipmap level blend, the engine's first point/spot shadows, and
//! the bias.
//!
//! Every arm here asserts the **frame**, and where it asserts an address it
//! asserts it against a re-derivation through `inf_render::vsm_receiver` rather
//! than against the shader agreeing with itself. What they are built to
//! falsify:
//!
//! * a receiver that reads **`VSM_ENTRY_NONE` as shadowed** — the fail-direction
//!   ruling that has lived in `inf_vsm::table`'s docs and three ledgers since
//!   P27.1 with no code able to violate it. This file is that code's gate
//!   (`a_missing_page_reads_lit_and_the_frame_says_so`), and it is the one
//!   obligation the P27.1 audit carried forward by name;
//! * a receiver that reads a **different page** from the one the pixel's own
//!   depth marked (the address walk twin, compared per pixel);
//! * a **bias** that leaves acne on a lit plane or peter-pans a contact shadow;
//! * a **point light** that reads the wrong cube face, or whose seam is a
//!   discontinuity in depth rather than only in address;
//! * a **spot** whose page grid is not its outer cone (the cone-to-fov
//!   derivation, which was CPU-covered only until this batch);
//! * a **level blend** that changes nothing, or that changes the whole frame
//!   rather than a band.
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.

use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance, RenderLight, RenderScene,
    RenderSettings, RenderView, VsmLightHandle, VsmPage, VsmSettings,
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

const FW: u32 = 256;
const FH: u32 = 144;

/// Frames every arm runs. Three is the floor — the marking ring is pinned at
/// `frame − 2`, so the first want cannot be applied before frame 2 — and six
/// gives the residency a frame to settle after it.
const FRAMES: u64 = 6;

/// Half the shipped clipmap, so a fixture's atlas is a handful of megabytes and
/// its levels are still real ones: 32 pages a side over a 24 m level 0 is a
/// **5.86 mm** level-0 texel, and 8 levels reach 3 km.
///
/// The grid was 8 pages over 12 m in this file's first draft and every
/// resolution-sensitive arm below failed against it — not because the receiver
/// was wrong but because a 94 mm shadow texel cannot resolve a 1.5 m caster, and
/// the derived bias correctly refused to try. The fixture is the thing that has
/// to be realistic; see `the_derived_bias_leaves_a_lit_plane_lit_and_a_shadow_attached`.
fn settings(slots: u64) -> VsmSettings {
    VsmSettings {
        enabled: true,
        budget_bytes: slots * 64 * 1024,
        clipmap_pages_per_side: 32,
        clipmap_levels: 8,
        first_level_extent_m: 12.0,
        spot_levels: 7,
        point_levels: 6,
        ..Default::default()
    }
}

/// **The level rule is one shadow texel per SCREEN pixel**, so a 256 x 144
/// fixture at a 60 degree field of view asks for 70 mm texels — coarser than any
/// shipped configuration, and coarse enough that the derived bias exceeds a
/// metre-scale caster's own depth separation at a grazing sun.
///
/// Every arm that measures shadow DETAIL therefore uses a narrow field of view,
/// which is the honest way to raise the pixel density of a small render target.
/// The arms that measure residency or engagement keep the wide one.
const NARROW_FOV_DEG: f32 = 25.0;

/// Straight down at the floor, so the view distance — and therefore the level
/// the rule picks — is nearly uniform across the frame. That is what keeps the
/// resident set inside ONE level, which is what makes "a page that is not
/// resident resolves to nothing" true rather than "resolves to its parent".
fn top_view(height: f64) -> RenderView {
    top_view_fov(height, 60.0)
}

fn top_view_fov(height: f64, fov_deg: f32) -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, height, 0.01),
        forward: glam::Vec3::new(0.0, -1.0, 0.0).normalize(),
        up: glam::Vec3::Z,
        fov_y: fov_deg.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// Every arm that looks along the ground uses [`look_view_fov`] directly: a
/// 60-degree wrapper existed in this file's first draft and every arm that used
/// it measured shadow detail, which a 60-degree field of view at 256 x 144
/// cannot carry (see `NARROW_FOV_DEG`).
fn look_view_fov(eye: glam::DVec3, at: glam::DVec3, fov_deg: f32) -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: eye,
        forward: (at - eye).as_vec3().normalize(),
        up: glam::Vec3::Y,
        fov_y: fov_deg.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// A white floor slab whose top face is at `y = 0`, and `casters` cubes standing
/// on it.
fn floor_scene(casters: &[(f64, f64)]) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        glam::DVec3::new(0.0, -0.5, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(60.0, 1.0, 60.0),
        [0.85, 0.85, 0.86, 1.0],
        1,
    ));
    for (i, (x, z)) in casters.iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            glam::DVec3::new(*x, 0.75, *z),
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.4, 1.5, 1.4),
            [0.75, 0.4, 0.3, 1.0],
            i as u32 + 2,
        ));
    }
    scene.mark_dirty();
    scene
}

/// A steep sun, so the shadows land near their casters and stay inside a small
/// clipmap.
fn sun(cast: bool) -> RenderLight {
    RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.94],
        intensity: 3.0,
        direction: glam::Vec3::new(0.30, 0.92, 0.25).normalize(),
        position: glam::DVec3::ZERO,
        range: 0.0,
        cast_shadows: cast,
        ..Default::default()
    }
}

/// Render `frames` frames and hand back the renderer **and** the last frame.
fn run(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    tune: impl FnOnce(&mut RenderSettings),
    frames: u64,
) -> (EngineRenderer, Vec<u8>) {
    let target = HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    tune(&mut s);
    renderer
        .try_set_settings(s)
        .expect("the arm's settings are legal");
    for _ in 0..frames {
        renderer.render(gpu, scene, view, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let img = target.read_rgba(gpu).expect("readback");
    (renderer, img)
}

/// The frame a scene renders with **no shadow term of any kind** — the control
/// every byte comparison below is against.
fn control(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    run(gpu, scene, view, |_| {}, FRAMES).1
}

fn differing(a: &[u8], b: &[u8]) -> Vec<usize> {
    (0..(FW * FH) as usize)
        .filter(|i| a[i * 4..i * 4 + 3] != b[i * 4..i * 4 + 3])
        .collect()
}

fn luma(img: &[u8], i: usize) -> f32 {
    let p = &img[i * 4..i * 4 + 3];
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

/// The whole page atlas, off the device, as `f32` depth — `tests/vsm_raster.rs`'s
/// door, kept identical so the two suites read the world the same way.
fn read_atlas(gpu: &GpuContext, renderer: &EngineRenderer) -> (u32, u32, Vec<f32>) {
    let sys = renderer.vsm().expect("a live vsm system");
    let tex = sys.pools().atlas();
    let (w, h) = (tex.width(), tex.height());
    let unpadded = w as usize * 4;
    let padded = unpadded.next_multiple_of(256);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vsm-atlas-readback"),
        size: (padded * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vsm-atlas-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map");
    let data = slice.get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((w * h) as usize);
    for row in data.chunks(padded).take(h as usize) {
        for texel in row[..unpadded].chunks_exact(4) {
            out.push(f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]));
        }
    }
    drop(data);
    buffer.unmap();
    (w, h, out)
}

// ── THE CARRIED GATE (P27.1 audit → P27.4) ──────────────────────────────────

/// **A page with no resident ancestor reads LIT, and the frame is the
/// assertion.**
///
/// The P27.1 ruling — *"a missing shadow page is a light LEAK, the opposite
/// fail-direction from virtual texturing's blurry-but-never-a-hole"* — has
/// lived in `inf_vsm::table`'s docs and in three STATUS ledgers, and the P27.1
/// audit found the thin spot: Phase 27's own "Done when" contains no
/// missing-fallback clause at all, *"so nothing anywhere will fail if P27.4's
/// receiver reads `VSM_ENTRY_NONE` as shadowed instead of lit"*. This is that
/// arm.
///
/// # The construction, and why it needs no door
///
/// A **budget of a few pages** over a marked set of many. The residency seats
/// what it can and defers the rest, so the frame has both populations at once:
/// pages that are resident and hold real rasterized caster depth, and pages that
/// resolve to `VSM_ENTRY_NONE`. Nothing is mocked and nothing is flushed — this
/// is the phase's own degradation, under its own budget pressure.
///
/// Three renders, and the assertions are between them:
///
/// * `control` — the same scene with no shadows at all;
/// * `full` — the same scene with a budget that seats everything;
/// * `starved` — the same scene with a handful of pages.
///
/// 1. `full ≠ control`, or the fixture cannot produce a shadow and every
///    equality below is about two identical frames.
/// 2. `starved ≠ control`, or the receiver read *nothing* and "it did not
///    darken the missing pages" is vacuous.
/// 3. **`starved` darkens only pixels `full` darkens.** This is the ruling. A
///    receiver that read `NONE` as slot 0 would sample the one page that *is*
///    resident all over the frame and darken pixels no shadow reaches; a
///    receiver that returned 0.0 on the sentinel would darken everything. Both
///    break the containment, and it is a byte comparison against a no-VSM
///    control rather than a counter.
/// 4. …and the counters say the two populations were both non-empty:
///    `admits > 0` **and** `deferred > 0`.
#[test]
fn a_missing_page_reads_lit_and_the_frame_says_so() {
    let Some(gpu) = gpu_or_skip("the P27.4 missing-page gate") else {
        return;
    };
    let scene = floor_scene(&[(-2.0, -1.0), (2.2, 1.4), (0.0, 3.0), (-3.4, 2.6)]);
    let mut scene = scene;
    scene.lights.push(sun(true));
    scene.mark_dirty();
    let view = top_view(9.0);

    let control = control(&gpu, &scene, &view);
    let (full_r, full) = run(&gpu, &scene, &view, |s| s.vsm = settings(64), FRAMES);
    let (thin_r, starved) = run(&gpu, &scene, &view, |s| s.vsm = settings(2), FRAMES);

    // (4) both populations exist, measured rather than hoped.
    let thin_stats = thin_r.vsm().expect("a live system").stats();
    assert!(
        thin_stats.admits > 0,
        "the starved run seated no page at all, so there is no resident \
         population to contrast the absent one with: {}",
        thin_stats.summary()
    );
    assert!(
        thin_stats.deferred > 0,
        "the starved run deferred nothing, so every page resolved and this arm \
         never reached a `VSM_ENTRY_NONE` at all: {}",
        thin_stats.summary()
    );
    // …and the residency itself says so, not only the counter.
    let absent = absent_pages(&thin_r);
    assert!(
        absent > 0,
        "no page of the starved run's tree resolves to NONE"
    );

    // (1) and (2): anti-vacuity on both frames.
    let full_diff = differing(&full, &control);
    let thin_diff = differing(&starved, &control);
    assert!(
        full_diff.len() > 200,
        "a full budget darkened {} pixels — the fixture casts no shadow, so \
         nothing below is a measurement",
        full_diff.len()
    );
    assert!(
        !thin_diff.is_empty(),
        "the starved run is byte-identical to the no-shadow control, so it read \
         no page at all and the containment below is vacuous"
    );

    // (3) THE RULING: a starved frame darkens only what a full one darkens.
    let full_set: std::collections::HashSet<usize> = full_diff.iter().copied().collect();
    let escaped: Vec<usize> = thin_diff
        .iter()
        .copied()
        .filter(|i| !full_set.contains(i))
        .collect();
    assert!(
        escaped.is_empty(),
        "{} pixels are shadowed with {} pages resident and LIT with all of them \
         — a page with no resident ancestor was read as shadow. The first is \
         pixel {} ({}, {}).",
        escaped.len(),
        thin_stats.admits,
        escaped[0],
        escaped[0] as u32 % FW,
        escaped[0] as u32 / FW,
    );
    // …and the starved frame really is the *smaller* of the two, which is what
    // says the missing pages went the LIT way rather than being coincidentally
    // equal.
    assert!(
        thin_diff.len() < full_diff.len(),
        "the starved frame darkened {} pixels and the full one {} — fewer \
         resident pages must mean less shadow, not more",
        thin_diff.len(),
        full_diff.len()
    );
    let _ = full_r;
}

/// Pages of every registered light that resolve to nothing.
fn absent_pages(renderer: &EngineRenderer) -> usize {
    let sys = renderer.vsm().expect("a live system");
    let res = sys.residency();
    inf_vsm::resolved_table(res)
        .values()
        .filter(|v| v.is_none())
        .count()
}

// ── the address walk, against its twin ──────────────────────────────────────

/// **The page a fragment reads is the page its own depth marked** — the shader's
/// address walk against `inf_render::vsm_receiver`'s, per pixel.
///
/// *Mirrored ≠ measured* (P26.5) applied to a walk rather than to a table. The
/// twin is evaluated on the CPU over the **atlas read back off the device**, so
/// what is compared is the shipped page content through two independent
/// address derivations — not the shader against itself.
///
/// The comparison is made where it can be made without inverting a PBR shade:
/// on a flat floor of one albedo under one directional light, luminance is
/// affine in the shadow factor, so *every pixel the twin calls fully lit must be
/// brighter than every pixel it calls fully shadowed*. A receiver reading the
/// wrong page puts a bright pixel inside the twin's shadow, and one wrong pixel
/// fails this.
#[test]
fn the_page_a_fragment_reads_is_the_page_its_own_depth_marked() {
    let Some(gpu) = gpu_or_skip("the P27.4 address twin") else {
        return;
    };
    let mut scene = floor_scene(&[(-1.5, -1.0), (1.8, 1.2)]);
    // 40 degrees of elevation: high enough that the derived bias is a fraction
    // of the caster's own depth separation, low enough that the shadow leaves
    // the caster's own footprint and lands on floor this arm can read.
    scene.lights.push(RenderLight {
        direction: glam::Vec3::new(0.0, 40f32.to_radians().sin(), 40f32.to_radians().cos())
            .normalize(),
        ..sun(true)
    });
    scene.mark_dirty();
    let view = top_view_fov(14.0, 35.0);
    let (renderer, img) = run(&gpu, &scene, &view, |s| s.vsm = settings(64), FRAMES);

    let sys = renderer.vsm().expect("a live system");
    let table = sys.residency().table_words().to_vec();
    let (aw, _ah, atlas) = read_atlas(&gpu, &renderer);
    let params = renderer
        .vsm_receiver_params()
        .expect("a live system packs its params");
    let desc = sys
        .residency()
        .desc(VsmLightHandle(0))
        .expect("the sun registered")
        .clone();
    let proj = sys.projections()[0];

    // Per pixel: the floor point it sees, then the twin's factor there.
    let vp = view.view_proj();
    let inv = vp.inverse();
    let eye = view.eye_local();
    let mut class = vec![-1i8; (FW * FH) as usize];
    let mut lit = Vec::new();
    let mut dark = Vec::new();
    for y in 0..FH {
        for x in 0..FW {
            let ndc = glam::Vec3::new(
                (x as f32 + 0.5) / FW as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / FH as f32 * 2.0,
                0.5,
            );
            // Two finite depths give the ray; the floor is the plane y = 0.
            let a = inv * glam::Vec4::new(ndc.x, ndc.y, 0.8, 1.0);
            let b = inv * glam::Vec4::new(ndc.x, ndc.y, 0.4, 1.0);
            let (a, b) = (a.truncate() / a.w, b.truncate() / b.w);
            let dir = (b - a).normalize();
            if dir.y.abs() < 1e-4 {
                continue;
            }
            let t = -a.y / dir.y;
            if t <= 0.0 {
                continue;
            }
            let world = a + dir * t;
            // Only the floor, and only well away from a caster's own silhouette
            // (a cube pixel is not a floor pixel and its luminance is a
            // different material).
            // The caster's own screen footprint is not floor: from above, a
            // cube's pixels are exactly its base rectangle, and its albedo is a
            // different material. The box is the CASTER's half-extent plus a
            // texel, NOT the shadow's — excluding the shadow would leave this
            // arm with nothing to separate.
            if scene.instances[1..].iter().any(|c| {
                let d =
                    world - glam::Vec3::new(c.translation.x as f32, 0.0, c.translation.z as f32);
                d.x.abs() < 0.8 && d.z.abs() < 0.8
            }) {
                continue;
            }
            let dist = (world - eye).length();
            let pixel_world = dist / inf_render::vt_stream::projection_scale(&view);
            let f = inf_render::vsm_shadow_factor(
                &table,
                &proj,
                &desc,
                world,
                glam::Vec3::Y,
                pixel_world,
                &params,
                |tx, ty| atlas[(ty * aw + tx) as usize],
            );
            let i = (y * FW + x) as usize;
            if f >= 0.999 {
                class[i] = 1;
            } else if f <= 0.001 {
                class[i] = 0;
            }
        }
    }
    // **The boundary is not the claim, and MSAA is why.** The scene resolves 4
    // samples a pixel, so a pixel straddling a shadow edge is a BLEND of a lit
    // and a shadowed shade — its luminance sits between the two populations by
    // construction, whatever the receiver did. So the comparison is made on the
    // INTERIORS: a pixel counts only when its whole 3 × 3 neighbourhood agrees
    // with it. A receiver that walked to a different page moves the boundary by
    // many pixels and still fails this; a receiver that is right by half a texel
    // does not.
    for y in 1..FH - 1 {
        for x in 1..FW - 1 {
            let i = (y * FW + x) as usize;
            if class[i] < 0 {
                continue;
            }
            let uniform = (-1i32..=1).all(|dy| {
                (-1i32..=1).all(|dx| {
                    let j = ((y as i32 + dy) as u32 * FW + (x as i32 + dx) as u32) as usize;
                    class[j] == class[i]
                })
            });
            if !uniform {
                continue;
            }
            if class[i] == 1 {
                lit.push(i);
            } else {
                dark.push(i);
            }
        }
    }

    // ANTI-VACUITY: both populations, and both large.
    assert!(
        lit.len() > 1_000,
        "the twin found {} fully-lit floor pixels",
        lit.len()
    );
    assert!(
        dark.len() > 100,
        "the twin found {} fully-shadowed floor pixels — the fixture's shadows \
         do not reach the floor, so the separation below is about one population",
        dark.len()
    );

    let brightest_dark = dark.iter().map(|i| luma(&img, *i)).fold(0.0f32, f32::max);
    let darkest_lit = lit
        .iter()
        .map(|i| luma(&img, *i))
        .fold(f32::INFINITY, f32::min);
    assert!(
        darkest_lit > brightest_dark,
        "the two halves overlap: the CPU twin's darkest LIT floor pixel is \
         {darkest_lit} and its brightest SHADOWED one is {brightest_dark} — the \
         shader and the twin walked to different pages"
    );
}

// ── the level blend ─────────────────────────────────────────────────────────

/// **The clipmap level blend replaces the cascade blend**: it acts, it acts in a
/// band, and zero restores the hard switch exactly.
///
/// The CSM's own `cascade_blend` is untouched by this batch and the arm says so
/// too — the two knobs are separate, and turning one off must not turn the other
/// off.
#[test]
fn the_clipmap_level_blend_acts_in_a_band_and_zero_is_the_hard_switch() {
    let Some(gpu) = gpu_or_skip("the P27.4 level blend") else {
        return;
    };
    // A long floor running away from the camera, so successive levels cover
    // successive screen rows — `csm_cascade_blend_softens_the_split_seam`'s
    // fixture shape, kept, because the claim is the same claim.
    let mut scene = floor_scene(&[
        (-2.0, -2.0),
        (2.0, -6.0),
        (-3.0, -12.0),
        (3.0, -20.0),
        (0.0, -30.0),
    ]);
    scene.lights.push(sun(true));
    scene.mark_dirty();
    let view = look_view_fov(
        glam::DVec3::new(0.0, 2.2, 6.0),
        glam::DVec3::new(0.0, 0.0, -14.0),
        NARROW_FOV_DEG,
    );

    let with = |blend: f32| {
        move |s: &mut RenderSettings| {
            s.vsm = VsmSettings {
                level_blend: blend,
                ..settings(256)
            };
        }
    };
    let (_, hard) = run(&gpu, &scene, &view, with(0.0), FRAMES);
    let (_, soft) = run(&gpu, &scene, &view, with(0.8), FRAMES);
    assert_ne!(hard, soft, "the level blend changed nothing at all");

    // It is a BAND, not a global change: the blend may not touch the near field,
    // which is well inside the finest level and has nothing to blend toward.
    let row_diffs: Vec<u32> = (0..FH)
        .map(|y| {
            (0..FW)
                .filter(|&x| {
                    let i = (y * FW + x) as usize;
                    hard[i * 4..i * 4 + 3] != soft[i * 4..i * 4 + 3]
                })
                .count() as u32
        })
        .collect();
    let touched: u32 = row_diffs.iter().sum();
    // **The blend can only act where the COARSER level is resident**, and the
    // marking pass asks for exactly one level per pixel — so the band is
    // populated only where some *other*, farther pixel happened to want the
    // level this one is blending toward. That is a real bound on the mechanism
    // and it is why this threshold is tens of pixels rather than thousands; it
    // is carried to P27.5 rather than papered over.
    eprintln!("the level blend moved {touched} pixels");
    assert!(
        touched > 30,
        "the blend barely touched anything ({touched})"
    );
    let bottom: u32 = row_diffs[(FH * 95 / 100) as usize..].iter().sum();
    assert_eq!(
        bottom, 0,
        "the blend reached the bottom rows — metres from the camera, in the \
         middle of the finest level's own band, with no seam to soften \
         ({bottom} pixels)"
    );

    // The escape hatch really is one, and it is deterministic.
    let (_, again) = run(&gpu, &scene, &view, with(0.0), FRAMES);
    assert_eq!(hard, again, "the zero-blend path is not deterministic");
}

// ── point and spot: the engine's first analytic shadows ─────────────────────

/// **A spot light casts a shadow**, and it is the first one this engine has ever
/// drawn from a spot.
///
/// The anti-vacuity is the light itself: a spot whose `cast_shadows` is false
/// must render the frame the control renders, so the difference below is the
/// page tree rather than the light.
#[test]
fn a_spot_light_casts_the_engines_first_spot_shadow() {
    let Some(gpu) = gpu_or_skip("the P27.4 spot receiver") else {
        return;
    };
    let mut scene = floor_scene(&[(0.0, 0.0)]);
    let spot = |cast: bool| RenderLight {
        kind: LightKind::Spot,
        color: [1.0, 0.95, 0.9],
        intensity: 220.0,
        // Straight down the beam from above and behind, so the cube's shadow
        // lands on the floor in front of it.
        position: glam::DVec3::new(0.0, 7.0, 4.5),
        direction: glam::Vec3::new(0.0, 0.82, 0.57).normalize(),
        range: 40.0,
        inner_cos: 32f32.to_radians().cos(),
        outer_cos: 40f32.to_radians().cos(),
        cast_shadows: cast,
    };
    scene.lights.push(spot(true));
    scene.mark_dirty();
    let view = look_view_fov(
        glam::DVec3::new(0.0, 5.0, 8.0),
        glam::DVec3::new(0.0, 0.0, -1.0),
        NARROW_FOV_DEG,
    );

    let (renderer, img) = run(&gpu, &scene, &view, |s| s.vsm = settings(256), FRAMES);
    let control = control(&gpu, &scene, &view);
    let diff = differing(&img, &control);
    assert!(
        diff.len() > 150,
        "a shadow-casting spot darkened {} pixels",
        diff.len()
    );

    // The tree really is a spot's — one quadtree, one face — and it is the one
    // the receiver was pointed at.
    let sys = renderer.vsm().expect("a live system");
    assert_eq!(sys.trees().len(), 1);
    assert_eq!(sys.trees()[0].kind, inf_vsm::VsmTreeKind::Quadtree);
    assert_eq!(sys.trees()[0].faces(), 1);
    let slots = renderer.vsm_light_slots();
    assert_eq!(slots, &[1], "the spot's projection slot");
    assert_eq!(
        renderer.vsm_receiver_params().expect("params").counts[0],
        0,
        "a scene with no directional light must have no sun slot"
    );

    // ANTI-VACUITY: the same spot with `cast_shadows` off is the control.
    let mut off = scene.clone();
    off.lights[0] = spot(false);
    off.mark_dirty();
    let (dark_r, unshadowed) = run(&gpu, &off, &view, |s| s.vsm = settings(256), FRAMES);
    assert!(
        dark_r.vsm().is_none(),
        "a scene whose only light does not cast still built a shadow system"
    );
    assert_eq!(
        unshadowed, control,
        "a non-casting spot changed the frame, so the difference above is not \
         the shadow"
    );
}

/// **A point light shadows through the cube face its own direction selects**,
/// and more than one face is engaged.
///
/// Two casters on opposite sides of the light, so the shadows leave through two
/// different faces: an arm with one caster is satisfied by a receiver that reads
/// face 0 for everything.
#[test]
fn a_point_light_shadows_through_more_than_one_of_its_faces() {
    let Some(gpu) = gpu_or_skip("the P27.4 point receiver") else {
        return;
    };
    let mut scene = floor_scene(&[(-2.4, 0.0), (2.4, 0.0), (0.0, -2.4), (0.0, 2.4)]);
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [1.0, 0.96, 0.92],
        intensity: 320.0,
        position: glam::DVec3::new(0.0, 3.2, 0.0),
        range: 40.0,
        cast_shadows: true,
        ..Default::default()
    });
    scene.mark_dirty();
    let view = top_view(11.0);

    let (renderer, img) = run(&gpu, &scene, &view, |s| s.vsm = settings(256), FRAMES);
    let control = control(&gpu, &scene, &view);
    assert!(
        differing(&img, &control).len() > 150,
        "a shadow-casting point light darkened nothing"
    );

    let sys = renderer.vsm().expect("a live system");
    assert_eq!(sys.trees()[0].kind, inf_vsm::VsmTreeKind::Cube);
    assert_eq!(sys.trees()[0].faces(), 6);
    // The residency's own answer to "which faces did the frame engage": the set
    // of faces with a resident page. Four casters around a light must reach more
    // than one, or the face selection is a constant.
    let faces: std::collections::BTreeSet<u32> = (0..sys.residency().geometry().slot_count())
        .filter_map(|s| sys.residency().slot_occupant(s))
        .map(|(_, p)| p.face)
        .collect();
    assert!(
        faces.len() > 1,
        "the point light's residency touched only face(s) {faces:?} — a cube \
         with one live face is a spot with five wasted quadtrees"
    );
    // …and the receiver's own face rule agrees about where each caster's shadow
    // leaves, through the shipped door rather than through a second copy.
    for (x, z) in [(-2.4f32, 0.0f32), (2.4, 0.0), (0.0, -2.4), (0.0, 2.4)] {
        let d = glam::Vec3::new(x, 0.0, z) - glam::Vec3::new(0.0, 3.2, 0.0);
        let f = inf_render::vsm_cube_face(d);
        assert!(
            faces.contains(&f),
            "the caster at ({x}, {z}) leaves through face {f}, which has no \
             resident page: {faces:?}"
        );
    }

    // **EVERY caster's own shadow, not the frame's total.** A residency face set
    // is the MARKING pass's answer, and a receiver that read face 0 for every
    // fragment would still leave the frame darker than the control — measured:
    // returning 0 from `vsm_cube_face` survived a whole-frame difference test,
    // because the one caster that really is on face 0 keeps its shadow. So each
    // of the four is probed at its own shadow, radially outward from the light.
    let vp = view.view_proj();
    let probe = |img: &[u8], world: glam::Vec3| -> f32 {
        let c = vp * world.extend(1.0);
        let n = c.truncate() / c.w;
        let x = ((n.x * 0.5 + 0.5) * FW as f32) as u32;
        let y = ((0.5 - n.y * 0.5) * FH as f32) as u32;
        luma(img, (y.min(FH - 1) * FW + x.min(FW - 1)) as usize)
    };
    for (x, z) in [(-2.4f32, 0.0f32), (2.4, 0.0), (0.0, -2.4), (0.0, 2.4)] {
        // The light is 3.2 m directly above the origin, so a caster's shadow
        // runs radially outward; 1.55x its radius clears the caster's own
        // 0.7 m half-extent and lands inside the shadow.
        let at = glam::Vec3::new(x * 1.55, 0.01, z * 1.55);
        let (lit, dark) = (probe(&control, at), probe(&img, at));
        assert!(
            dark < lit - 6.0,
            "the caster at ({x}, {z}) — cube face {} — casts no shadow at \
             {at:?}: {dark} against the control's {lit}",
            inf_render::vsm_cube_face(glam::Vec3::new(x, 0.0, z) - glam::Vec3::new(0.0, 3.2, 0.0)),
        );
    }
}

/// **A spot's page grid is exactly its outer cone** — the cone-to-fov
/// derivation's first device arm.
///
/// `spot_matrix` doubles `acos(outer_cos)` to get a vertical field of view, and
/// until this batch that was pinned only on the CPU (the P27.1 audit's
/// remainder: *"`spot_matrix`'s cone-to-fov derivation still has no device arm
/// of its own"*). Here a caster is placed at a **known fraction** of the cone's
/// half-width at a known distance, and the atlas is read back to find which page
/// column its depth landed in. Using the full angle instead of the half would
/// halve every NDC coordinate and move the column.
#[test]
fn a_spots_page_grid_is_exactly_its_outer_cone() {
    let Some(gpu) = gpu_or_skip("the P27.4 spot cone-to-fov arm") else {
        return;
    };
    // The beam points along −Y from 10 m up. The cone's half-width at depth `d`
    // is `d · tan(acos(outer_cos))`.
    let outer_deg = 35.0f32;
    let outer_cos = outer_deg.to_radians().cos();
    let depth = 9.0f32;
    let half_width = depth * (1.0 - outer_cos * outer_cos).sqrt() / outer_cos;
    // Half way out along +X: NDC x = +0.5.
    let cx = 0.5 * half_width;

    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        glam::DVec3::new(0.0, -0.5, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(60.0, 1.0, 60.0),
        [0.85, 0.85, 0.86, 1.0],
        1,
    ));
    scene.instances.push(MeshInstance::lit(
        glam::DVec3::new(cx as f64, 1.0, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(0.5, 2.0, 0.5),
        [0.7, 0.4, 0.3, 1.0],
        2,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Spot,
        color: [1.0; 3],
        intensity: 260.0,
        position: glam::DVec3::new(0.0, 10.0, 0.0),
        // `direction` is the direction TOWARD the light, so a beam pointing down
        // is +Y.
        direction: glam::Vec3::Y,
        range: 60.0,
        inner_cos: (outer_deg - 6.0).to_radians().cos(),
        outer_cos,
        cast_shadows: true,
    });
    scene.mark_dirty();
    let view = look_view_fov(
        glam::DVec3::new(0.0, 6.0, 9.0),
        glam::DVec3::ZERO,
        NARROW_FOV_DEG,
    );

    let (renderer, _) = run(&gpu, &scene, &view, |s| s.vsm = settings(256), FRAMES);
    let sys = renderer.vsm().expect("a live system");
    let (aw, _ah, atlas) = read_atlas(&gpu, &renderer);

    // The caster's tip, projected through the SHIPPED spot matrix, must land at
    // NDC x ≈ +0.5 — which is the whole of the cone-to-fov claim.
    let proj = sys.projections()[0];
    let vp = glam::Mat4::from_cols_array(&proj.view_proj);
    let tip = glam::Vec3::new(cx, 10.0 - depth, 0.0);
    let clip = vp * tip.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    // The MAGNITUDE is the cone-to-fov claim. The sign is the light basis's —
    // a beam pointing down has `right = fwd × up = −X` — and it is read off the
    // shipped `light_basis` rather than assumed, because an arm that hard-coded
    // `+0.5` would fail for a reason that has nothing to do with the cone.
    let (right, _, _) = inf_render::light_basis(glam::Vec3::Y);
    let want = right.x.signum() * 0.5;
    assert!(
        (ndc.x - want).abs() < 0.02,
        "a caster at half the outer cone's half-width projects to NDC x = {} \
         rather than {want} — the cone-to-fov derivation is not the half angle",
        ndc.x
    );
    assert!(
        ndc.y.abs() < 0.02,
        "the caster is on the beam's own axis in y: {}",
        ndc.y
    );

    // …and the atlas agrees: the page that page-space point names holds written
    // depth. This is the half a CPU projection cannot make.
    let desc = sys
        .residency()
        .desc(VsmLightHandle(0))
        .expect("registered")
        .clone();
    let mut found = None;
    for level in 0..desc.level_count() {
        let q = inf_render::vsm_level_ndc(&proj, ndc, level);
        let Some((px, py, local)) = inf_render::vsm_page_of(&desc, level, q) else {
            continue;
        };
        let page = VsmPage::new(0, level, px, py);
        let Some(r) = sys.residency().resolve(VsmLightHandle(0), page) else {
            continue;
        };
        let geom = sys.residency().geometry();
        let (ox, oy) = geom.slot_origin(r.slot).expect("a seated slot");
        let (tx, ty) = (ox + local.x as u32, oy + local.y as u32);
        let d = atlas[(ty * aw + tx) as usize];
        if d > inf_render::VSM_DEPTH_CLEAR {
            found = Some((level, px, py, d));
            break;
        }
    }
    let (level, px, py, d) = found.expect(
        "no level of the spot's quadtree holds written depth where its own \
         projection puts the caster — the page grid is not the cone",
    );
    // Reverse-Z: a caster 1 m below the tip is NEARER the light, so its stored
    // depth is smaller than the tip's own — and both are inside (0, 1].
    let tip_ndc = ndc.z;
    assert!(
        d > 0.0 && d <= 1.0,
        "level {level} page ({px},{py}) holds {d}, which is not a depth"
    );
    assert!(
        d >= tip_ndc - 1e-3,
        "the stored depth {d} is FURTHER from the light than the caster's own \
         tip {tip_ndc} — the page holds something else's depth"
    );
}

// ── the bias ────────────────────────────────────────────────────────────────

/// **The derived bias leaves a lit plane lit** — no acne — **and a contact
/// shadow attached** — no peter-panning.
///
/// Both halves in one arm, because a bias that fixes either alone is trivial: a
/// huge one removes acne and detaches every shadow, a zero one keeps contact and
/// stripes the ground.
///
/// * **Acne**: a plane lit by a *grazing* sun is where a page's texel density is
///   worst relative to the depth it stores. Every floor pixel outside a caster's
///   shadow must be exactly as bright as it is with no shadow at all.
/// * **Peter-panning**: the pixel at a caster's own foot must be darker than the
///   floor a metre away, or the shadow has walked off its object.
#[test]
fn the_derived_bias_leaves_a_lit_plane_lit_and_a_shadow_attached() {
    let Some(gpu) = gpu_or_skip("the P27.4 bias arm") else {
        return;
    };
    let mut scene = floor_scene(&[(0.0, 0.0)]);
    // **40 degrees of elevation, and the number is derived rather than chosen.**
    // The slope term is `(R + ½)·√2 · texel · tanθ` metres along the light, and a
    // caster of height `h` separates from its own shadow by `h·sin(elevation)`.
    // A shadow exists at all only where the second exceeds the first — and the
    // MARGIN between them is what the peter-panning half measures, so the arm
    // sits where the ratio is comfortable rather than at the edge.
    //
    // (This arm's first draft used 18 degrees against the file's first-draft
    // 94 mm texel, where the bias is **0.61 m** against a **0.46 m** separation
    // and the shadow correctly disappears. The derivation was right and the
    // fixture was asking a 94 mm shadow map to resolve a 1.5 m box at a grazing
    // angle. Recorded because it is the shape of the ruling: the slope bias is
    // a function of the PAGE's texel density, so a coarse page cannot hold a
    // shallow shadow, and no constant would have made it.)
    //
    // The sun comes from **−Z**, beyond the caster, so the shadow falls TOWARD
    // the camera. A sun behind the camera puts the shadow behind the caster,
    // where the caster itself occludes it — which is what this arm's first draft
    // did, and it cost an afternoon: every probe pixel was reading the CUBE, so
    // the receiver was measured at a point 2 m nearer than the arm believed and
    // at a level the twin never chose.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.94],
        intensity: 3.0,
        direction: glam::Vec3::new(0.0, 40f32.to_radians().sin(), -40f32.to_radians().cos())
            .normalize(),
        cast_shadows: true,
        ..Default::default()
    });
    scene.mark_dirty();
    let view = look_view_fov(
        glam::DVec3::new(0.0, 4.5, 8.5),
        glam::DVec3::new(0.0, 0.0, 0.5),
        NARROW_FOV_DEG,
    );

    let (_, img) = run(&gpu, &scene, &view, |s| s.vsm = settings(256), FRAMES);
    let control = control(&gpu, &scene, &view);
    let diff = differing(&img, &control);
    assert!(
        diff.len() > 200,
        "the grazing sun cast no shadow at all ({} pixels)",
        diff.len()
    );

    // **ACNE**: the shadow is one connected region behind one caster. Count the
    // darkened pixels in the half of the frame the shadow cannot reach — the
    // sun comes from +Z, so the shadow runs toward −Z, i.e. UP the frame; the
    // bottom rows are in front of the caster and must be untouched.
    let stray: usize = diff
        .iter()
        .filter(|i| (**i as u32) / FW < FH * 25 / 100)
        .count();
    assert_eq!(
        stray, 0,
        "{stray} pixels are darkened in front of the only caster, where no \
         shadow can fall — that is shadow acne, and it is what the slope term \
         exists to remove"
    );
    // …and the acne half is not vacuous: the region it scans is real floor that
    // the frame does light.
    let front: usize = (0..FH * 25 / 100)
        .flat_map(|y| (0..FW).map(move |x| (y * FW + x) as usize))
        .filter(|i| luma(&img, *i) > 40.0)
        .count();
    assert!(
        front > 1_000,
        "the top quarter of the frame holds {front} lit pixels, so \"no acne \
         there\" is a statement about the sky"
    );

    // **PETER-PANNING**: the floor immediately in front of the caster is in
    // shadow. The caster stands at the world origin and the shadow runs to +Z,
    // toward the camera, so this probe reads floor rather than caster.
    let vp = view.view_proj();
    let probe = |world: glam::Vec3| -> f32 {
        let c = vp * world.extend(1.0);
        let n = c.truncate() / c.w;
        let x = ((n.x * 0.5 + 0.5) * FW as f32) as u32;
        let y = ((0.5 - n.y * 0.5) * FH as f32) as u32;
        luma(&img, (y.min(FH - 1) * FW + x.min(FW - 1)) as usize)
    };
    let at_foot = probe(glam::Vec3::new(0.0, 0.01, 1.4));
    let away = probe(glam::Vec3::new(3.2, 0.01, 1.4));
    assert!(
        at_foot < away - 8.0,
        "the floor at the caster's own foot ({at_foot}) is not darker than the \
         open floor beside it ({away}) — the contact shadow detached"
    );
}

// ── engagement, and the defaults ────────────────────────────────────────────

/// **The three counters, and what each says on a scene that does not run virtual
/// shadows.**
///
/// P27.1's marking counter and P27.2's raster counter both already assert zero
/// on defaults. This adds the third — the receiver's — and the pattern is the
/// same one: it MOVES on a VSM-on scene, and it is zero on the default settings
/// *and* on a scene whose only light does not cast.
#[test]
fn the_receiver_engages_on_a_vsm_scene_and_is_inert_on_every_other() {
    let Some(gpu) = gpu_or_skip("the P27.4 engagement counters") else {
        return;
    };
    let mut scene = floor_scene(&[(0.0, 0.0)]);
    scene.lights.push(sun(true));
    scene.mark_dirty();
    let view = top_view(9.0);

    // ON: all three move, and the receiver's is one per frame.
    let (on, _) = run(&gpu, &scene, &view, |s| s.vsm = settings(64), FRAMES);
    assert_eq!(on.vsm_receiver_frames(), FRAMES);
    assert!(on.vsm_engaged_frames() > 0, "the marking pass never ran");
    assert!(on.vsm_raster_frames() > 0, "no page was ever rasterized");
    let params = on.vsm_receiver_params().expect("a live system");
    assert_eq!(params.counts[0], 1, "the sun's slot is projection 0 + 1");
    assert_eq!(params.counts[1], 1, "one light, one projection");
    assert!(
        (params.params[0] - VsmSettings::default().level_blend).abs() < 1e-6,
        "the blend band the shader read is not the setting"
    );

    // OFF by default: nothing is built and nothing is counted.
    let (off, _) = run(&gpu, &scene, &view, |_| {}, FRAMES);
    assert!(off.vsm().is_none());
    assert_eq!(off.vsm_receiver_frames(), 0);
    assert_eq!(off.vsm_engaged_frames(), 0);
    assert_eq!(off.vsm_raster_frames(), 0);
    assert!(off.vsm_receiver_params().is_none());
    assert!(off.vsm_light_slots().is_empty());

    // ON, but the only light does not cast: still nothing, and this is the case
    // a flag alone would get wrong.
    let mut idle = floor_scene(&[(0.0, 0.0)]);
    idle.lights.push(sun(false));
    idle.mark_dirty();
    let (quiet, _) = run(&gpu, &idle, &view, |s| s.vsm = settings(64), FRAMES);
    assert!(quiet.vsm().is_none());
    assert_eq!(quiet.vsm_receiver_frames(), 0);
    assert!(quiet.vsm_light_slots().is_empty());
}

/// **High and Medium run virtual shadows; Low keeps the cascaded path** — the
/// P27.5 clamp, wired at P27.1 and *reachable* now that the flag does something.
///
/// # One correction this arm's first draft needed, and it is worth writing down
///
/// The draft asserted that a Low-tier frame still shows shadows, on the reading
/// that "Low keeps CSM" means "Low keeps shadows". It does not:
/// `RenderTier::apply` has cleared `ShadowSettings::enabled` at Low since
/// P13.4.2, for cost, and that is a **pre-existing tier decision this batch does
/// not touch**. What the P27.5 clause means is which shadow *mechanism* a tier
/// runs, so the claim is made where it is true — on the settings and on the
/// system — and the CSM path's own liveness is asserted separately, by turning
/// virtual shadows off by hand and watching the cascade still shadow the frame.
/// That is the "demoted, not deleted" half, and it is the half a frame can see.
#[test]
fn the_tier_clamp_runs_vsm_on_high_and_medium_and_keeps_the_cascade_on_low() {
    let Some(gpu) = gpu_or_skip("the P27.5 tier clamp") else {
        return;
    };
    let mut scene = floor_scene(&[(0.0, 0.0), (2.4, 1.2)]);
    scene.lights.push(sun(true));
    scene.mark_dirty();
    let view = top_view(9.0);

    let asked = |s: &mut RenderSettings| {
        s.vsm = settings(64);
        s.shadows.enabled = true;
    };
    let tiered = |tier: inf_render::RenderTier| {
        move |s: &mut RenderSettings| {
            asked(s);
            *s = tier.apply(*s);
        }
    };

    let (high, high_img) = run(
        &gpu,
        &scene,
        &view,
        tiered(inf_render::RenderTier::High),
        FRAMES,
    );
    let (medium, _) = run(
        &gpu,
        &scene,
        &view,
        tiered(inf_render::RenderTier::Medium),
        FRAMES,
    );
    let (low, _) = run(
        &gpu,
        &scene,
        &view,
        tiered(inf_render::RenderTier::Low),
        FRAMES,
    );

    assert!(
        high.vsm().is_some(),
        "High tier did not run virtual shadows"
    );
    assert!(
        medium.vsm().is_some(),
        "Medium tier did not run virtual shadows — the clause is High/Medium"
    );
    assert!(
        low.vsm().is_none(),
        "Low tier built a virtual shadow system — the clamp that keeps the \
         cascaded path is not reachable"
    );
    assert_eq!(low.vsm_receiver_frames(), 0);
    assert!(high.vsm_receiver_frames() > 0);

    // Medium's clamp is a clamp and not a switch: it keeps the mechanism and
    // takes the grid and the budget down.
    let mut wanted = *high.settings();
    asked(&mut wanted);
    let m = inf_render::RenderTier::Medium.apply(wanted);
    assert!(m.vsm.enabled);
    assert!(m.vsm.clipmap_pages_per_side <= inf_render::VSM_CLIPMAP_PAGES_MEDIUM);
    assert!(m.vsm.budget_bytes <= inf_render::VSM_BUDGET_MEDIUM_BYTES);
    // …and Low's own shadow clamp is the PRE-EXISTING one, asserted here so the
    // frame comparison below is not read as this batch's doing.
    let l = inf_render::RenderTier::Low.apply(wanted);
    assert!(!l.vsm.enabled);
    assert!(!l.shadows.enabled, "Low's shadow clamp predates P27");

    // **Demoted, not deleted**: with virtual shadows off by hand and the cascade
    // on, the frame is still shadowed — the fallback P27.5 keeps is alive.
    let (csm, csm_img) = run(
        &gpu,
        &scene,
        &view,
        |s| {
            s.shadows.enabled = true;
        },
        FRAMES,
    );
    assert!(csm.vsm().is_none());
    let none = control(&gpu, &scene, &view);
    assert_ne!(
        csm_img, none,
        "the cascaded path no longer shadows anything — it is deleted rather \
         than demoted"
    );
    assert_ne!(
        high_img, csm_img,
        "the virtual and cascaded paths produced the same frame, so the clamp \
         changed nothing the eye could see"
    );
}
