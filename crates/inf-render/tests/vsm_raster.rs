//! **The P27.2 caster pass**, on a real device — the per-page GPU cull, the one
//! render pass that owns the atlas, and the viewport/scissor pair that pins each
//! page to its slot.
//!
//! Every arm here reads the atlas **back off the device** and compares texels.
//! That is deliberate and it is the standing law: the claim P27.2 makes is *depth
//! in a rectangle*, and a pass report, a draw counter or a residency snapshot can
//! all be perfect while the rectangle is empty. So the counters are used for
//! anti-vacuity and the assertions are made on depth.
//!
//! What these arms are built to falsify:
//!
//! * a raster that draws **nothing** (the atlas is all clear — which is exactly
//!   what P27.1 left, so every other assertion in the phase still passes);
//! * a raster whose page **matrix** is wrong (the depth is there but it is not the
//!   depth the page's own projection produces — checked against a CPU
//!   re-derivation, not against itself);
//! * a raster that writes **outside its page** (a missing or wrong
//!   `set_viewport`: the caster covers its whole page, so a viewport spanning the
//!   atlas paints every other slot);
//! * a cull that keeps everything (a caster far outside a page must not reach it);
//! * a masked material that shadows as a **solid** rather than as a cutout.
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.

use inf_render::{GpuContext, RenderScene, RenderView, VsmLightHandle, VsmPage, VsmSettings};

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
/// The caster cube's side, in metres.
const SIDE: f32 = 2.0;
/// Off-centre in y for `vsm_marking`'s reason — a marked set symmetric about the
/// clipmap centre cannot see the page grid's vertical flip.
const CUBE_XY: (f32, f32) = (0.0, 1.3);

/// A **small** atlas: 8 slots, so a page's rect is a large fraction of it and an
/// escaped draw is impossible to miss. 128-texel pages, 8 pages ⇒ 512 KiB.
fn settings() -> VsmSettings {
    settings_with(8)
}

/// The same clipmap with an atlas of `slots` pages. A 128-texel `Depth32Float`
/// page is exactly 64 KiB (the P27.1 page-geometry ruling), so a slot count is a
/// budget.
fn settings_with(slots: u64) -> VsmSettings {
    VsmSettings {
        enabled: true,
        budget_bytes: slots * 64 * 1024,
        clipmap_pages_per_side: 8,
        clipmap_levels: 6,
        first_level_extent_m: 6.0,
        ..Default::default()
    }
}

/// Where the opaque backdrop's near face sits, in metres along the light.
const BACKDROP_Z: f32 = -3.0;
/// Half its thickness.
const BACKDROP_T: f32 = 0.1;

/// A wide, thin, **opaque** slab well behind the caster — the surface whose depth
/// marks the pages, so an arm about a caster that discards is not an arm about a
/// frame with no depth in it at all.
fn backdrop() -> inf_render::MeshInstance {
    inf_render::MeshInstance::lit(
        glam::DVec3::new(0.0, 0.0, (BACKDROP_Z - BACKDROP_T) as f64),
        glam::Quat::IDENTITY,
        glam::Vec3::new(40.0, 40.0, 2.0 * BACKDROP_T),
        [1.0, 1.0, 1.0, 1.0],
        9,
    )
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

/// One cube under one directional shadow caster, on an empty sky.
fn scene(blend: u8, cutoff: f32, alpha: f32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    let mut inst = inf_render::MeshInstance::lit(
        glam::DVec3::new(CUBE_XY.0 as f64, CUBE_XY.1 as f64, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(SIDE),
        [1.0, 1.0, 1.0, alpha],
        1,
    );
    inst.blend = blend;
    inst.cutoff = cutoff;
    scene.instances.push(inst);
    scene.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        // Along +Z, so the light basis is unambiguous and a page's contents are
        // hand-checkable (`vsm.rs`'s clipmap twin arm uses the same direction for
        // the same reason).
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    scene.mark_dirty();
    scene
}

fn run(
    gpu: &GpuContext,
    scene: &RenderScene,
    v: &RenderView,
    set: &VsmSettings,
    frames: u64,
) -> inf_render::EngineRenderer {
    run_sequence(gpu, &[(scene, frames)], v, set)
}

/// The same renderer across a **sequence** of scenes — the door an arm about what
/// one frame leaves behind for the next has to come through (P27.2 audit). `run`
/// is this with one entry.
fn run_sequence(
    gpu: &GpuContext,
    steps: &[(&RenderScene, u64)],
    v: &RenderView,
    set: &VsmSettings,
) -> inf_render::EngineRenderer {
    run_tuned(gpu, steps, v, set, |_| {})
}

/// [`run_sequence`] with the rest of the render settings tunable — the door an arm
/// about a setting the caster pass *reads* (rather than one it owns) comes through.
fn run_tuned(
    gpu: &GpuContext,
    steps: &[(&RenderScene, u64)],
    v: &RenderView,
    set: &VsmSettings,
    tune: impl FnOnce(&mut inf_render::RenderSettings),
) -> inf_render::EngineRenderer {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    s.vsm = *set;
    tune(&mut s);
    renderer.set_settings(s);
    for (scene, frames) in steps {
        for _ in 0..*frames {
            renderer.render(gpu, scene, v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    renderer
}

/// The whole page atlas, off the device, as `f32` depth — **the WORLD**.
fn read_atlas(gpu: &GpuContext, renderer: &inf_render::EngineRenderer) -> (u32, u32, Vec<f32>) {
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

/// Every resident page, with the atlas rectangle it owns.
fn resident_pages(renderer: &inf_render::EngineRenderer) -> Vec<(u32, VsmPage, (u32, u32, u32))> {
    let sys = renderer.vsm().expect("a live vsm system");
    let res = sys.residency();
    let geom = res.geometry();
    let mut out = Vec::new();
    for slot in 0..geom.slot_count() {
        if let Some((light, page)) = res.slot_occupant(slot) {
            let (x, y) = geom.slot_origin(slot).expect("a seated slot has an origin");
            out.push((light.0, page, (x, y, geom.stored_page_size)));
        }
    }
    out
}

/// Texels of one rectangle that are not the reverse-Z clear value.
fn written(atlas: &(u32, u32, Vec<f32>), rect: (u32, u32, u32)) -> Vec<f32> {
    let mut out = Vec::new();
    for_each_written(atlas, rect, |_, _, d| out.push(d));
    out
}

/// Every written texel of one rectangle, with its **atlas coordinates** — the door
/// an arm about *where* in a page the depth landed comes through (P27.2 audit).
fn for_each_written(
    atlas: &(u32, u32, Vec<f32>),
    rect: (u32, u32, u32),
    mut f: impl FnMut(u32, u32, f32),
) {
    let (w, _, ref data) = *atlas;
    for y in rect.1..rect.1 + rect.2 {
        for x in rect.0..rect.0 + rect.2 {
            let d = data[(y * w + x) as usize];
            if d != inf_render::VSM_DEPTH_CLEAR {
                f(x, y, d);
            }
        }
    }
}

/// The render-local world point a written texel stands for: undo the viewport
/// transform into the page's own NDC, then the page matrix.
///
/// **The reconstruction is what makes a depth arm independent of the geometry that
/// wrote it** (P27.2 audit): it turns "there is depth in this page" into "the
/// surface is *here*, in metres", which is a claim the fixture's own heights can be
/// checked against rather than a claim about a texel count.
fn texel_world(
    vp_inv: glam::Mat4,
    rect: (u32, u32, u32),
    x: u32,
    y: u32,
    depth: f32,
) -> glam::Vec3 {
    let s = rect.2 as f32;
    // wgpu's viewport puts NDC +1 at the TOP of the rect, and a texel's centre is
    // half a texel in from its corner.
    let ndc_x = ((x - rect.0) as f32 + 0.5) / s * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y - rect.1) as f32 + 0.5) / s * 2.0;
    let h = vp_inv * glam::Vec4::new(ndc_x, ndc_y, depth, 1.0);
    h.truncate() / h.w
}

/// The page matrix of one resident page, through the shipped door.
fn page_vp(renderer: &inf_render::EngineRenderer, light: u32, page: VsmPage) -> glam::Mat4 {
    renderer
        .vsm()
        .expect("a live vsm system")
        .page_matrix(VsmLightHandle(light), page)
        .expect("a resident page has a matrix")
}

// ── (a) the pass runs, and the depth it writes is the depth the page's own
//        projection produces ────────────────────────────────────────────────────

/// **THE RASTER ARM**: a resident page holds real caster depth, at the value the
/// page's matrix says, and the pages nothing casts into hold the clear value.
///
/// The depth is checked against a **CPU re-derivation through
/// `vsm_page_matrix`** — the cube's near face, projected into the page — rather
/// than against "something non-zero", so a raster that drew the wrong geometry,
/// at the wrong scale, or through the wrong level fails here.
#[test]
fn a_resident_page_holds_the_depth_its_own_projection_produces() {
    let Some(gpu) = gpu_or_skip("the VSM page raster") else {
        return;
    };
    let set = settings();
    let v = view(5.0);
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &v, &set, 6);

    let stats = renderer
        .vsm_raster_stats()
        .expect("the system exists once a light casts");
    // ANTI-VACUITY, three ways: the pass ran, it saw pages, and it issued draws.
    // `vsm_raster_frames` counts frames that rasterized **at least one page**, and
    // since P27.3 that is a small number on a static scene by design — the pages
    // settle and the cache serves them. One is the floor the claim needs.
    assert!(renderer.vsm_raster_frames() >= 1, "{stats:?}");
    assert!(
        stats.pages > 0 && stats.draws > 0 && stats.casters > 0,
        "{stats:?}"
    );
    assert_eq!(stats.deferred_pages, 0, "the fixture outgrew the page cap");

    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty(), "nothing was resident to rasterize");
    let atlas = read_atlas(&gpu, &renderer);

    // The cube's near face in light space: the light looks along −Z (its
    // `direction` is the direction TO the light, +Z), so the surface a page sees
    // is the face at `z = +SIDE/2`.
    let mut checked = 0;
    let mut total_written = 0usize;
    for (light, page, rect) in &pages {
        let vp = page_vp(&renderer, *light, *page);
        let hits = written(&atlas, *rect);
        total_written += hits.len();
        if hits.is_empty() {
            continue;
        }
        // The cube's near face, at the page's own centre of coverage. Every texel
        // the raster wrote came off that face (the cube is the only caster and the
        // face is flat and axis-aligned), so its depth is a CONSTANT across the
        // page — which is what makes a single hand-computed value the right
        // assertion rather than a range.
        let p = glam::Vec3::new(CUBE_XY.0, CUBE_XY.1, SIDE * 0.5);
        let c = vp * p.extend(1.0);
        let want = c.z / c.w;
        for d in &hits {
            assert!(
                (d - want).abs() < 2e-3,
                "page {page:?} holds depth {d} where its own projection puts the \
                 caster's near face at {want}"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0 && total_written > 64,
        "the atlas held {total_written} written texels across {checked} pages — \
         the raster drew nothing"
    );
    // …and reverse-Z means a written texel is GREATER than the clear, always.
    // A forward-Z page would fail this and pass everything above.
    assert!(
        written(&atlas, (0, 0, atlas.0.min(atlas.1)))
            .iter()
            .all(|d| *d > inf_render::VSM_DEPTH_CLEAR),
        "a page holds depth below the reverse-Z clear"
    );
}

// ── (b) the scissor/viewport proof ──────────────────────────────────────────

/// **THE RECTANGLE PROOF**: a caster that fills its page writes **only** inside
/// that page's 128 × 128 rect, and every texel of every other slot is exactly the
/// clear value.
///
/// This is the arm the `set_viewport` / `set_scissor_rect` pair exists for, and it
/// is built to falsify them: the caster's projected footprint covers its whole
/// page, so with the viewport left at the attachment's default the same geometry
/// would be splattered across the entire atlas and every slot would be written.
/// Measured — see the ledger — deleting `set_viewport` fails this arm; deleting
/// `set_scissor_rect` does **not**, because clipping happens against the clip
/// volume before the viewport transform, so the scissor here is defence in depth
/// rather than the thing that pins the rect.
#[test]
fn a_caster_writes_inside_its_page_and_nowhere_else() {
    let Some(gpu) = gpu_or_skip("the VSM page rectangle") else {
        return;
    };
    let set = settings();
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &set, 6);
    let atlas = read_atlas(&gpu, &renderer);
    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty());

    let side = inf_render::VSM_PAGE_SIZE;
    let mut inside = 0usize;
    let mut occupied: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for (_, _, rect) in &pages {
        inside += written(&atlas, *rect).len();
        occupied.insert((rect.0, rect.1));
    }
    assert!(
        inside > 64,
        "the resident pages hold {inside} written texels — nothing to bound"
    );

    // Every texel of the atlas that is NOT inside a resident page's rect.
    let (w, h, ref data) = atlas;
    let mut escaped = 0usize;
    for y in 0..h {
        for x in 0..w {
            let corner = ((x / side) * side, (y / side) * side);
            if occupied.contains(&corner) {
                continue;
            }
            if data[(y * w + x) as usize] != inf_render::VSM_DEPTH_CLEAR {
                escaped += 1;
            }
        }
    }
    assert_eq!(
        escaped, 0,
        "{escaped} texels outside every page's rect were written — the page \
         viewport does not pin the 128×128 rectangle"
    );
    // ANTI-VACUITY: there really WERE slots outside the resident set, so "no
    // escapes" is a statement about a region that exists.
    let slots = (w / side) * (h / side);
    assert!(
        occupied.len() < slots as usize,
        "every slot of the atlas was resident ({} of {slots}), so the arm bounded \
         an empty region",
        occupied.len()
    );
}

/// **THE REGISTRATION PROOF** (P27.2 audit): a caster's silhouette lands on the
/// texels its page's own NDC says, not merely *somewhere* inside the slot.
///
/// `a_caster_writes_inside_its_page_and_nowhere_else` bounds the content to the
/// rect and nothing pinned it *to the rect's corner*: a viewport inset by two
/// texels — the shape a future border would introduce — is inside every rect,
/// writes every page it should, holds exactly the depth the CPU predicts, and
/// survived the whole file. It is a real defect, because P27.4 samples a page by
/// mapping a receiver's light-space position onto the slot and a two-texel shear
/// puts every shadow two texels off its caster.
///
/// So this arm compares the **bounding box of the written texels** against the
/// forward projection of the cube's own corners through `vsm_page_matrix` and the
/// viewport transform. One texel of tolerance, which is what a pixel-centre
/// coverage rule costs.
#[test]
fn a_pages_content_is_registered_to_its_slots_corner() {
    let Some(gpu) = gpu_or_skip("the VSM page registration") else {
        return;
    };
    let set = settings_with(64);
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &set, 6);
    let atlas = read_atlas(&gpu, &renderer);
    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty());

    let half = SIDE * 0.5;
    let mut checked = 0;
    // A page whose predicted box stops strictly inside the rect on some edge — the
    // anti-vacuity that says an EDGE was compared and not just "the whole slot".
    let mut partial = 0;
    for (light, page, rect) in &pages {
        let vp = page_vp(&renderer, *light, *page);
        let (mut ox0, mut oy0, mut ox1, mut oy1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut hits = 0;
        for_each_written(&atlas, *rect, |x, y, _| {
            ox0 = ox0.min(x);
            oy0 = oy0.min(y);
            ox1 = ox1.max(x);
            oy1 = oy1.max(y);
            hits += 1;
        });
        if hits == 0 {
            continue;
        }
        // The cube's near face, whose four corners are the silhouette: the light
        // looks along −Z and the face at `z = +SIDE/2` is the one a page sees.
        let (mut nx0, mut ny0, mut nx1, mut ny1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for (dx, dy) in [(-half, -half), (half, -half), (half, half), (-half, half)] {
            let c = vp * glam::Vec3::new(CUBE_XY.0 + dx, CUBE_XY.1 + dy, half).extend(1.0);
            let (nx, ny) = (c.x / c.w, c.y / c.w);
            nx0 = nx0.min(nx);
            nx1 = nx1.max(nx);
            ny0 = ny0.min(ny);
            ny1 = ny1.max(ny);
        }
        let s = rect.2 as f32;
        let to_x = |n: f32| rect.0 as f32 + (n * 0.5 + 0.5) * s;
        // NDC y is up and a viewport's y is down, so the NDC MAXIMUM is the top row.
        let to_y = |n: f32| rect.1 as f32 + (0.5 - n * 0.5) * s;
        let lo_x = to_x(nx0).max(rect.0 as f32);
        let hi_x = to_x(nx1).min((rect.0 + rect.2) as f32) - 1.0;
        let lo_y = to_y(ny1).max(rect.1 as f32);
        let hi_y = to_y(ny0).min((rect.1 + rect.2) as f32) - 1.0;
        for (label, want, got) in [
            ("left", lo_x, ox0),
            ("right", hi_x, ox1),
            ("top", lo_y, oy0),
            ("bottom", hi_y, oy1),
        ] {
            assert!(
                (got as f32 - want).abs() <= 1.0,
                "page {page:?} in rect {rect:?}: its {label} written texel is {got} \
                 where its own projection puts the caster's silhouette at {want:.2} \
                 — the page's content is not registered to its slot"
            );
        }
        if to_x(nx0) > rect.0 as f32 + 0.5 || to_y(ny1) > rect.1 as f32 + 0.5 {
            partial += 1;
        }
        checked += 1;
    }
    assert!(
        checked > 0 && partial > 0,
        "{checked} pages compared, {partial} of them with an edge strictly inside \
         the slot — an arm that only ever saw full pages cannot see a shear"
    );
}

// ── (c) the cull is subtractive ─────────────────────────

/// **THE CULL ARM**: the GPU's own per-page verdict, read back off the device and
/// compared against an independent CPU walk of the same spheres.
///
/// It has to be the args buffer rather than the atlas, and that is the finding
/// this arm is built around: **the atlas cannot tell a culled caster from a
/// clipped one.** A caster the page's frustum rejects writes no depth whether the
/// cull dropped it or the rasterizer did, so an image-side assertion is satisfied
/// by a cull that keeps everything — the failure that costs `pages × casters`
/// vertex invocations and is invisible everywhere else. The `instance_count`
/// words the cull wrote are the only record of what it decided, and reading them
/// off the device is *mirrored ≠ measured* (P26.5) applied to a decision.
#[test]
fn the_per_page_cull_drops_the_casters_the_page_cannot_see() {
    let Some(gpu) = gpu_or_skip("the VSM per-page cull") else {
        return;
    };
    let set = settings_with(64);
    // Two cubes, far apart in x, and a wide backdrop so pages exist between and
    // around them — without it every marked page sits under a cube and there is
    // nothing for the cull to reject.
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(-4.5, CUBE_XY.1 as f64, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(SIDE),
        [1.0, 1.0, 1.0, 1.0],
        2,
    ));
    s.instances.push(backdrop());
    s.mark_dirty();
    let renderer = run(&gpu, &s, &view(6.0), &set, 6);
    let sys = renderer.vsm().expect("live");
    let raster = sys.raster_state();
    let counts = raster.read_draw_counts(&gpu);
    let pages = raster.last_pages().to_vec();
    let groups = raster.last_groups();
    assert!(
        !pages.is_empty() && groups > 0,
        "the raster ran on no pages"
    );
    assert_eq!(counts.len(), pages.len() * groups);

    // The CPU twin: every caster's sphere, exactly as `pack_casters` derives it.
    let spheres: Vec<(glam::Vec3, f32)> = s
        .instances
        .iter()
        .map(|i| {
            let c = i.translation.as_vec3();
            let r = i.mesh.bounding_radius() * i.scale.x.max(i.scale.y).max(i.scale.z);
            (c, r)
        })
        .collect();

    let mut rejected = 0usize;
    let mut kept = 0usize;
    for (i, (light, page, _)) in pages.iter().enumerate() {
        let vp = page_vp(&renderer, light.0, *page);
        let want = spheres
            .iter()
            .filter(|(c, r)| inf_render::vsm_page_sees_sphere(&vp, *c, *r))
            .count() as u32;
        // Every caster in this fixture is a cube, so group 0 holds all of them and
        // the other four groups must be empty.
        let got = counts[i * groups];
        assert_eq!(
            got, want,
            "page {page:?}: the GPU cull kept {got} casters where the same test on \
             the CPU keeps {want}"
        );
        for g in 1..groups {
            assert_eq!(counts[i * groups + g], 0, "a non-cube group drew");
        }
        rejected += spheres.len() - want as usize;
        kept += want as usize;
    }
    // ANTI-VACUITY, both directions: the cull really rejected pairs and really
    // kept some. An agreement between two functions that both answer "all" is not
    // an agreement about culling.
    assert!(
        rejected > 0 && kept > 0,
        "the cull kept {kept} and rejected {rejected} (page, caster) pairs — one \
         of the two answers was never exercised"
    );
}

// ── (d) masked materials keep their alpha test in the page raster ───

/// **A cutout shadows as a cutout.** The same caster, masked with an alpha under
/// its cutoff, writes **no** depth of its own; opaque, it writes the page.
///
/// The scene carries an opaque backdrop, and that is not decoration: a masked
/// caster whose fragments all discard writes no *camera* depth either, so the
/// marking pass would mark nothing and the arm would pass with the whole feature
/// deleted. The backdrop is what marks the pages; the claim is then made about
/// depth **nearer the light than the backdrop**, which only the cube can produce.
#[test]
fn a_masked_caster_discards_in_the_page_raster() {
    let Some(gpu) = gpu_or_skip("the VSM masked caster") else {
        return;
    };
    let set = settings_with(64);
    let v = view(6.0);

    // Texels holding depth NEARER the light than the backdrop — i.e. the cube's.
    let cube_texels = |blend: u8, alpha: f32| -> (usize, inf_render::VsmRasterStats) {
        let mut s = scene(blend, 0.5, alpha);
        s.instances.push(backdrop());
        s.mark_dirty();
        let r = run(&gpu, &s, &v, &set, 6);
        let atlas = read_atlas(&gpu, &r);
        let mut n = 0usize;
        for (light, page, rect) in resident_pages(&r) {
            let vp = page_vp(&r, light, page);
            // The light looks along -Z, so a larger z is nearer it and — under
            // reverse-Z — holds the larger depth. Halfway between the two
            // surfaces separates them with room to spare.
            let at = |z: f32| {
                let c = vp * glam::Vec3::new(CUBE_XY.0, CUBE_XY.1, z).extend(1.0);
                c.z / c.w
            };
            let mid = 0.5 * (at(SIDE * 0.5) + at(BACKDROP_Z + BACKDROP_T));
            n += written(&atlas, rect).iter().filter(|d| **d > mid).count();
        }
        (n, r.vsm_raster_stats().expect("stats"))
    };

    // Opaque: the cube writes its own depth.
    let (solid, solid_stats) = cube_texels(0, 1.0);
    assert!(solid > 16, "the opaque control wrote {solid} cube texels");
    assert_eq!(
        solid_stats.masked_frames, 0,
        "an opaque scene bound the alpha-testing pipeline"
    );

    // Masked, alpha 0.2 under a 0.5 cutoff: every one of its fragments discards.
    let (cut, cut_stats) = cube_texels(1, 0.2);
    assert!(
        cut_stats.masked_frames > 0,
        "a masked caster did not reach the alpha-testing pipeline: {cut_stats:?}"
    );
    // The pass still RAN and still drew — this is a discard, not an absence.
    assert!(
        cut_stats.draws > 0 && cut_stats.casters > 0,
        "{cut_stats:?}"
    );
    assert_eq!(
        cut, 0,
        "a fully-cut-out caster wrote {cut} depth texels — its shadow is solid \
         where its material is a hole"
    );

    // …and the same masked material with alpha ABOVE its cutoff writes again, so
    // the arm above is the alpha test rather than "masked draws nothing".
    let (keep, _) = cube_texels(1, 0.9);
    assert!(
        keep > 16,
        "a masked caster whose alpha CLEARS its cutoff wrote {keep} texels"
    );
}

// ── (e) virtualized geometry casts ────────────────────────────

/// **THE HOLE THIS PHASE NAMES.** Phase 27's goal says "every caster path casts
/// (vgeom's 'casts no shadows' hole closes here)", and before this batch it was
/// total: `passes/shadow.rs` contains no occurrence of `vgeom` or `meshlet`, and
/// `passes/vgeom.rs` contains no shadow pipeline, no light-space matrix and no
/// caster registration.
///
/// A vmesh instance now writes depth into the pages its bounds touch. The arm is
/// built to falsify a path that merely *compiles*: the counters prove vgeom
/// casters were packed, and the atlas proves depth arrived where the CPU says the
/// asset's own bounding sphere puts it.
#[test]
fn a_virtualized_geometry_instance_casts_into_the_pages_it_touches() {
    let Some(gpu) = gpu_or_skip("the VSM meshlet-asset caster") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![
            inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh")
        ],
        ..Default::default()
    };
    // Standing up, so its silhouette faces the light along +Z.
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::new(0.0, 0.0, 0.0),
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(3.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(64);
    let renderer = run(&gpu, &s, &view(9.0), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(
        stats.vgeom_casters > 0,
        "no virtualized-geometry caster was packed: {stats:?}"
    );

    // The vmesh's own depth: nearer the light than the backdrop, which is the
    // only other caster in the scene. Anything the backdrop wrote is at the
    // backdrop's depth, so a texel past the midpoint can only be the vmesh's.
    let atlas = read_atlas(&gpu, &renderer);
    let mut vgeom_texels = 0usize;
    for (light, page, rect) in resident_pages(&renderer) {
        let vp = page_vp(&renderer, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        vgeom_texels += written(&atlas, rect).iter().filter(|d| **d > mid).count();
    }
    assert!(
        vgeom_texels > 16,
        "the vmesh wrote {vgeom_texels} texels nearer the light than the \
         backdrop — it is not casting"
    );

    // ANTI-VACUITY / control: the same scene with the vmesh instance removed
    // writes NOTHING past the backdrop, so the count above is the asset's own
    // depth and not the slab's.
    let mut bare = s.clone();
    bare.vgeom_instances.clear();
    bare.mark_dirty();
    let control = run(&gpu, &bare, &view(9.0), &set, 6);
    let control_atlas = read_atlas(&gpu, &control);
    let mut control_texels = 0usize;
    for (light, page, rect) in resident_pages(&control) {
        let vp = page_vp(&control, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        control_texels += written(&control_atlas, rect)
            .iter()
            .filter(|d| **d > mid)
            .count();
    }
    assert_eq!(
        control_texels, 0,
        "the backdrop alone wrote {control_texels} texels past its own depth"
    );
    assert_eq!(
        control.vsm_raster_stats().expect("stats").vgeom_casters,
        0,
        "a scene with no vmesh instance packed a vmesh caster"
    );
}

// ── (g) skinned and terrain cast ─────────────────────────────

/// Depth **nearer the light than the backdrop**, summed over every resident page
/// — the measurement every "does this path cast?" arm in this file makes.
fn depth_past_backdrop(gpu: &GpuContext, renderer: &inf_render::EngineRenderer) -> usize {
    let atlas = read_atlas(gpu, renderer);
    let mut n = 0usize;
    for (light, page, rect) in resident_pages(renderer) {
        let vp = page_vp(renderer, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        n += written(&atlas, rect).iter().filter(|d| **d > mid).count();
    }
    n
}

/// **A skinned caster casts, and it casts its POSE.** The skeleton's palette
/// reaches the page raster, so a character's shadow is the character's silhouette
/// rather than its bind pose sitting wherever the asset was authored.
///
/// The control is the same instance with a palette that translates it out of the
/// light's reach: same mesh, same instance transform, same everything but the
/// matrices — so what the arm measures is the skinning, not the presence of a
/// draw call.
#[test]
fn a_skinned_caster_casts_through_its_own_palette() {
    let Some(gpu) = gpu_or_skip("the VSM skinned caster") else {
        return;
    };
    // A unit quad in the XY plane, every vertex bound to joint 0.
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });

    let build = |palette: glam::Mat4| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        s.skinned.push(inf_render::SkinnedInstance {
            translation: glam::DVec3::new(0.0, CUBE_XY.1 as f64, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 3,
            mesh: 0,
            palette: vec![palette],
            vt: inf_render::VtTextureSet::NONE,
        });
        s.instances.push(backdrop());
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        s.mark_dirty();
        s
    };

    let set = settings_with(64);
    let posed = run(&gpu, &build(glam::Mat4::IDENTITY), &view(6.0), &set, 6);
    let stats = posed.vsm_raster_stats().expect("stats");
    assert!(
        stats.skinned_casters > 0,
        "no skinned caster was packed: {stats:?}"
    );
    let lit = depth_past_backdrop(&gpu, &posed);
    assert!(lit > 16, "the skinned quad wrote {lit} texels");

    // The SAME instance, posed 50 m behind the backdrop by its palette alone.
    // If the palette were ignored, the quad would sit where the bind pose puts it
    // and this count would match the one above.
    let far = run(
        &gpu,
        &build(glam::Mat4::from_translation(glam::Vec3::new(
            0.0, 0.0, -50.0,
        ))),
        &view(6.0),
        &set,
        6,
    );
    assert_eq!(
        depth_past_backdrop(&gpu, &far),
        0,
        "the joint palette did not move the caster — the page raster is drawing \
         the bind pose"
    );
    assert!(
        far.vsm_raster_stats().expect("stats").skinned_casters > 0,
        "the control packed no skinned caster, so it proves nothing"
    );
}

/// **A terrain tile casts**, out of its own heights rather than out of the
/// camera-fitted clipmap patch.
#[test]
fn a_terrain_tile_casts_from_its_own_heights() {
    let Some(gpu) = gpu_or_skip("the VSM terrain caster") else {
        return;
    };
    // One 33-sample tile, 0.5 m a sample, with a ridge along x that stands well
    // in front of the backdrop.
    const RES: u32 = 33;
    const MPS: f64 = 0.5;
    let span = (RES as f64 - 1.0) * MPS;
    let mut heights = vec![0f32; (RES * RES) as usize];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..RES {
        for i in 0..RES {
            // A ramp in z: the tile's own surface leans toward the light.
            let h = 2.0 - 3.0 * (j as f32 / (RES - 1) as f32);
            heights[(j * RES + i) as usize] = h;
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    let terrain = inf_render::RenderTerrain {
        id: 7,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: vec![inf_render::RenderTerrainTile {
            key: inf_render::TerrainTileKey::lod0((0, 0)),
            origin: glam::DVec3::new(-0.5 * span, 0.0, -0.5 * span),
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (lo, hi),
            holes: Vec::new(),
            version: 1,
        }],
        layers: Default::default(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    };

    let mut s = scene(0, 0.5, 1.0);
    s.terrains.push(terrain);
    s.mark_dirty();
    let set = settings_with(64);
    let renderer = run(&gpu, &s, &view(8.0), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(
        stats.terrain_casters > 0,
        "no terrain tile was packed as a caster: {stats:?}"
    );
    // The heightfield is the ONLY thing in this scene at those depths besides the
    // cube, and the cube is a metre across; the tile spans 16 m. So a page that
    // holds terrain depth holds far more written texels than the cube alone can
    // explain — measured against the same scene with the terrain removed.
    let with_terrain = {
        let atlas = read_atlas(&gpu, &renderer);
        resident_pages(&renderer)
            .iter()
            .map(|(_, _, r)| written(&atlas, *r).len())
            .sum::<usize>()
    };
    let mut bare = s.clone();
    bare.terrains.clear();
    bare.mark_dirty();
    let control = run(&gpu, &bare, &view(8.0), &set, 6);
    let without = {
        let atlas = read_atlas(&gpu, &control);
        resident_pages(&control)
            .iter()
            .map(|(_, _, r)| written(&atlas, *r).len())
            .sum::<usize>()
    };
    assert_eq!(
        control.vsm_raster_stats().expect("stats").terrain_casters,
        0,
        "a scene with no terrain packed a terrain caster"
    );
    assert!(
        with_terrain > without + 64,
        "the terrain added {} written texels over a control of {without} — it is \
         not casting",
        with_terrain.saturating_sub(without)
    );
}

/// One planar terrain tile: `height(u, v) = a + b·u + c·v` over the tile's own
/// samples, so the caster mesh's triangulation reproduces it **exactly** whatever
/// the decimation does and a depth arm can assert metres rather than texels.
#[derive(Clone, Copy)]
struct PlanarTile {
    key: inf_render::TerrainTileKey,
    origin: glam::DVec3,
    plane: (f32, f32, f32),
}

const TILE_RES: u32 = 33;
const TILE_MPS: f64 = 0.5;
const TILE_SPAN: f64 = (TILE_RES as f64 - 1.0) * TILE_MPS;

impl PlanarTile {
    fn height(&self, u: f32, v: f32) -> f32 {
        self.plane.0 + self.plane.1 * u + self.plane.2 * v
    }
    /// The world height of this tile's surface at render-local `(x, z)`, or `None`
    /// outside its own footprint.
    fn world_y(&self, x: f32, z: f32) -> Option<f32> {
        let u = (x as f64 - self.origin.x) / TILE_SPAN;
        let v = (z as f64 - self.origin.z) / TILE_SPAN;
        // A texel exactly on the edge is inside; the slack is one sample.
        let e = TILE_MPS / TILE_SPAN;
        ((-e..=1.0 + e).contains(&u) && (-e..=1.0 + e).contains(&v))
            .then(|| self.origin.y as f32 + self.height(u as f32, v as f32))
    }
    fn build(&self) -> inf_render::RenderTerrainTile {
        let mut heights = vec![0f32; (TILE_RES * TILE_RES) as usize];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for j in 0..TILE_RES {
            for i in 0..TILE_RES {
                let h = self.height(
                    i as f32 / (TILE_RES - 1) as f32,
                    j as f32 / (TILE_RES - 1) as f32,
                );
                heights[(j * TILE_RES + i) as usize] = h;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        inf_render::RenderTerrainTile {
            key: self.key,
            origin: self.origin,
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (lo, hi),
            holes: Vec::new(),
            version: 1,
        }
    }
}

fn planar_terrain(tiles: &[PlanarTile]) -> inf_render::RenderTerrain {
    inf_render::RenderTerrain {
        id: 7,
        tile_resolution: TILE_RES,
        meters_per_sample: TILE_MPS,
        tiles: tiles.iter().map(PlanarTile::build).collect(),
        layers: Default::default(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    }
}

/// A camera above the ground looking down at it, so the tiles mark pages.
fn terrain_view() -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 14.0, 14.0),
        forward: glam::Vec3::new(0.0, -1.0, -1.0).normalize(),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// A scene of nothing but planar terrain under an overhead sun — so **every**
/// written texel of the atlas is ground and there is nothing else for a residual
/// to be blamed on.
fn terrain_scene(tiles: &[PlanarTile]) -> RenderScene {
    let mut s = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    s.terrains.push(planar_terrain(tiles));
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        // Straight up: the direction TO the light, so the sun looks down and a
        // page's depth is world height. Nothing here depends on the light basis —
        // the page matrix is INVERTED rather than assumed.
        direction: glam::Vec3::Y,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();
    s
}

/// Every written atlas texel, reconstructed into render-local metres and compared
/// against the nearest tile surface below it. Returns `(texels checked, worst
/// residual in metres, texels that landed on no tile at all)`.
fn terrain_residuals(
    gpu: &GpuContext,
    renderer: &inf_render::EngineRenderer,
    tiles: &[PlanarTile],
) -> (usize, f32, usize) {
    let atlas = read_atlas(gpu, renderer);
    let (mut checked, mut worst, mut orphan) = (0usize, 0f32, 0usize);
    for (light, page, rect) in resident_pages(renderer) {
        let inv = page_vp(renderer, light, page).inverse();
        for_each_written(&atlas, rect, |x, y, d| {
            let p = texel_world(inv, rect, x, y, d);
            let best = tiles
                .iter()
                .filter_map(|t| t.world_y(p.x, p.z))
                .map(|y| (y - p.y).abs())
                .fold(f32::INFINITY, f32::min);
            if best.is_finite() {
                checked += 1;
                worst = worst.max(best);
            } else {
                orphan += 1;
            }
        });
    }
    (checked, worst, orphan)
}

/// **THE TERRAIN SURFACE ARM** (P27.2 audit): the depth a page holds over ground
/// is the tile's **own** surface, in metres, at the world height the tile's origin
/// and heights put it.
///
/// The first terrain arm counted texels against a control — which says the tile
/// casts *something* and nothing about *what*. Measured: transposing the height
/// index (`heights[si·res + sj]`) and dropping the tile origin's `y` from the
/// vertex both survived the whole file, the second one because the fixture's tile
/// origin was `y = 0` and a fixture that cannot distinguish is not a control. This
/// arm reconstructs every written texel through the page matrix's inverse and
/// compares it to `origin.y + height(u, v)`.
///
/// The fixture's height field is **planar** and its origin is off the ground plane
/// in all three axes, so the assertion is exact under any triangulation and no term
/// of the composition is zero.
#[test]
fn a_terrain_page_holds_the_tiles_own_surface_in_metres() {
    let Some(gpu) = gpu_or_skip("the VSM terrain surface") else {
        return;
    };
    let tiles = [PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        // Off-origin on every axis. `y = 3.5` is the term the old fixture zeroed.
        origin: glam::DVec3::new(-0.5 * TILE_SPAN, 3.5, -0.5 * TILE_SPAN),
        // Sloped on BOTH axes, and by different amounts, so a transposed height
        // index is a different surface rather than the same one.
        plane: (1.0, 2.5, -4.0),
    }];
    let set = settings_with(64);
    let renderer = run(&gpu, &terrain_scene(&tiles), &terrain_view(), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.terrain_casters > 0, "{stats:?}");

    let (checked, worst, orphan) = terrain_residuals(&gpu, &renderer, &tiles);
    assert!(
        checked > 512,
        "only {checked} ground texels were reconstructed — the arm bounded almost \
         nothing"
    );
    assert_eq!(
        orphan, 0,
        "{orphan} written texels reconstruct to a point over no tile at all"
    );
    assert!(
        worst < 0.05,
        "a page's depth puts the ground {worst} m off the tile's own surface"
    );
}

/// **The terrain caster cache is keyed on the tile, not on its place in a
/// streaming list** (P27.2 audit) — `b921fd3`'s fix, which shipped without an arm.
///
/// The list is a residency: tiles arrive and leave, and the entry that was index 0
/// last frame belongs to a different tile this frame. Keyed on the index, a tile
/// that streams in over an evicted one's slot inherits the evicted one's *mesh*
/// whenever the two share a version stamp — and the two always do, because a tile's
/// version starts at 1.
///
/// So: render a tile, evict it, stream a different tile into its place at the same
/// version, and assert the ground is at the NEW tile's height. Under the old key
/// the residual is the whole difference between the two.
#[test]
fn a_streamed_in_terrain_tile_does_not_inherit_the_evicted_ones_mesh() {
    let Some(gpu) = gpu_or_skip("the VSM terrain cache key") else {
        return;
    };
    // `keep` is resident throughout; `first` is evicted and `second` takes its
    // place in the list, at a DIFFERENT height and a different tile key.
    let keep = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((1, 0)),
        origin: glam::DVec3::new(0.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (1.0, 0.0, 0.0),
    };
    let first = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        origin: glam::DVec3::new(-1.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (0.0, 0.0, 0.0),
    };
    let second = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 1)),
        origin: glam::DVec3::new(-1.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        // Six metres above where the evicted tile's mesh sits.
        plane: (6.0, 0.0, 0.0),
    };
    let set = settings_with(64);
    let before = terrain_scene(&[first, keep]);
    let after_tiles = [second, keep];
    let after = terrain_scene(&after_tiles);
    let v = terrain_view();
    let renderer = run_sequence(&gpu, &[(&before, 6), (&after, 6)], &v, &set);
    assert!(
        renderer.vsm_raster_stats().expect("stats").terrain_casters > 0,
        "no terrain caster survived the swap"
    );

    let (checked, worst, orphan) = terrain_residuals(&gpu, &renderer, &after_tiles);
    assert!(checked > 512, "only {checked} ground texels after the swap");
    assert_eq!(orphan, 0, "{orphan} texels over no resident tile");
    assert!(
        worst < 0.05,
        "after a tile was evicted and another streamed into its place, the ground \
         is {worst} m off the resident tiles' own surfaces — the caster mesh is the \
         EVICTED tile's, inherited through a cache keyed on a streaming index"
    );
    // ANTI-VACUITY: the evicted tile's mesh sat six metres below the one that
    // replaced it, so a stale mesh would have been far outside the tolerance above.
    let over_the_swap = -(TILE_SPAN as f32);
    let now = second
        .world_y(over_the_swap, 0.0)
        .expect("inside the new tile");
    let then = first
        .world_y(over_the_swap, 0.0)
        .expect("inside the old one");
    assert!(
        (now - then).abs() > 5.0,
        "the two tiles are {} m apart, which the {} m tolerance would not have seen",
        (now - then).abs(),
        0.05
    );
}

// ── (h) off path, and the settings door ─────────────────────────────────────

/// With virtual shadows off, the caster pass never opens — the byte-stability
/// guarantee every golden rests on, as a counter rather than a hope.
#[test]
fn a_renderer_with_virtual_shadows_off_rasterizes_no_page() {
    let Some(gpu) = gpu_or_skip("the VSM off path") else {
        return;
    };
    let off = VsmSettings {
        enabled: false,
        ..settings()
    };
    let r = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &off, 4);
    assert_eq!(r.vsm_raster_frames(), 0);
    assert!(r.vsm_raster_stats().is_none());
    assert!(r.vsm().is_none());

    // …and a scene whose only light does not cast: the setting is ON and the pass
    // still never opens, which is the other half of the off path.
    let mut dark = scene(0, 0.5, 1.0);
    dark.lights[0].cast_shadows = false;
    dark.mark_dirty();
    let r = run(&gpu, &dark, &view(5.0), &settings(), 4);
    assert_eq!(r.vsm_raster_frames(), 0);
}

/// **The settings boundary, at the renderer's own door** (P27.2): an illegal
/// virtual-shadow configuration is refused and nothing is applied — not the legal
/// half either.
#[test]
fn the_renderer_refuses_an_illegal_shadow_configuration_whole() {
    let Some(gpu) = gpu_or_skip("the VSM settings boundary") else {
        return;
    };
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let before = *renderer.settings();
    let mut bad = before;
    // One legal change and one illegal one in the same block.
    bad.exposure = 3.25;
    bad.vsm = VsmSettings {
        clipmap_pages_per_side: 65_536,
        ..Default::default()
    };
    let err = renderer
        .try_set_settings(bad)
        .expect_err("a 2^32-page grid was accepted");
    assert!(
        matches!(err, inf_render::VsmSettingsError::PageSpace { .. }),
        "{err}"
    );
    assert_eq!(
        *renderer.settings(),
        before,
        "a refused settings block applied its legal half"
    );
    // The infallible door refuses too — it logs instead of returning.
    renderer.set_settings(bad);
    assert_eq!(*renderer.settings(), before);
    // ANTI-VACUITY: the same block minus the illegal field DOES apply.
    let mut good = bad;
    good.vsm = VsmSettings::default();
    assert!(renderer.try_set_settings(good).is_ok());
    assert_eq!(renderer.settings().exposure, 3.25);
}

// ── (i) the P27.2 audit's arms ──────────────────────────────────────────────

/// **THE POSE MARGIN, MEASURED** (P27.2 audit): the cull sphere a skinned caster
/// is tested with contains a pose that has **left** its bind-pose bound.
///
/// `SKINNED_POSE_MARGIN` carries the batch's own reasoning — "a skeleton moves
/// vertices, so a bind-pose bound is not conservative for an arbitrary pose, and
/// culling on a bound the pose escapes would delete a limb's shadow at exactly the
/// moment the limb moved" — and setting it to `0.0` survived every arm in this
/// file. It has to: a bound that is too tight only loses the pages the escaped limb
/// reached, and at the levels a 256 × 144 fixture marks, a page is metres wide.
///
/// So the assertion is on the **shipped caster record** — the sphere the GPU cull
/// actually ran — against the posed vertices the raster actually draws. Both halves
/// are needed: the pose is inside the margined sphere, and it is outside the
/// bind-pose one, which is what makes the margin the thing under test rather than
/// the sphere.
#[test]
fn a_skinned_casters_cull_sphere_contains_a_pose_that_left_the_bind_pose() {
    let Some(gpu) = gpu_or_skip("the VSM skinned pose margin") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let verts = [v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)];
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: verts.to_vec(),
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    // 0.6 m along x: the far corner ends up 1.89 m from the bind centre, which is
    // outside the bind sphere (1.41 m) and inside the margined one (2.12 m).
    let palette = glam::Mat4::from_translation(glam::Vec3::new(0.6, 0.0, 0.0));
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    s.skinned.push(inf_render::SkinnedInstance {
        translation: glam::DVec3::new(0.0, CUBE_XY.1 as f64, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0; 3],
        id: 3,
        mesh: 0,
        palette: vec![palette],
        vt: inf_render::VtTextureSet::NONE,
    });
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let renderer = run(&gpu, &s, &view(6.0), &settings_with(64), 6);
    let sys = renderer.vsm().expect("live");
    let casters = sys.raster_state().last_casters();
    // The rigid groups come first and are exactly one per primitive kind, so any
    // group past them is this scene's only other caster: the skinned quad.
    let c = casters
        .iter()
        .find(|c| c.ids[0] >= inf_render::VSM_RIGID_GROUPS)
        .expect("a skinned caster record was packed");
    let centre = glam::Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]);
    let radius = c.sphere[3];
    let model = glam::Mat4::from_cols_array(&c.model);
    let mut worst = 0f32;
    for vert in &verts {
        // What `vsm_skinned.wgsl` draws: the palette, then the caster's model.
        let skinned = (palette * glam::Vec3::from(vert.pos).extend(1.0)).truncate();
        worst = worst.max((model.transform_point3(skinned) - centre).length());
    }
    assert!(
        worst <= radius + 1e-4,
        "the posed quad reaches {worst} m from its cull sphere's centre and the \
         sphere is {radius} m — the page cull can delete this caster from a page \
         its own geometry covers"
    );
    // ANTI-VACUITY, and the whole point: the BIND pose's own bound does not contain
    // it. Without the margin the assertion above is the one that fails.
    let bind = radius / (1.0 + inf_render::SKINNED_POSE_MARGIN);
    assert!(
        worst > bind,
        "the fixture's pose stays inside the bind-pose bound ({worst} m against \
         {bind} m), so it proves nothing about the margin"
    );
    // …and the pose really is drawn, so this is a claim about a caster that casts.
    assert!(renderer.vsm_raster_stats().expect("stats").skinned_casters > 0);
}

/// **The caster ceiling counts what it refuses** (P27.2 audit) — `266acda`'s fix,
/// which shipped without an arm.
///
/// `VSM_MAX_CASTERS` was a bare `break` before that commit: a level past 16 384
/// casters had its tail stop casting with no counter and no log line. The fix added
/// the counter; nothing exercised it, and zeroing the increment survives every
/// other arm in this file because no fixture with one cube in it reaches the
/// ceiling. This one reaches it.
#[test]
fn the_caster_ceiling_counts_the_casters_it_refuses() {
    let Some(gpu) = gpu_or_skip("the VSM caster ceiling") else {
        return;
    };
    const OVER: u32 = 100;
    let mut s = scene(0, 0.5, 1.0);
    // A dense slab of cubes behind the fixture's own one. They do not have to be
    // visible — `pack_casters` walks the scene, not the frame — but they are packed
    // in scene order, so the ones past the ceiling are the ones refused.
    let total = inf_render::VSM_MAX_CASTERS + OVER;
    for i in 1..total {
        let (x, y) = ((i % 128) as f64 * 0.05 - 3.2, (i / 128) as f64 * 0.05 - 3.2);
        s.instances.push(inf_render::MeshInstance::lit(
            glam::DVec3::new(x, y, -1.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.04),
            [1.0, 1.0, 1.0, 1.0],
            100 + i,
        ));
    }
    s.mark_dirty();
    let renderer = run(&gpu, &s, &view(5.0), &settings(), 3);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.frames > 0, "the pass never opened: {stats:?}");
    assert_eq!(
        stats.casters,
        u64::from(inf_render::VSM_MAX_CASTERS) * stats.frames,
        "the ceiling did not bound the packed set: {stats:?}"
    );
    assert_eq!(
        stats.dropped_casters,
        u64::from(OVER) * stats.frames,
        "{} casters over the ceiling were refused and {} were counted — a silent \
         cap is how the far half of a level stops casting",
        u64::from(OVER) * stats.frames,
        stats.dropped_casters
    );
    // ANTI-VACUITY: the group ceiling is NOT what refused them, so the two counters
    // are telling different stories rather than one story twice.
    assert_eq!(stats.dropped_groups, 0, "{stats:?}");
    // …and the summary a host reads says the number.
    let line = renderer.vsm_summary().expect("a live system");
    assert!(
        line.contains(&format!("{} casters dropped", stats.dropped_casters)),
        "{line}"
    );
}

/// **A vgeom caster draws the level the CAMERA justifies** (P27.2 audit) — the
/// deviation memo's load-bearing sentence, measured.
///
/// `docs/memos/p27-2-vgeom-casters.md` rules that virtualized geometry casts
/// through "the same `pick_classic_level` against the same `lod_threshold`, at the
/// same `VgeomSettings::pixel_error`". Nothing could see it: a caster drawn from
/// the coarsest level of the chain lands in the same pages at almost the same
/// depths, so replacing the pick with `errors.len() - 1` survived every arm in this
/// file. `VsmRasterStats::vgeom_level_sum` is the counter that makes the sentence a
/// measurement, and `pixel_error` is the input the ruling names.
#[test]
fn a_vgeom_casters_level_is_the_one_its_pixel_error_justifies() {
    let Some(gpu) = gpu_or_skip("the VSM vgeom LOD") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![
            inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh")
        ],
        ..Default::default()
    };
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::ZERO,
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(3.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(64);
    let at = |pixel_error: f32| {
        let r = run_tuned(&gpu, &[(&s, 6)], &view(9.0), &set, move |rs| {
            rs.vgeom.pixel_error = pixel_error;
        });
        r.vsm_raster_stats().expect("stats")
    };
    // A tenth of a pixel of tolerated error: the finest level of the chain.
    let fine = at(0.1);
    // …and a tolerance no level can miss: the coarsest.
    let coarse = at(400.0);
    assert!(
        fine.vgeom_casters > 0 && coarse.vgeom_casters == fine.vgeom_casters,
        "the two runs packed different caster sets ({fine:?} / {coarse:?})"
    );
    assert_eq!(
        fine.vgeom_level_sum, 0,
        "a caster whose pixel error is a tenth of a pixel drew a level past the \
         finest: {fine:?}"
    );
    assert!(
        coarse.vgeom_level_sum > 0,
        "the same caster at 400 px of tolerated error drew the same level — the \
         page raster is not picking through `pick_classic_level` at all: {coarse:?}"
    );
}

/// **A frame with pages and no casters CLEARS them** (P27.2 audit).
///
/// The pass used to return early when nothing packed, on the reasoning that "the
/// pass below clears the whole atlas, so this early return is only taken when there
/// is no pass to open at all". It is taken when there are pages and no casters —
/// and the editor's infinite grid is exactly that configuration: it writes camera
/// depth, so it marks pages, and it is not a caster. Delete every object in a level
/// and the atlas goes on holding their shadows.
#[test]
fn a_frame_with_no_caster_clears_the_pages_the_last_one_filled() {
    let Some(gpu) = gpu_or_skip("the VSM caster-less clear") else {
        return;
    };
    let mut filled = scene(0, 0.5, 1.0);
    filled.grid_enabled = true;
    filled.mark_dirty();
    // The same scene with its one object deleted. The grid still writes depth, so
    // pages are still marked and still resident.
    let mut emptied = filled.clone();
    emptied.instances.clear();
    emptied.mark_dirty();

    let set = settings_with(64);
    let v = view(5.0);
    let renderer = run_sequence(&gpu, &[(&filled, 6), (&emptied, 6)], &v, &set);
    let stats = renderer.vsm_raster_stats().expect("stats");
    let pages = resident_pages(&renderer);
    // ANTI-VACUITY, three ways: pages are resident, the first half really drew
    // casters, and the pass opened **again** once they were gone.
    //
    // "Again" rather than "on every frame" since P27.3: the pass opens for the
    // frames whose content stamps moved, and deleting every object moves them all
    // exactly once. `frames >= 2` is therefore the claim — the filled half filled
    // them, the emptied half cleared them — and `cached_pages > 0` proves the
    // steady state in between really was the cache and not a pass drawing nothing.
    assert!(!pages.is_empty(), "nothing was resident to clear");
    assert!(
        stats.casters > 0,
        "the filled half packed nothing: {stats:?}"
    );
    assert!(
        stats.frames >= 2,
        "the pass stopped opening once the casters were gone: {stats:?}"
    );
    assert!(
        stats.cached_pages > 0,
        "no frame was served by the cache, so this is P27.2's every-frame raster \
         rather than P27.3's: {stats:?}"
    );

    let atlas = read_atlas(&gpu, &renderer);
    let left: usize = pages
        .iter()
        .map(|(_, _, r)| written(&atlas, *r).len())
        .sum();
    assert_eq!(
        left, 0,
        "{left} texels still hold a deleted object's depth — a caster-less frame \
         left the atlas as it found it"
    );
}

/// **The group ceiling counts what it refuses too** (P27.2 audit).
///
/// `VSM_MAX_GROUPS` is the ceiling the first write-up did not have: the
/// per-(page, group) draw uniform is `pages x groups x 256 B`, and a skinned
/// instance is a group because its palette is a bind group. A thousand characters
/// is a thousand groups, and without a ceiling that buffer passes what a default
/// device will allocate.
///
/// So the fixture is a thousand and thirty characters, sharing one two-triangle
/// mesh: the ceiling has to refuse eleven of them, and it has to say so in both
/// counters.
#[test]
fn the_group_ceiling_counts_the_groups_it_refuses() {
    let Some(gpu) = gpu_or_skip("the VSM group ceiling") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-0.1, -0.1), v(0.1, -0.1), v(0.1, 0.1), v(-0.1, 0.1)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    // Enough instances that the rigid groups plus the skinned ones overrun the
    // ceiling by a countable margin.
    const OVER: u32 = 11;
    let want = inf_render::VSM_MAX_GROUPS + OVER - inf_render::VSM_RIGID_GROUPS;
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    for i in 0..want {
        s.skinned.push(inf_render::SkinnedInstance {
            translation: glam::DVec3::new((i % 32) as f64 * 0.2 - 3.2, (i / 32) as f64 * 0.2, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 1_000 + i,
            mesh: 0,
            palette: vec![glam::Mat4::IDENTITY],
            vt: inf_render::VtTextureSet::NONE,
        });
    }
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let renderer = run(&gpu, &s, &view(6.0), &settings(), 3);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.frames > 0, "the pass never opened: {stats:?}");
    assert_eq!(
        stats.dropped_groups,
        u64::from(OVER) * stats.frames,
        "{} groups over the ceiling and {} counted: {stats:?}",
        u64::from(OVER) * stats.frames,
        stats.dropped_groups
    );
    // Every refused group took its caster with it, and the two counters agree.
    assert_eq!(stats.dropped_casters, stats.dropped_groups, "{stats:?}");
    assert_eq!(
        stats.skinned_casters,
        u64::from(want - OVER) * stats.frames,
        "{stats:?}"
    );
    // ANTI-VACUITY: the CASTER ceiling is not what refused them — a thousand
    // characters is nowhere near 16 384 — so this is the group ceiling's own arm.
    assert!(stats.casters < u64::from(inf_render::VSM_MAX_CASTERS) * stats.frames);
    // …and the pages still rasterized, so the ceiling refused a tail rather than
    // taking the frame down.
    assert!(
        stats.draws > 0 && renderer.vsm_raster_frames() > 0,
        "{stats:?}"
    );
}
