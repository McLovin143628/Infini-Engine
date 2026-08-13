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

use inf_render::{
    vsm_page_matrix, GpuContext, RenderScene, RenderView, VsmLightHandle, VsmPage, VsmSettings,
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
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    s.vsm = *set;
    renderer.set_settings(s);
    for _ in 0..frames {
        renderer.render(gpu, scene, v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
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
    let (w, _, ref data) = *atlas;
    let mut out = Vec::new();
    for y in rect.1..rect.1 + rect.2 {
        for x in rect.0..rect.0 + rect.2 {
            let d = data[(y * w + x) as usize];
            if d != inf_render::VSM_DEPTH_CLEAR {
                out.push(d);
            }
        }
    }
    out
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
    assert!(renderer.vsm_raster_frames() >= 3, "{stats:?}");
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
    let sys = renderer.vsm().expect("live");
    let mut checked = 0;
    let mut total_written = 0usize;
    for (light, page, rect) in &pages {
        let desc = sys
            .residency()
            .desc(VsmLightHandle(*light))
            .expect("registered");
        let g = desc.levels[page.level as usize];
        let base = sys.projections()[0];
        let vp = vsm_page_matrix(
            glam::Mat4::from_cols_array(&base.view_proj),
            desc.kind,
            page.level,
            g.pages_x,
            g.pages_y,
            page.x,
            page.y,
        );
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
        let desc = sys.residency().desc(*light).expect("registered");
        let g = desc.levels[page.level as usize];
        let vp = vsm_page_matrix(
            glam::Mat4::from_cols_array(&sys.projections()[0].view_proj),
            desc.kind,
            page.level,
            g.pages_x,
            g.pages_y,
            page.x,
            page.y,
        );
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
        let sys = r.vsm().expect("live");
        let mut n = 0usize;
        for (light, page, rect) in resident_pages(&r) {
            let desc = sys
                .residency()
                .desc(VsmLightHandle(light))
                .expect("registered");
            let g = desc.levels[page.level as usize];
            let vp = vsm_page_matrix(
                glam::Mat4::from_cols_array(&sys.projections()[0].view_proj),
                desc.kind,
                page.level,
                g.pages_x,
                g.pages_y,
                page.x,
                page.y,
            );
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
    let sys = renderer.vsm().expect("live");
    let mut vgeom_texels = 0usize;
    for (light, page, rect) in resident_pages(&renderer) {
        let desc = sys
            .residency()
            .desc(VsmLightHandle(light))
            .expect("registered");
        let g = desc.levels[page.level as usize];
        let vp = vsm_page_matrix(
            glam::Mat4::from_cols_array(&sys.projections()[0].view_proj),
            desc.kind,
            page.level,
            g.pages_x,
            g.pages_y,
            page.x,
            page.y,
        );
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
    let csys = control.vsm().expect("live");
    let mut control_texels = 0usize;
    for (light, page, rect) in resident_pages(&control) {
        let desc = csys
            .residency()
            .desc(VsmLightHandle(light))
            .expect("registered");
        let g = desc.levels[page.level as usize];
        let vp = vsm_page_matrix(
            glam::Mat4::from_cols_array(&csys.projections()[0].view_proj),
            desc.kind,
            page.level,
            g.pages_x,
            g.pages_y,
            page.x,
            page.y,
        );
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

// ── (f) off path, and the settings door ─────────────────────────────────────

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
