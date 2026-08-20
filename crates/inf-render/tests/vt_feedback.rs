//! **The P26.4 streaming loop** — the analytic floor, the GPU coverage feedback
//! that refines it, and the degradation between them.
//!
//! What these arms are built to falsify:
//!
//! * a feedback pass that marks **nothing** (an empty mask decodes to no wants,
//!   the floor stands, and every "PIE == shipping" comparison still passes —
//!   which is exactly the shape of a feature that does nothing);
//! * a feedback pass that marks the **wrong level** (the CPU floor and the GPU
//!   refinement would then disagree about what a frame needs, which reads as
//!   permanent pop-in rather than as a wrong number);
//! * a floor that is not a floor (a refinement taking a floor tile's page, or a
//!   dropped mask changing the trace).
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.

use std::sync::Arc;

use inf_render::vt::VtPools;
use inf_render::vt_stream::{FeedbackRequest, VtFeedback};
use inf_render::{
    analytic_floor, feedback_requests, justified_mip, projection_scale, screen_diameter_px,
    GpuContext, VtCoverage, VtMaterialMaps, VtTextures, VtTileSource, VT_FEEDBACK_MAX_TILES,
};
use inf_vt::{
    PageFormat, TileCoord, VtFeedbackLayout, VtPoolConfig, VtTextureHandle, VT_PRIORITY_FEEDBACK,
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

/// A real v2 container of `n × n`, through the one writer.
fn tiled(n: u32) -> Vec<u8> {
    inf_material::build_tiled_texture(
        vec![190u8; (n * n * 4) as usize],
        n,
        n,
        inf_material::TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// A registry holding one `n × n` texture bound to material `1`, warm.
fn library(n: u32, budget_pages: u64) -> VtTextures {
    let (mut lib, _) = VtTextures::new(VtPoolConfig {
        format: PageFormat::Rgba8,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        budget_bytes: PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * budget_pages,
        max_texture_dim: 8192,
        trilinear: false,
        // **Unthrottled** (IB-16): every arm in this file is about the WANT
        // half — the analytic floor, the coverage mask, the floor law — and a
        // per-frame upload budget bounds what is SEATED. Throttling here would
        // make each of them a statement about flow control on top of the thing
        // it names, and the floor law's own arm would stop distinguishing "the
        // floor was not asked for" from "the floor was asked for and deferred".
        upload_budget_bytes: 0,
    });
    let bytes = Arc::new(tiled(n)) as Arc<dyn VtTileSource>;
    let mut mats = std::collections::BTreeMap::new();
    mats.insert(
        1u128,
        VtMaterialMaps {
            albedo: Some(7),
            normal: None,
            orm: None,
            detail: None,
            detail_scale_q8: 0,
        },
    );
    assert_eq!(
        lib.register_materials(&mats, |_| Some(bytes.clone())),
        1,
        "the fixture texture did not register"
    );
    lib
}

/// A head-on view of a surface at the origin.
fn view(width: u32, height: u32, eye_z: f64) -> inf_render::RenderView {
    inf_render::RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 0.0, eye_z),
        forward: glam::Vec3::new(0.0, 0.0, -1.0),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width,
        height,
        ortho: None,
    }
}

// ── (a) the floor, on the CPU ───────────────────────────────────────────────

/// **The analytic floor tracks the camera, is bounded, and never falls below the
/// camera-free one** (P26.4, clause 4).
///
/// Three claims at once, and each is a different way the floor could be wrong:
/// it must contain `want_floor`'s camera-free levels for EVERY registered
/// texture (so an off-screen surface still resolves), it must add finer levels
/// for a surface that is close, and it must add nothing at all for one that is
/// behind the camera.
#[test]
fn the_analytic_floor_is_camera_driven_bounded_and_never_below_the_camera_free_one() {
    let lib = library(1024, 512);
    let handle = VtTextureHandle(0);
    let bare = lib.want_floor();
    assert!(!bare.is_empty());

    let near = VtCoverage {
        centre: glam::Vec3::ZERO,
        radius: 1.0,
        set: lib.set_for(Some(7), None, None),
        vgeom: false,
    };
    // Cold: the registry is not warm yet, so no set names anything and the
    // coverage contributes nothing — the warm gate, reaching this far.
    assert!(near.set.is_none());

    let mut lib = lib;
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(7), None, None);
    assert!(!set.is_none(), "the fixture never went warm");

    let close = analytic_floor(
        &lib,
        &view(1920, 1080, 2.0),
        &[VtCoverage {
            centre: glam::Vec3::ZERO,
            radius: 1.0,
            set,
            vgeom: false,
        }],
    );
    let far = analytic_floor(
        &lib,
        &view(1920, 1080, 400.0),
        &[VtCoverage {
            centre: glam::Vec3::ZERO,
            radius: 1.0,
            set,
            vgeom: false,
        }],
    );
    let behind = analytic_floor(
        &lib,
        &view(1920, 1080, 2.0),
        &[VtCoverage {
            centre: glam::Vec3::new(0.0, 0.0, 500.0),
            radius: 1.0,
            set,
            vgeom: false,
        }],
    );

    // Every camera-free want survives in all three.
    for w in &bare {
        for (label, set) in [("close", &close), ("far", &far), ("behind", &behind)] {
            assert!(
                set.iter()
                    .any(|x| x.texture == w.texture && x.tile == w.tile),
                "the {label} floor dropped a camera-free want {:?}",
                w.tile
            );
        }
    }
    // …a close surface asks for MORE, and a far one for less.
    assert!(
        close.len() > far.len(),
        "the floor did not track the camera: close {} vs far {}",
        close.len(),
        far.len()
    );
    assert_eq!(
        behind.len(),
        bare.len(),
        "a surface behind the camera contributed to the floor"
    );
    // …and the camera-driven part is BOUNDED: at most `VT_FLOOR_MAX_TILES` per
    // surface map, which is what lets it be claimed unconditionally.
    assert!(
        close.len() - bare.len() <= inf_render::VT_FLOOR_MAX_TILES as usize,
        "one surface claimed {} floor pages",
        close.len() - bare.len()
    );
    // …and every want it produced is a FLOOR want, or the priority split that
    // makes it a floor is a statement about nothing.
    assert!(close
        .iter()
        .all(|w| w.priority == inf_vt::VT_PRIORITY_FLOOR));
    // Anti-vacuity: the close floor really does reach a finer level than the
    // camera-free one — its finest mip is strictly below the coarsest three.
    let finest = |v: &[inf_vt::VtWant]| v.iter().map(|w| w.tile.mip).min().unwrap_or(u32::MAX);
    assert!(
        finest(&close) < finest(&bare),
        "the close floor reached no finer level than the camera-free one"
    );
    let _ = handle;
}

// ── (b) the GPU feedback marks the level the CPU rule justifies ─────────────

/// **THE FEEDBACK ARM**: the compute pass marks exactly the tiles of the level
/// `justified_mip` names, for a surface on screen, and marks **nothing** for one
/// that is off it.
///
/// This is the arm that fails when the pass does nothing (an empty mask decodes
/// to no wants and every other arm in the phase still passes), when it marks the
/// wrong level (the floor and the refinement then disagree for ever), and when
/// its frustum test is missing (an off-screen surface would page at full detail).
///
/// The mask comes back through the **ring**, at the pinned latency, so the arm
/// also exercises the path the renderer takes rather than a direct buffer read.
#[test]
fn the_feedback_pass_marks_the_level_the_camera_justifies() {
    let Some(gpu) = gpu_or_skip("the VT feedback pass") else {
        return;
    };
    let mut lib = library(1024, 512);
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let _ = lib.sync(&gpu.device, &gpu.queue, &mut pools, &[]);
    let set = lib.set_for(Some(7), None, None);
    assert!(!set.is_none());

    let layout = VtFeedbackLayout::for_residency(lib.residency());
    let mut feedback = VtFeedback::new(&gpu.device, layout, 64);
    let v = view(1920, 1080, 2.0);

    let run = |feedback: &mut VtFeedback,
               lib: &VtTextures,
               pools: &VtPools,
               coverage: &[VtCoverage],
               frame: u64| {
        let requests = feedback_requests(lib, feedback.layout(), coverage, false);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let n = feedback.record(
            &gpu.device,
            &gpu.queue,
            &mut enc,
            pools.table(),
            pools.table_generation(),
            &v,
            &requests,
        );
        feedback.finish(&mut enc, frame, n > 0);
        gpu.queue.submit([enc.finish()]);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        (n, requests.len())
    };

    // ON SCREEN, close: the pass must mark the level the CPU rule names.
    let on = [VtCoverage {
        centre: glam::Vec3::ZERO,
        radius: 1.0,
        set,
        vgeom: false,
    }];
    let (dispatched, requested) = run(&mut feedback, &lib, &pools, &on, 0);
    assert_eq!((dispatched, requested), (1, 1), "one surface, one map");
    // …read it back at the pinned latency: frame 0's mask is frame 2's answer.
    let mut wants = None;
    for _ in 0..64 {
        if let Some(w) = feedback.take_wants(&gpu.device, &lib, 2) {
            wants = Some(w);
            break;
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let wants = wants.expect("the ring never delivered frame 0's mask");
    assert!(
        !wants.is_empty(),
        "the feedback pass marked NOTHING — an empty mask is indistinguishable \
         from a scene with no feedback at all"
    );
    assert!(
        wants.iter().all(|w| w.priority == VT_PRIORITY_FEEDBACK),
        "a feedback want is not a refinement, so it could outrank the floor"
    );

    // The level it chose is the level the CPU rule chooses, from the same inputs.
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered");
    let px = screen_diameter_px(glam::Vec3::ZERO, 1.0, v.eye_local(), projection_scale(&v));
    let mut want_lv = justified_mip(
        desc.mips[0].width.max(desc.mips[0].height),
        px,
        desc.mip_count(),
    );
    while want_lv + 1 < desc.mip_count()
        && desc.mips[want_lv as usize].tile_count() > VT_FEEDBACK_MAX_TILES
    {
        want_lv += 1;
    }
    let levels: std::collections::BTreeSet<u32> = wants.iter().map(|w| w.tile.mip).collect();
    assert_eq!(
        levels,
        [want_lv].into_iter().collect(),
        "the GPU marked levels {levels:?} where the CPU rule says {want_lv} — the \
         floor and the refinement disagree about what this frame needs"
    );
    assert_eq!(
        wants.len(),
        desc.mips[want_lv as usize].tile_count() as usize,
        "the pass marked a partial level"
    );
    // ANTI-VACUITY on the level itself: the justified level is not mip 0 and not
    // the coarsest, or "it picked the right one" is a statement about a constant.
    assert!(
        want_lv > 0 && want_lv + 1 < desc.mip_count(),
        "the fixture's justified level is at an end of the pyramid ({want_lv} of \
         {}), so this arm cannot tell a rule from a clamp",
        desc.mip_count()
    );

    // OFF SCREEN, two ways, because the pass rejects them with two different
    // tests and one fixture exercises only one of them. Measured: deleting the
    // NDC test alone left the behind-the-camera case passing, because that one
    // is caught by the `clip.w <= 0` branch three lines earlier.
    //
    //  * BEHIND the eye            -> the `clip.w <= 0` branch;
    //  * far to the SIDE, in front -> the NDC-box branch.
    for (label, centre, frame, read_at) in [
        (
            "behind the camera",
            glam::Vec3::new(0.0, 0.0, 500.0),
            3u64,
            5u64,
        ),
        (
            "beside the frustum",
            glam::Vec3::new(400.0, 0.0, -20.0),
            6,
            8,
        ),
    ] {
        let off = [VtCoverage {
            centre,
            radius: 1.0,
            set,
            vgeom: false,
        }];
        let (n, _) = run(&mut feedback, &lib, &pools, &off, frame);
        assert_eq!(n, 1, "{label}: the request was not even dispatched");
        let mut got = None;
        for _ in 0..64 {
            if let Some(w) = feedback.take_wants(&gpu.device, &lib, read_at) {
                got = Some(w);
                break;
            }
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        assert_eq!(
            got.unwrap_or_else(|| panic!("{label}: the ring never delivered"))
                .len(),
            0,
            "a surface {label} was marked, so the pass pages at full detail whatever the camera is looking at"
        );
    }
}

// ── (c) the floor is a floor ────────────────────────────────────────────────

/// **A refinement can never take a floor tile's page, and a frame with no
/// feedback produces exactly the floor's transaction.**
///
/// The determinism claim, made executable. `apply_wants` is fed the floor alone
/// and then the floor plus a refinement set large enough to exhaust the pool;
/// every floor tile must still be resident afterwards, and the trace of the
/// floor-only frame must be the trace a dropped mask produces — which is what
/// "degrades to the floor, deterministically" has to mean.
#[test]
fn a_refinement_never_evicts_the_floor_and_a_dropped_mask_is_the_floors_trace() {
    // A pool with room for the floor and a little else.
    let mut lib = library(1024, 40);
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(7), None, None);
    let v = view(1920, 1080, 2.0);
    let coverage = [VtCoverage {
        centre: glam::Vec3::ZERO,
        radius: 1.0,
        set,
        vgeom: false,
    }];
    let floor = analytic_floor(&lib, &v, &coverage);

    // The floor alone.
    // The residency is `Clone`, so the three runs below are three copies of ONE
    // state — which is what makes them a comparison of the want sets rather than
    // of three separately-built worlds.
    let mut a = lib.residency().clone();
    let floor_only = a.apply_wants(&floor).trace();

    // The floor plus every mip-0 tile as a refinement — far more than the pool
    // can hold, so something MUST be deferred.
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered")
        .clone();
    let mut greedy = floor.clone();
    for y in 0..desc.mips[0].tiles_y {
        for x in 0..desc.mips[0].tiles_x {
            greedy.push(inf_vt::VtWant::refine(
                VtTextureHandle(0),
                TileCoord::new(0, x, y),
            ));
        }
    }
    let mut b = lib.residency().clone();
    let txn = b.apply_wants(&greedy);
    assert!(
        txn.deferred > 0,
        "the fixture's refinement fits the pool, so this arm cannot see a \
         refinement outrank the floor"
    );
    for w in &floor {
        assert!(
            b.is_resident(w.texture, w.tile),
            "a refinement took floor tile {:?}'s page",
            w.tile
        );
    }

    // …and a frame whose mask never arrived is the floor-only frame, exactly.
    let mut c = lib.residency().clone();
    assert_eq!(
        c.apply_wants(&floor).trace(),
        floor_only,
        "two runs of the floor produced different transactions"
    );
    assert!(
        !floor_only.is_empty(),
        "the floor's trace is empty, so the equality above is about nothing"
    );
}

/// A request list is one entry per (surface × bound map), carrying the table
/// offsets the shader would otherwise have to re-derive.
#[test]
fn a_request_is_one_entry_per_bound_map() {
    let mut lib = library(512, 128);
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(7), None, None);
    let layout = VtFeedbackLayout::for_residency(lib.residency());
    let coverage = vec![
        VtCoverage {
            centre: glam::Vec3::ZERO,
            radius: 1.0,
            set,
            vgeom: false,
        },
        VtCoverage {
            centre: glam::Vec3::X,
            radius: 2.0,
            set,
            vgeom: false,
        },
        // A surface that names nothing contributes no request at all.
        VtCoverage {
            centre: glam::Vec3::Y,
            radius: 1.0,
            set: inf_render::VtTextureSet::NONE,
            vgeom: false,
        },
    ];
    let reqs: Vec<FeedbackRequest> = feedback_requests(&lib, &layout, &coverage, false);
    assert_eq!(reqs.len(), 2, "one entry per bound map, and no more");
    assert_eq!(reqs[0].centre[3], 1.0);
    assert_eq!(reqs[1].centre[3], 2.0);
    // The block offset is the table's own, and the base bit the layout's — a
    // shader that re-derived either would be a second copy of the layout.
    let (block, _) = lib
        .residency()
        .table_block(VtTextureHandle(0))
        .expect("registered");
    assert_eq!(reqs[0].tex[0], block as u32);
    assert_eq!(
        reqs[0].tex[1],
        layout.texture_base(VtTextureHandle(0)).unwrap()
    );
}

// ── (d) end to end: the RENDERER runs the loop ──────────────────────────────

/// **The whole loop, through `EngineRenderer::render`** — the arm that fails if
/// the renderer never calls any of it (P26.4).
///
/// Everything above drives `VtFeedback`, `analytic_floor` and the residency
/// directly, which is where the rules live. What none of them can say is whether
/// the renderer *runs* them: a `vt_stream` that returned early on every frame
/// would leave every arm above green, the goldens green, and the feature dead —
/// which is the exact shape the P26.3 audit found when the batch that bound VT
/// into the lit passes had never rendered a fragment.
///
/// So this renders real frames and asserts on the WORLD afterwards: the loop ran
/// once per frame, it admitted pages, the ring landed, and residency reached a
/// level the camera-free floor never asks for. Plus the control that makes all
/// four measurements of the loop rather than of a renderer: a textureless
/// renderer's counters stay at zero.
#[test]
fn the_renderer_runs_the_streaming_loop_every_frame() {
    let Some(gpu) = gpu_or_skip("the VT streaming loop") else {
        return;
    };
    const FW: u32 = 320;
    const FH: u32 = 180;
    const FRAMES: u64 = 8;

    let mut lib = library(1024, 512);
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    // The transaction that carries the roots, so `set_for` may name the texture.
    let _ = lib.sync(&gpu.device, &gpu.queue, &mut pools, &[]);
    let set = lib.set_for(Some(7), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    // The camera-free floor's finest level — what residency would stop at if the
    // camera-driven half and the feedback both did nothing.
    let floor_finest = lib
        .want_floor()
        .iter()
        .map(|w| w.tile.mip)
        .min()
        .expect("a floor");

    let mut scene = inf_render::RenderScene::default();
    scene.instances.push(inf_render::MeshInstance {
        vt: set,
        translation: glam::DVec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::splat(2.0),
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 1,
        mesh: inf_render::PrimMesh::Cube,
        blend: 0,
        cutoff: 0.5,
    });
    scene.mark_dirty();
    let v = view(FW, FH, 1.8);

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    renderer.set_vt_level(Some((lib, pools)));
    for _ in 0..FRAMES {
        renderer.render(&gpu, &scene, &v, &target.view, (FW, FH));
        // A host's frame pacing; here, the poll that lets the ring's maps fire.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let stats = renderer.vt_pop_in();
    assert_eq!(
        stats.frames,
        FRAMES,
        "the renderer did not run the streaming loop once per frame: {}",
        stats.summary()
    );
    assert!(
        stats.admits > 0,
        "the loop admitted no page at all: {}",
        stats.summary()
    );
    assert!(
        stats.feedback_frames > 0,
        "the feedback ring never landed in {FRAMES} frames, so every frame took \
         the floor and the GPU pass is unobserved: {}",
        stats.summary()
    );
    assert_eq!(
        stats.feedback_frames + stats.feedback_misses,
        FRAMES,
        "a frame neither read a mask nor recorded a miss"
    );
    assert!(
        stats.floor_wants > 0 && stats.refine_wants > 0,
        "one of the two want classes is empty, so the per-class pop-in counters \
         are measuring one thing: {}",
        stats.summary()
    );

    // ASSERT THE WORLD: residency reached a level the camera-free floor never
    // asks for, which is the whole claim — a camera-driven floor plus feedback
    // is sharper than a pyramid tail.
    let lib = renderer.vt_textures().expect("the level is live");
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered")
        .clone();
    let mut finest = u32::MAX;
    for mip in 0..desc.mip_count() {
        let m = desc.mips[mip as usize];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                if lib
                    .residency()
                    .is_resident(VtTextureHandle(0), TileCoord::new(mip, x, y))
                {
                    finest = finest.min(mip);
                }
            }
        }
    }
    assert!(
        finest < floor_finest,
        "residency stopped at mip {finest} and the camera-free floor already \
         reaches {floor_finest} — nothing refined it"
    );

    // THE CONTROL: the same renderer with no VT level runs none of it. This is
    // what makes "the 50 goldens did not change" a measurement of the command
    // stream rather than an expectation.
    let mut bare = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut plain = inf_render::RenderScene::default();
    plain.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::ZERO,
        glam::Quat::IDENTITY,
        glam::Vec3::splat(2.0),
        [1.0, 1.0, 1.0, 1.0],
        1,
    ));
    plain.mark_dirty();
    for _ in 0..FRAMES {
        bare.render(&gpu, &plain, &v, &target.view, (FW, FH));
    }
    assert_eq!(
        bare.vt_pop_in(),
        inf_render::VtPopIn::default(),
        "a textureless frame entered the streaming loop: {}",
        bare.vt_pop_in().summary()
    );
    assert_eq!(bare.vt_engaged_frames(), 0);
}

// ── (e) the audit's arms ────────────────────────────────────────────────────

/// Record one feedback frame over `coverage` and read its mask back at the
/// pinned latency, pumping the device to completion so the arrival pattern is a
/// constant rather than a race. `frame` must advance by at least the ring's slot
/// count between calls.
fn run_feedback(
    gpu: &GpuContext,
    feedback: &mut VtFeedback,
    lib: &VtTextures,
    pools: &VtPools,
    v: &inf_render::RenderView,
    coverage: &[VtCoverage],
    frame: u64,
) -> Vec<inf_vt::VtWant> {
    let requests = feedback_requests(lib, feedback.layout(), coverage, false);
    assert!(!requests.is_empty(), "the fixture's coverage binds no map");
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let n = feedback.record(
        &gpu.device,
        &gpu.queue,
        &mut enc,
        pools.table(),
        pools.table_generation(),
        v,
        &requests,
    );
    feedback.finish(&mut enc, frame, n > 0);
    assert_eq!(
        n as usize,
        requests.len(),
        "the pass dispatched {n} of {} requests",
        requests.len()
    );
    gpu.queue.submit([enc.finish()]);
    for _ in 0..64 {
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(w) = feedback.take_wants(
            &gpu.device,
            lib,
            frame + inf_render::readback::READBACK_LATENCY_FRAMES,
        ) {
            return w;
        }
    }
    panic!("the ring never delivered frame {frame}'s mask");
}

/// **The feedback's camera term IS the floor's**, position by position (P26.4
/// audit).
///
/// `vt_stream::on_screen` is documented as *"the CPU twin of the shader's
/// frustum test"* and was not one. Two divergences, both measured on the shipped
/// pair, and both invisible to every arm the batch shipped because both are in
/// the *"the floor keeps more than the feedback does"* direction — which is
/// never a crash, never a wrong pixel, and always a surface that is quietly
/// stuck at the floor's level:
///
///  * the shader's NDC margin was `r_px / (height/2)` where the floor's is
///    `2·r_px / (height/2)`, so there is a band at the edge of the frustum the
///    floor claims and the feedback dropped;
///  * the shader returned unconditionally on `clip.w <= 0`, where `on_screen`
///    keeps a sphere that **straddles** the eye. That is the ordinary state of
///    the terrain-sized quad `docs/memos/p26-4-feedback-mechanism.md` names as
///    this signal's worst case: a camera standing on a 200 m ground plane is
///    *inside* its bounding sphere, so that surface was never refined at all.
///
/// Five positions, the last of them **bisected** into the old margin band rather
/// than guessed, so the fixture cannot drift into agreement as the projection
/// changes.
#[test]
fn the_feedback_and_the_floor_agree_about_what_is_on_screen() {
    let Some(gpu) = gpu_or_skip("the VT frustum twin") else {
        return;
    };
    let mut lib = library(1024, 512);
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let _ = lib.sync(&gpu.device, &gpu.queue, &mut pools, &[]);
    let set = lib.set_for(Some(7), None, None);
    assert!(!set.is_none(), "the fixture never went warm");

    let v = view(1920, 1080, 2.0);
    let vp = v.view_proj();
    let eye = v.eye_local();
    let ps = projection_scale(&v);
    let half_h = v.height as f32 * 0.5;
    let layout = VtFeedbackLayout::for_residency(lib.residency());
    let mut feedback = VtFeedback::new(&gpu.device, layout, 64);

    // The position inside the margin band the two used to disagree about: NDC x
    // three quarters of the way from the shader's old half-margin to the floor's
    // full one. Bisected, because `ndc.x` is monotone in world x.
    let z = -18.0f32;
    let (mut lo, mut hi) = (0.0f32, 400.0f32);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let c = glam::Vec3::new(mid, 0.0, z);
        let clip = vp * c.extend(1.0);
        let px = screen_diameter_px(c, 1.0, eye, ps);
        let target = 1.0 + 0.75 * (px / half_h);
        if clip.x / clip.w < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let band = glam::Vec3::new(0.5 * (lo + hi), 0.0, z);
    let band_px = screen_diameter_px(band, 1.0, eye, ps);
    // ANTI-VACUITY on the fixture: this position really is inside the band — the
    // floor keeps it and the shader's OLD margin would have dropped it.
    assert!(
        inf_render::on_screen(&vp, band, 1.0, band_px / half_h),
        "the bisection left the floor's frustum"
    );
    assert!(
        !inf_render::on_screen(&vp, band, 1.0, 0.5 * band_px / half_h),
        "the bisected position is not inside the margin the shader used to use, \
         so this arm cannot see that half of the divergence"
    );

    let cases: [(&str, glam::Vec3, f32, bool); 5] = [
        ("head-on", glam::Vec3::ZERO, 1.0, true),
        (
            "far behind the eye",
            glam::Vec3::new(0.0, 0.0, 500.0),
            1.0,
            false,
        ),
        (
            "beside the frustum",
            glam::Vec3::new(400.0, 0.0, -20.0),
            1.0,
            false,
        ),
        // The ground the camera stands on: centre behind, sphere containing the
        // eye. `clip.w < 0` and `radius > -clip.w`.
        (
            "straddling the eye",
            glam::Vec3::new(0.0, 0.0, 40.0),
            60.0,
            true,
        ),
        ("inside the old margin band", band, 1.0, true),
    ];
    let mut marked_any = false;
    let mut dropped_any = false;
    for (i, (label, centre, radius, want)) in cases.into_iter().enumerate() {
        // The floor's own verdict, from the same numbers the shader is handed.
        let px = screen_diameter_px(centre, radius, eye, ps);
        assert_eq!(
            inf_render::on_screen(&vp, centre, radius, px / half_h),
            want,
            "{label}: the CPU floor does not agree with the fixture"
        );
        let coverage = [VtCoverage {
            centre,
            radius,
            set,
            vgeom: false,
        }];
        let wants = run_feedback(
            &gpu,
            &mut feedback,
            &lib,
            &pools,
            &v,
            &coverage,
            i as u64 * 3,
        );
        assert_eq!(
            !wants.is_empty(),
            want,
            "{label}: the GPU marked {} tiles and the floor says on-screen = {want} — \
             the two disagree about which surfaces exist, so one of them is paying \
             for pages the other will not refine",
            wants.len()
        );
        marked_any |= !wants.is_empty();
        dropped_any |= wants.is_empty();
    }
    // ANTI-VACUITY: the sweep contains both answers, so it is not five copies of
    // one verdict.
    assert!(marked_any && dropped_any);
}

/// **The mask is order-independent on the GPU, not only in its CPU twin**
/// (P26.4 audit).
///
/// `inf_vt::feedback`'s own arm reorders `mark` calls on the CPU. What it cannot
/// say is whether the *pass* is order-independent: the request list is uploaded
/// in coverage order, the threads that read it run in whatever order the device
/// chooses, and the claim the whole replay-doctrine reconciliation rests on is
/// that neither matters. Reversing the list is the cheapest way to falsify it,
/// and it is a permutation the scene genuinely produces (the coverage list is
/// built from the render scene's own instance order).
#[test]
fn the_coverage_mask_does_not_depend_on_the_request_order() {
    let Some(gpu) = gpu_or_skip("the VT mask's order independence") else {
        return;
    };
    let mut lib = library(1024, 512);
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let _ = lib.sync(&gpu.device, &gpu.queue, &mut pools, &[]);
    let set = lib.set_for(Some(7), None, None);
    let v = view(1920, 1080, 2.0);
    let layout = VtFeedbackLayout::for_residency(lib.residency());
    let mut feedback = VtFeedback::new(&gpu.device, layout, 64);

    // Three surfaces at three distances, so three different levels are marked
    // and the mask is the OR of three threads rather than one thread's word.
    let mut coverage = vec![
        VtCoverage {
            centre: glam::Vec3::new(0.0, 0.0, -1.0),
            radius: 0.5,
            set,
            vgeom: false,
        },
        VtCoverage {
            centre: glam::Vec3::new(0.0, 0.0, -12.0),
            radius: 1.0,
            set,
            vgeom: false,
        },
        VtCoverage {
            centre: glam::Vec3::new(0.5, 0.0, -60.0),
            radius: 2.0,
            set,
            vgeom: false,
        },
    ];
    let forward = run_feedback(&gpu, &mut feedback, &lib, &pools, &v, &coverage, 0);
    coverage.reverse();
    let reversed = run_feedback(&gpu, &mut feedback, &lib, &pools, &v, &coverage, 3);
    assert_eq!(
        forward, reversed,
        "the coverage mask depends on the order the requests were uploaded in, \
         so the want set is a function of the scene's traversal and not of the \
         camera"
    );
    // ANTI-VACUITY: more than one thread wrote, at more than one level — an
    // agreement between two single-writer masks is an agreement about nothing.
    let levels: std::collections::BTreeSet<u32> = forward.iter().map(|w| w.tile.mip).collect();
    assert!(
        levels.len() > 1,
        "the fixture's three surfaces justify one level ({levels:?})"
    );
    // …and the scan is in virtual-address order whichever order they arrived in.
    let keys: Vec<_> = forward
        .iter()
        .map(|w| (w.texture.0, w.tile.payload_order()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "the scan is not in virtual-address order");
}

/// **Residency contains the floor after EVERY transaction of a scripted path,
/// and the path's floors repeat exactly** (P26.4 audit).
///
/// The batch's floor arm applies ONE transaction. The property the ledger claims
/// is stronger and is the one a streaming loop can violate: over a path, with a
/// refinement set that overflows the pool on every frame, residency never falls
/// below *that frame's* floor. And the determinism half — the floor is a pure
/// function of `(camera, bounds, registry)` — is asserted as sequence equality
/// over the whole path rather than inferred from the absence of a clock.
///
/// GPU-free: the floor is CPU arithmetic and the residency is a set operation,
/// which is exactly the split `inf-vt` exists to keep.
#[test]
fn the_floor_holds_across_a_whole_scripted_path_and_repeats_exactly() {
    // 48 pages: enough for any step's floor (21 camera-free + at most
    // `VT_FLOOR_MAX_TILES`) and nowhere near enough for it plus all 64 tiles of
    // mip 0, so every transaction below runs under real pressure.
    let mut lib = library(1024, 48);
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(7), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered")
        .clone();

    let path = |lib: &VtTextures| -> Vec<Vec<inf_vt::VtWant>> {
        (0..12)
            .map(|i| {
                let v = view(1920, 1080, 2.0 + f64::from(i) * 6.0);
                analytic_floor(
                    lib,
                    &v,
                    &[VtCoverage {
                        centre: glam::Vec3::ZERO,
                        radius: 1.0,
                        set,
                        vgeom: false,
                    }],
                )
            })
            .collect()
    };
    let a = path(&lib);
    let b = path(&lib);
    assert_eq!(
        a, b,
        "two runs of one scripted camera path produced two different floors"
    );
    // ANTI-VACUITY: the path moves. Twelve copies of one answer would satisfy
    // the equality above perfectly.
    let sizes: std::collections::BTreeSet<usize> = a.iter().map(Vec::len).collect();
    assert!(
        sizes.len() > 1,
        "every step of the path produced the same floor ({sizes:?})"
    );

    // …and the invariant, transaction after transaction, under pressure.
    let mut res = lib.residency().clone();
    let mut overflowed = false;
    for (i, floor) in a.iter().enumerate() {
        let mut wants = floor.clone();
        for y in 0..desc.mips[0].tiles_y {
            for x in 0..desc.mips[0].tiles_x {
                wants.push(inf_vt::VtWant::refine(
                    VtTextureHandle(0),
                    TileCoord::new(0, x, y),
                ));
            }
        }
        let txn = res.apply_wants(&wants);
        overflowed |= txn.deferred > 0;
        for w in floor {
            assert!(
                res.is_resident(w.texture, w.tile),
                "step {i}: residency fell below the floor at {:?} — {}",
                w.tile,
                txn.trace()
            );
        }
    }
    assert!(
        overflowed,
        "the refinement set never exhausted the pool, so the invariant above was \
         never under the pressure that could break it"
    );
}
