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
    });
    let bytes = Arc::new(tiled(n)) as Arc<dyn VtTileSource>;
    let mut mats = std::collections::BTreeMap::new();
    mats.insert(
        1u128,
        VtMaterialMaps {
            albedo: Some(7),
            normal: None,
            orm: None,
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
        }],
    );
    let far = analytic_floor(
        &lib,
        &view(1920, 1080, 400.0),
        &[VtCoverage {
            centre: glam::Vec3::ZERO,
            radius: 1.0,
            set,
        }],
    );
    let behind = analytic_floor(
        &lib,
        &view(1920, 1080, 2.0),
        &[VtCoverage {
            centre: glam::Vec3::new(0.0, 0.0, 500.0),
            radius: 1.0,
            set,
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
        let requests = feedback_requests(lib, feedback.layout(), coverage);
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
            frame,
        );
        gpu.queue.submit([enc.finish()]);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        (n, requests.len())
    };

    // ON SCREEN, close: the pass must mark the level the CPU rule names.
    let on = [VtCoverage {
        centre: glam::Vec3::ZERO,
        radius: 1.0,
        set,
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
        },
        VtCoverage {
            centre: glam::Vec3::X,
            radius: 2.0,
            set,
        },
        // A surface that names nothing contributes no request at all.
        VtCoverage {
            centre: glam::Vec3::Y,
            radius: 1.0,
            set: inf_render::VtTextureSet::NONE,
        },
    ];
    let reqs: Vec<FeedbackRequest> = feedback_requests(&lib, &layout, &coverage);
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
