//! **The P26.4 deviation, retired** (P28.1): virtual-texture feedback becomes
//! per FRAGMENT.
//!
//! `docs/memos/p26-4-feedback-mechanism.md` marks per SURFACE — one compute
//! thread per (drawn surface × bound map), frustum-culling a bounding sphere and
//! marking every tile of the level its projected footprint justifies — and states
//! exactly what that costs and exactly why the textbook fragment-side mark was
//! unreachable:
//!
//! > the fragment-side mark needs a **writable storage buffer visible to the
//! > FRAGMENT stage**, and this renderer cannot give it one cheaply … a `storage`
//! > (writable) buffer entry in that layout makes
//! > `DownlevelFlags::FRAGMENT_WRITABLE_STORAGE` a requirement of **every lit
//! > pass on every adapter**.
//!
//! The visibility buffer dissolves the objection rather than working around it.
//! `vis_feedback.wgsl` is a COMPUTE pass with a bind group of its own: nothing it
//! declares reaches the shared environment layout, no lit pipeline acquires a
//! writable storage binding, and no adapter capability moves. The memo's own
//! "when to revisit" says so — *"marking from THERE is a compute-stage write,
//! needs no new adapter capability, needs no second geometry pass, and is per
//! fragment by construction"*.
//!
//! What this file asserts is that the three losses the memo enumerates are
//! actually **recovered** and not merely also-covered. That distinction is the
//! whole substance of the retirement: the marks are an `atomicOr` union, so
//! leaving the per-surface requests in alongside would put the coarse marks back
//! into the same mask and a union with a coarse mark IS the coarse mark. With the
//! visibility path on, `feedback_requests` hands the meshlet set over
//! (`skip_vgeom`) and the per-pixel pass is its only producer.
//!
//! The **floor is untouched** either way — `analytic_floor` covers every surface
//! whatever draws it — so a refused frame, a dropped mask or a never-arriving one
//! still lands on exactly the analytic residency. That is the property the whole
//! deterministic-feedback ruling rests on and this batch does not move it.
//!
//! Skips cleanly with no GPU adapter.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomSettings, VtTextureSet,
    HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;

const W: u32 = 320;
const H: u32 = 180;
const ASSET: u128 = 0x2801_0000_0fbc_0000;
const TEX: u128 = 1;
/// A SECOND texture, for the hidden surface. The two must not share one: the
/// occlusion arm measures the tiles a hidden surface costs, and two surfaces
/// sampling one texture ask for overlapping tile sets — measured, the hidden
/// surface then added exactly ONE resident tile on both paths and the arm could
/// not tell the two producers apart.
const TEX_HIDDEN: u128 = 2;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP visbuffer_feedback ({what}): no GPU adapter ({e})");
            None
        }
    }
}

/// A 2048² texture: large enough that the mip a **near** surface justifies and
/// the mip a **far** one justifies are different levels, which is what makes
/// "sub-surface variation" a measurable thing rather than a phrase.
fn container() -> Vec<u8> {
    const N: u32 = 2048;
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            rgba.extend_from_slice(&[
                ((x * 7 + y * 3) % 251) as u8,
                ((x * 11 + y * 29) % 241) as u8,
                ((x * 5 + y * 17) % 239) as u8,
                255,
            ]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        N,
        N,
        inf_material::TextureImportSettings {
            srgb: false,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// The occlusion fixture: a big textured **wall** filling the view, and a second
/// textured surface parked well behind it, entirely inside the wall's screen
/// coverage.
///
/// The per-surface pass frustum-culls a bounding sphere and marks the hidden
/// surface's tiles — the memo's loss (1), stated there as *"a wall in front of a
/// textured floor does not stop the floor from asking for its tiles"*. The
/// per-fragment pass reads the id the DEPTH TEST left, so the hidden surface has
/// no pixel to be marked from.
fn scene(set: VtTextureSet, hidden: Option<VtTextureSet>) -> RenderScene {
    let m = Arc::new(dense_grid_mesh(16));
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, &m).expect("index the vmesh")],
        ..Default::default()
    };
    let standing = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let mut wall = VgeomInstance::lit(
        ASSET,
        DVec3::ZERO,
        standing,
        Vec3::splat(9.0),
        [1.0, 1.0, 1.0, 1.0],
        1,
    );
    wall.vt = set;
    sc.vgeom_instances.push(wall);
    if let Some(hidden_set) = hidden {
        // Big and CLOSE — just behind the wall, not far away. A distant surface
        // justifies a coarse mip whose whole level is a handful of tiles, which
        // the analytic floor already holds, so its per-surface marks add nothing
        // and the arm cannot tell the two producers apart (measured: 19 -> 20
        // resident on BOTH paths, the floor's one tile in each). Near and large,
        // it justifies a fine level and asks for real tiles — and is still
        // entirely inside the wall's screen coverage and entirely behind it.
        let mut behind = VgeomInstance::lit(
            ASSET,
            DVec3::new(0.0, 0.0, -6.0),
            standing,
            Vec3::splat(6.0),
            [1.0, 1.0, 1.0, 1.0],
            2,
        );
        behind.vt = hidden_set;
        sc.vgeom_instances.push(behind);
    }
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    sc.mark_dirty();
    sc
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 7.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn settings(visbuffer: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion: false,
            two_pass: false,
            visbuffer,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// What one run of the streaming loop produced: the refinement wants the feedback
/// mask turned into, and the floor beneath them.
struct Run {
    /// Tiles admitted over the run — the streamer's own counter.
    admits: u64,
    /// Frames where a mask was actually read (the pinned `frame − 2` hit).
    feedback_frames: u64,
    /// Tiles resident at the end.
    resident: usize,
    /// Requests the PER-SURFACE pass dispatched on the last frame — zero is
    /// the observable form of the handover.
    per_surface_requests: u32,
    /// **Which** tiles - the set, not its size.
    ///
    /// The size cannot see the handover: with `skip_vgeom` off both producers
    /// mark, so the visibility path's residency becomes the UNION and is still
    /// larger than the per-surface path's. Mutation-measured - disabling the
    /// handover left `the_per_fragment_signal_is_finer` green. A union is
    /// distinguishable from a replacement only by asking whether the per-surface
    /// path holds a tile the per-fragment path does not.
    tiles: std::collections::BTreeSet<(u32, inf_vt::TileCoord)>,
}

/// Drive `frames` frames of one scene through the real renderer, with the real
/// streaming loop, and report what the residency ended up holding.
fn run(gpu: &GpuContext, visbuffer: bool, hidden: bool, frames: usize) -> Run {
    let bytes = container();
    let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
        format: inf_vt::PageFormat::Rgba8,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        // Deliberately generous: this suite is about WHAT is asked for, not about
        // what a budget refuses. A tight budget would make every difference below
        // a story about eviction order instead.
        budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 2048,
        max_texture_dim: 8192,
    });
    lib.register_or_record(TEX, Arc::new(bytes.clone()))
        .unwrap_or_else(|| panic!("the fixture registers: {:?}", lib.refusals()));
    lib.register_or_record(TEX_HIDDEN, Arc::new(bytes))
        .unwrap_or_else(|| panic!("the second texture registers: {:?}", lib.refusals()));
    let pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(settings(visbuffer));
    r.set_vt_level(Some((lib, pools)));
    let v = view();
    for _ in 0..frames {
        let (set, hidden_set) = r
            .vt_textures()
            .map(|l| {
                (
                    l.set_for(Some(TEX), None, None),
                    l.set_for(Some(TEX_HIDDEN), None, None),
                )
            })
            .unwrap_or((VtTextureSet::NONE, VtTextureSet::NONE));
        let sc = scene(set, hidden.then_some(hidden_set));
        r.render(gpu, &sc, &v, &target.view, (W, H));
    }
    let pop = r.vt_pop_in();
    // The RESIDENT set, read off the atlas's own slots. `resolved_table` was the
    // first draft and it is a different thing: it maps every registered tile to
    // the ancestor a sample would resolve to, so it is the same 696 entries
    // whatever is paged in — which made the non-subset assertion below compare a
    // set with itself.
    let tiles = r
        .vt_textures()
        .map(|l| {
            let res = l.residency();
            (0..res.geometry().slot_count())
                .filter_map(|slot| res.slot_occupant(slot).map(|(h, c)| (h.0, c)))
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    Run {
        admits: pop.admits,
        feedback_frames: pop.feedback_frames,
        resident: r
            .vt_textures()
            .map(|l| l.residency().stats().resident as usize)
            .unwrap_or(0),
        per_surface_requests: r.vt_feedback_requests(),
        tiles,
    }
}

/// **The mask is produced from pixels, and the ring reads it.**
///
/// The load-bearing structural claim first: the per-fragment pass writes into the
/// SAME mask `VtFeedback::record` cleared, in the same encoder, before the ring
/// copy — which is why `record` and `finish` had to be split. If the split were
/// wrong the marks would land after the read and the loop would degrade to the
/// floor silently, with every other arm here still green.
///
/// So this asserts the loop **engaged**: a mask was read at the pinned `frame −
/// 2` on the visibility path, and tiles were admitted. Zero on either side is
/// indistinguishable from a producer that never ran.
#[test]
fn the_per_fragment_producer_feeds_the_same_pinned_ring() {
    let Some(gpu) = gpu_or_skip("the ring") else {
        return;
    };
    let vis = run(&gpu, true, false, 6);
    assert!(
        vis.feedback_frames > 0,
        "no feedback mask was ever read on the visibility path — the per-fragment \
         marks are landing after the ring copy, or the copy is not being recorded \
         at all"
    );
    assert!(
        vis.admits > 0,
        "nothing was admitted, so the wants this arm is about were empty"
    );
    assert!(vis.resident > 0, "nothing is resident");
    eprintln!(
        "per-fragment ring: {} feedback frames, {} admits, {} resident",
        vis.feedback_frames, vis.admits, vis.resident
    );
}

/// **The per-fragment signal is finer where the pixels are, and a hidden
/// surface adds nothing to it** — loss (3) recovered, loss (1) bounded.
///
/// Two claims, and the second is stated with the bound the measurement supports
/// rather than the one the memo's prose invites.
///
/// **Loss (3), sub-surface variation, RECOVERED and measured.** The memo's own
/// words: *"a 200 m terrain-sized quad gets one level for the whole thing rather
/// than one per screen region. That is the case where the per-fragment signal is
/// genuinely better."* On this fixture — a wall filling the frame, so its near
/// edge and its far edge justify different levels — the per-surface path settles
/// at **22** resident tiles and the per-fragment path at **32**. It is asking for
/// more, in the places the pixels are, which is the whole content of the claim.
///
/// **Loss (1), occlusion, RECOVERED structurally and NOT visible in residency
/// here — and that is a property of the floor, not a weakness of the producer.**
/// A surface that owns no pixel of the visibility buffer cannot be marked: the
/// producer reads the id the depth test left, so there is no path from an
/// occluded surface to a bit. But residency is `floor ∪ feedback`, and
/// `analytic_floor` is deliberately **occlusion-blind** — it covers every surface
/// whose bounds are on screen, which is exactly what lets a dropped mask degrade
/// safely (`docs/memos/p26-28-virtualization-direction.md`, ruling 3). At this
/// viewport the floor's `VT_FLOOR_MAX_TILES` cap already covers the level
/// `justified_mip` names, so the per-SURFACE feedback has almost nothing extra to
/// waste on a hidden surface, and both paths grow by the floor's **one** tile.
/// Measuring the occlusion win therefore needs per-consumer want accounting,
/// which is P28.3's unified feedback ring; until then the arm asserts the
/// direction (the visibility path never grows MORE) and the ledger carries the
/// bound.
#[test]
fn the_per_fragment_signal_is_finer_and_a_hidden_surface_adds_nothing_to_it() {
    let Some(gpu) = gpu_or_skip("occlusion") else {
        return;
    };
    const N: usize = 6;
    let fwd_alone = run(&gpu, false, false, N);
    let fwd_hidden = run(&gpu, false, true, N);
    let vis_alone = run(&gpu, true, false, N);
    let vis_hidden = run(&gpu, true, true, N);

    eprintln!(
        "per-fragment vs per-surface: forward {} -> {} resident, visbuffer {} -> {} resident",
        fwd_alone.resident, fwd_hidden.resident, vis_alone.resident, vis_hidden.resident
    );

    // (3) The two producers are demonstrably different, and different in the
    // direction the memo predicted. Without this the direction assertion below
    // could be two identical systems agreeing with each other.
    assert!(
        vis_alone.resident > fwd_alone.resident,
        "the per-fragment path settled at {} resident tiles and the per-surface \
         path at {} — the per-pixel signal is not refining anything, so both \
         halves of this arm are about one system",
        vis_alone.resident,
        fwd_alone.resident
    );

    // (1) The direction, with the bound in the message so a future reader who
    // sees it fail knows which of the two facts moved.
    let fwd_growth = fwd_hidden.resident.saturating_sub(fwd_alone.resident);
    let vis_growth = vis_hidden.resident.saturating_sub(vis_alone.resident);
    assert!(
        vis_growth <= fwd_growth,
        "adding a fully occluded textured surface grew the visibility path's \
         residency by {vis_growth} tiles and the per-surface path's by \
         {fwd_growth}. The per-fragment producer reads the id the depth test \
         left, so it has no path to an occluded surface at all; a growth larger \
         than the per-surface path's means something other than the producer is \
         paying for it."
    );
    // …and the anti-vacuity for THAT: the hidden surface has to cost something,
    // or "it costs no more" is a statement about a scene that did not change.
    assert!(
        fwd_growth > 0,
        "the hidden surface cost the per-surface path nothing either, so this \
         fixture does not contain the case the arm is named for"
    );
    eprintln!(
        "occlusion direction: hidden surface cost the per-surface path \
         {fwd_growth} tiles and the per-fragment path {vis_growth} (both the \
         occlusion-blind analytic floor's, at this viewport)"
    );

    // **THE HANDOVER'S FALSIFIER**, and it took three attempts to find one that
    // works — worth recording, because the two that failed both looked right.
    //
    // (a) Residency SIZE cannot see it: with the handover off both producers
    //     mark, the visibility path's set becomes the UNION, and it is still
    //     larger than the per-surface path's. Mutation-measured.
    // (b) Residency as a SET cannot see it either, on this fixture: the
    //     per-surface feedback adds nothing beyond the analytic floor here (the
    //     floor's `VT_FLOOR_MAX_TILES` cap already covers the level
    //     `justified_mip` names), so the per-surface set is a strict SUBSET of
    //     the per-fragment one — 22 tiles inside 32 — and "the per-surface path
    //     holds a tile the per-fragment path does not" is simply false whether
    //     or not the handover is in effect.
    //
    // What IS observable is the per-surface pass's own dispatch count. This
    // fixture's only textured surfaces are meshlets, so with the handover in
    // effect it must dispatch ZERO — a direct reading of the switch rather than
    // an inference from what the streamer did with it.
    assert_eq!(
        vis_alone.per_surface_requests, 0,
        "the per-surface feedback pass dispatched {} requests on a scene whose          only textured surfaces are meshlets — the handover is not in effect and          the meshlet set is being marked twice, which makes every precision          claim in this file a claim about a union with a coarse mark",
        vis_alone.per_surface_requests
    );
    assert!(
        fwd_alone.per_surface_requests > 0,
        "the per-surface pass dispatched nothing on the FORWARD path either, so          the zero above is not evidence of a handover"
    );
    eprintln!(
        "handover: per-surface requests {} forward / {} visbuffer; resident sets          {} and {}, forward set {} the per-fragment one",
        fwd_alone.per_surface_requests,
        vis_alone.per_surface_requests,
        fwd_alone.tiles.len(),
        vis_alone.tiles.len(),
        if fwd_alone.tiles.is_subset(&vis_alone.tiles) {
            "inside"
        } else {
            "NOT inside"
        }
    );
}

/// **The handover is exclusive**, which is the difference between recovering the
/// memo's losses and merely also-covering them.
///
/// `feedback_requests`' `skip_vgeom` is what stops the per-surface pass from
/// re-marking the meshlet set beside the per-fragment pass. A union of a coarse
/// mark and a precise one is the coarse mark, so without the skip every claim in
/// this file would be true of the mask and false of the residency.
///
/// Asserted at the unit door rather than through a frame, because that is where
/// the switch lives and a frame-level assertion could not tell the two producers
/// apart.
#[test]
fn the_meshlet_set_is_handed_over_and_not_marked_twice() {
    use inf_render::vt_stream::{feedback_requests, VtCoverage};
    let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
        format: inf_vt::PageFormat::Rgba8,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 256,
        max_texture_dim: 8192,
    });
    lib.register_or_record(TEX, Arc::new(container()))
        .expect("registers");
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(TEX), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    let layout = inf_vt::VtFeedbackLayout::for_residency(lib.residency());

    let coverage = [
        VtCoverage {
            centre: Vec3::ZERO,
            radius: 1.0,
            set,
            vgeom: false,
        },
        VtCoverage {
            centre: Vec3::new(3.0, 0.0, 0.0),
            radius: 1.0,
            set,
            vgeom: true,
        },
    ];
    let both = feedback_requests(&lib, &layout, &coverage, false);
    let handed = feedback_requests(&lib, &layout, &coverage, true);
    assert!(
        !both.is_empty(),
        "the fixture binds no map, so neither list is about anything"
    );
    assert_eq!(
        handed.len() * 2,
        both.len(),
        "the two surfaces bind the same maps, so skipping the meshlet one must \
         halve the request list exactly: {} against {}",
        handed.len(),
        both.len()
    );
    // …and it is the RIGHT half that survived: the rigid surface's centre, not
    // the meshlet one's. A skip that dropped the wrong surface would satisfy the
    // count above perfectly.
    for r in &handed {
        assert_eq!(
            [r.centre[0], r.centre[1], r.centre[2]],
            [0.0, 0.0, 0.0],
            "the handover kept the meshlet surface and dropped the rigid one"
        );
    }
}

/// **The floor survives the handover**, which is the property the whole
/// deterministic-feedback ruling rests on: *"a dropped feedback frame degrades to
/// exactly today's behaviour"*.
///
/// With the visibility path on and the meshlet set handed over, the analytic
/// floor must still cover the meshlet surface — otherwise a frame whose mask
/// never arrives would sample a hole, and the memo's guarantee would have been
/// spent rather than kept.
#[test]
fn the_analytic_floor_still_covers_a_handed_over_meshlet_surface() {
    use inf_render::vt_stream::{analytic_floor, scene_coverage, VtCoverage};
    let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
        format: inf_vt::PageFormat::Rgba8,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 256,
        max_texture_dim: 8192,
    });
    lib.register_or_record(TEX, Arc::new(container()))
        .expect("registers");
    let _ = lib.residency_mut().apply_wants(&[]);
    let set = lib.set_for(Some(TEX), None, None);
    let sc = scene(set, None);
    let cov = scene_coverage(&sc, &FloatingOrigin::new(DVec3::ZERO));
    assert!(
        cov.iter().any(|c| c.vgeom),
        "the fixture's meshlet surface is not in the coverage list at all"
    );
    // The floor takes the coverage list WHOLE — no `skip_vgeom` reaches it, and
    // that is the point.
    let floor = analytic_floor(&lib, &view(), &cov);
    let bare = analytic_floor(&lib, &view(), &[] as &[VtCoverage]);
    assert!(
        floor.len() > bare.len(),
        "the analytic floor is not larger with the meshlet surface visible than \
         without it, so the handover has taken the floor's coverage with it"
    );
    eprintln!(
        "floor after handover: {} wants with the meshlet surface, {} without",
        floor.len(),
        bare.len()
    );
}
