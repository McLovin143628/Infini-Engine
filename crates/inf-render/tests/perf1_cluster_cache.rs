//! **THE CLUSTER PAIRING, CACHED ACROSS FRAMES** (wave PERF1, clause 1).
//!
//! `VgeomNode::cluster_tile_wants` re-parsed every resident asset's page
//! directory and re-coupled every resident page on every frame. On the shipped
//! island that was **5.3 ms of CPU record per frame** (the PERF1 clause-0
//! profile, `cluster wants`), for answers that only change when a page is newly
//! seated or the texture registry changes. The cache keeps each page's coupled
//! tiles keyed on the source and on `VtTextures::registration_epoch`.
//!
//! What this file holds it to:
//!
//! * **equivalence** -- the same paired meshlet scene driven through two real
//!   renderers, one cached and one forced to re-walk every page every frame,
//!   leaves byte-identical frames, an identical coupling (every group and every
//!   member) and an identical streamer report on every frame;
//! * **engagement** -- the cached renderer serves pages from the cache
//!   (`cached > 0`) and stops walking once residency settles, while the forced
//!   one serves none;
//! * **invalidation** -- a new texture registry (a new epoch) makes the cached
//!   renderer walk its pages again.
//!
//! Skips cleanly with no GPU adapter.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, RenderLight, RenderScene,
    RenderSettings, RenderView, VgeomAsset, VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::{ClusterTexture, ClusterTextureSet, VgeomSource};

const W: u32 = 256;
const H: u32 = 144;
const ASSET: u128 = 0x9e1f_0000_0000_0001;
const TEX: u128 = 0x9e1f_0000_0000_0000_0000_0000_0000_00a1;

fn container() -> Vec<u8> {
    const N: u32 = 1024;
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
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

fn library(bytes: &[u8]) -> inf_render::VtTextures {
    let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
        format: inf_vt::PageFormat::Rgba8,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        budget_bytes: inf_vt::PageFormat::Rgba8.page_bytes(inf_vt::STORED_TILE_SIZE) * 1024,
        max_texture_dim: 8192,
        trilinear: false,
        upload_budget_bytes: 0,
    });
    lib.register_or_record(TEX, Arc::new(bytes.to_vec()))
        .unwrap_or_else(|| panic!("the fixture registers: {:?}", lib.refusals()));
    lib
}

/// A meshlet asset PAIRED against the registered image, so every resident page
/// carries real tile references and the coupling has members to compare.
fn paired(lib: &inf_render::VtTextures) -> Arc<VgeomSource> {
    let h = lib.handle(TEX).expect("registered");
    let desc = lib.residency().desc(h).expect("a descriptor").clone();
    let mesh = inf_vgeom::test_support::build_grid_tangented(
        24,
        0.3,
        inf_vgeom::test_support::GridNormals::Analytic,
        true,
    );
    let set = ClusterTextureSet {
        textures: vec![ClusterTexture::from_desc(
            inf_asset::AssetId(uuid::Uuid::from_u128(TEX)),
            &desc,
        )],
    };
    Arc::new(VgeomSource::from_mesh_paired(&mesh, &set).expect("the paired image builds"))
}

fn scene(src: &Arc<VgeomSource>, lib: Option<&inf_render::VtTextures>, step: usize) -> RenderScene {
    let mut sc = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::new(ASSET, src.clone())],
        ..Default::default()
    };
    // The camera walks in, so the streamer seats finer pages over the run and
    // the cache has a growing prefix to extend.
    let mut inst = VgeomInstance::lit(
        ASSET,
        DVec3::new(0.0, 0.0, -(step as f64) * 0.15),
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Vec3::splat(4.0),
        [1.0, 1.0, 1.0, 1.0],
        1,
    );
    if let Some(l) = lib {
        inst.vt = l.set_for(Some(TEX), None, None);
    }
    sc.vgeom_instances.push(inst);
    sc.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.55, 0.75).normalize(),
        ..RenderLight::default()
    });
    sc.mark_dirty();
    sc
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 9.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn settings() -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            occlusion: false,
            two_pass: false,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

struct Frame {
    bytes: Vec<u8>,
    coupling: Vec<((u128, usize), Vec<(u128, inf_vt::TileCoord)>)>,
    report: String,
}

fn drive(gpu: &GpuContext, cached: bool, frames: usize) -> (Vec<Frame>, (u64, u64), u64) {
    let bytes = container();
    let lib = library(&bytes);
    let src = paired(&lib);
    let pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    r.set_settings(settings());
    r.set_vt_level(Some((lib, pools)));
    r.set_cluster_tile_cache(cached);
    let v = view();
    let mut out = Vec::new();
    for step in 0..frames {
        let sc = scene(&src, r.vt_textures(), step);
        r.render(gpu, &sc, &v, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        out.push(Frame {
            bytes: target.read_rgba(gpu).expect("readback"),
            coupling: r.cluster_coupling_groups(),
            report: format!("{:?}", r.vgeom_stream_report()),
        });
    }
    let counts = r.cluster_tile_cache_counts();
    // INVALIDATION: a new registry is a new epoch -- the cached renderer walks
    // its resident pages again on the next frame.
    let before = counts.0;
    let lib2 = library(&bytes);
    let pools2 = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib2.residency(), false);
    r.set_vt_level(Some((lib2, pools2)));
    let sc = scene(&src, r.vt_textures(), frames);
    r.render(gpu, &sc, &v, &target.view, (W, H));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let rewalked = r.cluster_tile_cache_counts().0 - before;
    (out, counts, rewalked)
}

#[test]
fn the_cached_pairing_is_the_uncached_pairing_frame_for_frame() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP perf1_cluster_cache: no GPU adapter");
        return;
    };
    const FRAMES: usize = 24;
    let (cached, (walked, served), rewalked) = drive(&gpu, true, FRAMES);
    let (forced, (forced_walked, forced_served), _) = drive(&gpu, false, FRAMES);
    let members: usize = cached
        .last()
        .map(|f| f.coupling.iter().map(|(_, m)| m.len()).sum())
        .unwrap_or(0);
    println!(
        "PERF1 cluster cache: cached renderer walked {walked} pages and served {served} from the cache; \
         forced walked {forced_walked}, served {forced_served}; {} groups / {members} members on the last frame; \
         a new registry re-walked {rewalked}",
        cached.last().map_or(0, |f| f.coupling.len())
    );
    for (k, (a, b)) in cached.iter().zip(&forced).enumerate() {
        assert_eq!(
            a.coupling, b.coupling,
            "frame {k}: the cached coupling differs"
        );
        assert_eq!(a.report, b.report, "frame {k}: the streamer report differs");
        assert!(
            a.bytes == b.bytes,
            "frame {k}: the cached frame's pixels differ"
        );
    }
    // Anti-vacuity: there were pages, with members, to couple.
    assert!(
        members > 0,
        "the fixture's pages carry no tile reference -- the comparison is empty"
    );
    // Engagement: the cache served pages, and walked far fewer than the forced
    // renderer, which walked every resident page on every frame.
    assert!(served > 0, "the cache served nothing");
    assert_eq!(
        forced_served, 0,
        "the forced renderer must not use the cache"
    );
    assert!(
        walked * 4 < forced_walked,
        "the cache walked {walked} pages against the forced renderer's {forced_walked}"
    );
    assert!(
        rewalked > 0,
        "a new texture registry did not invalidate the cache"
    );
}

/// **THE MESHLET NODE KEEPS ITS BIND GROUPS** (wave PERF1, clause 1). The node
/// built four bind groups per resident asset per frame; on a settled scene it
/// now builds them on the first frames (two slots for the visibility
/// ping-pong) and re-uses them after, and a resize -- which reallocates the
/// HZB pyramid the late cull binds -- builds again. The equivalence half is the
/// arm above: its frames are drawn through these cached groups and match the
/// forced re-walk pixel for pixel.
#[test]
fn the_meshlet_node_reuses_its_bind_groups_across_frames() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP perf1 bind groups: no GPU adapter");
        return;
    };
    let bytes = container();
    let lib = library(&bytes);
    let src = paired(&lib);
    let pools = inf_render::vt::VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut s = settings();
    s.vgeom.occlusion = true;
    s.vgeom.two_pass = true;
    r.set_settings(s);
    r.set_vt_level(Some((lib, pools)));
    let v = view();
    let target = HeadlessTarget::new(&gpu, W, H);
    const FRAMES: usize = 20;
    for step in 0..FRAMES {
        let sc = scene(&src, r.vt_textures(), step.min(1));
        r.render(&gpu, &sc, &v, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let (built, reused) = r.vgeom_bind_group_counts();
    // A resize reallocates the pyramid: the late cull group must be rebuilt.
    let big = HeadlessTarget::new(&gpu, W * 2, H * 2);
    let mut v2 = v;
    v2.width = W * 2;
    v2.height = H * 2;
    let sc = scene(&src, r.vt_textures(), 1);
    r.render(&gpu, &sc, &v2, &big.view, (W * 2, H * 2));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let (built2, _) = r.vgeom_bind_group_counts();
    println!(
        "PERF1 vgeom bind groups over {FRAMES} frames: {built} built, {reused} re-used; a resize built {} more",
        built2 - built
    );
    assert!(
        built > 0,
        "the node built no bind group -- the fixture drew no meshlet asset"
    );
    assert!(
        built <= 8,
        "{built} bind groups built for one asset over {FRAMES} frames -- the cache is not holding"
    );
    assert!(
        reused >= 3 * (FRAMES as u64 - 3),
        "only {reused} re-used over {FRAMES} frames"
    );
    assert!(
        built2 > built,
        "a resized pyramid did not rebuild the late cull group"
    );
}
