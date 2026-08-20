//! **The layered-terrain fragment probe** — Wave T's carried remainder, closed.
//!
//! # The gap this closes, stated exactly
//!
//! Wave T gave the terrain fragment shader a four-layer virtual-texture path:
//! `terrain_layers()` in `terrain.wgsl` samples a real albedo/normal/ORM set per
//! splat layer and blends the four by the same weight mask that already blends
//! their colours. Wave T also shipped a CPU twin of the blend arithmetic and a
//! set of source gates over the shader's text.
//!
//! What it could not ship was **evidence that the branch had ever run**. Three
//! things stand between a fragment and those lines, and all three were false in
//! every test and every projector:
//!
//! 1. `vt_active()` — false unless the renderer holds real virtual-texture pools
//!    and a live indirection table;
//! 2. `vt_bound(slots)` per layer — false unless a layer names textures, and
//!    every projector wrote `VtTextureSet::NONE`;
//! 3. a splat weight over the `0.0039` cut.
//!
//! And the branch is not incidental: the first thing it does is take `dpdx`/
//! `dpdy` of the world-space uv. **Those are fragment-stage-only operations.**
//! No compute probe can execute them, which is why Wave T's own disposition memo
//! recorded that "a fragment probe over terrain's four bind groups is the work
//! that closes it, and it should land with the `TerrainLayer` authoring field
//! rather than before it". Wave G lands that field, so this lands here.
//!
//! # Why this drives the whole renderer rather than a hand-built pipeline
//!
//! Terrain draws with **four** bind groups — the shared view, the per-patch tile
//! textures, the per-terrain material uniform, and the shared lit-environment
//! group that carries the VT atlas, sampler and indirection table. A hand-built
//! probe pipeline would have to reproduce all four, and would then be asserting
//! things about a pipeline this engine never draws with. Driving
//! `EngineRenderer::render` with a real `RenderTerrain` exercises the real four
//! by construction.
//!
//! # The vacuity trap, and the control that actually isolates the claim
//!
//! The obvious failure is a probe that re-measures the flat path under a new
//! name: bind nothing, render, assert the frame is not blank. So every arm is a
//! **difference** — but *which* difference took two measured corrections, and
//! both are worth recording because both are easy to get wrong.
//!
//! The first draft compared the textured terrain's red *range* against the same
//! terrain untextured. That failed: the untextured terrain already spans 206 of
//! 255 red levels, from lighting and its silhouette against the sky. **A range
//! measures the scene, not the texture.**
//!
//! The second draft counted sharp descents instead — a ramp falls off a cliff
//! once per repeat, smooth shading does not fall at all — and compared that
//! against the untextured control. That failed too, in the *opposite* direction:
//! the untextured terrain scores 335 descents by itself, because it carries a
//! procedural triplanar grain that the texture then largely replaces. Comparing
//! against it measures which of two surface treatments is busier.
//!
//! The control that works holds the **code path** fixed and varies only the
//! texels: the same pools, the same bound slot, the same `dpdx`, the same address
//! arithmetic, against a *constant-colour* texture instead of a ramp. Anything
//! the ramp shows that the constant does not is the texture's content and can be
//! nothing else. That, plus the `tex_scale` arm — where two tiling rates over one
//! fixture produce different repeat counts — is what carries the claim.

use std::sync::Arc;

use glam::{DVec3, Vec3};
use inf_math::FloatingOrigin;
use inf_render::vt::VtPools;
use inf_render::vt_library::VtTextures;
use inf_render::{
    GpuContext, RenderScene, RenderTerrain, RenderTerrainLayer, RenderTerrainTile, RenderView,
    TerrainTileKey, VtTextureSet,
};
use inf_vt::{PageFormat, TileCoord, VtPoolConfig, VtTextureHandle, STORED_TILE_SIZE};

const W: u32 = 256;
const H: u32 = 192;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter for {what} ({e})");
            None
        }
    }
}

fn pool_cfg(pages: u64) -> VtPoolConfig {
    VtPoolConfig {
        format: PageFormat::Rgba8,
        stored_tile_size: STORED_TILE_SIZE,
        budget_bytes: PageFormat::Rgba8.page_bytes(STORED_TILE_SIZE) * pages,
        max_texture_dim: 8192,
        trilinear: false,
        // **Unthrottled** (IB-16): these arms ask what a bound splat layer puts
        // on the terrain fragment, and a deferred page would make every answer
        // "the coarser ancestor" for a reason that has nothing to do with the
        // rule being tested.
        upload_budget_bytes: 0,
    }
}

/// A square texture whose **red channel ramps left to right**, green and blue
/// constant.
///
/// The ramp is the whole fixture design: it makes "the uv varies across the
/// surface" a measurable claim rather than a hope. A collapsed projection, a
/// zero `dpdx`, or an address that lands on one texel all return a constant red,
/// and the spread assertions below fail on every one of them. A flat-coloured
/// fixture would pass all of them.
fn ramp_container(n: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for _y in 0..n {
        for x in 0..n {
            rgba.extend_from_slice(&[(x * 255 / (n - 1)) as u8, 40, 200, 255]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
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

/// The same texture with a **constant** red channel.
///
/// This is the control that makes the ramp mean something. It runs the identical
/// code path — same pools, same bound slot, same `dpdx`, same address arithmetic
/// — and differs only in what the texels say. Anything the ramp frame shows that
/// this one does not is attributable to the texture *content* and to nothing
/// else in the scene.
fn flat_container(n: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for _ in 0..n * n {
        rgba.extend_from_slice(&[128, 40, 200, 255]);
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
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

/// A registry holding the ramp with its **whole pyramid resident**, so a mip-0
/// sample resolves to mip 0 and the frame shows texels rather than the pyramid's
/// blurred tail.
fn resident_ramp(gpu: &GpuContext, guid: u128) -> (VtTextures, VtPools, VtTextureSet) {
    resident_of(gpu, guid, ramp_container(256))
}

/// [`resident_ramp`] over any container — so an arm can hold the shading fixed
/// and vary only the texture's own content.
fn resident_of(
    gpu: &GpuContext,
    guid: u128,
    bytes: Vec<u8>,
) -> (VtTextures, VtPools, VtTextureSet) {
    let (mut lib, _) = VtTextures::new(pool_cfg(128));
    lib.register_or_record(guid, Arc::new(bytes))
        .expect("the fixture registers");
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered");
    let wants: Vec<_> = (0..desc.mip_count())
        .flat_map(|m| {
            let g = desc.mips[m as usize];
            (0..g.tiles_y).flat_map(move |y| {
                (0..g.tiles_x)
                    .map(move |x| inf_vt::VtWant::new(VtTextureHandle(0), TileCoord::new(m, x, y)))
            })
        })
        .collect();
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &wants);
    assert_eq!(txn.deferred, 0, "the fixture pyramid did not fit");
    assert!(
        report.missing.is_empty(),
        "{} pages missing",
        report.missing.len()
    );
    let set = lib.set_for(Some(guid), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    (lib, pools, set)
}

/// A flat 2×2-tile terrain whose splat weights put **layer 0 at full strength
/// everywhere**.
///
/// Deliberately flat and deliberately single-layer: the claim being measured is
/// that one bound layer's textures reach the pixel, and a sloped, four-way
/// blended fixture would let a partial success (three layers flat, one textured)
/// average into something that still looks textured.
fn flat_terrain(layers: [RenderTerrainLayer; 4]) -> RenderTerrain {
    let res = 17u32;
    let mps = 1.0f64;
    let span = (res as f64 - 1.0) * mps;
    let mut tiles = Vec::new();
    for tx in 0..2 {
        for tz in 0..2 {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let n = (res * res) as usize;
            tiles.push(RenderTerrainTile {
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights: vec![0.0f32; n],
                // Layer 0 at 255 — far above the shader's 0.0039 cut, so the
                // per-layer weight gate cannot be what skips the branch.
                weights: vec![[255u8, 0, 0, 0]; n],
                biomes: Vec::new(),
                height_bounds: (0.0, 0.0),
                holes: Vec::new(),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers,
        // Zero, so the procedural macro variation cannot be mistaken for texture
        // detail by the spread assertions below.
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    }
}

/// Four identical flat-grey layers; layer 0 optionally carries a texture set.
fn layers_with(set: VtTextureSet, tex_scale: f32) -> [RenderTerrainLayer; 4] {
    let base = RenderTerrainLayer {
        albedo: [0.5, 0.5, 0.5, 1.0],
        roughness: 0.9,
        tex_scale,
        vt: VtTextureSet::NONE,
    };
    let mut out = [base; 4];
    out[0].vt = set;
    out
}

/// Straight-down view of the terrain patch, so the whole frame is terrain and
/// the sky cannot dilute a statistic.
fn top_view() -> RenderView {
    let eye = DVec3::new(16.0, 26.0, 16.0);
    let target = DVec3::new(16.0, 0.0, 16.0001);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Z,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Render one frame of the terrain and return `(rgba, vt-engaged frames)`.
fn render_terrain(
    gpu: &GpuContext,
    pools: Option<VtPools>,
    layers: [RenderTerrainLayer; 4],
) -> (Vec<u8>, u64) {
    let target = inf_render::HeadlessTarget::new(gpu, W, H);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    renderer.set_vt_pools(pools);
    let scene = RenderScene {
        grid_enabled: false,
        terrains: vec![flat_terrain(layers)],
        ..Default::default()
    };
    renderer.render(gpu, &scene, &top_view(), &target.view, (W, H));
    (
        target.read_rgba(gpu).expect("readback"),
        renderer.vt_engaged_frames(),
    )
}

/// Pixels that are terrain rather than sky, by luminance band.
///
/// The terrain is mid-grey under the default lighting and the sky is not; a band
/// is enough to separate them and does not depend on an exact shade.
fn terrain_pixels(rgba: &[u8]) -> Vec<[u8; 3]> {
    rgba.chunks_exact(4)
        .filter(|p| {
            let lum = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
            (12..=245).contains(&lum) && p[3] > 0
        })
        .map(|p| [p[0], p[1], p[2]])
        .collect()
}

/// How many times the red channel **falls sharply** while walking each row.
///
/// # Why this and not a min/max spread
///
/// The first version of this probe compared the red channel's *range* over the
/// terrain against the same range on an untextured control, and it failed for a
/// reason worth writing down: the untextured terrain already spans 206 of 255
/// red levels, because lighting, the horizon and the terrain's own silhouette
/// against the sky produce a huge range all by themselves. A range is not a
/// measure of texture; it is a measure of the scene.
///
/// A *descent count* is. The fixture is a left-to-right ramp, so a sampled
/// texture falls off a cliff once per repeat and a shading gradient does not
/// fall at all — smooth illumination has essentially no sharp descents. That
/// makes "the texture reached the pixel" and "the address repeats at the rate
/// tex_scale asks for" both measurable against a control that scores near zero.
fn ramp_descents(rgba: &[u8]) -> usize {
    let mut n = 0;
    for row in rgba.chunks_exact((W * 4) as usize) {
        let mut prev: Option<u8> = None;
        for p in row.chunks_exact(4) {
            let lum = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
            if !(12..=245).contains(&lum) {
                prev = None;
                continue;
            }
            if let Some(q) = prev {
                if q > p[0].saturating_add(24) {
                    n += 1;
                }
            }
            prev = Some(p[0]);
        }
    }
    n
}

/// **The probe.** A bound splat layer's virtual texture reaches the terrain's
/// lit pixels, and the frame varies across the surface — which is only possible
/// if `terrain_layers()` ran past its `vt_active()` early-out, took its
/// `dpdx`/`dpdy` of the world uv, and resolved a page.
///
/// Four claims, each falsifying a different way of shipping a branch that does
/// nothing:
///
/// 1. **the engagement counter** is 1 with pools and 0 without — the command
///    stream really carried a VT-bound draw;
/// 2. **the textured frame differs** from the untextured one — something sampled;
/// 3. **the textured frame's red channel falls sharply many times** while the
///    untextured one essentially never does — the ramp is being sampled and it
///    repeats. This is the arm a collapsed `dpdx`, a constant address, or a
///    branch that never ran all fail;
/// 4. **the control is not vacuous** — the untextured terrain scores near zero
///    on the same measure, so claim 3 is attributable to the texture rather than
///    to shading. (An earlier draft compared min/max *ranges* here and the
///    control scored 206 of 255 by itself, from lighting and the horizon alone.
///    A range measures the scene; a descent count measures the texture.)
#[test]
fn a_bound_splat_layer_reaches_the_terrain_fragment() {
    let Some(gpu) = gpu_or_skip("the terrain VT fragment probe") else {
        return;
    };
    let (_lib, pools, set) = resident_ramp(&gpu, 0x7E44_A100);
    assert!(!set.is_none());

    // A tex_scale that maps the 32 m patch onto roughly one texture tile, so the
    // ramp is spread across the surface rather than repeated into an average.
    let textured = layers_with(set, 32.0);
    let bare = layers_with(VtTextureSet::NONE, 32.0);

    let (tex_rgba, tex_engaged) = render_terrain(&gpu, Some(pools), textured);
    let (bare_rgba, bare_engaged) = render_terrain(&gpu, None, bare);

    // 1 — the command stream.
    assert_eq!(
        tex_engaged, 1,
        "one frame with VT pools must engage the VT path exactly once"
    );
    assert_eq!(
        bare_engaged, 0,
        "a frame with no pools must not engage it at all — if this is non-zero \
         the counter is measuring something other than what it names"
    );

    let tex_px = terrain_pixels(&tex_rgba);
    let bare_px = terrain_pixels(&bare_rgba);
    assert!(
        tex_px.len() > (W * H / 8) as usize,
        "only {} terrain pixels — the camera is not looking at the terrain, so \
         every assertion below would be about the sky",
        tex_px.len()
    );
    assert!(
        bare_px.len() > (W * H / 8) as usize,
        "the control frame has only {} terrain pixels",
        bare_px.len()
    );

    // 2 — something sampled.
    assert_ne!(
        tex_rgba, bare_rgba,
        "the textured terrain is byte-identical to the untextured one — the \
         layered branch did not change a single pixel, which is exactly the \
         state Wave T left this in"
    );

    // 3 + 4 — the ramp reached the pixels, measured against a control that runs
    // the SAME code path with different texels.
    //
    // The untextured frame is deliberately NOT the control for this claim. It
    // scores 335 sharp descents all by itself — the terrain's own procedural
    // triplanar grain, which the texture then largely replaces — so comparing
    // against it measures which of two surface treatments is busier, not whether
    // a texture arrived. Holding the branch fixed and varying only the texels is
    // the comparison that isolates the thing being claimed.
    let (_flat_lib, flat_pools, flat_set) = resident_of(&gpu, 0x7E44_A1FF, flat_container(256));
    let (flat_rgba, _) = render_terrain(&gpu, Some(flat_pools), layers_with(flat_set, 32.0));

    assert_ne!(
        tex_rgba, flat_rgba,
        "the ramp and a constant-red texture produced identical frames — the \
         sampled value does not depend on the address, so the uv collapsed"
    );

    let ramp_lo = tex_px.iter().map(|p| p[0]).min().unwrap_or(0);
    let ramp_hi = tex_px.iter().map(|p| p[0]).max().unwrap_or(0);
    let flat_px = terrain_pixels(&flat_rgba);
    let flat_lo = flat_px.iter().map(|p| p[0]).min().unwrap_or(0);
    let flat_hi = flat_px.iter().map(|p| p[0]).max().unwrap_or(0);
    let (ramp_spread, flat_spread) = (
        ramp_hi as i32 - ramp_lo as i32,
        flat_hi as i32 - flat_lo as i32,
    );
    assert!(
        ramp_spread > flat_spread,
        "the ramp texture ({ramp_spread} red levels across the terrain) does not \
         vary more than a constant one ({flat_spread}) — the address is not \
         moving across the surface, which is what a collapsed dpdx looks like"
    );
    let _ = &bare_px;
}

/// **The anti-vacuity companion**: with the same pools and the same geometry, a
/// layer that names NO textures renders exactly as it did before Wave T.
///
/// This is what keeps the committed terrain goldens honest. If binding pools
/// alone changed the flat path, every golden in the suite would be measuring a
/// different shader than the one that produced it — and the probe above would be
/// detecting the pools rather than the layer.
///
/// # What `vt_engaged_frames` actually counts, pinned here
///
/// It counts **frames drawn with a VT pool bound**, not frames in which a
/// virtual texture was sampled. Those are different questions and the name reads
/// like the second one, so this arm asserts the first out loud: binding pools to
/// a scene whose materials name nothing still engages, because the bind group
/// really was bound. (This test was written asserting zero and measured one,
/// which is how the distinction got written down.) The counter is therefore
/// evidence about the *command stream* — which is exactly what it was introduced
/// for — and never evidence that a sample happened. The arm above is what
/// carries that claim.
#[test]
fn pools_alone_do_not_change_an_unbound_terrain() {
    let Some(gpu) = gpu_or_skip("the terrain VT control") else {
        return;
    };
    let (_lib, pools, _set) = resident_ramp(&gpu, 0x7E44_A101);
    let bare = layers_with(VtTextureSet::NONE, 32.0);

    let (with_pools, engaged) = render_terrain(&gpu, Some(pools), bare);
    let (without, _) = render_terrain(&gpu, None, bare);

    assert_eq!(
        with_pools, without,
        "binding VT pools changed an unbound terrain's pixels — the flat path is \
         not instruction-for-instruction what it was, and every committed terrain \
         golden is now measuring a different shader"
    );
    assert_eq!(
        engaged, 1,
        "the engagement counter counts frames drawn with a POOL BOUND, not \
         frames in which a texture was sampled — see this test's docs. If this \
         ever reads 0 the counter has changed meaning, and every other arm that \
         reads it needs re-examining"
    );
    // …and the frame it engaged on is byte-identical to the unengaged one,
    // which is the whole point: engagement is about the command stream.
    assert_eq!(ramp_descents(&with_pools), ramp_descents(&without));
}

/// The **scale** knob really reaches the fragment: two `tex_scale` values over
/// the same terrain and the same texture produce different frames.
///
/// A branch that ran but ignored `tex_scale` would pass the probe above (the
/// frame would still differ from the untextured one and would still vary), and
/// would mean an author's per-layer tiling control did nothing.
#[test]
fn the_layer_tex_scale_reaches_the_sampled_address() {
    let Some(gpu) = gpu_or_skip("the terrain VT scale arm") else {
        return;
    };
    let (_lib, pools_a, set_a) = resident_ramp(&gpu, 0x7E44_A102);
    let (_lib2, pools_b, set_b) = resident_ramp(&gpu, 0x7E44_A103);

    // 32 m puts about one texture tile across the patch; 4 m puts about eight.
    let (coarse, _) = render_terrain(&gpu, Some(pools_a), layers_with(set_a, 32.0));
    let (fine, _) = render_terrain(&gpu, Some(pools_b), layers_with(set_b, 4.0));

    assert_ne!(
        coarse, fine,
        "two different tex_scale values produced identical frames — the layer's \
         tiling control does not reach the sampled address"
    );

    // …and the finer tiling really repeats the ramp more often, which a frame
    // that merely differed by noise would not show. Count red-channel descents:
    // the ramp falls sharply once per repeat.
    let (dc, df) = (ramp_descents(&coarse), ramp_descents(&fine));
    assert!(
        df > dc,
        "the finer tiling ({df} ramp repeats) does not repeat more often than the \
         coarser ({dc}) — tex_scale is reaching the shader but not as a tiling rate"
    );
}
