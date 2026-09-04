//! **The terrain-continuity gate** (wave CERT1): the clipmap's own surface must
//! be continuous — across a patch seam, across a morph band, and across a page
//! edge — because the subject of this wave is a terrain that is *"high resolution
//! and good-looking, not all jagged and sharp and glitched out"*.
//!
//! # What this file can and cannot assert, said plainly
//!
//! Everything measured here lives in `terrain.wgsl`'s vertex and fragment stages,
//! and the fragment stage cannot be executed from a compute probe (it reads a
//! per-tile bind group and the whole lit environment). So this file takes the
//! shape the house already uses for exactly that situation
//! (`terrain_layers::the_wgsl_implements_the_twin_above`,
//! `vt_sampling::the_wgsl_implements_the_twin_above`): a **CPU twin that carries
//! the measurement**, plus a **source gate scoped to the shipped functions** that
//! pins the twin and the shader to the same arithmetic.
//!
//! The twin is `f32` throughout, exactly as the shader is, and mirrors
//! `load_texel` / `sample_height` / `morph_at` / `coarse_height` /
//! `morphed_height` character for character. It does **not** mirror
//! `deform_depth`: `ground_height` is `sample_height − deform_depth`, and
//! `deform_depth` is identically zero in every scene without a deformation field,
//! which is every scene these arms build. That is stated rather than hidden — a
//! deformed terrain's continuity is the P22.1 field's problem, not this wave's.
//!
//! The CPU half of the morph (`morph_factor`, `morph_band`, `lod_thresholds`,
//! `assemble_patches`) is the **real** shipped code, called directly. That is the
//! point of the split: the band is one rule computed on the CPU and evaluated on
//! the GPU, not two rules that agree by luck.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use glam::{DVec3, Vec3};
use inf_math::{psin64, FloatingOrigin};
use inf_render::camera::RenderView;
use inf_render::passes::terrain::{
    assemble_patches, cells_at_lod, lod_for_distance, lod_thresholds, morph_band, morph_factor,
    patch_mesh_lod, ring_source_lod, TerrainPatch, TERRAIN_BASE_CELLS, TERRAIN_MORPH_REGION,
};
use inf_render::scene::{RenderTerrain, RenderTerrainTile, TerrainTileKey};
// DEV-ONLY, and only for the step-3 MEASUREMENTS below: the pyramid and the
// streamer's ladder are `inf-terrain`'s, and a measurement of them must call
// them rather than re-derive them here. `inf-render` names this crate as a dev
// dependency already (the `golden_deform` scene presses its footprints with the
// real Ring-0 field), so this adds nothing to the shipping renderer.
use inf_terrain::stream::RENDER_LOD0_RADIUS_TILES;
use inf_terrain::{downsample_block, RenderWantsParams, TerrainTile, DEFAULT_HYSTERESIS};

// ── the island's numbers ─────────────────────────────────────────────────────
//
// `inf_island::terrain`'s shipped recipe: `tile_resolution = 257`,
// `meters_per_sample = 1.0`, i.e. a **256 m tile at 1 m a sample**. Every arm
// below is built on those, so the metres it prints are the metres the island
// draws.

const RES: u32 = 257;
const MPS: f64 = 1.0;
const SPAN0: f64 = (RES - 1) as f64 * MPS;

/// The synthetic ground: two sines, ~60 m of relief over a 256 m tile.
///
/// **Bit-portable trig** (`inf_math::psin64`), not `f64::sin`, for the P14 law's
/// reason and for one more that is specific to this file: the numbers these arms
/// print are copied into doc comments, and a doc comment that says 0.77 m on
/// Windows and 0.78 m on Linux is prose ahead of its arms on one of the two.
///
/// The z octave's 64 m wavelength is the load-bearing choice. Ring 0's mesh is
/// 4 m a vertex and its morph target is 8 m a vertex, so a 64 m wave is sampled
/// 16×/8× — comfortably above Nyquist on both grids, which is what makes the
/// chord deviation between them a *measurement of the LOD ladder* rather than of
/// an aliased input.
///
/// # The 16 m x-phase, and why it is not decoration
///
/// The first draft had no phase, and the x octave's 128 m wavelength divides the
/// 256 m tile — so `x = 256`, the shared edge every arm below measures at, landed
/// exactly on an **inflection point**, where `∂²ground/∂x² = 0`. Two opposite
/// one-sided differences agree exactly there, and arm 3's cross-seam number came
/// out `0.000°` **before and after the fix**: a vacuous check, which is worse
/// than no check because it reads as a pass. The 16 m offset puts the seam at
/// `π/4` of the wave, where the gradient (0.694 m/m) and the curvature
/// (−0.0341 m/m²) are both substantial.
const X_PHASE: f64 = 16.0;

fn ground(x: f64, z: f64) -> f64 {
    20.0 * psin64((x + X_PHASE) * TAU / 128.0) + 10.0 * psin64(z * TAU / 64.0)
}

/// Analytic `∂ground/∂x`, `∂ground/∂z` — the ground truth arm 3 measures the
/// shader's estimate against.
fn ground_grad(x: f64, z: f64) -> (f64, f64) {
    let kx = TAU / 128.0;
    let kz = TAU / 64.0;
    // d/dx sin(kx·x) = kx·cos(kx·x) = kx·psin64(kx·x + π/2)
    (
        20.0 * kx * psin64((x + X_PHASE) * kx + TAU / 4.0),
        10.0 * kz * psin64(z * kz + TAU / 4.0),
    )
}

/// One level-0 page of [`ground`], at the level-0 grid pitch.
fn tile_of(coord: (i32, i32)) -> RenderTerrainTile {
    let ox = coord.0 as f64 * SPAN0;
    let oz = coord.1 as f64 * SPAN0;
    let mut heights = Vec::with_capacity((RES * RES) as usize);
    for j in 0..RES {
        for i in 0..RES {
            heights.push(ground(ox + i as f64 * MPS, oz + j as f64 * MPS) as f32);
        }
    }
    let lo = heights.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    RenderTerrainTile {
        key: TerrainTileKey::lod0(coord),
        origin: DVec3::new(ox, 0.0, oz),
        heights,
        weights: Vec::new(),
        biomes: Vec::new(),
        holes: Vec::new(),
        height_bounds: (lo, hi),
        version: 1,
    }
}

fn terrain_of(coords: &[(i32, i32)]) -> RenderTerrain {
    RenderTerrain {
        id: 0,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: coords.iter().copied().map(tile_of).collect(),
        layers: Default::default(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    }
}

/// A top-down-ish view from `eye`, wide enough that the fixtures' two tiles both
/// survive the frustum cull inside [`assemble_patches`].
fn view_from(eye: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: Vec3::new(0.0, -1.0, 0.001).normalize(),
        up: Vec3::Z,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    }
}

// ── the CPU twin of terrain.wgsl ─────────────────────────────────────────────

/// One bound height page, as the shader sees it at `@group(1) @binding(0)`.
struct Page<'a> {
    h: &'a [f32],
    res: u32,
}

impl Page<'_> {
    fn of(tile: &RenderTerrainTile) -> Page<'_> {
        Page {
            h: &tile.heights,
            res: RES,
        }
    }
    fn resf(&self) -> f32 {
        self.res as f32
    }
}

/// Twin of `load_texel`: **clamped** — a patch binds only its own page.
fn load_texel(p: &Page, i: i32, j: i32) -> f32 {
    let m = p.res as i32 - 1;
    let (ci, cj) = (i.clamp(0, m), j.clamp(0, m));
    p.h[(cj * p.res as i32 + ci) as usize]
}

/// Twin of `sample_height`: manual bilinear over `load_texel`, uv clamped to the
/// unit square first.
fn sample_height(p: &Page, uv: [f32; 2]) -> f32 {
    let r = p.resf() - 1.0;
    let px = uv[0].clamp(0.0, 1.0) * r;
    let py = uv[1].clamp(0.0, 1.0) * r;
    let (i0, j0) = (px.floor(), py.floor());
    let (fx, fy) = (px - i0, py - j0);
    let (ii, jj) = (i0 as i32, j0 as i32);
    let h00 = load_texel(p, ii, jj);
    let h10 = load_texel(p, ii + 1, jj);
    let h01 = load_texel(p, ii, jj + 1);
    let h11 = load_texel(p, ii + 1, jj + 1);
    let hx0 = h00 + (h10 - h00) * fx;
    let hx1 = h01 + (h11 - h01) * fx;
    hx0 + (hx1 - hx0) * fy
}

/// Twin of `ground_height`. `deform_depth` is identically 0 in a scene with no
/// deformation field — see the module header.
fn ground_height(p: &Page, uv: [f32; 2]) -> f32 {
    sample_height(p, uv)
}

/// Twin of `morph_at`: the smoothstep over a ring's `[start, end]` band, and a
/// non-positive width means "never morph" (the coarsest ring).
fn morph_at(dist: f32, band: [f32; 2]) -> f32 {
    let width = band[1] - band[0];
    if width <= 0.0 {
        return 0.0;
    }
    let t = ((dist - band[0]) / width).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Twin of `coarse_height`: bilinear on the coarse lattice — the CHORD the
/// coarser mesh rasterizes between its own vertices.
fn coarse_height(p: &Page, uv: [f32; 2], cells: f32) -> f32 {
    let step = 2.0 / cells.max(1.0);
    let gx = uv[0].clamp(0.0, 1.0) / step;
    let gy = uv[1].clamp(0.0, 1.0) / step;
    let (g0x, g0y) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - g0x, gy - g0y);
    let at = |a: f32, b: f32| ground_height(p, [a * step, b * step]);
    let h00 = at(g0x, g0y);
    let h10 = at(g0x + 1.0, g0y);
    let h01 = at(g0x, g0y + 1.0);
    let h11 = at(g0x + 1.0, g0y + 1.0);
    let hx0 = h00 + (h10 - h00) * fx;
    let hx1 = h01 + (h11 - h01) * fx;
    hx0 + (hx1 - hx0) * fy
}

/// Twin of `morphed_height`'s blend, with the morph supplied rather than derived
/// — the shader inlines this; the split exists so an arm can sweep the morph
/// directly instead of walking a camera around to produce one.
fn morphed_height_at(p: &Page, uv: [f32; 2], cells: f32, m: f32) -> f32 {
    let h_fine = ground_height(p, uv);
    if m <= 0.0 {
        return h_fine;
    }
    let h_coarse = coarse_height(p, uv, cells);
    h_fine + (h_coarse - h_fine) * m
}

/// Twin of `morphed_height`: the morph evaluated at **this vertex's own**
/// horizontal distance from the eye.
#[allow(clippy::too_many_arguments)]
fn morphed_height(
    p: &Page,
    uv: [f32; 2],
    origin_xz: [f32; 2],
    span: f32,
    cells: f32,
    band: [f32; 2],
    eye_xz: [f32; 2],
) -> f32 {
    let wx = origin_xz[0] + uv[0] * span;
    let wz = origin_xz[1] + uv[1] * span;
    let dist = ((wx - eye_xz[0]).powi(2) + (wz - eye_xz[1]).powi(2)).sqrt();
    morphed_height_at(p, uv, cells, morph_at(dist, band))
}

/// Twin of the shipped fragment stage's central-difference normal: four
/// `morphed_height` taps one texel apart, **with the tap uv clamped here** and
/// the difference divided by the distance the two taps are actually apart.
fn fragment_normal_at(p: &Page, uv: [f32; 2], span: f32, cells: f32, m: f32) -> Vec3 {
    let texel = 1.0 / (p.resf() - 1.0);
    let ul = (uv[0] - texel).max(0.0);
    let ur = (uv[0] + texel).min(1.0);
    let vd = (uv[1] - texel).max(0.0);
    let vu = (uv[1] + texel).min(1.0);
    let hl = morphed_height_at(p, [ul, uv[1]], cells, m);
    let hr = morphed_height_at(p, [ur, uv[1]], cells, m);
    let hd = morphed_height_at(p, [uv[0], vd], cells, m);
    let hu = morphed_height_at(p, [uv[0], vu], cells, m);
    let dhdx = (hr - hl) / ((ur - ul) * span).max(1e-6);
    let dhdz = (hu - hd) / ((vu - vd) * span).max(1e-6);
    Vec3::new(-dhdx, 1.0, -dhdz).normalize()
}

// ── the twin of what the shader did BEFORE wave CERT1 ────────────────────────
//
// Kept, and called by every arm below beside the shipped rule, so the defect each
// arm was written to falsify stays a NUMBER THIS SUITE PRINTS rather than a
// sentence in a commit message. If a later wave decides one of these fixes cost
// more than it bought, the before/after it needs is one `cargo test` away.

/// WGSL `round` is **round-half-to-even** and `f32::round` is half-away-from-zero.
/// The distinction is the defect: it is what decided which even neighbour an odd
/// vertex snapped to, and it alternated.
fn legacy_coarse_uv(uv: [f32; 2], cells: f32) -> [f32; 2] {
    let step = 2.0 / cells.max(1.0);
    [
        (uv[0] / step).round_ties_even() * step,
        (uv[1] / step).round_ties_even() * step,
    ]
}

/// Pre-CERT1 vertex height: `mix(h_fine, h_at_nearest_coarse_vertex, morph)`,
/// with **one morph for the whole patch**.
fn legacy_morphed_height(p: &Page, uv: [f32; 2], cells: f32, m: f32) -> f32 {
    let h_fine = ground_height(p, uv);
    let h_coarse = ground_height(p, legacy_coarse_uv(uv, cells));
    h_fine + (h_coarse - h_fine) * m
}

/// Pre-CERT1 fragment normal: four **un-morphed** `ground_height` taps, uv
/// clamped inside `sample_height`, always divided by `2 · world_step`.
fn legacy_fragment_normal(p: &Page, uv: [f32; 2], span: f32) -> Vec3 {
    let res = p.resf();
    let world_step = span / (res - 1.0);
    let texel = 1.0 / (res - 1.0);
    let hl = ground_height(p, [uv[0] - texel, uv[1]]);
    let hr = ground_height(p, [uv[0] + texel, uv[1]]);
    let hd = ground_height(p, [uv[0], uv[1] - texel]);
    let hu = ground_height(p, [uv[0], uv[1] + texel]);
    let dhdx = (hr - hl) / (2.0 * world_step);
    let dhdz = (hu - hd) / (2.0 * world_step);
    Vec3::new(-dhdx, 1.0, -dhdz).normalize()
}

/// Angle between two normals, in degrees — `atan2(|a × b|, a · b)` in `f64`, NOT
/// `acos(a · b)`.
///
/// The distinction is a measurement floor, not a style note. Both inputs are the
/// shader's own `f32` normals; a dot product one ulp below 1 lands `acos` in its
/// steep region, and `acos(1 − 6e-8) ≈ 3.5e-4 rad = 0.02°`. The first draft of
/// this file reported a **0.040° floor at morph 0**, where the two computations
/// are the same arithmetic and the true answer is exactly zero. `atan2` of the
/// cross product is well-conditioned there.
fn angle_deg(a: Vec3, b: Vec3) -> f64 {
    let a = DVec3::new(a.x as f64, a.y as f64, a.z as f64).normalize();
    let b = DVec3::new(b.x as f64, b.y as f64, b.z as f64).normalize();
    a.cross(b).length().atan2(a.dot(b)).to_degrees()
}

// ── arm 1 · the morph is a function of the VERTEX, not of the patch ──────────

/// The two adjacent level-0 patches of the seam fixture, plus the thresholds they
/// were assembled under.
struct Seam {
    thresholds: Vec<f64>,
    terrain: RenderTerrain,
    a: TerrainPatch,
    b: TerrainPatch,
}

/// Two adjacent level-0 tiles, both in ring 0, with the eye placed so their
/// **centres straddle the ring-0 morph ramp**: tile A's centre sits before the
/// ramp starts and tile B's sits one metre inside the ring-0 threshold.
fn seam_fixture() -> Seam {
    let terrain = terrain_of(&[(0, 0), (1, 0)]);
    let thresholds = lod_thresholds(SPAN0).to_vec();
    // Centres are (128, 128) and (384, 128); the eye on their line at x = 1 puts
    // them 127 m and 383 m away — both inside the 384 m ring-0 threshold, and on
    // opposite sides of the 249.6 m ramp start.
    let view = view_from(DVec3::new(1.0, 300.0, 128.0));
    let patches = assemble_patches(&terrain, &view, &view.origin);
    assert_eq!(patches.len(), 2, "both tiles must survive the cull");
    let a = patches[0];
    let b = patches[1];
    assert_eq!((a.ring, b.ring), (0, 0), "both patches must be in ring 0");
    Seam {
        thresholds,
        terrain,
        a,
        b,
    }
}

/// **The morph factor is a function of the VERTEX, not of the patch.**
///
/// Two adjacent same-ring patches share an edge, and along it their height
/// textures agree sample for sample (the pyramid's shared-edge invariant, and
/// here both pages are cut from one global `ground(x, z)`). So the *only* thing
/// that can separate the two surfaces at a shared vertex is the morph factor —
/// and the shipped rule computes it once per patch, from the distance to the
/// **tile centre**.
///
/// On the island that is not a small error, it is the maximum one. The ring-0
/// morph ramp is `TERRAIN_MORPH_REGION · band width` = 134.4 m wide while
/// adjacent tile centres are 256 m apart, so two neighbours can never both be
/// inside the ramp: one is pinned at 0 and the other at 1, and the seam opens by
/// the **full** deviation between the fine grid and its morph target.
#[test]
fn the_morph_factor_is_a_function_of_the_vertex_not_the_patch() {
    let f = seam_fixture();
    let (pa, pb) = (f.a, f.b);
    let ta = &f.terrain.tiles[pa.tile];
    let tb = &f.terrain.tiles[pb.tile];
    let (page_a, page_b) = (Page::of(ta), Page::of(tb));
    let cells = cells_at_lod(pa.mesh_lod) as f32;
    assert_eq!(
        cells_at_lod(pb.mesh_lod) as f32,
        cells,
        "same-ring neighbours"
    );
    let span = SPAN0 as f32;
    let eye = [1.0f32, 128.0f32];

    println!(
        "  ring-0 band [0, {:.1}] m, ramp starts at {:.1} m",
        f.thresholds[0],
        f.thresholds[0] * (1.0 - TERRAIN_MORPH_REGION)
    );
    println!(
        "  patch morphs at the TILE CENTRES (the pre-CERT1 rule): A = {:.6}, B = {:.6}",
        pa.morph, pb.morph
    );

    // The band is the REAL shipped rule — `morph_band`, the same function the
    // instance buffer is packed from — and both patches are ring 0, so both get
    // the same band and the vertex's own distance is the only input left.
    let (b0, b1) = morph_band(pa.ring, &f.thresholds).expect("ring 0 morphs");
    assert_eq!(
        morph_band(pb.ring, &f.thresholds),
        Some((b0, b1)),
        "same ring, same band"
    );
    let band = [b0 as f32, b1 as f32];
    // …and that band, evaluated on the CPU, is `morph_factor`. One rule.
    for probe in [0.0, 100.0, 300.0, 350.0, 384.0, 500.0] {
        let cpu = morph_factor(probe, pa.ring, &f.thresholds);
        let gpu = morph_at(probe as f32, band);
        assert!(
            (cpu - gpu).abs() < 1e-6,
            "the CPU definition and the shader's twin disagree at {probe} m: \
             {cpu} vs {gpu}"
        );
    }

    // Walk every shared-edge vertex of the two patches' grids. A's edge is
    // uv.x = 1, B's is uv.x = 0; the ODD indices are the ones the coarse target
    // moves, so they carry the whole gap.
    let n = cells as u32;
    let (mut worst, mut worst_j) = (0.0f32, 0);
    let (mut legacy_worst, mut legacy_j) = (0.0f32, 0);
    let oa = [ta.origin.x as f32, ta.origin.z as f32];
    let ob = [tb.origin.x as f32, tb.origin.z as f32];
    for j in 0..=n {
        let v = j as f32 / cells;
        let ha = morphed_height(&page_a, [1.0, v], oa, span, cells, band, eye);
        let hb = morphed_height(&page_b, [0.0, v], ob, span, cells, band, eye);
        let gap = (ha - hb).abs();
        if gap > worst {
            worst = gap;
            worst_j = j;
        }
        let la = legacy_morphed_height(&page_a, [1.0, v], cells, pa.morph);
        let lb = legacy_morphed_height(&page_b, [0.0, v], cells, pb.morph);
        let lg = (la - lb).abs();
        if lg > legacy_worst {
            legacy_worst = lg;
            legacy_j = j;
        }
    }
    let at_z = |j: u32| j as f64 * SPAN0 / cells as f64;
    println!(
        "  BEFORE (one morph per patch): worst shared-edge gap = {legacy_worst:.4} m \
         (v index {legacy_j} of {n}, world z = {:.1} m)",
        at_z(legacy_j)
    );
    println!(
        "  MEASURED worst shared-edge gap = {worst:.4} m (v index {worst_j} of {n}, \
         world z = {:.1} m)",
        at_z(worst_j)
    );
    // NON-VACUITY: the fixture must still put the two centres on opposite sides
    // of the ramp, or "0.0000 m" is a statement about a scene with no morph in it.
    assert!(
        legacy_worst > 1.0,
        "the fixture no longer straddles the ring-0 morph ramp — the pre-CERT1 \
         per-patch rule only opened {legacy_worst:.4} m here, so the shipped \
         rule's {worst:.4} m proves nothing"
    );
    assert!(
        worst < 1.0e-3,
        "two same-ring neighbours disagree by {worst:.4} m at their shared edge; \
         the morph must be a function of the vertex's own distance, not the patch's"
    );
}

// ── arm 2 · the fragment normal sees the surface the vertex moved ────────────

/// **The fragment's normal is the gradient of the surface the vertex stage
/// moved.**
///
/// `terrain.wgsl`'s own deformation doc states the principle: *"the fragment's
/// central-difference normal must see the same surface the vertex stage moved"*.
/// The vertex stage writes `mix(h_fine, h_coarse, morph)`; the fragment stage
/// central-differences the **un-morphed** height at full texel rate. Over the
/// last 35 % of every band the geometry moves toward the coarser grid and the
/// shading keeps lighting the finer one.
///
/// Ground truth here is **not** "whatever the fragment computes": it is the
/// product rule applied to the morphed field with the morph held constant,
/// `∇H = (1−m)·∇h_fine + m·∇h_coarse`, where `∇h_fine` is the texel-rate central
/// difference (what the fine surface's shading legitimately is) and `∇h_coarse`
/// is the **analytic** gradient of the coarse target — a different computation
/// from the one under test.
#[test]
fn the_fragment_normal_sees_the_surface_the_vertex_moved() {
    let terrain = terrain_of(&[(0, 0)]);
    let tile = &terrain.tiles[0];
    let page = Page::of(tile);
    let cells = TERRAIN_BASE_CELLS as f32; // ring 0
    let span = SPAN0 as f32;

    let mut worst = 0.0f64;
    let mut worst_m = 0.0f32;
    let mut legacy_worst = 0.0f64;
    println!("  morph   shipped    before (the un-morphed normal)");
    for step in 0..=10 {
        let m = step as f32 / 10.0;
        let (mut here, mut legacy_here) = (0.0f64, 0.0f64);
        // A grid of fragment positions at mesh-cell centres, so both the
        // interior and the cells whose corners the morph moves are seen.
        for jj in 0..64 {
            for ii in 0..64 {
                let uv = [ii as f32 / 64.0 + 0.5 / 64.0, jj as f32 / 64.0 + 0.5 / 64.0];
                let want = truth_normal(&page, uv, span, cells, m);
                here = here.max(angle_deg(
                    fragment_normal_at(&page, uv, span, cells, m),
                    want,
                ));
                legacy_here =
                    legacy_here.max(angle_deg(legacy_fragment_normal(&page, uv, span), want));
            }
        }
        println!("  {m:>5.2}   {here:>7.3}°   {legacy_here:>7.3}°");
        if here > worst {
            worst = here;
            worst_m = m;
        }
        legacy_worst = legacy_worst.max(legacy_here);
    }
    println!("  BEFORE  worst normal error = {legacy_worst:.3}°");
    println!("  MEASURED worst normal error = {worst:.3}° (at morph {worst_m:.2})");
    // NON-VACUITY. The green above is only worth reading while the fixture still
    // exercises a morph the m-blind normal gets WRONG — otherwise a shader that
    // stopped responding to the morph would pass. This is the number that fails
    // if the fixture goes flat, and it is also the size of the defect.
    assert!(
        legacy_worst > 5.0,
        "the fixture no longer exercises the morph — the pre-CERT1 m-blind normal \
         is only {legacy_worst:.3}° from the truth, so the shipped rule's \
         {worst:.3}° proves nothing"
    );
    // Exactly zero, and that is a property rather than luck: within one coarse
    // cell the chord is BILINEAR, so a central difference of it over ±1 texel is
    // its analytic gradient exactly, and `mix` is linear in the two samples. The
    // shipped central difference of the blended field therefore IS the product
    // rule, to the last bit. (The taps here are mesh-cell centres, which never
    // straddle a coarse-cell boundary; across a boundary the chord has a crease
    // and a central difference averages the two slopes, which is the better
    // shading normal there and not a defect this bound should police.)
    assert!(
        worst < 0.5,
        "the fragment lights a surface {worst:.3}° away from the one the vertex \
         stage moved (worst at morph {worst_m:.2})"
    );
}

/// `∇H = (1−m)·∇h_fine + m·∇h_coarse` at `uv`, with `∇h_fine` the texel-rate
/// central difference of the height page and `∇h_coarse` the **analytic**
/// gradient of the coarse morph target.
fn truth_normal(p: &Page, uv: [f32; 2], span: f32, cells: f32, m: f32) -> Vec3 {
    let res = p.resf();
    let texel = 1.0 / (res - 1.0);
    let world_step = span / (res - 1.0);
    // Fine gradient: the same central difference the fragment already had, and
    // the one a fine-grid surface legitimately shades with.
    let fx = (ground_height(p, [uv[0] + texel, uv[1]]) - ground_height(p, [uv[0] - texel, uv[1]]))
        / (2.0 * world_step);
    let fz = (ground_height(p, [uv[0], uv[1] + texel]) - ground_height(p, [uv[0], uv[1] - texel]))
        / (2.0 * world_step);
    // Coarse gradient: ANALYTIC over the coarse cell the fragment sits in — the
    // slope of the chord the coarser mesh rasterizes between its own vertices.
    let (cx, cz) = coarse_grad(p, uv, span, cells);
    Vec3::new(-(fx + (cx - fx) * m), 1.0, -(fz + (cz - fz) * m)).normalize()
}

/// The analytic gradient of the coarse morph target at `uv`: the slope of the
/// bilinear patch spanning the coarse cell `uv` falls in.
fn coarse_grad(p: &Page, uv: [f32; 2], span: f32, cells: f32) -> (f32, f32) {
    let step = 2.0 / cells.max(1.0);
    let gx = (uv[0].clamp(0.0, 1.0) / step).floor();
    let gy = (uv[1].clamp(0.0, 1.0) / step).floor();
    let fx = uv[0].clamp(0.0, 1.0) / step - gx;
    let fy = uv[1].clamp(0.0, 1.0) / step - gy;
    let at = |a: f32, b: f32| sample_height(p, [a * step, b * step]);
    let h00 = at(gx, gy);
    let h10 = at(gx + 1.0, gy);
    let h01 = at(gx, gy + 1.0);
    let h11 = at(gx + 1.0, gy + 1.0);
    let world = step * span; // metres between coarse lattice points
    let dx = ((h10 - h00) * (1.0 - fy) + (h11 - h01) * fy) / world;
    let dz = ((h01 - h00) * (1.0 - fx) + (h11 - h10) * fx) / world;
    (dx, dz)
}

// ── arm 3 · the fragment normal is continuous across a tile edge ─────────────

/// **The fragment normal does not step at a page edge, and does not halve there.**
///
/// `load_texel` clamps and `sample_height` clamps uv into `[0,1]`, and a patch
/// binds **only its own page** — there is no apron ring in the upload. So at
/// `uv.x < texel` the left tap is `h[0]` instead of the neighbour's `h[-1]`, and
/// `dhdx = (h[1] − h[0]) / (2·world_step)` is **exactly half** the true gradient.
/// The result is a flattened, discontinuous shading line one `world_step` wide
/// along every tile edge — 1 m every 256 m at ring 0, across the whole 7.2 km
/// island.
#[test]
fn the_fragment_normal_is_continuous_across_a_tile_edge() {
    let terrain = terrain_of(&[(0, 0), (1, 0)]);
    // The eye sits ON the shared edge, so every fragment measured here is inside
    // the un-morphed part of ring 0 and the morph plays no part in the number.
    let view = view_from(DVec3::new(SPAN0, 100.0, 128.0));
    let patches = assemble_patches(&terrain, &view, &view.origin);
    assert_eq!(patches.len(), 2);
    assert_eq!((patches[0].ring, patches[1].ring), (0, 0));
    assert_eq!((patches[0].morph, patches[1].morph), (0.0, 0.0));

    let page_a = Page::of(&terrain.tiles[patches[0].tile]);
    let page_b = Page::of(&terrain.tiles[patches[1].tile]);
    let cells = cells_at_lod(patches[0].mesh_lod) as f32;
    let span = SPAN0 as f32;
    let res = RES as f32;
    let texel = 1.0 / (res - 1.0);
    let world_step = span / (res - 1.0);

    /// Worst angle, and the world z it happened at.
    #[derive(Default, Clone, Copy)]
    struct Worst {
        deg: f64,
        z: f64,
    }
    impl Worst {
        fn see(&mut self, deg: f64, z: f64) {
            if deg > self.deg {
                *self = Worst { deg, z };
            }
        }
    }
    let (mut across, mut inward) = (Worst::default(), Worst::default());
    let (mut l_across, mut l_inward) = (Worst::default(), Worst::default());
    let (mut ratio_err, mut ratio) = (0.0f64, 1.0f64);
    let (mut l_ratio_err, mut l_ratio) = (0.0f64, 1.0f64);
    for k in 0..=256 {
        let v = k as f32 / 256.0;
        let z = v as f64 * SPAN0;
        let na = fragment_normal_at(&page_a, [1.0, v], span, cells, 0.0);
        let nb = fragment_normal_at(&page_b, [0.0, v], span, cells, 0.0);
        let la = legacy_fragment_normal(&page_a, [1.0, v], span);
        let lb = legacy_fragment_normal(&page_b, [0.0, v], span);
        // (1) ACROSS the seam: tile A's last column against tile B's first.
        across.see(angle_deg(na, nb), z);
        l_across.see(angle_deg(la, lb), z);
        // (2) INWARD from the seam: tile B's first column against the very next
        // one, ONE TEXEL away. This is the artefact a player sees — a flattened
        // shading line one texel wide down every tile edge — and it is INVISIBLE
        // to (1), because both sides halve by the same factor and so agree with
        // each other while both disagree with the ground.
        inward.see(
            angle_deg(
                nb,
                fragment_normal_at(&page_b, [texel, v], span, cells, 0.0),
            ),
            z,
        );
        l_inward.see(
            angle_deg(lb, legacy_fragment_normal(&page_b, [texel, v], span)),
            z,
        );
        // (3) The MAGNITUDE: at the edge the x-gradient must not be half of the
        // truth. Measured against the ANALYTIC gradient of `ground`.
        let (tx, _) = ground_grad(SPAN0, z);
        if tx.abs() > 0.2 {
            // Recover dh/dx from the normal: n = normalize(-dhdx, 1, -dhdz).
            let r = -(nb.x / nb.y) as f64 / tx;
            if (r - 1.0).abs() > ratio_err {
                ratio_err = (r - 1.0).abs();
                ratio = r;
            }
            let lr = -(lb.x / lb.y) as f64 / tx;
            if (lr - 1.0).abs() > l_ratio_err {
                l_ratio_err = (lr - 1.0).abs();
                l_ratio = lr;
            }
        }
    }
    let _ = world_step;
    println!("  step ACROSS the seam (A uv.x=1 vs B uv.x=0)");
    println!(
        "    BEFORE   {:>7.3}°   (both sides halve identically, so they agree \
         with each other while both are wrong)",
        l_across.deg
    );
    println!(
        "    MEASURED {:>7.3}°  (world z = {:.1} m)",
        across.deg, across.z
    );
    println!("  step INWARD from the seam (edge column vs one texel in) — the visible line");
    println!(
        "    BEFORE   {:>7.3}°  (world z = {:.1} m)",
        l_inward.deg, l_inward.z
    );
    println!(
        "    MEASURED {:>7.3}°  (world z = {:.1} m)",
        inward.deg, inward.z
    );
    println!("  |dh/dx| at the edge, as a multiple of the analytic gradient");
    println!("    BEFORE   {l_ratio:>7.4} ×");
    println!("    MEASURED {ratio:>7.4} ×");
    // NON-VACUITY: the seam must still be somewhere the ground has slope, or a
    // green "0.9996 ×" is a statement about flat ground.
    assert!(
        l_inward.deg > 5.0 && (l_ratio - 0.5).abs() < 0.05,
        "the fixture no longer exercises the clamped edge — the pre-CERT1 rule \
         stepped only {:.3}° and measured {l_ratio:.4}× (it should be ~0.5×)",
        l_inward.deg
    );
    assert!(
        inward.deg < 3.5,
        "the shading normal steps by {:.3}° between a tile's edge column and the \
         column one texel inside it — that is the flattened line down every seam",
        inward.deg
    );
    assert!(
        (ratio - 1.0).abs() < 0.10,
        "the edge gradient is {ratio:.4}× the true one — a clamped central \
         difference measures half the slope it should"
    );
    // The residual, and it is a RESIDUAL rather than a fix: the two sides now
    // estimate the slope with OPPOSITE one-sided differences, which disagree by
    // the surface's curvature over one texel. Closing it needs the neighbour's
    // sample — an apron ring in the page upload, a `.inf_terrain` change, out of
    // scope for this wave. The bound is what the fix actually achieves.
    assert!(
        across.deg < 3.5,
        "the shading normal steps by {:.3}° across a tile edge",
        across.deg
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// MEASUREMENTS — reported, not fixed (wave CERT1, step 3)
//
// Four properties of the shipped clipmap this wave was asked to QUANTIFY rather
// than change. Each prints a table and asserts only that it is not vacuous, so
// the numbers stay live under `--nocapture` and a later wave that routes one of
// them starts from a measurement instead of a guess.
// ═════════════════════════════════════════════════════════════════════════════

/// A rougher ground for the pyramid measurements: the two-sine relief above plus
/// two oblique octaves at 23 m and 7 m.
///
/// The 7 m octave is the point. At level 0 (1 m a sample) it is sampled 7×; at
/// level 2 (4 m a sample) it is 1.75× — **below Nyquist**, so a point-decimated
/// pyramid can neither represent it nor average it away. That is the
/// distance-shimmer source, and it is what the first table prices.
fn rough(x: f64, z: f64) -> f64 {
    ground(x, z)
        + 4.0 * psin64((x * 0.7 + z * 0.3) * TAU / 23.0)
        + 1.5 * psin64((x * 0.2 - z * 0.9) * TAU / 7.0)
}

/// A 4 × 4 block of level-0 pages of [`rough`] — enough for two real pyramid
/// levels.
fn rough_level0() -> Level {
    let mut out = BTreeMap::new();
    for tz in 0..4 {
        for tx in 0..4 {
            let mut t =
                TerrainTile::flat(RES, DVec3::new(tx as f64 * SPAN0, 0.0, tz as f64 * SPAN0));
            for j in 0..RES {
                for i in 0..RES {
                    let x = tx as f64 * SPAN0 + i as f64 * MPS;
                    let z = tz as f64 * SPAN0 + j as f64 * MPS;
                    t.set_sample(RES, i, j, rough(x, z) as f32);
                }
            }
            out.insert((tx, tz), t);
        }
    }
    out
}

/// One level of a pyramid, keyed the way `downsample_block` wants it.
type Level = BTreeMap<(i32, i32), TerrainTile>;

/// Level 1 and level 2 of the REAL pyramid over [`rough_level0`].
fn rough_pyramid() -> (Level, Level, TerrainTile) {
    let level0 = rough_level0();
    let mut level1 = BTreeMap::new();
    for cz in 0..2 {
        for cx in 0..2 {
            level1.insert((cx, cz), downsample_block(RES, MPS, (cx, cz), &level0));
        }
    }
    let level2 = downsample_block(RES, MPS * 2.0, (0, 0), &level1);
    (level0, level1, level2)
}

/// A tile's heights as the flat row-major slice the twin's [`Page`] wants.
fn tile_heights(t: &TerrainTile) -> Vec<f32> {
    let mut v = Vec::with_capacity((RES * RES) as usize);
    for j in 0..RES {
        for i in 0..RES {
            v.push(t.sample(RES, i, j));
        }
    }
    v
}

/// **MEASUREMENT 1 — the pyramid is point-decimated, and this is what it costs.**
///
/// `inf_terrain::pyramid` takes `combined_sample(.., 2i, 2j)` — a point sample,
/// not a filter — and its module doc gives the reason at length: a tent filter
/// needs fine samples from OUTSIDE the 2 × 2 block at the block's own edges,
/// terrain is sparse so those neighbours may not exist, and clamping instead
/// would make two coarse tiles disagree along their shared edge. Decimation is
/// the only 2:1 reduction that inherits the shared-edge invariant with no
/// neighbour fetch.
///
/// The cost of that choice has never been a number. Here it is: per level, the
/// RMS and MAX difference between the shipped decimated page and a 3 × 3 tent
/// over a field with real high-frequency content. The arm also **asserts the
/// shipped page IS the point sample**, so the table is a measurement of the
/// pyramid rather than of an assumption about it.
#[test]
fn the_pyramid_is_point_decimated_and_this_is_what_it_costs() {
    let (_l0, level1, level2) = rough_pyramid();
    println!(
        "  level  m/sample   RMS(decimated − tent)   MAX(decimated − tent)   \
         page == point sample?"
    );
    let mut any = false;
    for (level, tile) in [(1u32, &level1[&(0, 0)]), (2, &level2)] {
        let mps = MPS * (1u64 << level) as f64;
        let fine = mps * 0.5;
        let (mut sum_sq, mut max, mut exact) = (0.0f64, 0.0f64, true);
        for j in 0..RES {
            for i in 0..RES {
                let (x, z) = (i as f64 * mps, j as f64 * mps);
                let shipped = tile.sample(RES, i, j) as f64;
                if (shipped - rough(x, z)).abs() > 1e-3 {
                    exact = false;
                }
                // A separable 3 × 3 tent, [1 2 1]/4 per axis, over the level this
                // one was reduced FROM.
                let mut tent = 0.0;
                for (b, wb) in [(-1.0, 0.25), (0.0, 0.5), (1.0, 0.25)] {
                    for (a, wa) in [(-1.0, 0.25), (0.0, 0.5), (1.0, 0.25)] {
                        tent += wa * wb * rough(x + a * fine, z + b * fine);
                    }
                }
                let d = shipped - tent;
                sum_sq += d * d;
                max = max.max(d.abs());
            }
        }
        let rms = (sum_sq / (RES * RES) as f64).sqrt();
        println!("  {level:>5}  {mps:>8.0}   {rms:>21.4}   {max:>21.4}   {exact}");
        assert!(exact, "level {level} is not the point decimation any more");
        any |= rms > 0.0;
    }
    assert!(any, "the field carries no detail for the pyramid to lose");
}

/// **MEASUREMENT 2 — the streamer's ladder and the renderer's ladder do not
/// change gear together, and the doc comment that says they do is wrong.**
///
/// Two independent ladders decide what a patch of ground is made of:
///
/// * the RENDERER's clipmap rings, at `TERRAIN_LOD_SCALE · 2^r` tile spans;
/// * the STREAMER's quadtree cut, at `RENDER_LOD0_RADIUS_TILES · 2^L` tile spans
///   with a ±15 % dead band.
///
/// `ring_source_lod`'s doc claims that pairing them one-for-one keeps *"the
/// number of height texels per patch cell constant across the whole clipmap"*.
/// It does not — they change gear at different distances, so the ratio walks up
/// and back down. This arm prints the table and **asserts the claim is false**,
/// so the corrected doc comment cannot quietly drift back.
#[test]
fn the_streamer_and_renderer_ladders_do_not_change_gear_together() {
    let levels = 4u32;
    let wants = RenderWantsParams::geometric(RENDER_LOD0_RADIUS_TILES * SPAN0, levels);
    let ring = lod_thresholds(SPAN0);
    // A lod-`L` node refines into its lod-(L−1) children inside
    // `refine_radius(L) · (1 − h)`, so level-(L−1) pages live inside that radius
    // and level-L pages outside it. `refine_radius(0)` is never consulted — a
    // level-0 node has nothing to refine into — which is why the finest asset
    // switch sits at `refine_radius(1)` and not at the 2.5-tile anchor.
    let switch: Vec<f64> = (1..levels)
        .map(|l| wants.refine_radius(l) * (1.0 - DEFAULT_HYSTERESIS))
        .collect();
    println!(
        "  island grid: {RES} samples at {MPS} m ⇒ {SPAN0:.0} m tiles\n  \
         render rings   {ring:?} m\n  asset switches {switch:?} m \
         (fresh; the ±{:.0} % dead band makes the sticky bounds {:?} m)",
        DEFAULT_HYSTERESIS * 100.0,
        (1..levels)
            .map(|l| wants.refine_radius(l) * (1.0 + DEFAULT_HYSTERESIS))
            .collect::<Vec<_>>()
    );

    let mut edges: Vec<f64> = ring.to_vec();
    edges.extend(switch.iter().copied());
    edges.push(0.0);
    edges.push(switch.last().copied().unwrap_or(0.0) * 2.0);
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    edges.dedup();

    println!(
        "  {:>13}  {:>8}  {:>6}  {:>4}  {:>8}  {:>5}  {:>11}  {:>9}  {:>9}",
        "band (m)",
        "resident",
        "wanted",
        "ring",
        "mesh lod",
        "cells",
        "mesh cell",
        "texel",
        "tex/cell"
    );
    let mut ratios: Vec<f64> = Vec::new();
    for w in edges.windows(2) {
        let probe = w[0] + (w[1] - w[0]) * 0.5;
        // What the STREAMER's cut publishes here. A cut holds exactly one page
        // per point, so this IS the page that draws — the renderer reads what is
        // resident and `superseded` only steps a tile aside for an ancestor that
        // is *itself* resident.
        let asset = switch.iter().filter(|&&s| probe >= s).count() as u32;
        // What the RENDERER would ASK for here — `ring_source_lod`, called, so
        // the two ladders' divergence is in the table rather than in prose.
        let r = lod_for_distance(probe, &ring);
        let wanted = ring_source_lod(r, levels - 1);
        let mesh_lod = patch_mesh_lod(r, asset);
        let cells = cells_at_lod(mesh_lod) as f64;
        let span = SPAN0 * (1u64 << asset) as f64;
        let cell_m = span / cells;
        let texel_m = MPS * (1u64 << asset) as f64;
        println!(
            "  {:>5.0}–{:>7.0}  {asset:>8}  {wanted:>6}  {r:>4}  {mesh_lod:>8}  {cells:>5.0}  \
             {cell_m:>9.1} m  {texel_m:>7.1} m  {:>9.1}",
            w[0],
            w[1],
            cell_m / texel_m
        );
        ratios.push(cell_m / texel_m);
    }
    println!("  MEASURED texels per mesh cell across the ladder: {ratios:?}");
    let lo = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ratios.iter().copied().fold(0.0f64, f64::max);
    println!("  MEASURED range {lo} … {hi} — a {:.0}× spread", hi / lo);
    assert!(
        hi > lo,
        "the texels-per-mesh-cell ratio really is constant across the ladder, so \
         `ring_source_lod`'s doc comment was right and the correction this arm \
         justifies must be reverted with it"
    );
}

/// **MEASUREMENT 3 — the asset-LOD switch is not morphed, and this is its step.**
///
/// The clipmap morphs between two *mesh densities* over a 35 % band. It does not
/// morph between two *pages*: when the streamer's cut refines, the ground under a
/// point stops being sampled from level `n` and starts being sampled from level
/// `n − 1` in one frame, with no blend. The height changes by whatever the two
/// pages disagree by — which, since the pyramid decimates rather than filters, is
/// the fine profile's deviation from the coarse chord.
#[test]
fn the_asset_lod_switch_is_a_step_and_this_is_how_big() {
    let (level0, level1, level2) = rough_pyramid();
    let wants = RenderWantsParams::geometric(RENDER_LOD0_RADIUS_TILES * SPAN0, 4);
    println!("  switch at   levels    worst |Δheight| over the shared footprint");
    let mut any = 0.0f64;
    for (level, fine, coarse) in [
        (1u32, &level0[&(0, 0)], &level1[&(0, 0)]),
        (2, &level1[&(0, 0)], &level2),
    ] {
        let d = wants.refine_radius(level) * (1.0 - DEFAULT_HYSTERESIS);
        let fine_mps = MPS * (1u64 << (level - 1)) as f64;
        let (fh, ch) = (tile_heights(fine), tile_heights(coarse));
        let fp = Page { h: &fh, res: RES };
        let cp = Page { h: &ch, res: RES };
        let fine_span = (RES - 1) as f64 * fine_mps;
        let coarse_span = fine_span * 2.0;
        // Walk the FINE page's footprint and ask both pages for the height at the
        // same world point, through the same bilinear the shader uses.
        let mut worst = 0.0f64;
        for j in 0..=256 {
            for i in 0..=256 {
                let (x, z) = (i as f64 / 256.0 * fine_span, j as f64 / 256.0 * fine_span);
                let hf = sample_height(&fp, [(x / fine_span) as f32, (z / fine_span) as f32]);
                let hc = sample_height(&cp, [(x / coarse_span) as f32, (z / coarse_span) as f32]);
                worst = worst.max((hf - hc).abs() as f64);
            }
        }
        println!("  {d:>7.0} m   L{}→L{level}     {worst:>10.4} m", level - 1);
        any = any.max(worst);
    }
    assert!(
        any > 0.0,
        "the two levels agree everywhere — no detail to lose"
    );
}

/// **MEASUREMENT 4 — ring 0's mesh is four times coarser than its height data.**
///
/// `TERRAIN_BASE_CELLS = 64` over a 256 m tile is **4 m a vertex** against height
/// data at **1 m a sample**: three of every four surveyed metres reach a shading
/// normal and never a silhouette. The constant's own doc records what the last
/// density raise cost — *"four times the triangles at every ring costs about
/// 22.6 % more terrain time, because the pass is 92.5 % fragment-bound"* — so
/// this arm prices the other half of that trade: how much surface the mesh is
/// missing now, and how much of it 128 cells would recover.
#[test]
fn ring_zeros_mesh_is_coarser_than_its_height_data() {
    let terrain = terrain_of(&[(0, 0)]);
    let page = Page::of(&terrain.tiles[0]);
    println!(
        "  {:>5}  {:>10}  {:>12}  {:>12}",
        "cells", "m / vertex", "max |Δ| (m)", "RMS |Δ| (m)"
    );
    let mut max_by_cells: Vec<f64> = Vec::new();
    for cells in [TERRAIN_BASE_CELLS, TERRAIN_BASE_CELLS * 2] {
        let n = cells as f32;
        let (mut max, mut sum_sq) = (0.0f64, 0.0f64);
        // Every texel of the page: the surface the 1 m survey knows about.
        for j in 0..RES {
            for i in 0..RES {
                let uv = [i as f32 / (RES - 1) as f32, j as f32 / (RES - 1) as f32];
                let d = (mesh_surface(&page, uv, n) - sample_height(&page, uv)) as f64;
                max = max.max(d.abs());
                sum_sq += d * d;
            }
        }
        println!(
            "  {cells:>5}  {:>10.2}  {max:>12.4}  {:>12.4}",
            SPAN0 / cells as f64,
            (sum_sq / (RES * RES) as f64).sqrt()
        );
        max_by_cells.push(max);
    }
    println!(
        "  MEASURED: {} cells closes {:.0} % of the gap {} cells leaves. The file's \
         own 32→64 measurement prices 4× the triangles at ~22.6 % more terrain \
         time (the pass is 92.5 % fragment-bound), so this is that same trade \
         offered once more.",
        TERRAIN_BASE_CELLS * 2,
        (1.0 - max_by_cells[1] / max_by_cells[0]) * 100.0,
        TERRAIN_BASE_CELLS,
    );
    assert!(
        max_by_cells[0] > 0.0 && max_by_cells[1] < max_by_cells[0],
        "a finer mesh must track the heightfield more closely, not less"
    );
}

/// The height of a patch MESH at `uv`: linear interpolation over the triangle of
/// the `cells × cells` grid that contains it, with the diagonal
/// `build_lod_geometry` actually emits — `(i,j)-(i+1,j)-(i+1,j+1)` then
/// `(i,j)-(i+1,j+1)-(i,j+1)`, i.e. the split runs from the low corner to the high
/// one. (A bilinear stand-in would flatter the mesh: it interpolates the fourth
/// corner the triangle pair never sees.)
fn mesh_surface(p: &Page, uv: [f32; 2], cells: f32) -> f32 {
    let (gx, gy) = (uv[0] * cells, uv[1] * cells);
    let i0 = gx.floor().min(cells - 1.0);
    let j0 = gy.floor().min(cells - 1.0);
    let (fx, fy) = (gx - i0, gy - j0);
    let at = |a: f32, b: f32| sample_height(p, [a / cells, b / cells]);
    let h00 = at(i0, j0);
    let h11 = at(i0 + 1.0, j0 + 1.0);
    if fx >= fy {
        let h10 = at(i0 + 1.0, j0); // lower-right triangle
        h00 + (h10 - h00) * fx + (h11 - h10) * fy
    } else {
        let h01 = at(i0, j0 + 1.0); // upper-left triangle
        h00 + (h11 - h01) * fx + (h01 - h00) * fy
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// THE SOURCE GATE — the twin above and the shipped shader are one arithmetic
//
// Every number this file prints is a CPU twin's. A twin that has drifted from
// the shader measures nothing, so the shipped WGSL is read here and pinned
// function by function, on `terrain_layers.rs`'s pattern and for its recorded
// reason: a byte pin that greps the whole file cannot see where a spelling moved
// to, so each pin reads ONE function's body, extracted by brace matching from its
// signature.
// ═════════════════════════════════════════════════════════════════════════════

/// The shipped shader, read the way the gates read it. `.wgsl` is `text eol=lf`
/// in `.gitattributes`, so the substrings below mean the same thing on every
/// checkout — the P22 CRLF law.
const TERRAIN_WGSL: &str = include_str!("../src/shaders/terrain.wgsl");

/// The body of one WGSL function, by **brace matching from its signature**.
fn wgsl_fn(src: &str, signature: &str) -> String {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` is not in terrain.wgsl any more"));
    let open = src[start..]
        .find('{')
        .expect("a function signature with no body");
    let bytes = &src.as_bytes()[start + open..];
    let mut depth = 0i32;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return src[start..start + open + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("`{signature}` never closes");
}

/// Code only: comments are prose, and a gate that reads them is pinning a
/// paragraph.
fn code_of(body: &str) -> String {
    body.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **The shader implements the twin the arms above measure.**
///
/// Three functions, pinned to the arithmetic each twin mirrors. This is the arm
/// that fails when someone edits the shader and not the twin — at which point the
/// green numbers above stop being about the shipped renderer, silently, which is
/// the failure mode this file exists to make impossible.
#[test]
fn the_wgsl_implements_the_twins_above() {
    // ── `morph_at`: the smoothstep over the band, and the "no band, no morph"
    // clause that the coarsest ring rides in on.
    let m = code_of(&wgsl_fn(TERRAIN_WGSL, "fn morph_at("));
    for needle in [
        "let width = band.y - band.x;",
        "if (width <= 0.0) {",
        "let t = clamp((dist - band.x) / width, 0.0, 1.0);",
        "return t * t * (3.0 - 2.0 * t);",
    ] {
        assert!(
            m.contains(needle),
            "`morph_at` no longer contains `{needle}`; the twin's `morph_at` is \
             now measuring something the shader does not do:\n{m}"
        );
    }
    assert!(
        !m.contains("round("),
        "`morph_at` snaps something — the band is a smoothstep, not a snap:\n{m}"
    );

    // ── `coarse_height`: bilinear on the coarse lattice. `floor`, four taps and
    // a nested `mix` — and emphatically NOT `round`, which is the pre-CERT1
    // nearest-vertex snap this function replaced.
    let c = code_of(&wgsl_fn(TERRAIN_WGSL, "fn coarse_height("));
    for needle in [
        "let g = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) / coarse_step;",
        "let g0 = floor(g);",
        "let f = g - g0;",
        "return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);",
    ] {
        assert!(
            c.contains(needle),
            "`coarse_height` no longer contains `{needle}`:\n{c}"
        );
    }
    assert_eq!(
        c.matches("ground_height(").count(),
        4,
        "`coarse_height` must take exactly the four corners of one coarse cell — \
         a chord is bilinear, and a different tap count is a different \
         surface:\n{c}"
    );
    assert!(
        !c.contains("round("),
        "`coarse_height` snaps to a nearest coarse vertex again. That is the \
         pre-CERT1 target: it does not converge on the coarser mesh, it doubles \
         the local slope at every other vertex (measured 3.8262 m off the fine \
         surface against the chord's 0.29 m), and it makes the fragment's normal \
         go flat at full morph:\n{c}"
    );

    // ── `morphed_height`: ONE function, the morph read at THIS point's distance
    // from the eye, and the `mix(a, b, 0)` early-out spelled out.
    let h = code_of(&wgsl_fn(TERRAIN_WGSL, "fn morphed_height("));
    for needle in [
        "let world_xz = origin_xz + uv * span;",
        "let m = morph_at(length(world_xz - view.eye.xz), band);",
        "if (m <= 0.0) {",
        "return h_fine;",
        "return mix(h_fine, h_coarse, m);",
    ] {
        assert!(
            h.contains(needle),
            "`morphed_height` no longer contains `{needle}`:\n{h}"
        );
    }

    // ── BOTH stages call it, and neither computes the blend itself. This is the
    // whole of defect D, as a source property: `mix(` appearing in `fs` beside a
    // height, or `ground_height` reappearing in the normal, is the two stages
    // drifting apart again.
    let vs = code_of(&wgsl_fn(TERRAIN_WGSL, "fn vs(in: VIn)"));
    assert_eq!(
        vs.matches("morphed_height(").count(),
        1,
        "the vertex stage must displace through `morphed_height` exactly once:\n{vs}"
    );
    assert!(
        !vs.contains("ground_height("),
        "the vertex stage reads the un-morphed height directly again:\n{vs}"
    );
    let fs = wgsl_fn(TERRAIN_WGSL, "fn fs(in: VOut)");
    let normal_block = fs
        .split("let n = normalize")
        .next()
        .expect("the fragment always has a normal");
    let nb = code_of(normal_block);
    assert_eq!(
        nb.matches("morphed_height(").count(),
        4,
        "the fragment's central difference must take its four taps through \
         `morphed_height` — the same surface the vertex moved:\n{nb}"
    );
    assert!(
        !nb.contains("ground_height("),
        "the fragment central-differences the un-morphed height again — that is \
         defect D, measured at 10.586° of surface normal at full morph:\n{nb}"
    );

    // ── The tap spacing is MEASURED, not assumed: `max`/`min` clamps on the tap
    // uv, and a divisor built from the clamped taps rather than `2.0 *
    // world_step`. This is defect E, as a source property.
    for needle in [
        "let ul = max(in.uv.x - texel, 0.0);",
        "let ur = min(in.uv.x + texel, 1.0);",
        "let vd = max(in.uv.y - texel, 0.0);",
        "let vu = min(in.uv.y + texel, 1.0);",
        "let dhdx = (hr - hl) / max((ur - ul) * span, 1e-6);",
        "let dhdz = (hu - hd) / max((vu - vd) * span, 1e-6);",
    ] {
        assert!(
            nb.contains(needle),
            "the fragment normal no longer contains `{needle}`; a tap that clamps \
             inside `sample_height` and is then divided by the spacing it did NOT \
             travel measures half the gradient (0.4875×):\n{nb}"
        );
    }
    assert!(
        !nb.contains("2.0 * world_step"),
        "the fragment divides by a fixed `2 · world_step` again — at a page edge \
         the two taps are one texel apart, not two:\n{nb}"
    );
}

/// **The instance layout the CPU packs is the one the shader declares.**
///
/// `PatchRaw` is private, so this reads the two declarations that must agree and
/// pins the *meaning* of each slot. The failure it exists for is silent: swap
/// `params.x` and `params.z` on one side only and every patch morphs over a band
/// of `[64, 257]` metres, which draws a terrain rather than an error.
#[test]
fn the_instance_slots_mean_the_same_thing_on_both_sides() {
    let vin = wgsl_fn(TERRAIN_WGSL, "struct VIn");
    assert!(
        vin.contains("@location(2) params: vec4<f32>") && vin.contains("@location(3) skirt_depth"),
        "the instance attributes moved:\n{vin}"
    );
    let vs = code_of(&wgsl_fn(TERRAIN_WGSL, "fn vs(in: VIn)"));
    for needle in [
        "let band = in.params.xy;",
        "let cells = max(in.params.z, 1.0);",
        "let res = in.params.w;",
    ] {
        assert!(
            vs.contains(needle),
            "the vertex stage unpacks `params` differently now (`{needle}` is \
             gone). `PatchRaw::params` packs [band.start, band.end, cells, res] — \
             move one and the other must move with it:\n{vs}"
        );
    }
    assert!(
        vs.contains("in.skirt_depth"),
        "the skirt depth is no longer read from its own attribute:\n{vs}"
    );
}

/// **MEASUREMENT 5 — the footprint of this wave in a frame, so "the goldens did
/// not move" can be read for what it is worth.**
///
/// All 121 committed goldens pass under `INF_GOLDEN_STRICT=1` after this wave,
/// unblessed. That is a real result and it is a **weak** one, because
/// `inf_render::golden::image_diff` downscales both frames to **64 × 36** and
/// `within_tolerance` allows a 6 % mean and a 35 % max — the harness says so in
/// its own doc. A one-texel shading line every 256 m is below that instrument's
/// resolution by construction, so the goldens bound this change rather than
/// certify it.
///
/// This arm supplies the bound the goldens cannot: over one ring-0 patch, how
/// many fragments this wave changed and by how much, split by cause.
///
/// **The vertex stage is byte-identical wherever the morph is zero** and that is
/// not an approximation: the pre-CERT1 code evaluated `mix(h_fine, h_coarse, 0)`,
/// which is `h_fine + (h_coarse − h_fine) · 0` — exactly `h_fine` in IEEE for any
/// finite `h_coarse` — and the shipped early-out returns that same `h_fine`.
#[test]
fn what_this_wave_changed_in_a_frame() {
    let terrain = terrain_of(&[(0, 0)]);
    let page = Page::of(&terrain.tiles[0]);
    let span = SPAN0 as f32;
    let cells = TERRAIN_BASE_CELLS as f32;
    let texel = 1.0 / (RES - 1) as f32;

    // (a) The EDGE ring — fragments within one texel of a patch boundary, where
    // a tap used to clamp. Present in every frame that draws terrain at all,
    // morph or no morph.
    let (mut edge_n, mut edge_worst) = (0u32, 0.0f64);
    let (mut interior_n, mut interior_worst) = (0u32, 0.0f64);
    for j in 0..RES {
        for i in 0..RES {
            let uv = [i as f32 / (RES - 1) as f32, j as f32 / (RES - 1) as f32];
            let d = angle_deg(
                fragment_normal_at(&page, uv, span, cells, 0.0),
                legacy_fragment_normal(&page, uv, span),
            );
            let on_edge =
                uv[0] < texel || uv[0] > 1.0 - texel || uv[1] < texel || uv[1] > 1.0 - texel;
            if on_edge {
                edge_n += 1;
                edge_worst = edge_worst.max(d);
            } else {
                interior_n += 1;
                interior_worst = interior_worst.max(d);
            }
        }
    }
    let total = (RES * RES) as f64;
    println!(
        "  at morph 0 (every golden's near ground):\n    \
         edge ring    {edge_n:>6} fragments ({:.2} % of the patch), worst normal \
         change {edge_worst:.3}°\n    \
         interior     {interior_n:>6} fragments ({:.2} %), worst normal change \
         {interior_worst:.3}°",
        edge_n as f64 / total * 100.0,
        interior_n as f64 / total * 100.0,
    );
    // The interior at morph 0 is UNTOUCHED, to the bit — the tap uv only clamps
    // at an edge and `mix(a, b, 0)` is `a`. This is the claim that makes the
    // golden result meaningful at all, so it is asserted rather than asserted-in-
    // prose.
    assert_eq!(
        interior_worst, 0.0,
        "an interior fragment at morph 0 changed by {interior_worst}° — this wave \
         was supposed to be the identity there, and every golden's verdict rests \
         on it"
    );
    assert!(edge_worst > 1.0, "the edge ring did not change at all");

    // (b) The MORPH band. A golden whose terrain never morphs cannot see this at
    // all; the number is what a frame that DOES morph gains.
    let mut morph_worst = 0.0f64;
    for step in 1..=10 {
        let m = step as f32 / 10.0;
        for j in 0..64 {
            for i in 0..64 {
                let uv = [i as f32 / 64.0 + 0.5 / 64.0, j as f32 / 64.0 + 0.5 / 64.0];
                morph_worst = morph_worst.max(angle_deg(
                    fragment_normal_at(&page, uv, span, cells, m),
                    legacy_fragment_normal(&page, uv, span),
                ));
            }
        }
    }
    println!("  inside a morph band: worst normal change {morph_worst:.3}°");
    println!(
        "  the golden harness compares at 64 × 36 with a 6 % mean / 35 % max \
         tolerance, so it can see (b) and cannot see (a)."
    );
    assert!(morph_worst > 1.0, "the morph band did not change at all");
}
