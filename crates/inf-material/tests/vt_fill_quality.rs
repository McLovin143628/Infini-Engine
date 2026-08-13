//! **The missing-tile fill, measured before adoption** (P26.5, ROADMAP clause 2).
//!
//! `docs/memos/p26-28-virtualization-direction.md` §4 deferred Neural Texture
//! Compression and kept its intent: *"reconstruct detail while the real tile
//! streams"* ships as a deterministic edge-directed upscale of the finest
//! resident ancestor, **measured before adoption**. `inf_vt::fill` is the
//! upscale; this file is the measurement, and
//! `docs/memos/p26-5-missing-tile-fill.md` is the ruling it produced.
//!
//! It lives in `inf-material` because that is the crate that can build a real
//! pyramid: `inf-vt` is the *lower* crate (the P26.2 dependency inversion), so
//! it cannot reach the tiler that writes the very containers it reads. What is
//! measured is therefore end to end — a real image, through the shipped
//! `build_tiled_texture`, read back through the shipped `TiledTextureReader`,
//! reconstructed through the shipped `fill_from_ancestor`, and compared against
//! the tile that actually exists at that address.
//!
//! **Three reconstructions, one ground truth.** Every arm below scores mean
//! absolute error per channel against the real finer tile:
//!
//! * **nearest** — the ancestor's texel repeated, which is what a point-sampled
//!   fallback looks like;
//! * **box** — the ancestor's texels bilinearly doubled, which is what the
//!   hardware sampler already gives when it magnifies a coarse page, and
//!   therefore the bar the fill has to clear to be worth existing;
//! * **edge-directed** — `inf_vt::upscale2x`.
//!
//! A test that only asserted "edge-directed beats nearest" would be measuring
//! that interpolation exists.

use inf_material::{build_tiled_texture, TextureCompression, TextureImportSettings};
use inf_vt::{TileCoord, TiledTextureReader};

/// A deterministic test image with the three things that separate the arms:
/// smooth gradients (where a box filter is near-optimal and an over-eager
/// direction test staircases), **diagonal** edges (where a box filter fails and
/// the direction test is the whole point), and axis-aligned steps (where both
/// should be exact).
///
/// Integer arithmetic only, so the fixture is the same image on every machine —
/// the same law the thing it measures is held to.
fn fixture(n: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        for x in 0..n {
            let quadrant = (y * 2 / n) * 2 + (x * 2 / n);
            let px: [u8; 4] = match quadrant {
                // A smooth two-axis gradient.
                0 => [
                    (x * 255 / (n - 1)) as u8,
                    (y * 255 / (n - 1)) as u8,
                    64,
                    255,
                ],
                // A 45° wedge — the diagonal case.
                1 => {
                    if (x + y) % 64 < 32 {
                        [230, 220, 200, 255]
                    } else {
                        [30, 40, 50, 255]
                    }
                }
                // Axis-aligned bars.
                2 => {
                    if (x / 16) % 2 == 0 {
                        [200, 60, 60, 255]
                    } else {
                        [40, 40, 160, 255]
                    }
                }
                // A shallow diagonal (2:1) — the case a four-direction test
                // handles and a two-direction one only approximates.
                _ => {
                    if (x + y * 2) % 96 < 48 {
                        [180, 180, 40, 255]
                    } else {
                        [20, 90, 120, 255]
                    }
                }
            };
            // **Detail at the TEXEL scale**, which is the whole premise of a
            // missing tile: a mip level is a box downsample, so a feature wider
            // than two texels survives into the parent and *replicating* the
            // parent reproduces it exactly. Without this term the fixture is a
            // measurement of how well each filter copies information that was
            // never lost, and every arm scores near zero on three of four
            // regions. A deterministic integer hash, so the image is the same on
            // every machine.
            let h = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263)) >> 13;
            let d = (h % 41) as i32 - 20;
            let px = [
                (px[0] as i32 + d).clamp(0, 255) as u8,
                (px[1] as i32 + d).clamp(0, 255) as u8,
                (px[2] as i32 + d).clamp(0, 255) as u8,
                255,
            ];
            rgba.extend_from_slice(&px);
        }
    }
    rgba
}

/// Nearest-neighbour 2×: every source texel repeated into a 2×2 block.
fn nearest2x(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let dw = w as usize * 2;
    let mut out = vec![0u8; dw * h as usize * 2 * 4];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let s = &src[(y * w as usize + x) * 4..][..4];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let o = ((y * 2 + dy) * dw + x * 2 + dx) * 4;
                out[o..o + 4].copy_from_slice(s);
            }
        }
    }
    out
}

/// Box/bilinear 2× on the same quincunx lattice `inf_vt::upscale2x` uses — the
/// **same** filter with the direction test removed, so the comparison isolates
/// the direction test and nothing else.
fn box2x(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let dw = w as usize * 2;
    let mut out = vec![0u8; dw * h as usize * 2 * 4];
    let at = |x: u32, y: u32| -> &[u8] {
        let x = x.min(w - 1) as usize;
        let y = y.min(h - 1) as usize;
        &src[(y * w as usize + x) * 4..][..4]
    };
    let mean = |v: &[&[u8]]| -> [u8; 4] {
        let mut o = [0u8; 4];
        for c in 0..4 {
            let s: u32 = v.iter().map(|p| p[c] as u32).sum();
            o[c] = ((s + v.len() as u32 / 2) / v.len() as u32) as u8;
        }
        o
    };
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = (x as usize * 2, y as usize * 2);
            let put = |out: &mut Vec<u8>, dx: usize, dy: usize, v: [u8; 4]| {
                out[((oy + dy) * dw + ox + dx) * 4..][..4].copy_from_slice(&v);
            };
            put(&mut out, 0, 0, mean(&[at(x, y)]));
            put(&mut out, 1, 0, mean(&[at(x, y), at(x + 1, y)]));
            put(&mut out, 0, 1, mean(&[at(x, y), at(x, y + 1)]));
            put(
                &mut out,
                1,
                1,
                mean(&[at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)]),
            );
        }
    }
    out
}

/// Mean absolute error over the RGB channels of two equal-length RGBA8 buffers,
/// in `0..=255`.
fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0u64;
    let mut n = 0u64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            sum += (x[c] as i32 - y[c] as i32).unsigned_abs() as u64;
            n += 1;
        }
    }
    sum as f64 / n as f64
}

/// **Mean absolute error of the LOCAL GRADIENT** between two equal-sized RGBA8
/// tiles, in `0..=255` — the structure metric [`mae`] cannot see.
///
/// `mae` rewards the block-mean predictor, and a mip level is a box downsample,
/// so replicating a parent texel gives every child its own block's mean: the
/// minimum-error constant predictor, by construction. That is exactly what the
/// measurement below found, and it is also exactly what a *page* must not be —
/// a page is sampled 1:1 by a linear sampler, so block-constant texels are
/// visible blocking at the moment the surface is closest to the camera.
///
/// This scores how far the reconstruction's own texel-to-texel variation is from
/// the truth's: a blocky reconstruction has zero variation inside a block and
/// double at its seams, and both errors land here. Luma only (`tile` is square
/// and the border is already cropped away).
fn grad_mae(a: &[u8], b: &[u8], n: u32) -> f64 {
    let lum = |buf: &[u8], x: u32, y: u32| -> i32 {
        let p = &buf[((y * n + x) * 4) as usize..][..4];
        (77 * p[0] as i32 + 150 * p[1] as i32 + 29 * p[2] as i32) >> 8
    };
    let mut sum = 0u64;
    let mut count = 0u64;
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let ga =
                (lum(a, x + 1, y) - lum(a, x, y)).abs() + (lum(a, x, y + 1) - lum(a, x, y)).abs();
            let gb =
                (lum(b, x + 1, y) - lum(b, x, y)).abs() + (lum(b, x, y + 1) - lum(b, x, y)).abs();
            sum += (ga - gb).unsigned_abs() as u64;
            count += 1;
        }
    }
    sum as f64 / count as f64
}

/// Crop a `stored²` page's PAYLOAD (the `tile²` interior inside the border
/// ring), which is the only part a sample can legally reach.
fn payload(page: &[u8], stored: u32, tile: u32, border: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((tile * tile * 4) as usize);
    for y in 0..tile {
        let row = ((y + border) * stored + border) as usize * 4;
        out.extend_from_slice(&page[row..row + tile as usize * 4]);
    }
    out
}

/// **THE MEASUREMENT, and the ruling it produced.** Reconstruct every mip-0 tile
/// of a real container from its parent, three ways, and score all three against
/// the tile that is actually there.
///
/// The clause assumed the edge-directed upscale would be the answer. It is not:
/// **replicating the ancestor's texels beats every interpolation of them**, on
/// both a fidelity metric and a structure metric, because a mip level is a box
/// downsample and a parent texel therefore *is* the mean of its four children.
/// `docs/memos/p26-5-missing-tile-fill.md` is the ruling; this is where its
/// numbers come from.
///
/// Printed as well as asserted, because a number in a memo that no test prints
/// is a number that rots — and asserted in the direction the ruling actually
/// went, so the day it inverts the build says so.
#[test]
fn replicating_the_ancestor_beats_every_interpolation_of_it() {
    const N: u32 = 512;
    let bytes = build_tiled_texture(
        fixture(N),
        N,
        N,
        TextureImportSettings {
            srgb: false,
            generate_mips: true,
            // RGBA8 pages: a BC round trip would put its own error into every
            // arm equally and make the three numbers harder to read, and the
            // fill's own adoption is on the RGBA8 tier anyway (see the memo).
            compression: TextureCompression::None,
        },
    )
    .expect("the fixture tiles")
    .into_bytes();

    let reader = TiledTextureReader::new(bytes).expect("a v2 container");
    let desc = reader.vt_desc();
    let (stored, tile, border) = (
        reader.header().stored_tile_size(),
        reader.header().tile_size,
        reader.header().border,
    );
    assert!(desc.mip_count() >= 2, "the fixture has no parent level");
    let m0 = desc.mips[0];
    assert!(
        m0.tiles_x >= 4 && m0.tiles_y >= 4,
        "the fixture has too few mip-0 tiles ({}×{}) to average over",
        m0.tiles_x,
        m0.tiles_y
    );

    // Per REGION, not only in total. The P25 law -- *an average hides a station*
    // -- and here it hides the whole ruling: the four quadrants of the fixture
    // are four different reconstruction problems and they do not agree about
    // which filter wins.
    let mut per_region = [[0.0f64; 6]; 4];
    let mut per_region_n = [0u32; 4];
    let (mut e_near, mut e_box, mut e_edge, mut tiles) = (0.0f64, 0.0f64, 0.0f64, 0u32);
    let (mut g_near, mut g_box, mut g_edge) = (0.0f64, 0.0f64, 0.0f64);
    for ty in 0..m0.tiles_y {
        for tx in 0..m0.tiles_x {
            let region = ((ty * 2 / m0.tiles_y) * 2 + (tx * 2 / m0.tiles_x)) as usize;
            let want = TileCoord::new(0, tx, ty);
            let truth = reader.tile_rgba8(0, tx, ty).expect("the tile exists");
            let up = desc.parent(want).expect("mip 0 has a parent");
            let parent = reader
                .tile_rgba8(up.mip, up.x, up.y)
                .expect("the parent exists");
            let steps = desc.descent(want, up).expect("a one-step descent");
            assert_eq!(steps.len(), 1);

            // The three arms, cropped identically. `crop` is the fill's own crop,
            // written out here so the three differ ONLY in their filter.
            let crop = |doubled: &[u8]| -> Vec<u8> {
                let dw = stored as usize * 2;
                let (ox, oy) = (
                    (border + tile * steps[0].0) as usize,
                    (border + tile * steps[0].1) as usize,
                );
                let mut out = vec![0u8; (stored * stored * 4) as usize];
                for row in 0..stored as usize {
                    let s = ((oy + row) * dw + ox) * 4;
                    out[row * stored as usize * 4..][..stored as usize * 4]
                        .copy_from_slice(&doubled[s..s + stored as usize * 4]);
                }
                out
            };
            let near = crop(&nearest2x(&parent, stored, stored));
            let boxed = crop(&box2x(&parent, stored, stored));
            let edge = crop(&inf_vt::upscale2x(&parent, stored, stored).expect("upscales"));
            // …and THE SHIPPED FILL is the arm the measurement chose. Pinned
            // here rather than described, so the code and the ruling cannot
            // drift: a `fill_from_ancestor` that quietly switched filters would
            // otherwise be invisible to every test in the tree.
            assert_eq!(
                inf_vt::fill_from_ancestor(&parent, stored, tile, border, &steps)
                    .expect("the fill runs"),
                near,
                "the shipped fill is no longer the filter the measurement chose"
            );

            let t = payload(&truth, stored, tile, border);
            let (pn, pb, pe) = (
                payload(&near, stored, tile, border),
                payload(&boxed, stored, tile, border),
                payload(&edge, stored, tile, border),
            );
            let scores = [
                mae(&pn, &t),
                mae(&pb, &t),
                mae(&pe, &t),
                grad_mae(&pn, &t, tile),
                grad_mae(&pb, &t, tile),
                grad_mae(&pe, &t, tile),
            ];
            e_near += scores[0];
            e_box += scores[1];
            e_edge += scores[2];
            g_near += scores[3];
            g_box += scores[4];
            g_edge += scores[5];
            for k in 0..6 {
                per_region[region][k] += scores[k];
            }
            per_region_n[region] += 1;
            tiles += 1;
        }
    }
    let n = f64::from(tiles);
    let (e_near, e_box, e_edge) = (e_near / n, e_box / n, e_edge / n);
    let (g_near, g_box, g_edge) = (g_near / n, g_box / n, g_edge / n);
    const REGIONS: [&str; 4] = [
        "smooth gradient",
        "45 deg wedge",
        "axis-aligned bars",
        "shallow (2:1) diagonal",
    ];
    println!("vt missing-tile fill over {tiles} mip-0 tiles of a {N}sq fixture -- mean absolute error per channel (0..255)");
    for r in 0..4 {
        let m = f64::from(per_region_n[r]);
        println!(
            "  {:<24} texel: near {:>6.2} box {:>6.2} edge {:>6.2}   |   grad: near {:>6.2} box {:>6.2} edge {:>6.2}",
            REGIONS[r],
            per_region[r][0] / m,
            per_region[r][1] / m,
            per_region[r][2] / m,
            per_region[r][3] / m,
            per_region[r][4] / m,
            per_region[r][5] / m
        );
    }
    println!(
        "  {:<24} texel: near {e_near:>6.2} box {e_box:>6.2} edge {e_edge:>6.2}   |   grad: near {g_near:>6.2} box {g_box:>6.2} edge {g_edge:>6.2}",
        "ALL"
    );

    // ANTI-VACUITY FIRST: the fixture really does carry detail the parent level
    // cannot hold. A fixture whose features are all wider than two texels loses
    // nothing to a box downsample, so every arm scores near zero and the
    // ordering below is noise -- measured: without the texel-scale term in
    // `fixture` the bars region scores 0.00 for `nearest` and the whole
    // comparison collapses.
    assert!(
        e_near > 5.0 && e_box > 5.0 && e_edge > 5.0,
        "no arm scored above 5.0 (near {e_near:.2}, box {e_box:.2}, edge \
         {e_edge:.2}) -- the fixture has no sub-texel detail for a \
         reconstruction to be wrong about, so this measures nothing"
    );

    // THE RULING (P26.5, `docs/memos/p26-5-missing-tile-fill.md`).
    //
    // **Replicating the ancestor's texels beats every interpolation, on both
    // metrics.** That is not a quirk of the fixture, it is a property of the
    // pyramid: a mip level is a BOX DOWNSAMPLE, so a parent texel *is* the mean
    // of its four children and replication is the minimum-error constant
    // predictor for each block. Any interpolation spends that optimality moving
    // energy across block boundaries, to buy a smoothness the hardware sampler
    // already provides for free when it magnifies a coarse page.
    //
    // So the edge-directed upscale does NOT earn its adoption, and this arm
    // pins the ruling rather than the hope: the day someone makes it beat
    // replication, this fails BY NAME and says so.
    assert!(
        e_near < e_box && e_near < e_edge,
        "an interpolation beat replication (near {e_near:.2}, box {e_box:.2}, \
         edge {e_edge:.2}) -- the P26.5 ruling in \
         docs/memos/p26-5-missing-tile-fill.md rests on the opposite and must \
         be rewritten before the fill is adopted anywhere"
    );
    assert!(
        g_near < g_box && g_near < g_edge,
        "an interpolation beat replication on STRUCTURE too (near {g_near:.2}, \
         box {g_box:.2}, edge {g_edge:.2}) -- see the memo"
    );
    // …and the edge-directed filter does not clear the bilinear bar by a margin
    // that would be worth a page write even if a reachable page existed: it is
    // within one part in a hundred of `box` on texels and WORSE on structure.
    assert!(
        (e_box - e_edge).abs() * 100.0 < e_box,
        "the edge-directed filter has moved away from `box` (box {e_box:.2}, \
         edge {e_edge:.2}) -- re-read the memo before trusting its conclusion"
    );
}

/// **The fill is a pure function of its inputs** — two runs, byte-identical.
///
/// It writes bytes that are sampled, so this is the no-float law's observable
/// half: a reconstruction that depended on anything but its ancestor and its
/// descent would make a missing tile a property of the machine.
#[test]
fn the_fill_is_reproducible_byte_for_byte() {
    const N: u32 = 256;
    let bytes = build_tiled_texture(
        fixture(N),
        N,
        N,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::Bc1,
        },
    )
    .expect("the fixture tiles")
    .into_bytes();
    let reader = TiledTextureReader::new(bytes).expect("a v2 container");
    let desc = reader.vt_desc();
    let (stored, tile, border) = (
        reader.header().stored_tile_size(),
        reader.header().tile_size,
        reader.header().border,
    );
    let want = TileCoord::new(0, 1, 1);
    let up = desc.parent(want).expect("a parent");
    let steps = desc.descent(want, up).expect("a descent");
    let parent = reader
        .tile_rgba8(up.mip, up.x, up.y)
        .expect("the parent exists");
    let a = inf_vt::fill_from_ancestor(&parent, stored, tile, border, &steps).expect("fills");
    let b = inf_vt::fill_from_ancestor(&parent, stored, tile, border, &steps).expect("fills");
    assert_eq!(a, b, "two fills of one tile produced different bytes");
    // …and it is not the ancestor's page handed back, which would also be
    // reproducible.
    assert_ne!(a, parent, "the fill returned its input");
    assert_eq!(a.len(), parent.len());
}
