//! **Missing-tile fill** (P26.5, ROADMAP clause 2): reconstructing a page from
//! the finest resident ancestor, by a deterministic edge-directed upscale.
//!
//! # What this is for, and what it is not
//!
//! `docs/memos/p26-28-virtualization-direction.md` §4 rules on Neural Texture
//! Compression: *"deferred, with the intent kept. The doc's fallback idea
//! ('reconstruct detail while the real tile streams') ships as a deterministic
//! edge-directed upscale of the finest resident ancestor into the physical page
//! (measured before adoption). Weight-per-material NTC needs a training
//! pipeline and is deferred by memo, not silence."* This module is that upscale;
//! `docs/memos/p26-5-missing-tile-fill.md` is the measurement and the ruling on
//! where it is adopted.
//!
//! It is **not** the fallback itself. A tile that is not resident resolves
//! through the indirection table to its finest resident ancestor and is sampled
//! there — that is P26.2's law and nothing here changes it. What this answers is
//! the narrower question a page-fill faces: given the ancestor's page, what
//! should the missing child's page contain?
//!
//! # THE RULING: measured, and NOT adopted
//!
//! `inf-material`'s `tests/vt_fill_quality.rs` reconstructs every mip-0 tile of
//! a real container from its parent, three ways, and scores all three against
//! the tile that is actually there. The answer is not the one the clause
//! assumed:
//!
//! | | texel MAE | gradient MAE |
//! |---|---|---|
//! | replicate the ancestor's texels | **9.92** | **24.77** |
//! | bilinear (what the sampler already does when it magnifies) | 12.13 | 25.16 |
//! | edge-directed ([`upscale2x`]) | 12.05 | 25.55 |
//!
//! **Replication wins, and it is not a quirk of the fixture.** A mip level is a
//! BOX DOWNSAMPLE, so a parent texel *is* the mean of its four children —
//! the minimum-error constant predictor for that block, by construction. Any
//! interpolation spends that optimality moving energy across block boundaries,
//! to buy a smoothness the hardware sampler already provides for free when it
//! magnifies a coarse page. The edge-directed filter recovers about a fifth of
//! bilinear's deficit on diagonal content and loses on smooth content; it is
//! within one part in a hundred of bilinear overall, and worse on structure.
//!
//! **And there is no reachable window to adopt it into anyway.** A page-fill
//! needs a slot that has been allocated and bytes that cannot be produced. In
//! this implementation a tile read is a synchronous `read_ref` slice of an mmap
//! (`inf_render::VtTextures::sync`), so an admitted page's bytes are available
//! in the same call that admitted it: the only way `fetch` returns `None` is a
//! container declaring tiles it does not contain, which `container::parse`
//! refuses at the door (the P26.1 audit's three layout equalities). Wiring this
//! in would be a defensive branch no test could reach — the exact thing the
//! P26.4 audit withdrew `c78d2ff` for.
//!
//! So this module ships **declared and pinned, with no hot-path caller**, which
//! is the house rule for API whose caller has not arrived (`VtStats::summary`,
//! `is_warm`, `VT_BINDING_COUNT` before it). Its caller is P28.3's unified
//! streamer, where an admit's bytes genuinely arrive late and a slot genuinely
//! waits — and where the ruling above says the fill to write is the
//! **replication**, with [`upscale2x`] kept as the measured alternative so the
//! ruling can be re-measured rather than re-argued. See
//! `docs/memos/p26-5-missing-tile-fill.md`.
//!
//! # No floating point, and the reason is not style
//!
//! These bytes are **written into the atlas and sampled**, so they are content
//! in the sense the P26.1 law means: one `as f32` and the same missing tile is a
//! different image on two machines, with nothing anywhere comparing the two.
//! Every operation below is `u8` → `i32` → `u8`; the luma weights are the
//! integer BT.601 ones (77/150/29 over 256). `the_fill_never_touches_a_float`
//! enforces it on the source, on `container.rs`'s pattern.

/// Integer BT.601 luma of an RGBA8 texel, in `0..=255`.
///
/// `(77·r + 150·g + 29·b) >> 8`, the standard fixed-point weights. Only ever
/// compared against another luma, so the rounding is irrelevant and the
/// determinism is not.
#[inline]
fn luma(px: &[u8]) -> i32 {
    (77 * px[0] as i32 + 150 * px[1] as i32 + 29 * px[2] as i32) >> 8
}

/// The mean of two RGBA8 texels, per channel, rounding half up.
#[inline]
fn mix2(a: &[u8], b: &[u8], out: &mut [u8]) {
    for c in 0..4 {
        out[c] = ((a[c] as i32 + b[c] as i32 + 1) >> 1) as u8;
    }
}

/// The mean of four RGBA8 texels, per channel, rounding half up.
#[inline]
fn mix4(a: &[u8], b: &[u8], c: &[u8], d: &[u8], out: &mut [u8]) {
    for i in 0..4 {
        out[i] = ((a[i] as i32 + b[i] as i32 + c[i] as i32 + d[i] as i32 + 2) >> 2) as u8;
    }
}

/// How much smaller one difference must be than the other before the
/// interpolation commits to that direction, as a numerator over 8.
///
/// `5/8`: a diagonal wins only when it is meaningfully flatter than the other
/// one, so a smooth patch — where the two differences are within a few luma
/// levels of each other — takes the four-neighbour mean and comes out identical
/// to a box filter. Committing on any difference at all is what makes an
/// edge-directed filter produce staircases in gradients, which is the failure
/// mode this constant exists to avoid; the memo has the measurement.
const DIRECTION_MARGIN_NUM: i32 = 5;
const DIRECTION_MARGIN_DEN: i32 = 8;

/// **A deterministic edge-directed 2× upscale of an RGBA8 image.**
///
/// Two passes over a quincunx lattice, which is the classic
/// NEDI/DCCI structure:
///
/// 1. the **diagonal** samples — output `(2x+1, 2y+1)` — sit at the centre of
///    four source texels. Compare the two diagonals in luma; if one is
///    meaningfully flatter (see [`DIRECTION_MARGIN_NUM`]) an edge runs *across*
///    the other, so interpolate along the flat one. Otherwise take all four;
/// 2. the **axial** samples — `(2x+1, 2y)` and `(2x, 2y+1)` — sit between two
///    source texels on one axis and two just-filled diagonal samples on the
///    other. The same test, rotated 45°.
///
/// Source texels land on `(2x, 2y)` unchanged, so the upscale is exact on the
/// half of the output it inherits and every invented texel is a mean of real
/// neighbours: the result can never leave the source's own colour hull, which is
/// what makes it safe to sample as content.
///
/// Edges clamp (`min(x + 1, w - 1)`), the same rule the container's border ring
/// is baked with, so the two agree at a page boundary by construction.
pub fn upscale2x(src: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || src.len() != (w as usize * h as usize * 4) {
        return None;
    }
    let (dw, dh) = (w as usize * 2, h as usize * 2);
    let mut out = vec![0u8; dw * dh * 4];
    let at = |x: u32, y: u32| -> &[u8] {
        let x = x.min(w - 1) as usize;
        let y = y.min(h - 1) as usize;
        &src[(y * w as usize + x) * 4..][..4]
    };
    let put = |out: &mut Vec<u8>, x: usize, y: usize, v: [u8; 4]| {
        out[(y * dw + x) * 4..][..4].copy_from_slice(&v);
    };
    // Pass 0: the inherited samples.
    for y in 0..h {
        for x in 0..w {
            let mut v = [0u8; 4];
            v.copy_from_slice(at(x, y));
            put(&mut out, x as usize * 2, y as usize * 2, v);
        }
    }
    // Pass 1: the diagonal centres.
    for y in 0..h {
        for x in 0..w {
            let a = at(x, y);
            let b = at(x + 1, y);
            let c = at(x, y + 1);
            let d = at(x + 1, y + 1);
            let main = (luma(a) - luma(d)).abs();
            let anti = (luma(b) - luma(c)).abs();
            let mut v = [0u8; 4];
            if main * DIRECTION_MARGIN_DEN < anti * DIRECTION_MARGIN_NUM {
                mix2(a, d, &mut v);
            } else if anti * DIRECTION_MARGIN_DEN < main * DIRECTION_MARGIN_NUM {
                mix2(b, c, &mut v);
            } else {
                mix4(a, b, c, d, &mut v);
            }
            put(&mut out, x as usize * 2 + 1, y as usize * 2 + 1, v);
        }
    }
    // Pass 2: the axial samples.
    //
    // Every read here is either a SOURCE texel (through `at`, clamped in source
    // space) or a pass-1 CENTRE (through `centre`, clamped in centre space).
    // Never a pass-2 output: the first cut read `out` directly with an output
    // clamp, which at the right edge fetched the sample the loop was about to
    // write — zeros, averaged in — and produced texels outside the source's own
    // colour hull. The hull arm caught it, which is exactly what that arm is
    // for.
    let centres = out.clone();
    let centre = |i: i64, j: i64| -> [u8; 4] {
        let i = i.clamp(0, w as i64 - 1) as usize;
        let j = j.clamp(0, h as i64 - 1) as usize;
        let (x, y) = (i * 2 + 1, j * 2 + 1);
        let mut v = [0u8; 4];
        v.copy_from_slice(&centres[(y * dw + x) * 4..][..4]);
        v
    };
    for y in 0..h {
        for x in 0..w {
            let (ox, oy) = (x as usize * 2, y as usize * 2);
            let (i, j) = (x as i64, y as i64);
            // (2x+1, 2y): between the two horizontal source texels, and between
            // the two diagonal centres above and below.
            let mut v = [0u8; 4];
            axial(
                at(x, y),
                at(x + 1, y),
                &centre(i, j - 1),
                &centre(i, j),
                &mut v,
            );
            put(&mut out, ox + 1, oy, v);
            // (2x, 2y+1): the same, rotated — the axis pair is now vertical and
            // the perpendicular pair is the two centres left and right.
            let mut v2 = [0u8; 4];
            axial(
                at(x, y),
                at(x, y + 1),
                &centre(i - 1, j),
                &centre(i, j),
                &mut v2,
            );
            put(&mut out, ox, oy + 1, v2);
        }
    }
    Some(out)
}

/// One axial sample: `(a, b)` are the two neighbours on the sample's own axis,
/// `(p, q)` the two on the perpendicular one. Pass 1's test, rotated 45°.
#[inline]
fn axial(a: &[u8], b: &[u8], p: &[u8], q: &[u8], out: &mut [u8]) {
    let along = (luma(a) - luma(b)).abs();
    let across = (luma(p) - luma(q)).abs();
    if along * DIRECTION_MARGIN_DEN < across * DIRECTION_MARGIN_NUM {
        mix2(a, b, out);
    } else if across * DIRECTION_MARGIN_DEN < along * DIRECTION_MARGIN_NUM {
        mix2(p, q, out);
    } else {
        mix4(a, b, p, q, out);
    }
}

/// **Replicate an RGBA8 image 2×**: every source texel becomes a 2×2 block.
///
/// The filter the P26.5 measurement chose, and the one
/// [`fill_from_ancestor`] uses. See the module docs for why an interpolation
/// loses to it: a parent texel is the MEAN of its four children, so repeating it
/// gives each child its block's mean, which is the minimum-error constant
/// predictor by construction.
pub fn replicate2x(src: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || src.len() != (w as usize * h as usize * 4) {
        return None;
    }
    let dw = w as usize * 2;
    let mut out = vec![0u8; dw * h as usize * 2 * 4];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let s = &src[(y * w as usize + x) * 4..][..4];
            for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                let o = ((y * 2 + dy) * dw + x * 2 + dx) * 4;
                out[o..o + 4].copy_from_slice(s);
            }
        }
    }
    Some(out)
}

/// **Reconstruct a missing tile's page from its finest resident ancestor's
/// page** by walking `descent` down the tile tree, doubling at every step.
///
/// The doubling is [`replicate2x`], **not** [`upscale2x`], and that is the
/// P26.5 ruling rather than a default: the measurement in `inf-material`'s
/// `tests/vt_fill_quality.rs` scored replication best on both a fidelity and a
/// structure metric, and `inf-material`'s
/// `replicating_the_ancestor_beats_every_interpolation_of_it` asserts THIS call
/// site against its own nearest-neighbour arm — so the code and the memo cannot
/// drift, and a fill that quietly switched filters fails there by name.
///
/// `descent` is the quadrant path from the ancestor down to the wanted tile,
/// coarsest first — [`crate::VtTextureDesc::descent`] computes it from the same
/// clamped chain the indirection table and the shader both walk, so the page
/// this produces is the page the *address arithmetic* says belongs there and not
/// an approximation of it.
///
/// # The crop is exact, which is why the ring survives
///
/// A stored page is `tile_size + 2·border` texels a side. Double it and a child
/// quadrant's whole page — payload **and** border ring — is the sub-rectangle at
/// `(border + tile_size·qx, border + tile_size·qy)`, size `stored`, which is
/// inside the doubled page on every quadrant. So the reconstructed ring is
/// filled from real upscaled neighbours rather than from a clamp, and a bilinear
/// tap at the payload edge reads what it would have read.
///
/// `None` if the page is not `stored² · 4` bytes, if the geometry is not
/// `stored == tile_size + 2·border`, or if the descent is longer than
/// [`MAX_FILL_STEPS`] — a bound, not a policy: each step doubles the working
/// image, so five steps is a 32× blow-up of a 136² page and nothing beyond it
/// carries information the ancestor did not have.
pub fn fill_from_ancestor(
    page: &[u8],
    stored: u32,
    tile_size: u32,
    border: u32,
    descent: &[(u32, u32)],
) -> Option<Vec<u8>> {
    if stored != tile_size + 2 * border || descent.len() > MAX_FILL_STEPS {
        return None;
    }
    let want = stored as usize * stored as usize * 4;
    if page.len() != want {
        return None;
    }
    let mut cur = page.to_vec();
    for &(qx, qy) in descent {
        if qx > 1 || qy > 1 {
            return None;
        }
        let up = replicate2x(&cur, stored, stored)?;
        let (dw, ox, oy) = (
            stored as usize * 2,
            (border + tile_size * qx) as usize,
            (border + tile_size * qy) as usize,
        );
        let mut next = vec![0u8; want];
        for row in 0..stored as usize {
            let src = ((oy + row) * dw + ox) * 4;
            next[row * stored as usize * 4..][..stored as usize * 4]
                .copy_from_slice(&up[src..src + stored as usize * 4]);
        }
        cur = next;
    }
    Some(cur)
}

/// The most levels [`fill_from_ancestor`] will climb down.
///
/// Five: each step doubles the working image, so this is a 32× blow-up of a
/// 136² page (about 19 MB of intermediate at the last step) reconstructing a
/// tile whose ancestor holds 1/1024 of its texels. Past that the ancestor has
/// nothing left to say and the honest answer is the blur the sampler already
/// gives.
pub const MAX_FILL_STEPS: usize = 5;

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                out.extend_from_slice(&f(x, y));
            }
        }
        out
    }

    fn texel(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let mut v = [0u8; 4];
        v.copy_from_slice(&buf[((y * w + x) * 4) as usize..][..4]);
        v
    }

    /// **The source survives the upscale, exactly.**
    ///
    /// Every even output texel is its source texel, byte for byte — so half the
    /// reconstruction is not a reconstruction at all, and the invented half can
    /// only ever be a mean of real neighbours. That is what makes this safe to
    /// write into an atlas and sample.
    #[test]
    fn the_upscale_carries_every_source_texel_through_untouched() {
        let (w, h) = (9u32, 7u32);
        let src = img(w, h, |x, y| [(x * 7) as u8, (y * 11) as u8, 3, 255]);
        let up = upscale2x(&src, w, h).expect("upscales");
        assert_eq!(up.len(), (w * 2 * h * 2 * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    texel(&up, w * 2, x * 2, y * 2),
                    texel(&src, w, x, y),
                    "source texel ({x}, {y}) did not survive"
                );
            }
        }
    }

    /// **A flat image upscales to a flat image**, and a two-colour image
    /// invents no third colour outside their hull.
    ///
    /// The colour-hull property is the one that matters for content: an
    /// edge-directed filter that overshoots (a sharpening kernel, a Catmull-Rom)
    /// produces ringing, and ringing in a page that is *supposed* to be a
    /// placeholder for a sharper tile reads as a defect in the texture.
    #[test]
    fn the_upscale_never_leaves_the_colour_hull() {
        let flat = img(8, 8, |_, _| [40, 90, 200, 255]);
        let up = upscale2x(&flat, 8, 8).expect("upscales");
        assert!(
            up.chunks_exact(4).all(|p| p == [40, 90, 200, 255]),
            "a flat image gained a texel"
        );

        // A hard vertical edge: every texel must be one of the two colours or
        // between them, per channel.
        let edge = img(16, 16, |x, _| {
            if x < 8 {
                [10, 20, 30, 255]
            } else {
                [200, 210, 220, 255]
            }
        });
        let up = upscale2x(&edge, 16, 16).expect("upscales");
        for p in up.chunks_exact(4) {
            assert!(
                (10..=200).contains(&p[0]) && (20..=210).contains(&p[1]),
                "the upscale invented {p:?} outside the source's hull"
            );
        }
    }

    /// **The edge direction is actually followed.**
    ///
    /// A 45° step edge is the case a box filter cannot represent: it fills the
    /// centre sample with the mean of two dark and two light neighbours, i.e.
    /// mid-grey, staircasing the edge. The direction test takes the flat
    /// diagonal instead and the centre stays on its own side.
    ///
    /// Asserted against a box filter computed here, so this is a comparison and
    /// not a magic number — and the anti-vacuity half is that the box filter
    /// really does produce the grey, or the arm would pass on an implementation
    /// that also did.
    #[test]
    fn a_diagonal_edge_is_interpolated_along_the_edge() {
        const DARK: [u8; 4] = [20, 20, 20, 255];
        const LIGHT: [u8; 4] = [230, 230, 230, 255];
        // y > x ⇒ light. The anti-diagonal is the flat one.
        let (w, h) = (16u32, 16u32);
        let src = img(w, h, |x, y| if y > x { LIGHT } else { DARK });
        let up = upscale2x(&src, w, h).expect("upscales");

        // A centre sample well inside the image, straddling the edge: source
        // (x, y) = (7, 7) dark, (8, 7) dark, (7, 8) light, (8, 8) dark.
        let c = texel(&up, w * 2, 15, 15);
        let box_mean = ((DARK[0] as u32 * 3 + LIGHT[0] as u32) / 4) as u8;
        assert!(
            c[0] < box_mean,
            "the centre took the box mean ({box_mean}) rather than the flat \
             diagonal (got {})",
            c[0]
        );
        // Anti-vacuity: a box filter really would have produced something
        // brighter here, so the comparison is a measurement.
        assert!(box_mean > DARK[0] + 20);
    }

    /// A page reconstructed from its own ancestor is the **right size** and its
    /// quadrants differ — the crop really selects.
    #[test]
    fn the_fill_crops_the_quadrant_it_is_asked_for() {
        const STORED: u32 = 136;
        const TILE: u32 = 128;
        const BORDER: u32 = 4;
        // A page with a left-right ramp, so the two horizontal quadrants cannot
        // come out equal.
        let page = img(STORED, STORED, |x, _| {
            [(x * 255 / (STORED - 1)) as u8, 60, 90, 255]
        });
        let left = fill_from_ancestor(&page, STORED, TILE, BORDER, &[(0, 0)]).expect("fills");
        let right = fill_from_ancestor(&page, STORED, TILE, BORDER, &[(1, 0)]).expect("fills");
        assert_eq!(left.len(), page.len());
        assert_eq!(right.len(), page.len());
        assert_ne!(left, right, "both quadrants reconstructed the same page");
        // The left quadrant is darker than the right, because the ramp says so.
        let mean = |p: &[u8]| -> u32 {
            p.chunks_exact(4).map(|t| t[0] as u32).sum::<u32>() / (p.len() / 4) as u32
        };
        assert!(mean(&left) < mean(&right));
        // Two steps compose, and the geometry check refuses a page that is not
        // `tile + 2·border`.
        assert!(fill_from_ancestor(&page, STORED, TILE, BORDER, &[(0, 0), (1, 1)]).is_some());
        assert!(fill_from_ancestor(&page, STORED, TILE, 3, &[(0, 0)]).is_none());
        assert!(fill_from_ancestor(&page[..4], STORED, TILE, BORDER, &[(0, 0)]).is_none());
        assert!(fill_from_ancestor(&page, STORED, TILE, BORDER, &[(2, 0)]).is_none());
        let too_deep = vec![(0u32, 0u32); MAX_FILL_STEPS + 1];
        assert!(fill_from_ancestor(&page, STORED, TILE, BORDER, &too_deep).is_none());
        // An empty descent is the ancestor itself, unchanged — the identity that
        // makes "zero levels of fallback" not a special case.
        assert_eq!(
            fill_from_ancestor(&page, STORED, TILE, BORDER, &[]).expect("identity"),
            page
        );
    }

    /// **The fill never touches a float** — the P26.1 law, in the one module
    /// that invents texels rather than transporting them.
    ///
    /// It matters more here than in the decoders it is copied from: a decoded
    /// tile at least has a right answer stored somewhere, and these bytes exist
    /// nowhere but in the machine that made them. One `as f32` and a missing
    /// tile is a different image on a phone than on a desktop, invisibly.
    #[test]
    fn the_fill_never_touches_a_float() {
        let whole = include_str!("fill.rs");
        let (code, _) = whole
            .split_once("#[cfg(test)]")
            .expect("the test module marker moved; this gate scopes on it");
        let code: String = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("//") && !l.trim_start().starts_with("///"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["f32", "f64", "sqrt", "powf", "powi", "as f"] {
            assert!(
                !code.contains(banned),
                "`{banned}` reached the fill: a reconstructed page is no longer \
                 the same bytes on every target"
            );
        }
        let chars: Vec<char> = code.chars().collect();
        assert!(
            !chars
                .windows(3)
                .any(|w| w[0].is_ascii_digit() && w[1] == '.' && w[2].is_ascii_digit()),
            "a decimal literal reached the fill"
        );
        // Anti-vacuity: the filter left the code behind, not only the comments.
        assert!(
            code.contains("pub fn upscale2x(") && code.contains("pub fn fill_from_ancestor("),
            "the source filter ate the fill"
        );
    }
}
