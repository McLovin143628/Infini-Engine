//! Pure-Rust BC1 / BC3 (DXT1 / DXT5) block compression.
//!
//! GPU-native block-compressed textures are 4-8× smaller than RGBA8 and sample
//! for free, so imported textures are compressed at import time. We hand-roll
//! BC1 (opaque, 4 bits per texel — **8:1** against RGBA8's 32) and BC3 (full
//! alpha, 8 bits per texel — 4:1) rather than pull `intel_tex_2`, whose ISPC
//! build is a cross-OS CI liability (BC7 via intel_tex_2 is the documented
//! follow-up). The encoder uses the standard bounding-box endpoint selection
//! with per-pixel nearest-index assignment — correct, GPU-decodable output;
//! endpoint refinement (PCA/cluster fit) is a future quality pass.
//!
//! Blocks are 4×4; images whose dimensions aren't multiples of 4 are padded by
//! clamping edge texels (the standard approach), so any size compresses. The
//! same clamp rule fills a virtual-texture tile's border ring and its partial
//! right/bottom remainder (`crate::tiles`), which is what makes a v2 tile's
//! block grid the level's own block grid, byte for byte.
//!
//! **Everything here is integer arithmetic and must stay that way** — a `.inf_tex`
//! is content-hashed and cooked into a reproducible pack, so a float would make a
//! texture's bytes a property of the machine that imported it. Enforced by
//! `tests::the_encoder_never_touches_a_float`.

/// Compress an RGBA8 image to BC1 (color only; alpha discarded). 8 bytes/block.
pub fn compress_bc1(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    compress(rgba, width, height, false)
}

/// Compress an RGBA8 image to BC3 (color + interpolated alpha). 16 bytes/block.
pub fn compress_bc3(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    compress(rgba, width, height, true)
}

/// Bytes a compressed image occupies: `ceil(w/4) * ceil(h/4) * block_bytes`.
pub fn compressed_size(width: u32, height: u32, bc3: bool) -> usize {
    let blocks = width.div_ceil(4) as usize * height.div_ceil(4) as usize;
    blocks * if bc3 { 16 } else { 8 }
}

fn compress(rgba: &[u8], width: u32, height: u32, bc3: bool) -> Vec<u8> {
    let bw = width.div_ceil(4);
    let bh = height.div_ceil(4);
    let mut out = Vec::with_capacity((bw * bh) as usize * if bc3 { 16 } else { 8 });
    for by in 0..bh {
        for bx in 0..bw {
            let block = gather_block(rgba, width, height, bx * 4, by * 4);
            if bc3 {
                encode_alpha_block(&block, &mut out);
            }
            encode_color_block(&block, &mut out);
        }
    }
    out
}

/// The 16 RGBA texels of one 4×4 block, edge-clamped for partial blocks.
fn gather_block(rgba: &[u8], width: u32, height: u32, x0: u32, y0: u32) -> [[u8; 4]; 16] {
    let mut block = [[0u8; 4]; 16];
    for j in 0..4u32 {
        for i in 0..4u32 {
            let x = (x0 + i).min(width - 1);
            let y = (y0 + j).min(height - 1);
            let idx = ((y * width + x) * 4) as usize;
            block[(j * 4 + i) as usize] = [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]];
        }
    }
    block
}

fn to_565(c: [u8; 3]) -> u16 {
    ((c[0] as u16 >> 3) << 11) | ((c[1] as u16 >> 2) << 5) | (c[2] as u16 >> 3)
}

fn from_565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1f) as u8;
    let g = ((c >> 5) & 0x3f) as u8;
    let b = (c & 0x1f) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// Encode an 8-byte BC1 color block (4-color opaque mode).
fn encode_color_block(block: &[[u8; 4]; 16], out: &mut Vec<u8>) {
    // Bounding box of the RGB channels.
    let mut lo = [255u8; 3];
    let mut hi = [0u8; 3];
    for px in block {
        for c in 0..3 {
            lo[c] = lo[c].min(px[c]);
            hi[c] = hi[c].max(px[c]);
        }
    }
    let mut c0 = to_565(hi);
    let mut c1 = to_565(lo);
    // 4-color mode requires c0 > c1; if equal, the block is a flat color.
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    // Build the 4-entry palette in RGB888.
    let e0 = from_565(c0);
    let e1 = from_565(c1);
    let palette = [
        e0,
        e1,
        lerp(e0, e1, 1, 3), // 2/3 e0 + 1/3 e1
        lerp(e0, e1, 2, 3), // 1/3 e0 + 2/3 e1
    ];

    out.extend_from_slice(&c0.to_le_bytes());
    out.extend_from_slice(&c1.to_le_bytes());
    let mut indices: u32 = 0;
    for (n, px) in block.iter().enumerate() {
        let idx = nearest(&palette, [px[0], px[1], px[2]]);
        indices |= (idx as u32) << (2 * n);
    }
    out.extend_from_slice(&indices.to_le_bytes());
}

/// `a*(k)/(d) + b*(d-k)/(d)` … actually: weight `a` by `(d-k)/d`. We compute the
/// two interpolated palette colors: index 2 = (2*e0 + e1)/3, index 3 = (e0 + 2*e1)/3.
fn lerp(a: [u8; 3], b: [u8; 3], num: u8, den: u8) -> [u8; 3] {
    let mut o = [0u8; 3];
    for c in 0..3 {
        let av = a[c] as u32 * (den - num) as u32;
        let bv = b[c] as u32 * num as u32;
        o[c] = ((av + bv) / den as u32) as u8;
    }
    o
}

fn nearest(palette: &[[u8; 3]; 4], px: [u8; 3]) -> usize {
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let dr = p[0] as i32 - px[0] as i32;
        let dg = p[1] as i32 - px[1] as i32;
        let db = p[2] as i32 - px[2] as i32;
        let d = (dr * dr + dg * dg + db * db) as u32;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Encode an 8-byte BC3/DXT5 alpha block (8 interpolated alphas, a0 > a1 mode).
fn encode_alpha_block(block: &[[u8; 4]; 16], out: &mut Vec<u8>) {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for px in block {
        lo = lo.min(px[3]);
        hi = hi.max(px[3]);
    }
    let a0 = hi;
    let a1 = lo;
    // 8-alpha palette: a0, a1, then 6 interpolations (a0 > a1 branch).
    let mut palette = [0u8; 8];
    palette[0] = a0;
    palette[1] = a1;
    if a0 > a1 {
        for i in 1..7u32 {
            palette[(i + 1) as usize] = (((7 - i) * a0 as u32 + i * a1 as u32) / 7) as u8;
        }
    } else {
        // Degenerate (flat alpha): only two entries matter.
        for p in palette.iter_mut().skip(2) {
            *p = a0;
        }
    }

    out.push(a0);
    out.push(a1);
    // 16 texels × 3 bits = 48 bits packed into 6 bytes.
    let mut bits: u64 = 0;
    for (n, px) in block.iter().enumerate() {
        let idx = nearest_alpha(&palette, px[3]) as u64;
        bits |= idx << (3 * n);
    }
    for b in 0..6 {
        out.push(((bits >> (8 * b)) & 0xff) as u8);
    }
}

fn nearest_alpha(palette: &[u8; 8], a: u8) -> usize {
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, &p) in palette.iter().enumerate() {
        let d = (p as i32 - a as i32).unsigned_abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

// ── decoders (CPU) ───────────────────────────────────────────

/// The BC1/BC3 **decoders moved to [`inf_vt::container`]** in P26.3 and are
/// re-exported here, so the thumbnailer's `bc::decode_bc1` still resolves.
///
/// They went with the tile reader for one measured reason: an adapter without
/// `TEXTURE_COMPRESSION_BC` pages `TiledTextureReader::tile_rgba8` into an RGBA8
/// pool, and that arm is the **shipped player's** on every mobile target — which
/// does not link this crate. Leaving the decoder here would have left the
/// transcode tier reachable in the editor and unreachable in the player: the
/// exact shape of the P26.1 audit's capability-clamp finding, a second time.
///
/// The **encoder stays**, with the no-floating-point source gate below that
/// guards it — a texture's bytes are content-hashed into a reproducible pack, so
/// one `as f32` in an endpoint fit would make them a property of the machine that
/// imported it.
pub use inf_vt::container::{decode_bc1, decode_bc3};

#[cfg(test)]
mod tests {

    /// **The BC encoder never touches a float, and that is a LAW** (P26.1 audit).
    ///
    /// P26.1 makes this encoder the writer of every tile in every shipped
    /// `.inf_tex`, and the batch's upload proof rests out loud on "hand-rolled
    /// integer arithmetic with no floating point anywhere, so its output is the
    /// same bytes on every target". That is exactly the shape of P14's trig LAW —
    /// `f32` transcendentals are not bit-portable, and neither is `a * b + c`
    /// under a target's contraction rules — and it was a sentence in a commit
    /// message with nothing enforcing it.
    ///
    /// What one `as f32` in an endpoint fit would cost: a texture's content hash
    /// becomes machine-dependent, so two developers importing one PNG write
    /// different bytes, the import cache disagrees with itself, `.inf_pack` stops
    /// being reproducible, and every "re-import is byte-identical" arm in this
    /// repository passes on each machine separately while being false between
    /// them. Nothing here would fail; the CI is one machine per job.
    ///
    /// Scoped to the code ABOVE this test module — a ban list has to be able to
    /// name the tokens it bans.
    #[test]
    fn the_encoder_never_touches_a_float() {
        let whole = include_str!("bc.rs");
        let marker = "#[cfg(test)]";
        let (code, _) = whole
            .split_once(marker)
            .expect("the test module marker moved; this gate scopes on it");
        let code: String = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["f32", "f64", "sqrt", "powf", "powi", "as f"] {
            assert!(
                !code.contains(banned),
                "`{banned}` reached the BC encoder: its output is no longer the \
                 same bytes on every target, and a texture's content hash becomes \
                 a property of the machine that imported it"
            );
        }
        // A bare `0.5` is an f64 with no annotation to grep for.
        let chars: Vec<char> = code.chars().collect();
        assert!(
            !chars
                .windows(3)
                .any(|w| w[0].is_ascii_digit() && w[1] == '.' && w[2].is_ascii_digit()),
            "a decimal literal reached the BC encoder"
        );
        // Anti-vacuity: the filter left the code behind, not only the comments.
        // Both halves of the encoder — the entry point and the innermost block
        // fit — because a filter that ate the body would leave the signature.
        // (`fn decode_bc3` used to stand here; P26.3 moved the DECODERS to
        // `inf_vt::container` and left the encoder, so naming one would now
        // assert the gate onto code that is not in this file.)
        assert!(
            code.contains("fn compress_bc1") && code.contains("fn encode_alpha_block"),
            "the source filter ate the encoder"
        );
    }
    use super::*;

    #[test]
    fn bc1_size_is_correct() {
        assert_eq!(compressed_size(16, 16, false), 16 / 4 * 16 / 4 * 8);
        assert_eq!(compress_bc1(&vec![0u8; 16 * 16 * 4], 16, 16).len(), 128);
        // Non-multiple-of-4 rounds up.
        assert_eq!(compressed_size(5, 5, false), 2 * 2 * 8);
    }

    #[test]
    fn bc1_solid_color_round_trips_closely() {
        let color = [200u8, 100, 40, 255];
        let rgba: Vec<u8> = color.iter().cloned().cycle().take(16 * 16 * 4).collect();
        let comp = compress_bc1(&rgba, 16, 16);
        let dec = decode_bc1(&comp, 16, 16);
        // 565 quantization error is small and bounded.
        for c in 0..3 {
            let diff = (dec[c] as i32 - color[c] as i32).abs();
            assert!(diff <= 8, "channel {c}: {} vs {}", dec[c], color[c]);
        }
    }

    #[test]
    fn bc3_size_and_alpha_endpoints() {
        // A block with a clear alpha gradient: encoder should span the range.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for (n, px) in rgba.chunks_exact_mut(4).enumerate() {
            px[3] = (n * 17).min(255) as u8;
        }
        let comp = compress_bc3(&rgba, 4, 4);
        assert_eq!(comp.len(), 16);
        // First two bytes are the alpha endpoints (a0 = max, a1 = min).
        assert_eq!(comp[0], 255);
        assert_eq!(comp[1], 0);
    }
}
