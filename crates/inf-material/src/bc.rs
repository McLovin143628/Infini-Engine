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

/// Compress an RGBA8 image to **BC5 / RGTC2** — the normal-map format (Wave T).
/// 16 bytes/block: two independent one-channel blocks, R then G. The source's
/// blue and alpha are **discarded**, because that is the point.
///
/// # Why this is the right format for a normal map, in one paragraph
///
/// A tangent-space normal is a unit vector, so its Z is not data: it is
/// `sqrt(1 − x² − y²)`, recoverable exactly from the other two. Storing it costs
/// a third of the signal budget to carry a redundancy, and every endpoint scheme
/// that packs three channels into shared endpoints then spends part of X and Y's
/// precision describing it. BC1 is the worst case and is what a "just compress
/// it" import would reach for: its 5:6:5 endpoints quantise the red axis to 32
/// levels, which is *the* axis a normal map's X lives on. That is why normal maps
/// have shipped **uncompressed** in this engine until now — 73 984 B a page — and
/// why this format is a 4× saving rather than a quality trade: BC5 gives each of
/// X and Y its own pair of 8-bit endpoints and eight interpolated levels, which
/// is strictly more precision per surviving channel than the RGBA8 it replaces
/// has per texel of gradient.
///
/// Each half is byte-for-byte the BC4 block the BC3 alpha encoder below already
/// writes, which is why this is a short function and not a new encoder.
pub fn compress_bc5(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = width.div_ceil(4);
    let bh = height.div_ceil(4);
    let mut out = Vec::with_capacity((bw * bh) as usize * 16);
    for by in 0..bh {
        for bx in 0..bw {
            let block = gather_block(rgba, width, height, bx * 4, by * 4);
            encode_bc4_block(&block, 0, &mut out);
            encode_bc4_block(&block, 1, &mut out);
        }
    }
    out
}

/// Bytes a compressed image occupies: `ceil(w/4) * ceil(h/4) * block_bytes`.
pub fn compressed_size(width: u32, height: u32, bc3: bool) -> usize {
    let blocks = width.div_ceil(4) as usize * height.div_ceil(4) as usize;
    blocks * if bc3 { 16 } else { 8 }
}

/// Bytes a BC5 image occupies — two 8-byte blocks per 4×4, i.e. the BC3 size.
pub fn compressed_size_bc5(width: u32, height: u32) -> usize {
    compressed_size(width, height, true)
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
///
/// A BC3 alpha block and a BC4/RGTC1 block are the **same eight bytes** — which
/// is why BC5 above is two calls to [`encode_bc4_block`] and not a second
/// encoder.
fn encode_alpha_block(block: &[[u8; 4]; 16], out: &mut Vec<u8>) {
    encode_bc4_block(block, 3, out)
}

/// The one-channel block: 8 bytes describing `channel` of a 4×4 as two endpoints
/// plus 3-bit indices into an 8-entry interpolated palette.
fn encode_bc4_block(block: &[[u8; 4]; 16], channel: usize, out: &mut Vec<u8>) {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for px in block {
        lo = lo.min(px[channel]);
        hi = hi.max(px[channel]);
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
        let idx = nearest_alpha(&palette, px[channel]) as u64;
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
pub use inf_vt::container::{decode_bc1, decode_bc3, decode_bc5};

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
    /// different bytes, the import cache disagrees with itself, `.ipack` stops
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
        // (`fn encode_alpha_block` stood here until Wave T made it a one-line
        // forwarder onto `encode_bc4_block`; naming the forwarder would have
        // asserted the gate onto a signature and not onto the block fit.)
        assert!(
            code.contains("fn compress_bc1")
                && code.contains("fn compress_bc5")
                && code.contains("fn encode_bc4_block"),
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

    /// **BC5 is the normal-map format, and this is the number that says so**
    /// (Wave T).
    ///
    /// Three claims, measured rather than argued:
    ///
    /// 1. **Size.** A stored virtual-texture tile is 136² texels. As RGBA8 —
    ///    what every normal map in this engine shipped as until now — that is
    ///    73 984 B a page; as BC5 it is 18 496 B. Exactly 4×.
    /// 2. **Quality against the alternative that was actually available.** BC1
    ///    is what "just compress the normal map" reaches for, and its 5:6:5
    ///    endpoints quantise the red channel to 32 levels — the channel a
    ///    tangent-space X lives in. On a swept normal field BC5's per-channel
    ///    error is a fraction of BC1's, and the assertion is on the *ratio*, so
    ///    it cannot be satisfied by both being bad.
    /// 3. **Z is redundant, so discarding it is free.** A unit normal's Z is
    ///    `sqrt(1 − x² − y²)`; the reconstruction is asserted here on the CPU
    ///    against the source, which is the same arithmetic `vt_sample.wgsl`'s
    ///    `vt_normal_ts` does on the GPU.
    #[test]
    fn bc5_costs_a_quarter_of_rgba8_and_beats_bc1_on_a_normal_map() {
        // A swept tangent-space normal field: X and Y ramp independently, so no
        // block is flat and both channels carry a real gradient.
        let (w, h) = (64u32, 64u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        let mut truth: Vec<[f64; 3]> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let nx = (x as f64 / (w - 1) as f64) * 1.4 - 0.7;
                let ny = (y as f64 / (h - 1) as f64) * 1.4 - 0.7;
                let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
                truth.push([nx, ny, nz]);
                let enc = |v: f64| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                rgba.extend_from_slice(&[enc(nx), enc(ny), enc(nz), 255]);
            }
        }

        // (1) Size. A 136² stored tile, the unit the atlas allocates in.
        assert_eq!(compressed_size_bc5(136, 136), 34 * 34 * 16);
        assert_eq!(compressed_size_bc5(136, 136), 18_496);
        assert_eq!(136usize * 136 * 4, 73_984);
        assert_eq!(73_984 / compressed_size_bc5(136, 136), 4);
        assert_eq!(compress_bc5(&rgba, w, h).len(), compressed_size_bc5(w, h));

        // (2) Quality, per channel, against BC1 — the alternative that existed.
        let bc5 = decode_bc5(&compress_bc5(&rgba, w, h), w, h);
        let bc1 = decode_bc1(&compress_bc1(&rgba, w, h), w, h);
        let mut err5 = 0f64;
        let mut err1 = 0f64;
        for i in 0..(w * h) as usize {
            for c in 0..2 {
                err5 += (bc5[i * 4 + c] as f64 - rgba[i * 4 + c] as f64).abs();
                err1 += (bc1[i * 4 + c] as f64 - rgba[i * 4 + c] as f64).abs();
            }
        }
        let n = (w * h * 2) as f64;
        let (mae5, mae1) = (err5 / n, err1 / n);
        assert!(
            mae5 * 3.0 < mae1,
            "BC5 must beat BC1 on the two channels a normal map lives in by a \
             wide margin: BC5 MAE {mae5:.3}, BC1 MAE {mae1:.3}"
        );
        assert!(
            mae5 < 1.0,
            "BC5 MAE on X/Y is {mae5:.3}, expected well under 1"
        );

        // (3) The rebuild. `sqrt(1 - x^2 - y^2)` off the two stored channels is
        // the source normal back, to within the quantisation of X and Y.
        let mut worst = 0f64;
        let mut total = 0f64;
        for (i, t) in truth.iter().enumerate() {
            let x = bc5[i * 4] as f64 / 255.0 * 2.0 - 1.0;
            let y = bc5[i * 4 + 1] as f64 / 255.0 * 2.0 - 1.0;
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            let d = ((x - t[0]).powi(2) + (y - t[1]).powi(2) + (z - t[2]).powi(2)).sqrt();
            worst = worst.max(d);
            total += d;
        }
        let mean = total / truth.len() as f64;
        // **The worst case is the rim of the unit disc, and that is arithmetic
        // rather than a defect.** `z = sqrt(1 − x² − y²)` has an infinite slope
        // where `z → 0`, so a normal lying almost in the tangent plane amplifies
        // the quantisation of X and Y — this fixture deliberately sweeps out to
        // `|xy| = 0.99`, i.e. all the way onto that rim. Both numbers are
        // asserted because the mean is what a surface looks like and the worst
        // case is what a silhouette does.
        assert!(
            worst < 0.03 && mean < 0.005,
            "the rebuilt normal is {worst:.4} away from the source at worst, \
             {mean:.5} on average"
        );
        // The blue channel is genuinely gone — this is a two-channel format and
        // an arm that passed while B survived would be testing nothing.
        assert!(bc5.chunks_exact(4).all(|p| p[2] == 0));
    }

    /// The encoder is a **pure function of its input**, for the format Wave T
    /// added as much as for the two that came before — a texture's bytes are
    /// content-hashed into a reproducible pack.
    #[test]
    fn bc5_is_deterministic() {
        let rgba: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i * 37 % 253) as u8).collect();
        assert_eq!(compress_bc5(&rgba, 16, 16), compress_bc5(&rgba, 16, 16));
        // And it is NOT the BC3 encoding of the same block: BC5 carries R and G,
        // BC3 carries alpha and colour.
        assert_ne!(compress_bc5(&rgba, 16, 16), compress_bc3(&rgba, 16, 16));
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
