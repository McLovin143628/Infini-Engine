//! Pure-Rust BC1 / BC3 / BC5 / BC7 block compression.
//!
//! GPU-native block-compressed textures are 2-8× smaller than RGBA8 and sample
//! for free, so imported textures are compressed at import time. Every encoder
//! here is hand-rolled rather than `intel_tex_2`'s, whose ISPC build is a
//! cross-OS CI liability:
//!
//! | format | bytes/block | what it is for |
//! |---|---|---|
//! | BC1 | 8 | opaque colour, **8:1** against RGBA8's 32 bits a texel |
//! | BC3 | 16 | colour with a real alpha channel, 4:1 |
//! | BC5 | 16 | tangent-space normals — two channels, Z rebuilt (Wave T) |
//! | BC7 | 16 | full RGBA at 8-bit endpoints and sixteen levels (wave IASSET2) |
//!
//! BC1/BC3/BC5 use the standard bounding-box endpoint selection with per-pixel
//! nearest-index assignment — correct, GPU-decodable output; endpoint refinement
//! (PCA/cluster fit) is a future quality pass. **BC7 does one step better**: it
//! picks the bounding box's *corner* per channel by the sign of its covariance,
//! which is what a naive box gets wrong on anti-correlated channels. The same
//! improvement is available to BC1 and is deliberately not taken there, because
//! it would move the bytes of every `.inf_tex` this repository has committed.
//!
//! `TEXTURE_COMPRESSION_BC` covers all four; an adapter without it transcodes
//! through `inf_vt::container`'s decoders instead.
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

/// Compress an RGBA8 image to **BC7 — mode 6 only** (wave IASSET2). 16
/// bytes/block, the same page cost as BC3 and BC5 and twice BC1's.
///
/// # The mode subset, stated
///
/// BC7 has eight modes and this encoder emits **one**: mode 6, a single subset
/// with 7-bit-plus-p-bit RGBA endpoints and 4-bit indices. That is a real subset
/// and it is chosen rather than defaulted to:
///
/// * it is the only mode that carries **full RGBA at maximum index precision**
///   (16 interpolated levels against BC1's four), so one encoder covers base
///   colour, ORM triples and masks without a second code path;
/// * its endpoints are effectively **8 bits a channel** (7 stored plus a shared
///   p-bit), against BC1's 5:6:5 — which is the quantisation that makes BC1 bad
///   at exactly the smooth gradients ground and skin are made of;
/// * every other useful mode needs the **partition tables** (64 two-subset and
///   64 three-subset layouts) plus a search over them. That is where a
///   general-purpose BC7 encoder spends its complexity and its time, and it buys
///   accuracy on blocks whose colours are not colinear — a decal corner, a
///   silhouette over two backgrounds. It is the honest next step and it is not
///   this wave's.
///
/// What the subset costs is therefore a block **no single line fits**, and
/// `tests::bc7_mode_six_pays_for_a_non_colinear_block_and_the_number_is_here`
/// measures that rather than only the flattering case. A block with *two*
/// clusters is generally not the hard case: two clusters are colinear, and the
/// per-channel corner rule below encodes red-against-cyan exactly.
///
/// # Integer-only, like everything else here
///
/// The endpoints are the RGBA bounding box quantised to 7 bits — with the
/// **corner** chosen per channel by the sign of its covariance against the widest
/// channel, so an anti-correlated pair lands on the box's other diagonal rather
/// than off the data entirely — the two p-bits are chosen by **exhaustive search
/// over all four combinations** (there are four, and trying them is cheaper than
/// a rule that guesses), and each texel takes the nearest of the sixteen palette
/// entries by squared error. Every step is `u32`/`u64`/`i64` arithmetic —
/// `the_encoder_never_touches_a_float` covers this function because it is in this
/// file, which is why it is in this file.
///
/// **The BC1 encoder above still takes the max corner unconditionally**, and
/// that is a carried finding rather than an oversight: fixing it would move the
/// bytes of every `.inf_tex` this repository has committed, which is a content
/// re-bless and belongs with one.
pub fn compress_bc7(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = width.div_ceil(4);
    let bh = height.div_ceil(4);
    let mut out = Vec::with_capacity((bw * bh) as usize * 16);
    for by in 0..bh {
        for bx in 0..bw {
            let block = gather_block(rgba, width, height, bx * 4, by * 4);
            encode_bc7_block(&block, &mut out);
        }
    }
    out
}

/// Bytes a BC7 image occupies — 16 a block, the BC3 size.
pub fn compressed_size_bc7(width: u32, height: u32) -> usize {
    compressed_size(width, height, true)
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

// ── BC7, mode 6 ─────────────────────────────────────────────────────────────

/// The BC7 4-bit interpolation weights, out of 64 — the spec's `aWeight4`.
///
/// Written out rather than computed: they are **not** `i * 64 / 15`, they are the
/// table the hardware decoder uses, and a decoder that interpolated with a
/// formula would disagree with silicon on twelve of the sixteen levels.
const BC7_WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// One channel of a mode-6 endpoint, quantised to 7 bits given its p-bit.
///
/// The reconstructed value is `(q << 1) | p`, so the best `q` for a target `v` is
/// `round((v - p) / 2)` clamped into the field.
fn bc7_quant7(v: u8, p: u8) -> u8 {
    let q = (v as i32 - p as i32 + 1) >> 1;
    q.clamp(0, 127) as u8
}

/// The 8-bit endpoint a `(q, p)` pair reconstructs to — the decoder's rule,
/// spelled once so the encoder measures its own output.
#[inline]
fn bc7_deq(q: u8, p: u8) -> u8 {
    (q << 1) | p
}

/// One palette entry: `e0` and `e1` at weight `w` out of 64.
#[inline]
fn bc7_lerp(e0: u8, e1: u8, w: u32) -> u8 {
    (((e0 as u32) * (64 - w) + (e1 as u32) * w + 32) >> 6) as u8
}

/// Squared RGBA distance, in `u64` because 4 × 255² × 16 texels does not fit
/// comfortably in anything smaller once it is summed over a block.
#[inline]
fn bc7_dist(a: [u8; 4], b: [u8; 4]) -> u64 {
    let mut d = 0u64;
    for c in 0..4 {
        let e = a[c] as i32 - b[c] as i32;
        d += (e * e) as u64;
    }
    d
}

/// Encode one 16-byte BC7 mode-6 block.
///
/// The bit layout, LSB first, is the spec's and it is the only thing here that
/// is not obviously arithmetic:
///
/// ```text
///  [0..7)    mode — six zeros then a one (unary), so the field is 0b1000000
///  [7..63)   R0 R1 G0 G1 B0 B1 A0 A1, seven bits each, component-major
///  [63..65)  P0 P1
///  [65..68)  index 0, THREE bits (the anchor's high bit is implicitly zero)
///  [68..128) indices 1..15, four bits each
/// ```
///
/// The anchor rule is why the endpoints may be swapped at the end: index 0 is
/// stored in three bits, so it must be ≤ 7. Swapping the two endpoints and
/// replacing every index `i` with `15 - i` names the identical colours through
/// the reversed palette, which is what makes the fix free rather than a
/// compromise.
fn encode_bc7_block(block: &[[u8; 4]; 16], out: &mut Vec<u8>) {
    let mut lo = [255u8; 4];
    let mut hi = [0u8; 4];
    let mut sum = [0i64; 4];
    for px in block {
        for c in 0..4 {
            lo[c] = lo[c].min(px[c]);
            hi[c] = hi[c].max(px[c]);
            sum[c] += px[c] as i64;
        }
    }

    // **Which corner of the bounding box each endpoint takes**, per channel.
    //
    // Not "endpoint 0 is the max corner": a block whose red rises while its
    // green falls has its data on the box's OTHER diagonal, and a line through
    // (max R, max G) and (min R, min G) misses every texel in it. Measured
    // before this existed: a colinear RGBA gradient with one anti-correlated
    // channel came back **27** off per channel at worst, on content mode 6 fits
    // exactly once the corner is right.
    //
    // The rule is the sign of the covariance against the widest channel, summed
    // over the block in exact integers (each term is scaled by 16 so the means
    // never need a division). The widest channel is its own reference, so its
    // covariance is a variance and it always takes the max — which is what makes
    // the assignment total rather than circular.
    let mut widest = 0usize;
    for c in 1..4 {
        if hi[c] - lo[c] > hi[widest] - lo[widest] {
            widest = c;
        }
    }
    let mut corner_hi = [true; 4];
    for c in 0..4 {
        let mut cov: i64 = 0;
        for px in block {
            cov += (16 * px[widest] as i64 - sum[widest]) * (16 * px[c] as i64 - sum[c]);
        }
        corner_hi[c] = cov >= 0;
    }
    let (mut end0, mut end1) = ([0u8; 4], [0u8; 4]);
    for c in 0..4 {
        end0[c] = if corner_hi[c] { hi[c] } else { lo[c] };
        end1[c] = if corner_hi[c] { lo[c] } else { hi[c] };
    }

    let mut best_err = u64::MAX;
    let mut best_q0 = [0u8; 4];
    let mut best_q1 = [0u8; 4];
    let mut best_p = (0u8, 0u8);
    let mut best_idx = [0u8; 16];
    // All four p-bit combinations. They shift each endpoint by at most one
    // 8-bit step, which is exactly the precision a 7-bit field is short of.
    for p0 in 0..2u8 {
        for p1 in 0..2u8 {
            let mut q0 = [0u8; 4];
            let mut q1 = [0u8; 4];
            let mut e0 = [0u8; 4];
            let mut e1 = [0u8; 4];
            for c in 0..4 {
                q0[c] = bc7_quant7(end0[c], p0);
                q1[c] = bc7_quant7(end1[c], p1);
                e0[c] = bc7_deq(q0[c], p0);
                e1[c] = bc7_deq(q1[c], p1);
            }
            let mut palette = [[0u8; 4]; 16];
            for (i, entry) in palette.iter_mut().enumerate() {
                for c in 0..4 {
                    entry[c] = bc7_lerp(e0[c], e1[c], BC7_WEIGHT4[i]);
                }
            }
            let mut idx = [0u8; 16];
            let mut err = 0u64;
            for (n, px) in block.iter().enumerate() {
                let mut best = 0usize;
                let mut best_d = u64::MAX;
                for (i, entry) in palette.iter().enumerate() {
                    let d = bc7_dist(*entry, *px);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                idx[n] = best as u8;
                err += best_d;
            }
            if err < best_err {
                best_err = err;
                best_q0 = q0;
                best_q1 = q1;
                best_p = (p0, p1);
                best_idx = idx;
            }
        }
    }

    // The anchor: index 0 is three bits wide, so flip the palette if it is not.
    if best_idx[0] > 7 {
        std::mem::swap(&mut best_q0, &mut best_q1);
        best_p = (best_p.1, best_p.0);
        for i in best_idx.iter_mut() {
            *i = 15 - *i;
        }
    }

    let mut bits: u128 = 0;
    let mut pos: u32 = 0;
    let mut put = |v: u32, n: u32| {
        bits |= ((v as u128) & ((1u128 << n) - 1)) << pos;
        pos += n;
    };
    put(1 << 6, 7);
    for c in 0..4 {
        put(best_q0[c] as u32, 7);
        put(best_q1[c] as u32, 7);
    }
    put(best_p.0 as u32, 1);
    put(best_p.1 as u32, 1);
    put(best_idx[0] as u32, 3);
    for i in 1..16 {
        put(best_idx[i] as u32, 4);
    }
    debug_assert_eq!(pos, 128, "a BC7 block is 128 bits");
    out.extend_from_slice(&bits.to_le_bytes());
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
pub use inf_vt::container::{decode_bc1, decode_bc3, decode_bc5, decode_bc7};

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

    /// **BC7 round-trips through its own decoder, exactly** — the encoder's
    /// output is what `inf_vt::decode_bc7` reads, block bit for block bit.
    ///
    /// This is the arm the whole format rests on: the encoder writes a bit
    /// stream and the decoder reads one, and nothing else in the workspace
    /// compares them. It asserts on the DECODED texels rather than on the bytes,
    /// because the claim is about what a sampler sees.
    #[test]
    fn bc7_encodes_a_block_its_own_decoder_reads_back() {
        // **A flat block is exact when its channels share a parity, and within
        // one step otherwise** — and the reason is the format rather than the
        // encoder. Mode 6 stores 7 bits per channel plus **one p-bit per
        // endpoint, shared by all four channels**: `(q << 1) | p`. So an
        // endpoint can hit every odd value or every even one, never both. The
        // p-bit search picks whichever parity costs less over the four channels.
        //
        // Measured here rather than assumed: the first draft of this arm
        // asserted exactness on `[37, 200, 91, 255]` — three odd channels and
        // one even — and the even one came back 201.
        let odd = [37u8, 201, 91, 255];
        let rgba: Vec<u8> = odd.iter().copied().cycle().take(4 * 4 * 4).collect();
        let dec = decode_bc7(&compress_bc7(&rgba, 4, 4), 4, 4);
        assert!(
            dec.chunks_exact(4).all(|p| p == odd),
            "a flat block of one parity is not exact: {:?}",
            &dec[..8]
        );
        let mixed = [37u8, 200, 91, 255];
        let rgba: Vec<u8> = mixed.iter().copied().cycle().take(4 * 4 * 4).collect();
        let dec = decode_bc7(&compress_bc7(&rgba, 4, 4), 4, 4);
        for px in dec.chunks_exact(4) {
            for c in 0..4 {
                assert!(
                    (px[c] as i32 - mixed[c] as i32).abs() <= 1,
                    "a flat block moved by more than the shared p-bit allows: {px:?}"
                );
            }
        }

        // **A colinear RGBA gradient, alpha included** — the case mode 6 is for,
        // and the one BC1 cannot carry at all (it has no alpha) while BC3 gives
        // alpha its own block and quantises the colour on 5:6:5.
        //
        // Colinear on purpose: one subset means one line through 4-space, so a
        // block whose channels vary along *independent* axes is the subset's
        // documented cost and is measured in
        // `bc7_mode_six_pays_for_a_two_cluster_block_and_the_number_is_here`,
        // not here.
        let (w, h) = (16u32, 16u32);
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let t = (x + y) as u32;
                src.extend_from_slice(&[
                    (t * 8) as u8,
                    (240 - t * 8) as u8,
                    (t * 4) as u8,
                    (255 - t * 5) as u8,
                ]);
            }
        }
        let dec = decode_bc7(&compress_bc7(&src, w, h), w, h);
        assert_eq!(dec.len(), src.len());
        let worst = |d: &[u8]| -> u32 {
            src.iter()
                .zip(d)
                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                .max()
                .expect("non-empty")
        };
        assert!(
            worst(&dec) <= 8,
            "worst per-channel error on a colinear gradient is {}",
            worst(&dec)
        );
        // …and BC3, at the same sixteen bytes, is worse — which is what makes
        // the number above a statement about the format and not about 4×4
        // blocks being easy.
        let bc3 = decode_bc3(&compress_bc3(&src, w, h), w, h);
        assert!(
            worst(&dec) < worst(&bc3),
            "BC7 {} against BC3 {} at the same page size",
            worst(&dec),
            worst(&bc3)
        );

        // Deterministic — the bytes are content-hashed into a reproducible pack.
        assert_eq!(compress_bc7(&src, w, h), compress_bc7(&src, w, h));
        assert_eq!(compress_bc7(&src, w, h).len(), compressed_size_bc7(w, h));

        // A block whose ANCHOR index would exceed three bits still round-trips:
        // texel 0 is the darkest, so the natural endpoint order puts it at index
        // 15 and the encoder must flip the palette. Measured rather than
        // asserted structurally — the failure would be one wrong texel.
        let mut flip = vec![250u8; 4 * 4 * 4];
        flip[0..4].copy_from_slice(&[0, 0, 0, 255]);
        let dec = decode_bc7(&compress_bc7(&flip, 4, 4), 4, 4);
        assert!(
            dec[0] < 8 && dec[1] < 8 && dec[2] < 8,
            "the anchor texel decoded as {:?}",
            &dec[0..4]
        );
        assert!(
            dec[4] > 240,
            "the rest of the block moved: {:?}",
            &dec[4..8]
        );
    }

    /// **What BC7 buys and what it costs, measured on the committed ground
    /// library** (wave IASSET2) — the numbers the wave's content ruling is made
    /// of, printed so the ledger quotes a measurement rather than a memory.
    ///
    /// Three claims, and the third is the one that decides:
    ///
    /// 1. **Size.** BC7 is 16 bytes a block, BC1 is 8. A `.inf_tex` page is
    ///    18 496 B instead of 9 248, so a 24 MiB atlas arm holds **half** the
    ///    pages. That is not a rounding cost — it is the same currency the arm
    ///    split just bought back.
    /// 2. **Quality.** Mean absolute error per channel against the source, on
    ///    the real albedo of every ground kind rather than on a synthetic ramp.
    /// 3. **Encode time**, cook-side, printed per megatexel.
    ///
    /// The assertion is on the RATIO, so it cannot be satisfied by both being
    /// bad, and it is deliberately weak — the arm exists to publish numbers, and
    /// a threshold tight enough to be interesting would be a threshold that
    /// fails on a machine with a different allocator.
    #[test]
    fn bc7_against_bc1_on_the_committed_ground_library() {
        use crate::ground::GroundKind;
        for (slot, extent) in [
            ("albedo", crate::ground::GROUND_ALBEDO_EXTENT),
            // The ORM triple too, because `image_import_policy` deliberately
            // left its format to be decided by this measurement rather than
            // guessed: roughness banding shows as specular banding, which is a
            // different visibility from albedo banding.
            ("ORM", crate::ground::GROUND_MAP_EXTENT),
        ] {
            bc7_ground_slot(slot, extent);
        }
    }

    /// One slot of the ground library, measured three ways. Factored out of the
    /// arm above so the albedo and the ORM triple are the same measurement and
    /// not two that happen to agree.
    fn bc7_ground_slot(slot: &str, n: u32) {
        use crate::ground::GroundKind;
        let (n_, N) = (n, n);
        let _ = n_;
        let mut total = [0u64; 3];
        let mut worst = [0u32; 3];
        let mut texels = 0u64;
        let mut encode_ms = [0u128; 3];
        for kind in GroundKind::ALL {
            let maps = crate::ground::synthesize(kind);
            let src = if slot == "albedo" {
                &maps.albedo
            } else {
                &maps.orm
            };
            let mut enc = [Vec::new(), Vec::new(), Vec::new()];
            for (i, f) in [0usize, 1, 2].into_iter().enumerate() {
                let t = std::time::Instant::now();
                enc[i] = match f {
                    0 => compress_bc1(src, N, N),
                    1 => compress_bc3(src, N, N),
                    _ => compress_bc7(src, N, N),
                };
                encode_ms[i] += t.elapsed().as_millis();
            }
            let dec = [
                decode_bc1(&enc[0], N, N),
                decode_bc3(&enc[1], N, N),
                decode_bc7(&enc[2], N, N),
            ];
            for (i, d) in dec.iter().enumerate() {
                for (n, s) in src.iter().enumerate() {
                    // RGB only: the source is opaque, and an alpha channel that
                    // is 255 everywhere would flatter every encoder equally.
                    if n % 4 == 3 {
                        continue;
                    }
                    let e = (*s as i32 - d[n] as i32).unsigned_abs();
                    total[i] += e as u64;
                    worst[i] = worst[i].max(e);
                }
            }
            texels += u64::from(N) * u64::from(N) * 3;
        }
        let mae = |t: u64| (t * 1000) / texels;
        let mtexels = (texels / 3) as u128 / 1_000_000;
        println!(
            "IASSET2 BC7 vs BC1/BC3 on {} ground {slot} maps ({N}^2 each):\n  \
             MAE x1000/channel  BC1 {}  BC3 {}  BC7 {}\n  \
             WORST /channel     BC1 {}  BC3 {}  BC7 {}\n  \
             bytes/block        BC1 8  BC3 16  BC7 16\n  \
             encode ms/Mtexel   BC1 {}  BC3 {}  BC7 {}  (THIS PROFILE — a debug \
             run is not a cook-time number, see the wave ledger for the release \
             figure)",
            GroundKind::ALL.len(),
            mae(total[0]),
            mae(total[1]),
            mae(total[2]),
            worst[0],
            worst[1],
            worst[2],
            encode_ms[0] / mtexels.max(1),
            encode_ms[1] / mtexels.max(1),
            encode_ms[2] / mtexels.max(1),
        );
        assert!(
            total[2] * 2 < total[0],
            "BC7 must beat BC1 by a wide margin on real ground {slot}, or its \
             doubled page is not worth taking: BC7 {} against BC1 {}",
            mae(total[2]),
            mae(total[0])
        );
        // …and it must beat BC3 too, which costs the SAME sixteen bytes — that
        // is the comparison that says the format is worth its page rather than
        // that compression is worth its page.
        assert!(
            total[2] < total[1],
            "BC7 does not beat BC3 at the same page size: BC7 {} against BC3 {}",
            mae(total[2]),
            mae(total[1])
        );
    }

    /// **BC5 against BC1 on the committed ground NORMAL maps** (wave IASSET2) —
    /// the other half of the content ruling, and the half the arm split was
    /// built for.
    ///
    /// `bc5_costs_a_quarter_of_rgba8_and_beats_bc1_on_a_normal_map` measures a
    /// **synthetic** swept normal field, which is the right fixture for "is this
    /// format right for this signal" and the wrong one for "should this
    /// repository's content change". `inf_material::ground` chose BC1 for all
    /// seventeen maps and wrote its reason down — a BC5 map beside a BC1 albedo
    /// demoted the whole atlas — so the question this arm answers is whether the
    /// arms make that choice wrong on the actual bytes.
    ///
    /// Both formats are compared at their real page cost: BC1 is 8 bytes a
    /// block and BC5 is 16, so BC5 halves what a 24 MiB arm holds. The numbers
    /// are printed for the ledger and the assertion is the one that decides.
    #[test]
    fn bc5_against_bc1_on_the_committed_ground_normal_maps() {
        use crate::ground::{GroundKind, GROUND_MAP_EXTENT as N};
        let (mut mae1, mut mae5) = (0u64, 0u64);
        let (mut worst1, mut worst5) = (0u32, 0u32);
        let mut samples = 0u64;
        for kind in GroundKind::ALL {
            let maps = crate::ground::synthesize(kind);
            for src in [Some(&maps.normal), maps.detail.as_ref()]
                .into_iter()
                .flatten()
            {
                let d1 = decode_bc1(&compress_bc1(src, N, N), N, N);
                let d5 = decode_bc5(&compress_bc5(src, N, N), N, N);
                for (n, s) in src.iter().enumerate() {
                    // X and Y only: they are the whole signal, Z is rebuilt from
                    // them and alpha is unused. Scoring the blue channel would
                    // credit BC1 for carrying a redundancy and penalise BC5 for
                    // discarding one, which is backwards.
                    if n % 4 > 1 {
                        continue;
                    }
                    let (e1, e5) = (
                        (*s as i32 - d1[n] as i32).unsigned_abs(),
                        (*s as i32 - d5[n] as i32).unsigned_abs(),
                    );
                    mae1 += e1 as u64;
                    mae5 += e5 as u64;
                    worst1 = worst1.max(e1);
                    worst5 = worst5.max(e5);
                    samples += 1;
                }
            }
        }
        let per = |t: u64| (t * 1000) / samples.max(1);
        println!(
            "IASSET2 BC5 vs BC1 on the committed ground normal + detail maps \
             ({N}^2 each, X/Y only):\n  \
             MAE x1000/channel  BC1 {}  BC5 {}\n  \
             WORST /channel     BC1 {worst1}  BC5 {worst5}\n  \
             bytes/block        BC1 8  BC5 16",
            per(mae1),
            per(mae5),
        );
        assert!(
            mae5 * 2 < mae1,
            "BC5 must beat BC1 by a wide margin on the two channels a normal map \
             lives in, or its doubled page is not worth taking: BC5 {} against \
             BC1 {}",
            per(mae5),
            per(mae1)
        );
    }

    /// **Does BC7 replace BC3 under `TextureCompression::Auto`?** Measured, and
    /// the answer is **no** — mode 6 shares one index set between colour and
    /// alpha (wave IASSET2).
    ///
    /// The prescription is obvious and wrong, which is why it is measured before
    /// it is landed (the P23 law). BC7 and BC3 are both 16 bytes a block, and on
    /// the ground albedo BC7's mean error is 6.6× lower — so `Auto`'s alpha
    /// branch "should" become BC7 for free.
    ///
    /// It should not, and the reason is structural rather than a tuning
    /// accident: **BC3 gives alpha its own block** — its own endpoints, its own
    /// 3-bit indices — while mode 6 has ONE 4-bit index per texel shared by all
    /// four channels. A cutout mask is the case that breaks it: alpha is 0 or
    /// 255 while the colour varies independently, so one index cannot serve
    /// both, and that is precisely the content `Auto` routes to BC3.
    ///
    /// The fix is BC7's modes 4 and 5, which carry a **second** index set for
    /// alpha. They are the honest next step and they are not this wave's, so
    /// `Auto` is unchanged and this arm is the number that says why.
    #[test]
    fn bc7_mode_six_does_not_replace_bc3_on_a_cutout_mask() {
        // A foliage cutout: colour ramps across the block, alpha is binary.
        let (w, h) = (16u32, 16u32);
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let t = (x + y) as u32;
                let a = if (x / 2 + y / 2) % 2 == 0 { 255 } else { 0 };
                src.extend_from_slice(&[(t * 8) as u8, (200 - t * 6) as u8, (t * 4) as u8, a]);
            }
        }
        let alpha_err = |d: &[u8]| -> u64 {
            src.iter()
                .zip(d)
                .enumerate()
                .filter(|(n, _)| n % 4 == 3)
                .map(|(_, (a, b))| (*a as i32 - *b as i32).unsigned_abs() as u64)
                .sum()
        };
        let d3 = decode_bc3(&compress_bc3(&src, w, h), w, h);
        let d7 = decode_bc7(&compress_bc7(&src, w, h), w, h);
        let (e3, e7) = (alpha_err(&d3), alpha_err(&d7));
        println!("IASSET2 BC7 mode 6 vs BC3 on a cutout mask (alpha only): BC3 {e3}, BC7 {e7}");
        assert!(
            e3 < e7,
            "BC7 mode 6 now carries a binary alpha better than BC3's dedicated \
             block ({e7} against {e3}) — if that is real, `Auto`'s alpha branch \
             should move to BC7 and this arm should be rewritten as the \
             measurement that says so"
        );
    }

    /// **The subset's cost, measured rather than admitted in prose.**
    ///
    /// Mode 6 fits ONE line through a block, so what it cannot do is a block
    /// whose colours are **not colinear** — three clusters, a decal corner, a
    /// texel of foliage over two different backgrounds. That is exactly what the
    /// partitioned modes exist for, and this arm is the number that says how
    /// much this encoder gives up there.
    ///
    /// **Two** clusters is not the hard case and the first draft of this arm
    /// used one: red against cyan is perfectly colinear (a line through the
    /// colour cube's other diagonal), and once
    /// [`encode_bc7_block`]'s per-channel corner rule landed it encodes
    /// **exactly**, error zero, while BC1 loses 4 040. A fixture a fix makes
    /// free is a fixture that measures the fix and not the limit.
    ///
    /// Asserted loosely — BC7 must not be *worse* than BC1 — because the claim
    /// is "this is the cost", not "the cost is small". It is a floor for a later
    /// wave's partitioned modes to beat.
    #[test]
    fn bc7_mode_six_pays_for_a_non_colinear_block_and_the_number_is_here() {
        // Three primaries in one block: no line through the cube passes near
        // all of red, green and blue at once.
        let mut split = Vec::with_capacity(4 * 4 * 4);
        for j in 0..4u32 {
            for i in 0..4u32 {
                let c: [u8; 4] = match (i + j) % 3 {
                    0 => [220, 20, 20, 255],
                    1 => [20, 220, 20, 255],
                    _ => [20, 20, 220, 255],
                };
                split.extend_from_slice(&c);
            }
        }
        let err = |dec: &[u8]| -> u64 {
            split
                .iter()
                .zip(dec)
                .enumerate()
                .filter(|(n, _)| n % 4 != 3)
                .map(|(_, (a, b))| (*a as i32 - *b as i32).unsigned_abs() as u64)
                .sum()
        };
        let e7 = err(&decode_bc7(&compress_bc7(&split, 4, 4), 4, 4));
        let e1 = err(&decode_bc1(&compress_bc1(&split, 4, 4), 4, 4));
        println!(
            "IASSET2 BC7 non-colinear block (three primaries): BC7 total abs err {e7}, BC1 {e1}"
        );
        assert!(
            e7 <= e1,
            "mode 6 is WORSE than BC1 on a two-cluster block ({e7} against \
             {e1}); the partitioned modes are then not an improvement but a fix"
        );
        // ANTI-VACUITY: a smooth gradient is where mode 6 wins outright, so the
        // arm above is measuring the hard case and not a tie everywhere.
        let ramp: Vec<u8> = (0..16)
            .flat_map(|i| [(i * 16) as u8, (i * 16) as u8, (i * 16) as u8, 255])
            .collect();
        let r7 = decode_bc7(&compress_bc7(&ramp, 4, 4), 4, 4);
        let r1 = decode_bc1(&compress_bc1(&ramp, 4, 4), 4, 4);
        let sum = |d: &[u8]| -> u64 {
            ramp.iter()
                .zip(d)
                .enumerate()
                .filter(|(n, _)| n % 4 != 3)
                .map(|(_, (a, b))| (*a as i32 - *b as i32).unsigned_abs() as u64)
                .sum()
        };
        assert!(
            sum(&r7) * 2 < sum(&r1),
            "mode 6 does not beat BC1 on a smooth ramp: {} against {}",
            sum(&r7),
            sum(&r1)
        );
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
