//! **Per-block compression** (IASSET1): the codec a *streaming container* applies
//! to one tile / page / cell, as opposed to the whole-entry zstd
//! [`PackWriter`](crate::PackWriter) applies to an authored asset.
//!
//! # Why this exists beside the pack's own compression
//!
//! A `.ipack` entry is compressed **whole** (P9.2). That is right for
//! an authored payload read once at level load, and wrong for a container a
//! runtime pages *one unit at a time*: reaching one 581 KiB terrain tile out of a
//! 550 MB `.inf_terrain` would decode all 550 MB. So the streaming kinds opted out
//! of pack compression entirely ([`BlockCompressed`] / [`MappedInPlace`]) and
//! shipped **raw** — which bought the streaming
//! latency and paid for it in ship size, at 100% of the raw bytes.
//!
//! This module is the third option the two-way choice was hiding: compress each
//! **block** independently, record the codec in the container's own directory, and
//! decompress exactly the one block a page-in asked for. The ship-size win comes
//! back; the "decode the world to reach a tile" catastrophe does not.
//!
//! # What may and may not use it
//!
//! * **May**: a block the loader *already copies or decodes* on its way to being
//!   used. A `.inf_terrain` tile is `bincode`-decoded into a `TerrainTile` on
//!   every page-in — the borrowed slice is an input to a decoder, never a cast —
//!   so a decompress in front of that decode is an addition to an existing cost,
//!   not the destruction of a zero-copy path.
//! * **May NOT**: a block that is **cast in place**. `.inf_vmesh` parses its
//!   vertex/index sections with `bytemuck` casts straight off the mapping, and a
//!   `.inf_tex` tile goes to `write_texture` from the mapping. Those are the mmap
//!   doctrine's subjects and they stay raw — a compressed block cannot be cast,
//!   and (see [`encode_block`]) does not even keep its 16-byte address alignment.
//!
//! # Wire form
//!
//! A block whose codec is [`BlockCodec::Raw`] is *literally the raw bytes* — so a
//! container that compresses nothing is byte-identical to the same container
//! before this module existed, and the "compression is lossless" claim has a
//! trivial arm.
//!
//! A block under any other codec is:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ raw_len   u64 LE   the decompressed length, exactly        │
//! │ frame     [u8]     the codec's own bytes                   │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! `raw_len` is **written by whoever made the file**, so it is treated as a claim,
//! never as an allocation instruction: [`decode_block`] refuses a claim above the
//! caller's `ceiling`, decodes into exactly that many bytes, and requires the
//! output to come back the *same* length — not merely "no bigger". (The pack's own
//! decoder learned the weaker version of this lesson at round-2 finding B12; this
//! one is the strict form, because a block ceiling is a property of the container
//! and is therefore actually knowable.)
//!
//! [`BlockCompressed`]: crate::EntryPolicy::BlockCompressed
//! [`MappedInPlace`]: crate::EntryPolicy::MappedInPlace

use std::borrow::Cow;

use crate::error::{AssetError, Result};

/// The codec one block of a streaming container is stored under.
///
/// The discriminants are **wire values**: they live in a container's directory
/// and every asset ever cooked reads its blocks by them. Append, never insert or
/// renumber — the same law [`kind_code`](crate::pack) is held to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum BlockCodec {
    /// Stored verbatim. The only codec a *cast-in-place* block may use, and the
    /// one a compressible block falls back to when compression would inflate it.
    #[default]
    Raw = 0,
    /// LZ4 block format via `lz4_flex` — pure Rust, identical on native and
    /// wasm32.
    Lz4 = 1,
    /// Raw DEFLATE via `miniz_oxide` — pure Rust, identical on native and
    /// wasm32. Better ratio than LZ4, slower to decode.
    Deflate = 2,
    /// zstd — the C `zstd` natively, the pure-Rust `ruzstd` on wasm32. Best
    /// ratio; the only codec whose two implementations are different code.
    Zstd = 3,
}

impl BlockCodec {
    /// The wire value.
    #[inline]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The codec a wire value names, or `None` for one this build does not know
    /// (a container written by a newer engine — rejected, never guessed).
    #[inline]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Raw),
            1 => Some(Self::Lz4),
            2 => Some(Self::Deflate),
            3 => Some(Self::Zstd),
            _ => None,
        }
    }

    /// A short stable name, for reports and tables.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Lz4 => "lz4",
            Self::Deflate => "deflate",
            Self::Zstd => "zstd",
        }
    }

    /// Every codec, in wire order — the bake-off's subject list.
    pub const ALL: [BlockCodec; 4] = [Self::Raw, Self::Lz4, Self::Deflate, Self::Zstd];
}

/// Bytes of the `raw_len` prefix a compressed block carries.
pub const LEN_PREFIX: usize = 8;

/// Below this many bytes a block is stored [`Raw`](BlockCodec::Raw) whatever the
/// policy asks for: the prefix plus any codec's frame overhead reliably loses on
/// a tiny block, and a container full of inflated blocks is the one outcome
/// per-block compression must never produce.
pub const MIN_COMPRESSIBLE: usize = 256;

/// DEFLATE level used for cooked blocks. Cook is offline; 8 is within noise of 9
/// on the terrain corpus and materially faster to encode.
const DEFLATE_LEVEL: u8 = 8;

/// zstd level used for cooked blocks. **Not** the pack's 19: a per-block frame is
/// re-encoded on every cook of a 2 000-tile terrain, and the measured ratio
/// difference between 12 and 19 on tile blobs is under a point.
#[cfg(not(target_arch = "wasm32"))]
const ZSTD_BLOCK_LEVEL: i32 = 12;

/// Compress `raw` under `codec`, returning the block's **wire bytes**.
///
/// [`Raw`](BlockCodec::Raw) **borrows the input unchanged**. Every other codec
/// returns an owned `raw_len` prefix plus its frame — **and falls back to
/// [`Raw`](BlockCodec::Raw)** (reported in the returned codec, and borrowing)
/// whenever the result would not be strictly smaller than the input, or the input
/// is below [`MIN_COMPRESSIBLE`]. A block therefore never grows, and a container's
/// directory always states what actually happened rather than what was asked
/// for.
///
/// # Why this returns a `Cow` and not a `Vec`
///
/// The symmetric half of [`decode_block`]'s promise, and it was missed once: a
/// container that compresses **nothing** must not pay for the feature. Every
/// caller is a container builder walking its whole tile/chunk map, so a `Vec`
/// return made the loose (all-[`Raw`](BlockCodec::Raw)) writer materialize a
/// second full copy of the payload — a 550 MB `.inf_terrain` write-back went from
/// a ~2× transient to ~3×, for a copy whose only content was the bytes the
/// builder was already holding. Borrowing costs nothing on the compressed path
/// (that arm allocates its frame regardless) and removes the copy entirely on the
/// raw one.
///
/// # Alignment
///
/// A compressed block's payload starts at `+8`, so it is **not** 16-byte aligned
/// even when the block itself is. That is not a defect to fix — it is the reason
/// the mmap doctrine's cast-in-place kinds may not take this path at all. Nothing
/// casts a compressed block; the decompressed output is a fresh allocation whose
/// alignment the caller controls.
pub fn encode_block(codec: BlockCodec, raw: &[u8]) -> Result<(BlockCodec, Cow<'_, [u8]>)> {
    if codec == BlockCodec::Raw || raw.len() < MIN_COMPRESSIBLE {
        return Ok((BlockCodec::Raw, Cow::Borrowed(raw)));
    }
    let frame = match codec {
        BlockCodec::Raw => unreachable!("handled above"),
        BlockCodec::Lz4 => lz4_flex::block::compress(raw),
        BlockCodec::Deflate => miniz_oxide::deflate::compress_to_vec(raw, DEFLATE_LEVEL),
        BlockCodec::Zstd => zstd_encode_block(raw)?,
    };
    if frame.len() + LEN_PREFIX >= raw.len() {
        // Incompressible under this codec: never inflate.
        return Ok((BlockCodec::Raw, Cow::Borrowed(raw)));
    }
    let mut out = Vec::with_capacity(LEN_PREFIX + frame.len());
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    out.extend_from_slice(&frame);
    Ok((codec, Cow::Owned(out)))
}

/// Decode a block stored under `codec`, whose decompressed length the container
/// promises is at most `ceiling` bytes.
///
/// [`Raw`](BlockCodec::Raw) borrows — a container that compresses nothing keeps
/// the zero-copy read it always had. Every other codec allocates exactly the
/// declared length and requires the codec to fill it exactly.
pub fn decode_block(codec: BlockCodec, stored: &[u8], ceiling: usize) -> Result<Cow<'_, [u8]>> {
    if codec == BlockCodec::Raw {
        return Ok(Cow::Borrowed(stored));
    }
    if stored.len() < LEN_PREFIX {
        return Err(AssetError::Pack(format!(
            "a {} byte {} block is too short to hold its length prefix",
            stored.len(),
            codec.name()
        )));
    }
    let raw_len = u64::from_le_bytes(stored[..LEN_PREFIX].try_into().unwrap());
    // The claim is the file's, not ours. Bound it before it becomes an
    // allocation — a 40-byte block asking for gigabytes is the classic shape,
    // and this reader runs in the shipped player over a file it downloaded.
    if raw_len > ceiling as u64 {
        return Err(AssetError::Pack(format!(
            "a {} byte {} block declares {raw_len} decompressed bytes, past this \
             container's {ceiling} byte block ceiling (corrupt or hostile)",
            stored.len(),
            codec.name()
        )));
    }
    let raw_len = raw_len as usize;
    let frame = &stored[LEN_PREFIX..];
    let out = match codec {
        BlockCodec::Raw => unreachable!("handled above"),
        BlockCodec::Lz4 => lz4_flex::block::decompress(frame, raw_len)
            .map_err(|e| AssetError::Pack(format!("lz4: {e}")))?,
        BlockCodec::Deflate => miniz_oxide::inflate::decompress_to_vec_with_limit(frame, raw_len)
            .map_err(|e| AssetError::Pack(format!("deflate: {e:?}")))?,
        BlockCodec::Zstd => zstd_decode_block(frame, raw_len)?,
    };
    if out.len() != raw_len {
        return Err(AssetError::Pack(format!(
            "a {} block decompressed to {} bytes where its header declared {raw_len}",
            codec.name(),
            out.len()
        )));
    }
    Ok(Cow::Owned(out))
}

/// The decompressed length a block claims, without decompressing it — what a
/// container's `ls`/report path wants and a loader does not.
///
/// `None` when the block is [`Raw`](BlockCodec::Raw) (its stored length *is* its
/// decompressed length) or too short to carry a prefix.
pub fn declared_raw_len(codec: BlockCodec, stored: &[u8]) -> Option<u64> {
    if codec == BlockCodec::Raw || stored.len() < LEN_PREFIX {
        return None;
    }
    Some(u64::from_le_bytes(stored[..LEN_PREFIX].try_into().unwrap()))
}

#[cfg(not(target_arch = "wasm32"))]
fn zstd_encode_block(raw: &[u8]) -> Result<Vec<u8>> {
    zstd::bulk::compress(raw, ZSTD_BLOCK_LEVEL)
        .map_err(|e| AssetError::Pack(format!("zstd block encode: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn zstd_encode_block(_raw: &[u8]) -> Result<Vec<u8>> {
    Err(AssetError::Pack(
        "zstd block encoding is not supported on wasm — containers are cooked on desktop".into(),
    ))
}

/// zstd block decode: the C `zstd` natively, the pure-Rust `ruzstd` in a browser.
///
/// The two are different implementations of one format, which is exactly why
/// `Zstd` is the codec with a portability caveat where `Lz4`/`Deflate` have none
/// — see the bake-off table in the IASSET1 memo.
fn zstd_decode_block(frame: &[u8], raw_len: usize) -> Result<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        zstd::bulk::decompress(frame, raw_len)
            .map_err(|e| AssetError::Pack(format!("zstd block: {e}")))
    }
    #[cfg(target_arch = "wasm32")]
    {
        use std::io::Read;
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(frame)
            .map_err(|e| AssetError::Pack(format!("ruzstd init: {e}")))?;
        let mut out = Vec::with_capacity(raw_len);
        decoder
            .by_ref()
            .take(raw_len as u64 + 1)
            .read_to_end(&mut out)
            .map_err(|e| AssetError::Pack(format!("ruzstd: {e}")))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compressible block: repetitive enough that every codec wins on it, but
    /// not so uniform that the test proves nothing.
    fn corpus(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| ((i / 7) as u8).wrapping_add((i % 13) as u8 * 3))
            .collect()
    }

    #[test]
    fn wire_codes_are_frozen() {
        // A directory's codec byte is a wire value. If this table changes, every
        // container ever cooked reads its blocks under the wrong codec.
        assert_eq!(BlockCodec::Raw.code(), 0);
        assert_eq!(BlockCodec::Lz4.code(), 1);
        assert_eq!(BlockCodec::Deflate.code(), 2);
        assert_eq!(BlockCodec::Zstd.code(), 3);
        for c in BlockCodec::ALL {
            assert_eq!(BlockCodec::from_code(c.code()), Some(c));
        }
        assert_eq!(BlockCodec::from_code(4), None);
    }

    #[test]
    fn every_codec_round_trips_exactly() {
        let raw = corpus(64 * 1024);
        for codec in BlockCodec::ALL {
            let (used, stored) = encode_block(codec, &raw).unwrap();
            assert_eq!(used, codec, "{codec:?} should have won on this corpus");
            let back = decode_block(used, &stored, raw.len()).unwrap();
            assert_eq!(back.as_ref(), raw.as_slice(), "{codec:?}");
        }
    }

    #[test]
    fn raw_blocks_are_borrowed_and_byte_identical() {
        let raw = corpus(4096);
        let (used, stored) = encode_block(BlockCodec::Raw, &raw).unwrap();
        assert_eq!(used, BlockCodec::Raw);
        assert_eq!(
            stored.as_ref(),
            raw,
            "a raw block is literally the input bytes"
        );
        // **Both directions** (IASSET1 audit). The decode half was always
        // asserted; the encode half was not, and a `Vec` return made every
        // all-raw container build a second full copy of its own payload.
        assert!(
            matches!(stored, Cow::Borrowed(_)),
            "a raw ENCODE must not allocate either"
        );
        let back = decode_block(used, &stored, raw.len()).unwrap();
        assert!(matches!(back, Cow::Borrowed(_)), "raw must not allocate");
    }

    /// **The invariant is on the codec that was USED, not on the one asked for**:
    /// whenever `encode_block` reports [`Raw`](BlockCodec::Raw) it must be
    /// borrowing, whichever of the three fallback doors it came through.
    ///
    /// Written as a conditional rather than as "this corpus is incompressible for
    /// everyone", because that assumption is false and the first draft of this
    /// test proved it: `lz4_flex` finds something in a multiplicative-hash byte
    /// stream that DEFLATE and zstd also find, and the assertion that fired said
    /// `left: Lz4, right: Raw`. A test that has to be right about a codec's
    /// appetite is testing the codec; this one tests the return type.
    #[test]
    fn every_raw_result_is_a_borrow_whichever_door_it_came_through() {
        let noise: Vec<u8> = (0..8192u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        let tiny = corpus(MIN_COMPRESSIBLE - 1);
        let mut raws = 0usize;
        let mut owned = 0usize;
        for codec in BlockCodec::ALL {
            for input in [noise.as_slice(), tiny.as_slice()] {
                let (used, stored) = encode_block(codec, input).unwrap();
                match used {
                    BlockCodec::Raw => {
                        raws += 1;
                        assert!(
                            matches!(stored, Cow::Borrowed(_)),
                            "{codec:?} reported Raw and still allocated"
                        );
                        assert_eq!(stored.as_ref(), input);
                    }
                    _ => {
                        owned += 1;
                        assert!(matches!(stored, Cow::Owned(_)), "{codec:?}");
                    }
                }
            }
        }
        // Non-vacuity from both sides: the sub-threshold input forces a Raw for
        // every codec, and the noise compresses for at least one of them.
        assert!(raws >= BlockCodec::ALL.len(), "no Raw result was exercised");
        assert!(owned > 0, "no compressed result was exercised");
    }

    #[test]
    fn a_block_never_inflates() {
        // Incompressible (a counter through a permutation-ish mix) and small.
        let noise: Vec<u8> = (0..8192u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for codec in BlockCodec::ALL {
            let (used, stored) = encode_block(codec, &noise).unwrap();
            assert!(
                stored.len() <= noise.len(),
                "{codec:?} inflated {} -> {}",
                noise.len(),
                stored.len()
            );
            let back = decode_block(used, &stored, noise.len()).unwrap();
            assert_eq!(back.as_ref(), noise.as_slice());
        }
    }

    #[test]
    fn a_tiny_block_stays_raw() {
        let tiny = corpus(MIN_COMPRESSIBLE - 1);
        for codec in BlockCodec::ALL {
            let (used, stored) = encode_block(codec, &tiny).unwrap();
            assert_eq!(used, BlockCodec::Raw, "{codec:?} on a sub-threshold block");
            assert_eq!(stored.as_ref(), tiny);
        }
    }

    #[test]
    fn a_lying_length_prefix_is_refused_before_it_allocates() {
        let raw = corpus(32 * 1024);
        for codec in [BlockCodec::Lz4, BlockCodec::Deflate, BlockCodec::Zstd] {
            let (used, stored) = encode_block(codec, &raw).unwrap();
            assert_eq!(used, codec);
            // Claim 4 GiB out of a few kilobytes.
            let mut stored = stored.into_owned();
            stored[..LEN_PREFIX].copy_from_slice(&(4u64 << 30).to_le_bytes());
            let err = decode_block(used, &stored, raw.len()).unwrap_err();
            assert!(
                err.to_string().contains("block ceiling"),
                "{codec:?}: {err}"
            );
        }
    }

    #[test]
    fn a_prefix_under_the_ceiling_but_wrong_still_fails() {
        // The strict half: the codec must fill EXACTLY the declared length. A
        // short claim that is still under the ceiling has to be caught by the
        // decode, not by the bound.
        let raw = corpus(32 * 1024);
        for codec in [BlockCodec::Lz4, BlockCodec::Deflate, BlockCodec::Zstd] {
            let (used, stored) = encode_block(codec, &raw).unwrap();
            let mut stored = stored.into_owned();
            stored[..LEN_PREFIX].copy_from_slice(&((raw.len() / 2) as u64).to_le_bytes());
            assert!(
                decode_block(used, &stored, raw.len()).is_err(),
                "{codec:?} accepted a half-length claim"
            );
        }
    }

    #[test]
    fn a_truncated_block_is_refused() {
        for codec in [BlockCodec::Lz4, BlockCodec::Deflate, BlockCodec::Zstd] {
            let short = [0u8; LEN_PREFIX - 1];
            assert!(decode_block(codec, &short, 1 << 20).is_err(), "{codec:?}");
        }
    }

    /// **The LZ4 decoder this workspace compiles is the SAFE one** — pinned in
    /// the manifest, because nothing about the bytes can tell you which.
    ///
    /// `lz4_flex`'s `safe-decode` feature selects `block/decompress_safe.rs`
    /// (`forbid(unsafe_code)`) over `block/decompress.rs`, which writes its
    /// output through a raw pointer. Both decode a valid frame identically, so a
    /// round-trip test cannot see the difference and no assertion over
    /// [`decode_block`]'s behaviour ever will. What decides it is one line of
    /// `Cargo.toml`, and `default-features = false` — reached for to drop the
    /// `frame` format and its checksum crate — turns it off as a side effect.
    ///
    /// This reader parses a container the user downloaded, in the shipped
    /// player, and the `raw_len` ceiling bounds the *claim* and not the frame.
    /// The safe decoder is the reason a pure-Rust codec was preferred at all, so
    /// the pin is asserted where the decoder is, not left to a reviewer's memory
    /// of what a feature list used to say.
    #[test]
    fn the_lz4_decoder_is_pinned_to_the_safe_implementation() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("inf-asset lives two levels under the workspace root");
        let manifest =
            std::fs::read_to_string(root.join("Cargo.toml")).expect("the workspace manifest reads");
        // The pin spans lines, so normalize whitespace before looking for it.
        let flat: String = manifest.split_whitespace().collect::<Vec<_>>().join(" ");
        let start = flat
            .find("lz4_flex = {")
            .expect("lz4_flex is pinned in [workspace.dependencies]");
        let decl = &flat[start..start + flat[start..].find('}').expect("the pin closes") + 1];
        for feature in ["\"safe-decode\"", "\"safe-encode\""] {
            assert!(
                decl.contains(feature),
                "the workspace pin dropped {feature}: `{decl}`. Without `safe-decode` \
                 lz4_flex decodes an attacker-controlled frame through a raw pointer, \
                 in the shipped player, behind a ceiling that bounds the length CLAIM \
                 and not the frame."
            );
        }
    }

    #[test]
    fn declared_length_reads_without_decompressing() {
        let raw = corpus(20_000);
        let (used, stored) = encode_block(BlockCodec::Deflate, &raw).unwrap();
        assert_eq!(declared_raw_len(used, &stored), Some(20_000));
        assert_eq!(declared_raw_len(BlockCodec::Raw, &raw), None);
    }
}
