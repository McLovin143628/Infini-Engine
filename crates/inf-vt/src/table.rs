//! The **indirection table**: the wire layout the GPU mirror uploads and the
//! fragment shader reads.
//!
//! # Why a storage buffer and not an indirection *texture*
//!
//! The obvious representation is one small texture per virtual texture, sampled
//! with `textureLoad` at the tile coordinate. It is rejected here for a reason
//! that is structural rather than aesthetic: **bind-group pressure**. The
//! renderer runs on `Limits::default()`, which is 4 bind groups, and the shared
//! environment group is already occupied through index 13. A per-texture
//! indirection texture means a bind group that changes per draw, or a texture
//! array whose length is a pipeline constant; a single storage buffer holding
//! *every* texture's table, indexed by a per-instance texture id, means **one**
//! binding for any number of virtual textures.
//!
//! The whole VT surface P26.3 has to fold into the environment group is therefore
//! three bindings:
//!
//! | binding | resource | why |
//! |---|---|---|
//! | atlas | `texture_2d<f32>` | the one physical page atlas |
//! | sampler | `sampler` | linear, clamp — the border ring is what makes clamping correct |
//! | table | `var<storage, read> array<u32>` | this layout, every texture at once |
//!
//! # The layout
//!
//! Everything is `u32`, little-endian, and every offset in it is a **word**
//! offset from the start of the buffer, so a shader indexes `table[n]` with no
//! byte arithmetic at all.
//!
//! ```text
//! ┌ pool header ([`TABLE_HEADER_WORDS`] = 4) ─────────────────────────┐
//! │ [0] magic = [`TABLE_MAGIC`]                                       │
//! │ [1] texture_count                                                 │
//! │ [2] slots_x            (atlas slots per row)                      │
//! │ [3] stored_tile_size   (texels per side of a slot)                │
//! ├ texture directory (texture_count words) ──────────────────────────┤
//! │ [4 + t] = word offset of texture t's block                        │
//! ├ texture block, at `b` ([`TABLE_TEXTURE_HEADER_WORDS`] = 4) ───────┤
//! │ [b+0] mip_count                                                   │
//! │ [b+1] tile_size        (PAYLOAD texels per tile side)             │
//! │ [b+2] border                                                      │
//! │ [b+3] flags            bit 0 = srgb                               │
//! ├ mip records (mip_count × [`TABLE_MIP_REC_WORDS`] = 4) ────────────┤
//! │ width · height · tiles_x · first_entry (ABSOLUTE word offset)     │
//! ├ entries (one per tile of the pyramid, in (mip, y, x) order) ──────┤
//! │ [`pack_entry`]: slot in bits 0..16, resolved mip in bits 16..24   │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! An entry's index inside its texture is
//! [`VtTextureDesc::entry_index`](crate::VtTextureDesc::entry_index) — the same
//! number as the tile's index in the `.inf_tex` v2 **tile directory**. One index
//! addresses both the description of a tile and the bytes of a tile.
//!
//! # What P26.3's `vt_sample` does with it
//!
//! The entry an address resolves to is never "the tile you asked for": it is the
//! **finest resident ancestor** of the tile you asked for, which is why the entry
//! carries a mip as well as a slot. The shader therefore re-derives its texel
//! position at the mip it was given, rather than at the mip it wanted:
//!
//! ```wgsl
//! // uv: virtual UV in [0,1]²; want_mip: from the gradient
//! let b        = vt_table[VT_DIR + tex_id];          // this texture's block
//! let tile_sz  = f32(vt_table[b + 1u]);
//! let border   = f32(vt_table[b + 2u]);
//! let m        = min(want_mip, vt_table[b + 0u] - 1u);
//! let rec      = b + 4u + m * 4u;                    // the wanted mip's record
//! let w        = f32(vt_table[rec + 0u]);
//! let h        = f32(vt_table[rec + 1u]);
//! let tiles_x  =      vt_table[rec + 2u];
//! let tx       = min(u32(uv.x * w / tile_sz), tiles_x - 1u);
//! let ty       = u32(uv.y * h / tile_sz);
//! let entry    = vt_table[vt_table[rec + 3u] + ty * tiles_x + tx];
//!
//! let slot     = entry & 0xFFFFu;                    // physical page
//! let got      = (entry >> 16u) & 0xFFu;             // the mip we ACTUALLY got
//!
//! // Re-derive the texel at the resolved mip. This is the CPU
//! // `VtTextureDesc::ancestor` walk, done directly from uv — the two agree, and
//! // that agreement is the sampling contract (`address.rs`'s tests pin it).
//! let grec     = b + 4u + got * 4u;
//! let gp       = uv * vec2(f32(vt_table[grec]), f32(vt_table[grec + 1u]));
//! let local    = gp - floor(gp / tile_sz) * tile_sz;  // in [0, tile_size)
//!
//! let origin   = vec2(f32((slot % slots_x) * stored), f32((slot / slots_x) * stored));
//! let atlas_uv = (origin + vec2(border) + local) / vec2(atlas_size);
//! return textureSampleLevel(vt_atlas, vt_sampler, atlas_uv, 0.0);
//! ```
//!
//! The `+ border` is the whole reason the container bakes a border ring: a texel
//! at the payload edge filters against real neighbours from the same mip, not
//! against whatever page happens to sit beside it in the atlas.

use crate::address::{TileCoord, VtTextureDesc};

/// `b"VTB1"` little-endian — the first word of the table buffer, so a shader
/// reading a buffer that was never written reads something recognisable rather
/// than a plausible zero.
pub const TABLE_MAGIC: u32 = u32::from_le_bytes(*b"VTB1");

/// Words of the pool header, before the texture directory.
pub const TABLE_HEADER_WORDS: usize = 4;

/// Words of a texture block's own header, before its mip records.
pub const TABLE_TEXTURE_HEADER_WORDS: usize = 4;

/// Words of one mip record.
pub const TABLE_MIP_REC_WORDS: usize = 4;

/// The largest slot index a packed entry can carry (16 bits).
///
/// Far above the 3 600 slots one 8 192² atlas holds, so it is headroom for the
/// multi-atlas follow-up rather than a live constraint.
pub const MAX_SLOT_INDEX: u32 = 0xFFFF;

/// One decoded indirection entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtEntry {
    /// The physical page holding the resolved tile.
    pub slot: u32,
    /// The mip the resolved tile is at — **the finest resident ancestor's**, not
    /// necessarily the one that was asked for.
    pub mip: u32,
}

/// Pack a resolved `(slot, mip)` into one table word.
#[inline]
pub const fn pack_entry(slot: u32, mip: u32) -> u32 {
    (slot & 0xFFFF) | ((mip & 0xFF) << 16)
}

/// The inverse of [`pack_entry`].
#[inline]
pub const fn unpack_entry(word: u32) -> VtEntry {
    VtEntry {
        slot: word & 0xFFFF,
        mip: (word >> 16) & 0xFF,
    }
}

/// Where one texture's block lives in the table image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSpan {
    /// Word offset of the block header.
    pub base: usize,
    /// Word offset of entry 0.
    pub entries: usize,
    /// Words in the whole block (header + mip records + entries).
    pub len: usize,
}

/// The materialized table for a whole pool.
///
/// Kept live rather than rebuilt per frame: an admit or an evict patches the
/// entries it changes, and the mirror writes back exactly the blocks the
/// transaction names.
#[derive(Debug, Clone, Default)]
pub(crate) struct TableImage {
    pub words: Vec<u32>,
    pub blocks: Vec<BlockSpan>,
}

impl TableImage {
    /// Lay the image out for `descs`, leaving every entry zero. Callers fill the
    /// entries with a coarse-to-fine sweep afterwards.
    ///
    /// The layout is rebuilt whole on registration — never incrementally — because
    /// the texture directory sits *before* the blocks, so a new texture shifts
    /// every block that follows it. Registration happens at load, not per frame,
    /// and a `layout_rebuilt` transaction tells the mirror to re-create and
    /// re-upload the buffer rather than patch it.
    pub fn layout(descs: &[VtTextureDesc], slots_x: u32, stored_tile_size: u32) -> Self {
        let n = descs.len();
        let mut words = vec![0u32; TABLE_HEADER_WORDS + n];
        words[0] = TABLE_MAGIC;
        words[1] = n as u32;
        words[2] = slots_x;
        words[3] = stored_tile_size;

        let mut blocks = Vec::with_capacity(n);
        for (t, desc) in descs.iter().enumerate() {
            let base = words.len();
            words[TABLE_HEADER_WORDS + t] = base as u32;
            words.push(desc.mip_count());
            words.push(desc.tile_size);
            words.push(desc.border);
            words.push(u32::from(desc.srgb));
            let entries = base + TABLE_TEXTURE_HEADER_WORDS + TABLE_MIP_REC_WORDS * desc.mips.len();
            for (m, mip) in desc.mips.iter().enumerate() {
                words.push(mip.width);
                words.push(mip.height);
                words.push(mip.tiles_x);
                words.push((entries + desc.mip_first_entry(m as u32).unwrap_or(0) as usize) as u32);
            }
            debug_assert_eq!(words.len(), entries);
            let tiles = desc.tile_count() as usize;
            words.resize(words.len() + tiles, 0);
            blocks.push(BlockSpan {
                base,
                entries,
                len: words.len() - base,
            });
        }
        Self { words, blocks }
    }

    /// The word index of `at`'s entry in texture `t`.
    #[inline]
    pub fn entry_word(&self, t: usize, desc: &VtTextureDesc, at: TileCoord) -> Option<usize> {
        Some(self.blocks.get(t)?.entries + desc.entry_index(at)? as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::full_pyramid;

    #[test]
    fn an_entry_round_trips_and_leaves_its_top_byte_alone() {
        for (slot, mip) in [(0u32, 0u32), (1, 1), (2_713, 8), (MAX_SLOT_INDEX, 31)] {
            let w = pack_entry(slot, mip);
            assert_eq!(unpack_entry(w), VtEntry { slot, mip });
            assert_eq!(w >> 24, 0, "bits 24..32 are reserved and must stay zero");
        }
    }

    #[test]
    fn the_layout_addresses_every_tile_exactly_once() {
        let descs = vec![
            full_pyramid(320, 192, 128, 4, true),
            full_pyramid(129, 129, 128, 4, false),
        ];
        let img = TableImage::layout(&descs, 46, 136);
        assert_eq!(img.words[0], TABLE_MAGIC);
        assert_eq!(img.words[1], 2);
        assert_eq!(img.words[2], 46);
        assert_eq!(img.words[3], 136);

        let mut seen = std::collections::BTreeSet::new();
        for (t, desc) in descs.iter().enumerate() {
            let span = img.blocks[t];
            assert_eq!(img.words[TABLE_HEADER_WORDS + t], span.base as u32);
            assert_eq!(img.words[span.base], desc.mip_count());
            for (m, mip) in desc.mips.iter().enumerate() {
                let rec = span.base + TABLE_TEXTURE_HEADER_WORDS + TABLE_MIP_REC_WORDS * m;
                assert_eq!(img.words[rec], mip.width);
                assert_eq!(img.words[rec + 1], mip.height);
                assert_eq!(img.words[rec + 2], mip.tiles_x);
                // …and the shader's index arithmetic lands on the same word the
                // CPU's `entry_index` does, for every tile of every level.
                for y in 0..mip.tiles_y {
                    for x in 0..mip.tiles_x {
                        let at = TileCoord::new(m as u32, x, y);
                        let shader = img.words[rec + 3] as usize + (y * mip.tiles_x + x) as usize;
                        assert_eq!(Some(shader), img.entry_word(t, desc, at), "{at:?}");
                        assert!(seen.insert(shader), "word {shader} describes two tiles");
                        assert!(shader < span.base + span.len);
                    }
                }
            }
        }
        assert_eq!(
            seen.len(),
            (descs[0].tile_count() + descs[1].tile_count()) as usize
        );
    }
}
