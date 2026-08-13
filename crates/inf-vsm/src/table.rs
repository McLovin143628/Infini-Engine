//! The **indirection table**: the wire layout the GPU mirror uploads and the
//! P27.4 receiver reads.
//!
//! One storage buffer holding *every* light's table, for `inf_vt::table`'s
//! reason and it is if anything sharper here: the renderer runs on
//! `Limits::default()`'s **4 bind groups** and all four are spoken for, so the
//! whole VSM surface has to fold into the shared environment group. One buffer
//! plus one atlas is two bindings for any number of shadow-casting lights; a
//! table per light would be a bind group that changes per light.
//!
//! # The layout
//!
//! Everything is `u32`, little-endian, and every offset in it is a **word**
//! offset from the start of the buffer, so a shader indexes `table[n]` with no
//! byte arithmetic at all.
//!
//! ```text
//! ┌ atlas header ([`VSM_TABLE_HEADER_WORDS`] = 4) ─────────────────────┐
//! │ [0] magic = [`VSM_TABLE_MAGIC`]                                    │
//! │ [1] light_count                                                    │
//! │ [2] slots_x            (atlas slots per row)                       │
//! │ [3] stored_page_size   (texels per side of a slot)                 │
//! ├ light directory (light_count words) ───────────────────────────────┤
//! │ [4 + l] = word offset of light l's block                           │
//! ├ light block, at `b` ([`VSM_TABLE_LIGHT_HEADER_WORDS`] = 4) ────────┤
//! │ [b+0] level_count                                                  │
//! │ [b+1] page_size        (PAYLOAD texels per page side)              │
//! │ [b+2] border                                                       │
//! │ [b+3] kind | faces << 8   (`VsmTreeKind::wire`, and 1 or 6)        │
//! ├ level records (level_count × [`VSM_TABLE_LEVEL_REC_WORDS`] = 4) ───┤
//! │ pages_x · pages_y · first_entry (ABSOLUTE, face 0) · face_stride   │
//! ├ entries (one per page, in (face, level, y, x) order) ──────────────┤
//! │ [`pack_entry`], or [`VSM_ENTRY_NONE`]                              │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! A receiver reads `entry = table[rec.first_entry + face * rec.face_stride +
//! y * rec.pages_x + x]`, which is exactly
//! [`VsmLightDesc::entry_index`](crate::VsmLightDesc::entry_index) plus the
//! block's own entry base — one multiply-add, no sums.
//!
//! # `NONE` is a value, not a hole
//!
//! `inf-vt`'s every entry always resolves, because its coarsest level is pinned.
//! A shadow page has no pinned root (see [`crate::address`]), so an entry may
//! legitimately say **there is no shadow data for this page**, and a receiver
//! reads that as *lit*. [`VSM_ENTRY_NONE`] is `0xFFFF_FFFF` and a packed entry
//! leaves bits 24..32 zero, so the two can never be confused — asserted, because
//! the alternative is a light leak that looks like a bias problem.

use crate::address::{VsmLightDesc, VsmPage};

/// `b"VSMB"` little-endian — the first word of the table buffer, so a shader
/// reading a buffer that was never written reads something recognisable rather
/// than a plausible zero.
pub const VSM_TABLE_MAGIC: u32 = u32::from_le_bytes(*b"VSMB");

/// Words of the atlas header, before the light directory.
pub const VSM_TABLE_HEADER_WORDS: usize = 4;

/// Words of a light block's own header, before its level records.
pub const VSM_TABLE_LIGHT_HEADER_WORDS: usize = 4;

/// Words of one level record.
pub const VSM_TABLE_LEVEL_REC_WORDS: usize = 4;

/// The largest slot index a packed entry can carry (16 bits).
///
/// [`crate::plan_atlas`] floors the atlas against it and says so, which is where
/// the bound is enforced rather than assumed: [`pack_entry`] masks, so a slot
/// index past it would alias onto another page silently.
pub const MAX_SLOT_INDEX: u32 = 0xFFFF;

/// **No shadow data for this page.** See the module docs.
pub const VSM_ENTRY_NONE: u32 = 0xFFFF_FFFF;

/// The level field is 8 bits, and [`crate::MAX_VSM_LEVELS`] is what keeps it from
/// overflowing — checked here rather than reasoned about in a doc comment,
/// because the two constants live in different modules and the encoder masks.
const _: () = assert!(
    crate::address::MAX_VSM_LEVELS <= 0xFF,
    "a packed entry carries the resolved level in 8 bits"
);

/// One decoded indirection entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmEntry {
    /// The physical page holding the resolved page.
    pub slot: u32,
    /// The level the resolved page is at — **the finest resident ancestor's**,
    /// not necessarily the one that was asked for.
    pub level: u32,
}

/// Pack a resolved `(slot, level)` into one table word.
#[inline]
pub const fn pack_entry(slot: u32, level: u32) -> u32 {
    (slot & 0xFFFF) | ((level & 0xFF) << 16)
}

/// The inverse of [`pack_entry`] — `None` for [`VSM_ENTRY_NONE`].
#[inline]
pub const fn unpack_entry(word: u32) -> Option<VsmEntry> {
    if word == VSM_ENTRY_NONE {
        return None;
    }
    Some(VsmEntry {
        slot: word & 0xFFFF,
        level: (word >> 16) & 0xFF,
    })
}

/// Where one light's block lives in the table image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSpan {
    /// Word offset of the block header.
    pub base: usize,
    /// Word offset of entry 0.
    pub entries: usize,
    /// Words in the whole block (header + level records + entries).
    pub len: usize,
}

/// The materialized table for a whole atlas.
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
    /// Lay the image out for `descs`, leaving every entry [`VSM_ENTRY_NONE`].
    ///
    /// Rebuilt whole on registration — never incrementally — because the light
    /// directory sits *before* the blocks, so a new light shifts every block
    /// that follows it. Registration happens when a level's lights are resolved,
    /// not per frame, and a `layout_rebuilt` transaction tells the mirror to
    /// re-create and re-upload the buffer rather than patch it.
    pub fn layout(
        descs: &[VsmLightDesc],
        page_size: u32,
        border: u32,
        slots_x: u32,
        stored_page_size: u32,
    ) -> Self {
        let n = descs.len();
        let mut words = vec![0u32; VSM_TABLE_HEADER_WORDS + n];
        words[0] = VSM_TABLE_MAGIC;
        words[1] = n as u32;
        words[2] = slots_x;
        words[3] = stored_page_size;

        let mut blocks = Vec::with_capacity(n);
        for (l, desc) in descs.iter().enumerate() {
            let base = words.len();
            words[VSM_TABLE_HEADER_WORDS + l] = base as u32;
            words.push(desc.level_count());
            words.push(page_size);
            words.push(border);
            words.push(desc.kind.wire() | (desc.faces() << 8));
            let entries =
                base + VSM_TABLE_LIGHT_HEADER_WORDS + VSM_TABLE_LEVEL_REC_WORDS * desc.levels.len();
            let stride = desc.face_page_count();
            for (i, level) in desc.levels.iter().enumerate() {
                words.push(level.pages_x);
                words.push(level.pages_y);
                words.push(
                    (entries + desc.level_first_entry(0, i as u32).unwrap_or(0) as usize) as u32,
                );
                words.push(stride);
            }
            debug_assert_eq!(words.len(), entries);
            let pages = desc.page_count() as usize;
            words.resize(words.len() + pages, VSM_ENTRY_NONE);
            blocks.push(BlockSpan {
                base,
                entries,
                len: words.len() - base,
            });
        }
        Self { words, blocks }
    }

    /// The word index of `at`'s entry in light `l`.
    #[inline]
    pub fn entry_word(&self, l: usize, desc: &VsmLightDesc, at: VsmPage) -> Option<usize> {
        Some(self.blocks.get(l)?.entries + desc.entry_index(at)? as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::VsmTreeKind;

    #[test]
    fn an_entry_round_trips_and_can_never_be_mistaken_for_none() {
        for (slot, level) in [(0u32, 0u32), (1, 1), (1_023, 8), (MAX_SLOT_INDEX, 15)] {
            let w = pack_entry(slot, level);
            assert_eq!(unpack_entry(w), Some(VsmEntry { slot, level }));
            assert_eq!(w >> 24, 0, "bits 24..32 are reserved and must stay zero");
            assert_ne!(
                w, VSM_ENTRY_NONE,
                "a real page packed to the NONE sentinel — every receiver would \
                 read it as unlit ground"
            );
        }
        assert_eq!(unpack_entry(VSM_ENTRY_NONE), None);
        // …and slot 0 at level 0 is a legal entry rather than an accidental
        // "nothing", which is the trap `inf-vt`'s always-resolved table does not
        // have to avoid.
        assert_eq!(pack_entry(0, 0), 0);
        assert_eq!(
            unpack_entry(0),
            Some(VsmEntry { slot: 0, level: 0 }),
            "zero is a page, not an absence"
        );
    }

    #[test]
    fn the_layout_addresses_every_page_exactly_once() {
        let descs = vec![VsmLightDesc::clipmap(4, 8), VsmLightDesc::cube(3)];
        let img = TableImage::layout(&descs, 128, 0, 32, 128);
        assert_eq!(img.words[0], VSM_TABLE_MAGIC);
        assert_eq!(img.words[1], 2);
        assert_eq!(img.words[2], 32);
        assert_eq!(img.words[3], 128);

        let mut seen = std::collections::BTreeSet::new();
        for (l, desc) in descs.iter().enumerate() {
            let span = img.blocks[l];
            assert_eq!(img.words[VSM_TABLE_HEADER_WORDS + l], span.base as u32);
            assert_eq!(img.words[span.base], desc.level_count());
            assert_eq!(img.words[span.base + 1], 128);
            assert_eq!(img.words[span.base + 2], 0);
            assert_eq!(
                img.words[span.base + 3],
                desc.kind.wire() | (desc.faces() << 8)
            );
            for (i, level) in desc.levels.iter().enumerate() {
                let rec = span.base + VSM_TABLE_LIGHT_HEADER_WORDS + VSM_TABLE_LEVEL_REC_WORDS * i;
                assert_eq!(img.words[rec], level.pages_x);
                assert_eq!(img.words[rec + 1], level.pages_y);
                assert_eq!(img.words[rec + 3], desc.face_page_count());
                // …and the RECEIVER's index arithmetic lands on the same word the
                // CPU's `entry_index` does, for every page of every face.
                for face in 0..desc.faces() {
                    for y in 0..level.pages_y {
                        for x in 0..level.pages_x {
                            let at = VsmPage::new(face, i as u32, x, y);
                            let shader = img.words[rec + 2] as usize
                                + (face * img.words[rec + 3]) as usize
                                + (y * level.pages_x + x) as usize;
                            assert_eq!(Some(shader), img.entry_word(l, desc, at), "{at:?}");
                            assert!(seen.insert(shader), "word {shader} describes two pages");
                            assert!(shader < span.base + span.len);
                            assert_eq!(
                                img.words[shader], VSM_ENTRY_NONE,
                                "a fresh layout must resolve to NOTHING, not to slot 0"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            seen.len(),
            (descs[0].page_count() + descs[1].page_count()) as usize
        );
    }

    /// The block header carries the kind AND the face count in one word, and the
    /// two do not collide — a cube's `2 | 6 << 8` must not read as some other
    /// kind.
    #[test]
    fn the_kind_word_carries_both_fields_without_collision() {
        for (kind, desc) in [
            (VsmTreeKind::Clipmap, VsmLightDesc::clipmap(2, 4)),
            (VsmTreeKind::Quadtree, VsmLightDesc::quadtree(2)),
            (VsmTreeKind::Cube, VsmLightDesc::cube(2)),
        ] {
            let img = TableImage::layout(std::slice::from_ref(&desc), 128, 0, 4, 128);
            let w = img.words[img.blocks[0].base + 3];
            assert_eq!(w & 0xFF, kind.wire());
            assert_eq!(w >> 8, kind.faces());
        }
    }
}
