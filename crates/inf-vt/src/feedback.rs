//! The **feedback bitmask's layout** (P26.4): one bit per virtual tile of every
//! registered texture, addressed in exactly the order the residency addresses
//! them.
//!
//! # Why a bitmask, and why the layout is fixed
//!
//! The ruling that reconciles a GPU feedback signal with this repository's
//! replay doctrine lives at `inf_vgeom::stream`'s *"why the wants are CPU-side
//! and not cull feedback"*: a readback is frame-latent, so a want set derived
//! from *"whatever arrived"* depends on the frame **history**, and two runs
//! differing by one dropped frame diverge. P26.4 keeps the doctrine and pays its
//! two prices explicitly:
//!
//! 1. **The content is order-independent.** Producers `atomicOr` a bit; OR is
//!    commutative and idempotent, so the mask a frame produces is a pure
//!    function of `(camera, scene, residency)` and not of how many threads ran,
//!    in what order, or how many of them wrote the same bit. There is no counter
//!    to race and no compaction whose output depends on arrival order.
//! 2. **The latency is pinned, not observed.** The consumer reads frame `F − N`
//!    for a fixed `N` (`inf_render::readback`'s ring, `N = 2`) and takes the
//!    floor when that slot is not ready. So a trace is a function of the
//!    *committed* input at a *known* frame rather than of scheduling.
//!
//! And the third price, which is the one that makes a dropped frame harmless:
//! feedback only ever **refines**. Every want it produces is
//! [`VT_PRIORITY_FEEDBACK`](crate::VT_PRIORITY_FEEDBACK), so it is served after
//! the whole analytic floor and can never take a slot from it.
//!
//! # The layout
//!
//! Bit `n` of the mask and indirection entry `n` of the table describe **the
//! same tile**. Textures are laid out in handle order; within a texture, tiles
//! are in [`VtTextureDesc::entry_index`](crate::VtTextureDesc::entry_index)
//! order, which is the `.inf_tex` tile directory's own `(mip, y, x)` payload
//! order. That is not a convenience: [`VtFeedbackLayout::wants`] scans the mask
//! straight through and emits wants **in virtual-address order**, so a batch of
//! admits is a forward scan of the container's bytes rather than a scatter — the
//! same property `VtResidency::apply_wants`'s payload sort buys, arriving
//! already sorted.
//!
//! A texture's base bit is therefore a running sum of tile counts, and it is
//! handed to the producer (the GPU) per request rather than recomputed there:
//! one arithmetic rule, on the CPU, in this file.

use crate::address::TileCoord;
use crate::residency::{VtPriority, VtResidency, VtTextureHandle, VtWant, VT_PRIORITY_FEEDBACK};

/// Bits in one mask word. The mask is `u32` because `atomicOr` on a
/// `storage` buffer in WGSL is `u32`, and because a word is the unit a readback
/// copy is aligned to anyway.
pub const FEEDBACK_BITS_PER_WORD: u32 = 32;

/// Where each registered texture's tiles sit in the coverage bitmask.
///
/// Built from a residency, so it cannot disagree with the address space it
/// describes; rebuilt whenever a texture is registered (the same event that
/// relays the indirection table).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VtFeedbackLayout {
    /// First bit index of texture `t`. `bases.len()` is the texture count.
    bases: Vec<u32>,
    /// Tiles (= bits) of texture `t`, parallel to `bases`.
    tiles: Vec<u32>,
    bits: u32,
}

impl VtFeedbackLayout {
    /// Lay the mask out for everything `res` has registered.
    pub fn for_residency(res: &VtResidency) -> Self {
        let mut bases = Vec::with_capacity(res.texture_count());
        let mut tiles = Vec::with_capacity(res.texture_count());
        let mut bits = 0u32;
        for t in 0..res.texture_count() {
            let n = res
                .desc(VtTextureHandle(t as u32))
                .map_or(0, |d| d.tile_count());
            bases.push(bits);
            tiles.push(n);
            bits += n;
        }
        Self { bases, tiles, bits }
    }

    /// Total bits — one per virtual tile of every registered texture.
    #[inline]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Words the mask buffer needs. Never zero: a `wgpu` buffer may not be, and
    /// a pool with no textures still has to bind something legal.
    #[inline]
    pub fn words(&self) -> usize {
        (self.bits.div_ceil(FEEDBACK_BITS_PER_WORD) as usize).max(1)
    }

    /// How many textures the layout covers.
    #[inline]
    pub fn texture_count(&self) -> usize {
        self.bases.len()
    }

    /// The first bit of texture `tex` — what a producer is handed so the bit
    /// arithmetic on the GPU is `base + entry_index` and nothing more.
    #[inline]
    pub fn texture_base(&self, tex: VtTextureHandle) -> Option<u32> {
        self.bases.get(tex.index()).copied()
    }

    /// Bits texture `tex` owns (its tile count).
    #[inline]
    pub fn texture_bits(&self, tex: VtTextureHandle) -> Option<u32> {
        self.tiles.get(tex.index()).copied()
    }

    /// The bit describing `at` of `tex`, if both exist.
    pub fn bit_of(&self, res: &VtResidency, tex: VtTextureHandle, at: TileCoord) -> Option<u32> {
        let base = self.texture_base(tex)?;
        Some(base + res.desc(tex)?.entry_index(at)?)
    }

    /// **The CPU request scan** — decode a raw mask into wants, in
    /// virtual-address order, at [`VT_PRIORITY_FEEDBACK`].
    ///
    /// Straight through the words, low bit first, which *is* payload order by
    /// construction (see the module docs), so the wants come out already sorted
    /// the way `apply_wants` wants them and a batch of admits is a forward scan
    /// of the container.
    ///
    /// A mask longer than the layout is truncated and a shorter one is read to
    /// its end: a mask read out of a ring can predate a registration, and the
    /// honest answer to *"this mask describes a residency that no longer
    /// exists"* is fewer wants, not a panic. Bits past the last texture's tiles
    /// resolve to no address and are skipped — which is also what keeps a mask
    /// whose tail words were never cleared from inventing wants.
    pub fn wants(&self, res: &VtResidency, mask: &[u32]) -> Vec<VtWant> {
        self.wants_at(res, mask, VT_PRIORITY_FEEDBACK)
    }

    /// [`wants`](Self::wants) at an explicit rank — the seam a debug view or a
    /// future predictor (P28.4 speculates at a *lower* rank) needs.
    pub fn wants_at(&self, res: &VtResidency, mask: &[u32], priority: VtPriority) -> Vec<VtWant> {
        let mut out = Vec::new();
        for (t, (&base, &count)) in self.bases.iter().zip(self.tiles.iter()).enumerate() {
            let handle = VtTextureHandle(t as u32);
            let Some(desc) = res.desc(handle) else {
                continue;
            };
            let mut entry = 0u32;
            for mip in 0..desc.mip_count() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        let bit = base + entry;
                        entry += 1;
                        if entry > count {
                            break;
                        }
                        let word = (bit / FEEDBACK_BITS_PER_WORD) as usize;
                        let Some(w) = mask.get(word) else {
                            return out;
                        };
                        if w & (1u32 << (bit % FEEDBACK_BITS_PER_WORD)) != 0 {
                            out.push(
                                VtWant::new(handle, TileCoord::new(mip, x, y))
                                    .with_priority(priority),
                            );
                        }
                    }
                }
            }
        }
        out
    }

    /// Set the bit for `at` in `mask` — the **CPU twin of the shader's
    /// `atomicOr`**, so a test can build the mask the GPU would build and a
    /// debug view can mark by hand.
    ///
    /// Returns `false` when the address does not exist or the mask is too short,
    /// so a caller that expected a mark can say so rather than discovering a
    /// silent no-op (the anti-vacuity shape).
    pub fn mark(
        &self,
        res: &VtResidency,
        mask: &mut [u32],
        tex: VtTextureHandle,
        at: TileCoord,
    ) -> bool {
        let Some(bit) = self.bit_of(res, tex, at) else {
            return false;
        };
        let word = (bit / FEEDBACK_BITS_PER_WORD) as usize;
        match mask.get_mut(word) {
            Some(w) => {
                *w |= 1u32 << (bit % FEEDBACK_BITS_PER_WORD);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::full_pyramid;
    use crate::pool::{PageFormat, VtPoolConfig};

    fn residency(pages: u64) -> VtResidency {
        let (r, _) = VtResidency::new(VtPoolConfig {
            format: PageFormat::Bc1,
            stored_tile_size: 136,
            budget_bytes: PageFormat::Bc1.page_bytes(136) * pages,
            max_texture_dim: 8192,
        });
        r
    }

    /// **Bit `n` and indirection entry `n` describe one tile**, for every tile of
    /// every texture — the property the whole layout exists to have, and the one
    /// a hand-rolled base in the shader would break first.
    #[test]
    fn a_bit_and_a_table_entry_address_the_same_tile() {
        let mut r = residency(64);
        let a = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        let b = r
            .register_texture(full_pyramid(511, 3, 128, 4, false))
            .unwrap();
        let layout = VtFeedbackLayout::for_residency(&r);

        let tiles_a = r.desc(a).unwrap().tile_count();
        let tiles_b = r.desc(b).unwrap().tile_count();
        assert_eq!(layout.texture_base(a), Some(0));
        assert_eq!(layout.texture_base(b), Some(tiles_a));
        assert_eq!(layout.bits(), tiles_a + tiles_b);
        assert_eq!(layout.texture_base(VtTextureHandle(9)), None);

        // Every address maps to a distinct bit, and the bit is the texture's base
        // plus the SAME index the indirection table uses.
        let mut seen = std::collections::BTreeSet::new();
        for h in [a, b] {
            let desc = r.desc(h).unwrap().clone();
            for mip in 0..desc.mip_count() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        let at = TileCoord::new(mip, x, y);
                        let bit = layout.bit_of(&r, h, at).expect("in grid");
                        assert_eq!(
                            bit,
                            layout.texture_base(h).unwrap() + desc.entry_index(at).unwrap()
                        );
                        assert!(seen.insert(bit), "bit {bit} describes two tiles");
                    }
                }
            }
        }
        assert_eq!(seen.len() as u32, layout.bits());
    }

    /// The scan comes out **in virtual-address order** — coarse tiles of texture
    /// 0 before fine ones, texture 0 before texture 1 — which is what makes a
    /// batch of admits a forward scan of the container's bytes.
    #[test]
    fn the_scan_emits_wants_in_payload_order() {
        let mut r = residency(64);
        let a = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        let b = r
            .register_texture(full_pyramid(256, 256, 128, 4, true))
            .unwrap();
        let layout = VtFeedbackLayout::for_residency(&r);
        let mut mask = vec![0u32; layout.words()];

        // Marked in a deliberately scrambled order.
        let marks = [
            (b, TileCoord::new(0, 1, 1)),
            (a, TileCoord::new(1, 0, 0)),
            (b, TileCoord::new(1, 0, 0)),
            (a, TileCoord::new(0, 2, 1)),
            (a, TileCoord::new(0, 0, 0)),
        ];
        for (h, at) in marks {
            assert!(layout.mark(&r, &mut mask, h, at), "{at:?} did not mark");
        }
        let wants = layout.wants(&r, &mask);
        assert_eq!(wants.len(), marks.len());
        assert!(
            wants.iter().all(|w| w.priority == VT_PRIORITY_FEEDBACK),
            "the scan produced a want that is not a refinement"
        );
        let keys: Vec<_> = wants
            .iter()
            .map(|w| (w.texture.0, w.tile.payload_order()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "the scan is not in virtual-address order");
        assert_eq!(wants[0].texture, a);
        assert_eq!(wants.last().unwrap().texture, b);
    }

    /// **`atomicOr` is idempotent, and the decoded want set proves it**: marking
    /// the same tile twice — the ordinary case, since every fragment of a surface
    /// marks the same tiles — is one want, and marking in either order is the
    /// same mask.
    #[test]
    fn the_mask_is_order_independent_and_idempotent() {
        let mut r = residency(64);
        let h = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        let layout = VtFeedbackLayout::for_residency(&r);
        let tiles = [
            TileCoord::new(0, 0, 0),
            TileCoord::new(0, 1, 0),
            TileCoord::new(1, 0, 0),
        ];

        let build = |order: &[usize], repeats: usize| {
            let mut mask = vec![0u32; layout.words()];
            for _ in 0..repeats {
                for &i in order {
                    layout.mark(&r, &mut mask, h, tiles[i]);
                }
            }
            mask
        };
        let forward = build(&[0, 1, 2], 1);
        let reversed = build(&[2, 1, 0], 1);
        let repeated = build(&[1, 0, 2, 1, 0], 7);
        assert_eq!(forward, reversed);
        assert_eq!(forward, repeated);
        assert_eq!(layout.wants(&r, &forward).len(), 3);
        // Anti-vacuity: an empty mask is not what is being compared.
        assert!(forward.iter().any(|w| *w != 0));
    }

    /// A mask from **before** a registration decodes into the wants it can still
    /// describe and never invents one — the ring's stale-slot case, which is a
    /// normal event at a level switch rather than a corruption.
    #[test]
    fn a_short_or_stale_mask_is_truncated_rather_than_invented() {
        let mut r = residency(64);
        let a = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        let old = VtFeedbackLayout::for_residency(&r);
        let mut mask = vec![0u32; old.words()];
        assert!(old.mark(&r, &mut mask, a, TileCoord::new(0, 0, 0)));

        // A second texture arrives; the layout grows and the held mask is short.
        r.register_texture(full_pyramid(2048, 2048, 128, 4, true))
            .unwrap();
        let now = VtFeedbackLayout::for_residency(&r);
        assert!(
            now.words() > mask.len(),
            "the fixture did not grow the mask"
        );
        let wants = now.wants(&r, &mask);
        assert!(
            wants.iter().all(|w| w.texture == a),
            "a stale mask named a texture it could not have seen"
        );
        assert!(!wants.is_empty(), "the truncation ate everything");
        // …and the empty mask decodes to nothing at all.
        assert!(now.wants(&r, &vec![0u32; now.words()]).is_empty());
    }

    /// A layout over no textures is legal and binds one word — a pool exists
    /// before anything registers into it.
    #[test]
    fn an_empty_layout_still_binds_a_legal_buffer() {
        let r = residency(4);
        let layout = VtFeedbackLayout::for_residency(&r);
        assert_eq!(layout.bits(), 0);
        assert_eq!(layout.words(), 1, "a wgpu buffer may not be zero-sized");
        assert!(layout.wants(&r, &[0]).is_empty());
    }
}
