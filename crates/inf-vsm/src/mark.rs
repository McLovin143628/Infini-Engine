//! The **needed-page bitmask's layout** (P27.1 clause 1): one bit per virtual
//! page of every registered light, addressed in exactly the order the residency
//! addresses them.
//!
//! # Why a bitmask, and why the layout is fixed
//!
//! This is `inf_vt::feedback`'s doctrine applied to shadows, and the argument
//! transfers whole because it was never about textures. `inf_vgeom::stream`'s
//! ruling (lines 52–62) rejects GPU feedback as a residency input because *"a
//! readback is one frame latent, so the want set would depend on the frame
//! HISTORY, and two runs differing by a single dropped frame would diverge"*.
//! Three properties answer it:
//!
//! 1. **The content is order-independent.** Producers `atomicOr` a bit; OR is
//!    commutative and idempotent, so the mask a frame produces is a pure
//!    function of `(camera, depth buffer, lights, residency)` and not of how
//!    many threads ran, in what order, or how many of them wrote the same bit.
//!    There is no counter to race and no compaction that depends on arrival
//!    order.
//! 2. **The latency is pinned, not observed.** The consumer reads frame `F − N`
//!    for a fixed `N` (`inf_render::readback`'s ring, `N = 2` — built in P26.4
//!    naming this consumer) and takes an empty want set when that slot is not
//!    ready.
//! 3. **A page nothing marked is a page nothing samples**, so a dropped mask
//!    costs *shadow detail on the pages that were already resident*, never a
//!    hole and never a wrong pixel somewhere else.
//!
//! # The one place the shadow signal is BETTER than the texture one
//!
//! `docs/memos/p26-4-feedback-mechanism.md` records that VT's coverage is
//! per-**surface** rather than per-fragment, because a fragment-stage mark would
//! need a writable storage buffer in the FRAGMENT stage and the renderer's four
//! bind groups are all spoken for. That constraint does not reach here: the
//! marking pass is a **compute** pass that *consumes* the depth attachment, not
//! a lit-pass fragment writer. So this signal is per-**pixel**, occlusion
//! included by construction (the depth buffer is what it reads), with no new
//! adapter capability and no second geometry pass.
//!
//! # The layout
//!
//! Bit `n` of the mask and indirection entry `n` of the table describe **the
//! same page**. Lights are laid out in handle order; within a light, pages are in
//! [`VsmLightDesc::entry_index`](crate::VsmLightDesc::entry_index) order — face,
//! then level, then `(y, x)`. So [`VsmMarkLayout::wants`] scans the mask straight
//! through and emits wants **in entry order**, already sorted the way
//! `VsmResidency::apply_wants` wants them.

use crate::address::VsmPage;
use crate::residency::{VsmLightHandle, VsmPriority, VsmResidency, VsmWant, VSM_PRIORITY_MARKED};

/// Bits in one mask word. The mask is `u32` because `atomicOr` on a `storage`
/// buffer in WGSL is `u32`, and because a word is the unit a readback copy is
/// aligned to anyway.
pub const VSM_MARK_BITS_PER_WORD: u32 = 32;

/// Where each registered light's pages sit in the needed-page bitmask.
///
/// Built from a residency, so it cannot disagree with the address space it
/// describes; rebuilt whenever a light is registered (the same event that relays
/// the indirection table).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VsmMarkLayout {
    /// First bit index of light `l`. `bases.len()` is the light count.
    bases: Vec<u32>,
    /// Pages (= bits) of light `l`, parallel to `bases`.
    pages: Vec<u32>,
    bits: u32,
}

impl VsmMarkLayout {
    /// Lay the mask out for everything `res` has registered.
    pub fn for_residency(res: &VsmResidency) -> Self {
        let mut bases = Vec::with_capacity(res.light_count());
        let mut pages = Vec::with_capacity(res.light_count());
        let mut bits = 0u32;
        for l in 0..res.light_count() {
            let n = res
                .desc(VsmLightHandle(l as u32))
                .map_or(0, |d| d.page_count());
            bases.push(bits);
            pages.push(n);
            bits += n;
        }
        Self { bases, pages, bits }
    }

    /// Total bits — one per virtual page of every registered light.
    #[inline]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Words the mask buffer needs. Never zero: a `wgpu` buffer may not be, and
    /// an atlas exists before any light registers into it.
    #[inline]
    pub fn words(&self) -> usize {
        (self.bits.div_ceil(VSM_MARK_BITS_PER_WORD) as usize).max(1)
    }

    /// How many lights the layout covers.
    #[inline]
    pub fn light_count(&self) -> usize {
        self.bases.len()
    }

    /// The first bit of light `light` — what the marking pass is handed so its
    /// bit arithmetic is `base + entry_index` and nothing more.
    #[inline]
    pub fn light_base(&self, light: VsmLightHandle) -> Option<u32> {
        self.bases.get(light.index()).copied()
    }

    /// Bits light `light` owns (its page count).
    #[inline]
    pub fn light_bits(&self, light: VsmLightHandle) -> Option<u32> {
        self.pages.get(light.index()).copied()
    }

    /// The bit describing `at` of `light`, if both exist.
    pub fn bit_of(&self, res: &VsmResidency, light: VsmLightHandle, at: VsmPage) -> Option<u32> {
        let base = self.light_base(light)?;
        Some(base + res.desc(light)?.entry_index(at)?)
    }

    /// **The CPU request scan** — decode a raw mask into wants, in entry order,
    /// at [`VSM_PRIORITY_MARKED`].
    ///
    /// A mask longer than the layout is truncated and a shorter one is read to
    /// its end: a mask read out of a ring can predate a registration, and the
    /// honest answer to *"this mask describes a light set that no longer
    /// exists"* is fewer wants, not a panic.
    pub fn wants(&self, res: &VsmResidency, mask: &[u32]) -> Vec<VsmWant> {
        self.wants_at(res, mask, VSM_PRIORITY_MARKED)
    }

    /// [`wants`](Self::wants) at an explicit rank — the seam a debug view or
    /// P28.4's predictor (which speculates at a *lower* rank) needs.
    pub fn wants_at(
        &self,
        res: &VsmResidency,
        mask: &[u32],
        priority: VsmPriority,
    ) -> Vec<VsmWant> {
        let mut out = Vec::new();
        for (l, (&base, &count)) in self.bases.iter().zip(self.pages.iter()).enumerate() {
            let handle = VsmLightHandle(l as u32);
            let Some(desc) = res.desc(handle) else {
                continue;
            };
            let mut entry = 0u32;
            for face in 0..desc.faces() {
                for level in 0..desc.level_count() {
                    let g = desc.levels[level as usize];
                    for y in 0..g.pages_y {
                        for x in 0..g.pages_x {
                            let bit = base + entry;
                            entry += 1;
                            if entry > count {
                                break;
                            }
                            let word = (bit / VSM_MARK_BITS_PER_WORD) as usize;
                            let Some(w) = mask.get(word) else {
                                return out;
                            };
                            if w & (1u32 << (bit % VSM_MARK_BITS_PER_WORD)) != 0 {
                                out.push(
                                    VsmWant::new(handle, VsmPage::new(face, level, x, y))
                                        .with_priority(priority),
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Set the bit for `at` in `mask` — the **CPU twin of the shader's
    /// `atomicOr`**, so a test can build the mask the GPU would build and a debug
    /// view can mark by hand.
    ///
    /// Returns `false` when the address does not exist or the mask is too short,
    /// so a caller that expected a mark can say so rather than discovering a
    /// silent no-op.
    pub fn mark(
        &self,
        res: &VsmResidency,
        mask: &mut [u32],
        light: VsmLightHandle,
        at: VsmPage,
    ) -> bool {
        let Some(bit) = self.bit_of(res, light, at) else {
            return false;
        };
        let word = (bit / VSM_MARK_BITS_PER_WORD) as usize;
        match mask.get_mut(word) {
            Some(w) => {
                *w |= 1u32 << (bit % VSM_MARK_BITS_PER_WORD);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::VsmLightDesc;
    use crate::atlas::VsmAtlasConfig;

    fn residency(pages: u64) -> VsmResidency {
        let (r, _) = VsmResidency::new(VsmAtlasConfig {
            budget_bytes: VsmAtlasConfig::default().page_bytes() * pages,
            ..Default::default()
        });
        r
    }

    /// **Bit `n` and indirection entry `n` describe one page**, for every page of
    /// every light — the property the whole layout exists to have, and the one a
    /// hand-rolled base in the shader would break first.
    #[test]
    fn a_bit_and_a_table_entry_address_the_same_page() {
        let mut r = residency(64);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let b = r.register_light(VsmLightDesc::cube(3)).unwrap();
        let layout = VsmMarkLayout::for_residency(&r);

        let pages_a = r.desc(a).unwrap().page_count();
        let pages_b = r.desc(b).unwrap().page_count();
        assert_eq!(layout.light_base(a), Some(0));
        assert_eq!(layout.light_base(b), Some(pages_a));
        assert_eq!(layout.light_bits(b), Some(pages_b));
        assert_eq!(layout.bits(), pages_a + pages_b);
        assert_eq!(layout.light_base(VsmLightHandle(9)), None);
        assert_eq!(layout.light_count(), 2);

        let mut seen = std::collections::BTreeSet::new();
        for h in [a, b] {
            let desc = r.desc(h).unwrap().clone();
            for face in 0..desc.faces() {
                for level in 0..desc.level_count() {
                    let g = desc.levels[level as usize];
                    for y in 0..g.pages_y {
                        for x in 0..g.pages_x {
                            let at = VsmPage::new(face, level, x, y);
                            let bit = layout.bit_of(&r, h, at).expect("in tree");
                            assert_eq!(
                                bit,
                                layout.light_base(h).unwrap() + desc.entry_index(at).unwrap()
                            );
                            assert!(seen.insert(bit), "bit {bit} describes two pages");
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len() as u32, layout.bits());
    }

    /// The scan comes out **in entry order** — coarse-to-fine within a face,
    /// face by face, light by light — which is the order `apply_wants` sorts on,
    /// arriving already sorted.
    #[test]
    fn the_scan_emits_wants_in_entry_order() {
        let mut r = residency(64);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let b = r.register_light(VsmLightDesc::quadtree(3)).unwrap();
        let layout = VsmMarkLayout::for_residency(&r);
        let mut mask = vec![0u32; layout.words()];

        // Marked in a deliberately scrambled order.
        let marks = [
            (b, VsmPage::flat(0, 1, 1)),
            (a, VsmPage::flat(1, 3, 3)),
            (b, VsmPage::flat(1, 0, 0)),
            (a, VsmPage::flat(0, 2, 1)),
            (a, VsmPage::flat(0, 0, 0)),
        ];
        for (h, at) in marks {
            assert!(layout.mark(&r, &mut mask, h, at), "{at:?} did not mark");
        }
        let wants = layout.wants(&r, &mask);
        assert_eq!(wants.len(), marks.len());
        assert!(
            wants.iter().all(|w| w.priority == VSM_PRIORITY_MARKED),
            "the scan produced a want that is not a marked one"
        );
        let keys: Vec<_> = wants
            .iter()
            .map(|w| (w.light.0, w.page.entry_order()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "the scan is not in entry order");
        assert_eq!(wants[0].light, a);
        assert_eq!(wants.last().unwrap().light, b);
        // …and marking something that does not exist is a `false`, not a silent
        // no-op that a caller mistakes for a mark.
        assert!(!layout.mark(&r, &mut mask, a, VsmPage::flat(0, 99, 0)));
        assert!(!layout.mark(&r, &mut mask, VsmLightHandle(9), VsmPage::flat(0, 0, 0)));
        assert!(!layout.mark(&r, &mut [], a, VsmPage::flat(0, 0, 0)));
    }

    /// **Every bit decodes to the one page it names** — the *decoder's* half of
    /// the arm above, and the one the P27.1 audit found missing.
    ///
    /// [`VsmMarkLayout::mark`] addresses a bit through
    /// [`VsmLightDesc::entry_index`](crate::VsmLightDesc::entry_index)'s
    /// `(face, level, y, x)`; [`wants_at`](VsmMarkLayout::wants_at) walks the
    /// grid itself and counts. The two agree only if the walk is the *table's*
    /// order and not the derived `(…, x, y)` — this module's own opening finding,
    /// met on the consumer side. Measured: transposing the scan's two inner loops
    /// survived every other arm in this file, because the scrambled fixture below
    /// marks pages that sit on the diagonal, where `(x, y)` and `(y, x)` agree.
    ///
    /// A round trip over **every** page of two kinds, therefore, rather than a
    /// sample: one bit in, one want out, naming the page that set it.
    #[test]
    fn every_bit_decodes_to_the_one_page_it_names() {
        let mut r = residency(64);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let b = r.register_light(VsmLightDesc::cube(3)).unwrap();
        let layout = VsmMarkLayout::for_residency(&r);
        let mut checked = 0u32;
        for h in [a, b] {
            let desc = r.desc(h).unwrap().clone();
            for face in 0..desc.faces() {
                for level in 0..desc.level_count() {
                    let g = desc.levels[level as usize];
                    for y in 0..g.pages_y {
                        for x in 0..g.pages_x {
                            let at = VsmPage::new(face, level, x, y);
                            let mut mask = vec![0u32; layout.words()];
                            assert!(layout.mark(&r, &mut mask, h, at), "{at:?}");
                            let wants = layout.wants(&r, &mask);
                            assert_eq!(wants.len(), 1, "{at:?} decoded to {wants:?}");
                            assert_eq!(
                                (wants[0].light, wants[0].page),
                                (h, at),
                                "the bit {at:?} set decodes to a DIFFERENT page — the \
                                 scan's walk is not the table's (face, level, y, x)"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            checked,
            r.desc(a).unwrap().page_count() + r.desc(b).unwrap().page_count(),
            "the round trip did not reach every page"
        );
        // ANTI-VACUITY: the sweep really did visit pages whose x and y differ, so
        // it can tell the two orders apart at all.
        assert!(checked > 300);
    }

    /// **`atomicOr` is idempotent, and the decoded want set proves it**: marking
    /// the same page twice — the ordinary case, since thousands of pixels mark
    /// the same page — is one want, and marking in either order is one mask.
    #[test]
    fn the_mask_is_order_independent_and_idempotent() {
        let mut r = residency(64);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let layout = VsmMarkLayout::for_residency(&r);
        let pages = [
            VsmPage::flat(0, 0, 0),
            VsmPage::flat(0, 1, 0),
            VsmPage::flat(1, 4, 4),
        ];

        let build = |order: &[usize], repeats: usize| {
            let mut mask = vec![0u32; layout.words()];
            for _ in 0..repeats {
                for &i in order {
                    layout.mark(&r, &mut mask, h, pages[i]);
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
    /// normal event when a level's light set changes rather than a corruption.
    #[test]
    fn a_short_or_stale_mask_is_truncated_rather_than_invented() {
        let mut r = residency(64);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let old = VsmMarkLayout::for_residency(&r);
        let mut mask = vec![0u32; old.words()];
        assert!(old.mark(&r, &mut mask, a, VsmPage::flat(0, 0, 0)));

        r.register_light(VsmLightDesc::cube(4)).unwrap();
        let now = VsmMarkLayout::for_residency(&r);
        assert!(
            now.words() > mask.len(),
            "the fixture did not grow the mask"
        );
        let wants = now.wants(&r, &mask);
        assert!(
            wants.iter().all(|w| w.light == a),
            "a stale mask named a light it could not have seen"
        );
        assert!(!wants.is_empty(), "the truncation ate everything");
        assert!(now.wants(&r, &vec![0u32; now.words()]).is_empty());
    }

    /// A layout over no lights is legal and binds one word — an atlas exists
    /// before anything registers into it.
    #[test]
    fn an_empty_layout_still_binds_a_legal_buffer() {
        let r = residency(4);
        let layout = VsmMarkLayout::for_residency(&r);
        assert_eq!(layout.bits(), 0);
        assert_eq!(layout.words(), 1, "a wgpu buffer may not be zero-sized");
        assert!(layout.wants(&r, &[0]).is_empty());
    }
}
