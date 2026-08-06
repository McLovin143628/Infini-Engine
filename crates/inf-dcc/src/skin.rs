//! **The skin channel** (P24.2): which joints deform a vertex, and by how much.
//!
//! Until this batch the kernel had no place to put a weight, and said so out
//! loud: [`crate::build::from_mesh_asset`] **refused** a skinned `MeshAsset`
//! rather than dropping its weights, and [`crate::export::to_mesh_asset`] wrote
//! an empty skin stream because "the kernel has no skinning model, so an exported
//! submesh is rigid by construction, never by omission". Both refusals were
//! deliberate placeholders for this module, and both are now real flow.
//!
//! # The representation is `inf_mesh::VertexSkin`'s, deliberately mirrored
//!
//! Four influences, `u16` joint index, `f32` weight, `Σw = 1` — the same shape
//! the glTF importer produces and the same shape the GPU skinning pass uploads,
//! so import and export are a copy and nothing is re-derived on the way through.
//!
//! It is a **mirror**, not a re-export, for the reason [`crate::sculpt::SculptFalloff`]
//! is a mirror of `inf_terrain::Falloff`: this type rides inside a journalled
//! [`crate::ops::Op`] and inside [`crate::journal::SessionSave`], so a field
//! appended to `inf_mesh::VertexSkin` — a perfectly ordinary thing to do to a
//! render type — would silently change the shape of every session ever written
//! here. `the_wire_layout_matches_inf_mesh` pins the two against each other so
//! the mirror cannot drift *quietly*; it can only drift loudly.
//!
//! # `f32` weights in an `f64` kernel
//!
//! Architecture rule 3 is about **world positions**, and a weight is not one: it
//! is a dimensionless coefficient in `[0, 1]` whose whole life is spent between
//! an `f32` file and an `f32` vertex buffer. Storing it as `f64` would add a
//! narrowing at export whose rounding nobody can see or control, and would make
//! the import → kernel → export round trip lossy in the one direction that has to
//! be exact. The solvers (P24.2's heat diffusion) work in `f64` and convert
//! **once, at the wire** — the same discipline `inf_anim::template` records.
//!
//! # Two Options, and what each one means
//!
//! * [`Mesh::skin_binding`](crate::topo::Mesh::skin_binding) is `Option<SkinBinding>`:
//!   *is this mesh skinned at all*, and if so, to which skeleton and how many
//!   joints. Absent ⇒ the mesh is rigid.
//! * A vertex's [`VertWeights`] is **total**, never optional, and defaults to
//!   [`VertWeights::RIGID`]. That is what keeps every pre-existing op working
//!   unchanged: `Op::AddVertex` has no weight field (it is frozen on the wire),
//!   so a vertex born after a bind has to have *some* answer, and "all of joint
//!   0" is the only one that is always in range. The alternative — an
//!   `Option` per vertex plus a completeness invariant — makes the invariant
//!   fail on the very next `AddVertex`.
//!
//! [`crate::validate`] closes the gap the pair leaves: an **unbound** mesh whose
//! vertices carry non-rigid weights is a violation, so a mesh cannot hold orphan
//! weights that nothing can interpret.
//!
//! # New vertices inherit their weights from where they came from
//!
//! Every structural op that creates a vertex creates it *out of* existing ones,
//! and [`crate::topo::Mesh::alloc_vert_blended`] is the one door that takes that
//! provenance: a split-edge midpoint blends its two endpoints at `1 − t` / `t`, a
//! subdivision centroid blends its face's corners equally, an extruded or
//! mirrored copy inherits its source's weights outright. On an unskinned mesh
//! every source is [`VertWeights::RIGID`] and the blend is exactly `RIGID` again,
//! so nothing about a rigid edit moved by a single bit.
//!
//! # What is NOT here, and is ledgered
//!
//! * **A mirror op does not mirror joints.** `Op::Mirror` copies each vertex's
//!   weights verbatim, so a mirrored left arm is still weighted to the left
//!   arm's joints. Fixing it needs the skeleton (to pair `upper_arm_l` with
//!   `upper_arm_r`), which the kernel deliberately does not hold — it belongs to
//!   the auto-fit layer, and it is written down in ROADMAP §12's P24 block.
//! * **A weld keeps the surviving vertex's weights.** `WeldVerts`, `MergeVerts`
//!   and `CollapseEdge` all name a survivor, and the survivor keeps what it had.
//!   Averaging would be a silent third behaviour for an op whose contract is
//!   "this vertex wins".

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The most joints that may influence one vertex.
///
/// Four, because that is what `inf_mesh::VertexSkin` stores and what the GPU
/// skinning pass reads. It is a *format* limit, not a taste one: a fifth
/// influence has nowhere to be written.
pub const MAX_INFLUENCES: usize = 4;

/// How far a vertex's weights may sum from `1.0` before [`crate::validate`]
/// calls it a violation.
///
/// Not zero: renormalization divides in `f32`, so a four-way split lands a few
/// ULPs off by construction and a zero tolerance would refuse the solver's own
/// output. `1e-5` is ~80 ULPs at 1.0 — far tighter than any authoring error and
/// far looser than any rounding.
pub const WEIGHT_SUM_TOLERANCE: f32 = 1e-5;

/// One vertex's skinning influences: up to [`MAX_INFLUENCES`] joints and their
/// normalized weights.
///
/// A zero weight means "this slot is unused"; its joint index is then
/// meaningless but must still be **in range**, which is why the unused slots of
/// [`VertWeights::RIGID`] name joint 0 rather than `u16::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VertWeights {
    /// Joint indices into the bound skeleton's `joints` list.
    pub joints: [u16; MAX_INFLUENCES],
    /// Influence weights, summing to 1.
    pub weights: [f32; MAX_INFLUENCES],
}

impl Default for VertWeights {
    fn default() -> Self {
        Self::RIGID
    }
}

impl VertWeights {
    /// **All of joint 0** — the value every vertex carries on an unskinned mesh,
    /// and the value a fresh vertex is born with.
    ///
    /// Deliberately identical to `inf_mesh::VertexSkin::default()`, so a rigid
    /// vertex written through the export path is byte-for-byte what the glTF
    /// importer writes for an unweighted one.
    pub const RIGID: Self = Self {
        joints: [0; MAX_INFLUENCES],
        weights: [1.0, 0.0, 0.0, 0.0],
    };

    /// Whether this is exactly [`VertWeights::RIGID`].
    pub fn is_rigid(&self) -> bool {
        *self == Self::RIGID
    }

    /// The total weight on `joint` (summed over slots, since a caller may hand
    /// the same joint twice).
    pub fn weight_of(&self, joint: u16) -> f32 {
        self.joints
            .iter()
            .zip(&self.weights)
            .filter(|(j, _)| **j == joint)
            .map(|(_, w)| *w)
            .sum()
    }

    /// This vertex's influences as `(joint, weight)`, **zero-weight slots
    /// dropped**, in the canonical order: descending weight, ties broken by
    /// ascending joint index.
    ///
    /// The canonical order is what [`crate::topo::CanonicalMesh`] compares and
    /// what the solvers emit; the *stored* order is deliberately left alone, so
    /// an imported asset exports the influences back in the order it supplied
    /// them and the round trip is exact.
    pub fn influences(&self) -> Vec<(u16, f32)> {
        let mut out: Vec<(u16, f32)> = self
            .joints
            .iter()
            .zip(&self.weights)
            .filter(|(_, w)| **w != 0.0)
            .map(|(j, w)| (*j, *w))
            .collect();
        // Sort on the weight's BITS, not on the float: the ordering has to be
        // total (a NaN slipping through would make `sort_by` misbehave), and for
        // non-negative finite floats the bit pattern orders exactly as the value
        // does. Descending weight, then ascending joint.
        out.sort_by(|a, b| {
            b.1.to_bits()
                .cmp(&a.1.to_bits())
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// Build from `(joint, weight)` pairs: the top [`MAX_INFLUENCES`] by weight
    /// are kept and renormalized; the rest are dropped.
    ///
    /// Deterministic by construction — pairs are accumulated into a `BTreeMap`
    /// (so a joint given twice is summed, in joint order) and the top-four cut
    /// breaks ties by ascending joint index. Two runs on the same input produce
    /// the same bits.
    ///
    /// Degenerate input (empty, all-zero, non-finite or negative) yields
    /// [`VertWeights::RIGID`]: the caller has said nothing about this vertex, and
    /// the one answer that is always in range is the one the kernel already uses
    /// for "unweighted".
    pub fn from_pairs(pairs: &[(u16, f64)]) -> Self {
        let mut acc: BTreeMap<u16, f64> = BTreeMap::new();
        for &(j, w) in pairs {
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            *acc.entry(j).or_insert(0.0) += w;
        }
        let mut ranked: Vec<(u16, f64)> = acc.into_iter().collect();
        // Descending weight, ascending joint on a tie — and `total_cmp` because
        // the ordering must be total even though the filter above has already
        // excluded NaN.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(MAX_INFLUENCES);
        let sum: f64 = ranked.iter().map(|(_, w)| *w).sum();
        if ranked.is_empty() || sum <= 0.0 || !sum.is_finite() {
            return Self::RIGID;
        }
        let mut out = Self {
            joints: [0; MAX_INFLUENCES],
            weights: [0.0; MAX_INFLUENCES],
        };
        for (slot, (j, w)) in ranked.iter().enumerate() {
            out.joints[slot] = *j;
            out.weights[slot] = (*w / sum) as f32;
        }
        // The f64 → f32 narrowing above can leave the sum a ULP off 1; one f32
        // renormalization lands it inside `WEIGHT_SUM_TOLERANCE` for good.
        out.normalized()
    }

    /// Renormalize so the weights sum to 1, in `f32` — the exact arithmetic the
    /// exported stream will carry.
    ///
    /// A degenerate total (zero, negative, non-finite) falls back to
    /// [`VertWeights::RIGID`], which is `inf_mesh::VertexSkin::normalized`'s rule
    /// and the only one that cannot produce an out-of-range slot.
    pub fn normalized(mut self) -> Self {
        let sum: f32 = self.weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Self::RIGID;
        }
        for (j, w) in self.joints.iter_mut().zip(self.weights.iter_mut()) {
            if !w.is_finite() || *w < 0.0 {
                return Self::RIGID;
            }
            *w /= sum;
            if *w == 0.0 {
                // An emptied slot names joint 0, so an unused influence can never
                // be an out-of-range index (see the type docs).
                *j = 0;
            }
        }
        self
    }

    /// Whether every weight is finite and non-negative and they sum to 1 within
    /// [`WEIGHT_SUM_TOLERANCE`]. The predicate [`crate::validate`] reports on.
    pub fn is_normalized(&self) -> bool {
        if !self.weights.iter().all(|w| w.is_finite() && *w >= 0.0) {
            return false;
        }
        (self.weights.iter().sum::<f32>() - 1.0).abs() <= WEIGHT_SUM_TOLERANCE
    }

    /// The highest joint index this vertex names in a **non-zero** slot, or
    /// `None` when it names none.
    ///
    /// Zero-weight slots are excluded on purpose: they carry no influence, and
    /// range-checking a slot nobody reads would make a perfectly good weight
    /// table refuse against a smaller skeleton for no reason. (They are still
    /// pinned to joint 0 by [`VertWeights::normalized`], so this is belt and
    /// braces, not a loophole.)
    pub fn max_joint(&self) -> Option<u16> {
        self.joints
            .iter()
            .zip(&self.weights)
            .filter(|(_, w)| **w != 0.0)
            .map(|(j, _)| *j)
            .max()
    }

    /// **Paint**: set `joint`'s share to `target` and rescale the rest to fill
    /// what is left, then renormalize.
    ///
    /// The rescale is what makes a brush feel like a brush: raising one joint to
    /// 0.6 leaves the other influences in the same *proportion* to each other,
    /// so painting an arm does not silently re-rank a shoulder against a chest.
    /// When the other influences are all zero (or there are none) the remainder
    /// has nowhere to go and `joint` takes the whole vertex.
    ///
    /// A vertex already at [`MAX_INFLUENCES`] that does not name `joint`
    /// **drops its smallest influence** to make room — the standard four-bone
    /// budget rule, applied where the author can see it happen rather than
    /// silently at export.
    pub fn painted(mut self, joint: u16, target: f32) -> Self {
        let target = target.clamp(0.0, 1.0);
        let slot = match self.joints.iter().zip(&self.weights).position(|(j, w)| {
            // An existing slot for this joint, or an unused one. A zero-weight
            // slot never counts as "this joint's" even when its index matches,
            // because `normalized` parks unused slots on joint 0.
            *j == joint && *w != 0.0
        }) {
            Some(slot) => slot,
            None => match self.weights.iter().position(|w| *w == 0.0) {
                Some(free) => free,
                None => {
                    // Full: evict the smallest influence.
                    let mut worst = 0usize;
                    for i in 1..MAX_INFLUENCES {
                        if self.weights[i] < self.weights[worst] {
                            worst = i;
                        }
                    }
                    worst
                }
            },
        };
        let others: f32 = self
            .weights
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != slot)
            .map(|(_, w)| *w)
            .sum();
        let room = 1.0 - target;
        for (i, w) in self.weights.iter_mut().enumerate() {
            if i == slot {
                continue;
            }
            *w = if others > 0.0 {
                *w / others * room
            } else {
                0.0
            };
        }
        self.joints[slot] = joint;
        self.weights[slot] = target;
        self.normalized()
    }
}

/// What a skinned [`crate::topo::Mesh`]'s joint indices mean.
///
/// # Why the skeleton is raw bytes, and why it is optional
///
/// `[u8; 16]` is a GUID, and it is spelled as bytes because this crate has no
/// `uuid` dependency and does not want one — the same choice
/// `inf_anim::AnimClipAsset` makes for exactly the same reason. The editor
/// converts at its own edge.
///
/// It is `Option` because **a `.inf_mesh` does not record which skeleton it is
/// skinned to.** `inf_mesh::SubMesh::skin` is a bare weight stream; the pairing
/// lives one level up, in the scene's `SkeletalMesh { mesh, skeleton }`. So a
/// mesh opened from a bare asset is genuinely "bound to *a* skeleton, and this
/// file does not say which", and `None` says exactly that. A nil GUID would say
/// the same thing while looking like an answer.
///
/// # Why the joint count is stored
///
/// Without it "is this joint index in range" is unanswerable inside the kernel,
/// which holds no skeleton — so the range check would have to live in whichever
/// caller happened to have one, which is to say nowhere. Stored, it makes
/// [`crate::validate`]'s skin check a local property of the mesh, exactly like
/// every other check in that module.
///
/// The count is the skeleton's length **at bind time**. A skeleton that later
/// grows keeps every existing index valid; one that shrinks invalidates the
/// binding, which is a re-bind, not a silent clamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SkinBinding {
    /// The `.inf_skel` GUID this mesh's joint indices address, as raw bytes, or
    /// `None` when the mesh was read from an asset that does not record one.
    pub skeleton: Option<[u8; 16]>,
    /// How many joints the bound skeleton has. Every non-zero influence must
    /// name an index below it.
    ///
    /// On import this is the tightest count the *file* supports — one past the
    /// highest index any vertex names — because that is all a bare weight stream
    /// says. Re-binding against a real skeleton widens it to the real count,
    /// which is a [`crate::ops::Op::BindSkin`] and therefore a journal entry.
    pub joints: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_layout_matches_inf_mesh() {
        // The mirror, pinned in the direction that matters: a `VertWeights`
        // converted field-for-field into an `inf_mesh::VertexSkin` and back is
        // the same value, and both are the same size. A field appended to
        // either one fails this — loudly, which is the whole point of mirroring
        // rather than re-exporting.
        assert_eq!(
            std::mem::size_of::<VertWeights>(),
            std::mem::size_of::<inf_mesh::VertexSkin>(),
        );
        let mine = VertWeights {
            joints: [3, 7, 0, 0],
            weights: [0.6, 0.4, 0.0, 0.0],
        };
        let theirs = inf_mesh::VertexSkin {
            joints: mine.joints,
            weights: mine.weights,
        };
        assert_eq!(theirs.joints, mine.joints);
        assert_eq!(theirs.weights, mine.weights);
        // …and the two "unweighted" defaults are the same value, which is what
        // makes a rigid vertex export byte-identically to an imported one.
        let d = inf_mesh::VertexSkin::default();
        assert_eq!(
            (d.joints, d.weights),
            (VertWeights::RIGID.joints, VertWeights::RIGID.weights)
        );
    }

    #[test]
    fn from_pairs_keeps_the_top_four_and_normalizes() {
        let w = VertWeights::from_pairs(&[(9, 0.1), (2, 0.5), (5, 0.2), (7, 0.15), (1, 0.05)]);
        // Five influences in, four out — the two smallest are 1 (0.05) and
        // 9 (0.1), and the smaller of them is dropped.
        assert_eq!(w.joints, [2, 5, 7, 9]);
        assert!(w.is_normalized(), "{w:?}");
        assert!((w.weights[0] - 0.5 / 0.95).abs() < 1e-6, "{w:?}");
        assert_eq!(w.max_joint(), Some(9));
    }

    #[test]
    fn from_pairs_is_deterministic_and_ties_break_by_joint() {
        let a = VertWeights::from_pairs(&[(4, 0.25), (1, 0.25), (3, 0.25), (2, 0.25), (0, 0.25)]);
        let b = VertWeights::from_pairs(&[(0, 0.25), (2, 0.25), (3, 0.25), (1, 0.25), (4, 0.25)]);
        assert_eq!(a, b, "the order the pairs arrive in must not matter");
        assert_eq!(a.joints, [0, 1, 2, 3], "a tie breaks by ascending joint");
    }

    #[test]
    fn degenerate_pairs_fall_back_to_rigid() {
        for pairs in [
            vec![],
            vec![(1u16, 0.0f64)],
            vec![(1, -1.0)],
            vec![(1, f64::NAN)],
            vec![(1, f64::INFINITY)],
        ] {
            assert_eq!(
                VertWeights::from_pairs(&pairs),
                VertWeights::RIGID,
                "{pairs:?} must not produce an uninterpretable vertex"
            );
        }
    }

    #[test]
    fn a_blend_of_rigid_sources_is_exactly_rigid() {
        // The claim the whole "no bit moved on a rigid mesh" argument rests on.
        assert_eq!(
            VertWeights::from_pairs(&[(0, 0.5), (0, 0.5)]),
            VertWeights::RIGID
        );
        assert_eq!(VertWeights::from_pairs(&[(0, 1.0)]), VertWeights::RIGID);
        assert_eq!(
            VertWeights::from_pairs(&[(0, 1.0 / 3.0), (0, 1.0 / 3.0), (0, 1.0 / 3.0)]),
            VertWeights::RIGID
        );
    }

    #[test]
    fn painting_rescales_the_others_in_proportion() {
        let start = VertWeights::from_pairs(&[(1, 0.5), (2, 0.3), (3, 0.2)]);
        let painted = start.painted(1, 0.8);
        assert!((painted.weight_of(1) - 0.8).abs() < 1e-5, "{painted:?}");
        // 2 and 3 were 3:2 and must still be 3:2 inside the remaining 0.2.
        let (w2, w3) = (painted.weight_of(2), painted.weight_of(3));
        assert!((w2 - 0.12).abs() < 1e-5, "{painted:?}");
        assert!((w3 - 0.08).abs() < 1e-5, "{painted:?}");
        assert!(painted.is_normalized());
    }

    #[test]
    fn painting_a_full_vertex_evicts_its_smallest_influence() {
        let start = VertWeights::from_pairs(&[(1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]);
        let painted = start.painted(9, 0.5);
        assert_eq!(painted.weight_of(4), 0.0, "the smallest went: {painted:?}");
        assert!((painted.weight_of(9) - 0.5).abs() < 1e-5, "{painted:?}");
        assert!(painted.is_normalized());
        assert_eq!(painted.influences().len(), MAX_INFLUENCES);
    }

    #[test]
    fn painting_to_one_takes_the_whole_vertex() {
        let start = VertWeights::from_pairs(&[(1, 0.5), (2, 0.5)]);
        let painted = start.painted(2, 1.0);
        assert_eq!(painted.weight_of(2), 1.0);
        assert_eq!(painted.weight_of(1), 0.0);
        assert!(painted.is_normalized());
    }

    #[test]
    fn painting_to_zero_hands_the_vertex_to_the_others() {
        let start = VertWeights::from_pairs(&[(1, 0.5), (2, 0.5)]);
        let painted = start.painted(1, 0.0);
        assert_eq!(painted.weight_of(1), 0.0);
        assert!((painted.weight_of(2) - 1.0).abs() < 1e-6, "{painted:?}");
        // …and taking the LAST influence to zero cannot produce a vertex that
        // sums to nothing: the fallback is RIGID, not an empty table.
        let empty = VertWeights::from_pairs(&[(5, 1.0)]).painted(5, 0.0);
        assert!(empty.is_normalized(), "{empty:?}");
    }

    #[test]
    fn the_canonical_influence_order_is_independent_of_storage_order() {
        let a = VertWeights {
            joints: [7, 2, 0, 0],
            weights: [0.25, 0.75, 0.0, 0.0],
        };
        let b = VertWeights {
            joints: [2, 7, 0, 0],
            weights: [0.75, 0.25, 0.0, 0.0],
        };
        assert_ne!(a, b, "the STORED order differs, deliberately");
        assert_eq!(a.influences(), b.influences());
        assert_eq!(a.influences(), vec![(2, 0.75), (7, 0.25)]);
    }
}
