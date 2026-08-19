//! **Re-parameterizing an edit that is already in the past** (Wave D) — the
//! journal's answer to a modifier stack, and the one thing in this kernel that
//! Blender does not do for mesh editing.
//!
//! Blender's non-destructive story is an ordered list of parametric operators
//! evaluated on top of frozen base geometry. Ours is strictly more general and
//! was already sitting here unused: `base + Vec<Op> + cursor`, where
//! `replay(base, ops)` is property-tested to be a **pure function of the ops**
//! ([`crate::journal`]). A modifier stack is a *suffix* of parametric operations;
//! a journal is the whole edit history as data. So the feature is not a stack —
//! it is `ops[i] = new_op` followed by a replay of everything after it.
//!
//! Blender's F9 redo panel re-parameterizes the operation you just did. This
//! re-parameterizes the one you did twelve steps ago, and the rest of the model
//! re-derives.
//!
//! # The hazard, stated before the design
//!
//! Ops carry [`VertId`]/[`HalfId`]/[`FaceId`], and those are **arena slots**
//! ([`crate::topo`]'s id-reuse rule). A structural op rebuilds a local patch and
//! hands its ids back to a LIFO free list, so an amendment that changes *how
//! many* elements an op creates renumbers everything downstream — and a stale id
//! is not dead, so nothing downstream would notice. The next op in the journal
//! would apply cleanly to the wrong polygon.
//!
//! # Two gates, and neither is trusted alone
//!
//! **1. The classifier — [`op_amendable`] and [`op_structure`], both exhaustive
//! `match`es with no wildcard.** They mirror
//! [`crate::select::op_preserves_ids`]: a new [`Op`] stops the crate compiling
//! until an author has said which of its fields the id allocation depends on.
//! This is the *advertised* contract, and it is what a UI reads to decide which
//! ops to offer a slider for.
//!
//! **2. The verification — a topology comparison after EVERY replayed op.**
//! The amendment is applied to a clone, and at each step the new branch's
//! topology (which ids are live, and what each half-edge's `origin` / `twin` /
//! `next` / `face` are) is compared against the old branch's. Only *geometry* —
//! positions, UVs, normals, weights — may differ. If they ever diverge, the
//! amendment is refused and the session is byte-identical.
//!
//! The second gate is not belt-and-braces decoration. It is what makes
//! [`Op::Mirror`]'s `coord` amendable at all: the mirror welds at a tolerance of
//! **exactly zero**, so moving the plane changes *how many vertices land on it*,
//! which changes the id allocation — and there is no way to classify that
//! statically. The classifier says "coord may change"; the verification says
//! whether it did, on this mesh, this time. A classifier alone would have to
//! refuse the case outright; a verification alone would let a mis-declared new op
//! through on the meshes that happen not to trip it.
//!
//! # Refusals are values, and they are inert
//!
//! Every refusal leaves the session **byte-identical** — the same law the ops
//! themselves obey, for the same reason. Everything is computed on clones and
//! nothing is written until the last check passes.

use crate::ops::{Op, OpError};
use crate::topo::{FaceId, HalfId, Mesh, VertId};

/// Whether an op's parameters may be changed after the fact.
///
/// **Exhaustive on purpose**, exactly like
/// [`crate::select::op_preserves_ids`]: a new [`Op`] stops this compiling until
/// an author has decided whether re-parameterizing it can renumber the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Amendability {
    /// **Every field may change.** The op creates and destroys no element, so
    /// the ids it leaves behind are whatever the mesh already had. Precisely the
    /// ops [`crate::select::op_preserves_ids`] returns `true` for, and the
    /// coincidence is not one: "nothing is renumbered" is the same statement
    /// read from two directions.
    Free,
    /// **Some fields are fixed.** [`op_structure`] names them; the rest are the
    /// parameters an author may drag.
    Structural,
    /// **Nothing may change.** Either the op has no parameter that is not also
    /// its structure (a `CollapseEdge` *is* its half-edge), or the parameter
    /// decides how many elements exist (`LoopCut`'s count) — in which case an
    /// amendment is not a re-parameterization, it is a different edit.
    Never,
}

/// The part of an op the id allocation depends on — everything an amendment may
/// **not** change.
///
/// Compared for equality by [`amend_shape_ok`], so a field that belongs here and
/// is left out is a hole in the first gate. Also exhaustive, and deliberately a
/// second `match`: [`Amendability`] says *whether*, this says *which*, and a
/// single function answering both is a function nobody updates correctly.
#[derive(Debug, Clone, PartialEq)]
pub struct OpStructure {
    /// Two different variants are never amendable into each other, whatever
    /// their fields say.
    kind: std::mem::Discriminant<Op>,
    verts: Vec<VertId>,
    halfs: Vec<HalfId>,
    faces: Vec<FaceId>,
    /// Non-id fields that change *how much* is built: `individual`, `segments`,
    /// `cuts`, the mirror axis, a merge target. Encoded as integers because
    /// that is what they are.
    shape: Vec<u64>,
}

/// Why an amendment was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AmendError {
    /// `index` is not an applied op. Amending the **redo tail** is refused
    /// rather than silently truncating it: a redo the author has not performed
    /// is still their work, and rewriting it in place would change what
    /// pressing redo does with no record that anything happened.
    #[error("op {index} is not in the applied history ({cursor} of {ops} ops are applied)")]
    OutOfRange {
        index: usize,
        cursor: usize,
        ops: usize,
    },
    /// The replacement is a different kind of op.
    #[error("an amendment may re-parameterize an op, not replace it with a different one")]
    DifferentKind,
    /// The op is [`Amendability::Never`].
    #[error("a {kind} cannot be re-parameterized: {why}")]
    NotAmendable {
        kind: &'static str,
        why: &'static str,
    },
    /// A [`Amendability::Structural`] op whose fixed fields moved.
    #[error("that amendment changes which elements the op is aimed at, not just its parameters")]
    StructureChanged,
    /// The amended op itself refused.
    #[error("the amended op refused: {0}")]
    Refused(OpError),
    /// A later op refused once the amendment had landed — usually because the
    /// amendment made something it referred to impossible.
    #[error("op {index} in the history no longer applies after that amendment: {source}")]
    TailRefused { index: usize, source: OpError },
    /// **The amendment renumbered the mesh.** The classifier permitted it and
    /// the verification caught it, which is the case the two-gate design exists
    /// for — `Op::Mirror`'s coordinate is the canonical example: moving the
    /// plane changes how many vertices weld onto it.
    #[error("that amendment changes the mesh's structure, so op {index} in the history would land on different geometry")]
    IdAllocationChanged { index: usize },
    /// Replaying the amended history produced an invalid mesh. A bug in this
    /// crate rather than in the amendment, and still a refusal: the session is
    /// left exactly as it was.
    #[error("replaying the amended history produced an invalid mesh: {} violation(s)", .0.len())]
    InvalidResult(Vec<crate::validate::Violation>),
}

/// Whether an op's parameters may be changed after the fact. See
/// [`Amendability`].
pub fn op_amendable(op: &Op) -> Amendability {
    match op {
        // ── Free: `op_preserves_ids` is `true`, so nothing can renumber ─────
        Op::AddVertex { .. }
        | Op::TranslateVerts { .. }
        | Op::SetCornerUv { .. }
        | Op::SetCornerNormal { .. }
        | Op::SetEdgeSharp { .. }
        | Op::SetFaceSlot { .. }
        | Op::Sculpt { .. }
        | Op::RotateVerts { .. }
        | Op::ScaleVerts { .. }
        | Op::SetEdgeSeam { .. }
        | Op::Unwrap { .. }
        | Op::BindSkin { .. }
        | Op::AssignWeights { .. }
        | Op::AddMaterialSlots { .. }
        | Op::MoveVerts { .. }
        | Op::SetEdgesSharp { .. }
        | Op::SetEdgesSeam { .. } => Amendability::Free,

        // ── Structural: an operand set is fixed, a magnitude is not ─────────
        //
        // These are the amendments an author actually reaches for. "Make that
        // extrude 0.6 instead of 0.4" is the whole feature.
        Op::AddFace { .. }        // the vertex loop is fixed; corners and slot are not
        | Op::SplitEdge { .. }    // the edge is fixed; where along it is not
        | Op::ExtrudeFaces { .. } // the region is fixed; the distance is not
        | Op::ExtrudeEdges { .. } // the edges are fixed; the delta is not
        | Op::InsetFaces { .. }   // the region and `individual` are fixed; the amount is not
        | Op::BevelEdges { .. }   // the edges and the segment count are fixed; the amount is not
        | Op::Mirror { .. } => Amendability::Structural, // the axis is fixed; the coordinate is not

        // ── Never ──────────────────────────────────────────────────────────
        //
        // Each of these is *entirely* structure. Changing the only field is not
        // re-parameterizing the edit, it is doing a different one — which the
        // author can express by undoing and redoing it, with a history entry
        // that says so.
        Op::RemoveVertex { .. }
        | Op::RemoveFace { .. }
        | Op::CollapseEdge { .. }
        | Op::SplitFace { .. }
        | Op::WeldVerts { .. }
        | Op::Knife { .. }
        | Op::SubdivideFaces { .. }
        | Op::ClearSkin
        | Op::FlipFaces { .. }
        | Op::DissolveEdges { .. }
        | Op::BridgeLoops { .. }
        // …and these two have a second field that looks parametric and is not.
        // `LoopCut`'s `cuts` decides how many loops exist. `MergeVerts`'s
        // `target` decides which vertex SURVIVES (`ordered[0]` for `Center`,
        // `ordered[last]` for `Last` — `crate::model::merge_verts`), so it
        // renumbers as surely as the set does.
        | Op::LoopCut { .. }
        | Op::MergeVerts { .. } => Amendability::Never,
    }
}

/// A one-word name for an op, for a refusal message a UI can show.
pub fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::AddVertex { .. } => "vertex add",
        Op::RemoveVertex { .. } => "vertex delete",
        Op::AddFace { .. } => "face add",
        Op::RemoveFace { .. } => "face delete",
        Op::SplitEdge { .. } => "edge split",
        Op::CollapseEdge { .. } => "edge collapse",
        Op::SplitFace { .. } => "face split",
        Op::WeldVerts { .. } => "weld",
        Op::TranslateVerts { .. } => "translate",
        Op::SetCornerUv { .. } => "uv write",
        Op::SetCornerNormal { .. } => "normal write",
        Op::SetEdgeSharp { .. } => "sharpness",
        Op::SetFaceSlot { .. } => "material assign",
        Op::ExtrudeFaces { .. } => "extrude",
        Op::ExtrudeEdges { .. } => "edge extrude",
        Op::InsetFaces { .. } => "inset",
        Op::BevelEdges { .. } => "bevel",
        Op::LoopCut { .. } => "loop cut",
        Op::Knife { .. } => "knife",
        Op::MergeVerts { .. } => "merge",
        Op::SubdivideFaces { .. } => "subdivide",
        Op::Mirror { .. } => "mirror",
        Op::Sculpt { .. } => "sculpt stroke",
        Op::RotateVerts { .. } => "rotate",
        Op::ScaleVerts { .. } => "scale",
        Op::SetEdgeSeam { .. } => "seam",
        Op::Unwrap { .. } => "unwrap",
        Op::BindSkin { .. } => "skin bind",
        Op::AssignWeights { .. } => "weights",
        Op::ClearSkin => "skin clear",
        Op::AddMaterialSlots { .. } => "material slots",
        Op::MoveVerts { .. } => "soft move",
        Op::SetEdgesSharp { .. } => "shading",
        Op::SetEdgesSeam { .. } => "auto-seam",
        Op::FlipFaces { .. } => "flip",
        Op::DissolveEdges { .. } => "dissolve",
        Op::BridgeLoops { .. } => "bridge",
    }
}

/// **The one number an author may drag on an op**, and its unit — or `None`
/// when the op has no scalar to change.
///
/// Deliberately one number and not a form. The ops an author reaches back for
/// are the ones with a *magnitude* — how far did that extrude go, how big was
/// that bevel — and a general parameter editor over a wire enum would be a
/// second reflection system, which is the drift the P23 memo's no-DNA/RNA ruling
/// forbids. An op with more than one interesting number (a translate's three
/// components, a sculpt's radius *and* strength) reports `None` rather than
/// picking one arbitrarily and pretending the others do not exist.
///
/// Units: metres for a distance, **degrees** for an angle (the units doctrine —
/// degrees at the UI boundary, radians in the op), and a bare number for a
/// parameter.
pub fn op_scalar(op: &Op) -> Option<(f64, &'static str)> {
    match op {
        Op::ExtrudeFaces { distance, .. } => Some((*distance, "m")),
        Op::InsetFaces { amount, .. } => Some((*amount, "m")),
        Op::BevelEdges { amount, .. } => Some((*amount, "m")),
        Op::Mirror { coord, .. } => Some((*coord, "m")),
        Op::SplitEdge { t, .. } => Some((*t, "")),
        _ => None,
    }
}

/// Replace an op's scalar, leaving every other field alone. `None` when the op
/// has no scalar — which is the same set [`op_scalar`] answers `None` for, and
/// the two are asserted to agree.
pub fn with_scalar(op: &Op, value: f64) -> Option<Op> {
    Some(match op {
        Op::ExtrudeFaces { faces, .. } => Op::ExtrudeFaces {
            faces: faces.clone(),
            distance: value,
        },
        Op::InsetFaces {
            faces, individual, ..
        } => Op::InsetFaces {
            faces: faces.clone(),
            amount: value,
            individual: *individual,
        },
        Op::BevelEdges {
            edges, segments, ..
        } => Op::BevelEdges {
            edges: edges.clone(),
            amount: value,
            segments: *segments,
        },
        Op::Mirror { axis, .. } => Op::Mirror {
            axis: *axis,
            coord: value,
        },
        Op::SplitEdge { half, .. } => Op::SplitEdge {
            half: *half,
            t: value,
        },
        _ => return None,
    })
}

/// Why a [`Amendability::Never`] op is one, in words a panel can show.
pub fn why_never(op: &Op) -> &'static str {
    match op {
        Op::LoopCut { .. } => {
            "its cut count decides how many loops exist, so changing it is a different edit"
        }
        Op::MergeVerts { .. } => {
            "its target decides which vertex survives, so changing it renumbers the mesh"
        }
        Op::ClearSkin => "it has no parameters",
        _ => {
            "its parameters ARE the elements it names — undo it and make the edit you meant instead"
        }
    }
}

/// The fields an amendment may not change. See [`OpStructure`].
pub fn op_structure(op: &Op) -> OpStructure {
    let kind = std::mem::discriminant(op);
    let mut s = OpStructure {
        kind,
        verts: Vec::new(),
        halfs: Vec::new(),
        faces: Vec::new(),
        shape: Vec::new(),
    };
    match op {
        // `Free` ops have no structure to fix — the discriminant is the whole
        // constraint. Listed rather than wildcarded, because a wildcard here
        // would silently absorb a future op whose parameters DO renumber.
        Op::AddVertex { .. }
        | Op::TranslateVerts { .. }
        | Op::SetCornerUv { .. }
        | Op::SetCornerNormal { .. }
        | Op::SetEdgeSharp { .. }
        | Op::SetFaceSlot { .. }
        | Op::Sculpt { .. }
        | Op::RotateVerts { .. }
        | Op::ScaleVerts { .. }
        | Op::SetEdgeSeam { .. }
        | Op::Unwrap { .. }
        | Op::BindSkin { .. }
        | Op::AssignWeights { .. }
        | Op::ClearSkin
        | Op::AddMaterialSlots { .. }
        | Op::MoveVerts { .. }
        | Op::SetEdgesSharp { .. }
        | Op::SetEdgesSeam { .. } => {}

        Op::AddFace { verts, .. } => s.verts = verts.clone(),
        Op::SplitEdge { half, .. } => s.halfs = vec![*half],
        Op::ExtrudeFaces { faces, .. } => s.faces = faces.clone(),
        Op::ExtrudeEdges { edges, .. } => s.halfs = edges.clone(),
        Op::InsetFaces {
            faces, individual, ..
        } => {
            s.faces = faces.clone();
            s.shape = vec![u64::from(*individual)];
        }
        Op::BevelEdges {
            edges, segments, ..
        } => {
            s.halfs = edges.clone();
            s.shape = vec![u64::from(*segments)];
        }
        Op::Mirror { axis, .. } => {
            s.shape = vec![match axis {
                crate::model::MirrorAxis::X => 0,
                crate::model::MirrorAxis::Y => 1,
                crate::model::MirrorAxis::Z => 2,
            }]
        }

        // `Never` ops: everything about them is structure, so nothing needs
        // recording — `op_amendable` refuses them before this is consulted.
        Op::RemoveVertex { .. }
        | Op::RemoveFace { .. }
        | Op::CollapseEdge { .. }
        | Op::SplitFace { .. }
        | Op::WeldVerts { .. }
        | Op::LoopCut { .. }
        | Op::Knife { .. }
        | Op::MergeVerts { .. }
        | Op::SubdivideFaces { .. }
        | Op::FlipFaces { .. }
        | Op::DissolveEdges { .. }
        | Op::BridgeLoops { .. } => {}
    }
    s
}

/// Gate 1: may `new` replace `old` at all?
pub fn amend_shape_ok(old: &Op, new: &Op) -> Result<(), AmendError> {
    if std::mem::discriminant(old) != std::mem::discriminant(new) {
        return Err(AmendError::DifferentKind);
    }
    match op_amendable(old) {
        Amendability::Never => Err(AmendError::NotAmendable {
            kind: op_kind(old),
            why: why_never(old),
        }),
        Amendability::Free => Ok(()),
        Amendability::Structural => {
            if op_structure(old) == op_structure(new) {
                Ok(())
            } else {
                Err(AmendError::StructureChanged)
            }
        }
    }
}

/// Gate 2's evidence: **which elements exist and how they are connected**, with
/// no geometry and no attributes in it.
///
/// Positions, UVs, normals, weights, material slots — all deliberately absent.
/// An amendment's whole purpose is to change them, and a downstream op that
/// depended on one (a `SetFaceSlot` naming a slot the amended
/// `AddMaterialSlots` no longer creates) **refuses**, which is a better answer
/// than this: [`AmendError::TailRefused`] names the op and the reason, where
/// [`AmendError::IdAllocationChanged`] could only say "something moved".
///
/// What must not move is the id allocation and the connectivity, because that is
/// what the ops after it *name* — and naming is the failure with no symptom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    verts: Vec<VertId>,
    /// `(half, origin, twin, next, face)` for every live half-edge.
    halfs: Vec<(
        HalfId,
        Option<VertId>,
        Option<HalfId>,
        Option<HalfId>,
        Option<FaceId>,
    )>,
    /// `(face, its first half-edge)` for every live face.
    faces: Vec<(FaceId, Option<HalfId>)>,
}

/// The mesh's structure, with its geometry removed. See [`Topology`].
pub fn topology(mesh: &Mesh) -> Topology {
    Topology {
        verts: mesh.vert_ids().collect(),
        halfs: mesh
            .half_ids()
            .map(|h| {
                (
                    h,
                    mesh.origin(h),
                    mesh.twin(h),
                    mesh.next(h),
                    mesh.face_of(h).flatten(),
                )
            })
            .collect(),
        faces: mesh.face_ids().map(|f| (f, mesh.face_half(f))).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::cube;
    use crate::journal::MeshSession;
    use crate::model::MirrorAxis;

    /// The two classifiers must agree, in the direction that is actually a
    /// theorem: **`Free` ⟹ `op_preserves_ids`**. An op declared freely
    /// re-parameterizable that could renumber the mesh is the defect this whole
    /// module exists to prevent.
    ///
    /// The converse is *not* a theorem, and the exception is named rather than
    /// asserted around: `Op::ClearSkin` preserves ids and is `Never`, because it
    /// has no parameters at all — offering a UI slider for it would be offering
    /// a slider for nothing. Any *other* id-preserving op that is not `Free` is
    /// a classification somebody made too conservative, and this fails.
    #[test]
    fn free_implies_id_preserving_and_the_one_exception_is_named() {
        for op in crate::journal::tests_support::one_of_every_op() {
            let preserves = crate::select::op_preserves_ids(&op);
            let free = op_amendable(&op) == Amendability::Free;
            if free {
                assert!(
                    preserves,
                    "{} is declared freely amendable but renumbers",
                    op_kind(&op)
                );
            }
            if preserves && !free {
                assert!(
                    matches!(op, Op::ClearSkin),
                    "{} preserves ids but is not amendable — is that deliberate?",
                    op_kind(&op)
                );
            }
        }
    }

    /// The two scalar accessors answer for exactly the same ops, and the setter
    /// really sets what the getter reads. A drift between them is a slider that
    /// appears and does nothing.
    #[test]
    fn the_scalar_getter_and_setter_agree_on_which_ops_have_one() {
        for op in crate::journal::tests_support::one_of_every_op() {
            let has = op_scalar(&op).is_some();
            assert_eq!(
                has,
                with_scalar(&op, 1.0).is_some(),
                "{} disagrees about having a scalar",
                op_kind(&op)
            );
            if has {
                let changed = with_scalar(&op, 7.5).expect("it has one");
                assert_eq!(op_scalar(&changed).map(|(v, _)| v), Some(7.5));
                // …and the STRUCTURE is untouched, which is what makes the
                // slider legal under gate 1.
                assert!(
                    amend_shape_ok(&op, &changed).is_ok(),
                    "{} 's slider changes its structure",
                    op_kind(&op)
                );
            }
        }
        // Not vacuous: the ops an author actually reaches for really do have one.
        assert!(op_scalar(&Op::ExtrudeFaces {
            faces: vec![],
            distance: 0.4
        })
        .is_some());
    }

    #[test]
    fn every_op_has_a_kind_name_and_they_are_distinct_enough_to_be_useful() {
        let mut seen = std::collections::BTreeSet::new();
        for op in crate::journal::tests_support::one_of_every_op() {
            let k = op_kind(&op);
            assert!(!k.is_empty());
            seen.insert(k);
        }
        // 36 variants, and a couple deliberately share a family name.
        assert!(seen.len() >= 30, "only {} distinct names", seen.len());
    }

    #[test]
    fn a_different_kind_of_op_is_never_an_amendment() {
        let a = Op::ExtrudeFaces {
            faces: vec![FaceId(0)],
            distance: 1.0,
        };
        let b = Op::InsetFaces {
            faces: vec![FaceId(0)],
            amount: 1.0,
            individual: false,
        };
        assert_eq!(amend_shape_ok(&a, &b), Err(AmendError::DifferentKind));
    }

    #[test]
    fn a_structural_field_may_not_move_but_a_parameter_may() {
        let a = Op::ExtrudeFaces {
            faces: vec![FaceId(0)],
            distance: 0.4,
        };
        assert!(amend_shape_ok(
            &a,
            &Op::ExtrudeFaces {
                faces: vec![FaceId(0)],
                distance: 0.6,
            }
        )
        .is_ok());
        assert_eq!(
            amend_shape_ok(
                &a,
                &Op::ExtrudeFaces {
                    faces: vec![FaceId(1)],
                    distance: 0.4,
                }
            ),
            Err(AmendError::StructureChanged)
        );
        // …and a bevel's SEGMENT count is structure, while its amount is not.
        let b = Op::BevelEdges {
            edges: vec![HalfId(0)],
            amount: 0.1,
            segments: 2,
        };
        assert!(amend_shape_ok(
            &b,
            &Op::BevelEdges {
                edges: vec![HalfId(0)],
                amount: 0.2,
                segments: 2,
            }
        )
        .is_ok());
        assert_eq!(
            amend_shape_ok(
                &b,
                &Op::BevelEdges {
                    edges: vec![HalfId(0)],
                    amount: 0.1,
                    segments: 3,
                }
            ),
            Err(AmendError::StructureChanged)
        );
    }

    #[test]
    fn a_never_op_says_why() {
        let a = Op::LoopCut {
            half: HalfId(0),
            cuts: 1,
        };
        let b = Op::LoopCut {
            half: HalfId(0),
            cuts: 2,
        };
        let err = amend_shape_ok(&a, &b).expect_err("a loop cut is not amendable");
        assert!(
            matches!(&err, AmendError::NotAmendable { kind, why } if *kind == "loop cut" && why.contains("how many"))
        );
    }

    /// Topology is blind to geometry — which is exactly what makes it the right
    /// evidence for gate 2.
    #[test]
    fn topology_ignores_geometry_and_sees_structure() {
        let mut a = cube(2.0);
        let b = a.clone();
        let all: Vec<VertId> = a.vert_ids().collect();
        crate::ops::apply(
            &mut a,
            &Op::TranslateVerts {
                verts: all,
                delta: [5.0, 0.0, 0.0],
            },
        )
        .expect("translates");
        assert_eq!(
            topology(&a),
            topology(&b),
            "moving everything is not structure"
        );

        let mut c = b.clone();
        let one: Vec<FaceId> = c.face_ids().take(1).collect();
        crate::ops::apply(&mut c, &Op::SubdivideFaces { faces: one }).expect("subdivides");
        assert_ne!(topology(&c), topology(&b), "a subdivide IS structure");
    }

    /// The mirror case the two-gate design exists for: the classifier lets the
    /// coordinate move, and the verification refuses the move that renumbers.
    #[test]
    fn moving_a_mirror_plane_off_the_seam_is_caught_by_the_second_gate() {
        use crate::build::plane;
        let seam = |coord: f64| Op::Mirror {
            axis: MirrorAxis::X,
            coord,
        };
        // A plane spans x ∈ [−1, 1], so a mirror at x = 1 WELDS the shared edge
        // (6 vertices, 2 faces) and one at x = 5 does not (8 vertices, 2 faces).
        // Same op, same axis, different id allocation — which is precisely the
        // case no static classifier can decide.
        let mut welded = MeshSession::new(plane(2.0));
        welded.apply(seam(1.0)).expect("a mirror on the edge welds");
        let mut apart = MeshSession::new(plane(2.0));
        apart
            .apply(seam(5.0))
            .expect("a mirror away from it does not");
        assert_ne!(
            topology(welded.mesh()),
            topology(apart.mesh()),
            "the fixture does not exercise the hazard"
        );

        // Gate 1 permits the change — the axis is unchanged.
        assert!(amend_shape_ok(&welded.ops()[0], &seam(5.0)).is_ok());

        // With nothing after it there is nothing to renumber, so the amendment
        // lands — and lands on exactly the mesh the parameter would have made
        // from the start. (This is Blender's F9, as a special case.)
        let mut lone = MeshSession::new(plane(2.0));
        lone.apply(seam(1.0)).expect("mirrors");
        lone.amend(0, seam(5.0))
            .expect("with no tail there is nothing to break");
        assert_eq!(lone.mesh().encoded(), apart.mesh().encoded());

        // Put an op after it that names an id, and the verification fires.
        let mut s = welded;
        let f = s.mesh().face_ids().next().expect("a face");
        s.apply(Op::ExtrudeFaces {
            faces: vec![f],
            distance: 0.5,
        })
        .expect("extrudes");
        let before = s.mesh().encoded();
        let err = s
            .amend(0, seam(5.0))
            .expect_err("the tail would land on different geometry");
        assert!(
            matches!(err, AmendError::IdAllocationChanged { index: 1 }),
            "{err:?}"
        );
        assert_eq!(s.mesh().encoded(), before, "a refused amendment is inert");
    }

    /// A deterministic 40-op session whose **28th** op is the extrude under
    /// test — twelve ops from the end, with three structural ops among them so
    /// the tail genuinely re-derives rather than merely re-translating.
    ///
    /// Ids are resolved from the mesh as it stands at each step, which is the
    /// only way the fixture is honest: if the amendment renumbered anything, a
    /// later step would name something different and the two runs would diverge.
    fn forty_op_session(distance: f64) -> MeshSession {
        let mut s = MeshSession::new(cube(2.0));
        for i in 0..28 {
            let op = if i % 2 == 0 {
                let v = s.mesh().vert_ids().nth(i % 8).expect("a vertex");
                Op::TranslateVerts {
                    verts: vec![v],
                    delta: [0.001 * i as f64, 0.002, 0.0],
                }
            } else {
                let h = s.mesh().half_ids().nth(i % 24).expect("a half-edge");
                Op::SetEdgeSharp {
                    half: h,
                    sharp: i % 4 == 1,
                }
            };
            s.apply(op).expect("the warm-up applies");
        }
        assert_eq!(s.cursor(), 28);
        // The op under test. Picked by GEOMETRY (the face pointing most at +Y)
        // rather than by index, so it is the same face in both runs whatever
        // the arena did.
        let top = s
            .mesh()
            .face_ids()
            .max_by_key(|&f| {
                let vs = s.mesh().face_verts(f).unwrap_or_default();
                let y: f64 = vs
                    .iter()
                    .filter_map(|&v| s.mesh().position(v))
                    .map(|p| p.y)
                    .sum();
                (y * 1e6) as i64
            })
            .expect("a face");
        let out = s
            .apply(Op::ExtrudeFaces {
                faces: vec![top],
                distance,
            })
            .expect("the extrude applies");
        // …and eleven more, three of them structural. The cap comes from the
        // extrude's own `OpOutcome`, which is the only sound way to name it:
        // a structural op renumbers, so "the last face id" is a different face
        // in the two runs the moment anything before it differs.
        let cap = out.faces[0];
        let out = s
            .apply(Op::InsetFaces {
                faces: vec![cap],
                amount: 0.15,
                individual: false,
            })
            .expect("inset");
        let inner = out.faces[0];
        s.apply(Op::SubdivideFaces { faces: vec![inner] })
            .expect("subdivide");
        let e = s.mesh().half_ids().nth(3).expect("a half-edge");
        s.apply(Op::BevelEdges {
            edges: vec![e],
            amount: 0.03,
            segments: 2,
        })
        .expect("bevel");
        for i in 0..8 {
            let v = s.mesh().vert_ids().nth(i * 3).expect("a vertex");
            s.apply(Op::TranslateVerts {
                verts: vec![v],
                delta: [0.0, 0.001 * i as f64, 0.004],
            })
            .expect("a nudge");
        }
        assert_eq!(s.cursor(), 40, "the fixture is a forty-op session");
        s
    }

    /// **The wave's headline claim, as bytes.**
    ///
    /// Amend the extrude twelve ops from the end of a forty-op session, and the
    /// result is byte-identical to the same session authored with the new
    /// parameter from the start. Blender's redo panel re-parameterizes the
    /// operation you just did; this re-parameterizes one that has twenty-eight
    /// edits in front of it and eleven behind it, and the eleven re-derive.
    #[test]
    fn amending_twelve_steps_back_equals_authoring_it_that_way() {
        let mut edited = forty_op_session(0.4);
        let authored = forty_op_session(0.6);
        assert_ne!(
            edited.mesh().encoded(),
            authored.mesh().encoded(),
            "the two parameters must actually produce different meshes"
        );

        let top = match &edited.ops()[28] {
            Op::ExtrudeFaces { faces, distance } => {
                assert_eq!(*distance, 0.4);
                faces.clone()
            }
            other => panic!("op 28 is not the extrude: {other:?}"),
        };
        edited
            .amend(
                28,
                Op::ExtrudeFaces {
                    faces: top,
                    distance: 0.6,
                },
            )
            .expect("re-parameterizing an extrude twelve steps back");

        assert_eq!(
            edited.mesh().encoded(),
            authored.mesh().encoded(),
            "the amended session is not the mesh the parameter would have made"
        );
        assert_eq!(edited.ops(), authored.ops(), "…and neither is its history");
        assert_eq!(edited.cursor(), 40, "the cursor does not travel");
        assert_eq!(crate::validate(edited.mesh()), Ok(()));
    }

    /// The amendment survives a save/restore round trip — it is an edit to the
    /// journal, not a state beside it.
    #[test]
    fn an_amended_session_persists_as_the_amended_history() {
        let mut s = forty_op_session(0.4);
        let faces = match &s.ops()[28] {
            Op::ExtrudeFaces { faces, .. } => faces.clone(),
            other => panic!("{other:?}"),
        };
        s.amend(
            28,
            Op::ExtrudeFaces {
                faces,
                distance: 0.55,
            },
        )
        .expect("amends");
        let back = MeshSession::restore(s.save()).expect("restores");
        assert_eq!(back.mesh().encoded(), s.mesh().encoded());
        assert_eq!(back.ops(), s.ops());
    }

    /// Undo still walks the **amended** history, and the checkpoints ahead of
    /// the amendment were dropped rather than left describing a mesh that no
    /// longer exists.
    #[test]
    fn undo_after_an_amendment_walks_the_new_history() {
        let mut s = forty_op_session(0.4);
        let faces = match &s.ops()[28] {
            Op::ExtrudeFaces { faces, .. } => faces.clone(),
            other => panic!("{other:?}"),
        };
        s.amend(
            28,
            Op::ExtrudeFaces {
                faces,
                distance: 0.6,
            },
        )
        .expect("amends");
        assert!(
            s.checkpoint_keys().iter().all(|&k| k <= 28),
            "a checkpoint past the amendment describes a mesh that no longer \
             exists: {:?}",
            s.checkpoint_keys()
        );
        for _ in 0..12 {
            assert!(s.undo());
        }
        assert_eq!(s.cursor(), 28);
        // Replaying the amended prefix by hand must land on the same mesh —
        // which is the statement that undo did not walk the OLD history.
        let want = MeshSession::replay(s.base(), &s.ops()[..28]).expect("the prefix replays");
        assert_eq!(s.mesh().encoded(), want.encoded());
        // …and redoing forward lands back on the amended result.
        for _ in 0..12 {
            assert!(s.redo());
        }
        assert_eq!(s.mesh().encoded(), forty_op_session(0.6).mesh().encoded());
    }

    #[test]
    fn amending_past_the_cursor_is_refused() {
        let mut s = MeshSession::new(cube(2.0));
        let v = s.mesh().vert_ids().next().expect("a vertex");
        s.apply(Op::TranslateVerts {
            verts: vec![v],
            delta: [0.1, 0.0, 0.0],
        })
        .expect("translates");
        s.undo();
        let err = s
            .amend(
                0,
                Op::TranslateVerts {
                    verts: vec![v],
                    delta: [0.2, 0.0, 0.0],
                },
            )
            .expect_err("op 0 is in the redo tail now");
        assert!(
            matches!(err, AmendError::OutOfRange { index: 0, .. }),
            "{err:?}"
        );
    }

    /// An amendment that makes a later op impossible is refused **by name**, and
    /// the session is byte-identical afterwards.
    #[test]
    fn an_amendment_that_breaks_a_later_op_is_refused_inertly() {
        let mut s = MeshSession::new(cube(2.0));
        s.apply(Op::AddMaterialSlots {
            names: vec!["A".into(), "B".into()],
        })
        .expect("two slots");
        let f = s.mesh().face_ids().next().expect("a face");
        s.apply(Op::SetFaceSlot {
            face: f,
            slot: Some(1),
        })
        .expect("assigns the second slot");
        let before = s.mesh().encoded();
        let err = s
            .amend(
                0,
                Op::AddMaterialSlots {
                    names: vec!["A".into()],
                },
            )
            .expect_err("slot 1 would not exist any more");
        assert!(
            matches!(err, AmendError::TailRefused { index: 1, .. }),
            "{err:?}"
        );
        assert_eq!(s.mesh().encoded(), before);
        assert_eq!(s.ops().len(), 2);
    }
}
