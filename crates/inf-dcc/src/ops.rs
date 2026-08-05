//! The **core** op set — the substrate P23.4's extrude / inset / bevel /
//! loop-cut are built out of, and nothing more.
//!
//! Three rules hold for every op here, and they are what P23.4 gets to assume:
//!
//! 1. **A refusal is a value.** Preconditions come back as a typed [`OpError`],
//!    never a panic and never a partial edit. Structural ops run inside
//!    `Mesh::transact`, so a refused op leaves the mesh **byte-identical** —
//!    property-tested, because an op that half-applies before noticing it cannot
//!    is worse than one that never ran.
//! 2. **The mesh stays valid.** Every op leaves [`validate()`](crate::validate::validate)
//!    green. Ops do *not* call it (that would make the property test assert a
//!    call it just made — see the note at the top of [the `validate` module](mod@crate::validate)); they
//!    enforce local preconditions and a local manifold test, and the property
//!    test audits them independently.
//! 3. **Every op is a journal entry.** [`Op`] is the serialized form; applying it
//!    through `MeshSession` (the op journal) records it, and replaying the
//!    record reproduces the mesh byte for byte. Nothing mutates a mesh outside an
//!    `Op` — that is what makes the journal a complete description of a session.
//!
//! # How the structural ops are built
//!
//! `SplitEdge`, `SplitFace`, `CollapseEdge` and `WeldVerts` all work the same
//! way: **capture the local patch of faces, remove it, rebuild it**. Link surgery
//! written per-op is where half-edge kernels go wrong — a forgotten `prev` is
//! invisible until a traversal walks off the surface three ops later — whereas
//! `remove_face_raw` + `add_face_raw` + `finish_patch` is one path that every op
//! shares and every property test exercises.
//!
//! The cost is that a rebuilt patch's half-edge and face **ids change**. That is
//! sound (ids are arena slots, and the id-reuse rule in [`crate::topo`] already
//! scopes them to one journal generation) and it is why the ops return an
//! [`OpOutcome`] naming what they made.
//!
//! # `meshopt` is not here, and never will be
//!
//! No op calls it, no op's result depends on it, and there is no implicit cleanup
//! pass. The P18 law — meshopt is not cross-platform — makes it poison for a
//! replay structure: one op routed through it and "replaying this journal gives
//! the same mesh" becomes true only on the machine that recorded it. Optimization
//! happens at export, once, behind a flag that says so
//! (`ExportOptions::optimize`, on the writer).

use std::collections::BTreeSet;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::topo::{CornerData, FaceId, FacePatch, HalfId, Mesh, VertId};

/// Why an op refused. Every variant names the elements involved, so a refusal is
/// something a UI can explain rather than something it has to translate.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OpError {
    #[error("no such vertex: {0}")]
    NoSuchVert(VertId),
    #[error("no such half-edge: {0}")]
    NoSuchHalf(HalfId),
    #[error("no such face: {0}")]
    NoSuchFace(FaceId),
    #[error("a face needs at least 3 vertices, got {verts}")]
    FaceTooSmall { verts: usize },
    #[error("{verts} vertices but {corners} corner records")]
    CornerCountMismatch { verts: usize, corners: usize },
    #[error("vertex {0} appears twice in the same face loop")]
    RepeatedVertex(VertId),
    #[error("the directed edge {from}→{to} already carries a face")]
    EdgeAlreadyFaced { from: VertId, to: VertId },
    #[error("vertex {0}'s one-ring would not be a single fan (a bowtie)")]
    NonManifoldVertex(VertId),
    #[error("vertex {vert} still has {incident} incident half-edges")]
    VertexNotIsolated { vert: VertId, incident: usize },
    #[error("the split parameter {t} is not strictly between 0 and 1")]
    SplitParameterOutOfRange { t: f64 },
    #[error("half-edge {0} is a boundary, so it has no corner attributes")]
    BoundaryCorner(HalfId),
    #[error("half-edges {from} and {to} are not corners of the same face")]
    CornersNotInSameFace { from: HalfId, to: HalfId },
    #[error("half-edges {from} and {to} are already adjacent in their face loop")]
    CornersAdjacent { from: HalfId, to: HalfId },
    #[error("{0} was given twice")]
    SameCorner(HalfId),
    #[error("{0} was given twice")]
    SameVertex(VertId),
    /// The link condition failed: `a` and `b` share neighbours the collapse
    /// cannot account for, so collapsing would fold the surface onto itself.
    #[error("edge {half} fails the link condition (unexpected shared neighbours)")]
    NotCollapsible { half: HalfId, shared: Vec<VertId> },
    #[error("material slot {slot} is out of range ({slots} named slots)")]
    UnknownSlot { slot: u32, slots: usize },
}

/// What an op created. Ids of rebuilt elements change (see the module docs), so
/// a caller that wants to keep working on what it just made reads them here.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpOutcome {
    pub verts: Vec<VertId>,
    pub faces: Vec<FaceId>,
    pub halfs: Vec<HalfId>,
}

/// One journalled mesh operation.
///
/// The enum is **externally tagged** (serde's default). `#[serde(tag = ...)]` is
/// not an option: bincode cannot encode internally-tagged enums (the P4 finding),
/// and this type has to survive both codecs.
///
/// There is deliberately **no `AddEdge`**: a wire edge has no loop to live in
/// (see [`crate::topo`]). Edges are born from `AddFace`, `SplitEdge` and
/// `SplitFace`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// Create an isolated vertex — the staging state a face is then built from.
    AddVertex { position: [f64; 3] },
    /// Remove an isolated vertex. Refuses while anything is still attached.
    RemoveVertex { vert: VertId },
    /// Add a face over an existing vertex loop.
    AddFace {
        verts: Vec<VertId>,
        corners: Vec<CornerData>,
        slot: Option<u32>,
    },
    /// Remove a face. Its vertices survive (possibly isolated); edges that end up
    /// face-less on both sides are freed.
    RemoveFace { face: FaceId },
    /// Insert a vertex at `lerp(origin, dest, t)` on an edge, splitting the ≤ 2
    /// faces beside it. Corner attributes at the new vertex are interpolated.
    SplitEdge { half: HalfId, t: f64 },
    /// Collapse an edge onto its **origin** vertex, which keeps its position.
    /// (Moving it afterwards is a separate, separately-undoable `TranslateVerts`
    /// — an op that silently did both would make "merge at centre" and "merge to
    /// first" the same journal entry.)
    CollapseEdge { half: HalfId },
    /// Cut a face in two along a new edge from `origin(from)` to `origin(to)`.
    SplitFace { from: HalfId, to: HalfId },
    /// Merge `merge` into `keep`, rebuilding every face that touched it. The
    /// general form of `CollapseEdge`, without requiring an edge between them.
    WeldVerts { keep: VertId, merge: VertId },
    /// Move vertices by a delta, in metres.
    TranslateVerts { verts: Vec<VertId>, delta: [f64; 3] },
    /// Set a face corner's UV.
    SetCornerUv { half: HalfId, uv: [f64; 2] },
    /// Set (`Some`) or clear (`None`) a face corner's **authored** normal.
    /// Clearing hands the corner back to export's smooth-fan rule.
    SetCornerNormal {
        half: HalfId,
        normal: Option<[f64; 3]>,
    },
    /// Mark an undirected edge sharp or smooth. Written through both halves.
    SetEdgeSharp { half: HalfId, sharp: bool },
    /// Retarget a face's material slot.
    SetFaceSlot { face: FaceId, slot: Option<u32> },
}

/// Apply one op. See the module docs for the three rules this upholds.
pub fn apply(mesh: &mut Mesh, op: &Op) -> Result<OpOutcome, OpError> {
    match op {
        Op::AddVertex { position } => {
            let v = mesh.alloc_vert(*position);
            Ok(OpOutcome {
                verts: vec![v],
                ..Default::default()
            })
        }

        Op::RemoveVertex { vert } => {
            let out = mesh
                .vert_outgoing(*vert)
                .ok_or(OpError::NoSuchVert(*vert))?;
            if !out.is_empty() {
                return Err(OpError::VertexNotIsolated {
                    vert: *vert,
                    incident: out.len(),
                });
            }
            mesh.free_vert(*vert);
            Ok(OpOutcome::default())
        }

        Op::AddFace {
            verts,
            corners,
            slot,
        } => {
            check_slot(mesh, *slot)?;
            let touched: BTreeSet<VertId> = verts.iter().copied().collect();
            let f = mesh.transact(|m| {
                let f = m.add_face_raw(verts, corners, *slot)?;
                m.finish_patch(&touched)?;
                Ok(f)
            })?;
            Ok(OpOutcome {
                faces: vec![f],
                ..Default::default()
            })
        }

        Op::RemoveFace { face } => {
            if !mesh.has_face(*face) {
                return Err(OpError::NoSuchFace(*face));
            }
            mesh.transact(|m| {
                let patch = m.remove_face_raw(*face);
                let touched: BTreeSet<VertId> = patch.verts.iter().copied().collect();
                m.finish_patch(&touched)
            })?;
            Ok(OpOutcome::default())
        }

        Op::SplitEdge { half, t } => split_edge(mesh, *half, *t),
        Op::CollapseEdge { half } => collapse_edge(mesh, *half),
        Op::SplitFace { from, to } => split_face(mesh, *from, *to),
        Op::WeldVerts { keep, merge } => {
            if !mesh.has_vert(*keep) {
                return Err(OpError::NoSuchVert(*keep));
            }
            if !mesh.has_vert(*merge) {
                return Err(OpError::NoSuchVert(*merge));
            }
            if keep == merge {
                return Err(OpError::SameVertex(*keep));
            }
            let faces = mesh.transact(|m| weld_impl(m, *keep, *merge))?;
            Ok(OpOutcome {
                faces,
                ..Default::default()
            })
        }

        Op::TranslateVerts { verts, delta } => {
            // Every id is checked before the first write, so a refusal is inert
            // without needing a transaction.
            for v in verts {
                if !mesh.has_vert(*v) {
                    return Err(OpError::NoSuchVert(*v));
                }
            }
            for v in verts {
                let p = mesh.vert_mut(*v).position;
                mesh.vert_mut(*v).position = [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]];
            }
            Ok(OpOutcome::default())
        }

        Op::SetCornerUv { half, uv } => {
            corner_of(mesh, *half)?;
            mesh.half_mut(*half).uv = *uv;
            Ok(OpOutcome::default())
        }

        Op::SetCornerNormal { half, normal } => {
            corner_of(mesh, *half)?;
            mesh.half_mut(*half).normal = *normal;
            Ok(OpOutcome::default())
        }

        Op::SetEdgeSharp { half, sharp } => {
            if !mesh.has_half(*half) {
                return Err(OpError::NoSuchHalf(*half));
            }
            mesh.set_sharp_pair(*half, *sharp);
            Ok(OpOutcome::default())
        }

        Op::SetFaceSlot { face, slot } => {
            if !mesh.has_face(*face) {
                return Err(OpError::NoSuchFace(*face));
            }
            check_slot(mesh, *slot)?;
            mesh.face_mut(*face).material_slot = *slot;
            Ok(OpOutcome::default())
        }
    }
}

fn check_slot(mesh: &Mesh, slot: Option<u32>) -> Result<(), OpError> {
    if let Some(s) = slot {
        let slots = mesh.material_slots().len();
        if slots > 0 && s as usize >= slots {
            return Err(OpError::UnknownSlot { slot: s, slots });
        }
    }
    Ok(())
}

/// The half-edge must be a live **face** corner.
fn corner_of(mesh: &Mesh, h: HalfId) -> Result<FaceId, OpError> {
    match mesh.face_of(h) {
        None => Err(OpError::NoSuchHalf(h)),
        Some(None) => Err(OpError::BoundaryCorner(h)),
        Some(Some(f)) => Ok(f),
    }
}

fn lerp_corner(a: &CornerData, b: &CornerData, t: f64) -> CornerData {
    let uv = [
        a.uv[0] + (b.uv[0] - a.uv[0]) * t,
        a.uv[1] + (b.uv[1] - a.uv[1]) * t,
    ];
    // Two authored normals interpolate; anything else hands the new corner to the
    // smooth-fan rule rather than inventing an authored value from one side.
    let normal = match (a.normal, b.normal) {
        (Some(x), Some(y)) => {
            let v = DVec3::from_array(x).lerp(DVec3::from_array(y), t);
            let len = v.length();
            if len > 1e-12 {
                Some((v / len).to_array())
            } else {
                None
            }
        }
        _ => None,
    };
    CornerData { uv, normal }
}

fn split_edge(mesh: &mut Mesh, half: HalfId, t: f64) -> Result<OpOutcome, OpError> {
    if !mesh.has_half(half) {
        return Err(OpError::NoSuchHalf(half));
    }
    if !(t > 0.0 && t < 1.0) {
        return Err(OpError::SplitParameterOutOfRange { t });
    }
    let twin = mesh.twin(half).expect("live half-edge id");
    let a = mesh.origin(half).expect("live half-edge id");
    let b = mesh.origin(twin).expect("live half-edge id");
    let pa = mesh.position(a).expect("live vertex id");
    let pb = mesh.position(b).expect("live vertex id");
    let mid = (pa + (pb - pa) * t).to_array();

    let mut incident: Vec<FaceId> = Vec::new();
    for h in [half, twin] {
        if let Some(Some(f)) = mesh.face_of(h) {
            incident.push(f);
        }
    }

    let out = mesh.transact(|m| {
        let sharp = m.capture_sharp(&incident);
        let patches: Vec<FacePatch> = incident.iter().map(|&f| m.remove_face_raw(f)).collect();
        let mid_vert = m.alloc_vert(mid);
        let mut touched: BTreeSet<VertId> = [a, b, mid_vert].into_iter().collect();
        for patch in &patches {
            touched.extend(patch.verts.iter().copied());
        }
        let mut rebuilt = Vec::with_capacity(patches.len());
        for patch in &patches {
            let n = patch.verts.len();
            let at = (0..n)
                .find(|&i| {
                    let (x, y) = (patch.verts[i], patch.verts[(i + 1) % n]);
                    (x == a && y == b) || (x == b && y == a)
                })
                .expect("the edge's own face contains the edge");
            let forward = patch.verts[at] == a;
            let corner = lerp_corner(
                &patch.corners[at],
                &patch.corners[(at + 1) % n],
                if forward { t } else { 1.0 - t },
            );
            let mut verts = patch.verts.clone();
            let mut corners = patch.corners.clone();
            verts.insert(at + 1, mid_vert);
            corners.insert(at + 1, corner);
            touched.extend(verts.iter().copied());
            rebuilt.push(m.add_face_raw(&verts, &corners, patch.slot)?);
        }
        m.apply_sharp(&sharp);
        // The two halves of the split edge inherit the original's sharpness.
        let was_sharp = sharp
            .iter()
            .any(|&(x, y, s)| s && ((x == a && y == b) || (x == b && y == a)));
        if was_sharp {
            for (x, y) in [(a, mid_vert), (mid_vert, b)] {
                if let Some(h) = m.find_half(x, y) {
                    m.set_sharp_pair(h, true);
                }
            }
        }
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: vec![mid_vert],
            faces: rebuilt,
            ..Default::default()
        })
    })?;
    Ok(out)
}

fn split_face(mesh: &mut Mesh, from: HalfId, to: HalfId) -> Result<OpOutcome, OpError> {
    if from == to {
        return Err(OpError::SameCorner(from));
    }
    let fa = corner_of(mesh, from)?;
    let fb = corner_of(mesh, to)?;
    if fa != fb {
        return Err(OpError::CornersNotInSameFace { from, to });
    }
    if mesh.next(from) == Some(to) || mesh.next(to) == Some(from) {
        return Err(OpError::CornersAdjacent { from, to });
    }
    let halfs = mesh.face_loop(fa).expect("live face id");
    let i = halfs.iter().position(|&h| h == from).expect("in its face");
    let j = halfs.iter().position(|&h| h == to).expect("in its face");
    let (i, j) = if i < j { (i, j) } else { (j, i) };

    let out = mesh.transact(|m| {
        let sharp = m.capture_sharp(&[fa]);
        let patch = m.remove_face_raw(fa);
        let n = patch.verts.len();
        let mut loops: Vec<(Vec<VertId>, Vec<CornerData>)> = Vec::with_capacity(2);
        // First half: i..=j. Second half: j..n then 0..=i.
        let first: Vec<usize> = (i..=j).collect();
        let second: Vec<usize> = (j..n).chain(0..=i).collect();
        for idx in [first, second] {
            let verts = idx.iter().map(|&k| patch.verts[k]).collect();
            let corners = idx.iter().map(|&k| patch.corners[k]).collect();
            loops.push((verts, corners));
        }
        let touched: BTreeSet<VertId> = patch.verts.iter().copied().collect();
        let mut made = Vec::with_capacity(2);
        for (verts, corners) in &loops {
            made.push(m.add_face_raw(verts, corners, patch.slot)?);
        }
        m.apply_sharp(&sharp);
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            faces: made,
            ..Default::default()
        })
    })?;
    Ok(out)
}

fn collapse_edge(mesh: &mut Mesh, half: HalfId) -> Result<OpOutcome, OpError> {
    if !mesh.has_half(half) {
        return Err(OpError::NoSuchHalf(half));
    }
    let twin = mesh.twin(half).expect("live half-edge id");
    let a = mesh.origin(half).expect("live half-edge id");
    let b = mesh.origin(twin).expect("live half-edge id");

    // The link condition. `a` and `b` may share exactly the apex vertices of the
    // triangles beside the edge — those triangles vanish with the collapse. Any
    // other shared neighbour means the collapse would identify two edges that
    // are not the same edge, folding the surface.
    let neighbours = |v: VertId| -> BTreeSet<VertId> {
        mesh.vert_outgoing(v)
            .expect("live vertex id")
            .iter()
            .filter_map(|&h| mesh.dest(h))
            .collect()
    };
    let shared: BTreeSet<VertId> = neighbours(a)
        .intersection(&neighbours(b))
        .copied()
        .collect();
    let mut expected: BTreeSet<VertId> = BTreeSet::new();
    for h in [half, twin] {
        if let Some(Some(f)) = mesh.face_of(h) {
            let verts = mesh.face_verts(f).expect("live face id");
            if verts.len() == 3 {
                for v in verts {
                    if v != a && v != b {
                        expected.insert(v);
                    }
                }
            }
        }
    }
    if shared != expected {
        return Err(OpError::NotCollapsible {
            half,
            shared: shared.difference(&expected).copied().collect(),
        });
    }

    let faces = mesh.transact(|m| weld_impl(m, a, b))?;
    Ok(OpOutcome {
        faces,
        ..Default::default()
    })
}

/// Merge `merge` into `keep`: rebuild every face that touched `merge` with the
/// substitution applied, dropping loops that lose their surface, then remove the
/// now-isolated vertex.
///
/// Runs inside a `Mesh::transact` — the caller opens it.
fn weld_impl(mesh: &mut Mesh, keep: VertId, merge: VertId) -> Result<Vec<FaceId>, OpError> {
    let incident: Vec<FaceId> = mesh
        .vert_outgoing(merge)
        .ok_or(OpError::NoSuchVert(merge))?
        .iter()
        .filter_map(|&h| match mesh.face_of(h) {
            Some(Some(f)) => Some(f),
            _ => None,
        })
        .collect();
    let sharp: Vec<(VertId, VertId, bool)> = mesh
        .capture_sharp(&incident)
        .into_iter()
        .map(|(x, y, s)| {
            (
                if x == merge { keep } else { x },
                if y == merge { keep } else { y },
                s,
            )
        })
        .collect();

    let patches: Vec<FacePatch> = incident.iter().map(|&f| mesh.remove_face_raw(f)).collect();
    // EVERY vertex of every removed face, not only the ones that end up in a
    // rebuilt loop. A face that loses its surface in the merge is not rebuilt, so
    // its vertices contribute nothing to the loops below — but their edges were
    // still freed, and a boundary half-edge somewhere else still points at one.
    // Relinking only the rebuilt vertices left those `next`/`prev` naming dead
    // slots, which is `DeadLink` and is what the validity property caught.
    let mut touched: BTreeSet<VertId> = [keep].into_iter().collect();
    for patch in &patches {
        touched.extend(patch.verts.iter().copied());
    }
    let mut made = Vec::new();
    for patch in &patches {
        let substituted: Vec<VertId> = patch
            .verts
            .iter()
            .map(|&v| if v == merge { keep } else { v })
            .collect();
        // A run of two identical vertices can only be produced by the
        // substitution, so exactly one of the pair is the merged corner — and it
        // is the one to drop. Dropping "the first of the run" instead is
        // direction-dependent: it keeps the surviving vertex's own attributes on
        // one incident face and the vanishing vertex's on the other, which is how
        // a collapse that should undo a split left one UV at the midpoint.
        let from_merge: Vec<bool> = patch.verts.iter().map(|&v| v == merge).collect();
        let mut keep_idx: Vec<usize> = (0..substituted.len()).collect();
        loop {
            let n = keep_idx.len();
            if n < 2 {
                break;
            }
            let Some(i) =
                (0..n).find(|&i| substituted[keep_idx[i]] == substituted[keep_idx[(i + 1) % n]])
            else {
                break;
            };
            let j = (i + 1) % n;
            let drop_at = if from_merge[keep_idx[j]] { j } else { i };
            keep_idx.remove(drop_at);
        }
        if keep_idx.len() < 3 {
            continue; // the face lost its surface in the merge
        }
        let verts: Vec<VertId> = keep_idx.iter().map(|&k| substituted[k]).collect();
        let corners: Vec<crate::topo::CornerData> =
            keep_idx.iter().map(|&k| patch.corners[k]).collect();
        touched.extend(verts.iter().copied());
        made.push(mesh.add_face_raw(&verts, &corners, patch.slot)?);
    }

    let leftover = mesh.vert_outgoing(merge).map(<[HalfId]>::len).unwrap_or(0);
    if leftover == 0 {
        mesh.free_vert(merge);
    } else {
        return Err(OpError::VertexNotIsolated {
            vert: merge,
            incident: leftover,
        });
    }
    mesh.apply_sharp(&sharp);
    mesh.finish_patch(&touched)?;
    Ok(made)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::validate::validate;

    fn expect_ok(mesh: &mut Mesh, op: Op) -> OpOutcome {
        let out = apply(mesh, &op).unwrap_or_else(|e| panic!("{op:?} refused: {e}"));
        assert_eq!(validate(mesh), Ok(()), "after {op:?}");
        out
    }

    fn refuses(mesh: &mut Mesh, op: Op, want: OpError) {
        let before = mesh.encoded();
        assert_eq!(apply(mesh, &op), Err(want));
        assert_eq!(
            mesh.encoded(),
            before,
            "a refused op must leave the mesh byte-identical"
        );
    }

    #[test]
    fn split_edge_adds_one_vertex_and_keeps_the_surface_closed() {
        let mut m = cube(1.0);
        let (v0, e0, f0) = (m.vert_count(), m.edge_count(), m.face_count());
        let h = m.half_ids().next().unwrap();
        let out = expect_ok(&mut m, Op::SplitEdge { half: h, t: 0.25 });
        assert_eq!(out.verts.len(), 1);
        assert_eq!(m.vert_count(), v0 + 1);
        assert_eq!(m.edge_count(), e0 + 1, "one edge becomes two");
        assert_eq!(m.face_count(), f0, "the two faces are rebuilt, not added");
        // χ is unchanged: a closed genus-0 surface stays one.
        assert_eq!(
            m.vert_count() as i64 - m.edge_count() as i64 + m.face_count() as i64,
            2
        );
    }

    #[test]
    fn split_edge_places_the_vertex_at_the_parameter_it_was_given() {
        let mut m = plane(2.0);
        let h = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .unwrap();
        let a = m.position(m.origin(h).unwrap()).unwrap();
        let b = m.position(m.dest(h).unwrap()).unwrap();
        let out = expect_ok(&mut m, Op::SplitEdge { half: h, t: 0.25 });
        let p = m.position(out.verts[0]).unwrap();
        assert_eq!(p, a + (b - a) * 0.25);
    }

    #[test]
    fn split_edge_refuses_a_parameter_off_the_edge() {
        let mut m = cube(1.0);
        let h = m.half_ids().next().unwrap();
        refuses(
            &mut m,
            Op::SplitEdge { half: h, t: 0.0 },
            OpError::SplitParameterOutOfRange { t: 0.0 },
        );
        refuses(
            &mut m,
            Op::SplitEdge { half: h, t: 1.5 },
            OpError::SplitParameterOutOfRange { t: 1.5 },
        );
        refuses(
            &mut m,
            Op::SplitEdge {
                half: HalfId(9_999),
                t: 0.5,
            },
            OpError::NoSuchHalf(HalfId(9_999)),
        );
    }

    #[test]
    fn split_face_cuts_a_quad_into_two_triangles() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        let halfs = m.face_loop(f).unwrap();
        let out = expect_ok(
            &mut m,
            Op::SplitFace {
                from: halfs[0],
                to: halfs[2],
            },
        );
        assert_eq!(out.faces.len(), 2);
        assert_eq!(m.face_count(), 2);
        assert_eq!(m.vert_count(), 4, "no new vertices");
        assert_eq!(m.edge_count(), 5, "the four sides plus the diagonal");
    }

    #[test]
    fn split_face_refuses_adjacent_corners_and_foreign_ones() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        let halfs = m.face_loop(f).unwrap();
        refuses(
            &mut m,
            Op::SplitFace {
                from: halfs[0],
                to: halfs[1],
            },
            OpError::CornersAdjacent {
                from: halfs[0],
                to: halfs[1],
            },
        );
        let boundary = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(true))
            .unwrap();
        refuses(
            &mut m,
            Op::SplitFace {
                from: halfs[0],
                to: boundary,
            },
            OpError::BoundaryCorner(boundary),
        );
    }

    /// A square pyramid with a triangulated base — five vertices, eight faces,
    /// closed. Its base diagonal is the edge whose collapse the link condition
    /// must refuse (the apex is a third shared neighbour).
    fn pyramid() -> Mesh {
        let mut m = Mesh::new();
        let v: Vec<VertId> = [
            [0.0, 1.0, 0.0],   // 0 apex
            [-1.0, 0.0, -1.0], // 1
            [1.0, 0.0, -1.0],  // 2
            [1.0, 0.0, 1.0],   // 3
            [-1.0, 0.0, 1.0],  // 4
        ]
        .iter()
        .map(|p| {
            apply(&mut m, &Op::AddVertex { position: *p })
                .unwrap()
                .verts[0]
        })
        .collect();
        for tri in [
            [1, 2, 3],
            [1, 3, 4], // base, facing −Y
            [2, 1, 0],
            [3, 2, 0],
            [4, 3, 0],
            [1, 4, 0], // sides
        ] {
            apply(
                &mut m,
                &Op::AddFace {
                    verts: tri.iter().map(|&i| v[i]).collect(),
                    corners: vec![CornerData::default(); 3],
                    slot: None,
                },
            )
            .unwrap();
        }
        assert_eq!(validate(&m), Ok(()));
        m
    }

    #[test]
    fn collapse_edge_undoes_a_split_exactly() {
        // Split a cube edge, then collapse the new vertex back into the one it
        // came from. `CollapseEdge` keeps the ORIGIN, so the half-edge has to
        // point from the survivor into the midpoint.
        let mut m = cube(1.0);
        let before = m.canonical();
        let h = m.half_ids().next().unwrap();
        let split = expect_ok(&mut m, Op::SplitEdge { half: h, t: 0.5 });
        let mid = split.verts[0];
        let into_mid = m.twin(m.vert_outgoing(mid).unwrap()[0]).unwrap();
        expect_ok(&mut m, Op::CollapseEdge { half: into_mid });
        assert_eq!(m.vert_count(), 8);
        assert_eq!(m.face_count(), 6);
        // Not the same *ids* — the patch was rebuilt — but the same mesh.
        assert_eq!(m.canonical(), before);
    }

    #[test]
    fn collapse_refuses_when_the_link_condition_fails() {
        // The pyramid's base diagonal: its endpoints share the apex as well as
        // the two triangle apexes, so collapsing it would fold the solid.
        let mut m = pyramid();
        let verts: Vec<VertId> = m.vert_ids().collect();
        let diagonal = m.find_half(verts[1], verts[3]).expect("the base diagonal");
        let before = m.encoded();
        let err = apply(&mut m, &Op::CollapseEdge { half: diagonal }).unwrap_err();
        match err {
            OpError::NotCollapsible { half, ref shared } => {
                assert_eq!(half, diagonal);
                assert_eq!(shared, &vec![verts[0]], "the apex is the extra neighbour");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(m.encoded(), before, "a refusal is inert");
    }

    #[test]
    fn collapsing_a_tetrahedron_edge_is_allowed_and_stays_valid() {
        // Documented behaviour rather than a hidden one: the link condition is
        // about TOPOLOGY, and a tetrahedron edge collapse leaves a legal — if
        // geometrically degenerate — two-face surface. Refusing it would need a
        // geometric rule, and a geometric rule in a topology kernel is where
        // "why did my merge do nothing" bugs come from.
        let mut m = Mesh::new();
        let v: Vec<VertId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| {
            apply(&mut m, &Op::AddVertex { position: *p })
                .unwrap()
                .verts[0]
        })
        .collect();
        for tri in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]] {
            expect_ok(
                &mut m,
                Op::AddFace {
                    verts: tri.iter().map(|&i| v[i]).collect(),
                    corners: vec![CornerData::default(); 3],
                    slot: None,
                },
            );
        }
        let h = m.find_half(v[0], v[1]).unwrap();
        expect_ok(&mut m, Op::CollapseEdge { half: h });
        assert_eq!(m.vert_count(), 3);
        assert_eq!(m.face_count(), 2);
    }

    #[test]
    fn add_face_refuses_a_second_face_on_the_same_side_of_an_edge() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        let verts = m.face_verts(f).unwrap();
        refuses(
            &mut m,
            Op::AddFace {
                verts: verts.clone(),
                corners: vec![CornerData::default(); 4],
                slot: None,
            },
            OpError::EdgeAlreadyFaced {
                from: verts[0],
                to: verts[1],
            },
        );
    }

    #[test]
    fn add_face_refuses_a_bowtie_and_leaves_nothing_behind() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        let shared = m.face_verts(f).unwrap()[0];
        let a = apply(
            &mut m,
            &Op::AddVertex {
                position: [5.0, 0.0, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        let b = apply(
            &mut m,
            &Op::AddVertex {
                position: [5.0, 0.0, 5.0],
            },
        )
        .unwrap()
        .verts[0];
        refuses(
            &mut m,
            Op::AddFace {
                verts: vec![shared, a, b],
                corners: vec![CornerData::default(); 3],
                slot: None,
            },
            OpError::NonManifoldVertex(shared),
        );
    }

    #[test]
    fn remove_face_leaves_isolated_vertices_that_can_then_be_removed() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        expect_ok(&mut m, Op::RemoveFace { face: f });
        assert_eq!(m.face_count(), 0);
        assert_eq!(m.half_count(), 0, "no wire edges survive a face");
        assert_eq!(m.vert_count(), 4);
        let v = m.vert_ids().next().unwrap();
        expect_ok(&mut m, Op::RemoveVertex { vert: v });
        assert_eq!(m.vert_count(), 3);
    }

    #[test]
    fn remove_vertex_refuses_while_anything_is_attached() {
        let mut m = plane(2.0);
        let v = m.vert_ids().next().unwrap();
        refuses(
            &mut m,
            Op::RemoveVertex { vert: v },
            OpError::VertexNotIsolated {
                vert: v,
                incident: 2,
            },
        );
    }

    #[test]
    fn translate_refuses_the_whole_set_when_one_id_is_dead() {
        let mut m = plane(2.0);
        let good = m.vert_ids().next().unwrap();
        refuses(
            &mut m,
            Op::TranslateVerts {
                verts: vec![good, VertId(9_999)],
                delta: [1.0, 0.0, 0.0],
            },
            OpError::NoSuchVert(VertId(9_999)),
        );
    }

    #[test]
    fn corner_attribute_ops_refuse_a_boundary_half() {
        let mut m = plane(2.0);
        let b = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(true))
            .unwrap();
        refuses(
            &mut m,
            Op::SetCornerUv {
                half: b,
                uv: [0.5, 0.5],
            },
            OpError::BoundaryCorner(b),
        );
        let c = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .unwrap();
        expect_ok(
            &mut m,
            Op::SetCornerUv {
                half: c,
                uv: [0.25, 0.75],
            },
        );
        assert_eq!(m.corner_uv(c), Some([0.25, 0.75]));
    }

    #[test]
    fn sharpness_is_written_through_both_halves() {
        let mut m = plane(2.0);
        let h = m.half_ids().next().unwrap();
        let t = m.twin(h).unwrap();
        expect_ok(
            &mut m,
            Op::SetEdgeSharp {
                half: h,
                sharp: true,
            },
        );
        assert_eq!(m.is_sharp(h), Some(true));
        assert_eq!(m.is_sharp(t), Some(true), "the twin agrees");
    }

    #[test]
    fn welding_two_corners_of_a_quad_makes_a_triangle() {
        let mut m = plane(2.0);
        let verts = m.face_verts(m.face_ids().next().unwrap()).unwrap();
        expect_ok(
            &mut m,
            Op::WeldVerts {
                keep: verts[0],
                merge: verts[1],
            },
        );
        assert_eq!(m.vert_count(), 3);
        assert_eq!(m.face_count(), 1);
        assert_eq!(m.face_verts(m.face_ids().next().unwrap()).unwrap().len(), 3);
    }

    #[test]
    fn an_out_of_range_material_slot_is_refused() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().unwrap();
        m.set_material_slots(vec!["Default".into()]);
        refuses(
            &mut m,
            Op::SetFaceSlot {
                face: f,
                slot: Some(3),
            },
            OpError::UnknownSlot { slot: 3, slots: 1 },
        );
    }
}
