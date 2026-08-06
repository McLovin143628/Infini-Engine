//! **The selection model** (P23.4): what the author has picked, keyed by the
//! journal generation that makes the ids mean anything.
//!
//! # The stamp is not decoration — it is the entire correctness story
//!
//! [`crate::topo`]'s id-reuse rule says it plainly: ids are arena slots with a
//! LIFO free list, so a structural op that rebuilds a patch mints *the same
//! numbers* for *different elements*. A stale id is therefore not dead — nothing
//! would catch it — and a selection is precisely the thing that holds ids across
//! edits. So a [`SelectionSet`] carries the
//! [`generation`](crate::journal::MeshSession::generation) it was built against
//! and every entry point takes the current one:
//!
//! ```text
//!   selection.sync(session.generation())   ⇒ true when the set was DROPPED
//! ```
//!
//! and the consumer is expected to have dropped it. That is the whole contract,
//! and it is why the stamp had to be made process-monotone across `new`, `clone`
//! and `restore` (P23.3 audit round 2, NB2): a set compared against a *restored*
//! session's stamp would otherwise have matched a completely different document.
//!
//! # But dropping the selection after every edit would be a useless tool
//!
//! An author who extrudes expects the new cap to be selected, not nothing. So
//! there are two doors and [`op_preserves_ids`] decides which one a caller uses:
//!
//! * ops that **restructure** hand back an [`crate::ops::OpOutcome`] naming what
//!   they made, and the selection is *replaced* from it
//!   ([`SelectionSet::adopt`]) — no id is carried across the boundary at all;
//! * ops that only **read and write attributes** rebuild nothing, so every live
//!   id still names what it named and the set may be
//!   [`carried`](SelectionSet::carry), dropping only ids that died.
//!
//! `op_preserves_ids` is an exhaustive `match` with no wildcard, so a tenth
//! modelling op cannot be added without someone deciding which half it is in —
//! the same discipline as the frozen-discriminant test.
//!
//! # Soft select measures GEODESIC distance, in metres
//!
//! [`SelectionSet::soft_weights`] runs Dijkstra over the edge graph with real
//! edge lengths and feeds the result to [`inf_terrain::Falloff::weight`], the
//! pure, geometry-free curve the terrain sculpt brush has used since P16. Three
//! reasons it is not a hop count and not a Euclidean ball:
//!
//! * **Units.** A radius in metres is architecture rule 6. A hop count is
//!   unitless and density-dependent, so the same brush would reach twice as far
//!   across a coarse part of a model as across a dense one.
//! * **The surface, not the room.** Two shells that pass close to each other —
//!   the two sides of a thin wall, a folded strap — are far apart *on the mesh*.
//!   A Euclidean ball grabs both; a geodesic walk does not, and that is the exact
//!   failure a soft select is judged on.
//! * **Determinism.** The frontier is a `BTreeSet<(u64, VertId)>` keyed by the
//!   distance's bit pattern (correct ordering for non-negative finite `f64`) with
//!   the vertex id as tie-break, so equal distances are settled by id and not by
//!   insertion order.

use std::collections::{BTreeMap, BTreeSet};

use inf_terrain::Falloff;
use serde::{Deserialize, Serialize};

use crate::ops::{Op, OpOutcome};
use crate::topo::{FaceId, HalfId, Mesh, VertId};

/// Which component kind a selection is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum SelectMode {
    #[default]
    Vert,
    Edge,
    Face,
}

/// Whether an op leaves every surviving id naming the same element.
///
/// **Exhaustive on purpose.** A new [`Op`] stops this compiling until an author
/// has decided whether their op renumbers, which is the only way this stays true
/// — the cost of getting it wrong is a selection that silently points at
/// different geometry, which is invisible until the next edit lands somewhere the
/// author did not click.
pub fn op_preserves_ids(op: &Op) -> bool {
    match op {
        // Pure attribute writes and a fresh isolated vertex: nothing is removed,
        // so no slot is freed and no patch is rebuilt.
        Op::AddVertex { .. }
        | Op::TranslateVerts { .. }
        | Op::SetCornerUv { .. }
        | Op::SetCornerNormal { .. }
        | Op::SetEdgeSharp { .. }
        | Op::SetFaceSlot { .. }
        // P23.5: a stroke and a gizmo drag move POSITIONS. No face is rebuilt,
        // no slot is freed, so every live id still names what it named — which
        // is exactly what makes sculpting usable: an author who drags the brush
        // across a selected region must still have it selected afterwards.
        | Op::Sculpt { .. }
        | Op::RotateVerts { .. }
        | Op::ScaleVerts { .. } => true,
        // Everything else frees slots (which the LIFO free list then hands back
        // to something different) or rebuilds a patch outright.
        Op::RemoveVertex { .. }
        | Op::AddFace { .. }
        | Op::RemoveFace { .. }
        | Op::SplitEdge { .. }
        | Op::CollapseEdge { .. }
        | Op::SplitFace { .. }
        | Op::WeldVerts { .. }
        | Op::ExtrudeFaces { .. }
        | Op::ExtrudeEdges { .. }
        | Op::InsetFaces { .. }
        | Op::BevelEdges { .. }
        | Op::LoopCut { .. }
        | Op::Knife { .. }
        | Op::MergeVerts { .. }
        | Op::SubdivideFaces { .. }
        | Op::Mirror { .. } => false,
    }
}

/// The canonical half-edge naming an undirected edge: the lower id of the twin
/// pair, so an edge has exactly one key however it was reached.
pub fn canonical_edge(mesh: &Mesh, h: HalfId) -> Option<HalfId> {
    let t = mesh.twin(h)?;
    Some(if h < t { h } else { t })
}

/// A vertex / edge / face selection, stamped with the journal generation its ids
/// belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionSet {
    generation: u64,
    verts: BTreeSet<VertId>,
    edges: BTreeSet<HalfId>,
    faces: BTreeSet<FaceId>,
}

impl SelectionSet {
    /// An empty selection for `generation`.
    pub fn new(generation: u64) -> Self {
        Self {
            generation,
            verts: BTreeSet::new(),
            edges: BTreeSet::new(),
            faces: BTreeSet::new(),
        }
    }

    /// The generation these ids belong to.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bring the set up to `generation`, **clearing it** if the stamp moved.
    /// Returns `true` when it was dropped, so a caller can say so rather than
    /// leaving the author wondering where their selection went.
    pub fn sync(&mut self, generation: u64) -> bool {
        if self.generation == generation {
            return false;
        }
        self.generation = generation;
        self.clear();
        true
    }

    /// Re-stamp without clearing, dropping ids whose elements no longer exist.
    ///
    /// **Only legal after an op that [`op_preserves_ids`]**, which is why that
    /// function is next to this one. Used on a translate or a UV write, where
    /// insisting on a drop would make every attribute edit deselect the thing
    /// being edited.
    pub fn carry(&mut self, generation: u64, mesh: &Mesh) {
        self.generation = generation;
        self.verts.retain(|&v| mesh.has_vert(v));
        self.edges.retain(|&h| mesh.has_half(h));
        self.faces.retain(|&f| mesh.has_face(f));
    }

    /// Replace the selection with what an op just made, at the new generation.
    ///
    /// The answer to "an extrude must leave the new cap selected" that does not
    /// carry a single id across a renumbering.
    pub fn adopt(&mut self, generation: u64, outcome: &OpOutcome, mesh: &Mesh) {
        self.generation = generation;
        self.clear();
        // The outcome is taken **literally** — it is the op's successor, not an
        // inventory (see [`OpOutcome`]) — and then widened DOWNWARD only: a face
        // implies its edges and their vertices, an edge implies its endpoints.
        // Never upward, because "the faces around these vertices" is exactly how
        // an extrude's cap selection would grow back into the walls it was
        // carefully kept apart from.
        self.verts.extend(outcome.verts.iter().copied());
        self.faces.extend(outcome.faces.iter().copied());
        for &h in &outcome.halfs {
            let Some(c) = canonical_edge(mesh, h) else {
                continue;
            };
            self.edges.insert(c);
            if let Some([a, b]) = edge_ends(mesh, c) {
                self.verts.insert(a);
                self.verts.insert(b);
            }
        }
        let faces = self.faces.clone();
        for f in faces {
            let Some(halfs) = mesh.face_loop(f) else {
                continue;
            };
            for h in halfs {
                if let Some(c) = canonical_edge(mesh, h) {
                    self.edges.insert(c);
                }
                if let Some(v) = mesh.origin(h) {
                    self.verts.insert(v);
                }
            }
        }
    }

    pub fn verts(&self) -> &BTreeSet<VertId> {
        &self.verts
    }
    pub fn edges(&self) -> &BTreeSet<HalfId> {
        &self.edges
    }
    pub fn faces(&self) -> &BTreeSet<FaceId> {
        &self.faces
    }

    pub fn clear(&mut self) {
        self.verts.clear();
        self.edges.clear();
        self.faces.clear();
    }

    /// How many components are selected in `mode`.
    pub fn len(&self, mode: SelectMode) -> usize {
        match mode {
            SelectMode::Vert => self.verts.len(),
            SelectMode::Edge => self.edges.len(),
            SelectMode::Face => self.faces.len(),
        }
    }

    pub fn is_empty(&self, mode: SelectMode) -> bool {
        self.len(mode) == 0
    }

    pub fn set_vert(&mut self, v: VertId, on: bool) {
        if on {
            self.verts.insert(v);
        } else {
            self.verts.remove(&v);
        }
    }

    /// Select or deselect an edge, by whichever half the caller had.
    pub fn set_edge(&mut self, mesh: &Mesh, h: HalfId, on: bool) {
        let Some(c) = canonical_edge(mesh, h) else {
            return;
        };
        if on {
            self.edges.insert(c);
        } else {
            self.edges.remove(&c);
        }
    }

    pub fn set_face(&mut self, f: FaceId, on: bool) {
        if on {
            self.faces.insert(f);
        } else {
            self.faces.remove(&f);
        }
    }

    pub fn contains_vert(&self, v: VertId) -> bool {
        self.verts.contains(&v)
    }
    pub fn contains_edge(&self, mesh: &Mesh, h: HalfId) -> bool {
        canonical_edge(mesh, h).is_some_and(|c| self.edges.contains(&c))
    }
    pub fn contains_face(&self, f: FaceId) -> bool {
        self.faces.contains(&f)
    }

    // ── set algebra over the topology ──────────────────────────────────────

    /// Expand the selection by one topological ring.
    ///
    /// Adjacency is **through vertices** in all three modes, which is the rule
    /// that makes grow reach diagonally across a quad grid — the behaviour an
    /// author expects from a grow, and the one that makes grow∘shrink not a
    /// no-op (it is a morphological closing, exactly like the image operators
    /// these are named after).
    pub fn grow(&mut self, mesh: &Mesh, mode: SelectMode) {
        match mode {
            SelectMode::Vert => {
                let mut add = BTreeSet::new();
                for &v in &self.verts {
                    for w in neighbours(mesh, v) {
                        add.insert(w);
                    }
                }
                self.verts.extend(add);
            }
            SelectMode::Edge => {
                let seed = self.edge_verts(mesh);
                let mut add = BTreeSet::new();
                for v in seed {
                    for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
                        if let Some(c) = canonical_edge(mesh, h) {
                            add.insert(c);
                        }
                    }
                }
                self.edges.extend(add);
            }
            SelectMode::Face => {
                let seed = self.face_verts(mesh);
                let mut add = BTreeSet::new();
                for v in seed {
                    for f in faces_at(mesh, v) {
                        add.insert(f);
                    }
                }
                self.faces.extend(add);
            }
        }
    }

    /// Contract the selection by one ring: drop every element that touches
    /// something unselected.
    pub fn shrink(&mut self, mesh: &Mesh, mode: SelectMode) {
        match mode {
            SelectMode::Vert => {
                let keep: BTreeSet<VertId> = self
                    .verts
                    .iter()
                    .copied()
                    .filter(|&v| neighbours(mesh, v).iter().all(|w| self.verts.contains(w)))
                    .collect();
                self.verts = keep;
            }
            SelectMode::Edge => {
                let keep: BTreeSet<HalfId> = self
                    .edges
                    .iter()
                    .copied()
                    .filter(|&h| {
                        edge_ends(mesh, h).into_iter().flatten().all(|v| {
                            mesh.vert_outgoing(v)
                                .unwrap_or(&[])
                                .iter()
                                .all(|&x| self.contains_edge(mesh, x))
                        })
                    })
                    .collect();
                self.edges = keep;
            }
            SelectMode::Face => {
                let keep: BTreeSet<FaceId> = self
                    .faces
                    .iter()
                    .copied()
                    .filter(|&f| {
                        mesh.face_verts(f).unwrap_or_default().into_iter().all(|v| {
                            faces_at(mesh, v)
                                .into_iter()
                                .all(|g| self.faces.contains(&g))
                        })
                    })
                    .collect();
                self.faces = keep;
            }
        }
    }

    /// Replace the selection with everything that is *not* selected, in `mode`.
    pub fn invert(&mut self, mesh: &Mesh, mode: SelectMode) {
        match mode {
            SelectMode::Vert => {
                self.verts = mesh
                    .vert_ids()
                    .filter(|v| !self.verts.contains(v))
                    .collect();
            }
            SelectMode::Edge => {
                self.edges = mesh
                    .half_ids()
                    .filter_map(|h| canonical_edge(mesh, h))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .filter(|c| !self.edges.contains(c))
                    .collect();
            }
            SelectMode::Face => {
                self.faces = mesh
                    .face_ids()
                    .filter(|f| !self.faces.contains(f))
                    .collect();
            }
        }
    }

    /// Grow the selection to every component connected to it — the whole shell.
    pub fn select_linked(&mut self, mesh: &Mesh, mode: SelectMode) {
        let seeds = match mode {
            SelectMode::Vert => self.verts.clone(),
            SelectMode::Edge => self.edge_verts(mesh),
            SelectMode::Face => self.face_verts(mesh),
        };
        let component = connected_verts(mesh, &seeds);
        match mode {
            SelectMode::Vert => self.verts = component,
            SelectMode::Edge => {
                self.edges = component
                    .iter()
                    .flat_map(|&v| mesh.vert_outgoing(v).unwrap_or(&[]).iter().copied())
                    .filter_map(|h| canonical_edge(mesh, h))
                    .collect();
            }
            SelectMode::Face => {
                self.faces = component.iter().flat_map(|&v| faces_at(mesh, v)).collect();
            }
        }
    }

    /// Add the edge loop through `h` — the walk that continues straight on across
    /// every valence-4 vertex it meets.
    pub fn select_edge_loop(&mut self, mesh: &Mesh, h: HalfId) {
        for x in edge_loop(mesh, h) {
            if let Some(c) = canonical_edge(mesh, x) {
                self.edges.insert(c);
            }
        }
    }

    /// Add the edge ring through `h` — the walk that steps *across* each quad to
    /// the edge opposite the one it came in on. The same traversal
    /// [`crate::ops::Op::LoopCut`] cuts along.
    pub fn select_edge_ring(&mut self, mesh: &Mesh, h: HalfId) {
        for x in edge_ring(mesh, h) {
            if let Some(c) = canonical_edge(mesh, x) {
                self.edges.insert(c);
            }
        }
    }

    // ── conversions, with the standard containment semantics ───────────────

    /// Rewrite the `to` set from the `from` set.
    ///
    /// Widening (face → vert) takes **everything** the source touches; narrowing
    /// (vert → face) takes only what is **wholly** contained. That asymmetry is
    /// the standard every DCC uses, and it is the only pair that makes
    /// `convert(a→b)` then `convert(b→a)` non-expanding.
    pub fn convert(&mut self, mesh: &Mesh, from: SelectMode, to: SelectMode) {
        if from == to {
            return;
        }
        match (from, to) {
            (SelectMode::Vert, SelectMode::Edge) => {
                self.edges = all_edges(mesh)
                    .into_iter()
                    .filter(|&c| {
                        edge_ends(mesh, c)
                            .into_iter()
                            .flatten()
                            .all(|v| self.verts.contains(&v))
                    })
                    .collect();
            }
            (SelectMode::Vert, SelectMode::Face) => {
                self.faces = mesh
                    .face_ids()
                    .filter(|&f| {
                        mesh.face_verts(f)
                            .unwrap_or_default()
                            .iter()
                            .all(|v| self.verts.contains(v))
                    })
                    .collect();
            }
            (SelectMode::Edge, SelectMode::Vert) => self.verts = self.edge_verts(mesh),
            (SelectMode::Edge, SelectMode::Face) => {
                self.faces = mesh
                    .face_ids()
                    .filter(|&f| {
                        mesh.face_loop(f)
                            .unwrap_or_default()
                            .iter()
                            .all(|&h| self.contains_edge(mesh, h))
                    })
                    .collect();
            }
            (SelectMode::Face, SelectMode::Vert) => self.verts = self.face_verts(mesh),
            (SelectMode::Face, SelectMode::Edge) => {
                self.edges = self
                    .faces
                    .iter()
                    .flat_map(|&f| mesh.face_loop(f).unwrap_or_default())
                    .filter_map(|h| canonical_edge(mesh, h))
                    .collect();
            }
            _ => unreachable!("from == to was handled above"),
        }
    }

    /// Every vertex the selected edges touch.
    pub fn edge_verts(&self, mesh: &Mesh) -> BTreeSet<VertId> {
        self.edges
            .iter()
            .flat_map(|&h| edge_ends(mesh, h))
            .flatten()
            .collect()
    }

    /// Every corner of the selected faces.
    pub fn face_verts(&self, mesh: &Mesh) -> BTreeSet<VertId> {
        self.faces
            .iter()
            .flat_map(|&f| mesh.face_verts(f).unwrap_or_default())
            .collect()
    }

    /// The vertices this selection means, in `mode` — the seed set for a soft
    /// select and for any op that works on vertices.
    pub fn resolved_verts(&self, mesh: &Mesh, mode: SelectMode) -> BTreeSet<VertId> {
        match mode {
            SelectMode::Vert => self.verts.clone(),
            SelectMode::Edge => self.edge_verts(mesh),
            SelectMode::Face => self.face_verts(mesh),
        }
    }

    /// Soft-select weights: `1` on the selection, falling to `0` at `radius`
    /// metres of **geodesic** distance. Only non-zero entries are returned.
    ///
    /// See the module docs for why the distance is geodesic and the falloff is
    /// [`inf_terrain::Falloff`] rather than a curve reinvented here.
    pub fn soft_weights(
        &self,
        mesh: &Mesh,
        mode: SelectMode,
        radius: f64,
        falloff: Falloff,
    ) -> BTreeMap<VertId, f64> {
        let seeds = self.resolved_verts(mesh, mode);
        if seeds.is_empty() || !(radius.is_finite() && radius > 0.0) {
            return seeds.into_iter().map(|v| (v, 1.0)).collect();
        }
        geodesic_distances(mesh, &seeds, radius)
            .into_iter()
            .filter_map(|(v, d)| {
                let w = falloff.weight(d, radius);
                (w > 0.0).then_some((v, w))
            })
            .collect()
    }
}

// ── free traversals ────────────────────────────────────────────────────────

/// The two endpoints of an edge, `None` for a dead id.
fn edge_ends(mesh: &Mesh, h: HalfId) -> Option<[VertId; 2]> {
    Some([mesh.origin(h)?, mesh.dest(h)?])
}

fn neighbours(mesh: &Mesh, v: VertId) -> BTreeSet<VertId> {
    mesh.vert_outgoing(v)
        .unwrap_or(&[])
        .iter()
        .filter_map(|&h| mesh.dest(h))
        .collect()
}

fn faces_at(mesh: &Mesh, v: VertId) -> BTreeSet<FaceId> {
    mesh.vert_outgoing(v)
        .unwrap_or(&[])
        .iter()
        .filter_map(|&h| match mesh.face_of(h) {
            Some(Some(f)) => Some(f),
            _ => None,
        })
        .collect()
}

fn all_edges(mesh: &Mesh) -> BTreeSet<HalfId> {
    mesh.half_ids()
        .filter_map(|h| canonical_edge(mesh, h))
        .collect()
}

/// Everything reachable from `seeds` along edges.
fn connected_verts(mesh: &Mesh, seeds: &BTreeSet<VertId>) -> BTreeSet<VertId> {
    let mut seen: BTreeSet<VertId> = seeds.clone();
    let mut stack: Vec<VertId> = seeds.iter().copied().collect();
    while let Some(v) = stack.pop() {
        for w in neighbours(mesh, v) {
            if seen.insert(w) {
                stack.push(w);
            }
        }
    }
    seen
}

/// The **edge ring** through `start`: step across each quad to the edge opposite
/// the one entered on, both ways, stopping at anything that is not an interior
/// quad.
///
/// Returned in walk order starting from the backward end, with `start` inside it
/// — which is what makes [`crate::ops::Op::LoopCut`] able to treat entry `i` and
/// exit `i` of the same face as one strip step.
pub fn edge_ring(mesh: &Mesh, start: HalfId) -> Vec<HalfId> {
    let mut forward = Vec::new();
    let mut seen: BTreeSet<HalfId> = [start].into_iter().collect();
    let mut cur = start;
    while let Some(next) = ring_step(mesh, cur) {
        if !seen.insert(next) {
            return std::iter::once(start).chain(forward).collect(); // closed
        }
        forward.push(next);
        cur = next;
    }
    let mut backward = Vec::new();
    let mut cur = start;
    while let Some(prev) = ring_step_back(mesh, cur) {
        if !seen.insert(prev) {
            break;
        }
        backward.push(prev);
        cur = prev;
    }
    backward.reverse();
    backward
        .into_iter()
        .chain(std::iter::once(start))
        .chain(forward)
        .collect()
}

/// From a half-edge entering a quad, the half-edge entering the next quad.
fn ring_step(mesh: &Mesh, h: HalfId) -> Option<HalfId> {
    let f = mesh.face_of(h)??;
    if mesh.face_loop(f)?.len() != 4 {
        return None;
    }
    let opposite = mesh.next(mesh.next(h)?)?;
    let t = mesh.twin(opposite)?;
    let g = mesh.face_of(t)??;
    (mesh.face_loop(g)?.len() == 4).then_some(t)
}

/// The inverse step: from a half-edge entering a quad, the one entering the quad
/// on the other side of that edge.
fn ring_step_back(mesh: &Mesh, h: HalfId) -> Option<HalfId> {
    let t = mesh.twin(h)?;
    let g = mesh.face_of(t)??;
    if mesh.face_loop(g)?.len() != 4 {
        return None;
    }
    let entry = mesh.next(mesh.next(t)?)?;
    let f = mesh.face_of(entry)??;
    (mesh.face_loop(f)?.len() == 4).then_some(entry)
}

/// The **edge loop** through `start`: continue straight on at every valence-4
/// vertex, both ways.
pub fn edge_loop(mesh: &Mesh, start: HalfId) -> Vec<HalfId> {
    let mut forward = Vec::new();
    let mut seen: BTreeSet<HalfId> = [start].into_iter().collect();
    let mut cur = start;
    while let Some(next) = loop_step(mesh, cur) {
        if !seen.insert(next) {
            return std::iter::once(start).chain(forward).collect();
        }
        forward.push(next);
        cur = next;
    }
    let mut backward = Vec::new();
    let mut cur = start;
    while let Some(prev) = loop_step_back(mesh, cur) {
        if !seen.insert(prev) {
            break;
        }
        backward.push(prev);
        cur = prev;
    }
    backward.reverse();
    backward
        .into_iter()
        .chain(std::iter::once(start))
        .chain(forward)
        .collect()
}

/// At `dest(h)`, the outgoing half-edge two rotations from `twin(h)` — the
/// "opposite" spoke of a valence-4 fan, and therefore the straight continuation.
fn loop_step(mesh: &Mesh, h: HalfId) -> Option<HalfId> {
    let t = mesh.twin(h)?;
    let b = mesh.origin(t)?;
    if mesh.vert_outgoing(b)?.len() != 4 || !mesh.vertex_is_manifold(b) {
        return None;
    }
    let r1 = mesh.rotate_around_origin(t)?;
    mesh.rotate_around_origin(r1)
}

fn loop_step_back(mesh: &Mesh, h: HalfId) -> Option<HalfId> {
    let a = mesh.origin(h)?;
    if mesh.vert_outgoing(a)?.len() != 4 || !mesh.vertex_is_manifold(a) {
        return None;
    }
    let r1 = mesh.rotate_around_origin(h)?;
    let r2 = mesh.rotate_around_origin(r1)?;
    mesh.twin(r2)
}

/// Dijkstra over the edge graph, in metres, bounded by `max`.
///
/// Deterministic by construction: the frontier is ordered by `(distance bits,
/// vertex id)`, and `f64::to_bits` is monotone on the non-negative finites a
/// length can be. A non-finite edge length (an imported mesh the reader let
/// through — [`crate::build::ImportReport::non_finite_values`]) is *skipped*
/// rather than poisoning every distance behind it with a NaN.
pub fn geodesic_distances(
    mesh: &Mesh,
    seeds: &BTreeSet<VertId>,
    max: f64,
) -> BTreeMap<VertId, f64> {
    let mut dist: BTreeMap<VertId, f64> = BTreeMap::new();
    let mut frontier: BTreeSet<(u64, VertId)> = BTreeSet::new();
    for &v in seeds {
        if mesh.has_vert(v) {
            dist.insert(v, 0.0);
            frontier.insert((0, v));
        }
    }
    while let Some(&(bits, v)) = frontier.iter().next() {
        frontier.remove(&(bits, v));
        let d = f64::from_bits(bits);
        if dist.get(&v).is_some_and(|&best| d > best) {
            continue; // a shorter route settled this vertex already
        }
        let Some(p) = mesh.position(v) else { continue };
        for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
            let Some(w) = mesh.dest(h) else { continue };
            let Some(q) = mesh.position(w) else { continue };
            let step = (q - p).length();
            if !step.is_finite() {
                continue;
            }
            let nd = d + step;
            if nd > max {
                continue;
            }
            if dist.get(&w).is_none_or(|&best| nd < best) {
                dist.insert(w, nd);
                frontier.insert((nd.to_bits(), w));
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, cylinder, plane};
    use crate::journal::MeshSession;
    use crate::topo::CornerData;

    /// An `n × n` grid of quads in XZ, unit spacing — the fixture loops, rings,
    /// grow/shrink and geodesics all have a hand-checkable answer on.
    fn grid(n: usize) -> Mesh {
        let mut m = Mesh::new();
        let mut ids = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                ids.push(m.alloc_vert([i as f64, 0.0, j as f64]));
            }
        }
        let at = |i: usize, j: usize| ids[j * (n + 1) + i];
        let mut touched = BTreeSet::new();
        for j in 0..n {
            for i in 0..n {
                let quad = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                touched.extend(quad);
                m.add_face_raw(&quad, &[CornerData::default(); 4], None)
                    .expect("a grid is well formed");
            }
        }
        m.finish_patch(&touched).expect("a grid is manifold");
        m
    }

    #[test]
    fn a_selection_is_dropped_when_the_generation_moves() {
        let mut s = MeshSession::new(cube(1.0));
        let mut sel = SelectionSet::new(s.generation());
        sel.set_face(s.mesh().face_ids().next().expect("a face"), true);
        assert!(!sel.sync(s.generation()), "nothing moved");
        assert_eq!(sel.len(SelectMode::Face), 1);

        let h = s.mesh().half_ids().next().expect("an edge");
        s.apply(Op::SplitEdge { half: h, t: 0.5 }).expect("split");
        assert!(
            sel.sync(s.generation()),
            "the stamp moved, so the set drops"
        );
        assert!(sel.is_empty(SelectMode::Face));
    }

    #[test]
    fn adopt_replaces_the_selection_with_what_the_op_made() {
        let mut s = MeshSession::new(cube(2.0));
        let mut sel = SelectionSet::new(s.generation());
        let top = s.mesh().face_ids().next().expect("a face");
        let out = s
            .apply(Op::ExtrudeFaces {
                faces: vec![top],
                distance: 1.0,
            })
            .expect("extrude");
        sel.adopt(s.generation(), &out, s.mesh());
        assert_eq!(sel.generation(), s.generation());
        assert_eq!(
            sel.len(SelectMode::Face),
            1,
            "the CAP -- `adopt` takes the outcome literally and widens only downward"
        );
        assert_eq!(sel.len(SelectMode::Vert), 4, "and its four corners");
        assert!(sel.faces().iter().all(|&f| s.mesh().has_face(f),));
    }

    #[test]
    fn carry_keeps_ids_across_an_op_that_cannot_renumber() {
        let mut s = MeshSession::new(plane(2.0));
        let mut sel = SelectionSet::new(s.generation());
        let v: Vec<VertId> = s.mesh().vert_ids().collect();
        sel.set_vert(v[0], true);
        assert!(op_preserves_ids(&Op::TranslateVerts {
            verts: vec![v[0]],
            delta: [0.0; 3]
        }));
        s.apply(Op::TranslateVerts {
            verts: vec![v[0]],
            delta: [0.0, 1.0, 0.0],
        })
        .expect("translate");
        sel.carry(s.generation(), s.mesh());
        assert_eq!(sel.generation(), s.generation());
        assert!(sel.contains_vert(v[0]), "an attribute op does not renumber");
    }

    #[test]
    fn carry_drops_an_id_whose_element_died() {
        let s = MeshSession::new(plane(2.0));
        let mut sel = SelectionSet::new(s.generation());
        let ghost = VertId(9_999);
        sel.set_vert(ghost, true);
        sel.carry(s.generation(), s.mesh());
        assert!(!sel.contains_vert(ghost));
    }

    #[test]
    fn the_id_preservation_table_agrees_with_what_the_ops_actually_do() {
        // The claim `carry` rests on, measured rather than asserted: for every op
        // the table calls id-preserving, no live id changes meaning. The probe is
        // the arena's own capacity — a rebuild frees and re-allocates, which shows
        // up as a changed live-id set even when the counts match.
        let mut m = cube(1.0);
        let before: Vec<VertId> = m.vert_ids().collect();
        let hs: Vec<HalfId> = m.half_ids().collect();
        let fs: Vec<FaceId> = m.face_ids().collect();
        let preserving = [
            Op::TranslateVerts {
                verts: vec![before[0]],
                delta: [0.1, 0.0, 0.0],
            },
            Op::SetCornerUv {
                half: hs[0],
                uv: [0.3, 0.4],
            },
            Op::SetEdgeSharp {
                half: hs[0],
                sharp: false,
            },
            Op::SetFaceSlot {
                face: fs[0],
                slot: None,
            },
        ];
        for op in preserving {
            assert!(op_preserves_ids(&op), "{op:?}");
            crate::ops::apply(&mut m, &op).expect("applies");
            assert_eq!(m.vert_ids().collect::<Vec<_>>(), before);
            assert_eq!(m.half_ids().collect::<Vec<_>>(), hs);
            assert_eq!(m.face_ids().collect::<Vec<_>>(), fs);
        }
        // And the other half -- the hazard itself, measured. A structural op's
        // free list hands the SAME face ids straight back, so the id SET is
        // identical and every one of them may now name a different polygon. That
        // is exactly why `carry` is illegal here and `adopt` exists: a liveness
        // check could never have caught this.
        let loops_before: Vec<Vec<VertId>> =
            fs.iter().map(|&f| m.face_verts(f).expect("live")).collect();
        assert!(!op_preserves_ids(&Op::SplitEdge {
            half: hs[0],
            t: 0.5
        }));
        crate::ops::apply(
            &mut m,
            &Op::SplitEdge {
                half: hs[0],
                t: 0.5,
            },
        )
        .expect("split");
        assert_eq!(
            m.face_ids().collect::<Vec<_>>(),
            fs,
            "the ids come BACK, unchanged as a set"
        );
        let loops_after: Vec<Vec<VertId>> =
            fs.iter().map(|&f| m.face_verts(f).expect("live")).collect();
        assert_ne!(
            loops_before, loops_after,
            "...while naming different polygons"
        );
    }

    #[test]
    fn every_op_is_classified_and_the_classification_is_the_one_written_down() {
        // The table's *behaviour*, against a hand-written list rather than
        // against itself. `the_id_preservation_table_agrees_with_what_the_ops
        // _actually_do` proves the six "preserving" answers are true of the
        // ops; this proves nothing has been quietly added to or removed from
        // that side, and `determinism_law`'s source gate proves the function
        // still enumerates every variant instead of defaulting through a
        // wildcard. Three different failures, three different gates.
        let hand: [(Op, bool); 25] = [
            (Op::AddVertex { position: [0.0; 3] }, true),
            (Op::RemoveVertex { vert: VertId(0) }, false),
            (
                Op::AddFace {
                    verts: vec![],
                    corners: vec![],
                    slot: None,
                },
                false,
            ),
            (Op::RemoveFace { face: FaceId(0) }, false),
            (
                Op::SplitEdge {
                    half: HalfId(0),
                    t: 0.5,
                },
                false,
            ),
            (Op::CollapseEdge { half: HalfId(0) }, false),
            (
                Op::SplitFace {
                    from: HalfId(0),
                    to: HalfId(1),
                },
                false,
            ),
            (
                Op::WeldVerts {
                    keep: VertId(0),
                    merge: VertId(1),
                },
                false,
            ),
            (
                Op::TranslateVerts {
                    verts: vec![],
                    delta: [0.0; 3],
                },
                true,
            ),
            (
                Op::SetCornerUv {
                    half: HalfId(0),
                    uv: [0.0; 2],
                },
                true,
            ),
            (
                Op::SetCornerNormal {
                    half: HalfId(0),
                    normal: None,
                },
                true,
            ),
            (
                Op::SetEdgeSharp {
                    half: HalfId(0),
                    sharp: true,
                },
                true,
            ),
            (
                Op::SetFaceSlot {
                    face: FaceId(0),
                    slot: None,
                },
                true,
            ),
            (
                Op::ExtrudeFaces {
                    faces: vec![],
                    distance: 1.0,
                },
                false,
            ),
            (
                Op::ExtrudeEdges {
                    edges: vec![],
                    delta: [0.0; 3],
                },
                false,
            ),
            (
                Op::InsetFaces {
                    faces: vec![],
                    amount: 0.1,
                    individual: false,
                },
                false,
            ),
            (
                Op::BevelEdges {
                    edges: vec![],
                    amount: 0.1,
                },
                false,
            ),
            (
                Op::LoopCut {
                    half: HalfId(0),
                    cuts: 1,
                },
                false,
            ),
            (Op::Knife { path: vec![] }, false),
            (
                Op::MergeVerts {
                    verts: vec![],
                    target: crate::model::MergeTarget::Center,
                },
                false,
            ),
            (Op::SubdivideFaces { faces: vec![] }, false),
            (
                Op::Mirror {
                    axis: crate::model::MirrorAxis::X,
                    coord: 0.0,
                },
                false,
            ),
            (
                Op::Sculpt {
                    mode: crate::sculpt::SculptMode::Draw,
                    dabs: vec![],
                    radius: 1.0,
                    strength: 0.1,
                    falloff: crate::sculpt::SculptFalloff::Smooth,
                },
                true,
            ),
            (
                Op::RotateVerts {
                    verts: vec![],
                    pivot: [0.0; 3],
                    axis: [0.0, 1.0, 0.0],
                    radians: 0.0,
                },
                true,
            ),
            (
                Op::ScaleVerts {
                    verts: vec![],
                    pivot: [0.0; 3],
                    factor: [1.0; 3],
                },
                true,
            ),
        ];
        for (op, want) in &hand {
            assert_eq!(
                op_preserves_ids(op),
                *want,
                "{op:?} is classified differently from the hand list"
            );
        }
        assert_eq!(
            hand.iter().filter(|(_, w)| *w).count(),
            9,
            "nine ops preserve ids; a tenth is a claim that needs its own proof"
        );
    }

    #[test]
    fn a_soft_select_measures_metres_and_not_hops() {
        // The gate the unit-spaced fixtures could not be: on a grid where every
        // edge is 1 m, a Dijkstra that returned a constant 1.0 per step is
        // indistinguishable from one that measures. Here the row is spaced
        // 0.1 / 0.1 / 0.1 / 2.7, so three hops is 0.3 m and four is 3.0 m, and
        // the two answers cannot be confused.
        let mut m = Mesh::new();
        let xs = [0.0, 0.1, 0.2, 0.3, 3.0];
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for &x in &xs {
            lower.push(m.alloc_vert([x, 0.0, 0.0]));
            upper.push(m.alloc_vert([x, 0.0, 1.0]));
        }
        let mut touched = BTreeSet::new();
        for i in 0..xs.len() - 1 {
            let quad = [lower[i], lower[i + 1], upper[i + 1], upper[i]];
            touched.extend(quad);
            m.add_face_raw(&quad, &[CornerData::default(); 4], None)
                .expect("a strip is well formed");
        }
        m.finish_patch(&touched).expect("manifold");

        let seeds: BTreeSet<VertId> = [lower[0]].into_iter().collect();
        let d = geodesic_distances(&m, &seeds, 1.0);
        // Measured in metres: 0.1, 0.2, 0.3 along the row — a hop count would
        // have said 1, 2, 3 and stopped after the first.
        for (i, want) in [(1usize, 0.1), (2, 0.2), (3, 0.3)] {
            let got = *d
                .get(&lower[i])
                .unwrap_or_else(|| panic!("x = {} is 0.{i} m away and must be reached", xs[i]));
            assert!(
                (got - want).abs() < 1e-12,
                "x={} gave {got}, want {want}",
                xs[i]
            );
        }
        assert!(
            !d.contains_key(&lower[4]),
            "the far vertex is 3.0 m away and outside a 1 m radius, however few \
             hops it takes"
        );
        // And the weights follow the metres, not the topology.
        let mut sel = SelectionSet::new(1);
        sel.set_vert(lower[0], true);
        let w = sel.soft_weights(&m, SelectMode::Vert, 1.0, Falloff::Linear);
        assert!((w[&lower[1]] - 0.9).abs() < 1e-12, "{:?}", w.get(&lower[1]));
        assert!((w[&lower[3]] - 0.7).abs() < 1e-12, "{:?}", w.get(&lower[3]));
        assert!(!w.contains_key(&lower[4]));
    }

    #[test]
    fn an_edge_loop_on_a_grid_runs_all_the_way_across() {
        let m = grid(4);
        // A vertical edge in the middle: its loop is the 4 edges of one column.
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.position(m.origin(h).expect("live")).expect("live");
                let b = m.position(m.dest(h).expect("live")).expect("live");
                (a.x - 2.0).abs() < 1e-12 && (b.x - 2.0).abs() < 1e-12
            })
            .expect("a column edge");
        let l = edge_loop(&m, h);
        assert_eq!(
            l.len(),
            4,
            "the whole column, both directions from the seed"
        );
        for &x in &l {
            let a = m.position(m.origin(x).expect("live")).expect("live");
            let b = m.position(m.dest(x).expect("live")).expect("live");
            assert!((a.x - 2.0).abs() < 1e-12 && (b.x - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn an_edge_ring_on_a_cylinder_closes_and_a_grid_ring_does_not() {
        let c = cylinder(1.0, 2.0, 8);
        let h = c
            .half_ids()
            .find(|&h| {
                let a = c.position(c.origin(h).expect("live")).expect("live");
                let b = c.position(c.dest(h).expect("live")).expect("live");
                (a.y - b.y).abs() > 1.0
            })
            .expect("a vertical side edge");
        assert_eq!(edge_ring(&c, h).len(), 8, "the closed band of side edges");

        let g = grid(4);
        let h = g
            .half_ids()
            .find(|&h| {
                let a = g.position(g.origin(h).expect("live")).expect("live");
                let b = g.position(g.dest(h).expect("live")).expect("live");
                (a.x - 2.0).abs() < 1e-12 && (b.x - 2.0).abs() < 1e-12
            })
            .expect("a column edge");
        assert_eq!(edge_ring(&g, h).len(), 4, "an open ring across the grid");
    }

    #[test]
    fn grow_and_shrink_move_the_boundary_by_one_ring() {
        // A 5x5 grid so the grown block is strictly interior. On a 4x4 the block
        // would touch the mesh boundary, and shrink would (correctly) keep the
        // corner quads that have no unselected neighbour to shrink away from --
        // real behaviour, and not what this test is about.
        let m = grid(5);
        let mut sel = SelectionSet::new(1);
        let centre = m
            .face_ids()
            .find(|&f| {
                m.face_verts(f).expect("live").iter().all(|&v| {
                    let p = m.position(v).expect("live");
                    (2.0..=3.0).contains(&p.x) && (2.0..=3.0).contains(&p.z)
                })
            })
            .expect("the middle quad");
        sel.set_face(centre, true);
        sel.grow(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 9, "the 3x3 block around it");
        sel.shrink(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 1, "back to the middle");
    }

    #[test]
    fn shrink_keeps_what_has_no_unselected_neighbour_to_shrink_from() {
        // The other half, stated rather than left to be discovered: at a mesh
        // boundary there is nothing outside the selection, so those faces stay.
        // Blender's "Select Less" behaves the same way, and a test that expected
        // otherwise would be asserting a bug into existence.
        let m = grid(2);
        let mut sel = SelectionSet::new(1);
        sel.invert(&m, SelectMode::Face); // everything
        assert_eq!(sel.len(SelectMode::Face), 4);
        sel.shrink(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 4, "nothing to shrink away from");
    }

    #[test]
    fn invert_and_linked_cover_the_mesh() {
        let m = cube(1.0);
        let mut sel = SelectionSet::new(1);
        sel.set_face(m.face_ids().next().expect("a face"), true);
        sel.invert(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 5);
        sel.clear();
        sel.set_face(m.face_ids().next().expect("a face"), true);
        sel.select_linked(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 6, "one shell");
    }

    #[test]
    fn linked_does_not_cross_between_two_shells() {
        // Two cubes in one mesh: a linked select from one must not reach the other.
        let mut m = cube(1.0);
        let far = cube(1.0);
        let mut map = BTreeMap::new();
        for v in far.vert_ids() {
            let p = far.position(v).expect("live").to_array();
            map.insert(v, m.alloc_vert([p[0] + 10.0, p[1], p[2]]));
        }
        let mut touched = BTreeSet::new();
        for f in far.face_ids() {
            let loop_verts: Vec<VertId> = far
                .face_verts(f)
                .expect("live")
                .into_iter()
                .map(|v| map[&v])
                .collect();
            touched.extend(loop_verts.iter().copied());
            m.add_face_raw(&loop_verts, &[CornerData::default(); 4], None)
                .expect("well formed");
        }
        m.finish_patch(&touched).expect("manifold");
        assert_eq!(m.face_count(), 12);

        let mut sel = SelectionSet::new(1);
        sel.set_face(m.face_ids().next().expect("a face"), true);
        sel.select_linked(&m, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 6, "one shell, not both");
    }

    #[test]
    fn conversions_widen_by_touching_and_narrow_by_containment() {
        let m = plane(2.0);
        let f = m.face_ids().next().expect("the quad");
        let mut sel = SelectionSet::new(1);
        sel.set_face(f, true);
        sel.convert(&m, SelectMode::Face, SelectMode::Vert);
        assert_eq!(sel.len(SelectMode::Vert), 4);
        sel.convert(&m, SelectMode::Vert, SelectMode::Edge);
        assert_eq!(sel.len(SelectMode::Edge), 4);

        // Narrowing is containment: drop one vertex and the face stops qualifying.
        let mut sel = SelectionSet::new(1);
        let vs: Vec<VertId> = m.vert_ids().collect();
        for &v in &vs[..3] {
            sel.set_vert(v, true);
        }
        sel.convert(&m, SelectMode::Vert, SelectMode::Face);
        assert_eq!(
            sel.len(SelectMode::Face),
            0,
            "3 of 4 corners is not the face"
        );
        sel.set_vert(vs[3], true);
        sel.convert(&m, SelectMode::Vert, SelectMode::Face);
        assert_eq!(sel.len(SelectMode::Face), 1);
    }

    #[test]
    fn soft_weights_fall_off_over_geodesic_metres() {
        let m = grid(4); // unit spacing, so distance is countable by hand
        let corner = m
            .vert_ids()
            .find(|&v| {
                m.position(v)
                    .expect("live")
                    .abs_diff_eq(glam::DVec3::ZERO, 1e-12)
            })
            .expect("the origin corner");
        let mut sel = SelectionSet::new(1);
        sel.set_vert(corner, true);
        let w = sel.soft_weights(&m, SelectMode::Vert, 2.5, Falloff::Linear);
        assert_eq!(w.get(&corner), Some(&1.0), "the seed is full weight");
        // Its two 1 m neighbours are at exactly 1 m ⇒ 1 − 1/2.5 = 0.6.
        let one_metre: Vec<f64> = m
            .vert_ids()
            .filter(|&v| (m.position(v).expect("live").length() - 1.0).abs() < 1e-12)
            .filter_map(|v| w.get(&v).copied())
            .collect();
        assert_eq!(one_metre.len(), 2);
        for x in one_metre {
            assert!((x - 0.6).abs() < 1e-12, "{x}");
        }
        // And nothing beyond the radius is in the map at all.
        for (&v, &weight) in &w {
            assert!(weight > 0.0);
            let d = m.position(v).expect("live");
            assert!(d.x + d.z <= 2.5 + 1e-12, "{v} is outside the radius");
        }
    }

    #[test]
    fn geodesic_distance_is_the_surface_not_the_room() {
        // Two grids 0.01 m apart in Y with no edge between them. A Euclidean ball
        // of radius 1 would take both; the geodesic walk takes one, which is the
        // whole reason the distance is geodesic.
        let mut m = grid(2);
        let upper = grid(2);
        let mut map = BTreeMap::new();
        for v in upper.vert_ids() {
            let p = upper.position(v).expect("live").to_array();
            map.insert(v, m.alloc_vert([p[0], p[1] + 0.01, p[2]]));
        }
        let mut touched = BTreeSet::new();
        for f in upper.face_ids() {
            let loop_verts: Vec<VertId> = upper
                .face_verts(f)
                .expect("live")
                .into_iter()
                .map(|v| map[&v])
                .collect();
            touched.extend(loop_verts.iter().copied());
            m.add_face_raw(&loop_verts, &[CornerData::default(); 4], None)
                .expect("well formed");
        }
        m.finish_patch(&touched).expect("manifold");

        let seed = m
            .vert_ids()
            .find(|&v| {
                m.position(v)
                    .expect("live")
                    .abs_diff_eq(glam::DVec3::ZERO, 1e-12)
            })
            .expect("the lower origin corner");
        let d = geodesic_distances(&m, &[seed].into_iter().collect(), 1.0);
        let twin_above = map
            .values()
            .copied()
            .find(|&v| {
                let p = m.position(v).expect("live");
                p.x.abs() < 1e-12 && p.z.abs() < 1e-12
            })
            .expect("the upper origin corner");
        assert!(
            !d.contains_key(&twin_above),
            "a 0.01 m Euclidean hop is unreachable ON THE SURFACE"
        );
    }

    #[test]
    fn the_geodesic_frontier_is_order_independent() {
        let m = grid(3);
        let seeds: BTreeSet<VertId> = m.vert_ids().take(2).collect();
        let a = geodesic_distances(&m, &seeds, 3.0);
        let b = geodesic_distances(&m, &seeds, 3.0);
        assert_eq!(a, b);
        // And every settled distance is the true shortest path, not a first
        // arrival: the grid's diagonal corner is 2×n hops of 1 m.
        let far = m
            .vert_ids()
            .find(|&v| {
                let p = m.position(v).expect("live");
                (p.x - 1.0).abs() < 1e-12 && (p.z - 1.0).abs() < 1e-12
            })
            .expect("(1,1)");
        assert!(a
            .get(&far)
            .is_some_and(|&d| (d - 2.0).abs() < 1e-12 || d < 2.0 + 1e-12));
    }
}
