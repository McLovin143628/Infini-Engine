//! The half-edge store: arenas, links, traversal and the low-level surgery the
//! ops in [`crate::ops`] are assembled from.
//!
//! # The structure
//!
//! ```text
//!            next ──▶
//!   ┌───────────────────────────┐
//!   │  h (face side)            │      face(h) = Some(f)   ── a corner of f
//!   ├───────────────────────────┤
//!   │  twin(h)                  │      face(twin(h)) = Some(g) or None
//!   └───────────────────────────┘                           ── a boundary
//!            ◀── prev
//! ```
//!
//! * A **vertex** owns a position (`f64` metres) and the ascending list of
//!   half-edges leaving it. That list is redundant with the links and is
//!   therefore *checked* by [the `validate` module](mod@crate::validate) — a redundancy an invariant
//!   polices is a cheap index; one nothing polices is a desync waiting to ship.
//! * A **half-edge** owns its origin vertex, its twin, its `next`/`prev` within
//!   one closed loop, an optional face, its corner attributes (UV + optional
//!   authored normal) and the sharp flag of the *undirected edge* it is half of
//!   (kept equal on both halves; [the `validate` module](mod@crate::validate) checks that too).
//! * A **face** owns one of its half-edges and a material slot.
//!
//! # Boundaries
//!
//! `twin` is total. A half-edge whose `face` is `None` is a boundary half-edge
//! and is linked into a boundary loop — the same `next`/`prev` machinery as a
//! face loop, walked the same way. So a plane is not a special case, and neither
//! is a hole.
//!
//! **There is no `add_edge`.** A wire edge — one with no face on *either* side —
//! has no loop to live in, so every traversal that assumes "each half-edge lies
//! in exactly one closed cycle" would need a second code path for it. Edges are
//! born from `Mesh::add_face_raw`, [`crate::ops::Op::SplitEdge`] and
//! [`crate::ops::Op::SplitFace`], and they die with the faces that hold them.
//! `WireEdge` is a [`crate::validate::Violation`], not a state.
//!
//! # The id-reuse rule, and what it means for the journal
//!
//! Ids are indices into arenas with a **LIFO free list**: freeing pushes the
//! slot, allocating pops the most recently freed one. A freed id therefore *will*
//! come back, attached to a different element.
//!
//! That is not a hazard for replay — it is what makes replay work. Allocation is
//! a pure function of the op sequence, so replaying the same ops from the same
//! base performs the same allocations in the same order and mints the same ids;
//! that is exactly why a journal can name an element by id at all
//! (`replay(base, ops)` is byte-identical to the incremental edit, property-
//! tested in `tests/property.rs`).
//!
//! It *is* a hazard for anything holding an id **across** a journal move. After
//! an undo, id `v7` may name a different vertex than the one a selection
//! remembered — and it will not be dead, so nothing would catch it. Hence
//! [`crate::journal::MeshSession::generation`]: it bumps on every history move,
//! and a consumer that caches ids (P23.4's selection) must discard its cache
//! when the generation changes. The rule stated once: **an id is meaningful only
//! within one journal generation.**

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::ops::OpError;
use crate::skin::{SkinBinding, VertWeights};

macro_rules! mesh_id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            /// The raw arena index.
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

mesh_id!(
    /// A vertex id. Stable while it lives; reusable after it dies (see the
    /// module docs' id-reuse rule).
    VertId,
    "v"
);
mesh_id!(
    /// A **half-edge** id — one direction of one edge. `twin` pairs them; the
    /// undirected edge is the pair.
    HalfId,
    "h"
);
mesh_id!(
    /// A face id. Faces may be n-gons; triangulation happens at export.
    FaceId,
    "f"
);

/// A slot arena with a LIFO free list. Deterministic by construction: iteration
/// is an integer range and allocation order is a pure function of the call
/// sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Arena<T> {
    slots: Vec<Option<T>>,
    free: Vec<u32>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<T> Arena<T> {
    pub(crate) fn alloc(&mut self, value: T) -> u32 {
        match self.free.pop() {
            Some(i) => {
                self.slots[i as usize] = Some(value);
                i
            }
            None => {
                self.slots.push(Some(value));
                (self.slots.len() - 1) as u32
            }
        }
    }

    pub(crate) fn remove(&mut self, i: u32) -> Option<T> {
        let taken = self.slots.get_mut(i as usize)?.take();
        if taken.is_some() {
            self.free.push(i);
        }
        taken
    }

    pub(crate) fn get(&self, i: u32) -> Option<&T> {
        self.slots.get(i as usize)?.as_ref()
    }

    pub(crate) fn get_mut(&mut self, i: u32) -> Option<&mut T> {
        self.slots.get_mut(i as usize)?.as_mut()
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn live(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.slots.len() as u32).filter(move |&i| self.slots[i as usize].is_some())
    }
}

/// A vertex: a position in metres plus the ascending list of half-edges leaving
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vert {
    /// Local-space position, **metres**, `f64` (architecture rule 3).
    pub position: [f64; 3],
    /// Outgoing half-edges, kept sorted ascending. Redundant with the links and
    /// validated against them.
    pub(crate) out: Vec<HalfId>,
    /// **The skin channel** (P24.2): which joints deform this vertex.
    ///
    /// Total, never optional, and [`VertWeights::RIGID`] on an unskinned mesh —
    /// see [the `skin` module](mod@crate::skin) for why a per-vertex `Option`
    /// would fail its own completeness invariant on the next `AddVertex`.
    ///
    /// **Positional tail field.** bincode writes this struct's fields in
    /// declaration order, so adding it changed the shape of every `Mesh` on the
    /// wire and took [`crate::journal::SessionSave::CURRENT_VERSION`] from 2 to 3
    /// with it. Anything appended after it does the same.
    pub(crate) skin: VertWeights,
}

/// One half-edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HalfEdge {
    pub(crate) origin: VertId,
    pub(crate) twin: HalfId,
    pub(crate) next: HalfId,
    pub(crate) prev: HalfId,
    /// `None` ⇒ this is a **boundary** half-edge.
    pub(crate) face: Option<FaceId>,
    /// Corner UV. Meaningful only when `face` is `Some`; a boundary half-edge
    /// carries the default and nothing ever reads it.
    pub(crate) uv: [f64; 2],
    /// Corner normal. `Some` ⇒ **authored** (imported, or set by
    /// [`crate::ops::Op::SetCornerNormal`]) and preserved verbatim by the default
    /// export policy. `None` ⇒ **derived**: export computes it from the corner's
    /// smooth fan.
    pub(crate) normal: Option<[f64; 3]>,
    /// Sharpness of the undirected edge. Held on both halves and required to
    /// agree ([`crate::validate::Violation::SharpDisagrees`]).
    pub(crate) sharp: bool,
    /// **UV seam** of the undirected edge (P23.5): a cut the unwrapper opens the
    /// surface along. Same storage discipline as `sharp` -- on the twin pair,
    /// held on both halves, required to agree
    /// ([`crate::validate::Violation::SeamDisagrees`]) -- because it describes
    /// one *undirected* edge and must not depend on which half you ask.
    ///
    /// It is a different flag from `sharp` on purpose, even though authors often
    /// mark the same edges: sharpness is a **shading** statement the writer turns
    /// into split normals, and a seam is a **parameterization** statement the
    /// unwrapper turns into a chart boundary. Tying them would make marking a
    /// crease silently re-cut the UVs.
    pub(crate) seam: bool,
}

/// A face: one of its half-edges, plus the material slot it draws with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Face {
    pub(crate) half: HalfId,
    pub(crate) material_slot: Option<u32>,
}

/// Per-corner attributes, as carried through a structural rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CornerData {
    pub uv: [f64; 2],
    pub normal: Option<[f64; 3]>,
}

/// A face captured for rebuild: its vertex loop, its corner attributes and its
/// slot. The unit [`crate::ops`] removes and re-adds when an op changes a local
/// patch of topology.
#[derive(Debug, Clone, PartialEq)]
pub struct FacePatch {
    pub verts: Vec<VertId>,
    pub corners: Vec<CornerData>,
    pub slot: Option<u32>,
}

/// The half-edge mesh.
///
/// Cloning is cheap enough to be a correctness tool: the structural ops run
/// inside `Mesh::transact`, which mutates a clone and commits only if the
/// local manifold checks pass, so **a refused op leaves the mesh byte-identical**
/// (property-tested). The cost is `O(|mesh|)` per structural op; the honest
/// alternative — a slot-level undo log — is a second correctness surface the
/// invariants would then have to police, and `transact` is the single seam where
/// it would land if P23.4 measures a need for it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Mesh {
    pub(crate) verts: Arena<Vert>,
    pub(crate) halfs: Arena<HalfEdge>,
    pub(crate) faces: Arena<Face>,
    /// Material slot names, in slot order — mirrors [`inf_mesh::MeshAsset::material_slots`].
    pub(crate) material_slots: Vec<String>,
    /// Submesh name per slot key, sorted by key. A `Vec` of pairs rather than a
    /// `BTreeMap<Option<u32>, _>` because a non-string map key is not
    /// representable in JSON, and the dual-format rule means every struct here
    /// must survive a self-describing encoder.
    pub(crate) slot_names: Vec<(Option<u32>, String)>,
    /// **What this mesh's joint indices mean** (P24.2), or `None` for a rigid
    /// mesh. See [the `skin` module](mod@crate::skin).
    ///
    /// Positional tail field, appended at v3 — same rule as [`Vert::skin`].
    pub(crate) skin: Option<SkinBinding>,
}

impl Mesh {
    /// An empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    // ── counts ─────────────────────────────────────────────────────────────

    pub fn vert_count(&self) -> usize {
        self.verts.len()
    }
    pub fn half_count(&self) -> usize {
        self.halfs.len()
    }
    /// Undirected edges — half the half-edges, always (twin is total).
    pub fn edge_count(&self) -> usize {
        self.halfs.len() / 2
    }
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    // ── ids ────────────────────────────────────────────────────────────────

    /// Live vertex ids, ascending.
    pub fn vert_ids(&self) -> impl Iterator<Item = VertId> + '_ {
        self.verts.live().map(VertId)
    }
    /// Live half-edge ids, ascending.
    pub fn half_ids(&self) -> impl Iterator<Item = HalfId> + '_ {
        self.halfs.live().map(HalfId)
    }
    /// Live face ids, ascending.
    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> + '_ {
        self.faces.live().map(FaceId)
    }

    pub fn has_vert(&self, v: VertId) -> bool {
        self.verts.get(v.0).is_some()
    }
    pub fn has_half(&self, h: HalfId) -> bool {
        self.halfs.get(h.0).is_some()
    }
    pub fn has_face(&self, f: FaceId) -> bool {
        self.faces.get(f.0).is_some()
    }

    // ── THE ARENA-INDEXING CONTRACT ────────────────────────────────────────
    //
    // Three things can be wrong with an id, and they get three different
    // answers. Stating them together because the `expect`s further down are only
    // honest if the third case cannot reach them:
    //
    // 1. **Dead** — never allocated, or freed. Every public accessor returns
    //    `None`; every op returns a typed refusal (`NoSuchVert`/`NoSuchHalf`/
    //    `NoSuchFace`). Never a panic. This is the case a UI selection hits
    //    after a delete, and it is a value.
    //
    // 2. **Stale but live** — freed and reallocated, or renumbered by a
    //    structural op that rebuilt its patch. The id now names a *different*
    //    element and nothing about the id says so. Not detectable here by
    //    construction; that is the entire reason
    //    `MeshSession::generation` (the op journal) exists, and a consumer
    //    caching ids must compare it.
    //
    // 3. **Internally inconsistent** — a live half-edge whose `twin`, `next`,
    //    `prev`, `origin` or `face` names a dead slot. The `*_ref` helpers below
    //    PANIC on this, deliberately: it is an assertion about an invariant this
    //    crate maintains, not a condition a caller can cause, and turning it
    //    into a refusal would mean inventing an error variant for "the kernel is
    //    broken" and threading it through every traversal. What makes that safe
    //    is that a mesh can only reach the ops layer from three places — a
    //    primitive, `from_mesh_asset`, or another session — and the one door
    //    that accepts a mesh from *outside the process*,
    //    `MeshSession::restore`, runs `validate` in full before any op sees it.
    //    That check is not a nicety: without it a save with one mangled `twin`
    //    panics inside `split_edge`.

    pub fn position(&self, v: VertId) -> Option<DVec3> {
        self.verts.get(v.0).map(|x| DVec3::from_array(x.position))
    }
    pub fn origin(&self, h: HalfId) -> Option<VertId> {
        self.halfs.get(h.0).map(|x| x.origin)
    }
    pub fn twin(&self, h: HalfId) -> Option<HalfId> {
        self.halfs.get(h.0).map(|x| x.twin)
    }
    pub fn next(&self, h: HalfId) -> Option<HalfId> {
        self.halfs.get(h.0).map(|x| x.next)
    }
    pub fn prev(&self, h: HalfId) -> Option<HalfId> {
        self.halfs.get(h.0).map(|x| x.prev)
    }
    pub fn face_of(&self, h: HalfId) -> Option<Option<FaceId>> {
        self.halfs.get(h.0).map(|x| x.face)
    }
    /// The vertex a half-edge points at.
    pub fn dest(&self, h: HalfId) -> Option<VertId> {
        let t = self.twin(h)?;
        self.origin(t)
    }
    /// `true` for a boundary half-edge (no face on this side). `None` for a dead id.
    pub fn is_boundary(&self, h: HalfId) -> Option<bool> {
        self.halfs.get(h.0).map(|x| x.face.is_none())
    }
    pub fn corner_uv(&self, h: HalfId) -> Option<[f64; 2]> {
        self.halfs.get(h.0).map(|x| x.uv)
    }
    pub fn corner_normal(&self, h: HalfId) -> Option<Option<[f64; 3]>> {
        self.halfs.get(h.0).map(|x| x.normal)
    }
    pub fn is_sharp(&self, h: HalfId) -> Option<bool> {
        self.halfs.get(h.0).map(|x| x.sharp)
    }

    /// Whether the undirected edge is a **UV seam** (P23.5).
    pub fn is_seam(&self, h: HalfId) -> Option<bool> {
        self.halfs.get(h.0).map(|x| x.seam)
    }
    pub fn face_slot(&self, f: FaceId) -> Option<Option<u32>> {
        self.faces.get(f.0).map(|x| x.material_slot)
    }
    pub fn face_half(&self, f: FaceId) -> Option<HalfId> {
        self.faces.get(f.0).map(|x| x.half)
    }
    /// Half-edges leaving `v`, ascending.
    pub fn vert_outgoing(&self, v: VertId) -> Option<&[HalfId]> {
        self.verts.get(v.0).map(|x| x.out.as_slice())
    }
    /// The material slot names, in slot order.
    pub fn material_slots(&self) -> &[String] {
        &self.material_slots
    }
    // ── the skin channel (P24.2) ───────────────────────────────────────────

    /// What this mesh's joint indices mean, or `None` when it is rigid.
    pub fn skin_binding(&self) -> Option<SkinBinding> {
        self.skin
    }

    /// Whether this mesh is bound to a skeleton.
    pub fn is_skinned(&self) -> bool {
        self.skin.is_some()
    }

    /// A vertex's skinning influences. `None` for a dead id — never a silent
    /// default, so a stale selection reads as absent rather than as "rigid".
    pub fn vert_weights(&self, v: VertId) -> Option<VertWeights> {
        self.verts.get(v.0).map(|x| x.skin)
    }

    /// Bind (or re-bind) the skin channel. Internal: [`crate::ops::Op::BindSkin`]
    /// is the only door, so a binding cannot appear without a journal entry.
    pub(crate) fn set_skin_binding(&mut self, binding: Option<SkinBinding>) {
        self.skin = binding;
    }

    /// Overwrite a vertex's influences. Internal, for the same reason: the
    /// journalled ops are the only writers.
    pub(crate) fn set_vert_weights(&mut self, v: VertId, w: VertWeights) {
        if let Some(rec) = self.verts.get_mut(v.0) {
            rec.skin = w;
        }
    }

    /// The submesh name recorded for a slot, if the mesh was imported from an
    /// asset that had one.
    pub fn slot_name(&self, slot: Option<u32>) -> Option<&str> {
        self.slot_names
            .iter()
            .find(|(k, _)| *k == slot)
            .map(|(_, n)| n.as_str())
    }

    /// The half-edge from `a` to `b`, if the directed edge exists. `O(deg a)`.
    pub fn find_half(&self, a: VertId, b: VertId) -> Option<HalfId> {
        let va = self.verts.get(a.0)?;
        va.out.iter().copied().find(|&h| self.dest(h) == Some(b))
    }

    /// The half-edges of a face's loop, starting at [`Mesh::face_half`], in
    /// `next` order. `None` for a dead face.
    pub fn face_loop(&self, f: FaceId) -> Option<Vec<HalfId>> {
        let start = self.face_half(f)?;
        Some(self.walk_loop(start).0)
    }

    /// The vertex loop of a face.
    pub fn face_verts(&self, f: FaceId) -> Option<Vec<VertId>> {
        Some(
            self.face_loop(f)?
                .into_iter()
                .map(|h| self.half_ref(h).origin)
                .collect(),
        )
    }

    /// Walk a loop from `start` following `next`. Returns the half-edges visited
    /// and whether the loop closed back on `start` within budget — a broken loop
    /// must be reportable as data, not a hang.
    pub(crate) fn walk_loop(&self, start: HalfId) -> (Vec<HalfId>, bool) {
        let budget = self.halfs.capacity() + 1;
        let mut out = vec![start];
        let mut h = start;
        for _ in 0..budget {
            let Some(rec) = self.halfs.get(h.0) else {
                return (out, false);
            };
            h = rec.next;
            if h == start {
                return (out, true);
            }
            out.push(h);
        }
        (out, false)
    }

    /// Rotate to the next outgoing half-edge around `origin(h)`: `twin(prev(h))`.
    /// Crossing the edge `prev(h)`.
    pub(crate) fn rotate_around_origin(&self, h: HalfId) -> Option<HalfId> {
        let p = self.halfs.get(h.0)?.prev;
        Some(self.halfs.get(p.0)?.twin)
    }

    /// Walk the fan of outgoing half-edges around `v` starting at `h`. Returns
    /// the half-edges visited and whether the walk closed.
    pub(crate) fn walk_fan(&self, h: HalfId) -> (Vec<HalfId>, bool) {
        let budget = self.halfs.capacity() + 1;
        let mut out = vec![h];
        let mut cur = h;
        for _ in 0..budget {
            let Some(nxt) = self.rotate_around_origin(cur) else {
                return (out, false);
            };
            cur = nxt;
            if cur == h {
                return (out, true);
            }
            out.push(cur);
        }
        (out, false)
    }

    /// Is `v`'s one-ring a single closed fan? An isolated vertex is manifold.
    ///
    /// This is the exact bowtie test: the fan walk must close and must cover
    /// *every* outgoing half-edge. Two cones meeting at a point have zero
    /// boundary half-edges, so a "at most one outgoing boundary half-edge" test
    /// would wave them through — which is why the walk is what ops check.
    pub fn vertex_is_manifold(&self, v: VertId) -> bool {
        let Some(rec) = self.verts.get(v.0) else {
            return false;
        };
        let Some(&first) = rec.out.first() else {
            return true;
        };
        let (fan, closed) = self.walk_fan(first);
        closed && fan.len() == rec.out.len()
    }

    // ── internal, panicking accessors ──────────────────────────────────────

    #[inline]
    pub(crate) fn vert_ref(&self, v: VertId) -> &Vert {
        self.verts.get(v.0).expect("live vertex id")
    }
    #[inline]
    pub(crate) fn half_ref(&self, h: HalfId) -> &HalfEdge {
        self.halfs.get(h.0).expect("live half-edge id")
    }
    #[inline]
    pub(crate) fn face_ref(&self, f: FaceId) -> &Face {
        self.faces.get(f.0).expect("live face id")
    }
    #[inline]
    pub(crate) fn half_mut(&mut self, h: HalfId) -> &mut HalfEdge {
        self.halfs.get_mut(h.0).expect("live half-edge id")
    }
    #[inline]
    pub(crate) fn vert_mut(&mut self, v: VertId) -> &mut Vert {
        self.verts.get_mut(v.0).expect("live vertex id")
    }
    #[inline]
    pub(crate) fn face_mut(&mut self, f: FaceId) -> &mut Face {
        self.faces.get_mut(f.0).expect("live face id")
    }

    // ── low-level surgery ──────────────────────────────────────────────────

    /// Create an isolated vertex, **rigid**.
    ///
    /// The right door for a vertex with no provenance — `Op::AddVertex`, and the
    /// primitive generators, which build a mesh out of nothing. A vertex a
    /// structural op derives from existing geometry goes through
    /// [`Mesh::alloc_vert_blended`] instead.
    pub(crate) fn alloc_vert(&mut self, position: [f64; 3]) -> VertId {
        VertId(self.verts.alloc(Vert {
            position,
            out: Vec::new(),
            skin: VertWeights::RIGID,
        }))
    }

    /// Create an isolated vertex whose skinning influences are **blended from
    /// the vertices it came out of**.
    ///
    /// `sources` is `(vertex, coefficient)`; the coefficients need not sum to 1
    /// (the blend renormalizes), and a dead source contributes nothing. A
    /// split-edge midpoint passes its two endpoints at `1 − t` / `t`, a
    /// subdivision centroid its face's corners equally, an extruded or mirrored
    /// copy its single source at 1.
    ///
    /// **On a rigid mesh this is `alloc_vert`, bit for bit**: every source is
    /// [`VertWeights::RIGID`], so the accumulated table is `{0: Σcoeff}` and
    /// normalizing it gives `RIGID` back exactly. That is what lets every
    /// structural op adopt this door without moving a single byte of an
    /// unskinned session.
    pub(crate) fn alloc_vert_blended(
        &mut self,
        position: [f64; 3],
        sources: &[(VertId, f64)],
    ) -> VertId {
        let mut pairs: Vec<(u16, f64)> = Vec::new();
        for &(v, coeff) in sources {
            let Some(rec) = self.verts.get(v.0) else {
                continue;
            };
            for (j, w) in rec.skin.joints.iter().zip(&rec.skin.weights) {
                pairs.push((*j, *w as f64 * coeff));
            }
        }
        let skin = VertWeights::from_pairs(&pairs);
        VertId(self.verts.alloc(Vert {
            position,
            out: Vec::new(),
            skin,
        }))
    }

    /// Remove an **isolated** vertex. Caller checks isolation.
    pub(crate) fn free_vert(&mut self, v: VertId) {
        self.verts.remove(v.0);
    }

    fn insert_out(&mut self, v: VertId, h: HalfId) {
        let out = &mut self.verts.get_mut(v.0).expect("live vertex id").out;
        let at = out.partition_point(|&x| x < h);
        out.insert(at, h);
    }

    fn remove_out(&mut self, v: VertId, h: HalfId) {
        if let Some(rec) = self.verts.get_mut(v.0) {
            rec.out.retain(|&x| x != h);
        }
    }

    /// Allocate the twin pair `a→b` / `b→a`. Both halves start as boundary
    /// half-edges linked to themselves; the caller links them properly.
    fn alloc_pair(&mut self, a: VertId, b: VertId) -> HalfId {
        let blank = |origin: VertId| HalfEdge {
            origin,
            twin: HalfId(u32::MAX),
            next: HalfId(u32::MAX),
            prev: HalfId(u32::MAX),
            face: None,
            uv: [0.0, 0.0],
            normal: None,
            sharp: false,
            seam: false,
        };
        let h = HalfId(self.halfs.alloc(blank(a)));
        let t = HalfId(self.halfs.alloc(blank(b)));
        {
            let rec = self.half_mut(h);
            rec.twin = t;
            rec.next = h;
            rec.prev = h;
        }
        {
            let rec = self.half_mut(t);
            rec.twin = h;
            rec.next = t;
            rec.prev = t;
        }
        self.insert_out(a, h);
        self.insert_out(b, t);
        h
    }

    /// Free a twin pair. Both halves must already be face-less.
    fn free_pair(&mut self, h: HalfId) {
        let t = self.half_ref(h).twin;
        let (oh, ot) = (self.half_ref(h).origin, self.half_ref(t).origin);
        self.remove_out(oh, h);
        self.remove_out(ot, t);
        // Free the higher id first so a LIFO pop returns the lower one first —
        // one fixed order, so allocation stays a pure function of the op sequence.
        let (lo, hi) = if h.0 < t.0 { (h, t) } else { (t, h) };
        self.halfs.remove(hi.0);
        self.halfs.remove(lo.0);
    }

    /// Rebuild the boundary `next`/`prev` links through `v`.
    ///
    /// A manifold vertex lies on at most one boundary loop, so it has at most one
    /// outgoing and one incoming boundary half-edge and the link between them is
    /// forced. Returns `false` when that is not the case — the caller (always
    /// inside a `Mesh::transact`) then refuses and the mutation is discarded.
    fn relink_boundary_at(&mut self, v: VertId) -> bool {
        let out = self.vert_ref(v).out.clone();
        let mut outgoing = None;
        let mut incoming = None;
        for h in out {
            if self.half_ref(h).face.is_none() {
                if outgoing.is_some() {
                    return false;
                }
                outgoing = Some(h);
            }
            let t = self.half_ref(h).twin;
            if self.half_ref(t).face.is_none() {
                if incoming.is_some() {
                    return false;
                }
                incoming = Some(t);
            }
        }
        match (incoming, outgoing) {
            (Some(i), Some(o)) => {
                self.half_mut(i).next = o;
                self.half_mut(o).prev = i;
                true
            }
            (None, None) => true,
            // One without the other cannot happen on a consistent twin pairing:
            // a boundary loop that enters a vertex leaves it again.
            _ => false,
        }
    }

    /// Add a face over an existing vertex loop.
    ///
    /// Preconditions, all checked before any mutation:
    /// * at least 3 vertices, all live, pairwise distinct;
    /// * no directed edge of the loop already carries a face (that would put two
    ///   faces on the same side of an edge — non-manifold).
    ///
    /// Vertex-manifoldness is checked *after* linking by the caller's
    /// `Mesh::transact`, because a bowtie is a property of the resulting fan,
    /// not of any pre-state predicate.
    pub(crate) fn add_face_raw(
        &mut self,
        loop_verts: &[VertId],
        corners: &[CornerData],
        slot: Option<u32>,
    ) -> Result<FaceId, OpError> {
        let n = loop_verts.len();
        if n < 3 {
            return Err(OpError::FaceTooSmall { verts: n });
        }
        if corners.len() != n {
            return Err(OpError::CornerCountMismatch {
                verts: n,
                corners: corners.len(),
            });
        }
        for (i, &v) in loop_verts.iter().enumerate() {
            if !self.has_vert(v) {
                return Err(OpError::NoSuchVert(v));
            }
            if loop_verts[..i].contains(&v) {
                return Err(OpError::RepeatedVertex(v));
            }
        }
        for i in 0..n {
            let (a, b) = (loop_verts[i], loop_verts[(i + 1) % n]);
            if let Some(h) = self.find_half(a, b) {
                if self.half_ref(h).face.is_some() {
                    return Err(OpError::EdgeAlreadyFaced { from: a, to: b });
                }
            }
        }

        let f = FaceId(self.faces.alloc(Face {
            half: HalfId(0),
            material_slot: slot,
        }));
        let mut halfs = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (loop_verts[i], loop_verts[(i + 1) % n]);
            let h = match self.find_half(a, b) {
                Some(h) => h,
                None => self.alloc_pair(a, b),
            };
            halfs.push(h);
        }
        for i in 0..n {
            let h = halfs[i];
            let nx = halfs[(i + 1) % n];
            let pv = halfs[(i + n - 1) % n];
            let rec = self.half_mut(h);
            rec.face = Some(f);
            rec.next = nx;
            rec.prev = pv;
            rec.uv = corners[i].uv;
            rec.normal = corners[i].normal;
        }
        self.faces.get_mut(f.0).expect("just allocated").half = halfs[0];
        Ok(f)
    }

    /// Detach a face, capturing what it takes with it. The half-edges it owned
    /// become boundary half-edges; any edge that ends up face-less on both sides
    /// is freed (no wire edges).
    ///
    /// Does **not** relink boundaries — the caller does that once for the whole
    /// patch it is rebuilding.
    pub(crate) fn remove_face_raw(&mut self, f: FaceId) -> FacePatch {
        let halfs = self.face_loop(f).expect("live face id");
        let mut verts = Vec::with_capacity(halfs.len());
        let mut corners = Vec::with_capacity(halfs.len());
        for &h in &halfs {
            let rec = self.half_ref(h);
            verts.push(rec.origin);
            corners.push(CornerData {
                uv: rec.uv,
                normal: rec.normal,
            });
        }
        let slot = self.face_ref(f).material_slot;
        for &h in &halfs {
            self.half_mut(h).face = None;
        }
        for &h in &halfs {
            let t = self.half_ref(h).twin;
            if self.half_ref(t).face.is_none() {
                self.free_pair(h);
            }
        }
        self.faces.remove(f.0);
        FacePatch {
            verts,
            corners,
            slot,
        }
    }

    /// The per-edge flags of the undirected edges a set of faces touches, as
    /// `(from, to, flags)` triples — captured before a rebuild so they survive
    /// an op that re-creates the local patch.
    ///
    /// **One capture for both flags** (P23.5). Sharpness had this treatment from
    /// P23.3 and the seam flag needed exactly the same one at exactly the same
    /// dozen call sites; a second `capture_seam`/`apply_seam` pair would have
    /// been a dozen more places for an op to remember one and forget the other,
    /// and the forgetting would be invisible until an author's UV cut vanished
    /// three bevels later.
    pub(crate) fn capture_edge_flags(&self, faces: &[FaceId]) -> Vec<(VertId, VertId, EdgeFlags)> {
        let mut out = Vec::new();
        for &f in faces {
            let Some(loop_halfs) = self.face_loop(f) else {
                continue;
            };
            for h in loop_halfs {
                let rec = self.half_ref(h);
                if rec.sharp || rec.seam {
                    let d = self.dest(h).expect("live half-edge id");
                    out.push((
                        rec.origin,
                        d,
                        EdgeFlags {
                            sharp: rec.sharp,
                            seam: rec.seam,
                        },
                    ));
                }
            }
        }
        out
    }

    /// Set the sharp flag on **both** halves of an edge — the only way the flag
    /// is ever written, so the twin agreement invariant cannot be broken by a
    /// caller that forgot the other side.
    pub(crate) fn set_sharp_pair(&mut self, h: HalfId, sharp: bool) {
        let t = self.half_ref(h).twin;
        self.half_mut(h).sharp = sharp;
        self.half_mut(t).sharp = sharp;
    }

    /// Set the seam flag on **both** halves. Same rule, same reason.
    pub(crate) fn set_seam_pair(&mut self, h: HalfId, seam: bool) {
        let t = self.half_ref(h).twin;
        self.half_mut(h).seam = seam;
        self.half_mut(t).seam = seam;
    }

    /// Re-apply captured edge flags to whatever edges still exist.
    pub(crate) fn apply_edge_flags(&mut self, captured: &[(VertId, VertId, EdgeFlags)]) {
        for &(a, b, flags) in captured {
            if !self.has_vert(a) || !self.has_vert(b) {
                continue;
            }
            if let Some(h) = self.find_half(a, b) {
                let t = self.half_ref(h).twin;
                for x in [h, t] {
                    let rec = self.half_mut(x);
                    rec.sharp = flags.sharp;
                    rec.seam = flags.seam;
                }
            }
        }
    }

    /// Finish a structural edit: relink the boundary loops through `touched` and
    /// check that every touched vertex is still manifold.
    pub(crate) fn finish_patch(&mut self, touched: &BTreeSet<VertId>) -> Result<(), OpError> {
        for &v in touched {
            if !self.has_vert(v) {
                continue;
            }
            if !self.relink_boundary_at(v) {
                return Err(OpError::NonManifoldVertex(v));
            }
        }
        for &v in touched {
            if !self.has_vert(v) {
                continue;
            }
            if !self.vertex_is_manifold(v) {
                return Err(OpError::NonManifoldVertex(v));
            }
        }
        Ok(())
    }

    /// Run a structural edit on a clone and commit only on success — the seam
    /// that makes **a refused op inert**.
    pub(crate) fn transact<T>(
        &mut self,
        edit: impl FnOnce(&mut Mesh) -> Result<T, OpError>,
    ) -> Result<T, OpError> {
        let mut scratch = self.clone();
        let out = edit(&mut scratch)?;
        *self = scratch;
        Ok(out)
    }

    // ── slot bookkeeping ───────────────────────────────────────────────────

    pub(crate) fn set_material_slots(&mut self, slots: Vec<String>) {
        self.material_slots = slots;
    }

    pub(crate) fn set_slot_name(&mut self, slot: Option<u32>, name: String) {
        match self.slot_names.binary_search_by(|(k, _)| k.cmp(&slot)) {
            Ok(_) => {} // first name wins — see `build::from_mesh_asset`
            Err(at) => self.slot_names.insert(at, (slot, name)),
        }
    }

    // ── canonical form ─────────────────────────────────────────────────────

    /// A **labelling-independent** normal form: the comparison the round-trip
    /// gates use.
    ///
    /// Ids are arena indices, so two meshes that differ only in *when* their
    /// elements were allocated are not `==` even though they are the same mesh.
    /// The canonical form removes the labels: faces are sorted by their
    /// rotation-minimal corner-attribute sequence, vertices are then renumbered
    /// by first appearance in that order, and isolated vertices are appended
    /// sorted by position.
    ///
    /// **Honest limit.** The tie-break between two faces whose corner data is
    /// *bit-identical* (same positions, UVs, normals, sharpness, slot) falls back
    /// on arena order, so two isomorphic meshes that both contain such a pair
    /// could canonicalize differently. The error is conservative — it reports a
    /// difference where an isomorphism exists, so a gate fails loudly rather than
    /// passing a broken round trip — and no fixture or generator in this crate
    /// produces indistinguishable faces.
    pub fn canonical(&self) -> CanonicalMesh {
        // Sort key per face: the lexicographically smallest rotation of its
        // corner records. Vertex ids are deliberately absent from the key.
        struct Keyed {
            key: Vec<CornerKey>,
            slot: Option<u32>,
            halfs: Vec<HalfId>,
            rot: usize,
        }
        let mut keyed: Vec<Keyed> = Vec::with_capacity(self.face_count());
        for f in self.face_ids() {
            let halfs = self.face_loop(f).expect("live face id");
            let recs: Vec<CornerKey> = halfs.iter().map(|&h| self.corner_key(h)).collect();
            let rot = min_rotation(&recs);
            let n = recs.len();
            let key: Vec<CornerKey> = (0..n).map(|i| recs[(rot + i) % n].clone()).collect();
            keyed.push(Keyed {
                key,
                slot: self.face_ref(f).material_slot,
                halfs,
                rot,
            });
        }
        keyed.sort_by(|a, b| a.key.cmp(&b.key).then(a.slot.cmp(&b.slot)));

        let mut number: BTreeMap<VertId, u32> = BTreeMap::new();
        let mut verts: Vec<[u64; 3]> = Vec::new();
        let mut faces: Vec<CanonicalFace> = Vec::with_capacity(keyed.len());
        for entry in &keyed {
            let n = entry.halfs.len();
            let mut corners = Vec::with_capacity(n);
            for i in 0..n {
                let h = entry.halfs[(entry.rot + i) % n];
                let origin = self.half_ref(h).origin;
                let next_id = verts.len() as u32;
                let vid = *number.entry(origin).or_insert(next_id);
                if vid == next_id {
                    verts.push(bits3(self.vert_ref(origin).position));
                }
                corners.push(CanonicalCorner {
                    vert: vid,
                    uv: entry.key[i].uv,
                    normal: entry.key[i].normal,
                    sharp: entry.key[i].sharp,
                    seam: entry.key[i].seam,
                });
            }
            faces.push(CanonicalFace {
                corners,
                slot: entry.slot,
            });
        }

        // The skin channel rides the same numbering: `vert_skin[i]` is the
        // canonical vertex `i`'s influences, in the LABELLING-INDEPENDENT order
        // (`VertWeights::influences`), so two isomorphic meshes that stored the
        // same influences in different slots canonicalize the same. The stored
        // order is deliberately not normalized — that is what keeps the asset
        // round trip byte-exact — so the normalization has to happen here.
        let mut vert_skin: Vec<CanonicalWeights> = vec![Vec::new(); verts.len()];
        for (&vid, &canonical) in &number {
            vert_skin[canonical as usize] = canonical_weights(self.vert_ref(vid).skin);
        }

        let mut isolated: Vec<([u64; 3], CanonicalWeights)> = self
            .vert_ids()
            .filter(|v| !number.contains_key(v))
            .map(|v| {
                let rec = self.vert_ref(v);
                (bits3(rec.position), canonical_weights(rec.skin))
            })
            .collect();
        isolated.sort_unstable();
        for (p, w) in isolated {
            verts.push(p);
            vert_skin.push(w);
        }

        CanonicalMesh {
            verts,
            faces,
            material_slots: self.material_slots.clone(),
            skin: self.skin,
            vert_skin,
        }
    }

    fn corner_key(&self, h: HalfId) -> CornerKey {
        let rec = self.half_ref(h);
        CornerKey {
            position: bits3(self.vert_ref(rec.origin).position),
            uv: bits2(rec.uv),
            normal: rec.normal.map(bits3),
            sharp: rec.sharp,
            seam: rec.seam,
        }
    }

    /// The bincode encoding of the whole mesh — the crate's definition of
    /// "byte-identical", used by every determinism gate.
    pub fn encoded(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("a Mesh has no un-encodable state")
    }
}

/// The labelling-independent normal form produced by [`Mesh::canonical`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonicalMesh {
    /// Vertex positions as normalized bit patterns, in canonical order.
    pub verts: Vec<[u64; 3]>,
    pub faces: Vec<CanonicalFace>,
    pub material_slots: Vec<String>,
    /// The skin binding (P24.2), or `None` for a rigid mesh.
    pub skin: Option<SkinBinding>,
    /// Each canonical vertex's influences as `(joint, weight bits)`, zero-weight
    /// slots dropped and sorted descending by weight — **index-aligned to
    /// `verts`**, so a mesh's weights are compared in the same labelling-free
    /// frame as its geometry.
    ///
    /// Bits rather than `f32` because this type is `Eq` + `Ord` (it is the
    /// comparison every round-trip gate makes), and because a weight that
    /// differs by one ULP is a different weight — a tolerance here would hide
    /// exactly the drift the gates exist to catch.
    pub vert_skin: Vec<CanonicalWeights>,
}

/// One vertex's influences in [`CanonicalMesh`]'s labelling-free form:
/// `(joint, weight bits)`, zero slots dropped, descending weight.
///
/// A named alias rather than the bare `Vec<(u16, u32)>` because the pair means
/// nothing on sight — `(u16, u32)` reads as two indices — and because it appears
/// in three signatures that have to agree.
pub type CanonicalWeights = Vec<(u16, u32)>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonicalFace {
    pub corners: Vec<CanonicalCorner>,
    pub slot: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonicalCorner {
    pub vert: u32,
    pub uv: [u64; 2],
    pub normal: Option<[u64; 3]>,
    pub sharp: bool,
    pub seam: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CornerKey {
    position: [u64; 3],
    uv: [u64; 2],
    normal: Option<[u64; 3]>,
    sharp: bool,
    seam: bool,
}

/// The per-edge boolean flags carried across a structural rebuild.
///
/// A struct rather than two loose `bool`s so [`Mesh::capture_edge_flags`] and
/// [`Mesh::apply_edge_flags`] cannot be extended for one flag and not the other:
/// adding a third stops every call site compiling until it is filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct EdgeFlags {
    pub(crate) sharp: bool,
    pub(crate) seam: bool,
}

/// `f64` → a comparable bit pattern, with `-0.0` folded onto `+0.0` so an op
/// that produces a negative zero does not read as a different mesh.
fn bits(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}
fn bits2(v: [f64; 2]) -> [u64; 2] {
    [bits(v[0]), bits(v[1])]
}
fn bits3(v: [f64; 3]) -> [u64; 3] {
    [bits(v[0]), bits(v[1]), bits(v[2])]
}

/// One vertex's influences in the canonical (labelling-independent) form used by
/// [`CanonicalMesh::vert_skin`]: zero slots dropped, descending weight, ties by
/// ascending joint, weights as `f32` bit patterns with `-0.0` folded.
fn canonical_weights(w: VertWeights) -> CanonicalWeights {
    w.influences()
        .into_iter()
        .map(|(j, x)| (j, if x == 0.0 { 0 } else { x.to_bits() }))
        .collect()
}

/// Index of the rotation that makes `items` lexicographically smallest.
fn min_rotation<T: Ord>(items: &[T]) -> usize {
    let n = items.len();
    let mut best = 0usize;
    for start in 1..n {
        let mut ord = std::cmp::Ordering::Equal;
        for i in 0..n {
            ord = items[(start + i) % n].cmp(&items[(best + i) % n]);
            if ord != std::cmp::Ordering::Equal {
                break;
            }
        }
        if ord == std::cmp::Ordering::Less {
            best = start;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::validate::validate;

    #[test]
    fn twin_is_total_and_involutive_on_a_cube() {
        let m = cube(1.0);
        for h in m.half_ids() {
            let t = m.twin(h).unwrap();
            assert_ne!(t, h, "no half-edge is its own twin");
            assert_eq!(m.twin(t), Some(h));
            assert_eq!(m.origin(t), m.dest(h));
        }
        assert_eq!(m.half_count(), 24, "12 edges × 2");
        assert_eq!(m.edge_count(), 12);
    }

    #[test]
    fn a_plane_has_a_boundary_loop_and_a_cube_has_none() {
        let p = plane(1.0);
        let b: Vec<HalfId> = p
            .half_ids()
            .filter(|&h| p.is_boundary(h) == Some(true))
            .collect();
        assert_eq!(b.len(), 4, "the quad's four outer half-edges");
        let (walked, closed) = p.walk_loop(b[0]);
        assert!(closed, "the boundary is a closed loop");
        assert_eq!(walked.len(), 4);

        let c = cube(1.0);
        assert!(c.half_ids().all(|h| c.is_boundary(h) == Some(false)));
    }

    #[test]
    fn dead_ids_read_as_none_rather_than_panicking() {
        let m = cube(1.0);
        let ghost = VertId(9_999);
        assert_eq!(m.position(ghost), None);
        assert_eq!(m.vert_outgoing(ghost), None);
        assert_eq!(m.find_half(ghost, VertId(0)), None);
        assert_eq!(m.face_loop(FaceId(9_999)), None);
    }

    #[test]
    fn free_list_is_lifo_so_allocation_is_a_pure_function_of_the_call_order() {
        let mut m = Mesh::new();
        let a = m.alloc_vert([0.0; 3]);
        let b = m.alloc_vert([1.0, 0.0, 0.0]);
        assert_eq!((a, b), (VertId(0), VertId(1)));
        m.free_vert(a);
        let c = m.alloc_vert([2.0, 0.0, 0.0]);
        assert_eq!(c, a, "the freed slot comes back, by id");
        assert_eq!(m.vert_count(), 2);
    }

    #[test]
    fn canonical_form_ignores_labelling() {
        // The same square built with its loop started at a different corner is
        // the same mesh; the arenas differ, the canonical forms do not.
        let mut a = Mesh::new();
        let av: Vec<VertId> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
            .iter()
            .map(|p| a.alloc_vert(*p))
            .collect();
        a.add_face_raw(&av, &[CornerData::default(); 3], None)
            .unwrap();
        a.finish_patch(&av.iter().copied().collect()).unwrap();

        let mut b = Mesh::new();
        // Allocate a decoy vertex first so every id differs, then start the loop
        // at a different corner.
        let decoy = b.alloc_vert([9.0, 9.0, 9.0]);
        b.free_vert(decoy);
        let bv: Vec<VertId> = [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            .iter()
            .map(|p| b.alloc_vert(*p))
            .collect();
        b.add_face_raw(&bv, &[CornerData::default(); 3], None)
            .unwrap();
        b.finish_patch(&bv.iter().copied().collect()).unwrap();

        assert!(validate(&a).is_ok());
        assert!(validate(&b).is_ok());
        assert_eq!(a.canonical(), b.canonical());
    }

    #[test]
    fn a_bowtie_vertex_is_not_manifold() {
        // Two triangles sharing exactly one vertex.
        let mut m = Mesh::new();
        let shared = m.alloc_vert([0.0, 0.0, 0.0]);
        let a1 = m.alloc_vert([1.0, 0.0, 0.0]);
        let a2 = m.alloc_vert([1.0, 1.0, 0.0]);
        let b1 = m.alloc_vert([-1.0, 0.0, 0.0]);
        let b2 = m.alloc_vert([-1.0, -1.0, 0.0]);
        m.add_face_raw(&[shared, a1, a2], &[CornerData::default(); 3], None)
            .unwrap();
        m.add_face_raw(&[shared, b1, b2], &[CornerData::default(); 3], None)
            .unwrap();
        let touched: BTreeSet<VertId> = [shared, a1, a2, b1, b2].into_iter().collect();
        assert_eq!(
            m.finish_patch(&touched),
            Err(OpError::NonManifoldVertex(shared)),
            "the shared vertex carries two boundary loops"
        );
    }

    #[test]
    fn encoded_is_stable_across_two_runs() {
        assert_eq!(cube(1.0).encoded(), cube(1.0).encoded());
    }
}
