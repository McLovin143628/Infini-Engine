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
//!    through [`crate::journal::MeshSession`] records it, and replaying the
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
//! ([`crate::export::ExportOptions::optimize`]).

use std::collections::BTreeSet;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::model::{self, KnifePoint, MergeTarget, MirrorAxis};
use crate::topo::{CornerData, EdgeFlags, FaceId, FacePatch, HalfId, Mesh, VertId};

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
    /// A NaN or infinity was handed to an op.
    ///
    /// Refused rather than stored, because a non-finite coordinate is not a
    /// value the author can see or find later: it prints as `NaN` in a Details
    /// panel, it makes the mesh's bounds exclude its own geometry, and every
    /// comparison it takes part in is false — including the ones a later
    /// refusal would rely on. The read path already treats it as a
    /// [`crate::validate::Violation`]; this is the same rule at the edit door.
    #[error("{what} is not finite: {value:?}")]
    NonFinite { what: &'static str, value: Vec<f64> },
    /// An authored corner normal that is not a unit vector.
    ///
    /// "Authored" means the writer copies it out verbatim, so it has to be a
    /// real normal. A caller that wants to *store a direction* should normalize
    /// it; a caller that wants shading derived from the surface should pass
    /// `None` and let the smooth fan do it.
    #[error("an authored corner normal must be unit length, got |n| = {length}")]
    NormalNotUnit { length: f64 },

    // ── the modelling ops (P23.4) ──────────────────────────────────────────
    //
    // `OpError` is not a wire type (it derives neither `Serialize` nor
    // `Deserialize`), so unlike `Op` it may be extended freely — a refusal
    // crosses the IPC boundary as its `Display` string.
    /// An op was handed nothing to work on.
    #[error("{what} was empty")]
    EmptyOperand { what: &'static str },
    /// A size that has to be strictly positive was not.
    #[error("{what} must be positive, got {value}")]
    AmountNotPositive { what: &'static str, value: f64 },
    /// A count outside the range its op accepts.
    #[error("{what} must be between {min} and {max}, got {value}")]
    CountOutOfRange {
        what: &'static str,
        value: u32,
        min: u32,
        max: u32,
    },
    /// A face region whose area-weighted normal cancels out (two opposite faces
    /// selected together) or is degenerate, so an extrude has no direction.
    #[error("the region's {faces} face(s) have no usable normal")]
    DegenerateRegion { faces: usize },
    /// A single face with no usable normal.
    #[error("face {0} is degenerate: it has no usable normal")]
    DegenerateFace(FaceId),
    /// An edge whose two endpoints are at the same place, so it has no direction.
    #[error("edge {0} has zero length")]
    ZeroLengthEdge(HalfId),
    /// An edge extrude was aimed at an edge with a face on both sides. Tearing
    /// the surface open is a different tool (a rip) and is not this one.
    #[error("half-edge {0} is not on a boundary, so there is nothing to extrude into")]
    EdgeNotBoundary(HalfId),
    /// A bevel needs a face on both sides to offset into.
    #[error("edge {0} is on a boundary; a bevel needs a face on both sides")]
    BevelBoundaryEdge(HalfId),
    /// **The bevel would build a mesh its own writer cannot save.**
    ///
    /// Two beveled edges that share an endpoint each push an offset copy of that
    /// endpoint into their far faces, and on a right-angled corner those two
    /// copies land at *exactly the same place*. The kernel can hold two vertices
    /// at one position; a `MeshAsset` cannot — [`crate::from_mesh_asset`] welds
    /// positions at a tolerance of **exactly zero**, so the pair fuses on the way
    /// back in and an edge ends up used twice, which the reader refuses as
    /// non-manifold.
    ///
    /// Refused rather than emitted, because a modelling op that hands back a file
    /// nobody can re-open — with only a save-time advisory in between — is the
    /// failure shape this codebase keeps paying for. The refusal is **inert**
    /// (`Mesh::transact`), so the mesh is byte-identical afterwards.
    ///
    /// The remedy is in the message and it is honest: bevel edges that do not
    /// meet. A **corner join** for meeting bevels — dissolving the shared vertex
    /// into a patch — is the missing feature, and it is not built.
    #[error(
        "beveling edge {edge} would put a second vertex at ({}, {}, {}), where this \
         bevel has already placed one: two beveled edges meeting at a corner each \
         offset that corner into their far face, and on a right angle both land in \
         the same place. Saving that produces a mesh the reader refuses as \
         non-manifold. Bevel edges that do not share an endpoint; a corner join for \
         meeting bevels is not built.",
        at[0], at[1], at[2]
    )]
    BevelCoincidentVertex { edge: HalfId, at: [f64; 3] },
    /// A region border that passes through one vertex twice — a pinch — so the
    /// inset corner has no single miter.
    #[error("vertex {0} is on the region border more than once, so its inset corner is ambiguous")]
    AmbiguousRegionVertex(VertId),
    /// A border corner that folds back on itself (its two edge normals oppose),
    /// where the miter offset is unbounded.
    #[error("vertex {0}'s inset corner folds back on itself")]
    DegenerateInsetCorner(VertId),
    /// Every edge of the region has a region face on both sides, so there is no
    /// outline to inset.
    #[error("the region has no border, so there is nothing to inset")]
    NoRegionBorder,
    /// An edge ring can only start inside a quad.
    #[error("face {0} is not a quad, so an edge ring cannot start there")]
    NotAQuad(FaceId),
    /// Two consecutive knife waypoints are not corners of any one face.
    #[error("vertices {from} and {to} share no face, so the knife cannot cut between them")]
    KnifeNoCommonFace { from: VertId, to: VertId },
    /// An offset large enough to fold a rebuilt face through itself.
    ///
    /// `validate` is a **topology** auditor and is blind to winding, so an
    /// inside-out solid is a perfectly valid mesh to it. Refused rather than
    /// clamped: the limit belongs to the author's geometry, not to the op, and a
    /// clamp would silently produce a shape they did not ask for.
    #[error("{what} of {amount} turns the geometry inside out; use a smaller one")]
    WouldInvert { what: &'static str, amount: f64 },

    // ── the sculpt / transform / UV ops (P23.5) ────────────────────────────
    /// A rotation whose axis has no direction, so there is nothing to rotate
    /// about. Refused rather than treated as "no rotation": a caller that handed
    /// over a zero axis has a bug, and silently applying the identity hides it.
    #[error("a rotation axis must have a direction, got {axis:?}")]
    ZeroAxis { axis: [f64; 3] },
    /// An angle so large that reducing it modulo 2π is no longer meaningful.
    ///
    /// Refused rather than reduced-anyway, because beyond `2^52` an `f64` has
    /// less than one radian of resolution: the value is not an angle the author
    /// expressed, it is a magnitude that happens to be stored in the field. The
    /// P23.5 audit found what the alternative was — `psin64` and `pcos64` both
    /// return exactly zero past ~2e16, so Rodrigues quietly became an **axis
    /// projection**, every coordinate stayed finite, `validate` passed (it audits
    /// topology, not geometry) and a quad came back collinear.
    #[error("a rotation of {radians} rad is past the limit of ±{limit}; angles that large are not angles")]
    AngleOutOfRange { radians: f64, limit: f64 },
    /// An unwrap could not parameterize a chart. Carries the chart's index and
    /// the reason, because "unwrap failed" on a 40-chart model is not actionable.
    #[error("chart {chart} cannot be unwrapped: {why}")]
    DegenerateChart { chart: usize, why: &'static str },
    /// An [`Op::Unwrap`] naming a half-edge that is not a live face corner.
    ///
    /// Separate from [`OpError::BoundaryCorner`] so a *replayed* unwrap says what
    /// is wrong with the recording rather than what is wrong with a click.
    #[error("the unwrap names {0}, which is not a live face corner")]
    UnwrapCornerMissing(HalfId),

    // ── the skin ops (P24.2) ───────────────────────────────────────────────
    /// A skin op ran on a mesh that is not bound to a skeleton.
    ///
    /// Refused rather than auto-binding, because a weight table's joint indices
    /// only mean something against a *named* skeleton, and inventing one from an
    /// `AssignWeights` would let two halves of a character bind to two rigs.
    #[error("this mesh is not bound to a skeleton; bind it before assigning weights")]
    NotSkinned,
    /// A binding with no joints. A zero-joint skeleton cannot be a skeleton
    /// ([`inf_anim::SkeletonError::Empty`] says the same thing one crate over),
    /// and every joint index would be out of range against it.
    ///
    /// [`inf_anim::SkeletonError::Empty`]: https://docs.rs/inf-anim
    #[error("a skin binding needs at least one joint")]
    SkinBindingEmpty,
    /// An influence naming a joint the bound skeleton does not have.
    ///
    /// This is also what a **re-bind onto a smaller skeleton** refuses with: the
    /// alternative is clamping the index, which silently re-weights the vertex to
    /// a different bone.
    #[error("vertex {vert} is weighted to joint {joint}, but the skeleton has {joints}")]
    SkinJointOutOfRange {
        vert: VertId,
        joint: u16,
        joints: u32,
    },
    /// An influence set that cannot be normalized: every weight zero, or one of
    /// them negative or non-finite.
    ///
    /// Not silently replaced with [`crate::skin::VertWeights::RIGID`]. That
    /// fallback exists inside the blending helpers, where the caller has said
    /// nothing about the vertex; here the caller has said something specific and
    /// it does not describe a vertex, so it comes back as a refusal.
    #[error("the weights on vertex {vert} do not describe an influence set ({why})")]
    SkinWeightsDegenerate { vert: VertId, why: &'static str },
}

/// Every value crossing an op boundary is checked here, once.
pub(crate) fn finite(what: &'static str, values: &[f64]) -> Result<(), OpError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(OpError::NonFinite {
            what,
            value: values.to_vec(),
        })
    }
}

/// **The op's successor** — what a caller should now be holding, not an
/// inventory of everything the op touched.
///
/// The distinction is the whole value of the type, and getting it wrong is not a
/// cosmetic error. An extrude *creates* a cap and four walls; if it reports both,
/// the selection built from it contains the base vertices, and the very next
/// drag flattens the extrusion the author just made. So:
///
/// | op | outcome |
/// | --- | --- |
/// | [`Op::ExtrudeFaces`] | the moved **cap**, and its vertices |
/// | [`Op::InsetFaces`] | the **inner** faces |
/// | [`Op::BevelEdges`] | the chamfer **strip** |
/// | [`Op::LoopCut`] | the **new loop's edges**, and the cut vertices |
/// | [`Op::Knife`] | the **cut edges** |
/// | [`Op::SubdivideFaces`] | the region's sub-faces (never the fringe it re-cut) |
///
/// Ids of rebuilt elements change (see the module docs), which is why this is the
/// only sound way for a caller to keep working: the ids it held a moment ago may
/// name something else, and these do not.
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

    // ── the modelling ops (P23.4) — APPENDED, never inserted ───────────────
    //
    // These nine variants take wire positions 13..=21. They were **appended**
    // deliberately: `frozen_discriminant` in `crate::journal`'s tests is a
    // `match` with no wildcard precisely so that adding one of these stopped the
    // crate compiling until an author gave it the next index by hand, and the
    // existing pins (`Op::CollapseEdge { half: 7 }` encodes `[5, 7]`) are
    // unchanged by construction. `SessionSave::CURRENT_VERSION` therefore stays
    // at 1: every session ever written is still read correctly by this build.
    // A session written by THIS build and read by an older one fails its bincode
    // decode on an unknown discriminant, loudly, which is the right answer.
    /// Extrude a face region along its own area-weighted normal, walling only
    /// the region **border** so a multi-face selection moves as one block.
    ExtrudeFaces { faces: Vec<FaceId>, distance: f64 },
    /// Extrude boundary edges into new faces by an explicit delta (metres) — an
    /// edge has no canonical direction, so the caller supplies one.
    ExtrudeEdges { edges: Vec<HalfId>, delta: [f64; 3] },
    /// Inset a face region's outline (or every face's own outline when
    /// `individual`), filling the gap with a ring of quads.
    InsetFaces {
        faces: Vec<FaceId>,
        amount: f64,
        individual: bool,
    },
    /// Chamfer interior edges — one segment (see [`crate::model`]).
    BevelEdges { edges: Vec<HalfId>, amount: f64 },
    /// Cut `cuts` parallel loops across the quad strip through `half`.
    LoopCut { half: HalfId, cuts: u32 },
    /// Cut across faces along a path of vertices and points on edges, atomically.
    Knife { path: Vec<KnifePoint> },
    /// Fuse vertices into one, at the set's centre or onto its last member.
    MergeVerts {
        verts: Vec<VertId>,
        target: MergeTarget,
    },
    /// Split each face into one quad per corner — simple midpoint, no smoothing.
    SubdivideFaces { faces: Vec<FaceId> },
    /// Reflect the whole mesh across an axis-aligned plane, welding the seam.
    Mirror { axis: MirrorAxis, coord: f64 },

    // ── the sculpt / gizmo ops (P23.5) — APPENDED, never inserted ──────────
    //
    // Wire positions 22..=24, taken by the same discipline as P23.4's nine: the
    // `frozen_discriminant` match in `crate::journal`'s tests has no wildcard, so
    // each of these stopped the crate compiling until it was given the next free
    // index by hand. `Op::CollapseEdge { half: 7 }` still encodes `[5, 7]`.
    /// **A whole brush stroke as one entry** — every dab centre of one
    /// mouse-down→up gesture, replayed as data (see [`crate::sculpt`]).
    ///
    /// The journal has no transactions, so the op *is* the transaction: one
    /// journal entry per stroke and one undo step per stroke, which is the
    /// terrain `Stroke::begin`/`add_dab`/`finish` doctrine adapted to a structure
    /// whose atom is already an op.
    Sculpt {
        mode: crate::sculpt::SculptMode,
        /// Dab centres in metres, **already** arc-length resampled by
        /// [`crate::sculpt::stroke_dabs`], so replay does not depend on the
        /// resampler.
        dabs: Vec<[f64; 3]>,
        radius: f64,
        strength: f64,
        falloff: crate::sculpt::SculptFalloff,
    },
    /// Rotate vertices about a pivot. See [`crate::xform`] for why the pivot is
    /// on the wire and the soft-select weight is not.
    RotateVerts {
        verts: Vec<VertId>,
        pivot: [f64; 3],
        axis: [f64; 3],
        radians: f64,
    },
    /// Scale vertices about a pivot, per axis.
    ScaleVerts {
        verts: Vec<VertId>,
        pivot: [f64; 3],
        factor: [f64; 3],
    },

    // ── the UV ops (P23.5) — appended at 25 and 26 ─────────────────────────
    /// Mark or clear a **UV seam** on an undirected edge. Written through both
    /// halves, exactly like [`Op::SetEdgeSharp`].
    ///
    /// One op with a bool rather than the `MarkSeam` / `ClearSeam` pair the spec
    /// sketched: `SetEdgeSharp` is the same statement about the same storage in
    /// the same enum, and two conventions for one idea is how a wire format
    /// starts drifting. "Mark" and "clear" are the two values of `seam`.
    SetEdgeSeam { half: HalfId, seam: bool },
    /// **The result of an unwrap**, as per-corner UVs.
    ///
    /// The op carries the computed layout, and replay does **not** re-solve. The
    /// solver is deterministic today (see [`crate::uv`]) and this is not a
    /// hedge against that — it is the meshopt lesson applied before it can bite:
    /// a journal is replayed by a *different build* than wrote it, so an op whose
    /// effect is "whatever the current solver says" means improving the solver
    /// silently rewrites every session ever recorded. Carrying the values makes
    /// an unwrap a fact, and the solver a tool that produced it.
    ///
    /// `corners` is `(half-edge, uv)` sorted by half-edge — deterministic on the
    /// wire and cheap to verify.
    Unwrap { corners: Vec<(HalfId, [f64; 2])> },

    // ── the skin ops (P24.2) — appended at 27, 28 and 29 ───────────────────
    //
    // Same discipline, fourth time: `frozen_discriminant` has no wildcard, so
    // each of these stopped the crate compiling until it was given the next free
    // index by hand, and `Op::CollapseEdge { half: 7 }` still encodes `[5, 7]`.
    // `SessionSave::CURRENT_VERSION` DID move for this batch — not for these
    // three, but because `Vert` and `Mesh` grew skin fields and the `base` a save
    // carries is therefore a different shape.
    /// **Bind the mesh to a skeleton**, giving its joint indices a meaning.
    ///
    /// Every vertex keeps the weights it had — which, on a mesh that has never
    /// been skinned, is [`crate::skin::VertWeights::RIGID`] everywhere: 100 % of
    /// joint 0. A newly bound character therefore rides its root joint, which is
    /// wrong-looking and *valid*, and is exactly the state the auto-fit and the
    /// weight solve exist to replace.
    ///
    /// Re-binding is allowed and is how a rig change lands; it refuses only when
    /// the new skeleton has too few joints for the weights already on the mesh
    /// ([`OpError::SkinJointOutOfRange`]), because silently clamping an index
    /// re-weights a character to a different bone.
    BindSkin {
        /// The `.inf_skel` GUID, raw bytes (this crate has no `uuid` dep), or
        /// `None` when the caller genuinely does not know — see
        /// [`crate::skin::SkinBinding`].
        skeleton: Option<[u8; 16]>,
        /// The skeleton's joint count at bind time.
        joints: u32,
    },
    /// **The weight table, as a fact** — `(vertex, influences)` pairs, sorted by
    /// vertex.
    ///
    /// This is the ledgered weight-table op P23 wrote down and did not build.
    /// It carries the *values*, and replay does **not** re-solve, for the reason
    /// [`Op::Unwrap`] carries its UVs: a journal is replayed by a different build
    /// than wrote it, so an op whose effect is "whatever the current solver says"
    /// means improving the solver silently rewrites every session ever recorded.
    /// The heat solve and the paint brush both *produce* one of these; neither is
    /// in the journal.
    ///
    /// Weights are renormalized on apply, so a caller may hand raw sums; the
    /// normalization is `f32` arithmetic on stated values and is therefore still
    /// a pure function of the op.
    AssignWeights {
        /// `(vertex, influences)`, sorted by vertex id — deterministic on the
        /// wire and cheap to verify.
        weights: Vec<(VertId, crate::skin::VertWeights)>,
    },
    /// **Unbind**, and reset every vertex to
    /// [`crate::skin::VertWeights::RIGID`].
    ///
    /// Both halves, deliberately: an unbind that left the weights behind would
    /// leave a mesh carrying joint indices nothing can interpret, which
    /// [`crate::validate`] calls a violation
    /// ([`crate::validate::Violation::SkinWithoutBinding`]). "Clear the skin"
    /// means the mesh is rigid again, and rigid has one representation.
    ClearSkin,
}

/// Apply one op. See the module docs for the three rules this upholds.
pub fn apply(mesh: &mut Mesh, op: &Op) -> Result<OpOutcome, OpError> {
    match op {
        Op::AddVertex { position } => {
            finite("a vertex position", position)?;
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
            for c in corners {
                check_corner(c)?;
            }
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
            finite("a translation delta", delta)?;
            for v in verts {
                if !mesh.has_vert(*v) {
                    return Err(OpError::NoSuchVert(*v));
                }
            }
            // The RESULT, not just the operand. Two finite values can add to an
            // infinity — a vertex at 1e308 translated by 1e308 — so checking the
            // inputs and storing the sum is a gate that reads the wrong number.
            // Computed for every vertex before the first write, so the refusal
            // stays inert without a transaction.
            let moved: Vec<[f64; 3]> = verts
                .iter()
                .map(|v| {
                    let p = mesh.vert_ref(*v).position;
                    [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]]
                })
                .collect();
            for p in &moved {
                finite("a translated vertex position", p)?;
            }
            for (v, p) in verts.iter().zip(moved) {
                mesh.vert_mut(*v).position = p;
            }
            Ok(OpOutcome::default())
        }

        Op::SetCornerUv { half, uv } => {
            corner_of(mesh, *half)?;
            finite("a corner uv", uv)?;
            mesh.half_mut(*half).uv = *uv;
            Ok(OpOutcome::default())
        }

        Op::SetCornerNormal { half, normal } => {
            corner_of(mesh, *half)?;
            check_corner(&CornerData {
                uv: [0.0, 0.0],
                normal: *normal,
            })?;
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

        // The modelling set. Each one is a `Mesh::transact` internally, so the
        // three rules at the top of this module hold for them unchanged.
        Op::ExtrudeFaces { faces, distance } => model::extrude_faces(mesh, faces, *distance),
        Op::ExtrudeEdges { edges, delta } => model::extrude_edges(mesh, edges, *delta),
        Op::InsetFaces {
            faces,
            amount,
            individual,
        } => model::inset_faces(mesh, faces, *amount, *individual),
        Op::BevelEdges { edges, amount } => model::bevel_edges(mesh, edges, *amount),
        Op::LoopCut { half, cuts } => model::loop_cut(mesh, *half, *cuts),
        Op::Knife { path } => model::knife(mesh, path),
        Op::MergeVerts { verts, target } => model::merge_verts(mesh, verts, *target),
        Op::SubdivideFaces { faces } => model::subdivide_faces(mesh, faces),
        Op::Mirror { axis, coord } => model::mirror(mesh, *axis, *coord),

        // The sculpt / gizmo set (P23.5). Same three rules: a refusal is a typed
        // value, the mesh stays valid, and the whole thing is one journal entry.
        Op::Sculpt {
            mode,
            dabs,
            radius,
            strength,
            falloff,
        } => crate::sculpt::sculpt(mesh, *mode, dabs, *radius, *strength, *falloff),
        Op::RotateVerts {
            verts,
            pivot,
            axis,
            radians,
        } => crate::xform::rotate_verts(mesh, verts, *pivot, *axis, *radians),
        Op::ScaleVerts {
            verts,
            pivot,
            factor,
        } => crate::xform::scale_verts(mesh, verts, *pivot, *factor),

        Op::SetEdgeSeam { half, seam } => {
            if !mesh.has_half(*half) {
                return Err(OpError::NoSuchHalf(*half));
            }
            mesh.set_seam_pair(*half, *seam);
            Ok(OpOutcome::default())
        }
        Op::Unwrap { corners } => {
            // Every id and every value is checked before the first write, so the
            // refusal is inert without needing a transaction — the
            // `TranslateVerts` shape.
            for (h, uv) in corners {
                match mesh.face_of(*h) {
                    Some(Some(_)) => {}
                    _ => return Err(OpError::UnwrapCornerMissing(*h)),
                }
                finite("an unwrapped corner uv", uv)?;
            }
            for (h, uv) in corners {
                mesh.half_mut(*h).uv = *uv;
            }
            Ok(OpOutcome::default())
        }

        // ── the skin ops (P24.2) ───────────────────────────────────────────
        //
        // All three are attribute writes: no face is rebuilt, no slot is freed,
        // so every live id still names what it named (`op_preserves_ids`), and
        // all three check everything before the first write, so a refusal is
        // inert without needing a `transact` (the `TranslateVerts` shape).
        Op::BindSkin { skeleton, joints } => {
            if *joints == 0 {
                return Err(OpError::SkinBindingEmpty);
            }
            // A re-bind must not orphan a weight that is already on the mesh.
            for v in mesh.vert_ids() {
                if let Some(j) = mesh.vert_ref(v).skin.max_joint() {
                    if j as u32 >= *joints {
                        return Err(OpError::SkinJointOutOfRange {
                            vert: v,
                            joint: j,
                            joints: *joints,
                        });
                    }
                }
            }
            mesh.set_skin_binding(Some(crate::skin::SkinBinding {
                skeleton: *skeleton,
                joints: *joints,
            }));
            Ok(OpOutcome::default())
        }

        Op::AssignWeights { weights } => {
            let Some(binding) = mesh.skin_binding() else {
                return Err(OpError::NotSkinned);
            };
            for (v, w) in weights {
                if !mesh.has_vert(*v) {
                    return Err(OpError::NoSuchVert(*v));
                }
                if !w.weights.iter().all(|x| x.is_finite()) {
                    return Err(OpError::SkinWeightsDegenerate {
                        vert: *v,
                        why: "a weight is not finite",
                    });
                }
                if w.weights.iter().any(|x| *x < 0.0) {
                    return Err(OpError::SkinWeightsDegenerate {
                        vert: *v,
                        why: "a weight is negative",
                    });
                }
                if w.weights.iter().sum::<f32>() <= 0.0 {
                    return Err(OpError::SkinWeightsDegenerate {
                        vert: *v,
                        why: "every weight is zero, so the vertex is influenced by nothing",
                    });
                }
                if let Some(j) = w.max_joint() {
                    if j as u32 >= binding.joints {
                        return Err(OpError::SkinJointOutOfRange {
                            vert: *v,
                            joint: j,
                            joints: binding.joints,
                        });
                    }
                }
            }
            for (v, w) in weights {
                mesh.set_vert_weights(*v, w.normalized());
            }
            Ok(OpOutcome::default())
        }

        Op::ClearSkin => {
            if mesh.skin_binding().is_none() {
                return Err(OpError::NotSkinned);
            }
            mesh.set_skin_binding(None);
            let ids: Vec<VertId> = mesh.vert_ids().collect();
            for v in ids {
                mesh.set_vert_weights(v, crate::skin::VertWeights::RIGID);
            }
            Ok(OpOutcome::default())
        }
    }
}

/// A corner's UV must be finite and its AUTHORED normal must be a finite unit
/// vector — see [`OpError::NormalNotUnit`].
fn check_corner(c: &CornerData) -> Result<(), OpError> {
    finite("a corner uv", &c.uv)?;
    if let Some(n) = c.normal {
        finite("a corner normal", &n)?;
        let length = DVec3::from_array(n).length();
        if (length - 1.0).abs() > 1e-6 {
            return Err(OpError::NormalNotUnit { length });
        }
    }
    Ok(())
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

pub(crate) fn lerp_corner(a: &CornerData, b: &CornerData, t: f64) -> CornerData {
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
    // Same rule as `TranslateVerts`: the stored value is what has to be finite.
    // Two vertices at opposite ends of the f64 range have a midpoint that is not.
    finite("a split-edge midpoint", &mid)?;

    let mut incident: Vec<FaceId> = Vec::new();
    for h in [half, twin] {
        if let Some(Some(f)) = mesh.face_of(h) {
            incident.push(f);
        }
    }

    let out = mesh.transact(|m| {
        let sharp = m.capture_edge_flags(&incident);
        let patches: Vec<FacePatch> = incident.iter().map(|&f| m.remove_face_raw(f)).collect();
        // The midpoint's skin is its endpoints' skins at the same `t` the
        // position uses (P24.2) — one lerp, two channels, so a split on a
        // weighted arm does not put a rigid vertex in the middle of it.
        let mid_vert = m.alloc_vert_blended(mid, &[(a, 1.0 - t), (b, t)]);
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
            // The same finiteness law the modelling ops obey: an interpolated
            // corner is a value this crate STORES, and two legal extremes
            // interpolate to an infinity.
            crate::model::finite_corner(&corner, "a split-edge corner")?;
            let mut verts = patch.verts.clone();
            let mut corners = patch.corners.clone();
            verts.insert(at + 1, mid_vert);
            corners.insert(at + 1, corner);
            touched.extend(verts.iter().copied());
            rebuilt.push(m.add_face_raw(&verts, &corners, patch.slot)?);
        }
        m.apply_edge_flags(&sharp);
        // The two halves of the split edge inherit the original's flags — its
        // sharpness AND (since P23.5) its seam, which is what stops splitting an
        // edge on a UV cut from quietly healing the cut.
        let flags = crate::model::flags_of(&sharp, a, b);
        if flags != EdgeFlags::default() {
            for (x, y) in [(a, mid_vert), (mid_vert, b)] {
                if let Some(h) = m.find_half(x, y) {
                    m.set_sharp_pair(h, flags.sharp);
                    m.set_seam_pair(h, flags.seam);
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
        let sharp = m.capture_edge_flags(&[fa]);
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
        m.apply_edge_flags(&sharp);
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
pub(crate) fn weld_impl(
    mesh: &mut Mesh,
    keep: VertId,
    merge: VertId,
) -> Result<Vec<FaceId>, OpError> {
    let incident: Vec<FaceId> = mesh
        .vert_outgoing(merge)
        .ok_or(OpError::NoSuchVert(merge))?
        .iter()
        .filter_map(|&h| match mesh.face_of(h) {
            Some(Some(f)) => Some(f),
            _ => None,
        })
        .collect();
    let sharp: Vec<(VertId, VertId, EdgeFlags)> = mesh
        .capture_edge_flags(&incident)
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
    mesh.apply_edge_flags(&sharp);
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
    fn weld_relinks_every_vertex_the_patch_touched() {
        // Found by `validity_holds_after_every_op`, and the reason the touched
        // set is built from every REMOVED face rather than every REBUILT one.
        //
        // A quad with one triangle hanging off an outer edge. Welding the
        // triangle's free corner into one of the shared ones makes that triangle
        // lose its surface entirely, so it is removed and never rebuilt — and
        // its OTHER shared vertex therefore never reached `finish_patch`, while
        // the edge it lost had already been freed. The boundary half-edge
        // arriving at that vertex kept pointing at a dead slot: `DeadLink`, with
        // no other symptom.
        let mut m = plane(2.0);
        let quad = m.face_verts(m.face_ids().next().unwrap()).unwrap();
        let (v0, v1) = (quad[0], quad[1]);
        let e = apply(
            &mut m,
            &Op::AddVertex {
                position: [0.0, 2.0, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        // The free side of the quad's v0→v1 edge is v1→v0.
        expect_ok(
            &mut m,
            Op::AddFace {
                verts: vec![v1, v0, e],
                corners: vec![CornerData::default(); 3],
                slot: None,
            },
        );
        assert_eq!(m.face_count(), 2);

        expect_ok(&mut m, Op::WeldVerts { keep: v1, merge: e });
        assert_eq!(m.face_count(), 1, "the fin folded onto an edge and went");
        assert_eq!(m.vert_count(), 4, "and took its own vertex with it");
        assert_eq!(m.edge_count(), 4, "back to the bare quad");
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
    fn non_finite_values_are_refused_at_every_op_that_takes_one() {
        // M6, the door side. The writer counts non-finite values because it
        // cannot refuse an author's imported file without losing their work; an
        // *edit* has no such excuse, so no op in this module will STORE a
        // non-finite coordinate.
        //
        // "Store" is the load-bearing word, and the first version of this got it
        // wrong: it checked the operands and let the arithmetic through, so a
        // vertex at 1e308 translated by 1e308 was written as an infinity by an op
        // whose own test claimed the kernel "cannot be made to hold a NaN".
        // `a_finite_op_that_computes_a_non_finite_result_is_refused` covers that
        // half; this one covers the operands.
        let mut m = plane(2.0);
        let v = m.vert_ids().next().unwrap();
        let c = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .unwrap();
        for (op, what) in [
            (
                Op::AddVertex {
                    position: [f64::NAN, 0.0, 0.0],
                },
                "a vertex position",
            ),
            (
                Op::TranslateVerts {
                    verts: vec![v],
                    delta: [0.0, f64::INFINITY, 0.0],
                },
                "a translation delta",
            ),
            (
                Op::SetCornerUv {
                    half: c,
                    uv: [f64::NAN, 0.0],
                },
                "a corner uv",
            ),
            (
                Op::SetCornerNormal {
                    half: c,
                    normal: Some([f64::NAN, 1.0, 0.0]),
                },
                "a corner normal",
            ),
        ] {
            let before = m.encoded();
            match apply(&mut m, &op) {
                Err(OpError::NonFinite { what: got, .. }) => assert_eq!(got, what),
                other => panic!("{op:?} was not refused: {other:?}"),
            }
            assert_eq!(m.encoded(), before, "and the refusal is inert");
        }
    }

    #[test]
    fn a_finite_op_that_computes_a_non_finite_result_is_refused() {
        // Every operand here is finite; every RESULT is not. Checking the
        // operands and storing the arithmetic is a gate aimed at the wrong
        // number, and it shipped: these two ops wrote infinities into the mesh
        // while the suite asserted the kernel could not be made to hold one.
        let huge = 1.0e308_f64;

        let mut m = plane(2.0);
        let v: Vec<VertId> = m.vert_ids().collect();
        let before = m.encoded();
        expect_ok(
            &mut m,
            Op::TranslateVerts {
                verts: v.clone(),
                delta: [huge, 0.0, 0.0],
            },
        );
        assert!(m.position(v[0]).unwrap().x.is_finite());
        match apply(
            &mut m,
            &Op::TranslateVerts {
                verts: v.clone(),
                delta: [huge, 0.0, 0.0],
            },
        ) {
            Err(OpError::NonFinite { what, .. }) => {
                assert_eq!(what, "a translated vertex position")
            }
            other => panic!("an overflowing translate was not refused: {other:?}"),
        }
        assert!(
            m.vert_ids().all(|v| m.position(v).unwrap().is_finite()),
            "and nothing was stored"
        );

        // The split midpoint is the other computed store.
        let mut m = Mesh::new();
        let a = apply(
            &mut m,
            &Op::AddVertex {
                position: [-huge, 0.0, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        let b = apply(
            &mut m,
            &Op::AddVertex {
                position: [huge, 0.0, huge],
            },
        )
        .unwrap()
        .verts[0];
        let c = apply(
            &mut m,
            &Op::AddVertex {
                position: [0.0, huge, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        expect_ok(
            &mut m,
            Op::AddFace {
                verts: vec![a, b, c],
                corners: vec![CornerData::default(); 3],
                slot: None,
            },
        );
        let h = m.find_half(a, b).unwrap();
        let before_split = m.encoded();
        match apply(&mut m, &Op::SplitEdge { half: h, t: 0.5 }) {
            Err(OpError::NonFinite { what, .. }) => assert_eq!(what, "a split-edge midpoint"),
            other => panic!("an overflowing split was not refused: {other:?}"),
        }
        assert_eq!(m.encoded(), before_split, "and the refusal is inert");
        let _ = before;
    }

    #[test]
    fn an_authored_normal_must_be_a_real_normal() {
        // The writer copies an authored normal out VERBATIM, so it has to be
        // unit. A caller who wants shading from the surface passes `None`.
        let mut m = plane(2.0);
        let c = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .unwrap();
        match apply(
            &mut m,
            &Op::SetCornerNormal {
                half: c,
                normal: Some([0.0, 5.0, 0.0]),
            },
        ) {
            Err(OpError::NormalNotUnit { length }) => assert!((length - 5.0).abs() < 1e-12),
            other => panic!("{other:?}"),
        }
        // Unit is accepted, and so is "derive it".
        expect_ok(
            &mut m,
            Op::SetCornerNormal {
                half: c,
                normal: Some([0.0, 1.0, 0.0]),
            },
        );
        expect_ok(
            &mut m,
            Op::SetCornerNormal {
                half: c,
                normal: None,
            },
        );
        // AddFace checks its corner records by the same door.
        let a = apply(
            &mut m,
            &Op::AddVertex {
                position: [4.0, 0.0, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        let b = apply(
            &mut m,
            &Op::AddVertex {
                position: [4.0, 0.0, 4.0],
            },
        )
        .unwrap()
        .verts[0];
        let d = apply(
            &mut m,
            &Op::AddVertex {
                position: [8.0, 0.0, 0.0],
            },
        )
        .unwrap()
        .verts[0];
        let bad = CornerData {
            uv: [0.0, 0.0],
            normal: Some([0.0, 3.0, 0.0]),
        };
        assert!(matches!(
            apply(
                &mut m,
                &Op::AddFace {
                    verts: vec![a, b, d],
                    corners: vec![bad, bad, bad],
                    slot: None,
                },
            ),
            Err(OpError::NormalNotUnit { .. })
        ));
    }

    // ── the skin ops (P24.2) ───────────────────────────────────────────────

    use crate::skin::{SkinBinding, VertWeights};

    /// A cube bound to a 6-joint skeleton.
    fn skinned_cube() -> Mesh {
        let mut m = cube(1.0);
        apply(
            &mut m,
            &Op::BindSkin {
                skeleton: Some([9; 16]),
                joints: 6,
            },
        )
        .expect("binds");
        m
    }

    #[test]
    fn binding_leaves_every_vertex_rigid_and_valid() {
        let m = skinned_cube();
        assert_eq!(
            m.skin_binding(),
            Some(SkinBinding {
                skeleton: Some([9; 16]),
                joints: 6
            })
        );
        assert!(
            m.vert_ids()
                .all(|v| m.vert_weights(v).unwrap() == VertWeights::RIGID),
            "a fresh bind rides joint 0 — wrong-looking and VALID, which is the \
             state auto-fit and the weight solve replace"
        );
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_zero_joint_binding_is_refused_inertly() {
        let mut m = cube(1.0);
        refuses(
            &mut m,
            Op::BindSkin {
                skeleton: None,
                joints: 0,
            },
            OpError::SkinBindingEmpty,
        );
        assert!(!m.is_skinned());
    }

    #[test]
    fn re_binding_onto_a_smaller_skeleton_refuses_rather_than_clamping() {
        let mut m = skinned_cube();
        let v = m.vert_ids().next().unwrap();
        expect_ok(
            &mut m,
            Op::AssignWeights {
                weights: vec![(v, VertWeights::from_pairs(&[(5, 1.0)]))],
            },
        );
        // Joint 5 exists in a 6-joint rig and not in a 3-joint one. Clamping the
        // index would silently re-weight this vertex to a different bone.
        refuses(
            &mut m,
            Op::BindSkin {
                skeleton: Some([1; 16]),
                joints: 3,
            },
            OpError::SkinJointOutOfRange {
                vert: v,
                joint: 5,
                joints: 3,
            },
        );
        // …and a WIDER re-bind is fine, which is how a rig change lands.
        expect_ok(
            &mut m,
            Op::BindSkin {
                skeleton: Some([1; 16]),
                joints: 40,
            },
        );
    }

    #[test]
    fn assigning_weights_refuses_every_way_it_can_and_stays_inert() {
        let v = cube(1.0).vert_ids().next().unwrap();
        let good = VertWeights::from_pairs(&[(1, 1.0)]);

        // On an unbound mesh there is nothing for an index to mean.
        let mut rigid = cube(1.0);
        refuses(
            &mut rigid,
            Op::AssignWeights {
                weights: vec![(v, good)],
            },
            OpError::NotSkinned,
        );

        let mut m = skinned_cube();
        refuses(
            &mut m,
            Op::AssignWeights {
                weights: vec![(VertId(9_999), good)],
            },
            OpError::NoSuchVert(VertId(9_999)),
        );
        refuses(
            &mut m,
            Op::AssignWeights {
                weights: vec![(v, VertWeights::from_pairs(&[(6, 1.0)]))],
            },
            OpError::SkinJointOutOfRange {
                vert: v,
                joint: 6,
                joints: 6,
            },
        );
        for (bad, why) in [
            (
                VertWeights {
                    joints: [1, 0, 0, 0],
                    weights: [f32::NAN, 0.0, 0.0, 0.0],
                },
                "a weight is not finite",
            ),
            (
                VertWeights {
                    joints: [1, 2, 0, 0],
                    weights: [1.5, -0.5, 0.0, 0.0],
                },
                "a weight is negative",
            ),
            (
                VertWeights {
                    joints: [1, 0, 0, 0],
                    weights: [0.0; 4],
                },
                "every weight is zero, so the vertex is influenced by nothing",
            ),
        ] {
            refuses(
                &mut m,
                Op::AssignWeights {
                    weights: vec![(v, bad)],
                },
                OpError::SkinWeightsDegenerate { vert: v, why },
            );
        }
        // The whole batch is inert: a good pair alongside a bad one writes
        // NOTHING, because every pair is checked before the first write.
        let others: Vec<VertId> = m.vert_ids().take(2).collect();
        refuses(
            &mut m,
            Op::AssignWeights {
                weights: vec![
                    (others[1], good),
                    (
                        others[0],
                        VertWeights {
                            joints: [1, 0, 0, 0],
                            weights: [0.0; 4],
                        },
                    ),
                ],
            },
            OpError::SkinWeightsDegenerate {
                vert: others[0],
                why: "every weight is zero, so the vertex is influenced by nothing",
            },
        );
        assert_eq!(m.vert_weights(others[1]), Some(VertWeights::RIGID));
    }

    #[test]
    fn assigned_weights_are_renormalized_on_apply() {
        let mut m = skinned_cube();
        let v = m.vert_ids().next().unwrap();
        expect_ok(
            &mut m,
            Op::AssignWeights {
                weights: vec![(
                    v,
                    VertWeights {
                        joints: [1, 2, 0, 0],
                        weights: [3.0, 1.0, 0.0, 0.0],
                    },
                )],
            },
        );
        let w = m.vert_weights(v).unwrap();
        assert!(w.is_normalized(), "{w:?}");
        assert!((w.weight_of(1) - 0.75).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn clearing_the_skin_unbinds_and_resets_every_vertex() {
        let mut m = skinned_cube();
        let v = m.vert_ids().next().unwrap();
        expect_ok(
            &mut m,
            Op::AssignWeights {
                weights: vec![(v, VertWeights::from_pairs(&[(2, 0.4), (4, 0.6)]))],
            },
        );
        expect_ok(&mut m, Op::ClearSkin);
        assert!(!m.is_skinned());
        assert!(
            m.vert_ids()
                .all(|v| m.vert_weights(v).unwrap() == VertWeights::RIGID),
            "an unbind that left weights behind is a `SkinWithoutBinding`"
        );
        // Idempotence is NOT the rule: a second clear is a refusal, so the UI
        // cannot report "cleared" for an op that did nothing.
        refuses(&mut m, Op::ClearSkin, OpError::NotSkinned);
    }

    /// **A structural op carries the channel through** (P24.2): a split-edge
    /// midpoint is its endpoints' weights at the same `t` as its position.
    #[test]
    fn a_split_blends_the_weights_of_the_edge_it_cuts() {
        let mut m = skinned_cube();
        let h = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .unwrap();
        let (a, b) = (m.origin(h).unwrap(), m.dest(h).unwrap());
        expect_ok(
            &mut m,
            Op::AssignWeights {
                weights: vec![
                    (a, VertWeights::from_pairs(&[(1, 1.0)])),
                    (b, VertWeights::from_pairs(&[(2, 1.0)])),
                ],
            },
        );
        let out = expect_ok(&mut m, Op::SplitEdge { half: h, t: 0.25 });
        let mid = out.verts[0];
        let w = m.vert_weights(mid).unwrap();
        assert!((w.weight_of(1) - 0.75).abs() < 1e-6, "{w:?}");
        assert!((w.weight_of(2) - 0.25).abs() < 1e-6, "{w:?}");
        // The CONTROL: the same split on a rigid mesh produces a rigid vertex,
        // bit for bit — which is what makes "no unskinned edit moved" true.
        let mut rigid = cube(1.0);
        let h = rigid
            .half_ids()
            .find(|&h| rigid.is_boundary(h) == Some(false))
            .unwrap();
        let out = expect_ok(&mut rigid, Op::SplitEdge { half: h, t: 0.25 });
        assert_eq!(
            rigid.vert_weights(out.verts[0]),
            Some(VertWeights::RIGID),
            "a blend of rigid sources must be exactly RIGID"
        );
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
