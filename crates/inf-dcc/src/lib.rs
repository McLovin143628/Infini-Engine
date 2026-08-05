//! **The embedded DCC's mesh kernel** (P23.3): a half-edge structure, the
//! invariants it promises, the `MeshAsset` reader/writer pair, a small set of
//! core topology operations, and the deterministic op journal that is both the
//! undo/redo story and the test story.
//!
//! This crate is a *kernel*, not an editor. It has no panel, no commands, no
//! selection model and no modelling ops beyond the primitives P23.4's extrude /
//! inset / bevel / loop-cut are built out of. It has no format of its own
//! either: it imports from and exports to [`inf_mesh::MeshAsset`], which is the
//! same type the glTF importer produces, so "model in engine" and "import from
//! Blender" converge on one representation and there is no in-engine format an
//! author can get trapped in (`docs/memos/p23-dcc-design.md` §7).
//!
//! # The six decisions this kernel is made of
//!
//! **1. Boundaries are real half-edges, not absent ones.** `twin` is TOTAL —
//! every half-edge has exactly one twin, always — and it is `face` that is
//! [`Option`]: a half-edge with `face == None` is a *boundary* half-edge, linked
//! into a boundary loop exactly like a face loop. Open meshes are first-class
//! (a plane has one boundary loop). The alternative — a `twin: Option<HalfId>`
//! sentinel — makes the involution partial and puts a branch in every traversal
//! P23.4 will ever write. See [`topo`].
//!
//! **2. Attributes live where seams live.** Positions are on *vertices*; UVs and
//! (optional, authored) normals are on *corners* — that is, on the face-side
//! half-edges. A `MeshAsset` stores a *split* vertex per UV/normal seam, so
//! importing it position-welds the topology back together while keeping every
//! attribute on the corner it was authored for. Nothing is averaged, nothing is
//! lost, and export re-splits from exactly the same information. See [`build`]
//! and [`export`].
//!
//! **3. The mesh is edge- AND vertex-manifold, and ops refuse rather than
//! corrupt.** At most one half-edge runs from `a` to `b` (so at most two faces
//! meet at an edge), and every vertex's incident half-edges form ONE cycle (no
//! bowties). Vertex-manifoldness is not decoration: it is what makes a vertex's
//! boundary loop unique, which is what makes the boundary relink after a
//! structural edit well-defined — and what will make P23.4's loop cut have a
//! defined path. Import does not refuse a bowtie, it *splits* it (§[`build`]);
//! ops refuse it, as a typed value.
//!
//! **4. Refusals are values and they are inert.** Every op returns
//! `Result<_, OpError>`, a refused op leaves the mesh **byte-identical**, and
//! that is property-tested — because an op that half-applies before noticing it
//! cannot is worse than one that never ran.
//!
//! **5. Positions are `f64`.** Architecture rule 3 stops at the crate boundary:
//! `f32` exists here only in a `MeshVertex` being read or written. Every
//! intermediate — face normals, tangent accumulation, ear-clipping — is `f64`.
//!
//! **6. `meshopt` never enters the op journal.** The P18 law (meshopt is *not*
//! cross-platform: identical input, different bytes on `x86_64-msvc` vs
//! `aarch64-apple-darwin`) makes it poison for a structure whose entire contract
//! is "replaying these ops on any machine produces the same mesh". It is
//! available exactly once, behind [`ExportOptions::optimize`], on the way OUT —
//! and that door is documented as the non-deterministic one.
//!
//! # What "deterministic" means here, precisely
//!
//! * Two runs of the same op sequence produce **byte-identical** encoded meshes.
//! * `replay(base, ops)` equals the incrementally-edited mesh, byte for byte —
//!   which is only true because id allocation is itself a pure function of the
//!   op sequence (see the id-reuse rule in [`topo`]).
//! * Iteration is over integer ranges and `BTreeMap`/`BTreeSet`. There is no
//!   `HashMap` on any path that feeds a mesh, an export or a journal.
//! * No `std` trigonometry: the primitive generators call
//!   [`inf_math::psin64`]/[`inf_math::pcos64`]. `sqrt` *is* used freely — it is
//!   correctly rounded by IEEE-754 and therefore bit-portable, which `sin` and
//!   `cbrt` are not.
//!
//! # Units
//!
//! Positions are **metres** (1 world unit = 1 metre, architecture rule 6). UVs
//! are dimensionless. Normals and tangents are unit vectors; a tangent's `w` is
//! the bitangent handedness sign (±1), the glTF convention `MeshVertex` already
//! carries.

pub mod build;
pub mod export;
pub mod journal;
pub mod ops;
pub mod topo;
pub mod validate;

pub use build::{
    cube, cylinder, from_mesh_asset, plane, torus, ImportError, ImportReport, MeshImport,
    WELD_TOLERANCE,
};
pub use export::{to_mesh_asset, ExportOptions, ExportReport, NormalPolicy, TANGENT_FALLBACK};
pub use journal::{MeshSession, SessionError, SessionSave, CHECKPOINT_INTERVAL, MAX_CHECKPOINTS};
pub use ops::{Op, OpError, OpOutcome};
pub use topo::{CanonicalMesh, CornerData, FaceId, HalfId, Mesh, VertId};
pub use validate::{validate, Violation};
