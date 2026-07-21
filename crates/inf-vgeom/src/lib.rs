//! Virtualized geometry (Phase 13.1a): the offline meshlet **clusterization +
//! simplification DAG builder** — the offline half of the engine's Nanite-class
//! virtualized geometry. This is an **original** implementation of the
//! well-published technique (no Epic code).
//!
//! # What this crate produces
//!
//! [`build_vgeom`] turns a mesh's vertex streams + index buffer into a
//! [`VgeomMesh`]: a level-of-detail **DAG of meshlets** (small clusters of
//! ~64 vertices / ~124 triangles) where each meshlet carries culling bounds
//! (sphere + normal cone) and two errors that make runtime LOD selection a
//! branchless per-meshlet cut — `draw iff error ≤ threshold < parent_error`. See
//! [`model`] for the data model and the *cut invariant* that guarantees the cut
//! yields a complete, non-overlapping surface at every threshold.
//!
//! # Where it fits (cook-derived, not authoring)
//!
//! `.inf_vmesh` is **derived at cook time** from an authoring `.inf_mesh`
//! ([`inf_mesh::MeshAsset`](../inf_mesh/struct.MeshAsset.html)) — the preferred
//! split (roadmap P13.1 read #3): source meshes stay authoring-clean, and the
//! cook derives the render-optimized meshlet form beside them, wired by a
//! dependency edge (`runtime/inf-packager`). It is a separate [`AssetKind`] and
//! `.inf_vmesh` extension, **not** an embedded `MeshAsset` field, so no existing
//! mesh payload changes.
//!
//! ## Runtime pick logic (next wave)
//!
//! A cooked pack carries **both** the `.inf_mesh` and its derived `.inf_vmesh`.
//! The next-wave renderer picks the `.inf_vmesh` when virtualized geometry is
//! enabled *and* present for a mesh (deriving the vmesh id from the mesh id — see
//! the cook), and falls back to the classic `.inf_mesh` LOD path otherwise
//! (roadmap risk #3: the engine ships without virtualized geometry). The GPU
//! culling + LOD-selection **compute** passes ([`VgeomMesh::select`] is their
//! reference rule) are the next wave; this crate only *designs their data*.
//!
//! # Deferred (documented follow-ups)
//!
//! * **Quantization** — v1 stores full-precision f32 positions; per-cluster
//!   quantized positions are a later schema bump.
//! * **METIS-style grouping** — v1 groups meshlets with a deterministic greedy
//!   shared-edge agglomeration; a graph-partitioner (balanced min-cut) is the
//!   quality follow-up.
//! * **Streaming/paging** — the payload is already laid out coarse→fine
//!   ([`VgeomMesh::levels`]); the actual range-request streamer is P13.1
//!   deliverable 2 (next wave).
//! * **Per-material meshlets** — v1 flattens all submeshes into one geometry;
//!   material-slot tagging per meshlet is a follow-up.

pub mod build;
pub mod model;

pub use build::{build_vgeom, BuildParams};
pub use model::{Group, LevelRange, Meshlet, VgeomMesh, VgeomVertex};
