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
//! * **Per-material meshlets** — v1 flattens all submeshes into one geometry;
//!   material-slot tagging per meshlet is a follow-up.
//!
//! # Streaming (P18.2 — landed)
//!
//! [`model`] is the *shape* of the DAG; [`asset`] is the **container** it ships
//! in (a paged, 16-byte-aligned image sliced straight out of an mmap'd pack), and
//! [`stream`] is the residency state machine + suballocator that decides which
//! pages are in VRAM. `inf-render` owns nothing but the four `wgpu` pools and the
//! shader that reads them, so the whole policy is testable with no adapter.

// The meshlet DAG BUILDER (uses meshopt's C++). Cook-time only — the browser
// player loads pre-cooked DAGs, so `build` (and its `meshopt` dep) is host-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod build;

pub mod asset;
pub mod model;
pub mod stream;

// The shared fixture meshes. Test-only twice over: `cfg(test)` inside this crate,
// the `test-support` feature for everyone else's test targets — and host-only,
// because it calls the `meshopt`-backed builder above.
#[cfg(all(not(target_arch = "wasm32"), any(test, feature = "test-support")))]
pub mod test_support;

pub use asset::{
    build_vgeom_asset, tile_mip_for_lod, ClusterTexture, ClusterTextureSet, ClusterTileRef,
    MeshletRec, VgeomAssetError, VgeomAssetHeader, VgeomAssetImage, VgeomAssetReader,
    VgeomAssetView, VgeomPageEntry, VgeomPageSections, VgeomSource, TILE_REF_LEN,
    VMESH_ASSET_SCHEMA_VERSION,
};
#[cfg(not(target_arch = "wasm32"))]
pub use build::{build_vgeom, BuildParams};
pub use model::{
    pack_tangent, pick_classic_level, unpack_tangent, ClassicLod, Group, LevelRange, Meshlet,
    VgeomMesh, VgeomVertex, NO_TANGENT, TANGENT_EXP,
};
pub use stream::{
    ideal_page_count, AssetResidency, PageBlocks, PageUpload, PoolAllocator, PoolBlock, VgeomPools,
    VgeomStreamBudget, VgeomStreamPlan, VgeomStreamStats, VgeomStreamer, VgeomWant,
    DEFAULT_VGEOM_BUDGET_BYTES, NOT_RESIDENT,
};

// ── The derived-asset id rule (Ring 0, P18.3) ───────────────────────────────
//
// `.inf_vmesh` is *derived* from a `.inf_mesh`, and every host that wants to find
// a mesh's virtualized form does it by **computing** the id rather than by
// consulting a side index. Before P18.3 the salt was written out three times —
// once in the cook, once in the shipped player, once (about to be) in the editor
// — each with its own drift test. It lives here now, in the crate that owns the
// format, so there is exactly one constant to be wrong about. The cook and the
// player delegate to it and keep their public names.

/// The fixed salt XORed into a mesh GUID to derive its `.inf_vmesh` GUID.
///
/// XOR with a constant is a bijection, so distinct mesh ids always yield distinct
/// vmesh ids; the salt makes a collision with any *authored* asset id vanishingly
/// unlikely (and the cook guards the remaining case).
pub const VMESH_ID_SALT: u128 = 0x7635_4e56_4d45_5348_1f13_1a2b_3c4d_5e6f;

/// Derive the deterministic `.inf_vmesh` asset id for a mesh id.
///
/// Involutive (`derived_vmesh_id(derived_vmesh_id(x)) == x`), because the salt is
/// its own inverse under XOR.
pub fn derived_vmesh_id(mesh_id: inf_asset::AssetId) -> inf_asset::AssetId {
    inf_asset::AssetId(uuid::Uuid::from_u128(
        mesh_id.uuid().as_u128() ^ VMESH_ID_SALT,
    ))
}
