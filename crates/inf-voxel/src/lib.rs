//! **Sparse SDF voxel chunk volumes** (P21.1): the geometry a heightfield
//! cannot express — caves, tunnels, overhangs and excavation pits — carried in
//! chunks that *locally extend* the terrain rather than replacing it.
//!
//! # The design stance, stated once
//!
//! The planet-scale base **stays a heightfield**. The P16 streaming clipmap's
//! economics are unbeatable at that scale, and nothing here voxelizes the world.
//! What arrives is the hybrid every serious open-world engine uses: SDF chunk
//! volumes anchored in `f64` world space that override and extend the
//! heightfield where an author has actually dug. A level with no volumes pays
//! nothing; a level with a cave system pays for the cave system.
//!
//! # The three types that matter
//!
//! * [`VoxelChunk`] — a dense `CHUNK_DIM³` block of `f32` signed distances plus a
//!   sparse per-sample material index. Format-aware serde, so its bincode blob is
//!   the same bytes in every container it ever lands in.
//! * [`VoxelData`] — the **resident** chunk set: a `BTreeMap<ChunkKey, VoxelChunk>`
//!   with per-chunk monotone stamps and a dirty set, grown and shrunk against a
//!   [`ChunkStore`]. Shaped exactly like `inf_terrain::TerrainData`, and for the
//!   same reasons.
//! * [`VoxelAsset`] — the `.inf_voxel` container: a header, a chunk directory and
//!   16-byte-aligned per-chunk blobs a reader sub-slices out of an mmap with no
//!   decode and no copy.
//!
//! Around them: a **deterministic Surface-Nets mesher** ([`mesh_chunk`]) whose
//! chunk seams are watertight by construction, and **carve/fill ops**
//! ([`VoxelOp`]) that return exact per-material voxel counts and a reversible
//! [`VoxelDelta`]. [`coupling`] is where a cut meets the heightfield: the
//! exactly-invertible rule that decides which terrain samples a carve opens and a
//! fill closes again, so a cave has a mouth and undoing it seals one.
//!
//! # Determinism is structural, not aspirational
//!
//! Everything here obeys the engine-wide law the GI voxelizer paid for
//! (`inf_render::gi`): every pass is a **gather** — one output element per thread
//! of iteration, never a scatter into shared words — iteration order comes from
//! `BTreeMap`/`BTreeSet` and integer ranges, there is **no `HashMap` anywhere on a
//! path that feeds committed content**, world math is `f64`, and no `std` trig is
//! called (nothing here needs any; `inf-math`'s `psin64`/`pcos64` are the door if
//! that ever changes). Two runs of the mesher over the same chunks produce
//! byte-identical vertex buffers, and so do two runs that inserted those chunks in
//! different orders.
//!
//! # Units
//!
//! Sample **values** are signed distances in **voxels** (see [`SDF_BAND`]), not
//! metres — the field is scale-free, and the one place metres enter is
//! [`VoxelData::voxel_size_m`]. Positions crossing this crate's public surface are
//! either `f64` **world metres** (the ops, [`VoxelData::sdf_at`]) or `f64`
//! **global grid units** (the mesher's vertices, [`VoxelMesh::positions`]), and
//! every signature says which. 1 world unit = 1 metre, SI everywhere
//! (architecture rule 6).

pub mod asset;
pub mod chunk;
pub mod coupling;
pub mod data;
pub mod delta;
pub mod ground;
pub mod mesh;
pub mod ops;
pub mod residency;
pub mod runtime;
pub mod spoil;
pub mod store;
pub mod wants;
pub mod writeback;

pub use asset::{
    build_voxel_asset, chunk_raw_ceiling, read_voxel_asset, recompress_voxel_asset,
    write_voxel_asset, ChunkDirEntry, RecompressReport, VoxelAsset, VoxelAssetBuilder,
    VoxelAssetError, VoxelAssetHeader, VoxelAssetReader, VoxelAssetView, CHUNK_ALIGN,
    COOK_CHUNK_CODEC, DIR_ENTRY_LEN, FIRST_CODEC_SCHEMA_VERSION, HEADER_LEN, HEADER_LEN_V2,
    VOXEL_ASSET_MAGIC, VOXEL_ASSET_SCHEMA_VERSION,
};
pub use chunk::{
    ChunkKey, VoxelChunk, CHUNK_DIM, CHUNK_VOXELS, DEFAULT_MATERIAL, EMPTY_SDF, MATERIAL_COUNT,
    SDF_BAND,
};
pub use coupling::{
    apply_surface_cut, cut_crosses_surface, inline_hole_advisory, surface_cut_samples,
    touch_surface_cut, InlineHoleAdvisory, SurfaceSample,
};
pub use data::VoxelData;
pub use delta::{ChunkPatch, VoxelDelta};
pub use ground::{ground_height_at, sim_volume, topmost_voxel_surface, voxel_surface_y_at};
pub use mesh::{
    mesh_chunk, mesh_keys_for, source_key, MeshSourceKey, MeshSyncReport, VoxelMesh, VoxelMeshCache,
};
pub use ops::{trench_basis, OpReport, VoxelDeltaBuilder, VoxelOp, VoxelOpKind, VoxelShape};
pub use residency::{chunk_range, ChunkStore, MemoryChunkStore, ResidencyReport};
pub use runtime::{
    runtime_carve, RuntimeCarveOutcome, RuntimeCarveReport, MAX_RUNTIME_CARVE_SAMPLES,
};
pub use spoil::{
    cbrt_det, default_spoil_site, pile_base_radius_m, SpoilPlan, SpoilReport, REPOSE_COS,
    REPOSE_DEG, REPOSE_TAN, SPOIL_GAP_M,
};
pub use store::{VolumeSlot, VoxelStreamReport, VoxelVolumes};
pub use wants::{
    advance_wants, chunk_wants, clamp_wants, ChunkCatalog, ChunkGrid, ChunkIndex,
    VoxelStreamBudget, VoxelWantsParams,
};
pub use writeback::{rewrite_voxel_asset, VoxelEdits};
