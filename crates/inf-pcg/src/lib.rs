//! Procedural content generation runtime (ROADMAP §P10.5): the deterministic,
//! massively-scalable substrate that scatters 1M+ instances across terrain by
//! rules.
//!
//! ## Pipeline
//!
//! ```text
//!   SamplerDef  ─build(height)─▶  DensityField  ┐
//!                                                ├─▶ scatter_region ─▶ Vec<PcgInstance>
//!   HeightProvider (terrain seam) ───────────────┘
//! ```
//!
//! * [`hash`] — counter-based hashing: every random draw is a pure function of an
//!   integer coordinate tuple, so nothing is a stateful RNG.
//! * [`noise`] — hand-rolled seedable value-noise + fBm (the procedural density).
//! * [`height`] — the [`HeightProvider`] terrain seam (bridged to
//!   `inf_terrain::HeightSource` next batch — see the module docs).
//! * [`fields`] — the [`TerrainFields`] seam (P19.3): a terrain's *per-sample
//!   layers*, i.e. the P19.1 erosion data maps and the P19.2 painted biome ids.
//! * [`binding`] — [`BiomeBinding`] (P19.3): painted biomes dispatch their own
//!   `.inf_pcg` graphs over the regions their ids own, feathered at the borders.
//! * [`sampler`] — [`DensityField`] sources, terrain filters, and combinators.
//! * [`scatter`] — the deterministic, `parallel_map`-parallel scatter kernel.
//! * [`rules`] — the serializable [`PcgDocument`] rule model + [`evaluate`].
//! * [`graph`] — the editor node kit over `inf-graph` + [`lower_graph`] (the
//!   `.inf_pcg` graph → the stable `PcgDocument`; editor preview == runtime).
//! * [`asset`] — the `.inf_pcg` [`PcgAssetPayload`] envelope (stores the graph).
//!
//! ## Determinism doctrine
//!
//! Same document + same seed + same terrain ⇒ **identical** instance lists,
//! independent of thread count. Sampling is counter-based; parallelism is
//! [`inf_core::parallel_map`] over cells (a deterministic in-order pure map).
//!
//! ## Deferred (later P10.5 batches)
//!
//! GPU-instanced scattering + per-instance culling (P10.5.3), LOD/impostor fade
//! (P10.5.4), the `.inf_pcg` **editor** node kit on the shared `inf-graph` canvas
//! (P10.5b), and PCG debug visualization (P10.5.5).

pub mod asset;
pub mod binding;
pub mod fields;
pub mod graph;
pub mod hash;
pub mod height;
pub mod noise;
pub mod rules;
pub mod sampler;
pub mod scatter;

pub use asset::{PcgAssetPayload, PcgError};
pub use binding::{bind_document, biome_seed, BiomeBinding, BiomeGraph, DEFAULT_BIOME_FEATHER};
pub use fields::{NoFields, OffsetTerrain, TerrainFields};
pub use graph::{lower_graph, pcg_registry, LoweredPcg, PcgGraphIssue, PcgSeverity};
pub use height::{FnHeight, HeightProvider};
pub use noise::ValueNoise;
pub use rules::{evaluate, evaluate_with, PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
pub use sampler::{
    AltitudeFilter, BiomeMask, Constant, DataMapMask, DensityField, Invert, MaskImage, Max, Min,
    Multiply, Noise, SlopeFilter,
};
pub use scatter::{
    scatter_region, scatter_region_in, PcgInstance, Region, RotationMode, ScatterParams,
};
