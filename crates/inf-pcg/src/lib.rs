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
//! * [`grammar`] — the P19.4 **rule-rewriting grammar**: a rule text DSL, spline
//!   and footprint spans, and an exact-fill layout that turns them into placed
//!   modular-mesh instances on the same instancing path as scatter.
//! * [`building`] — the P19.5 **building & interior grammar**: a footprint
//!   becomes a floor stack, a room partition, walls with real openings and
//!   furnished rooms. The 2-D half is its own slice tree; the 1-D half is
//!   [`grammar`] verbatim (a wall *is* a span).
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
pub mod building;
pub mod fields;
pub mod grammar;
pub mod graph;
pub mod hash;
pub mod height;
pub mod noise;
pub mod rules;
pub mod sampler;
pub mod scatter;
pub mod volume;

pub use asset::{PcgAssetPayload, PcgError};
pub use binding::{
    bind_document, biome_seed, neighbour_rings, scatter_reach_m, BiomeBinding, BiomeGraph,
    BiomeScatterCache, DEFAULT_BIOME_FEATHER,
};
pub use building::{
    archetype, archetypes, evaluate_buildings, evaluate_buildings_in, plans_of,
    society::{PcgSlot, SlotPosture, SlotRole, SlotShift},
    station::{PcgStation, StationUse},
    subdivide_block, ArchetypeId, BlockLot, BlockSubdivision, BuildingArchetype, BuildingOutput,
    BuildingParams, BuildingPass, BuildingPlan, LotRules, Opening, OpeningKind, Rect2, Room,
    RoomType, Stair, StructureGroup, StructureTier, Wall, DEFAULT_STRUCTURE_LOD_M,
};
pub use fields::{NoFields, OffsetTerrain, TerrainFields};
pub use grammar::{
    evaluate_grammars, evaluate_grammars_in, footprint_perimeter, footprint_rows, FootprintMode,
    Grammar, GrammarContext, GrammarError, GrammarPass, Ground, NoSplines, RowAxis, Span, SpanSet,
    SpanSource, SplineInterp, SplinePath, SplineSource,
};
pub use graph::{
    grammar_mesh_refs, lower_graph, pcg_registry, LoweredPcg, PcgGraphIssue, PcgSeverity,
};
pub use height::{FnHeight, HeightProvider, FN_HEIGHT_NORMAL_EPS};
pub use noise::ValueNoise;
pub use rules::{evaluate, evaluate_with, PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
pub use sampler::{
    AltitudeFilter, BiomeMask, Constant, DataMapMask, DensityField, Invert, MaskImage, Max, Min,
    Multiply, Noise, SlopeFilter, MAX_FEATHER_SAMPLES,
};
pub use scatter::{
    scatter_region, scatter_region_in, PcgCollider, PcgInstance, PcgSurface, Region, RotationMode,
    ScatterParams,
};
pub use volume::{compose_volume, VolumeOutput};
