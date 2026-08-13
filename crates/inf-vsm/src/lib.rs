//! **Virtual shadow maps** (P27.1): the virtual page space of a shadow-casting
//! light, the physical `Depth32Float` page atlas that backs a *fraction* of it,
//! and the residency that decides which fraction.
//!
//! This is the **GPU-free half**, deliberately, exactly as `inf-vt` is the
//! GPU-free half of virtual texturing, `inf_vgeom::stream` of meshlet streaming
//! and `inf_terrain::residency` of terrain paging. `inf-render`'s `vsm` modules
//! own the one atlas texture and the one indirection buffer, and do nothing but
//! execute a [`VsmTransaction`]. Everything interesting here is therefore
//! **unit-testable with no adapter**, on every CI leg.
//!
//! # A new crate rather than a widened `inf-vt` — the measurement
//!
//! The ROADMAP's P27.1 clause 2 says *"Ring-0 crate first (`inf-vsm` or inside a
//! widened `inf-vt` — measure which)"*. Measured, and the answer is a new crate:
//!
//! * **`inf-vt` has 3 290 non-test lines. Nineteen of them are reusable
//!   unchanged** — `pack_entry`, `MAX_SLOT_INDEX`, `lru_victim` and
//!   `VtPoolGeometry::slot_origin`, 0.6 % — and even `unpack_entry` is not among
//!   them, because a shadow entry has a [`VSM_ENTRY_NONE`](table::VSM_ENTRY_NONE)
//!   state a virtual texture's cannot have. Everything else is keyed on
//!   `VtTextureDesc`, `TileCoord`, `PageFormat` or the `.inf_tex` container.
//! * **The address spaces are different shapes, not different sizes.** A mip
//!   level has *fewer* tiles than the level below it; a clipmap level has
//!   exactly as many and covers twice the ground, and its parent is
//!   `N/4 + x/2` rather than `x/2`. A widened `inf-vt` would carry two
//!   descriptors, two parent rules, two table layouts and two residencies behind
//!   one name.
//! * **`inf-material` depends on `inf-vt`.** Widening it would put the shadow
//!   page allocator inside the texture importer, transitively, for nothing — the
//!   exact hazard that crate's own docs set its dependency direction to make a
//!   compile error rather than a review question.
//! * **The three laws that DO transfer are laws, not code**: order-independent
//!   marks at a fixed layout, pinned readback latency, and stamps that never
//!   enter a trace. They are restated in [`mark`] and [`residency`], with the
//!   deviations named where they are taken.
//!
//! P28.3 merges the two *brains* — one budget arbiter, one feedback ring, one
//! stamp domain, one want pipeline. That merge is a rewrite of both crates'
//! residency loops, and the types here are named so it is mechanical:
//! [`VsmWant`]/[`VsmAdmit`]/[`VsmEvict`]/[`VsmTransaction`]/[`VsmPriority`]
//! mirror `inf-vt`'s spelling for spelling.
//!
//! # The three laws, and the one that does not transfer
//!
//! **1. A transaction is a pure function of `(state, wants)`.**
//! [`VsmResidency::apply_wants`] sorts its wants into **entry order** — the order
//! the indirection table and the marking bitmask share — evicts by
//! least-recently-stamped with a total tie-break, and allocates the lowest free
//! slot. Two runs of one want sequence produce byte-identical
//! [`VsmTransaction::trace`] output.
//!
//! **2. A stamp is not a measurement.** Slot stamps come from a process-global
//! monotone counter, for the reason `inf_vgeom::stream`, `inf_terrain` and
//! `inf_vt` all give. A stamp must never enter a trace, a golden or a comparison
//! between runs; [`VsmTransaction`] contains none.
//!
//! **3. What `inf-vt` calls the mandatory floor does not exist here, and that is
//! the load-bearing difference.** A virtual texture pins its coarsest mip
//! because a fragment may sample *any* tile and a miss would be a hole. A shadow
//! page is only ever sampled by pixels whose depth marked it, so a page nothing
//! marked is a page nothing reads — and pinning a clipmap's coarsest level would
//! claim `N²` pages per directional light before any evidence at all (**4 096**
//! at the shipped 8 k grid — four times the whole 1 024-page default atlas — and
//! **16 384** at 16 k, which is sixteen times it; the P27.1 audit found this
//! sentence's first draft pairing the 16 k count with the 8 k multiplier, and
//! `address.rs` now pins both). A page with no
//! resident ancestor therefore resolves to `NONE`, which a receiver reads as
//! **lit**: a light leak under budget pressure rather than a black hole, and the
//! honest degradation for a signal whose producer is a rasterizer.
//!
//! # What this crate does NOT decide
//!
//! It does not decide what to want, and it holds **no matrices**. Where a
//! clipmap is centred, which level a pixel justifies and how a light's frustum
//! maps to pages are the renderer's, and they change every frame; what is here is
//! the shape of the address space, which does not. The `inf_terrain::residency`
//! split, kept: *"the caller computes the wants; there is deliberately no camera,
//! no radius and no budget policy in Ring 0."*

pub mod address;
pub mod atlas;
pub mod mark;
pub mod residency;
pub mod table;

pub use address::{VsmDescError, VsmLevelDesc, VsmLightDesc, VsmPage, VsmTreeKind, MAX_VSM_LEVELS};
pub use atlas::{
    plan_atlas, VsmAdvisory, VsmAtlasConfig, VsmAtlasGeometry, DEFAULT_VSM_BUDGET_BYTES,
    VSM_DEFAULT_MAX_TEXTURE_DIM, VSM_PAGE_BORDER, VSM_PAGE_BYTES_PER_TEXEL, VSM_PAGE_SIZE,
};
pub use mark::{VsmMarkLayout, VSM_MARK_BITS_PER_WORD};
pub use residency::{
    resolved_table, VsmAdmit, VsmError, VsmEvict, VsmLightHandle, VsmPriority, VsmResidency,
    VsmResolved, VsmStats, VsmTransaction, VsmWant, MAX_VSM_PAGES_PER_LIGHT, VSM_PRIORITY_MARKED,
    VSM_PRIORITY_SPECULATIVE,
};
pub use table::{
    pack_entry, unpack_entry, VsmEntry, MAX_SLOT_INDEX, VSM_ENTRY_NONE, VSM_TABLE_HEADER_WORDS,
    VSM_TABLE_LEVEL_REC_WORDS, VSM_TABLE_LIGHT_HEADER_WORDS, VSM_TABLE_MAGIC,
};
