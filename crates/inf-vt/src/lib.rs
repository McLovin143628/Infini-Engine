//! **Streaming virtual texturing** (P26.2): the virtual address space of a tiled
//! texture, the physical page pool that backs a *fraction* of it, and the
//! residency state machine that decides which fraction.
//!
//! This is the **GPU-free half**, deliberately, exactly as
//! [`inf_vgeom::stream`](../inf_vgeom/stream/index.html) is the GPU-free half of
//! meshlet streaming and `inf_terrain::residency` is the GPU-free half of terrain
//! paging. `inf-render`'s `vt` module owns the one atlas texture and the one
//! indirection buffer, and does nothing but execute a [`VtTransaction`].
//! Everything interesting here is therefore **unit-testable with no adapter**, on
//! every CI leg, which is what lets the residency gates be exhaustive rather than
//! adapter-gated.
//!
//! # The dependency direction is the Ring rule, made structural
//!
//! `inf-material` depends on **this** crate, not the other way round: the address
//! of a tile is more primitive than the file that stores it, so
//! [`TileCoord`] lives here and `inf_material::tiles`
//! re-exports it. That ordering is not a taste question. `inf-material` pulls
//! `image` and `naga`, and the shipped player does not link it (it is a **dev**
//! dependency of `inf-render`); if `inf-vt` named `inf-material`, then P26.3
//! binding VT into the lit passes would drag an image decoder and a shader
//! validator into every shipped build, transitively and invisibly. Making
//! `inf-material` the *upper* crate turns that mistake into a dependency cycle —
//! a compile error rather than a review question.
//!
//! What crosses the seam instead is bytes and geometry: the container hands a
//! residency its [`VtTextureDesc`] (through
//! [`TiledTextureReader::vt_desc`](container::TiledTextureReader::vt_desc)), and
//! the caller hands the GPU mirror one tile's stored blob (through
//! [`tile`](container::TiledTextureReader::tile) or, on an adapter without
//! `TEXTURE_COMPRESSION_BC`,
//! [`tile_rgba8`](container::TiledTextureReader::tile_rgba8)) — **the same door,
//! one format decision earlier**, which is the premise the P26.1 transcode arm
//! was built on.
//!
//! **P26.3 moved the container's READ half down here** ([`container`]), leaving
//! the writer in `inf-material`. The reason is the same one that set the
//! dependency direction: the shipped player has to read tiles to sample them, and
//! it does not link `inf-material`. See the [`container`] module docs for the
//! exact split and for the one deliberate deviation (the BC *decoder* came with
//! the reader; the encoder did not).
//!
//! # The three laws
//!
//! **1. The coarsest mip is always resident.** Every registered texture's
//! coarsest level — one tile for a full pyramid, the whole grid for a texture
//! imported without mips — is admitted at registration, pinned, and never
//! evicted. This is `inf-vgeom`'s "page 0 holds every root and is never evicted"
//! law applied to textures, and it buys the same thing: **every virtual tile
//! always resolves to some resident tile**, so a sample can be blurry and can
//! never be a hole. [`VtResidency::register_texture`] therefore *refuses*, by
//! name ([`VtError::MandatoryFloorExceedsBudget`]), a registration whose root set
//! would not fit — the floor is claimed up front rather than discovered as a
//! missing texture at frame time.
//!
//! **2. A transaction is a pure function of `(state, wants)`.**
//! [`VtResidency::apply_wants`] sorts its wants into **payload order** (the P26.1 audit's finding:
//! `TileCoord`'s derived `Ord` is `(mip, x, y)` while the tile directory is
//! `(mip, y, x)`, so a residency table keyed on the derived order walks the file
//! backwards), evicts by least-recently-stamped with a total tie-break, and
//! allocates the lowest free slot. Nothing consults a clock, a map iteration
//! order, or a thread. There is **no parallelism here at all** — this is
//! bookkeeping, not compute, and a worker-pool matrix over a serial program
//! measures nothing (the P22.4 law). Two runs of one want sequence therefore
//! produce byte-identical [`VtTransaction::trace`] output, which is what the
//! gates pin.
//!
//! **3. A stamp is not a measurement.** Slot stamps come from a
//! **process-global** monotone counter — `inf_stream::next_stamp` since P28.3,
//! where this crate's own counter merged with `inf-vgeom`'s and `inf-vsm`'s —
//! for the reason `inf_terrain`'s `NEXT_TILE_VERSION` also gives: a GPU mirror caches "the table I last uploaded
//! for texture T was generation N", and a per-residency counter restarting at 1
//! would let a freshly-built residency mint a generation a stale cache already
//! holds — after a budget change, or after a level switch. A global counter never
//! decreases, so that is unreachable. The consequence is the same one those
//! crates carry: **a stamp must never enter a trace, a golden or a comparison
//! between runs**; the only legal operation on it is `!=` against a
//! previously-observed stamp, plus the `<` that orders two stamps *within one
//! process*. [`VtTransaction`] contains none.
//!
//! # What a transaction is, and what "transactional" means
//!
//! A [`VtTransaction`] is the complete difference between two residency states:
//! the pages to write into the atlas, the slots whose contents became
//! unreachable, and the indirection blocks that changed. The mirror applies it
//! **whole, before any frame samples it** — see `inf_render::vt` for the
//! ordering contract P26.3 relies on. There is no partial visibility: a frame
//! either sees the state before a transaction or the state after it, never a
//! half.
//!
//! An **evict is a table operation, not a texture operation.** Nothing is written
//! to the atlas when a page leaves; its texels simply become unreachable, because
//! no live indirection entry names that slot any more. The slot is overwritten
//! the next time it is admitted. That is why [`VtTransaction::evicts`] carries
//! what *left* rather than what to erase.
//!
//! # What this crate does NOT decide
//!
//! It does not decide what to want. The want set is the caller's — analytic
//! (camera + bounds, P26.4's floor), and later refined by the deterministic
//! feedback bitmask. This layer only executes a want set against a budget, which
//! is precisely the split `inf_terrain::residency` documents ("the caller computes
//! the wants; there is deliberately no camera, no radius and no budget policy in
//! Ring 0") and it keeps residency testable as a pure set operation.
//!
//! It does not *choose* a rank either, but since P26.4 it **honours** one: a
//! [`VtWant`] carries a [`VtPriority`], the sort is priority-primary with payload
//! order as the tie-break, and the two ranks that exist are
//! [`VT_PRIORITY_FLOOR`] and [`VT_PRIORITY_FEEDBACK`]. That is what makes the
//! caller's analytic floor a **floor**: a whole priority class is seated before
//! the next is offered a slot, and `apply_wants` never evicts what it has
//! already touched, so a refinement cannot take a floor tile's page. Payload
//! order still holds *inside* a class, so each class is one forward scan of the
//! mmap rather than a scatter.
//!
//! The feedback signal itself is `inf_render`'s — the GPU produces an
//! order-independent coverage bitmask whose **layout** is [`feedback`]'s, and
//! [`VtFeedbackLayout::wants`] is the CPU request scan that turns it back into
//! wants in virtual-address order.

pub mod address;
pub mod container;
pub mod feedback;
/// P26.5's missing-tile fill: the deterministic, integer-only edge-directed
/// upscale a page takes when its own bytes cannot be produced.
pub mod fill;
pub mod pool;
pub mod residency;
pub mod table;

pub use address::{full_pyramid, DescError, TileCoord, VtMipDesc, VtTextureDesc, MAX_VT_MIPS};
pub use container::{
    decode_bc1, decode_bc3, is_v2, stored_page_format, stored_tile_bytes, TexMipEntry,
    TexTileEntry, TiledTextureError, TiledTextureHeader, TiledTextureReader, TiledTextureView,
    STORED_TILE_SIZE, TEX_ASSET_MAGIC, TEX_ASSET_SCHEMA_VERSION, TILE_BORDER, TILE_SIZE,
};
pub use feedback::{VtFeedbackLayout, FEEDBACK_BITS_PER_WORD};
pub use fill::{fill_from_ancestor, replicate2x, upscale2x, MAX_FILL_STEPS};
pub use pool::{
    plan_pool, PageFormat, VtAdvisory, VtPoolConfig, VtPoolGeometry, DEFAULT_MAX_TEXTURE_DIM,
    DEFAULT_VT_BUDGET_BYTES, DEFAULT_VT_UPLOAD_BUDGET_BYTES, VT_SUSTAINED_THROTTLE_FRAMES,
};
pub use residency::{
    resolved_table, VtAdmit, VtError, VtEvict, VtPriority, VtResidency, VtResolved, VtStats,
    VtTextureHandle, VtTransaction, VtWant, VT_PRIORITY_FEEDBACK, VT_PRIORITY_FLOOR,
    VT_PRIORITY_PREDICT,
};
pub use table::{
    pack_entry, unpack_entry, VtEntry, TABLE_HEADER_WORDS, TABLE_MAGIC, TABLE_MIP_REC_WORDS,
    TABLE_TEXTURE_HEADER_WORDS,
};
