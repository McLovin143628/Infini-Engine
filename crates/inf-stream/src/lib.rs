//! **The one virtual streamer** (P28.3): the arbitration half of virtual
//! geometry, virtual texturing and virtual shadow mapping, written once.
//!
//! UE5 runs three separate page systems with three budgets, three feedback
//! paths and three eviction policies, and its known failure is that they can
//! disagree — a high-poly mesh with a blurry texture is three streamers each
//! doing the locally right thing. P28.2 made that state unrepresentable for one
//! *pair* by binding a cluster page's geometry and its texture tiles into one
//! transaction. This crate is the general form: the three consumers keep their
//! **domain brains** (what to want) and surrender **arbitration** (what to get).
//!
//! # The split, precisely
//!
//! | kept by the consumer | surrendered to here |
//! |---|---|
//! | which pages/tiles a camera justifies | the order they are admitted in |
//! | how a slot is seated and a table maintained | which slot may be taken from whom |
//! | how many bytes a page costs | how many bytes the consumer may spend |
//! | which counters it reports | the stamp every counter is keyed on |
//!
//! Five modules, one per surrendered thing:
//!
//! * [`stamp`] — **one stamp domain.** The three process-global monotone
//!   counters merge into one. A stamp is not a measurement and never enters a
//!   trace.
//! * [`lane`] + [`admit`] — **one want pipeline.** A want carries a [`Lane`]
//!   (analytic floor ∪ feedback refinement ∪ predictor), and
//!   [`admit_by_lane`] is the single admission walk both
//!   slot-pool consumers run. It is where the P28.2 audit's **priority-blind
//!   protection order** is fixed: a floor want outranks a resident refinement.
//! * [`budget`] — **one budget arbiter.** One unified byte ceiling divided into
//!   three grants, deterministically, floors first.
//! * [`couple`] — **cross-system aging.** A group (a cluster page) and its
//!   members (its texture tiles) are wanted together, touched together, and
//!   dropped together, so they cannot age apart.
//! * [`ring`] — **one readback ledger.** The pinned feedback latency and the
//!   slot arithmetic, shared by every consumer that reads the GPU back, so two
//!   consumers reading at frame `F` read the same source frame.
//!
//! # What this crate does NOT decide
//!
//! It holds no camera, no matrix, no page format and no residency. It never
//! decides *what* to want — that is the consumer's, exactly as
//! `inf_terrain::residency` and `inf_vt::residency` both already say. Every
//! function here is a pure function of its arguments: there is no clock, no
//! thread, no map iteration order and no parallelism (bookkeeping is not
//! compute — the P22.4 law). Two routes to one residency triple therefore
//! produce identical arbitration, which is the property
//! `inf-render`'s `tests/stream_arbiter.rs` pins as an oracle over the three
//! real residencies rather than as a unit assertion here.
//!
//! # The three consumers, and the one that shares only half
//!
//! `inf-vt` and `inf-vsm` are **slot pools**: a flat array of interchangeable
//! slots, a residency map, an LRU victim. Both implement
//! [`SlotPool`] and both run [`admit_by_lane`], so the
//! admission order is one implementation.
//!
//! `inf-vgeom` is **not** a slot pool and is not made into one here. Its
//! residency is a *prefix* of a page order over a four-pool suballocator, and
//! its arbitration is a byte auction ordered worst-error-first — a different
//! shape, not a different size. What it surrenders is the byte ceiling
//! ([`budget`]), the stamp domain ([`stamp`]) and the coupling ([`couple`]);
//! what it keeps is the auction. Forcing it through `SlotPool` would mean
//! calling a page a slot, and a page is a variable number of bytes in four
//! pools.

pub mod admit;
pub mod budget;
pub mod couple;
pub mod lane;
pub mod ring;
pub mod stamp;

pub use admit::{admit_by_lane, Acquired, AdmitBudget, AdmitLog, SlotPool};
pub use budget::{arbitrate, BudgetGrant, BudgetRequest, Consumer, CONSUMERS};
pub use couple::{breaches, Coupling};
pub use lane::{normalize, Lane, LANE_FEEDBACK, LANE_FLOOR, LANE_PREDICT};
pub use ring::{
    read_frame, record_slot, ring_slots, RingLedger, RingReader, FEEDBACK_LATENCY_FRAMES,
};
pub use stamp::{next_stamp, Stamp};

/// A refusal the streamer states by name rather than absorbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    #[error(
        "the consumers' mandatory floors are {floor_bytes} B and the unified streaming budget \
         is {total_bytes} B. A floor is what residency may never fall below — the virtual \
         texture's pinned coarsest mips, the meshlet streamer's root pages — so this is a \
         refusal and not a preference: raise the budget, or load content with a smaller floor"
    )]
    FloorExceedsBudget { floor_bytes: u64, total_bytes: u64 },
}
