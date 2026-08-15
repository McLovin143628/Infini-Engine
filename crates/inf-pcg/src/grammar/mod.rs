//! **Rule-rewriting grammar** (P19.4): splines and footprints become placed
//! modular-mesh instances.
//!
//! Scatter answers *how many of these, over this area*. A grammar answers *what
//! sequence of pieces goes along this line* — a fence following a road spline, a
//! wall around a footprint, a colonnade, a repeating façade bay. Both are
//! deterministic, both are counter-hashed, and both end up as
//! `ScatteredInstance`s on the same GPU-instanced path, so the whole downstream
//! half of the engine needed no change at all.
//!
//! ```text
//!   grammar.rules ─┐                                 ┌─ Span (spline arc length)
//!                  ├─ grammar.expand ─ SCATTER wire ─┤
//!   grammar.spline ┘                                 └─ SpanSet (footprint edges/rows)
//!     grammar.footprint
//! ```
//!
//! * [`dsl`] — the authored rule text: lexer, parser, rule table, module
//!   palette, node-anchored errors. **The grammar of the grammar is documented
//!   in that module's header.**
//! * [`span`] — the 1-D domains: an arc-length polyline from a spline, a
//!   footprint's four edges (plus corner frames) or its parallel rows.
//! * [`expand`] — derivation, **the exact-fill layout**, and placement.
//!
//! # What this is *not* (yet)
//!
//! P19.5 grows this into buildings: floor stacks, room partitioning, doors and
//! windows, per-room furniture, and building-type palettes. The seams that batch
//! needs are already the shape it wants — a [`SpanSet`](span::SpanSet) can carry
//! a floor plate's partition lines, a [`ModuleDef`](dsl::ModuleDef) is one entry
//! of what becomes a building set — but nothing here knows about interiors.
//!
//! # Where a pass lives, and why it is not in `PcgDocument`
//!
//! A [`GrammarPass`](expand::GrammarPass) rides
//! [`LoweredPcg::grammars`](crate::graph::LoweredPcg), **not** the serialized
//! [`PcgDocument`](crate::rules::PcgDocument). `PcgDocument` is the frozen v2
//! `.inf_pcg` wire and bincode is positional, so growing it by one field would
//! make every committed graph fail to decode — a schema bump with a frozen
//! record, for a value nothing needs on disk. Since P19.3 the **authored graph
//! JSON is the source of truth** and every evaluation site re-lowers it (the
//! player included); the stored `document` is explicitly "a convenience mirror".
//! Lowering therefore produces two things — the document and the passes — and
//! the mirror stays exactly what it was.
//!
//! The consequence is stated rather than buried: a `PcgAssetPayload::new(doc)`
//! payload, which carries no graph, carries **no grammar**. That is the same
//! shape as a v1 payload having no image mask and no merge tree.

pub mod dsl;
pub mod expand;
pub mod span;

pub use dsl::{
    Alternative, Element, Grammar, GrammarError, ModuleDef, Primary, Repeat, Rule, SizeSpec,
};
pub use expand::{
    build_spans, derive, evaluate_grammars, evaluate_grammars_in, expand_span, layout, pass_seed,
    place_corners, FootprintMode, GrammarContext, GrammarOutput, GrammarPass, Ground, Layout, Slot,
    SpanSource, LAYOUT_UNITS, MAX_DEPTH, MAX_SLOTS,
};
pub use span::{
    axis_quat, euler_deg_to_quat, footprint_perimeter, footprint_rows, yaw_onto, Frame, NoSplines,
    RowAxis, Span, SpanSet, SplineInterp, SplinePath, SplineSource, DEFAULT_SPLINE_SAMPLES,
};
