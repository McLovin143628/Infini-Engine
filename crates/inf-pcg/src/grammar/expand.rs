//! Expansion: derivation, the exact-fill layout, and placement into instances.
//!
//! Three stages, in this order, each a pure function of its inputs:
//!
//! 1. **Derivation** — rewrite the axiom into a flat slot list. Alternatives are
//!    picked by a hashed weighted draw; repetition counts are resolved greedily
//!    left to right against the span's length. No world position exists yet.
//! 2. **Layout** — assign each slot an interval of the span. This is where the
//!    exact fill lives (below).
//! 3. **Placement** — turn `(slot, interval)` into a [`PcgInstance`] through the
//!    span's frame and the module palette.
//!
//! # THE EXACT-FILL ALGORITHM
//!
//! The requirement is that the slot intervals **partition the span exactly** —
//! not "to within an epsilon", because a float epsilon is a tolerance somebody
//! later has to widen. It is met by never accumulating a position:
//!
//! * Each slot gets a **desired** length: a fixed size as authored, a flexible
//!   size drawn from `[min, max]` by the counter hash. The flexible slots are
//!   then water-filled (bounded, deterministic passes) so the desired lengths
//!   sum to the span length as closely as float arithmetic allows.
//! * Those desired lengths are **apportioned into integer ticks** over a fixed
//!   `LAYOUT_UNITS = 2^40` budget by the largest-remainder (Hamilton) method:
//!   `q_i = floor(d_i / Σd · UNITS)`, then the `UNITS − Σq_i` leftover ticks go
//!   one each to the largest fractional remainders, ties by ascending index.
//!   **The tick sum is exactly `UNITS` — an integer identity, not a float
//!   comparison.** (The degenerate all-zero-weight branch has to hand its own
//!   remainder to the last slot to keep that true: `UNITS / n · n` undershoots
//!   for every `n` that is not a power of two.)
//! * A boundary is then `b_k = L · (T_k / UNITS)` where `T_k` is the *prefix* of
//!   ticks. `UNITS` is a power of two and `T_k ≤ 2^40 < 2^53`, so `T_k / UNITS`
//!   is an **exact** f64. Every interior boundary is one multiply off the span
//!   length — never a running sum — so there is no drift to accumulate. (This is
//!   the P17.4 exact-linear lesson: derive the k-th value from k, do not add k
//!   times.)
//! * **The two endpoints are assigned, not computed**, and that — not the tick
//!   arithmetic — is what makes the partition exact: `b_0` is the literal `0.0`
//!   and `b_n` is the literal `span_len`. The apportionment's job is to place
//!   the *interior* boundaries proportionally and monotonically; the ends are
//!   true by construction, so no float identity has to be relied upon for them.
//!
//! The price is stated rather than hidden: a *fixed* size is honoured to within
//! one tick, `L / 2^40` — about **0.9 nanometres on a kilometre-long span**, or
//! 1 part in 10¹². In exchange the span is filled with no residue at all.
//!
//! ## Overflow and underflow
//!
//! * **Overflow** (the minimum lengths exceed the span): slots are dropped from
//!   the **end** until the minimum fits. A wall that cannot fit its last post is
//!   short a post; it is never squeezed, because squeezing would silently
//!   shrink authored module sizes.
//! * **Underflow** (even at maximum length the slots do not reach the end): the
//!   slack becomes a **tail** folded into the final slot's interval. A module is
//!   anchored at the *start* of its interval, so an oversized final interval is
//!   invisible — the span simply ends with empty space, which is the honest
//!   result of a greedy fill. `Layout::tail` reports how much.
//!
//! # Anchoring, and why flexible slots move things rather than stretch them
//!
//! A module is placed at the **start** of its interval, oriented by the span
//! frame there. `ScatteredInstance::scale` is a single `f64`, so there is no
//! non-uniform stretch available on the instancing path a grammar rides; a
//! flexible size therefore changes **spacing**, never mesh size. A palette entry
//! recentres itself with `offset 0,0,<half-length>` when its mesh is modelled
//! around its own origin. Stretching to fit needs a per-instance scale *vector*
//! (an ECS + two-projector change) and is a stated remainder.
//!
//! # Determinism
//!
//! Every draw is `hash(pass_seed, span_index, salt, index)` — the counter-hash
//! doctrine, with the span index folded in so two edges of one footprint make
//! different choices. Nothing is order-dependent: the derivation is a
//! left-to-right DFS, the layout indexes slots, and the tie-break in the
//! apportionment is by ascending index.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::scatter::{PcgCollider, PcgInstance};

use super::dsl::{Alternative, Element, Grammar, ModuleDef, Primary, Repeat, SizeSpec};
use super::span::{
    footprint_perimeter, footprint_rows, positive, Frame, RowAxis, Span, SpanSet, SplineSource,
    DEFAULT_SPLINE_SAMPLES,
};

/// The integer budget one span is apportioned into. A power of two below `2^53`,
/// so `ticks / UNITS` is an exact `f64` and the boundary multiply is the only
/// rounding in the whole layout.
pub const LAYOUT_UNITS: u64 = 1 << 40;

/// The most slots one span may produce. A guard, not a design limit: a fence of
/// 0.5 m panels would need a 50 km span to reach it.
pub const MAX_SLOTS: usize = 100_000;

/// The most copies one `*` / `+` may produce.
pub const MAX_FILL_REPEAT: u64 = MAX_SLOTS as u64;

/// The deepest a derivation nests through rules and groups.
///
/// **This is the termination backstop, not a tuning knob.** The derivation walks
/// natively (`emit_alternatives` → `emit_elements` → `emit_element` →
/// `emit_primary` → `emit_alternatives`), so an unbounded descent is a stack
/// overflow — an uncatchable abort in the editor, the cook and the shipped
/// player alike. Two separate guards stand in front of it and neither is
/// sufficient alone: the parser rejects *cyclic rules* (so a grammar cannot
/// recurse forever) and caps `( … )` nesting at
/// [`MAX_NESTING`](super::dsl::MAX_NESTING) (so one statement cannot nest
/// forever) — but a long **acyclic rule chain** (`R0 -> R1 -> … -> R2000`, which
/// parses and validates happily) descends once per link and is bounded by
/// nothing else. Beyond this depth the branch stops and
/// [`Layout::depth_capped`] is set, so the outcome is truncated content rather
/// than a dead process.
pub const MAX_DEPTH: u32 = 128;

// Per-draw salts, so a choice, an optional and a flexible size drawn for the
// same index are decorrelated (the scatter kernel's own convention).
const SALT_CHOICE: u64 = 0x6742;
const SALT_OPTIONAL: u64 = 0x7853;
const SALT_FLEX: u64 = 0x8964;
/// Separates a grammar pass from every other consumer of the counter hash.
const GRAMMAR_SALT: u64 = 0x0067_7261_6D6D_6172; // "grammar"

/// Where a slot takes its ground height from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ground {
    /// The span's own Y (an authored spline height, or the footprint's plane).
    Span,
    /// The [`HeightProvider`] sampled at the slot's XZ — the PCG idiom, and the
    /// default, so a fence follows the hill the road climbs.
    #[default]
    Terrain,
}

/// How a footprint's rectangle is cut into spans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FootprintMode {
    /// The four sides, plus a corner frame at each vertex.
    Perimeter { corner_size: f64 },
    /// `rows` evenly spaced parallel spans.
    Rows { rows: u32, axis: RowAxis },
}

/// Where a pass's span comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanSource {
    /// A scene entity carrying a `Spline`. `entity: None` means *the evaluating
    /// entity's own spline*, so the common case (a spline and a PCG volume on
    /// one actor) needs no GUID typed anywhere.
    Spline {
        entity: Option<Uuid>,
        samples_per_segment: usize,
    },
    /// A rectangle centred on the evaluating entity. A zero `size` axis falls
    /// back to the volume's own extent, so a footprint dropped on a `PcgVolume`
    /// matches the box the author already sized.
    Footprint { size: DVec2, mode: FootprintMode },
    /// **Explicit world-space points** (Wave G) — a road centreline, a river, a
    /// coastline, a parcel boundary.
    ///
    /// # The 80% of "the engine has no polygon"
    ///
    /// Before this the grammar could reach a span only two ways: a `Spline`
    /// entity, or an axis-aligned rectangle. That is why a GIS polygon fed into
    /// this engine collapsed to its bounding box — not because a polygon ring is
    /// hard, but because there was no way to *hand it in*. `Span::from_points`
    /// has been public the whole time; nothing could call it.
    ///
    /// A polygon ring is a closed polyline, so this one variant unlocks polygon
    /// perimeters, road centrelines, stream courses and coastlines all at once —
    /// which is why it is the highest leverage-to-cost item in the Wave G plan.
    ///
    /// # It costs no schema bump, and that is deliberate
    ///
    /// [`SpanSource`] is not part of [`PcgDocument`] (see the type's own note):
    /// it rides `LoweredPcg::grammars`, and the player re-lowers the stored graph
    /// JSON, which is the source of truth. The graph JSON is self-describing, so
    /// an added variant and an added node are additive.
    ///
    /// Points are **world metres** — already through the GIS import door, so
    /// nothing here knows what a CRS is.
    Polyline {
        points: Vec<DVec3>,
        /// `true` for a ring (a polygon boundary walked as a line), `false` for
        /// an open path.
        closed: bool,
    },
}

/// One lowered grammar pass — the runtime shape of a `grammar.expand` node.
///
/// Deliberately **not** part of [`PcgDocument`](crate::rules::PcgDocument): that
/// type is the frozen v2 `.inf_pcg` wire, bincode-positional, and growing it
/// would break every committed graph for a value the player never reads (it
/// re-lowers the stored graph JSON, which *is* the source of truth). A pass
/// therefore rides [`LoweredPcg::grammars`](crate::graph::LoweredPcg) beside the
/// document. See the P19.4 ROADMAP block for the full argument.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarPass {
    /// The pass name (the node's `name` param) — diagnostics and ordering.
    pub name: String,
    /// The layer this pass was lowered under, so a disabled layer disables its
    /// grammars exactly as it disables its rules.
    pub layer: String,
    pub enabled: bool,
    pub seed: u64,
    pub grammar: Grammar,
    /// The symbol expansion starts from (already resolved: the node's `axiom`
    /// param, or the grammar's first rule).
    pub axiom: String,
    pub span: SpanSource,
    /// The palette entry stamped on a perimeter's corners; empty for none.
    pub corner_module: String,
    pub ground: Ground,
    pub altitude_offset: f64,
}

/// What the evaluating entity contributes to a pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrammarContext {
    /// The evaluating entity's stable GUID, for `SpanSource::Spline`'s
    /// self-reference. `None` disables the self-reference (nothing resolves).
    pub entity: Option<Uuid>,
    /// World-space anchor: a footprint's centre.
    pub center: DVec3,
    /// The volume's XZ **half**-extent; a zero footprint axis uses `2 × extent`.
    pub extent: DVec2,
    /// The volume's own seed, folded into every pass so two volumes sharing one
    /// graph build different things.
    pub seed_offset: u64,
}

impl Default for GrammarContext {
    fn default() -> Self {
        Self {
            entity: None,
            center: DVec3::ZERO,
            extent: DVec2::splat(50.0),
            seed_offset: 0,
        }
    }
}

/// One terminal occurrence in a derivation.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    /// The terminal symbol, kept so a test can assert an exact expansion and a
    /// debug view can name what it placed.
    pub symbol: String,
    /// The palette index, or `None` for a gap (a terminal with no module).
    pub module: Option<u32>,
    pub size: SizeSpec,
}

/// A derivation laid out on a span.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    pub slots: Vec<Slot>,
    /// `slots.len() + 1` boundaries in arc length. `bounds[0] == 0.0` always,
    /// and when there is at least one slot `*bounds.last() == span_length`
    /// **bit-for-bit** (see the module docs).
    ///
    /// An **empty** layout — nothing derived, or everything truncated — is
    /// `[0.0]`: there is nothing to fill, `tail` carries the whole span, and the
    /// exactness statement is vacuous rather than false.
    pub bounds: Vec<f64>,
    /// Slots dropped from the end because their minimum lengths overflowed the
    /// span.
    pub truncated: usize,
    /// Metres of the final interval left empty (the underflow tail).
    pub tail: f64,
    /// A branch stopped at [`MAX_DEPTH`].
    pub depth_capped: bool,
    /// Derivation stopped at [`MAX_SLOTS`].
    pub slot_capped: bool,
}

impl Layout {
    /// The span length this layout fills — by construction, its last boundary.
    pub fn length(&self) -> f64 {
        self.bounds.last().copied().unwrap_or(0.0)
    }

    /// The interval `[start, end)` of slot `i`.
    pub fn interval(&self, i: usize) -> (f64, f64) {
        (self.bounds[i], self.bounds[i + 1])
    }
}

/// A counter-hash draw stream: pure in `(base, salt, index)`, never stateful
/// beyond a monotone walk counter.
#[derive(Clone, Copy)]
struct Draw {
    base: Hash64,
    counter: u64,
}

impl Draw {
    fn new(base: Hash64) -> Self {
        Self { base, counter: 0 }
    }

    /// The next draw in the walk (choices and optionals, whose *position* in the
    /// derivation is what identifies them).
    fn walk(&mut self, salt: u64) -> f64 {
        self.counter = self.counter.wrapping_add(1);
        self.base.mix_u64(salt).mix_u64(self.counter).unit()
    }

    /// A draw indexed by something stable (a slot index), so it does not depend
    /// on how many walk draws preceded it.
    fn indexed(&self, salt: u64, index: u64) -> f64 {
        self.base.mix_u64(salt).mix_u64(index).unit()
    }
}

// ── stage 1: derivation ─────────────────────────────────────────────────────

struct Deriver<'a> {
    g: &'a Grammar,
    draw: Draw,
    slots: Vec<Slot>,
    /// Nominal length committed so far — what a greedy `*` measures against.
    used: f64,
    span_len: f64,
    /// Each rule's nominal length, computed **once, iteratively, bottom-up**
    /// before any derivation runs — see [`rule_nominals`].
    rule_nominal: Vec<f64>,
    depth_capped: bool,
    slot_capped: bool,
}

/// Every rule's nominal length, in declaration order, computed bottom-up over
/// the (acyclic, parser-enforced) rule-reference graph.
///
/// **This exists to remove an exponential, not to save a few cycles.** A rule's
/// nominal length is the weighted mean of its alternatives, which needs the
/// nominal length of every rule it references — so computing it by recursive
/// descent re-walks a *diamond* once per path. `A -> B B`, `B -> C C`,
/// `C -> D D`, … is thirty readable lines and 2³⁰ walks, i.e. a hang in the
/// cook and in the shipped player. Evaluating each rule exactly once, in reverse
/// topological order, makes it linear.
///
/// The traversal is an **explicit-stack post-order DFS** for the same reason the
/// cycle check is: a legal grammar may be a 2000-link chain, and native
/// recursion over it is a stack overflow. A reference the walk has not resolved
/// yet contributes `0`, which cannot happen for an acyclic graph visited
/// post-order and is simply the honest answer if one ever slipped through.
fn rule_nominals(g: &Grammar) -> Vec<f64> {
    let n = g.rules().len();
    let mut out = vec![0.0f64; n];
    let mut done = vec![false; n];
    let mut on_stack = vec![false; n];

    for start in 0..n {
        if done[start] {
            continue;
        }
        // (rule index, next reference to visit).
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        on_stack[start] = true;
        while let Some(&(node, next)) = stack.last() {
            let kids = rule_refs(g, node);
            if next < kids.len() {
                stack.last_mut().expect("non-empty").1 += 1;
                let kid = kids[next];
                if !done[kid] && !on_stack[kid] {
                    on_stack[kid] = true;
                    stack.push((kid, 0));
                }
                continue;
            }
            // Every reference is resolved, so this rule's own value is now a
            // pure table read.
            out[node] = nominal_of_alternatives(g, &g.rules()[node].alternatives, &out);
            done[node] = true;
            on_stack[node] = false;
            stack.pop();
        }
    }
    out
}

/// The rule indices rule `i` references, ascending and deduplicated.
fn rule_refs(g: &Grammar, i: usize) -> Vec<usize> {
    fn walk(g: &Grammar, alts: &[Alternative], out: &mut Vec<usize>) {
        for alt in alts {
            for el in &alt.elements {
                match &el.primary {
                    Primary::Symbol { name, .. } => {
                        if let Some(j) = g.rule_position(name) {
                            out.push(j);
                        }
                    }
                    Primary::Group(inner) => walk(g, inner, out),
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(g, &g.rules()[i].alternatives, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

/// The weighted-mean nominal length of `alts`, reading every rule reference out
/// of the already-computed `table` instead of recursing into it.
fn nominal_of_alternatives(g: &Grammar, alts: &[Alternative], table: &[f64]) -> f64 {
    if alts.is_empty() {
        return 0.0;
    }
    // The weighted mean over the alternatives: what one occurrence costs on
    // average, which is the right reservation for a stochastic rule.
    let total: f64 = alts.iter().map(|a| a.weight.max(0.0)).sum();
    if !positive(total) {
        return 0.0;
    }
    alts.iter()
        .map(|a| {
            let len: f64 = a
                .elements
                .iter()
                .map(|e| nominal_of_element(g, e, table))
                .sum();
            len * a.weight.max(0.0) / total
        })
        .sum()
}

/// One occurrence of a primary, before its repetition is applied.
fn nominal_of_primary(g: &Grammar, p: &Primary, table: &[f64]) -> f64 {
    match p {
        Primary::Group(inner) => nominal_of_alternatives(g, inner, table),
        Primary::Symbol { name, size } => match g.rule_position(name) {
            // A rule reference is a TABLE READ — the whole point of the table.
            Some(i) => table[i],
            // A terminal's size is authored; the parser already guaranteed it
            // resolves.
            None => g
                .terminal_size(name, *size)
                .map(|s| s.nominal())
                .unwrap_or(0.0),
        },
    }
}

fn nominal_of_element(g: &Grammar, el: &Element, table: &[f64]) -> f64 {
    let body = nominal_of_primary(g, &el.primary, table);
    match el.repeat {
        // A `*` takes whatever is left, so it reserves nothing; an optional
        // reserves its full body (reserving too much leaves a tail, which is
        // gentler than the truncation reserving too little would cause).
        Repeat::Fill => 0.0,
        Repeat::One | Repeat::Optional | Repeat::FillAtLeastOne => body,
        Repeat::Exactly(n) => body * n as f64,
    }
}

impl Deriver<'_> {
    /// The length a construct is *expected* to occupy, used only to decide how
    /// many copies a repetition takes and how much to reserve for what follows.
    /// Never a placement quantity. Rule references resolve through the
    /// precomputed [`rule_nominals`] table, so this is linear in the AST it is
    /// handed rather than exponential in the rule graph's depth.
    fn nominal_element(&self, el: &Element) -> f64 {
        nominal_of_element(self.g, el, &self.rule_nominal)
    }

    fn nominal_primary(&self, p: &Primary) -> f64 {
        nominal_of_primary(self.g, p, &self.rule_nominal)
    }

    /// Pick one alternative by hashed weight and emit it. `reserve` is the
    /// nominal length owed to everything that follows this construct.
    fn emit_alternatives(&mut self, alts: &[Alternative], reserve: f64, depth: u32) {
        if depth >= MAX_DEPTH {
            self.depth_capped = true;
            return;
        }
        if alts.is_empty() {
            return;
        }
        let chosen = if alts.len() == 1 {
            0
        } else {
            let total: f64 = alts.iter().map(|a| a.weight.max(0.0)).sum();
            if !positive(total) {
                0
            } else {
                let u = self.draw.walk(SALT_CHOICE) * total;
                let mut acc = 0.0;
                let mut pick = alts.len() - 1;
                for (i, a) in alts.iter().enumerate() {
                    acc += a.weight.max(0.0);
                    if u < acc {
                        pick = i;
                        break;
                    }
                }
                pick
            }
        };
        self.emit_elements(&alts[chosen].elements, reserve, depth);
    }

    fn emit_elements(&mut self, els: &[Element], reserve: f64, depth: u32) {
        // Suffix reservations, computed once: element `i` must leave room for
        // everything after it, plus whatever the enclosing construct owes.
        let noms: Vec<f64> = els.iter().map(|e| self.nominal_element(e)).collect();
        let mut suffix = vec![0.0; els.len() + 1];
        for i in (0..els.len()).rev() {
            suffix[i] = suffix[i + 1] + noms[i];
        }
        for (i, el) in els.iter().enumerate() {
            if self.slots.len() >= MAX_SLOTS {
                self.slot_capped = true;
                return;
            }
            self.emit_element(el, reserve + suffix[i + 1], depth);
        }
    }

    fn emit_element(&mut self, el: &Element, reserve: f64, depth: u32) {
        let count: u64 = match el.repeat {
            Repeat::One => 1,
            Repeat::Exactly(n) => n as u64,
            Repeat::Optional => u64::from(self.draw.walk(SALT_OPTIONAL) < 0.5),
            Repeat::Fill | Repeat::FillAtLeastOne => {
                let body = self.nominal_primary(&el.primary);
                let avail = self.span_len - self.used - reserve;
                let mut c = if body > 0.0 && avail > 0.0 {
                    let f = (avail / body).floor();
                    if f.is_finite() && f > 0.0 {
                        (f as u64).min(MAX_FILL_REPEAT)
                    } else {
                        0
                    }
                } else {
                    0
                };
                if el.repeat == Repeat::FillAtLeastOne {
                    c = c.max(1);
                }
                c
            }
        };
        for _ in 0..count {
            if self.slots.len() >= MAX_SLOTS {
                self.slot_capped = true;
                return;
            }
            self.emit_primary(&el.primary, reserve, depth);
        }
    }

    fn emit_primary(&mut self, p: &Primary, reserve: f64, depth: u32) {
        match p {
            Primary::Group(alts) => self.emit_alternatives(alts, reserve, depth + 1),
            Primary::Symbol { name, size } => match self.g.rule(name) {
                Some(rule) => self.emit_alternatives(&rule.alternatives, reserve, depth + 1),
                None => {
                    let spec = self
                        .g
                        .terminal_size(name, *size)
                        .unwrap_or(SizeSpec::Fixed(0.0));
                    self.used += spec.nominal();
                    self.slots.push(Slot {
                        symbol: name.clone(),
                        module: self.g.module_index(name),
                        size: spec,
                    });
                }
            },
        }
    }
}

/// Derive the slot sequence for `axiom` over a span of `span_len` metres.
///
/// Pure in `(grammar, axiom, span_len, base)`. An axiom that names no rule is
/// treated as a single terminal occurrence, so a one-module grammar
/// (`module Bollard = … size 2` + axiom `Bollard`) works without writing a rule.
pub fn derive(grammar: &Grammar, axiom: &str, span_len: f64, base: Hash64) -> Deriverd {
    let mut d = Deriver {
        g: grammar,
        draw: Draw::new(base),
        slots: Vec::new(),
        used: 0.0,
        span_len,
        rule_nominal: rule_nominals(grammar),
        depth_capped: false,
        slot_capped: false,
    };
    match grammar.rule(axiom) {
        Some(rule) => {
            let alts = rule.alternatives.clone();
            d.emit_alternatives(&alts, 0.0, 0);
        }
        // An axiom naming a declared module places that module once — the
        // one-piece grammar (`module Bollard = … size 2`, axiom `Bollard`) needs
        // no rule. An axiom naming NOTHING places nothing, rather than a
        // zero-length gap that would silently look like success.
        None if grammar.module_index(axiom).is_some() => {
            d.emit_primary(
                &Primary::Symbol {
                    name: axiom.to_string(),
                    size: None,
                },
                0.0,
                0,
            );
        }
        None => {}
    }
    Deriverd {
        slots: d.slots,
        depth_capped: d.depth_capped,
        slot_capped: d.slot_capped,
    }
}

/// The result of [`derive`]: the slot sequence plus the guards it tripped.
#[derive(Debug, Clone, PartialEq)]
pub struct Deriverd {
    pub slots: Vec<Slot>,
    pub depth_capped: bool,
    pub slot_capped: bool,
}

// ── stage 2: the exact-fill layout ──────────────────────────────────────────

/// Lay `slots` out over `span_len` metres. See the module docs for the
/// algorithm, the overflow/underflow policies and the exactness statement.
pub fn layout(mut slots: Vec<Slot>, span_len: f64, base: Hash64) -> Layout {
    let draw = Draw::new(base);
    if !positive(span_len) || slots.is_empty() {
        return Layout {
            slots: Vec::new(),
            bounds: vec![0.0],
            truncated: slots.len(),
            tail: span_len.max(0.0),
            depth_capped: false,
            slot_capped: false,
        };
    }

    // Overflow: keep the longest prefix whose MINIMUM lengths fit.
    let mut keep = slots.len();
    let mut acc = 0.0;
    for (i, s) in slots.iter().enumerate() {
        acc += s.size.min();
        if acc > span_len {
            keep = i;
            break;
        }
    }
    let truncated = slots.len() - keep;
    slots.truncate(keep);
    if slots.is_empty() {
        return Layout {
            slots,
            bounds: vec![0.0],
            truncated,
            tail: span_len,
            depth_capped: false,
            slot_capped: false,
        };
    }

    // Desired lengths: fixed as authored, flexible drawn in range.
    let mut d: Vec<f64> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| match s.size {
            SizeSpec::Fixed(v) => v,
            SizeSpec::Flex { min, max } => min + draw.indexed(SALT_FLEX, i as u64) * (max - min),
        })
        .collect();

    let sum_max: f64 = slots.iter().map(|s| s.size.max()).sum();
    let mut tail = 0.0;
    if sum_max < span_len {
        // Underflow: everything at maximum, the slack becomes the tail.
        for (i, s) in slots.iter().enumerate() {
            d[i] = s.size.max();
        }
        tail = span_len - sum_max;
    } else {
        water_fill(&slots, &mut d, span_len);
    }
    // The tail lives in the final interval; a module anchored at the interval's
    // start does not notice, and the exactness identity is preserved.
    if tail > 0.0 {
        if let Some(last) = d.last_mut() {
            *last += tail;
        }
    }

    let bounds = apportion(&d, span_len);
    Layout {
        slots,
        bounds,
        truncated,
        tail,
        depth_capped: false,
        slot_capped: false,
    }
}

/// Move the desired lengths toward `target` within each slot's `[min, max]`,
/// proportionally to the headroom each slot has, in a bounded number of passes.
///
/// Deterministic: every quantity is computed in slot order from float values
/// that are themselves pure functions of the inputs. Called only when
/// `Σ min ≤ target ≤ Σ max`, so it always converges to within float noise.
fn water_fill(slots: &[Slot], d: &mut [f64], target: f64) {
    for _ in 0..8 {
        let sum: f64 = d.iter().sum();
        let delta = target - sum;
        if delta == 0.0 {
            return;
        }
        // Room available in the direction we need to move.
        let room: f64 = slots
            .iter()
            .zip(d.iter())
            .map(|(s, v)| {
                if delta > 0.0 {
                    (s.size.max() - v).max(0.0)
                } else {
                    (v - s.size.min()).max(0.0)
                }
            })
            .sum();
        if !positive(room) {
            return;
        }
        let scale = (delta.abs() / room).min(1.0);
        for (i, s) in slots.iter().enumerate() {
            let step = if delta > 0.0 {
                (s.size.max() - d[i]).max(0.0) * scale
            } else {
                -(d[i] - s.size.min()).max(0.0) * scale
            };
            d[i] += step;
        }
    }
}

/// Largest-remainder apportionment of `LAYOUT_UNITS` ticks over `weights`, then
/// the boundaries `L · (prefix / UNITS)`.
///
/// The returned vector has `weights.len() + 1` entries; its first is exactly
/// `0.0` and its last exactly `span_len`.
fn apportion(weights: &[f64], span_len: f64) -> Vec<f64> {
    let n = weights.len();
    let total: f64 = weights.iter().map(|w| w.max(0.0)).sum();
    let mut ticks = vec![0u64; n];
    if !positive(total) {
        // Degenerate: every slot is zero-length (or the weights are NaN). Split
        // evenly and give the **remainder to the last slot**, because `UNITS / n
        // · n` undershoots for every `n` that is not a power of two — an even
        // split alone would leave the ticks summing to less than `UNITS`.
        let each = LAYOUT_UNITS / n as u64;
        for t in ticks.iter_mut() {
            *t = each;
        }
        ticks[n - 1] += LAYOUT_UNITS - each * n as u64;
    } else {
        let mut fracs: Vec<(f64, usize)> = Vec::with_capacity(n);
        let mut used: u64 = 0;
        for (i, w) in weights.iter().enumerate() {
            let exact = w.max(0.0) / total * LAYOUT_UNITS as f64;
            let q = if exact.is_finite() && exact > 0.0 {
                (exact.floor() as u64).min(LAYOUT_UNITS)
            } else {
                0
            };
            let q = q.min(LAYOUT_UNITS - used);
            ticks[i] = q;
            used += q;
            fracs.push((exact - exact.floor(), i));
        }
        // Leftover ticks to the largest fractional remainders, ties by index —
        // so the apportionment is a pure function of the weights alone.
        let mut rem = LAYOUT_UNITS - used;
        if rem > 0 {
            fracs.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            for &(_, i) in fracs.iter() {
                if rem == 0 {
                    break;
                }
                ticks[i] += 1;
                rem -= 1;
            }
            // More leftovers than slots (only reachable when every weight is
            // sub-tick): the remainder lands on the last slot.
            if rem > 0 {
                ticks[n - 1] += rem;
            }
        }
    }

    let mut bounds = Vec::with_capacity(n + 1);
    let inv = 1.0 / LAYOUT_UNITS as f64;
    let mut prefix: u64 = 0;
    bounds.push(0.0);
    for (i, t) in ticks.iter().enumerate() {
        prefix += t;
        // The final boundary is the span length ITSELF, not `L · 1.0` — the
        // identity is by construction, not by arithmetic luck.
        bounds.push(if i + 1 == n {
            span_len
        } else {
            span_len * (prefix as f64 * inv)
        });
    }
    bounds
}

// ── stage 3: placement ──────────────────────────────────────────────────────

/// What an expansion produces: the visible instances and the solid boxes the
/// modules that declared a `collider` contribute (P19.5).
///
/// The two lists are parallel in *origin* but not in length — a module without a
/// `collider` attribute contributes an instance and no box, which is every
/// module authored before P19.5 — so they are kept as two vectors rather than as
/// one list of optional pairs. Concatenation is associative, which is what lets
/// [`evaluate_grammars_in`] fold a `parallel_map`'s per-job outputs in job order
/// and stay pool-size invariant.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GrammarOutput {
    pub instances: Vec<PcgInstance>,
    /// Solid boxes, in the same order their instances were placed.
    pub colliders: Vec<PcgCollider>,
}

impl GrammarOutput {
    /// Append `other`'s lists to this one's.
    pub fn extend(&mut self, other: GrammarOutput) {
        self.instances.extend(other.instances);
        self.colliders.extend(other.colliders);
    }

    /// `true` when nothing was placed at all.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty() && self.colliders.is_empty()
    }
}

/// Place one frame's module, or `None` when the ground is missing there.
///
/// The optional second half is the module's collision box: centred on the same
/// world point as the instance (the module's `offset` is already applied there),
/// scaled by the module's uniform `scale`, and oriented by the **slot yaw
/// alone** — see [`ModuleDef::collider`] for why the authored euler is
/// deliberately not folded in.
fn place_module(
    frame: &Frame,
    module: &ModuleDef,
    kind: u32,
    ground: Ground,
    altitude_offset: f64,
    height: &dyn HeightProvider,
) -> Option<(PcgInstance, Option<PcgCollider>)> {
    let yaw = frame.rotation();
    let offset = yaw * module.offset;
    let x = frame.position.x + offset.x;
    let z = frame.position.z + offset.z;
    let base = match ground {
        Ground::Span => frame.position.y,
        Ground::Terrain => height.height(x, z)?,
    };
    let pos = DVec3::new(x, base + offset.y + altitude_offset, z);
    let solid = module.collider.map(|h| PcgCollider {
        center: pos,
        half_extents: h * module.scale,
        rotation: yaw,
    });
    Some((
        PcgInstance {
            pos,
            rotation: yaw * super::span::euler_deg_to_quat(module.rotation_deg),
            scale: module.scale,
            kind_index: kind,
        },
        solid,
    ))
}

/// Expand `pass` along one span and place the result.
pub fn expand_span(
    pass: &GrammarPass,
    span: &Span,
    base: Hash64,
    height: &dyn HeightProvider,
) -> GrammarOutput {
    if span.is_degenerate() {
        return GrammarOutput::default();
    }
    let derived = derive(&pass.grammar, &pass.axiom, span.length(), base);
    let lay = layout(derived.slots, span.length(), base);
    let modules = pass.grammar.modules();
    let mut out = GrammarOutput {
        instances: Vec::with_capacity(lay.slots.len()),
        colliders: Vec::new(),
    };
    for (i, slot) in lay.slots.iter().enumerate() {
        let Some(kind) = slot.module else { continue };
        let Some(module) = modules.get(kind as usize) else {
            continue;
        };
        let frame = span.frame_at(lay.bounds[i]);
        if let Some((inst, solid)) = place_module(
            &frame,
            module,
            kind,
            pass.ground,
            pass.altitude_offset,
            height,
        ) {
            out.instances.push(inst);
            out.colliders.extend(solid);
        }
    }
    out
}

/// Stamp the corner module on every corner frame of a footprint perimeter.
pub fn place_corners(
    pass: &GrammarPass,
    corners: &[Frame],
    height: &dyn HeightProvider,
) -> GrammarOutput {
    let mut out = GrammarOutput::default();
    if pass.corner_module.is_empty() || corners.is_empty() {
        return out;
    }
    let Some(kind) = pass.grammar.module_index(&pass.corner_module) else {
        return out;
    };
    let Some(module) = pass.grammar.modules().get(kind as usize) else {
        return out;
    };
    for f in corners {
        if let Some((inst, solid)) =
            place_module(f, module, kind, pass.ground, pass.altitude_offset, height)
        {
            out.instances.push(inst);
            out.colliders.extend(solid);
        }
    }
    out
}

/// Build the span set a pass expands along, resolving a spline reference
/// through `splines` and a footprint against `cx`.
pub fn build_spans(pass: &GrammarPass, splines: &dyn SplineSource, cx: &GrammarContext) -> SpanSet {
    match &pass.span {
        SpanSource::Spline {
            entity,
            samples_per_segment,
        } => {
            let guid = entity.or(cx.entity);
            let Some(path) = guid.and_then(|g| splines.spline(g)) else {
                return SpanSet::default();
            };
            let per = if *samples_per_segment == 0 {
                DEFAULT_SPLINE_SAMPLES
            } else {
                *samples_per_segment
            };
            SpanSet::one(Span::from_spline(&path, per))
        }
        SpanSource::Footprint { size, mode } => {
            // A zero axis means "use the volume's own box".
            let sx = if size.x > 0.0 {
                size.x
            } else {
                cx.extent.x * 2.0
            };
            let sz = if size.y > 0.0 {
                size.y
            } else {
                cx.extent.y * 2.0
            };
            let size = DVec2::new(sx, sz);
            match *mode {
                FootprintMode::Perimeter { corner_size } => {
                    footprint_perimeter(cx.center, size, corner_size)
                }
                FootprintMode::Rows { rows, axis } => footprint_rows(cx.center, size, rows, axis),
            }
        }
        SpanSource::Polyline { points, closed } => {
            // Guarded here rather than trusted: these points come from a file,
            // and a span built on a NaN produces module placements whose
            // transforms are non-finite — which `f32::min`/`max` then report
            // healthy bounds for. Refusing is not an option at this seam (a
            // grammar pass has no error channel and an erroring pass takes its
            // whole handler down — the P21 law that a gameplay refusal is a
            // VALUE), so a bad polyline yields NO spans, which is the same
            // outcome an unresolvable spline reference already has.
            if points.len() < 2 || points.iter().any(|p| !p.is_finite()) {
                return SpanSet::default();
            }
            SpanSet::one(Span::from_points(points.iter().copied(), *closed))
        }
    }
}

/// The counter-hash base one pass runs under: its authored seed folded with the
/// grammar salt and the evaluating volume's own seed.
///
/// Stated in one place so the editor and the player cannot derive it
/// differently — the whole reason the biome binding keeps `biome_seed` public.
pub fn pass_seed(pass_seed: u64, volume_seed: u64) -> Hash64 {
    Hash64::new(pass_seed)
        .mix_u64(GRAMMAR_SALT)
        .mix_u64(volume_seed)
}

/// Evaluate every enabled pass on the process-wide job pool.
pub fn evaluate_grammars(
    passes: &[GrammarPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> GrammarOutput {
    evaluate_grammars_in(inf_core::global(), passes, splines, height, cx)
}

/// [`evaluate_grammars`] on a caller-supplied pool — the seam the determinism
/// guard drives, mirroring [`scatter_region_in`](crate::scatter::scatter_region_in).
///
/// Spans are mapped through [`inf_core::parallel_map`] (a deterministic in-order
/// pure map) and concatenated in pass-then-span order, so the population is
/// byte-identical for any worker count.
pub fn evaluate_grammars_in(
    pool: &inf_core::JobPool,
    passes: &[GrammarPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> GrammarOutput {
    /// One unit of work: a pass, and either one of its spans or its corners.
    enum Job {
        Span(usize, Span, Hash64),
        Corners(usize, Vec<Frame>),
    }

    let mut jobs: Vec<Job> = Vec::new();
    for (pi, pass) in passes.iter().enumerate() {
        if !pass.enabled {
            continue;
        }
        let set = build_spans(pass, splines, cx);
        let base = pass_seed(pass.seed, cx.seed_offset);
        for (si, span) in set.spans.into_iter().enumerate() {
            jobs.push(Job::Span(pi, span, base.mix_u64(si as u64)));
        }
        if !set.corners.is_empty() {
            jobs.push(Job::Corners(pi, set.corners));
        }
    }
    if jobs.is_empty() {
        return GrammarOutput::default();
    }
    let per: Vec<GrammarOutput> = pool.parallel_map(jobs, |job| match job {
        Job::Span(pi, span, base) => expand_span(&passes[pi], &span, base, height),
        Job::Corners(pi, corners) => place_corners(&passes[pi], &corners, height),
    });
    let mut out = GrammarOutput::default();
    for chunk in per {
        out.extend(chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    //! The P19.4 tests below were written when expansion returned a bare
    //! `Vec<PcgInstance>`. P19.5 made it return instances **and** solids, so
    //! these four shims hand back just the instances — keeping every existing
    //! assertion about *placement* readable, and leaving the collider half to
    //! the tests that are about it (`dsl::tests::a_collider_declaration_places_a_solid`
    //! and the whole `building` module).
    fn placed(
        pass: &GrammarPass,
        span: &Span,
        base: Hash64,
        height: &dyn HeightProvider,
    ) -> Vec<PcgInstance> {
        super::expand_span(pass, span, base, height).instances
    }

    fn placed_corners(
        pass: &GrammarPass,
        corners: &[Frame],
        height: &dyn HeightProvider,
    ) -> Vec<PcgInstance> {
        super::place_corners(pass, corners, height).instances
    }

    fn evaluated(
        passes: &[GrammarPass],
        splines: &dyn SplineSource,
        height: &dyn HeightProvider,
        cx: &GrammarContext,
    ) -> Vec<PcgInstance> {
        super::evaluate_grammars(passes, splines, height, cx).instances
    }

    fn evaluated_in(
        pool: &inf_core::JobPool,
        passes: &[GrammarPass],
        splines: &dyn SplineSource,
        height: &dyn HeightProvider,
        cx: &GrammarContext,
    ) -> Vec<PcgInstance> {
        super::evaluate_grammars_in(pool, passes, splines, height, cx).instances
    }

    use super::*;
    use crate::height::FnHeight;
    use inf_math::spline::SplineInterp;

    fn flat() -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(|_, _| Some(0.0))
    }

    const POST: &str = "6f9619ff-8b86-d011-b42d-00c04fc964ff";
    const PANEL: &str = "6f9619ff-8b86-d011-b42d-00c04fc964f0";

    fn fence_grammar() -> Grammar {
        Grammar::parse(&format!(
            "module Post = mesh {POST} size 0.2\n\
             module Panel = mesh {PANEL} size 2\n\
             Fence -> Post Panel* Post\n"
        ))
        .unwrap()
    }

    fn pass_with(grammar: Grammar, axiom: &str, span: SpanSource) -> GrammarPass {
        GrammarPass {
            name: "g".into(),
            layer: "layer".into(),
            enabled: true,
            seed: 7,
            grammar,
            axiom: axiom.into(),
            span,
            corner_module: String::new(),
            ground: Ground::Terrain,
            altitude_offset: 0.0,
        }
    }

    fn line_span(len: f64) -> Span {
        Span::from_points([DVec3::ZERO, DVec3::new(0.0, 0.0, len)], false)
    }

    fn names(lay: &Layout) -> Vec<&str> {
        lay.slots.iter().map(|s| s.symbol.as_str()).collect()
    }

    fn lay_out(g: &Grammar, axiom: &str, len: f64, seed: u64) -> Layout {
        let base = Hash64::new(seed);
        let d = derive(g, axiom, len, base);
        let mut lay = layout(d.slots, len, base);
        lay.depth_capped = d.depth_capped;
        lay.slot_capped = d.slot_capped;
        lay
    }

    // ── the exactness statement ─────────────────────────────────────────────

    /// **The parametric identity, not a tolerance.** For every corpus grammar
    /// and every span length, the boundaries start at exactly `0` and end at
    /// exactly the span length — compared on `to_bits()`, so a future sloppy
    /// `PartialEq` cannot hide a near-miss — and they are non-decreasing.
    #[test]
    fn the_layout_fills_the_span_exactly() {
        let g = fence_grammar();
        let mixed = Grammar::parse(&format!(
            "module Panel = mesh {PANEL} size 2\n\
             Wall -> (Panel | Gap[0.5..1.5])* Panel\n"
        ))
        .unwrap();
        for grammar in [&g, &mixed] {
            let axiom = grammar.default_axiom().unwrap().to_string();
            for len in [
                0.5,
                1.0,
                2.0,
                2.4,
                7.0,
                11.3,
                100.0,
                1000.0,
                12_345.678_9,
                1e6,
            ] {
                for seed in [0u64, 1, 42, 9_999] {
                    let lay = lay_out(grammar, &axiom, len, seed);
                    assert_eq!(lay.bounds.len(), lay.slots.len() + 1);
                    assert_eq!(lay.bounds[0].to_bits(), 0.0f64.to_bits(), "start != 0");
                    if lay.slots.is_empty() {
                        // Nothing fits: the span is untouched, all tail.
                        assert_eq!(lay.bounds, vec![0.0]);
                        assert_eq!(lay.tail, len);
                        continue;
                    }
                    assert_eq!(
                        lay.bounds.last().unwrap().to_bits(),
                        len.to_bits(),
                        "len={len} seed={seed}: the span is not filled exactly"
                    );
                    for w in 1..lay.bounds.len() {
                        assert!(
                            lay.bounds[w] >= lay.bounds[w - 1],
                            "boundaries went backwards at {w}"
                        );
                    }
                }
            }
        }
    }

    /// A fixed size survives the quantization to well within a micrometre —
    /// the stated price of exact fill, asserted rather than assumed.
    #[test]
    fn fixed_sizes_survive_quantization_to_a_nanometre() {
        let g = fence_grammar();
        for len in [10.0, 137.5, 1000.0] {
            let lay = lay_out(&g, "Fence", len, 3);
            for (i, slot) in lay.slots.iter().enumerate() {
                if let SizeSpec::Fixed(want) = slot.size {
                    // The last interval absorbs the underflow tail by design.
                    if i + 1 == lay.slots.len() && lay.tail > 0.0 {
                        continue;
                    }
                    let got = lay.bounds[i + 1] - lay.bounds[i];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "slot {i} ({}) wanted {want} got {got}",
                        slot.symbol
                    );
                }
            }
        }
    }

    /// The tick apportionment sums to `LAYOUT_UNITS` for any weight vector,
    /// including the degenerate ones — the integer identity underneath the
    /// float identity above.
    ///
    /// **The degenerate branch is the interesting one.** An all-zero weight
    /// vector cannot be normalized, so it splits the budget evenly — and
    /// `UNITS / n · n` *undershoots* for every `n` that is not a power of two,
    /// which would silently leave the last boundary short of the span. `n = 3`
    /// (and 5, and 7) is exactly that case, so it is enumerated rather than
    /// implied. The endpoints are still assigned literally, which is what makes
    /// them exact; the ticks are what make the *interior* proportional.
    #[test]
    fn apportionment_is_exact_for_any_weights() {
        for weights in [
            vec![1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0 / 3.0; 7],
            vec![0.0, 0.0],
            // Degenerate at counts that do NOT divide 2^40.
            vec![0.0, 0.0, 0.0],
            vec![0.0; 5],
            vec![0.0; 7],
            vec![0.0; 100],
            vec![f64::NAN, f64::NAN, f64::NAN],
            vec![-1.0, -2.0, -3.0],
            vec![1e-30, 1.0, 1e-30],
            vec![f64::MIN_POSITIVE; 5],
            (1..=97).map(|i| i as f64).collect(),
        ] {
            let n = weights.len();
            let b = apportion(&weights, 10.0);
            assert_eq!(b.len(), n + 1);
            assert_eq!(b[0].to_bits(), 0.0f64.to_bits());
            assert_eq!(b.last().unwrap().to_bits(), 10.0f64.to_bits());
            for w in 1..b.len() {
                assert!(b[w] >= b[w - 1], "{weights:?} produced {b:?}");
                assert!(b[w].is_finite());
            }
            // A degenerate vector must still divide the span EVENLY rather than
            // collapsing everything into the last interval — the failure mode an
            // undershooting even split hides behind the assigned endpoint.
            if weights.iter().all(|w| !positive(*w)) && n <= 100 {
                let expect = 10.0 / n as f64;
                for w in 1..n {
                    assert!(
                        (b[w] - b[w - 1] - expect).abs() < 1e-9,
                        "degenerate n={n} split unevenly at {w}: {b:?}"
                    );
                }
            }
        }
    }

    // ── derivation ──────────────────────────────────────────────────────────

    /// **A known small grammar expands to an exact expected sequence.**
    #[test]
    fn a_known_grammar_expands_to_the_expected_sequence() {
        let g = fence_grammar();
        // 0.2 + 0.2 posts reserved; 10 − 0.4 = 9.6 ⇒ four 2 m panels.
        let lay = lay_out(&g, "Fence", 10.0, 1);
        assert_eq!(
            names(&lay),
            vec!["Post", "Panel", "Panel", "Panel", "Panel", "Post"]
        );
        // Too short for even one panel: just the two posts.
        assert_eq!(names(&lay_out(&g, "Fence", 1.0, 1)), vec!["Post", "Post"]);
        // Too short for the second post: truncated from the END.
        let tiny = lay_out(&g, "Fence", 0.3, 1);
        assert_eq!(names(&tiny), vec!["Post"]);
        assert_eq!(tiny.truncated, 1);
        // Nothing fits at all.
        let none = lay_out(&g, "Fence", 0.1, 1);
        assert!(none.slots.is_empty());
        assert_eq!(none.truncated, 2);
    }

    #[test]
    fn repeat_operators_do_what_they_say() {
        let g = Grammar::parse(&format!(
            "module P = mesh {PANEL} size 1\n\
             Exact -> P{{3}}\n\
             Plus -> P+\n\
             Fill -> P*\n\
             Opt -> P?\n"
        ))
        .unwrap();
        assert_eq!(lay_out(&g, "Exact", 100.0, 1).slots.len(), 3);
        assert_eq!(
            lay_out(&g, "Exact", 2.5, 1).slots.len(),
            2,
            "then truncated"
        );
        assert_eq!(lay_out(&g, "Fill", 7.0, 1).slots.len(), 7);
        assert_eq!(lay_out(&g, "Fill", 0.5, 1).slots.len(), 0);
        // `+` places one even when it cannot fit … and then the layout truncates.
        assert_eq!(lay_out(&g, "Plus", 7.0, 1).slots.len(), 7);
        assert_eq!(lay_out(&g, "Plus", 0.5, 1).truncated, 1);
        // `?` is an even coin over seeds.
        let heads = (0..64)
            .filter(|s| !lay_out(&g, "Opt", 100.0, *s).slots.is_empty())
            .count();
        assert!((16..48).contains(&heads), "biased optional: {heads}/64");
    }

    #[test]
    fn weighted_alternatives_respect_their_weights() {
        let g = Grammar::parse(&format!(
            "module A = mesh {POST} size 1\n\
             module B = mesh {PANEL} size 1\n\
             Row -> Pick*\n\
             Pick -> A | B@3\n"
        ))
        .unwrap();
        let lay = lay_out(&g, "Row", 4000.0, 5);
        assert_eq!(lay.slots.len(), 4000);
        let a = lay.slots.iter().filter(|s| s.symbol == "A").count();
        let b = lay.slots.len() - a;
        // 1:3 weighting — allow generous slack, this is a distribution check.
        assert!(b > a * 2, "a={a} b={b}");
        assert!(a > 500, "alternative A never fired: a={a}");
    }

    #[test]
    fn a_terminal_with_no_module_is_a_gap_that_still_consumes_span() {
        let g = Grammar::parse(&format!(
            "module P = mesh {PANEL} size 2\nWall -> (P Gap[1])*\n"
        ))
        .unwrap();
        let lay = lay_out(&g, "Wall", 12.0, 1);
        assert_eq!(lay.slots.len(), 8, "4 × (P + Gap)");
        assert_eq!(lay.slots.iter().filter(|s| s.module.is_none()).count(), 4);
        // The gaps really occupy their metre.
        for (i, s) in lay.slots.iter().enumerate() {
            if s.symbol == "Gap" {
                assert!((lay.bounds[i + 1] - lay.bounds[i] - 1.0).abs() < 1e-6);
            }
        }
        assert_eq!(g.gaps(), vec!["Gap"]);
    }

    /// Flexible slots absorb the remainder instead of leaving a tail, and stay
    /// inside their authored range.
    #[test]
    fn flexible_slots_absorb_the_remainder_within_their_range() {
        let g = Grammar::parse(&format!(
            "module P = mesh {PANEL} size 2\nWall -> P Flex[0.5..3] P Flex[0.5..3] P\n"
        ))
        .unwrap();
        for len in [7.5, 8.0, 9.9, 11.9] {
            let lay = lay_out(&g, "Wall", len, 2);
            assert_eq!(lay.slots.len(), 5);
            assert_eq!(
                lay.tail, 0.0,
                "len={len} left a tail it could have absorbed"
            );
            for (i, s) in lay.slots.iter().enumerate() {
                let got = lay.bounds[i + 1] - lay.bounds[i];
                assert!(
                    got >= s.size.min() - 1e-6 && got <= s.size.max() + 1e-6,
                    "len={len} slot {i} ({}) got {got}, range {}..{}",
                    s.symbol,
                    s.size.min(),
                    s.size.max()
                );
            }
        }
        // Beyond the flexible range the slack becomes an honest tail.
        let long = lay_out(&g, "Wall", 20.0, 2);
        assert!(long.tail > 0.0);
        assert_eq!(long.bounds.last().unwrap().to_bits(), 20.0f64.to_bits());
    }

    #[test]
    fn derivation_is_a_pure_function_of_its_seed() {
        let g = Grammar::parse(&format!(
            "module A = mesh {POST} size 1\nmodule B = mesh {PANEL} size 1\n\
             R -> (A | B)*\n"
        ))
        .unwrap();
        let a = lay_out(&g, "R", 60.0, 11);
        assert_eq!(a, lay_out(&g, "R", 60.0, 11));
        assert_ne!(names(&a), names(&lay_out(&g, "R", 60.0, 12)));
    }

    #[test]
    fn an_axiom_that_names_a_module_places_it_directly() {
        let g = Grammar::parse(&format!("module Bollard = mesh {POST} size 2\n")).unwrap();
        let lay = lay_out(&g, "Bollard", 5.0, 1);
        assert_eq!(names(&lay), vec!["Bollard"]);
        assert_eq!(lay.slots[0].module, Some(0));
        // An axiom naming nothing at all produces nothing.
        assert!(lay_out(&g, "Nope", 5.0, 1).slots.is_empty());
        assert!(lay_out(&g, "", 5.0, 1).slots.is_empty());
    }

    /// **A diamond-shaped rule graph must be LINEAR to expand, not exponential.**
    ///
    /// `A -> B B`, `B -> C C`, … is thirty readable lines whose derivation tree
    /// has 2³⁰ leaves; the nominal length of `A` needs `B`'s, which needs `C`'s.
    /// Computing that by recursive descent re-walks every path — a hang in the
    /// cook and in the shipped player, on content that parses and validates
    /// cleanly. `rule_nominals` evaluates each rule once, bottom-up, so this
    /// returns instantly. If the table is ever removed this test does not fail;
    /// it *hangs*, which is why it is bounded by the harness rather than by an
    /// assertion, and why the depth is chosen so the honest answer is fast and
    /// the naive one is not (2³⁰ ≈ 10⁹ walks).
    #[test]
    fn a_diamond_rule_graph_does_not_expand_exponentially() {
        const DEPTH: usize = 30;
        let mut src = format!("module P = mesh {PANEL} size 1\n");
        for i in 0..DEPTH {
            src.push_str(&format!("R{i} -> R{} R{}\n", i + 1, i + 1));
        }
        src.push_str(&format!("R{DEPTH} -> P\n"));
        let g = Grammar::parse(&src).unwrap();

        // The nominal table itself is the thing under test: R{DEPTH} is one
        // panel, and each level doubles.
        let noms = rule_nominals(&g);
        assert_eq!(noms[DEPTH], 1.0);
        assert_eq!(noms[DEPTH - 1], 2.0);
        assert_eq!(noms[0], (1u64 << DEPTH) as f64);

        // …and a derivation over a span far too small for it truncates rather
        // than materializing 2^30 slots.
        let lay = lay_out(&g, "R0", 10.0, 1);
        assert!(lay.slots.len() <= MAX_SLOTS);
        assert_eq!(lay.bounds[0].to_bits(), 0.0f64.to_bits());
    }

    /// A long **acyclic** rule chain parses and validates happily (the cycle
    /// check is explicit-stack), so nothing upstream bounds the derivation's own
    /// native recursion — [`MAX_DEPTH`] is what does, and it truncates rather
    /// than aborting the process.
    #[test]
    fn a_deep_rule_chain_is_capped_rather_than_overflowing_the_stack() {
        const N: usize = 4000;
        let mut src = format!("module P = mesh {PANEL} size 1\n");
        for i in 0..N {
            src.push_str(&format!("R{i} -> R{}\n", i + 1));
        }
        src.push_str(&format!("R{N} -> P\n"));
        let g = Grammar::parse(&src).unwrap();
        assert_eq!(rule_nominals(&g)[0], 1.0, "the table survives the chain");

        let lay = lay_out(&g, "R0", 100.0, 1);
        assert!(lay.depth_capped, "the derivation must report the cap");
        assert!(lay.slots.is_empty(), "the capped branch places nothing");
        // A chain just inside the cap still reaches its terminal.
        let mut short = format!("module P = mesh {PANEL} size 1\n");
        for i in 0..(MAX_DEPTH as usize - 2) {
            short.push_str(&format!("R{i} -> R{}\n", i + 1));
        }
        short.push_str(&format!("R{} -> P\n", MAX_DEPTH as usize - 2));
        let g = Grammar::parse(&short).unwrap();
        let lay = lay_out(&g, "R0", 100.0, 1);
        assert!(!lay.depth_capped);
        assert_eq!(names(&lay), vec!["P"]);
    }

    #[test]
    fn the_slot_cap_bounds_a_pathological_span() {
        let g = Grammar::parse(&format!("module P = mesh {PANEL} size 0.001\nR -> P*\n")).unwrap();
        let lay = lay_out(&g, "R", 1e9, 1);
        assert!(lay.slots.len() <= MAX_SLOTS);
        assert_eq!(lay.bounds.last().unwrap().to_bits(), 1e9f64.to_bits());
    }

    #[test]
    fn a_zero_size_repeat_body_cannot_loop_forever() {
        let g = Grammar::parse(&format!(
            "module Z = mesh {POST} size 0\nmodule P = mesh {PANEL} size 1\nR -> Z* P\n"
        ))
        .unwrap();
        let lay = lay_out(&g, "R", 100.0, 1);
        assert_eq!(names(&lay), vec!["P"], "a zero-nominal `*` must place none");
    }

    // ── placement ───────────────────────────────────────────────────────────

    #[test]
    fn modules_are_anchored_at_their_slot_start_and_face_along_the_span() {
        let g = fence_grammar();
        let pass = pass_with(
            g,
            "Fence",
            SpanSource::Footprint {
                size: DVec2::ONE,
                mode: FootprintMode::Perimeter { corner_size: 0.0 },
            },
        );
        let span = line_span(10.0);
        let out = placed(&pass, &span, Hash64::new(1), &flat());
        assert_eq!(out.len(), 6);
        // The span runs +Z from the origin, so anchors sit at the boundaries.
        assert_eq!(out[0].pos, DVec3::ZERO);
        assert!((out[1].pos.z - 0.2).abs() < 1e-6);
        assert!((out[5].pos.z - 8.2).abs() < 1e-6);
        // +Z tangent ⇒ identity yaw.
        for i in &out {
            assert!(
                (i.rotation.to_array()[3] - 1.0).abs() < 1e-12,
                "{:?}",
                i.rotation
            );
            assert_eq!(i.pos.y, 0.0);
        }
        // `kind_index` IS the palette index, so a viewer can tell posts from panels.
        assert_eq!(out[0].kind_index, 0);
        assert_eq!(out[1].kind_index, 1);
    }

    #[test]
    fn a_module_offset_is_expressed_in_the_slot_frame() {
        // `offset 0,0,1` centres a 2 m panel; on a span running +X the offset
        // must move it along +X, not +Z.
        let g = Grammar::parse(&format!(
            "module P = mesh {PANEL} size 2 offset 0,0.5,1\nR -> P\n"
        ))
        .unwrap();
        let pass = pass_with(
            g,
            "R",
            SpanSource::Spline {
                entity: None,
                samples_per_segment: 8,
            },
        );
        let span = Span::from_points([DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0)], false);
        let out = placed(&pass, &span, Hash64::new(1), &flat());
        assert_eq!(out.len(), 1);
        assert!(
            (out[0].pos - DVec3::new(1.0, 0.5, 0.0)).length() < 1e-9,
            "{:?}",
            out[0].pos
        );
    }

    /// **P19.5's solid half.** A module that declares a `collider` emits one
    /// box per instance, centred exactly where the instance is, scaled with it,
    /// and oriented by the **slot yaw** rather than by the module's authored
    /// euler — and a module without one emits nothing, so a P19.4 grammar's
    /// output is unchanged.
    #[test]
    fn a_collider_declaration_places_a_solid_beside_its_instance() {
        let g = Grammar::parse(
            "module Solid = size 2 offset 0,1.5,1 scale 2 rot 0,90,0 collider 0.1,1.5,1\n\
             module Airy  = size 2\n\
             W -> Solid Airy Solid\n",
        )
        .unwrap();
        let mut pass = pass_with(
            g,
            "W",
            SpanSource::Spline {
                entity: None,
                samples_per_segment: 8,
            },
        );
        pass.ground = Ground::Span;
        let span = Span::from_points([DVec3::ZERO, DVec3::new(0.0, 4.0, 6.0)], false);
        let out = super::expand_span(&pass, &span, Hash64::new(2), &flat());
        assert_eq!(out.instances.len(), 3, "every module places an instance");
        assert_eq!(out.colliders.len(), 2, "only the declared ones are solid");
        // The box sits on its instance and carries the SCALED half-extents.
        let solids: Vec<&PcgInstance> =
            out.instances.iter().filter(|i| i.kind_index == 0).collect();
        for (inst, solid) in solids.iter().zip(out.colliders.iter()) {
            assert_eq!(solid.center, inst.pos);
            assert_eq!(solid.half_extents, DVec3::new(0.2, 3.0, 2.0));
            // The instance carries the authored 90° yaw; the box does NOT.
            assert_ne!(solid.rotation, inst.rotation);
            assert_eq!(solid.rotation, span.frame_at(0.0).rotation());
        }
        // A grammar with no collider at all is exactly what it was in P19.4.
        let plain = Grammar::parse("module P = size 2\nW -> P+\n").unwrap();
        let plain_pass = pass_with(
            plain,
            "W",
            SpanSource::Spline {
                entity: None,
                samples_per_segment: 8,
            },
        );
        assert!(
            super::expand_span(&plain_pass, &span, Hash64::new(2), &flat())
                .colliders
                .is_empty()
        );
    }

    #[test]
    fn ground_mode_chooses_between_the_span_and_the_terrain() {
        let g = Grammar::parse(&format!("module P = mesh {PANEL} size 2\nR -> P\n")).unwrap();
        let span = Span::from_points(
            [DVec3::new(0.0, 5.0, 0.0), DVec3::new(0.0, 5.0, 10.0)],
            false,
        );
        let ramp = FnHeight::new(|_, z| Some(z * 0.5 + 100.0));
        let mut pass = pass_with(
            g,
            "R",
            SpanSource::Spline {
                entity: None,
                samples_per_segment: 8,
            },
        );
        pass.ground = Ground::Span;
        assert_eq!(placed(&pass, &span, Hash64::new(1), &ramp)[0].pos.y, 5.0);
        pass.ground = Ground::Terrain;
        assert_eq!(placed(&pass, &span, Hash64::new(1), &ramp)[0].pos.y, 100.0);
        pass.altitude_offset = 2.0;
        assert_eq!(placed(&pass, &span, Hash64::new(1), &ramp)[0].pos.y, 102.0);
        // No ground there ⇒ no instance (the scatter kernel's own rule).
        let hole = FnHeight::new(|_, _| None);
        assert!(placed(&pass, &span, Hash64::new(1), &hole).is_empty());
    }

    #[test]
    fn corners_are_stamped_with_the_named_module() {
        let g = Grammar::parse(&format!(
            "module Corner = mesh {POST} size 1\nmodule P = mesh {PANEL} size 2\nR -> P*\n"
        ))
        .unwrap();
        let mut pass = pass_with(
            g,
            "R",
            SpanSource::Footprint {
                size: DVec2::new(10.0, 10.0),
                mode: FootprintMode::Perimeter { corner_size: 1.0 },
            },
        );
        pass.corner_module = "Corner".into();
        let set = build_spans(
            &pass,
            &super::super::span::NoSplines,
            &GrammarContext::default(),
        );
        let corners = placed_corners(&pass, &set.corners, &flat());
        assert_eq!(corners.len(), 4);
        assert!(corners.iter().all(|c| c.kind_index == 0));
        // An unknown corner module stamps nothing rather than panicking.
        pass.corner_module = "Nope".into();
        assert!(placed_corners(&pass, &set.corners, &flat()).is_empty());
        pass.corner_module = String::new();
        assert!(placed_corners(&pass, &set.corners, &flat()).is_empty());
    }

    #[test]
    fn a_footprint_with_zero_size_uses_the_volume_extent() {
        let g = fence_grammar();
        let pass = pass_with(
            g,
            "Fence",
            SpanSource::Footprint {
                size: DVec2::ZERO,
                mode: FootprintMode::Perimeter { corner_size: 0.0 },
            },
        );
        let cx = GrammarContext {
            extent: DVec2::new(6.0, 4.0),
            ..GrammarContext::default()
        };
        let set = build_spans(&pass, &super::super::span::NoSplines, &cx);
        let lens: Vec<f64> = set.spans.iter().map(|s| s.length()).collect();
        assert_eq!(lens, vec![12.0, 8.0, 12.0, 8.0]);
    }

    #[test]
    fn a_spline_span_resolves_the_entity_or_the_self_reference() {
        use std::collections::HashMap;
        let guid = Uuid::from_u128(0xABC);
        let mut map: HashMap<Uuid, super::super::span::SplinePath> = HashMap::new();
        map.insert(
            guid,
            super::super::span::SplinePath {
                points: vec![DVec3::ZERO, DVec3::new(0.0, 0.0, 20.0)],
                closed: false,
                interp: SplineInterp::Linear,
            },
        );
        let g = fence_grammar();
        // Explicit reference.
        let pass = pass_with(
            g.clone(),
            "Fence",
            SpanSource::Spline {
                entity: Some(guid),
                samples_per_segment: 4,
            },
        );
        let set = build_spans(&pass, &map, &GrammarContext::default());
        assert!((set.spans[0].length() - 20.0).abs() < 1e-9);
        // Self reference through the context.
        let pass = pass_with(
            g,
            "Fence",
            SpanSource::Spline {
                entity: None,
                samples_per_segment: 0,
            },
        );
        let cx = GrammarContext {
            entity: Some(guid),
            ..GrammarContext::default()
        };
        assert!((build_spans(&pass, &map, &cx).spans[0].length() - 20.0).abs() < 1e-9);
        // Unresolvable ⇒ no span at all (fails closed).
        assert!(build_spans(&pass, &map, &GrammarContext::default()).is_empty());
        assert!(build_spans(&pass, &super::super::span::NoSplines, &cx).is_empty());
    }

    /// **The Wave G span door**: explicit world points become a span, a closed
    /// one walks its return leg, and bad input fails closed rather than
    /// producing a degenerate span.
    ///
    /// This is the variant that unlocks GIS geometry for the grammar — a road
    /// centreline, a river, a coastline, a parcel boundary — because a polygon
    /// ring is a closed polyline and `Span::from_points` was already public with
    /// nothing able to call it.
    #[test]
    fn a_polyline_span_takes_explicit_world_points() {
        let g = fence_grammar();
        let cx = GrammarContext::default();

        // An open path: a 3-4-5 dogleg is 5 + 12 = 17 m and not 17-something,
        // so a length that arrives right is arriving through arc length rather
        // than through a straight line between the ends.
        let pass = pass_with(
            g.clone(),
            "Fence",
            SpanSource::Polyline {
                points: vec![
                    DVec3::ZERO,
                    DVec3::new(3.0, 0.0, 4.0),
                    DVec3::new(3.0, 0.0, 16.0),
                ],
                closed: false,
            },
        );
        let set = build_spans(&pass, &super::super::span::NoSplines, &cx);
        assert_eq!(set.spans.len(), 1);
        assert!(
            (set.spans[0].length() - 17.0).abs() < 1e-9,
            "got {}",
            set.spans[0].length()
        );

        // A ring: the same three points closed walk the return leg home too.
        let ring = pass_with(
            g.clone(),
            "Fence",
            SpanSource::Polyline {
                points: vec![
                    DVec3::ZERO,
                    DVec3::new(3.0, 0.0, 4.0),
                    DVec3::new(3.0, 0.0, 16.0),
                ],
                closed: true,
            },
        );
        let ring_len = build_spans(&ring, &super::super::span::NoSplines, &cx).spans[0].length();
        let home = (DVec3::new(3.0, 0.0, 16.0) - DVec3::ZERO).length();
        assert!(
            (ring_len - (17.0 + home)).abs() < 1e-9,
            "a closed ring is {ring_len}, expected {} — the return leg is missing",
            17.0 + home
        );

        // The points really are world positions: a translated polyline puts its
        // span somewhere else rather than being re-centred on the context.
        let moved = pass_with(
            g.clone(),
            "Fence",
            SpanSource::Polyline {
                points: vec![DVec3::new(500.0, 0.0, 500.0), DVec3::new(500.0, 0.0, 510.0)],
                closed: false,
            },
        );
        let mset = build_spans(&moved, &super::super::span::NoSplines, &cx);
        assert!((mset.spans[0].length() - 10.0).abs() < 1e-9);
        let f = mset.spans[0].frame_at(0.0);
        assert!(
            (f.position.x - 500.0).abs() < 1e-6 && (f.position.z - 500.0).abs() < 1e-6,
            "the span starts at {:?}, not at the world position it was given",
            f.position
        );

        // **Fails closed**, exactly as an unresolvable spline reference does: a
        // grammar pass has no error channel, so a bad polyline places nothing
        // rather than taking its handler down.
        for bad in [
            vec![],
            vec![DVec3::ZERO],
            vec![DVec3::ZERO, DVec3::new(f64::NAN, 0.0, 1.0)],
            vec![
                DVec3::new(0.0, f64::INFINITY, 0.0),
                DVec3::new(1.0, 0.0, 1.0),
            ],
        ] {
            let p = pass_with(
                g.clone(),
                "Fence",
                SpanSource::Polyline {
                    points: bad.clone(),
                    closed: false,
                },
            );
            assert!(
                build_spans(&p, &super::super::span::NoSplines, &cx).is_empty(),
                "{bad:?} produced a span"
            );
        }
    }

    // ── the pass-level evaluation ───────────────────────────────────────────

    fn perimeter_pass() -> GrammarPass {
        let g = Grammar::parse(&format!(
            "module Corner = mesh {POST} size 1\n\
             module Panel = mesh {PANEL} size 2\n\
             Wall -> Panel*\n"
        ))
        .unwrap();
        let mut p = pass_with(
            g,
            "Wall",
            SpanSource::Footprint {
                size: DVec2::new(20.0, 12.0),
                mode: FootprintMode::Perimeter { corner_size: 1.0 },
            },
        );
        p.corner_module = "Corner".into();
        p
    }

    /// **Pool-size invariance through the real path** — every pass, every span,
    /// the corners, the concatenation order.
    #[test]
    fn the_population_is_invariant_under_pool_size() {
        use inf_core::JobPool;
        let passes = vec![perimeter_pass(), {
            let mut p = perimeter_pass();
            p.name = "second".into();
            p.seed = 99;
            p.span = SpanSource::Footprint {
                size: DVec2::new(30.0, 30.0),
                mode: FootprintMode::Rows {
                    rows: 5,
                    axis: RowAxis::X,
                },
            };
            p.corner_module = String::new();
            p
        }];
        let cx = GrammarContext::default();
        let runs: Vec<Vec<PcgInstance>> = [1usize, 2, 4, 8]
            .into_iter()
            .map(|n| {
                evaluated_in(
                    &JobPool::new(n),
                    &passes,
                    &super::super::span::NoSplines,
                    &flat(),
                    &cx,
                )
            })
            .collect();
        assert!(runs[0].len() > 50, "only {} instances", runs[0].len());
        for (n, r) in [1usize, 2, 4, 8].into_iter().zip(&runs).skip(1) {
            assert_eq!(r, &runs[0], "population differs on a {n}-worker pool");
        }
        assert_eq!(
            evaluated(&passes, &super::super::span::NoSplines, &flat(), &cx),
            runs[0]
        );
    }

    #[test]
    fn a_disabled_pass_places_nothing_and_the_volume_seed_moves_things() {
        let mut p = perimeter_pass();
        let cx = GrammarContext::default();
        let base = evaluated(&[p.clone()], &super::super::span::NoSplines, &flat(), &cx);
        assert!(!base.is_empty());
        p.enabled = false;
        assert!(evaluated(&[p.clone()], &super::super::span::NoSplines, &flat(), &cx).is_empty());
        p.enabled = true;
        // Two volumes sharing one graph must not build the same thing.
        let other = GrammarContext {
            seed_offset: 1,
            ..cx
        };
        let moved = evaluated(&[p], &super::super::span::NoSplines, &flat(), &other);
        assert_eq!(moved.len(), base.len(), "the structure is the same size");
        assert_ne!(pass_seed(7, 0).finish(), pass_seed(7, 1).finish());
    }

    /// Two spans of one set draw differently — the span index really is folded
    /// into the counter hash, so the four sides of a footprint are not clones.
    #[test]
    fn each_span_of_a_set_draws_independently() {
        let g = Grammar::parse(&format!(
            "module A = mesh {POST} size 1\nmodule B = mesh {PANEL} size 1\nR -> (A | B)*\n"
        ))
        .unwrap();
        let pass = pass_with(
            g,
            "R",
            SpanSource::Footprint {
                size: DVec2::new(40.0, 40.0),
                mode: FootprintMode::Rows {
                    rows: 2,
                    axis: RowAxis::X,
                },
            },
        );
        let base = pass_seed(pass.seed, 0);
        let set = build_spans(
            &pass,
            &super::super::span::NoSplines,
            &GrammarContext::default(),
        );
        let kinds = |i: usize| -> Vec<u32> {
            placed(&pass, &set.spans[i], base.mix_u64(i as u64), &flat())
                .iter()
                .map(|x| x.kind_index)
                .collect()
        };
        assert_eq!(kinds(0).len(), 40);
        assert_ne!(kinds(0), kinds(1), "both rows drew the identical sequence");
    }

    /// **Cross-run bit identity.** Two evaluations of the same inputs produce
    /// byte-identical positions and rotations — compared on `to_bits()`.
    #[test]
    fn evaluation_is_bit_identical_across_runs() {
        let passes = vec![perimeter_pass()];
        let cx = GrammarContext::default();
        let bits = |v: &[PcgInstance]| -> Vec<[u64; 8]> {
            v.iter()
                .map(|i| {
                    let r = i.rotation.to_array();
                    [
                        i.pos.x.to_bits(),
                        i.pos.y.to_bits(),
                        i.pos.z.to_bits(),
                        r[0].to_bits(),
                        r[1].to_bits(),
                        r[2].to_bits(),
                        r[3].to_bits(),
                        i.scale.to_bits(),
                    ]
                })
                .collect()
        };
        let a = evaluated(&passes, &super::super::span::NoSplines, &flat(), &cx);
        let b = evaluated(&passes, &super::super::span::NoSplines, &flat(), &cx);
        assert!(!a.is_empty());
        assert_eq!(bits(&a), bits(&b));
    }
}
