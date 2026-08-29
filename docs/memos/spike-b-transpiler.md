# Spike B memo — bidirectional Infini Blueprints ↔ Rust transpiler

**Status:** GO. The graph↔code round trip works with the guarantees the
product needs; the proptest suite and a 38-case hand-edit corpus are green
and stay in CI permanently. Crates: `inf-blueprint` (model),
`inf-transpile` (emit/lift/fingerprint).

## The contract (what "bidirectional" means here)

- The **graph is the source of truth**; the Rust file is a faithful,
  hand-editable projection.
- `lift(generate(g)) == g` for every well-formed graph — proptest-enforced
  isomorphism (512 cases per property run, statement/expression recursion).
- `generate ∘ lift` is **idempotent** byte-for-byte, so the watch-loop
  converges instead of ping-ponging.
- Lifting **never fails** on parseable Rust and **never loses text**:
  anything outside the liftable subset survives as an opaque snippet
  (statement level) or verbatim item (item level).

## Architecture decisions

1. **Emit builds `syn` ASTs directly, never token streams parsed back.**
   Tokens re-associate by operator precedence (`quote!(#a * #b)` with
   `a = x + y` silently becomes `x + y * b`-shaped); AST nodes cannot.
   `prettyplease` inserts exactly the parentheses concrete syntax needs, so
   `(alpha + 1.0) * 2.0` round-trips and `((alpha))` normalizes away.
2. **Node ids live in identifiers** (`n7`, `speed_n7`) — identifiers survive
   `syn`; comments don't. Control-flow nodes are identified structurally by
   position. Fn identity is item-level: `#[infinity::blueprint(id = "…")]`
   (shipped later as a no-op passthrough proc-macro so user code compiles).
3. **Hand-written binders stay bare (`Binding::Raw`).** Renaming `x` →
   `x_n7` on regen would silently unbind snippets that still say `x`. Raw
   binders keep their spelling; their node id is re-derived deterministically
   on each lift (max explicit id + k, lexical encounter order), which the
   proptest isomorphism pins down.
4. **Snippet poisoning.** A name bound by an unliftable statement poisons
   that name: later statements referencing it also stay snippets. Otherwise
   the graph would wire references to a node that doesn't exist (or worse, a
   shadowed older node) and regen would change semantics. Corpus-pinned.
5. **Canonical forms** (each enforced at emit or normalized at lift):
   negative numbers live in literals (never `Neg(lit)`); float spelling is
   Rust's shortest-round-trip `{:?}`; int suffixes normalize away; the final
   `Return` renders as a tail expression; `else if` chains nest as
   single-`If` else-bodies and print idiomatically.
6. **Sidecar fingerprint** — `fingerprint_fn` = xxh3 of the generated
   projection. The editor stores it per fn; a differing hash on file change
   means "hand-edited → lift". Stable across round trips (property-tested).

## Liftable subset (Spike B scope)

f64/i64/bool/String/unit; params; `let`/`let mut` with optional annotation;
assignment; nested unary/binary expression trees (arith, comparison,
`&&`/`||`); free-fn calls with opaque multi-segment paths; `if`/`else if`/
`else`; `while`; early and tail returns; call statements. Everything else —
method calls, macros, `for`/`match`/closures, `if let`, unknown identifiers,
value-carrying nested-block tails, duplicate-id binders — becomes a snippet;
non-fn items and unliftable fns stay verbatim at file level (order
preserved). P6 grows the subset (user types via `.inf_struct`/`.inf_enum`,
methods on engine handles, `for` sugar) — each addition must land with its
round-trip cases.

## Accepted losses / open items (be honest in-product)

- **Plain `//` comments inside a blueprint fn body are dropped on the next
  regeneration** (syn discards them; doc comments are attributes and DO keep
  the whole fn verbatim instead). The editor must say this: bodies of
  synced fns are generated artifacts. A span-based comment re-attachment
  pass is a possible P6+ enhancement.
- `i64::MIN` and non-finite floats have no literal spelling — editor
  validation rejects them (emit errors exist as a backstop).
- Graph edits that wire a *raw*-named node into scopes where its bare
  identifier is shadowed would render wrong text; the P6 editor "adopts" a
  raw binder into `Named` (id-carrying) the first time it's wired.
- Formatting is prettyplease-canonical; a hand-run `rustfmt` with different
  settings will diff once and then converge.

## Verification (in CI from this commit on)

- `roundtrip.rs` — recipe-based proptest (scope-correct reference
  resolution, unique ids, canonical raw-id numbering): isomorphism,
  idempotence, fingerprint stability.
- `hand_edits.rs` — 38 corpus cases: value edits, renames, control flow,
  every documented snippet fallback, poisoning, item preservation, multi-fn
  files. Every case additionally asserts regen convergence.
