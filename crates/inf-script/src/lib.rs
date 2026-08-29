//! **InfiniScript** — the `.infini` text face on the Infini Blueprints IR.
//!
//! A `.infini` file is not a program in a new language. It is a **third face on
//! one program**: the same `inf_blueprint::BlueprintFn` a graph lowers to and the
//! transpiler renders to Rust. Text parses *into* that IR and prints back *out*
//! of it, so a script and a graph are the same thing seen from two sides, and the
//! shipped build is the same Rust either way.
//!
//! ```text
//!                                    ┌─▶ interpreter ──▶ in-editor, hot-swapped
//!                                    │        ⇕  PARITY: interpreted == compiled
//!   .infini text ──parse──▶ IR ──────┼─▶ transpile ──▶ Rust ──▶ the shipped binary
//!        ▲                           │
//!        └──────── emit ─────────────┤
//!                                    └─▶ raise ──▶ graph ──▶ lower ──▶ IR
//! ```
//!
//! # The three doors
//!
//! | door | what it is for |
//! |---|---|
//! | [`source::compile_path`] | **a file on disk** → a class; the encoding contract lives here |
//! | [`compile`] | a whole file's *text* → a `BlueprintClass` the editor can instantiate |
//! | [`parse::parse_fn`] / [`emit::emit_fn`] | one handler, the graph↔text bridge's Ring-0 half |
//! | [`emit::emit_class`] | a class → the file that re-parses to it |
//!
//! [`source`] is the **file door** SCRIPT1b added and SCRIPT1a did not have: the
//! crate read no filesystem at all, so "the encoding is UTF-8" was a claim about
//! whoever called it. The watcher, the cook and the PIE payload builder all
//! enter through it.
//!
//! # What the round trip guarantees, stated exactly
//!
//! Three laws, each gated in `tests/roundtrip.rs`, and the third is the honest
//! replacement for a claim the first two cannot make:
//!
//! 1. **`parse(emit(f)) == f`, exactly, for every `f` the parser produces.**
//!    Ids, binding kinds, type annotations and mutability all survive. This is
//!    every `.infini` file anyone writes.
//! 2. **`emit(parse(emit(f))) == emit(f)`, exactly, for every `f`** — including
//!    IR that came from a graph. The text is a fixed point: open a Blueprint as
//!    text, save it, and the bytes do not move.
//! 3. **`parse(emit(f))` runs identically to `f`** — same trace, byte for byte,
//!    on the same host. For graph-lowered IR the *ids* of synthetic locals may
//!    renumber into the parser's walk order; the **program** does not change, and
//!    the gate compares interpreter traces rather than asserting the equality it
//!    cannot have.
//!
//! # Determinism
//!
//! A file's IR is a pure function of its bytes. The lexer normalises line
//! endings first, so a CRLF checkout and an LF checkout lower to the same IR
//! (`tests/determinism.rs` hashes both). Nothing in the pipeline reads a clock,
//! a path, an environment variable or a hash seed; the parser's only counter is
//! its own local ids.
//!
//! And a script **cannot name a transcendental — only a verb**. Calls resolve
//! against the node kit ([`verbs`]), whose `math.*` builtins route to
//! `inf_math::portable`, and whose everything-else crosses the one `Host` trait.
//! There is no path from `.infini` text to `std` or `libm`.
//!
//! # Where the honest bounds are written down
//!
//! `docs/memos/infiniscript-direction.md`, appendix A — the language spec — has
//! the grammar and the **per-construct** table of what raises to a graph and what
//! only exists as text. [`emit`]'s module docs carry the short version: the text
//! face is total over the IR, and the graph face is not.

pub mod emit;
pub mod lex;
pub mod parse;
pub mod source;
pub mod verbs;

use inf_blueprint::BlueprintClass;

pub use emit::{emit_class, emit_fn, EmitError};
pub use lex::Span;
pub use parse::{parse_fn, parse_unit, Unit};
pub use source::{compile_bytes, compile_path, is_script_path, MAX_SOURCE_BYTES, SCRIPT_EXT};
pub use verbs::{Verb, VerbError, Verbs};

/// How much a diagnostic matters. A warning does not stop a script compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

/// One thing the compiler has to say about a source file, with the place it
/// happened. **Refusals are values** (P21): nothing here panics, and every
/// message names a line, a column and — wherever there is one — a remedy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        write!(f, "{}: {kind}: {}", self.span, self.message)
    }
}

/// Render diagnostics as one block of text, one per line — what a CLI prints
/// and what a test asserts on.
pub fn render(diags: &[Diagnostic]) -> String {
    diags
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compile a whole `.infini` source into an actor class under `id`.
///
/// `Ok` carries the class **and** any warnings; `Err` carries the diagnostics of
/// a file that could not be compiled at all.
pub fn compile(
    source: &str,
    id: impl Into<String>,
) -> Result<(BlueprintClass, Vec<Diagnostic>), Vec<Diagnostic>> {
    let (unit, diags) = parse_unit(source)?;
    Ok((unit.into_class(id), diags))
}
