//! Bidirectional graph ↔ Rust transpiler (syn/quote/prettyplease).
//!
//! Spike B (docs/ROADMAP.md §5). The graph ([`inf_blueprint`] model) is the
//! source of truth; the generated Rust file is a faithful projection that
//! users may hand-edit. The invariants this crate maintains:
//!
//! - **generate → lift is the identity** on well-formed graphs (proptest-
//!   enforced isomorphism).
//! - **lift never fails on parseable Rust** — anything outside the liftable
//!   subset becomes an opaque [`inf_blueprint::Stmt::Snippet`] (statement
//!   granularity) or a [`FileEntry::Verbatim`] item (item granularity). Hand
//!   edits are never lost.
//! - **regeneration is idempotent**: `generate(lift(generate(lift(src))))`
//!   equals `generate(lift(src))` byte-for-byte, so watch-loop syncs
//!   converge instead of ping-ponging.
//!
//! Codegen builds `syn` ASTs directly (never token streams parsed back),
//! so expression structure can't be re-associated by precedence, and
//! `prettyplease` inserts any parentheses the concrete syntax needs.
//! Known, accepted loss: plain `//` comments inside hand edits do not
//! survive `syn` and are dropped on the next regeneration (doc comments are
//! attributes and DO survive on Verbatim items; see the Spike B memo).

mod emit;
mod lift;

pub use emit::{generate_file, generate_fn, normalize_snippet, EmitError};
pub use lift::{lift_file, LiftError, Lifted};

use inf_blueprint::BlueprintFn;

/// A blueprint source file: blueprint functions interleaved with verbatim
/// hand-written items (helpers, `use` lines, …), in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct BlueprintFile {
    pub entries: Vec<FileEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileEntry {
    Blueprint(BlueprintFn),
    /// A non-blueprint item preserved verbatim (prettyplease-normalized).
    Verbatim(String),
}

/// Stable fingerprint of a function's generated projection. The editor's
/// sidecar stores this per function; on file change, functions whose current
/// source hash differs from the stored fingerprint were hand-edited and get
/// lifted.
pub fn fingerprint_fn(f: &BlueprintFn) -> Result<u64, EmitError> {
    Ok(xxhash_rust::xxh3::xxh3_64(generate_fn(f)?.as_bytes()))
}
