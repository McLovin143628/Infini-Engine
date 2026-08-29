//! **The file door** — the one place a `.infini` file becomes a `&str`.
//!
//! # Why there is a door at all
//!
//! SCRIPT1a shipped a parser and no reader. `parse_unit` takes a `&str`, and a
//! grep for `std::fs` across the crate returned nothing, so the appendix's *"the
//! encoding is UTF-8"* was a statement about **the caller** — and the caller did
//! not exist yet. The SCRIPT1a audit routed that here by name: *"a `.infini`
//! holding invalid bytes must decode into a diagnostic with a line, not an
//! `unwrap`"*. P21's law at the door rather than inside it.
//!
//! SCRIPT1b gives the crate three callers at once — a watcher, a cook and a PIE
//! payload builder — which is exactly the shape this house has been burned by
//! before (*one door for three paths*, P22: the cook, the PIE payload and the
//! Simulate seeder each derived fractures, two of them separately, and they
//! agreed "by construction" and did not). So all three read a script through
//! [`compile_path`], and everything below it is one function deep.
//!
//! # What the door refuses, and what it silently repairs
//!
//! | | |
//! |---|---|
//! | the file will not open | a diagnostic naming the path and the OS error |
//! | more than [`MAX_SOURCE_BYTES`] | a diagnostic naming **both numbers** — LOUD, because the alternative is a parser that appears to hang |
//! | not UTF-8 | a diagnostic naming the **byte offset** of the first bad byte, at the line and column that offset lands on |
//! | a leading byte-order mark | **repaired** — dropped, exactly as [`crate::lex::strip_bom`] does |
//! | CRLF, or a lone CR | left alone here; the lexer normalises them, and it is the lexer's spans that a human reads |
//!
//! The BOM is repaired rather than refused for the reason the audit gave when it
//! found the parser refusing one: it is what a Windows editor puts in a file it
//! saved, and the wave's own determinism claim is that a Windows checkout and a
//! Unix one lower identically. Dropping it *here* as well as in the lexer is not
//! belt-and-braces — the door's output is what a cook hashes and what a diff
//! shows, so the two must agree about where the text starts.

use std::path::Path;

use inf_blueprint::BlueprintClass;

use crate::lex::{strip_bom, Span};
use crate::{Diagnostic, Severity};

/// The extension a script file carries, without the dot.
///
/// Spelled once. `inf_asset::AssetKind::Script` maps the same string, and
/// `tests/file_door.rs` pins the two against each other so a rename cannot make
/// the asset database and the compiler disagree about what a script is.
pub const SCRIPT_EXT: &str = "infini";

/// The largest `.infini` file the door will read: **1 MiB**.
///
/// Not a performance bound — the SCRIPT1a audit measured a twenty-million
/// character line through the lexer at 0.09 s and the whole hostile corpus at
/// 0.38 s, so the parser is not what a large file threatens. It is a
/// **plausibility** bound: 1 MiB is on the order of thirty thousand lines of a
/// language with no modules and no `require`, and a `.infini` bigger than that
/// is an accident — a binary renamed, a log redirected, a merge that
/// concatenated a repository. The failure mode without a bound is the worst
/// kind: the editor's watcher parses it on every save and nothing says why the
/// editor got slow.
///
/// So the refusal is deliberately **LOUD**: it names the file's real size and
/// the limit, in bytes, rather than saying "too large".
pub const MAX_SOURCE_BYTES: usize = 1024 * 1024;

/// A refusal shaped like every other one this crate makes: a severity, a place
/// and a message. Built here rather than inline so the three door failures
/// cannot drift into three different spellings.
fn refuse(span: Span, message: String) -> Diagnostic {
    Diagnostic {
        severity: Severity::Error,
        span,
        message,
    }
}

/// The 1-based line and column a byte offset lands on, counted over the bytes
/// that *did* decode.
///
/// Columns are in **characters**, matching [`Span`]'s contract, so the column of
/// a bad byte sitting after an accented letter is the column a human counts.
/// The offset itself is carried in the message, because a byte offset is what
/// `iconv`, a hex editor and `git` all speak and a line/column is not.
fn place_of(valid: &str) -> Span {
    let line = valid.bytes().filter(|b| *b == b'\n').count() as u32 + 1;
    let col = valid
        .rsplit('\n')
        .next()
        .map(|l| l.chars().count() as u32 + 1)
        .unwrap_or(1);
    Span { line, col, len: 0 }
}

/// Decode the bytes of a `.infini` file into source text.
///
/// The whole encoding contract, in one function: a size bound, UTF-8 validation
/// that names the byte offset it failed at, and the byte-order-mark strip.
/// `label` is what the diagnostic calls the thing being decoded — a path, for a
/// file on disk.
pub fn decode<'a>(bytes: &'a [u8], label: &str) -> Result<&'a str, Diagnostic> {
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(refuse(
            Span {
                line: 1,
                col: 1,
                len: 0,
            },
            format!(
                "{label} is {} bytes, over the {MAX_SOURCE_BYTES}-byte limit for a \
                 .infini file — a script this size is usually a file that is not a \
                 script; split it, or check what wrote it",
                bytes.len()
            ),
        ));
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(strip_bom(text)),
        Err(e) => {
            let at = e.valid_up_to();
            // `valid_up_to` is by definition a char boundary of a valid prefix,
            // so this cannot panic — and the prefix is what tells us where the
            // bad byte is in a human's terms.
            let valid = std::str::from_utf8(&bytes[..at]).unwrap_or("");
            let bad = bytes.get(at).copied().unwrap_or(0);
            Err(refuse(
                place_of(valid),
                format!(
                    "{label} is not valid UTF-8: byte {at} is 0x{bad:02x} — save the \
                     file as UTF-8 (InfiniScript has no other encoding)"
                ),
            ))
        }
    }
}

/// Read a `.infini` file off disk and decode it.
///
/// An unreadable file is a diagnostic, not an `io::Error`: every caller of this
/// door already has a `Vec<Diagnostic>` channel to a human, and a second error
/// type would mean a second rendering of the same failure.
pub fn read(path: &Path) -> Result<String, Diagnostic> {
    let label = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| {
        refuse(
            Span {
                line: 1,
                col: 1,
                len: 0,
            },
            format!("{label} could not be read: {e}"),
        )
    })?;
    decode(&bytes, &label).map(|s| s.to_owned())
}

/// **The door**: bytes on disk → an actor class, or the diagnostics of why not.
///
/// The three SCRIPT1b callers — the editor's hot-reload watcher, the cook and
/// the PIE payload builder — all come through here, so a script that compiles in
/// one of them compiles in all three by construction rather than by three
/// matching implementations.
pub fn compile_path(
    path: &Path,
    id: impl Into<String>,
) -> Result<(BlueprintClass, Vec<Diagnostic>), Vec<Diagnostic>> {
    let text = read(path).map_err(|d| vec![d])?;
    crate::compile(&text, id)
}

/// The door's byte-level half, for a caller that already holds the bytes — the
/// cook reads an asset's payload out of the database and never re-opens the file.
pub fn compile_bytes(
    bytes: &[u8],
    label: &str,
    id: impl Into<String>,
) -> Result<(BlueprintClass, Vec<Diagnostic>), Vec<Diagnostic>> {
    let text = decode(bytes, label).map_err(|d| vec![d])?;
    crate::compile(text, id)
}

/// Is this path a `.infini` script? Case-insensitive, matching
/// `AssetKind::from_extension`.
pub fn is_script_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(SCRIPT_EXT))
}
