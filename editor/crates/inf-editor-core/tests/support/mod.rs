//! **Shared source-text anchoring for the gates that read this tree as text.**
//!
//! Several gates in this crate assert things a compiler cannot see — that two
//! hand-maintained mirrors have not drifted, that a fallible call flows into the
//! door that reports it. All of them have to find an *item* in a `.rs` file, and
//! all of them get that wrong the same way if each writes its own finder.
//!
//! # Why this module exists (P24.1 re-audit F2)
//!
//! `projector_mirror.rs` learned the hard way that
//! `source.find("fn <name>(")` matches the first **mention**, which for
//! `project_voxel` was a backticked self-reference inside that function's own doc
//! comment — extraction began mid-comment and two blocks drifted to 23 lines
//! against 17 with the gate green (the P21.2-F1 defect). [`item_start`] is the
//! positional fix.
//!
//! It was then re-introduced **in the same commit that fixed it**: the B1 gate
//! lives in a different test binary, could not see the helper, and open-coded
//! `src.find("pub fn resolve_anim_assets<H>(")`. Measured consequence: with the
//! signature mentioned in that function's module docs *and* a real `.ok()`
//! restored in its body, the ban did not fire at all — only an unrelated count
//! assertion did, dumping the wrong span.
//!
//! A helper that has to be written twice will be written twice, and the second
//! copy will be the naive one. So it lives here, and both binaries `mod support;`
//! it. (`tests/support/mod.rs` is not itself a test target — cargo auto-discovers
//! `tests/*.rs`, not a subdirectory's `mod.rs`.)

#![allow(dead_code)] // each test binary uses a different subset

/// Byte offset of the **real item** `fn <name>(` — never a mention of the
/// signature inside a comment, a string or a backticked cross-reference.
///
/// The test is positional rather than lexical: everything between the start of
/// the matched line and the `fn` keyword must be item qualifiers and nothing
/// else. A `///`, a backtick or prose fails it, so a doc block can name the
/// signature as often as it likes without moving the anchor. Indented `impl`
/// methods work too — `    pub fn foo(` leaves `pub`.
pub fn item_start(source: &str, name: &str) -> usize {
    // `fn <name>` followed by `(` **or `<`**: a generic item is still an item, and
    // the B1 reporting door (`fn decode_anim<T: AssetPayload>(…)`) is one. A needle
    // that hard-codes `(` reports "occurs nowhere as an item", which reads as a
    // rename and is how this was first found.
    let needle = format!("fn {name}");
    let mut from = 0usize;
    while let Some(rel) = source[from..].find(&needle) {
        let at = from + rel;
        let after = &source[at + needle.len()..];
        if !(after.starts_with('(') || after.starts_with('<')) {
            from = at + 1;
            continue;
        }
        let line_start = source[..at].rfind('\n').map_or(0, |i| i + 1);
        let is_item = source[line_start..at].split_whitespace().all(|w| {
            matches!(w, "pub" | "async" | "const" | "unsafe" | "extern")
                || w.starts_with("pub(")
                || w.starts_with('"')
        });
        if is_item {
            return at;
        }
        from = at + 1;
    }
    panic!("`fn {name}` occurs nowhere as an item — was it renamed?");
}

/// The **item text** of a free function — the signature line (qualifiers
/// included) through the closing brace at column 0 — with line endings
/// normalized.
pub fn item_text(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = item_start(&source, name);
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let rest = &source[line_start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`fn {name}(` does not terminate at column 0"))
        + 3;
    rest[..end].to_string()
}

/// Read a source file under this crate, line endings normalized.
///
/// Normalization is not cosmetic: `core.autocrlf = true` checks `.rs` out CRLF on
/// Windows, and a gate that searches for a `\n`-delimited substring finds nothing
/// there (the P22 law, met on the trig gate).
pub fn read_crate_source(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// `source` with every comment and string/char literal replaced by spaces, and
/// **nothing else moved** — every byte offset and line number is preserved, so a
/// span reported off the stripped text points at the real one.
///
/// A gate that matches raw source reads *prose as code*: the previous B1 ban
/// would have fired on a doc comment that quoted `.ok()` while describing why
/// `.ok()` is wrong, which is the sentence such a gate is most likely to be
/// written next to.
///
/// Handles `//` and `/* … */` (nested, as Rust does), `"…"`, `'…'`, raw strings
/// `r"…"` / `r#"…"#`, and escapes. It is a lexer, not a parser — it has no
/// opinion about syntax, only about which bytes are *text*.
pub fn strip_comments_and_strings(source: &str) -> String {
    let b = source.as_bytes();
    let mut out: Vec<u8> = source.as_bytes().to_vec();
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for x in out.iter_mut().take(to).skip(from) {
            if *x != b'\n' {
                *x = b' ';
            }
        }
    };
    let mut i = 0usize;
    while i < b.len() {
        // Line comment.
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            let end = source[i..].find('\n').map_or(b.len(), |r| i + r);
            blank(&mut out, i, end);
            i = end;
            continue;
        }
        // Block comment (nested).
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let start = i;
            let mut depth = 1usize;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            blank(&mut out, start, i);
            continue;
        }
        // Raw string: r"…" or r#…"…"#…
        if b[i] == b'r' && i + 1 < b.len() && (b[i + 1] == b'"' || b[i + 1] == b'#') {
            let start = i;
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < b.len() && b[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < b.len() && b[j] == b'"' {
                j += 1;
                let close: Vec<u8> = std::iter::once(b'"')
                    .chain(std::iter::repeat_n(b'#', hashes))
                    .collect();
                while j < b.len() && !b[j..].starts_with(&close) {
                    j += 1;
                }
                j = (j + close.len()).min(b.len());
                blank(&mut out, start, j);
                i = j;
                continue;
            }
        }
        // String / char literal.
        if b[i] == b'"' || b[i] == b'\'' {
            let quote = b[i];
            let start = i;
            let mut j = i + 1;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == quote {
                    j += 1;
                    break;
                }
                // A lifetime (`'a`) is not a literal: bail out rather than eat the
                // rest of the file looking for a closing quote that never comes.
                if quote == b'\'' && b[j] == b'\n' {
                    j = start + 1;
                    break;
                }
                j += 1;
            }
            blank(&mut out, start, j);
            i = j.max(start + 1);
            continue;
        }
        i += 1;
    }
    String::from_utf8(out).expect("blanking preserves UTF-8 boundaries of code bytes")
}

/// Every **call site whose callee path ends in `decode`** in `code`, as
/// `(line number, the callee path as written)`.
///
/// The unit of an allowlist gate: what matters is *which function is being
/// called*, not how its result is then spelled. `.ok()`, `.map_or(None, Some)`,
/// `if let Ok(x) = …`, `let Ok(x) = … else` and a hand-rolled `match` are five
/// spellings of one call, and a ban list catches only the spellings whoever wrote
/// it thought of.
///
/// **Free functions only.** `inf_asset`'s codec is a free function
/// (`inf_asset::decode::<T>(bytes)`), so a *method* call — `a.decode()` — is
/// something else entirely: `AudioAsset::decode` turns a compressed blob into
/// PCM and has nothing to do with the schema ladder. That is a structural
/// distinction (a `.` before the path), not a name on a list, which is what keeps
/// this an allowlist rather than the ban it replaced. A silent `.decode().ok()`
/// on an inherent method is a different failure class and out of this gate's
/// scope; it is not silently *permitted* here, it is not what this gate is about.
///
/// `code` must already be [`strip_comments_and_strings`]ed.
pub fn decode_call_sites(code: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let bytes = code.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut i = 0usize;
    while let Some(rel) = code[i..].find("decode") {
        let at = i + rel;
        i = at + 6;
        // Must be a whole identifier: `decode`, `decode_anim`, … all count, but
        // `decoded` as a *variable* does not — it is only a call if a `(` or a
        // turbofish follows the identifier.
        let mut end = at;
        while end < bytes.len() && is_ident(bytes[end]) {
            end += 1;
        }
        if at > 0 && is_ident(bytes[at - 1]) {
            continue; // mid-identifier (`re_decode`), handled at its own start
        }
        let after = code[end..].trim_start();
        if !(after.starts_with('(') || after.starts_with("::<")) {
            continue;
        }
        // Walk backwards over the fully-qualified path (`inf_asset::decode`).
        let mut start = at;
        loop {
            let head = &code[..start];
            if let Some(stripped) = head.strip_suffix("::") {
                let mut s = stripped.len();
                while s > 0 && is_ident(bytes[s - 1]) {
                    s -= 1;
                }
                if s == stripped.len() {
                    break;
                }
                start = s;
            } else {
                break;
            }
        }
        // A method call (`value.decode()`) is not the asset codec — see the docs.
        if code[..start].trim_end().ends_with('.') {
            continue;
        }
        let line = code[..at].bytes().filter(|c| *c == b'\n').count() + 1;
        out.push((line, code[start..end].to_string()));
    }
    out
}
