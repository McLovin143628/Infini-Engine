//! **A report field that never reaches the author is a field that does not
//! exist** (P24.2 audit M-DTO).
//!
//! `ImportReport::skin_conflicts` shipped in P24.2's first commit, was documented
//! as an advisory the author needs, and reached nobody: the DTO was not updated,
//! the hand-written `import_dto` mirror was not updated, and the panel had no row
//! for it. That is the **GpuLight triplication law** — three hand-maintained
//! copies of one shape — meeting its first real trigger since it was written
//! down.
//!
//! A compiler cannot catch it. `import_dto` builds the DTO field by field, so a
//! new field on the *kernel* struct changes nothing the compiler has to check;
//! only a new field on the *DTO* breaks the build. The forgetting therefore
//! always goes in the same direction, and this gate is that direction.
//!
//! # What is compared, and why by source text
//!
//! The kernel's `ImportReport` / `ExportReport` field lists against the DTOs'.
//! Both are plain `pub name: T,` structs, so the comparison is exact and cheap —
//! and it is a *source* comparison because there is no runtime value that knows
//! a field was omitted.

/// Field names of a `pub struct` in `src`, in declaration order.
///
/// Doc comments and attributes are skipped, and so is anything after the
/// struct's closing brace at its own indentation. Line endings are normalized
/// first — the P22 CRLF law, met on every gate in this repo that reads a `.rs`.
fn fields_of(src: &str, decl: &str) -> Vec<String> {
    let src = src.replace("\r\n", "\n");
    let start = src
        .find(decl)
        .unwrap_or_else(|| panic!("`{decl}` occurs nowhere — was it renamed?"));
    let body = &src[start + decl.len()..];
    let end = body
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{decl}` does not terminate at column 0"));
    body[..end]
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("pub ") && l.ends_with(','))
        .filter_map(|l| l.trim_start_matches("pub ").split(':').next())
        .map(|n| n.trim().to_string())
        .collect()
}

const KERNEL_IMPORT: &str = include_str!("../../../../crates/inf-dcc/src/build.rs");
const KERNEL_EXPORT: &str = include_str!("../../../../crates/inf-dcc/src/export.rs");
const DTOS: &str = include_str!("../src/ipc.rs");
const MIRROR: &str = include_str!("../../../studio/src-tauri/src/commands/dcc.rs");

/// Every counter the kernel reports reaches the DTO, and the hand-written mirror
/// really copies it.
#[test]
fn every_import_and_export_counter_reaches_the_author() {
    for (what, kernel_src, kernel_decl, dto_decl, mirror_fn) in [
        (
            "import",
            KERNEL_IMPORT,
            "pub struct ImportReport {",
            "pub struct DccImportDto {",
            "fn import_dto(",
        ),
        (
            "export",
            KERNEL_EXPORT,
            "pub struct ExportReport {",
            "pub struct DccExportDto {",
            "fn export_dto(",
        ),
    ] {
        let kernel = fields_of(kernel_src, kernel_decl);
        let dto = fields_of(DTOS, dto_decl);
        assert!(
            kernel.len() >= 6,
            "{what}: only {} fields parsed out of the kernel report — the parser \
             is looking at the wrong thing",
            kernel.len()
        );
        assert_eq!(
            kernel, dto,
            "{what}: the kernel's report and its DTO have drifted. A counter the \
             kernel computes and the DTO omits reaches NOBODY — no panel row, no \
             log line, nothing — which is exactly how `skin_conflicts` shipped \
             invisible at P24.2. Add it to `{dto_decl}` and to `{mirror_fn}`."
        );

        // …and the hand-written mirror really assigns each one. `import_dto`
        // builds the DTO field by field, so a field the kernel gained and the
        // mirror ignores would still compile if the DTO had a `Default`.
        let mirror = MIRROR.replace("\r\n", "\n");
        let at = mirror
            .find(mirror_fn)
            .unwrap_or_else(|| panic!("`{mirror_fn}` occurs nowhere"));
        let body = &mirror[at..at + mirror[at..].find("\n}\n").expect("terminates")];
        for f in &kernel {
            assert!(
                body.contains(&format!("{f}:")),
                "{what}: `{mirror_fn}` does not copy `{f}`"
            );
        }
    }
}

/// NOT VACUOUS: the gate can tell a missing field from a present one.
#[test]
fn the_drift_pin_would_notice_a_missing_field() {
    let a = fields_of(KERNEL_IMPORT, "pub struct ImportReport {");
    assert!(
        a.contains(&"skin_conflicts".to_string()),
        "the field this gate was written for is gone: {a:?}"
    );
    // A synthetic DTO one field short must not compare equal.
    let mut b = a.clone();
    b.pop();
    assert_ne!(a, b, "the comparison cannot see a dropped field");
}
