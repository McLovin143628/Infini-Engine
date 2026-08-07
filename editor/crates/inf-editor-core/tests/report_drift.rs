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

/// **…and every DTO field reaches the PANEL** (P24.2 re-audit minor 1).
///
/// The kernel→DTO→mirror chain above stops one hop short of the law it is named
/// for: `skin_conflicts` was found because it reached nobody, and `optimized`
/// then reached the DTO and the TypeScript binding and stopped there — still
/// invisible to an author, still "a field that does not exist" by this file's own
/// headline.
///
/// So the last hop is checked too: each DTO field's **camelCase** spelling must
/// appear in the Model Editor panel, which is the only surface that renders these
/// reports. A grep, necessarily — the seam crosses a language boundary and no
/// Rust type can see the other side.
///
/// It is a *presence* check, not a rendering check: nothing here can prove a row
/// is laid out correctly, only that the field is referenced at all. That bound is
/// the honest one, and it is exactly the bound that would have caught all three
/// of this batch's misses.
///
/// # The search reads code, never comments
///
/// The first cut searched the panel's raw text, and the panel — like every file
/// in this repository — explains itself in comments *directly above the row it is
/// about*. The comment above the `meshopt` row contains the word `optimized`
/// three times, so **deleting that row entirely left this gate green**: measured,
/// not reasoned about. A grep across a language boundary that reads comments is
/// the P24.1 F1 finding once more — a check satisfied by the sentence explaining
/// why the check exists.
///
/// **A correction to the round that landed this arm.** Its mutation matrix listed
/// seven severings and reported all seven firing, and each row is true as written
/// — but the severing that matters most to *this* arm was not among them. The one
/// tested was "a **new** DTO field with no panel row", which fires because a new
/// name appears nowhere in the file. Deleting an **existing** row is the severing
/// an author will actually commit, and pre-fix it did not fire: the comment above
/// it kept the name alive. Seven of seven fired; the eighth was not run. That is
/// the difference between a matrix that covers a gate and one that samples it.
const PANEL: &str = include_str!("../../../studio/src/panels/model/ModelEditor.tsx");

/// The panel's **code**, with every comment blanked out.
///
/// One stripper, used by every read of the panel in this file, because a second
/// reader that skipped it would be exactly the hole described above with a
/// different line number.
///
/// * **Block comments** — JSX's `{/* … */}` and TypeScript's `/** … */` are the
///   same lexical form, so a single pass removes both. Newlines inside a comment
///   are kept so the stripped text still has the shape of the original.
/// * **Line comments** — blanked only when the comment is the *whole* line, which
///   is the conservative reading: `//` also occurs mid-line inside string literals
///   (a URL), and blanking from there would delete real code. Measured on this
///   panel: it has no mid-line `//` at all, and every whole-line one is a comment.
///
/// Line endings are normalized first — the P22 CRLF law, met on every gate here
/// that reads a file. Carriage returns are *filtered out* rather than rewritten as
/// a pair: a char filter carries no escape sequence for a scripted edit to mangle.
fn code_only(src: &str) -> String {
    let src: String = src.chars().filter(|c| *c != '\r').collect();
    let mut out = String::with_capacity(src.len());
    let mut rest = src.as_str();
    while let Some(at) = rest.find("/*") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after.find("*/").map_or(after.len(), |e| e + 2);
        out.extend(after[..end].chars().filter(|c| *c == '\n'));
        rest = &after[end..];
    }
    out.push_str(rest);
    out.lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// [`code_only`], with **string literals blanked too** (P24.3 audit F8).
///
/// The comment stripper closed one masking channel and left its twin open: a
/// user-facing string can contain a field's name, so deleting the reader while
/// keeping the tooltip that explains it leaves the gate green. It is the P24.2
/// comment hole relocated into a literal.
///
/// Template-literal **interpolations** (`${...}`) are kept — those are code, and
/// blanking them would delete real readers (`${sel.name}` is a reader). Newlines
/// inside a literal are preserved so line numbers in a failure still line up.
///
/// # What this measured, which is not what was assumed
///
/// Run over the Skeleton Editor when it landed: **zero** of its nineteen DTO
/// names are masked — every one appears in real code as well as, in some cases,
/// a tooltip. So `SKEL_NOT_SHOWN` stays empty and this function changes no
/// verdict today. It is here because the channel is real and the next field may
/// not be so lucky, which is the same reason `code_only` exists.
fn code_and_no_strings(src: &str) -> String {
    let code = code_only(src);
    let b: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == '"' || c == '\'' || c == '`' {
            let quote = c;
            let mut j = i + 1;
            while j < b.len() {
                if b[j] == '\\' {
                    j += 2;
                    continue;
                }
                if b[j] == quote {
                    break;
                }
                // `${ … }` inside a template literal is CODE.
                if quote == '`' && b[j] == '$' && b.get(j + 1) == Some(&'{') {
                    let mut k = j + 2;
                    let mut depth = 1usize;
                    while k < b.len() && depth > 0 {
                        if b[k] == '{' {
                            depth += 1;
                        } else if b[k] == '}' {
                            depth -= 1;
                        }
                        k += 1;
                    }
                    out.extend(&b[j..k.min(b.len())]);
                    j = k;
                    continue;
                }
                if b[j] == '\n' {
                    out.push('\n');
                }
                j += 1;
            }
            i = j + 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn camel(snake: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for c in snake.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// Fields the panel deliberately does **not** show, frozen from a measurement.
///
/// Extending the pin across this seam was cheap and immediately measured **ten
/// pre-existing gaps** — every one of them older than P24.2. They are recorded
/// here rather than either (a) silently permitted, which is the state that let
/// `skin_conflicts` ship invisible, or (b) fixed in a repair round, which would
/// mean designing ten new readout rows against no author's request.
///
/// The value of freezing them is entirely about the *next* field: a new report
/// counter must now either get a panel row or be added to this list by hand,
/// which is the decision the seam was making by default and wrongly. Retiring
/// entries is a UI task, ledgered in ROADMAP section 12's P24 block.
const NOT_SHOWN: [&str; 10] = [
    // Import: totals an author reads off the mesh stats instead.
    "sourceVertices",
    "weldedPositions",
    "sharpEdges",
    // Export: the shape of what was written, already visible as mesh stats, plus
    // three writer advisories that have never had rows.
    "submeshes",
    "fanFallbacks",
    "fallbackTangents",
    "coincidentVertices",
    "reusedDiagonals",
    "nonFiniteWritten",
    "nonUnitNormalsWritten",
];

#[test]
fn every_report_field_reaches_the_panel() {
    let panel = code_only(PANEL);
    for (what, decl) in [
        ("import", "pub struct DccImportDto {"),
        ("export", "pub struct DccExportDto {"),
    ] {
        let fields = fields_of(DTOS, decl);
        assert!(fields.len() >= 7, "{what}: only {} fields", fields.len());
        for f in &fields {
            let js = camel(f);
            if NOT_SHOWN.contains(&js.as_str()) {
                continue;
            }
            assert!(
                panel.contains(&js),
                "{what}: `{f}` (`{js}`) reaches the DTO and the TS binding but is \
                 never referenced by the Model Editor's CODE — by this gate's own \
                 law it is a field that does not exist. Add a row for it. (A \
                 comment naming it does not count, and used to.)"
            );
        }
    }
    // NOT VACUOUS: the panel really is the file this thinks it is, and the
    // camelCase conversion really converts. Read through the same stripper as
    // everything above — a vacuity arm satisfied by a comment certifies nothing.
    assert!(
        panel.contains("Verdict"),
        "the panel has no readout rows at all"
    );
    // …and the stripper is not idle on this file: the panel really does explain
    // itself in comments, which is the whole reason the searches above go through
    // `code_only`.
    let raw: String = PANEL.chars().filter(|c| *c != '\r').collect();
    assert!(
        panel.len() < raw.len(),
        "nothing was stripped from the panel — either it has no comments any more \
         or `code_only` has stopped working; the arm below says which"
    );
    // Every frozen exemption must name a field that still EXISTS — an entry for
    // a field somebody deleted is a hole the next field can fall through.
    let all: Vec<String> = ["pub struct DccImportDto {", "pub struct DccExportDto {"]
        .iter()
        .flat_map(|d| fields_of(DTOS, d))
        .map(|f| camel(&f))
        .collect();
    for x in NOT_SHOWN {
        assert!(
            all.iter().any(|f| f == x),
            "the frozen exemption {x:?} names no DTO field any more; delete it"
        );
    }
    assert_eq!(camel("skin_conflicts"), "skinConflicts");
    assert_eq!(camel("optimize_skipped_skinned"), "optimizeSkippedSkinned");
    assert_eq!(camel("optimized"), "optimized");
}

/// **The stripper removes what the search must not read, and nothing else.**
///
/// The shape under test is the one that was actually measured in the panel: a JSX
/// comment naming a field, sitting directly above the row that renders it. Before
/// `code_only`, deleting the row left the field's name behind in the comment and
/// the gate stayed green; the count below is the difference.
#[test]
fn the_panel_search_cannot_be_satisfied_by_a_comment() {
    let sample = "\
{/* `optimized` reached the DTO and the
    binding and stopped there. */}
// optimized
<Verdict label=\"meshopt\" value={save.export.optimized ? \"ran\" : \"off\"} />
";
    let stripped = code_only(sample);
    assert_eq!(
        sample.matches("optimized").count(),
        3,
        "the fixture no longer poses the problem it was written for"
    );
    assert_eq!(
        stripped.matches("optimized").count(),
        1,
        "a comment naming a field still satisfies the panel search: {stripped:?}"
    );
    // The row is what survives, unmangled.
    assert!(stripped.contains("<Verdict label=\"meshopt\""));
    // …and with the row deleted, the comment alone does NOT satisfy it. This is
    // the exact severing the P24.2 mutation matrix did not perform.
    let row_deleted =
        code_only("{/* `optimized` reached the DTO and the binding and stopped there. */}\n");
    assert!(
        !row_deleted.contains("optimized"),
        "deleting the row while keeping its comment leaves the field 'present'"
    );
    // Line structure survives, so a stripped panel is still readable in a failure.
    assert_eq!(stripped.lines().count(), sample.lines().count());
}

// ── the Skeleton Editor's own root pair (P24.3) ─────────────────────────────
//
// **Its OWN pair, deliberately.** The arms above are hard-wired to the two DCC
// report DTOs and the Model Editor's file, and extending them by adding a third
// tuple would have made one gate answer for two panels: a failure would name a
// file the reader is not editing, and the `NOT_SHOWN` list would become a shared
// bucket where a mesh counter and a rig field exempt each other. So the skeleton
// gets a second reader over the same rules.
//
// **Two files, two claims**, because the DTOs land in two places:
//
//  * `SkelJointDto` / `SkelSocketDto` / `SkelDocDto` describe the *document* and
//    are what the PANEL renders;
//  * `SkelApplyDto` is the *envelope* every mutating call returns — `ok`,
//    `refusal`, `warning`, `doc` — and the panel never sees it, because the STORE
//    unwraps it. Checking it against the panel would fail for a field that is
//    handled correctly; checking it against the store is the honest seam.

const SKEL_PANEL: &str = include_str!("../../../studio/src/panels/skeleton/SkeletonEditor.tsx");
const SKEL_STORE: &str = include_str!("../../../studio/src/stores/skelStore.ts");

/// Skeleton DTO fields the panel deliberately does **not** render.
///
/// Two entries, both structural rather than "we ran out of time":
///
/// * `index` on a **socket** is `joint`, and the panel shows the joint's *name*
///   (`joints[s.joint]?.name`) — a number an author cannot act on is not a row.
///   It is referenced, so it is not here; this list is only the genuine gaps.
/// * `scale` on a **socket**: `place_socket` writes the identity, and there is no
///   control that produces anything else. It is displayed in the socket row's
///   `title` and therefore reaches the code — also not here.
///
/// What IS here is the one field with no reader at all. Recorded rather than
/// silently permitted, which is the state that let `skin_conflicts` ship
/// invisible.
/// **Measured empty, and that is the finding.** Every one of the nineteen
/// skeleton DTO names reaches the panel's code — including under
/// `code_and_no_strings`, so none of them is propped up by a tooltip.
///
/// # The rename weakness, and why it is CLOSED elsewhere
///
/// This gate is a substring search, so renaming `sidedWithoutTwin` to
/// `sidedWithoutTwinXX` in the panel would still satisfy it. That is a real
/// limit and it is **not** load-bearing here: the panel is TypeScript reading a
/// ts-rs-generated interface, so a field renamed on one side and not the other
/// is a `tsc --noEmit` error, and `npm run typecheck` is part of CI. The
/// substring search is the backstop for *deletion*, which the compiler cannot
/// see because dropping a read is always valid TypeScript. Written down rather
/// than assumed, because "the compiler has it" is exactly the kind of claim that
/// stops being true quietly.
const SKEL_NOT_SHOWN: [&str; 0] = [];

/// **Every skeleton DTO field reaches the panel's CODE.**
///
/// Same law, same stripper, same bound as the Model Editor arm: a presence check,
/// not a rendering check, and a comment naming a field does not count.
#[test]
fn every_skeleton_dto_field_reaches_the_panel() {
    // Strings blanked too (F8): a tooltip naming a field must not keep a deleted
    // reader "present". Measured when this landed — zero of the nineteen names
    // are masked, so the list below stays empty and this only guards the future.
    let panel = code_and_no_strings(SKEL_PANEL);
    for decl in [
        "pub struct SkelJointDto {",
        "pub struct SkelSocketDto {",
        "pub struct SkelDocDto {",
    ] {
        let fields = fields_of(DTOS, decl);
        assert!(
            fields.len() >= 5,
            "{decl}: only {} fields parsed — the parser is looking at the wrong \
             thing",
            fields.len()
        );
        for f in &fields {
            let js = camel(f);
            if SKEL_NOT_SHOWN.contains(&js.as_str()) {
                continue;
            }
            assert!(
                panel.contains(&js),
                "`{f}` (`{js}`) reaches the DTO and the TS binding but is never \
                 referenced by the Skeleton Editor's CODE — by this gate's own law \
                 it is a field that does not exist. Give it a row, or add it to \
                 `SKEL_NOT_SHOWN` with a reason. (A comment naming it does not \
                 count.)"
            );
        }
    }
    // NOT VACUOUS: the panel is the file this thinks it is, and it really is
    // stripped — read through the same `code_only` as the assertions above.
    assert!(
        panel.contains("projectRig"),
        "the skeleton panel has no renderer at all"
    );
    let raw: String = SKEL_PANEL.chars().filter(|c| *c != '\r').collect();
    assert!(
        panel.len() < raw.len(),
        "nothing was stripped from the skeleton panel — either it has no comments \
         any more or `code_only` has stopped working"
    );
    // Every frozen exemption must name a field that still EXISTS.
    let all: Vec<String> = [
        "pub struct SkelJointDto {",
        "pub struct SkelSocketDto {",
        "pub struct SkelDocDto {",
    ]
    .iter()
    .flat_map(|d| fields_of(DTOS, d))
    .map(|f| camel(&f))
    .collect();
    for x in SKEL_NOT_SHOWN {
        assert!(
            all.iter().any(|f| f == x),
            "the frozen exemption {x:?} names no skeleton DTO field any more; \
             delete it"
        );
    }
    assert_eq!(camel("sided_without_twin"), "sidedWithoutTwin");
    assert_eq!(camel("limit_min_deg"), "limitMinDeg");
}

/// **…and every `SkelApplyDto` field reaches the STORE**, which is the layer that
/// unwraps the envelope.
///
/// `ok`, `refusal` and `warning` are three different answers and a store that
/// dropped one would silently swallow either the refusal an author needs to read
/// or the cost of a rename they just made.
#[test]
fn every_skeleton_apply_field_reaches_the_store() {
    let store = SKEL_STORE
        .chars()
        .filter(|c| *c != '\r')
        .collect::<String>();
    let fields = fields_of(DTOS, "pub struct SkelApplyDto {");
    assert_eq!(fields.len(), 4, "the envelope changed shape: {fields:?}");
    for f in &fields {
        let js = camel(f);
        assert!(
            store.contains(&format!("res.{js}")),
            "`SkelApplyDto::{f}` is never read out of the reply in `skelStore` — \
             the panel cannot show what the store did not unwrap"
        );
    }
}
