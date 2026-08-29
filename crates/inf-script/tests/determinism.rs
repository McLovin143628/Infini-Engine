//! **A `.infini` file's IR is a pure function of its bytes.**
//!
//! The determinism law this arc inherits, made a fixture rather than a
//! sentence. Three things are asserted, and the third is the one that has to
//! hold on three operating systems at once:
//!
//! 1. **Purity.** Lowering the same source twice in one process gives the same
//!    IR. Nothing reads a clock, a path, an environment variable or a hash seed.
//! 2. **Line endings do not change a program.** A CRLF checkout and an LF
//!    checkout of the same file lower to the same IR — the lexer normalises
//!    first. This is the `.rs is read by TESTS, so it needs text eol=lf` lesson
//!    met from the other side: rather than depend on a `.gitattributes` entry for
//!    a file *users* author, the reader is made insensitive.
//! 3. **The cross-host pin.** A committed digest of the fixture's lowered IR.
//!    CI runs this on Windows, Linux and macOS; a lowering that depended on the
//!    host would redden exactly one leg, with a line number instead of a
//!    mystery.
//!
//! # Why a hand-rolled FNV rather than a hasher from the standard library
//!
//! `DefaultHasher` is explicitly documented as unspecified and free to change
//! between releases, so a constant pinned against it says nothing about *hosts*
//! — only about toolchains. FNV-1a over the canonical JSON bytes is six lines,
//! has no state beyond two constants, and will give the same answer in ten
//! years. The bytes it eats are `serde_json`'s, under the workspace's
//! `float_roundtrip` pin, which is what keeps a `Lit::Float` from moving a bit on
//! the way through.

use inf_script::{compile, emit_class, parse_fn, render};

/// The determinism fixture. Deliberately dense: every literal kind, a float
/// with a full 17-digit mantissa, both loop forms, a branch and the math
/// builtins (which route to `inf_math::portable`).
///
/// It carries **no `rust` block**, which the first draft of this comment said it
/// did (the audit's correction). That is the right shape rather than an
/// omission: a snippet's contents are preserved byte for byte, so a fixture
/// holding one could not also be used by
/// [`indentation_and_trailing_whitespace_are_not_part_of_the_program`], whose
/// whole point is re-indenting the source. The snippet law has its own arm,
/// [`a_rust_block_survives_verbatim`].
const FIXTURE: &str = r#"actor "Determinism"

var angle: float = -114668.51350953568 exposed
var count: int = -9223372036854775808
var flag: bool = true
var label: string = "a\tb\u{1f600}"

on tick(dt)
    local a = angle + dt * math.sin(dt) - math.cos(dt) / 2.0
    local b = math.clamp(math.lerp(a, 10.0, 0.5), 0.0, 6.0)
    angle = b
    if b > 1.0 and not flag then
        for i = 0, 9 do
            count = count + i
        end
    elseif b < -1.0 then
        while count > 0 do
            count = count - 1
        end
    else
        debug.print(label)
    end
    engine.set_rotation(b)
end
"#;

/// FNV-1a, 64-bit. Two constants and a loop; the same answer on every host and
/// in every future toolchain.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The canonical bytes of a source's IR.
fn ir_bytes(source: &str) -> Vec<u8> {
    let (class, _) = compile(source, "act:determinism")
        .unwrap_or_else(|d| panic!("the fixture did not compile:\n{}", render(&d)));
    serde_json::to_vec(&class).expect("the IR serialises")
}

/// **The pin.** Regenerate only with a stated reason: this number moving means
/// either the fixture changed, the IR's shape changed, or a host disagreed —
/// and the third is the one this file exists to catch.
const FIXTURE_IR_DIGEST: u64 = 0xc47b_dc74_0874_3308;

#[test]
fn the_fixture_lowers_to_the_pinned_digest_on_every_host() {
    let digest = fnv1a(&ir_bytes(FIXTURE));
    assert_eq!(
        digest, FIXTURE_IR_DIGEST,
        "the determinism fixture lowered to {digest:#018x}, not the pinned \
         {FIXTURE_IR_DIGEST:#018x}. If the fixture or the IR changed on purpose, \
         re-pin with the reason in the commit message. If neither did, this host \
         lowers `.infini` differently from the one that pinned it, which is the \
         thing the pin is for"
    );
}

#[test]
fn lowering_is_pure() {
    assert_eq!(ir_bytes(FIXTURE), ir_bytes(FIXTURE));
}

/// **CRLF and LF are the same program.**
#[test]
fn a_windows_checkout_and_a_unix_one_lower_identically() {
    let crlf = FIXTURE.replace('\n', "\r\n");
    assert_ne!(
        crlf.as_bytes(),
        FIXTURE.as_bytes(),
        "the inputs differ in bytes"
    );
    assert_eq!(
        ir_bytes(&crlf),
        ir_bytes(FIXTURE),
        "a carriage return changed the IR"
    );
    // …and a lone `\r`, which an ancient tool can still produce.
    assert_eq!(ir_bytes(&FIXTURE.replace('\n', "\r")), ir_bytes(FIXTURE));
}

/// **A byte-order mark is not a syntax error.**
///
/// The CRLF rule's twin, and the same law: a file authored on Windows and one
/// authored on Unix must lower to the same IR. A BOM is the *other* thing a
/// Windows editor puts in a text file it saves, and before the SCRIPT1a audit it
/// was `unexpected character `\u{feff}`` at 1:1 — a refusal naming a character
/// that is invisible in the message, on a file that looks perfectly fine.
///
/// Only the leading one is stripped: U+FEFF elsewhere is a zero-width no-break
/// space, which inside a string literal is content and outside one is still an
/// unexpected character. Both halves are asserted, because a strip that ate them
/// everywhere would silently change a string.
#[test]
fn a_file_saved_with_a_byte_order_mark_lowers_identically() {
    let with_bom = format!("\u{feff}{FIXTURE}");
    assert_ne!(with_bom.as_bytes(), FIXTURE.as_bytes());
    assert_eq!(
        ir_bytes(&with_bom),
        ir_bytes(FIXTURE),
        "a byte-order mark changed the IR"
    );
    // …and a CRLF file *with* a BOM, which is what Notepad actually writes.
    assert_eq!(
        ir_bytes(&format!("\u{feff}{}", FIXTURE.replace('\n', "\r\n"))),
        ir_bytes(FIXTURE)
    );
    // Inside a string it is content, and survives the round trip as content.
    let f = parse_fn("on begin_play()\n    debug.print(\"a\u{feff}b\")\nend\n").unwrap();
    let text = emit_class(&{
        let (c, _) = compile(
            "on begin_play()\n    debug.print(\"a\u{feff}b\")\nend\n",
            "act:bom",
        )
        .unwrap();
        c
    })
    .unwrap();
    assert!(
        text.contains('\u{feff}'),
        "the mark was eaten from a string"
    );
    assert_eq!(parse_fn(&text).unwrap(), f);
    // Outside one it is still an unexpected character, with a place.
    let d = compile(
        "on begin_play()\n    \u{feff}debug.print(\"x\")\nend\n",
        "act:bom",
    )
    .unwrap_err();
    assert_eq!((d[0].span.line, d[0].span.col), (2, 5), "{}", d[0]);
}

/// **Nor is the layout part of the program.** Tabs for spaces, doubled
/// indentation, and trailing whitespace on every line all lower to the same IR.
///
/// The audit's extension of the line-ending law in the direction it points:
/// "a file's IR is a pure function of its bytes" is trivially true and says
/// nothing a designer cares about. What they care about is that *incidental
/// formatting* — which editor did the indenting, whether it strips trailing
/// space on save — cannot change a program or a committed digest. Every one of
/// these is a byte difference the lexer is required not to see.
///
/// It works on this fixture precisely because the fixture holds **no `rust`
/// block**: a snippet's contents are the one place in the language where bytes
/// rather than tokens survive (`a_rust_block_survives_verbatim`), so
/// re-indenting one legitimately *does* change the program, and a version of
/// this arm run over a fixture containing one would be asserting the opposite
/// law.
#[test]
fn indentation_and_trailing_whitespace_are_not_part_of_the_program() {
    let tabs: String = FIXTURE
        .lines()
        .map(|l| {
            let n = l.len() - l.trim_start().len();
            format!("{}{}\n", "\t".repeat(n), l.trim_start())
        })
        .collect();
    assert_ne!(tabs.as_bytes(), FIXTURE.as_bytes());
    assert_eq!(ir_bytes(&tabs), ir_bytes(FIXTURE), "a tab changed the IR");

    let doubled: String = FIXTURE
        .lines()
        .map(|l| {
            let n = l.len() - l.trim_start().len();
            format!("{}{}\n", " ".repeat(n * 2), l.trim_start())
        })
        .collect();
    assert_eq!(ir_bytes(&doubled), ir_bytes(FIXTURE));

    let trailing: String = FIXTURE.lines().map(|l| format!("{l}   \n")).collect();
    assert_ne!(trailing.as_bytes(), FIXTURE.as_bytes());
    assert_eq!(ir_bytes(&trailing), ir_bytes(FIXTURE));
}

/// A source containing a `rust` block round-trips its opaque contents **byte
/// for byte**, including the trailing newline and the `]]` inside it — the one
/// place in the language where bytes rather than tokens are preserved.
#[test]
fn a_rust_block_survives_verbatim() {
    let code = "    let v = vec![[1u8, 2][0]];\n    let _ = v + 1;\n";
    let src = format!("on begin_play()\n    rust [=[\n{code}]=]\nend\n");
    let f = parse_fn(&src).unwrap_or_else(|d| panic!("{}", render(&d)));
    let inf_blueprint::Stmt::Snippet(got) = &f.body[0] else {
        panic!("not a snippet: {:?}", f.body[0]);
    };
    assert_eq!(got, code, "the snippet's bytes moved");
}

/// **The determinism fixture is not trivial.** A pin over an empty program is a
/// constant nobody can break.
#[test]
fn the_fixture_is_dense_enough_to_be_worth_pinning() {
    let (class, warnings) = compile(FIXTURE, "act:determinism").unwrap();
    assert!(warnings.is_empty(), "{}", render(&warnings));
    assert_eq!(class.variables.len(), 4);
    let body = &class.events[0].body.body;
    // 4 statements at the top level: two lets, the member-variable set, the
    // branch, and the rotation — plus everything nested inside the branch.
    assert!(body.len() >= 5, "{body:#?}");
    let json = String::from_utf8(ir_bytes(FIXTURE)).unwrap();
    assert!(
        json.len() > 2000,
        "the fixture's IR is only {} bytes of JSON",
        json.len()
    );
    // The full-mantissa float really is in there, undisturbed.
    assert!(json.contains("-114668.51350953568"), "the float moved");
    // And so is `i64::MIN`, the literal a naive lexer cannot read.
    assert!(json.contains("-9223372036854775808"), "the int moved");
}

/// The emitted text is a pure function of the IR too — the other half of the
/// same law, and what a cook writing a `.infini` back to disk depends on.
#[test]
fn emission_is_pure() {
    let (class, _) = compile(FIXTURE, "act:determinism").unwrap();
    let a = emit_class(&class).unwrap();
    let b = emit_class(&class).unwrap();
    assert_eq!(a, b);
    // The emitter writes LF, whatever the source used.
    assert!(!a.contains('\r'), "the emitter wrote a carriage return");
}
