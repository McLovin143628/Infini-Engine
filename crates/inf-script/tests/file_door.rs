//! **The file door** (SCRIPT1b clause 1) — every way bytes on disk can fail to
//! be a script, as a *value*.
//!
//! The SCRIPT1a audit's routing item, verbatim: *"There is no file door yet, so
//! invalid UTF-8 has no refusal yet… a `.infini` holding invalid bytes must
//! decode into a diagnostic with a line, not an `unwrap`."* Every arm here is
//! that sentence for one class of bad file.
//!
//! The anti-vacuity rule this file is held to: a door that refused *everything*
//! would pass every refusal arm, so the good-file arms sit beside them and the
//! bytes they read are the same bytes the bad arms mutate.

use std::path::Path;

use inf_script::source::{self, MAX_SOURCE_BYTES, SCRIPT_EXT};

/// A real script, small and complete: a member variable, a handler, arithmetic,
/// a comparison and a host call. Mutated below rather than replaced, so a
/// refusal arm cannot pass by refusing something that was never a script.
const GOOD: &str = "\
actor \"Gate\"

var open_frac: float = 0.0

on tick(dt)
  local step = dt * 0.5
  open_frac = open_frac + step
  if open_frac > 1.0 then
    open_frac = 1.0
  end
  engine.set_rotation(open_frac * 90.0)
end
";

fn write(dir: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).expect("write the fixture");
    p
}

#[test]
fn a_real_script_goes_through_the_door() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(tmp.path(), "Gate.infini", GOOD.as_bytes());
    let (class, warnings) = source::compile_path(&p, "script:gate").expect("the door opens");
    assert_eq!(class.id, "script:gate");
    assert_eq!(class.name, "Gate");
    assert_eq!(class.variables.len(), 1);
    assert_eq!(class.events.len(), 1, "the tick handler survived the door");
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// **The routed item.** Invalid UTF-8 names the byte offset — which is what a
/// hex editor, `iconv` and `git` all speak — at the line and column that offset
/// lands on.
#[test]
fn invalid_utf8_is_a_diagnostic_naming_the_byte_offset() {
    let tmp = tempfile::tempdir().unwrap();
    // A real script with one byte of Latin-1 in it: `0xE9` where `é` should be.
    // This is exactly what an editor that saved as cp1252 produces.
    let mut bytes = b"actor \"Caf\xe9\"\n\non tick(dt)\n  engine.set_rotation(dt)\nend\n".to_vec();
    let bad_at = bytes.iter().position(|b| *b == 0xe9).unwrap();
    let p = write(tmp.path(), "Cafe.infini", &bytes);

    let diags = source::compile_path(&p, "script:cafe").expect_err("not UTF-8");
    assert_eq!(diags.len(), 1, "{diags:?}");
    let d = &diags[0];
    println!("the door said: {d}");
    assert!(
        d.message.contains(&format!("byte {bad_at}")),
        "the refusal must name the byte offset: {}",
        d.message
    );
    assert!(d.message.contains("0xe9"), "{}", d.message);
    assert!(d.message.contains("UTF-8"), "{}", d.message);
    // …and the place is the line and column that byte sits on, not 1:1.
    assert_eq!((d.span.line, d.span.col), (1, 11), "{:?}", d.span);

    // The same bytes as valid UTF-8 compile, so the arm is about the ENCODING
    // and not about the script.
    bytes.splice(bad_at..bad_at + 1, "é".bytes());
    let ok = write(tmp.path(), "Cafe2.infini", &bytes);
    let (class, _) = source::compile_path(&ok, "script:cafe").expect("utf-8 opens");
    assert_eq!(class.name, "Café");
}

/// A bad byte on a later line is placed on that line, so the offset and the
/// span are two views of one position rather than one of them being a constant.
#[test]
fn the_place_of_a_bad_byte_follows_the_lines_before_it() {
    let tmp = tempfile::tempdir().unwrap();
    let mut bytes = GOOD.as_bytes().to_vec();
    // Overwrite a byte inside the string-free body, on line 6.
    let line6 = GOOD
        .split_inclusive('\n')
        .take(5)
        .map(|l| l.len())
        .sum::<usize>();
    bytes[line6 + 2] = 0xff;
    let p = write(tmp.path(), "Gate.infini", &bytes);
    let d = source::compile_path(&p, "x")
        .expect_err("not UTF-8")
        .remove(0);
    println!("the door said: {d}");
    assert_eq!(d.span.line, 6, "{:?}", d.span);
    assert_eq!(d.span.col, 3, "{:?}", d.span);
    assert!(d.message.contains(&format!("byte {}", line6 + 2)));
}

/// The size bound is **LOUD**: both numbers, in bytes, and what to do.
#[test]
fn an_over_long_file_is_refused_by_both_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    // One byte over, so the arm proves the bound rather than proving that a
    // hundred megabytes is a lot.
    let big = vec![b'-'; MAX_SOURCE_BYTES + 1];
    let p = write(tmp.path(), "Huge.infini", &big);
    let d = source::compile_path(&p, "x")
        .expect_err("too big")
        .remove(0);
    println!("the door said: {d}");
    assert!(d.message.contains(&format!("{}", MAX_SOURCE_BYTES + 1)));
    assert!(d.message.contains(&format!("{MAX_SOURCE_BYTES}")));

    // Exactly at the bound is accepted — the refusal is `>`, not `>=`, and a
    // gate that only tested "much too big" could not tell the two apart. The
    // content is comment lines, so it is also a real (empty) script.
    let at = vec![b'-'; MAX_SOURCE_BYTES];
    let p = write(tmp.path(), "AtBound.infini", &at);
    let (class, _) = source::compile_path(&p, "x").expect("exactly at the bound opens");
    assert!(class.events.is_empty());
}

/// A byte-order mark is **repaired**, not refused — the audit's finding, met at
/// the door as well as in the lexer, because the door's output is what a cook
/// hashes.
#[test]
fn a_byte_order_mark_is_dropped_at_the_door() {
    let tmp = tempfile::tempdir().unwrap();
    let mut bom = "\u{feff}".as_bytes().to_vec();
    bom.extend_from_slice(GOOD.as_bytes());
    let p = write(tmp.path(), "Bom.infini", &bom);
    let text = source::read(&p).expect("a BOM is not an error");
    assert_eq!(text, GOOD, "the door hands on the text without the mark");

    let (with_bom, _) = source::compile_path(&p, "x").expect("compiles");
    let plain = write(tmp.path(), "Plain.infini", GOOD.as_bytes());
    let (without, _) = source::compile_path(&plain, "x").expect("compiles");
    assert_eq!(
        with_bom, without,
        "a Windows editor's BOM must not change the program"
    );
}

/// A missing file is a diagnostic naming the path, not an `io::Error` a caller
/// has to render a second way.
#[test]
fn a_file_that_will_not_open_is_a_value() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("NotThere.infini");
    let d = source::compile_path(&p, "x")
        .expect_err("missing")
        .remove(0);
    println!("the door said: {d}");
    assert!(d.message.contains("NotThere.infini"), "{}", d.message);
    assert!(d.message.contains("could not be read"), "{}", d.message);
}

/// A file that decodes and does not parse still comes back as diagnostics with
/// places — the door adds a layer and does not swallow the one under it.
#[test]
fn a_parse_error_still_reaches_the_caller_with_its_place() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(
        tmp.path(),
        "Broken.infini",
        b"on tick(dt)\n  engine.set_rotation(\nend\n",
    );
    let diags = source::compile_path(&p, "x").expect_err("does not parse");
    assert!(!diags.is_empty());
    for d in &diags {
        println!("the door said: {d}");
        assert!(d.span.line >= 1);
    }
}

/// The extension test is case-insensitive and spelled once.
#[test]
fn the_extension_is_recognised_the_way_a_filesystem_spells_it() {
    assert_eq!(SCRIPT_EXT, "infini");
    for name in ["a.infini", "a.INFINI", "a.Infini", "deep/dir/a.infini"] {
        assert!(source::is_script_path(Path::new(name)), "{name}");
    }
    for name in ["a.inf_act", "a.infini.toml", "a", "infini"] {
        assert!(!source::is_script_path(Path::new(name)), "{name}");
    }
}

/// **Nothing the door reads can panic.** The hostile corpus SCRIPT1a built for
/// the *parser* is fed through the *reader* here, as raw bytes, plus the byte
/// sequences a text file cannot hold at all.
#[test]
fn no_byte_sequence_makes_the_door_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let mut refused = 0usize;
    let mut opened = 0usize;
    // The comments sit ABOVE their bytes rather than trailing them: a run of
    // spaces aligning a trailing comment after a string literal is
    // indistinguishable from an eaten `\`-continuation, and `inf_packager`'s
    // workspace-wide sweep says so. Rewritten rather than exempted — the
    // SCRIPT1a precedent, where its raise-coverage fixtures collided with the
    // same sweep.
    let cases: Vec<Vec<u8>> = vec![
        // empty, and one NUL
        vec![],
        vec![0x00],
        // the byte-order marks of the encodings InfiniScript is not
        vec![0xff, 0xfe, 0x00, 0x00],
        vec![0xff, 0xfe],
        vec![0xfe, 0xff],
        // a UTF-8 BOM alone: a valid, empty script
        b"\xef\xbb\xbf".to_vec(),
        // a stray continuation byte inside a real script
        b"on tick(dt)\x80\nend\n".to_vec(),
        // a truncated two-byte sequence
        b"\xc3".to_vec(),
        // a lone surrogate (CESU-8)
        b"\xed\xa0\x80".to_vec(),
        // past U+10FFFF
        b"\xf4\x90\x80\x80".to_vec(),
        // CRLF throughout
        b"actor \"x\"\r\non tick(dt)\r\nend\r\n".to_vec(),
        b"\x00\x00\x00\x00".repeat(64),
        GOOD.as_bytes().to_vec(),
    ];
    for (i, bytes) in cases.iter().enumerate() {
        let p = write(tmp.path(), &format!("case{i}.infini"), bytes);
        match source::compile_path(&p, "x") {
            Ok(_) => opened += 1,
            Err(d) => {
                assert!(!d.is_empty(), "case {i} refused with no diagnostic");
                refused += 1;
            }
        }
    }
    println!(
        "the door: {opened} opened, {refused} refused, over {} cases",
        cases.len()
    );
    // Anti-vacuity in both directions: a door that opened everything would prove
    // nothing about the encoding, and one that refused everything would prove
    // nothing about the parser behind it.
    assert!(refused >= 6, "only {refused} refused");
    assert!(opened >= 3, "only {opened} opened");
}
