//! **The libm source gate for `inf-material`** (wave TER2a audit).
//!
//! # The claim this file is here to keep honest
//!
//! `inf_material::ground` synthesises **7.39 MB of committed bytes** — the
//! seventeen `.inf_tex` containers under `samples/ground/` that the island's four
//! `TerrainLayer`s bind — and its module header spends four paragraphs on why:
//!
//! > *"a compute bake's output is a fact about the adapter that ran it … **no
//! > transcendental at all** — no `sin`, `cos`, `tan`, `powf`, `exp`, `ln`,
//! > `cbrt`. Ripples are triangle waves, anisotropy is per-axis frequency, and
//! > the one square root is `f64::sqrt` … So two builds on two platforms produce
//! > the same bytes."*
//!
//! That is a load-bearing sentence. It is the whole reason the library is a CPU
//! generator rather than the P7 GPU bake, and `inf_editor_core::samples`'s byte
//! lock compares the committed bytes against a fresh generation on **every CI
//! leg** on the strength of it. Without a gate it is a sentence.
//!
//! # Why it was missing, and why that shape is the dangerous one
//!
//! Every sibling that commits derived bytes carries one — `inf-terrain`,
//! `inf-pcg`, `inf-physics`, `inf-dcc`, `inf-anim`, `inf-island`, `inf-gis`,
//! `inf-render`'s trig law, and `inf-editor-core`'s recursive scan (which is what
//! covers `cover.rs`, the *other* generator wave TER2a landed). `inf-material`
//! did not, so of the three modules TER2a added to the committed-bytes path two
//! were gated and the largest was not.
//!
//! The failure mode is not that somebody writes `.sin()` on purpose. It is that
//! a later wave adds a ripple, a rotation or a curve to a ground set — the most
//! natural edit in the world in a texture synthesiser — and the byte lock then
//! goes red on **one** leg of CI with a diff of seven megabytes of texels and no
//! line number. This gate names the line instead.
//!
//! # There is no module exemption
//!
//! Measured at the time of writing: every `.rs` under `src/` — `bc.rs`,
//! `derive.rs`, `error.rs`, `ground.rs`, `instance.rs`, `lib.rs`, `mapset.rs`,
//! `material.rs`, `texture.rs`, `tiles.rs` and the four under `graph/` — is
//! clean of the whole canonical list, which is a genuinely good state that
//! nothing was protecting. The BC1 compressor, the mip chain and the v2 tiler are
//! integer and `sqrt` arithmetic; the graph half emits **WGSL text** and never
//! evaluates it, so the transcendentals it can write are the driver's problem at
//! run time and not a byte this crate commits.
//!
//! The one exemption is a **literal line inside a test**, enumerated below with
//! its reason — never a vocabulary and never a whole file (the P24.2 `M-F32LAW`
//! law: a token rule such as "a line mentioning `truth` is fine" is a ban list
//! wearing an allowlist's clothes).
//!
//! # The scan reads the directory, not a list
//!
//! `inf-island`'s sibling gate keeps a hand-written `SOURCES` table and pays for
//! it with an `the_source_table_covers_every_module` meta-arm; `inf-editor-core`
//! walks `src/` at run time instead, for the reason the I2 audit found the hard
//! way — a list is a standing invitation to rot, and a module added after it is
//! a module nobody checked. This crate follows the walk.
//!
//! **The worktree constraint, stated rather than discovered**: `CARGO_MANIFEST_DIR`
//! is baked in at *compile* time, so a test binary built in one git worktree and
//! run from another — which a shared `CARGO_TARGET_DIR` makes possible — reads
//! the building worktree's sources. Build and run this crate's tests in the same
//! worktree.

use std::path::{Path, PathBuf};

/// `(file, the exact trimmed line, why it is allowed to call libm)`.
///
/// **Exemptions are FILES and LITERAL LINES — never a vocabulary.**
/// `every_frozen_exemption_still_matches_a_line` fails if any of these stops
/// matching, because an exemption for a line somebody deleted is a hole nobody is
/// using and nobody will notice.
const EXEMPT: [(&str, &str, &str); 1] = [(
    "ground.rs",
    "1.055 * v.powf(1.0 / 2.4) - 0.055",
    // The sRGB encode this crate ships IS the IEC curve rather than an
    // approximation of it: `1/2.4` is exactly `5/12`, so `linear_to_srgb` takes a
    // twelfth root by four exact square roots and three Newton steps. The arm
    // `the_srgb_encode_tracks_the_real_curve` measures it against `powf` over
    // 4 097 points and demands agreement inside a thousandth of a level of 255 —
    // and the first draft of that root was 8.2 levels out, which is the finding
    // the arm exists to have caught. A reference a test compares against is the
    // one legitimate use of a transcendental in this crate, and it reaches no
    // committed byte: it is inside `#[cfg(test)]`, and its value is compared,
    // never written.
    "the reference curve `the_srgb_encode_tracks_the_real_curve` measures the \
     shipped twelfth root against; inside #[cfg(test)], compared and never written",
)];

/// Every `.rs` under `src/`, recursively, as `(relative name, contents)`.
fn sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, root, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let name = p
                    .strip_prefix(root)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .replace('\\', "/");
                let body = std::fs::read_to_string(&p).unwrap_or_default();
                out.push((name, body));
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out
}

/// Lines of `src` containing `needle`, ignoring comment lines — the bans are on
/// code, and the module docs necessarily *name* the things they ban.
///
/// CRLF-safe by construction (`str::lines` strips a trailing carriage return),
/// which matters because `.rs` is `text eol=lf` in `.gitattributes` precisely so
/// a Windows checkout hands a gate the same bytes a Linux one does — the P22
/// lesson, met by a gate that reads the filesystem.
fn code_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let t = line.trim_start();
            !t.starts_with("//") && !t.starts_with('*') && line.contains(needle)
        })
        .map(|(i, line)| (i + 1, line.trim().to_string()))
        .collect()
}

/// **No module in this crate calls a std transcendental**, outside the one
/// enumerated test line.
#[test]
fn no_std_transcendentals_outside_the_named_exemptions() {
    const GATE: &str = "inf-material/tests/portable_math_law.rs";
    // `powi` is not banned and is not on the list: it is repeated multiplication
    // and exact for an integer exponent. `powf` IS on it.
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    let files = sources();
    assert!(
        files.len() >= 10,
        "{GATE}: the walk found only {} sources under src/, so it is not \
         reaching the crate — a gate that sweeps nothing passes for ever",
        files.len()
    );
    let mut offences: Vec<String> = Vec::new();
    for (name, src) in &files {
        for needle in &banned {
            for (line_no, line) in code_hits(src, needle) {
                let stem = name.rsplit('/').next().unwrap_or(name);
                if EXEMPT.iter().any(|(f, l, _)| *f == stem && *l == line) {
                    continue;
                }
                offences.push(format!("src/{name}:{line_no} calls `{needle}`: {line}"));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "{GATE}: {} transcendental call(s) in a crate that synthesises COMMITTED \
         bytes. `inf_material::ground` writes the seventeen `.inf_tex` under \
         `samples/ground/`, and `inf_editor_core::samples`'s lock compares them \
         against a fresh generation on every CI leg — so a call that is not \
         bit-portable across targets turns one leg red with a seven-megabyte \
         diff and no line number. Use an `inf_math::portable` replacement, or a \
         square root, which IEEE-754 requires to be correctly rounded. If the \
         call genuinely reaches no committed byte, add it to `EXEMPT` above as a \
         literal line with its reason. Offences: {offences:#?}",
        offences.len()
    );
}

/// **Every frozen exemption still matches a line.**
///
/// An exemption for a line somebody has since deleted or reflowed is a hole
/// nobody is using and nobody will notice — and the next transcendental written
/// on that shape walks straight through it.
#[test]
fn every_frozen_exemption_still_matches_a_line() {
    let files = sources();
    for (file, line, why) in EXEMPT {
        let src = files
            .iter()
            .find(|(n, _)| n.rsplit('/').next().unwrap_or(n) == file)
            .map(|(_, s)| s.as_str())
            .unwrap_or_else(|| panic!("the exempt file src/{file} is gone"));
        assert!(
            src.lines().any(|l| l.trim() == line),
            "the frozen exemption `{line}` (src/{file} — {why}) no longer \
             matches any line: it was deleted or reflowed, and the hole it \
             leaves is one nothing is watching"
        );
    }
}

/// The one exemption is inside `#[cfg(test)]`, which is what makes it
/// survivable — a reference curve a test compares against reaches no committed
/// byte. Asserted rather than asserted-in-prose: the day somebody moves that
/// arithmetic into the shipped half of the module, this fails.
#[test]
fn the_exempt_line_is_inside_the_test_module() {
    let files = sources();
    for (file, line, _) in EXEMPT {
        let (_, src) = files
            .iter()
            .find(|(n, _)| n.rsplit('/').next().unwrap_or(n) == file)
            .unwrap_or_else(|| panic!("the exempt file src/{file} is gone"));
        let at = src
            .lines()
            .position(|l| l.trim() == line)
            .unwrap_or_else(|| panic!("the exemption no longer matches src/{file}"));
        let gate = src
            .lines()
            .position(|l| l.trim() == "#[cfg(test)]")
            .unwrap_or_else(|| panic!("src/{file} has no `#[cfg(test)]` module"));
        assert!(
            at > gate,
            "the exempt line in src/{file} is at line {} and the `#[cfg(test)]` \
             gate is at {} — the exemption's whole justification is that it \
             reaches no shipped byte, and it now does",
            at + 1,
            gate + 1
        );
    }
}
