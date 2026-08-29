//! **The libm source gate for `inf-script`**, and the structural half of the
//! same law.
//!
//! # The claim
//!
//! *"A script cannot name a transcendental — only a verb."* That sentence is the
//! reason this arc refuses `mlua`: a Luau script can write `math.sin` and reach
//! the platform's libm, and the P14 trig law and its P22 `cbrt` extension say
//! what that costs a `PIE == shipping` gate that compares bytes across three
//! operating systems.
//!
//! The claim has two halves and both are asserted here:
//!
//! 1. **The compiler itself calls no transcendental.** The usual source scan,
//!    over `inf_math::libm_ban::ALL` — the one list, imported rather than
//!    hand-copied (six copies diverged once already, which is why the list moved
//!    into `inf-math`). A `.infini` file's IR is compared byte for byte across
//!    hosts by `tests/determinism.rs`, so a non-portable call *in the parser*
//!    would move that digest on one leg of CI.
//! 2. **The surface has no door to one.** A call must resolve against the node
//!    kit, and the kit's `math.sin`/`math.cos` are
//!    `inf_blueprint::math_builtins`, which route to `inf_math::portable`. So
//!    the guarantee is a property of name resolution rather than of a review —
//!    [`a_script_that_writes_math_sin_gets_the_portable_one`] takes the long way
//!    round to prove it: parse a script, run it, and compare the result's **bits**
//!    against `psin64`.
//!
//! The second arm is the one that would survive somebody deleting the first.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inf_blueprint::interp::{Host, RunError, Value};
use inf_script::{parse_fn, render};

/// Every `.rs` under `src/`, recursively — the directory is walked rather than
/// listed, because a list is a standing invitation to add a module nobody
/// checked (the I2 audit's lesson, and `inf-editor-core`'s pattern).
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
                out.push((name, std::fs::read_to_string(&p).unwrap_or_default()));
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out
}

/// Lines of `src` containing `needle`, ignoring comment lines — the ban is on
/// code, and this crate's module docs necessarily *name* the things they ban.
///
/// CRLF-safe by construction (`str::lines` strips a trailing carriage return).
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

#[test]
fn no_module_in_the_compiler_calls_a_std_transcendental() {
    const GATE: &str = "inf-script/tests/portable_math_law.rs";
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    let files = sources();
    assert!(
        files.len() >= 5,
        "{GATE}: the walk found only {} sources under src/, so it is not \
         reaching the crate — a gate that sweeps nothing passes for ever",
        files.len()
    );
    let mut offences: Vec<String> = Vec::new();
    for (name, src) in &files {
        for needle in &banned {
            for (line_no, line) in code_hits(src, needle) {
                offences.push(format!("src/{name}:{line_no} calls `{needle}`: {line}"));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "{GATE}: {} transcendental call(s) in the compiler that turns a \
         `.infini` file into IR. `tests/determinism.rs` pins that IR's digest \
         and CI runs it on three operating systems, so a call that is not \
         bit-portable reddens one leg with a hash and no line number. There is \
         no exemption list here on purpose: the crate is a lexer, a parser and a \
         printer, and none of the three has any business computing a sine. \
         Offences: {offences:#?}",
        offences.len()
    );
}

/// **The structural half.** A script writes `math.sin(x)`; the value it gets is
/// `inf_math::portable::psin64(x)`, bit for bit — not `f64::sin`.
///
/// The long way round on purpose: parse real text, run the real interpreter over
/// a host that refuses everything (so nothing can be intercepted), and compare
/// bits. If somebody ever routes `math.*` through a `Host` that a project could
/// override, or points the builtin at `std`, this fails.
#[test]
fn a_script_that_writes_math_sin_gets_the_portable_one() {
    struct NoHost;
    impl Host for NoHost {
        fn call(&mut self, path: &[String], _: &[Value]) -> Result<Value, RunError> {
            panic!("a math builtin reached the Host as `{}`", path.join("::"))
        }
    }
    let f =
        parse_fn("function trig(x: float) -> float\n    return math.sin(x) + math.cos(x)\nend\n")
            .unwrap_or_else(|d| panic!("{}", render(&d)));

    for x in [0.0_f64, 1.0, -1.0, 2.0, 6.283185307179586, 1e6] {
        let args: HashMap<String, Value> = [("x".to_string(), Value::Float(x))].into();
        let got = inf_blueprint::eval_fn(&f, &args, &mut NoHost)
            .expect("the math palette is hostless")
            .as_float()
            .unwrap();
        let want = inf_math::portable::psin64(x) + inf_math::portable::pcos64(x);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "math.sin({x}) + math.cos({x}) gave {got:?}, not the portable \
             {want:?} — the script reached a libm"
        );
    }
}

/// …and a script **cannot** write anything else. `std.sin`, `libm.sin` and a
/// bare `sin` are all refusals with a line, because name resolution goes through
/// the node kit and there is no other door.
#[test]
fn there_is_no_spelling_that_reaches_std_or_libm() {
    for src in [
        "on tick(dt)\n    local a = std.sin(dt)\nend\n",
        "on tick(dt)\n    local a = libm.sin(dt)\nend\n",
        "on tick(dt)\n    local a = sin(dt)\nend\n",
        "on tick(dt)\n    local a = f64.sin(dt)\nend\n",
        "on tick(dt)\n    local a = math.tan(dt)\nend\n",
        "on tick(dt)\n    local a = math.exp(dt)\nend\n",
        "on tick(dt)\n    local a = math.ln(dt)\nend\n",
        "on tick(dt)\n    local a = math.cbrt(dt)\nend\n",
        "on tick(dt)\n    local a = math.atan2(dt, dt)\nend\n",
    ] {
        let err = parse_fn(src).expect_err(&format!("`{src}` compiled"));
        assert!(!err.is_empty(), "a refusal with no diagnostic for `{src}`");
    }
    // `math.pow` IS in the kit and IS `f64::powf`, which the node kit's own docs
    // name as not bit-portable. That is a *known* hole in the palette, not a hole
    // this crate opens, and it is carried by name rather than hidden: a script
    // can reach it, exactly as a graph can.
    assert!(parse_fn("on tick(dt)\n    local a = math.pow(dt, 2.0)\nend\n").is_ok());
}
