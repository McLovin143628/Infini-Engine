//! The structural half of "replaying this journal on any machine produces the
//! same mesh" — pinned by reading the crate's own source.
//!
//! Why a source grep and not a runtime assertion: the claim is about **two
//! machines**, and nothing in a test process compares two machines. The property
//! tests catch order-dependence *within* one run; what makes the cross-target
//! claim true is that the code contains none of the four things known to differ
//! between targets. Those are enforceable by reading, and nothing else here
//! enforces them:
//!
//! 1. **`meshopt`** — the P18 law. Identical input, different bytes on
//!    `x86_64-msvc` versus `aarch64-apple-darwin`. It may appear once, in
//!    `export.rs`, behind the opt-in flag that documents itself as the
//!    non-deterministic door — and nowhere else, and never inside an `Op`.
//! 2. **`std` transcendentals** — the P14 law, widened at P22.4 (`cbrt` routes
//!    through the `libm` crate on wasm32). `sin`/`cos`/`tan`/`cbrt`/`powf`/
//!    `atan2` are banned; the primitive generators call `inf_math::psin64` /
//!    `pcos64`. `sqrt` is deliberately *not* banned: IEEE-754 specifies it
//!    exactly, so it is bit-portable.
//! 3. **Hash containers** — iteration order is not a specified property, and
//!    `RandomState` is seeded per process, so a `HashMap` on a path that feeds a
//!    mesh makes two runs on the *same* machine disagree. `BTreeMap`/`BTreeSet`
//!    everywhere.
//! 4. **`f32` world coordinates** — architecture rule 3. `f32` exists in this
//!    crate at the `MeshAsset` boundary — `build.rs` (reading) and `export.rs`
//!    (writing) — and, since P24.2, in the **skin channel**: `skin.rs` in full,
//!    plus the lines of `topo.rs`/`ops.rs`/`validate.rs` that carry a weight. A
//!    skinning weight is a dimensionless coefficient in `[0, 1]` whose whole life
//!    is spent between an `f32` file and an `f32` vertex buffer; it is not a world
//!    coordinate and rule 3 does not reach it. The ban keeps its teeth by staying
//!    a **line**-level allowlist rather than a file exemption: an `f32` position
//!    or UV in those three files still fails, because the line would not mention
//!    a weight.
//!
//! The P22.4 lesson this gate is built to survive: a ban enumerates what you
//! thought of. So the checks below are on *tokens that compile*, and each one
//! names the file and line it found, rather than reporting a bare `false`.

/// **Every source file in the crate**, and it has to stay that way.
///
/// The P23.3 audit's NB1 law generalizes: a gate that does not cover a file does
/// not cover it, and a new module is exactly where the next `.sin()` or `HashMap`
/// would land. `every_source_file_is_covered` reads the directory and fails if
/// this list has fallen behind, so the coverage cannot rot silently.
const SOURCES: [(&str, &str); 12] = [
    ("lib.rs", include_str!("../src/lib.rs")),
    ("topo.rs", include_str!("../src/topo.rs")),
    ("validate.rs", include_str!("../src/validate.rs")),
    ("build.rs", include_str!("../src/build.rs")),
    ("export.rs", include_str!("../src/export.rs")),
    ("ops.rs", include_str!("../src/ops.rs")),
    ("model.rs", include_str!("../src/model.rs")),
    ("select.rs", include_str!("../src/select.rs")),
    ("sculpt.rs", include_str!("../src/sculpt.rs")),
    ("xform.rs", include_str!("../src/xform.rs")),
    ("uv.rs", include_str!("../src/uv.rs")),
    ("skin.rs", include_str!("../src/skin.rs")),
];

const JOURNAL: &str = include_str!("../src/journal.rs");

/// Lines of `src` that contain `needle`, ignoring `//` comment lines — the bans
/// are on code, and the module docs necessarily *name* the things they ban.
fn code_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let t = line.trim_start();
            !t.starts_with("//") && !t.starts_with("*") && line.contains(needle)
        })
        .map(|(i, line)| (i + 1, line.trim().to_string()))
        .collect()
}

#[test]
fn no_std_transcendentals_anywhere() {
    // `sqrt` is absent from this list on purpose: it is exactly specified by
    // IEEE-754 and therefore bit-portable, which is why the tangent frame may
    // normalize freely while the primitive generators may not call `sin`.
    // The four glam constructors at the end are the P23.5 addition, and they are
    // the reason the ban list needed to grow rather than merely be re-run: a
    // rotation op is committed content, and `DQuat::from_axis_angle` /
    // `DMat3::from_axis_angle` reach `sin_cos` **inside another crate**, where no
    // grep of this one would ever see them. `crate::xform` builds its rotation by
    // hand from `inf_math`'s portable pair for exactly that reason.
    let banned = [
        ".sin()",
        ".cos()",
        ".tan()",
        ".cbrt()",
        ".powf(",
        ".atan2(",
        ".exp()",
        ".ln()",
        "f64::sin",
        "f64::cos",
        "f32::sin",
        "f32::cos",
        ".sin_cos()",
        "from_axis_angle",
        "from_rotation_x",
        "from_rotation_y",
        "from_rotation_z",
    ];
    for (name, src) in SOURCES
        .iter()
        .chain(std::iter::once(&("journal.rs", JOURNAL)))
    {
        for needle in banned {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "{name} calls `{needle}` — use inf_math's bit-portable pair \
                 (P14 law). Found: {hits:?}"
            );
        }
    }
}

#[test]
fn the_primitives_go_through_inf_maths_portable_trig() {
    // The positive half of the ban: the generators that need trigonometry must
    // actually be calling the portable pair, or the ban above is satisfied by a
    // crate that simply has no curves in it.
    let (_, build) = SOURCES[3];
    assert!(
        build.contains("psin64("),
        "build.rs must use inf_math::psin64"
    );
    assert!(
        build.contains("pcos64("),
        "build.rs must use inf_math::pcos64"
    );
    // The other crate with a curve in it (P23.5): a gizmo rotation is committed
    // content, so `xform.rs` must be *calling* the portable pair and not merely
    // failing to call `sin`. Without this half, deleting the rotation entirely
    // would satisfy the ban.
    let (_, xform) = SOURCES[9];
    assert!(
        xform.contains("psin64(") && xform.contains("pcos64("),
        "xform.rs must build its rotation from inf_math's portable trig"
    );
}

#[test]
fn no_hash_containers_on_any_path_that_feeds_a_mesh() {
    for (name, src) in SOURCES
        .iter()
        .chain(std::iter::once(&("journal.rs", JOURNAL)))
    {
        for needle in ["HashMap<", "HashSet<", "HashMap::", "HashSet::"] {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "{name} uses `{needle}` — iteration order is unspecified and \
                 `RandomState` is per-process. Found: {hits:?}"
            );
        }
    }
}

#[test]
fn meshopt_appears_only_at_export_and_never_in_an_op() {
    for (name, src) in SOURCES
        .iter()
        .chain(std::iter::once(&("journal.rs", JOURNAL)))
    {
        assert!(
            code_hits(src, "meshopt::").is_empty(),
            "{name} names `meshopt::` directly; the only sanctioned door is \
             `inf_mesh::optimize` behind `ExportOptions::optimize`"
        );
        if *name != "export.rs" {
            let hits = code_hits(src, "inf_mesh::optimize");
            assert!(
                hits.is_empty(),
                "{name} calls the optimizer. It is NOT cross-platform (P18 law), \
                 so it may run once, on the way out, behind a flag. Found: {hits:?}"
            );
        }
    }
    // And inside export.rs it must be gated, not unconditional.
    let (_, export) = SOURCES[4];
    let hits = code_hits(export, "inf_mesh::optimize");
    assert_eq!(hits.len(), 1, "one call site only: {hits:?}");
    assert!(
        export.contains("if opts.optimize {"),
        "the optimizer must sit behind the opt-in flag"
    );
}

/// **`f32` is a boundary, not a habit** — and P24.2 gave it a second boundary.
///
/// Positions, normals and every intermediate are `f64` (architecture rule 3).
/// `build.rs` reads `f32` out of a `MeshVertex` and `export.rs` writes it back;
/// `skin.rs` *is* the weight representation (a deliberate mirror of
/// `inf_mesh::VertexSkin`, whose weights are `f32` in the file and in the GPU
/// buffer). Everywhere else an `f32` would mean a world coordinate that lost
/// precision inside the kernel.
///
/// The three files the skin channel reaches into keep the ban, at **line**
/// granularity: an `f32` there is legal only on a line that is about a weight.
/// A file-level exemption would have retired the law for `topo.rs` — the file
/// that holds every position in the crate — in exchange for one struct field.
#[test]
fn f32_lives_only_at_the_asset_boundary_and_the_skin_channel() {
    // Tokens that make an `f32` line a *weight* line rather than a coordinate
    // one. Deliberately short: `skin` and `weight` are the channel's whole
    // vocabulary, and `joint` is what a weight is attached to.
    const SKIN_TOKENS: [&str; 5] = ["skin", "Skin", "weight", "Weight", "joint"];

    for (name, src) in [
        SOURCES[1],  // topo.rs
        SOURCES[2],  // validate.rs
        SOURCES[5],  // ops.rs
        SOURCES[6],  // model.rs
        SOURCES[7],  // select.rs
        SOURCES[8],  // sculpt.rs
        SOURCES[9],  // xform.rs
        SOURCES[10], // uv.rs
        ("journal.rs", JOURNAL),
    ] {
        let stray: Vec<(usize, String)> = code_hits(src, "f32")
            .into_iter()
            .filter(|(_, line)| !SKIN_TOKENS.iter().any(|t| line.contains(t)))
            .collect();
        assert!(
            stray.is_empty(),
            "{name} mentions f32 on a line that is not about a skinning weight; \
             the kernel is f64 and the only f32 boundaries are build.rs/export.rs \
             (the asset) and skin.rs (the weight channel). Found: {stray:?}"
        );
    }
    // NOT VACUOUS, twice over. The channel really does put `f32` in those files
    // (so the allowlist is doing work rather than covering an empty set), and
    // `skin.rs` really is the weight representation (so exempting it is a
    // statement about that file and not a blanket).
    let permitted: Vec<(&str, usize, String)> = [SOURCES[2], SOURCES[5]]
        .into_iter()
        .flat_map(|(name, src)| {
            code_hits(src, "f32")
                .into_iter()
                .filter(|(_, line)| SKIN_TOKENS.iter().any(|t| line.contains(t)))
                .map(move |(n, line)| (name, n, line))
        })
        .collect();
    assert!(
        !permitted.is_empty(),
        "no file above carries a permitted f32 line — the allowlist is \
         permitting nothing, and would keep passing if it permitted everything"
    );
    assert!(
        code_hits(SOURCES[11].1, "f32").len() >= 3,
        "skin.rs is exempt because it IS the f32 weight representation; it no \
         longer is"
    );
}

#[test]
fn every_source_file_is_covered_by_the_bans_above() {
    // The gate on the gate. `SOURCES` is a hand-written list of `include_str!`s
    // (it has to be — the macro takes a literal), so the failure mode is a new
    // module that nothing greps. That is the P23.3 NB1 law in its general form:
    // when the code grows, every gate downstream has to be re-proven rather than
    // assumed, and here "re-proven" can be automatic.
    // **Recursive**, because a module in a subdirectory is a module: reading only
    // the top level meant the day `src/uv/` appears (P23.5), every law above
    // silently stops applying to it and this gate keeps saying it does not.
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir)
            .expect("a readable directory")
            .flatten()
        {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .expect("a utf-8 file name")
                        .to_string(),
                );
            }
        }
    }
    // **Worktree constraint, stated rather than discovered.** `CARGO_MANIFEST_DIR`
    // is baked in at COMPILE time, so a test binary built in one git worktree and
    // run from another — which a shared `CARGO_TARGET_DIR` makes possible — reads
    // the *building* worktree's sources. Every law in this file would then be
    // enforced against the wrong tree, silently and greenly. `include_str!` has
    // the same property and no runtime alternative at all, so the honest fix is
    // the constraint: **build and run this crate's tests in the same worktree.**
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut on_disk: Vec<String> = Vec::new();
    walk(&dir, &mut on_disk);
    on_disk.sort();
    let mut covered: Vec<String> = SOURCES
        .iter()
        .map(|(n, _)| (*n).to_string())
        .chain(std::iter::once("journal.rs".to_string()))
        .collect();
    covered.sort();
    assert_eq!(
        on_disk, covered,
        "a source file is not in this test's ban list — add it, or the laws do \
         not apply to it"
    );
}

/// The **forcing function** behind `op_preserves_ids` — the property that makes
/// a new modelling op impossible to add without classifying it.
///
/// Its *answers* are checked by two tests in `select`; what neither of them can
/// see is a `_ => false` wildcard, which produces identical answers today and
/// silently defaults every future op to "restructuring". That is behaviourally
/// safe and structurally fatal: the whole point of the function is that it does
/// not compile until somebody decides, and a wildcard removes exactly that.
///
/// Scanned **line by line** rather than by searching for a brace-newline
/// pattern. `include_str!` hands back the file's own bytes, and on a Windows
/// checkout under `core.autocrlf = true` those are CRLF — the P22 law, met
/// again: a needle containing a newline simply is not there, and the gate
/// aborts with a message about braces instead of failing.
#[test]
fn the_id_preservation_classifier_has_no_wildcard_and_names_every_op() {
    let body = block_after(SOURCES[7].1, "pub fn op_preserves_ids");
    assert!(
        !body.iter().any(|l| l.contains("_ =>")),
        "op_preserves_ids has a wildcard arm; a new op would default silently \
         instead of stopping the build"
    );

    // Every variant of `Op`, read out of the enum itself rather than re-listed
    // here, must be named by the classifier.
    let decl = block_after(SOURCES[5].1, "pub enum Op {");
    let variants: Vec<String> = decl
        .iter()
        .map(|l| l.trim())
        .filter(|l| {
            !l.starts_with("//")
                && !l.starts_with('#')
                && l.chars().next().is_some_and(char::is_uppercase)
        })
        .filter_map(|l| l.split([' ', '{', '(', ',']).next())
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .collect();
    assert_eq!(variants.len(), 30, "found {variants:?}");
    let joined = body.join(" ");
    for v in &variants {
        assert!(
            joined.contains(&format!("Op::{v} ")),
            "op_preserves_ids does not name `Op::{v}`"
        );
    }
}

/// The lines of the block that opens on the first line containing `needle`, up
/// to (not including) the next line that is a bare `}` at column zero.
fn block_after(source: &str, needle: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line contains `{needle}`"));
    lines[start + 1..]
        .iter()
        .take_while(|l| l.trim_end() != "}")
        .map(|l| l.to_string())
        .collect()
}
#[test]
fn no_skip_serializing_if_on_a_bincode_struct() {
    // The P10 law, caught three times: bincode is positional, so a conditionally
    // omitted field desyncs the stream for every field after it. A session save
    // is bincode.
    for (name, src) in SOURCES
        .iter()
        .chain(std::iter::once(&("journal.rs", JOURNAL)))
    {
        assert!(
            code_hits(src, "skip_serializing_if").is_empty(),
            "{name} uses skip_serializing_if on a type that is encoded with bincode"
        );
    }
}

/// **`Op::Unwrap` must replay its VALUES, never re-run the solver.**
///
/// This one is a source grep because it *cannot* be a runtime assertion: the
/// solver is deterministic today, so an `apply` that re-solved would produce the
/// same UVs and every property test in the crate would stay green. The claim is
/// about a **future build** — a journal is replayed by a different build than
/// wrote it, and an op defined by a computation is an op whose recorded effect
/// changes when the computation improves. That is the meshopt law (P18) one level
/// up, and the only thing that can enforce it is reading the code.
///
/// # The first version of this gate was defeatable, and was defeated
///
/// It banned **one spelling** (`uv::unwrap`) across the whole file and checked
/// its positive halves with `.contains()` on the whole file too. The P23.5 audit
/// walked straight through it: a `pub fn recompute` wrapper re-solved on replay
/// with 208 of 208 tests green, because the ban never saw the new spelling and
/// the positive halves were satisfied by the *unchanged* lines elsewhere — a
/// comment would have satisfied them.
///
/// So: the region is the **arm**, found by balancing braces from its own `match`
/// pattern, and the ban is on the **module** (`uv::`) rather than on a function
/// name. A wrapper, a rename and a re-export are all `uv::` something, and none
/// of them can hide in a file this no longer reads whole.
#[test]
fn the_unwrap_op_replays_its_values_and_never_calls_the_solver() {
    let (_, ops) = SOURCES[5];
    let arm = braced_block(ops, "Op::Unwrap { corners } => {");
    let body = arm.join(
        "
",
    );

    // Negative: the whole MODULE, not one function in it.
    let hits: Vec<&str> = arm
        .iter()
        .map(String::as_str)
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//") && l.contains("uv::")
        })
        .collect();
    assert!(
        hits.is_empty(),
        "the `Op::Unwrap` arm names `uv::`. It must write the corners it CARRIES          — see `crate::uv`. Found: {hits:?}"
    );

    // Positive, scoped to the arm: it has to actually write them, or the ban
    // above is satisfied by an op that does nothing.
    assert!(
        body.contains("mesh.half_mut(*h).uv = *uv;"),
        "the `Op::Unwrap` arm no longer writes the UVs it carries:
{body}"
    );
    // And the scoping itself is checked, because a `braced_block` that returned
    // the whole file would make both assertions above meaningless.
    assert!(
        arm.len() < 40,
        "the arm scope has slipped and now covers {} lines; the ban is only          meaningful while it is the ARM",
        arm.len()
    );
    assert!(
        !body.contains("Op::Sculpt"),
        "the arm scope leaked into a neighbouring arm"
    );
}

/// The lines of the block that opens on the first line containing `needle`, up to
/// and including the line where its braces balance again.
///
/// Brace-counting rather than an indentation or bare-`}` rule: a match arm closes
/// at its own indentation, which `block_after`'s column-zero rule cannot see, and
/// that is precisely how the first version of the gate above ended up reading the
/// whole of `apply`.
fn braced_block(source: &str, needle: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line contains `{needle}`"));
    let mut depth: i32 = 0;
    // See the note on `body_of` in `commands/dcc.rs`: a "stop at depth zero" rule
    // that has not yet SEEN a brace stops on the first line it is handed.
    let mut opened = false;
    let mut out = Vec::new();
    for line in &lines[start..] {
        out.push((*line).to_string());
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if opened && depth <= 0 {
            break;
        }
    }
    out
}
