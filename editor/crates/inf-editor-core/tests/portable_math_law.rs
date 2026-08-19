//! **The P14 libm law, over the editor's Ring-1 crate** — the gate the Wave-D
//! audit re-carried by name.
//!
//! Its words were:
//!
//! > A `libm`/trig source gate does not cover `editor/.../dcc.rs`. Its three
//! > `.sin()`/`.cos()` sites are display-only today and its two `pcos64` sites
//! > (`shade_edges`, `auto_seam_edges`) correctly feed journalled *results*; the
//! > file now contains both kinds and no gate distinguishes them.
//!
//! Two corrections the measurement made to that sentence, both recorded because
//! a re-carry that undercounts is how the next one gets missed:
//!
//! 1. **`dcc.rs` has five banned sites, not three.** The two the re-carry did not
//!    name are `glam::Quat::from_axis_angle` and `from_rotation_arc` in
//!    `gizmo_orientation`, which reach `sin_cos` *inside glam* where no grep of
//!    this crate would see them. That is the exact half
//!    [`inf_math::libm_ban::GLAM`] exists for, and a hand-written gate scoped to
//!    "the three `.sin()` sites" would have shipped with both of them uncovered.
//! 2. **`dcc.rs` was not the only file.** A crate-wide scan found seven more in
//!    `thumbnail/scene_render.rs` and — the ones that mattered — **three on paths
//!    that write committed content**, in `scene/doc.rs` and `samples.rs`. Those
//!    three are fixed rather than exempted; see the positive half below.
//!
//! # The distinction this gate draws
//!
//! Not "does the file call `sin`" but **"can the value it computes reach two
//! machines that are claimed to agree"**. Three answers, and the frozen list
//! below records which one each site got:
//!
//! * **Committed content** — a value serialized into an `.inf_lvl`, a
//!   `.inf_terrain`, a cooked pack or a journalled op. Must use
//!   [`inf_math`]'s portable pair. No exemptions; the asserted half is that
//!   these sites really do call it.
//! * **Display geometry** — screen-space overlay drawing, a preview camera, a
//!   projection matrix, a preview sphere. Redrawn every frame from live state
//!   and never written anywhere two machines compare. Exempt, by line.
//! * **The interaction frame** — the gizmo's basis quaternion. It is `f32`, it
//!   is consumed by hit-testing and by `GizmoDrag::begin`, and the drag's
//!   *result* is journalled as explicit numbers rather than as the basis that
//!   produced it. Exempt, by line, and this is the class worth naming out loud
//!   because it is the one that looks like committed content and is not.
//!
//! # Why the coverage is a WALK and not a list
//!
//! `inf-dcc`'s own `determinism_law` keeps a hand-written `SOURCES` list because
//! `include_str!` takes a literal, and pays for it with an
//! `every_source_file_is_covered` meta-arm. This crate is far larger and the same
//! list would be a standing invitation to rot. So the scan reads the directory at
//! run time: **every `.rs` under `src/`, recursively**, with no list to fall
//! behind. A new module is covered the moment it exists.
//!
//! **The worktree constraint, stated rather than discovered** (the same one
//! `determinism_law` records): `CARGO_MANIFEST_DIR` is baked in at *compile*
//! time, so a test binary built in one git worktree and run from another — which
//! a shared `CARGO_TARGET_DIR` makes possible — reads the building worktree's
//! sources. Build and run this crate's tests in the same worktree.

use std::path::{Path, PathBuf};

/// `(file, the exact trimmed line, why it is allowed to call libm)`.
///
/// **Exemptions are FILES and LITERAL LINES — never a vocabulary** (the P24.2
/// audit's `M-F32LAW` law). A token rule such as "a line mentioning `preview` is
/// fine" is a ban list wearing an allowlist's clothes: the day a committed-content
/// site happens to mention the token, it walks straight through. Every line here
/// was read and classified by hand, and a new one has to be added the same way.
///
/// `every_frozen_exemption_still_matches_a_line` fails if any of these stops
/// matching, because an exemption for a line somebody deleted is a hole nobody is
/// using and nobody will notice.
const EXEMPT: [(&str, &str, &str); 12] = [
    // ── dcc.rs: display geometry ────────────────────────────────────────────
    (
        "dcc.rs",
        "view.distance = radius / (view.fov_deg * 0.5).to_radians().sin() * 1.15;",
        "the Model Editor's frame-selection camera distance. A view field, \
         recomputed on every F press, serialized nowhere.",
    ),
    (
        "dcc.rs",
        "let p = centre + (u * a.cos() + v * a.sin()) * radius;",
        "the sculpt hover brush ring — an overlay polyline rebuilt each frame \
         from the cursor's current hit. Nothing downstream of it is stored.",
    ),
    (
        "dcc.rs",
        "let p = proj.point(origin + (u * t.cos() + v * t.sin()) * g);",
        "the rotate gizmo's ring, same class as the brush ring: screen-space \
         geometry drawn for the author and thrown away.",
    ),
    // ── dcc.rs: the interaction frame ───────────────────────────────────────
    (
        "dcc.rs",
        "glam::Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)",
        "gizmo_orientation's 180 degree case. The BASIS a drag is expressed in, \
         not the drag: `Op::Transform` records the resulting numbers, so a \
         replay never re-derives this quaternion. f32 throughout.",
    ),
    (
        "dcc.rs",
        "Some(n) => glam::Quat::from_rotation_arc(Vec3::Z, n),",
        "gizmo_orientation's normal-aligned basis; same class as the line above, \
         and the reason the GLAM half of the ban list exists at all.",
    ),
    // ── thumbnail/scene_render.rs: display geometry, all of it ──────────────
    (
        "thumbnail/scene_render.rs",
        "yaw_deg: dir.x.atan2(dir.z).to_degrees(),",
        "PreviewView::default's fixed framing, expressed as angles.",
    ),
    (
        "thumbnail/scene_render.rs",
        "pitch_deg: dir.y.asin().to_degrees(),",
        "the pitch half of the same default framing.",
    ),
    (
        "thumbnail/scene_render.rs",
        "let (sy, cy) = yaw.sin_cos();",
        "PreviewView::eye — where the preview camera stands.",
    ),
    (
        "thumbnail/scene_render.rs",
        "let (sp, cp) = pitch.sin_cos();",
        "the pitch half of PreviewView::eye.",
    ),
    (
        "thumbnail/scene_render.rs",
        "let g = 1.0 / (fovy * 0.5).tan();",
        "the preview's projection matrix. A matrix, not a mesh.",
    ),
    (
        "thumbnail/scene_render.rs",
        "let (sp, cp) = phi.sin_cos();",
        "unit_sphere's stack angle. The material-preview ball is generated for \
         one offscreen PNG and is never an asset: `bake_texture` writes its \
         .inf_tex from a compute pass over the material graph, not from this.",
    ),
    (
        "thumbnail/scene_render.rs",
        "let (st, ct) = theta.sin_cos();",
        "unit_sphere's sector angle, same ball.",
    ),
];

/// **The committed-content sites, asserted POSITIVELY.**
///
/// `(file, the exact trimmed line that must be present, what it produces)`.
///
/// The ban above is only half a law: a crate with no curves in it satisfies it
/// perfectly. These are the lines that were calling `std`'s libm on a path whose
/// output two machines are claimed to agree about, and the claim is only true
/// while they keep calling the portable pair. Stated as whole lines because
/// "the file mentions `psin64` somewhere" is satisfied by a comment.
const MUST_BE_PORTABLE: [(&str, &str, &str); 5] = [
    (
        "scene/doc.rs",
        "2.0 * inf_math::psin64(x * 0.1) * inf_math::pcos64(z * 0.1)",
        "SpawnKind::Terrain's starter hill. Written into the author's `Terrain` \
         component, serialized into their .inf_lvl and cooked into a pack.",
    ),
    (
        "samples.rs",
        "6.0 * inf_math::psin64(x * 0.08) * inf_math::pcos64(z * 0.08)",
        "terrain_demo_height. `committed_sample_matches_generators` re-runs it \
         and asserts BYTE equality against the committed TerrainDemo.inf_lvl.",
    ),
    (
        "samples.rs",
        "3.0 * inf_math::psin64(x * 0.08) * inf_math::pcos64(z * 0.08)",
        "character_demo_height, byte-locked the same way, and additionally the \
         ground the character-demo PIE-vs-shipping trace is compared on.",
    ),
    (
        "dcc.rs",
        "let threshold = inf_math::pcos64(deg.clamp(0.0, 180.0) * std::f64::consts::PI / 180.0);",
        "shade_edges' sharp-edge threshold — the sharp flags are journalled.",
    ),
    (
        "dcc.rs",
        "let threshold = inf_math::pcos64(angle_deg.clamp(0.0, 180.0) * std::f64::consts::PI / 180.0);",
        "auto_seam_edges' seam threshold — the seams become an Op::SetEdgesSeam.",
    ),
];

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` under `src/`, recursively, as `(relative slash-path, contents)`.
///
/// Sorted, so a failure list is stable between runs on a filesystem whose
/// `read_dir` order is not (the P26 "NTFS hides sorts" law).
fn sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, String)>) {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("a readable src directory")
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, base, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let rel = p
                    .strip_prefix(base)
                    .expect("under base")
                    .to_string_lossy()
                    .replace('\\', "/");
                let text = std::fs::read_to_string(&p).expect("utf-8 source");
                out.push((rel, text));
            }
        }
    }
    let base = src_root();
    let mut out = Vec::new();
    walk(&base, &base, &mut out);
    out
}

/// A file's **production** half: everything before a column-zero `#[cfg(test)]`,
/// with line numbers preserved so a reported hit points at the real line.
///
/// A test fixture that builds a sine wave to check a refusal is not committed
/// content — it is the thing that checks the code that must not have one. Cutting
/// the tail keeps the ban aimed at what it names, which is the same cut
/// `inf-dcc`'s `determinism_law` and `inf-anim`'s `portable_pose` both make.
///
/// Carriage returns are **filtered out** rather than rewritten as a pair: the P22
/// law needs them gone before a line-wise search, and a char filter carries no
/// escape sequence for a scripted edit to mangle.
fn production_only(src: &str) -> Vec<(usize, String)> {
    let src: String = src.chars().filter(|c| *c != '\r').collect();
    let lines: Vec<&str> = src.lines().collect();
    let cut = lines
        .iter()
        .position(|l| *l == "#[cfg(test)]")
        .unwrap_or(lines.len());
    lines[..cut]
        .iter()
        .enumerate()
        .map(|(i, l)| (i + 1, (*l).to_string()))
        .collect()
}

/// Whether a line is code rather than a comment — the bans are on code, and the
/// module docs necessarily *name* the things they ban.
fn is_code(line: &str) -> bool {
    let t = line.trim_start();
    !t.starts_with("//") && !t.starts_with('*')
}

#[test]
fn no_std_transcendental_reaches_committed_content_from_this_crate() {
    const GATE: &str = "inf-editor-core/tests/portable_math_law.rs";
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    // The list is the canonical one, and it is complete in both spellings plus
    // the glam constructors. Without this the gate would be enforcing whatever
    // somebody happened to think of — which is the failure mode `libm_ban` was
    // hoisted to end.
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    let mut stray: Vec<String> = Vec::new();
    for (name, src) in sources() {
        for (n, line) in production_only(&src) {
            if !is_code(&line) {
                continue;
            }
            let trimmed = line.trim();
            if EXEMPT.iter().any(|(f, l, _)| *f == name && *l == trimmed) {
                continue;
            }
            for needle in &banned {
                if line.contains(needle) {
                    stray.push(format!("{name}:{n}  `{needle}`  {trimmed}"));
                    break;
                }
            }
        }
    }
    assert!(
        stray.is_empty(),
        "a std transcendental is reachable from Ring-1 code that this gate does \
         not classify. Every call has to be one of three things: COMMITTED \
         CONTENT (use inf_math's portable pair — psin64/pcos64/pacos64/pcbrt), \
         DISPLAY geometry, or the INTERACTION frame; the latter two go in \
         `EXEMPT` by exact line with the reason. See this file's header. \
         Found:\n{}",
        stray.join("\n")
    );
}

#[test]
fn the_committed_content_sites_really_call_the_portable_pair() {
    // The positive half. Without it, deleting the terrain generators entirely —
    // or reverting them to `.sin()` and adding an exemption — satisfies the ban
    // above perfectly.
    let files = sources();
    for (file, line, what) in MUST_BE_PORTABLE {
        let (_, src) = files
            .iter()
            .find(|(n, _)| n == file)
            .unwrap_or_else(|| panic!("`{file}` is gone from src/"));
        let found = production_only(src)
            .iter()
            .any(|(_, l)| is_code(l) && l.trim() == line);
        assert!(
            found,
            "{file} no longer contains `{line}`.\nThat line produces: {what}\n\
             If it moved, update this pin; if it went back to std trig, the P14 \
             law is broken on a path two machines are compared on."
        );
    }
}

#[test]
fn every_frozen_exemption_still_matches_a_line() {
    // An exemption for a line somebody deleted is a hole nobody is using and
    // nobody will notice — and the next banned call to land on that text gets in
    // free. Same arm `inf-dcc`'s f32 law carries, for the same reason.
    let files = sources();
    for (file, line, why) in EXEMPT {
        let (_, src) = files
            .iter()
            .find(|(n, _)| n == file)
            .unwrap_or_else(|| panic!("the exemption names `{file}`, which is gone"));
        let found = production_only(src)
            .iter()
            .any(|(_, l)| is_code(l) && l.trim() == line);
        assert!(
            found,
            "the frozen exemption {file} / {line:?} matches no line any more \
             (it was allowed because: {why}). Delete it rather than leaving a \
             hole."
        );
    }
}

/// **NOT VACUOUS**, four ways — the scan reads what it thinks it reads, and the
/// classification is doing work rather than covering an empty set.
#[test]
fn the_scan_is_not_looking_at_an_empty_set() {
    let files = sources();
    // (a) The walk really found this crate. A `read_dir` that silently returned
    //     nothing would make every arm above pass.
    assert!(
        files.len() >= 20,
        "only {} source files walked out of src/ — the walk is looking at the \
         wrong directory",
        files.len()
    );
    // (b) It is recursive: `thumbnail/scene_render.rs` lives one level down, and
    //     a top-level-only reader would miss seven of the twelve exemptions.
    assert!(
        files.iter().any(|(n, _)| n.contains('/')),
        "no nested module was walked; the reader is not recursive"
    );
    // (c) The exemptions are load-bearing: with them removed, the ban FIRES. If
    //     this stops being true the crate has no libm calls left and the whole
    //     list should be deleted rather than kept as decoration.
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    let hits = files
        .iter()
        .flat_map(|(_, src)| production_only(src))
        .filter(|(_, l)| is_code(l))
        .filter(|(_, l)| banned.iter().any(|b| l.contains(b)))
        .count();
    assert_eq!(
        hits,
        EXEMPT.len(),
        "the number of banned calls in the crate ({hits}) is not the number of \
         frozen exemptions ({}). If it is larger the main arm should have caught \
         it; if it is smaller, an exemption is stale.",
        EXEMPT.len()
    );
    // (d) The production cut really cuts. `samples.rs` has a large test module,
    //     and a `production_only` that returned the whole file would drag every
    //     fixture in the crate into the ban.
    let (_, samples) = files
        .iter()
        .find(|(n, _)| n == "samples.rs")
        .expect("samples.rs");
    assert!(
        production_only(samples).len() < samples.lines().count(),
        "nothing was cut from samples.rs — either it has no #[cfg(test)] any \
         more or the cut has stopped working"
    );
}

/// **The gate can fail.** A gate that cannot fail is not a gate — measured here
/// rather than reasoned about, on the two shapes that matter.
#[test]
fn the_classification_would_notice_a_new_call_and_a_renamed_one() {
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    let exempt = |file: &str, line: &str| {
        EXEMPT
            .iter()
            .any(|(f, l, _)| *f == file && *l == line.trim())
    };

    // A NEW committed-content call, in a file that already holds exemptions, is
    // not covered by any of them.
    let intruder = "let h = (x * 0.05).sin() * amplitude;";
    assert!(banned.iter().any(|b| intruder.contains(b)));
    assert!(
        !exempt("dcc.rs", intruder),
        "a new `.sin()` in dcc.rs is silently permitted"
    );

    // …and an exempt line that has been EDITED is no longer exempt, which is what
    // makes the pin a pin rather than a licence for the file.
    let edited = "let p = centre + (u * a.cos() + v * a.sin()) * radius * 2.0;";
    assert!(
        exempt(
            "dcc.rs",
            "let p = centre + (u * a.cos() + v * a.sin()) * radius;"
        ),
        "the fixture no longer poses the problem it was written for"
    );
    assert!(
        !exempt("dcc.rs", edited),
        "an edited exemption line still matches; the pin is a file exemption in \
         disguise"
    );

    // The file half is real too: the same text in another file is not exempt.
    assert!(
        !exempt("samples.rs", "let g = 1.0 / (fovy * 0.5).tan();"),
        "an exemption granted to one file is being honoured in another"
    );
}
