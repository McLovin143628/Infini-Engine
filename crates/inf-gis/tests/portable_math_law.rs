//! **The libm source gate for `inf-gis`** (Wave-G audit).
//!
//! # Why this file exists
//!
//! Seven crates in this tree carry a source gate over `inf_math::libm_ban::ALL`
//! — `inf-terrain`, `inf-pcg`, `inf-physics`, `inf-dcc`, `inf-anim`,
//! `inf-editor-core`'s erosion mirror, `inf-render`'s trig law. `inf-gis` did
//! not, and the Wave-G audit found that the module whose header spends three
//! paragraphs on why adjacent tile edges must be **bit-identical** computes the
//! lat/lon half of the same arithmetic with `.tan()`, `.ln()`, `.exp()`,
//! `.atan()`, `.cos()` and `.log2()` — the last of which was not even on the
//! canonical ban list until this audit put it there.
//!
//! The law is not "no transcendentals". It is **a value two machines re-derive
//! independently may not depend on one**. So this gate does two things:
//!
//! 1. It holds every module that is *already* clean to the full list, so the
//!    clean ones cannot quietly stop being clean. `classify`, `feature`,
//!    `roads`, `triangulate`, `vector` and `terrarium` are in that set today —
//!    a genuinely good state that nothing was protecting.
//! 2. It pins the exempt set to **exactly** the two modules that are not, by
//!    name, with the reason and the release condition written down. An exemption
//!    that is enumerated is a debt; an exemption that is invisible is a defect.
//!
//! # The exemption, and what retires it
//!
//! `crs.rs` and `tilemath.rs` are projection code. A projection *is*
//! transcendental — there is no `psin`-shaped replacement for an inverse
//! Mercator or for `proj4rs`, which is a third-party crate this gate cannot see
//! inside anyway. What makes the exemption survivable today is that **nothing in
//! the shipped tree cooks anything from either module**: `tilemath`'s tile
//! selection has no caller outside its own tests, and `crs::Transform` is used
//! at an import door whose output an author then saves as an ordinary
//! `.inf_lvl` — one machine's numbers, committed once, never re-derived
//! elsewhere and compared.
//!
//! **The day a cook or a PIE-==-shipping trace re-derives a coordinate through
//! either module, this exemption has to go** — and the way that is enforced is
//! `inf_gis_is_not_linked_by_the_cook_or_the_runtime` below, which fails if any
//! manifest a **cooking or shipping** crate is built from starts naming this
//! crate.
//!
//! # The CLI, and why its ban is a use-site ban (I2)
//!
//! `tools/inf-cli` was on that manifest list, because `inf cook` lives in it.
//! The island's IB-3 put `inf gis info` / `inf gis plan` in the same binary, and
//! linkage alone then failed the gate — for a verb that runs at **author** time
//! and writes an ordinary `.inf_lvl`, which is precisely what the editor's
//! wizard already did through the same code. The condition the exemption
//! actually rests on is about *cooking*, not about linking, so the CLI's ban now
//! aims at that: `the_cli_reaches_inf_gis_only_from_its_gis_verbs` reads the
//! bodies of `cmd_cook`, `cmd_cook_mods`, `cmd_export`, `cmd_pack`, `cmd_new`
//! and `report_export` and fails if any of them mentions `inf_gis`. The cook
//! itself is `inf_packager::cook`, and `inf-packager`'s manifest ban is
//! unchanged and absolute.

/// Every `inf-gis` source, and whether it is exempt from the transcendental ban.
///
/// Kept as an exhaustive table rather than a directory walk on purpose: a new
/// module is a deliberate decision about which half of this list it joins, and a
/// walk would silently put it in the clean half and then fail confusingly.
const SOURCES: &[(&str, &str, bool)] = &[
    ("buildings.rs", include_str!("../src/buildings.rs"), false),
    ("classify.rs", include_str!("../src/classify.rs"), false),
    ("crs.rs", include_str!("../src/crs.rs"), true),
    ("import.rs", include_str!("../src/import.rs"), false),
    ("epsg.rs", include_str!("../src/epsg.rs"), false),
    ("feature.rs", include_str!("../src/feature.rs"), false),
    ("lib.rs", include_str!("../src/lib.rs"), false),
    ("roads.rs", include_str!("../src/roads.rs"), false),
    ("terrarium.rs", include_str!("../src/terrarium.rs"), false),
    ("tilemath.rs", include_str!("../src/tilemath.rs"), true),
    (
        "triangulate.rs",
        include_str!("../src/triangulate.rs"),
        false,
    ),
    ("vector.rs", include_str!("../src/vector.rs"), false),
];

/// Lines of `src` containing `needle`, ignoring comment lines — the bans are on
/// code, and the module docs necessarily *name* the things they ban.
///
/// CRLF-safe by construction (`str::lines` strips a trailing `\r`), which
/// matters because `.rs` is `text eol=lf` in `.gitattributes` precisely so a
/// Windows checkout hands a gate the same bytes a Linux one does.
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

/// Every module outside the named exempt set is clean of std transcendentals.
#[test]
fn no_std_transcendentals_outside_the_named_exemptions() {
    const GATE: &str = "inf-gis/tests/portable_math_law.rs";
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    for (name, src, exempt) in SOURCES {
        if *exempt {
            continue;
        }
        for needle in &banned {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "{GATE}: inf-gis/src/{name} calls `{needle}`, which is not \
                 bit-portable across targets (the P14 law). Use the \
                 `inf_math::portable` replacement, or — if this module genuinely \
                 has to project — add it to this file's exempt table WITH the \
                 reason and the condition that retires it. Hits: {hits:?}"
            );
        }
    }
}

/// **The exemption is a debt, so it is measured.**
///
/// A gate that merely skipped two files would be indistinguishable from a gate
/// somebody had switched off. This one asserts that the exempt modules really
/// are the ones that need it (they contain projection math) and — the part that
/// matters — that the exempt set has not silently grown.
#[test]
fn the_exempt_set_is_exactly_the_projection_modules() {
    let exempt: Vec<&str> = SOURCES
        .iter()
        .filter(|(_, _, e)| *e)
        .map(|(n, _, _)| *n)
        .collect();
    assert_eq!(
        exempt,
        vec!["crs.rs", "tilemath.rs"],
        "the transcendental exemption has moved. Adding a module to it is a \
         deliberate act that needs its reason written down in this file's header \
         and its release condition named; removing one is good news that should \
         come with the portable replacement it now uses."
    );

    // …and each exempt module actually contains what it is exempt for, so an
    // exemption cannot outlive the code that justified it.
    for (name, src, _) in SOURCES.iter().filter(|(_, _, e)| *e) {
        let uses_transcendental = inf_math::libm_ban::ALL
            .iter()
            .any(|needle| !code_hits(src, needle).is_empty());
        assert!(
            uses_transcendental,
            "inf-gis/src/{name} is on the exempt list and no longer calls a \
             banned function — take it off the list rather than leaving a \
             standing exemption nothing needs"
        );
    }
}

/// **The condition that retires the exemption**, asserted rather than promised.
///
/// The exemption rests on one fact: nothing that *cooks* or that a
/// PIE-==-shipping trace compares re-derives a coordinate through `crs` or
/// `tilemath`. `inf-gis` is a host-only crate and the shipped player never links
/// it — that is stated in the crate docs and in the workspace manifest, and this
/// is the arm that keeps it true.
#[test]
fn inf_gis_is_not_linked_by_the_cook_or_the_runtime() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    // Every manifest a **cooking or shipping** binary is built from.
    //
    // `tools/inf-cli` used to be on this list and is not any more; see
    // `the_cli_reaches_inf_gis_only_from_its_gis_verbs` for what replaced it and
    // why. The three below are unchanged and absolute: `inf-packager` IS the
    // cook, and if it cannot name this crate then `inf cook` cannot reach it
    // however the binary around it is linked.
    for rel in [
        "runtime/inf-player/Cargo.toml",
        "runtime/inf-packager/Cargo.toml",
        "crates/inf-runtime/Cargo.toml",
    ] {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // Only the DEPENDENCY sections matter; a comment naming the crate is
        // fine and is in fact how the refusal is recorded.
        let dep_lines: Vec<&str> = src
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with('#'))
            .collect();
        assert!(
            !dep_lines.iter().any(|l| l.starts_with("inf-gis")),
            "{rel} names `inf-gis` as a dependency. That crate's projection math \
             is exempt from the portability gate on the grounds that nothing \
             which cooks or ships re-derives a coordinate through it — linking it \
             into a cooked or shipped path retires that exemption, and \
             `crs.rs`/`tilemath.rs` need portable replacements before it can."
        );
    }
}

/// The body of `fn NAME(` in `src`, by brace matching. `None` when absent.
fn fn_body<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let at = src.find(&format!("fn {name}("))?;
    let rest = &src[at..];
    let open = rest.find('{')?;
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    for (i, b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[open..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// **The CLI may LINK `inf-gis`; the cooking verbs may not REACH it** (I2).
///
/// # Why the manifest ban moved to a use-site ban
///
/// `tools/inf-cli` was on the linkage list above because `inf cook` lives in it.
/// The island's IB-3 put `inf gis info` / `inf gis plan` in the same binary —
/// the headless half of the one import door — and linkage alone then failed the
/// gate.
///
/// The exemption's own condition is not "nothing links this crate"; the header
/// says it in these words: *"the day a cook or a PIE-==-shipping trace
/// re-derives a coordinate through either module, this exemption has to go"*.
/// `inf gis` runs at **author** time and writes an ordinary `.inf_lvl` /
/// `.inf_mesh` — one machine's numbers, committed once — which is exactly what
/// the editor's wizard already did through the same code. `inf cook` calls
/// `inf_packager::cook`, and `inf-packager` still cannot name this crate, so the
/// cook cannot reach it whatever else is in the binary.
///
/// So the ban aims at the thing it names: **the cooking and exporting verbs must
/// not mention `inf_gis`**, checked per function body rather than per manifest.
/// If `cmd_cook` ever reprojects a coordinate, this fails and the exemption goes
/// with it.
#[test]
fn the_cli_reaches_inf_gis_only_from_its_gis_verbs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let main = std::fs::read_to_string(root.join("tools/inf-cli/src/main.rs"))
        .expect("read tools/inf-cli/src/main.rs");

    // The verbs that cook, export or ship bytes, plus the one that scaffolds.
    for verb in [
        "cmd_cook",
        "cmd_cook_mods",
        "cmd_export",
        "cmd_pack",
        "cmd_new",
        "report_export",
    ] {
        let body = fn_body(&main, verb)
            .unwrap_or_else(|| panic!("`fn {verb}(` is gone from the CLI — this gate names it"));
        assert!(
            !body.contains("inf_gis"),
            "`{verb}` names `inf_gis`. The portability exemption on `crs.rs` and \
             `tilemath.rs` rests on nothing that COOKS re-deriving a coordinate \
             through them; a cooking verb that reaches this crate retires it."
        );
    }

    // …and the GIS verbs really do reach it, or this arm is measuring a CLI that
    // has no GIS half at all.
    let gis_body = fn_body(&main, "cmd_gis_plan").expect("`fn cmd_gis_plan(` exists");
    assert!(
        gis_body.contains("inf_gis"),
        "the GIS verb does not reach inf-gis, so the exemption above is vacuous"
    );
}

/// **Every module is in the table** — the gap a new module walked through.
///
/// `SOURCES` is an exhaustive hand-written list, on purpose ("a new module is a
/// deliberate decision about which half of this list it joins"). Nothing was
/// checking that it stayed exhaustive: I2 added `import.rs` and `buildings.rs`
/// and the gate ran green over the eight modules it already knew about, which is
/// the exact shape of an arm that cannot fail. A directory walk here does not
/// weaken the deliberate-decision property — it enforces it, because the walk
/// only says a module is MISSING, never which half it belongs in.
#[test]
fn the_source_table_covers_every_module() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("read inf-gis/src")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".rs"))
        .collect();
    on_disk.sort();
    let mut tabled: Vec<String> = SOURCES.iter().map(|(n, _, _)| n.to_string()).collect();
    tabled.sort();
    assert_eq!(
        on_disk, tabled,
        "inf-gis/src and this file's SOURCES table disagree. Add the new module \
         to the table WITH its exempt flag — `false` unless it genuinely needs a \
         transcendental, in which case the header's exemption note and its \
         release condition have to grow too."
    );
}
