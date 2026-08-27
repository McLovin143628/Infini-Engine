//! **The libm source gate for `inf-island`** (wave I7).
//!
//! # The claim this file is here to keep honest
//!
//! The island's crate header says: *"the sampling step is not bit-portable and
//! everything after it is."* That is a load-bearing sentence — it is why the
//! `.inf_terrain` is a build artifact of one machine and why the derived layers
//! are *verified* rather than re-derived — and without a gate it is a sentence.
//!
//! So: **every module in this crate is clean of `std` transcendentals, with no
//! exemption.** The non-portable half is not in here at all; it is
//! `inf_gis::crs` and `inf_gis::tilemath`, which that crate's own gate exempts by
//! name, and this crate reaches them only from the sampling step — the
//! projection lattice and the tile plan, both named in `PROJECTION_ALLOWED`.
//!
//! # The hole this file had, and what it cost (the I7 CI-red)
//!
//! Being clean of transcendentals was never the whole claim. A crate can call no
//! `sin` of its own and still put one in a committed file by asking a library to
//! do it — and that is exactly what happened. `IslandRecipe::anchor` called
//! `inf_gis::anchor_at`, which inverts an easting/northing through `proj4rs`;
//! the three degrees it returns are serialized into the island's committed
//! `.inf_lvl`; macOS and Windows disagreed about the last ulp of one of them and
//! CI went red on one platform of three.
//!
//! The arm that should have caught it enumerated four modules and three needles.
//! `recipe.rs` was not one of the four and `anchor_at(` was not one of the three.
//! Both halves are now the other way up: an **allowlist** of modules with
//! reasons, and a separate ban on the anchor door across every module.
//!
//! Everything this crate computes *itself* — the carve, the priority flood, the
//! D8 routing, the accumulation, the Jenks dispatch, the grade audit, the segment
//! index — is `+ - * /`, comparisons and `sqrt`. `sqrt` is exact in IEEE-754 and
//! therefore bit-identical everywhere, which is why the router measures a chord
//! with it and the slope query goes through `inf_math::portable::patan2_64`
//! instead of `.atan()`.
//!
//! # And the exemption's own release condition, extended
//!
//! `inf-gis`'s gate rests on nothing that **cooks or ships** re-deriving a
//! coordinate through its projection modules. `inf-island` depends on `inf-gis`,
//! so linking `inf-island` into a cooking or shipping crate would link `inf-gis`
//! too and retire that exemption at one remove. The arm below is what keeps that
//! from happening quietly.

/// Every `inf-island` source. Exhaustive by hand, and
/// `the_source_table_covers_every_module` is what keeps it exhaustive — the hole
/// the I2 audit found in the sibling gate, which ran green over eight modules
/// while two new ones went unchecked.
const SOURCES: &[(&str, &str)] = &[
    ("biome.rs", include_str!("../src/biome.rs")),
    ("build.rs", include_str!("../src/build.rs")),
    ("detail.rs", include_str!("../src/detail.rs")),
    ("hydro.rs", include_str!("../src/hydro.rs")),
    ("layers.rs", include_str!("../src/layers.rs")),
    ("lib.rs", include_str!("../src/lib.rs")),
    ("recipe.rs", include_str!("../src/recipe.rs")),
    ("report.rs", include_str!("../src/report.rs")),
    ("roads.rs", include_str!("../src/roads.rs")),
    ("shape.rs", include_str!("../src/shape.rs")),
    ("source.rs", include_str!("../src/source.rs")),
    ("splat.rs", include_str!("../src/splat.rs")),
    ("terrain.rs", include_str!("../src/terrain.rs")),
];

/// Lines of `src` containing `needle`, ignoring comment lines — the bans are on
/// code, and the module docs necessarily *name* the things they ban.
///
/// CRLF-safe by construction (`str::lines` strips a trailing carriage return),
/// which matters because `.rs` is `text eol=lf` in `.gitattributes` precisely so
/// a Windows checkout hands a gate the same bytes a Linux one does — the P22
/// lesson, met by a gate that reads `include_str!`.
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

/// **No module in this crate calls a std transcendental. There is no exemption.**
#[test]
fn the_whole_crate_is_clean_of_std_transcendentals() {
    const GATE: &str = "inf-island/tests/portable_math_law.rs";
    // `powi` is not banned and is not on the list: it is repeated multiplication
    // and exact for an integer exponent. `powf` IS on it.
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    for (name, src) in SOURCES {
        for needle in &banned {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "{GATE}: inf-island/src/{name} calls `{needle}`, which is not \
                 bit-portable across targets (the P14 law). The island's own \
                 header claims everything after the sampling step IS portable, \
                 and the derived layers are committed on the strength of it. Use \
                 the `inf_math::portable` replacement — or, for a width from an \
                 area, a square root, which is exact. Hits: {hits:?}"
            );
        }
    }
}

/// The table is exhaustive.
#[test]
fn the_source_table_covers_every_module() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("read inf-island/src")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".rs"))
        .collect();
    on_disk.sort();
    let mut tabled: Vec<String> = SOURCES.iter().map(|(n, _)| n.to_string()).collect();
    tabled.sort();
    assert_eq!(
        on_disk, tabled,
        "a module was added to inf-island/src without joining this gate's table"
    );
}

/// **The gate can fail** — the anti-vacuity arm the sibling gate learned to want.
///
/// A source scan that never matched anything would be indistinguishable from one
/// whose needles are all typos.
#[test]
fn the_scan_finds_what_it_is_looking_for_when_it_is_there() {
    let sample = "let a = x.sin();\n// x.sin() in a comment\nlet b = 2.0;\n";
    let hits = code_hits(sample, ".sin(");
    assert_eq!(hits.len(), 1, "the comment line must not count: {hits:?}");
    assert_eq!(hits[0].0, 1);
    assert!(code_hits(sample, ".cos(").is_empty());
    // …and the crate really is being scanned, not an empty list.
    assert!(SOURCES.len() >= 10);
    assert!(SOURCES.iter().all(|(_, s)| s.len() > 500));
    assert!(inf_math::libm_ban::ALL.len() > 40);
}

/// The coordinate doors of `inf-gis`'s exempt modules — the calls that hand this
/// crate a number `proj4rs` computed.
const PROJECTION_DOORS: [&str; 4] = [
    "to_source(",
    "to_world(",
    "lonlat_to_mercator(",
    "mercator_to_lonlat(",
];

/// The modules allowed to name one, **and why**.
///
/// An allowlist, not a ban list — the I7 CI-red is the fourth time this house
/// has paid for the difference. The arm this replaces enumerated four modules
/// (`hydro`, `roads`, `biome`, `shape`) and three needles, and seven modules
/// plus the door that actually mattered walked past it: `recipe.rs` called
/// `inf_gis::anchor_at`, whose answer went into a **committed** `.inf_lvl` and
/// differed between macOS and Windows in its last ulp.
const PROJECTION_ALLOWED: [(&str, &str); 3] = [
    (
        "terrain.rs",
        "the projection lattice — the sampling step itself, which this crate's \
         header names as the non-portable half",
    ),
    (
        "source.rs",
        "the tile plan: which XYZ tiles the sampling step must fetch is a \
         question in the source's own frame, and it is answered before any \
         sample exists",
    ),
    (
        "recipe.rs",
        "one test, which reports the longitude of the world's east and west \
         edges so a site outside them is refused by name. It reaches no writer: \
         `anchor_at` is banned in this module by the arm below, which is the \
         door whose output was reaching a committed file",
    ),
];

/// **The projection is reached from the sampling step and nowhere else.**
///
/// `inf-gis`'s exemption covers `crs.rs` and `tilemath.rs`. This crate reaches
/// them, and the modules that may are named above with their reason; every other
/// module in the table is banned, so a module added tomorrow is banned by
/// default rather than unlisted by accident.
#[test]
fn the_projection_is_reached_from_the_sampling_step_and_nowhere_that_derives() {
    for (name, src) in SOURCES {
        if let Some((_, why)) = PROJECTION_ALLOWED.iter().find(|(n, _)| n == name) {
            assert!(!why.is_empty());
            continue;
        }
        for needle in PROJECTION_DOORS {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "inf-island/src/{name} calls `{needle}`, and it is not on this \
                 gate's allowlist. The derivations run on the CARVED HEIGHTS, and \
                 a projection call in one of them would make a committed stream \
                 vertex a fact about the host's libm. If the call is genuinely \
                 part of the sampling step, add the module to \
                 `PROJECTION_ALLOWED` **with its reason**. Hits: {hits:?}"
            );
        }
    }
    // …and the lattice really does reach it, or the boundary above is vacuous.
    let terrain = SOURCES
        .iter()
        .find(|(n, _)| *n == "terrain.rs")
        .expect("terrain.rs is tabled")
        .1;
    assert!(
        !code_hits(terrain, "to_source(").is_empty(),
        "the projection lattice does not call the projection, so the ban above \
         bans nothing"
    );
    assert!(!code_hits(terrain, "lonlat_to_mercator(").is_empty());
    // …and every allowlisted module is a module that exists, so an entry cannot
    // rot into a licence for nothing.
    for (name, _) in PROJECTION_ALLOWED {
        assert!(
            SOURCES.iter().any(|(n, _)| *n == name),
            "`{name}` is allowlisted here and is not in this crate's source table"
        );
    }
}

/// **THE ANCHOR IS STATED, NEVER INVERTED** — the I7 CI-red, as a law.
///
/// `inf_gis::anchor_at` derives a latitude, a longitude and a grid convergence
/// by inverting an easting/northing through `proj4rs`: a series over
/// `sin`/`cos`/`atan2`, i.e. the platform's libm. Those three numbers are
/// serialized into `.inf_lvl`, and `.inf_lvl` is **committed**, so the door is a
/// direct route from one machine's libm into a byte three platforms compare.
/// It took one: `origin_latitude_deg` read 49.34307562364773 on Windows and
/// 49.34307562364772 on macOS — one ulp, one byte at offset 14 788, one red CI.
///
/// So the recipe **states** its geodetic origin (`[anchor] latitude_deg`, …) and
/// `tests/stated_anchor.rs` checks the statement against the projection with a
/// tolerance. Nothing in `inf-island/src` may call the anchor door — including
/// a test helper, because a helper is the shortest path back.
#[test]
fn no_module_inverts_an_anchor_out_of_a_projection() {
    for (name, src) in SOURCES {
        let hits = code_hits(src, "anchor_at(");
        assert!(
            hits.is_empty(),
            "inf-island/src/{name} calls `anchor_at(`. That door inverts through \
             `proj4rs` and its three degrees are written into a COMMITTED \
             `.inf_lvl`, where macOS and Windows have already been measured to \
             disagree in the last ulp. State the origin in the recipe \
             (`[anchor] latitude_deg` / `longitude_deg` / `convergence_deg`) and \
             let `tests/stated_anchor.rs` check it. Hits: {hits:?}"
        );
    }
    // The replacement door is really the one in use, or the ban above is a ban on
    // a crate that stopped anchoring at all.
    let recipe = SOURCES
        .iter()
        .find(|(n, _)| *n == "recipe.rs")
        .expect("recipe.rs is tabled")
        .1;
    assert!(
        !code_hits(recipe, "require_projected_crs(").is_empty(),
        "recipe.rs no longer checks its CRS at all — the anchor ban must not be \
         satisfied by dropping the door instead of replacing it"
    );
    assert!(
        !code_hits(recipe, "latitude_deg").is_empty()
            && !code_hits(recipe, "convergence_deg").is_empty(),
        "the recipe does not carry a stated geodetic origin, so the ban above \
         leaves the anchor with nowhere to come from"
    );
    // …and the scan can see the spelling it bans, on a sample that has it.
    assert_eq!(
        code_hits("let a = inf_gis::anchor_at(x);", "anchor_at(").len(),
        1
    );
}

/// **The exemption's release condition, at one remove.**
///
/// Linking `inf-island` into a cooking or shipping crate links `inf-gis` with it.
#[test]
fn inf_island_is_not_linked_by_the_cook_or_the_runtime() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    for rel in [
        "runtime/inf-player/Cargo.toml",
        "runtime/inf-packager/Cargo.toml",
        "crates/inf-runtime/Cargo.toml",
    ] {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // **Only the SHIPPING sections.** A `[dev-dependencies]` entry is not
        // linked by the shipped player — `island_gate` builds the island through
        // `inf island build`'s own door and drives it on both hosts, which is
        // exactly the kind of test that has to name this crate. What must stay
        // clean is what a cooked or shipped binary links.
        let mut section = "";
        let mut dep_lines: Vec<&str> = Vec::new();
        for raw in src.lines() {
            let l = raw.trim();
            if l.starts_with('[') {
                section = l;
                continue;
            }
            if l.starts_with('#') {
                continue;
            }
            if section.contains("dev-dependencies") || section.contains("build-dependencies") {
                continue;
            }
            if section.contains("dependencies") {
                dep_lines.push(l);
            }
        }
        assert!(
            !dep_lines.is_empty(),
            "{rel} parsed to no dependency lines — this scan is reading nothing"
        );
        for banned in ["inf-island", "inf-gis"] {
            assert!(
                !dep_lines.iter().any(|l| l.starts_with(banned)),
                "{rel} names `{banned}` as a SHIPPING dependency. `inf-island` \
                 depends on `inf-gis`, whose projection modules are exempt from \
                 the portability gate on the grounds that nothing which cooks or \
                 ships re-derives a coordinate through them. Linking either \
                 retires that exemption. (A `[dev-dependencies]` entry is fine \
                 and is how `island_gate` reaches the island.)"
            );
        }
    }
}

/// The **cooking** verbs of the CLI do not reach the island either.
///
/// The island's verbs may — `inf island build` is author-time and writes one
/// machine's numbers, exactly as `inf gis plan` does — and this is where that
/// distinction is written down rather than assumed.
#[test]
fn the_cli_reaches_inf_island_only_from_its_island_verbs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let main = std::fs::read_to_string(root.join("tools/inf-cli/src/main.rs"))
        .expect("read tools/inf-cli/src/main.rs");

    let fn_body = |name: &str| -> Option<String> {
        let at = main.find(&format!("fn {name}("))?;
        let open = main[at..].find('{')? + at;
        let mut depth = 0usize;
        for (i, c) in main[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(main[open..open + i + 1].to_string());
                    }
                }
                _ => {}
            }
        }
        None
    };

    for verb in [
        "cmd_cook",
        "cmd_cook_mods",
        "cmd_export",
        "cmd_pack",
        "cmd_new",
    ] {
        let body =
            fn_body(verb).unwrap_or_else(|| panic!("`fn {verb}(` is gone — this gate names it"));
        assert!(
            !body.contains("inf_island"),
            "`{verb}` names `inf_island`, which reaches `inf_gis`'s exempt \
             projection modules. A cooking verb that does retires the exemption."
        );
    }
    // …and the island verbs really do reach it.
    let build = fn_body("cmd_island_build").expect("`fn cmd_island_build(` exists");
    assert!(
        build.contains("inf_island"),
        "the island verb does not reach inf-island, so the ban above is vacuous"
    );
}
